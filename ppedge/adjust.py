"""Apply capped, explicitly logged situational adjustments."""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .names import norm_name
from .pricing import PricedProp

# NFL narratives live in adjustments.json; other leagues get their own file
ADJ_FILE = Path(__file__).resolve().parent.parent / (
    "adjustments.json" if config.LEAGUE == "nfl" else f"adjustments_{config.LEAGUE}.json")


def load(path: Path = ADJ_FILE) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("adjustments", [])
    except (OSError, ValueError):
        return []


def _matches(adj: dict, pp: PricedProp) -> bool:
    p = pp.prop
    if p.spec_key not in adj.get("stat", []):
        return False
    if adj.get("only_tier") and adj["only_tier"] != pp.tier:
        return False
    if adj.get("team") and adj["team"] != p.team:
        return False
    if adj.get("positions") and (p.position or "").upper() not in adj["positions"]:
        return False
    if adj.get("player"):
        want = norm_name(adj["player"])
        have = norm_name(p.player)
        if want != have and want.split()[-1] != have.split()[-1]:
            return False
    return True


def apply_narratives(priced: list[PricedProp], adjustments: list[dict]) -> list[dict]:
    """Narrative mode: apply every narrative entry in full (capped) to the probabilities.

    Used only for the separate Narrative Board; the main board stays pure-market.
    """
    log = []
    for adj in adjustments:
        if not (adj.get("narrative") or adj.get("user_edge")):
            continue
        hits = [pp for pp in priced if _matches(adj, pp)]
        raw = float(adj.get("delta_pp", 0.0)) / 100.0
        delta = (1.0 if adj.get("side", "Over") == "Over" else -1.0) * max(
            -config.ADJ_CAP_EACH, min(config.ADJ_CAP_EACH, raw))
        thought = adj.get("thought") or adj.get("reason", "")
        for pp in hits:
            room = config.ADJ_CAP_TOTAL - abs(sum(d for _, d in pp.adjustments))
            pp.adjustments.append((adj.get("id", "?"), max(-room, min(room, delta)) if room > 0 else 0.0))
            pp.thoughts = getattr(pp, "thoughts", []) + [thought]
        log.append({"id": adj.get("id"), "matched": [f"{h.prop.player} {h.prop.stat_type}" for h in hits],
                    "applied_pp": delta * 100, "reason": thought, "priced_in": False, "user_edge": False})
    return log


def apply(priced: list[PricedProp], adjustments: list[dict]) -> list[dict]:
    """Mutate priced props in place; return a log of every adjustment considered."""
    log = []
    for adj in adjustments:
        hits = [pp for pp in priced if _matches(adj, pp)]
        raw = float(adj.get("delta_pp", 0.0)) / 100.0
        each = max(-config.ADJ_CAP_EACH, min(config.ADJ_CAP_EACH, raw))
        narrative = bool(adj.get("user_edge") or adj.get("narrative"))
        if not narrative:
            each *= config.ADJ_SITUATIONAL_WEIGHT
        applied = 0.0 if adj.get("priced_in") else each
        sign = 1.0 if adj.get("side", "Over") == "Over" else -1.0
        if narrative:
            # narratives go to their own board and never change rankings or lineups
            thought = adj.get("thought") or adj.get("reason", "")
            for pp in hits:
                pp.assumed.append((adj.get("id", "?"), sign * each, thought))
            log.append({"id": adj.get("id"), "matched": [f"{h.prop.player} {h.prop.stat_type}" for h in hits],
                        "requested_pp": raw * 100, "applied_pp": 0.0, "priced_in": False,
                        "user_edge": True, "assumed_pp": sign * each * 100, "reason": adj.get("reason", "")})
            continue
        for pp in hits:
            current = sum(d for _, d in pp.adjustments)
            delta = sign * applied
            room = config.ADJ_CAP_TOTAL - abs(current)
            delta = max(-room, min(room, delta)) if room > 0 else 0.0
            pp.adjustments.append((adj.get("id", "?"), delta))
        log.append({
            "id": adj.get("id"),
            "matched": [f"{h.prop.player} {h.prop.stat_type}" for h in hits],
            "requested_pp": raw * 100,
            "applied_pp": applied * 100 * sign,
            "priced_in": bool(adj.get("priced_in")),
            "user_edge": bool(adj.get("user_edge")),
            "reason": adj.get("reason", ""),
        })
    return log
