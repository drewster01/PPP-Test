"""Write data/board.json for the live HTML board (web/index.html)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from . import config, headshots
from .devig import prob_to_american
from .fetch import DATA_DIR
from .lineups import LineupResult, breakeven, outcome_rows, payout_scale
from .pricing import PricedProp
from .report import _reason, _side

BOARD = DATA_DIR / "board.json"
BOARD_NARR = DATA_DIR / "board_narr.json"


def _need(pp: PricedProp) -> float:
    """Per-leg hit rate needed to be +EV, after PrizePicks' reduced payout on adjusted lines."""
    base = min(breakeven("flex", 6), breakeven("power", 6))
    return base / config.ADJUSTED_LEG_FACTOR if pp.prop.adjusted else base


def _num(x: float) -> float | None:
    return round(x, 4) if x == x else None  # NaN -> null


def kelly(outcomes: list[dict], fraction: float = config.KELLY_FRACTION) -> float:
    """Fractional Kelly stake (share of bankroll) maximizing E[log wealth].

    Shrunk per Baker & McHale (2013) and capped per entry. 0 when EV <= 1.
    """
    probs = np.array([o["prob"] for o in outcomes])
    mult = np.array([o["mult"] for o in outcomes])
    if float((probs * mult).sum()) <= 1.0:
        return 0.0
    res = minimize_scalar(lambda f: -float((probs * np.log1p(f * (mult - 1.0))).sum()),
                          bounds=(0.0, 0.99), method="bounded")
    return min(fraction * float(res.x), config.MAX_EXPOSURE_PER_ENTRY)


def _prop(pp: PricedProp) -> dict:
    old = pp.p_over_normal if pp.pick == "Over" else 1 - pp.p_over_normal
    return {
        "id": pp.prop.pp_id, "player": pp.prop.player, "team": pp.prop.team,
        "pos": pp.prop.position, "stat": pp.prop.stat_type, "line": pp.prop.line,
        "pick": _side(pp), "fair": round(pp.fair_prob, 4), "adj": round(pp.prob, 4),
        "old": round(old, 4) if old == old else None, "american": round(prob_to_american(pp.prob)),
        "books": pp.books, "n_books": pp.n_books, "book_line": pp.book_line,
        "push": round(pp.p_push, 4), "method": pp.method, "more_only": not pp.prop.under_allowed,
        "flag": pp.prob >= config.FLAG_THRESHOLD and pp.tier == "sharp", "reason": _reason(pp),
        "tier": pp.tier, "edge": round(pp.edge, 4),
        "strong": pp.tier == "sharp" and pp.prob >= config.FLAG_THRESHOLD + config.EDGE_MARGIN,
        "assumed": [{"id": k, "pp": round(d * 100, 2), "reason": r} for k, d, r in pp.assumed],
        "assumed_prob": round(pp.assumed_prob, 4) if pp.assumed else None,
        "adjustments": [{"id": k, "pp": round(d * 100, 2)} for k, d in pp.adjustments if abs(d) > 1e-9],
        "updated_at": pp.prop.updated_at,
        "jersey": pp.prop.jersey, "opp": pp.prop.opponent, "start": pp.prop.start_time,
        "photo": getattr(pp, "photo", None), "thoughts": list(dict.fromkeys(getattr(pp, "thoughts", []))), "market_prob": round(pp.fair_prob, 4), "market_american": round(prob_to_american(pp.fair_prob)),
        # +EV leg: clears the lowest per-leg breakeven of any entry type, raised for
        # reduced-payout (adjusted) lines
        "adjusted": pp.prop.adjusted,
        "derived": pp.method in ("combo-components", "ppr-components"),
        "news": getattr(pp, "news", None),
        "situations": getattr(pp, "situations", []),
        "by_source": getattr(pp, "by_source", {}),
        "moved": getattr(pp, "moved", None),
        "need": round(_need(pp), 4),
        "plus_ev": pp.prob >= _need(pp),
    }


def _lineup(name: str, r: LineupResult) -> dict:
    n = len(r.legs)
    return {
        "name": name,
        "legs": [l.prop.pp_id for l in r.legs],
        "optimized_for": r.fmt,
        "formats": {
            f: {**r.by_format[f], "outcomes": outcome_rows(r.dist, n, f),
                "kelly": kelly(outcome_rows(r.dist, n, f)),
                "top_mult": config.POWER_PAYOUTS[n] if f == "power" else config.FLEX_PAYOUTS[n][n]}
            for f in ("flex", "power")
        },
    }


def _menu_item(r: LineupResult) -> dict:
    n = len(r.legs)
    scale = payout_scale(r.legs)
    rows = [{**o, "mult": round(o["mult"] * scale, 2) if o["mult"] != 1.0 else 1.0}
            for o in outcome_rows(r.dist, n, r.fmt)]
    # fold push-shortened results into a simple "N of n" view for display
    simple = {}
    for o in rows:
        if o["mult"] <= 0 or "push" in o["label"]:
            continue
        label = o["label"].split(" (")[0]
        s = simple.setdefault(label, {"label": label, "prob": 0.0, "mult": o["mult"]})
        s["prob"] += o["prob"]
    return {
        "fmt": r.fmt, "size": n,
        "scale": round(scale, 3),
        "legs": [{"player": l.prop.player, "team": l.prop.team, "photo": getattr(l, "photo", None),
                  "adjusted": l.prop.adjusted,
                  "stat": l.prop.stat_type, "line": l.prop.line,
                  "pick": _side(l), "american": round(prob_to_american(l.prob))} for l in r.legs],
        "ev": round(r.ev, 4), "p_profit": round(r.p_profit, 4),
        "breakeven_leg": round(breakeven(r.fmt, n), 4),
        "payouts": sorted(simple.values(), key=lambda s: -s["mult"]),
        # full outcome distribution so the page can recompute EV with the app's real multipliers
        "outcomes": [{"m": o["m"], "h": o["h"], "prob": round(o["prob"], 5), "mult": o["mult"]} for o in rows],
        "key": "|".join(f"{l.prop.player}:{l.prop.spec_key}:{l.prop.line}:{_side(l)}" for l in r.legs) + f"|{r.fmt}",
    }


def _narrative(pp: PricedProp) -> dict:
    """Books' odds vs odds with the user's narrative applied, for the Narrative board."""
    total = sum(d for _, d, _ in pp.assumed)
    side = "Over" if total >= 0 else "Under"
    p_book = pp.p_adj if side == "Over" else 1.0 - pp.p_adj
    p_narr = min(max(p_book + abs(total), 0.01), 0.99)
    thoughts = list(dict.fromkeys(t for _, _, t in pp.assumed if t))
    return {
        "player": pp.prop.player, "team": pp.prop.team, "pos": pp.prop.position,
        "photo": getattr(pp, "photo", None), "stat": pp.prop.stat_type, "line": pp.prop.line,
        "pick": "More" if side == "Over" else "Less",
        "book_prob": round(p_book, 4), "book_american": round(prob_to_american(p_book)),
        "narr_prob": round(p_narr, 4), "narr_american": round(prob_to_american(p_narr)),
        "agrees_with_books": (side == "Over") == (pp.pick == "Over"),
        "plus_ev": p_narr >= min(breakeven("flex", 6), breakeven("power", 6)),
        "more_only": not pp.prop.under_allowed,
        "thought": " ".join(thoughts),
    }


def write(priced, top, extras, lineups, unmatched, adj_log, meta, notes, run_at, menu=(), photos=None,
          portfolio=(), path: Path = BOARD, mode: str = "market") -> Path:
    photos = photos or {}
    for pp in priced:
        pp.photo = headshots.find(photos, pp.prop.player, pp.prop.team)
    narratives = sorted((_narrative(pp) for pp in priced if pp.assumed
                         and not (not pp.prop.under_allowed and sum(d for _, d, _ in pp.assumed) < 0)),
                        key=lambda n: -n["narr_prob"])
    doc = {
        "generated_at": run_at.isoformat(),
        "kickoff": config.FIRST_START or config.KICKOFF_ISO, "league": config.LEAGUE,
        "game": config.GAME_LABEL, "mode": mode,
        "sgo_fetched_at": meta.get("sgo_fetched_at"),
        "pp_fetched_at": meta.get("pp_fetched_at"),
        "sgo_key_set": bool(os.environ.get("SGO_KEY")),
        "refresh_min": config.LIVE_REFRESH_MIN,
        "bankroll": config.BANKROLL,
        "unit": config.UNIT,
        "breakeven": {"flex6": breakeven("flex", 6), "power2": breakeven("power", 2),
                      "power6": breakeven("power", 6)},
        "payouts": {"power": config.POWER_PAYOUTS, "flex": config.FLEX_PAYOUTS},
        "props": [_prop(pp) for pp in sorted(priced, key=lambda x: -x.prob)],
        "top10": [pp.prop.pp_id for pp in top],
        "extras": [pp.prop.pp_id for pp in extras],
        "lineups": [_lineup("AB"[i] if i < 2 else str(i + 1), r) for i, r in enumerate(lineups)],
        "menu": [_menu_item(r) for r in menu],
        "narratives": narratives,
        "portfolio": [_menu_item(r) for r in portfolio],
        "unmatched": unmatched,
        "adjustments": adj_log,
        "notes": notes,
    }
    DATA_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)  # atomic, so the page never reads a half-written file
    return path
