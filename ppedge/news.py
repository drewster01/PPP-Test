"""Tonight's injury and lineup news (data/<league>/news.json), used to flag
players on every board and keep ruled-out players out of lineups.

news.json: [{"player", "team", "status", "note", "beneficiaries": [...]}]
Status OUT / DOUBTFUL removes the player from lineups; QUESTIONABLE and
MINUTES_LIMIT are shown as warnings.
"""

from __future__ import annotations

import json

from .fetch import DATA_DIR
from .names import norm_name

NEWS = DATA_DIR / "news.json"
AVOID = {"OUT", "DOUBTFUL"}
WARN = {"QUESTIONABLE", "MINUTES_LIMIT"}


def _today_file(path) -> bool:
    """News is slate-specific: only use a file saved on today's date (US Eastern)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, et).date() == datetime.now(et).date()
    except OSError:
        return False


def load() -> dict[str, dict]:
    if not _today_file(NEWS):
        return {}
    try:
        items = json.loads(NEWS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {norm_name(i["player"]): i for i in items if i.get("player")}


SITUATIONS = DATA_DIR / "situations.json"


def situations() -> dict[str, list[dict]]:
    """Game-script flags by team: BLOWOUT, PACE_UP, PACE_DOWN."""
    if not _today_file(SITUATIONS):
        return {}
    try:
        items = json.loads(SITUATIONS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, list[dict]] = {}
    for it in items:
        for t in it.get("teams", []):
            out.setdefault(t, []).append({"flag": it.get("flag"), "note": it.get("note", "")})
    return out


def previous_probs(path) -> dict:
    """Last run's fair probability per prop, to spot moves since then."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {(p["player"], p["stat"]): (p["line"], p["pick"], p["adj"], doc.get("generated_at"))
            for p in doc.get("props", [])}


def status(news: dict, player: str) -> dict | None:
    return news.get(norm_name(player))


def to_narratives(news: dict) -> list[dict]:
    """Turn news beneficiaries into Narrative Board entries (adjustments format)."""
    out = []
    for item in news.values():
        for b in item.get("beneficiaries") or []:
            out.append({
                "id": f"news-{norm_name(item['player']).replace(' ', '-')}-{norm_name(b['player']).replace(' ', '-')}",
                "narrative": True, "player": b["player"], "team": b.get("team"),
                "stat": b.get("stats") or [], "side": b.get("side", "Over"),
                "delta_pp": float(b.get("lean_pp", 1.0)),
                "thought": f"{item['player']} {item.get('status', '').lower()}: {b.get('why', '')} ({item.get('note', '')})",
            })
    return out
