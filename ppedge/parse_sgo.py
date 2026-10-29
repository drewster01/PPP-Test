"""Parse SportsGameOdds events into per-player, per-stat two-way quotes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from . import config
from .fetch import _is_target_game
from .names import norm_name, norm_team, short_key


@dataclass
class Quote:
    book: str
    line: float
    over: float  # American odds
    under: float
    updated: datetime | None
    is_alt: bool
    quality: str = "two_sided"  # two_sided | one_sided | line_only | prob
    prob: float | None = None  # precomputed fair P(over) for non two-sided quotes
    weight: float = 1.0  # source-specific multiplier (e.g. Kalshi bid/ask tightness)
    raw: dict | None = None  # the price exactly as the source shows it (e.g. Kalshi bid/ask)


@dataclass
class PlayerMarket:
    player: str
    team: str | None
    spec_key: str
    sgo_stat: str
    quotes: list[Quote]


def _ts(v) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def _alt_entries(entry: dict) -> list[dict]:
    alts = entry.get("altLines") or entry.get("alternateLines") or []
    if isinstance(alts, dict):
        alts = list(alts.values())
    return [a for a in alts if isinstance(a, dict)]


def find_event(sgo: dict) -> dict | None:
    for e in sgo.get("data") or []:
        if _is_target_game(e):
            return e
    return None


def parse(sgo: dict, log) -> tuple[list[PlayerMarket], dict]:
    event = find_event(sgo)
    if event is None:
        names = []
        for e in sgo.get("data") or []:
            t = e.get("teams") or {}
            names.append(" @ ".join(str(((t.get(s) or {}).get("names") or {}).get("short", "?"))
                                    for s in ("away", "home")))
        log(f"SportsGameOdds: target game not found. Events returned: {names}")
        return [], {"event_found": False}

    team_by_id = {}
    for side in ("home", "away"):
        t = (event.get("teams") or {}).get(side) or {}
        team_by_id[t.get("teamID")] = norm_team((t.get("names") or {}).get("short") or t.get("teamID"))
    players = event.get("players") or {}

    diag = Counter()
    unmapped = Counter()
    # (playerID, statID) -> {"over": odd, "under": odd}
    pairs: dict[tuple[str, str], dict] = defaultdict(dict)
    for odd in (event.get("odds") or {}).values():
        if odd.get("betTypeID") != "ou" or odd.get("periodID") != "game":
            continue
        pid = odd.get("playerID") or odd.get("statEntityID")
        if pid not in players:
            continue
        stat = odd.get("statID")
        if stat not in config.SGO_STAT_INDEX:
            unmapped[stat] += 1
            continue
        side = odd.get("sideID")
        if side in ("over", "under"):
            pairs[(pid, stat)][side] = odd

    markets: list[PlayerMarket] = []
    for (pid, stat), sides in pairs.items():
        if "over" not in sides or "under" not in sides:
            diag["one_sided_market"] += 1
            continue
        p = players[pid]
        name = p.get("name") or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
        quotes: list[Quote] = []
        ob = sides["over"].get("byBookmaker") or {}
        ub = sides["under"].get("byBookmaker") or {}
        for book in set(ob) & set(ub):
            if book.lower() in config.EXCLUDED_BOOKS:
                continue
            o, u = ob[book], ub[book]
            if o.get("available") is False or u.get("available") is False:
                diag["unavailable"] += 1
                continue
            try:
                lo, lu = float(o.get("overUnder")), float(u.get("overUnder"))
                if lo == lu:
                    quotes.append(Quote(book, lo, float(o["odds"]), float(u["odds"]),
                                        _ts(o.get("lastUpdatedAt") or u.get("lastUpdatedAt")), False))
                else:
                    diag["main_line_mismatch"] += 1
            except (TypeError, ValueError, KeyError):
                diag["bad_quote"] += 1
            # alternate lines: pair over and under at the same line
            u_alt = {}
            for a in _alt_entries(u):
                try:
                    u_alt[float(a.get("overUnder"))] = a
                except (TypeError, ValueError):
                    pass
            for a in _alt_entries(o):
                try:
                    line = float(a.get("overUnder"))
                    b = u_alt.get(line)
                    if b is None or a.get("available") is False or b.get("available") is False:
                        continue
                    quotes.append(Quote(book, line, float(a["odds"]), float(b["odds"]),
                                        _ts(a.get("lastUpdatedAt") or o.get("lastUpdatedAt")), True))
                    diag["alt_quotes"] += 1
                except (TypeError, ValueError, KeyError):
                    diag["bad_alt"] += 1
        if quotes:
            markets.append(PlayerMarket(
                player=name,
                team=team_by_id.get(p.get("teamID")) or norm_team(p.get("teamID")),
                spec_key=config.SGO_STAT_INDEX[stat].key,
                sgo_stat=stat,
                quotes=quotes,
            ))
    diag["markets"] = len(markets)
    if unmapped:
        diag["unmapped_player_stats"] = dict(unmapped)
    return markets, dict(diag)


def index_markets(markets: list[PlayerMarket]) -> tuple[dict, dict]:
    full, short = {}, {}
    for m in markets:
        full.setdefault((norm_name(m.player), m.team, m.spec_key), []).append(m)
        short.setdefault((short_key(m.player), m.team, m.spec_key), []).append(m)
    return full, short
