"""Free, no-login sharp sources: Kalshi exchange and Pinnacle's public guest feed.

Kalshi:   official public market data (no auth for reads). Each player has
          several thresholds, which feed the ladder fit directly.
Pinnacle: the public guest feed the website uses, league-level endpoints only
          (per-matchup endpoints require a token; we do not use one).
Blocked requests are reported and skipped, never worked around.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

from . import config
from .names import norm_team
from .parse_sgo import PlayerMarket, Quote

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_EVENT_SUFFIX = "26OCT08TBDAL"
KALSHI_SERIES = {
    "KXNFLRECYDS": "rec_yds", "KXNFLREC": "receptions", "KXNFLRSHYDS": "rush_yds",
    "KXNFLPASSYDS": "pass_yds", "KXNFLLONGREC": "long_rec", "KXNFLRSHATT": "rush_att",
    "KXNFLRRYDS": "rush_rec_yds", "KXNFLPASSTDS": "pass_td", "KXNFLPASSATT": "pass_att",
    "KXNFLPASSCOMP": "pass_comp", "KXNFLPASSINT": "pass_int", "KXNFLTD": "tds",
}
KALSHI_MAX_SPREAD = 0.15  # skip markets with bid/ask wider than this
KALSHI_MIN_MID = 0.12  # skip deep-tail contracts (mid below 12% or above 88%)

PINNACLE = "https://guest.api.arcadia.pinnacle.com/0.1"
PINNACLE_NFL = 889
PINNACLE_UNITS = {
    "Receiving Yards": "rec_yds", "Receptions": "receptions", "Rushing Yards": "rush_yds",
    "Rush Attempts": "rush_att", "Pass Completions": "pass_comp", "Interceptions": "pass_int",
    "Passing Yards": "pass_yds", "Pass Attempts": "pass_att", "Touchdown Passes": "pass_td",
    "Touchdowns": "tds", "Field Goals": "fg_made", "Longest Reception": "long_rec",
    "Rushing + Receiving Yards": "rush_rec_yds",
}
HEADERS = {"User-Agent": config.BROWSER_UA, "Accept": "application/json"}

if config.LEAGUE == "nba":
    KALSHI_SERIES = {
        "KXNBAPTS": "pts", "KXNBAREB": "reb", "KXNBAAST": "ast", "KXNBA3PT": "fg3m",
        "KXNBAPRA": "pra", "KXNBASTL": "stl", "KXNBABLK": "blk",
    }
    PINNACLE_NFL = 487  # NBA league id on the guest feed
    PINNACLE_UNITS = {"Points": "pts", "Rebounds": "reb", "Assists": "ast",
                      "Threes Made": "fg3m", "Pts & Rebs & Asts": "pra",
                      "Steals": "stl", "Blocks": "blk"}


def _slate_date_code() -> str:
    """Kalshi event-ticker date code for today's slate in US Eastern, e.g. 26OCT21."""
    from zoneinfo import ZoneInfo
    d = datetime.now(ZoneInfo("America/New_York"))
    return d.strftime("%y%b%d").upper()


def _get(url: str, params: dict | None, source: str, log):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    except requests.RequestException as exc:
        log(f"{source}: network error ({type(exc).__name__}).")
        return None
    if r.status_code != 200:
        log(f"{source}: HTTP {r.status_code}; skipped (not retried or bypassed).")
        return None
    return r.json()


# ---------------------------------------------------------------- Kalshi
def fetch_kalshi(log) -> dict | None:
    out = {}
    code = _slate_date_code()
    for series in KALSHI_SERIES:
        if config.GAME_TEAMS is None:
            # whole slate: every open market in the series, kept to today's games
            params = {"series_ticker": series, "status": "open", "limit": 1000}
        else:
            params = {"event_ticker": f"{series}-{KALSHI_EVENT_SUFFIX}", "limit": 1000}
        body = _get(f"{KALSHI}/markets", params, "Kalshi", log)
        if body is None:
            continue
        ms = body.get("markets") or []
        if config.GAME_TEAMS is None:
            ms = [m for m in ms if f"-{code}" in str(m.get("event_ticker", ""))]
        out[series] = ms
        time.sleep(0.2)
    if not out:
        return None
    log(f"Kalshi: {sum(len(v) for v in out.values())} market(s) across {len(out)} series.")
    return {"fetched_at": datetime.now(timezone.utc).isoformat(), "series": out}


# Every Kalshi contract as posted, for display: (normalized player, stat key) -> {N: prices}.
# Pricing still uses only the filtered, informative contracts.
KALSHI_LADDER: dict[tuple, dict[int, dict]] = {}


def _build_ladder(raw: dict) -> None:
    from .names import norm_name
    KALSHI_LADDER.clear()
    for series, markets in (raw.get("series") or {}).items():
        spec_key = KALSHI_SERIES.get(series)
        for m in markets:
            if m.get("status") not in ("active", None):
                continue
            player = str(m.get("title", "")).split(":")[0].strip()
            try:
                n = int(float(m.get("floor_strike")) + 0.5)  # "N+" contract: floor_strike = N - 0.5
                bid = float(m.get("yes_bid_dollars") or 0)
                ask = float(m.get("yes_ask_dollars") or 0)
            except (TypeError, ValueError):
                continue
            if player and 0 < ask < 1:
                KALSHI_LADDER.setdefault((norm_name(player), spec_key), {})[n] = {"yes_bid": bid, "yes_ask": ask}


def parse_kalshi(raw: dict, log) -> list[PlayerMarket]:
    _build_ladder(raw)
    asof = datetime.fromisoformat(raw["fetched_at"]) if raw.get("fetched_at") else None
    teams = config.GAME_TEAMS
    grouped: dict[tuple, PlayerMarket] = {}
    skipped = 0
    for series, markets in (raw.get("series") or {}).items():
        spec_key = KALSHI_SERIES.get(series)
        for m in markets:
            try:
                if m.get("status") not in ("active", None):
                    raise ValueError
                bid = float(m.get("yes_bid_dollars"))
                ask = float(m.get("yes_ask_dollars"))
                line = float(m.get("floor_strike"))
            except (TypeError, ValueError):
                skipped += 1
                continue
            if bid <= 0 or ask >= 1 or ask - bid > KALSHI_MAX_SPREAD:
                skipped += 1
                continue
            mid = (bid + ask) / 2.0
            if not KALSHI_MIN_MID <= mid <= 1.0 - KALSHI_MIN_MID:
                # deep tails: longshot contracts are systematically overpriced on
                # exchanges (favorite-longshot bias) and distort the fitted mean
                skipped += 1
                continue
            player = str(m.get("title", "")).split(":")[0].strip()
            seg = str(m.get("ticker", "")).split("-")
            # multi-game slates: Kalshi team codes differ from PrizePicks', match on name
            team = (next((t for t in teams if len(seg) > 2 and seg[2].startswith(t)), None)
                    if teams else None)
            if not player:
                skipped += 1
                continue
            # tighter markets, and thresholds near the middle, carry more information
            w = max(0.3, 1.0 - (ask - bid) / KALSHI_MAX_SPREAD) * (4 * mid * (1 - mid))
            q = Quote("kalshi", line, 0, 0, asof, False, "prob", mid, w,
                      {"yes_bid": bid, "yes_ask": ask})
            key = (player, team, spec_key)
            if key not in grouped:
                grouped[key] = PlayerMarket(player, team, spec_key, series, [])
            grouped[key].quotes.append(q)
    if skipped:
        log(f"Kalshi: skipped {skipped} one-sided, wide or inactive market(s).")
    return list(grouped.values())


# ---------------------------------------------------------------- Pinnacle
def fetch_pinnacle(log) -> dict | None:
    matchups = _get(f"{PINNACLE}/leagues/{PINNACLE_NFL}/matchups", {"brandId": 0}, "Pinnacle", log)
    if matchups is None:
        return None
    if config.GAME_TEAMS is None:
        # whole slate: every game starting within the next 18 hours
        now = datetime.now(timezone.utc)
        games = [m for m in matchups if m.get("type") == "matchup" and m.get("startTime")
                 and 0 <= (datetime.fromisoformat(m["startTime"].replace("Z", "+00:00")) - now).total_seconds() < 18 * 3600]
        if not games:
            log("Pinnacle: no games on today's slate.")
            return None
        game = {"id": [g["id"] for g in games]}
        ids_parent = set(game["id"])
    else:
        game = next((m for m in matchups if m.get("type") == "matchup"
                     and {p.get("name") for p in m.get("participants", [])}
                     >= {"Dallas Cowboys", "Tampa Bay Buccaneers"}), None)
        if game is None:
            log("Pinnacle: Bucs-Cowboys matchup not listed.")
            return None
        ids_parent = {game["id"]}
    specials = [m for m in matchups if m.get("parentId") in ids_parent and m.get("type") == "special"]
    time.sleep(0.5)
    straight = _get(f"{PINNACLE}/leagues/{PINNACLE_NFL}/markets/straight", None, "Pinnacle", log)
    if straight is None:
        return None
    ids = {s["id"] for s in specials}
    markets = [m for m in straight if m.get("matchupId") in ids]
    log(f"Pinnacle: {len(specials)} special(s), {len(markets)} priced market(s) for the game.")
    return {"fetched_at": datetime.now(timezone.utc).isoformat(), "game": game,
            "specials": specials, "markets": markets}


def parse_pinnacle(raw: dict, log) -> list[PlayerMarket]:
    asof = datetime.fromisoformat(raw["fetched_at"]) if raw.get("fetched_at") else None
    by_id = {m["matchupId"]: m for m in raw.get("markets", [])
             if m.get("period") == 0 and m.get("type") == "total"}
    out = []
    unknown = set()
    for s in raw.get("specials", []):
        sp = s.get("special") or {}
        if sp.get("category") != "Player Props":
            continue
        units = s.get("units") or ""
        spec_key = PINNACLE_UNITS.get(units)
        if spec_key is None:
            unknown.add(units)
            continue
        desc = sp.get("description", "")
        player = desc.split(" Total ")[0].strip() if " Total " in desc else desc.replace(units, "").strip()
        side_by_pid = {p["id"]: p.get("name") for p in s.get("participants", [])}
        m = by_id.get(s["id"])
        if not m:
            continue
        over = under = None
        line = None
        for pr in m.get("prices", []):
            side = side_by_pid.get(pr.get("participantId"))
            if side == "Over":
                over, line = pr.get("price"), pr.get("points")
            elif side == "Under":
                under = pr.get("price")
        if over is None or under is None or line is None:
            continue
        out.append(PlayerMarket(player, None, spec_key, units,
                                [Quote("pinnacle", float(line), float(over), float(under), asof, False)]))
    if unknown:
        log(f"Pinnacle: unmapped prop types {sorted(unknown)}.")
    return out
