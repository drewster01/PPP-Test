"""More free public sources: Bovada, Kambi (BetRivers / Unibet), PointsBet ON,
Smarkets exchange, Sleeper and Dabble pick'em.

All are read-only public JSON with no login or key. Blocked requests are
reported and skipped. Every source becomes PlayerMarket quotes; PrizePicks
lines are still the only lines we pick from.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone

import requests

from . import config
from .fetch import DATA_DIR
from .names import norm_team
from .parse_sgo import PlayerMarket, Quote

HEADERS = {"User-Agent": config.BROWSER_UA, "Accept": "application/json"}
HOME, AWAY = "Dallas Cowboys", "Tampa Bay Buccaneers"
_PARTIAL = re.compile(r"\b(1h|2h|1st|2nd|3rd|4th|half|quarter|q[1-4]|first\s+\d+|1q|2q|3q|4q)\b", re.I)


def _get(url, source, log, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=40)
    except requests.RequestException as exc:
        log(f"{source}: network error ({type(exc).__name__}).")
        return None
    if r.status_code != 200:
        log(f"{source}: HTTP {r.status_code}; skipped.")
        return None
    r.encoding = "utf-8"
    try:
        return r.json()
    except ValueError:
        log(f"{source}: response was not JSON; skipped.")
        return None


def stat_key(text: str) -> str | None:
    """Map any source's stat label to our spec key (full-game stats only)."""
    t = re.sub(r"[_\-]", " ", (text or "").lower())
    if _PARTIAL.search(t):
        return None
    t = t.replace("player ", "").replace("total ", "")
    if "longest" in t:
        if "rec" in t:
            return "long_rec"
        if "rush" in t:
            return "long_rush"
        if "comp" in t or "pass" in t:
            return "long_comp"
        return None
    if ("rush" in t and "rec" in t) or "rushing and receiving" in t:
        return "rush_rec_yds" if "yard" in t or "yds" in t else None
    if "pass" in t and "rush" in t:
        return "pass_rush_yds" if "yard" in t or "yds" in t else None
    if "first" in t or "per " in t or "percentage" in t or "rating" in t:
        return None
    if "rec" in t and ("yard" in t or "yds" in t):
        return "rec_yds"
    if "reception" in t:
        return "receptions"
    if "rush" in t and ("yard" in t or "yds" in t):
        return "rush_yds"
    if "rush" in t and "attempt" in t:
        return "rush_att"
    if "pass" in t and ("yard" in t or "yds" in t):
        return "pass_yds"
    if "completion" in t:
        return "pass_comp"
    if "pass" in t and "attempt" in t:
        return "pass_att"
    if ("pass" in t and ("touchdown" in t or "td" in t.split())) or "passing tds" in t:
        return "pass_td"
    if "interception" in t or t.strip() in ("ints", "int"):
        return "pass_int"
    if "tackles and assists" in t or "tackles + assists" in t:
        return "tackles"
    if "kicking points" in t:
        return "kick_pts"
    if "field goal" in t:
        return "fg_made"
    if "anytime" in t and "touchdown" in t:
        return "tds"
    if t.strip() == "sacks":
        return "sacks"
    return None


def _swap(name: str) -> str:
    """'Lamb, CeeDee' -> 'CeeDee Lamb'."""
    if "," in name:
        last, first = [s.strip() for s in name.split(",", 1)]
        return f"{first} {last}"
    return name.strip()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _add(groups, player, team, key, quote, label):
    if not player or key is None:
        return
    k = (player, team, key)
    if k not in groups:
        groups[k] = PlayerMarket(player, team, key, label, [])
    groups[k].quotes.append(quote)


def _pair_markets(groups, book, asof, rows, label):
    """rows: iterable of (player, team, key, side, line, decimal_odds)."""
    sides = {}
    for player, team, key, side, line, dec in rows:
        if key is None or not player or dec is None or dec <= 1.0:
            continue
        sides.setdefault((player, team, key, float(line)), {})[side] = dec
    for (player, team, key, line), s in sides.items():
        if "over" in s and "under" in s:
            _add(groups, player, team, key,
                 Quote(book, line, _dec_to_am(s["over"]), _dec_to_am(s["under"]), asof, False), label)


def _dec_to_am(d: float) -> float:
    return (d - 1) * 100 if d >= 2 else -100 / (d - 1)


# ------------------------------------------------------------------ Bovada
def fetch_bovada(log):
    base = "https://www.bovada.lv/services/sports/event/coupon/events/A/description/football/nfl"
    lst = _get(base, "Bovada", log, {"marketFilterId": "def", "preMatchOnly": "true", "lang": "en"})
    if not lst:
        return None
    ev = next((e for grp in lst for e in grp.get("events", [])
               if "Buccaneers" in e.get("description", "") and "Cowboys" in e.get("description", "")), None)
    if not ev:
        log("Bovada: game not listed.")
        return None
    time.sleep(1)
    body = _get(f"https://www.bovada.lv/services/sports/event/coupon/events/A/description{ev['link']}",
                "Bovada", log, {"lang": "en"})
    return {"fetched_at": _now().isoformat(), "event": body} if body else None


def parse_bovada(raw, log):
    asof = datetime.fromisoformat(raw["fetched_at"])
    rows = []
    for grp in raw.get("event") or []:
        for e in grp.get("events", []):
            for dg in e.get("displayGroups", []):
                for m in dg.get("markets", []):
                    if m.get("key") != "2W-OU" or (m.get("period") or {}).get("abbreviation") != "G":
                        continue
                    desc = m.get("description", "")
                    tm = re.search(r"\((TB|DAL)\)\s*$", desc)
                    dk = m.get("descriptionKey") or desc
                    if not tm or " - " not in dk:
                        continue
                    stat_txt, player = dk.split(" - ", 1)
                    key = stat_key(stat_txt.replace(" O/U", ""))
                    player = re.sub(r"\s*\((TB|DAL)\)\s*$", "", player).strip()
                    for o in m.get("outcomes", []):
                        p = o.get("price") or {}
                        try:
                            rows.append((player, tm.group(1), key, "over" if o.get("type") == "O" else "under",
                                         float(p["handicap"]), float(p["decimal"])))
                        except (KeyError, TypeError, ValueError):
                            continue
    groups = {}
    _pair_markets(groups, "bovada", asof, rows, "bovada")
    return list(groups.values())


# ------------------------------------------------------------------ Kambi
KAMBI_BRANDS = {"ub": "unibet", "rsiuspa": "betrivers"}


def fetch_kambi(log):
    out = {}
    for brand in KAMBI_BRANDS:
        lst = _get(f"https://eu-offering-api.kambicdn.com/offering/v2018/{brand}/listView/american_football/nfl.json",
                   f"Kambi {brand}", log, {"lang": "en_US", "market": "US"})
        if not lst:
            continue
        ev = next((e["event"] for e in lst.get("events", [])
                   if "Cowboys" in e["event"].get("homeName", "") and "Buccaneers" in e["event"].get("awayName", "")), None)
        if not ev:
            continue
        time.sleep(1)
        body = _get(f"https://eu-offering-api.kambicdn.com/offering/v2018/{brand}/betoffer/event/{ev['id']}.json",
                    f"Kambi {brand}", log, {"lang": "en_US", "market": "US", "includeParticipants": "true"})
        if body:
            out[brand] = body
        time.sleep(1)
    return {"fetched_at": _now().isoformat(), "brands": out} if out else None


def parse_kambi(raw, log):
    asof = datetime.fromisoformat(raw["fetched_at"])
    groups = {}
    for brand, body in (raw.get("brands") or {}).items():
        teams = {}
        for e in body.get("events", []):
            for p in e.get("participants", []):
                teams[p.get("participantId")] = norm_team(p.get("name"))
        rows = []
        for bo in body.get("betOffers", []):
            if (bo.get("betOfferType") or {}).get("id") != 127:
                continue
            label = (bo.get("criterion") or {}).get("label", "")
            if re.match(r"^\d+\+", label):
                continue
            key = stat_key(label.replace("by the Player", "").replace("- Including Overtime", ""))
            for o in bo.get("outcomes", []):
                if o.get("status") != "OPEN" or o.get("type") not in ("OT_OVER", "OT_UNDER"):
                    continue
                try:
                    rows.append((_swap(o.get("participant", "")), teams.get(o.get("eventParticipantId")), key,
                                 "over" if o["type"] == "OT_OVER" else "under",
                                 o["line"] / 1000.0, o["odds"] / 1000.0))
                except (KeyError, TypeError):
                    continue
        _pair_markets(groups, KAMBI_BRANDS[brand], asof, rows, KAMBI_BRANDS[brand])
    return list(groups.values())


# ------------------------------------------------------------------ PointsBet ON
def fetch_pointsbet(log):
    evs = _get("https://api.on.pointsbet.com/api/v2/competitions/6/events/featured", "PointsBet", log,
               {"includeLive": "false", "page": "1"})
    if not evs:
        return None
    ev = next((e for e in evs.get("events", []) if "Buccaneers" in e.get("name", "")
               and "Cowboys" in e.get("name", "")), None)
    if not ev:
        log("PointsBet: game not listed.")
        return None
    time.sleep(1)
    body = _get(f"https://api.on.pointsbet.com/api/mes/v3/events/{ev['key']}", "PointsBet", log)
    if not body:
        return None
    keep = {k: body.get(k) for k in ("team1Id", "team2Id", "homeTeam", "awayTeam")}
    keep["fixedOddsMarkets"] = [{"eventClass": m.get("eventClass"), "outcomes": m.get("outcomes", [])}
                                for m in body.get("fixedOddsMarkets", [])]
    return {"fetched_at": _now().isoformat(), "event": keep}


def parse_pointsbet(raw, log):
    asof = datetime.fromisoformat(raw["fetched_at"])
    ev = raw.get("event") or {}
    teams = {str(ev.get("team1Id")): "DAL", str(ev.get("team2Id")): "TB"}
    rows = []
    for m in ev.get("fixedOddsMarkets", []):
        key = stat_key(m.get("eventClass", "").replace("Quarterback ", ""))
        for o in m.get("outcomes", []):
            mm = re.match(r"^(.+) (Over|Under) ([\d.]+)$", o.get("name", ""))
            if not mm or not o.get("isOpenForBetting", True):
                continue
            try:
                rows.append((mm.group(1), teams.get(str(o.get("teamId"))), key, mm.group(2).lower(),
                             float(o.get("points") or mm.group(3)), float(o["price"])))
            except (KeyError, TypeError, ValueError):
                continue
    groups = {}
    _pair_markets(groups, "pointsbet", asof, rows, "pointsbet")
    return list(groups.values())


# ------------------------------------------------------------------ Smarkets
SMARKETS_STATS = {
    "PLAYER_RECEPTIONS": "receptions", "PLAYER_RECEIVING_YARDS": "rec_yds", "PLAYER_RUSHING_YARDS": "rush_yds",
    "PLAYER_PASSING_YARDS": "pass_yds", "PLAYER_LONGEST_RECEPTION": "long_rec", "PLAYER_LONGEST_RUSH": "long_rush",
    "PLAYER_LONGEST_COMPLETION": "long_comp", "PLAYER_PASSING_TOUCHDOWNS": "pass_td",
    "PLAYER_KICKING_POINTS": "kick_pts", "PLAYER_FIELD_GOALS": "fg_made", "PLAYER_SACKS": "sacks",
    "PLAYER_RECEIVING_TOUCHDOWNS": "rec_td", "PLAYER_RUSHING_TOUCHDOWNS": "rush_td",
}


def fetch_smarkets(log):
    evs = _get("https://api.smarkets.com/v3/events/", "Smarkets", log,
               {"type": "american_football_match", "state": "upcoming"})
    if not evs:
        return None
    ev = next((e for e in evs.get("events", []) if "Buccaneers" in e.get("name", "")
               and "Cowboys" in e.get("name", "")), None)
    if not ev:
        log("Smarkets: game not listed.")
        return None
    time.sleep(1)
    mk = _get(f"https://api.smarkets.com/v3/events/{ev['id']}/markets/", "Smarkets", log)
    if not mk:
        return None
    markets = [m for m in mk.get("markets", []) if ((m.get("market_type") or {}).get("params") or {}).get("PLAYER_NAME")]
    ids = [m["id"] for m in markets]
    contracts, quotes = [], {}
    for i in range(0, len(ids), 90):
        time.sleep(1)
        c = _get(f"https://api.smarkets.com/v3/markets/{','.join(ids[i:i + 90])}/contracts/", "Smarkets", log)
        if c:
            contracts += c.get("contracts", [])
        time.sleep(1)
        q = _get(f"https://api.smarkets.com/v3/markets/{','.join(ids[i:i + 90])}/quotes/", "Smarkets", log)
        if q:
            quotes.update(q)
    return {"fetched_at": _now().isoformat(), "markets": markets, "contracts": contracts, "quotes": quotes}


def parse_smarkets(raw, log):
    asof = datetime.fromisoformat(raw["fetched_at"])
    by_market = {}
    for c in raw.get("contracts", []):
        by_market.setdefault(c.get("market_id"), {})[(c.get("contract_type") or {}).get("name")] = c.get("id")
    groups = {}
    for m in raw.get("markets", []):
        mt = m.get("market_type") or {}
        params = mt.get("params") or {}
        key = SMARKETS_STATS.get(mt.get("name"))
        cs = by_market.get(m.get("id"), {})
        if key is None or "OVER" not in cs:
            continue
        q = raw.get("quotes", {}).get(str(cs["OVER"])) or {}
        bids, offers = q.get("bids") or [], q.get("offers") or []
        if not bids or not offers:
            continue
        bid = max(b["price"] for b in bids) / 10000.0
        ask = min(o["price"] for o in offers) / 10000.0
        if ask - bid > 0.15:
            continue
        w = max(0.3, 1.0 - (ask - bid) / 0.15) * (0.6 if (m.get("info") or {}).get("is_alternate") else 1.0)
        try:
            line = float(params.get("TOTAL"))
        except (TypeError, ValueError):
            continue
        _add(groups, params["PLAYER_NAME"], None, key,
             Quote("smarkets", line, 0, 0, asof, False, "prob", (bid + ask) / 2.0, w), "smarkets")
    return list(groups.values())


# ------------------------------------------------------------------ Sleeper
SLEEPER_PLAYERS = DATA_DIR / "sleeper_players.json"


def _sleeper_names(log) -> dict:
    try:
        doc = json.loads(SLEEPER_PLAYERS.read_text(encoding="utf-8"))
        if _now() - datetime.fromisoformat(doc["fetched_at"]) < timedelta(days=1):
            return doc["names"]
    except (OSError, ValueError, KeyError):
        pass
    body = _get("https://api.sleeper.app/v1/players/nfl", "Sleeper players", log)
    if not body:
        return {}
    names = {pid: p.get("full_name") for pid, p in body.items()
             if p.get("full_name") and p.get("team") in config.GAME_TEAMS}
    DATA_DIR.mkdir(exist_ok=True)
    SLEEPER_PLAYERS.write_text(json.dumps({"fetched_at": _now().isoformat(), "names": names}), encoding="utf-8")
    return names


def fetch_sleeper(log):
    body = _get("https://api.sleeper.app/lines/available", "Sleeper", log,
                {"dynamic": "true", "include_preseason": "true"})
    if body is None:
        return None
    lines = [l for l in body if l.get("sport") == "nfl"
             and {o.get("subject_team") for o in l.get("options", [])} & set(config.GAME_TEAMS)]
    for l in lines:
        l.pop("recent_performance", None)
    return {"fetched_at": _now().isoformat(), "lines": lines, "names": _sleeper_names(log)}


def parse_sleeper(raw, log):
    asof = datetime.fromisoformat(raw["fetched_at"])
    names = raw.get("names") or {}
    rows = []
    for l in raw.get("lines", []):
        if l.get("status") != "active":
            continue
        key = stat_key(l.get("wager_type", ""))
        for o in l.get("options", []):
            if o.get("status") != "active":
                continue
            try:
                rows.append((names.get(str(o.get("subject_id"))), o.get("subject_team"), key,
                             o.get("outcome"), float(o["outcome_value"]), float(o["payout_multiplier"])))
            except (KeyError, TypeError, ValueError):
                continue
    groups = {}
    _pair_markets(groups, "sleeper", asof, rows, "sleeper")
    return list(groups.values())


# ------------------------------------------------------------------ Dabble
DABBLE_NFL = "7f7c2f8f-ff04-4e5a-a3c8-d503a523d9ad"


def fetch_dabble(log):
    body = _get(f"https://api.dabble.com/competitions/{DABBLE_NFL}/sport-fixtures", "Dabble", log)
    if not body:
        return None
    fx = next((f for f in body.get("data", []) if "Buccaneers" in f.get("name", "")
               and "Cowboys" in f.get("name", "")), None)
    if not fx:
        log("Dabble: game not listed.")
        return None
    return {"fetched_at": _now().isoformat(), "fixture": {k: fx.get(k) for k in ("name", "markets", "selections", "prices")}}


def parse_dabble(raw, log):
    asof = datetime.fromisoformat(raw["fetched_at"])
    fx = raw.get("fixture") or {}
    markets = {m["id"]: m for m in fx.get("markets") or []}
    price = {p["selectionId"]: p.get("price") for p in fx.get("prices") or []}
    # selections do not carry their market id; the price records link the two
    sel_market = {p["selectionId"]: p.get("marketId") for p in fx.get("prices") or []}
    rows = []
    for s in fx.get("selections") or []:
        m = markets.get(s.get("marketId") or sel_market.get(s.get("id")))
        if not m:
            continue
        slug = m.get("resultingType") or ""
        mm = re.match(rf"^(.+?) {re.escape(slug)} ([\d.]+) (over|under)$", s.get("name", ""))
        if not mm:
            continue
        try:
            rows.append((mm.group(1), None, stat_key(slug), mm.group(3), float(mm.group(2)), float(price[s["id"]])))
        except (KeyError, TypeError, ValueError):
            continue
    groups = {}
    _pair_markets(groups, "dabble", asof, rows, "dabble")
    return list(groups.values())


SOURCES = {
    "bovada": (fetch_bovada, parse_bovada),
    "kambi": (fetch_kambi, parse_kambi),
    "pointsbet": (fetch_pointsbet, parse_pointsbet),
    "smarkets": (fetch_smarkets, parse_smarkets),
    "sleeper": (fetch_sleeper, parse_sleeper),
    "dabble": (fetch_dabble, parse_dabble),
}
