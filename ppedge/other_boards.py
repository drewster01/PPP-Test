"""Underdog and DraftKings Pick6 boards, priced from the same sharp markets.

Underdog: public v1 lobby feed (no login). Each pick side carries its own
  payout_multiplier, so EV uses Underdog's real per-pick payout.
Pick6:    the public pick6.draftkings.com page embeds its board as React Router
  "turbo-stream" data (no login). Only the categories the public page serves
  (currently Points) are available.
Requests use a normal browser User-Agent only; blocks are reported and skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config, nba_combos
from .devig import prob_to_american
from .fetch import DATA_DIR
from .parse_pp import PPProp
from .pricing import find_market, price
from .turbo_stream import decode, extract

UA = config.BROWSER_UA
# Per-leg breakeven at a 1.0x pick multiplier for a typical 3-5 pick entry on
# these apps (about 55%); a pick's own multiplier scales the requirement.
BASE_NEED = 0.55
MAX_GAP_SD = 1.0  # skip lines more than 1 SD away from every book line

UD_URL = "https://api.underdogfantasy.com/v1/over_under_lines"
UD_SPORT = {"nba": "NBA", "nfl": "NFL"}
# Underdog stat id -> (our spec key or None, PrizePicks-style stat name)
UD_STATS = {
    "points": ("pts", "Points"), "rebounds": ("reb", "Rebounds"), "assists": ("ast", "Assists"),
    "three_points_made": ("fg3m", "3-PT Made"), "pts_rebs_asts": ("pra", "Pts+Rebs+Asts"),
    "steals": ("stl", "Steals"), "blocks": ("blk", "Blocked Shots"),
    "pts_rebs": (None, "Pts+Rebs"), "pts_asts": (None, "Pts+Asts"), "rebs_asts": (None, "Rebs+Asts"),
    "blks_stls": (None, "Blks+Stls"),
    # NFL
    "receiving_yds": ("rec_yds", "Receiving Yards"), "receptions": ("receptions", "Receptions"),
    "rushing_yds": ("rush_yds", "Rush Yards"), "passing_yds": ("pass_yds", "Pass Yards"),
    "rush_attempts": ("rush_att", "Rush Attempts"), "passing_comps": ("pass_comp", "Pass Completions"),
    "passing_att": ("pass_att", "Pass Attempts"), "passing_tds": ("pass_td", "Pass TDs"),
}
PICK6_URL = "https://pick6.draftkings.com/"
PICK6_STATS = {"Points": ("pts", "Points"), "Rebounds": ("reb", "Rebounds"), "Assists": ("ast", "Assists"),
               "3-Pointers Made": ("fg3m", "3-PT Made"), "Pts + Reb + Ast": ("pra", "Pts+Rebs+Asts")}


@dataclass
class OtherLine:
    platform: str
    line_id: str
    player: str
    team: str | None
    opp: str | None
    start: str | None
    stat: str
    spec_key: str | None
    line: float
    mult_over: float | None
    mult_under: float | None
    photo: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- Underdog
def fetch_underdog(log):
    try:
        r = requests.get(UD_URL, headers={"User-Agent": UA, "Accept": "application/json"}, timeout=40)
    except requests.RequestException as exc:
        log(f"Underdog: network error ({type(exc).__name__}).")
        return None
    if r.status_code != 200:
        log(f"Underdog: HTTP {r.status_code}; skipped.")
        return None
    body = r.json()
    sport = UD_SPORT[config.LEAGUE]
    players = {p["id"]: p for p in body.get("players", []) if p.get("sport_id") == sport}
    apps = {a["id"]: a for a in body.get("appearances", []) if a.get("player_id") in players}
    lines = [l for l in body.get("over_under_lines", [])
             if ((l.get("over_under") or {}).get("appearance_stat") or {}).get("appearance_id") in apps]
    games = [g for g in body.get("games", []) if g.get("sport_id") == sport] or body.get("games", [])
    log(f"Underdog: {len(lines)} {sport} line(s).")
    return {"fetched_at": _now(), "players": list(players.values()), "appearances": list(apps.values()),
            "lines": lines, "games": games}


def parse_underdog(raw) -> list[OtherLine]:
    players = {p["id"]: p for p in raw.get("players", [])}
    apps = {a["id"]: a for a in raw.get("appearances", [])}
    team_abbr = {}
    games = {}
    for g in raw.get("games", []):
        title = str(g.get("abbreviated_title") or "")
        if " @ " in title:
            away, home = title.split(" @ ", 1)
            team_abbr[g.get("away_team_id")] = away.strip()
            team_abbr[g.get("home_team_id")] = home.strip()
            games[g.get("id")] = g
    out = []
    for l in raw.get("lines", []):
        if l.get("status") != "active":
            continue
        st = (l.get("over_under") or {}).get("appearance_stat") or {}
        stat_id = st.get("stat")
        if stat_id not in UD_STATS:
            continue
        a = apps.get(st.get("appearance_id")) or {}
        p = players.get(a.get("player_id")) or {}
        g = games.get(a.get("match_id")) or {}
        team = team_abbr.get(a.get("team_id"))
        opp = next((v for k, v in team_abbr.items() if k in (g.get("home_team_id"), g.get("away_team_id"))
                    and v != team), None)
        mult = {}
        for o in l.get("options", []):
            if o.get("status") == "active":
                try:
                    mult[o.get("choice")] = float(o.get("payout_multiplier") or 1.0)
                except (TypeError, ValueError):
                    pass
        try:
            line = float(l.get("stat_value"))
        except (TypeError, ValueError):
            continue
        key, name = UD_STATS[stat_id]
        out.append(OtherLine("underdog", l.get("id"), f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                             team, opp, g.get("scheduled_at"), name, key, line,
                             mult.get("higher"), mult.get("lower"), p.get("dark_image_url") or p.get("image_url")))
    return out


# ---------------------------------------------------------------- Pick6
def fetch_pick6(log):
    try:
        r = requests.get(PICK6_URL, params={"sport": config.LEAGUE.upper()},
                         headers={"User-Agent": UA, "Accept": "text/html"}, timeout=40)
    except requests.RequestException as exc:
        log(f"Pick6: network error ({type(exc).__name__}).")
        return None
    if r.status_code != 200:
        log(f"Pick6: HTTP {r.status_code}; skipped.")
        return None
    try:
        pcr = decode(extract(r.text))["loaderData"]["routes/_homeShared"]["pickCardsResponse"]
    except (KeyError, TypeError, ValueError, IndexError):
        log("Pick6: page data not found; skipped.")
        return None
    keep = {k: pcr.get(k) for k in ("pickCardByPickableId", "entityInfoByDkId", "displayTeamById",
                                    "competitionById", "pickSixMarketById")}
    log(f"Pick6: {len(keep['pickCardByPickableId'] or {})} card(s).")
    return {"fetched_at": _now(), "pcr": keep}


def parse_pick6(raw) -> list[OtherLine]:
    pcr = raw.get("pcr") or {}
    ents = pcr.get("entityInfoByDkId") or {}
    teams = pcr.get("displayTeamById") or {}
    comps = pcr.get("competitionById") or {}
    mkts = pcr.get("pickSixMarketById") or {}
    out = []
    for pid, card in (pcr.get("pickCardByPickableId") or {}).items():
        if not card.get("entities"):
            continue
        ent = card["entities"][0]
        dk = str(ent.get("dkId"))
        info = ents.get(dk) or {}
        comp = comps.get(str((ent.get("compIds") or [None])[0])) or {}
        team_id = ((comp.get("entityCompByDkId") or {}).get(dk) or {}).get("teamId")
        team = (teams.get(str(team_id)) or {}).get("name")
        opp_id = comp.get("awayTeamId") if team_id == comp.get("homeTeamId") else comp.get("homeTeamId")
        opp = (teams.get(str(opp_id)) or {}).get("name")
        for m in card.get("activePickableMarkets") or []:
            if m.get("pickableMarketId") != card.get("defaultPickableMarketId") or m.get("isPaused"):
                continue
            if m.get("promoPickTypeId") not in (None, 1):
                continue  # promos like "Stat Slice" use altered lines and special rules
            name = (mkts.get(str(m.get("pickSixMarketId"))) or {}).get("name")
            if name not in PICK6_STATS:
                continue
            mult = {s.get("statLinePropositionId"): s.get("standingsMultiplier") for s in m.get("activeSelections") or []}
            key, stat = PICK6_STATS[name]
            out.append(OtherLine("pick6", str(m.get("pickableMarketId")), info.get("fullName") or info.get("name", "?"),
                                 team, opp, comp.get("startTime"), stat, key, float(m.get("targetValue")),
                                 mult.get(1) if 1 in mult else None, mult.get(2) if 2 in mult else None,
                                 (info.get("imageUrls") or {}).get("160")))
    return out


# ---------------------------------------------------------------- pricing + board
def price_lines(lines: list[OtherLine], full: dict, short: dict, photos: dict, find_photo) -> list[dict]:
    from .parse_pp import _tonight
    out = []
    for ol in lines:
        if config.GAME_TEAMS is None and not _tonight(ol.start):
            continue  # tomorrow's games are on these boards too
        allowed = "under_or_over" if ol.mult_under is not None else "over"
        prop = PPProp(ol.line_id, ol.player, ol.team, None, ol.stat, ol.line, "standard", ol.start,
                      ol.spec_key, allowed)
        if ol.spec_key:
            m = find_market(prop, full, short)
            pp = price(prop, m) if m else None
        else:
            pp = nba_combos.price_combo(prop, full, short) if config.LEAGUE == "nba" else None
        if pp is None or pp.tier != "sharp":
            continue
        # Pinnacle's posted line and price, for the straight-up Mores view
        pin_post = None
        if ol.spec_key:
            pins = [q for q in (m.quotes if m else []) if q.book == "pinnacle"]
            if pins:
                q = min(pins, key=lambda q: (abs(q.line - ol.line), q.is_alt))
                pin_post = {"line": q.line, "over": q.over, "under": q.under}
        if pp.line_gap_sd > MAX_GAP_SD:
            continue  # line far outside anything the books price: too uncertain
        # Shrinking toward 50% is cautious on even lines but flatters long-shot
        # alt lines that pay big multipliers, so use the less favorable of the
        # raw and shrunk probability for each side.
        over = min(pp.p_over, pp.p_over_raw)
        under = min(1.0 - pp.p_over, 1.0 - pp.p_over_raw)
        sides = []
        if ol.mult_over is not None:
            sides.append(("More", over, ol.mult_over or 1.0))
        if ol.mult_under is not None:
            sides.append(("Less", under, ol.mult_under or 1.0))
        if not sides:
            continue
        side, p, mult = max(sides, key=lambda s: s[1] * s[2])
        need = min(0.99, BASE_NEED / mult) if mult > 0 else 0.99
        out.append({
            "id": ol.line_id, "player": ol.player, "team": ol.team, "opp": ol.opp, "start": ol.start,
            "pos": None, "stat": ol.stat, "line": ol.line, "pick": side, "adj": round(p, 4),
            "american": round(prob_to_american(p)), "mult": mult, "need": round(need, 4),
            "plus_ev": p >= need, "tier": "sharp", "more_only": ol.mult_under is None,
            "photo": find_photo(photos, ol.player, ol.team) or ol.photo,
            "derived": pp.method in ("combo-components", "ppr-components"), "adjusted": False,
            "by_source": {"pinnacle": pin_post} if pin_post else {}, "mult_over": ol.mult_over,
            "mult_under": ol.mult_under,
        })
    out.sort(key=lambda d: -(d["adj"] - d["need"]))
    return out


def line_shop(pp_props: list[dict], boards: dict[str, list[dict]]) -> list[dict]:
    """Same player + stat across apps: every app's line, side, payout and value.

    Value = fair probability of the side minus the hit rate that app's payout
    needs. The best app for a pick is the one with the highest value.
    """
    from .names import norm_name
    rows: dict[tuple, dict] = {}
    for platform, props in (("prizepicks", pp_props), *boards.items()):
        for p in props:
            key = (norm_name(p["player"]), config.norm_stat(p["stat"]))
            r = rows.setdefault(key, {"player": p["player"], "team": p.get("team"), "opp": p.get("opp"),
                                      "stat": p["stat"], "photo": p.get("photo"), "start": p.get("start"),
                                      "apps": []})
            r["photo"] = r["photo"] or p.get("photo")
            r["apps"].append({"platform": platform, "line": p["line"], "pick": p["pick"], "prob": p["adj"],
                              "american": p["american"], "need": p["need"], "mult": p.get("mult"),
                              "value": round(p["adj"] - p["need"], 4), "plus_ev": p["plus_ev"]})
    out = []
    for r in rows.values():
        if len({a["platform"] for a in r["apps"]}) < 2:
            continue  # nothing to shop
        r["apps"].sort(key=lambda a: -a["value"])
        r["best"] = r["apps"][0]
        lines = [a["line"] for a in r["apps"]]
        r["line_gap"] = round(max(lines) - min(lines), 2)
        out.append(r)
    out.sort(key=lambda r: -r["best"]["value"])
    return out


def write_board(platform: str, props: list[dict], path: Path, meta: dict) -> None:
    doc = {"generated_at": _now(), "kickoff": config.FIRST_START or config.KICKOFF_ISO,
           "league": config.LEAGUE, "game": config.GAME_LABEL, "platform": platform, "mode": "market",
           "unit": config.UNIT, "bankroll": config.BANKROLL, "props": props, "menu": [], "portfolio": [],
           "narratives": [], "pp_fetched_at": meta.get(f"{platform}_fetched_at")}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
