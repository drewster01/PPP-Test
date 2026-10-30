"""League-wide NBA data from NBA.com (nba_api), saved once a day so the model makes a few bulk requests
instead of hundreds of per-player ones.

- game_logs(season): every player's box score line for every game of a season (one request),
- tracking(season): rebound chances and usage per player (two requests).

NBA.com is reached with nba_api's normal browser headers, a timeout and a few spaced retries. If it
still fails, callers fall back to ESPN / Basketball-Reference.
"""
from __future__ import annotations

import json
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CACHE = Path(__file__).resolve().parent.parent / "data" / "nba" / "cache"
SEASON_NAME = {2027: "2026-27", 2026: "2025-26", 2025: "2024-25"}
RETRIES = 3
TIMEOUT = 45
CURRENT = 2027  # the season still being played is re-downloaded once a day; finished seasons once

_mem: dict = {}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv"))


def _cached(key: str, season: int, fetch) -> list[dict] | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    day = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    path = CACHE / (f"{key}_{season}_{day}.json" if season == CURRENT else f"{key}_{season}.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(RETRIES):
        try:
            rows = fetch()
            path.write_text(json.dumps(rows), encoding="utf-8")
            return rows
        except Exception:  # network or NBA.com refusal: wait and try again, then give up
            time.sleep(3 * (attempt + 1))
    return None


def _frame_rows(endpoint) -> list[dict]:
    df = endpoint.get_data_frames()[0]
    return json.loads(df.to_json(orient="records"))


def game_logs(season: int) -> dict[str, list[dict]] | None:
    """{normalized player name: [games oldest first, PrizePicks stat names]} for a regular season."""
    if ("logs", season) in _mem:
        return _mem[("logs", season)]
    from nba_api.stats.endpoints import playergamelogs

    rows = _cached("gamelogs", season, lambda: _frame_rows(playergamelogs.PlayerGameLogs(
        season_nullable=SEASON_NAME[season], season_type_nullable="Regular Season", timeout=TIMEOUT)))
    if rows is None:
        _mem[("logs", season)] = None
        return None
    out: dict[str, list[dict]] = {}
    for r in rows:
        fgm, fga, tpm, tpa = r["FGM"], r["FGA"], r["FG3M"], r["FG3A"]
        pts, reb, ast = r["PTS"], r["REB"], r["AST"]
        out.setdefault(norm(r["PLAYER_NAME"]), []).append({
            "date": r["GAME_DATE"], "minutes": float(r["MIN"] or 0), "Points": pts, "Rebounds": reb,
            "Assists": ast, "Steals": r["STL"], "Blocked Shots": r["BLK"], "Turnovers": r["TOV"],
            "Personal Fouls": r["PF"], "Offensive Rebounds": r["OREB"], "Defensive Rebounds": r["DREB"],
            "FG Made": fgm, "FG Attempted": fga, "3-PT Made": tpm, "3-PT Attempted": tpa,
            "Two Pointers Made": fgm - tpm, "Two Pointers Attempted": fga - tpa,
            "Free Throws Made": r["FTM"], "Free Throws Attempted": r["FTA"],
            "Pts+Rebs": pts + reb, "Pts+Asts": pts + ast, "Rebs+Asts": reb + ast, "Pts+Rebs+Asts": pts + reb + ast,
            "Blks+Stls": r["BLK"] + r["STL"], "team": r["TEAM_ABBREVIATION"], "matchup": r["MATCHUP"],
        })
    for games in out.values():
        games.sort(key=lambda g: g["date"])
    _mem[("logs", season)] = out
    return out


def tracking(season: int) -> dict[str, dict] | None:
    """{player: {reb_chances, oreb_chances, dreb_chances, usage}} per game, from NBA.com tracking."""
    if ("trk", season) in _mem:
        return _mem[("trk", season)]
    from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptstats

    reb = _cached("rebounding", season, lambda: _frame_rows(leaguedashptstats.LeagueDashPtStats(
        season=SEASON_NAME[season], pt_measure_type="Rebounding", player_or_team="Player", timeout=TIMEOUT)))
    adv = _cached("advanced", season, lambda: _frame_rows(leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON_NAME[season], measure_type_detailed_defense="Advanced", timeout=TIMEOUT)))
    out: dict[str, dict] = {}
    for r in reb or []:
        out.setdefault(norm(r["PLAYER_NAME"]), {}).update(
            reb_chances=r.get("REB_CHANCES"), oreb_chances=r.get("OREB_CHANCES"), dreb_chances=r.get("DREB_CHANCES"))
    for r in adv or []:
        out.setdefault(norm(r["PLAYER_NAME"]), {}).update(usage=r.get("USG_PCT"), minutes=r.get("MIN"))
    _mem[("trk", season)] = out or None
    return _mem[("trk", season)]


def defense_factors(season: int) -> dict[str, dict[str, float]]:
    """{team: {stat: what opponents get against this team per game / league average}} from the logs."""
    if ("def", season) in _mem:
        return _mem[("def", season)]
    logs = game_logs(season) or {}
    per_game: dict[tuple, dict] = {}  # (date, offense team, defense team) -> stat totals
    for games in logs.values():
        for g in games:
            m = g.get("matchup") or ""
            opp = m.replace("@", "vs.").split("vs.")[-1].strip()
            key = (g["date"], g.get("team"), opp)
            tot = per_game.setdefault(key, {})
            for k, v in g.items():
                if isinstance(v, (int, float)) and k not in ("minutes",):
                    tot[k] = tot.get(k, 0) + v
    allowed: dict[str, dict[str, list]] = {}
    league: dict[str, list] = {}
    for (_, _, defense), tot in per_game.items():
        for k, v in tot.items():
            allowed.setdefault(defense, {}).setdefault(k, []).append(v)
            league.setdefault(k, []).append(v)
    avg = {k: sum(v) / len(v) for k, v in league.items() if v}
    out = {t: {k: (sum(v) / len(v)) / avg[k] for k, v in st.items() if avg.get(k)} for t, st in allowed.items()}
    _mem[("def", season)] = out
    return out
