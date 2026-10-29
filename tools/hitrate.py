"""How often a lean would have hit, from the player's own game logs (ESPN), and a count-model chance.

Low lines (0.5 steals, 1.5 free throws, a 5.5-yard longest rush) swing on one event, so a research
estimate alone is not enough. For each lean this adds:

- hit_rate: share of his games this season and last season that beat the line on the lean's side
  (this season counts double; NBA games under 15 minutes are skipped as not comparable),
- stat_prob (NBA counting stats): the chance from a negative binomial built from his per-minute rate
  x expected minutes, with the spread measured from his own games,
- volatility: how much one event moves the result (high when the line is within one standard deviation
  of zero, or the stat's spread is large next to the line).
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request

SPORT = {"nba": "basketball/nba", "nfl": "football/nfl"}
SEASON = {"nba": (2027, 2026), "nfl": (2026, 2025)}  # ESPN season ids: this season, last season
MIN_MINUTES = 15
THIS_SEASON_WEIGHT = 2.0
# Small samples: a player's per-minute rate is shrunk toward the league rate as if he had also played
# PRIOR_MINUTES at it, and his hit rate toward 50% as if he had PRIOR_GAMES more games split evenly.
PRIOR_MINUTES = 120  # volume stats (shots, points, rebounds) are stable per player: light prior
PRIOR_GAMES = 6
# Stats that swing on one event need a heavier prior (about 15-20 games); free throws about 10
PRIOR_MINUTES_BY_STAT = {"Steals": 480, "Blocked Shots": 480, "Turnovers": 480, "Blks+Stls": 480,
                         "Personal Fouls": 400, "Free Throws Attempted": 300, "Free Throws Made": 300}
MINUTES_SD = 5.0  # tonight's minutes are uncertain by about this much
# NBA league-average rates per minute played (rotation players), for the shrinkage above
LEAGUE_RATE = {
    "Points": 0.46, "Rebounds": 0.18, "Assists": 0.11, "Steals": 0.028, "Blocked Shots": 0.017,
    "Turnovers": 0.054, "Personal Fouls": 0.072, "FG Made": 0.17, "FG Attempted": 0.37,
    "3-PT Made": 0.054, "3-PT Attempted": 0.15, "Two Pointers Made": 0.12, "Two Pointers Attempted": 0.22,
    "Free Throws Made": 0.066, "Free Throws Attempted": 0.085, "Pts+Rebs": 0.64, "Pts+Asts": 0.57,
    "Rebs+Asts": 0.29, "Pts+Rebs+Asts": 0.75, "Blks+Stls": 0.045,
    "Offensive Rebounds": 0.042, "Defensive Rebounds": 0.14,
}

_cache: dict = {}
CUTOFF: str | None = None  # ISO date (UTC); when set, games on or after it are ignored (backtests)


def _get(url: str) -> dict:
    if url not in _cache:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        _cache[url] = json.load(urllib.request.urlopen(req, timeout=30))
    return _cache[url]


def athlete_id(photo: str | None) -> str | None:
    m = re.search(r"/players/full/(\d+)\.png", photo or "")
    return m.group(1) if m else None


def _nba_game(row: dict) -> dict:
    split = lambda v: tuple(int(x) for x in v.split("-"))  # noqa: E731
    fgm, fga = split(row["FG"])
    tpm, tpa = split(row["3PT"])
    ftm, fta = split(row["FT"])
    n = lambda k: float(row.get(k) or 0)  # noqa: E731
    pts, reb, ast = n("PTS"), n("REB"), n("AST")
    return {"minutes": n("MIN"), "Points": pts, "Rebounds": reb, "Assists": ast, "Steals": n("STL"),
            "Blocked Shots": n("BLK"), "Turnovers": n("TO"), "Personal Fouls": n("PF"),
            "FG Made": fgm, "FG Attempted": fga, "3-PT Made": tpm, "3-PT Attempted": tpa,
            "Two Pointers Made": fgm - tpm, "Two Pointers Attempted": fga - tpa,
            "Free Throws Made": ftm, "Free Throws Attempted": fta,
            "Pts+Rebs": pts + reb, "Pts+Asts": pts + ast, "Rebs+Asts": reb + ast,
            "Pts+Rebs+Asts": pts + reb + ast, "Blks+Stls": n("BLK") + n("STL")}


NFL_NAMES = {
    "passingAttempts": "Pass Attempts", "completions": "Pass Completions", "passingYards": "Pass Yards",
    "passingTouchdowns": "Pass TDs", "interceptions": "INT", "longPassing": "Longest Completion",
    "rushingAttempts": "Rush Attempts", "rushingYards": "Rush Yards", "longRushing": "Longest Rush",
    "receptions": "Receptions", "receivingTargets": "Rec Targets", "receivingYards": "Receiving Yards",
    "longReception": "Longest Reception", "fieldGoalsMade": "FG Made", "totalKickingPoints": "Kicking Points",
    "totalTackles": "Tackles+Ast", "sacks": "Sacks",
}


def _nfl_game(names: list[str], stats: list[str]) -> dict:
    out = {}
    for k, v in zip(names, stats):
        try:
            out[NFL_NAMES.get(k, k)] = float(str(v).split("/")[0].replace(",", ""))
        except ValueError:
            pass
    if "Pass Completions" in out and out.get("Pass Attempts"):
        out["Completion Percentage"] = 100 * out["Pass Completions"] / out["Pass Attempts"]
    out["Rush+Rec Yds"] = out.get("Rush Yards", 0) + out.get("Receiving Yards", 0)
    out["Pass+Rush Yds"] = out.get("Pass Yards", 0) + out.get("Rush Yards", 0)
    return out


BREF = "https://www.basketball-reference.com"
_last_bref = [0.0]


def _bref(path: str) -> str:
    """Basketball-Reference page (polite: one request every 3.5 s, as the site asks)."""
    import time
    url = BREF + path
    if url not in _cache:
        wait = 3.5 - (time.time() - _last_bref[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        _cache[url] = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        _last_bref[0] = time.time()
    return _cache[url]


def bref_slug(name: str) -> str | None:
    page = _bref("/search/search.fcgi?search=" + urllib.parse.quote(name))
    m = re.search(r'/players/[a-z]/([a-z]+\d\d)\.html', page)
    return m.group(1) if m else None


def rebound_games(name: str, season: int) -> list[dict]:
    """Offensive / defensive rebounds per game from Basketball-Reference (ESPN logs don't split them)."""
    try:
        slug = bref_slug(name)
        if not slug:
            return []
        page = _bref(f"/players/{slug[0]}/{slug}/gamelog/{season}")
    except OSError:
        return []
    out = []
    for row in re.findall(r'<tr id="player_game_log_reg[^"]*"[^>]*>(.*?)</tr>', page, re.S):
        cell = dict(re.findall(r'data-stat="([a-z_]+)"[^>]*>(?:<a[^>]*>)?([^<]*)', row))
        if not cell.get("mp"):
            continue  # did not play
        mm, _, ss = cell["mp"].partition(":")
        date = cell.get("date", "")
        if CUTOFF and date and date >= CUTOFF[:10]:
            continue
        out.append({"date": date, "minutes": int(mm) + (int(ss) / 60 if ss else 0),
                    "Offensive Rebounds": float(cell.get("orb") or 0), "Defensive Rebounds": float(cell.get("drb") or 0)})
    return out


def games(league: str, aid: str, season: int) -> list[dict]:
    """Regular-season games for one season, oldest first."""
    try:
        log = _get(f"https://site.web.api.espn.com/apis/common/v3/sports/{SPORT[league]}/athletes/{aid}/gamelog?season={season}")
    except (OSError, ValueError):
        return []
    reg = next((st for st in log.get("seasonTypes", []) if "Regular" in st.get("displayName", "")), None)
    if not reg:
        return []
    out = []
    for c in reg.get("categories", []):
        for e in c.get("events", []):
            date = (log.get("events", {}).get(e["eventId"]) or {}).get("gameDate", "")
            if CUTOFF and date >= CUTOFF:
                continue  # backtests: only games played before the lean's date
            g = _nba_game(dict(zip(log["labels"], e["stats"]))) if league == "nba" else _nfl_game(log.get("names", []), e["stats"])
            g["date"] = date
            out.append(g)
    return sorted(out, key=lambda g: g["date"])


def _nb_cdf(k: int, mean: float, var: float) -> float:
    """P(X <= k) for a negative binomial with this mean and variance (Poisson if var <= mean)."""
    if mean <= 0:
        return 1.0
    if var <= mean * 1.02:
        return sum(math.exp(-mean) * mean ** i / math.factorial(i) for i in range(k + 1))
    r = mean * mean / (var - mean)
    p = r / (r + mean)
    return sum(math.exp(math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) + r * math.log(p) + i * math.log(1 - p))
               for i in range(k + 1))


def analyze(league: str, lean: dict) -> dict | None:
    aid = athlete_id(lean.get("photo"))
    if not aid:
        return None
    stat, line, side = lean["stat"], lean.get("current_line", lean["line"]), lean["side"]
    if league == "nba" and stat in ("Offensive Rebounds", "Defensive Rebounds"):
        now, last = (rebound_games(lean["player"], s) for s in SEASON[league])
    else:
        now, last = (games(league, aid, s) for s in SEASON[league])
    if league == "nba":
        now = [g for g in now if g["minutes"] >= MIN_MINUTES]
        last = [g for g in last if g["minutes"] >= MIN_MINUTES]
    rows = [(g, THIS_SEASON_WEIGHT) for g in now if stat in g] + [(g, 1.0) for g in last if stat in g]
    if len(rows) < 2:
        return None
    win = lambda v: v > line if side == "More" else v < line  # noqa: E731
    decided = [(g, w) for g, w in rows if g[stat] != line]
    wins, total = sum(w for g, w in decided if win(g[stat])), sum(w for _, w in decided)
    hit_rate = (wins + PRIOR_GAMES * 0.5) / (total + PRIOR_GAMES)
    vals = [g[stat] for g, _ in rows]
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
    out = {"hit_rate": round(hit_rate * 100), "games": len(rows), "games_now": len(now),
           "avg": round(mean, 1), "sd": round(sd, 1)}
    # count model (NBA): per-minute rate (this season weighted double, shrunk toward the league rate)
    # x tonight's minutes, as a negative binomial averaged over a spread of minutes
    if league == "nba":
        mins = sum(g["minutes"] * w for g, w in rows)
        prior, k_min = LEAGUE_RATE.get(stat), PRIOR_MINUTES_BY_STAT.get(stat, PRIOR_MINUTES)
        if prior is None:
            k_min = 0
        rate = (sum(g[stat] * w for g, w in rows) + (prior or 0) * k_min) / (mins + k_min)
        exp_min = _expected_minutes(lean, now or last)
        # research can move his rate for tonight (a teammate out, a new role): usage_adj 1.10 = 10% more
        lam = rate * exp_min * float(lean.get("usage_adj") or 1.0)
        per = [g[stat] / g["minutes"] * exp_min for g, _ in rows]
        disp = max(sum((v - lam) ** 2 for v in per) / max(len(per) - 1, 1) / max(lam, 1e-9), 1.0)
        prob = lambda mean: _side_prob(side, line, mean, exp_min, disp)  # noqa: E731
        bad = 1.15 if side == "Less" else 0.85  # one-event stress test: the mean moves 15% against the lean
        out.update(stat_prob=round(100 * prob(lam)), stress_prob=round(100 * prob(lam * bad)),
                   exp_min=round(exp_min), proj=round(lam, 1))
    # volatility: one event decides it when the line is near zero next to the spread
    z = abs(line - mean) / max(sd, 0.5)
    out["volatility"] = "high" if (line <= 2.5 and league == "nba") or z < 0.35 else "medium" if z < 0.75 else "low"
    return out


def _side_prob(side: str, line: float, lam: float, exp_min: float, disp: float) -> float:
    """Chance the lean wins (pushes refunded), averaged over minutes exp_min +/- MINUTES_SD."""
    total = 0.0
    for dm, w in ((-MINUTES_SD, 0.25), (0.0, 0.5), (MINUTES_SD, 0.25)):
        mean = max(lam * (exp_min + dm) / exp_min, 0.01)
        var = mean * disp
        if float(line).is_integer():
            below, at = _nb_cdf(int(line) - 1, mean, var), _nb_cdf(int(line), mean, var)
            p_less, p_more, push = below, 1 - at, at - below
            p_less, p_more = p_less / (1 - push), p_more / (1 - push)
        else:
            p_less = _nb_cdf(math.floor(line), mean, var)
            p_more = 1 - p_less
        total += w * (p_more if side == "More" else p_less)
    return total


def _expected_minutes(lean: dict, recent: list[dict]) -> float:
    em = lean.get("expected_minutes")
    try:
        if em:
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(em))]
            if nums:
                return sum(nums[:2]) / len(nums[:2])
    except ValueError:
        pass
    last5 = [g["minutes"] for g in recent[-5:]]
    return sum(last5) / len(last5) if last5 else 28.0
