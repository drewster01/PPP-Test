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
import sys
from pathlib import Path
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbastats  # noqa: E402
from scipy.stats import nbinom, poisson  # noqa: E402

SPORT = {"nba": "basketball/nba", "nfl": "football/nfl", "cfb": "football/college-football"}
SEASON = {"nba": (2027, 2026), "nfl": (2026, 2025), "cfb": (2026, 2025)}  # ESPN season ids: this season, last season
MIN_MINUTES = 15
THIS_SEASON_WEIGHT = 3.0  # backtest 10/29: 3x beat 1x and 2x on both seasons
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
            g["opp"] = ((log.get("events", {}).get(e["eventId"]) or {}).get("opponent") or {}).get("abbreviation")
            out.append(g)
    return sorted(out, key=lambda g: g["date"])


def _nb_cdf(k: int, mean: float, var: float) -> float:
    """P(X <= k) for a negative binomial with this mean and variance (Poisson if var <= mean)."""
    if mean <= 0 or k < 0:
        return 1.0 if k >= 0 else 0.0
    if var <= mean * 1.02:
        return float(poisson.cdf(k, mean))
    r = mean * mean / (var - mean)
    return float(nbinom.cdf(k, r, r / (r + mean)))


def analyze(league: str, lean: dict) -> dict | None:
    aid = athlete_id(lean.get("photo"))
    if not aid and league != "nba":
        return None
    stat, line, side = lean["stat"], (lean["line"] if lean.get("current_line") is None else lean["current_line"]), lean["side"]
    now = last = None
    if league == "nba":
        logs = [nbastats.game_logs(s) for s in SEASON[league]]
        if all(lg is not None for lg in logs):
            now, last = (lg.get(nbastats.norm(lean["player"]), []) for lg in logs)
            now = [g for g in now if not CUTOFF or g["date"] < CUTOFF]
    if now is None:
        if league == "nba" and stat in ("Offensive Rebounds", "Defensive Rebounds"):
            now, last = (rebound_games(lean["player"], s) for s in SEASON[league])
        else:
            now, last = (games(league, aid, s) for s in SEASON[league])
    return model_from_games(league, lean, now, last)


def model_from_games(league: str, lean: dict, now: list[dict], last: list[dict]) -> dict | None:
    """The model on given game histories (oldest first). Backtests call this with past games only."""
    stat, line, side = lean["stat"], (lean["line"] if lean.get("current_line") is None else lean["current_line"]), lean["side"]
    if league == "nba":
        now = [g for g in now if g["minutes"] >= MIN_MINUTES]
        last = [g for g in last if g["minutes"] >= MIN_MINUTES]
    now = [g for g in now if stat in g]
    last = [g for g in last if stat in g]
    if league in ("nfl", "cfb") and len(now) >= NFL_CURRENT_ONLY:
        last = []  # NFL roles shift week to week: once he has 5 games, this season only (synthesis rule 10)
    n_now = len(now)
    w_now = [0.5 ** ((n_now - 1 - i) / HALF_LIFE_GAMES) * THIS_SEASON_WEIGHT for i in range(n_now)]
    w_last = [0.5 ** ((len(last) - 1 - i) / HALF_LIFE_GAMES) * 0.5 + 0.5 for i in range(len(last))]
    role = (role_change(now, last) or rate_shift(now, last, stat)) if league == "nba" else None
    if role and last and now:
        cap = ROLE_CHANGE_LAST_SHARE / (1 - ROLE_CHANGE_LAST_SHARE) * sum(w_now)
        scale = min(1.0, cap / sum(w_last))
        w_last = [w * scale for w in w_last]
    rows = list(zip(now, w_now)) + list(zip(last, w_last))
    if league == "nba" and SOS_ADJUST:
        rows = [(_sos(g, stat), w) for g, w in rows]
    elif league in ("nfl", "cfb") and SOS_ADJUST:
        rows = _football_sos(league, lean, stat, rows)
    if len(rows) < 2:
        return None
    win = lambda v: v > line if side == "More" else v < line  # noqa: E731
    decided = [(g, w) for g, w in rows if g[stat] != line]
    wins, total = sum(w for g, w in decided if win(g[stat])), sum(w for _, w in decided)
    hit_rate = (wins + PRIOR_GAMES * 0.5) / (total + PRIOR_GAMES)
    if league in ("nfl", "cfb"):
        hit_rate = 0.5 + FOOTBALL_CALIBRATION * (hit_rate - 0.5)
    vals = [g[stat] for g, _ in rows]
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
    out = {"hit_rate": round(hit_rate * 100), "games": len(rows), "games_now": len(now),
           "avg": round(mean, 1), "sd": round(sd, 1)}
    if role:
        out["role_change"] = role
    # count model (NBA): per-minute rate (this season weighted double, shrunk toward the league rate)
    # x tonight's minutes, as a negative binomial averaged over a spread of minutes
    if league == "nba":
        mins = sum(g["minutes"] * w for g, w in rows)
        prior, k_min = LEAGUE_RATE.get(stat), PRIOR_MINUTES_BY_STAT.get(stat, PRIOR_MINUTES)
        if prior is None:
            k_min = 0
        rate = (sum(g[stat] * w for g, w in rows) + (prior or 0) * k_min) / (mins + k_min)
        mk = MAKES.get(stat) if MAKES_FROM_ATTEMPTS else None
        if mk and all(mk[0] in g for g, _ in rows):
            # made shots = attempt rate x shooting %, the % shrunk hard toward the league: a cold or hot
            # start moves attempts a little and makes a lot, and only the attempts are a stable skill
            att_stat, league_pct, k_att = mk
            att_prior = LEAGUE_RATE.get(att_stat, 0)
            att_w = sum(g[att_stat] * w for g, w in rows)
            att_rate = (att_w + att_prior * PRIOR_MINUTES) / (mins + PRIOR_MINUTES)
            pct = (sum(g[stat] * w for g, w in rows) + league_pct * k_att) / (att_w + k_att)
            rate = att_rate * pct
            out["shooting_pct"] = round(pct, 3)
        exp_min = _expected_minutes(lean, now or last)
        # research can move his rate for tonight (a teammate out, a new role): usage_adj 1.10 = 10% more
        mf = matchup_factor(lean.get("opp"), stat)
        lam = rate * exp_min * float(lean.get("usage_adj") or 1.0) * mf
        if mf != 1.0:
            out["matchup"] = round(mf, 3)
        per = [g[stat] / g["minutes"] * exp_min for g, _ in rows]
        disp = max(sum((v - lam) ** 2 for v in per) / max(len(per) - 1, 1) / max(lam, 1e-9), 1.0)
        p_blow = blowout_prob(lean.get("team"))
        out["blowout"] = round(p_blow * 100)
        prob = lambda mean: _side_prob(side, line, mean, exp_min, disp, p_blow)  # noqa: E731
        bad = 1.15 if side == "Less" else 0.85  # one-event stress test: the mean moves 15% against the lean
        cal = lambda q: 0.5 + CALIBRATION * (q - 0.5)  # noqa: E731  (backtest: raw chances run a bit hot)
        out.update(stat_prob=round(100 * cal(prob(lam))), stress_prob=round(100 * cal(prob(lam * bad))),
                   exp_min=round(exp_min), proj=round(lam, 1))
    # volatility: one event decides it when the line is near zero next to the spread
    z = abs(line - mean) / max(sd, 0.5)
    out["volatility"] = "high" if (line <= 2.5 and league == "nba") or z < 0.35 else "medium" if z < 0.75 else "low"
    return out


def _side_prob(side: str, line: float, lam: float, exp_min: float, disp: float, p_blow: float = 0.0) -> float:
    """Chance the lean wins (pushes refunded), averaged over a minutes mixture: a normal game
    (minutes +/- MINUTES_SD) and, with chance p_blow, a blowout where rotation players sit late."""
    total = 0.0
    cut = BLOWOUT_MINUTES_CUT if exp_min >= 24 else 0.0
    # recent minutes already include blowouts: lift normal games so the average stays at exp_min
    # (SPY audit 10/29: without this the blowout cut counted twice and every Under ran ~4 pts hot)
    lift = p_blow * cut / (1 - p_blow) if p_blow < 1 else 0.0
    draws = [(dm + lift, w * (1 - p_blow)) for dm, w in ((-MINUTES_SD, 0.25), (0.0, 0.5), (MINUTES_SD, 0.25))]
    draws.append((-cut, p_blow))
    for dm, w in draws:
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


HALF_LIFE_GAMES = 25
MAKES_FROM_ATTEMPTS = True
# made stat -> (attempt stat, league make %, prior attempts the % is shrunk with)
MAKES = {"3-PT Made": ("3-PT Attempted", 0.36, 300), "FG Made": ("FG Attempted", 0.47, 250),
         "Two Pointers Made": ("Two Pointers Attempted", 0.54, 250), "Free Throws Made": ("Free Throws Attempted", 0.78, 150)}
NFL_CURRENT_ONLY = 8  # TLT backtest 10/30: 8 beat 5 on both seeds and both player halves
CALIBRATION = 0.9  # backtest 10/29: 0.85-0.9 scored best on all three samples
ROLE_CHANGE_LAST_SHARE = 0.40
BLOWOUT_MARGIN = 15
MARGIN_SD = 12.5
BLOWOUT_MINUTES_CUT = 8.0
MATCHUP_REGRESS = 0.5  # opponent factors are noisy: use half of each
MATCHUPS = Path(__file__).resolve().parent.parent / "data" / "nba" / "matchups.json"
# PrizePicks stat -> opponent factor key (factors are keyed by the defending team)
MATCHUP_KEY = {
    "Points": "pts", "Rebounds": "reb", "Assists": "ast", "Steals": "stl", "Blocked Shots": "blk",
    "Turnovers": "tov", "Offensive Rebounds": "oreb", "Defensive Rebounds": "dreb",
    "FG Made": "fga", "FG Attempted": "fga", "3-PT Made": "fg3a", "3-PT Attempted": "fg3a",
    "Two Pointers Made": "fg2a", "Two Pointers Attempted": "fg2a", "Free Throws Made": "fta",
    "Free Throws Attempted": "fta", "Pts+Rebs": ("pts", "reb"), "Pts+Asts": ("pts", "ast"),
    "Rebs+Asts": ("reb", "ast"), "Pts+Rebs+Asts": ("pts", "reb", "ast"), "Blks+Stls": ("blk", "stl"),
}


def _matchups() -> dict:
    if "matchups" not in _cache:
        try:
            _cache["matchups"] = json.loads(MATCHUPS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache["matchups"] = {}
    return _cache["matchups"]


def matchup_factor(opp: str | None, stat: str) -> float:
    team = (_matchups().get("teams") or {}).get(opp or "")
    key = MATCHUP_KEY.get(stat)
    if not team or not key:
        return 1.0
    keys = key if isinstance(key, tuple) else (key,)
    vals = [float(team[k]) for k in keys if isinstance(team.get(k), (int, float))]
    if not vals:
        return 1.0
    return 1 + MATCHUP_REGRESS * (sum(vals) / len(vals) - 1)


def blowout_prob(team: str | None) -> float:
    """P(final margin >= BLOWOUT_MARGIN either way), margin ~ Normal(spread, MARGIN_SD)."""
    alias = {"WAS": "WSH", "WSH": "WAS", "GSW": "GS", "GS": "GSW", "NYK": "NY", "NY": "NYK",
             "NOP": "NO", "NO": "NOP", "SAS": "SA", "SA": "SAS", "UTA": "UTAH", "UTAH": "UTA"}
    for name, g in (_matchups().get("games") or {}).items():
        if team and (team in name.split("@") or alias.get(team) in name.split("@")):
            m = re.search(r"([A-Z]{2,3})\s*([-+]\d+(?:\.\d+)?)", str(g.get("spread", "")))
            if not m:
                return 0.0
            fav_margin = -float(m.group(2))  # "POR -7.5": POR expected to win by 7.5
            cdf = lambda x: 0.5 * (1 + math.erf((x - fav_margin) / (MARGIN_SD * math.sqrt(2))))  # noqa: E731
            return (1 - cdf(BLOWOUT_MARGIN)) + cdf(-BLOWOUT_MARGIN)
    return 0.0


RATE_SHIFT = 0.25  # this season's per-minute rate 25%+ away from last season's
RATE_SHIFT_MIN_GAMES = 8  # fewer games: too noisy (backtest), research sets usage_adj instead


def rate_shift(now: list[dict], last: list[dict], stat: str) -> str | None:
    if len(now) < RATE_SHIFT_MIN_GAMES or not last:
        return None
    r_now = sum(g[stat] for g in now) / max(sum(g["minutes"] for g in now), 1)
    r_last = sum(g[stat] for g in last) / max(sum(g["minutes"] for g in last), 1)
    if r_last > 0 and abs(r_now / r_last - 1) >= RATE_SHIFT:
        return f"{stat} per minute {'up' if r_now > r_last else 'down'} {abs(r_now / r_last - 1):.0%} from last season"
    return None


SOS_ADJUST = True
SOS_REGRESS = 0.5  # use half of each past opponent's defensive factor (team factors are noisy)


def _sos(g: dict, stat: str) -> dict:
    """The game restated against an average defense: his number divided by how much that night's
    opponent usually gives up of this stat (strength of schedule, Popovich 10/30)."""
    season = 2027 if g.get("date", "") >= "2026-10-01" else 2026
    opp = (g.get("matchup") or "").replace("@", "vs.").split("vs.")[-1].strip()
    f = (nbastats.defense_factors(season).get(opp) or {}).get(stat)
    if not f or stat not in g:
        return g
    f = 1 + SOS_REGRESS * (min(max(f, 0.8), 1.25) - 1)
    return {**g, stat: g[stat] / f}


# PrizePicks team abbreviation -> ESPN's, where they differ (college)
PP_TO_ESPN = {"SCAR": "SC", "IND": "IU", "MIZZ": "MIZ", "BSU": "BOIS", "UGA": "UGA", "OKLA": "OU",
              "WASH": "WASH", "FRES": "FRES", "TXST": "TXST", "CHAR": "CLT", "KENT": "KENT", "SAC": "SAC",
              "KU": "KU", "NW": "NU", "UVA": "UVA", "WAKE": "WAKE", "MINN": "MINN", "RUTG": "RUTG", "MICH": "MICH"}


# TLT backtest 10/30: schedule adjustment helped pass stats on both seeds and hurt rush stats on both
FOOTBALL_SOS_FAMILIES = {"pass"}
FOOTBALL_CALIBRATION = 0.85  # football chances above 75% hit about 71%: pull them toward 50%


def _football_sos(league: str, lean: dict, stat: str, rows: list) -> list:
    """Restate each past game against an average defense, then apply tonight's defense (half strength)."""
    import defense
    fam = defense.FAMILY.get(stat)
    if fam not in FOOTBALL_SOS_FAMILIES:
        return rows
    table = defense.factors(league) if fam else {}
    if not table:
        return rows
    reg = lambda f: 1 + SOS_REGRESS * (min(max(f, 0.6), 1.6) - 1)  # noqa: E731
    opp = lean.get("opp") or ""
    alias = PP_TO_ESPN if league == "cfb" else {}  # NFL abbreviations already match ESPN (IND = Colts)
    tonight = (table.get(alias.get(opp, opp)) or {}).get(fam)
    t = reg(tonight) if tonight else 1.0
    out = []
    for g, w in rows:
        f = (table.get(g.get("opp") or "") or {}).get(fam)
        if f and stat in g:
            g = {**g, stat: g[stat] / reg(f) * t}
        elif stat in g and t != 1.0:
            g = {**g, stat: g[stat] * t}
        out.append((g, w))
    return out


def role_change(now: list[dict], last: list[dict]) -> str | None:
    """Why last season's numbers may not describe him now: new team or minutes moved 6+."""
    if not now or not last:
        return None
    if now[-1].get("team") and last[-1].get("team") and now[-1]["team"] != last[-1]["team"]:
        return f"new team ({last[-1]['team']} to {now[-1]['team']})"
    m_now = sum(g["minutes"] for g in now) / len(now)
    m_last = sum(g["minutes"] for g in last) / len(last)
    if abs(m_now - m_last) >= 6:
        return f"minutes {m_last:.0f} last season, {m_now:.0f} now"
    return None


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
