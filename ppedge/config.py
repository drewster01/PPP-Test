"""Static configuration: endpoints, stat mappings, distribution priors,
bookmaker weights and PrizePicks payout tables.

Everything a user might reasonably want to tune lives here (or in
adjustments.json) so the pricing code stays free of magic numbers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Which league this process prices. Set PPEDGE_LEAGUE=nba (props_edge.py and
# live.py do this from --league) before importing anything from ppedge.
LEAGUE = os.environ.get("PPEDGE_LEAGUE", "nfl").lower()

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
SGO_EVENTS_URL = "https://api.sportsgameodds.com/v2/events"
SGO_PARAMS = {
    "leagueID": "NFL",
    "oddsAvailable": "true",
    "includeAltLines": "true",
    "limit": "10",
}
PP_URL = "https://partner-api.prizepicks.com/projections"
PP_PARAMS = {"league_id": {"nfl": "9", "nba": "7", "cfb": "15"}[LEAGUE], "per_page": "500", "single_stat": "true"}
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)
SGO_MIN_SECONDS_BETWEEN_REQUESTS = 6.5  # free tier: 10 req/min
SGO_MAX_PAGES = 3  # hard cap on requests per run
CACHE_MAX_AGE_MIN = 10  # reuse raw files younger than this unless --refresh

# --------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------
# Whole slate: every team on today's PrizePicks board (both leagues)
GAME_TEAMS = None
# CFB plays mostly on Saturday: its slate covers today and tomorrow
SLATE_DAYS = 2 if LEAGUE == "cfb" else 1
TEAM_ALIASES = {}
GAME_LABEL = f"{LEAGUE.upper()} slate"

# --------------------------------------------------------------------------
# Stat specifications
# --------------------------------------------------------------------------
# family:
#   "gamma"  continuous, right skewed, sd scales with mean (sd = cv * mean)
#   "negbin" counts, var = mu + mu^2 / k   (k large -> Poisson)
# normal_sd: the fixed SD from the original brief, used only for the
#   comparison column so the old and new models can be compared side by side.


@dataclass(frozen=True)
class StatSpec:
    key: str
    pp_names: tuple[str, ...]
    sgo_ids: tuple[str, ...]
    family: str
    cv: float = 0.0  # gamma
    k: float = 0.0  # negbin
    normal_sd: float = 0.0
    # position-specific overrides: {"QB": {"cv": 0.85}}
    by_position: dict = field(default_factory=dict)
    low_confidence: bool = False


STATS: tuple[StatSpec, ...] = (
    StatSpec("pass_yds", ("Pass Yards", "Passing Yards"), ("passing_yards",),
             "gamma", cv=0.26, normal_sd=58),
    StatSpec("pass_att", ("Pass Attempts", "Passing Attempts"), ("passing_attempts",),
             "negbin", k=80, normal_sd=6.0),
    # receptions/completions/rush attempts are often under-dispersed (variance <
    # mean); a binomial handles that, a negative binomial cannot
    # (Florez, Guindani & Vannucci 2024, JQAS). k here is the binomial trial count.
    StatSpec("pass_comp", ("Pass Completions", "Completions"), ("passing_completions",),
             "binom", k=70, normal_sd=4.5),
    StatSpec("pass_td", ("Pass TDs", "Passing TDs", "Pass Touchdowns"), ("passing_touchdowns",),
             "negbin", k=1000, normal_sd=0.9),
    StatSpec("pass_int", ("INT", "Pass INTs", "Interceptions", "Interceptions Thrown"),
             ("passing_interceptions",), "negbin", k=1000, normal_sd=0.8),
    StatSpec("long_comp", ("Longest Completion", "Longest Pass"), ("passing_longestCompletion",),
             "gamma", cv=0.35, normal_sd=12, low_confidence=True),
    StatSpec("rush_yds", ("Rush Yards", "Rushing Yards"), ("rushing_yards",),
             "gamma", cv=0.55, normal_sd=22, by_position={"QB": {"cv": 0.85, "normal_sd": 20}}),
    StatSpec("rush_att", ("Rush Attempts", "Rushing Attempts"), ("rushing_attempts",),
             "binom", k=80, normal_sd=3.5),
    StatSpec("long_rush", ("Longest Rush",), ("rushing_longestRush",),
             "gamma", cv=0.7, normal_sd=8, low_confidence=True),
    StatSpec("rec_yds", ("Receiving Yards", "Rec Yards"), ("receiving_yards",),
             "gamma", cv=0.65, normal_sd=26),
    StatSpec("receptions", ("Receptions",), ("receiving_receptions",),
             "binom", k=15, normal_sd=1.8),
    StatSpec("targets", ("Rec Targets", "Targets"), ("receiving_targets",),
             "negbin", k=25, normal_sd=2.2),
    StatSpec("long_rec", ("Longest Reception",), ("receiving_longestReception",),
             "gamma", cv=0.6, normal_sd=9, low_confidence=True),
    StatSpec("rush_rec_yds", ("Rush+Rec Yds", "Rush + Rec Yds", "Rush+Rec Yards"),
             ("rushing+receiving_yards",), "gamma", cv=0.45, normal_sd=28),
    StatSpec("pass_rush_yds", ("Pass+Rush Yds", "Pass + Rush Yds", "Pass+Rush Yards"),
             ("passing+rushing_yards",), "gamma", cv=0.25, normal_sd=60),
    StatSpec("rush_td", ("Rush TDs", "Rushing TDs"), ("rushing_touchdowns",),
             "negbin", k=1000, normal_sd=0.6),
    StatSpec("rec_td", ("Rec TDs", "Receiving TDs"), ("receiving_touchdowns",),
             "negbin", k=1000, normal_sd=0.5),
    StatSpec("tds", ("Rush+Rec TDs", "Touchdowns", "Anytime TDs"), ("touchdowns",),
             "negbin", k=1000, normal_sd=0.6),
    StatSpec("sacks_taken", ("Sacks Taken",), ("passing_sacksTaken",),
             "negbin", k=1000, normal_sd=1.4),
    StatSpec("punts", ("Punts",), ("punting_numPunts",), "negbin", k=60, normal_sd=1.5,
             low_confidence=True),
    StatSpec("punts_in20", ("Punts Inside 20",), ("punting_puntsInside20",),
             "negbin", k=1000, normal_sd=1.0, low_confidence=True),
    StatSpec("kick_pts", ("Kicking Points",), ("kicking_totalPoints",),
             "negbin", k=30, normal_sd=3.0, low_confidence=True),
    StatSpec("fg_made", ("FG Made", "Field Goals Made"), ("fieldGoals_made",),
             "negbin", k=1000, normal_sd=1.0),
    StatSpec("tackles", ("Tackles+Ast", "Tackles + Assists", "Tackles"),
             ("defense_combinedTackles", "defense_tackles"), "negbin", k=20, normal_sd=2.2),
    StatSpec("sacks", ("Sacks",), ("defense_sacks",), "negbin", k=1000, normal_sd=0.5,
             low_confidence=True),
)
# Deliberately NOT matched: PrizePicks "Fantasy Score" uses PrizePicks' own
# scoring, which differs from sportsbook fantasy-score props.

# NBA: game-level counts are over-dispersed (var > mean), so negative binomial.
# k chosen so typical starter lines match observed spread, e.g. 20 pts -> sd ~7.
# sgo_ids here are just internal ids (Kalshi/Pinnacle map onto spec keys).
NBA_STATS: tuple[StatSpec, ...] = (
    StatSpec("pts", ("Points",), ("nba_points",), "negbin", k=12, normal_sd=6.5),
    StatSpec("reb", ("Rebounds",), ("nba_rebounds",), "negbin", k=20, normal_sd=2.8),
    StatSpec("ast", ("Assists",), ("nba_assists",), "negbin", k=20, normal_sd=2.4),
    StatSpec("fg3m", ("3-PT Made", "3-Pointers Made"), ("nba_threes",), "negbin", k=40, normal_sd=1.4),
    StatSpec("pra", ("Pts+Rebs+Asts",), ("nba_pra",), "negbin", k=20, normal_sd=8.5),
    StatSpec("stl", ("Steals",), ("nba_steals",), "negbin", k=1000, normal_sd=1.0),
    StatSpec("blk", ("Blocked Shots", "Blocks"), ("nba_blocks",), "negbin", k=1000, normal_sd=0.9),
)
if LEAGUE == "nba":
    STATS = NBA_STATS


def norm_stat(s: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", s.lower())


PP_STAT_INDEX = {norm_stat(n): spec for spec in STATS for n in spec.pp_names}
SGO_STAT_INDEX = {sid: spec for spec in STATS for sid in spec.sgo_ids}


def spec_disp(spec: StatSpec, params: dict) -> float:
    return params["cv"] if spec.family == "gamma" else params["k"]


def spec_params(spec: StatSpec, position: str | None) -> dict:
    p = {"cv": spec.cv, "k": spec.k, "normal_sd": spec.normal_sd}
    if position and position.upper() in spec.by_position:
        p.update(spec.by_position[position.upper()])
    return p


# --------------------------------------------------------------------------
# Bookmakers
# --------------------------------------------------------------------------
# Relative weight in the consensus. Sharper, higher-limit books get more say.
BOOK_WEIGHTS = {
    "pinnacle": 3.0, "circa": 3.0, "bookmakereu": 2.5, "bookmaker": 2.5,
    "betonline": 2.0, "betcris": 2.0,
    "fanduel": 1.6, "draftkings": 1.5, "bet365": 1.3, "caesars": 1.1,
    "betmgm": 1.0, "espnbet": 1.0, "fanatics": 1.0, "hardrockbet": 1.0,
    "bovada": 0.9, "betrivers": 0.8, "unibet": 0.8, "williamhill": 1.0,
}
DEFAULT_BOOK_WEIGHT = 0.8
# Pick'em / DFS operators are the thing being evaluated, not the consensus.
# Only PrizePicks (the board we pick from) is excluded from the consensus.
# Other pick'em apps price both sides, so they are useful inputs.
EXCLUDED_BOOKS = {"prizepicks"}
# Prediction markets are real two-sided markets, so they count (from chat notes).
BOOK_WEIGHTS.update({"kalshi": 1.6, "polymarket": 1.0, "smarkets": 1.3,
                     "pointsbet": 0.8, "sleeper": 0.6, "dabble": 0.5})
# Retail and pick'em prices are blended in at this fraction of their weight when
# sharp prices exist (sharp anchor + soft consensus; Kaunitz et al. 2017).
RETAIL_WITH_SHARP_FACTOR = 0.35
# Pinnacle keeps full weight. Its logged-out guest feed can lag, so a Pinnacle
# price is dropped only where it disagrees with Kalshi's live mid by more than
# PINNACLE_STALE_GAP at the same line.
PINNACLE_STALE_GAP = 0.08
# Quotes hand-entered from articles (manual_lines.json) are older and noisier.
MANUAL_WEIGHT_FACTOR = 0.6
LINE_ONLY_WEIGHT_FACTOR = 0.4  # line published without prices: assume 50/50
ONE_SIDED_HOLD = 0.045  # assumed total margin when only one side's price is known
STALE_MINUTES = 45  # drop a book's price if older than this vs. the freshest

# Sharp tier: market makers and zero/low-vig exchanges. When any sharp price
# exists for a prop, retail books are ignored for that prop entirely.
SHARP_BOOKS = {
    "pinnacle", "circa", "bookmaker", "bookmakereu", "betonline", "betcris",
    "prophetx", "sporttrade", "kalshi", "polymarket", "smarkets",
}
RETAIL_ONLY_RELIABILITY = 0.6  # extra shrink toward 50% when only retail books price a prop
LINEUPS_SHARP_ONLY = True  # top picks and lineups use only sharp-priced props

# --------------------------------------------------------------------------
# Pricing controls
# --------------------------------------------------------------------------
DEVIG_METHOD = "power"  # multiplicative | additive | power | shin | logit
# +EV discipline: evaluate each pick at the least favorable fair price among
# these methods, so an edge only counts if it survives every reasonable de-vig.
WORST_CASE_DEVIG = True
DEVIG_CANDIDATES = ("multiplicative", "shin")
# Shrink toward 50% when the PrizePicks line differs from the book line:
#   reliability = 1 / (1 + LINE_SHIFT_PENALTY * (distance_in_sds)^2)
LINE_SHIFT_PENALTY = 1.5
EXTRAPOLATION_PENALTY = 2.0  # gap multiplier when the PP line is outside all source lines
# Yardage stats that are 0 when the player records no catch; capped at P(1+ catch).
CATCH_CAPPED = {"rec_yds", "long_rec"}
# Shrink toward 50% when few books price the prop:
#   reliability *= n / (n + BOOK_COUNT_PRIOR)
BOOK_COUNT_PRIOR = 1.0
# Fit a two-parameter (mean + dispersion) distribution when at least this many
# distinct lines exist for one prop (alt lines or books disagreeing on lines).
LADDER_MIN_LINES = 3

# --------------------------------------------------------------------------
# PrizePicks payouts (verify in the app; PrizePicks changes these)
# --------------------------------------------------------------------------
POWER_PAYOUTS = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 37.5}
FLEX_PAYOUTS = {
    3: {3: 3.0, 2: 1.0},
    4: {4: 6.0, 3: 1.5},
    5: {5: 10.0, 4: 2.0, 3: 0.4},
    6: {6: 25.0, 5: 2.0, 4: 0.4},
}
FLAG_THRESHOLD = 0.542  # 6-pick Flex per-leg breakeven
# PrizePicks flags many lines "adjusted_odds": they pay a reduced multiplier.
# Calibrated from the app: a 6-pick Flex with 4 adjusted legs paid 15x instead
# of 25x  ->  (15/25)^(1/4) = 0.88 per adjusted leg. Applied to every payout tier.
ADJUSTED_LEG_FACTOR = (15 / 25) ** (1 / 4)
POWER2_THRESHOLD = 0.577

# --------------------------------------------------------------------------
# Adjustments
# --------------------------------------------------------------------------
# Pure market mode: odds come only from the books; adjustments.json is ignored
# (no narratives, no research leans). Set True to bring them back.
USE_ADJUSTMENTS = False
# Price only from these sources (None = use every source). User choice: Kalshi + Pinnacle.
ONLY_SOURCES = {"kalshi", "pinnacle"}
# Stat types left off every board (normalized names). User choice: no Fantasy Score.
EXCLUDE_STATS = {"fantasyscore"}
ADJ_CAP_EACH = 0.03  # max +/- 3 percentage points per adjustment
ADJ_CAP_TOTAL = 0.05  # max +/- 5 percentage points per prop
# Public situational info is largely priced (Sapra 2008; Woodland & Woodland
# 2001), so situational leans are blended at w = 0.35 vs. the market
# (forecast-combination weight, Clemen 1989; Hubacek et al. 2019).
# Entries marked "user_edge" are the user's deliberate leans and apply in full.
ADJ_SITUATIONAL_WEIGHT = 0.35
# "Strong edge" requires clearing breakeven by a margin, not just touching it.
EDGE_MARGIN = 0.02
# Alert when the books' fair probability moves this much between refreshes
# while the app's line stays put (stale line after news: the top pro edge).
MOVE_ALERT = 0.03

# --------------------------------------------------------------------------
# Bet sizing (Baker & McHale 2013: shrink Kelly for estimation error)
# --------------------------------------------------------------------------
KELLY_FRACTION = 0.25
MAX_EXPOSURE_PER_ENTRY = 0.05  # of bankroll

# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
SIM_TRIALS = 200_000
SIM_SEED = 20261008
LINEUP_CANDIDATES = 16
LINEUP_SIZE = 6
MAX_LEG_CORR = 0.25  # reject a lineup if any two legs are this positively linked
# Diversified slips: any two slips share at most this many players, and no
# player appears in more than this share of all slips.
PORTFOLIO_MAX_SHARED = 2
PORTFOLIO_MAX_EXPOSURE = 0.5

# --------------------------------------------------------------------------
# Live board
# --------------------------------------------------------------------------
BANKROLL = 98.00  # user, 2026-10-29
UNIT = 10.00  # user's standard entry size
# refresh until the end of today's slate (US Eastern); overwritten at runtime
# with the last start time once the board is parsed
from datetime import datetime as _dt
from zoneinfo import ZoneInfo as _Zi
KICKOFF_ISO = _dt.now(_Zi("America/New_York")).replace(hour=23, minute=59, second=0, microsecond=0).isoformat()
FIRST_START: str | None = None  # multi-game slates: earliest upcoming start (set at runtime)
LIVE_REFRESH_MIN = 10  # matches the free tier's ~10 minute refresh
BOARD_PORT = 8766  # one server for both leagues

if LEAGUE == "nba":
    CATCH_CAPPED = set()
