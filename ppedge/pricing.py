"""Turn sportsbook quotes into a fair probability at the PrizePicks line."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from . import config
from .devig import american_to_prob, devig_two_way, prob_to_american
from .dists import Dist, fit_ladder, fit_mean, normal_shift
from .names import norm_name, short_key
from .parse_pp import PPProp
from .parse_sgo import PlayerMarket, Quote

SPEC_BY_KEY = {s.key: s for s in config.STATS}


@dataclass
class PricedProp:
    prop: PPProp
    book_line: float  # weighted median of main lines
    n_books: int
    n_quotes: int
    method: str  # "ladder" | "per-book"
    p_over_raw: float  # pure market, no shrinkage (P over | no push)
    p_push: float
    reliability: float
    p_over: float  # after reliability shrinkage
    p_over_normal: float  # original brief's method, for comparison
    line_gap_sd: float
    books: list[str]
    tier: str = "retail"  # "sharp" if priced by sharp books (retail ignored), else "retail"
    implied_over: float = float("nan")  # with vig, at the book line
    implied_under: float = float("nan")
    hold: float = float("nan")  # book margin (overround - 1)
    adjustments: list[tuple[str, float]] = field(default_factory=list)
    assumed: list[tuple[str, float, str]] = field(default_factory=list)  # user's assumed edges

    @property
    def assumed_prob(self) -> float:
        """Probability of the model's side if the user's assumed edges were applied."""
        p = min(max(self.p_adj + sum(d for _, d, _ in self.assumed), 0.01), 0.99)
        return p if self.pick == "Over" else 1.0 - p

    @property
    def edge(self) -> float:
        """Probability above the 6-pick Flex per-leg breakeven (positive = +EV leg)."""
        return self.prob - config.FLAG_THRESHOLD

    @property
    def pick(self) -> str:
        if not self.prop.under_allowed:
            return "Over"  # PrizePicks only offers More on this line
        return "Over" if self.p_adj >= 0.5 else "Under"

    @property
    def p_adj(self) -> float:
        return min(max(self.p_over + sum(d for _, d in self.adjustments), 0.01), 0.99)

    @property
    def prob(self) -> float:
        """Probability of the chosen side (given no push), after adjustments."""
        return self.p_adj if self.pick == "Over" else 1.0 - self.p_adj

    @property
    def fair_prob(self) -> float:
        """Probability of the chosen side before situational adjustments."""
        return self.p_over if self.pick == "Over" else 1.0 - self.p_over

    @property
    def fair_american(self) -> float:
        return prob_to_american(self.fair_prob)


def weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def _book_weight(q: Quote) -> float:
    w = config.BOOK_WEIGHTS.get(q.book.lower().removeprefix("manual:"), config.DEFAULT_BOOK_WEIGHT)
    w *= (0.6 if q.is_alt else 1.0) * q.weight
    if q.book.startswith("manual:"):
        w *= config.MANUAL_WEIGHT_FACTOR
    if q.quality == "line_only":
        w *= config.LINE_ONLY_WEIGHT_FACTOR
    elif q.quality == "one_sided":
        w *= 0.7
    return w


def _quote_prob(q: Quote, method: str) -> float:
    """Fair P(over | no push) for one quote."""
    if q.prob is not None:
        return q.prob
    return devig_two_way(q.over, q.under, method)


def _drop_stale(quotes: list[Quote]) -> list[Quote]:
    stamps = [q.updated for q in quotes if q.updated]
    if not stamps:
        return quotes
    cutoff = max(stamps) - timedelta(minutes=config.STALE_MINUTES)
    return [q for q in quotes if q.updated is None or q.updated >= cutoff]


def find_market(prop: PPProp, full: dict, short: dict) -> PlayerMarket | None:
    """Gather every source's market for this player/stat into one combined market.

    Sources without a team field (Pinnacle) are matched on name alone; the
    short key (first initial + last name) catches Kenny/Kenneth-style variants.
    """
    cands = []
    want_first = norm_name(prop.player).split()[0] if prop.player else ""
    for team in (prop.team, None):
        hit = full.get((norm_name(prop.player), team, prop.spec_key))
        if not hit:
            s = short.get((short_key(prop.player), team, prop.spec_key))
            # initial + last name alone is ambiguous (Aaron vs Andrew Wiggins):
            # first names must share their first 3 letters (Kenny / Kenneth is fine)
            s = [m for m in s or [] if norm_name(m.player).split()[0][:3] == want_first[:3]]
            hit = s if s and len({norm_name(m.player) for m in s}) == 1 else None
        cands += hit or []
    if not cands:
        return None
    spec = SPEC_BY_KEY[prop.spec_key]
    # if a sportsbook lists both combined and plain tackles, keep the preferred stat
    sgo_pref = [m for m in cands if m.sgo_stat in spec.sgo_ids]
    if len({m.sgo_stat for m in sgo_pref}) > 1:
        best = min(spec.sgo_ids.index(m.sgo_stat) for m in sgo_pref)
        cands = [m for m in cands if m.sgo_stat not in spec.sgo_ids or spec.sgo_ids.index(m.sgo_stat) == best]
    quotes = [q for m in cands for q in m.quotes]
    return PlayerMarket(cands[0].player, prop.team, prop.spec_key, cands[0].sgo_stat, quotes)


def fit_dist(market: PlayerMarket, position: str | None,
             devig_method: str = config.DEVIG_METHOD) -> tuple[Dist, bool] | None:
    """Fitted distribution for one player/stat market, plus whether it is sharp-priced."""
    spec = SPEC_BY_KEY[market.spec_key]
    disp0 = config.spec_disp(spec, config.spec_params(spec, position))
    quotes = _drop_stale(market.quotes)
    sharp = [q for q in quotes if q.book.lower().removeprefix("manual:") in config.SHARP_BOOKS]
    factor = config.RETAIL_WITH_SHARP_FACTOR if sharp else 1.0
    pts = []
    for q in quotes:
        is_sharp = q.book.lower().removeprefix("manual:") in config.SHARP_BOOKS
        try:
            pts.append((q.line, _quote_prob(q, devig_method), _book_weight(q) * (1.0 if is_sharp else factor)))
        except (ValueError, ZeroDivisionError):
            continue
    if not pts:
        return None
    if len({p[0] for p in pts}) >= config.LADDER_MIN_LINES:
        return fit_ladder(spec.family, disp0, pts), bool(sharp)
    mus = [fit_mean(spec.family, disp0, l, p).mu for l, p, _ in pts]
    return Dist(spec.family, float(np.average(mus, weights=[w for *_, w in pts])), disp0), bool(sharp)


def price(prop: PPProp, market: PlayerMarket, devig_method: str = config.DEVIG_METHOD,
          cap_over: float | None = None) -> PricedProp | None:
    """cap_over: upper bound on P(over), e.g. P(at least one catch) for receiving yards."""
    spec = SPEC_BY_KEY[prop.spec_key]
    params = config.spec_params(spec, prop.position)
    disp0 = config.spec_disp(spec, params)

    quotes = _drop_stale(market.quotes)
    sharp = [q for q in quotes if q.book.lower().removeprefix("manual:") in config.SHARP_BOOKS]
    tier = "sharp" if sharp else "retail"
    retail_factor = config.RETAIL_WITH_SHARP_FACTOR if sharp else 1.0
    points = []  # (line, p_over_no_push, weight, quote)
    for q in quotes:
        try:
            is_sharp = q.book.lower().removeprefix("manual:") in config.SHARP_BOOKS
            points.append((q.line, _quote_prob(q, devig_method),
                           _book_weight(q) * (1.0 if is_sharp else retail_factor), q))
        except (ValueError, ZeroDivisionError):
            continue
    if not points:
        return None
    # guest-feed Pinnacle can lag: drop it where Kalshi's live mid at the same
    # line disagrees by more than the configured gap
    kal = {l: p for l, p, _, q in points if q.book == "kalshi"}
    points = [pt for pt in points
              if not (pt[3].book == "pinnacle" and pt[0] in kal
                      and abs(pt[1] - kal[pt[0]]) > config.PINNACLE_STALE_GAP)]
    mains = [p for p in points if not p[3].is_alt] or points
    books = sorted({p[3].book.removeprefix("manual:") for p in points})
    distinct_lines = {p[0] for p in points}

    if len(distinct_lines) >= config.LADDER_MIN_LINES:
        dist = fit_ladder(spec.family, disp0, [(l, p, w) for l, p, w, _ in points])
        o, push, u = dist.outcome(prop.line)
        p_raw = o / (o + u)
        method = "ladder"
        sd = dist.sd
    else:
        ests, pushes, ws, sds = [], [], [], []
        for l, p, w, _ in mains:
            d: Dist = fit_mean(spec.family, disp0, l, p)
            o, push_i, u = d.outcome(prop.line)
            ests.append(o / (o + u))
            pushes.append(push_i)
            ws.append(w)
            sds.append(d.sd)
        p_raw = weighted_median(ests, ws)
        push = float(np.average(pushes, weights=ws))
        sd = float(np.average(sds, weights=ws))
        method = "per-book"

    if cap_over is not None and p_raw > cap_over:
        # the gamma has no mass at zero; a player with no catches has 0 yards
        p_raw = cap_over

    nearest = min(distinct_lines, key=lambda l: abs(l - prop.line))
    gap = abs(nearest - prop.line) / max(sd, 1e-6)
    if not min(distinct_lines) <= prop.line <= max(distinct_lines):
        gap *= config.EXTRAPOLATION_PENALTY  # outside every source's lines: a guess
    n_books = len(books)
    reliability = (1.0 / (1.0 + config.LINE_SHIFT_PENALTY * gap * gap)) * (
        n_books / (n_books + config.BOOK_COUNT_PRIOR))
    if tier == "retail":
        reliability *= config.RETAIL_ONLY_RELIABILITY
    p_shrunk = 0.5 + (p_raw - 0.5) * reliability

    # original method: multiplicative devig + fixed-SD normal shift, simple mean
    normal_vals = []
    for l, pq, _, q in mains:
        try:
            pm = pq if q.prob is not None else devig_two_way(q.over, q.under, "multiplicative")
        except ValueError:
            continue
        normal_vals.append(normal_shift(l, pm, prop.line, params["normal_sd"]))
    p_normal = float(np.mean(normal_vals)) if normal_vals else float("nan")

    # vig vs no-vig, from two-sided main quotes: raw implied Over % and the hold
    two = [q for _, _, _, q in mains if q.quality == "two_sided"]
    if two:
        qo = [american_to_prob(q.over) for q in two]
        qu = [american_to_prob(q.under) for q in two]
        implied_over = float(np.mean(qo))
        implied_under = float(np.mean(qu))
        hold = float(np.mean([a + b - 1 for a, b in zip(qo, qu)]))
    else:
        implied_over = implied_under = hold = float("nan")

    return PricedProp(
        prop=prop,
        book_line=weighted_median([p[0] for p in mains], [p[2] for p in mains]),
        n_books=n_books,
        n_quotes=len(points),
        method=method,
        p_over_raw=p_raw,
        p_push=push,
        reliability=reliability,
        p_over=p_shrunk,
        p_over_normal=p_normal,
        line_gap_sd=gap if math.isfinite(gap) else 0.0,
        books=books,
        tier=tier,
        implied_over=implied_over,
        implied_under=implied_under,
        hold=hold,
    )
