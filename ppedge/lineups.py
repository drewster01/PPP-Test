"""Monte Carlo lineup scoring and selection under PrizePicks payout rules."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import config
from .correlate import matrix, pair_rho, sign
from .names import norm_name
from .pricing import PricedProp

HIT, PUSH, MISS = 1, 2, 0


def _last(pp: PricedProp) -> str:
    return norm_name(pp.prop.player).split()[-1]


def banned_pair(a: PricedProp, b: PricedProp) -> str | None:
    """User's explicit do-not-combine rules (from the brief), plus same player."""
    if norm_name(a.prop.player) == norm_name(b.prop.player):
        return "same player"
    for x, y in ((a, b), (b, a)):
        kx, ky = x.prop.spec_key, y.prop.spec_key
        px, py = (x.prop.position or "").upper(), (y.prop.position or "").upper()
        # QB passing Over with his own WR/TE receiving Over
        if (px == "QB" and kx in {"pass_yds", "pass_comp", "pass_att", "pass_td", "pass_rush_yds"}
                and x.pick == "Over" and y.prop.team == x.prop.team and py in {"WR", "TE"}
                and ky in {"rec_yds", "receptions", "targets", "long_rec"} and y.pick == "Over"):
            return "QB passing Over + own receiver Over"
        lx, ly = _last(x), _last(y)
        rec = {"rec_yds", "receptions", "targets", "long_rec"}
        if lx == "godwin" and x.pick == "Over" and kx in rec and ky in rec:
            if (ly == "egbuka" and y.pick == "Under") or (ly == "otton" and y.pick == "Over"):
                return "Godwin Over with Egbuka Under / Otton Over"
        if lx == "williams" and kx == "rush_yds" and x.pick == "Under":
            if ly == "goodson" and y.pick == "Over":
                return "Williams rush Under + Goodson Over"
        if lx == "gainwell" and kx == "receptions" and x.pick == "Over":
            if ly == "irving" and ky == "receptions" and y.pick == "Under":
                return "Gainwell rec Over + Irving rec Under"
    return None


def simulate(legs: list[PricedProp], trials: int, seed: int = config.SIM_SEED,
             correlated: bool = True) -> np.ndarray:
    """Return a (trials, n_legs) int8 matrix of HIT/PUSH/MISS outcomes."""
    rng = np.random.default_rng(seed)
    c = matrix(legs) if correlated else np.eye(len(legs))
    z = rng.multivariate_normal(np.zeros(len(legs)), c, size=trials, method="cholesky")
    u = stats.norm.cdf(z)
    p_hit = np.array([pp.prob * (1 - pp.p_push) for pp in legs])
    p_push = np.array([pp.p_push for pp in legs])
    out = np.full(u.shape, MISS, dtype=np.int8)
    out[u < p_hit + p_push] = PUSH
    out[u < p_hit] = HIT
    return out


def payout_table(n: int, fmt: str) -> np.ndarray:
    """table[n_effective, hits] -> payout multiple. Pushes shrink the lineup."""
    t = np.zeros((n + 1, n + 1))
    for m in range(n + 1):
        if m < 2:
            t[m, :] = 1.0  # voided below the minimum size: refund
        elif fmt == "power" or m == 2:
            t[m, m] = config.POWER_PAYOUTS.get(m, 0.0)
        else:
            for h, mult in config.FLEX_PAYOUTS.get(m, {}).items():
                t[m, h] = mult
    return t


def payout(hits: np.ndarray, pushes: np.ndarray, n: int, fmt: str) -> np.ndarray:
    return payout_table(n, fmt)[n - pushes, hits]


def leg_value(pp: PricedProp) -> float:
    """Hit probability discounted by PrizePicks' reduced payout on adjusted lines."""
    return pp.prob * (config.ADJUSTED_LEG_FACTOR if pp.prop.adjusted else 1.0)


def payout_scale(legs: list[PricedProp]) -> float:
    """Estimated multiplier scale for PrizePicks' reduced-payout (adjusted) legs."""
    return config.ADJUSTED_LEG_FACTOR ** sum(1 for l in legs if l.prop.adjusted)


@dataclass
class LineupResult:
    legs: list[PricedProp]
    fmt: str
    ev: float  # expected return per $1 (1.0 = break even)
    p_profit: float
    p_all: float
    dist: dict | None = None  # {(n_effective, hits): probability}
    ev_independent: float | None = None
    by_format: dict | None = None  # fmt -> {"ev", "p_profit", "ev_independent"}


def _dist(sub: np.ndarray) -> dict:
    n = sub.shape[1]
    hits = (sub == HIT).sum(axis=1)
    n_eff = n - (sub == PUSH).sum(axis=1)
    keys, counts = np.unique(n_eff * 100 + hits, return_counts=True)
    return {(int(k) // 100, int(k) % 100): c / len(hits) for k, c in zip(keys, counts)}


def evaluate(idx: tuple[int, ...], outcomes: np.ndarray, legs: list[PricedProp], fmt: str) -> LineupResult:
    sub = outcomes[:, idx]
    hits = (sub == HIT).sum(axis=1)
    pushes = (sub == PUSH).sum(axis=1)
    chosen = [legs[i] for i in idx]
    pay = payout(hits, pushes, len(idx), fmt)
    # reduced multipliers on adjusted legs; a voided (refund) result stays 1x
    scale = payout_scale(chosen)
    pay = np.where(pay == 1.0, pay, pay * scale) if fmt == "flex" else pay * scale
    return LineupResult(chosen, fmt, float(pay.mean()),
                        float((pay > 1.0).mean()), float((hits == len(idx)).mean()))


def outcome_rows(dist: dict, n: int, fmt: str) -> list[dict]:
    """Every result with its probability and payout multiple, for display."""
    table = payout_table(n, fmt)
    rows = []
    for (m, h), p in sorted(dist.items(), key=lambda kv: (-kv[0][0], -kv[0][1])):
        rows.append({"label": f"{h}/{m}" + (f" ({n - m} push)" if m < n else ""),
                     "prob": p, "mult": float(table[m, h]), "m": m, "h": h})
    return rows


def finalize(legs: list[PricedProp], trials: int, seed: int, fmt: str) -> LineupResult:
    idx = tuple(range(len(legs)))
    corr = simulate(legs, trials, seed=seed)
    indep = simulate(legs, trials, seed=seed, correlated=False)
    res = evaluate(idx, corr, legs, fmt)
    res.dist = _dist(corr)
    res.by_format = {}
    for f in ("flex", "power"):
        rc, ri = evaluate(idx, corr, legs, f), evaluate(idx, indep, legs, f)
        res.by_format[f] = {"ev": rc.ev, "p_profit": rc.p_profit, "ev_independent": ri.ev}
    res.ev_independent = res.by_format[fmt]["ev_independent"]
    return res


def effective_rho(a: PricedProp, b: PricedProp) -> float:
    return pair_rho(a, b) * sign(a) * sign(b)


def _valid(combo: tuple[int, ...], legs: list[PricedProp]) -> bool:
    if len({legs[i].prop.team for i in combo}) < 2:  # PrizePicks: 2+ teams
        return False
    for i, j in itertools.combinations(combo, 2):
        if banned_pair(legs[i], legs[j]):
            return False
        # brief: no strongly correlated legs (positive links inflate all-hit odds)
        if effective_rho(legs[i], legs[j]) >= config.MAX_LEG_CORR:
            return False
    return True


def build(priced: list[PricedProp], fmt: str = "flex", n_lineups: int = 2,
          size: int = config.LINEUP_SIZE, pool_size: int = config.LINEUP_CANDIDATES,
          trials: int = config.SIM_TRIALS) -> list[LineupResult]:
    eligible = [pp for pp in priced if pp.prob > 0.5
                and (pp.tier == "sharp" or not config.LINEUPS_SHARP_ONLY)]
    eligible.sort(key=lambda pp: -leg_value(pp))
    pool, seen = [], {}
    for pp in eligible:  # at most two props per player in the pool
        key = norm_name(pp.prop.player)
        if seen.get(key, 0) < 2:
            pool.append(pp)
            seen[key] = seen.get(key, 0) + 1
        if len(pool) >= pool_size:
            break
    if len(pool) < size:
        return []
    search = simulate(pool, min(trials, 40_000))
    used: set[str] = set()
    chosen: list[frozenset] = []
    results: list[LineupResult] = []
    for _ in range(n_lineups):
        best = None
        # prefer lineups sharing no players with earlier ones; relax if the board is thin
        for max_overlap in (0, 2, 3):
            for combo in itertools.combinations(range(len(pool)), size):
                if sum(norm_name(pool[i].prop.player) in used for i in combo) > max_overlap:
                    continue
                if frozenset(combo) in chosen or not _valid(combo, pool):
                    continue
                r = evaluate(combo, search, pool, fmt)
                if best is None or r.ev > best[1].ev:
                    best = (combo, r)
            if best is not None:
                break
        if best is None:
            break
        chosen.append(frozenset(best[0]))
        combo = best[0]
        results.append(finalize([pool[i] for i in combo], trials, config.SIM_SEED + len(results) + 1, fmt))
        used |= {norm_name(pool[i].prop.player) for i in combo}
    return results


def build_menu(priced: list[PricedProp], pool_size: int = 14, trials: int = config.SIM_TRIALS,
               keep: int = 6) -> list[LineupResult]:
    """Best lineup for every entry type (2-6 pick Power, 3-6 pick Flex), ranked by EV."""
    eligible = sorted((pp for pp in priced if pp.prob > 0.5
                       and (pp.tier == "sharp" or not config.LINEUPS_SHARP_ONLY)),
                      key=lambda pp: -leg_value(pp))
    pool, seen = [], set()
    for pp in eligible:  # one prop per player keeps the menu readable
        k = norm_name(pp.prop.player)
        if k not in seen:
            pool.append(pp)
            seen.add(k)
        if len(pool) >= pool_size:
            break
    if len(pool) < 2:
        return []
    search = simulate(pool, 20_000)
    out = []
    for size in range(2, min(6, len(pool)) + 1):
        for fmt in ("power", "flex"):
            if fmt == "flex" and size < 3:
                continue
            best = None
            for combo in itertools.combinations(range(len(pool)), size):
                if not _valid(combo, pool):
                    continue
                r = evaluate(combo, search, pool, fmt)
                if best is None or r.ev > best[1].ev:
                    best = (combo, r)
            if best:
                r = finalize([pool[i] for i in best[0]], trials, config.SIM_SEED + size, fmt)
                # conservative EV: the correlation model is a judgment prior, so
                # only credit the lower of correlated vs. independent EV
                r.ev = min(r.ev, r.ev_independent)
                out.append(r)
    out.sort(key=lambda r: -r.ev)
    return out[:keep]


def build_portfolio(priced: list[PricedProp], sizes=(6, 5), per_size: int = 4,
                    max_shared: int = config.PORTFOLIO_MAX_SHARED,
                    max_exposure: float = config.PORTFOLIO_MAX_EXPOSURE, pool_size: int = 20,
                    trials: int = config.SIM_TRIALS) -> list[LineupResult]:
    """Several diversified Flex lineups per size, each +EV under the conservative rule.

    Greedy: score every valid combo once (min of correlated and independent EV),
    then take the best ones subject to (a) at most `max_shared` legs in common
    with any lineup already chosen and (b) no player in more than `max_exposure`
    of all lineups.
    """
    eligible = sorted((pp for pp in priced if pp.prob > 0.5
                       and (pp.tier == "sharp" or not config.LINEUPS_SHARP_ONLY)),
                      key=lambda pp: -leg_value(pp))
    pool, seen = [], set()
    for pp in eligible:
        k = norm_name(pp.prop.player)
        if k not in seen:
            pool.append(pp)
            seen.add(k)
        if len(pool) >= pool_size:
            break
    if len(pool) < min(sizes):
        return []
    corr = simulate(pool, 12_000)
    indep = simulate(pool, 12_000, correlated=False)
    total = per_size * len(sizes)
    cap = max(1, int(max_exposure * total))
    exposure: dict[int, int] = {}
    chosen: list[tuple[int, ...]] = []
    results: list[LineupResult] = []
    for size in sizes:
        scored = []
        for combo in itertools.combinations(range(len(pool)), size):
            if not _valid(combo, pool):
                continue
            ev = min(evaluate(combo, corr, pool, "flex").ev, evaluate(combo, indep, pool, "flex").ev)
            if ev > 1.0:
                scored.append((ev, combo))
        scored.sort(reverse=True)
        taken = 0
        for ev, combo in scored:
            if taken >= per_size:
                break
            if any(len(set(combo) & set(c)) > max_shared for c in chosen):
                continue
            if any(exposure.get(i, 0) >= cap for i in combo):
                continue
            r = finalize([pool[i] for i in combo], trials, config.SIM_SEED + len(chosen) + 1, "flex")
            r.ev = min(r.ev, r.ev_independent)
            if r.ev < 1.0:
                continue  # looked +EV in the quick search but not in the full simulation
            chosen.append(combo)
            for i in combo:
                exposure[i] = exposure.get(i, 0) + 1
            results.append(r)
            taken += 1
    return results


def breakeven(fmt: str, n: int) -> float:
    """Per-leg probability (independent legs, no pushes) that makes EV = 1."""
    from scipy.optimize import brentq
    from math import comb

    def ev(p: float) -> float:
        if fmt == "power":
            return p ** n * config.POWER_PAYOUTS[n] - 1
        t = config.FLEX_PAYOUTS[n]
        return sum(comb(n, h) * p ** h * (1 - p) ** (n - h) * m for h, m in t.items()) - 1

    return brentq(ev, 0.3, 0.99)
