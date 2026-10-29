"""Price NBA combo lines PrizePicks posts but the sharp sources do not price
directly (Pts+Rebs, Pts+Asts, Rebs+Asts, Blks+Stls, Fantasy Score), by
combining the player's component markets.

Mean = weighted sum of fitted component means; spread combines component
SDs with typical same-player correlations; the total is modeled as a gamma
and shrunk like any line priced from a model rather than a market.
"""

from __future__ import annotations

import math

from scipy import stats

from . import config
from .parse_pp import PPProp
from .pricing import PricedProp, find_market, fit_dist

# PrizePicks NBA Fantasy Score: pts 1, reb 1.2, ast 1.5, blk 3, stl 3, TO -1
COMBOS = {
    "pts+rebs": {"pts": 1.0, "reb": 1.0},
    "pts+asts": {"pts": 1.0, "ast": 1.0},
    "rebs+asts": {"reb": 1.0, "ast": 1.0},
    "blks+stls": {"blk": 1.0, "stl": 1.0},
    "fantasyscore": {"pts": 1.0, "reb": 1.2, "ast": 1.5, "blk": 3.0, "stl": 3.0},
}
REQUIRED = {
    "pts+rebs": {"pts", "reb"}, "pts+asts": {"pts", "ast"}, "rebs+asts": {"reb", "ast"},
    "blks+stls": {"blk", "stl"}, "fantasyscore": {"pts", "reb", "ast"},
}
RHO = {frozenset({"pts", "reb"}): 0.15, frozenset({"pts", "ast"}): 0.20, frozenset({"reb", "ast"}): 0.10,
       frozenset({"pts", "stl"}): 0.10, frozenset({"blk", "stl"}): 0.05, frozenset({"reb", "blk"}): 0.20}
# Steals/blocks without a market fall back to small league-typical means.
FALLBACK = {"stl": (0.9, 0.95), "blk": (0.5, 0.75)}
MODEL_RELIABILITY = 0.7


def key_for(stat_type: str) -> str | None:
    k = config.norm_stat(stat_type)
    return k if k in COMBOS else None


def price_combo(prop: PPProp, full: dict, short: dict) -> PricedProp | None:
    combo = key_for(prop.stat_type)
    if combo is None:
        return None
    weights = COMBOS[combo]
    parts = {}  # key -> (mean, sd, sharp, from_market)
    for key in weights:
        probe = PPProp("probe", prop.player, prop.team, prop.position, key, 0.5, "standard", None, key)
        m = find_market(probe, full, short)
        fd = fit_dist(m, prop.position) if m else None
        if fd:
            parts[key] = (fd[0].mu, fd[0].sd, fd[1], True)
        elif key in FALLBACK and combo == "fantasyscore":
            parts[key] = (*FALLBACK[key], True, False)
    if not REQUIRED[combo] <= {k for k, v in parts.items() if v[3]}:
        return None
    mean = sum(weights[k] * v[0] for k, v in parts.items())
    if combo == "fantasyscore":
        # turnovers (-1 each) have no market: typical rate from usage
        tov = 0.3 + 0.25 * parts["ast"][0] + 0.04 * parts["pts"][0]
        mean -= tov
    keys = list(parts)
    var = 0.0
    for i, a in enumerate(keys):
        var += (weights[a] * parts[a][1]) ** 2
        for b in keys[i + 1:]:
            var += 2 * RHO.get(frozenset({a, b}), 0.0) * weights[a] * parts[a][1] * weights[b] * parts[b][1]
    if mean <= 0:
        return None
    sd = math.sqrt(max(var, 1e-6))
    shape = (mean / sd) ** 2
    p_raw = float(stats.gamma.sf(prop.line, shape, scale=mean / shape))
    n = sum(1 for v in parts.values() if v[3])
    reliability = MODEL_RELIABILITY * (n / (n + 1.0))
    sharp = all(v[2] for v in parts.values() if v[3])
    detail = ", ".join(f"{k} {v[0]:.1f}" for k, v in parts.items())
    return PricedProp(
        prop=prop, book_line=round(mean, 1), n_books=n, n_quotes=n, method="combo-components",
        p_over_raw=p_raw, p_push=0.0, reliability=reliability, p_over=0.5 + (p_raw - 0.5) * reliability,
        p_over_normal=float(stats.norm.sf((prop.line - mean) / sd)), line_gap_sd=abs(prop.line - mean) / sd,
        books=[f"projection {mean:.1f} from {detail}"], tier="sharp" if sharp else "retail",
    )
