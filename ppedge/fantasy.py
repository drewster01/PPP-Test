"""Price PrizePicks Fantasy Score lines from sharp component markets (standard PPR).

Projection = sum of (scoring weight x fitted mean) across the player's
component markets (receptions, yards, TDs...). Spread combines component SDs
with same-player correlations. The total is modeled as a gamma.
"""

from __future__ import annotations

import math

from scipy import stats

from .names import norm_name
from .parse_pp import PPProp
from .pricing import PricedProp, fit_dist, find_market

# standard PPR
SCORING = {
    "receptions": 1.0, "rec_yds": 0.1, "rush_yds": 0.1, "tds": 6.0,
    "pass_yds": 0.04, "pass_td": 4.0, "pass_int": -1.0,
}
COMPONENTS = {
    "QB": ["pass_yds", "pass_td", "pass_int", "rush_yds", "tds"],
    "RB": ["receptions", "rec_yds", "rush_yds", "tds"],
    "WR": ["receptions", "rec_yds", "rush_yds", "tds"],
    "TE": ["receptions", "rec_yds", "tds"],
}
# components a projection cannot do without
# TDs are required too: leaving them out would bias every projection toward Less
REQUIRED = {"QB": {"pass_yds", "pass_td"}, "RB": {"rush_yds", "tds"},
            "WR": {"receptions", "rec_yds", "tds"}, "TE": {"receptions", "rec_yds", "tds"}}
RHO = {
    frozenset({"receptions", "rec_yds"}): 0.70, frozenset({"rec_yds", "tds"}): 0.30,
    frozenset({"rush_yds", "tds"}): 0.30, frozenset({"receptions", "tds"}): 0.20,
    frozenset({"pass_yds", "pass_td"}): 0.50, frozenset({"pass_yds", "pass_int"}): 0.10,
    frozenset({"rush_yds", "rec_yds"}): -0.05,
}
MODEL_RELIABILITY = 0.7  # built from components, not priced directly: shrink harder


def price_fantasy(prop: PPProp, full: dict, short: dict) -> PricedProp | None:
    pos = (prop.position or "").upper()
    comps = COMPONENTS.get(pos)
    if not comps:
        return None
    parts = {}  # key -> (mean, sd, sharp)
    for key in comps:
        probe = PPProp("probe", prop.player, prop.team, prop.position, key, 0.5, "standard", None, key)
        m = find_market(probe, full, short)
        fd = fit_dist(m, prop.position) if m else None
        if fd:
            parts[key] = (fd[0].mu, fd[0].sd, fd[1])
    if not REQUIRED[pos] <= set(parts):
        return None
    mean = sum(SCORING[k] * mu for k, (mu, _, _) in parts.items())
    var = 0.0
    keys = list(parts)
    for i, a in enumerate(keys):
        var += (SCORING[a] * parts[a][1]) ** 2
        for b in keys[i + 1:]:
            rho = RHO.get(frozenset({a, b}), 0.0)
            var += 2 * rho * SCORING[a] * parts[a][1] * SCORING[b] * parts[b][1]
    if mean <= 0:
        return None
    sd = math.sqrt(max(var, 1e-6))
    shape = (mean / sd) ** 2
    p_raw = float(stats.gamma.sf(prop.line, shape, scale=mean / shape))
    sharp = all(s for _, _, s in parts.values())
    n = len(parts)
    reliability = MODEL_RELIABILITY * (n / (n + 1.0))
    p = 0.5 + (p_raw - 0.5) * reliability
    detail = ", ".join(f"{k} {mu:.1f}" for k, (mu, _, _) in parts.items())
    pp = PricedProp(
        prop=prop, book_line=round(mean, 1), n_books=n, n_quotes=n, method="ppr-components",
        p_over_raw=p_raw, p_push=0.0, reliability=reliability, p_over=p,
        p_over_normal=float(stats.norm.sf((prop.line - mean) / sd)), line_gap_sd=abs(prop.line - mean) / sd,
        books=[f"PPR projection {mean:.1f} from {detail}"], tier="sharp" if sharp else "retail",
    )
    return pp
