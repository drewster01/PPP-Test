"""Odds conversion and two-way vig removal.

The multiplicative method assumes the margin is spread proportionally across
both sides. Prop books usually shade the popular side (the Over) harder, which
the power, Shin and logit methods handle better because they take more margin
out of the higher-priced side. "power" is the default.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq


def american_to_prob(odds: float | str) -> float:
    o = float(odds)
    if o == 0:
        raise ValueError("American odds of 0 are invalid")
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def prob_to_american(p: float) -> float:
    if not 0.0 < p < 1.0:
        return float("nan")
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def _multiplicative(q1: float, q2: float) -> float:
    return q1 / (q1 + q2)


def _additive(q1: float, q2: float) -> float:
    return min(max(q1 - (q1 + q2 - 1.0) / 2.0, 1e-9), 1 - 1e-9)


def _power(q1: float, q2: float) -> float:
    if abs(q1 + q2 - 1.0) < 1e-12:
        return q1
    f = lambda k: q1 ** k + q2 ** k - 1.0  # noqa: E731
    lo, hi = (1.0, 50.0) if q1 + q2 > 1 else (0.02, 1.0)
    k = brentq(f, lo, hi)
    return q1 ** k


def _shin(q1: float, q2: float) -> float:
    s = q1 + q2
    if s <= 1.0:
        return q1 / s

    def probs(z: float) -> tuple[float, float]:
        out = []
        for q in (q1, q2):
            out.append((math.sqrt(z * z + 4.0 * (1.0 - z) * q * q / s) - z) / (2.0 * (1.0 - z)))
        return out[0], out[1]

    z = brentq(lambda z: sum(probs(z)) - 1.0, 0.0, 0.99)
    return probs(z)[0]


def _logit(q1: float, q2: float) -> float:
    l1 = math.log(q1 / (1 - q1))
    l2 = math.log(q2 / (1 - q2))
    return 1.0 / (1.0 + math.exp(-(l1 - l2) / 2.0))


METHODS = {
    "multiplicative": _multiplicative,
    "additive": _additive,
    "power": _power,
    "shin": _shin,
    "logit": _logit,
}


def devig_two_way(over_odds, under_odds, method: str = "power") -> float:
    """Return the fair probability of the Over from American odds on both sides."""
    q1 = american_to_prob(over_odds)
    q2 = american_to_prob(under_odds)
    if q1 + q2 < 0.98:  # arbitrage-level or bad data: refuse to price
        raise ValueError(f"implied probabilities sum to {q1 + q2:.3f}; likely bad data")
    return METHODS[method](q1, q2)
