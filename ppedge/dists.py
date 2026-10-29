"""Stat distributions used to move a sportsbook price to a different line.

Counts (receptions, attempts, completions, TDs) use a negative binomial;
yardage uses a gamma whose SD scales with its mean. Both are integer-valued
in reality, so every probability is computed on whole-number outcomes and a
whole-number line can push.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize


@dataclass(frozen=True)
class Dist:
    family: str  # "gamma" | "negbin" | "binom"
    mu: float
    disp: float  # gamma: coefficient of variation; negbin: k (size); binom: trials n

    @property
    def sd(self) -> float:
        if self.family == "gamma":
            return self.disp * self.mu
        if self.family == "binom":
            return math.sqrt(self.mu * max(0.0, 1.0 - self.mu / self.disp))
        return math.sqrt(self.mu + self.mu * self.mu / self.disp)

    def _ge(self, n: float) -> float:
        """P(outcome >= n) for integer n."""
        if self.family == "gamma":
            shape = 1.0 / (self.disp * self.disp)
            # continuity correction: whole-number yards
            return float(stats.gamma.sf(n - 0.5, shape, scale=self.mu / shape))
        if self.family == "binom":
            # under-dispersed counts (variance < mean), e.g. receptions
            trials = int(round(self.disp))
            return float(stats.binom.sf(n - 1, trials, min(self.mu / trials, 0.999)))
        k = self.disp
        return float(stats.nbinom.sf(n - 1, k, k / (k + self.mu)))

    def outcome(self, line: float) -> tuple[float, float, float]:
        """(P over, P push, P under) at a line."""
        if float(line).is_integer():
            n = int(line)
            p_over = self._ge(n + 1)
            p_push = self._ge(n) - p_over
        else:
            p_over = self._ge(math.floor(line) + 1)
            p_push = 0.0
        return p_over, p_push, max(0.0, 1.0 - p_over - p_push)

    def p_over_given_no_push(self, line: float) -> float:
        o, p, u = self.outcome(line)
        return o / (o + u) if o + u > 0 else 0.5


def _bounds(family: str) -> tuple[float, float]:
    return {"gamma": (0.15, 1.2), "binom": (8.0, 200.0)}.get(family, (2.0, 1000.0))


def fit_mean(family: str, disp: float, line: float, p_over: float) -> Dist:
    """Find the mean that reproduces a no-push Over probability at `line`."""
    p_over = min(max(p_over, 0.005), 0.995)

    def f(mu: float) -> float:
        return Dist(family, mu, disp).p_over_given_no_push(line) - p_over

    lo, hi = 0.01, max(10.0, line * 6 + 10)
    if family == "binom":
        hi = min(hi, disp * 0.99)
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return Dist(family, lo, disp)
    if fhi < 0:
        return Dist(family, hi, disp)
    return Dist(family, brentq(f, lo, hi, xtol=1e-6), disp)


def fit_ladder(family: str, disp0: float, points: list[tuple[float, float, float]],
               prior_strength: float = 0.5) -> Dist:
    """Fit mean and dispersion to several (line, p_over_no_push, weight) points.

    A weak prior keeps the dispersion near the stat's default so that a couple
    of noisy prices cannot produce an absurd shape.
    """
    lines = np.array([p[0] for p in points], dtype=float)
    probs = np.clip(np.array([p[1] for p in points], dtype=float), 0.01, 0.99)
    w = np.array([p[2] for p in points], dtype=float)
    target = np.log(probs / (1 - probs))
    lo_d, hi_d = _bounds(family)

    # start from the mean implied by the highest-weight point
    i0 = int(np.argmax(w))
    start = fit_mean(family, disp0, lines[i0], probs[i0])

    def loss(x: np.ndarray) -> float:
        mu, d = math.exp(x[0]), math.exp(x[1])
        if family == "binom" and mu >= d * 0.99:
            return 1e6
        dist = Dist(family, mu, d)
        model = np.array([dist.p_over_given_no_push(l) for l in lines])
        model = np.clip(model, 1e-4, 1 - 1e-4)
        err = np.log(model / (1 - model)) - target
        return float(np.sum(w * err * err) / np.sum(w)) + prior_strength * (x[1] - math.log(disp0)) ** 2

    res = minimize(
        loss,
        x0=np.array([math.log(start.mu), math.log(disp0)]),
        method="L-BFGS-B",
        bounds=[(math.log(0.05), math.log(1000.0)), (math.log(lo_d), math.log(hi_d))],
    )
    return Dist(family, math.exp(res.x[0]), math.exp(res.x[1]))


def normal_shift(book_line: float, p_over: float, pp_line: float, sd: float) -> float:
    """The original brief's normal-approximation shift, kept for comparison."""
    p_over = min(max(p_over, 1e-4), 1 - 1e-4)
    mu = book_line + sd * stats.norm.ppf(p_over)
    return float(stats.norm.sf((pp_line - mu) / sd))
