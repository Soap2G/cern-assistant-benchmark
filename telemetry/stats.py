"""Small statistics helpers for quality reporting.

Pass rates are binomial proportions, often over small n and at extreme values
(0% or 100%), where the normal approximation is poor. We use the Wilson score
interval, which stays inside [0, 1] and is well-behaved at the extremes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# 95% two-sided normal quantile
Z_95 = 1.959963984540054


@dataclass
class PassStat:
    n_pass: int
    n_scored: int       # pass + fail (infra errors excluded)
    n_infra: int        # runs excluded as infrastructure failures
    rate: float         # n_pass / n_scored
    lo: float           # Wilson lower bound
    hi: float           # Wilson upper bound

    def pct(self) -> str:
        if self.n_scored == 0:
            return "n/a"
        return (f"{100*self.rate:.0f}% [{100*self.lo:.0f}-{100*self.hi:.0f}%]")


def wilson_interval(n_pass: int, n_scored: int, z: float = Z_95) -> tuple[float, float, float]:
    """Return (rate, lo, hi) using the Wilson score interval."""
    if n_scored == 0:
        return 0.0, 0.0, 0.0
    p = n_pass / n_scored
    z2 = z * z
    denom = 1 + z2 / n_scored
    centre = (p + z2 / (2 * n_scored)) / denom
    half = (z * math.sqrt(p * (1 - p) / n_scored + z2 / (4 * n_scored * n_scored))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(statuses: list[str]) -> PassStat:
    """Aggregate a list of Score.status values into a PassStat.

    "infra_error" rows are counted separately and excluded from the denominator.
    """
    n_pass = sum(1 for s in statuses if s == "pass")
    n_infra = sum(1 for s in statuses if s == "infra_error")
    n_scored = len(statuses) - n_infra
    rate, lo, hi = wilson_interval(n_pass, n_scored)
    return PassStat(n_pass=n_pass, n_scored=n_scored, n_infra=n_infra, rate=rate, lo=lo, hi=hi)
