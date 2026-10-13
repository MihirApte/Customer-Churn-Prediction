"""Is a paid retention offer worth it for this customer? Decided in code, not by the LLM.

Why not use the model's probability directly: the model was trained with SMOTE, so its
probabilities are inflated (scores of 0.70 and above really churned about 61% of the
time on the held-out test set). Instead we use the REAL churn rate of held-out customers
in the same risk tier, and show how uncertain that rate is.

    expected value of one offer = real churn rate x retention rate x annual revenue - offer cost

An offer only pays for itself when that is above zero. We also compute a 95% range for
the real churn rate (Wilson interval), because a tier with few customers gives a shaky
estimate, and report a verdict:
    JUSTIFIED      even the pessimistic end of the range is above zero
    NOT_JUSTIFIED  even the optimistic end of the range is at or below zero
    MARGINAL       the range crosses zero, so the benefit is uncertain
"""

import math
from enum import Enum
from typing import NamedTuple

import numpy as np

from ..config import THRESH_CRITICAL, THRESH_HIGH, THRESH_MEDIUM
from ..cost import annual_revenue

TIER_BANDS = (
    ("LOW", 0.0, THRESH_MEDIUM),
    ("MEDIUM", THRESH_MEDIUM, THRESH_HIGH),
    ("HIGH", THRESH_HIGH, THRESH_CRITICAL),
    ("CRITICAL", THRESH_CRITICAL, 1.0 + 1e-9),
)


class Verdict(str, Enum):
    JUSTIFIED = "JUSTIFIED"
    MARGINAL = "MARGINAL"
    NOT_JUSTIFIED = "NOT_JUSTIFIED"


class Decision(NamedTuple):
    verdict: Verdict
    tier: str
    n: int                 # held-out customers in this tier
    churn_rate: float      # real churn rate of those customers
    rate_low: float        # 95% range for the real rate
    rate_high: float
    annual_revenue: float  # this customer's 12 months of charges
    offer_cost: float
    retention_rate: float
    expected_value: float  # EUR per offer, at the observed rate
    ev_low: float          # at the pessimistic end of the range
    ev_high: float         # at the optimistic end of the range
    break_even_rate: float # real churn rate at which an offer breaks even


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% confidence range for a proportion (works for small n and rates near 0 or 1)."""
    if n == 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def tier_for(prob: float) -> tuple:
    """(name, low, high) of the risk tier a probability falls in."""
    for name, lo, hi in TIER_BANDS:
        if lo <= prob < hi:
            return name, lo, hi
    raise ValueError(f"probability out of range: {prob}")


def expected_value(churn_rate: float, retention_rate: float, annual_rev: float, offer_cost: float) -> float:
    return churn_rate * retention_rate * annual_rev - offer_cost


def decide_intervention(
    y_true,
    y_prob,
    prob: float,
    monthly_charge: float,
    retention_rate: float,
    offer_cost: float,
) -> Decision:
    """Decide whether a paid offer is worth it for a customer with churn score `prob`."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    tier, lo, hi = tier_for(prob)
    mask = (y_prob >= lo) & (y_prob < hi)
    n = int(mask.sum())
    if n == 0:
        raise ValueError(f"No held-out customers in the {tier} tier to estimate a churn rate from")

    churners = int(y_true[mask].sum())
    rate = churners / n
    rate_low, rate_high = wilson_interval(churners, n)

    annual_rev = annual_revenue(monthly_charge)
    ev = expected_value(rate, retention_rate, annual_rev, offer_cost)
    ev_low = expected_value(rate_low, retention_rate, annual_rev, offer_cost)
    ev_high = expected_value(rate_high, retention_rate, annual_rev, offer_cost)

    if ev_low > 0:
        verdict = Verdict.JUSTIFIED
    elif ev_high <= 0:
        verdict = Verdict.NOT_JUSTIFIED
    else:
        verdict = Verdict.MARGINAL

    break_even = offer_cost / (retention_rate * annual_rev) if annual_rev > 0 else float("inf")
    return Decision(
        verdict, tier, n, rate, rate_low, rate_high, annual_rev, offer_cost,
        retention_rate, ev, ev_low, ev_high, break_even,
    )
