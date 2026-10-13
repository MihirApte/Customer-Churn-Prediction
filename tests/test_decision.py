import pytest

from churn.advisor.decision import (
    Verdict,
    decide_intervention,
    expected_value,
    tier_for,
    wilson_interval,
)


def test_wilson_interval_is_sensible_at_the_extremes():
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and low == pytest.approx(0.7225, abs=1e-3)


def test_wilson_interval_is_centred_and_narrows_with_more_data():
    low_small, high_small = wilson_interval(5, 10)
    low_big, high_big = wilson_interval(500, 1000)
    assert (low_small + high_small) / 2 == pytest.approx(0.5)
    assert (high_big - low_big) < (high_small - low_small)


def test_wilson_interval_rejects_empty_sample():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


@pytest.mark.parametrize("prob, tier", [
    (0.0, "LOW"), (0.109, "LOW"), (0.11, "MEDIUM"), (0.399, "MEDIUM"),
    (0.40, "HIGH"), (0.699, "HIGH"), (0.70, "CRITICAL"), (1.0, "CRITICAL"),
])
def test_tier_for_boundaries(prob, tier):
    assert tier_for(prob)[0] == tier


def test_expected_value_formula():
    # 40% churn x 30% saved x 768 revenue - 80 offer
    assert expected_value(0.40, 0.30, 768, 80) == pytest.approx(0.4 * 0.3 * 768 - 80)


@pytest.mark.parametrize("prob, verdict", [
    (0.05, Verdict.NOT_JUSTIFIED),   # LOW: real churn 3.6%
    (0.25, Verdict.NOT_JUSTIFIED),   # MEDIUM: real churn 16.8%
    (0.50, Verdict.MARGINAL),        # HIGH: 38.0%, range crosses break-even
    (0.80, Verdict.JUSTIFIED),       # CRITICAL: 60.6%
])
def test_verdict_by_tier_for_a_64_euro_customer(prob, verdict, holdout):
    y_true, y_prob = holdout
    assert decide_intervention(y_true, y_prob, prob, 64, 0.30, 80).verdict == verdict


def test_the_customers_own_charge_changes_the_verdict(holdout):
    y_true, y_prob = holdout
    at_64 = decide_intervention(y_true, y_prob, 0.5, 64, 0.30, 80)
    at_95 = decide_intervention(y_true, y_prob, 0.5, 95, 0.30, 80)
    assert at_64.verdict == Verdict.MARGINAL and at_95.verdict == Verdict.JUSTIFIED
    assert at_95.expected_value > at_64.expected_value


def test_decision_reports_the_numbers_it_used(holdout):
    y_true, y_prob = holdout
    d = decide_intervention(y_true, y_prob, 0.8, 64, 0.30, 80)
    assert d.tier == "CRITICAL" and d.n == 315
    assert d.churn_rate == pytest.approx(0.606, abs=0.002)
    assert d.annual_revenue == 768
    assert d.break_even_rate == pytest.approx(80 / (0.30 * 768))
    assert d.rate_low < d.churn_rate < d.rate_high
    assert d.ev_low < d.expected_value < d.ev_high


def test_a_tier_with_no_heldout_customers_raises(holdout):
    y_true, y_prob = holdout
    keep = y_prob < 0.7  # drop every CRITICAL customer
    with pytest.raises(ValueError, match="No held-out customers"):
        decide_intervention(y_true[keep], y_prob[keep], 0.8, 64, 0.30, 80)
