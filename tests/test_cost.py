import numpy as np
import pytest

from churn.cost import annual_revenue, compute_roi, confusion_counts


def test_annual_revenue_is_twelve_months():
    assert annual_revenue(64) == 768


def test_confusion_counts_known_example():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
    assert confusion_counts(y_true, y_pred) == (4, 2, 1, 3)  # (tn, fp, fn, tp)


# 10 customers, 40% churn: threshold 0.5 gives TP=3, FP=2, FN=1, TN=4
Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PROB = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.55, 0.1, 0.1, 0.1, 0.1])


def test_compute_roi_matches_hand_calculation():
    r = compute_roi(Y_TRUE, Y_PROB, threshold=0.5, monthly_charge=100, retention_pct=50,
                    offer_cost=100, customers=10, churn_rate_pct=40)
    assert (r.tp_scaled, r.fp_scaled, r.fn_scaled, r.n_churners) == (3, 2, 1, 4)
    assert r.recall == pytest.approx(0.75)
    assert r.precision == pytest.approx(0.6)
    assert r.annual_rev == 1200
    assert r.saved_customers == 1                      # int(3 * 50%) = 1
    assert r.revenue_saved == 1200
    assert r.offer_spend == 500                        # (3 + 2) offers x 100
    assert r.net_saving == 700
    assert r.missed_cost == 1200                       # 1 missed churner x 1200
    assert r.no_model_cost == 4800                     # 4 churners x 1200
    assert r.model_cost == 1700


def test_higher_threshold_flags_fewer_customers():
    low = compute_roi(Y_TRUE, Y_PROB, 0.3, 100, 50, 100, 10, 40)
    high = compute_roi(Y_TRUE, Y_PROB, 0.75, 100, 50, 100, 10, 40)
    assert high.tp_scaled + high.fp_scaled < low.tp_scaled + low.fp_scaled


def test_nothing_flagged_does_not_divide_by_zero():
    r = compute_roi(Y_TRUE, Y_PROB, 1.5, 100, 50, 100, 10, 40)
    assert r.tp_scaled == 0 and r.fp_scaled == 0
    assert r.precision == 0 and r.recall == 0
    assert r.offer_spend == 0 and r.net_saving == 0


def test_results_scale_with_customer_base():
    small = compute_roi(Y_TRUE, Y_PROB, 0.5, 100, 50, 100, 10, 40)
    big = compute_roi(Y_TRUE, Y_PROB, 0.5, 100, 50, 100, 20, 40)
    assert big.tp_scaled == 2 * small.tp_scaled
    assert big.n_churners == 2 * small.n_churners


def test_cheaper_offers_never_reduce_the_saving():
    pricey = compute_roi(Y_TRUE, Y_PROB, 0.5, 100, 50, 300, 10, 40)
    cheap = compute_roi(Y_TRUE, Y_PROB, 0.5, 100, 50, 50, 10, 40)
    assert cheap.net_saving > pricey.net_saving
