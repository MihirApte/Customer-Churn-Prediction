"""Business cost / ROI maths, moved out of the Streamlit app so it can be tested."""

from typing import NamedTuple

import numpy as np


class RoiResult(NamedTuple):
    n_churners: int
    tp_scaled: int
    fp_scaled: int
    fn_scaled: int
    recall: float
    precision: float
    annual_rev: float
    saved_customers: int
    revenue_saved: float
    offer_spend: float
    net_saving: float
    missed_cost: float
    no_model_cost: float
    model_cost: float


def annual_revenue(monthly_charge: float) -> float:
    """Revenue lost if a customer churns: 12 months of their monthly charge."""
    return monthly_charge * 12


def confusion_counts(y_true, y_pred):
    """Return (tn, fp, fn, tp) for binary 0/1 arrays."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def compute_roi(
    y_true,
    y_prob,
    threshold: float,
    monthly_charge: float,
    retention_pct: float,
    offer_cost: float,
    customers: int,
    churn_rate_pct: float,
) -> RoiResult:
    """Scale held-out test-set performance at `threshold` to a customer base.

    Recall and precision come from the actual test set. Offers go to every flagged
    customer (TP + FP); only retention_pct of the true positives are actually kept.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    _, fp, fn, tp = confusion_counts(y_true, y_pred)

    n_test = len(y_true)
    test_churn_rate = y_true.mean()
    user_churn_rate = churn_rate_pct / 100
    scale_factor = (customers * user_churn_rate) / (n_test * test_churn_rate)

    tp_scaled = int(tp * scale_factor)
    fp_scaled = int(fp * scale_factor)
    fn_scaled = int(fn * scale_factor)
    n_churners = int(customers * user_churn_rate)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    annual_rev = annual_revenue(monthly_charge)
    saved_customers = int(tp_scaled * retention_pct / 100)
    revenue_saved = saved_customers * annual_rev
    offer_spend = (tp_scaled + fp_scaled) * offer_cost
    net_saving = revenue_saved - offer_spend
    missed_cost = fn_scaled * annual_rev

    no_model_cost = n_churners * annual_rev
    model_cost = missed_cost + offer_spend

    return RoiResult(
        n_churners, tp_scaled, fp_scaled, fn_scaled, recall, precision, annual_rev,
        saved_customers, revenue_saved, offer_spend, net_saving, missed_cost,
        no_model_cost, model_cost,
    )
