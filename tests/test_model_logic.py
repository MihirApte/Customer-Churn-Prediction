import numpy as np
import pytest

from churn import config
from churn.features import FEATURE_COLS
from churn.model import risk_tier, top_churn_drivers


@pytest.mark.parametrize("prob, label, level", [
    (0.0, "LOW RISK", "success"),
    (0.109, "LOW RISK", "success"),
    (0.11, "MEDIUM RISK - Monitor", "warning"),
    (0.399, "MEDIUM RISK - Monitor", "warning"),
    (0.40, "HIGH RISK - Act", "error"),
    (0.699, "HIGH RISK - Act", "error"),
    (0.70, "CRITICAL RISK - Urgent", "error"),
    (1.0, "CRITICAL RISK - Urgent", "error"),
])
def test_risk_tier_boundaries(prob, label, level):
    tier = risk_tier(prob)
    assert tier.label == label
    assert tier.level == level


def test_tier_thresholds_are_ordered():
    assert 0 < config.THRESH_MEDIUM < config.THRESH_HIGH < config.THRESH_CRITICAL < 1


def _shap(values_by_feature):
    arr = np.zeros(len(FEATURE_COLS))
    for feature, value in values_by_feature.items():
        arr[FEATURE_COLS.index(feature)] = value
    return arr


def test_top_drivers_are_the_largest_positive_values_in_order():
    shap_vals = _shap({"tenure": 0.5, "MonthlyCharges": 0.9, "Contract_One year": 0.1,
                       "PaperlessBilling_Yes": -2.0})
    drivers = top_churn_drivers(shap_vals, n=2)
    assert [d.feature for d in drivers] == ["MonthlyCharges", "tenure"]
    assert drivers[0].shap_value == pytest.approx(0.9)


def test_negative_contributions_are_never_drivers():
    drivers = top_churn_drivers(_shap({"tenure": -1.0, "MonthlyCharges": -0.2}), n=3)
    assert drivers == []


def test_fewer_positive_features_than_n_returns_fewer():
    drivers = top_churn_drivers(_shap({"tenure": 0.4}), n=3)
    assert [d.feature for d in drivers] == ["tenure"]


def test_excluded_features_are_skipped_before_taking_the_top_n():
    shap_vals = _shap({"gender_Male": 2.0, "tenure": 0.5, "MonthlyCharges": 0.4, "TotalCharges": 0.3})
    drivers = top_churn_drivers(shap_vals, n=2, exclude=frozenset({"gender_Male"}))
    assert [d.feature for d in drivers] == ["tenure", "MonthlyCharges"]
