"""Model loading, prediction, risk tiers and SHAP-based churn drivers."""

from typing import NamedTuple

import numpy as np
import pandas as pd

from .config import (
    MODEL_PATH,
    THRESH_CRITICAL,
    THRESH_HIGH,
    THRESH_MEDIUM,
    X_TEST_PATH,
    Y_TEST_PATH,
)
from .features import FEATURE_COLS


class RiskTier(NamedTuple):
    label: str
    action: str
    level: str  # "success" | "warning" | "error" (maps to a Streamlit banner)


class Driver(NamedTuple):
    feature: str
    shap_value: float  # log-odds contribution; positive pushes toward churn


def risk_tier(prob: float) -> RiskTier:
    """Map a churn probability to a risk tier using the configured thresholds."""
    if prob < THRESH_MEDIUM:
        return RiskTier("LOW RISK", "No action required.", "success")
    if prob < THRESH_HIGH:
        return RiskTier("MEDIUM RISK - Monitor", "Monitor and use low-cost touches only; a paid offer loses money on average at this risk level.", "warning")
    if prob < THRESH_CRITICAL:
        return RiskTier("HIGH RISK - Act", "Prioritise for personalised retention outreach.", "error")
    return RiskTier("CRITICAL RISK - Urgent", "Escalate immediately - aggressive retention offer.", "error")


def load_model(path=MODEL_PATH):
    import joblib  # imported here so the pure functions below (and their tests) need no heavy packages

    return joblib.load(path)


def build_explainer(model):
    import shap

    return shap.TreeExplainer(model)


def predict_churn_probability(model, X: pd.DataFrame) -> float:
    """Churn probability for a single-row feature frame."""
    return float(model.predict_proba(X)[0][1])


def shap_values_for(explainer, X: pd.DataFrame) -> np.ndarray:
    """1-D array of SHAP values (one per feature) for a single-row frame."""
    return np.asarray(explainer.shap_values(X)[0])


def top_churn_drivers(
    shap_vals: np.ndarray,
    feature_names: list = FEATURE_COLS,
    n: int = 3,
    exclude: frozenset = frozenset(),
) -> list:
    """The n features pushing hardest TOWARD churn (positive SHAP only), largest first.

    Features in `exclude` are skipped before taking the top n (used to keep
    non-actionable or sensitive features such as gender out of the advice).
    Returns fewer than n if fewer features have a positive contribution.
    """
    order = np.argsort(shap_vals)[::-1]
    drivers = [
        Driver(feature_names[i], float(shap_vals[i]))
        for i in order
        if shap_vals[i] > 0 and feature_names[i] not in exclude
    ]
    return drivers[:n]


def load_holdout(model):
    """Held-out test labels and the model's predicted probabilities on them."""
    X_test = pd.read_csv(X_TEST_PATH)[FEATURE_COLS]
    y_test = pd.read_csv(Y_TEST_PATH).squeeze().values
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_test, y_prob
