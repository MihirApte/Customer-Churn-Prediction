"""Shared setup for the evaluation scripts: real held-out customers with their SHAP drivers.

The evaluation uses the 1,407 customers the model never trained on, rebuilt from the
cleaned dataset with the same split as notebook 03 (80/20, random_state=42, stratified).
A check confirms each rebuilt customer matches the saved X_test.csv row exactly, so the
profiles, scores and SHAP drivers all belong to the same real customers.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config
from churn import model as churn_model
from churn.advisor.decision import decide_intervention
from churn.advisor.query import NON_ACTIONABLE, build_query
from churn.features import FEATURE_COLS, build_feature_row

EVAL_DIR = ROOT / "evaluation"
LABELS_DIR = EVAL_DIR / "labels"


@dataclass
class Customer:
    cid: int              # row number in telco_churn_cleaned.csv
    profile: dict         # raw customer facts
    score: float          # model churn score
    drivers: list         # top 3 positive SHAP drivers, gender excluded
    decision: object      # cost decision (Decision)
    actual_churn: int


@dataclass
class Context:
    model: object
    roi_params: dict
    y_true: np.ndarray
    y_prob: np.ndarray
    customers: list


def load_context() -> Context:
    config.load_env_file()
    model = churn_model.load_model()
    explainer = churn_model.build_explainer(model)
    scaler_params = config.load_scaler_params()
    roi_params = config.load_roi_params()

    df = pd.read_csv(config.ROOT / "data" / "processed" / "telco_churn_cleaned.csv")
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    encoded = pd.get_dummies(X, columns=categorical, drop_first=True)
    _, test_part, _, _ = train_test_split(encoded, y, test_size=0.2, random_state=42, stratify=y)
    raw_test = X.loc[test_part.index]

    X_test = pd.read_csv(config.X_TEST_PATH)[FEATURE_COLS]
    y_test = pd.read_csv(config.Y_TEST_PATH).squeeze().values

    rebuilt = np.vstack([
        build_feature_row(row.to_dict(), scaler_params).iloc[0].values.astype(float)
        for _, row in raw_test.iterrows()
    ])
    if not np.allclose(rebuilt, X_test.values.astype(float), atol=1e-6):
        raise SystemExit("Alignment check failed: rebuilt customers do not match X_test.csv. "
                         "Nothing was evaluated.")

    y_prob = model.predict_proba(X_test)[:, 1]
    shap_matrix = np.asarray(explainer.shap_values(X_test))

    customers = []
    for i, (cid, row) in enumerate(raw_test.iterrows()):
        drivers = churn_model.top_churn_drivers(shap_matrix[i], n=3, exclude=NON_ACTIONABLE)
        decision = decide_intervention(
            y_test, y_prob, float(y_prob[i]), float(row["MonthlyCharges"]),
            roi_params["retention_rate"], roi_params["offer_cost"],
        )
        customers.append(Customer(int(cid), row.to_dict(), float(y_prob[i]), drivers, decision, int(y_test[i])))
    return Context(model, roi_params, y_test, y_prob, customers)


def baseline_retrieve(retriever, drivers, profile, k: int = 3) -> list:
    """Embedding-only baseline: one combined sentence, top k by similarity, no metadata gate."""
    try:
        query = build_query(drivers, profile)
    except ValueError:
        return []
    return retriever.retrieve(query, k=k)


def describe_customer(profile: dict, score: float, tier: str) -> str:
    """Plain facts a person needs to judge whether a playbook entry fits (no gender)."""
    p = profile
    facts = [
        f"Model score {score:.2f} ({tier} tier)",
        f"tenure {p['tenure']} months",
        f"contract: {p['Contract']}",
        f"internet: {p['InternetService']}",
        f"payment: {p['PaymentMethod']}",
        f"monthly charge EUR {p['MonthlyCharges']:.0f}",
        f"paperless billing: {p['PaperlessBilling']}",
        f"multiple lines: {p['MultipleLines']}",
        f"streaming TV: {p['StreamingTV']}, streaming movies: {p['StreamingMovies']}",
        f"tech support: {p['TechSupport']}, online security: {p['OnlineSecurity']}",
        f"senior citizen: {'yes' if p['SeniorCitizen'] == 1 else 'no'}",
        f"partner: {p['Partner']}, dependents: {p['Dependents']}",
    ]
    return "; ".join(facts)
