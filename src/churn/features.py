"""Turn a customer profile (plain dict) into the exact row the model was trained on."""

import pandas as pd

# Column order the model expects. Must match training.
FEATURE_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]

# Numeric profile fields. The first is a 0/1 flag; the rest are z-scored below.
SENIOR_FIELD = "SeniorCitizen"
CONTINUOUS_FIELDS = ("tenure", "MonthlyCharges", "TotalCharges")

# Allowed values for every categorical field. The first value of each field is the
# "baseline" category, which has no one-hot column (all its columns stay 0).
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

PROFILE_FIELDS = (SENIOR_FIELD, *CONTINUOUS_FIELDS, *CATEGORICAL_OPTIONS)


def scale(col: str, raw_value: float, scaler_params: dict) -> float:
    """Standardise a raw value using the training-data mean and std."""
    return (raw_value - scaler_params[col]["mean"]) / scaler_params[col]["std"]


def build_feature_row(profile: dict, scaler_params: dict) -> pd.DataFrame:
    """Convert a raw customer profile into a one-row DataFrame in FEATURE_COLS order.

    Raises ValueError if a field is missing or has a value the model never saw,
    instead of silently producing a wrong row.
    """
    missing = [f for f in PROFILE_FIELDS if f not in profile]
    if missing:
        raise ValueError(f"Profile is missing fields: {missing}")

    if profile[SENIOR_FIELD] not in (0, 1):
        raise ValueError(f"{SENIOR_FIELD} must be 0 or 1, got {profile[SENIOR_FIELD]!r}")

    row = dict.fromkeys(FEATURE_COLS, 0)
    row[SENIOR_FIELD] = int(profile[SENIOR_FIELD])

    for col in CONTINUOUS_FIELDS:
        row[col] = scale(col, profile[col], scaler_params)

    for col, options in CATEGORICAL_OPTIONS.items():
        value = profile[col]
        if value not in options:
            raise ValueError(f"{col} must be one of {options}, got {value!r}")
        one_hot_col = f"{col}_{value}"
        if one_hot_col in row:  # baseline categories have no column
            row[one_hot_col] = 1

    return pd.DataFrame([row])[FEATURE_COLS]
