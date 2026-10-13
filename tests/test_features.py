import json

import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.features import CATEGORICAL_OPTIONS, FEATURE_COLS, build_feature_row, scale


def test_row_has_every_feature_column_in_training_order(profile, scaler_params):
    row = build_feature_row(profile, scaler_params)
    assert list(row.columns) == FEATURE_COLS
    assert row.shape == (1, 30)


def test_continuous_features_are_standardised_with_training_stats(profile, scaler_params):
    row = build_feature_row({**profile, "tenure": 56}, scaler_params)
    assert row["tenure"].iloc[0] == pytest.approx((56 - 32.0) / 24.0)
    assert scale("MonthlyCharges", 95.0, scaler_params) == pytest.approx(1.0)


def test_baseline_categories_have_no_column(profile, scaler_params):
    # Month-to-month, DSL and bank transfer are the dropped baselines: all their columns stay 0
    row = build_feature_row(
        {**profile, "Contract": "Month-to-month", "InternetService": "DSL",
         "PaymentMethod": "Bank transfer (automatic)"}, scaler_params).iloc[0]
    assert row["Contract_One year"] == 0 and row["Contract_Two year"] == 0
    assert row["InternetService_Fiber optic"] == 0 and row["InternetService_No"] == 0
    assert row["PaymentMethod_Electronic check"] == 0
    assert row["PaymentMethod_Mailed check"] == 0
    assert row["PaymentMethod_Credit card (automatic)"] == 0


def test_exactly_one_hot_column_is_set_for_a_non_baseline_value(profile, scaler_params):
    row = build_feature_row({**profile, "Contract": "Two year"}, scaler_params).iloc[0]
    assert row["Contract_Two year"] == 1
    assert row["Contract_One year"] == 0


def test_no_internet_service_sets_the_dependent_flags(profile, scaler_params):
    no_net = {**profile, "InternetService": "No", "OnlineSecurity": "No internet service",
              "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
              "TechSupport": "No internet service", "StreamingTV": "No internet service",
              "StreamingMovies": "No internet service"}
    row = build_feature_row(no_net, scaler_params).iloc[0]
    assert row["InternetService_No"] == 1
    assert row["OnlineSecurity_No internet service"] == 1
    assert row["StreamingMovies_No internet service"] == 1
    assert row["OnlineSecurity_Yes"] == 0


def test_binary_flags(profile, scaler_params):
    row = build_feature_row({**profile, "gender": "Male", "SeniorCitizen": 1, "Partner": "Yes"},
                            scaler_params).iloc[0]
    assert row["gender_Male"] == 1 and row["SeniorCitizen"] == 1 and row["Partner_Yes"] == 1
    row = build_feature_row({**profile, "gender": "Female", "SeniorCitizen": 0, "Partner": "No"},
                            scaler_params).iloc[0]
    assert row["gender_Male"] == 0 and row["SeniorCitizen"] == 0 and row["Partner_Yes"] == 0


@pytest.mark.parametrize("field", list(CATEGORICAL_OPTIONS))
def test_every_allowed_category_value_builds_a_row(field, profile, scaler_params):
    for value in CATEGORICAL_OPTIONS[field]:
        row = build_feature_row({**profile, field: value}, scaler_params)
        assert row.shape == (1, 30)
        assert not row.isna().any().any()


def test_missing_field_raises(profile, scaler_params):
    del profile["tenure"]
    with pytest.raises(ValueError, match="missing fields"):
        build_feature_row(profile, scaler_params)


def test_unknown_category_value_raises_instead_of_silently_zeroing(profile, scaler_params):
    with pytest.raises(ValueError, match="Contract must be one of"):
        build_feature_row({**profile, "Contract": "Weekly"}, scaler_params)


def test_senior_flag_must_be_zero_or_one(profile, scaler_params):
    with pytest.raises(ValueError, match="SeniorCitizen"):
        build_feature_row({**profile, "SeniorCitizen": "Yes"}, scaler_params)


def test_reproduces_the_saved_training_test_matrix():
    """The app's feature builder must give exactly the rows the model was tested on.
    This guards against train/serve skew (it caught a scaler mismatch)."""
    raw_path = config.ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
    if not (raw_path.exists() and config.X_TEST_PATH.exists() and config.SCALER_PARAMS_PATH.exists()):
        pytest.skip("processed data files not available")

    scaler_params = json.loads(config.SCALER_PARAMS_PATH.read_text())
    raw = pd.read_csv(raw_path)
    built = {
        tuple(np.round(build_feature_row(r.drop("Churn").to_dict(), scaler_params)
                       .iloc[0].values.astype(float), 5))
        for _, r in raw.iterrows()
    }
    x_test = pd.read_csv(config.X_TEST_PATH)[FEATURE_COLS].values.astype(float)
    missing = [i for i, row in enumerate(x_test) if tuple(np.round(row, 5)) not in built]
    assert not missing, f"{len(missing)} of {len(x_test)} test rows were not reproduced"
