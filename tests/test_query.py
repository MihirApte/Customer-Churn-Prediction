import pytest
from helpers import Driver

from churn.advisor.query import (
    DESCRIBERS,
    NON_ACTIONABLE,
    base_field,
    build_query,
    describe_driver,
    driver_descriptions,
    friendly_driver_names,
    skipped_drivers,
)
from churn.features import FEATURE_COLS


@pytest.mark.parametrize("feature, change, should_describe", [
    # A describer fires only when the customer's value is one the playbook addresses.
    ("MonthlyCharges", {"MonthlyCharges": 58.0}, False),
    ("MonthlyCharges", {"MonthlyCharges": 95.0}, True),
    ("TotalCharges", {"TotalCharges": 4000.0}, False),
    ("TotalCharges", {"TotalCharges": 400.0}, True),
    ("SeniorCitizen", {"SeniorCitizen": 0}, False),
    ("SeniorCitizen", {"SeniorCitizen": 1}, True),
    ("Contract_Two year", {"Contract": "One year"}, False),
    ("Contract_Two year", {"Contract": "Month-to-month"}, True),
    ("InternetService_Fiber optic", {"InternetService": "DSL"}, False),
    ("InternetService_Fiber optic", {"InternetService": "Fiber optic"}, True),
    ("MultipleLines_Yes", {"MultipleLines": "No"}, False),
    ("MultipleLines_Yes", {"MultipleLines": "Yes"}, True),
    ("Partner_Yes", {"Partner": "Yes"}, False),
    ("Partner_Yes", {"Partner": "No"}, True),
    ("Dependents_Yes", {"Dependents": "Yes"}, False),
    ("StreamingTV_Yes", {"StreamingTV": "No"}, False),
    ("StreamingTV_Yes", {"StreamingTV": "Yes"}, True),
    ("TechSupport_Yes", {"TechSupport": "Yes"}, False),
    ("TechSupport_Yes", {"TechSupport": "No"}, True),
    ("PaperlessBilling_Yes", {"PaperlessBilling": "No"}, False),
    ("PaymentMethod_Credit card (automatic)", {"PaymentMethod": "Credit card (automatic)"}, False),
    ("PaymentMethod_Electronic check", {"PaymentMethod": "Electronic check"}, True),
    ("PaymentMethod_Mailed check", {"PaymentMethod": "Mailed check"}, True),
])
def test_describer_fires_only_for_values_the_playbook_addresses(feature, change, should_describe, profile):
    assert (describe_driver(feature, {**profile, **change}) is not None) is should_describe


def test_tenure_is_always_described_in_plain_bands(profile):
    assert "new customer" in describe_driver("tenure", {**profile, "tenure": 3})
    assert "first anniversary" in describe_driver("tenure", {**profile, "tenure": 10})
    assert "moderate" in describe_driver("tenure", {**profile, "tenure": 20})
    assert "long tenure" in describe_driver("tenure", {**profile, "tenure": 40})


def test_gender_is_never_described(profile):
    assert "gender_Male" in NON_ACTIONABLE
    assert describe_driver("gender_Male", profile) is None


def test_every_feature_family_has_a_describer_except_gender():
    families = {base_field(f) for f in FEATURE_COLS}
    assert families - set(DESCRIBERS) == {"gender"}


def test_query_joins_descriptions_capitalised_and_ends_with_a_full_stop(profile):
    drivers = [Driver("InternetService_Fiber optic", 0.5), Driver("tenure", 0.4)]
    query = build_query(drivers, profile)
    assert query.startswith("Fibre optic internet service.")
    assert "New customer" in query and query.endswith(".")


def test_drivers_from_the_same_field_are_described_once(profile):
    drivers = [Driver("Contract_Two year", 0.9), Driver("Contract_One year", 0.5)]
    parts = driver_descriptions(drivers, profile)
    assert len(parts) == 1 and parts[0][0] == "Contract_Two year"


def test_query_never_contains_gender(profile):
    query = build_query([Driver("gender_Male", 2.0), Driver("tenure", 0.5)], profile)
    assert "male" not in query.lower() and "gender" not in query.lower()


def test_query_raises_when_no_driver_is_usable(profile):
    with pytest.raises(ValueError, match="No usable"):
        build_query([Driver("gender_Male", 1.0)], profile)


def test_skipped_drivers_lists_inapplicable_drivers_but_not_gender(profile):
    drivers = [Driver("gender_Male", 1.0), Driver("MonthlyCharges", 0.5), Driver("tenure", 0.4)]
    assert skipped_drivers(drivers, {**profile, "MonthlyCharges": 58.0}) == ["MonthlyCharges"]


def test_friendly_names_are_readable_and_deduplicated():
    names = friendly_driver_names(["PaymentMethod_Mailed check", "PaymentMethod_Electronic check",
                                   "MonthlyCharges", "Contract_Two year"])
    assert names == ["Payment method", "Monthly charge", "Contract type"]


@pytest.mark.parametrize("feature, method, should_describe", [
    ("PaymentMethod_Electronic check", "Electronic check", True),
    ("PaymentMethod_Mailed check", "Electronic check", False),
    ("PaymentMethod_Mailed check", "Mailed check", True),
    ("PaymentMethod_Credit card (automatic)", "Electronic check", False),
])
def test_payment_drivers_are_described_only_for_the_customers_own_method(feature, method, should_describe, profile):
    assert (describe_driver(feature, {**profile, "PaymentMethod": method}) is not None) is should_describe
