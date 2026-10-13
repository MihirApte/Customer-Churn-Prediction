from types import SimpleNamespace

import numpy as np
import pytest
from helpers import Driver

from churn.advisor.retriever import Retriever
from churn.advisor.search import COMBINED, retrieve_for_customer


# ---- Retriever (cosine similarity search) --------------------------------------------

def _entry(entry_id, text, targets=()):
    return SimpleNamespace(id=entry_id, retrieval_text=text, target_features=tuple(targets))


def _axis_embedder(texts):
    """'alpha' texts point along x, 'beta' along y; 'both' is diagonal."""
    vectors = []
    for t in texts:
        t = t.lower()
        if "both" in t:
            vectors.append([1.0, 1.0])
        elif "alpha" in t:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return np.array(vectors)


def test_retriever_ranks_by_cosine_similarity():
    r = Retriever([_entry("A", "alpha"), _entry("B", "beta"), _entry("C", "both")], _axis_embedder)
    hits = r.retrieve("alpha", k=3)
    assert [h.entry.id for h in hits] == ["A", "C", "B"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(1 / np.sqrt(2))
    assert hits[2].score == pytest.approx(0.0)


def test_scores_ignore_vector_length():
    r = Retriever([_entry("A", "alpha"), _entry("B", "beta")], lambda texts: 7 * _axis_embedder(texts))
    assert r.retrieve("alpha", k=1)[0].score == pytest.approx(1.0)


def test_k_larger_than_playbook_returns_everything():
    r = Retriever([_entry("A", "alpha"), _entry("B", "beta")], _axis_embedder)
    assert len(r.retrieve("alpha", k=50)) == 2


def test_allowed_filter_removes_entries_even_if_they_are_the_closest():
    r = Retriever([_entry("A", "alpha"), _entry("B", "beta")], _axis_embedder)
    hits = r.retrieve("alpha", k=2, allowed=lambda e: e.id != "A")
    assert [h.entry.id for h in hits] == ["B"]
    assert r.retrieve("alpha", k=2, allowed=lambda e: False) == []


def test_retriever_rejects_bad_arguments():
    with pytest.raises(ValueError, match="at least one"):
        Retriever([], _axis_embedder)
    r = Retriever([_entry("A", "alpha")], _axis_embedder)
    with pytest.raises(ValueError, match="k must"):
        r.retrieve("alpha", k=0)


# ---- Hybrid customer search (metadata gate + embeddings) -----------------------------

FIBRE_TENURE_ECHECK = [
    Driver("InternetService_Fiber optic", 0.54),
    Driver("tenure", 0.49),
    Driver("PaymentMethod_Electronic check", 0.37),
]


def test_every_driver_with_guidance_gets_an_entry(retriever, profile):
    result = retrieve_for_customer(retriever, FIBRE_TENURE_ECHECK, profile)
    covered = {label for h in result.hits for label in h.matched_on if label != COMBINED}
    assert covered == {d.feature for d in FIBRE_TENURE_ECHECK}
    assert result.uncovered == []


def test_every_returned_entry_addresses_at_least_one_driver(retriever, profile):
    result = retrieve_for_customer(retriever, FIBRE_TENURE_ECHECK, profile)
    driver_features = {d.feature for d in FIBRE_TENURE_ECHECK}
    for h in result.hits:
        assert driver_features & set(h.entry.target_features), h.entry.id


def test_credit_card_payer_never_gets_the_move_to_automatic_payment_entry(retriever, profile):
    customer = {**profile, "PaymentMethod": "Credit card (automatic)", "StreamingTV": "Yes"}
    drivers = [Driver("StreamingTV_Yes", 0.6), Driver("PaymentMethod_Credit card (automatic)", 0.59),
               Driver("InternetService_Fiber optic", 0.58)]
    result = retrieve_for_customer(retriever, drivers, customer)
    assert "PB-08" not in {h.entry.id for h in result.hits}
    assert "PaymentMethod_Credit card (automatic)" in result.uncovered


def test_mailed_check_gets_the_mailed_check_entry_not_the_electronic_check_one(retriever, profile):
    customer = {**profile, "PaymentMethod": "Mailed check"}
    result = retrieve_for_customer(retriever, [Driver("PaymentMethod_Mailed check", 0.5)], customer)
    ids = [h.entry.id for h in result.hits]
    assert ids[0] == "PB-09" and "PB-08" not in ids


def test_a_charge_that_is_not_high_never_retrieves_the_high_bill_entries(retriever, profile):
    customer = {**profile, "MonthlyCharges": 58.0}
    drivers = [Driver("PaymentMethod_Electronic check", 0.5), Driver("tenure", 0.4),
               Driver("MonthlyCharges", 0.3)]
    result = retrieve_for_customer(retriever, drivers, customer)
    ids = {h.entry.id for h in result.hits}
    assert not ({"PB-06", "PB-07"} & ids)
    assert result.uncovered == ["MonthlyCharges"]


def test_a_customer_already_on_a_one_year_contract_gets_no_month_to_month_entries(retriever, profile):
    customer = {**profile, "Contract": "One year"}
    result = retrieve_for_customer(retriever, [Driver("Contract_Two year", 0.6), Driver("tenure", 0.4)], customer)
    assert not ({"PB-03", "PB-04", "PB-16", "PB-17"} & {h.entry.id for h in result.hits})
    assert "Contract_Two year" in result.uncovered


def test_the_low_cost_entry_is_never_retrieved_by_search(retriever, profile):
    result = retrieve_for_customer(retriever, FIBRE_TENURE_ECHECK, profile)
    assert "PB-19" not in {h.entry.id for h in result.hits}


def test_result_size_respects_max_entries(retriever, profile):
    result = retrieve_for_customer(retriever, FIBRE_TENURE_ECHECK, profile, max_entries=3)
    assert len(result.hits) <= 3


def test_every_driver_keeps_its_entry_even_when_max_entries_is_smaller(retriever, profile):
    result = retrieve_for_customer(retriever, FIBRE_TENURE_ECHECK, profile, max_entries=1)
    assert len(result.hits) == 3  # driver coverage takes precedence over the cap


def test_entries_are_unique_and_those_answering_more_drivers_come_first(retriever, profile):
    result = retrieve_for_customer(retriever, FIBRE_TENURE_ECHECK, profile)
    ids = [h.entry.id for h in result.hits]
    assert len(ids) == len(set(ids))
    counts = [sum(1 for label in h.matched_on if label != COMBINED) for h in result.hits]
    assert counts == sorted(counts, reverse=True)


def test_search_raises_when_no_driver_can_be_used(retriever, profile):
    with pytest.raises(ValueError, match="No usable"):
        retrieve_for_customer(retriever, [Driver("gender_Male", 1.0)], profile)
    with pytest.raises(ValueError, match="max_entries"):
        retrieve_for_customer(retriever, FIBRE_TENURE_ECHECK, profile, max_entries=0)


# ---- Eligibility rules for stacked and multi-condition entries ------------------------

def test_a_stacked_entry_needs_all_its_conditions_to_hold(retriever, profile):
    # PB-16 is "new fibre customer on month-to-month": a one-year-contract customer must not get it
    on_one_year = {**profile, "Contract": "One year"}
    drivers = [Driver("tenure", 0.5), Driver("InternetService_Fiber optic", 0.4)]
    ids = {h.entry.id for h in retrieve_for_customer(retriever, drivers, on_one_year).hits}
    assert "PB-16" not in ids
    ids = {h.entry.id for h in retrieve_for_customer(retriever, drivers, profile).hits}
    assert "PB-16" in ids  # the same drivers on a month-to-month customer do qualify


def test_the_new_customer_entry_is_not_offered_to_a_customer_with_large_total_charges(retriever, profile):
    established = {**profile, "tenure": 14, "MonthlyCharges": 100.0, "TotalCharges": 1400.0}
    ids = {h.entry.id for h in retrieve_for_customer(retriever, [Driver("tenure", 0.5)], established).hits}
    assert "PB-01" not in ids


def test_an_any_entry_applies_when_only_one_of_its_conditions_holds(retriever, profile):
    only_tv = {**profile, "StreamingTV": "Yes", "StreamingMovies": "No"}
    result = retrieve_for_customer(retriever, [Driver("StreamingTV_Yes", 0.6)], only_tv)
    assert result.hits[0].entry.id == "PB-10"


def test_a_payment_driver_must_be_the_customers_own_payment_method(retriever, profile):
    # the customer pays by electronic check; "Mailed check = 0" is not something the playbook can act on
    drivers = [Driver("PaymentMethod_Mailed check", 0.5), Driver("tenure", 0.3)]
    result = retrieve_for_customer(retriever, drivers, profile)
    assert "PB-09" not in {h.entry.id for h in result.hits}
    assert "PaymentMethod_Mailed check" in result.uncovered
