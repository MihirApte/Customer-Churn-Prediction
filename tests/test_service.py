import re

import pytest
from helpers import BAD_PLAN, GOOD_PLAN, Driver, FakeLLM, valid_plan

from churn.advisor.citations import check_citations
from churn.advisor.decision import Verdict
from churn.advisor.llm import LLMError
from churn.advisor.service import MAX_LLM_ATTEMPTS, RetentionAdvisor

DRIVERS = [
    Driver("InternetService_Fiber optic", 0.54),
    Driver("tenure", 0.49),
    Driver("PaymentMethod_Electronic check", 0.37),
]
CRITICAL, HIGH, MEDIUM, LOW = 0.80, 0.50, 0.25, 0.05


def _advisor(retriever, llm, holdout, roi_params):
    y_true, y_prob = holdout
    return RetentionAdvisor(retriever, llm, y_true, y_prob, roi_params)


def test_unprofitable_customers_never_reach_the_llm(retriever, holdout, roi_params, profile):
    llm = FakeLLM(["must not be called"])
    for score in (LOW, MEDIUM):
        advice = _advisor(retriever, llm, holdout, roi_params).advise(profile, DRIVERS, score)
        assert advice.mode == "no_offer"
        assert advice.decision.verdict == Verdict.NOT_JUSTIFIED
        assert "PB-19" in advice.plan
    assert llm.calls == 0


def test_no_offer_plan_explains_the_money(retriever, holdout, roi_params, profile):
    advice = _advisor(retriever, FakeLLM([]), holdout, roi_params).advise(profile, DRIVERS, MEDIUM)
    assert "No paid retention offer" in advice.plan
    assert "MEDIUM" in advice.plan and "break" in advice.plan


def test_justified_customer_gets_a_verified_llm_plan(retriever, holdout, roi_params, profile):
    llm = FakeLLM([valid_plan])
    advice = _advisor(retriever, llm, holdout, roi_params).advise(profile, DRIVERS, CRITICAL)
    assert advice.mode == "llm" and advice.attempts == 1 and llm.calls == 1
    assert advice.plan.startswith("Recommended actions")
    assert advice.citation.ok
    assert advice.hits  # the evidence shown to the user


def test_prompt_contains_the_computed_numbers_and_only_offered_entries(retriever, holdout, roi_params, profile):
    llm = FakeLLM([valid_plan])
    advice = _advisor(retriever, llm, holdout, roi_params).advise(profile, DRIVERS, CRITICAL)
    prompt = llm.last_user
    assert "Verdict: JUSTIFIED" in prompt
    assert f"n={advice.decision.n}" in prompt and "break-even" in prompt
    for hit in advice.hits:
        assert f"[{hit.entry.id}]" in prompt
    assert "[PB-19]" not in prompt  # the low-cost fallback is never offered for a justified offer


def test_customer_data_in_the_prompt_never_mentions_gender(retriever, holdout, roi_params, profile):
    llm = FakeLLM([valid_plan])
    _advisor(retriever, llm, holdout, roi_params).advise(profile, DRIVERS, CRITICAL)
    user = llm.last_user.lower()
    assert "gender" not in user and not re.search(r"\b(male|female)\b", user)
    assert "gender" in llm.last_system.lower()  # the rule forbidding its use is stated to the model


def test_system_prompt_forbids_invented_numbers_and_unretrieved_ids(retriever, holdout, roi_params, profile):
    llm = FakeLLM([valid_plan])
    _advisor(retriever, llm, holdout, roi_params).advise(profile, DRIVERS, CRITICAL)
    assert "Never invent an id" in llm.last_system
    assert "Do not invent numbers" in llm.last_system


def test_marginal_customer_prompt_asks_for_the_cheapest_actions(retriever, holdout, roi_params, profile):
    llm = FakeLLM([valid_plan])
    advice = _advisor(retriever, llm, holdout, roi_params).advise({**profile, "MonthlyCharges": 64.0}, DRIVERS, HIGH)
    assert advice.decision.verdict == Verdict.MARGINAL
    assert "cheapest actions" in llm.last_user


def test_a_bad_citation_triggers_one_retry_that_can_succeed(retriever, holdout, roi_params, profile):
    llm = FakeLLM([BAD_PLAN, valid_plan])
    advice = _advisor(retriever, llm, holdout, roi_params).advise(profile, DRIVERS, CRITICAL)
    assert advice.mode == "llm" and advice.attempts == 2
    assert "rejected" in llm.last_user and "PB-99" in llm.last_user


def test_two_bad_plans_fall_back_to_a_plan_built_from_the_playbook(retriever, holdout, roi_params, profile):
    llm = FakeLLM([BAD_PLAN, BAD_PLAN])
    advice = _advisor(retriever, llm, holdout, roi_params).advise(profile, DRIVERS, CRITICAL)
    assert advice.mode == "fallback" and advice.attempts == MAX_LLM_ATTEMPTS
    assert "citation check" in advice.note
    assert "PB-99" not in advice.plan


def test_the_user_never_sees_an_unverified_plan(retriever, holdout, roi_params, profile):
    """Whatever the LLM does, the plan shown cites only entries that were retrieved."""
    replies = {"good": [valid_plan], "bad twice": [BAD_PLAN, BAD_PLAN], "error": [LLMError("down")],
               "no citations": ["Just do nice things.", "Still nothing."]}
    for name, scripted in replies.items():
        advice = _advisor(retriever, FakeLLM(scripted), holdout, roi_params).advise(profile, DRIVERS, CRITICAL)
        retrieved = [h.entry.id for h in advice.hits]
        assert check_citations(advice.plan, retrieved).ok, name


def test_llm_failure_falls_back_instead_of_crashing(retriever, holdout, roi_params, profile):
    llm = FakeLLM([LLMError("rate limited")])
    advice = _advisor(retriever, llm, holdout, roi_params).advise(profile, DRIVERS, CRITICAL)
    assert advice.mode == "fallback" and advice.attempts == 1
    assert "rate limited" in advice.note
    assert advice.plan.startswith("Recommended actions (taken directly from the playbook)")


def test_drivers_without_guidance_are_reported_and_passed_to_the_prompt(retriever, holdout, roi_params, profile):
    customer = {**profile, "MonthlyCharges": 58.0}
    drivers = [Driver("PaymentMethod_Electronic check", 0.5), Driver("tenure", 0.4), Driver("MonthlyCharges", 0.3)]
    llm = FakeLLM([valid_plan])
    advice = _advisor(retriever, llm, holdout, roi_params).advise(customer, drivers, HIGH)
    assert advice.uncovered == ["MonthlyCharges"]
    assert "no playbook entry" in llm.last_user and "MonthlyCharges" in llm.last_user


def test_no_usable_drivers_gives_a_clear_fallback_without_calling_the_llm(retriever, holdout, roi_params, profile):
    llm = FakeLLM(["must not be called"])
    advice = _advisor(retriever, llm, holdout, roi_params).advise(profile, [Driver("gender_Male", 1.0)], CRITICAL)
    assert advice.mode == "fallback" and llm.calls == 0
    assert "No actionable churn drivers" in advice.plan


def test_no_eligible_entries_gives_a_clear_fallback_without_calling_the_llm(retriever, holdout, roi_params, profile):
    customer = {**profile, "PaymentMethod": "Credit card (automatic)"}
    llm = FakeLLM(["must not be called"])
    advice = _advisor(retriever, llm, holdout, roi_params).advise(
        customer, [Driver("PaymentMethod_Credit card (automatic)", 0.6)], CRITICAL)
    # the only driver is not actionable for this customer, so nothing is usable to search with
    assert advice.mode == "fallback" and llm.calls == 0
