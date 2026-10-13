import pytest

from churn.advisor.query import applicable_families
from churn.advisor.search import entry_applies
from churn.evaluation_metrics import numbers_in, percentile, proportion_with_ci, unsupported_numbers


def test_numbers_are_extracted_without_thousands_separators():
    assert numbers_in("EUR 1,140 at 60.6% with n=315.") == ["1140", "60.6", "315"]


def test_numbers_present_in_the_prompt_are_supported():
    prompt = "Expected value EUR +73; break-even 31.7%; similar customers churn 60.6% (n=315)."
    plan = "EV is EUR 73, above the 31.7% break-even, vs 60.6% for similar customers."
    assert unsupported_numbers(plan, prompt) == []


def test_a_rounded_prompt_number_is_still_supported():
    assert unsupported_numbers("about 61% churn", "churn rate 60.6%") == []


def test_an_invented_number_is_flagged():
    prompt = "Expected value EUR +73; break-even 31.7%."
    assert unsupported_numbers("Offer a 15% discount worth EUR 99", prompt) == ["15", "99"]


def test_action_numbering_is_not_counted_as_a_claim():
    assert unsupported_numbers("1. Do A\n2. Do B\n3. Do C\n4. Do D", "no numbers here") == []


def test_proportion_with_ci():
    rate, low, high = proportion_with_ci(8, 10)
    assert rate == pytest.approx(0.8) and low < 0.8 < high
    assert proportion_with_ci(0, 0) == (0.0, 0.0, 0.0)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10, 20], 50) == pytest.approx(15)
    assert percentile([5], 95) == 5
    with pytest.raises(ValueError):
        percentile([], 50)


def test_entry_applies_uses_all_or_any_semantics(entries, profile):
    by_id = {e.id: e for e in entries}
    families = applicable_families(profile)  # fibre, month-to-month, new, e-check, paperless ...
    assert entry_applies(by_id["PB-16"], families)                       # all of its conditions hold
    assert not entry_applies(by_id["PB-16"], families - {"Contract"})    # one condition missing
    assert entry_applies(by_id["PB-10"], {"StreamingTV"})                # "any" needs only one
    assert not entry_applies(by_id["PB-19"], families)                   # no target features
