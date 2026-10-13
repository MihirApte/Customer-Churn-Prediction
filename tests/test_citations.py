import pytest

from churn.advisor.citations import check_citations, extract_citations


def test_extracts_unique_ids_in_order_of_first_appearance():
    text = "1. Do A [PB-05]\n2. Do B [PB-01]\n3. Do C [PB-05]"
    assert extract_citations(text) == ("PB-05", "PB-01")


def test_passes_when_every_citation_was_retrieved():
    check = check_citations("[PB-01] and [PB-05]", ["PB-01", "PB-05", "PB-08"])
    assert check.ok and check.cited == ("PB-01", "PB-05") and check.invalid == ()


def test_fails_when_nothing_is_cited():
    check = check_citations("A plan with no citations at all.", ["PB-01"])
    assert not check.ok and "no playbook entries" in check.reason


def test_fails_on_an_entry_that_was_not_retrieved():
    check = check_citations("Do X [PB-99]", ["PB-01"])
    assert not check.ok and check.invalid == ("PB-99",)


def test_one_bad_citation_fails_the_whole_plan_even_if_others_are_valid():
    check = check_citations("[PB-01] and also [PB-07]", ["PB-01"])
    assert not check.ok and check.invalid == ("PB-07",)


@pytest.mark.parametrize("text", ["see PB-5", "see PB-005", "see pb-01"])
def test_near_miss_ids_are_not_accepted(text):
    assert not check_citations(text, ["PB-05", "PB-01"]).ok


def test_unbracketed_ids_are_still_detected():
    assert check_citations("Use PB-01 here", ["PB-01"]).ok
    assert not check_citations("Use PB-42 here", ["PB-01"]).ok


def test_typographic_hyphens_are_read_as_normal_hyphens():
    assert check_citations("Do it [PB‑05] then [PB–01]", ["PB-05", "PB-01"]).ok
    assert not check_citations("bad [PB‑99]", ["PB-05"]).ok
