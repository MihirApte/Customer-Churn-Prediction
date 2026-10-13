import json
import re

import pytest

from churn.advisor.playbook import load_playbook
from churn.features import FEATURE_COLS


def test_real_playbook_loads_with_unique_ids(entries):
    ids = [e.id for e in entries]
    assert len(ids) >= 15
    assert len(ids) == len(set(ids))


def test_every_entry_is_complete(entries):
    for e in entries:
        assert e.title.strip() and e.situation.strip() and e.caution.strip(), e.id
        assert len(e.actions) >= 1 and all(a.strip() for a in e.actions), e.id
        assert re.search(r"\d", e.evidence), f"{e.id} evidence should contain a measured number"


def test_target_features_are_real_model_features(entries):
    for e in entries:
        for feature in e.target_features:
            assert feature in FEATURE_COLS, f"{e.id} targets unknown feature {feature}"


def test_low_cost_fallback_entry_exists_and_targets_nothing(entries):
    by_id = {e.id: e for e in entries}
    assert "PB-19" in by_id
    assert by_id["PB-19"].target_features == ()


def test_retrieval_text_uses_title_and_situation_but_not_actions(entries):
    e = entries[0]
    assert e.title in e.retrieval_text and e.situation in e.retrieval_text
    assert e.actions[0] not in e.retrieval_text


def test_playbook_never_mentions_gender(entries):
    text = " ".join(f"{e.title} {e.situation} {' '.join(e.actions)} {e.caution}" for e in entries).lower()
    assert not re.search(r"\b(gender|male|female|men|women)\b", text)


def _write(tmp_path, data):
    path = tmp_path / "playbook.json"
    path.write_text(json.dumps(data))
    return path


GOOD_ENTRY = {"id": "PB-01", "title": "t", "target_features": [], "situation": "s",
              "actions": ["a"], "evidence": "e 1", "caution": "c"}


def test_loader_rejects_missing_keys(tmp_path):
    bad = {k: v for k, v in GOOD_ENTRY.items() if k != "actions"}
    with pytest.raises(ValueError, match="missing keys"):
        load_playbook(_write(tmp_path, {"entries": [bad]}))


def test_loader_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        load_playbook(_write(tmp_path, {"entries": [GOOD_ENTRY, GOOD_ENTRY]}))


def test_loader_rejects_empty_playbook(tmp_path):
    with pytest.raises(ValueError, match="no entries"):
        load_playbook(_write(tmp_path, {"entries": []}))


def test_match_defaults_to_all_and_streaming_entry_is_any(entries):
    by_id = {e.id: e for e in entries}
    assert all(e.match in ("all", "any") for e in entries)
    assert by_id["PB-10"].match == "any"
    assert by_id["PB-16"].match == "all"


def test_loader_rejects_an_unknown_match_value(tmp_path):
    bad = {**GOOD_ENTRY, "match": "some"}
    with pytest.raises(ValueError, match="match"):
        load_playbook(_write(tmp_path, {"entries": [bad]}))
