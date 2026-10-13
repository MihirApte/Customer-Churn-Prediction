from types import SimpleNamespace

import pytest

from churn import config
from churn.advisor.llm import (
    DEFAULT_GROQ_MODEL,
    GroqClient,
    LLMError,
    MissingAPIKeyError,
    UnavailableLLM,
)


def test_missing_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
        GroqClient()


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    with pytest.raises(MissingAPIKeyError):
        GroqClient()


def test_key_is_read_from_the_environment_and_default_model_is_used(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert GroqClient().model == DEFAULT_GROQ_MODEL


def test_model_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")
    assert GroqClient().model == "some/other-model"


class _FakeChat:
    def __init__(self, content=None, error=None):
        self._content, self._error = content, error
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client_with(chat):
    client = GroqClient(api_key="test-key", model="m")
    client._client = SimpleNamespace(chat=chat)  # skip the real SDK and the network
    return client


def test_complete_returns_the_stripped_reply():
    assert _client_with(_FakeChat(content="  hello  ")).complete("sys", "usr") == "hello"


def test_complete_wraps_sdk_errors_in_llmerror():
    with pytest.raises(LLMError, match="Groq request failed"):
        _client_with(_FakeChat(error=RuntimeError("rate limited"))).complete("sys", "usr")


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_reply_is_an_error(content):
    with pytest.raises(LLMError, match="empty"):
        _client_with(_FakeChat(content=content)).complete("sys", "usr")


def test_unavailable_llm_always_fails_with_its_reason():
    with pytest.raises(LLMError, match="no key here"):
        UnavailableLLM("no key here").complete("sys", "usr")


def test_loading_a_missing_env_file_is_harmless(tmp_path):
    config.load_env_file(tmp_path / "does-not-exist.env")  # must not raise
