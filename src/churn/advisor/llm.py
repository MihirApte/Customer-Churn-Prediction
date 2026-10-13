"""LLM access behind a tiny interface, so the provider can change and tests need no network."""

import os
from typing import Protocol

# llama-3.3-70b-versatile was retired by Groq on 16 Aug 2026. Check console.groq.com/docs/models
# for current model ids and override with the GROQ_MODEL environment variable if this one changes.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class LLMError(Exception):
    """The LLM could not be reached or returned nothing usable."""


class MissingAPIKeyError(LLMError):
    """GROQ_API_KEY is not set."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's reply to a system + user prompt."""


class UnavailableLLM:
    """Stand-in used when no API key is configured. Every call fails with a clear
    reason, so the advisor falls back to a plan built directly from the playbook
    and the app keeps working."""

    def __init__(self, reason: str):
        self.reason = reason

    def complete(self, system: str, user: str) -> str:
        raise LLMError(self.reason)


class GroqClient:
    """Calls Groq's chat API. The key is read from the GROQ_API_KEY environment
    variable (set locally in .env, or as a Space secret on HuggingFace). It is never
    written in code or logged."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,  # generous: reasoning models spend part of this on hidden thinking
        timeout: float = 30.0,
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        if not self._api_key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Add it to .env locally, or as a secret on HuggingFace Spaces."
            )
        self.model = model or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def complete(self, system: str, user: str) -> str:
        try:
            if self._client is None:
                from groq import Groq

                self._client = Groq(api_key=self._api_key, timeout=self.timeout)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # network, rate limit, bad model name, etc.
            raise LLMError(f"Groq request failed: {type(exc).__name__}: {exc}") from exc

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("Groq returned an empty reply.")
        return text
