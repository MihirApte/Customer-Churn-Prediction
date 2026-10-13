"""Shared test helpers: a fake embedder, a fake LLM and synthetic held-out data.

None of these touch the network, the real embedding model or the Groq API, so the
whole suite runs in seconds with no API key.
"""

import hashlib
import re
from collections import namedtuple

import numpy as np

from churn.advisor.llm import LLMError

Driver = namedtuple("Driver", "feature shap_value")

BASE_PROFILE = {
    "SeniorCitizen": 0, "tenure": 3, "MonthlyCharges": 70.0, "TotalCharges": 210.0,
    "gender": "Male", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
}


def hashing_embedder(dim: int = 512):
    """Deterministic bag-of-words embedder. It only captures shared words, not meaning,
    so tests built on it check the retrieval LOGIC (filters, ordering, caps), never
    semantic quality."""

    def embed(texts):
        out = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1
        return out

    return embed


def make_holdout():
    """Synthetic held-out set whose four risk tiers have the same size and churn rate as
    the real test set: LOW 473 / 3.6%, MEDIUM 327 / 16.8%, HIGH 292 / 38.0%, CRITICAL 315 / 60.6%."""
    y_true, y_prob = [], []
    for n, rate, lo, hi in [(473, 0.036, 0.0, 0.11), (327, 0.168, 0.11, 0.40),
                            (292, 0.380, 0.40, 0.70), (315, 0.606, 0.70, 1.0)]:
        churners = round(n * rate)
        y_true += [1] * churners + [0] * (n - churners)
        y_prob += list(np.linspace(lo + 1e-3, hi - 1e-3, n))
    return np.array(y_true), np.array(y_prob)


class FakeLLM:
    """Returns scripted replies in order. A reply that is an Exception is raised instead.
    Records how many calls were made and the last prompts it received."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_system = None
        self.last_user = None

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.last_system, self.last_user = system, user
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):  # a reply built from the prompt it receives
            return reply(user)
        return reply


def valid_plan(user_prompt: str) -> str:
    """A plan that cites exactly the ids the prompt allows, whatever retrieval chose."""
    allowed = re.search(r"Allowed ids: ([^.]*)\.", user_prompt).group(1).split(", ")
    lines = [f"{i}. Do something sensible [{entry_id}]" for i, entry_id in enumerate(allowed, start=1)]
    return "Recommended actions:\n" + "\n".join(lines) + "\nCost check: as computed."


GOOD_PLAN = (
    "Recommended actions:\n"
    "1. Make a proactive welcome check-in call [PB-01]\n"
    "2. Run a proactive line and speed check [PB-05]\n"
    "Cost check: the expected value is positive."
)
BAD_PLAN = "Recommended actions:\n1. Do something [PB-99]\nCost check: fine."

__all__ = ["Driver", "BASE_PROFILE", "hashing_embedder", "make_holdout", "FakeLLM", "valid_plan",
           "GOOD_PLAN", "BAD_PLAN", "LLMError"]
