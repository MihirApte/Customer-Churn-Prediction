"""Small, tested helpers used by the evaluation scripts in evaluation/."""

import math
import re

from .advisor.decision import wilson_interval

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_LIST_NUMBERS = {"1", "2", "3", "4"}  # numbering of the plan's actions, not claims


def numbers_in(text: str) -> list:
    """Every number written in `text`, as plain strings ('1,140' -> '1140')."""
    found = []
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        found.append(cleaned)
    return found


def unsupported_numbers(plan: str, prompt: str) -> list:
    """Numbers the plan states that do not appear in the prompt it was written from.

    A plan number is supported if it equals a prompt number, or equals a prompt number
    rounded to a whole number (the model may say 61% for 60.6%). The numbers 1-4 are
    ignored because they number the actions. Anything left may have been invented."""
    prompt_numbers = set(numbers_in(prompt))
    rounded = set()
    for n in prompt_numbers:
        try:
            rounded.add(str(int(round(float(n)))))
        except ValueError:
            pass
    return [
        n for n in numbers_in(plan)
        if n not in prompt_numbers and n not in rounded and n not in _LIST_NUMBERS
    ]


def proportion_with_ci(successes: int, n: int) -> tuple:
    """(rate, low, high): a proportion with its 95% Wilson range. All zeros if n == 0."""
    if n == 0:
        return 0.0, 0.0, 0.0
    low, high = wilson_interval(successes, n)
    return successes / n, low, high


def percentile(values, q: float) -> float:
    """q-th percentile (0-100) by linear interpolation. Raises on an empty list."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
