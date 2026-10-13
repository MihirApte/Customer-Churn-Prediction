"""Check that an LLM-written plan cites only playbook entries that were retrieved."""

import re
from typing import NamedTuple

# Any token that looks like a playbook id, bracketed or not. Strict on purpose:
# "PB-5" or "PB-99" counts as an invalid citation, never as a harmless typo.
ID_PATTERN = re.compile(r"\bPB-\d+\b")


class CitationCheck(NamedTuple):
    ok: bool            # at least one citation, and every citation was retrieved
    cited: tuple        # ids cited, in order of first appearance
    invalid: tuple      # cited ids that were NOT retrieved
    reason: str         # empty if ok


# LLMs sometimes write typographic hyphens (U+2010 to U+2015, minus sign U+2212),
# so "PB‑05" must still be read as "PB-05".
_FANCY_HYPHENS = re.compile("[‐‑‒–—―−]")


def extract_citations(text: str) -> tuple:
    """Playbook ids mentioned in `text`, unique, in order of first appearance."""
    text = _FANCY_HYPHENS.sub("-", text)
    seen = []
    for match in ID_PATTERN.findall(text):
        if match not in seen:
            seen.append(match)
    return tuple(seen)


def check_citations(text: str, retrieved_ids) -> CitationCheck:
    """Pass only if the plan cites at least one entry and cites nothing outside `retrieved_ids`."""
    allowed = set(retrieved_ids)
    cited = extract_citations(text)
    invalid = tuple(c for c in cited if c not in allowed)

    if not cited:
        return CitationCheck(False, cited, invalid, "The plan cites no playbook entries.")
    if invalid:
        return CitationCheck(False, cited, invalid, f"The plan cites entries that were not retrieved: {', '.join(invalid)}.")
    return CitationCheck(True, cited, invalid, "")
