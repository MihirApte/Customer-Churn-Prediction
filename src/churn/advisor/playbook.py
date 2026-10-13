"""Load and validate the retention playbook."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import PLAYBOOK_PATH

REQUIRED_KEYS = {"id", "title", "target_features", "situation", "actions", "evidence", "caution"}


@dataclass(frozen=True)
class PlaybookEntry:
    id: str
    title: str
    target_features: tuple  # model feature names this entry addresses
    situation: str          # WHO the entry applies to (this is what gets embedded)
    actions: tuple          # what to do
    evidence: str           # measured churn rates from the dataset
    caution: str
    match: str = "all"      # "all": every target family must apply to the customer; "any": one is enough

    @property
    def retrieval_text(self) -> str:
        """Text that is embedded for search: the title and the situation.

        Actions are left out on purpose. The query describes the customer, and the
        situation describes which customers an entry applies to, so those two
        should be compared. Actions describe what to do, not who it is for.
        """
        return f"{self.title}. {self.situation}"


def load_playbook(path: Path = PLAYBOOK_PATH) -> list:
    """Read playbook.json into PlaybookEntry objects, validating structure."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    seen_ids = set()
    for raw in data["entries"]:
        missing = REQUIRED_KEYS - set(raw)
        if missing:
            raise ValueError(f"Playbook entry {raw.get('id', '?')} is missing keys: {sorted(missing)}")
        if raw["id"] in seen_ids:
            raise ValueError(f"Duplicate playbook id: {raw['id']}")
        seen_ids.add(raw["id"])
        match = raw.get("match", "all")
        if match not in ("all", "any"):
            raise ValueError(f"Playbook entry {raw['id']} has match={match!r}; use 'all' or 'any'")
        entries.append(
            PlaybookEntry(
                id=raw["id"],
                title=raw["title"],
                target_features=tuple(raw["target_features"]),
                situation=raw["situation"],
                actions=tuple(raw["actions"]),
                evidence=raw["evidence"],
                caution=raw["caution"],
                match=match,
            )
        )
    if not entries:
        raise ValueError("Playbook has no entries")
    return entries
