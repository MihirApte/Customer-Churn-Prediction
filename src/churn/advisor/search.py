"""Find playbook entries for a customer: hybrid search (metadata filter + embeddings).

Two lessons shaped this module.

1. Searching with all drivers blended into one sentence lets the strongest topic
   crowd out the others. So each driver is searched on its own first, which
   guarantees every driver with guidance gets its best entry. A combined search then
   adds entries that fit the overall pattern (the "stacked risk" entries).

2. Embeddings alone can return advice that contradicts the customer: "pays by
   automatic credit card" retrieved "move to automatic payment", and "mailed check"
   retrieved the electronic-check entry. So an entry is only eligible if its
   `target_features` include the driver being searched (or, for the combined search,
   at least one of the customer's drivers). Embeddings then rank the eligible entries.
   A driver with no eligible entry is reported as uncovered instead of getting a
   wrong match.
"""

from typing import NamedTuple

from .query import applicable_families, base_field, build_query, driver_descriptions, skipped_drivers
from .retriever import Hit, Retriever

COMBINED = "combined"


def entry_applies(entry, applicable: set) -> bool:
    """Does the situation the entry describes hold for this customer?

    `applicable` is the set of feature families the customer has a playbook-relevant value
    for (see query.applicable_families). A stacked entry (for example fibre + month-to-month
    + new) needs ALL its conditions; an entry marked match="any" needs one. An entry with no
    target features (the low-cost fallback) is never returned by search."""
    families = {base_field(f) for f in entry.target_features}
    if not families:
        return False
    return bool(families & applicable) if entry.match == "any" else families <= applicable


class RetrievalResult(NamedTuple):
    hits: list        # list[Hit], best first
    uncovered: list   # driver features with no eligible playbook entry


def retrieve_for_customer(
    retriever: Retriever,
    drivers,
    profile: dict,
    combined_k: int = 2,
    max_entries: int = 4,
) -> RetrievalResult:
    """Choose the playbook entries to advise on for one customer.

    1. For every usable driver, take the best-ranked entry that targets that driver.
    2. Search with the combined sentence, restricted to entries that target at least
       one of the customer's drivers, and add the top `combined_k` while there is
       room (an entry already chosen just gains the extra label).
    Entries that answer more drivers come first, then higher similarity.
    Driver coverage takes precedence over `max_entries`.
    """
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")

    parts = driver_descriptions(drivers, profile)
    if not parts:
        raise ValueError("No usable churn drivers to search with")

    # Only drivers that apply to this customer's actual values take part in the search.
    usable_bases = {base_field(f) for f, _ in parts}
    all_driver_features = {d.feature for d in drivers if base_field(d.feature) in usable_bases}
    applicable = applicable_families(profile)

    def applies(entry) -> bool:
        return entry_applies(entry, applicable)

    selected: dict = {}
    uncovered: list = list(skipped_drivers(drivers, profile))

    def add(hit: Hit, label: str) -> None:
        existing = selected.get(hit.entry.id)
        if existing is None:
            selected[hit.entry.id] = Hit(hit.entry, hit.score, (label,))
        else:
            labels = existing.matched_on
            if label not in labels:
                labels = labels + (label,)
            selected[hit.entry.id] = Hit(existing.entry, max(existing.score, hit.score), labels)

    for feature, sentence in parts:
        hits = retriever.retrieve(
            sentence + ".", k=1, allowed=lambda e, f=feature: f in e.target_features and applies(e)
        )
        if hits:
            add(hits[0], feature)
        else:
            uncovered.append(feature)

    combined_query = build_query(drivers, profile)
    for hit in retriever.retrieve(
        combined_query,
        k=combined_k,
        allowed=lambda e: bool(all_driver_features & set(e.target_features)) and applies(e),
    ):
        if hit.entry.id in selected or len(selected) < max_entries:
            add(hit, COMBINED)

    def rank(h: Hit):
        driver_matches = sum(1 for label in h.matched_on if label != COMBINED)
        return (driver_matches, h.score)

    return RetrievalResult(sorted(selected.values(), key=rank, reverse=True), uncovered)
