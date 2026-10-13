"""The retention advisor: ties the cost decision, retrieval, the LLM and the citation check together.

Flow for one customer:
  1. Decide in code whether a paid offer is worth it (decision.py).
  2. NOT_JUSTIFIED  -> no search, no LLM. Recommend the low-cost touch (playbook PB-19).
  3. Otherwise search the playbook (hybrid retrieval), ask the LLM for a plan that cites
     only the retrieved entries, and check the citations.
  4. If the citation check fails, retry once. If it fails again, or the LLM is
     unreachable, fall back to a plan built directly from the retrieved entries, so the
     user never sees an unchecked LLM plan and the app never breaks.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from .citations import CitationCheck, check_citations
from .decision import Decision, Verdict, decide_intervention
from .llm import LLMClient, LLMError
from .prompts import SYSTEM_PROMPT, build_retry_prompt, build_user_prompt
from .query import build_query, driver_descriptions
from .retriever import Retriever
from .search import retrieve_for_customer

LOW_COST_ENTRY_ID = "PB-19"
MAX_LLM_ATTEMPTS = 2


@dataclass
class Advice:
    mode: str                      # "no_offer", "llm" or "fallback"
    decision: Decision
    plan: str
    query: str = ""
    hits: list = field(default_factory=list)
    uncovered: list = field(default_factory=list)
    citation: Optional[CitationCheck] = None
    attempts: int = 0              # LLM calls made
    llm_seconds: float = 0.0
    note: str = ""                 # plain-English explanation for the user


def _eur(x: float, signed: bool = False) -> str:
    return f"EUR {x:+,.0f}" if signed else f"EUR {x:,.0f}"


def format_entries_plan(hits: list, limit: int = 4) -> str:
    """A plan built straight from the retrieved entries (no LLM): the first action of each."""
    lines = ["Recommended actions (taken directly from the playbook):"]
    for i, hit in enumerate(hits[:limit], start=1):
        lines.append(f"{i}. {hit.entry.actions[0]} [{hit.entry.id}]")
    return "\n".join(lines)


class RetentionAdvisor:
    def __init__(self, retriever: Retriever, llm: LLMClient, y_true, y_prob, roi_params: dict):
        self.retriever = retriever
        self.llm = llm
        self.y_true = y_true
        self.y_prob = y_prob
        self.retention_rate = roi_params["retention_rate"]
        self.offer_cost = roi_params["offer_cost"]
        self._entries = {e.id: e for e in retriever.entries}

    def _no_offer_plan(self, d: Decision) -> str:
        head = (
            f"No paid retention offer is recommended. For a customer in the {d.tier} tier an offer is expected "
            f"to lose money: {_eur(d.expected_value, True)} per offer (real churn rate of similar customers "
            f"{d.churn_rate:.1%}, n={d.n}; an offer only breaks even above {d.break_even_rate:.1%})."
        )
        entry = self._entries.get(LOW_COST_ENTRY_ID)
        if entry is None:
            return head
        actions = "\n".join(f"{i}. {a} [{entry.id}]" for i, a in enumerate(entry.actions, start=1))
        return f"{head}\n\nLow-cost alternative:\n{actions}"

    def advise(self, profile: dict, drivers: list, churn_score: float) -> Advice:
        d = decide_intervention(
            self.y_true, self.y_prob, churn_score, profile["MonthlyCharges"],
            self.retention_rate, self.offer_cost,
        )

        if d.verdict == Verdict.NOT_JUSTIFIED:
            return Advice(
                mode="no_offer", decision=d, plan=self._no_offer_plan(d),
                note="Decided by the cost check, so no search or LLM call was made.",
            )

        parts = driver_descriptions(drivers, profile)
        if not parts:
            return Advice(
                mode="fallback", decision=d,
                plan="No actionable churn drivers were found for this customer, so no playbook advice applies.",
                note="No usable drivers.",
            )

        query = build_query(drivers, profile)
        result = retrieve_for_customer(self.retriever, drivers, profile)
        if not result.hits:
            return Advice(
                mode="fallback", decision=d, query=query, uncovered=result.uncovered,
                plan="No playbook entry covers this customer's churn drivers, so no advice is given.",
                note="Retrieval found nothing eligible.",
            )

        allowed_ids = [h.entry.id for h in result.hits]
        prompt = build_user_prompt(
            d, churn_score, profile["MonthlyCharges"], [text for _, text in parts],
            result.hits, result.uncovered,
        )

        attempts, llm_seconds, check, error = 0, 0.0, None, ""
        current = prompt
        while attempts < MAX_LLM_ATTEMPTS:
            attempts += 1
            start = time.perf_counter()
            try:
                text = self.llm.complete(SYSTEM_PROMPT, current)
            except LLMError as exc:
                error = str(exc)
                llm_seconds += time.perf_counter() - start
                break
            llm_seconds += time.perf_counter() - start

            check = check_citations(text, allowed_ids)
            if check.ok:
                return Advice(
                    mode="llm", decision=d, plan=text, query=query, hits=result.hits,
                    uncovered=result.uncovered, citation=check, attempts=attempts,
                    llm_seconds=llm_seconds,
                    note="Plan written by the LLM from the retrieved entries; citations verified.",
                )
            current = build_retry_prompt(prompt, check.reason, allowed_ids)

        note = (
            f"The LLM could not be used ({error}), so the plan is built directly from the playbook."
            if error
            else f"The LLM plan failed the citation check ({check.reason}) after {attempts} attempts, "
                 "so the plan is built directly from the playbook."
        )
        return Advice(
            mode="fallback", decision=d, plan=format_entries_plan(result.hits), query=query,
            hits=result.hits, uncovered=result.uncovered, citation=check, attempts=attempts,
            llm_seconds=llm_seconds, note=note,
        )
