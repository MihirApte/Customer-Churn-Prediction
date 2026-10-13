"""Prompts for the retention plan. The numbers in them are computed by the system;
the LLM is told to use them as given and never to calculate its own."""

from .decision import Decision, Verdict

SYSTEM_PROMPT = """You are a retention advisor for a telecom company. You write a short action plan for ONE customer, using ONLY the playbook entries given in the user message.

Rules:
1. Every action must come from the "Actions" of a provided playbook entry and must end with that entry's id in square brackets, for example [PB-05].
2. Cite only ids that appear in the user message. Never invent an id.
3. Do not invent numbers, discounts, percentages or prices. Use only the figures given in the message.
4. Do not recommend different prices or eligibility because of age, gender or household status.
5. Follow the cost verdict in the message.
6. Write at most 4 actions. Be concrete and brief.
7. Churn rates in the message describe SIMILAR customers, not this customer. Say "similar customers", never "this customer's churn risk".

Reply in exactly this format:
Recommended actions:
1. <action> [PB-xx]
2. <action> [PB-xx]
Cost check: <one or two sentences using only the figures given>"""

VERDICT_INSTRUCTIONS = {
    Verdict.JUSTIFIED: "A paid retention offer is justified for this customer. You may recommend offers from the entries.",
    Verdict.MARGINAL: (
        "A paid retention offer is only marginal for this customer: the financial benefit is uncertain. "
        "Prefer the cheapest actions (calls, checks, reviews) over paid offers, and say plainly in the "
        "cost check that the benefit is uncertain."
    ),
}


def _eur(value: float, signed: bool = False) -> str:
    return f"EUR {value:+,.0f}" if signed else f"EUR {value:,.0f}"


def format_entry(entry) -> str:
    actions = "\n".join(f"  - {a}" for a in entry.actions)
    return (
        f"[{entry.id}] {entry.title}\n"
        f"Applies when: {entry.situation}\n"
        f"Actions:\n{actions}\n"
        f"Evidence: {entry.evidence}\n"
        f"Caution: {entry.caution}"
    )


def build_user_prompt(
    decision: Decision,
    churn_score: float,
    monthly_charge: float,
    driver_texts: list,
    hits: list,
    uncovered: list,
) -> str:
    d = decision
    lines = [
        "Customer",
        f"- Risk tier: {d.tier} (model score {churn_score:.2f})",
        f"- Monthly charge: {_eur(monthly_charge)}",
        "- Main churn drivers for this customer:",
    ]
    lines += [f"  {i}. {text}" for i, text in enumerate(driver_texts, start=1)]

    lines += [
        "",
        "Cost check (computed by the system, do not recalculate)",
        f"- Real churn rate of similar held-out customers: {d.churn_rate:.1%} "
        f"(95% range {d.rate_low:.1%} to {d.rate_high:.1%}, n={d.n})",
        f"- Annual revenue at risk: {_eur(d.annual_revenue)}",
        f"- Offer cost: {_eur(d.offer_cost)}; assumed retention rate: {d.retention_rate:.0%}",
        f"- Expected value of one paid offer: {_eur(d.expected_value, True)} "
        f"(range {_eur(d.ev_low, True)} to {_eur(d.ev_high, True)}); "
        f"break-even needs a real churn rate of {d.break_even_rate:.1%}",
        f"- Verdict: {d.verdict.value}. {VERDICT_INSTRUCTIONS[d.verdict]}",
        "",
        "Playbook entries you may use",
    ]
    lines += [format_entry(h.entry) + "\n" for h in hits]

    if uncovered:
        lines.append(f"Drivers with no playbook entry (do not give advice on these): {', '.join(uncovered)}")
        lines.append("")

    ids = ", ".join(h.entry.id for h in hits)
    lines.append(f"Allowed ids: {ids}. Write the plan now in the required format.")
    return "\n".join(lines)


def build_retry_prompt(original_prompt: str, reason: str, allowed_ids: list) -> str:
    return (
        f"{original_prompt}\n\nYour previous reply was rejected: {reason} "
        f"Cite only these ids: {', '.join(allowed_ids)}. Rewrite the plan in the required format."
    )
