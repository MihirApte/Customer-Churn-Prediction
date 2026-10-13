"""Measure the retention advisor on real held-out customers and write evaluation/results.md.

Three parts (every number printed is measured here; nothing is typed in by hand):
  1. Coverage and rule violations, automatic, on every held-out customer that reaches retrieval.
  2. Hand-labelled precision and pooled recall, from the label sheet you filled in
     (make it first with: python evaluation/make_label_sheet.py).
  3. LLM behaviour with real Groq calls (only if you pass --llm N).

Run from the project root:
    python evaluation/run_eval.py              # parts 1 and 2
    python evaluation/run_eval.py --llm 40     # also part 3 (needs GROQ_API_KEY in .env)
"""

import argparse
import csv
import datetime
import random
import statistics
import time

from common import EVAL_DIR, LABELS_DIR, baseline_retrieve, load_context

from churn.advisor.decision import Verdict
from churn.advisor.llm import GroqClient, LLMError
from churn.advisor.playbook import load_playbook
from churn.advisor.query import applicable_families, driver_descriptions
from churn.advisor.retriever import Retriever, SentenceTransformerEmbedder
from churn.advisor.search import entry_applies, retrieve_for_customer
from churn.advisor.service import RetentionAdvisor
from churn.evaluation_metrics import percentile, proportion_with_ci, unsupported_numbers

parser = argparse.ArgumentParser()
parser.add_argument("--llm", type=int, default=0, help="number of customers for the real-LLM run (0 = skip)")
parser.add_argument("--no-labels", action="store_true", help="skip the hand-labelled part")
args = parser.parse_args()

report = []


def out(line=""):
    print(line)
    report.append(line)


def pct(rate, low, high):
    return f"{rate:.1%} (95% range {low:.1%} to {high:.1%})"


ctx = load_context()
retriever = Retriever(load_playbook(), SentenceTransformerEmbedder())
pool = [c for c in ctx.customers
        if c.decision.verdict != Verdict.NOT_JUSTIFIED and driver_descriptions(c.drivers, c.profile)]

out("# Retention advisor evaluation")
out(f"Run on {datetime.date.today()} using {len(ctx.customers)} held-out test customers "
    f"(the model never trained on them). {len(pool)} of them reach retrieval; the others are "
    f"ruled out by the cost check or have no describable driver.")
out()

# ---------------------------------------------------------------- 1. automatic coverage
out("## 1. Coverage and rule violations (automatic, all customers that reach retrieval)")
out()
stats = {name: dict(entries=0, with_hits=0, covered=0, inapplicable=0) for name in ("hybrid", "baseline")}
total_drivers = 0
for c in pool:
    hybrid = retrieve_for_customer(retriever, c.drivers, c.profile).hits
    baseline = baseline_retrieve(retriever, c.drivers, c.profile, k=3)
    families = applicable_families(c.profile)
    total_drivers += len(c.drivers)
    for name, hits in (("hybrid", hybrid), ("baseline", baseline)):
        s = stats[name]
        s["entries"] += len(hits)
        s["with_hits"] += 1 if hits else 0
        s["inapplicable"] += sum(1 for h in hits if not entry_applies(h.entry, families))
        s["covered"] += sum(1 for d in c.drivers if any(d.feature in h.entry.target_features for h in hits))

n = len(pool)
out(f"Customers: {n}. Top SHAP drivers considered (gender excluded): {total_drivers}.")
out()
out("| Measure | Hybrid retrieval (app) | Embedding-only baseline |")
out("|---|---|---|")
rows = [
    ("Customers with at least one entry",
     *[pct(*proportion_with_ci(stats[k]["with_hits"], n)) for k in ("hybrid", "baseline")]),
    ("Top drivers that got matching guidance",
     *[pct(*proportion_with_ci(stats[k]["covered"], total_drivers)) for k in ("hybrid", "baseline")]),
    ("Returned entries whose conditions do not hold for the customer",
     *[pct(*proportion_with_ci(stats[k]["inapplicable"], stats[k]["entries"])) for k in ("hybrid", "baseline")]),
    ("Average entries per customer",
     *[f"{stats[k]['entries'] / n:.2f}" for k in ("hybrid", "baseline")]),
]
for label, a, b in rows:
    out(f"| {label} | {a} | {b} |")
out()
out("Note: the hybrid method rules out inapplicable entries by construction, so its 0% on the "
    "third row confirms the rule is enforced, not that the method is accurate. The baseline "
    "figure shows how often embeddings alone return an entry that does not fit the customer.")
out()

# ---------------------------------------------------------------- 2. hand-labelled precision
if not args.no_labels:
    out("## 2. Hand-labelled retrieval quality")
    out()
    sheet_path, key_path = LABELS_DIR / "label_sheet.csv", LABELS_DIR / "label_key.csv"
    if not (sheet_path.exists() and key_path.exists()):
        out("Label sheet not found. Run `python evaluation/make_label_sheet.py` first.")
    else:
        with open(sheet_path, newline="", encoding="utf-8-sig") as f:
            labels = {r["pair_id"]: r["relevant"].strip() for r in csv.DictReader(f)}
        with open(key_path, newline="", encoding="utf-8") as f:
            key = list(csv.DictReader(f))
        bad = [pid for pid, v in labels.items() if v not in ("0", "1")]
        if bad:
            raise SystemExit(f"{len(bad)} rows in label_sheet.csv are not labelled 0 or 1 "
                             f"(first: {bad[:5]}). Fill every `relevant` cell, save, and run again.")
        joined = [dict(r, relevant=int(labels[r["pair_id"]]), hybrid=int(r["in_hybrid"]),
                       baseline=int(r["in_baseline"])) for r in key]
        n_customers = len({r["customer_id"] for r in joined})
        relevant_total = sum(r["relevant"] for r in joined)
        out(f"{len(joined)} (customer, entry) pairs judged by the project author for {n_customers} customers. "
            f"The sheet did not say which method returned which pair.")
        out()
        out("| Measure | Hybrid retrieval (app) | Embedding-only baseline |")
        out("|---|---|---|")
        prec, rec_, sizes = [], [], []
        for system in ("hybrid", "baseline"):
            mine = [r for r in joined if r[system]]
            sizes.append(len(mine))
            prec.append(pct(*proportion_with_ci(sum(r["relevant"] for r in mine), len(mine))))
            found = sum(1 for r in joined if r["relevant"] and r[system])
            rec_.append(pct(*proportion_with_ci(found, relevant_total)))
        out(f"| Pairs returned (n) | {sizes[0]} | {sizes[1]} |")
        out(f"| Precision: returned entries judged relevant | {prec[0]} | {prec[1]} |")
        out(f"| Pooled recall: relevant entries found (of all judged relevant) | {rec_[0]} | {rec_[1]} |")
        out()
        out("Pooled recall counts only entries that at least one method returned, so it understates "
            "what neither method found. The sample is small and the labeller also wrote the playbook, "
            "so treat the ranges as the honest uncertainty.")
        wrong = [r for r in joined if r["hybrid"] and not r["relevant"]]
        if wrong:
            out()
            out("Entries the hybrid method returned that you judged not relevant (review these):")
            for r in wrong:
                out(f"- customer {r['customer_id']}: {r['entry_id']}")
    out()

# ---------------------------------------------------------------- 3. real LLM run
if args.llm > 0:
    out("## 3. LLM behaviour (real Groq calls)")
    out()

    class RecordingLLM:
        def __init__(self, inner):
            self.inner, self.prompts, self.latencies = inner, [], []

        def complete(self, system, user):
            start = time.perf_counter()
            try:
                return self.inner.complete(system, user)
            finally:
                self.latencies.append(time.perf_counter() - start)
                self.prompts.append(user)

    try:
        groq = GroqClient()
    except LLMError as exc:
        raise SystemExit(f"{exc}\nCannot run part 3 without a key.")
    recorder = RecordingLLM(groq)
    advisor = RetentionAdvisor(retriever, recorder, ctx.y_true, ctx.y_prob, ctx.roi_params)
    sample = random.Random(11).sample(pool, min(args.llm, len(pool)))

    records, all_latencies = [], []
    for i, c in enumerate(sample, start=1):
        recorder.prompts.clear()
        recorder.latencies.clear()
        advice = advisor.advise(c.profile, c.drivers, c.score)
        llm_error = "could not be used" in advice.note
        invented = unsupported_numbers(advice.plan, recorder.prompts[-1]) if advice.mode == "llm" else []
        actions = sum(1 for line in advice.plan.splitlines() if line[:2].strip().rstrip(".").isdigit())
        all_latencies += recorder.latencies
        records.append(dict(cid=c.cid, score=round(c.score, 3), tier=c.decision.tier,
                            verdict=c.decision.verdict.value, mode=advice.mode, attempts=advice.attempts,
                            llm_error=int(llm_error), actions=actions,
                            unsupported_numbers=" ".join(invented),
                            error=advice.note if llm_error else "", plan=advice.plan))
        print(f"  {i}/{len(sample)} customer {c.cid}: {advice.mode}, {advice.attempts} call(s)")
        time.sleep(3)  # stay inside the free-tier rate limit

    called = [r for r in records if r["attempts"] > 0]
    errors = sum(r["llm_error"] for r in called)
    replied = [r for r in called if not r["llm_error"]]  # the model actually answered
    first_pass = sum(1 for r in replied if r["mode"] == "llm" and r["attempts"] == 1)
    retry_pass = sum(1 for r in replied if r["mode"] == "llm" and r["attempts"] > 1)
    cite_fallback = sum(1 for r in replied if r["mode"] == "fallback")
    llm_plans = [r for r in replied if r["mode"] == "llm"]
    with_invented = sum(1 for r in llm_plans if r["unsupported_numbers"])

    out(f"Model: {groq.model}. {len(records)} customers sampled (seed 11); {len(called)} reached the LLM "
        f"({len(records) - len(called)} had no eligible playbook entries); the model answered "
        f"{len(replied)} of them and {errors} call(s) failed with a Groq error.")
    out()
    out("| Measure (among customers where the model answered) | Result |")
    out("|---|---|")
    out(f"| Plan passed the citation check on the first attempt | "
        f"{first_pass} of {len(replied)}: {pct(*proportion_with_ci(first_pass, len(replied)))} |")
    out(f"| Passed only after one retry | {retry_pass} of {len(replied)} |")
    out(f"| Fell back to a playbook-built plan after failed citations | {cite_fallback} of {len(replied)} |")
    out(f"| LLM plans containing a number not given in the prompt | "
        f"{pct(*proportion_with_ci(with_invented, len(llm_plans)))} |")
    if all_latencies:
        out(f"| LLM call latency, median / 95th percentile | "
            f"{statistics.median(all_latencies):.1f}s / {percentile(all_latencies, 95):.1f}s "
            f"(n={len(all_latencies)} calls) |")
    out()
    out("The plan the user sees always passes the citation check: failures are replaced by a "
        "playbook-built plan, so the first-attempt rate measures the model, not the safety net. "
        "Numbers are matched against the prompt allowing whole-number rounding; see "
        "evaluation/results_llm.csv for the flagged cases.")
    if errors:
        out()
        out(f"Note: {errors} call(s) failed with a Groq error (see the `error` column in "
            "evaluation/results_llm.csv). The app falls back to a playbook-built plan in that case. "
            "Those customers are left out of the rates above because the model never answered.")
    with open(EVAL_DIR / "results_llm.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

(EVAL_DIR / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print(f"\nWrote {EVAL_DIR / 'results.md'}")
