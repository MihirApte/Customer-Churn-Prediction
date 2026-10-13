"""Build the hand-labelling sheet for the retrieval evaluation.

Picks 25 real high-risk customers from the held-out test set. For each, it collects the
playbook entries returned by two retrieval methods:
  - the hybrid retrieval used in the app, and
  - an embedding-only baseline (one combined sentence, top 3, no metadata gate).
Every unique (customer, entry) pair appears once, in shuffled order, with NOTHING saying
which method returned it. You judge each pair; later, the key file is used to score both
methods fairly.

Run from the project root:
    python evaluation/make_label_sheet.py

Then open evaluation/labels/label_sheet.csv and fill the `relevant` column:
    1 = this entry's situation fits this customer AND its advice would be sensible for them
    0 = it does not fit this customer, or the advice would be odd for them
Judge only from the customer facts and the entry. Do not try to guess which method found it.
"""

import csv
import random

from common import LABELS_DIR, baseline_retrieve, describe_customer, load_context

from churn.advisor.decision import Verdict
from churn.advisor.playbook import load_playbook
from churn.advisor.query import driver_descriptions
from churn.advisor.retriever import Retriever, SentenceTransformerEmbedder
from churn.advisor.search import retrieve_for_customer

N_CUSTOMERS = 25
SAMPLE_SEED = 7

ctx = load_context()
retriever = Retriever(load_playbook(), SentenceTransformerEmbedder())

# Customers the advisor would actually search for: a paid offer is not ruled out by the cost check,
# and at least one churn driver can be described.
pool = [c for c in ctx.customers
        if c.decision.verdict != Verdict.NOT_JUSTIFIED and driver_descriptions(c.drivers, c.profile)]
print(f"{len(ctx.customers)} held-out customers; {len(pool)} reach retrieval; sampling {N_CUSTOMERS}.")
sample = random.Random(SAMPLE_SEED).sample(pool, N_CUSTOMERS)

pairs = {}
for c in sample:
    hybrid = retrieve_for_customer(retriever, c.drivers, c.profile).hits
    baseline = baseline_retrieve(retriever, c.drivers, c.profile, k=3)
    for system, hits in (("hybrid", hybrid), ("baseline", baseline)):
        for hit in hits:
            row = pairs.setdefault((c.cid, hit.entry.id), {"customer": c, "entry": hit.entry,
                                                           "hybrid": False, "baseline": False})
            row[system] = True

rows = list(pairs.values())
random.Random(SAMPLE_SEED).shuffle(rows)

LABELS_DIR.mkdir(parents=True, exist_ok=True)
with open(LABELS_DIR / "label_sheet.csv", "w", newline="", encoding="utf-8-sig") as f_sheet, \
     open(LABELS_DIR / "label_key.csv", "w", newline="", encoding="utf-8") as f_key:
    sheet = csv.writer(f_sheet)
    key = csv.writer(f_key)
    sheet.writerow(["pair_id", "customer_id", "customer_facts", "entry_id", "entry_title",
                    "entry_applies_when", "relevant"])
    key.writerow(["pair_id", "customer_id", "entry_id", "in_hybrid", "in_baseline"])
    for i, r in enumerate(rows, start=1):
        c, e = r["customer"], r["entry"]
        pair_id = f"P{i:03d}"
        sheet.writerow([pair_id, c.cid,
                        describe_customer(c.profile, c.score, c.decision.tier),
                        e.id, e.title, e.situation, ""])
        key.writerow([pair_id, c.cid, e.id, int(r["hybrid"]), int(r["baseline"])])

print(f"Wrote {len(rows)} pairs to {LABELS_DIR / 'label_sheet.csv'}")
print("Fill the `relevant` column with 1 or 0 (see the instructions at the top of this script),")
print("save the file, then run: python evaluation/run_eval.py")
