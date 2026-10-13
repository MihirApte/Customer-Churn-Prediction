"""Read-only audit: which decision threshold is best under three different cost models?

Notebook 06 picked the 0.11 threshold using a cost matrix that gives a caught churner
a cost of (offer - retention x annual revenue) = -150 EUR, while a missed churner costs
+768 EUR. Those two numbers are not on the same footing: the -150 is a SAVING relative to
doing nothing, the 768 is an absolute loss. This script compares three accountings on
the held-out test set and prints the best threshold for each. It changes nothing.

  A) Simple:     missed churner = 12 months revenue, false alarm = 50 EUR, caught = 0
                 (the cost function described on the resume)
  B) Notebook:   the matrix in notebook 06 (caught churner = -150, false alarm = 80,
                 missed churner = 768)
  C) Corrected:  expected cost, everything measured against the same baseline.
                 Caught churner = offer + (1 - retention) x annual revenue
                                = 80 + 0.7 x 768 = 617.6 EUR (30% are kept, 70% still leave)
                 False alarm = 80 EUR, missed churner = 768 EUR.
                 This is the same accounting the ROI tab uses.

Run from the project root:
    python scripts/audit_threshold.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn import config, cost
from churn import model as churn_model

y_true, y_prob = churn_model.load_holdout(churn_model.load_model())
p = config.load_roi_params()
ANNUAL = cost.annual_revenue(p["monthly_charge"])
OFFER, RET = p["offer_cost"], p["retention_rate"]
n = len(y_true)

COST_MODELS = {
    "A) Simple (768 vs 50, no TP term)": dict(tp=0.0, fp=50.0, fn=ANNUAL),
    "B) Notebook 06 matrix": dict(tp=OFFER - ANNUAL * RET, fp=OFFER, fn=ANNUAL),
    "C) Corrected expected cost": dict(tp=OFFER + (1 - RET) * ANNUAL, fp=OFFER, fn=ANNUAL),
}

thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
counts = {}
for t in thresholds:
    _, fp, fn, tp = cost.confusion_counts(y_true, (y_prob >= t).astype(int))
    counts[t] = (tp, fp, fn)

baseline = float(y_true.sum()) * ANNUAL  # doing nothing: every churner is lost


def total_cost(t, m):
    tp, fp, fn = counts[t]
    return tp * m["tp"] + fp * m["fp"] + fn * m["fn"]


print(f"Test set: {n} customers. Annual revenue {ANNUAL:.0f}, offer {OFFER:.0f}, retention {RET:.0%}")
print(f"Doing nothing costs {baseline:,.0f} EUR on this test set.\n")
print(f"{'cost model':36s} {'best thr':>8s} {'saving @best':>13s} {'saving @0.11':>13s} {'saving @0.50':>13s}")
for name, m in COST_MODELS.items():
    best = min(thresholds, key=lambda t: total_cost(t, m))
    sav = lambda t: baseline - total_cost(t, m)
    print(f"{name:36s} {best:8.2f} {sav(best):13,.0f} {sav(0.11):13,.0f} {sav(0.50):13,.0f}")

print("\nSaving per 1,000 customers under the corrected accounting (C), by threshold:")
m = COST_MODELS["C) Corrected expected cost"]
for t in (0.11, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
    print(f"  threshold {t:.2f}: {(baseline - total_cost(t, m)) / n * 1000:10,.0f} EUR")
