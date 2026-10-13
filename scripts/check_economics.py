"""Read-only check: does a retention offer pay for itself, and at which risk level?

Uses the held-out test set and the assumptions in reports/roi_params.json
(offer cost, retention rate, monthly charge). It changes nothing.

It answers three questions the retention advisor depends on:
  1. What does the ROI tab show at the default settings and other thresholds?
  2. How well do predicted probabilities match real churn rates (calibration)?
  3. At what real churn rate does an offer break even, and which tiers clear it?

Run from the project root:
    python scripts/check_economics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn import config, cost
from churn import model as churn_model

y_true, y_prob = churn_model.load_holdout(churn_model.load_model())
params = config.load_roi_params()

MONTHLY = params["monthly_charge"]
RETENTION = params["retention_rate"]
OFFER = params["offer_cost"]
ANNUAL = cost.annual_revenue(MONTHLY)
BREAK_EVEN = OFFER / (RETENTION * ANNUAL)

print(f"Test set: {len(y_true)} customers, real churn rate {y_true.mean():.1%}")
print(f"Assumptions: monthly EUR {MONTHLY}, annual EUR {ANNUAL}, retention {RETENTION:.0%}, offer EUR {OFFER}")
print(f"Break-even: an offer pays for itself only if the customer's REAL churn chance is above "
      f"{BREAK_EVEN:.1%}\n  (= offer cost / (retention rate x annual revenue))\n")

print("1) ROI tab at different thresholds (7,000 customers, 26% churn, as in the app)")
print(f"{'threshold':>9} {'flagged':>8} {'precision':>10} {'recall':>7} {'net saving EUR':>15}")
for thr in (0.11, 0.2, 0.3, 0.35, 0.4, 0.5):
    roi = cost.compute_roi(y_true, y_prob, thr, MONTHLY, RETENTION * 100, OFFER, 7000, 26)
    flagged = (y_prob >= thr).mean()
    print(f"{thr:9.2f} {flagged:8.1%} {roi.precision:10.1%} {roi.recall:7.1%} {roi.net_saving:15,.0f}")


def table(title, edges, labels):
    print(f"\n{title}")
    print(f"{'band':>14} {'n':>5} {'mean predicted':>15} {'real churn':>11} {'EV per offer EUR':>17}  worth it?")
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"{label:>14} {n:5d}  (no customers)")
            continue
        real = float(y_true[mask].mean())
        ev = real * RETENTION * ANNUAL - OFFER
        print(f"{label:>14} {n:5d} {y_prob[mask].mean():15.1%} {real:11.1%} {ev:17,.0f}  {'yes' if ev > 0 else 'no'}")


edges = [0, 0.11, 0.2, 0.3, 0.4, 0.55, 0.7, 1.0001]
table("2) Calibration: predicted probability vs real churn rate",
      edges, ["<0.11", "0.11-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.55", "0.55-0.70", ">=0.70"])

t = [0, config.THRESH_MEDIUM, config.THRESH_HIGH, config.THRESH_CRITICAL, 1.0001]
table("3) By the app's risk tiers (expected value of one offer for a EUR "
      f"{MONTHLY} / month customer)", t, ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
print("\nEV per offer = real churn rate x retention rate x annual revenue - offer cost. Small n means a noisy estimate.")
