"""Recompute reports/scaler_params.json so it matches how the model was actually trained.

Why: notebook 03 fits StandardScaler on the 80% TRAINING split only. The old
scaler_params.json held mean/std from the FULL dataset, so the app scaled inputs
slightly differently from the data the model saw (about 0.006 std for tenure).

This script replays the notebook's exact steps (same cleaning file, same
train_test_split seed), fits the scaler on the training split, and CHECKS that
transforming the test split reproduces data/processed/X_test.csv. It only writes
the JSON if that check passes.

Run from the project root:
    python scripts/recompute_scaler_params.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

df = pd.read_csv(ROOT / "data" / "processed" / "telco_churn_cleaned.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Same arguments as notebook 03, cell 6
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train[NUMERICAL_COLS])

# Safety check: must reproduce the saved test matrix exactly
expected = pd.read_csv(ROOT / "data" / "processed" / "X_test.csv")[NUMERICAL_COLS].values
reproduced = scaler.transform(X_test[NUMERICAL_COLS])
max_diff = float(np.abs(reproduced - expected).max())
if not np.allclose(reproduced, expected, atol=1e-6):
    sys.exit(f"CHECK FAILED: does not reproduce X_test.csv (max diff {max_diff:.2e}). "
             "scaler_params.json was NOT changed.")
print(f"Check passed: reproduces X_test.csv (max diff {max_diff:.2e})")

path = ROOT / "reports" / "scaler_params.json"
old = json.loads(path.read_text())
new = {
    col: {"mean": float(m), "std": float(s)}
    for col, m, s in zip(NUMERICAL_COLS, scaler.mean_, scaler.scale_)
}

print(f"\n{'column':16s} {'old mean':>12s} {'new mean':>12s} {'old std':>12s} {'new std':>12s}")
for col in NUMERICAL_COLS:
    print(f"{col:16s} {old[col]['mean']:12.4f} {new[col]['mean']:12.4f} "
          f"{old[col]['std']:12.4f} {new[col]['std']:12.4f}")

path.write_text(json.dumps(new, indent=2))
print(f"\nWrote {path}")
