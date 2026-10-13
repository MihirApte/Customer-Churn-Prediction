"""Paths, risk-tier thresholds and loaders for the JSON parameter files.

Paths are resolved from this file's location, so the code works no matter
which directory the app or the tests are launched from.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "best_model.pkl"
ROI_PARAMS_PATH = ROOT / "reports" / "roi_params.json"
SCALER_PARAMS_PATH = ROOT / "reports" / "scaler_params.json"
PLAYBOOK_PATH = ROOT / "data" / "playbook" / "playbook.json"
X_TEST_PATH = ROOT / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = ROOT / "data" / "processed" / "y_test.csv"

# Risk tier boundaries on churn probability.
# 0.11 is the monitoring boundary (flag for watching, low-cost touches only). A paid offer
# only pays off at higher risk; the advisor decides that per customer in advisor/decision.py.
THRESH_MEDIUM = 0.11
THRESH_HIGH = 0.40
THRESH_CRITICAL = 0.70

# Sentence embedding model used to match a customer to playbook entries.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_env_file(path: Path = ROOT / ".env") -> None:
    """Load local secrets (such as GROQ_API_KEY) from .env into the environment.

    Does nothing if python-dotenv is missing or the file does not exist, so it is
    safe on HuggingFace Spaces, where secrets are real environment variables.
    Values already set in the environment are never overwritten.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path, override=False)


def load_roi_params(path: Path = ROI_PARAMS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def load_scaler_params(path: Path = SCALER_PARAMS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)
