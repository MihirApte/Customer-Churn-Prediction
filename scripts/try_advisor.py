"""Run the full retention advisor (cost decision, retrieval, LLM, citation check) on example customers.

Needs GROQ_API_KEY in .env (see .env.example). Customers whose offer is not worth it
never reach the LLM, so you will see some "no_offer" results without any API call.

Run from the project root:
    python scripts/try_advisor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _examples import BASE, EXAMPLES
from churn import config
from churn import model as churn_model
from churn.advisor.llm import GroqClient, LLMError
from churn.advisor.playbook import load_playbook
from churn.advisor.query import NON_ACTIONABLE
from churn.advisor.retriever import Retriever, SentenceTransformerEmbedder
from churn.advisor.service import RetentionAdvisor
from churn.features import build_feature_row

config.load_env_file()

model = churn_model.load_model()
explainer = churn_model.build_explainer(model)
scaler_params = config.load_scaler_params()
y_true, y_prob = churn_model.load_holdout(model)

try:
    llm = GroqClient()
except LLMError as exc:
    sys.exit(f"{exc}\nCreate .env from .env.example and paste your key, then run again.")
print(f"Using Groq model: {llm.model}")

advisor = RetentionAdvisor(
    Retriever(load_playbook(), SentenceTransformerEmbedder()),
    llm, y_true, y_prob, config.load_roi_params(),
)

for name, overrides in EXAMPLES.items():
    profile = {**BASE, **overrides}
    X = build_feature_row(profile, scaler_params)
    score = churn_model.predict_churn_probability(model, X)
    drivers = churn_model.top_churn_drivers(
        churn_model.shap_values_for(explainer, X), n=3, exclude=NON_ACTIONABLE
    )
    advice = advisor.advise(profile, drivers, score)
    d = advice.decision

    print("=" * 78)
    print(name)
    print(f"Score {score:.2f} | tier {d.tier} | verdict {d.verdict.value} | "
          f"EV per offer EUR {d.expected_value:+,.0f} (n={d.n}, real rate {d.churn_rate:.1%})")
    print(f"Mode: {advice.mode} | LLM calls: {advice.attempts} | LLM time: {advice.llm_seconds:.1f}s")
    print(f"Note: {advice.note}")
    if advice.hits:
        print("Retrieved:", ", ".join(h.entry.id for h in advice.hits))
    if advice.citation:
        print(f"Citations: {advice.citation.cited} ok={advice.citation.ok}")
    print("\n" + advice.plan + "\n")
