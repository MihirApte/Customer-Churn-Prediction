"""Try the retrieval on a few example customers and print what it finds.

Needs the trained model and sentence-transformers installed. The first run
downloads the embedding model (about 90 MB), so it takes a minute.

Run from the project root:
    python scripts/try_retrieval.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _examples import BASE, EXAMPLES
from churn import config
from churn import model as churn_model
from churn.advisor.playbook import load_playbook
from churn.advisor.query import NON_ACTIONABLE, build_query
from churn.advisor.retriever import Retriever, SentenceTransformerEmbedder
from churn.advisor.search import retrieve_for_customer
from churn.features import build_feature_row

model = churn_model.load_model()
explainer = churn_model.build_explainer(model)
scaler_params = config.load_scaler_params()

print("Loading playbook and embedding model ...")
t0 = time.perf_counter()
retriever = Retriever(load_playbook(), SentenceTransformerEmbedder())
print(f"Ready in {time.perf_counter() - t0:.1f}s\n")

for name, overrides in EXAMPLES.items():
    profile = {**BASE, **overrides}
    X = build_feature_row(profile, scaler_params)
    prob = churn_model.predict_churn_probability(model, X)
    drivers = churn_model.top_churn_drivers(
        churn_model.shap_values_for(explainer, X), n=3, exclude=NON_ACTIONABLE
    )
    query = build_query(drivers, profile)

    t0 = time.perf_counter()
    result = retrieve_for_customer(retriever, drivers, profile)
    ms = (time.perf_counter() - t0) * 1000

    print("=" * 78)
    print(name)
    print(f"Churn probability: {prob:.1%} | tier: {churn_model.risk_tier(prob).label}")
    print("Top drivers:", ", ".join(f"{d.feature} ({d.shap_value:+.2f})" for d in drivers))
    print("Query:", query)
    print(f"Retrieved in {ms:.0f} ms:")
    for h in result.hits:
        print(f"  {h.score:.3f}  {h.entry.id}  {h.entry.title}  <- {', '.join(h.matched_on)}")
    if result.uncovered:
        print("  No playbook guidance for:", ", ".join(result.uncovered))
    print()
