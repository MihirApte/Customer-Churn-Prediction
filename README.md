# Customer Churn Prediction with SHAP Explainability and a RAG Retention Advisor

**Author:** Mihir Apte | MSc Computer Science (Data Science), Trinity College Dublin
**GitHub:** [MihirApte](https://github.com/MihirApte)
**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/mihir-apte/CustomerChurnAnalyser)

---

## What This Project Is

An end-to-end churn project: an XGBoost model predicts which telecom customers are likely to leave, SHAP explains why for each customer, and a Retention Advisor turns those reasons into a cited, cost-checked action plan.

The Streamlit app has three tabs:

- **Customer Risk Predictor**: fill in a customer profile and get a risk score, a 4-tier classification, and a SHAP waterfall chart of the features behind that score. A "Get retention plan" button sends the same customer to the advisor.
- **ROI Calculator**: set your own business assumptions and see what the model is worth per year compared to doing nothing.
- **Retention Advisor**: for the current customer, decides whether a paid offer is worth it, retrieves matching playbook entries, and writes a plan that cites them.

---

## Retention Advisor (RAG)

SHAP says *why* a customer is at risk. The advisor answers *what to do about it, and is it worth the money*.

```
customer profile
   -> model score + top 3 SHAP drivers (gender excluded)
   -> cost decision, in code: JUSTIFIED / MARGINAL / NOT_JUSTIFIED
   -> hybrid retrieval over a 19-entry retention playbook
   -> LLM (Groq) writes a plan using only the retrieved entries
   -> citation check; on failure retry once, then fall back to a plan built directly from the playbook
```

**Hybrid retrieval.** A metadata gate first keeps only entries that target the customer's driver and whose conditions actually hold for that customer (for example, no month-to-month offers for a customer on a one-year contract). The remaining entries are ranked by sentence-embedding similarity (all-MiniLM-L6-v2, NumPy cosine similarity). Each driver is searched separately plus one combined query, so no driver is crowded out.

**Cost decision in code, not by the LLM.** Expected value = real churn rate of held-out customers in the same risk tier x retention rate (30%) x the customer's annual revenue - offer cost (EUR 80), with a Wilson 95% interval. If a paid offer does not pay off, the LLM is not called and a low-cost plan is returned. Retention rate and offer cost are assumptions, adjustable in the ROI tab.

**Citation check.** Every playbook ID (PB-xx) in the plan must be one that was retrieved for that customer. Anything else is rejected.

**No key, no problem.** Without a Groq key, or if the API fails, the app returns a playbook-built plan instead of an error.

The playbook (`data/playbook/playbook.json`) is illustrative: entries are written for this project, entries reviewed and evaluated by author, and the evidence figures describe associations, not proven causes.

### Evaluation (measured, `evaluation/results.md`)

Run on real held-out customers (the 1,407 the model never trained on). 538 reach retrieval; the rest are ruled out by the cost check.

| Measure | Hybrid retrieval (app) | Embedding-only baseline |
|---|---|---|
| Top drivers that got matching guidance (538 customers, 1,614 drivers) | 88.2% | 66.1% |
| Returned entries whose conditions do not hold for the customer | 0.0% | 12.0% |
| Precision, hand-labelled (95 pairs, 25 customers) | 97.5% (95% range 91.3 to 99.3) | 84.0% (74.1 to 90.6) |
| Pooled recall, hand-labelled | 95.1% (88.1 to 98.1) | 76.8% (66.6 to 84.6) |

Real Groq calls (`openai/gpt-oss-120b`, 40 customers): 40 of 40 plans passed the citation check on the first attempt, 0 of 40 contained a number not given in the prompt, latency 7.0 s median and 12.0 s at the 95th percentile.

Limits to keep in mind: the labels come from one person who also wrote the playbook; the labelled sample is small (25 customers) and the ranges treat pairs as independent although they are clustered by customer; the 0.0% inapplicable rate is enforced by design; 88.2% driver coverage is capped by what the playbook covers; the first-attempt citation rate measures the model, not the safety net around it.

To reproduce: `python evaluation/make_label_sheet.py` then `python evaluation/run_eval.py --llm 40`.

---

## Risk Tiers and an Honest Note on Scores

| Tier | Score range | Meaning |
|---|---|---|
| Low Risk | Below 0.11 | No action required |
| Medium Risk | 0.11 to 0.40 | Monitor, low-cost touches only; a paid offer loses money on average |
| High Risk | 0.40 to 0.70 | Prioritise for personalised outreach |
| Critical Risk | Above 0.70 | Escalate immediately |

The model was trained with SMOTE, which inflates its output scores, so they are risk rankings, not true probabilities. That is why the advisor's cost decision uses the actual churn rate of held-out customers in the same tier instead of the raw score.

**Threshold.** Notebook 06 originally treated a caught churner as a saving while counting a missed churner as an absolute loss, which overstated the benefit. It was corrected so a flagged churner costs the offer plus the share who leave anyway (offer + 70% x EUR 768). On the 1,407 test customers the corrected analysis gives a saving of EUR 22,426 versus doing nothing (about EUR 16 per customer) at a threshold of 0.60, and the saving is fairly flat between about 0.4 and 0.6. The 0.11 boundary is kept as the monitoring line, and 0.40 (the HIGH tier boundary) is where a paid offer roughly breaks even at EUR 64 per month. These figures depend on the assumed 30% retention rate and EUR 80 offer cost.

---

## Model Card

| Property | Detail |
|---|---|
| Model | XGBoost Classifier |
| Training data | IBM Telco Customer Churn (7,043 customers) |
| Features | 30 (after one-hot encoding of 21 raw features) |
| Class imbalance handling | SMOTE oversampling on training fold |
| Feature scaling | StandardScaler on tenure, MonthlyCharges, TotalCharges (train-split statistics) |
| Explainability | SHAP TreeExplainer |
| Deployment | Streamlit on HuggingFace Spaces |

Logistic Regression and Random Forest were also trained; XGBoost was selected on held-out performance.

---

## Project Structure

```
|-- app.py                     # Streamlit UI only (three tabs)
|-- src/churn/
|   |-- config.py              # paths, tier thresholds, env loading
|   |-- features.py            # profile -> model features (validated)
|   |-- model.py               # model loading, risk tiers, SHAP drivers
|   |-- cost.py                # ROI maths
|   |-- evaluation_metrics.py  # CIs, percentiles, unsupported-number check
|   |-- advisor/               # playbook, query, retriever, search, decision, llm, prompts, citations, service
|-- data/playbook/playbook.json  # 19 retention entries
|-- tests/                     # pytest suite (174 tests, no network or API key needed)
|-- evaluation/                # scripts, labels, results.md, results_llm.csv
|-- scripts/                   # one-off checks (economics, threshold audit, retrieval trials)
|-- models/, reports/, notebooks/, data/processed/
|-- .env.example               # template for the Groq key
```

---

## Run Locally

```bash
git clone https://github.com/MihirApte/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python -m venv venv
venv\Scripts\activate           # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

**LLM key (optional).** Copy `.env.example` to `.env` and set `GROQ_API_KEY=...` (free key at console.groq.com). `.env` is gitignored. Without a key the advisor still works using playbook-built plans.

On HuggingFace Spaces the key is stored as a Space secret named `GROQ_API_KEY`, never in the code.

**Data.** `data/processed/X_test.csv` and `y_test.csv` are needed for the ROI tab and the advisor's tier statistics; regenerate them with the notebooks if missing.

## Run the Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite covers feature building, tiers, cost maths, retrieval rules, the citation check, the LLM client and the advisor service, using a fake embedder and a fake LLM so it runs offline.

---

## Key Design Decisions

**Why a RAG advisor and not just SHAP?** SHAP names the drivers; it does not say what to offer or whether the offer pays for itself. The advisor grounds its advice in a fixed playbook and checks the arithmetic in code.

**Why is the cost decision not given to the LLM?** Money maths should be deterministic and testable. The LLM only writes the plan.

**Why XGBoost?** Tree models handle mixed feature types and interactions well and work with TreeExplainer.

---

*Built as a portfolio project for end-to-end ML engineering, business-oriented evaluation, and deployment.*
