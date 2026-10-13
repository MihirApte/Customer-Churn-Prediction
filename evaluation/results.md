# Retention advisor evaluation
Run on 2026-10-13 using 1407 held-out test customers (the model never trained on them). 538 of them reach retrieval; the others are ruled out by the cost check or have no describable driver.

## 1. Coverage and rule violations (automatic, all customers that reach retrieval)

Customers: 538. Top SHAP drivers considered (gender excluded): 1614.

| Measure | Hybrid retrieval (app) | Embedding-only baseline |
|---|---|---|
| Customers with at least one entry | 100.0% (95% range 99.3% to 100.0%) | 100.0% (95% range 99.3% to 100.0%) |
| Top drivers that got matching guidance | 88.2% (95% range 86.5% to 89.7%) | 66.1% (95% range 63.8% to 68.4%) |
| Returned entries whose conditions do not hold for the customer | 0.0% (95% range 0.0% to 0.2%) | 12.0% (95% range 10.5% to 13.6%) |
| Average entries per customer | 3.03 | 3.00 |

Note: the hybrid method rules out inapplicable entries by construction, so its 0% on the third row confirms the rule is enforced, not that the method is accurate. The baseline figure shows how often embeddings alone return an entry that does not fit the customer.

## 2. Hand-labelled retrieval quality

95 (customer, entry) pairs judged by the project author for 25 customers. The sheet did not say which method returned which pair.

| Measure | Hybrid retrieval (app) | Embedding-only baseline |
|---|---|---|
| Pairs returned (n) | 80 | 75 |
| Precision: returned entries judged relevant | 97.5% (95% range 91.3% to 99.3%) | 84.0% (95% range 74.1% to 90.6%) |
| Pooled recall: relevant entries found (of all judged relevant) | 95.1% (95% range 88.1% to 98.1%) | 76.8% (95% range 66.6% to 84.6%) |

Pooled recall counts only entries that at least one method returned, so it understates what neither method found. The sample is small and the labeller also wrote the playbook, so treat the ranges as the honest uncertainty.

Entries the hybrid method returned that you judged not relevant (review these):
- customer 1389: PB-05
- customer 5404: PB-18

## 3. LLM behaviour (real Groq calls)

Model: openai/gpt-oss-120b. 40 customers sampled (seed 11); 40 reached the LLM (0 had no eligible playbook entries); the model answered 40 of them and 0 call(s) failed with a Groq error.

| Measure (among customers where the model answered) | Result |
|---|---|
| Plan passed the citation check on the first attempt | 40 of 40: 100.0% (95% range 91.2% to 100.0%) |
| Passed only after one retry | 0 of 40 |
| Fell back to a playbook-built plan after failed citations | 0 of 40 |
| LLM plans containing a number not given in the prompt | 0.0% (95% range 0.0% to 8.8%) |
| LLM call latency, median / 95th percentile | 7.0s / 12.0s (n=40 calls) |

The plan the user sees always passes the citation check: failures are replaced by a playbook-built plan, so the first-attempt rate measures the model, not the safety net. Numbers are matched against the prompt allowing whole-number rounding; see evaluation/results_llm.csv for the flagged cases.
