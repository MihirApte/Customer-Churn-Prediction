# Customer Churn Prediction with SHAP Explainability

**Author:** Mihir Apte | MSc Computer Science (Data Science), Trinity College Dublin
**GitHub:** [MihirApte](https://github.com/MihirApte)
**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/mihir-apte/CustomerChurnAnalyser)

---

## What This Project Is

Most churn projects stop at a model accuracy number. This one goes further. It is a full end-to-end ML pipeline that predicts which telecom customers are likely to churn, explains *why* the model thinks so for each individual customer, and translates the model output into a real business decision with an estimated EUR impact.

The Streamlit app has two tabs:

- **Customer Risk Predictor** - fill in any customer profile and get a churn probability, a 4-tier risk classification, and a SHAP waterfall chart showing which features drove that specific prediction
- **ROI Calculator** - plug in your business assumptions (customer base size, monthly charge, retention offer cost) and see what the model is worth in EUR per year compared to doing nothing

---

## Problem Statement

The dataset is the IBM Telco Customer Churn dataset (7,043 customers, 21 features). Around 26% of customers churn, creating a class imbalance problem. The business goal is not to maximise accuracy but to catch as many churners as possible at an acceptable cost of false alarms.

Key business framing:

- Losing a churner costs: average monthly charge x 12 months of lost revenue
- Sending a retention offer to a non-churner costs: a fixed offer amount (wasted spend)
- The decision threshold was optimised to minimise total business cost, not to hit 0.5 by default

The optimised threshold came out at **0.11**, meaning the model flags customers with as low as 11% predicted churn probability. This is much more aggressive than the default 0.5, and is justified by the cost maths.

---

## Risk Tier System

| Tier | Probability Range | Action |
|---|---|---|
| Low Risk | Below 0.11 | No action required |
| Medium Risk | 0.11 to 0.40 | Flag for standard retention offer |
| High Risk | 0.40 to 0.70 | Prioritise for personalised outreach |
| Critical Risk | Above 0.70 | Escalate immediately |

---

## Model Card

| Property | Detail |
|---|---|
| Model | XGBoost Classifier |
| Training data | IBM Telco Customer Churn (Kaggle) |
| Features | 30 (after one-hot encoding of 21 raw features) |
| Class imbalance handling | SMOTE oversampling on training fold |
| Feature scaling | StandardScaler on tenure, MonthlyCharges, TotalCharges |
| Explainability | SHAP TreeExplainer (per-prediction waterfall charts) |
| Decision threshold | 0.11 (business-cost optimised, not default 0.5) |
| Deployment | Streamlit on HuggingFace Spaces |

Three other models were trained and compared (Logistic Regression, Random Forest, XGBoost). XGBoost was selected as the final model based on performance at the business-optimised threshold.

---

## Tech Stack

| Purpose | Library |
|---|---|
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Modelling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Class imbalance | imbalanced-learn (SMOTE) |
| App and deployment | Streamlit |
| Hosting | HuggingFace Spaces |

---

## Project Structure

```
customer-churn-prediction/
|-- app.py                  # Streamlit application (single entry point)
|-- requirements.txt        # Minimal dependency list for deployment
|-- models/
|   |-- best_model.pkl      # Final XGBoost model used by the app
|-- reports/
|   |-- scaler_params.json  # Mean and std from training data (for inference-time scaling)
|   |-- roi_params.json     # Default business assumption values for ROI tab
|   |-- figures/            # Charts generated during model evaluation
|-- data/
|   |-- processed/          # Cleaned and encoded train/test splits
|-- notebooks/              # Jupyter notebooks (EDA, training, evaluation)
|-- .gitignore
|-- README.md
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/MihirApte/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

Note: the processed data files (`data/processed/X_test.csv` and `y_test.csv`) are required for the ROI Calculator tab. These are not tracked by git because they are derived from the raw Kaggle dataset. To regenerate them, run the training notebook in `notebooks/`.

---

## HuggingFace Spaces

The app is deployed as a public Streamlit Space. No setup required - just open the link and interact with the predictor directly.

Live link: [HuggingFace Spaces](https://huggingface.co/spaces/mihir-apte/CustomerChurnAnalyser)

---

## Key Design Decisions

**Why 0.11 and not 0.5 as the threshold?**
At 0.5, the model misses too many churners. Each missed churner costs 12 months of revenue. At 0.11, more churners are flagged and given a retention offer. The false alarm cost (wasted offer) is much lower than the missed churner cost, so being aggressive pays off. The ROI tab lets you verify this with your own numbers.

**Why XGBoost?**
Tree-based models handle mixed feature types (binary flags, continuous charges) and feature interactions naturally. XGBoost with SMOTE outperformed Logistic Regression and Random Forest at the business-relevant threshold on the held-out test set.

**Why SHAP?**
Churn prediction is only useful if a retention team can act on it. A model that just says "this customer will churn" is not enough. SHAP shows exactly which features drove each individual prediction, letting a retention rep have an informed conversation with the customer rather than a generic script.

---

*Built as a portfolio project to demonstrate end-to-end ML engineering, business-oriented model evaluation, and production-style deployment.*
