"""
Customer Churn Analyser - Streamlit App (UI layer only).

Two tabs:
  1. Customer Risk Predictor  - sidebar inputs -> churn probability + SHAP waterfall
  2. ROI Calculator           - adjustable business assumptions -> annual saving estimate

All logic lives in src/churn/. This file only builds widgets and draws results.

Run with:
    streamlit run app.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
import streamlit.components.v1 as components

from churn import config, cost, model as churn_model
from churn.advisor.decision import Verdict
from churn.advisor.llm import GroqClient, LLMError, UnavailableLLM
from churn.advisor.playbook import load_playbook
from churn.advisor.query import NON_ACTIONABLE, friendly_driver_names
from churn.advisor.retriever import Retriever, SentenceTransformerEmbedder
from churn.advisor.service import RetentionAdvisor
from churn.features import FEATURE_COLS, build_feature_row

config.load_env_file()  # local .env (GROQ_API_KEY); on HuggingFace the key is a Space secret
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Churn Analyser",
    layout="wide",
)


# Cached loaders (thin wrappers around src/churn)
@st.cache_resource
def load_model():
    return churn_model.load_model()


@st.cache_resource
def get_explainer(_mdl):
    return churn_model.build_explainer(_mdl)


@st.cache_data
def load_roi_params():
    return config.load_roi_params()


@st.cache_data
def load_scaler_params():
    return config.load_scaler_params()


@st.cache_data
def load_test_data():
    return churn_model.load_holdout(load_model())


model = load_model()
explainer = get_explainer(model)
roi_params = load_roi_params()
scaler_params = load_scaler_params()
y_test_arr, y_prob_arr = load_test_data()


@st.cache_resource(show_spinner=False)  # the Retention Advisor tab shows its own spinner
def load_advisor():
    """Built only when a plan is first requested (loading the embedding model takes a while).
    Without GROQ_API_KEY the advisor still works, using playbook text instead of LLM text."""
    try:
        llm = GroqClient()
    except LLMError as exc:
        llm = UnavailableLLM(str(exc))
    retriever = Retriever(load_playbook(), SentenceTransformerEmbedder())
    return RetentionAdvisor(retriever, llm, y_test_arr, y_prob_arr, roi_params)


# App title
st.title("Telecom Customer Churn Analyser")

tab1, tab2, tab3 = st.tabs(["Customer Risk Predictor", "ROI Calculator", "Retention Advisor"])

# Streamlit cannot switch tabs from Python, so this invisible script clicks the third tab.
# Best effort only: if it does not work in a browser, the plan is still waiting in that tab.
SWITCH_TO_ADVISOR_TAB = """
<script>
const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
if (tabs.length > 2) { tabs[2].click(); }
</script>
"""


# TAB 1 - CUSTOMER RISK PREDICTOR
with tab1:
    st.header("Customer Churn Risk Predictor")
    st.markdown(
        "Adjust the customer profile in the **sidebar** to get a real-time churn "
        "probability and a SHAP explanation of the individual prediction."
    )

    # -- Sidebar inputs --------------------------------------------------------
    with st.sidebar:
        st.header("Customer Profile")

        st.subheader("Demographics")
        gender  = st.selectbox("Gender", ["Male", "Female"])
        senior  = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        deps    = st.selectbox("Has Dependents", ["No", "Yes"])

        st.subheader("Account Details")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges (EUR)", 18.0, 120.0, 64.0, step=0.5)
        default_total   = round(tenure * monthly_charges, 2)
        total_charges   = st.slider(
            "Total Charges (EUR)", 0.0, 8700.0,
            float(min(default_total, 8700.0)), step=10.0
        )
        contract  = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment   = st.selectbox("Payment Method", [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ])

        st.subheader("Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        if phone == "Yes":
            multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        else:
            multi_lines = "No phone service"
            st.caption("Multiple lines: not applicable")

        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        if internet != "No":
            online_sec    = st.selectbox("Online Security",   ["No", "Yes"])
            online_bkp    = st.selectbox("Online Backup",     ["No", "Yes"])
            device_prot   = st.selectbox("Device Protection", ["No", "Yes"])
            tech_sup      = st.selectbox("Tech Support",      ["No", "Yes"])
            streaming_tv  = st.selectbox("Streaming TV",      ["No", "Yes"])
            streaming_mov = st.selectbox("Streaming Movies",  ["No", "Yes"])
        else:
            online_sec = online_bkp = device_prot = "No internet service"
            tech_sup = streaming_tv = streaming_mov = "No internet service"
            st.caption("Internet-dependent services: not applicable")

    # Raw customer profile -> model input row
    profile = {
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "gender":           gender,
        "Partner":          partner,
        "Dependents":       deps,
        "PhoneService":     phone,
        "MultipleLines":    multi_lines,
        "InternetService":  internet,
        "OnlineSecurity":   online_sec,
        "OnlineBackup":     online_bkp,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_sup,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_mov,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
    }

    X_input    = build_feature_row(profile, scaler_params)
    churn_prob = churn_model.predict_churn_probability(model, X_input)
    tier       = churn_model.risk_tier(churn_prob)
    label, action, st_fn = tier.label, tier.action, tier.level

    # Prediction display
    col_pred, col_meta = st.columns([3, 1])

    with col_pred:
        st.subheader("Prediction")

        if st_fn == "success":
            st.success(f"{label}    ({churn_prob:.1%} probability)")
        elif st_fn == "warning":
            st.warning(f"{label}    ({churn_prob:.1%} probability)")
        else:
            st.error(f"{label}    ({churn_prob:.1%} probability)")

        st.caption(f"Recommended action: {action}")

        if st.button("Get retention plan", key="get_plan_tab1",
                     help="Opens the Retention Advisor tab with a plan for this customer"):
            st.session_state["request_plan"] = True
            components.html(SWITCH_TO_ADVISOR_TAB, height=0)

        # Probability bar with tier boundaries
        fig_bar, ax_bar = plt.subplots(figsize=(7, 1.3))
        bar_color = {"success": "#27ae60", "warning": "#f39c12", "error": "#c0392b"}[st_fn]
        ax_bar.barh(0, churn_prob, color=bar_color, height=0.5)
        ax_bar.barh(0, 1 - churn_prob, left=churn_prob, color="#ecf0f1", height=0.5)
        ax_bar.axvline(config.THRESH_MEDIUM,   color="#27ae60", linewidth=1.8, linestyle="--", label=f"Monitor ({config.THRESH_MEDIUM:.2f})")
        ax_bar.axvline(config.THRESH_HIGH,     color="#f39c12", linewidth=1.8, linestyle="--", label=f"Act ({config.THRESH_HIGH:.2f})")
        ax_bar.axvline(config.THRESH_CRITICAL, color="#c0392b", linewidth=1.8, linestyle="--", label=f"Urgent ({config.THRESH_CRITICAL:.2f})")
        ax_bar.set_xlim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Churn Probability")
        ax_bar.set_title(f"Churn Probability: {churn_prob:.1%}", fontsize=13, fontweight="bold")
        ax_bar.legend(loc="lower right", fontsize=7.5, title="Tier thresholds")
        fig_bar.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.caption(
            "Tier boundaries: below 0.11 = Low Risk | 0.11-0.40 = Medium Risk | "
            "0.40-0.70 = High Risk | above 0.70 = Critical Risk. "
            "Scores are risk rankings, not true probabilities (SMOTE inflates them). "
            "0.11 is the monitoring boundary; a paid offer only breaks even at higher risk (see the Retention Advisor tab)."
        )

    with col_meta:
        st.subheader("Key Metrics")
        st.metric("Churn Probability",          f"{churn_prob:.1%}")
        st.metric("Risk Tier",                  label.split(" - ")[0])
        st.metric("Annual Revenue at Risk",     f"EUR {cost.annual_revenue(monthly_charges):,.0f}")

    # -- SHAP Waterfall --------------------------------------------------------
    st.divider()
    st.subheader("Why this prediction? - SHAP Feature Explanation")
    st.markdown(
        "The waterfall chart shows how each feature pushed the model output away from "
        "the average prediction (base value). "
        "**Red bars** push toward churn; **blue bars** push away from churn. "
        "Values are in log-odds space -- larger magnitude means larger influence."
    )

    shap_vals = churn_model.shap_values_for(explainer, X_input)
    drivers = churn_model.top_churn_drivers(shap_vals, n=3, exclude=NON_ACTIONABLE)
    profile_key = json.dumps(profile, sort_keys=True)
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=float(explainer.expected_value),
        data=X_input.iloc[0].values,
        feature_names=FEATURE_COLS,
    )

    try:
        shap.plots.waterfall(explanation, max_display=12, show=False)
        fig_shap = plt.gcf()
        fig_shap.set_size_inches(10, 6)
        st.pyplot(fig_shap)
        plt.close("all")
    except Exception:
        plt.close("all")
        # Fallback: horizontal bar chart of SHAP values sorted by magnitude
        indices = np.argsort(np.abs(shap_vals))[-12:]
        fig_fb, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_vals[indices]]
        ax.barh([FEATURE_COLS[i] for i in indices], shap_vals[indices], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (log-odds)")
        ax.set_title("Feature impact on churn prediction")
        st.pyplot(fig_fb)
        plt.close("all")

    st.caption(
        "Base value = the model average output across all training customers. "
        "Each bar shows how far one feature moved the prediction from that baseline."
    )


# TAB 2 - ROI CALCULATOR
with tab2:
    st.header("Business ROI Calculator")
    st.markdown(
        "Adjust the business assumptions below to see how model performance translates "
        "into financial value. Recall and precision are derived from actual model "
        "performance on the held-out test set, then scaled to your customer base."
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Business Assumptions")

        roi_monthly = st.slider(
            "Avg Monthly Charge (EUR)", 20, 150,
            int(roi_params["monthly_charge"]), step=1
        )
        roi_retention = st.slider(
            "Retention Rate - % of correctly flagged churners saved",
            5, 80, int(roi_params["retention_rate"] * 100), step=5
        )
        roi_offer_cost = st.slider(
            "Retention Offer Cost (EUR per customer)",
            10, 300, int(roi_params["offer_cost"]), step=5
        )
        roi_customers = st.slider(
            "Total Customer Base Size", 500, 50000, 7000, step=500
        )
        roi_churn_rate = st.slider(
            "Expected Annual Churn Rate (%)", 5, 50, 26, step=1
        )
        roi_thresh_roi = st.slider(
            "Decision Threshold",
            0.01, 0.99, float(roi_params["optimal_threshold"]), step=0.01
        )

    roi = cost.compute_roi(
        y_true=y_test_arr,
        y_prob=y_prob_arr,
        threshold=roi_thresh_roi,
        monthly_charge=roi_monthly,
        retention_pct=roi_retention,
        offer_cost=roi_offer_cost,
        customers=roi_customers,
        churn_rate_pct=roi_churn_rate,
    )

    with col_b:
        st.subheader("Estimated Annual Impact")

        st.metric("Churners in Customer Base", f"{roi.n_churners:,}")
        st.metric(
            "Churners Caught by Model", f"{roi.tp_scaled:,}",
            delta=f"{roi.recall:.0%} recall (from test set)"
        )
        st.metric(
            "False Alarms (wasted offers)", f"{roi.fp_scaled:,}",
            delta=f"{roi.precision:.0%} precision (from test set)"
        )
        st.metric("Customers Retained",    f"{roi.saved_customers:,}")
        st.metric("Revenue Saved",         f"EUR {roi.revenue_saved:,.0f}")
        st.metric("Total Offer Spend",     f"EUR {roi.offer_spend:,.0f}")
        st.metric(
            "Net Annual Saving", f"EUR {roi.net_saving:,.0f}",
            delta="vs EUR 0 with no model"
        )
        st.metric("Revenue Lost (missed churners)", f"EUR {roi.missed_cost:,.0f}")

    # Bar chart
    st.divider()
    st.subheader("Total Annual Churn Cost: Model vs No Model")

    fig_roi, ax_roi = plt.subplots(figsize=(7, 4))
    scenarios = [
        "No Model\n(lose all churners)",
        f"With Model\n(threshold = {roi_thresh_roi:.2f})",
    ]
    values_k    = [roi.no_model_cost / 1000, roi.model_cost / 1000]
    color_model = "#27ae60" if roi.net_saving > 0 else "#e67e22"
    colors_roi  = ["#c0392b", color_model]

    bars = ax_roi.bar(scenarios, values_k, color=colors_roi, width=0.4, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values_k):
        ax_roi.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values_k) * 0.015,
            f"EUR {val:,.0f}k",
            ha="center", fontsize=12, fontweight="bold",
        )
    ax_roi.set_ylabel("Total Annual Churn Cost (EUR thousands)")
    ax_roi.set_title("Annual Churn Cost Comparison", fontsize=13)
    ax_roi.set_ylim(0, max(values_k) * 1.18)
    fig_roi.tight_layout()
    st.pyplot(fig_roi)
    plt.close(fig_roi)

    saving_k = (roi.no_model_cost - roi.model_cost) / 1000
    if saving_k > 0:
        st.success(
            f"The model saves an estimated EUR {saving_k:,.0f}k per year under these assumptions."
        )
    else:
        st.warning(
            "Under these assumptions the model costs more than it saves. "
            "Try lowering the threshold, reducing offer cost, or increasing the retention rate."
        )

    st.caption(
        "How the numbers are calculated: "
        "Recall and precision come from the actual held-out test set at the chosen threshold, not estimated. "
        "Scaling uses (customer_base x churn_rate) / (test_size x test_churn_rate) to adjust for "
        "different base sizes and churn rates. "
        "Offers are sent to all flagged customers (TP + FP); "
        "only the TP x retention_rate fraction are actually retained."
    )


# TAB 3 - RETENTION ADVISOR
def _eur(value, signed=False):
    return f"EUR {value:+,.0f}" if signed else f"EUR {value:,.0f}"


def render_advice(advice):
    """Draw one RetentionAdvisor result."""
    d = advice.decision

    # 1. The money verdict, decided in code before any LLM is involved
    if d.verdict == Verdict.JUSTIFIED:
        st.success(
            f"A paid offer is justified. Expected value {_eur(d.expected_value, True)} per offer "
            f"(similar customers in the {d.tier} tier churned at {d.churn_rate:.1%})."
        )
    elif d.verdict == Verdict.MARGINAL:
        st.warning(
            f"A paid offer is only marginal. Expected value {_eur(d.expected_value, True)} per offer, "
            f"and the range ({_eur(d.ev_low, True)} to {_eur(d.ev_high, True)}) crosses zero. "
            "Prefer low-cost actions."
        )
    else:
        st.info(
            f"No paid offer recommended. Expected value {_eur(d.expected_value, True)} per offer "
            f"(similar customers in the {d.tier} tier churned at {d.churn_rate:.1%})."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real churn rate, similar customers", f"{d.churn_rate:.1%}",
              help=f"Held-out test customers in the {d.tier} tier (n={d.n}). 95% range: "
                   f"{d.rate_low:.1%} to {d.rate_high:.1%}.")
    c2.metric("Expected value per offer", _eur(d.expected_value, True))
    c3.metric("Break-even churn rate", f"{d.break_even_rate:.1%}",
              help="An offer pays for itself only if similar customers churn more often than this.")
    c4.metric("Annual revenue at risk", _eur(d.annual_revenue))

    # 2. The plan
    st.subheader("Recommended plan")
    plan_markdown = advice.plan.replace("\nCost check:", "\n\nCost check:")
    st.markdown(plan_markdown)
    if advice.mode == "llm":
        st.caption(
            f"Written by the LLM from the retrieved playbook entries. Citations verified: "
            f"{', '.join(advice.citation.cited)} (all were retrieved). "
            f"LLM time {advice.llm_seconds:.1f}s."
        )
    elif advice.mode == "fallback":
        st.warning(advice.note)
    else:
        st.caption(advice.note)

    if advice.uncovered:
        st.caption(
            "These factors also raise this customer's churn risk, but the playbook has no advice for "
            "their current value: " + ", ".join(friendly_driver_names(advice.uncovered)) + "."
        )

    # 3. Evidence
    if advice.hits:
        with st.expander("Playbook entries retrieved for this customer"):
            for hit in advice.hits:
                e = hit.entry
                st.markdown(f"**{e.id}: {e.title}**")
                st.caption(f"Matched on: {', '.join(hit.matched_on)} | similarity {hit.score:.2f}")
                st.markdown(f"*Applies when:* {e.situation}")
                st.markdown("*Actions:*\n" + "\n".join(f"- {a}" for a in e.actions))
                st.markdown(f"*Evidence:* {e.evidence}")
                st.markdown(f"*Caution:* {e.caution}")
                st.divider()

    with st.expander("How this was decided"):
        st.markdown(
            f"- Expected value of one offer = real churn rate x retention rate x annual revenue - offer cost\n"
            f"- = {d.churn_rate:.1%} x {d.retention_rate:.0%} x {_eur(d.annual_revenue)} - {_eur(d.offer_cost)} "
            f"= **{_eur(d.expected_value, True)}**\n"
            f"- The churn rate is the real rate of held-out test customers in the {d.tier} tier "
            f"(n={d.n}, 95% range {d.rate_low:.1%} to {d.rate_high:.1%}), not the model's raw score, "
            f"because the model's probabilities are inflated by SMOTE.\n"
            f"- Verdict: {d.verdict.value} (justified only if even the low end of the range is above zero)."
        )
        if advice.query:
            st.markdown(f"**Search query built from the top SHAP drivers:** {advice.query}")
        if advice.attempts:
            st.caption(f"LLM calls made: {advice.attempts}")


with tab3:
    st.header("Retention Advisor")
    st.markdown(
        "Turns the prediction into an action plan. The system first checks in code whether a paid offer "
        "is worth its cost for this kind of customer. If it is, it searches a retention playbook for the "
        "entries that match this customer's top churn drivers, and an LLM writes a short plan that may "
        "cite only those entries. Every citation is checked."
    )
    st.caption(
        f"Current customer: {label} ({churn_prob:.1%} model score). "
        "Change the profile in the sidebar, then generate a plan."
    )

    if st.button("Generate retention plan", type="primary", key="get_plan_tab3"):
        st.session_state["request_plan"] = True

    if st.session_state.pop("request_plan", False):
        with st.spinner("Checking the economics, searching the playbook and writing the plan..."):
            try:
                st.session_state["advice"] = {
                    "profile_key": profile_key,
                    "advice": load_advisor().advise(profile, drivers, churn_prob),
                }
            except Exception as exc:
                st.session_state.pop("advice", None)
                st.error(f"The advisor hit an error: {exc}")

    stored = st.session_state.get("advice")
    if stored is None:
        st.info("No plan yet. Press **Generate retention plan** here, or the button in the first tab.")
    elif stored["profile_key"] != profile_key:
        st.warning("The customer profile changed since this plan was generated. "
                   "Press **Generate retention plan** to refresh it.")
    else:
        render_advice(stored["advice"])

    st.caption(
        "The playbook is an illustrative set of retention ideas written by the project author, not company "
        "data, and has not been tested in the field. Churn rates quoted come from the IBM Telco dataset "
        "and show association, not proof that an action changes behaviour. This is a research demo."
    )
