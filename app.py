"""
Customer Churn Analyser - Streamlit App
Two tabs:
  1. Customer Risk Predictor  - sidebar sliders -> churn probability + SHAP waterfall
  2. ROI Calculator           - adjustable business assumptions -> annual saving estimate

Run with:
    streamlit run app.py
"""

import json
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

#  Page config 
st.set_page_config(
    page_title="Customer Churn Analyser",
    layout="wide",
)

#  Feature column order 
FEATURE_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]

#  Risk tier boundaries 
THRESH_MEDIUM   = 0.11   
THRESH_HIGH     = 0.40   
THRESH_CRITICAL = 0.70   


def risk_tier(prob):
    """Return (label, action, streamlit_fn) for a given churn probability."""
    if prob < THRESH_MEDIUM:
        return "LOW RISK", "No action required.", "success"
    elif prob < THRESH_HIGH:
        return "MEDIUM RISK - Monitor", "Flag for standard retention offer.", "warning"
    elif prob < THRESH_CRITICAL:
        return "HIGH RISK - Act", "Prioritise for personalised retention outreach.", "error"
    else:
        return "CRITICAL RISK - Urgent", "Escalate immediately - aggressive retention offer.", "error"


#  Cached loaders 
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


@st.cache_resource
def get_explainer(_mdl):
    return shap.TreeExplainer(_mdl)


@st.cache_data
def load_roi_params():
    with open("reports/roi_params.json") as f:
        return json.load(f)


@st.cache_data
def load_scaler_params():
    with open("reports/scaler_params.json") as f:
        return json.load(f)


@st.cache_data
def load_test_data():
    X_test = pd.read_csv("data/processed/X_test.csv")[FEATURE_COLS]
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze().values
    mdl = load_model()
    y_prob = mdl.predict_proba(X_test)[:, 1]
    return y_test, y_prob


model = load_model()
explainer = get_explainer(model)
roi_params = load_roi_params()
scaler_params = load_scaler_params()
y_test_arr, y_prob_arr = load_test_data()


def scale(col, raw_value):
    """Standardise a raw user input using training-data mean and std."""
    return (raw_value - scaler_params[col]["mean"]) / scaler_params[col]["std"]


#  App title 
st.title("Telecom Customer Churn Analyser")

#  Tabs 
tab1, tab2 = st.tabs(["Customer Risk Predictor", "ROI Calculator"])


 
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

    #  Build feature vector 
    def build_features():
        row = {col: 0 for col in FEATURE_COLS}

        # Continuous features must be z-scored to match training data
        row["SeniorCitizen"]    = 1 if senior == "Yes"  else 0
        row["tenure"]           = scale("tenure", tenure)
        row["MonthlyCharges"]   = scale("MonthlyCharges", monthly_charges)
        row["TotalCharges"]     = scale("TotalCharges", total_charges)
        row["gender_Male"]      = 1 if gender == "Male" else 0
        row["Partner_Yes"]      = 1 if partner == "Yes" else 0
        row["Dependents_Yes"]   = 1 if deps == "Yes"    else 0
        row["PhoneService_Yes"] = 1 if phone == "Yes"   else 0

        if multi_lines == "No phone service":
            row["MultipleLines_No phone service"] = 1
        elif multi_lines == "Yes":
            row["MultipleLines_Yes"] = 1

        if internet == "Fiber optic":
            row["InternetService_Fiber optic"] = 1
        elif internet == "No":
            row["InternetService_No"] = 1

        for base, val in [
            ("OnlineSecurity",   online_sec),
            ("OnlineBackup",     online_bkp),
            ("DeviceProtection", device_prot),
            ("TechSupport",      tech_sup),
            ("StreamingTV",      streaming_tv),
            ("StreamingMovies",  streaming_mov),
        ]:
            if val == "No internet service":
                row[f"{base}_No internet service"] = 1
            elif val == "Yes":
                row[f"{base}_Yes"] = 1

        if contract == "One year":
            row["Contract_One year"] = 1
        elif contract == "Two year":
            row["Contract_Two year"] = 1

        row["PaperlessBilling_Yes"] = 1 if paperless == "Yes" else 0

        if payment == "Credit card (automatic)":
            row["PaymentMethod_Credit card (automatic)"] = 1
        elif payment == "Electronic check":
            row["PaymentMethod_Electronic check"] = 1
        elif payment == "Mailed check":
            row["PaymentMethod_Mailed check"] = 1

        return pd.DataFrame([row])[FEATURE_COLS]

    X_input    = build_features()
    churn_prob = float(model.predict_proba(X_input)[0][1])
    label, action, st_fn = risk_tier(churn_prob)

    # Prediction display 
    col_pred, col_meta = st.columns([3, 1])

    with col_pred:
        st.subheader("Prediction")

        # Risk tier banner
        if st_fn == "success":
            st.success(f"{label}    ({churn_prob:.1%} probability)")
        elif st_fn == "warning":
            st.warning(f"{label}    ({churn_prob:.1%} probability)")
        else:
            st.error(f"{label}    ({churn_prob:.1%} probability)")

        st.caption(f"Recommended action: {action}")

        # Probability bar with tier boundaries
        fig_bar, ax_bar = plt.subplots(figsize=(7, 1.3))
        bar_color = {"success": "#27ae60", "warning": "#f39c12", "error": "#c0392b"}[st_fn]
        ax_bar.barh(0, churn_prob, color=bar_color, height=0.5)
        ax_bar.barh(0, 1 - churn_prob, left=churn_prob, color="#ecf0f1", height=0.5)
        # Draw tier boundary lines
        ax_bar.axvline(THRESH_MEDIUM,   color="#27ae60", linewidth=1.8, linestyle="--", label=f"Monitor ({THRESH_MEDIUM:.2f})")
        ax_bar.axvline(THRESH_HIGH,     color="#f39c12", linewidth=1.8, linestyle="--", label=f"Act ({THRESH_HIGH:.2f})")
        ax_bar.axvline(THRESH_CRITICAL, color="#c0392b", linewidth=1.8, linestyle="--", label=f"Urgent ({THRESH_CRITICAL:.2f})")
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
            "The 0.11 trigger was optimised to minimise total business cost."
        )

    with col_meta:
        st.subheader("Key Metrics")
        st.metric("Churn Probability",          f"{churn_prob:.1%}")
        st.metric("Risk Tier",                  label.split(" - ")[0])
        st.metric("Annual Revenue at Risk",     f"EUR {monthly_charges * 12:,.0f}")

    # -- SHAP Waterfall --------------------------------------------------------
    st.divider()
    st.subheader("Why this prediction? - SHAP Feature Explanation")
    st.markdown(
        "The waterfall chart shows how each feature pushed the model output away from "
        "the average prediction (base value). "
        "**Red bars** push toward churn; **blue bars** push away from churn. "
        "Values are in log-odds space -- larger magnitude means larger influence."
    )

    shap_vals = explainer.shap_values(X_input)
    explanation = shap.Explanation(
        values=shap_vals[0],
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
        import numpy as np
        vals = shap_vals[0]
        indices = np.argsort(np.abs(vals))[-12:]
        fig_fb, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals[indices]]
        ax.barh([FEATURE_COLS[i] for i in indices], vals[indices], color=colors)
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

    with col_b:
        st.subheader("Estimated Annual Impact")

        y_pred_roi = (y_prob_arr >= roi_thresh_roi).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test_arr, y_pred_roi).ravel()
        n_test = len(y_test_arr)

        test_churn_rate = y_test_arr.mean()
        user_churn_rate = roi_churn_rate / 100
        scale_factor = (roi_customers * user_churn_rate) / (n_test * test_churn_rate)

        tp_scaled = int(tp_t * scale_factor)
        fp_scaled = int(fp_t * scale_factor)
        fn_scaled = int(fn_t * scale_factor)
        n_churners = int(roi_customers * user_churn_rate)

        recall_actual    = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        precision_actual = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0

        annual_rev      = roi_monthly * 12
        saved_customers = int(tp_scaled * roi_retention / 100)
        revenue_saved   = saved_customers * annual_rev
        offer_spend     = (tp_scaled + fp_scaled) * roi_offer_cost
        net_saving      = revenue_saved - offer_spend
        missed_cost     = fn_scaled * annual_rev

        st.metric("Churners in Customer Base", f"{n_churners:,}")
        st.metric(
            "Churners Caught by Model", f"{tp_scaled:,}",
            delta=f"{recall_actual:.0%} recall (from test set)"
        )
        st.metric(
            "False Alarms (wasted offers)", f"{fp_scaled:,}",
            delta=f"{precision_actual:.0%} precision (from test set)"
        )
        st.metric("Customers Retained",    f"{saved_customers:,}")
        st.metric("Revenue Saved",         f"EUR {revenue_saved:,.0f}")
        st.metric("Total Offer Spend",     f"EUR {offer_spend:,.0f}")
        st.metric(
            "Net Annual Saving", f"EUR {net_saving:,.0f}",
            delta="vs EUR 0 with no model"
        )
        st.metric("Revenue Lost (missed churners)", f"EUR {missed_cost:,.0f}")

    # Bar chart 
    st.divider()
    st.subheader("Total Annual Churn Cost: Model vs No Model")

    no_model_cost = n_churners * annual_rev
    model_cost    = missed_cost + offer_spend

    fig_roi, ax_roi = plt.subplots(figsize=(7, 4))
    scenarios = [
        "No Model\n(lose all churners)",
        f"With Model\n(threshold = {roi_thresh_roi:.2f})",
    ]
    values_k    = [no_model_cost / 1000, model_cost / 1000]
    color_model = "#27ae60" if net_saving > 0 else "#e67e22"
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

    saving_k = (no_model_cost - model_cost) / 1000
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
