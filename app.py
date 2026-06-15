import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap

st.set_page_config(page_title="Credit Default Risk Scorer", page_icon="🏦", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@st.cache_resource
def load_artifacts():
    """Load the fitted scoring pipeline once and cache it across reruns."""
    cluster_scaler = joblib.load(os.path.join(MODELS_DIR, "cluster_scaler.joblib"))
    kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    meta = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))
    explainer = shap.TreeExplainer(model)
    return cluster_scaler, kmeans, model, meta, explainer


cluster_scaler, kmeans, model, meta, explainer = load_artifacts()
features = meta["features"]
cluster_features = meta["cluster_features"]
THRESHOLD = meta["threshold"]

# Human-readable labels for the model's raw feature names
LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving credit utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 days late (count)",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "90+ days late (count)",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days late (count)",
    "NumberOfDependents": "Dependents",
    "cluster": "Borrower segment",
}

st.title("🏦 Credit Default Risk Scorer")
st.markdown("Enter borrower details to get an instant, **explainable** default-probability score.")
st.caption(
    f"Model: tuned XGBoost · Hold-out AUC {meta['test_auc']:.3f} · "
    f"Decision threshold {THRESHOLD:.2f} · Data: Give Me Some Credit (Kaggle)"
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 45)
    monthly_income = st.number_input("Monthly Income ($)", 0, 500000, 5400)
    debt_ratio = st.number_input("Debt Ratio", 0.0, 10.0, 0.37)
    revolving_util = st.number_input("Revolving Credit Utilization", 0.0, 2.0, 0.15)
    num_dependents = st.number_input("Number of Dependents", 0, 20, 0)
with col2:
    late_30_59 = st.number_input("Times 30-59 Days Late", 0, 98, 0)
    late_60_89 = st.number_input("Times 60-89 Days Late", 0, 98, 0)
    late_90 = st.number_input("Times 90+ Days Late", 0, 98, 0)
    open_credit_lines = st.number_input("Open Credit Lines", 0, 58, 8)
    real_estate_loans = st.number_input("Real Estate Loans", 0, 54, 1)

st.divider()

if st.button("Calculate Default Risk", type="primary", use_container_width=True):
    # Assign the borrower to a K-Means risk segment, then score in training feature order.
    cluster_row = pd.DataFrame(
        [[revolving_util, age, monthly_income, debt_ratio, late_90]],
        columns=cluster_features,
    )
    cluster = int(kmeans.predict(cluster_scaler.transform(cluster_row))[0])

    borrower = pd.DataFrame(
        [[revolving_util, age, late_30_59, debt_ratio, monthly_income,
          open_credit_lines, late_90, real_estate_loans, late_60_89,
          num_dependents, cluster]],
        columns=features,
    )
    prob = float(model.predict_proba(borrower)[0][1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Default Probability", f"{prob:.1%}")
    if prob < THRESHOLD:
        risk, rec = "🟢 LOW RISK", "APPROVE"
    elif prob < 0.40:
        risk, rec = "🟡 MEDIUM RISK", "REVIEW"
    else:
        risk, rec = "🔴 HIGH RISK", "DECLINE"
    c2.metric("Risk Level", risk)
    c3.metric("Recommendation", rec)
    st.progress(min(prob, 1.0))

    # SHAP explanation for this single decision
    shap_row = explainer(borrower)
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Why this score? (SHAP waterfall)")
        shap.plots.waterfall(shap_row[0], max_display=8, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        st.pyplot(fig, clear_figure=True)
