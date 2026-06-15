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
