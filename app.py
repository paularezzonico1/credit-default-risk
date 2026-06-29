"""Streamlit scoring dashboard — a thin client over the scoring API.

This UI no longer loads the model. It calls the FastAPI service over HTTP, so
the presentation layer and the model-serving layer are fully decoupled (the way
a real system separates them). If the API is unreachable, the app shows a clear
error instead of crashing.

Point it at the API with the ``CDR_API_URL`` env var (default localhost:8000).
"""

import os

import matplotlib.pyplot as plt
import requests
import streamlit as st

API_URL = os.environ.get("CDR_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("CDR_API_TIMEOUT", "10"))

st.set_page_config(page_title="Credit Default Risk Scorer", page_icon="🏦", layout="wide")


def _api_get(path: str):
    """GET helper returning parsed JSON, or None on any connection failure."""
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


# Confirm the API is reachable and the model is loaded before rendering inputs.
info = _api_get("/model/info")
if info is None:
    st.title("🏦 Credit Default Risk Scorer")
    st.error(
        f"Cannot reach the scoring API at `{API_URL}`.\n\n"
        "Start it with `uvicorn api.main:app --reload` (or `docker compose up`), "
        "then set `CDR_API_URL` if it runs elsewhere."
    )
    st.stop()

st.title("🏦 Credit Default Risk Scorer")
st.markdown("Enter borrower details to get an instant, **explainable** default-probability score.")
st.caption(
    f"Model {info['model_version']} · Hold-out AUC {info['holdout_auc']:.3f} · "
    f"Decision threshold {info['threshold']:.2f} · Served via API at {API_URL}"
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
    payload = {
        "revolving_utilization": revolving_util,
        "age": int(age),
        "num_30_59_days_late": int(late_30_59),
        "debt_ratio": debt_ratio,
        "monthly_income": monthly_income,
        "num_open_credit_lines": int(open_credit_lines),
        "num_90_days_late": int(late_90),
        "num_real_estate_loans": int(real_estate_loans),
        "num_60_89_days_late": int(late_60_89),
        "num_dependents": int(num_dependents),
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        st.error(f"Lost connection to the scoring API at `{API_URL}`. Is it still running?")
        st.stop()

    if resp.status_code == 422:
        st.error("The API rejected these inputs as invalid:")
        st.json(resp.json())
        st.stop()
    if resp.status_code != 200:
        st.error(f"Scoring failed (HTTP {resp.status_code}).")
        st.json(resp.json())
        st.stop()

    result = resp.json()
    prob = result["probability_of_default"]
    decision = result["decision"]
    risk_label = {"APPROVE": "🟢 LOW RISK", "REVIEW": "🟡 MEDIUM RISK",
                  "DECLINE": "🔴 HIGH RISK"}[decision]

    c1, c2, c3 = st.columns(3)
    c1.metric("Default Probability", f"{prob:.1%}")
    c2.metric("Risk Level", risk_label)
    c3.metric("Recommendation", decision)
    st.progress(min(prob, 1.0))
    st.caption(f"Decision id `{result['request_id']}` · model {result['model_version']}")

    st.divider()
    left, right = st.columns([3, 2])
    factors = result["top_risk_factors"]
    with left:
        st.subheader("Top risk factors (SHAP contributions)")
        labels = [f["label"] for f in factors][::-1]
        values = [f["shap_value"] for f in factors][::-1]
        colors = ["#c0392b" if v >= 0 else "#27ae60" for v in values]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="#444", linewidth=0.8)
        ax.set_xlabel("SHAP value (log-odds) — positive pushes toward default")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("Why this score?")
        for f in factors:
            color = "#c0392b" if f["shap_value"] >= 0 else "#27ae60"
            arrow = "▲" if f["shap_value"] >= 0 else "▼"
            st.markdown(
                f"**{f['label']}**  \n"
                f"<span style='color:{color}'>{arrow} {f['shap_value']:+.3f} {f['direction']}</span>",
                unsafe_allow_html=True,
            )
        st.caption("SHAP values are in log-odds units. Positive = pushes toward default.")

st.divider()
st.caption("UI decoupled from model serving · scores come from the FastAPI service · "
           "see MODEL_CARD.md for limitations.")
