import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import pickle

st.set_page_config(page_title="Credit Default Risk Scorer", page_icon="🏦", layout="centered")

st.title("🏦 Credit Default Risk Scorer")
st.markdown("Enter borrower details below to get an instant default probability.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=45)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=500000, value=5400)
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=10.0, value=0.37)
    revolving_util = st.number_input("Revolving Credit Utilization", min_value=0.0, max_value=1.0, value=0.15)
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

with col2:
    late_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, max_value=98, value=0)
    late_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, max_value=98, value=0)
    late_90 = st.number_input("Times 90+ Days Late", min_value=0, max_value=98, value=0)
    open_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=58, value=8)
    real_estate_loans = st.number_input("Real Estate Loans", min_value=0, max_value=54, value=1)

st.divider()

if st.button("Calculate Default Risk", type="primary", use_container_width=True):

    # Load and prepare training data to fit models
    with st.spinner("Running model..."):
        DATA_PATH = '/Users/paularezzonico/Desktop/credit-default-risk/data/cs-training.csv'
        df = pd.read_csv(DATA_PATH, index_col=0)
        df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
        df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)

        # Fit cluster model
        cluster_features = ['RevolvingUtilizationOfUnsecuredLines', 'age',
                            'MonthlyIncome', 'DebtRatio', 'NumberOfTimes90DaysLate']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[cluster_features])
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        km.fit(X_scaled)

        # Get cluster for this borrower
        borrower_cluster_input = np.array([[revolving_util, age, monthly_income, debt_ratio, late_90]])
        borrower_cluster_scaled = scaler.transform(borrower_cluster_input)
        cluster = km.predict(borrower_cluster_scaled)[0]

        # Fit XGBoost
        features = ['RevolvingUtilizationOfUnsecuredLines', 'age',
                    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents', 'cluster']
        df['cluster'] = km.predict(scaler.transform(df[cluster_features]))
        X = df[features]
        y = df['SeriousDlqin2yrs']
        split = int(len(df) * 0.8)
        model_scaler = StandardScaler()
        X_train_sc = model_scaler.fit_transform(X.iloc[:split])
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                        eval_metric='logloss', verbosity=0)
        xgb_model.fit(X_train_sc, y.iloc[:split])

        # Score the borrower
        borrower = np.array([[revolving_util, age, late_30_59, debt_ratio,
                               monthly_income, open_credit_lines, late_90,
                               real_estate_loans, late_60_89, num_dependents, cluster]])
        borrower_sc = model_scaler.transform(borrower)
        prob = xgb_model.predict_proba(borrower_sc)[0][1]

    # Display result
    st.subheader("Risk Assessment Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Default Probability", f"{prob:.1%}")
    with col2:
        if prob < 0.20:
            risk = "🟢 LOW RISK"
        elif prob < 0.40:
            risk = "🟡 MEDIUM RISK"
        else:
            risk = "🔴 HIGH RISK"
        st.metric("Risk Level", risk)
    with col3:
        recommendation = "APPROVE" if prob < 0.20 else ("REVIEW" if prob < 0.40 else "DECLINE")
        st.metric("Recommendation", recommendation)

    st.progress(float(prob))

    st.divider()
    st.markdown("**Key Risk Factors:**")
    factors = {
        "Late Payments (90+ days)": late_90,
        "Late Payments (30-59 days)": late_30_59,
        "Debt Ratio": debt_ratio,
        "Age": age,
        "Monthly Income": f"${monthly_income:,}"
    }
    for k, v in factors.items():
        st.markdown(f"- **{k}:** {v}")

st.divider()
st.caption("Model: XGBoost | AUC: 0.863 | Threshold: 0.20 | Data: Give Me Some Credit (Kaggle)")