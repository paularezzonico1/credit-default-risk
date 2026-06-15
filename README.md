# 🏦 Credit Default Risk Predictor

An end-to-end, **explainable**, and **stress-tested** machine-learning system that predicts retail
loan default on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset
(150,000 borrowers). Built to demonstrate the full modelling lifecycle a bank credit-risk team
actually runs — from EDA to a governed, deployable scoring app.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![SHAP](https://img.shields.io/badge/SHAP-explainable-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Why this project stands out
- **11-stage pipeline:** EDA → segmentation → regularized baseline → model bake-off (incl. a neural
  network) → cross-validation & tuning → evaluation visuals → SHAP → LIME → stress testing → productionization.
- **Rigorous validation:** 5-fold stratified cross-validation with mean ± std, not a single lucky split.
- **Fully explainable:** SHAP waterfalls and LIME explanations for every decision — audit-ready.
- **Stress-tested:** the portfolio is re-scored under three macroeconomic scenarios (CCAR-style).
- **Deployable:** a Streamlit app loads serialized artifacts and returns an instant, explained score.
- **Governed:** a complete [model card](MODEL_CARD.md) documents intended use, fairness, and limitations.

---

## Results

**Hold-out test ROC-AUC**

| Model | Test ROC-AUC |
|---|---|
| Decision Tree (deep) | 0.613 |
| Decision Tree (shallow) | 0.818 |
| Logistic Regression (L1/L2) | 0.801 |
| Random Forest | 0.844 |
| MLP neural network (64, 32) | 0.844 |
| XGBoost (default) | 0.864 |
| **XGBoost (tuned)** | **0.871** |

**5-fold cross-validation (mean ± std AUC):** Logistic 0.790 ± 0.004 · MLP 0.834 ± 0.003 ·
Random Forest 0.839 ± 0.005 · **XGBoost 0.858 ± 0.004**.

**Final model:** tuned XGBoost at threshold **0.20** (recall ≈ 51%), optimized for the lending cost
asymmetry — a missed default costs far more than a false alarm.
