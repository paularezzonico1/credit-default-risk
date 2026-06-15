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
