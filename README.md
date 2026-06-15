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

---

## Explainability (SHAP + LIME)
Banking regulation (SR 11-7, ECOA / Reg B) requires lenders to explain credit decisions. This project
provides two independent methods:
- **SHAP** — global beeswarm/bar importance plus per-borrower waterfall plots whose contributions
  provably sum to the prediction.
- **LIME** — a local linear surrogate as a second opinion on the same borrower.

Both agree with the logistic-regression coefficients: **past delinquency** and **revolving
utilization** dominate default risk.

---

## Macroeconomic Stress Testing
The test portfolio is re-scored under adverse scenarios (income falls, utilization/debt rise,
delinquencies climb). Expected loss assumes 65% LGD on a $10,000 exposure.

| Scenario | Mean PD | Approval @0.20 | Expected loss / loan |
|---|---|---|---|
| Baseline | 6.6% | 91.7% | $431 |
| Mild Recession | 8.1% | 89.1% | $525 |
| Severe Recession | 44.4% | 0.4% | $2,889 |

---

## Live Scoring Dashboard
`app.py` is a Streamlit app that loads the serialized model and, for any borrower, returns:
1. a **default probability** and approve / review / decline recommendation,
2. a **SHAP waterfall** explaining the score, and
3. the **top 3 risk factors** driving the decision.

```bash
streamlit run app.py
```
The app loads artifacts from `models/` and never retrains on a click.
