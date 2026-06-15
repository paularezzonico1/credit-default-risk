# Model Card — Credit Default Risk Scorer

**Model name:** Credit Default Risk Scorer (tuned XGBoost)
**Version:** 1.0
**Owner:** Paula Rezzonico
**Date:** 2026-06-14
**Status:** Demonstration / educational — not approved for production lending decisions

A gradient-boosted decision-tree model that estimates the probability a retail borrower becomes
90+ days delinquent within two years, trained on the public *Give Me Some Credit* dataset
(150,000 borrowers). The model is paired with SHAP/LIME explanations and a macroeconomic stress test.

---

## Model Details
- **Architecture:** XGBoost gradient-boosted trees (`learning_rate=0.1`, `max_depth=3`,
  `n_estimators=200`), selected by grid search under 3-fold CV.
- **Inputs (11 features):** revolving utilization, age, debt ratio, monthly income, number of open
  credit lines, real-estate loans, dependents, three delinquency counts (30-59 / 60-89 / 90+ days
  past due), and a K-Means borrower-segment label engineered upstream.
- **Output:** calibrated-ish probability of default in [0, 1]; a binary decision is produced by
  thresholding at **0.20**.
- **Feature scaling:** none for the production model — tree ensembles are scale-invariant, which also
  keeps SHAP/LIME attributions in real borrower units.
- **Frameworks:** scikit-learn 1.3, XGBoost 2.0, SHAP 0.46, LIME 0.2.

---

## Intended Use
- **Primary use:** decision-support for a retail-credit risk analyst — ranking and triaging consumer
  loan applications, and portfolio-level what-if / stress analysis.
- **Intended users:** bank credit-risk and model-validation teams.
- **Decision flow:** scores below 0.20 → *approve*, 0.20-0.40 → *manual review*, above 0.40 →
  *decline*. The model augments, and never replaces, human adjudication.

---

## Out-of-Scope & Prohibited Uses
- Not for fully automated, human-out-of-the-loop credit denial.
- Not validated for commercial, mortgage, or non-US-consumer lending.
- Must not use prohibited-basis attributes (race, sex, religion, national origin, etc.) or close
  proxies as inputs — doing so would violate ECOA / Reg B.
- The training data is a static 2011 competition dataset; the model is **not** fit for live
  production use without retraining on current, representative, and governed data.
