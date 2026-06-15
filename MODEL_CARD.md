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

---

## Training Data
- **Source:** Kaggle *Give Me Some Credit* (`cs-training.csv`), 150,000 borrowers, 11 features.
- **Target:** `SeriousDlqin2yrs` — 90+ days past due within 2 years. **Class balance: 6.7% positive.**
- **Missing data:** `MonthlyIncome` (~20%) and `NumberOfDependents` (~2.6%) imputed with the median
  (robust to the heavily right-skewed income distribution).
- **Split:** time-ordered 80/20 (first 120k rows train, last 30k test) to mimic deployment — training
  on past loans and predicting future ones, avoiding look-ahead leakage.

---

## Evaluation

**Hold-out test ROC-AUC (single 80/20 split)**

| Model | Test ROC-AUC |
|---|---|
| Decision Tree (deep) | 0.613 |
| Decision Tree (shallow) | 0.818 |
| Logistic Regression (L1/L2) | 0.801 |
| Random Forest | 0.844 |
| MLP neural network (64, 32) | 0.844 |
| XGBoost (default) | 0.864 |
| **XGBoost (tuned, production)** | **0.871** |

**5-fold stratified cross-validation (mean ± std AUC)**

| Model | CV ROC-AUC |
|---|---|
| Logistic Regression | 0.790 ± 0.004 |
| MLP (64, 32) | 0.834 ± 0.003 |
| Random Forest | 0.839 ± 0.005 |
| XGBoost | 0.858 ± 0.004 |

The tight CV standard deviations (≤0.005) confirm the ranking is stable, not an artifact of one split.

---

## Operating Point (threshold = 0.20)
Accuracy is meaningless on a 93/7 imbalanced target (predicting "no default" always scores 93%).
The threshold is a **business decision** driven by cost asymmetry: a missed default loses the loan
principal, while a false alarm only forgoes interest. At 0.20 on the 30,000-loan test set:

- **Recall (defaulters caught): 51.1%** — 1,044 of 2,044 true defaulters flagged.
- **Precision: 42.0%** — 1,440 false alarms among 27,956 good borrowers.
- This deliberately trades precision for recall, consistent with lending loss economics.

---

## Stress Testing
The test portfolio was re-scored under three macroeconomic scenarios (income shocks, rising
utilization/debt, higher delinquency). Expected loss assumes 65% LGD on a $10,000 exposure.

| Scenario | Mean predicted PD | Approval rate @0.20 | Expected loss / loan |
|---|---|---|---|
| Baseline | 6.6% | 91.7% | $431 |
| Mild Recession | 8.1% | 89.1% | $525 |
| Severe Recession | 44.4% | 0.4% | $2,889 |

The sharp, non-linear deterioration under severe stress is the early-warning behaviour a risk
committee uses to size capital buffers and tighten underwriting.

---

## Explainability
Every prediction is explainable, supporting ECOA adverse-action requirements and SR 11-7 model risk
management:
- **SHAP** (`TreeExplainer`) — global feature importance and per-borrower waterfall attributions whose
  contributions provably sum to the prediction.
- **LIME** — an independent local linear surrogate used as a second opinion per decision.
- Both methods, and the logistic-regression coefficients, agree that **past delinquency** and
  **revolving utilization** are the dominant default drivers.
