# Model Card — Credit Default Risk Scorer

**Model name:** Credit Default Risk Scorer (tuned XGBoost)
**Version:** 1.0
**Owner:** Paula Rezzonico
**Date:** 2026-06-14
**Status:** Demonstration / educational — not approved for production lending decisions

A gradient-boosted decision-tree model that estimates the probability a retail borrower becomes
90+ days delinquent within two years, trained on the public *Give Me Some Credit* dataset
(150,000 borrowers). The model is paired with SHAP/LIME explanations and a macroeconomic stress test.
