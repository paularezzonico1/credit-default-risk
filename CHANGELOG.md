# Changelog

## Unreleased
- Added `docs/requirements.md`: a formal business and functional requirements document
  (BRD/FRD) with numbered traceable requirements, acceptance criteria, a requirements
  traceability matrix, and a sign-off/RACI table.

## 1.0 — 2026-06-14
- Added 5-fold cross-validation and `GridSearchCV` hyperparameter tuning.
- Added an MLP neural network to the model bake-off.
- Added evaluation visuals: confusion matrix, ROC curve, gain + permutation feature importance.
- Added SHAP (global + per-borrower waterfall) and LIME explanations.
- Added a three-scenario macroeconomic stress test.
- Serialized the scoring pipeline; rebuilt the Streamlit app to load artifacts and show a SHAP
  waterfall plus the top-3 risk factors per decision.
- Added a model card, a test suite, and comprehensive documentation.
