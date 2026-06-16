"""Train the credit-default-risk scoring pipeline and persist its artifacts.

This is the standalone, importable counterpart to the project notebook
(`notebooks/Credit Default Risk Analysis.ipynb`). It reproduces the modelling
phases that matter for production:

    1. Load & preprocess the "Give Me Some Credit" data.
    2. Segment borrowers with K-Means and attach the cluster as a feature.
    3. Cross-validate candidate models, then tune XGBoost with GridSearchCV.
    4. Train the production XGBoost on raw features and set up a SHAP explainer.
    5. Serialize the scaler, K-Means, model, and metadata to ``models/`` so that
       ``app.py`` and the test-suite can load them unchanged.

Run it directly to regenerate the artifacts::

    python train_model.py

or import individual functions to reuse the pipeline (e.g. ``stress_test.py``).
"""

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Configuration ----------------------------------------------------------

TARGET = "SeriousDlqin2yrs"

# The 11 model features, in the order the serialized model expects them.
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "cluster",
]

# Features used for unsupervised borrower segmentation (K-Means).
CLUSTER_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "MonthlyIncome",
    "DebtRatio",
    "NumberOfTimes90DaysLate",
]

# Business-driven decision threshold (a missed default costs far more than a
# false alarm), and the global random seed for reproducibility.
THRESHOLD = 0.20
RANDOM_STATE = 42

# Hyperparameter grid for tuning XGBoost.
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.1, 0.3],
}


# --- Pipeline steps ----------------------------------------------------------

def find_project_root(markers=("requirements.txt", "README.md")):
    """Walk up from the cwd until a directory containing a marker file is found.

    Lets the script run from the repo root or from a subdirectory.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if any(os.path.exists(os.path.join(path, m)) for m in markers):
            return path
        path = os.path.dirname(path)
    return os.path.dirname(os.path.abspath(__file__))


def load_data(data_path):
    """Load the raw training CSV (first column is the borrower index)."""
    df = pd.read_csv(data_path, index_col=0)
    print(f"Loaded {df.shape[0]:,} borrowers x {df.shape[1]} columns")
    return df


def preprocess(df):
    """Median-impute the two columns with missing values."""
    df = df.copy()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )
    assert df.isnull().sum().sum() == 0, "missing values remain after imputation"
    return df


def add_cluster_feature(df):
    """Fit K-Means (k=4) on scaled segmentation features and attach ``cluster``.

    Returns the augmented frame, the fitted K-Means, and a ``cluster_scaler``
    re-fit on the full frame so it can be reused at scoring time (the app and
    the stress test scale raw borrower inputs with it before ``kmeans.predict``).
    """
    df = df.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Re-fit a clean scaler on the full data for downstream scoring.
    cluster_scaler = StandardScaler().fit(df[CLUSTER_FEATURES])
    return df, kmeans, cluster_scaler


def split_data(df):
    """Time-ordered 80/20 split (mirrors deployment; a random split would leak).

    The data is ordered by loan time, so the first 80% trains and the last 20%
    is held out for evaluation.
    """
    X = df[FEATURES]
    y = df[TARGET]
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test


def cross_validate(X, y):
    """5-fold stratified CV (mean ± std AUC) for the candidate models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=200)),
        ]),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0,
        ),
        "MLP (64,32)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                  max_iter=80, random_state=RANDOM_STATE)),
        ]),
    }

    cv_results = {}
    print(f"\n{'Model':<22}{'Mean AUC':>10}{'Std':>9}")
    print("-" * 41)
    for name, model in cv_models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
        print(f"{name:<22}{scores.mean():>10.4f}{scores.std():>9.4f}")
    return cv_results


def tune_xgboost(X_train, y_train):
    """Grid-search XGBoost hyperparameters; return the best parameter dict."""
    grid = GridSearchCV(
        xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0),
        PARAM_GRID, cv=3, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV AUC:     {grid.best_score_:.4f}")
    return grid.best_params_


def train_production_model(X_train, y_train, best_params):
    """Fit the production XGBoost on raw features (trees are scale-invariant,
    which keeps SHAP/LIME explanations in real borrower units)."""
    model = xgb.XGBClassifier(
        **best_params, random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Return the hold-out test ROC-AUC."""
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nProduction XGBoost  |  Test ROC-AUC: {auc:.4f}")
    return auc


def build_shap_explainer(model):
    """Construct a SHAP TreeExplainer for the fitted model."""
    import shap

    explainer = shap.TreeExplainer(model)
    print("SHAP TreeExplainer ready")
    return explainer


def save_artifacts(cluster_scaler, kmeans, model, metadata, models_dir):
    """Serialize the scoring pipeline so app.py / tests can load it unchanged."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(cluster_scaler, os.path.join(models_dir, "cluster_scaler.joblib"))
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans.joblib"))
    joblib.dump(model, os.path.join(models_dir, "xgb_model.joblib"))
    joblib.dump(metadata, os.path.join(models_dir, "metadata.joblib"))
    print(f"\nSaved artifacts to {models_dir}")
    print(sorted(os.listdir(models_dir)))


def main():
    """Run the full training pipeline and persist the artifacts."""
    root = find_project_root()
    data_path = os.path.join(root, "data", "cs-training.csv")
    models_dir = os.path.join(root, "models")

    df = load_data(data_path)
    df = preprocess(df)
    df, kmeans, cluster_scaler = add_cluster_feature(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # Robustness check across candidate models, then tune the chosen one.
    cross_validate(df[FEATURES], df[TARGET])
    best_params = tune_xgboost(X_train, y_train)

    model = train_production_model(X_train, y_train, best_params)
    test_auc = evaluate(model, X_test, y_test)
    build_shap_explainer(model)

    metadata = {
        "features": FEATURES,
        "cluster_features": CLUSTER_FEATURES,
        "threshold": THRESHOLD,
        "test_auc": float(test_auc),
    }
    save_artifacts(cluster_scaler, kmeans, model, metadata, models_dir)

    # Sanity check: a reloaded model must reproduce the test AUC exactly.
    reloaded = joblib.load(os.path.join(models_dir, "xgb_model.joblib"))
    reload_auc = roc_auc_score(y_test, reloaded.predict_proba(X_test)[:, 1])
    assert abs(reload_auc - test_auc) < 1e-9, "reloaded model must match"
    print(f"Reloaded model reproduces test AUC = {reload_auc:.4f}  [OK]")


if __name__ == "__main__":
    main()
