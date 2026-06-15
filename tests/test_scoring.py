"""Validate the production model scores and clears the documented AUC bar."""
import os
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


def _load():
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    meta = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))
    return model, meta


def test_model_auc_bar():
    _, meta = _load()
    assert meta["test_auc"] > 0.85, "production AUC regressed below 0.85"


def test_high_risk_borrower_scores_high():
    model, meta = _load()
    cluster_scaler = joblib.load(os.path.join(MODELS_DIR, "cluster_scaler.joblib"))
    kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.joblib"))
    features, cluster_features = meta["features"], meta["cluster_features"]
    # delinquent, high-utilization, low-income borrower
    cl = pd.DataFrame([[0.95, 29, 1800, 0.8, 3]], columns=cluster_features)
    cluster = int(kmeans.predict(cluster_scaler.transform(cl))[0])
    borrower = pd.DataFrame([[0.95, 29, 4, 0.8, 1800, 5, 3, 0, 2, 2, cluster]], columns=features)
    prob = float(model.predict_proba(borrower)[0][1])
    assert prob > meta["threshold"], "obvious high-risk borrower should breach the threshold"
