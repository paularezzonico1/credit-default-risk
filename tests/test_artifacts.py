"""Smoke tests for the serialized scoring pipeline."""
import os
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def test_all_artifacts_present():
    for name in ["cluster_scaler.joblib", "kmeans.joblib", "xgb_model.joblib", "metadata.joblib"]:
        assert os.path.exists(os.path.join(MODELS_DIR, name)), f"missing {name}"


def test_metadata_schema():
    meta = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))
    assert set(["features", "cluster_features", "threshold", "test_auc"]) <= set(meta)
    assert len(meta["features"]) == 11
    assert 0.0 < meta["threshold"] < 1.0
