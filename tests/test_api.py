"""API tests using FastAPI's TestClient (Phase A)."""

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app, service

# Ensure the model is loaded for the test session (startup events only fire
# inside a `with TestClient(...)` context, but loading explicitly is robust).
if not service.is_loaded and service.artifacts_present():
    service.load()

client = TestClient(app)

VALID_BORROWER = {
    "revolving_utilization": 0.15,
    "age": 45,
    "num_30_59_days_late": 0,
    "debt_ratio": 0.37,
    "monthly_income": 5400,
    "num_open_credit_lines": 8,
    "num_90_days_late": 0,
    "num_real_estate_loans": 1,
    "num_60_89_days_late": 0,
    "num_dependents": 0,
}

artifacts_required = pytest.mark.skipif(
    not service.artifacts_present(),
    reason="model artifacts not present; run train_model.py",
)


@artifacts_required
def test_health_reports_loaded():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith("xgb-")


@artifacts_required
def test_model_info_schema():
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["holdout_auc"] > 0.85
    assert body["n_features"] == len(body["features"]) == 11
    assert 0.0 < body["threshold"] < 1.0
    assert body["training_date"]


@artifacts_required
def test_predict_valid_request():
    r = client.post("/predict", json=VALID_BORROWER)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert len(body["top_risk_factors"]) == 3
    assert body["request_id"]
    assert body["model_version"].startswith("xgb-")


@artifacts_required
def test_predict_high_risk_borrower_declines_or_reviews():
    """A delinquent, over-extended, low-income borrower should not be APPROVE."""
    risky = dict(VALID_BORROWER)
    risky.update(
        revolving_utilization=0.95, age=29, monthly_income=1800,
        debt_ratio=0.8, num_90_days_late=3, num_30_59_days_late=2,
        num_60_89_days_late=2,
    )
    r = client.post("/predict", json=risky)
    assert r.status_code == 200
    assert r.json()["decision"] in {"REVIEW", "DECLINE"}


def test_predict_missing_field_returns_422():
    bad = dict(VALID_BORROWER)
    del bad["age"]
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_out_of_range_returns_422():
    for field, value in [
        ("age", 5),                       # below 18
        ("monthly_income", -100),         # negative
        ("num_90_days_late", 999),        # above 98
        ("revolving_utilization", -0.1),  # negative
    ]:
        bad = dict(VALID_BORROWER)
        bad[field] = value
        r = client.post("/predict", json=bad)
        assert r.status_code == 422, f"{field}={value} should be rejected"
