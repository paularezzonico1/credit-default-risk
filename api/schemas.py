"""Pydantic request/response schemas for the scoring API (pydantic v1).

The input schema mirrors the borrower fields the Streamlit app collects and the
model consumes, but exposes them under clear snake_case names with field-level
validation. The ``cluster`` feature is intentionally absent: it is derived
server-side by the K-Means segmenter, never supplied by the caller.
"""

from typing import List

from pydantic import BaseModel, Field, validator


class BorrowerInput(BaseModel):
    """A single borrower's features, with sensible validation bounds.

    Bounds reject malformed input (negative incomes, impossible ages) while
    staying wide enough to accept the genuine outliers present in the Give Me
    Some Credit data (e.g. utilization and debt-ratio values far above 1).
    """

    revolving_utilization: float = Field(
        ..., ge=0.0, le=60000.0,
        description="Revolving credit utilization ratio (balance / limit).",
    )
    age: int = Field(..., ge=18, le=120, description="Borrower age in years.")
    num_30_59_days_late: int = Field(
        ..., ge=0, le=98, description="Times 30-59 days past due.",
    )
    debt_ratio: float = Field(
        ..., ge=0.0, le=500000.0,
        description="Monthly debt payments / monthly gross income.",
    )
    monthly_income: float = Field(
        ..., ge=0.0, le=10_000_000.0, description="Monthly gross income ($).",
    )
    num_open_credit_lines: int = Field(
        ..., ge=0, le=100, description="Open credit lines and loans.",
    )
    num_90_days_late: int = Field(
        ..., ge=0, le=98, description="Times 90+ days past due.",
    )
    num_real_estate_loans: int = Field(
        ..., ge=0, le=100, description="Real-estate loans or lines.",
    )
    num_60_89_days_late: int = Field(
        ..., ge=0, le=98, description="Times 60-89 days past due.",
    )
    num_dependents: int = Field(
        ..., ge=0, le=50, description="Number of dependents.",
    )

    @validator("revolving_utilization", "debt_ratio", "monthly_income")
    def _reject_nan(cls, v: float) -> float:
        """Reject NaN/inf, which slip past numeric range checks."""
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("must be a finite number")
        return v

    class Config:
        schema_extra = {
            "example": {
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
        }


class RiskFactor(BaseModel):
    """One feature's contribution to a decision, in SHAP log-odds units."""

    feature: str = Field(..., description="Raw model feature name.")
    label: str = Field(..., description="Human-readable feature label.")
    shap_value: float = Field(..., description="SHAP contribution (log-odds).")
    direction: str = Field(..., description="'toward default' or 'away from default'.")


class PredictionResponse(BaseModel):
    """Scoring result for a single borrower."""

    request_id: str = Field(..., description="Unique id for this decision (audit key).")
    probability_of_default: float = Field(..., description="Model PD in [0, 1].")
    decision: str = Field(..., description="APPROVE, REVIEW, or DECLINE.")
    threshold: float = Field(..., description="Approve/decline threshold applied.")
    top_risk_factors: List[RiskFactor] = Field(
        ..., description="Top 3 SHAP drivers of this decision.",
    )
    model_version: str = Field(..., description="Version id of the scoring model.")


class HealthResponse(BaseModel):
    """Liveness/readiness payload for ``GET /health``."""

    status: str = Field(..., description="'healthy' when artifacts are loaded.")
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    """Model governance metadata for ``GET /model/info``."""

    model_version: str
    training_date: str = Field(..., description="ISO timestamp the artifact was built.")
    holdout_auc: float = Field(..., description="Documented hold-out ROC-AUC.")
    threshold: float
    n_features: int
    features: List[str]
