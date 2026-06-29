"""Model-serving layer: load artifacts once and score borrowers.

This isolates everything that touches the serialized model so the FastAPI
endpoints stay thin and the same logic is reused by the test-suite. It does NOT
retrain or alter the model — it only loads the artifacts produced by
``train_model.py`` and applies them, exactly as the Streamlit app used to.

``model_version`` is derived from a SHA-256 content hash of the model artifact.
That gives an immutable, reproducible identifier without modifying the training
pipeline: the same model file always yields the same version string.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import shap

# Human-readable labels for the model's raw feature names (moved here from the
# Streamlit app so the API can return labelled risk factors).
FEATURE_LABELS: Dict[str, str] = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving credit utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 days late (count)",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "90+ days late (count)",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days late (count)",
    "NumberOfDependents": "Dependents",
    "cluster": "Borrower segment",
}

# Map validated API input field names -> raw model feature names.
INPUT_TO_FEATURE: Dict[str, str] = {
    "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines",
    "age": "age",
    "num_30_59_days_late": "NumberOfTime30-59DaysPastDueNotWorse",
    "debt_ratio": "DebtRatio",
    "monthly_income": "MonthlyIncome",
    "num_open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "num_90_days_late": "NumberOfTimes90DaysLate",
    "num_real_estate_loans": "NumberRealEstateLoansOrLines",
    "num_60_89_days_late": "NumberOfTime60-89DaysPastDueNotWorse",
    "num_dependents": "NumberOfDependents",
}

# REVIEW band upper bound (mirrors the Streamlit app: <0.20 approve,
# 0.20-0.40 review, >=0.40 decline). The lower bound is the model threshold.
_REVIEW_UPPER = 0.40

_ARTIFACTS = ("cluster_scaler.joblib", "kmeans.joblib", "xgb_model.joblib", "metadata.joblib")


def _short_hash(path: Path) -> str:
    """Return the first 12 hex chars of a file's SHA-256 digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def decide(probability: float, threshold: float) -> str:
    """Map a default probability to an APPROVE / REVIEW / DECLINE decision."""
    if probability < threshold:
        return "APPROVE"
    if probability < _REVIEW_UPPER:
        return "REVIEW"
    return "DECLINE"


class ModelService:
    """Loads the scoring pipeline once and scores individual borrowers."""

    def __init__(self, models_dir: Path, threshold_override: float | None = None) -> None:
        self.models_dir = Path(models_dir)
        self._threshold_override = threshold_override
        self._loaded = False
        self.cluster_scaler = None
        self.kmeans = None
        self.model = None
        self.meta: Dict[str, Any] = {}
        self.explainer = None
        self.features: List[str] = []
        self.cluster_features: List[str] = []
        self.threshold: float = 0.0
        self.model_version: str = "unloaded"
        self.training_date: str = ""

    # --- lifecycle ---------------------------------------------------------

    def artifacts_present(self) -> bool:
        """True when every required artifact file exists on disk."""
        return all((self.models_dir / name).exists() for name in _ARTIFACTS)

    def load(self) -> None:
        """Load all artifacts and build the SHAP explainer (call once at startup)."""
        if not self.artifacts_present():
            missing = [n for n in _ARTIFACTS if not (self.models_dir / n).exists()]
            raise FileNotFoundError(
                f"Missing model artifacts in {self.models_dir}: {missing}. "
                "Run `python train_model.py` to generate them."
            )
        self.cluster_scaler = joblib.load(self.models_dir / "cluster_scaler.joblib")
        self.kmeans = joblib.load(self.models_dir / "kmeans.joblib")
        self.model = joblib.load(self.models_dir / "xgb_model.joblib")
        self.meta = joblib.load(self.models_dir / "metadata.joblib")
        self.explainer = shap.TreeExplainer(self.model)

        self.features = list(self.meta["features"])
        self.cluster_features = list(self.meta["cluster_features"])
        self.threshold = (
            self._threshold_override
            if self._threshold_override is not None
            else float(self.meta["threshold"])
        )

        model_path = self.models_dir / "xgb_model.joblib"
        self.model_version = f"xgb-{_short_hash(model_path)}"
        self.training_date = datetime.fromtimestamp(
            os.path.getmtime(model_path), tz=timezone.utc
        ).isoformat()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # --- scoring -----------------------------------------------------------

    def _to_feature_frame(self, borrower: Dict[str, Any]) -> pd.DataFrame:
        """Build the model-ready single-row frame, deriving the cluster feature."""
        # K-Means segment from the scaled cluster features.
        cluster_row = pd.DataFrame(
            [[
                borrower["revolving_utilization"],
                borrower["age"],
                borrower["monthly_income"],
                borrower["debt_ratio"],
                borrower["num_90_days_late"],
            ]],
            columns=self.cluster_features,
        )
        cluster = int(self.kmeans.predict(self.cluster_scaler.transform(cluster_row))[0])

        values = {INPUT_TO_FEATURE[k]: v for k, v in borrower.items()}
        values["cluster"] = cluster
        return pd.DataFrame([[values[f] for f in self.features]], columns=self.features)

    def predict(self, borrower: Dict[str, Any], top_n: int = 3) -> Dict[str, Any]:
        """Score one borrower and return PD, decision, and top SHAP drivers."""
        if not self._loaded:
            raise RuntimeError("ModelService.load() must be called before predict().")

        frame = self._to_feature_frame(borrower)
        probability = float(self.model.predict_proba(frame)[0][1])
        decision = decide(probability, self.threshold)

        shap_row = self.explainer(frame)
        contributions = sorted(
            zip(self.features, shap_row.values[0]), key=lambda x: -x[1]
        )
        top = [
            {
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "shap_value": float(val),
                "direction": "toward default" if val >= 0 else "away from default",
            }
            for feat, val in contributions[:top_n]
        ]
        return {
            "probability_of_default": probability,
            "decision": decision,
            "threshold": self.threshold,
            "top_risk_factors": top,
            "model_version": self.model_version,
        }

    def model_info(self) -> Dict[str, Any]:
        """Return governance metadata for the loaded model."""
        return {
            "model_version": self.model_version,
            "training_date": self.training_date,
            "holdout_auc": float(self.meta["test_auc"]),
            "threshold": self.threshold,
            "n_features": len(self.features),
            "features": self.features,
        }
