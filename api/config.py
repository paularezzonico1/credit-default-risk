"""Application configuration, driven entirely by environment variables.

Uses pydantic v1's built-in ``BaseSettings`` (the project is pinned to
pydantic 1.10). Every operational knob — where the model artifacts live, the
decision threshold, the database URL, the API host/port — is read from the
environment so nothing is hardcoded and the same image can run in any
environment by changing config alone.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field

# Repo root = two levels up from this file (api/config.py -> api -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment (prefix ``CDR_``)."""

    # Where the serialized model artifacts live.
    models_dir: Path = Field(default=_REPO_ROOT / "models")

    # Optional override of the decision threshold. When None, the threshold
    # baked into the model metadata is used (the documented 0.20).
    threshold_override: Optional[float] = Field(default=None)

    # API server binding.
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Database connection (Postgres in compose/CI). Used from Phase B onward.
    # Example: postgresql+psycopg2://user:pass@db:5432/credit_risk
    database_url: Optional[str] = Field(default=None)

    class Config:
        env_prefix = "CDR_"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read the environment once)."""
    return Settings()
