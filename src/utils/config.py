"""Application configuration and environment loader."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable project settings."""

    # HeadHunter API
    hh_client_id: str = os.getenv("HH_CLIENT_ID", "")
    hh_client_secret: str = os.getenv("HH_CLIENT_SECRET", "")
    hh_redirect_uri: str = os.getenv("HH_REDIRECT_URI", "http://localhost:8000/callback")
    hh_access_token: str = os.getenv("HH_ACCESS_TOKEN", "")
    hh_user_agent: str = os.getenv(
        "HH_USER_AGENT",
        "JobPulse/1.0 (academic-research-cis; https://github.com/tarasssevich/jobPulse)",
    )

    # Storage paths
    sqlite_db_path: Path = BASE_DIR / os.getenv("SQLITE_DB_PATH", "data/staging.db")
    raw_data_dir: Path = BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")

    # Rate limiting & timeouts
    request_delay_seconds: float = float(os.getenv("REQUEST_DELAY_SECONDS", "0.35"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10.0"))

    # Log level
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
