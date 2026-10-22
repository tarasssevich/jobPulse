"""Test configuration and settings loader."""

from pathlib import Path

from src.utils.config import Settings


def test_settings_default_paths():
    """Verify default paths are set and point inside repository."""
    s = Settings()
    assert isinstance(s.raw_data_dir, Path)
    assert isinstance(s.sqlite_db_path, Path)
    assert s.request_delay_seconds > 0
    assert "JobPulse" in s.hh_user_agent
