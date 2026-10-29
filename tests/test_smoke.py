"""Smoke test to verify test harness and package import."""

import src


def test_package_metadata():
    """Verify package version is defined."""
    assert src.__version__ == "0.1.0"
