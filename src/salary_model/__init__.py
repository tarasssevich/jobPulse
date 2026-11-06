"""JobPulse Econometric & Salary Modeling Module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.salary_model.econometrics import (
    PREMIUM_COLUMNS,
    calculate_skill_premiums,
    get_canonical_skill_name,
)
from src.salary_model.models import (
    ModelMetrics,
    SalaryModelingPipeline,
)

if TYPE_CHECKING:
    from src.salary_model.runner import run_salary_analysis


def __getattr__(name: str) -> Any:
    """Lazy import for runner module to avoid runpy duplicate import warnings."""
    if name == "run_salary_analysis":
        from src.salary_model.runner import run_salary_analysis

        return run_salary_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PREMIUM_COLUMNS",
    "ModelMetrics",
    "SalaryModelingPipeline",
    "calculate_skill_premiums",
    "get_canonical_skill_name",
    "run_salary_analysis",
]
