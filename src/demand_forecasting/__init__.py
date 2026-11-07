"""JobPulse Step 4: Time-Series Skill Demand Modeling & Forecasting.

Public API exports for daily demand construction, Holt linear trend forecasting,
and academic publication visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.demand_forecasting.forecaster import (
    TREND_DECLINING,
    TREND_HIGH_GROWTH,
    TREND_STABLE,
    SkillDemandForecaster,
    categorize_trend,
    forecast_top_skills,
)
from src.demand_forecasting.timeseries import (
    SERIES_COLUMNS,
    build_daily_skill_demand_series,
)
from src.demand_forecasting.visualizer import DemandVisualizer

if TYPE_CHECKING:
    from src.demand_forecasting.runner import run_demand_forecasting


def __getattr__(name: str) -> Any:
    """Lazy import for runner module to avoid runpy duplicate import warnings."""
    if name == "run_demand_forecasting":
        from src.demand_forecasting.runner import run_demand_forecasting

        return run_demand_forecasting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SERIES_COLUMNS",
    "TREND_DECLINING",
    "TREND_HIGH_GROWTH",
    "TREND_STABLE",
    "DemandVisualizer",
    "SkillDemandForecaster",
    "build_daily_skill_demand_series",
    "categorize_trend",
    "forecast_top_skills",
    "run_demand_forecasting",
]
