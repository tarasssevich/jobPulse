"""Publication-Grade Scientific Visualizer for Skill Demand Forecasting.

Generates multi-panel figures (300 DPI, academic typography, high contrast)
displaying historical demand points, smoothed trend lines, Holt forecasts,
and 95% confidence intervals.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib

# Headless backend to avoid GUI window popups during CI/CD and automated pipelines
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.demand_forecasting.forecaster import (
    TREND_DECLINING,
    TREND_HIGH_GROWTH,
    SkillDemandForecaster,
    _resolve_skill_column,
)
from src.demand_forecasting.timeseries import build_daily_skill_demand_series
from src.salary_model.econometrics import get_canonical_skill_name

logger = logging.getLogger(__name__)

# Academic color constants
NAVY_PRIMARY: Final[str] = "#1f4e79"
COLOR_GROWTH: Final[str] = "#2e7d32"
COLOR_DECLINE: Final[str] = "#c62828"
COLOR_STABLE: Final[str] = "#ef6c00"
COLOR_SCATTER: Final[str] = "#7f7f7f"
GRID_COLOR: Final[str] = "#d0d0d0"
TEXT_DARK: Final[str] = "#212529"

DEFAULT_PANELS: Final[tuple[tuple[str, str], ...]] = (
    ("skill_python", "Python (Core Backend / Data)"),
    ("skill_docker", "Docker (Cloud / DevOps)"),
    ("skill_postgresql", "PostgreSQL (Databases)"),
    ("skill_kafka", "Kafka (High Growth / Backend)"),
)


def _apply_academic_theme() -> None:
    """Configure publication-ready typography and axes properties."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Liberation Sans",
                "Helvetica",
                "sans-serif",
            ],
            "axes.edgecolor": "#595959",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "text.color": TEXT_DARK,
            "axes.labelcolor": TEXT_DARK,
            "xtick.color": TEXT_DARK,
            "ytick.color": TEXT_DARK,
            "figure.autolayout": False,
        }
    )


class DemandVisualizer:
    """Publication-grade visualizer for time-series skill demand forecasting.

    Attributes:
        output_dir: Directory where generated figure artifacts are stored.
        dpi: Dots per inch for raster image generation (default: 300).
    """

    def __init__(
        self,
        output_dir: Path | str = "reports/figures",
        dpi: int = 300,
    ) -> None:
        """Initialize the visualizer and ensure output directories exist."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        _apply_academic_theme()

    def plot_skill_demand_forecast(
        self,
        df: pd.DataFrame,
        skills: Sequence[str | tuple[str, str]] | None = None,
        horizon_days: int = 14,
        filename: str = "fig6_skill_demand_forecast.png",
    ) -> Path:
        """Generate 4-panel (2x2) academic publication chart of skill demand forecasts.

        Args:
            df: Vacancies DataFrame with `published_at` and skill binary columns.
            skills: Exactly 4 skill identifiers or (col, title) tuples (defaults to DEFAULT_PANELS).
            horizon_days: Forecast horizon in calendar days.
            filename: Target output image filename.

        Returns:
            Path to the persisted PNG figure.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If df is empty, lacks required columns, or skills count != 4.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
        if type(horizon_days) is not int or horizon_days < 1:
            raise ValueError("horizon_days must be an integer >= 1.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if "published_at" not in df.columns:
            raise ValueError("Required column 'published_at' is missing.")

        panels_config: list[tuple[str, str]] = []
        if skills is None:
            panels_config = list(DEFAULT_PANELS)
        else:
            if len(skills) != 4:
                raise ValueError(f"DemandVisualizer requires exactly 4 skills, got {len(skills)}.")
            for item in skills:
                if isinstance(item, tuple) and len(item) == 2:
                    panels_config.append(item)
                elif isinstance(item, str):
                    canonical = get_canonical_skill_name(item)
                    panels_config.append((item, canonical))
                else:
                    raise TypeError("Each skill entry must be a string or a (skill, title) tuple.")

        available_cols = set(df.columns)
        forecaster = SkillDemandForecaster()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        axes_flat = axes.flatten()

        try:
            for ax, (raw_col, title) in zip(axes_flat, panels_config):
                col = _resolve_skill_column(raw_col, available_cols)
                if col is None:
                    raise ValueError(f"Skill column '{raw_col}' not found in DataFrame.")

                daily_df = build_daily_skill_demand_series(df, col)
                if len(daily_df) < 2:
                    raise ValueError(f"Insufficient historical data points for column '{col}'.")

                series = daily_df["demand_share_smoothed"]
                res = forecaster.fit_predict(series, horizon_days=horizon_days)

                forecast = res["forecast"]
                ci_lower = res["ci_lower"]
                ci_upper = res["ci_upper"]
                growth = res["growth_rate_pct"]
                trend_cat = res["trend_category"]

                # Accent color based on trend category
                if trend_cat == TREND_HIGH_GROWTH:
                    accent_color = COLOR_GROWTH
                elif trend_cat == TREND_DECLINING:
                    accent_color = COLOR_DECLINE
                else:
                    accent_color = COLOR_STABLE

                # 1. Historical actual daily demand points (subtle scatter dots)
                ax.scatter(
                    daily_df.index,
                    daily_df["demand_share"],
                    color=COLOR_SCATTER,
                    alpha=0.45,
                    s=18,
                    label="Фактический спрос (день)",
                    zorder=2,
                )

                # 2. Historical smoothed trend line
                ax.plot(
                    series.index,
                    series.values,
                    color=NAVY_PRIMARY,
                    linewidth=2.2,
                    label="Тренд (7-дн. сглаж.)",
                    zorder=3,
                )

                # Bridge to eliminate gap between historical series and forecast
                bridge_x = [series.index[-1], forecast.index[0]]
                bridge_y = [series.iloc[-1], forecast.iloc[0]]
                ax.plot(
                    bridge_x,
                    bridge_y,
                    color=accent_color,
                    linestyle="--",
                    linewidth=2.0,
                    zorder=3,
                )

                # 3. Forecast line
                ax.plot(
                    forecast.index,
                    forecast.values,
                    color=accent_color,
                    linestyle="--",
                    linewidth=2.0,
                    label=f"Прогноз ({horizon_days} дн.)",
                    zorder=3,
                )

                # 4. 95% Confidence Interval shaded band (alpha=0.25)
                ax.fill_between(
                    forecast.index,
                    ci_lower.values,
                    ci_upper.values,
                    color=accent_color,
                    alpha=0.25,
                    label="95% дов. интервал",
                    zorder=1,
                )

                # 5. Annotated text box with projected growth rate and trend category
                sign = "+" if growth > 0 else ""
                annotation_text = rf"$\Delta = {sign}{growth:.1f}\%${chr(10)}{trend_cat}"
                ax.text(
                    0.04,
                    0.92,
                    annotation_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round,pad=0.45",
                        "facecolor": "#f8f9fa",
                        "edgecolor": "#ced4da",
                        "alpha": 0.92,
                    },
                    fontsize=9.5,
                    fontweight="bold",
                    color=TEXT_DARK,
                    zorder=4,
                )

                ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY_PRIMARY, pad=10)
                ax.set_xlabel("Дата", fontsize=10)
                ax.set_ylabel("Доля в вакансиях, %", fontsize=10)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
                ax.set_ylim(bottom=0.0)

            plt.tight_layout()
            output_path = self.output_dir / filename
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Successfully generated publication forecast figure at %s", output_path)
            return output_path

        finally:
            plt.close(fig)
