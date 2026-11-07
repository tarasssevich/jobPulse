"""Orchestration Runner for JobPulse Step 4: Time-Series Skill Demand Forecasting.

Loads enriched vacancies, executes Holt linear trend forecasting for top IT skills,
persists summary CSV and JSON metrics, and renders publication-grade Fig 6.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.demand_forecasting.forecaster import forecast_top_skills
from src.demand_forecasting.visualizer import DEFAULT_PANELS, DemandVisualizer
from src.salary_model.econometrics import get_canonical_skill_name

logger = logging.getLogger("demand_forecasting.runner")


def run_demand_forecasting(
    input_path: Path | str = "data/processed/vacancies_enriched.parquet",
    output_dir: Path | str = "data/processed",
    figures_dir: Path | str = "reports/figures",
    top_n: int = 15,
    horizon_days: int = 14,
) -> dict[str, Any]:
    """Execute end-to-end skill demand time-series modeling and forecasting pipeline.

    Args:
        input_path: Path to `vacancies_enriched.parquet`.
        output_dir: Directory where summary CSV and JSON metrics will be saved.
        figures_dir: Directory where publication figure will be rendered.
        top_n: Number of leading market technologies to forecast (default: 15).
        horizon_days: Future horizon in days for forecasting (default: 14).

    Returns:
        Structured dictionary with paths, summary DataFrame, and aggregated metrics.
    """
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n must be an integer >= 1.")
    if type(horizon_days) is not int or horizon_days < 1:
        raise ValueError("horizon_days must be an integer >= 1.")

    in_file = Path(input_path)
    out_dir = Path(output_dir)
    fig_dir = Path(figures_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found at: {in_file}")

    logger.info("Loading enriched vacancy dataset from %s ...", in_file)
    df = pd.read_parquet(in_file)
    logger.info("Loaded %d vacancy records.", len(df))

    if df.empty:
        raise ValueError(f"Input vacancy dataset at {in_file} is empty.")
    if "published_at" not in df.columns:
        raise ValueError("Required column 'published_at' is missing from vacancies dataset.")

    # Identify all binary skill indicators
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    if not skill_cols:
        raise ValueError("No 'skill_*' columns found in vacancies dataset.")

    # Select top N skills by overall vacancy frequency
    top_skills_series = df[skill_cols].sum().sort_values(ascending=False).head(top_n)
    top_skills = top_skills_series.index.tolist()
    logger.info("Selected Top %d skills for forecasting: %s", len(top_skills), top_skills)

    # Calculate date span
    parsed_dates = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dropna()
    min_date_str = parsed_dates.min().strftime("%Y-%m-%d") if not parsed_dates.empty else "N/A"
    max_date_str = parsed_dates.max().strftime("%Y-%m-%d") if not parsed_dates.empty else "N/A"
    total_days_span = (parsed_dates.max() - parsed_dates.min()).days + 1 if not parsed_dates.empty else 0

    if total_days_span < 2:
        raise ValueError(
            f"At least 2 distinct calendar days required for time-series forecasting, found {total_days_span}."
        )

    # Execute forecasting across selected top skills
    logger.info("Executing Holt time-series forecasting (horizon=%d days)...", horizon_days)
    summary_df = forecast_top_skills(df, top_skills, horizon_days=horizon_days)

    # Persist summary CSV
    csv_path = out_dir / "demand_forecast_summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Saved forecast summary CSV to %s", csv_path)

    # Compute high-level aggregated metrics
    high_growth_count = int((summary_df["trend_category"] == "High Growth / Rising").sum())
    stable_count = int((summary_df["trend_category"] == "Stable / Evergreen").sum())
    declining_count = int((summary_df["trend_category"] == "Declining").sum())
    mean_mae = float(np.round(summary_df["mae"].mean(), 4)) if not summary_df.empty else 0.0
    mean_mape = float(np.round(summary_df["mape"].mean(), 4)) if not summary_df.empty else 0.0

    metrics_payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {
            "input_file": str(in_file),
            "total_vacancies": len(df),
            "date_start": min_date_str,
            "date_end": max_date_str,
            "total_days_span": total_days_span,
            "skills_analyzed_count": len(summary_df),
        },
        "forecasting_parameters": {
            "model": "Holt Linear Trend (statsmodels.tsa.api.Holt)",
            "initialization_method": "estimated",
            "smoothing_window_days": 7,
            "horizon_days": horizon_days,
            "confidence_level": 0.95,
        },
        "summary": {
            "high_growth_count": high_growth_count,
            "stable_count": stable_count,
            "declining_count": declining_count,
            "mean_mae": mean_mae,
            "mean_mape": mean_mape,
        },
        "forecasts": summary_df.to_dict(orient="records"),
    }

    # Persist JSON metrics
    json_path = out_dir / "demand_forecast_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved forecast metrics JSON to %s", json_path)

    # Generate publication-grade 4-panel visualizer figure
    logger.info("Rendering publication figure (fig6_skill_demand_forecast.png)...")
    visualizer = DemandVisualizer(output_dir=fig_dir, dpi=300)

    # Determine 4 panels: prefer default publication panels if present in df,
    # otherwise pick top 4 forecasted skills.
    default_panel_cols = [p[0] for p in DEFAULT_PANELS]
    if all(c in df.columns for c in default_panel_cols):
        viz_panels: list[tuple[str, str]] | None = None
    elif len(top_skills) >= 4:
        viz_panels = [
            (col, get_canonical_skill_name(col))
            for col in top_skills[:4]
        ]
    else:
        raise ValueError(
            f"At least 4 skill columns are required to generate the 4-panel figure, found {len(top_skills)}."
        )

    fig_path = visualizer.plot_skill_demand_forecast(
        df=df,
        skills=viz_panels,
        horizon_days=horizon_days,
        filename="fig6_skill_demand_forecast.png",
    )
    logger.info("Saved publication figure to %s", fig_path)

    # Print publication-formatted summary table
    print("\n" + "=" * 98)
    print(f" JOBPULSE STEP 4: TIME-SERIES SKILL DEMAND FORECAST ({horizon_days} DAYS HORIZON)".center(98))
    print("=" * 98)
    header = f"{'Skill Name':<16} | {'Current %':>10} | {'Forecast %':>10} | {'Delta %':>10} | {'Trend Category':<22} | {'MAE':>7} | {'MAPE %':>8}"
    print(header)
    print("-" * 98)
    for _, row in summary_df.iterrows():
        print(
            f"{row['skill_name']:<16} | {row['current_share_pct']:>9.2f}% | "
            f"{row['forecast_share_pct']:>9.2f}% | {row['delta_pct']:>+9.2f}% | "
            f"{row['trend_category']:<22} | {row['mae']:>7.3f} | {row['mape']:>7.2f}%"
        )
    print("=" * 98)
    print(
        f"Summary: High Growth: {high_growth_count} | Stable: {stable_count} | "
        f"Declining: {declining_count} | Mean MAE: {mean_mae:.3f} | Mean MAPE: {mean_mape:.2f}%\n"
    )

    return {
        "summary_df": summary_df,
        "csv_path": csv_path,
        "json_path": json_path,
        "figure_path": fig_path,
        "metrics": metrics_payload,
    }


def main() -> None:
    """CLI Entrypoint for demand forecasting runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="JobPulse Step 4: Time-Series Skill Demand Modeling & Forecasting"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/vacancies_enriched.parquet",
        help="Path to input enriched vacancies parquet dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save forecast summary CSV and JSON metrics",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="reports/figures",
        help="Directory to save publication figures",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of leading technologies to forecast",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Forecast horizon in days",
    )
    args = parser.parse_args()

    try:
        run_demand_forecasting(
            input_path=args.input,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            top_n=args.top_n,
            horizon_days=args.horizon,
        )
    except Exception:
        logger.exception("Demand forecasting pipeline execution failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
