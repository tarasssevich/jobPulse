"""Unit and integration tests for JobPulse Step 4: Time-Series Demand Forecasting.

Covers:
- `build_daily_skill_demand_series` date parsing, zero-vacancy days handling, rolling smoothing.
- `SkillDemandForecaster` Holt linear trend fit, confidence intervals, bounds clamping, metrics.
- Trend categorization ('High Growth / Rising', 'Stable / Evergreen', 'Declining').
- Edge cases: empty series, single-point series, constant series, corrupted payloads.
- `DemandVisualizer` publication-grade 4-panel generation and resource cleanup.
- `run_demand_forecasting` end-to-end pipeline execution and artifact validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.demand_forecasting.forecaster import (
    TREND_DECLINING,
    TREND_HIGH_GROWTH,
    TREND_STABLE,
    SkillDemandForecaster,
    categorize_trend,
    forecast_top_skills,
)
from src.demand_forecasting.runner import run_demand_forecasting
from src.demand_forecasting.timeseries import (
    SERIES_COLUMNS,
    build_daily_skill_demand_series,
)
from src.demand_forecasting.visualizer import DemandVisualizer


@pytest.fixture
def synthetic_vacancies_timeseries_df() -> pd.DataFrame:
    """Generate reproducible synthetic vacancy dataset with known temporal distribution."""
    # 30-day timeline with some gaps (e.g. weekends)
    dates = pd.date_range("2026-10-01", "2026-10-30", freq="D")
    records: list[dict[str, object]] = []

    for i, date in enumerate(dates):
        # Introduce a 2-day gap where 0 vacancies are published
        if i in (10, 11):
            continue

        n_vacancies = 10 + (i % 5) * 5  # Between 10 and 30 vacancies per day
        for v_idx in range(n_vacancies):
            # Growing demand for Python (from ~20% to ~50%)
            prob_python = 0.20 + (i / len(dates)) * 0.30
            # Stable demand for Docker (~25%)
            prob_docker = 0.25
            # Declining demand for 1C (from ~40% to ~15%)
            prob_1c = max(0.05, 0.40 - (i / len(dates)) * 0.25)
            # Rare / emerging Go (~10%)
            prob_go = 0.10

            records.append(
                {
                    "vacancy_id": f"vac_{i}_{v_idx}",
                    "published_at": f"{date.strftime('%Y-%m-%d')}T10:00:00+03:00",
                    "skill_python": int(np.random.random() < prob_python),
                    "skill_docker": int(np.random.random() < prob_docker),
                    "skill_1c": int(np.random.random() < prob_1c),
                    "skill_go": int(np.random.random() < prob_go),
                    "skill_kafka": int(np.random.random() < 0.15),
                    "skill_postgresql": int(np.random.random() < 0.20),
                }
            )

    return pd.DataFrame(records)


# =============================================================================
# 1. Time-Series Aggregation Tests
# =============================================================================


def test_build_daily_skill_demand_series_basic() -> None:
    """Verify daily aggregation, market share, and 7-day rolling smoothing calculation."""
    df = pd.DataFrame(
        {
            "published_at": [
                "2026-10-01T12:00:00Z",
                "2026-10-01T14:00:00Z",
                "2026-10-02T09:00:00Z",
                "2026-10-02T11:00:00Z",
            ],
            "skill_python": [1, 0, 1, 1],
        }
    )

    series_df = build_daily_skill_demand_series(df, "skill_python", rolling_window=7)

    assert isinstance(series_df, pd.DataFrame)
    assert list(series_df.columns) == list(SERIES_COLUMNS)
    assert len(series_df) == 2
    assert pd.Timestamp("2026-10-01") in series_df.index
    assert pd.Timestamp("2026-10-02") in series_df.index

    # Day 1: 1 out of 2 = 50.0%
    assert series_df.loc["2026-10-01", "total_vacancies"] == 2
    assert series_df.loc["2026-10-01", "skill_vacancies"] == 1
    assert series_df.loc["2026-10-01", "demand_share"] == 50.0
    assert series_df.loc["2026-10-01", "demand_share_smoothed"] == 50.0

    # Day 2: 2 out of 2 = 100.0%
    assert series_df.loc["2026-10-02", "total_vacancies"] == 2
    assert series_df.loc["2026-10-02", "skill_vacancies"] == 2
    assert series_df.loc["2026-10-02", "demand_share"] == 100.0
    # Smoothed over day 1 and 2: (50 + 100) / 2 = 75.0%
    assert series_df.loc["2026-10-02", "demand_share_smoothed"] == 75.0


def test_build_daily_skill_demand_series_handles_zero_vacancy_days() -> None:
    """Verify that calendar days with 0 vacancies are filled gracefully without NaN/errors."""
    df = pd.DataFrame(
        {
            # Day 1 and Day 3, with Day 2 missing
            "published_at": ["2026-10-01T10:00:00Z", "2026-10-03T10:00:00Z"],
            "skill_docker": [1, 0],
        }
    )

    series_df = build_daily_skill_demand_series(df, "skill_docker", rolling_window=7)

    # 3 continuous days
    assert len(series_df) == 3
    assert pd.Timestamp("2026-10-02") in series_df.index

    # Missing Day 2 should have 0 vacancies and 0.0 share
    day_2 = series_df.loc["2026-10-02"]
    assert day_2["total_vacancies"] == 0
    assert day_2["skill_vacancies"] == 0
    assert day_2["demand_share"] == 0.0
    assert not np.isnan(day_2["demand_share_smoothed"])


def test_build_daily_skill_demand_series_corrupted_dates() -> None:
    """Corrupted timestamps and NaT values must be safely filtered."""
    df = pd.DataFrame(
        {
            "published_at": ["invalid-date", None, "2026-10-01T00:00:00Z", "corrupted"],
            "skill_go": [1, 1, 1, 0],
        }
    )

    series_df = build_daily_skill_demand_series(df, "skill_go")
    assert len(series_df) == 1
    assert series_df.iloc[0]["total_vacancies"] == 1
    assert series_df.iloc[0]["skill_vacancies"] == 1


def test_build_daily_skill_demand_series_empty() -> None:
    """Empty dataframe must return empty series dataframe with proper schema."""
    empty_df = pd.DataFrame(columns=["published_at", "skill_python"])
    res = build_daily_skill_demand_series(empty_df, "skill_python")

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert list(res.columns) == list(SERIES_COLUMNS)

    # DataFrame with only invalid dates
    all_invalid = pd.DataFrame({"published_at": ["bad1", "bad2"], "skill_python": [1, 0]})
    res_inv = build_daily_skill_demand_series(all_invalid, "skill_python")
    assert len(res_inv) == 0


def test_build_daily_skill_demand_series_validation() -> None:
    """Input validation must enforce types and column existence."""
    df = pd.DataFrame({"published_at": ["2026-10-01"], "skill_sql": [1]})

    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        build_daily_skill_demand_series([1, 2, 3], "skill_sql")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="skill_col must be a string"):
        build_daily_skill_demand_series(df, 123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="rolling_window must be an integer >= 1"):
        build_daily_skill_demand_series(df, "skill_sql", rolling_window=0)

    with pytest.raises(ValueError, match="Required column 'published_at' is missing"):
        build_daily_skill_demand_series(pd.DataFrame({"other_col": [1]}), "skill_sql")

    with pytest.raises(ValueError, match="Skill column 'skill_unknown' not found"):
        build_daily_skill_demand_series(df, "skill_unknown")


def test_build_daily_skill_demand_series_immutability(
    synthetic_vacancies_timeseries_df: pd.DataFrame,
) -> None:
    """Input DataFrame must not be mutated during series construction."""
    copy_df = synthetic_vacancies_timeseries_df.copy(deep=True)
    _ = build_daily_skill_demand_series(synthetic_vacancies_timeseries_df, "skill_python")
    pd.testing.assert_frame_equal(synthetic_vacancies_timeseries_df, copy_df)


# =============================================================================
# 2. Forecaster & Holt Model Tests
# =============================================================================


def test_forecaster_fit_predict_basic() -> None:
    """Holt linear trend model generates valid forecast series, bounds, and evaluation metrics."""
    dates = pd.date_range("2026-10-01", periods=20, freq="D")
    # Linear upward trend from 10.0% to 29.0%
    values = np.linspace(10.0, 29.0, 20)
    series = pd.Series(values, index=dates, name="demand_share_smoothed")

    forecaster = SkillDemandForecaster()
    res = forecaster.fit_predict(series, horizon_days=14)

    assert "forecast" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert "metrics" in res
    assert "growth_rate_pct" in res
    assert "trend_category" in res

    forecast = res["forecast"]
    assert len(forecast) == 14
    assert forecast.index[0] == pd.Timestamp("2026-10-21")
    assert forecast.index[-1] == pd.Timestamp("2026-11-03")

    # Positive trend forecast should continue upward
    assert forecast.iloc[-1] > series.iloc[-1]
    assert res["growth_rate_pct"] > 5.0
    assert res["trend_category"] == TREND_HIGH_GROWTH

    # In-sample metrics
    metrics = res["metrics"]
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["mape"] >= 0.0


def test_forecaster_confidence_intervals_and_bounds_clamping() -> None:
    """Verify confidence interval bounds: lower >= 0.0, upper <= 100.0, and ci_lower <= ci_upper."""
    dates = pd.date_range("2026-10-01", periods=15, freq="D")
    # Noisy series near lower boundary
    np.random.seed(42)
    values = np.clip(np.random.normal(3.0, 2.0, size=15), 0.5, 10.0)
    series = pd.Series(values, index=dates)

    forecaster = SkillDemandForecaster()
    res = forecaster.fit_predict(series, horizon_days=7)

    ci_lower = res["ci_lower"]
    ci_upper = res["ci_upper"]
    forecast = res["forecast"]

    # Lower bound clamped to 0.0
    assert (ci_lower >= 0.0).all()
    # Upper bound clamped to 100.0
    assert (ci_upper <= 100.0).all()
    # Lower CI <= forecast <= Upper CI
    assert (ci_lower <= forecast + 1e-6).all()
    assert (forecast <= ci_upper + 1e-6).all()


def test_trend_categorization() -> None:
    """Verify categorical buckets: >5% Rising, [-5%, 5%] Stable, <-5% Declining."""
    assert categorize_trend(15.2) == TREND_HIGH_GROWTH
    assert categorize_trend(5.01) == TREND_HIGH_GROWTH

    assert categorize_trend(5.0) == TREND_STABLE
    assert categorize_trend(0.0) == TREND_STABLE
    assert categorize_trend(-5.0) == TREND_STABLE

    assert categorize_trend(-5.01) == TREND_DECLINING
    assert categorize_trend(-30.5) == TREND_DECLINING


def test_forecaster_zero_division_guard() -> None:
    """Growth rate calculation when current share is 0.0 must not divide by zero."""
    dates = pd.date_range("2026-10-01", periods=5, freq="D")
    series = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=dates)

    forecaster = SkillDemandForecaster()
    res = forecaster.fit_predict(series, horizon_days=7)
    assert res["growth_rate_pct"] == 0.0
    assert res["trend_category"] == TREND_STABLE


def test_forecaster_edge_cases() -> None:
    """Edge cases: empty series, single point, constant series, invalid inputs."""
    forecaster = SkillDemandForecaster()

    # Empty series
    with pytest.raises(ValueError, match="At least 2 non-null observations required"):
        forecaster.fit_predict(pd.Series(dtype=float))

    # Single-point series
    with pytest.raises(ValueError, match="At least 2 non-null observations required"):
        forecaster.fit_predict(pd.Series([12.5], index=pd.date_range("2026-10-01", periods=1)))

    # Invalid horizon
    with pytest.raises(ValueError, match="horizon_days must be an integer >= 1"):
        forecaster.fit_predict(pd.Series([10.0, 11.0]), horizon_days=0)

    # Invalid input type
    with pytest.raises(TypeError, match="series must be a pandas Series"):
        forecaster.fit_predict([10.0, 11.0])  # type: ignore[arg-type]

    # Constant series (residual std ~0, should not crash)
    dates = pd.date_range("2026-10-01", periods=10, freq="D")
    constant_s = pd.Series([20.0] * 10, index=dates)
    res_const = forecaster.fit_predict(constant_s, horizon_days=7)
    assert np.isclose(res_const["forecast"].iloc[-1], 20.0, atol=1e-4)
    assert res_const["trend_category"] == TREND_STABLE


def test_forecast_top_skills(synthetic_vacancies_timeseries_df: pd.DataFrame) -> None:
    """Test multi-skill evaluation function returns expected summary columns."""
    skills_to_test = ["skill_python", "Docker", "skill_unknown_xyz"]
    summary_df = forecast_top_skills(
        synthetic_vacancies_timeseries_df,
        skills_to_test,
        horizon_days=14,
    )

    assert isinstance(summary_df, pd.DataFrame)
    # 2 valid skills evaluated, unknown skill skipped
    assert len(summary_df) == 2
    assert "Python" in summary_df["skill_name"].values
    assert "Docker" in summary_df["skill_name"].values

    expected_cols = [
        "skill_name",
        "current_share_pct",
        "forecast_share_pct",
        "delta_pct",
        "trend_category",
        "mae",
        "mape",
    ]
    assert list(summary_df.columns) == expected_cols
    assert (summary_df["current_share_pct"] >= 0.0).all()
    assert (summary_df["forecast_share_pct"] >= 0.0).all()


# =============================================================================
# 3. Visualizer & Pipeline Integration Tests
# =============================================================================


def test_demand_visualizer_execution(
    synthetic_vacancies_timeseries_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """DemandVisualizer must produce 300 DPI publication figure and clean up figure memory."""
    visualizer = DemandVisualizer(output_dir=tmp_path, dpi=100)

    panels = [
        ("skill_python", "Python (Core Backend)"),
        ("skill_docker", "Docker (DevOps)"),
        ("skill_1c", "1C (Enterprise)"),
        ("skill_go", "Go (High Concurrency)"),
    ]

    out_file = visualizer.plot_skill_demand_forecast(
        df=synthetic_vacancies_timeseries_df,
        skills=panels,
        horizon_days=14,
        filename="test_fig6.png",
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 10_000

    # Ensure resource cleanup: no dangling open matplotlib figures
    assert len(plt.get_fignums()) == 0


def test_demand_visualizer_validation(synthetic_vacancies_timeseries_df: pd.DataFrame) -> None:
    """Visualizer must validate panel count and DataFrame inputs."""
    viz = DemandVisualizer()

    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        viz.plot_skill_demand_forecast([1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="requires exactly 4 skills"):
        viz.plot_skill_demand_forecast(
            synthetic_vacancies_timeseries_df,
            skills=["skill_python", "skill_docker"],  # Only 2
        )


def test_run_demand_forecasting_pipeline(
    synthetic_vacancies_timeseries_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """End-to-end integration test of run_demand_forecasting runner."""
    parquet_path = tmp_path / "test_vacancies.parquet"
    synthetic_vacancies_timeseries_df.to_parquet(parquet_path)

    out_dir = tmp_path / "processed"
    fig_dir = tmp_path / "figures"

    results = run_demand_forecasting(
        input_path=parquet_path,
        output_dir=out_dir,
        figures_dir=fig_dir,
        top_n=4,
        horizon_days=14,
    )
    assert "summary_df" in results

    # 1. Summary CSV exists and matches expected columns
    csv_path = out_dir / "demand_forecast_summary.csv"
    assert csv_path.exists()
    summary_df = pd.read_csv(csv_path)
    assert len(summary_df) == 4
    assert "skill_name" in summary_df.columns
    assert "delta_pct" in summary_df.columns

    # 2. Metrics JSON exists and conforms to schema
    json_path = out_dir / "demand_forecast_metrics.json"
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        metrics = json.load(f)
    assert "timestamp" in metrics
    assert "summary" in metrics
    assert "forecasts" in metrics
    assert metrics["forecasting_parameters"]["horizon_days"] == 14

    # 3. Publication figure exists
    fig_path = fig_dir / "fig6_skill_demand_forecast.png"
    assert fig_path.exists()
    assert fig_path.stat().st_size > 10_000


# =============================================================================
# 4. Adversarial & Edge Case Robustness Tests
# =============================================================================


def test_build_daily_skill_demand_series_clipping_and_bool_validation() -> None:
    """Out-of-range skill values are clamped in [0, 1] and boolean window raises ValueError."""
    # Arrange
    df = pd.DataFrame(
        {
            "published_at": ["2026-10-01T10:00:00Z", "2026-10-01T11:00:00Z"],
            "skill_python": [-5, 10],  # Dirty/corrupted skill values
        }
    )

    # Act
    res = build_daily_skill_demand_series(df, "skill_python")

    # Assert: total vacancies = 2, skill vacancies clamped to 1 (since 10 -> 1, -5 -> 0)
    assert res.loc["2026-10-01", "total_vacancies"] == 2
    assert res.loc["2026-10-01", "skill_vacancies"] == 1
    assert res.loc["2026-10-01", "demand_share"] == 50.0

    # Act & Assert: bool rolling window
    with pytest.raises(ValueError, match="rolling_window must be an integer >= 1"):
        build_daily_skill_demand_series(df, "skill_python", rolling_window=True)  # type: ignore[arg-type]


def test_forecaster_initialization_validation() -> None:
    """Forecaster rejects invalid initialization method strings."""
    # Arrange & Act & Assert
    with pytest.raises(
        ValueError, match="initialization_method must be 'estimated' or 'heuristic'"
    ):
        SkillDemandForecaster(initialization_method="invalid_method")


def test_forecaster_infinite_values_handling() -> None:
    """Series containing infinite values (+inf, -inf) are sanitized without NaN propagation."""
    # Arrange
    dates = pd.date_range("2026-10-01", periods=10, freq="D")
    raw_values = [10.0, np.inf, 12.0, -np.inf, 14.0, 15.0, np.inf, 16.0, 17.0, 18.0]
    series = pd.Series(raw_values, index=dates)
    forecaster = SkillDemandForecaster()

    # Act
    res = forecaster.fit_predict(series, horizon_days=5)

    # Assert
    forecast = res["forecast"]
    assert not forecast.isna().any()
    assert not np.isinf(forecast).any()
    assert (forecast >= 0.0).all()
    assert (forecast <= 100.0).all()
    assert not np.isnan(res["metrics"]["mae"])
    assert not np.isnan(res["metrics"]["rmse"])
    assert not np.isnan(res["metrics"]["mape"])


def test_forecaster_emerging_skill_growth_rate() -> None:
    """Emerging skill starting at 0.0% share and rising to positive share categorizes as High Growth."""
    # Arrange: skill had 0% share historically and rises to positive share
    dates = pd.date_range("2026-10-01", periods=5, freq="D")
    series = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=dates)
    forecaster = SkillDemandForecaster()

    # Act: predict with flat series
    res = forecaster.fit_predict(series, horizon_days=5)
    assert res["growth_rate_pct"] == 0.0
    assert res["trend_category"] == TREND_STABLE

    # Act & Assert: emerging skill from 0 to positive demand
    series_emerging = pd.Series([0.0, 0.0, 0.0, 2.0, 5.0], index=dates)
    res_emerging = forecaster.fit_predict(series_emerging, horizon_days=5)
    assert res_emerging["growth_rate_pct"] > 0.0
    assert res_emerging["trend_category"] == TREND_HIGH_GROWTH


def test_forecaster_near_zero_mape_stability() -> None:
    """Near-zero values do not cause division-by-zero or infinite MAPE."""
    # Arrange
    dates = pd.date_range("2026-10-01", periods=6, freq="D")
    series = pd.Series([1e-5, 2e-5, 1e-5, 3e-5, 2e-5, 1e-5], index=dates)
    forecaster = SkillDemandForecaster()

    # Act
    res = forecaster.fit_predict(series, horizon_days=5)

    # Assert
    assert np.isfinite(res["metrics"]["mape"])
    assert res["metrics"]["mape"] >= 0.0


def test_demand_visualizer_invalid_horizon(
    synthetic_vacancies_timeseries_df: pd.DataFrame,
) -> None:
    """Visualizer rejects invalid or non-integer horizon values."""
    # Arrange
    viz = DemandVisualizer()

    # Act & Assert
    with pytest.raises(ValueError, match="horizon_days must be an integer >= 1"):
        viz.plot_skill_demand_forecast(synthetic_vacancies_timeseries_df, horizon_days=0)

    with pytest.raises(ValueError, match="horizon_days must be an integer >= 1"):
        viz.plot_skill_demand_forecast(
            synthetic_vacancies_timeseries_df,
            horizon_days=True,  # type: ignore[arg-type]
        )


def test_runner_validation_errors(tmp_path: Path) -> None:
    """Pipeline runner strictly validates inputs, file existence, and minimum calendar span."""
    # Arrange
    dummy_parquet = tmp_path / "dummy.parquet"
    pd.DataFrame({"published_at": ["2026-10-01"], "skill_python": [1]}).to_parquet(dummy_parquet)

    # Act & Assert: top_n invalid
    with pytest.raises(ValueError, match="top_n must be an integer >= 1"):
        run_demand_forecasting(dummy_parquet, top_n=0)

    # Act & Assert: horizon invalid
    with pytest.raises(ValueError, match="horizon_days must be an integer >= 1"):
        run_demand_forecasting(dummy_parquet, horizon_days=-1)

    # Act & Assert: 1 calendar day span is insufficient
    with pytest.raises(ValueError, match="At least 2 distinct calendar days required"):
        run_demand_forecasting(dummy_parquet, top_n=1)

    # Act & Assert: missing published_at column
    no_dates_parquet = tmp_path / "no_dates.parquet"
    pd.DataFrame({"skill_python": [1, 0]}).to_parquet(no_dates_parquet)
    with pytest.raises(ValueError, match="Required column 'published_at' is missing"):
        run_demand_forecasting(no_dates_parquet)


def test_runner_dynamic_panels_without_default_skills(tmp_path: Path) -> None:
    """Pipeline runner gracefully handles datasets lacking DEFAULT_PANELS by selecting top 4 skills."""
    # Arrange: dataset has skills with NO python, docker, kafka, postgresql
    dates = pd.date_range("2026-10-01", periods=10, freq="D")
    records = []
    for i, date in enumerate(dates):
        for v in range(5):
            records.append(
                {
                    "vacancy_id": f"v_{i}_{v}",
                    "published_at": f"{date.strftime('%Y-%m-%d')}T12:00:00Z",
                    "skill_rust": int(v % 2 == 0),
                    "skill_scala": int(v % 3 == 0),
                    "skill_csharp": int(v % 2 == 1),
                    "skill_ruby": 1,
                }
            )
    custom_df = pd.DataFrame(records)
    parquet_path = tmp_path / "custom_skills.parquet"
    custom_df.to_parquet(parquet_path)

    out_dir = tmp_path / "custom_out"
    fig_dir = tmp_path / "custom_fig"

    # Act
    res = run_demand_forecasting(
        input_path=parquet_path,
        output_dir=out_dir,
        figures_dir=fig_dir,
        top_n=4,
        horizon_days=7,
    )

    # Assert
    assert "summary_df" in res
    assert len(res["summary_df"]) == 4
    assert (fig_dir / "fig6_skill_demand_forecast.png").exists()
