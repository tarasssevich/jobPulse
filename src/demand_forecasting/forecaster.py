"""Time-Series Skill Demand Forecaster using Holt Linear Trend Models.

Implements econometric time-series forecasting, 95% confidence interval estimation,
growth rate derivation, and trend categorization across tech skills.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Final

import numpy as np
import pandas as pd
from statsmodels.tsa.api import Holt, SimpleExpSmoothing

from src.demand_forecasting.timeseries import build_daily_skill_demand_series
from src.salary_model.econometrics import get_canonical_skill_name
from src.skills.enricher import skill_to_column_name

logger = logging.getLogger(__name__)

TREND_HIGH_GROWTH: Final[str] = "High Growth / Rising"
TREND_STABLE: Final[str] = "Stable / Evergreen"
TREND_DECLINING: Final[str] = "Declining"

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "skill_name",
    "current_share_pct",
    "forecast_share_pct",
    "delta_pct",
    "trend_category",
    "mae",
    "mape",
)


def categorize_trend(growth_rate_pct: float) -> str:
    """Categorize growth rate percentage into publication-standard trend buckets.

    Args:
        growth_rate_pct: Percentage change between forecast and current level.

    Returns:
        One of 'High Growth / Rising', 'Stable / Evergreen', or 'Declining'.
    """
    if growth_rate_pct > 5.0:
        return TREND_HIGH_GROWTH
    if growth_rate_pct < -5.0:
        return TREND_DECLINING
    return TREND_STABLE


class _NaiveFitResult:
    """Fallback model result mimicking statsmodels result object during severe degeneracy."""

    def __init__(self, ts: pd.Series, horizon_days: int) -> None:
        self.fittedvalues = ts.copy()
        self.resid = pd.Series(0.0, index=ts.index, dtype=float)
        last_val = float(ts.iloc[-1]) if len(ts) > 0 else 0.0

        if isinstance(ts.index, pd.DatetimeIndex) and ts.index.freq is not None:
            future_idx = pd.date_range(
                start=ts.index[-1] + ts.index.freq,
                periods=horizon_days,
                freq=ts.index.freq,
            )
        elif isinstance(ts.index, pd.DatetimeIndex) and len(ts.index) > 1:
            step = ts.index[-1] - ts.index[-2]
            future_idx = pd.date_range(
                start=ts.index[-1] + step,
                periods=horizon_days,
                freq=step,
            )
        elif isinstance(ts.index, pd.DatetimeIndex):
            future_idx = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=horizon_days,
                freq="D",
            )
        else:
            future_idx = pd.RangeIndex(start=len(ts), stop=len(ts) + horizon_days)

        self._forecast = pd.Series(last_val, index=future_idx, dtype=float)

    def forecast(self, steps: int) -> pd.Series:
        """Return projected constant baseline forecast."""
        return self._forecast.iloc[:steps]


class SkillDemandForecaster:
    """Holt linear trend model forecaster with 95% confidence intervals and metrics.

    Attributes:
        initialization_method: Initialization method for Holt smoothing ('estimated' or 'heuristic').
    """

    def __init__(self, initialization_method: str = "estimated") -> None:
        """Initialize the forecaster with smoothing parameters."""
        if not isinstance(initialization_method, str) or initialization_method not in (
            "estimated",
            "heuristic",
        ):
            raise ValueError("initialization_method must be 'estimated' or 'heuristic'.")
        self.initialization_method = initialization_method

    def fit_predict(
        self,
        series: pd.Series,
        horizon_days: int = 14,
    ) -> dict[str, Any]:
        """Fit Holt model on historical time series and project demand share.

        Args:
            series: Input 1D time-series of smoothed demand share (e.g. demand_share_smoothed).
            horizon_days: Number of calendar days to forecast into the future.

        Returns:
            Dictionary containing:
            - `forecast`: pd.Series of length horizon_days bounded [0.0, 100.0].
            - `ci_lower`: pd.Series of 95% lower confidence interval clamped [0.0, 100.0].
            - `ci_upper`: pd.Series of 95% upper confidence interval clamped [0.0, 100.0].
            - `metrics`: dict with 'mae', 'rmse', 'mape'.
            - `growth_rate_pct`: projected growth rate percentage guarded for div by zero.
            - `trend_category`: 'High Growth / Rising', 'Stable / Evergreen', or 'Declining'.
            - `fitted_values`: in-sample fitted values.
            - `resid_std`: residual standard deviation.
            - `model`: fitted statsmodels result object.

        Raises:
            TypeError: If series is not a pandas Series or horizon_days is not an int.
            ValueError: If series has fewer than 2 valid observations or horizon_days < 1.
        """
        if not isinstance(series, pd.Series):
            raise TypeError(f"series must be a pandas Series, got {type(series).__name__}.")
        if type(horizon_days) is not int or horizon_days < 1:
            raise ValueError("horizon_days must be an integer >= 1.")

        clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_series) < 2:
            raise ValueError(
                f"At least 2 non-null observations required for Holt forecasting, got {len(clean_series)}."
            )

        ts = clean_series.astype(float).clip(lower=0.0, upper=100.0).copy()

        # Determine future date index
        if isinstance(ts.index, pd.DatetimeIndex):
            future_idx = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=horizon_days,
                freq="D",
            )
        else:
            future_idx = pd.RangeIndex(start=len(ts), stop=len(ts) + horizon_days)

        # Multi-tier fit attempt
        fit_result: Any = None
        raw_forecast_vals: np.ndarray | None = None
        fitted_vals_series: pd.Series | None = None
        resid_series: pd.Series | None = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 1. Try fitting directly on indexed series
            for init_m in (self.initialization_method, "heuristic"):
                try:
                    model = Holt(ts, initialization_method=init_m)
                    res = model.fit()
                    raw_fc = res.forecast(horizon_days)
                    raw_forecast_vals = np.asarray(raw_fc.values, dtype=float)
                    fitted_vals_series = pd.Series(
                        res.fittedvalues.values, index=ts.index, dtype=float
                    )
                    resid_series = pd.Series(res.resid.values, index=ts.index, dtype=float)
                    fit_result = res
                    break
                except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, RuntimeError) as err:
                    logger.debug(
                        "Holt fit with '%s' on indexed series failed (%s).", init_m, err
                    )

            # 2. If indexed fit failed (e.g. irregular DatetimeIndex gaps), fit on plain array series
            if raw_forecast_vals is None:
                ts_plain = pd.Series(ts.values, dtype=float)
                for init_m in (self.initialization_method, "heuristic"):
                    try:
                        model = Holt(ts_plain, initialization_method=init_m)
                        res = model.fit()
                        raw_fc = res.forecast(horizon_days)
                        raw_forecast_vals = np.asarray(raw_fc.values, dtype=float)
                        fitted_vals_series = pd.Series(
                            res.fittedvalues.values, index=ts.index, dtype=float
                        )
                        resid_series = pd.Series(
                            res.resid.values, index=ts.index, dtype=float
                        )
                        fit_result = res
                        break
                    except (
                        ValueError,
                        np.linalg.LinAlgError,
                        ZeroDivisionError,
                        RuntimeError,
                    ) as err:
                        logger.debug(
                            "Holt fit with '%s' on plain series failed (%s).", init_m, err
                        )

            # 3. Fallback to SimpleExpSmoothing if Holt failed
            if raw_forecast_vals is None:
                ts_plain = pd.Series(ts.values, dtype=float)
                try:
                    ses_model = SimpleExpSmoothing(ts_plain, initialization_method="heuristic")
                    res = ses_model.fit()
                    raw_fc = res.forecast(horizon_days)
                    raw_forecast_vals = np.asarray(raw_fc.values, dtype=float)
                    fitted_vals_series = pd.Series(
                        res.fittedvalues.values, index=ts.index, dtype=float
                    )
                    resid_series = pd.Series(res.resid.values, index=ts.index, dtype=float)
                    fit_result = res
                except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, RuntimeError) as err:
                    logger.warning(
                        "SimpleExpSmoothing optimization failed (%s), using naive fallback.",
                        err,
                    )

            # 4. Fallback to Naive baseline if all optimizations failed
            if raw_forecast_vals is None:
                naive_res = _NaiveFitResult(ts, horizon_days)
                raw_fc = naive_res.forecast(horizon_days)
                raw_forecast_vals = np.asarray(raw_fc.values, dtype=float)
                fitted_vals_series = naive_res.fittedvalues
                resid_series = naive_res.resid
                fit_result = naive_res

            last_obs = float(ts.iloc[-1])
            raw_fc_series = pd.Series(raw_forecast_vals, index=future_idx, dtype=float)
            forecast = (
                raw_fc_series.replace([np.inf, -np.inf], np.nan)
                .fillna(last_obs)
                .clip(lower=0.0, upper=100.0)
            )
            residuals = resid_series.replace([np.inf, -np.inf], np.nan).dropna()
            fitted_vals = fitted_vals_series.replace([np.inf, -np.inf], np.nan).fillna(ts)

        if len(residuals) > 1:
            resid_std = float(residuals.std(ddof=1))
        else:
            resid_std = 0.0

        if np.isnan(resid_std) or np.isinf(resid_std) or resid_std < 0.0:
            resid_std = 0.0

        # Compute 95% confidence intervals and clamp bounds between 0.0% and 100.0%
        ci_lower = (forecast - 1.96 * resid_std).clip(lower=0.0, upper=100.0)
        ci_upper = (forecast + 1.96 * resid_std).clip(lower=0.0, upper=100.0)

        # In-sample evaluation metrics
        y_true = ts.to_numpy()
        y_pred = fitted_vals.to_numpy()
        diff = y_true - y_pred
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))

        nonzero_mask = np.abs(y_true) > 1e-4
        if np.any(nonzero_mask):
            mape = float(
                np.mean(np.abs(diff[nonzero_mask] / y_true[nonzero_mask]))
                * 100.0
            )
        else:
            mape = 0.0

        if np.isnan(mae) or np.isinf(mae):
            mae = 0.0
        if np.isnan(rmse) or np.isinf(rmse):
            rmse = 0.0
        if np.isnan(mape) or np.isinf(mape):
            mape = 0.0

        # Calculate projected growth rate guarded against division by zero
        current_share = float(ts.iloc[-1])
        forecast_end = float(forecast.iloc[-1])

        if current_share > 1e-6:
            growth_rate_pct = ((forecast_end - current_share) / current_share) * 100.0
        elif forecast_end > 1e-6:
            # Emerging skill transitioning from 0% share to positive demand
            growth_rate_pct = 100.0
        else:
            growth_rate_pct = 0.0

        if np.isnan(growth_rate_pct) or np.isinf(growth_rate_pct):
            growth_rate_pct = 0.0

        trend_cat = categorize_trend(growth_rate_pct)

        return {
            "forecast": forecast,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "metrics": {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4),
            },
            "growth_rate_pct": round(growth_rate_pct, 4),
            "trend_category": trend_cat,
            "fitted_values": fitted_vals,
            "resid_std": round(resid_std, 4),
            "model": fit_result,
        }


def _resolve_skill_column(skill_identifier: str, available_columns: set[str]) -> str | None:
    """Resolve user-supplied skill name to existing binary DataFrame column."""
    if not isinstance(skill_identifier, str):
        return None
    if skill_identifier in available_columns:
        return skill_identifier
    candidate_1 = f"skill_{skill_identifier.lower()}"
    if candidate_1 in available_columns:
        return candidate_1
    candidate_2 = skill_to_column_name(skill_identifier)
    if candidate_2 in available_columns:
        return candidate_2
    return None


def forecast_top_skills(
    df: pd.DataFrame,
    top_skills: list[str],
    horizon_days: int = 14,
) -> pd.DataFrame:
    """Evaluate and summarize Holt time-series forecasting across a list of skills.

    Args:
        df: Input vacancies DataFrame with `published_at` and skill binary columns.
        top_skills: List of skill column names or names to evaluate.
        horizon_days: Forecast horizon in days (default: 14).

    Returns:
        DataFrame with columns:
        `[skill_name, current_share_pct, forecast_share_pct, delta_pct, trend_category, mae, mape]`.

    Raises:
        TypeError: If df is not a DataFrame or top_skills is not a sequence.
        ValueError: If horizon_days < 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    if not isinstance(top_skills, (list, tuple)):
        raise TypeError(f"top_skills must be a list or tuple, got {type(top_skills).__name__}.")
    if type(horizon_days) is not int or horizon_days < 1:
        raise ValueError("horizon_days must be an integer >= 1.")

    available_cols = set(df.columns)
    forecaster = SkillDemandForecaster()
    rows: list[dict[str, Any]] = []

    for raw_skill in top_skills:
        col = _resolve_skill_column(raw_skill, available_cols)
        if col is None:
            logger.warning("Skill column for '%s' not found in DataFrame. Skipping.", raw_skill)
            continue

        series_df = build_daily_skill_demand_series(df, col)
        if len(series_df) < 2:
            logger.warning("Insufficient time-series points for skill '%s'. Skipping.", raw_skill)
            continue

        ts = series_df["demand_share_smoothed"]
        result = forecaster.fit_predict(ts, horizon_days=horizon_days)

        current_share = round(float(ts.iloc[-1]), 2)
        forecast_share = round(float(result["forecast"].iloc[-1]), 2)
        delta_pct = round(float(result["growth_rate_pct"]), 2)
        trend_cat = result["trend_category"]
        mae = round(float(result["metrics"]["mae"]), 4)
        mape = round(float(result["metrics"]["mape"]), 4)
        canonical_name = get_canonical_skill_name(col)

        rows.append(
            {
                "skill_name": canonical_name,
                "current_share_pct": current_share,
                "forecast_share_pct": forecast_share,
                "delta_pct": delta_pct,
                "trend_category": trend_cat,
                "mae": mae,
                "mape": mape,
            }
        )

    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
