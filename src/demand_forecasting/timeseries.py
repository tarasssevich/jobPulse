"""Time-Series Aggregation & Daily Skill Demand Construction.

Provides pure functional routines for extracting, aggregating, and smoothing
daily skill demand market shares from job market vacancy postings.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SERIES_COLUMNS: Final[tuple[str, ...]] = (
    "total_vacancies",
    "skill_vacancies",
    "demand_share",
    "demand_share_smoothed",
)


def build_daily_skill_demand_series(
    df: pd.DataFrame,
    skill_col: str,
    rolling_window: int = 7,
) -> pd.DataFrame:
    """Build a continuous daily time-series of skill demand share and smoothed trend.

    Args:
        df: Input vacancies DataFrame with `published_at` and `skill_col` columns.
        skill_col: Name of the binary or numeric skill column (e.g. 'skill_python').
        rolling_window: Rolling window in days for moving average smoothing (default: 7).

    Returns:
        DataFrame indexed by calendar date with columns:
        - `total_vacancies`: total number of vacancies published on that day.
        - `skill_vacancies`: count of vacancies requiring the skill.
        - `demand_share`: daily market share percentage (0.0 to 100.0).
        - `demand_share_smoothed`: N-day moving average of demand share.

    Raises:
        TypeError: If df is not a DataFrame or skill_col is not a string.
        ValueError: If required columns are missing or rolling_window < 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    if not isinstance(skill_col, str):
        raise TypeError(f"skill_col must be a string, got {type(skill_col).__name__}.")
    if type(rolling_window) is not int or rolling_window < 1:
        raise ValueError("rolling_window must be an integer >= 1.")
    if "published_at" not in df.columns:
        raise ValueError("Required column 'published_at' is missing from DataFrame.")
    if skill_col not in df.columns:
        raise ValueError(f"Skill column '{skill_col}' not found in DataFrame.")

    empty_df = pd.DataFrame(
        {
            "total_vacancies": pd.Series(dtype="int64"),
            "skill_vacancies": pd.Series(dtype="int64"),
            "demand_share": pd.Series(dtype="float64"),
            "demand_share_smoothed": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], name="date", freq="D"),
    )

    if df.empty:
        return empty_df

    # Parse timestamps cleanly, coercing corrupted or missing values to NaT
    parsed_dates = pd.to_datetime(
        df["published_at"],
        format="mixed",
        utc=True,
        errors="coerce",
    )
    valid_mask = parsed_dates.notna()

    if not valid_mask.any():
        logger.warning("No valid timestamps found in 'published_at' column.")
        return empty_df

    dates = parsed_dates.loc[valid_mask].dt.tz_convert(None).dt.floor("D")
    raw_skills = df.loc[valid_mask, skill_col]
    skill_vals = (
        pd.to_numeric(raw_skills, errors="coerce")
        .fillna(0)
        .clip(lower=0.0, upper=1.0)
    )

    # Daily aggregation
    daily_totals = dates.groupby(dates).size()
    daily_skills = skill_vals.groupby(dates).sum()

    # Reindex to continuous daily calendar range to prevent gaps
    min_date = dates.min()
    max_date = dates.max()
    full_date_index = pd.date_range(start=min_date, end=max_date, freq="D", name="date")

    total_vacancies = daily_totals.reindex(full_date_index, fill_value=0).astype("int64")
    skill_vacancies = daily_skills.reindex(full_date_index, fill_value=0).astype("int64")

    # Graceful calculation of market share percentage, handling days with 0 vacancies
    demand_share_values = np.where(
        total_vacancies > 0,
        np.clip((skill_vacancies / total_vacancies) * 100.0, 0.0, 100.0),
        0.0,
    )
    demand_share = pd.Series(
        demand_share_values,
        index=full_date_index,
        name="demand_share",
        dtype="float64",
    )

    demand_share_smoothed = demand_share.rolling(
        window=rolling_window,
        min_periods=1,
    ).mean()

    return pd.DataFrame(
        {
            "total_vacancies": total_vacancies,
            "skill_vacancies": skill_vacancies,
            "demand_share": demand_share,
            "demand_share_smoothed": demand_share_smoothed,
        },
        index=full_date_index,
    )
