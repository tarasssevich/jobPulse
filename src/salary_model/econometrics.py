"""Econometric Analysis & Skill Premium Evaluation for IT Job Market.

Implements non-parametric statistical evaluation (Mann-Whitney U test)
to measure skill-based wage premiums and statistical significance across vacancies.
"""

from __future__ import annotations

import logging
import math
from types import MappingProxyType
from typing import Final

import pandas as pd
from scipy.stats import mannwhitneyu

from src.skills.enricher import skill_to_column_name
from src.skills.taxonomy import CANONICAL_SKILLS

logger = logging.getLogger(__name__)

# Precompute mapping from binary column name (e.g. 'skill_ci_cd') to canonical name ('CI/CD')
_COLUMN_TO_CANONICAL: Final[MappingProxyType[str, str]] = MappingProxyType(
    {skill_to_column_name(s): s for s in CANONICAL_SKILLS}
)

# Standard result column schema
PREMIUM_COLUMNS: Final[tuple[str, ...]] = (
    "skill",
    "skill_name",
    "N_with",
    "n_with",
    "median_with",
    "median_without",
    "mean_with",
    "mean_without",
    "diff_median",
    "premium_pct",
    "p_value",
    "is_significant",
)


def get_canonical_skill_name(column_name: str) -> str:
    """Resolve a feature column name to its canonical display name.

    Args:
        column_name: Column name, e.g. 'skill_python' or 'skill_ci_cd'.

    Returns:
        Canonical skill name, e.g. 'Python' or 'CI/CD'.

    Examples:
        >>> get_canonical_skill_name("skill_python")
        'Python'
        >>> get_canonical_skill_name("skill_ci_cd")
        'CI/CD'
    """
    if column_name in _COLUMN_TO_CANONICAL:
        return _COLUMN_TO_CANONICAL[column_name]
    raw = column_name.removeprefix("skill_").replace("_", " ").strip()
    return raw.title() if raw else column_name


def calculate_skill_premiums(
    df: pd.DataFrame,
    min_vacancies: int = 15,
) -> pd.DataFrame:
    """Calculate econometric wage premiums and statistical significance for skills.

    For each skill binary column (`skill_*`), compares the salary distribution of
    vacancies requiring the skill versus those that do not, using the non-parametric
    two-sided Mann-Whitney U test.

    This function is pure and does not mutate the input DataFrame.

    Args:
        df: Input DataFrame containing `salary_mid_byn` and binary `skill_*` columns.
        min_vacancies: Minimum number of vacancies requiring the skill (skill == 1)
            required to calculate the premium. Defaults to 15.

    Returns:
        pd.DataFrame: Econometric evaluation table sorted by `premium_pct` descending,
            containing:
            - skill: Column name (e.g. 'skill_python')
            - skill_name: Canonical display name (e.g. 'Python')
            - N_with: Count of vacancies requiring the skill (skill == 1)
            - n_with: Alias for N_with
            - median_with: Median salary (BYN) with skill
            - median_without: Median salary (BYN) without skill
            - mean_with: Mean salary (BYN) with skill
            - mean_without: Mean salary (BYN) without skill
            - diff_median: Difference in median salary (median_with - median_without)
            - premium_pct: Relative wage premium percentage ((median_with - median_without) / median_without * 100)
            - p_value: Two-sided Mann-Whitney U test p-value
            - is_significant: Boolean flag (p_value < 0.05)

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If `salary_mid_byn` is missing or `min_vacancies` < 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if "salary_mid_byn" not in df.columns:
        raise ValueError("Column 'salary_mid_byn' not found in DataFrame.")
    if min_vacancies < 1:
        raise ValueError("min_vacancies must be at least 1.")

    # Pure filter: records where salary_mid_byn is present and positive
    sal_numeric = pd.to_numeric(df["salary_mid_byn"], errors="coerce")
    valid_salary_mask = sal_numeric.notnull() & (sal_numeric > 0)
    df_sal = df.loc[valid_salary_mask].copy()
    df_sal["salary_mid_byn"] = sal_numeric.loc[valid_salary_mask]

    # Identify all binary skill columns
    skill_cols = [c for c in df_sal.columns if c.startswith("skill_")]

    if df_sal.empty or not skill_cols:
        return pd.DataFrame(columns=list(PREMIUM_COLUMNS))

    records: list[dict[str, object]] = []

    for col in skill_cols:
        series_skill = df_sal[col]
        # Allow boolean, integer/float binary, or string indicators cleanly
        if pd.api.types.is_bool_dtype(series_skill):
            mask_with = series_skill.fillna(False).astype(bool)
        else:
            mask_with = series_skill.isin([1, 1.0, True, "1", "True"]).fillna(False).astype(bool)

        sal_with = df_sal.loc[mask_with, "salary_mid_byn"].dropna()
        sal_without = df_sal.loc[~mask_with, "salary_mid_byn"].dropna()

        n_with = len(sal_with)
        n_without = len(sal_without)

        # Skip if sample size is below threshold or control group is empty
        if n_with < min_vacancies or n_without == 0:
            continue

        median_with = float(sal_with.median())
        median_without = float(sal_without.median())
        mean_with = float(sal_with.mean())
        mean_without = float(sal_without.mean())

        diff_median = median_with - median_without
        if median_without > 0:
            premium_pct = (diff_median / median_without) * 100.0
        else:
            premium_pct = 0.0

        # Perform two-sided Mann-Whitney U test
        try:
            mwu_res = mannwhitneyu(sal_with, sal_without, alternative="two-sided")
            p_val = float(mwu_res.pvalue)
            # Handle total ties across groups where SciPy produces NaN
            if math.isnan(p_val):
                p_val = 1.0
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            logger.warning("Mann-Whitney U test error for column '%s': %s", col, exc)
            p_val = 1.0

        is_significant = bool(p_val < 0.05)
        canonical_name = get_canonical_skill_name(col)

        records.append(
            {
                "skill": col,
                "skill_name": canonical_name,
                "N_with": n_with,
                "n_with": n_with,
                "median_with": round(median_with, 2),
                "median_without": round(median_without, 2),
                "mean_with": round(mean_with, 2),
                "mean_without": round(mean_without, 2),
                "diff_median": round(diff_median, 2),
                "premium_pct": round(premium_pct, 2),
                "p_value": p_val,
                "is_significant": is_significant,
            }
        )

    if not records:
        return pd.DataFrame(columns=list(PREMIUM_COLUMNS))

    result_df = pd.DataFrame(records)
    # Sort by wage premium descending
    sorted_df = result_df.sort_values(by="premium_pct", ascending=False).reset_index(drop=True)
    return sorted_df
