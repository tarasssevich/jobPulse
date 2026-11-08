"""Academic Publication Figures & Data Visualization for JobPulse Research.

Generates high-contrast, publication-grade figures (300 DPI, academic typography,
deep navy blue and slate gray palette) suitable for scientific conferences and journal papers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import matplotlib

# Use non-interactive Agg backend to avoid GUI dependencies
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from src.salary_model.econometrics import (
    calculate_skill_premiums,
    get_canonical_skill_name,
)

logger = logging.getLogger(__name__)

# Academic color palette constants
NAVY_PRIMARY: Final[str] = "#1f4e79"  # Deep navy blue
SLATE_SECONDARY: Final[str] = "#595959"  # Slate gray
SLATE_LIGHT: Final[str] = "#e0e0e0"  # Subtle grid gray
TEXT_DARK: Final[str] = "#262626"  # High contrast dark text
BOX_BG: Final[str] = "#dbe5f1"  # Soft slate-blue tint


def format_feature_label(raw_feature: str) -> str:
    """Transform raw encoded feature column names into publication-friendly labels."""
    if raw_feature.startswith("experience_level_"):
        lvl = raw_feature.removeprefix("experience_level_")
        return f"Grade: {lvl}"
    if raw_feature.startswith("work_format_"):
        fmt = raw_feature.removeprefix("work_format_").capitalize()
        return f"Format: {fmt}"
    if raw_feature.startswith("country_"):
        cntry = raw_feature.removeprefix("country_")
        translation = {
            "Беларусь": "Belarus",
            "Россия": "Russia",
            "Казахстан": "Kazakhstan",
        }
        return f"Country: {translation.get(cntry, cntry)}"
    if raw_feature == "skills_count":
        return "Total Skills Count"
    if raw_feature.startswith("skill_"):
        return f"Skill: {get_canonical_skill_name(raw_feature)}"
    return raw_feature.replace("_", " ").title()


class AcademicVisualizer:
    """Generates publication-quality scientific charts adhering to academic standards.

    Attributes:
        output_dir: Directory where figures will be saved (e.g. reports/figures/).
        dpi: Dots per inch for saved raster figures (default: 300).
    """

    def __init__(
        self,
        output_dir: Path | str = "reports/figures",
        dpi: int = 300,
    ) -> None:
        """Initialize visualizer and configure matplotlib publication styles."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._apply_publication_theme()

    def _apply_publication_theme(self) -> None:
        """Configure clean academic style for matplotlib."""
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
                "axes.edgecolor": SLATE_SECONDARY,
                "axes.linewidth": 0.8,
                "axes.grid": False,
                "grid.color": SLATE_LIGHT,
                "grid.linestyle": "--",
                "grid.alpha": 0.6,
                "text.color": TEXT_DARK,
                "axes.labelcolor": TEXT_DARK,
                "xtick.color": TEXT_DARK,
                "ytick.color": TEXT_DARK,
                "figure.autolayout": False,
            }
        )

    def plot_salary_distribution_by_grade(
        self,
        df: pd.DataFrame,
        filename: str = "fig1_salary_distribution_by_grade.png",
    ) -> Path:
        """Generate Fig 1: Boxplot of salaries (BYN) by experience level with annotated medians.

        Args:
            df: Input DataFrame containing `experience_level` and `salary_mid_byn`.
            filename: Target output filename.

        Returns:
            Path to saved figure.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If required columns are missing, or no valid records exist.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if "salary_mid_byn" not in df.columns or "experience_level" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'salary_mid_byn' and 'experience_level' columns."
            )

        sal_numeric = pd.to_numeric(df["salary_mid_byn"], errors="coerce")
        valid_mask = sal_numeric.notnull() & (sal_numeric > 0)
        valid = df.loc[valid_mask].copy()
        valid["salary_mid_byn"] = sal_numeric.loc[valid_mask]

        if valid.empty:
            raise ValueError("No valid records with positive salary_mid_byn found.")

        target_order = ["Junior", "Middle", "Middle+", "Senior"]
        order = [lvl for lvl in target_order if lvl in valid["experience_level"].unique()]
        if not order:
            order = [
                str(lvl) for lvl in valid["experience_level"].dropna().unique() if str(lvl).strip()
            ]
        if not order:
            raise ValueError(
                "No valid experience levels found in dataset for salary distribution plot."
            )

        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=self.dpi)
        try:
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", color=SLATE_LIGHT, alpha=0.7)

            # Build boxplot data
            data_by_grade = [
                valid.loc[valid["experience_level"] == lvl, "salary_mid_byn"].values
                for lvl in order
            ]

            ax.boxplot(
                data_by_grade,
                tick_labels=order,
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markeredgecolor": NAVY_PRIMARY,
                    "markerfacecolor": "white",
                    "markersize": 6,
                },
                medianprops={"color": NAVY_PRIMARY, "linewidth": 2.0},
                boxprops={"facecolor": BOX_BG, "edgecolor": NAVY_PRIMARY, "linewidth": 1.2},
                whiskerprops={"color": SLATE_SECONDARY, "linewidth": 1.0, "linestyle": "--"},
                capprops={"color": SLATE_SECONDARY, "linewidth": 1.2},
                flierprops={
                    "marker": "o",
                    "markersize": 3.5,
                    "markerfacecolor": SLATE_SECONDARY,
                    "markeredgecolor": "none",
                    "alpha": 0.4,
                },
            )

            # Annotate median and count for each grade
            for i, lvl in enumerate(order, start=1):
                subset = valid.loc[valid["experience_level"] == lvl, "salary_mid_byn"]
                med = subset.median()
                n_count = len(subset)
                ax.annotate(
                    f"Median: {med:,.0f} BYN\n(N={n_count})",
                    xy=(i, med),
                    xytext=(0, 22),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color=NAVY_PRIMARY,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": SLATE_LIGHT,
                        "alpha": 0.9,
                    },
                )

            ax.set_title(
                "Figure 1: Salary Distribution by Professional Grade (BYN)",
                fontsize=13,
                fontweight="bold",
                pad=15,
                color=TEXT_DARK,
            )
            ax.set_xlabel("Experience Level", fontsize=11, fontweight="bold", labelpad=8)
            ax.set_ylabel("Monthly Salary (BYN)", fontsize=11, fontweight="bold", labelpad=8)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved Figure 1 to %s", out_path)
            return out_path
        finally:
            plt.close(fig)

    def plot_top_skills_demand(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        filename: str = "fig2_top_skills_demand.png",
    ) -> Path:
        """Generate Fig 2: Horizontal bar chart of top 20 skills by percentage of market demand.

        Args:
            df: Vacancies dataset with binary `skill_*` columns.
            top_n: Number of top skills to display (default: 20).
            filename: Target output filename.

        Returns:
            Path to saved figure.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If df is empty or has no skill columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        skill_cols = [c for c in df.columns if c.startswith("skill_")]
        if not skill_cols:
            raise ValueError("No skill_* columns found in dataset.")

        total_vacancies = len(df)
        counts = (
            df[skill_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().nlargest(top_n)
        )

        # Prepare labels and percentages
        labels = [get_canonical_skill_name(c) for c in counts.index]
        pcts = [(cnt / total_vacancies) * 100.0 for cnt in counts.values]

        # Reverse to have highest skill at the top of horizontal chart
        labels = labels[::-1]
        pcts = pcts[::-1]
        counts_list = counts.values[::-1]

        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=self.dpi)
        try:
            ax.set_axisbelow(True)
            ax.grid(axis="x", linestyle="--", color=SLATE_LIGHT, alpha=0.7)

            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, pcts, color=NAVY_PRIMARY, edgecolor=NAVY_PRIMARY, height=0.65)

            # Annotate bars with percentage and raw count
            for bar, pct, cnt in zip(bars, pcts, counts_list):
                ax.text(
                    pct + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}% ({cnt:,})",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color=TEXT_DARK,
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9.5)
            max_pct = max(pcts) if pcts and max(pcts) > 0 else 1.0
            ax.set_xlim(0, max_pct * 1.18)
            ax.set_title(
                f"Figure 2: Top {top_n} IT Hard Skills by Market Demand (% of Vacancies)",
                fontsize=13,
                fontweight="bold",
                pad=15,
                color=TEXT_DARK,
            )
            ax.set_xlabel(
                f"Market Penetration (% of Total Vacancies, N = {total_vacancies:,})",
                fontsize=11,
                fontweight="bold",
                labelpad=8,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved Figure 2 to %s", out_path)
            return out_path
        finally:
            plt.close(fig)

    def plot_skill_premiums(
        self,
        df_premiums: pd.DataFrame,
        top_n: int = 15,
        filename: str = "fig3_skill_premium.png",
    ) -> Path:
        """Generate Fig 3: Horizontal bar chart of top 15 skills by wage premium (%) with p < 0.05 annotation.

        Args:
            df_premiums: Skill premiums DataFrame from calculate_skill_premiums.
            top_n: Number of top premium skills to display (default: 15).
            filename: Target output filename.

        Returns:
            Path to saved figure.

        Raises:
            TypeError: If df_premiums is not a DataFrame.
            ValueError: If df_premiums is empty.
        """
        if not isinstance(df_premiums, pd.DataFrame):
            raise TypeError("df_premiums must be a pandas DataFrame.")
        if df_premiums.empty:
            raise ValueError("df_premiums is empty.")

        top_df = df_premiums.head(top_n).copy()

        # Reverse order for horizontal plot (highest at top)
        top_df = top_df.iloc[::-1].reset_index(drop=True)

        labels = [
            str(top_df.loc[i, "skill_name"])
            if "skill_name" in top_df.columns
            else get_canonical_skill_name(str(top_df.loc[i, "skill"]))
            for i in range(len(top_df))
        ]
        premiums = pd.to_numeric(top_df["premium_pct"], errors="coerce").fillna(0.0).values
        is_sig = top_df["is_significant"].astype(bool).values
        p_vals = pd.to_numeric(top_df["p_value"], errors="coerce").fillna(1.0).values

        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=self.dpi)
        try:
            ax.set_axisbelow(True)
            ax.grid(axis="x", linestyle="--", color=SLATE_LIGHT, alpha=0.7)

            y_pos = np.arange(len(labels))
            colors = [NAVY_PRIMARY if sig else SLATE_SECONDARY for sig in is_sig]
            bars = ax.barh(y_pos, premiums, color=colors, edgecolor=NAVY_PRIMARY, height=0.65)

            min_p = float(min(premiums)) if len(premiums) > 0 else 0.0
            max_p = float(max(premiums)) if len(premiums) > 0 else 1.0
            if max_p <= 0.0 and min_p >= 0.0:
                max_p = 1.0
            x_min = min_p * 1.15 if min_p < 0 else 0.0
            x_max = max_p * 1.20 if max_p > 0 else 1.0
            ax.set_xlim(x_min, x_max)

            for bar, prem, sig, p in zip(bars, premiums, is_sig, p_vals):
                sig_marker = (
                    "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else " (ns)"))
                )
                prefix = "+" if prem > 0 else ""
                label_text = f"{prefix}{prem:.1f}% {sig_marker}"
                offset = 1.2 if prem >= 0 else -1.2
                ha = "left" if prem >= 0 else "right"
                ax.text(
                    prem + offset,
                    bar.get_y() + bar.get_height() / 2,
                    label_text,
                    va="center",
                    ha=ha,
                    fontsize=8.5,
                    fontweight="bold" if sig else "normal",
                    color=TEXT_DARK,
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9.5)
            ax.set_title(
                f"Figure 3: Top {top_n} Skills by Econometric Wage Premium (%)",
                fontsize=13,
                fontweight="bold",
                pad=15,
                color=TEXT_DARK,
            )
            ax.set_xlabel(
                "Median Salary Premium over Baseline (%) [Mann-Whitney U Test: * p<0.05, ** p<0.01, *** p<0.001]",
                fontsize=9.5,
                fontweight="bold",
                labelpad=8,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved Figure 3 to %s", out_path)
            return out_path
        finally:
            plt.close(fig)

    def plot_skill_cooccurrence_matrix(
        self,
        df: pd.DataFrame,
        top_n: int = 15,
        filename: str = "fig4_skill_cooccurrence_matrix.png",
    ) -> Path:
        """Generate Fig 4: Heatmap of co-occurrence frequencies between top 15 technologies.

        Args:
            df: Vacancies dataset with binary skill columns.
            top_n: Number of top technologies (default: 15).
            filename: Target output filename.

        Returns:
            Path to saved figure.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If df is empty or has no skill columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        skill_cols = [c for c in df.columns if c.startswith("skill_")]
        if not skill_cols:
            raise ValueError("No skill_* columns found in dataset.")

        top_cols = (
            df[skill_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum()
            .nlargest(top_n)
            .index.tolist()
        )
        top_names = [get_canonical_skill_name(c) for c in top_cols]

        sub_df = df[top_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        sub_df.columns = top_names

        # Compute co-occurrence matrix C = X^T X
        cooc = sub_df.T.dot(sub_df)

        fig, ax = plt.subplots(figsize=(10, 8.5), dpi=self.dpi)
        try:
            ax.grid(False)
            cmap = sns.light_palette(NAVY_PRIMARY, as_cmap=True)

            sns.heatmap(
                cooc,
                annot=True,
                fmt="d",
                cmap=cmap,
                cbar=True,
                ax=ax,
                linewidths=0.5,
                linecolor="white",
                annot_kws={"size": 8.5, "color": TEXT_DARK},
                cbar_kws={"label": "Co-Occurrence Frequency (Number of Vacancies)"},
            )

            ax.set_title(
                f"Figure 4: Technology Co-Occurrence Matrix (Top {top_n} Skills)",
                fontsize=13,
                fontweight="bold",
                pad=15,
                color=TEXT_DARK,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5)

            fig.tight_layout()
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved Figure 4 to %s", out_path)
            return out_path
        finally:
            plt.close(fig)

    def plot_feature_importances(
        self,
        feature_importances: pd.DataFrame | list[dict[str, Any]] | pd.Series,
        top_n: int = 15,
        filename: str = "fig5_feature_importance_model.png",
    ) -> Path:
        """Generate Fig 5: Top 15 most important features for salary prediction from Random Forest.

        Args:
            feature_importances: Feature importances from SalaryModelingPipeline.
            top_n: Number of top features to display (default: 15).
            filename: Target output filename.

        Returns:
            Path to saved figure.

        Raises:
            TypeError: If feature_importances is of unsupported type.
            ValueError: If feature_importances is empty.
        """
        if isinstance(feature_importances, list):
            fi_df = pd.DataFrame(feature_importances)
        elif isinstance(feature_importances, pd.Series):
            fi_df = pd.DataFrame(
                {
                    "feature": feature_importances.index,
                    "importance": feature_importances.values,
                }
            )
        elif isinstance(feature_importances, pd.DataFrame):
            fi_df = feature_importances.copy()
        else:
            raise TypeError("feature_importances must be a DataFrame, Series, or list of dicts.")

        if fi_df.empty:
            raise ValueError("feature_importances is empty.")

        top_df = fi_df.head(top_n).copy()
        top_df = top_df.iloc[::-1].reset_index(drop=True)

        labels = [format_feature_label(str(feat)) for feat in top_df["feature"]]
        importances = pd.to_numeric(top_df["importance"], errors="coerce").fillna(0.0).values

        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=self.dpi)
        try:
            ax.set_axisbelow(True)
            ax.grid(axis="x", linestyle="--", color=SLATE_LIGHT, alpha=0.7)

            y_pos = np.arange(len(labels))
            bars = ax.barh(
                y_pos, importances, color=NAVY_PRIMARY, edgecolor=NAVY_PRIMARY, height=0.65
            )

            max_imp = (
                float(max(importances)) if len(importances) > 0 and max(importances) > 0 else 1.0
            )
            ax.set_xlim(0, max_imp * 1.25)

            for bar, val in zip(bars, importances):
                ax.text(
                    val + (max_imp * 0.015),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f} ({val * 100:.1f}%)",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color=TEXT_DARK,
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9.5)
            ax.set_title(
                f"Figure 5: Random Forest Top {top_n} Feature Importances (Salary Model)",
                fontsize=13,
                fontweight="bold",
                pad=15,
                color=TEXT_DARK,
            )
            ax.set_xlabel(
                "Relative Feature Importance (Mean Decrease in Impurity)",
                fontsize=11,
                fontweight="bold",
                labelpad=8,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved Figure 5 to %s", out_path)
            return out_path
        finally:
            plt.close(fig)

    def generate_all_figures(
        self,
        df: pd.DataFrame,
        df_premiums: pd.DataFrame | None = None,
        feature_importances: Any = None,
    ) -> dict[str, Path]:
        """Generate all 5 academic figures and return dictionary of file paths.

        Args:
            df: Vacancies dataset.
            df_premiums: Skill premiums DataFrame (computed automatically if None).
            feature_importances: Random forest feature importances.

        Returns:
            Dictionary mapping figure identifiers to file Paths.
        """
        logger.info("Generating all 5 publication-ready figures in %s...", self.output_dir)

        if df_premiums is None:
            df_premiums = calculate_skill_premiums(df)

        generated: dict[str, Path] = {
            "fig1": self.plot_salary_distribution_by_grade(df),
            "fig2": self.plot_top_skills_demand(df, top_n=20),
            "fig3": self.plot_skill_premiums(df_premiums, top_n=15),
            "fig4": self.plot_skill_cooccurrence_matrix(df, top_n=15),
        }

        if feature_importances is not None:
            generated["fig5"] = self.plot_feature_importances(feature_importances, top_n=15)

        logger.info("All academic publication figures successfully generated.")
        return generated
