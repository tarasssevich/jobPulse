"""Orchestrator for Step 3: Econometric Analysis, Salary ML Models & Academic Publication Figures.

Executes:
1. Loads enriched vacancies dataset (`data/processed/vacancies_enriched.parquet`).
2. Calculates econometric skill premiums and tests significance -> saves to `data/processed/skill_premiums.csv`.
3. Trains Baseline, Ridge, and Random Forest models with 5-fold CV -> saves to `data/processed/model_metrics.json`.
4. Generates 5 academic publication figures at 300 DPI -> saves to `reports/figures/`.
5. Prints publication summary tables.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.analysis.visualizer import AcademicVisualizer
from src.salary_model.econometrics import calculate_skill_premiums
from src.salary_model.models import SalaryModelingPipeline

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DATASET: Final[str] = "data/processed/vacancies_enriched.parquet"
DEFAULT_PREMIUMS_OUTPUT: Final[str] = "data/processed/skill_premiums.csv"
DEFAULT_METRICS_OUTPUT: Final[str] = "data/processed/model_metrics.json"
DEFAULT_FIGURES_DIR: Final[str] = "reports/figures"


def run_salary_analysis(
    input_path: Path | str = DEFAULT_INPUT_DATASET,
    premiums_output_path: Path | str = DEFAULT_PREMIUMS_OUTPUT,
    metrics_output_path: Path | str = DEFAULT_METRICS_OUTPUT,
    figures_dir: Path | str = DEFAULT_FIGURES_DIR,
    min_vacancies_for_premium: int = 15,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute the complete Step 3 salary analysis workflow.

    Args:
        input_path: Path to vacancies_enriched.parquet dataset.
        premiums_output_path: Output CSV path for skill premiums.
        metrics_output_path: Output JSON path for model evaluation metrics.
        figures_dir: Output directory for academic publication figures.
        min_vacancies_for_premium: Threshold sample size for econometric tests.
        random_state: Random state for reproducibility.

    Returns:
        Structured dictionary containing premiums, model results, and generated figures.
    """
    in_file = Path(input_path)
    prem_file = Path(premiums_output_path)
    metrics_file = Path(metrics_output_path)
    fig_dir = Path(figures_dir)

    if not in_file.exists():
        raise FileNotFoundError(f"Input dataset not found at {in_file.resolve()}")

    logger.info("Step 3: Loading enriched vacancies dataset from %s", in_file)
    df = pd.read_parquet(in_file)
    total_vacancies = len(df)
    if "salary_mid_byn" not in df.columns:
        raise ValueError(f"Column 'salary_mid_byn' not found in dataset {in_file}")

    sal_numeric = pd.to_numeric(df["salary_mid_byn"], errors="coerce")
    n_salary = int((sal_numeric.notnull() & (sal_numeric > 0)).sum())

    if n_salary == 0:
        raise ValueError(f"No valid records with positive salary_mid_byn found in {in_file}")

    logger.info(
        "Dataset loaded: %d total vacancies, %d with valid salary observations.",
        total_vacancies,
        n_salary,
    )

    # 1. Econometric Analysis: Skill Premiums & Mann-Whitney U tests
    logger.info(
        "Computing econometric skill wage premiums (min_vacancies=%d)...", min_vacancies_for_premium
    )
    df_premiums = calculate_skill_premiums(df, min_vacancies=min_vacancies_for_premium)

    prem_file.parent.mkdir(parents=True, exist_ok=True)
    df_premiums.to_csv(prem_file, index=False)
    logger.info("Saved %d skill premiums to %s", len(df_premiums), prem_file)

    # 2. Machine Learning: Salary Modeling Pipeline
    logger.info("Training salary ML models (Baseline, Ridge, Random Forest)...")
    pipeline = SalaryModelingPipeline(random_state=random_state)
    model_results = pipeline.run(df)

    # Prepare JSON serializable metrics
    metrics_json_data = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "dataset": {
            "input_file": str(in_file),
            "total_vacancies": total_vacancies,
            "vacancies_with_salary": n_salary,
            "train_size": model_results["train_size"],
            "test_size": model_results["test_size"],
            "features_count": len(model_results["feature_names"]),
        },
        "models": {
            "baseline": model_results["metrics"]["baseline"],
            "ridge": model_results["metrics"]["ridge"],
            "random_forest": model_results["metrics"]["random_forest"],
        },
        "feature_importances_top20": model_results["feature_importances_list"],
        "econometrics_summary": {
            "total_skills_evaluated": len(df_premiums),
            "statistically_significant_count": int(df_premiums["is_significant"].sum()),
            "top_10_premiums": df_premiums[
                ["skill_name", "N_with", "diff_median", "premium_pct", "p_value", "is_significant"]
            ]
            .head(10)
            .to_dict(orient="records"),
        },
    }

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_json_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved model metrics to %s", metrics_file)

    # 3. Academic Visualizations (5 Figures, 300 DPI)
    logger.info("Generating 5 academic publication figures in %s...", fig_dir)
    visualizer = AcademicVisualizer(output_dir=fig_dir, dpi=300)
    generated_figures = visualizer.generate_all_figures(
        df=df,
        df_premiums=df_premiums,
        feature_importances=model_results["feature_importances"],
    )

    # 4. Print Summary Tables
    print_summary_report(model_results, df_premiums, generated_figures)

    return {
        "df_premiums": df_premiums,
        "model_results": model_results,
        "metrics_json": metrics_json_data,
        "generated_figures": generated_figures,
    }


def print_summary_report(
    model_results: dict[str, Any],
    df_premiums: pd.DataFrame,
    generated_figures: dict[str, Path],
) -> None:
    """Print clean academic publication summary tables to standard output."""
    sep = "=" * 80
    subsep = "-" * 80

    print("\n" + sep)
    print(" JOBPULSE STEP 3: ECONOMETRICS, SALARY ML MODELS & PUBLICATION FIGURES ")
    print(sep)

    # 1. Model Performance Table
    print("\n[1] SALARY REGRESSION MODELS PERFORMANCE (TEST SET & 5-FOLD CV)")
    print(subsep)
    header = f"{'Model':<18} | {'Test R^2':<8} | {'MAE (BYN)':<10} | {'RMSE (BYN)':<11} | {'CV R^2 (Train)':<18}"
    print(header)
    print(subsep)

    metrics = model_results["metrics"]
    for key, name in [
        ("baseline", "Baseline (Median)"),
        ("ridge", "Ridge (L2 Reg)"),
        ("random_forest", "Random Forest"),
    ]:
        m = metrics[key]
        r2_str = f"{m['r2']:.4f}"
        mae_str = f"{m['mae']:,.1f}"
        rmse_str = f"{m['rmse']:,.1f}"
        cv_str = f"{m['cv_r2_mean']:.4f} (+/- {m['cv_r2_std']:.4f})"
        print(f"{name:<18} | {r2_str:<8} | {mae_str:<10} | {rmse_str:<11} | {cv_str:<18}")
    print(subsep)

    # 2. Econometric Premiums Table
    print("\n[2] TOP 10 HARD SKILLS BY ECONOMETRIC WAGE PREMIUM (MANN-WHITNEY U TEST)")
    print(subsep)
    prem_hdr = f"{'Skill':<18} | {'N':<5} | {'Diff Med (BYN)':<14} | {'Premium %':<10} | {'p-value':<12} | {'Significant'}"
    print(prem_hdr)
    print(subsep)

    top_prem = df_premiums.head(10)
    for _, row in top_prem.iterrows():
        s_name = str(row["skill_name"])[:18]
        n_w = int(row["N_with"])
        diff_m = f"{row['diff_median']:+,.1f}"
        prem = f"{row['premium_pct']:+.1f}%"
        pval = f"{row['p_value']:.2e}"
        sig = "YES (p<0.05)" if row["is_significant"] else "No"
        print(f"{s_name:<18} | {n_w:<5} | {diff_m:<14} | {prem:<10} | {pval:<12} | {sig}")
    print(subsep)

    # 3. Top Random Forest Features
    print("\n[3] TOP 10 MOST INFLUENTIAL SALARY PREDICTORS (RANDOM FOREST)")
    print(subsep)
    print(f"{'Rank':<5} | {'Feature':<35} | {'Importance':<12}")
    print(subsep)

    top_fi = model_results["feature_importances"].head(10)
    for idx, (_, row) in enumerate(top_fi.iterrows(), start=1):
        feat = str(row["feature"])[:35]
        imp = f"{row['importance']:.4f} ({row['importance'] * 100:.1f}%)"
        print(f"{idx:<5} | {feat:<35} | {imp:<12}")
    print(subsep)

    # 4. Publication Figures
    print("\n[4] GENERATED ACADEMIC PUBLICATION FIGURES (300 DPI)")
    print(subsep)
    for fig_id, fig_path in generated_figures.items():
        print(f"- {fig_id.upper()}: {fig_path.resolve()} ({fig_path.stat().st_size / 1024:.1f} KB)")
    print(sep + "\n")


def main() -> None:
    """CLI entrypoint for Step 3 salary analysis orchestrator."""
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="JobPulse Step 3: Econometric Analysis, Salary ML Models & Academic Publication Figures"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_INPUT_DATASET,
        help=f"Path to input enriched Parquet dataset (default: {DEFAULT_INPUT_DATASET})",
    )
    parser.add_argument(
        "--premiums-output",
        "-p",
        type=str,
        default=DEFAULT_PREMIUMS_OUTPUT,
        help=f"Path to output skill premiums CSV (default: {DEFAULT_PREMIUMS_OUTPUT})",
    )
    parser.add_argument(
        "--metrics-output",
        "-m",
        type=str,
        default=DEFAULT_METRICS_OUTPUT,
        help=f"Path to output model metrics JSON (default: {DEFAULT_METRICS_OUTPUT})",
    )
    parser.add_argument(
        "--figures-dir",
        "-f",
        type=str,
        default=DEFAULT_FIGURES_DIR,
        help=f"Directory for generated publication figures (default: {DEFAULT_FIGURES_DIR})",
    )
    parser.add_argument(
        "--min-vacancies",
        type=int,
        default=15,
        help="Minimum number of vacancies with skill required for premium test (default: 15)",
    )

    args = parser.parse_args()

    try:
        run_salary_analysis(
            input_path=args.input,
            premiums_output_path=args.premiums_output,
            metrics_output_path=args.metrics_output,
            figures_dir=args.figures_dir,
            min_vacancies_for_premium=args.min_vacancies,
        )
    except Exception:
        logger.exception("Salary analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
