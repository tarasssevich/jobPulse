"""Dataset enrichment pipeline with extracted hard skills and binary ML feature columns.

Enriches the preprocessed analytical dataset (vacancies_clean.parquet) with:
1. `extracted_skills`: Canonical list of skills found per vacancy.
2. `skills_count`: Total number of canonical skills.
3. `skills_by_category`: JSON-serialized taxonomy categorization.
4. Top-N binary indicator columns (`skill_<name>`: 0 or 1) ready for ML models.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Final

import pandas as pd

from src.skills.extractor import SkillExtractor

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PARQUET: Final[str] = "data/processed/vacancies_clean.parquet"
DEFAULT_OUTPUT_PARQUET: Final[str] = "data/processed/vacancies_enriched.parquet"
DEFAULT_TOP_N_FEATURES: Final[int] = 50


def skill_to_column_name(skill: str) -> str:
    """Convert canonical skill name to standardized ML feature column name.

    Examples:
        >>> skill_to_column_name("Python")
        'skill_python'
        >>> skill_to_column_name("C++")
        'skill_cpp'
        >>> skill_to_column_name("C#")
        'skill_csharp'
        >>> skill_to_column_name(".NET")
        'skill_dotnet'
        >>> skill_to_column_name("CI/CD")
        'skill_ci_cd'
        >>> skill_to_column_name("Node.js")
        'skill_node_js'
        >>> skill_to_column_name("REST API")
        'skill_rest_api'
        >>> skill_to_column_name("1C")
        'skill_1c'
    """
    clean = skill.strip().lower()
    clean = clean.replace("c++", "cpp")
    clean = clean.replace("c#", "csharp")
    clean = clean.replace(".net", "dotnet")
    clean = clean.replace("/", "_")
    clean = clean.replace(".", "_")
    clean = clean.replace("-", "_")
    clean = clean.replace(" ", "_")
    clean = re.sub(r"[^a-z0-9_]", "", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return f"skill_{clean}"


def enrich_dataset(
    input_parquet: Path | str = DEFAULT_INPUT_PARQUET,
    output_parquet: Path | str = DEFAULT_OUTPUT_PARQUET,
    top_n_features: int = DEFAULT_TOP_N_FEATURES,
) -> pd.DataFrame:
    """Enrich clean vacancies dataset with extracted skills and binary indicator features.

    Args:
        input_parquet: Path to input cleaned Parquet dataset.
        output_parquet: Destination path for enriched Parquet dataset.
        top_n_features: Number of most frequent skills to create binary indicator columns for.

    Returns:
        Enriched pandas DataFrame.
    """
    in_path = Path(input_parquet)
    out_parquet_path = Path(output_parquet)
    out_csv_path = out_parquet_path.with_suffix(".csv")

    if not in_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {in_path.resolve()}")

    logger.info("Reading input dataset from %s", in_path)
    df = pd.read_parquet(in_path).copy()

    extractor = SkillExtractor()

    # Extract skills row by row using both key_skills and description_clean
    key_skills_series = df["key_skills"] if "key_skills" in df.columns else [None] * len(df)
    desc_series = (
        df["description_clean"]
        if "description_clean" in df.columns
        else (df["description"] if "description" in df.columns else [""] * len(df))
    )

    extracted_skills_list: list[list[str]] = []
    skills_count_list: list[int] = []
    skills_by_category_list: list[str] = []
    skills_sets: list[set[str]] = []

    for keys, text in zip(key_skills_series, desc_series):
        res = extractor.extract(key_skills=keys, text=text)
        skills = res["skills"]
        extracted_skills_list.append(skills)
        skills_count_list.append(res["count"])
        skills_by_category_list.append(json.dumps(res["by_category"], ensure_ascii=False))
        skills_sets.append(set(skills))

    # Add core extraction columns
    df["extracted_skills"] = extracted_skills_list
    df["skills_count"] = pd.Series(skills_count_list, dtype="int64", index=df.index)
    df["skills_by_category"] = skills_by_category_list

    # Determine top N most frequent skills across the dataset
    skill_counter: Counter[str] = Counter()
    for skills in extracted_skills_list:
        skill_counter.update(skills)

    top_skills_with_counts = skill_counter.most_common(top_n_features)
    top_skills = [skill for skill, _ in top_skills_with_counts]

    # Generate binary indicator columns (0 or 1) for ML models
    for skill in top_skills:
        col_name = skill_to_column_name(skill)
        binary_vals = [1 if skill in s_set else 0 for s_set in skills_sets]
        df[col_name] = pd.Series(binary_vals, dtype="int64", index=df.index)

    # Persist outputs to Parquet and CSV
    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving enriched dataset to Parquet: %s", out_parquet_path)
    df.to_parquet(out_parquet_path, index=False)

    logger.info("Saving enriched dataset to CSV: %s", out_csv_path)
    df.to_csv(out_csv_path, index=False)

    return df


# Alias for backward and roadmap consensus compatibility
enrich_vacancies_dataset = enrich_dataset


def main() -> None:
    """CLI entrypoint for dataset skills enrichment."""
    parser = argparse.ArgumentParser(
        description="Enrich cleaned vacancy dataset with extracted skills and binary ML features."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_INPUT_PARQUET,
        help=f"Path to input cleaned Parquet file (default: {DEFAULT_INPUT_PARQUET})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_PARQUET,
        help=f"Path to output enriched Parquet file (default: {DEFAULT_OUTPUT_PARQUET})",
    )
    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=DEFAULT_TOP_N_FEATURES,
        help=f"Number of top skills to create binary indicator columns for (default: {DEFAULT_TOP_N_FEATURES})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n=======================================================")
    print(" JobPulse: Hard Skills Enrichment Pipeline (Step 2)   ")
    print("=======================================================")
    print(f"Input:         {args.input}")
    print(f"Output:        {args.output}")
    print(f"Top N Skills:  {args.top_n}\n")

    enriched_df = enrich_dataset(
        input_parquet=args.input,
        output_parquet=args.output,
        top_n_features=args.top_n,
    )

    total_vacancies = len(enriched_df)
    vacancies_with_skills = int((enriched_df["skills_count"] > 0).sum())
    coverage_pct = (vacancies_with_skills / total_vacancies * 100) if total_vacancies > 0 else 0.0
    mean_skills = enriched_df["skills_count"].mean() if total_vacancies > 0 else 0.0

    print("-------------------------------------------------------")
    print(f"Total Vacancies Processed:   {total_vacancies:,}")
    print(f"Vacancies with Skills:       {vacancies_with_skills:,} ({coverage_pct:.1f}%)")
    print(f"Average Skills per Vacancy:  {mean_skills:.2f}")
    print("-------------------------------------------------------")

    # Calculate and display top 20 skills
    counter: Counter[str] = Counter()
    for skills in enriched_df["extracted_skills"]:
        counter.update(skills)

    print("\nTop 20 Most Frequent Skills:")
    print(f"{'Rank':<5} {'Skill':<25} {'Vacancies':<12} {'Share (%)':<10}")
    print("-" * 52)
    for rank, (skill, count) in enumerate(counter.most_common(20), 1):
        pct = (count / total_vacancies * 100) if total_vacancies > 0 else 0.0
        print(f"{rank:<5} {skill:<25} {count:<12} {pct:.1f}%")

    out_p = Path(args.output)
    out_c = out_p.with_suffix(".csv")
    print("\nSaved files:")
    if out_p.exists():
        print(f"  Parquet: {out_p} ({out_p.stat().st_size / 1024 / 1024:.2f} MB)")
    if out_c.exists():
        print(f"  CSV:     {out_c} ({out_c.stat().st_size / 1024 / 1024:.2f} MB)")
    print("Enrichment complete successfully.\n")


if __name__ == "__main__":
    main()
