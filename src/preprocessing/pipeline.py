"""Data preprocessing and normalization pipeline for JobPulse.

Transforms raw vacancy payloads from staging SQLite storage into clean,
standardized datasets saved in Parquet and CSV analytical formats.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.experience_mapper import map_experience
from src.preprocessing.location_mapper import resolve_location
from src.preprocessing.salary_normalizer import SalaryNormalizer, parse_salary_bound
from src.preprocessing.text_cleaner import clean_html_text
from src.utils.config import settings

logger = logging.getLogger(__name__)


def normalize_iso_timestamp(ts: Any) -> str:
    """Normalize timestamp string into ISO 8601 format, or empty string if malformed."""
    if ts is None or isinstance(ts, bool):
        return ""
    ts_str = str(ts).strip()
    if not ts_str or ts_str.lower() in ("none", "null"):
        return ""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.isoformat()
    except (ValueError, TypeError):
        try:
            parsed = pd.to_datetime(ts_str)
            if pd.isna(parsed):
                return ""
            return parsed.isoformat()
        except (ValueError, TypeError, pd.errors.ParserError, OverflowError):
            return ""


def normalize_employment_type(payload: Mapping[str, Any] | None) -> str:
    """Extract and normalize employment type from vacancy payload.

    Args:
        payload: Parsed vacancy JSON dictionary.

    Returns:
        Standardized employment string (e.g. 'full', 'part', 'project', 'probation').
    """
    if not payload or not isinstance(payload, Mapping):
        return "full"

    emp_id: str | None = None

    # Check 'employment' field
    emp = payload.get("employment")
    if isinstance(emp, Mapping):
        emp_id = emp.get("id") or emp.get("name")
    elif isinstance(emp, str):
        emp_id = emp

    # Fallback to 'employment_form'
    if not emp_id:
        ef = payload.get("employment_form")
        if isinstance(ef, Mapping):
            emp_id = ef.get("id") or ef.get("name")
        elif isinstance(ef, str):
            emp_id = ef

    if not emp_id:
        return "full"

    norm = str(emp_id).lower().strip()
    if norm in ("full", "full_time", "full-time", "полная", "полная занятость"):
        return "full"
    if norm in ("part", "part_time", "part-time", "side_job", "частичная", "частичная занятость"):
        return "part"
    if norm in ("project", "temporary", "contract", "проект", "проектная работа"):
        return "project"
    if norm in ("probation", "internship", "стажировка"):
        return "probation"
    if norm in ("fly_in_fly_out", "вахта", "вахтовый метод"):
        return "fly_in_fly_out"

    return norm


def normalize_work_format(payload: Mapping[str, Any] | None) -> str:
    """Extract and normalize work format (office, hybrid, remote).

    Args:
        payload: Parsed vacancy JSON dictionary.

    Returns:
        One of 'office', 'hybrid', 'remote'.
    """
    if not payload or not isinstance(payload, Mapping):
        return "office"

    wf = payload.get("work_format")
    format_ids: set[str] = set()

    if isinstance(wf, Sequence) and not isinstance(wf, (str, bytes)):
        for item in wf:
            if isinstance(item, Mapping):
                val = item.get("id") or item.get("name")
                if val:
                    format_ids.add(str(val).upper().strip())
            elif isinstance(item, str):
                format_ids.add(item.upper().strip())
    elif isinstance(wf, Mapping):
        val = wf.get("id") or wf.get("name")
        if val:
            format_ids.add(str(val).upper().strip())
    elif isinstance(wf, str):
        format_ids.add(wf.upper().strip())

    # Check matches in format_ids
    has_remote = any("REMOTE" in fid or "УДАЛЕН" in fid for fid in format_ids)
    has_hybrid = any("HYBRID" in fid or "ГИБРИД" in fid for fid in format_ids)
    has_office = any("ON_SITE" in fid or "FIELD_WORK" in fid or "ОФИС" in fid for fid in format_ids)

    # Determine format priority
    if has_hybrid or (has_remote and has_office):
        return "hybrid"
    if has_remote:
        return "remote"
    if has_office:
        return "office"

    # Fallback to schedule
    sched = payload.get("schedule")
    sched_str = ""
    if isinstance(sched, Mapping):
        sched_str = str(sched.get("id") or sched.get("name") or "").lower().strip()
    elif isinstance(sched, str):
        sched_str = sched.lower().strip()

    if "remote" in sched_str or "удален" in sched_str:
        return "remote"
    if "hybrid" in sched_str or "гибрид" in sched_str:
        return "hybrid"

    return "office"


def extract_key_skills(payload: Mapping[str, Any] | None) -> list[str]:
    """Extract canonical key skill names from vacancy payload.

    Args:
        payload: Parsed vacancy JSON dictionary.

    Returns:
        List of distinct non-empty skill name strings.
    """
    if not payload or not isinstance(payload, Mapping):
        return []

    raw_skills = payload.get("key_skills")
    if not isinstance(raw_skills, Sequence) or isinstance(raw_skills, (str, bytes)):
        return []

    skills: list[str] = []
    seen: set[str] = set()

    for item in raw_skills:
        skill_name: str | None = None
        if isinstance(item, Mapping):
            val = item.get("name")
            if isinstance(val, str):
                skill_name = val.strip()
        elif isinstance(item, str):
            skill_name = item.strip()

        if skill_name:
            lower = skill_name.lower()
            if lower not in seen:
                seen.add(lower)
                skills.append(skill_name)

    return skills


class DataCleaningPipeline:
    """ETL Pipeline for staging to clean analytical dataset transformation."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        salary_normalizer: SalaryNormalizer | None = None,
    ) -> None:
        """Initialize pipeline with storage paths and normalizer services.

        Args:
            db_path: Path to staging SQLite database. Defaults to settings.sqlite_db_path.
            output_dir: Directory where processed files will be written.
            salary_normalizer: Custom SalaryNormalizer instance.
        """
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self.output_dir = Path(output_dir or (self.db_path.parent / "processed"))
        self.salary_normalizer = salary_normalizer or SalaryNormalizer()

    def _read_staging_records(self) -> list[dict[str, Any]]:
        """Fetch all rows from raw_vacancies SQLite staging table.

        Returns:
            List of dictionaries with vacancy_id, source, collected_at, and payload_json.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Staging database not found at {self.db_path}")

        records: list[dict[str, Any]] = []
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            # Check table existence safely
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_vacancies';"
            ).fetchone()
            if not table_check:
                logger.warning("Table 'raw_vacancies' does not exist in %s", self.db_path)
                return []

            cursor = conn.execute(
                "SELECT vacancy_id, source, payload_json, collected_at FROM raw_vacancies;"
            )
            for row in cursor:
                records.append(
                    {
                        "vacancy_id": str(row["vacancy_id"]),
                        "source": str(row["source"]),
                        "payload_json": row["payload_json"],
                        "collected_at": str(row["collected_at"]),
                    }
                )
        finally:
            conn.close()

        logger.info("Fetched %d raw vacancy records from %s", len(records), self.db_path)
        return records

    def process_record(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """Transform a single staging record into analytical schema contract.

        Args:
            record: Raw database record containing payload_json.

        Returns:
            Clean analytical dictionary, or None if payload is unparseable.
        """
        if not record or not isinstance(record, Mapping):
            return None

        payload_raw = record.get("payload_json")
        if not payload_raw:
            return None

        try:
            if isinstance(payload_raw, str):
                payload = json.loads(payload_raw)
            elif isinstance(payload_raw, Mapping):
                payload = dict(payload_raw)
            else:
                return None
        except (json.JSONDecodeError, TypeError) as err:
            logger.warning("Corrupted JSON for vacancy %s: %s", record.get("vacancy_id"), err)
            return None

        if not isinstance(payload, Mapping):
            logger.warning("Payload is not a JSON object for vacancy %s", record.get("vacancy_id"))
            return None

        vacancy_id = str(payload.get("id") or record.get("vacancy_id") or "").strip()
        source = str(record.get("source") or "hh.ru").strip()
        title = clean_html_text(str(payload.get("name") or ""))

        # Company
        employer = payload.get("employer")
        company_name = ""
        if isinstance(employer, Mapping):
            raw_cname = employer.get("name")
            if raw_cname:
                company_name = clean_html_text(str(raw_cname))

        # Location (Country & City)
        country, city = resolve_location(
            area=payload.get("area"),
            employer=employer if isinstance(employer, Mapping) else None,
            address=payload.get("address"),
        )

        # Salary
        salary = payload.get("salary")
        sal_from: float | None = None
        sal_to: float | None = None
        sal_curr: str | None = None
        sal_gross: bool | None = None

        if isinstance(salary, Mapping):
            sal_from = parse_salary_bound(salary.get("from"))
            sal_to = parse_salary_bound(salary.get("to"))
            raw_curr = salary.get("currency")
            raw_gross = salary.get("gross")

            if isinstance(raw_curr, str) and raw_curr.strip():
                sal_curr = raw_curr.strip().upper()

            if raw_gross is True:
                sal_gross = True
            elif raw_gross is False:
                sal_gross = False
            elif isinstance(raw_gross, str):
                lower_gross = raw_gross.strip().lower()
                if lower_gross in ("true", "1"):
                    sal_gross = True
                elif lower_gross in ("false", "0"):
                    sal_gross = False

            # Invert if bounds were entered backwards
            if sal_from is not None and sal_to is not None and sal_from > sal_to:
                sal_from, sal_to = sal_to, sal_from

        sal_mid_byn = self.salary_normalizer.normalize(
            salary_from=sal_from,
            salary_to=sal_to,
            currency=sal_curr,
            gross=sal_gross,
        )

        # Experience
        exp_raw, exp_level = map_experience(payload.get("experience"))

        # Employment and Work Format
        employment_type = normalize_employment_type(payload)
        work_format = normalize_work_format(payload)

        # Text cleaner
        description_clean = clean_html_text(payload.get("description"))

        # Key skills
        key_skills = extract_key_skills(payload)

        # Timestamps and URL - handle malformed dates gracefully
        published_at = normalize_iso_timestamp(payload.get("published_at"))
        if not published_at:
            published_at = normalize_iso_timestamp(record.get("collected_at"))

        url = str(payload.get("alternate_url") or f"https://hh.ru/vacancy/{vacancy_id}")

        return {
            "vacancy_id": vacancy_id,
            "source": source,
            "title": title,
            "company_name": company_name,
            "country": country,
            "city": city,
            "salary_from": sal_from,
            "salary_to": sal_to,
            "salary_currency": sal_curr,
            "salary_gross": sal_gross,
            "salary_mid_byn": sal_mid_byn,
            "experience_raw": exp_raw,
            "experience_level": exp_level,
            "employment_type": employment_type,
            "work_format": work_format,
            "description_clean": description_clean,
            "key_skills": key_skills,
            "published_at": published_at,
            "url": url,
        }

    def run(self) -> pd.DataFrame:
        """Execute complete pipeline: extract, clean, validate, and persist.

        Returns:
            Cleaned and normalized pandas DataFrame.
        """
        records = self._read_staging_records()

        cleaned_rows: list[dict[str, Any]] = []
        for r in records:
            clean_item = self.process_record(r)
            if clean_item is not None:
                cleaned_rows.append(clean_item)

        if not cleaned_rows:
            logger.warning("No valid records processed.")
            df = pd.DataFrame(
                columns=[
                    "vacancy_id",
                    "source",
                    "title",
                    "company_name",
                    "country",
                    "city",
                    "salary_from",
                    "salary_to",
                    "salary_currency",
                    "salary_gross",
                    "salary_mid_byn",
                    "experience_raw",
                    "experience_level",
                    "employment_type",
                    "work_format",
                    "description_clean",
                    "key_skills",
                    "published_at",
                    "url",
                ]
            )
        else:
            df = pd.DataFrame(cleaned_rows)

        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = self.output_dir / "vacancies_clean.parquet"
        csv_path = self.output_dir / "vacancies_clean.csv"

        logger.info("Persisting %d records to %s", len(df), parquet_path)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)

        # In CSV, key_skills is represented as JSON string for clean interoperability
        df_csv = df.copy()
        df_csv["key_skills"] = df_csv["key_skills"].apply(
            lambda k: json.dumps(k, ensure_ascii=False)
        )
        df_csv.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info("Persisting records to %s", csv_path)

        return df


def main() -> None:
    """CLI runner for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="JobPulse Preprocessing Pipeline")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to staging SQLite database (default from settings)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Path to output directory (default: data/processed)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pipeline = DataCleaningPipeline(
        db_path=args.db,
        output_dir=args.out_dir,
    )

    print("\n[START] Running JobPulse Preprocessing Pipeline...")
    df = pipeline.run()

    total_rows = len(df)
    salary_rows = df["salary_mid_byn"].notna().sum()
    pct_salary = (salary_rows / total_rows * 100) if total_rows > 0 else 0.0

    print("\n" + "=" * 60)
    print("           JOBPULSE DATA PIPELINE EXECUTION STATS")
    print("=" * 60)
    print(f"Total processed vacancies:  {total_rows}")
    print(f"Vacancies with salary:      {salary_rows} ({pct_salary:.1f}%)")

    if salary_rows > 0:
        overall_median = df["salary_mid_byn"].median()
        print(f"Overall median salary BYN:  {overall_median:.2f} BYN")
        print("\nMedian Salary (BYN) by Experience Level:")
        median_by_exp = (
            df[df["salary_mid_byn"].notna()]
            .groupby("experience_level")["salary_mid_byn"]
            .agg(["count", "median", "mean"])
            .reindex(["Junior", "Middle", "Middle+", "Senior", "Unknown"])
            .dropna(how="all")
        )
        print(median_by_exp.to_string())

    print("\nVacancies by Country:")
    print(df["country"].value_counts().to_string())

    print("\nVacancies by Work Format:")
    print(df["work_format"].value_counts().to_string())

    print("=" * 60)
    print(
        f"Output files generated:\n - {pipeline.output_dir / 'vacancies_clean.parquet'}\n - {pipeline.output_dir / 'vacancies_clean.csv'}\n"
    )


if __name__ == "__main__":
    main()
