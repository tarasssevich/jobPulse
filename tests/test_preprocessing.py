"""Comprehensive unit tests for preprocessing and normalization pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.preprocessing.experience_mapper import map_experience
from src.preprocessing.location_mapper import resolve_location
from src.preprocessing.pipeline import (
    DataCleaningPipeline,
    extract_key_skills,
    normalize_employment_type,
    normalize_iso_timestamp,
    normalize_work_format,
)
from src.preprocessing.salary_normalizer import (
    SalaryNormalizer,
    normalize_salary,
    parse_salary_bound,
)
from src.preprocessing.text_cleaner import clean_html_text

# ============================================================================
# 1. Salary Normalizer Tests
# ============================================================================


class TestSalaryNormalizer:
    """Tests for SalaryNormalizer domain service."""

    @pytest.fixture
    def normalizer(self) -> SalaryNormalizer:
        return SalaryNormalizer()

    def test_midpoint_calculation(self, normalizer: SalaryNormalizer) -> None:
        """Test midpoint when both bounds, only from, or only to are present."""
        # Both bounds
        assert normalizer.normalize(2000, 4000, "BYN") == 3000.0
        # Only from
        assert normalizer.normalize(3500, None, "BYN") == 3500.0
        # Only to
        assert normalizer.normalize(None, 5000, "BYN") == 5000.0
        # Both None
        assert normalizer.normalize(None, None, "BYN") is None

    def test_currency_conversion(self, normalizer: SalaryNormalizer) -> None:
        """Test conversion rates to BYN baseline."""
        # BYN (1.0)
        assert normalizer.normalize(1000, 1000, "BYN") == 1000.0
        # BYR (legacy HH code for BYN) (1.0)
        assert normalizer.normalize(1000, 1000, "BYR") == 1000.0
        # RUR & RUB (0.034)
        assert normalizer.normalize(100_000, 100_000, "RUR") == 3400.0
        assert normalizer.normalize(100_000, 100_000, "RUB") == 3400.0
        # USD (3.30)
        assert normalizer.normalize(2000, 2000, "USD") == 6600.0
        # EUR (3.55)
        assert normalizer.normalize(2000, 2000, "EUR") == 7100.0
        # KZT (0.0068)
        assert normalizer.normalize(500_000, 500_000, "KZT") == 3400.0

    def test_currency_case_and_whitespace(self, normalizer: SalaryNormalizer) -> None:
        """Test currency string normalization (case, surrounding spaces)."""
        assert normalizer.normalize(1000, 1000, "byn") == 1000.0
        assert normalizer.normalize(1000, 1000, " Usd ") == 3300.0
        assert normalizer.normalize(100_000, 100_000, "rur\t") == 3400.0

    def test_unknown_or_missing_currency(self, normalizer: SalaryNormalizer) -> None:
        """Unknown or missing currency should return None."""
        assert normalizer.normalize(1000, 2000, "GBP") is None
        assert normalizer.normalize(1000, 2000, "") is None
        assert normalizer.normalize(1000, 2000, None) is None
        assert normalizer.normalize(1000, 2000, "INVALID") is None

    def test_gross_net_conversion(self, normalizer: SalaryNormalizer) -> None:
        """Gross=True should apply 13% tax shield (0.87 factor)."""
        # Gross True: 1000 * 0.87 = 870.0
        assert normalizer.normalize(1000, 1000, "BYN", gross=True) == 870.0
        # Gross False: 1000 (net as is)
        assert normalizer.normalize(1000, 1000, "BYN", gross=False) == 1000.0
        # Gross None: 1000 (defaults to net)
        assert normalizer.normalize(1000, 1000, "BYN", gross=None) == 1000.0

    def test_gross_with_foreign_currency(self, normalizer: SalaryNormalizer) -> None:
        """Gross conversion combined with foreign currency rate."""
        # 100_000 RUB gross -> 100_000 * 0.87 = 87_000 net -> 87_000 * 0.034 = 2958.0 BYN
        assert normalizer.normalize(100_000, 100_000, "RUB", gross=True) == 2958.0

    def test_normalize_dict(self, normalizer: SalaryNormalizer) -> None:
        """Test dictionary convenience method."""
        payload = {"from": 2000, "to": 3000, "currency": "BYN", "gross": False}
        assert normalizer.normalize_dict(payload) == 2500.0

        empty_or_none: dict[str, Any] = {}
        assert normalizer.normalize_dict(empty_or_none) is None
        assert normalizer.normalize_dict(None) is None

    def test_standalone_normalize_salary_function(self) -> None:
        """Test pure function wrapper."""
        res = normalize_salary(salary_from=1000, salary_to=2000, currency="USD", gross=False)
        assert res == 4950.0

    def test_custom_rates_and_shield(self) -> None:
        """Custom exchange rates and tax shield factor support."""
        custom_rates = {"XYZ": 2.5}
        norm = SalaryNormalizer(exchange_rates=custom_rates, tax_shield_factor=0.80)
        # 1000 XYZ gross -> 800 net -> 800 * 2.5 = 2000.0
        assert norm.normalize(1000, 1000, "XYZ", gross=True) == 2000.0

    def test_invalid_and_edge_inputs(self, normalizer: SalaryNormalizer) -> None:
        """Malformed bounds should be ignored safely."""
        # Negative bound ignored
        assert normalizer.normalize(-500, 2000, "BYN") == 2000.0
        # String numeric bounds coerced
        assert normalizer.normalize("1500", "2500", "BYN") == 2000.0
        # Non-numeric bound handled
        assert normalizer.normalize("invalid", 3000, "BYN") == 3000.0
        assert normalizer.normalize("abc", "xyz", "BYN") is None

    def test_zero_and_negative_salaries(self, normalizer: SalaryNormalizer) -> None:
        """Zero and negative salaries must be treated as missing."""
        # Arrange / Act / Assert
        assert normalizer.normalize(0, 0, "BYN") is None
        assert normalizer.normalize(0, None, "BYN") is None
        assert normalizer.normalize(-100, -200, "BYN") is None
        assert normalizer.normalize(0, 3000, "BYN") == 3000.0

    def test_inverted_bounds(self, normalizer: SalaryNormalizer) -> None:
        """Inverted bounds (from > to) should be swapped automatically."""
        # Arrange / Act / Assert
        assert normalizer.normalize(5000, 2000, "BYN") == 3500.0

    def test_formatted_string_bounds(self, normalizer: SalaryNormalizer) -> None:
        """Numeric strings with thousands separators and commas."""
        # Arrange / Act / Assert
        assert normalizer.normalize("2 500", "3 500", "BYN") == 3000.0
        assert normalizer.normalize("2,500.50", "2,500.50", "BYN") == 2500.5

    def test_non_finite_and_boolean_bounds(self, normalizer: SalaryNormalizer) -> None:
        """Infinity, NaN, and boolean bounds rejected."""
        # Arrange / Act / Assert
        assert normalizer.normalize(float("inf"), 2000, "BYN") == 2000.0
        assert normalizer.normalize(float("inf"), float("inf"), "BYN") is None
        assert normalizer.normalize(float("nan"), 2000, "BYN") == 2000.0
        assert normalizer.normalize(True, 2000, "BYN") == 2000.0
        assert normalizer.normalize(False, False, "BYN") is None

    def test_gross_flag_representations(self, normalizer: SalaryNormalizer) -> None:
        """Gross flag accepted as string or integer."""
        # Arrange / Act / Assert
        assert normalizer.normalize(1000, 1000, "BYN", gross="true") == 870.0
        assert normalizer.normalize(1000, 1000, "BYN", gross=1) == 870.0
        assert normalizer.normalize(1000, 1000, "BYN", gross="false") == 1000.0
        assert normalizer.normalize(1000, 1000, "BYN", gross=0) == 1000.0

    def test_tax_shield_validation(self) -> None:
        """Tax shield factor must be in (0.0, 1.0]."""
        with pytest.raises(ValueError):
            SalaryNormalizer(tax_shield_factor=0.0)
        with pytest.raises(ValueError):
            SalaryNormalizer(tax_shield_factor=1.5)

    def test_parse_salary_bound_standalone(self) -> None:
        """Test parse_salary_bound helper directly."""
        assert parse_salary_bound(" 3 000 ") == 3000.0
        assert parse_salary_bound(0) is None
        assert parse_salary_bound(-50) is None
        assert parse_salary_bound(float("inf")) is None
        assert parse_salary_bound(True) is None


# ============================================================================
# 2. Text Cleaner Tests
# ============================================================================


class TestTextCleaner:
    """Tests for clean_html_text function."""

    def test_none_and_empty_inputs(self) -> None:
        """None, non-string, or empty strings return empty string."""
        assert clean_html_text(None) == ""
        assert clean_html_text("") == ""
        assert clean_html_text("   \n\t  ") == ""
        assert clean_html_text(123) == ""  # type: ignore[arg-type]

    def test_strips_html_tags(self) -> None:
        """All HTML tags stripped cleanly with space separator."""
        html_sample = "<p>Python <strong>developer</strong> with <em>FastAPI</em> experience.</p>"
        expected = "Python developer with FastAPI experience."
        assert clean_html_text(html_sample) == expected

    def test_decodes_html_entities(self) -> None:
        """Standard and numeric HTML entities unescaped."""
        html_sample = "Requirements &amp; skills: Python &gt; 3.12 &quot;JobPulse&quot; &#39;test&#39;&nbsp;ok"
        expected = "Requirements & skills: Python > 3.12 \"JobPulse\" 'test' ok"
        assert clean_html_text(html_sample) == expected

    def test_collapses_whitespace_and_newlines(self) -> None:
        """Multiple newlines and tabs collapsed into a single space."""
        raw = "Line 1\n\n\n<div>Line 2</div>\t\t\t<span>Line 3</span>"
        expected = "Line 1 Line 2 Line 3"
        assert clean_html_text(raw) == expected

    def test_nested_and_malformed_html(self) -> None:
        """Nested, self-closing, and unclosed tags handled gracefully."""
        raw = "<div><ul><li>Item 1</li><br/><li>Item 2</li></ul><p>Unclosed paragraph"
        expected = "Item 1 Item 2 Unclosed paragraph"
        assert clean_html_text(raw) == expected

    def test_mathematical_entities_not_accidentally_stripped(self) -> None:
        """Entities representing math operators like &lt; and &gt; should not form fake tags."""
        # Arrange
        raw = "<p>Python &gt;= 3.10 &amp; Salary &lt; 5000</p>"
        # Act
        res = clean_html_text(raw)
        # Assert
        assert res == "Python >= 3.10 & Salary < 5000"

    def test_null_bytes_and_empty_tags(self) -> None:
        """Null bytes sanitized and empty HTML tags return empty string."""
        assert clean_html_text("Python\x00Developer") == "PythonDeveloper"
        assert clean_html_text("<p>  <br/>  <div></div> </p>") == ""


# ============================================================================
# 3. Experience Mapper Tests
# ============================================================================


class TestExperienceMapper:
    """Tests for map_experience function."""

    def test_hh_standard_ids_from_dict(self) -> None:
        """Official HeadHunter experience IDs inside dictionary."""
        assert map_experience({"id": "noExperience", "name": "Нет опыта"}) == (
            "Нет опыта",
            "Junior",
        )
        assert map_experience({"id": "between1And3", "name": "От 1 года до 3 лет"}) == (
            "От 1 года до 3 лет",
            "Middle",
        )
        assert map_experience({"id": "between3And6", "name": "От 3 до 6 лет"}) == (
            "От 3 до 6 лет",
            "Middle+",
        )
        assert map_experience({"id": "moreThan6", "name": "Более 6 лет"}) == (
            "Более 6 лет",
            "Senior",
        )

    def test_dict_without_name(self) -> None:
        """Dict with only ID."""
        assert map_experience({"id": "between1And3"}) == ("between1And3", "Middle")

    def test_string_inputs_by_id(self) -> None:
        """Direct ID passed as string."""
        assert map_experience("noExperience") == ("noExperience", "Junior")
        assert map_experience("between1And3") == ("between1And3", "Middle")
        assert map_experience("between3And6") == ("between3And6", "Middle+")
        assert map_experience("moreThan6") == ("moreThan6", "Senior")

    def test_string_inputs_by_russian_text(self) -> None:
        """Russian natural text classifications."""
        assert map_experience("Требуется опыт от 1 года до 3 лет")[1] == "Middle"
        assert map_experience("Опыт работы 3-6 лет")[1] == "Middle+"
        assert map_experience("Более 6 лет в разработке")[1] == "Senior"
        assert map_experience("Без опыта работы")[1] == "Junior"

    def test_none_and_unknown_inputs(self) -> None:
        """Missing or unrecognized experience returns Unknown."""
        assert map_experience(None) == ("", "Unknown")
        assert map_experience("") == ("", "Unknown")
        assert map_experience("Random non-matching text") == ("Random non-matching text", "Unknown")
        assert map_experience({"id": "weird_id", "name": "Непонятный"}) == ("Непонятный", "Unknown")

    def test_direct_canonical_level_names(self) -> None:
        """Passing standard level name directly maps correctly."""
        assert map_experience("Middle") == ("Middle", "Middle")
        assert map_experience("Middle+") == ("Middle+", "Middle+")
        assert map_experience("Junior") == ("Junior", "Junior")
        assert map_experience("Senior") == ("Senior", "Senior")
        assert map_experience("Unknown") == ("Unknown", "Unknown")

    def test_specialized_it_roles_and_keywords(self) -> None:
        """Intern, trainee, lead, and russian keyword classification."""
        assert map_experience("Intern Software Engineer") == ("Intern Software Engineer", "Junior")
        assert map_experience("Стажер-разработчик") == ("Стажер-разработчик", "Junior")
        assert map_experience("Team Lead Backend") == ("Team Lead Backend", "Senior")
        assert map_experience("Tech Lead") == ("Tech Lead", "Senior")
        assert map_experience("Middle+ Python Dev") == ("Middle+ Python Dev", "Middle+")

    def test_mapping_with_null_and_boolean(self) -> None:
        """Null values inside mapping or boolean types should yield Unknown."""
        assert map_experience({"id": None, "name": None}) == ("", "Unknown")
        assert map_experience(True) == ("", "Unknown")
        assert map_experience(False) == ("", "Unknown")


# ============================================================================
# 4. Location Mapper Tests
# ============================================================================


class TestLocationMapper:
    """Tests for resolve_location function."""

    def test_known_cis_area_ids(self) -> None:
        """Resolves known area IDs from staging database."""
        # Minsk (1002) -> Belarus
        country, city = resolve_location(area={"id": "1002", "name": "Минск"})
        assert country == "Беларусь"
        assert city == "Минск"

        # Moscow (1) -> Russia
        country, city = resolve_location(area={"id": "1", "name": "Москва"})
        assert country == "Россия"
        assert city == "Москва"

        # Almaty (159) -> Kazakhstan
        country, city = resolve_location(area={"id": "159", "name": "Алматы"})
        assert country == "Казахстан"
        assert city == "Алматы"

    def test_employer_country_fallback(self) -> None:
        """Fallback to employer country_id when area is unknown."""
        country, city = resolve_location(
            area={"id": "999999", "name": "CustomCity"},
            employer={"country_id": 4, "name": "TestCorp"},
        )
        assert country == "Беларусь"
        assert city == "CustomCity"

    def test_city_fallback_from_address(self) -> None:
        """City resolved from address if area name is empty."""
        country, city = resolve_location(
            area={"id": "1002", "name": ""},
            address={"city": "Минск"},
        )
        assert country == "Беларусь"
        assert city == "Минск"

    def test_completely_unknown_location(self) -> None:
        """Missing area and employer returns Unknown."""
        country, city = resolve_location(None, None)
        assert country == "Unknown"
        assert city == ""

    def test_null_and_sentinel_string_values(self) -> None:
        """Area or employer with None or sentinel 'None'/'null' strings."""
        country, city = resolve_location(area={"id": None, "name": None})
        assert country == "Unknown"
        assert city == ""

        country2, city2 = resolve_location(area={"id": "None", "name": "null"})
        assert country2 == "Unknown"
        assert city2 == ""

    def test_employer_country_variants(self) -> None:
        """Employer country specified as float ID or nested dict."""
        # Float country_id
        country, city = resolve_location(
            area={"id": "9999", "name": "CustomCity"},
            employer={"country_id": 4.0},
        )
        assert country == "Беларусь"
        assert city == "CustomCity"

        # Nested country dict in employer
        country2, _ = resolve_location(
            area={"id": "9999", "name": "CustomCity"},
            employer={"country": {"name": "Казахстан"}},
        )
        assert country2 == "Казахстан"

        # Nested country in area
        country3, _ = resolve_location(
            area={"id": "9999", "name": "CustomCity", "country": "Россия"},
        )
        assert country3 == "Россия"


# ============================================================================
# 5. Pipeline Component & Extraction Tests
# ============================================================================


class TestPipelineHelpers:
    """Tests for helper extraction functions in pipeline."""

    def test_normalize_employment_type(self) -> None:
        """Test employment normalization."""
        assert normalize_employment_type({"employment": {"id": "full"}}) == "full"
        assert normalize_employment_type({"employment_form": {"id": "FULL"}}) == "full"
        assert normalize_employment_type({"employment": {"id": "part"}}) == "part"
        assert normalize_employment_type({"employment_form": {"id": "PROJECT"}}) == "project"
        assert normalize_employment_type({}) == "full"  # default fallback

    def test_normalize_work_format(self) -> None:
        """Test work format normalization."""
        # Single formats
        assert normalize_work_format({"work_format": [{"id": "REMOTE"}]}) == "remote"
        assert normalize_work_format({"work_format": [{"id": "ON_SITE"}]}) == "office"
        assert normalize_work_format({"work_format": [{"id": "HYBRID"}]}) == "hybrid"

        # Multi-format combinations
        assert (
            normalize_work_format({"work_format": [{"id": "REMOTE"}, {"id": "ON_SITE"}]})
            == "hybrid"
        )
        assert (
            normalize_work_format({"work_format": [{"id": "REMOTE"}, {"id": "HYBRID"}]}) == "hybrid"
        )

        # Schedule fallback
        assert normalize_work_format({"schedule": {"id": "remote"}}) == "remote"
        assert normalize_work_format({"schedule": {"id": "fullDay"}}) == "office"

    def test_extract_key_skills(self) -> None:
        """Test skill extraction and deduplication."""
        payload = {
            "key_skills": [
                {"name": "Python"},
                {"name": "FastAPI"},
                {"name": "Python"},  # duplicate
                {"name": "  Docker  "},
                {"name": ""},  # empty
            ]
        }
        assert extract_key_skills(payload) == ["Python", "FastAPI", "Docker"]
        assert extract_key_skills({}) == []

    def test_normalize_iso_timestamp(self) -> None:
        """ISO 8601 parsing and malformed date fallback."""
        assert normalize_iso_timestamp("2026-10-25T12:00:00+0300") == "2026-10-25T12:00:00+03:00"
        assert normalize_iso_timestamp("2026-10-25T12:00:00Z") == "2026-10-25T12:00:00+00:00"
        assert normalize_iso_timestamp("malformed_date") == ""
        assert normalize_iso_timestamp(None) == ""
        assert normalize_iso_timestamp(True) == ""

    def test_extract_key_skills_case_insensitive_dedup(self) -> None:
        """Case-insensitive skill deduplication preserving first casing."""
        payload = {
            "key_skills": [
                {"name": "Python"},
                {"name": "python"},
                {"name": "PYTHON"},
                {"name": "Docker"},
            ]
        }
        assert extract_key_skills(payload) == ["Python", "Docker"]

    def test_pipeline_helpers_none_guards(self) -> None:
        """None or non-dict inputs to pipeline helpers must not crash."""
        assert normalize_employment_type(None) == "full"
        assert normalize_employment_type("not-a-dict") == "full"  # type: ignore[arg-type]
        assert normalize_work_format(None) == "office"
        assert normalize_work_format(123) == "office"  # type: ignore[arg-type]
        assert extract_key_skills(None) == []


# ============================================================================
# 6. End-to-End DataCleaningPipeline Tests
# ============================================================================


@pytest.fixture
def sample_staging_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with diverse sample vacancies."""
    db_path = tmp_path / "test_staging.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE raw_vacancies (
            vacancy_id TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'hh.ru',
            payload_json TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'raw'
        );
        """
    )

    vacancies = [
        # 1. Full BYN vacancy
        {
            "id": "1001",
            "name": "Backend Python Developer",
            "employer": {"name": "Tech Corp", "country_id": 4},
            "area": {"id": "1002", "name": "Минск"},
            "salary": {"from": 3000, "to": 5000, "currency": "BYN", "gross": False},
            "experience": {"id": "between3And6", "name": "От 3 до 6 лет"},
            "employment_form": {"id": "FULL"},
            "work_format": [{"id": "REMOTE"}],
            "description": "<p>FastAPI &amp; <b>Docker</b></p>",
            "key_skills": [{"name": "Python"}, {"name": "FastAPI"}],
            "published_at": "2026-10-25T12:00:00+0300",
            "alternate_url": "https://hh.ru/vacancy/1001",
        },
        # 2. Russian RUB vacancy with gross salary
        {
            "id": "1002",
            "name": "Data Scientist",
            "employer": {"name": "Yandex", "country_id": 1},
            "area": {"id": "1", "name": "Москва"},
            "salary": {"from": 200_000, "to": 300_000, "currency": "RUR", "gross": True},
            "experience": {"id": "between1And3", "name": "От 1 года до 3 лет"},
            "employment_form": {"id": "FULL"},
            "work_format": [{"id": "HYBRID"}],
            "description": "<div>Machine Learning &amp; PyTorch</div>",
            "key_skills": [{"name": "Python"}, {"name": "PyTorch"}],
            "published_at": "2026-10-26T14:30:00+0300",
            "alternate_url": "https://hh.ru/vacancy/1002",
        },
        # 3. Vacancy without salary
        {
            "id": "1003",
            "name": "Junior QA Engineer",
            "employer": {"name": "QA Lab", "country_id": 4},
            "area": {"id": "1003", "name": "Гомель"},
            "salary": None,
            "experience": {"id": "noExperience", "name": "Нет опыта"},
            "employment_form": {"id": "PART"},
            "work_format": [{"id": "ON_SITE"}],
            "description": "Testing mobile apps",
            "key_skills": [{"name": "Postman"}],
            "published_at": "2026-10-27T09:00:00+0300",
            "alternate_url": "https://hh.ru/vacancy/1003",
        },
        # 4. Vacancy with corrupted JSON
        "CORRUPTED_JSON_LINE",
    ]

    for i, v in enumerate(vacancies):
        vid = f"100{i + 1}"
        payload_str = json.dumps(v) if isinstance(v, dict) else v
        conn.execute(
            """
            INSERT INTO raw_vacancies (vacancy_id, source, payload_json, collected_at, status)
            VALUES (?, 'hh.ru', ?, '2026-10-29T10:00:00', 'raw');
            """,
            (vid, payload_str),
        )

    conn.commit()
    conn.close()
    return db_path


class TestDataCleaningPipeline:
    """Integration tests for DataCleaningPipeline."""

    def test_pipeline_execution(self, sample_staging_db: Path, tmp_path: Path) -> None:
        """Pipeline should read staging DB, clean records, and export files."""
        output_dir = tmp_path / "processed"
        pipeline = DataCleaningPipeline(
            db_path=sample_staging_db,
            output_dir=output_dir,
        )

        df = pipeline.run()

        # 4 total records in db, 1 corrupted -> 3 valid cleaned records
        assert len(df) == 3
        assert isinstance(df, pd.DataFrame)

        # Expected columns check
        expected_cols = [
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
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column: {col}"

        # Verify values of row 1 (BYN)
        row1 = df[df["vacancy_id"] == "1001"].iloc[0]
        assert row1["title"] == "Backend Python Developer"
        assert row1["country"] == "Беларусь"
        assert row1["city"] == "Минск"
        assert row1["salary_mid_byn"] == 4000.0  # (3000 + 5000) / 2
        assert row1["experience_level"] == "Middle+"
        assert row1["work_format"] == "remote"
        assert row1["description_clean"] == "FastAPI & Docker"
        assert list(row1["key_skills"]) == ["Python", "FastAPI"]

        # Verify values of row 2 (RUB gross)
        row2 = df[df["vacancy_id"] == "1002"].iloc[0]
        assert row2["country"] == "Россия"
        assert row2["experience_level"] == "Middle"
        assert row2["work_format"] == "hybrid"
        # Midpoint = 250_000 * 0.87 = 217_500 net -> * 0.034 = 7395.0 BYN
        assert row2["salary_mid_byn"] == 7395.0

        # Verify values of row 3 (No salary)
        row3 = df[df["vacancy_id"] == "1003"].iloc[0]
        assert pd.isna(row3["salary_mid_byn"])
        assert row3["experience_level"] == "Junior"
        assert row3["work_format"] == "office"
        assert row3["employment_type"] == "part"

        # Verify parquet export
        parquet_path = output_dir / "vacancies_clean.parquet"
        assert parquet_path.exists()
        df_parquet = pd.read_parquet(parquet_path)
        assert len(df_parquet) == 3

        # Verify csv export
        csv_path = output_dir / "vacancies_clean.csv"
        assert csv_path.exists()
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 3

    def test_nonexistent_database(self, tmp_path: Path) -> None:
        """FileNotFoundError raised if DB does not exist."""
        pipeline = DataCleaningPipeline(db_path=tmp_path / "nonexistent.db")
        with pytest.raises(FileNotFoundError):
            pipeline.run()

    def test_pipeline_edge_cases_and_non_dict_json(self, tmp_path: Path) -> None:
        """Pipeline handles inverted bounds, zero salaries, and non-object JSON payloads."""
        # Arrange
        db_path = tmp_path / "edge_cases.db"
        output_dir = tmp_path / "edge_output"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE raw_vacancies (
                    vacancy_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL DEFAULT 'hh.ru',
                    payload_json TEXT NOT NULL,
                    collected_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'raw'
                );
                """
            )
            records = [
                # 1. Inverted salary bounds
                (
                    "2001",
                    json.dumps(
                        {
                            "id": "2001",
                            "name": "Senior Go Dev",
                            "salary": {"from": 6000, "to": 3000, "currency": "BYN"},
                            "experience": "moreThan6",
                            "published_at": "invalid-date-string",
                        }
                    ),
                    "2026-10-29T10:00:00",
                ),
                # 2. Zero salary bounds
                (
                    "2002",
                    json.dumps(
                        {
                            "id": "2002",
                            "name": "Intern React",
                            "salary": {"from": 0, "to": 0, "currency": "BYN"},
                            "experience": "noExperience",
                        }
                    ),
                    "2026-10-29T10:00:00",
                ),
                # 3. Negative lower bound
                (
                    "2003",
                    json.dumps(
                        {
                            "id": "2003",
                            "name": "Middle QA",
                            "salary": {"from": -500, "to": 3500, "currency": "BYN"},
                            "experience": "between1And3",
                        }
                    ),
                    "2026-10-29T10:00:00",
                ),
                # 4-7. Non-object valid JSONs that should be skipped safely
                ("2004", "12345", "2026-10-29T10:00:00"),
                ("2005", '"just a string"', "2026-10-29T10:00:00"),
                ("2006", "[1, 2, 3]", "2026-10-29T10:00:00"),
                ("2007", "null", "2026-10-29T10:00:00"),
            ]
            for vid, payload, collected in records:
                conn.execute(
                    "INSERT INTO raw_vacancies VALUES (?, 'hh.ru', ?, ?, 'raw');",
                    (vid, payload, collected),
                )
            conn.commit()
        finally:
            conn.close()

        # Act
        pipeline = DataCleaningPipeline(db_path=db_path, output_dir=output_dir)
        df = pipeline.run()

        # Assert
        assert len(df) == 3

        # Inverted bounds swapped
        row1 = df[df["vacancy_id"] == "2001"].iloc[0]
        assert row1["salary_from"] == 3000.0
        assert row1["salary_to"] == 6000.0
        assert row1["salary_mid_byn"] == 4500.0
        # Malformed published_at fell back to collected_at
        assert row1["published_at"] == "2026-10-29T10:00:00"

        # Zero salary treated as None / NaN in pandas
        row2 = df[df["vacancy_id"] == "2002"].iloc[0]
        assert pd.isna(row2["salary_from"])
        assert pd.isna(row2["salary_to"])
        assert pd.isna(row2["salary_mid_byn"])

        # Negative lower bound sanitized to None / NaN in pandas
        row3 = df[df["vacancy_id"] == "2003"].iloc[0]
        assert pd.isna(row3["salary_from"])
        assert row3["salary_to"] == 3500.0
        assert row3["salary_mid_byn"] == 3500.0

    def test_pipeline_empty_db_or_missing_table(self, tmp_path: Path) -> None:
        """Database without raw_vacancies table returns empty DataFrame cleanly."""
        # Arrange
        db_path = tmp_path / "empty_schema.db"
        conn = sqlite3.connect(db_path)
        conn.close()

        # Act
        pipeline = DataCleaningPipeline(db_path=db_path, output_dir=tmp_path / "empty_out")
        df = pipeline.run()

        # Assert
        assert len(df) == 0
        assert "vacancy_id" in df.columns
        assert "salary_mid_byn" in df.columns
