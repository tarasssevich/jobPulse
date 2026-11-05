"""Comprehensive unit and integration tests for JobPulse hard skills module.

Covers:
- Taxonomy definitions, consistency, collision freedom, category integrity.
- Key skills extraction with case variations and punctuation.
- Special character token extraction (C++, C#, .NET, Node.js, CI/CD, Vue.js, Next.js).
- False positive rejection (Go, R, C, SOLID, REST API).
- Categorization mapping and aggregation.
- Full enrichment pipeline with dummy and edge-case datasets (missing values, NaNs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.skills.enricher import (
    enrich_dataset,
    skill_to_column_name,
)
from src.skills.extractor import SkillExtractor
from src.skills.taxonomy import (
    ALIAS_TO_CANONICAL,
    CANONICAL_SKILLS,
    SKILL_TAXONOMY,
    SKILL_TO_CATEGORY,
    SKILLS_BY_CATEGORY,
    SkillCategory,
    get_canonical_skill,
    get_skill_category,
)

# =============================================================================
# 1. Taxonomy & Integrity Tests
# =============================================================================


def test_taxonomy_size_and_canonical_count() -> None:
    """Taxonomy must contain at least 120+ canonical IT skills."""
    assert len(CANONICAL_SKILLS) >= 120
    assert len(SKILL_TAXONOMY) >= 120
    assert len(ALIAS_TO_CANONICAL) > len(CANONICAL_SKILLS)


def test_all_seven_categories_populated() -> None:
    """All 7 canonical categories must be defined and contain canonical skills."""
    expected_categories = {
        SkillCategory.LANGUAGES,
        SkillCategory.FRAMEWORKS,
        SkillCategory.DATABASES,
        SkillCategory.DEVOPS_CLOUD,
        SkillCategory.ARCHITECTURE_API,
        SkillCategory.TESTING_QA,
        SkillCategory.DATA_AI,
    }
    assert set(SkillCategory) == expected_categories
    for cat in SkillCategory:
        skills = SKILLS_BY_CATEGORY.get(cat)
        assert skills is not None
        assert len(skills) > 0, f"Category {cat} has no assigned skills"


def test_core_technologies_exist_in_taxonomy() -> None:
    """Mandatory core technologies from specifications must be present with explicit names."""
    mandatory_skills = [
        "Python",
        "Java",
        "C++",
        "C#",
        "TypeScript",
        "JavaScript",
        "Go",
        "Rust",
        "Kotlin",
        "Swift",
        "PHP",
        "SQL",
        "React",
        "Angular",
        "Vue.js",
        "Next.js",
        "Django",
        "FastAPI",
        "Flask",
        "Spring",
        ".NET",
        "Node.js",
        "PyTorch",
        "TensorFlow",
        "PostgreSQL",
        "MySQL",
        "Redis",
        "MongoDB",
        "ClickHouse",
        "Elasticsearch",
        "Kafka",
        "RabbitMQ",
        "Docker",
        "Kubernetes",
        "CI/CD",
        "Git",
        "Linux",
        "Ansible",
        "Terraform",
        "AWS",
        "Azure",
        "GCP",
        "REST API",
        "GraphQL",
        "gRPC",
        "Microservices",
        "SOLID",
        "Pytest",
        "JUnit",
        "Selenium",
        "Postman",
        "Pandas",
        "NumPy",
        "Scikit-Learn",
        "Airflow",
        "Spark",
    ]
    for skill in mandatory_skills:
        assert skill in CANONICAL_SKILLS, f"Mandatory skill '{skill}' missing from taxonomy"
        assert skill in SKILL_TO_CATEGORY
        assert SKILL_TO_CATEGORY[skill] in SkillCategory


def test_get_canonical_skill_and_category_helpers() -> None:
    """Helper functions must resolve aliases, handle case-insensitivity and None."""
    assert get_canonical_skill("postgres") == "PostgreSQL"
    assert get_canonical_skill("POSTGRESQL") == "PostgreSQL"
    assert get_canonical_skill("k8s") == "Kubernetes"
    assert get_canonical_skill("fastapi") == "FastAPI"
    assert get_canonical_skill("c++") == "C++"
    assert get_canonical_skill("c#") == "C#"
    assert get_canonical_skill(".net") == ".NET"
    assert get_canonical_skill("cicd") == "CI/CD"
    assert get_canonical_skill(None) is None
    assert get_canonical_skill("") is None
    assert get_canonical_skill("non_existent_fake_tech") is None

    assert get_skill_category("Python") == SkillCategory.LANGUAGES
    assert get_skill_category("python") == SkillCategory.LANGUAGES
    assert get_skill_category("FastAPI") == SkillCategory.FRAMEWORKS
    assert get_skill_category("PostgreSQL") == SkillCategory.DATABASES
    assert get_skill_category("Docker") == SkillCategory.DEVOPS_CLOUD
    assert get_skill_category("REST API") == SkillCategory.ARCHITECTURE_API
    assert get_skill_category("Pytest") == SkillCategory.TESTING_QA
    assert get_skill_category("Pandas") == SkillCategory.DATA_AI
    assert get_skill_category(None) is None


# =============================================================================
# 2. Key Skills Extraction Tests
# =============================================================================


def test_extract_from_key_skills_casings() -> None:
    """Extract skills from key_skills with diverse casing and alias variations."""
    extractor = SkillExtractor()
    tags = [
        "postgres",
        "POSTGRESQL",
        "k8s",
        "fastapi",
        "PYTHON",
        "Docker",
        "c++",
        "C#",
        ".net",
        "node.js",
        "ci/cd",
    ]
    extracted = extractor.extract_from_key_skills(tags)
    expected = {
        "PostgreSQL",
        "Kubernetes",
        "FastAPI",
        "Python",
        "Docker",
        "C++",
        "C#",
        ".NET",
        "Node.js",
        "CI/CD",
    }
    assert extracted == expected


def test_extract_from_key_skills_short_tokens() -> None:
    """Short tags in key_skills ('go', 'r', 'c') must map cleanly to canonical skills."""
    extractor = SkillExtractor()
    assert extractor.extract_from_key_skills(["go"]) == {"Go"}
    assert extractor.extract_from_key_skills(["golang"]) == {"Go"}
    assert extractor.extract_from_key_skills(["r"]) == {"R"}
    assert extractor.extract_from_key_skills(["rlang"]) == {"R"}
    assert extractor.extract_from_key_skills(["c"]) == {"C"}


def test_extract_from_key_skills_robustness() -> None:
    """Handle None, empty lists, numpy arrays, NaNs and corrupted payloads safely."""
    extractor = SkillExtractor()
    assert extractor.extract_from_key_skills(None) == set()
    assert extractor.extract_from_key_skills([]) == set()
    assert extractor.extract_from_key_skills(["", "   ", "nan", "None", None]) == set()
    assert extractor.extract_from_key_skills(float("nan")) == set()

    # Numpy array input
    np_tags = np.array(["Python", "PostgreSQL", None, "Docker"])
    assert extractor.extract_from_key_skills(np_tags) == {"Python", "PostgreSQL", "Docker"}


# =============================================================================
# 3. Special Characters Extraction in Text
# =============================================================================


@pytest.mark.parametrize(
    ("text", "expected_skills"),
    [
        (
            "Looking for a Senior C++ Developer with modern C++17 experience.",
            {"C++"},
        ),
        (
            "We require strong C# and .NET Core skills for backend services.",
            {"C#", ".NET"},
        ),
        (
            "Hands-on experience with Node.js and Vue.js frontends.",
            {"Node.js", "Vue.js"},
        ),
        (
            "Familiarity with Next.js, React, and TypeScript.",
            {"Next.js", "React", "TypeScript"},
        ),
        (
            "Setting up CI/CD pipelines using GitLab CI and Docker.",
            {"CI/CD", "GitLab CI", "Docker"},
        ),
        (
            "Разработка конфигураций на платформе 1С:Предприятие 8.3.",
            {"1C"},
        ),
        (
            "Building REST APIs with FastAPI and PostgreSQL.",
            {"REST API", "FastAPI", "PostgreSQL"},
        ),
    ],
)
def test_special_characters_extraction_in_text(text: str, expected_skills: set[str]) -> None:
    """Extract skills containing punctuation (+, #, ., /) safely from free text."""
    extractor = SkillExtractor()
    extracted = extractor.extract_from_text(text)
    for skill in expected_skills:
        assert skill in extracted, f"Expected '{skill}' in extracted skills for text: '{text}'"


# =============================================================================
# 4. False Positive Rejection Tests
# =============================================================================


@pytest.mark.parametrize(
    "false_text",
    [
        "We go to school and work every day.",
        "Let's go forward with the project.",
        "Go ahead and apply today.",
        "On the go solutions for busy teams.",
        "In general, r is one of many letters in the alphabet.",
        "Please note that an r. ruble exchange fee applies.",
        "We need a candidate with solid experience and a solid background in math.",
        "Take a well-deserved rest and join our friendly team.",
        "The rest of the team will assist you during onboarding.",
        "C level executives will review the proposal.",
        "Daily vitamin c supplies in the office.",
    ],
)
def test_false_positive_rejection(false_text: str) -> None:
    """Engine must not match Go, R, C, SOLID, or REST API on colloquial speech."""
    extractor = SkillExtractor()
    extracted = extractor.extract_from_text(false_text)
    ambiguous_skills = {"Go", "R", "C", "SOLID", "REST API"}
    matched_ambiguous = extracted.intersection(ambiguous_skills)
    assert not matched_ambiguous, f"False positive(s) {matched_ambiguous} detected in: '{false_text}'"


def test_legitimate_go_and_r_extraction_in_text() -> None:
    """Legitimate occurrences of Go and R with technical context must be matched."""
    extractor = SkillExtractor()

    # Go in programming context
    assert "Go" in extractor.extract_from_text("Senior Go developer needed for backend.")
    assert "Go" in extractor.extract_from_text("Разработка микросервисов на Go.")
    assert "Go" in extractor.extract_from_text("Proficiency in Golang.")
    assert "Go" in extractor.extract_from_text("Our tech stack: Python, Go, and Kafka.")
    assert "Go" in extractor.extract_from_text("Go 1.22 features understanding.")

    # R in data science context
    assert "R" in extractor.extract_from_text("Data analysis using Python and R.")
    assert "R" in extractor.extract_from_text("Experience with R programming and RStudio.")
    assert "R" in extractor.extract_from_text("Отличное знание rlang.")

    # SOLID principles context
    assert "SOLID" in extractor.extract_from_text("Thorough understanding of SOLID principles.")
    assert "SOLID" in extractor.extract_from_text("Знание SOLID, KISS, DRY.")
    assert "SOLID" in extractor.extract_from_text("Writing clean code following SOLID.")


# =============================================================================
# 5. Extract Method & Categorization Tests
# =============================================================================


def test_extract_combined_method() -> None:
    """extract() method must combine tags and text, sorting skills and organizing by category."""
    extractor = SkillExtractor()
    key_skills = ["PostgreSQL", "Docker", "k8s"]
    text = "We are seeking a Python and FastAPI backend developer with CI/CD experience."

    result = extractor.extract(key_skills=key_skills, text=text)

    assert isinstance(result, dict)
    assert "skills" in result
    assert "by_category" in result
    assert "count" in result

    expected_skills = sorted(
        ["PostgreSQL", "Docker", "Kubernetes", "Python", "FastAPI", "CI/CD"]
    )
    assert result["skills"] == expected_skills
    assert result["count"] == len(expected_skills)

    by_cat = result["by_category"]
    assert by_cat["languages"] == ["Python"]
    assert by_cat["frameworks"] == ["FastAPI"]
    assert by_cat["databases"] == ["PostgreSQL"]
    assert by_cat["devops_cloud"] == ["CI/CD", "Docker", "Kubernetes"]


def test_extract_empty_and_none_inputs() -> None:
    """extract() must return consistent empty structure when inputs are None or empty."""
    extractor = SkillExtractor()
    empty_res = extractor.extract(None, None)
    assert empty_res == {"skills": [], "by_category": {}, "count": 0}

    empty_res_str = extractor.extract([], "")
    assert empty_res_str == {"skills": [], "by_category": {}, "count": 0}


# =============================================================================
# 6. Feature Name Slugification Tests
# =============================================================================


@pytest.mark.parametrize(
    ("skill", "expected_column"),
    [
        ("Python", "skill_python"),
        ("Docker", "skill_docker"),
        ("PostgreSQL", "skill_postgresql"),
        ("C++", "skill_cpp"),
        ("C#", "skill_csharp"),
        (".NET", "skill_dotnet"),
        ("CI/CD", "skill_ci_cd"),
        ("Node.js", "skill_node_js"),
        ("Vue.js", "skill_vue_js"),
        ("Next.js", "skill_next_js"),
        ("REST API", "skill_rest_api"),
        ("1C", "skill_1c"),
        ("Scikit-Learn", "skill_scikit_learn"),
        ("Ruby on Rails", "skill_ruby_on_rails"),
        ("Microsoft SQL Server", "skill_microsoft_sql_server"),
    ],
)
def test_skill_to_column_name(skill: str, expected_column: str) -> None:
    """Canonical skill names must convert to clean ML-compliant feature column names."""
    assert skill_to_column_name(skill) == expected_column


# =============================================================================
# 7. Dataset Enrichment Pipeline Tests
# =============================================================================


def test_enrich_dataset_pipeline(tmp_path: Path) -> None:
    """Full enrichment pipeline test on dummy DataFrame verifying parquet/csv persistence and features."""
    input_parquet = tmp_path / "test_clean.parquet"
    output_parquet = tmp_path / "test_enriched.parquet"
    output_csv = tmp_path / "test_enriched.csv"

    dummy_data: dict[str, Any] = {
        "vacancy_id": ["1", "2", "3", "4"],
        "title": [
            "Senior Python Developer",
            "DevOps / SRE Engineer",
            "Fullstack JavaScript / Node.js Developer",
            "Junior Data Analyst",
        ],
        "key_skills": [
            ["Python", "PostgreSQL", "Docker", "FastAPI"],
            ["Kubernetes", "Docker", "CI/CD", "Linux"],
            ["JavaScript", "Node.js", "React", "PostgreSQL"],
            ["SQL", "Python", "Pandas"],
        ],
        "description_clean": [
            "Разработка микросервисов на Python и FastAPI, работа с PostgreSQL и Docker.",
            "Настройка Kubernetes кластеров, сопровождение CI/CD пайплайнов и Linux систем.",
            "Разработка веб-приложений на React, Node.js, интеграция с REST API.",
            "Анализ данных с использованием Python, SQL и Pandas.",
        ],
    }
    pd.DataFrame(dummy_data).to_parquet(input_parquet, index=False)

    enriched_df = enrich_dataset(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        top_n_features=5,
    )

    # File persistence verification
    assert output_parquet.exists()
    assert output_csv.exists()

    # Core columns verification
    assert "extracted_skills" in enriched_df.columns
    assert "skills_count" in enriched_df.columns
    assert "skills_by_category" in enriched_df.columns

    # Verify extracted content in row 0
    row_0_skills = enriched_df["extracted_skills"].iloc[0]
    assert "Python" in row_0_skills
    assert "FastAPI" in row_0_skills
    assert "PostgreSQL" in row_0_skills
    assert "Docker" in row_0_skills
    assert enriched_df["skills_count"].iloc[0] == len(row_0_skills)

    # Verify json-serialized category dictionary
    row_0_cat = json.loads(enriched_df["skills_by_category"].iloc[0])
    assert "languages" in row_0_cat
    assert "Python" in row_0_cat["languages"]

    # Verify binary indicator columns (top 5 skills: e.g. skill_python, skill_docker, skill_postgresql)
    for col in ["skill_python", "skill_docker", "skill_postgresql"]:
        assert col in enriched_df.columns
        assert set(enriched_df[col].unique()).issubset({0, 1})

    # Python should be 1 for row 0 and row 3, 0 for row 1
    assert enriched_df["skill_python"].iloc[0] == 1
    assert enriched_df["skill_python"].iloc[1] == 0
    assert enriched_df["skill_python"].iloc[3] == 1


def test_enrich_dataset_missing_and_corrupted_columns(tmp_path: Path) -> None:
    """Enrichment pipeline must handle missing/null columns without raising exceptions."""
    input_parquet = tmp_path / "corrupted_clean.parquet"
    output_parquet = tmp_path / "corrupted_enriched.parquet"

    corrupted_data: dict[str, Any] = {
        "vacancy_id": ["10", "20"],
        "key_skills": [None, ["Python"]],
        "description_clean": [None, "C++ developer"],
    }
    pd.DataFrame(corrupted_data).to_parquet(input_parquet, index=False)

    enriched_df = enrich_dataset(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        top_n_features=3,
    )

    assert len(enriched_df) == 2
    assert enriched_df["skills_count"].iloc[0] == 0
    assert enriched_df["extracted_skills"].iloc[0] == []
    assert set(enriched_df["extracted_skills"].iloc[1]) == {"Python", "C++"}
