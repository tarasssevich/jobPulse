"""JobPulse Hard Skills Taxonomy, Extraction & Enrichment Module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.skills.extractor import SkillExtractor
from src.skills.taxonomy import (
    ALIAS_TO_CANONICAL,
    ARCHITECTURE_API,
    CANONICAL_SKILLS,
    DATA_AI,
    DATABASES,
    DEVOPS_CLOUD,
    FRAMEWORKS,
    LANGUAGES,
    SKILL_TAXONOMY,
    SKILL_TO_CATEGORY,
    SKILLS_BY_CATEGORY,
    TESTING_QA,
    SkillCategory,
    SkillDefinition,
    get_canonical_skill,
    get_skill_category,
)

if TYPE_CHECKING:
    from src.skills.enricher import (
        DEFAULT_INPUT_PARQUET,
        DEFAULT_OUTPUT_PARQUET,
        DEFAULT_TOP_N_FEATURES,
        enrich_dataset,
        enrich_vacancies_dataset,
        skill_to_column_name,
    )

__all__ = [
    "ALIAS_TO_CANONICAL",
    "ARCHITECTURE_API",
    "CANONICAL_SKILLS",
    "DATABASES",
    "DATA_AI",
    "DEFAULT_INPUT_PARQUET",
    "DEFAULT_OUTPUT_PARQUET",
    "DEFAULT_TOP_N_FEATURES",
    "DEVOPS_CLOUD",
    "FRAMEWORKS",
    "LANGUAGES",
    "SKILLS_BY_CATEGORY",
    "SKILL_TAXONOMY",
    "SKILL_TO_CATEGORY",
    "TESTING_QA",
    "SkillCategory",
    "SkillDefinition",
    "SkillExtractor",
    "enrich_dataset",
    "enrich_vacancies_dataset",
    "get_canonical_skill",
    "get_skill_category",
    "skill_to_column_name",
]


def __getattr__(name: str) -> Any:
    if name in {
        "enrich_dataset",
        "enrich_vacancies_dataset",
        "skill_to_column_name",
        "DEFAULT_INPUT_PARQUET",
        "DEFAULT_OUTPUT_PARQUET",
        "DEFAULT_TOP_N_FEATURES",
    }:
        import src.skills.enricher as enricher_module

        return getattr(enricher_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
