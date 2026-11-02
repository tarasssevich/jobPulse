"""JobPulse Preprocessing and Normalization Module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.preprocessing.experience_mapper import map_experience
from src.preprocessing.location_mapper import resolve_location
from src.preprocessing.salary_normalizer import (
    DEFAULT_EXCHANGE_RATES_TO_BYN,
    TAX_SHIELD_FACTOR,
    SalaryNormalizer,
    normalize_salary,
    parse_salary_bound,
)
from src.preprocessing.text_cleaner import clean_html_text

if TYPE_CHECKING:
    from src.preprocessing.pipeline import DataCleaningPipeline

__all__ = [
    "DEFAULT_EXCHANGE_RATES_TO_BYN",
    "TAX_SHIELD_FACTOR",
    "DataCleaningPipeline",
    "SalaryNormalizer",
    "clean_html_text",
    "map_experience",
    "normalize_salary",
    "parse_salary_bound",
    "resolve_location",
]


def __getattr__(name: str) -> Any:
    if name == "DataCleaningPipeline":
        from src.preprocessing.pipeline import DataCleaningPipeline

        return DataCleaningPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
