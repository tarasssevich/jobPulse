"""Experience level mapping utility for IT vacancies.

Standardizes heterogeneous raw experience requirements into canonical levels:
Junior, Middle, Middle+, Senior, or Unknown.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

# Canonical experience levels according to analytical specification
STANDARD_LEVELS: Final[frozenset[str]] = frozenset(
    {"Junior", "Middle", "Middle+", "Senior", "Unknown"}
)

# Canonical level lowercase lookup
_CANONICAL_LEVELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "junior": "Junior",
        "middle": "Middle",
        "middle+": "Middle+",
        "middle plus": "Middle+",
        "middle-plus": "Middle+",
        "middleplus": "Middle+",
        "senior": "Senior",
        "unknown": "Unknown",
    }
)

# Official HeadHunter experience IDs to standard levels
_RAW_HH_MAP: dict[str, str] = {
    "noExperience": "Junior",
    "between1And3": "Middle",
    "between3And6": "Middle+",
    "moreThan6": "Senior",
}
HH_EXPERIENCE_ID_MAP: Final[Mapping[str, str]] = MappingProxyType(_RAW_HH_MAP)

# Case-insensitive HH ID map
_HH_ID_LOWER_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {k.lower(): v for k, v in _RAW_HH_MAP.items()}
)

# Heuristics for free-form text fallback
_YEARS_BETWEEN_1_3 = re.compile(r"1\s*(?:[-–—]|год[а-я]*\s*до|до)\s*3", re.IGNORECASE)
_YEARS_BETWEEN_3_6 = re.compile(r"3\s*(?:[-–—]|год[а-я]*\s*до|лет\s*до|до)\s*6", re.IGNORECASE)
_YEARS_MORE_THAN_6 = re.compile(r"(?:от|>|более)\s*6", re.IGNORECASE)

_JUNIOR_KEYWORDS = (
    "нет опыта",
    "без опыта",
    "no experience",
    "не требуется",
    "junior",
    "джун",
    "стажер",
    "стажёр",
    "intern",
    "trainee",
)
_MIDDLE_PLUS_KEYWORDS = ("middle+", "middle plus", "мидл+", "мидл плюс", "миддл+", "миддл плюс")
_MIDDLE_KEYWORDS = ("middle", "мидл", "миддл")
_SENIOR_KEYWORDS = ("senior", "сеньор", "синьор", "lead", "лид", "principal")


def _classify_from_text(text: str) -> str:
    """Classify free-form text into standard experience level."""
    lower = text.lower().strip()
    if not lower:
        return "Unknown"

    # Direct canonical level match
    if lower in _CANONICAL_LEVELS:
        return _CANONICAL_LEVELS[lower]

    # Direct HeadHunter ID match (case-insensitive)
    if lower in _HH_ID_LOWER_MAP:
        return _HH_ID_LOWER_MAP[lower]

    # Keyword check: Middle+ must be checked before Middle
    if any(kw in lower for kw in _MIDDLE_PLUS_KEYWORDS) or _YEARS_BETWEEN_3_6.search(lower):
        return "Middle+"

    if any(kw in lower for kw in _MIDDLE_KEYWORDS) or _YEARS_BETWEEN_1_3.search(lower):
        return "Middle"

    if any(kw in lower for kw in _SENIOR_KEYWORDS) or _YEARS_MORE_THAN_6.search(lower):
        return "Senior"

    if any(kw in lower for kw in _JUNIOR_KEYWORDS):
        return "Junior"

    return "Unknown"


def map_experience(exp_data: Mapping[str, Any] | str | None) -> tuple[str, str]:
    """Map raw vacancy experience data into a standardized tuple (raw_str, standard_level).

    Args:
        exp_data: Raw experience information from API payload. Can be:
            - Dictionary: e.g. {"id": "between1And3", "name": "От 1 года до 3 лет"}
            - String: e.g. "between1And3" or "От 3 до 6 лет"
            - None or invalid type

    Returns:
        tuple[str, str]:
            - raw_str: original human-readable or identifier string (empty string if missing)
            - standard_level: one of 'Junior', 'Middle', 'Middle+', 'Senior', 'Unknown'
    """
    if exp_data is None or isinstance(exp_data, bool):
        return ("", "Unknown")

    if isinstance(exp_data, Mapping):
        raw_id = exp_data.get("id")
        raw_name = exp_data.get("name")

        exp_id = str(raw_id).strip() if raw_id is not None else ""
        exp_name = str(raw_name).strip() if raw_name is not None else ""

        # Determine raw string representation
        raw_str = exp_name if exp_name else exp_id
        if not raw_str or raw_str.lower() in ("none", "null"):
            return ("", "Unknown")

        # First check official HH experience ID
        if exp_id and exp_id in HH_EXPERIENCE_ID_MAP:
            return (raw_str, HH_EXPERIENCE_ID_MAP[exp_id])
        if exp_id and exp_id.lower() in _HH_ID_LOWER_MAP:
            return (raw_str, _HH_ID_LOWER_MAP[exp_id.lower()])

        # Fallback to textual parsing of name or id
        classified = _classify_from_text(raw_str)
        return (raw_str, classified)

    if isinstance(exp_data, str):
        raw_str = exp_data.strip()
        if not raw_str or raw_str.lower() in ("none", "null"):
            return ("", "Unknown")

        # Check if string matches known HH experience ID
        if raw_str in HH_EXPERIENCE_ID_MAP:
            return (raw_str, HH_EXPERIENCE_ID_MAP[raw_str])
        if raw_str.lower() in _HH_ID_LOWER_MAP:
            return (raw_str, _HH_ID_LOWER_MAP[raw_str.lower()])

        # Heuristic classification
        return (raw_str, _classify_from_text(raw_str))

    return (str(exp_data), "Unknown")
