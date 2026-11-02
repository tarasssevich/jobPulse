"""Location (country and city) resolution service for CIS vacancies."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

# Default fallback mapping for employer country_id in HeadHunter
_RAW_EMPLOYER_COUNTRY_ID_MAP: dict[int, str] = {
    1: "Россия",
    3: "Казахстан",
    4: "Беларусь",
    5: "Украина",
    6: "Узбекистан",
    7: "Кыргызстан",
    8: "Беларусь",
    9: "Грузия",
    16: "Беларусь",
    28: "Армения",
    40: "Казахстан",
    113: "Россия",
}
EMPLOYER_COUNTRY_ID_MAP: Final[Mapping[int, str]] = MappingProxyType(_RAW_EMPLOYER_COUNTRY_ID_MAP)

# Top-level country area IDs
_RAW_COUNTRY_AREA_IDS: dict[str, str] = {
    "16": "Беларусь",
    "113": "Россия",
    "40": "Казахстан",
    "5": "Украина",
    "6": "Узбекистан",
    "7": "Кыргызстан",
    "9": "Грузия",
    "28": "Армения",
}
COUNTRY_AREA_IDS: Final[Mapping[str, str]] = MappingProxyType(_RAW_COUNTRY_AREA_IDS)

_CURRENT_DIR = Path(__file__).resolve().parent
_MAPPING_FILE = _CURRENT_DIR / "area_to_country.json"

# Load local area mapping if present
_RAW_CACHED_AREA_TO_COUNTRY: dict[str, str] = {}
if _MAPPING_FILE.exists():
    try:
        with open(_MAPPING_FILE, encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                _RAW_CACHED_AREA_TO_COUNTRY = {str(k): str(v) for k, v in loaded.items()}
    except (OSError, json.JSONDecodeError):
        _RAW_CACHED_AREA_TO_COUNTRY = {}

_CACHED_AREA_TO_COUNTRY: Final[Mapping[str, str]] = MappingProxyType(_RAW_CACHED_AREA_TO_COUNTRY)


def _extract_clean_str(mapping: Mapping[str, Any] | None, key: str) -> str:
    """Safely extract and clean string from mapping, handling None and null values."""
    if not mapping or not isinstance(mapping, Mapping):
        return ""
    val = mapping.get(key)
    if val is None or isinstance(val, (bool, list, dict)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("none", "null") else s


def _extract_country_name(source: Mapping[str, Any] | None) -> str:
    """Extract country name from nested country field if present."""
    if not source or not isinstance(source, Mapping):
        return ""
    country_obj = source.get("country")
    if isinstance(country_obj, Mapping):
        cname = _extract_clean_str(country_obj, "name")
        if cname:
            return cname
    elif isinstance(country_obj, str):
        cname = country_obj.strip()
        if cname and cname.lower() not in ("none", "null"):
            return cname
    return ""


def resolve_location(
    area: Mapping[str, Any] | None = None,
    employer: Mapping[str, Any] | None = None,
    address: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Resolve standardized (country, city) tuple from vacancy payload data.

    Args:
        area: Area dict from payload containing 'id' and 'name'.
        employer: Employer dict containing 'country_id' and 'name'.
        address: Address dict containing 'city' and 'raw'.

    Returns:
        tuple[str, str]: (country, city). Returns ('Unknown', '') if unresolved.
    """
    area_id = _extract_clean_str(area, "id")
    city_name = _extract_clean_str(area, "name")

    # If city is missing from area, try address
    if not city_name:
        city_name = _extract_clean_str(address, "city")

    country = "Unknown"

    # 1. Check known area_id mapping
    if area_id:
        if area_id in _CACHED_AREA_TO_COUNTRY:
            country = _CACHED_AREA_TO_COUNTRY[area_id]
        elif area_id in COUNTRY_AREA_IDS:
            country = COUNTRY_AREA_IDS[area_id]

    # 2. Check area object country field
    if country == "Unknown":
        area_country = _extract_country_name(area)
        if area_country:
            country = area_country

    # 3. Check employer country_id if country is still unresolved
    if country == "Unknown" and employer and isinstance(employer, Mapping):
        cid = employer.get("country_id")
        if cid is not None and not isinstance(cid, bool):
            try:
                cid_int = int(float(cid))
                if cid_int in EMPLOYER_COUNTRY_ID_MAP:
                    country = EMPLOYER_COUNTRY_ID_MAP[cid_int]
            except (ValueError, TypeError, OverflowError):
                pass

    # 4. Check employer object country field
    if country == "Unknown":
        emp_country = _extract_country_name(employer)
        if emp_country:
            country = emp_country

    return (country, city_name)
