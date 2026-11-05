"""Hard skills extraction engine for vacancy tags and natural language descriptions.

Compiles high-performance regex patterns with boundary verification, handling tricky
punctuation (C++, C#, .NET, Node.js, CI/CD, Vue.js) and preventing false positives
on short language tokens (Go, R, C, SOLID, REST API).
"""

from __future__ import annotations

import re
from typing import Any, Final

from src.skills.taxonomy import (
    ALIAS_TO_CANONICAL,
    SKILL_TO_CATEGORY,
)

# Skills with complex punctuation or contextual constraints that require bespoke regex patterns
_SPECIAL_SKILL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "C++",
        "C#",
        ".NET",
        "CI/CD",
        "Node.js",
        "Vue.js",
        "Next.js",
        "Nuxt.js",
        "Go",
        "R",
        "C",
        "SOLID",
        "REST API",
        "1C",
    }
)


class SkillExtractor:
    """Extraction engine for canonical hard skills from structured tags and free text."""

    def __init__(self) -> None:
        """Initialize and precompile extraction patterns for rapid batch execution."""
        # 1. Compile bespoke regex patterns for special tokens and contextual constraints
        self._cpp_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])c\+\+(?:11|14|17|20|23)?(?![a-zA-Z0-9_+#])",
            re.IGNORECASE,
        )
        self._csharp_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:c#|csharp|c-sharp)(?![a-zA-Z0-9_+#])",
            re.IGNORECASE,
        )
        self._dotnet_regex: Final[re.Pattern[str]] = re.compile(
            r"(?:(?<![a-zA-Z0-9_])\.net(?:\s+(?:core|\d+(?:\.\d+)?))?(?![a-zA-Z0-9_])|"
            r"(?<![a-zA-Z0-9_])dotnet(?:\s*core)?(?![a-zA-Z0-9_])|"
            r"(?<![a-zA-Z0-9_])asp\.net(?:\s+core)?(?![a-zA-Z0-9_]))",
            re.IGNORECASE,
        )
        self._cicd_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:ci\s*/\s*cd|cicd)(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )
        self._nodejs_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:node\.js|nodejs|node\s+js)(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )
        self._vuejs_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:vue(?:\.js)?|vuejs|vue\s*[23])(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )
        self._nextjs_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:next\.js|nextjs|next\s+js)(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )
        self._nuxtjs_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:nuxt(?:\.js)?|nuxtjs|nuxt\s+js)(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )
        self._go_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])golang(?![a-zA-Z0-9_])|"
            r"\b(?:backend|junior|middle|senior|lead|head\s+of|разработка\s+на|разработчик|"
            r"программист|язык|языке|знание|опыт|стек|на|на\s+языке)\s+go(?![a-zA-Z0-9_])|"
            r"(?<![a-zA-Z0-9_])go\s+(?:developer|backend|dev|engineer|разработчик|программист|language|lang|stack|core|microservices)(?![a-zA-Z0-9_])|"
            r"\b(?:python|java|c\+\+|rust|kotlin|scala|php)\s*(?:/|,|\+|или|and|or)\s*go(?![a-zA-Z0-9_])|"
            r"(?<![a-zA-Z0-9_])go\s*(?:/|,|\+|или|and|or)\s*(?:python|java|c\+\+|rust|kotlin|scala|php)\b|"
            r"(?<![a-zA-Z0-9_])go\s*1\.\d+",
            re.IGNORECASE,
        )
        self._r_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:rlang|r-project)(?![a-zA-Z0-9_])|"
            r"(?<![a-zA-Z0-9_])r\s+(?:language|programming|developer|script|studio|shiny)(?![a-zA-Z0-9_])|"
            r"\b(?:язык|языке)\s+r(?![a-zA-Z0-9_])|"
            r"\b(?:python|sql|matlab|spss|sas|julia)\s*(?:/|,|\+|или|and|or)\s*r(?![a-zA-Z0-9_])|"
            r"(?<![a-zA-Z0-9_])r\s*(?:/|,|\+|или|and|or)\s*(?:python|sql|matlab|spss|sas|julia)\b",
            re.IGNORECASE,
        )
        self._c_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])c\s+(?:language|programming|developer|разработчик|программист)(?![a-zA-Z0-9_])|"
            r"\b(?:язык|языке)\s+c(?![a-zA-Z0-9_])|"
            r"(?<![a-zA-Z0-9_])c\s*(?:/|,)\s*c\+\+|"
            r"(?<![a-zA-Z0-9_])c\+\+\s*(?:/|,)\s*c(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )
        self._solid_upper_regex: Final[re.Pattern[str]] = re.compile(r"\bSOLID\b")
        self._solid_lower_regex: Final[re.Pattern[str]] = re.compile(
            r"\b(?:solid\s+principles|принцип(?:ы|ов)?\s+solid|"
            r"solid\s*[\/,]\s*(?:kiss|dry|yagni)|(?:kiss|dry|yagni)\s*[\/,]\s*solid)\b",
            re.IGNORECASE,
        )
        self._rest_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:rest\s*apis?|restful|rest-apis?|рест\s*апи|"
            r"rest\s*сервис|rest\s*web\s*services)(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )
        self._one_c_regex: Final[re.Pattern[str]] = re.compile(
            r"(?<![a-zA-Z0-9_])(?:1c|1с)(?::\s*enterprise|:предприятие|\s+enterprise|"
            r"\s+предприятие|\s+программирование|\s+разработчик|\s+программист|\s+8\.3|\s+8)?(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )

        # 2. Compile unified single-pass regex for all standard aliases
        # Exclude special skills and single-character aliases to guarantee boundary precision
        self._standard_alias_to_canonical: dict[str, str] = {}
        for alias, canonical in ALIAS_TO_CANONICAL.items():
            if canonical not in _SPECIAL_SKILL_NAMES and len(alias) >= 2:
                self._standard_alias_to_canonical[alias] = canonical

        # Sort longer aliases first to avoid premature prefix matching (e.g., 'docker compose' before 'docker')
        sorted_aliases = sorted(self._standard_alias_to_canonical.keys(), key=len, reverse=True)
        escaped_aliases = (re.escape(alias) for alias in sorted_aliases)
        pattern_body = "|".join(escaped_aliases)
        self._standard_regex: Final[re.Pattern[str]] = re.compile(
            rf"(?<![a-zA-Z0-9_а-яА-ЯёЁ])({pattern_body})(?![a-zA-Z0-9_а-яА-ЯёЁ])",
            re.IGNORECASE,
        )

    def extract_from_key_skills(self, key_skills: list[str] | Any) -> set[str]:
        """Extract canonical skills from structured key_skills tag list.

        Args:
            key_skills: An iterable of skill tag strings or None/corrupted value.

        Returns:
            Set of extracted canonical skill names.
        """
        if key_skills is None:
            return set()

        # Handle potential float NaN or non-iterables safely
        try:
            import numpy as np

            if isinstance(key_skills, float) and np.isnan(key_skills):
                return set()
        except ImportError:
            pass

        if not hasattr(key_skills, "__iter__") or isinstance(key_skills, (str, bytes)):
            return set()

        extracted: set[str] = set()
        for raw_item in key_skills:
            if raw_item is None:
                continue
            item_str = str(raw_item).strip()
            if not item_str or item_str.lower() in ("nan", "none"):
                continue

            # Fast direct alias lookup
            canonical = ALIAS_TO_CANONICAL.get(item_str.lower())
            if canonical is not None:
                extracted.add(canonical)
            else:
                # Fall back to text extraction for compound tags (e.g. 'Node.js / Express', 'Python 3')
                from_text = self.extract_from_text(item_str)
                extracted.update(from_text)

        return extracted

    def extract_from_text(self, text: str | None) -> set[str]:
        """Extract canonical skills from free-form plain text or cleaned description.

        Args:
            text: Sanitized text string or None.

        Returns:
            Set of extracted canonical skill names.
        """
        if text is None or not isinstance(text, str):
            return set()
        if not text.strip():
            return set()

        extracted: set[str] = set()

        # 1. Match standard skills using unified high-speed regex
        for match in self._standard_regex.finditer(text):
            alias = match.group(1).lower()
            canonical = self._standard_alias_to_canonical.get(alias)
            if canonical is not None:
                extracted.add(canonical)

        # 2. Check special skills with custom boundary/context patterns
        if self._cpp_regex.search(text):
            extracted.add("C++")
        if self._csharp_regex.search(text):
            extracted.add("C#")
        if self._dotnet_regex.search(text):
            extracted.add(".NET")
        if self._cicd_regex.search(text):
            extracted.add("CI/CD")
        if self._nodejs_regex.search(text):
            extracted.add("Node.js")
        if self._vuejs_regex.search(text):
            extracted.add("Vue.js")
        if self._nextjs_regex.search(text):
            extracted.add("Next.js")
        if self._nuxtjs_regex.search(text):
            extracted.add("Nuxt.js")
        if self._go_regex.search(text):
            extracted.add("Go")
        if self._r_regex.search(text):
            extracted.add("R")
        if self._c_regex.search(text):
            extracted.add("C")
        if self._solid_upper_regex.search(text) or self._solid_lower_regex.search(text):
            extracted.add("SOLID")
        if self._rest_regex.search(text):
            extracted.add("REST API")
        if self._one_c_regex.search(text):
            extracted.add("1C")

        return extracted

    def extract(
        self,
        key_skills: list[str] | Any = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        """Extract canonical skills from both structured tags and free text, aggregated by category.

        Args:
            key_skills: Structured tag collection or None.
            text: Vacancy description plain text or None.

        Returns:
            Dictionary with keys:
                - 'skills': Sorted list of unique canonical skill names.
                - 'by_category': Mapping from category name string to sorted list of skills.
                - 'count': Total number of unique skills extracted.
        """
        all_skills: set[str] = set()

        if key_skills is not None:
            all_skills.update(self.extract_from_key_skills(key_skills))

        if text is not None:
            all_skills.update(self.extract_from_text(text))

        sorted_skills = sorted(all_skills)

        by_category: dict[str, list[str]] = {}
        for skill in sorted_skills:
            category = SKILL_TO_CATEGORY.get(skill)
            if category is not None:
                by_category.setdefault(category.value, []).append(skill)

        # Ensure all category skill lists are sorted
        for cat_skills in by_category.values():
            cat_skills.sort()

        return {
            "skills": sorted_skills,
            "by_category": by_category,
            "count": len(sorted_skills),
        }
