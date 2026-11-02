"""Text cleaning and normalization utility for vacancy descriptions."""

from __future__ import annotations

import html
import re

# Regex matching any HTML tags
_HTML_TAG_PATTERN: re.Pattern[str] = re.compile(r"<[^>]+>")

# Regex matching one or more whitespace characters
_WHITESPACE_PATTERN: re.Pattern[str] = re.compile(r"\s+")


def clean_html_text(text: str | None) -> str:
    """Clean raw HTML description text into clean, normalized plain text.

    Steps:
        1. Handle None / non-string input by returning empty string.
        2. Strip null bytes and check for empty content.
        3. Strip HTML tags by replacing them with spaces to prevent word collisions.
        4. Decode HTML character entities via html.unescape.
        5. Replace multiple whitespace characters (spaces, tabs, newlines) with a single space.
        6. Strip leading and trailing whitespace.

    Args:
        text: Raw vacancy description string, potentially containing HTML.

    Returns:
        Sanitized plain text string, or empty string if input is None or empty.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Strip null bytes that could corrupt SQLite or downstream Parquet
    cleaned = text.replace("\x00", "")
    if not cleaned.strip():
        return ""

    # Strip HTML tags BEFORE unescaping to prevent literal entities (&lt;, &gt;)
    # from becoming fake HTML tags that delete enclosed content
    tagless = _HTML_TAG_PATTERN.sub(" ", cleaned)

    # Decode HTML entities (e.g., &amp; -> &, &quot; -> ", &nbsp; -> space, &gt; -> >)
    unescaped = html.unescape(tagless)

    # Collapse multiple whitespace characters into a single space
    collapsed = _WHITESPACE_PATTERN.sub(" ", unescaped)

    return collapsed.strip()
