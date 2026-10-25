"""Raw storage layer: streaming JSONL logging and SQLite staging database.

Provides immutable persistence of raw API responses and deduplication by vacancy_id.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RawStorage:
    """Manages raw vacancy data storage in JSONL format and SQLite staging database."""

    def __init__(self, raw_data_dir: Path, sqlite_db_path: Path) -> None:
        """Initialize storage paths and ensure schema exists.

        Args:
            raw_data_dir: Directory where JSONL files are archived.
            sqlite_db_path: Path to the SQLite staging database file.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.sqlite_db_path = Path(sqlite_db_path)

        # Ensure parent directories exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_sqlite_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection."""
        conn = sqlite3.connect(self.sqlite_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency & safety
        return conn

    def _init_sqlite_schema(self) -> None:
        """Create staging table and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_vacancies (
                    vacancy_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL DEFAULT 'hh.ru',
                    payload_json TEXT NOT NULL,
                    collected_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'raw'
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_raw_vacancies_collected_at
                ON raw_vacancies (collected_at);
                """
            )

    def _get_daily_jsonl_path(self, target_date: datetime | None = None) -> Path:
        """Generate path for daily JSONL archive file."""
        dt = target_date or datetime.now(UTC)
        filename = f"hh_raw_{dt.strftime('%Y%m%d')}.jsonl"
        return self.raw_data_dir / filename

    def is_already_collected(self, vacancy_id: str) -> bool:
        """Check if vacancy was already saved in staging database.

        Args:
            vacancy_id: Unique vacancy identifier.

        Returns:
            True if vacancy exists in staging DB, False otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM raw_vacancies WHERE vacancy_id = ? LIMIT 1;",
                (str(vacancy_id),),
            )
            return cursor.fetchone() is not None

    def save_vacancy(
        self,
        vacancy_id: str,
        payload: dict[str, Any],
        source: str = "hh.ru",
    ) -> bool:
        """Persist a single raw vacancy into both JSONL archive and SQLite staging.

        Args:
            vacancy_id: Unique vacancy identifier.
            payload: Full raw JSON dictionary received from API.
            source: Source platform name.

        Returns:
            True if vacancy was newly inserted, False if it was already present.
        """
        str_id = str(vacancy_id)
        payload_str = json.dumps(payload, ensure_ascii=False)
        now_iso = datetime.now(UTC).isoformat()

        # 1. Insert into SQLite staging (with deduplication)
        is_new = False
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO raw_vacancies (
                    vacancy_id, source, payload_json, collected_at, status
                ) VALUES (?, ?, ?, ?, 'raw');
                """,
                (str_id, source, payload_str, now_iso),
            )
            if cursor.rowcount > 0:
                is_new = True

        # 2. Append to immutable JSONL archive if it's a new entry
        if is_new:
            jsonl_record = {
                "vacancy_id": str_id,
                "source": source,
                "collected_at": now_iso,
                "payload": payload,
            }
            jsonl_line = json.dumps(jsonl_record, ensure_ascii=False) + "\n"
            jsonl_path = self._get_daily_jsonl_path()
            with open(jsonl_path, mode="a", encoding="utf-8") as f:
                f.write(jsonl_line)

        return is_new

    def count_collected(self) -> int:
        """Return total count of raw vacancies in staging DB."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM raw_vacancies;")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def iter_raw_vacancies(self) -> Iterator[dict[str, Any]]:
        """Iterate over all raw vacancies stored in SQLite staging.

        Yields:
            Dict containing vacancy_id, source, collected_at, and parsed payload.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT vacancy_id, source, collected_at, payload_json FROM raw_vacancies;"
            )
            for row in cursor:
                yield {
                    "vacancy_id": row["vacancy_id"],
                    "source": row["source"],
                    "collected_at": row["collected_at"],
                    "payload": json.loads(row["payload_json"]),
                }
