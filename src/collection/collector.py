"""Collection orchestrator: coordinates HH API fetching, validation, and raw storage."""

from __future__ import annotations

import logging

from tqdm import tqdm

from src.collection.hh_client import HeadHunterClient, HHApiError
from src.collection.schemas import RawVacancy
from src.storage.raw_storage import RawStorage

logger = logging.getLogger(__name__)


# Standard HeadHunter Area IDs for CIS countries
CIS_AREAS = {
    "BY": 16,  # Беларусь
    "RU": 113,  # Россия
    "KZ": 40,  # Казахстан
}

# Standard IT professional role IDs from HeadHunter
# 96: Программист, разработчик
# 160: DevOps-инженер
# 165: Дата-сайентист
# 124: Тестировщик
# 104: Руководитель группы разработки
# 157: Системный аналитик
DEFAULT_IT_ROLES = [96, 160, 165, 124, 157]


class VacancyCollector:
    """Orchestrates fetching vacancies from HeadHunter and saving to raw storage."""

    def __init__(
        self,
        client: HeadHunterClient | None = None,
        storage: RawStorage | None = None,
    ) -> None:
        """Initialize collector with API client and raw storage.

        Args:
            client: Configured HeadHunterClient instance.
            storage: Configured RawStorage instance.
        """
        self.client = client or HeadHunterClient()
        if storage is None:
            from src.utils.config import settings

            self.storage = RawStorage(
                raw_data_dir=settings.raw_data_dir,
                sqlite_db_path=settings.sqlite_db_path,
            )
        else:
            self.storage = storage

    def collect(
        self,
        area_code: str = "BY",
        text: str | None = None,
        professional_roles: list[int | str] | None = None,
        max_vacancies: int = 100,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """Execute a collection run for a specified country and criteria.

        Args:
            area_code: CIS country code ('BY', 'RU', 'KZ') or integer area ID.
            text: Optional keyword search filter.
            professional_roles: List of professional role IDs to query.
            max_vacancies: Maximum number of new vacancies to collect in this run.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            Dictionary with statistics: scanned, saved_new, skipped_existing, errors.
        """
        area_id = CIS_AREAS.get(area_code.upper(), area_code)
        roles = professional_roles or [None]

        stats = {
            "scanned": 0,
            "saved_new": 0,
            "skipped_existing": 0,
            "errors": 0,
        }

        pbar = tqdm(total=max_vacancies, desc=f"Collecting {area_code}", disable=not show_progress)

        for role in roles:
            if stats["saved_new"] >= max_vacancies:
                break

            for page in range(20):  # HH API limits search pagination to pages 0..19 (max 2000)
                if stats["saved_new"] >= max_vacancies:
                    break

                try:
                    search_result = self.client.search_vacancies(
                        text=text,
                        area=area_id,
                        professional_role=role,
                        page=page,
                        per_page=100,
                    )
                except HHApiError as e:
                    logger.error("Search failed on page %d for role %s: %s", page, role, e)
                    stats["errors"] += 1
                    break

                items = search_result.get("items", [])
                if not items:
                    break  # No more items on this search branch

                for item in items:
                    if stats["saved_new"] >= max_vacancies:
                        break

                    stats["scanned"] += 1
                    vacancy_id = str(item.get("id"))

                    # Deduplication check before making network request for details
                    if self.storage.is_already_collected(vacancy_id):
                        stats["skipped_existing"] += 1
                        continue

                    # Fetch full details
                    try:
                        detail = self.client.get_vacancy_details(vacancy_id)
                        # Validate against schema
                        RawVacancy.model_validate(detail)

                        # Persist to raw storage (JSONL + SQLite)
                        is_new = self.storage.save_vacancy(
                            vacancy_id=vacancy_id,
                            payload=detail,
                            source="hh.ru",
                        )
                        if is_new:
                            stats["saved_new"] += 1
                            pbar.update(1)
                    except (HHApiError, ValueError, KeyError) as err:
                        logger.warning("Failed to fetch or save vacancy %s: %s", vacancy_id, err)
                        stats["errors"] += 1

                # If this was the last page, stop querying further pages
                if page >= search_result.get("pages", 1) - 1:
                    break

        pbar.close()
        logger.info("Collection finished: %s", stats)
        return stats
