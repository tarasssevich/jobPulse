"""Unit tests for VacancyCollector orchestrator."""

from unittest.mock import MagicMock

from src.collection.collector import VacancyCollector
from src.storage.raw_storage import RawStorage


def test_collector_flow(tmp_path):
    """Test collector orchestrator with mocked HH client."""
    storage = RawStorage(
        raw_data_dir=tmp_path / "raw",
        sqlite_db_path=tmp_path / "test.db",
    )

    mock_client = MagicMock()
    mock_client.search_vacancies.return_value = {
        "items": [{"id": "2001"}, {"id": "2002"}],
        "pages": 1,
        "found": 2,
    }
    mock_client.get_vacancy_details.side_effect = [
        {
            "id": "2001",
            "name": "Frontend React Dev",
            "published_at": "2026-10-25T10:00:00",
            "area": {"id": "16", "name": "Minsk"},
            "key_skills": [{"name": "React"}, {"name": "TypeScript"}],
        },
        {
            "id": "2002",
            "name": "Backend Python Dev",
            "published_at": "2026-10-25T11:00:00",
            "area": {"id": "16", "name": "Minsk"},
            "key_skills": [{"name": "Python"}],
        },
    ]

    collector = VacancyCollector(client=mock_client, storage=storage)

    stats = collector.collect(
        area_code="BY",
        professional_roles=[96],
        max_vacancies=10,
        show_progress=False,
    )

    assert stats["saved_new"] == 2
    assert stats["skipped_existing"] == 0
    assert storage.count_collected() == 2

    # Second run should skip already collected IDs
    stats_2 = collector.collect(
        area_code="BY",
        professional_roles=[96],
        max_vacancies=10,
        show_progress=False,
    )
    assert stats_2["saved_new"] == 0
    assert stats_2["skipped_existing"] == 2
    assert storage.count_collected() == 2
