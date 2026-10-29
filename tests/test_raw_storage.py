"""Unit tests for RawStorage (JSONL & SQLite staging)."""

import json

from src.storage.raw_storage import RawStorage


def test_raw_storage_lifecycle(tmp_path):
    """Test full cycle: save, deduplicate, count and iterate."""
    raw_dir = tmp_path / "raw"
    db_file = tmp_path / "test_staging.db"

    storage = RawStorage(raw_data_dir=raw_dir, sqlite_db_path=db_file)

    payload_1 = {
        "id": "1001",
        "name": "Backend Python Developer",
        "salary": {"from": 3000, "to": 5000, "currency": "BYN"},
        "key_skills": [{"name": "Python"}, {"name": "FastAPI"}],
    }

    # 1. First save should return True
    assert not storage.is_already_collected("1001")
    is_saved = storage.save_vacancy(vacancy_id="1001", payload=payload_1)
    assert is_saved is True
    assert storage.is_already_collected("1001")
    assert storage.count_collected() == 1

    # 2. Saving the exact same ID again should return False (deduplication)
    is_saved_again = storage.save_vacancy(vacancy_id="1001", payload=payload_1)
    assert is_saved_again is False
    assert storage.count_collected() == 1

    # 3. Check JSONL file content
    jsonl_files = list(raw_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # Should only have 1 line because duplicate was ignored

    record = json.loads(lines[0])
    assert record["vacancy_id"] == "1001"
    assert record["payload"]["name"] == "Backend Python Developer"

    # 4. Save second vacancy
    payload_2 = {"id": "1002", "name": "Data Analyst"}
    storage.save_vacancy(vacancy_id="1002", payload=payload_2)
    assert storage.count_collected() == 2

    # 5. Test iterator
    items = list(storage.iter_raw_vacancies())
    assert len(items) == 2
    ids = {item["vacancy_id"] for item in items}
    assert ids == {"1001", "1002"}
