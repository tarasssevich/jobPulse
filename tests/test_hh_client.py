"""Unit tests for HeadHunterClient using mocked HTTP responses."""

import pytest
import responses

from src.collection.auth import obtain_app_token
from src.collection.hh_client import HeadHunterClient, HHApiError


@responses.activate
def test_obtain_app_token_success():
    """Verify headless client_credentials token acquisition."""
    responses.add(
        responses.POST,
        "https://api.hh.ru/token",
        json={"access_token": "APPL_TEST_TOKEN_123", "token_type": "bearer"},
        status=200,
    )

    token = obtain_app_token(
        client_id="TEST_ID",
        client_secret="TEST_SECRET",
        save_to_env=False,
        force_refresh=True,
    )
    assert token == "APPL_TEST_TOKEN_123"


@responses.activate
def test_search_vacancies_success():
    """Verify search_vacancies correctly formats params and parses JSON."""
    client = HeadHunterClient(request_delay=0.0)

    responses.add(
        responses.GET,
        "https://api.hh.ru/vacancies",
        json={"items": [{"id": "123", "name": "Python Dev"}], "pages": 1, "found": 1},
        status=200,
    )

    result = client.search_vacancies(area=16, text="Python", page=0, per_page=10)

    assert result["found"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0]["id"] == "123"


@responses.activate
def test_get_vacancy_details_success():
    """Verify fetching full vacancy detail by ID."""
    client = HeadHunterClient(request_delay=0.0)

    responses.add(
        responses.GET,
        "https://api.hh.ru/vacancies/999",
        json={
            "id": "999",
            "name": "Senior DevOps",
            "description": "<p>Docker & Kubernetes</p>",
            "key_skills": [{"name": "Docker"}],
        },
        status=200,
    )

    detail = client.get_vacancy_details("999")
    assert detail["id"] == "999"
    assert detail["name"] == "Senior DevOps"
    assert "description" in detail


@responses.activate
def test_get_vacancy_details_404():
    """Verify 404 response raises HHApiError."""
    client = HeadHunterClient(request_delay=0.0)

    responses.add(
        responses.GET,
        "https://api.hh.ru/vacancies/000",
        json={"description": "Not found"},
        status=404,
    )

    with pytest.raises(HHApiError) as exc_info:
        client.get_vacancy_details("000")

    assert exc_info.value.status_code == 404
