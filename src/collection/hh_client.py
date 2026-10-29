"""HeadHunter API client with rate limiting, error handling, and retries."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.config import settings

logger = logging.getLogger(__name__)


class HHApiError(Exception):
    """Base exception for HeadHunter API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HHRateLimitError(HHApiError):
    """Raised when rate limit (HTTP 429) is exceeded."""


class HeadHunterClient:
    """Robust client for querying HeadHunter public API."""

    BASE_URL = "https://api.hh.ru"

    def __init__(
        self,
        user_agent: str | None = None,
        access_token: str | None = None,
        request_delay: float | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize client with session and retry policy.

        Args:
            user_agent: Required User-Agent identifying application.
            access_token: Optional OAuth access token for authenticated requests.
            request_delay: Minimum delay in seconds between requests (polite scraping).
            timeout: Default request timeout in seconds.
        """
        self.user_agent = user_agent or settings.hh_user_agent
        self.access_token = access_token or settings.hh_access_token
        self.request_delay = (
            request_delay if request_delay is not None else settings.request_delay_seconds
        )
        self.timeout = timeout if timeout is not None else settings.request_timeout_seconds

        self.session = requests.Session()
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        self.session.headers.update(headers)

        # Configure network level retries for transient connection errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._last_request_time: float = 0.0

    def _throttle(self) -> None:
        """Enforce minimum delay between consecutive API requests."""
        if self.request_delay <= 0:
            return
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Execute HTTP request with throttling and 429 backoff handling.

        Args:
            method: HTTP method (GET, POST).
            endpoint: API endpoint path (e.g. '/vacancies').
            params: Query string parameters.
            max_retries: Maximum attempts for rate-limited requests.

        Returns:
            Parsed JSON dictionary from API response.

        Raises:
            HHRateLimitError: If rate limit persists after retries.
            HHApiError: On client/server HTTP errors.
        """
        url = (
            f"{self.BASE_URL}{endpoint}"
            if endpoint.startswith("/")
            else f"{self.BASE_URL}/{endpoint}"
        )
        attempt = 0

        while attempt < max_retries:
            attempt += 1
            self._throttle()

            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                logger.warning(
                    "Network error on %s %s (attempt %d/%d): %s",
                    method,
                    url,
                    attempt,
                    max_retries,
                    e,
                )
                if attempt >= max_retries:
                    raise HHApiError(f"Network error: {e}") from e
                time.sleep(2.0 * attempt)
                continue

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                wait_time = float(response.headers.get("Retry-After", 2.0 * attempt))
                logger.warning("Rate limit hit (429) for %s. Sleeping %s seconds.", url, wait_time)
                time.sleep(wait_time)
                continue

            if response.status_code == 403 and settings.hh_client_id and settings.hh_client_secret:
                logger.info("Received 403 Forbidden; attempting automatic app token acquisition...")
                try:
                    from src.collection.auth import obtain_app_token

                    new_token = obtain_app_token(save_to_env=True)
                    self.access_token = new_token
                    self.session.headers["Authorization"] = f"Bearer {new_token}"
                    continue
                except Exception as auth_err:  # noqa: BLE001
                    logger.warning("Automatic token acquisition failed: %s", auth_err)

            if response.status_code == 404:
                raise HHApiError(f"Resource not found (404): {url}", status_code=404)

            if response.status_code >= 400:
                error_msg = (
                    f"HH API returned error {response.status_code} for {url}: {response.text[:200]}"
                )
                logger.error(error_msg)
                raise HHApiError(error_msg, status_code=response.status_code)

        raise HHRateLimitError(
            f"Exceeded max retries ({max_retries}) due to rate limiting on {url}", status_code=429
        )

    def search_vacancies(
        self,
        text: str | None = None,
        area: str | int | None = None,
        professional_role: str | int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 0,
        per_page: int = 100,
    ) -> dict[str, Any]:
        """Search vacancies matching query criteria.

        Args:
            text: Keyword search query (e.g. 'Python').
            area: Area ID (16=Belarus, 113=Russia, 40=Kazakhstan).
            professional_role: Professional role ID (e.g. 96=Programmer/Developer).
            date_from: ISO 8601 publication start date.
            date_to: ISO 8601 publication end date.
            page: Zero-based page index (0 to 19).
            per_page: Results per page (up to 100).

        Returns:
            Dictionary with 'items' (list of vacancy summaries), 'found', 'pages', 'page'.
        """
        params: dict[str, Any] = {
            "page": page,
            "per_page": min(per_page, 100),
        }
        if text:
            params["text"] = text
        if area:
            params["area"] = area
        if professional_role:
            params["professional_role"] = professional_role
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        return self._request("GET", "/vacancies", params=params)

    def get_vacancy_details(self, vacancy_id: str | int) -> dict[str, Any]:
        """Fetch complete vacancy payload by ID.

        Args:
            vacancy_id: HeadHunter vacancy identifier.

        Returns:
            Full vacancy payload dictionary containing description and key_skills.
        """
        return self._request("GET", f"/vacancies/{vacancy_id}")

    def get_professional_roles(self) -> list[dict[str, Any]]:
        """Fetch professional roles directory from HeadHunter.

        Returns:
            List of categories and their respective professional roles.
        """
        result = self._request("GET", "/professional_roles")
        return result.get("categories", [])
