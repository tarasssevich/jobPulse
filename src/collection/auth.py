"""OAuth 2.0 Authentication for HeadHunter API.

Supports automatic headless Application Token generation (grant_type=client_credentials)
without requiring browser interaction.
"""

from __future__ import annotations

import logging

import requests
from dotenv import set_key

from src.utils.config import BASE_DIR, settings

logger = logging.getLogger(__name__)

ENV_FILE = BASE_DIR / ".env"


def obtain_app_token(
    client_id: str | None = None,
    client_secret: str | None = None,
    save_to_env: bool = True,
    force_refresh: bool = False,
) -> str:
    """Obtain an application-level OAuth access token via client_credentials grant.

    Note: HeadHunter application tokens (APPL...) are long-lived and HH enforces a cooldown
    ("app token refresh too early") if you request a new token while an existing one is active.
    Therefore, if HH_ACCESS_TOKEN is already present in settings and force_refresh is False,
    the existing token is returned.

    Args:
        client_id: HeadHunter application Client ID.
        client_secret: HeadHunter application Client Secret.
        save_to_env: Whether to persist the obtained token into local .env file.
        force_refresh: Force requesting a new token even if one is configured.

    Returns:
        Bearer access_token string.

    Raises:
        RuntimeError: If token request fails.
    """
    if settings.hh_access_token and not force_refresh:
        return settings.hh_access_token

    cid = client_id or settings.hh_client_id
    csecret = client_secret or settings.hh_client_secret

    if not cid or not csecret:
        raise RuntimeError(
            "HH_CLIENT_ID and HH_CLIENT_SECRET must be set in .env to obtain an application token."
        )

    token_url = "https://api.hh.ru/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": cid,
        "client_secret": csecret,
    }
    headers = {
        "User-Agent": settings.hh_user_agent,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    response = requests.post(token_url, data=payload, headers=headers, timeout=10.0)
    if response.status_code != 200:
        raise RuntimeError(
            f"App token request failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError(f"No access_token in response: {data}")

    if save_to_env and ENV_FILE.exists():
        set_key(str(ENV_FILE), "HH_ACCESS_TOKEN", access_token)
        logger.info("Application access token saved to .env")

    return str(access_token)


def run_auth_flow() -> str:
    """Acquire or verify application token from HeadHunter API."""
    print("\n--- JobPulse: Проверка токена приложения HeadHunter API ---")
    token = obtain_app_token(save_to_env=True)
    masked = f"{token[:8]}...{token[-6:]}" if len(token) > 14 else "***"
    print(f"[OK] Активный токен приложения готов к работе ({masked})!\n")
    return token


if __name__ == "__main__":
    run_auth_flow()
