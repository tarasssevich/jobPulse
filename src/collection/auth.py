"""OAuth 2.0 Authorization Code flow for HeadHunter API.

Runs a temporary local callback server on http://localhost:8000/callback
to capture the authorization code and exchange it for an access token.
"""

from __future__ import annotations

import logging
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from dotenv import set_key

from src.utils.config import BASE_DIR, settings

logger = logging.getLogger(__name__)

ENV_FILE = BASE_DIR / ".env"


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Temporary HTTP handler to capture authorization code from HH redirect."""

    auth_code: str | None = None
    auth_error: str | None = None

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/callback":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            if "code" in query_params:
                OAuthCallbackHandler.auth_code = query_params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = """
                <html>
                <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 50px;">
                    <h2 style="color: #2e7d32;">✅ Авторизация JobPulse успешна!</h2>
                    <p>Токен получен. Вы можете закрыть эту вкладку и вернуться в терминал.</p>
                </body>
                </html>
                """
                self.wfile.write(html.encode("utf-8"))
            else:
                OAuthCallbackHandler.auth_error = query_params.get("error", ["unknown"])[0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Authorization failed.")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default HTTP request logs in console."""
        return


def get_authorize_url(oauth_host: str | None = None) -> str:
    """Generate HH OAuth authorize URL."""
    host = oauth_host or settings.hh_oauth_host
    params = {
        "response_type": "code",
        "client_id": settings.hh_client_id,
        "redirect_uri": settings.hh_redirect_uri,
    }
    return f"https://{host}/oauth/authorize?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(code: str, oauth_host: str | None = None) -> dict:
    """Exchange authorization code for access and refresh tokens.

    Args:
        code: Authorization code received from redirect.
        oauth_host: OAuth host domain ('rabota.by' or 'hh.ru').

    Returns:
        JSON response with access_token and refresh_token.
    """
    host = oauth_host or settings.hh_oauth_host
    token_url = f"https://{host}/oauth/token"
    payload = {
        "grant_type": "authorization_code",
        "client_id": settings.hh_client_id,
        "client_secret": settings.hh_client_secret,
        "redirect_uri": settings.hh_redirect_uri,
        "code": code,
    }
    headers = {
        "User-Agent": settings.hh_user_agent,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    response = requests.post(token_url, data=payload, headers=headers, timeout=10.0)
    if response.status_code != 200:
        raise RuntimeError(f"Token exchange failed ({response.status_code}): {response.text}")

    data = response.json()
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")

    # Persist into local .env securely
    if access_token and ENV_FILE.exists():
        set_key(str(ENV_FILE), "HH_ACCESS_TOKEN", access_token)
        if refresh_token:
            set_key(str(ENV_FILE), "HH_REFRESH_TOKEN", refresh_token)
        logger.info("Tokens successfully written to .env")

    return data


def run_auth_flow() -> str:
    """Run interactive local server and prompt user to authorize in browser."""
    auth_url = get_authorize_url()
    print("\n--- JobPulse: HeadHunter OAuth Авторизация ---")
    print("Открываем страницу авторизации в браузере...")
    print(f"Если браузер не открылся автоматически, перейдите по ссылке:\n{auth_url}\n")

    webbrowser.open(auth_url)

    # Start temporary local HTTP server on port 8000
    server_address = ("localhost", 8000)
    httpd = HTTPServer(server_address, OAuthCallbackHandler)
    print("Ожидание подтверждения авторизации на http://localhost:8000/callback ...")

    while OAuthCallbackHandler.auth_code is None and OAuthCallbackHandler.auth_error is None:
        httpd.handle_request()

    httpd.server_close()

    if OAuthCallbackHandler.auth_error:
        raise RuntimeError(f"Ошибка авторизации: {OAuthCallbackHandler.auth_error}")

    code = OAuthCallbackHandler.auth_code
    print("Код авторизации получен! Обмениваем на токен доступа...")
    tokens = exchange_code_for_token(code)
    print("✅ Токен успешно получен и сохранен в .env!\n")
    return tokens["access_token"]


if __name__ == "__main__":
    run_auth_flow()
