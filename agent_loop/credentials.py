"""Backend credential access for trusted clients. No credentials are exposed as tools."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import quote, urlsplit

try:
    import httpx2 as httpx
except ModuleNotFoundError:
    import httpx


class CredentialError(RuntimeError):
    pass


@dataclass(frozen=True)
class Grant:
    session_id: str
    token: str = field(repr=False)


class CredentialClient:
    def __init__(self, url: str, read_grant: Callable[[], Grant], http=None):
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise CredentialError("Credential backend requires a plain HTTPS URL")
        self.url = url.rstrip("/")
        self._read_grant = read_grant
        self._http = http or httpx.Client(timeout=10, follow_redirects=False, trust_env=False)

    def _get(self, path: str) -> dict:
        grant = self._read_grant()
        try:
            response = self._http.get(self.url + path, headers={
                "Authorization": "Bearer " + grant.token, "X-Session-ID": grant.session_id,
            })
            if response.status_code in (401, 403):
                raise CredentialError("Container grant expired or revoked; renew it through the launching client")
            if response.status_code in (404, 409):
                raise CredentialError("User credential missing or expired")
            if response.status_code != 200:
                raise CredentialError("Credential backend unavailable")
            value = response.json()
            if not isinstance(value, dict):
                raise CredentialError("Invalid credential response")
            return value
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("Credential backend unavailable") from None

    def settings(self) -> dict[str, str]:
        value = self._get("/settings").get("settings")
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            raise CredentialError("Invalid user settings")
        return value

    def connections(self) -> list[dict]:
        value = self._get("/connections").get("connections")
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise CredentialError("Invalid connection metadata")
        return value

    def credential(self, connection_id: str) -> str:
        value = self._get("/connections/" + quote(connection_id, safe="") + "/credential")
        secret = value.get("credential")
        try:
            expires = datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
            provider_expiry = value.get("credential_expires_at")
            if provider_expiry:
                expires = min(expires, datetime.fromisoformat(provider_expiry.replace("Z", "+00:00")))
            if expires <= datetime.now(timezone.utc) or not isinstance(secret, str) or not secret:
                raise ValueError()
        except (ValueError, TypeError, KeyError, AttributeError):
            raise CredentialError("User credential missing or expired") from None
        return secret

    def close(self) -> None:
        self._http.close()


def sdk_client(connection: str, lookup: Callable[[str], str], *, base_url: str | None = None, timeout: float = 180):
    from openai import OpenAI

    def authorize(request):
        request.headers["Authorization"] = "Bearer " + lookup(connection)

    def forget(response):
        response.request.headers.pop("Authorization", None)

    http = httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False,
                        event_hooks={"request": [authorize], "response": [forget]})
    # The SDK never stores the user's key. Look it up again for every outbound request.
    return OpenAI(api_key="container-managed", base_url=base_url, timeout=timeout, max_retries=0, http_client=http)
