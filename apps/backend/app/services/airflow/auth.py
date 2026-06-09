import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
import jwt

from .config import AirflowSettings
from .errors import AirflowAuthFailed, AirflowUnavailable

logger = logging.getLogger(__name__)

_REFRESH_LEEWAY_SECONDS = 60
_FALLBACK_TTL_SECONDS = 600


@dataclass(frozen=True)
class _CachedToken:
    token: str
    expires_at: float

    def is_fresh(self, *, now: float, leeway: int = _REFRESH_LEEWAY_SECONDS) -> bool:
        return self.expires_at - now > leeway


class AirflowAuth:
    def __init__(self, settings: AirflowSettings) -> None:
        self._settings = settings
        self._cached: _CachedToken | None = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        now = time.time()
        cached = self._cached
        if cached is not None and cached.is_fresh(now=now):
            return cached.token

        async with self._lock:
            now = time.time()
            cached = self._cached
            if cached is not None and cached.is_fresh(now=now):
                return cached.token
            token, expires_at = await self._exchange_credentials()
            self._cached = _CachedToken(token=token, expires_at=expires_at)
            return token

    async def invalidate(self) -> None:
        async with self._lock:
            self._cached = None

    async def _exchange_credentials(self) -> tuple[str, float]:
        url = f"{self._settings.base_url.rstrip('/')}{self._settings.auth_token_path}"
        payload = {
            "username": self._settings.username,
            "password": self._settings.password,
        }
        try:
            async with httpx.AsyncClient(timeout=self._settings.http_timeout_seconds) as client:
                response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise AirflowUnavailable(f"Cannot reach Airflow auth endpoint ({url}): {exc}") from exc

        if response.status_code == 401:
            raise AirflowAuthFailed(
                "Airflow rejected the service-account credentials "
                "(AIRFLOW_USERNAME / AIRFLOW_PASSWORD)."
            )
        if response.status_code >= 500:
            raise AirflowUnavailable(f"Airflow auth endpoint returned {response.status_code}")
        if response.status_code >= 400:
            raise AirflowAuthFailed(
                f"Airflow auth endpoint returned {response.status_code}: " f"{response.text[:200]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise AirflowAuthFailed("Airflow auth endpoint returned non-JSON body") from exc

        token = self._extract_token(body)
        expires_at = self._extract_expiry(token)
        logger.info(
            "Acquired Airflow JWT for user=%s, expires_at=%s",
            self._settings.username,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_at)),
        )
        return token, expires_at

    @staticmethod
    def _extract_token(body: dict) -> str:
        for key in ("access_token", "token", "jwt", "id_token"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
        raise AirflowAuthFailed(
            f"Airflow auth response did not contain a token field " f"(keys={list(body)[:5]})"
        )

    @staticmethod
    def _extract_expiry(token: str) -> float:
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError:
            return time.time() + _FALLBACK_TTL_SECONDS
        exp = unverified.get("exp")
        if isinstance(exp, int | float) and exp > 0:
            return float(exp)
        return time.time() + _FALLBACK_TTL_SECONDS
