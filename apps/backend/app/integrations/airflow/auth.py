"""Service-account JWT issuance for Airflow.

Backend mints a short-lived JWT signed with ``AIRFLOW_JWT_SECRET`` (the same
secret Airflow expects via ``AIRFLOW__API_AUTH__JWT_SECRET``). Airflow 3.x
defaults to HS512 with audience ``apache-airflow``; we honour the env-var
overrides so this also works against custom deployments.

The token is cached in memory and refreshed proactively 60s before expiry.
Refreshing is serialised through an ``asyncio.Lock`` so we never mint two
tokens concurrently under load.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import jwt

from .config import AirflowSettings

_REFRESH_LEEWAY_SECONDS = 60


@dataclass(frozen=True)
class _CachedToken:
    token: str
    expires_at: float

    def is_fresh(self, *, now: float, leeway: int = _REFRESH_LEEWAY_SECONDS) -> bool:
        return self.expires_at - now > leeway


class AirflowAuth:
    """Issues and caches the Airflow service-account JWT.

    The class is intentionally framework-agnostic: it knows nothing about
    FastAPI. ``AirflowClient`` calls :py:meth:`get_token` before every request
    and ``invalidate()`` after a 401 to force a re-mint.
    """

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
            token, expires_at = self._mint(now=now)
            self._cached = _CachedToken(token=token, expires_at=expires_at)
            return token

    async def invalidate(self) -> None:
        async with self._lock:
            self._cached = None

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────
    def _mint(self, *, now: float) -> tuple[str, float]:
        issued_at = int(now)
        expires_at = issued_at + self._settings.jwt_ttl_seconds
        payload = {
            "iss": self._settings.jwt_issuer,
            "aud": self._settings.jwt_audience,
            "sub": self._settings.service_account_sub,
            "iat": issued_at,
            "nbf": issued_at,
            "exp": expires_at,
        }
        token = jwt.encode(
            payload,
            self._settings.jwt_secret,
            algorithm=self._settings.jwt_algorithm,
        )
        return token, float(expires_at)
