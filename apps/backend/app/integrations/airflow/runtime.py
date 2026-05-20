"""Process-wide AirflowAuth + AirflowClient lifecycle.

The client is created once at startup and closed at shutdown. We expose
``start()``/``close()`` helpers that the FastAPI lifespan invokes, plus a
``get_airflow_client()`` / ``get_airflow_service()`` getter pair used as
dependency-injection factories in the API router.

Why a module-level singleton? Because ``httpx.AsyncClient`` has expensive
connection-pool state that we want to share across requests; spinning up
a fresh client per request would be wasteful.
"""

from __future__ import annotations

from typing import Optional

from app.services.airflow import AirflowService

from .auth import AirflowAuth
from .client import AirflowClient
from .config import get_airflow_settings

_client: Optional[AirflowClient] = None
_service: Optional[AirflowService] = None


async def start_airflow_runtime() -> None:
    global _client, _service
    if _client is not None:
        return
    settings = get_airflow_settings()
    auth = AirflowAuth(settings)
    client = AirflowClient(settings, auth)
    await client.start()
    _client = client
    _service = AirflowService(client=client, settings=settings)


async def stop_airflow_runtime() -> None:
    global _client, _service
    if _client is not None:
        await _client.close()
    _client = None
    _service = None


def get_airflow_client() -> AirflowClient:
    if _client is None:
        raise RuntimeError("Airflow runtime not started; check FastAPI lifespan.")
    return _client


def get_airflow_service() -> AirflowService:
    if _service is None:
        raise RuntimeError("Airflow runtime not started; check FastAPI lifespan.")
    return _service
