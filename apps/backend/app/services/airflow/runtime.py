from app.services.airflow.service import AirflowService

from .auth import AirflowAuth
from .client import AirflowClient
from .config import get_airflow_settings

_client: AirflowClient | None = None
_service: AirflowService | None = None


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
