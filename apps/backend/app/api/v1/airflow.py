from typing import Any

from fastapi import APIRouter, Depends

from app.core.auth import require_auth
from app.services.airflow.client import AirflowClient
from app.services.airflow.errors import AirflowIntegrationError, AirflowUnavailable
from app.services.airflow.runtime import get_airflow_client

router = APIRouter(prefix="/airflow", tags=["airflow"])


def _client() -> AirflowClient:
    return get_airflow_client()


@router.get("/health")
async def airflow_health(
    _user: dict[str, Any] = Depends(require_auth),
    client: AirflowClient = Depends(_client),
) -> dict[str, Any]:
    try:
        body = await client.healthcheck()
    except AirflowIntegrationError:
        raise
    except Exception as exc:
        raise AirflowUnavailable(f"Healthcheck failed: {exc}") from exc
    return {"status": "ok", "airflow": body}
