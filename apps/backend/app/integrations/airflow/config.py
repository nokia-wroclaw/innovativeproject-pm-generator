"""Airflow integration configuration.

All settings come from environment variables. We keep this file deliberately
small and side-effect free; ``get_airflow_settings()`` is cached with ``lru_cache``
so the values are read once per process.

The defaults align with the values used in ``infra/airflow-docker-compose.yml``
(see ``_AIRFLOW_WWW_USER_USERNAME`` / ``_AIRFLOW_WWW_USER_PASSWORD`` for the
service-account credentials we exchange for a JWT at ``POST /auth/token``).
"""

from dataclasses import dataclass
from functools import lru_cache
import os

from fastapi import HTTPException, status


def _required_env(name: str) -> str:
    if value := os.getenv(name):
        return value
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Missing required Airflow configuration: {name}",
    )


@dataclass(frozen=True)
class AirflowSettings:
    base_url: str
    api_prefix: str
    auth_token_path: str
    username: str
    password: str
    http_timeout_seconds: float
    log_stream_max_duration_seconds: int
    log_stream_heartbeat_seconds: int
    dag_list_cache_seconds: int

    @property
    def api_base(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.api_prefix}"


@lru_cache
def get_airflow_settings() -> AirflowSettings:
    return AirflowSettings(
        base_url=_required_env("AIRFLOW_URL").rstrip("/"),
        api_prefix=os.getenv("AIRFLOW_API_PREFIX", "/api/v2"),
        auth_token_path=os.getenv("AIRFLOW_AUTH_TOKEN_PATH", "/auth/token"),
        username=_required_env("AIRFLOW_USERNAME"),
        password=_required_env("AIRFLOW_PASSWORD"),
        http_timeout_seconds=float(os.getenv("AIRFLOW_HTTP_TIMEOUT_SECONDS", "15")),
        log_stream_max_duration_seconds=int(
            os.getenv("LOG_STREAM_MAX_DURATION_SECONDS", "7200")
        ),
        log_stream_heartbeat_seconds=int(
            os.getenv("LOG_STREAM_HEARTBEAT_SECONDS", "15")
        ),
        dag_list_cache_seconds=int(os.getenv("DAG_LIST_CACHE_SECONDS", "30")),
    )
