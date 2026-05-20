"""Airflow integration configuration.

All settings come from environment variables. We keep this file deliberately
small and side-effect free; ``get_airflow_settings()`` is cached with ``lru_cache``
so the values are read once per process.

The defaults align with the values used in ``infra/airflow-docker-compose.yml``
(see ``AIRFLOW__API_AUTH__JWT_SECRET``, ``AIRFLOW__API_AUTH__JWT_ISSUER``).
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
    jwt_secret: str
    jwt_algorithm: str
    jwt_issuer: str
    jwt_audience: str
    service_account_sub: str
    jwt_ttl_seconds: int
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
        jwt_secret=os.getenv("AIRFLOW_JWT_SECRET", "airflow_jwt_secret"),
        jwt_algorithm=os.getenv("AIRFLOW_JWT_ALGORITHM", "HS512"),
        jwt_issuer=os.getenv("AIRFLOW_JWT_ISSUER", "airflow"),
        jwt_audience=os.getenv("AIRFLOW_JWT_AUDIENCE", "apache-airflow"),
        service_account_sub=os.getenv("AIRFLOW_SERVICE_ACCOUNT_SUB", "genpm-backend"),
        jwt_ttl_seconds=int(os.getenv("AIRFLOW_JWT_TTL_SECONDS", "600")),
        http_timeout_seconds=float(os.getenv("AIRFLOW_HTTP_TIMEOUT_SECONDS", "15")),
        log_stream_max_duration_seconds=int(
            os.getenv("LOG_STREAM_MAX_DURATION_SECONDS", "7200")
        ),
        log_stream_heartbeat_seconds=int(
            os.getenv("LOG_STREAM_HEARTBEAT_SECONDS", "15")
        ),
        dag_list_cache_seconds=int(os.getenv("DAG_LIST_CACHE_SECONDS", "30")),
    )
