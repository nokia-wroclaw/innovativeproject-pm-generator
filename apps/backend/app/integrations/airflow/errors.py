"""Exceptions raised by the Airflow integration layer.

The API router maps these to ``ApiError`` payloads (see
``docs/architecture/dag-management.md`` §6).
"""

from typing import Any


class AirflowIntegrationError(Exception):
    """Base class for all Airflow integration failures."""

    code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class AirflowUnavailable(AirflowIntegrationError):
    code = "AIRFLOW_UNAVAILABLE"
    http_status = 502


class AirflowAuthFailed(AirflowIntegrationError):
    code = "AIRFLOW_AUTH_FAILED"
    http_status = 502


class AirflowNotFound(AirflowIntegrationError):
    code = "DAG_NOT_FOUND"
    http_status = 404


class AirflowConflict(AirflowIntegrationError):
    code = "AIRFLOW_CONFLICT"
    http_status = 409
