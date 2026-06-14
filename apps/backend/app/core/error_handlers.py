import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models.dags import ApiError
from app.services.airflow.errors import (
    AirflowAuthFailed,
    AirflowConflict,
    AirflowIntegrationError,
    AirflowNotFound,
    AirflowUnavailable,
)
from app.services.s3.visualization import VisualizationSchemaError
from app.services.s3.visualization_artifacts import VisualizationStorageError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return str(rid) if rid else "unknown"


def _envelope(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None,
    request: Request,
    status_code: int,
) -> JSONResponse:
    body = ApiError(
        error=code,
        message=message,
        details=details,
        request_id=_request_id(request),
    ).model_dump(mode="json")
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={"X-Request-ID": body["request_id"]},
    )


async def airflow_unavailable_handler(request: Request, exc: AirflowUnavailable) -> JSONResponse:
    logger.warning("Airflow unavailable: %s", exc.message)
    return _envelope(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        request=request,
        status_code=exc.http_status,
    )


async def airflow_auth_handler(request: Request, exc: AirflowAuthFailed) -> JSONResponse:
    logger.error("Airflow rejected service-account token: %s", exc.message)
    return _envelope(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        request=request,
        status_code=exc.http_status,
    )


async def airflow_notfound_handler(request: Request, exc: AirflowNotFound) -> JSONResponse:
    path = (exc.details or {}).get("path", "")
    if "/dagRuns/" in path and "/taskInstances/" in path:
        code = "TASK_NOT_FOUND"
    elif "/dagRuns/" in path:
        code = "RUN_NOT_FOUND"
    else:
        code = "DAG_NOT_FOUND"
    return _envelope(
        code=code,
        message=exc.message,
        details=exc.details,
        request=request,
        status_code=404,
    )


async def airflow_conflict_handler(request: Request, exc: AirflowConflict) -> JSONResponse:
    return _envelope(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        request=request,
        status_code=exc.http_status,
    )


async def airflow_integration_handler(
    request: Request, exc: AirflowIntegrationError
) -> JSONResponse:
    logger.exception("Unhandled Airflow integration error: %s", exc.message)
    return _envelope(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        request=request,
        status_code=exc.http_status,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _envelope(
        code="VALIDATION_ERROR",
        message="Request payload failed validation.",
        details={"errors": exc.errors()},
        request=request,
        status_code=422,
    )


async def visualization_schema_handler(
    request: Request, exc: VisualizationSchemaError
) -> JSONResponse:
    return _envelope(
        code="VISUALIZATION_SCHEMA_ERROR",
        message=str(exc),
        details=None,
        request=request,
        status_code=400,
    )


async def visualization_storage_handler(
    request: Request, exc: VisualizationStorageError
) -> JSONResponse:
    logger.error("Visualization storage error: %s", exc)
    return _envelope(
        code="VISUALIZATION_STORAGE_ERROR",
        message="An error occurred while accessing visualization storage.",
        details=None,
        request=request,
        status_code=500,
    )


async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    message = str(exc)
    if "Airflow runtime not started" in message or "Airflow configuration" in message:
        return _envelope(
            code="AIRFLOW_UNAVAILABLE",
            message=message,
            details=None,
            request=request,
            status_code=503,
        )
    return _envelope(
        code="INTERNAL_ERROR",
        message=message,
        details=None,
        request=request,
        status_code=500,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code == 401:
        code = "UNAUTHENTICATED"
    elif exc.status_code == 403:
        code = "FORBIDDEN"
    else:
        code = "INTERNAL_ERROR" if exc.status_code >= 500 else "VALIDATION_ERROR"
    return _envelope(
        code=code,
        message=str(exc.detail),
        details=None,
        request=request,
        status_code=exc.status_code,
    )


async def fallback_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception in request %s", _request_id(request))
    return _envelope(
        code="INTERNAL_ERROR",
        message="An unexpected error occurred.",
        details=None,
        request=request,
        status_code=500,
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(VisualizationSchemaError, visualization_schema_handler)
    app.add_exception_handler(VisualizationStorageError, visualization_storage_handler)
    app.add_exception_handler(AirflowUnavailable, airflow_unavailable_handler)
    app.add_exception_handler(AirflowAuthFailed, airflow_auth_handler)
    app.add_exception_handler(AirflowNotFound, airflow_notfound_handler)
    app.add_exception_handler(AirflowConflict, airflow_conflict_handler)
    app.add_exception_handler(AirflowIntegrationError, airflow_integration_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(RuntimeError, runtime_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, fallback_exception_handler)
