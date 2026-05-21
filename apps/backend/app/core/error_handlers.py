import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.services.airflow.errors import (
    AirflowAuthFailed,
    AirflowConflict,
    AirflowIntegrationError,
    AirflowNotFound,
    AirflowUnavailable,
)
from app.models.dags import ApiError

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
    ).model_dump()
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={"X-Request-ID": body["request_id"]},
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AirflowUnavailable)
    async def _airflow_unavailable(request: Request, exc: AirflowUnavailable) -> JSONResponse:
        logger.warning("Airflow unavailable: %s", exc.message)
        return _envelope(
            code=exc.code, message=exc.message, details=exc.details,
            request=request, status_code=exc.http_status,
        )

    @app.exception_handler(AirflowAuthFailed)
    async def _airflow_auth(request: Request, exc: AirflowAuthFailed) -> JSONResponse:
        logger.error("Airflow rejected service-account token: %s", exc.message)
        return _envelope(
            code=exc.code, message=exc.message, details=exc.details,
            request=request, status_code=exc.http_status,
        )

    @app.exception_handler(AirflowNotFound)
    async def _airflow_notfound(request: Request, exc: AirflowNotFound) -> JSONResponse:
        path = (exc.details or {}).get("path", "")
        if "/dagRuns/" in path and "/taskInstances/" in path:
            code = "TASK_NOT_FOUND"
        elif "/dagRuns/" in path:
            code = "RUN_NOT_FOUND"
        else:
            code = "DAG_NOT_FOUND"
        return _envelope(
            code=code, message=exc.message, details=exc.details,
            request=request, status_code=404,
        )

    @app.exception_handler(AirflowConflict)
    async def _airflow_conflict(request: Request, exc: AirflowConflict) -> JSONResponse:
        return _envelope(
            code=exc.code, message=exc.message, details=exc.details,
            request=request, status_code=exc.http_status,
        )

    @app.exception_handler(AirflowIntegrationError)
    async def _airflow_integration(request: Request, exc: AirflowIntegrationError) -> JSONResponse:
        logger.exception("Unhandled Airflow integration error: %s", exc.message)
        return _envelope(
            code=exc.code, message=exc.message, details=exc.details,
            request=request, status_code=exc.http_status,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope(
            code="VALIDATION_ERROR",
            message="Request payload failed validation.",
            details={"errors": exc.errors()},
            request=request,
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def _http_exception(request: Request, exc: HTTPException) -> JSONResponse:
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

    @app.exception_handler(Exception)
    async def _fallback(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception in request %s", _request_id(request))
        return _envelope(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred.",
            details=None,
            request=request,
            status_code=500,
        )

    _ = (
        _airflow_unavailable, _airflow_auth, _airflow_notfound,
        _airflow_conflict, _airflow_integration, _validation,
        _http_exception, _fallback,
    )
