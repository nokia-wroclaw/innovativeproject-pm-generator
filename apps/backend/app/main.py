import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from app.api.v1 import airflow, dags, generation, modeling, pipeline, s3
from app.core.error_handlers import register_error_handlers
from app.core.logging import setup_logging
from app.db import schemas
from app.db.database import db_manager
from app.services.airflow.runtime import (
    start_airflow_runtime,
    stop_airflow_runtime,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    schemas.Base.metadata.create_all(bind=db_manager.engine)
    setup_logging()
    try:
        await start_airflow_runtime()
    except Exception:
        logger.exception(
            "Airflow runtime failed to start — DAG endpoints will be unavailable "
            "until AIRFLOW_URL / AIRFLOW_USERNAME / AIRFLOW_PASSWORD are configured."
        )
    try:
        yield
    finally:
        await stop_airflow_runtime()


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
allowed_origins = [_frontend_origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["ETag", "X-Request-ID"],
)


@app.middleware("http")
async def add_security_and_request_id(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Request-ID"] = request_id
    if response.headers.get("Content-Type", "").startswith("text/event-stream"):
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
    else:
        response.headers["Cache-Control"] = "no-store"
    return response


register_error_handlers(app)

app.include_router(generation.router, prefix="/api/v1")
app.include_router(airflow.router, prefix="/api/v1")
app.include_router(dags.router, prefix="/api/v1")
app.include_router(s3.router, prefix="/api/v1")
app.include_router(pipeline.router, prefix="/api/v1")
app.include_router(modeling.router, prefix="/api/v1")
