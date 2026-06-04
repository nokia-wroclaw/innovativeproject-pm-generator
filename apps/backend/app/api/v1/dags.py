import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from sse_starlette.sse import EventSourceResponse

from app.core.auth import require_admin
from app.models.dags import (
    ActionResponse,
    DagDetails,
    DagRunSummary,
    DagSummary,
    LogChunk,
    TaskInstance,
    TaskTry,
    TriggerRequest,
)
from app.services.airflow.config import get_airflow_settings
from app.services.airflow.errors import AirflowIntegrationError
from app.services.airflow.runtime import get_airflow_service
from app.services.airflow.service import AirflowService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dags",
    tags=["dags"],
    dependencies=[Depends(require_admin)],
)


def _service() -> AirflowService:
    return get_airflow_service()


def _identity(payload: dict[str, Any]) -> str | None:
    return payload.get("preferred_username") or payload.get("email") or payload.get("sub")


@router.get("", response_model=list[DagSummary])
async def list_dags(
    service: AirflowService = Depends(_service),
) -> list[DagSummary]:
    return await service.list_dags()


@router.get("/{dag_id}", response_model=DagDetails)
async def get_dag_details(
    dag_id: str,
    service: AirflowService = Depends(_service),
) -> DagDetails:
    return await service.get_dag_details(dag_id)


@router.get("/{dag_id}/runs", response_model=list[DagRunSummary])
async def list_dag_runs(
    dag_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: AirflowService = Depends(_service),
) -> list[DagRunSummary]:
    return await service.list_dag_runs(dag_id, limit=limit, offset=offset)


@router.get("/{dag_id}/runs/{run_id}/tasks", response_model=list[TaskInstance])
async def list_task_instances(
    dag_id: str,
    run_id: str,
    service: AirflowService = Depends(_service),
) -> list[TaskInstance]:
    return await service.list_task_instances(dag_id, run_id)


@router.get("/{dag_id}/runs/{run_id}/tasks/{task_id}", response_model=TaskInstance)
async def get_task_instance(
    dag_id: str,
    run_id: str,
    task_id: str,
    service: AirflowService = Depends(_service),
) -> TaskInstance:
    return await service.get_task_instance(dag_id, run_id, task_id)


@router.get(
    "/{dag_id}/runs/{run_id}/tasks/{task_id}/tries",
    response_model=list[TaskTry],
)
async def list_task_tries(
    dag_id: str,
    run_id: str,
    task_id: str,
    service: AirflowService = Depends(_service),
) -> list[TaskTry]:
    return await service.list_task_tries(dag_id, run_id, task_id)


@router.get("/{dag_id}/runs/{run_id}/tasks/{task_id}/logs", response_model=LogChunk)
async def get_task_logs(
    dag_id: str,
    run_id: str,
    task_id: str,
    try_number: int = Query(..., ge=1),
    token: str | None = Query(None),
    service: AirflowService = Depends(_service),
) -> LogChunk:
    return await service.get_task_logs_page(
        dag_id, run_id, task_id, try_number=try_number, token=token
    )


@router.post("/{dag_id}/runs", response_model=ActionResponse)
async def trigger_dag(
    dag_id: str,
    body: TriggerRequest | None = None,
    user: dict[str, Any] = Depends(require_admin),
    service: AirflowService = Depends(_service),
) -> ActionResponse:
    return await service.trigger_dag(
        dag_id,
        body=body or TriggerRequest(),
        triggered_by=_identity(user),
    )


@router.post("/{dag_id}/runs/{run_id}/stop", response_model=ActionResponse)
async def stop_dag_run(
    dag_id: str,
    run_id: str,
    user: dict[str, Any] = Depends(require_admin),
    service: AirflowService = Depends(_service),
) -> ActionResponse:
    return await service.stop_dag_run(dag_id, run_id, triggered_by=_identity(user))


@router.post("/{dag_id}/runs/{run_id}/clear", response_model=ActionResponse)
async def clear_dag_run(
    dag_id: str,
    run_id: str,
    user: dict[str, Any] = Depends(require_admin),
    service: AirflowService = Depends(_service),
) -> ActionResponse:
    return await service.clear_dag_run(dag_id, run_id, triggered_by=_identity(user))


@router.post(
    "/{dag_id}/runs/{run_id}/tasks/{task_id}/clear",
    response_model=ActionResponse,
)
async def clear_task_instance(
    dag_id: str,
    run_id: str,
    task_id: str,
    downstream: bool = Query(False),
    user: dict[str, Any] = Depends(require_admin),
    service: AirflowService = Depends(_service),
) -> ActionResponse:
    return await service.clear_task_instance(
        dag_id,
        run_id,
        task_id,
        downstream=downstream,
        triggered_by=_identity(user),
    )


@router.get("/{dag_id}/runs/{run_id}/tasks/{task_id}/logs/stream")
async def stream_task_logs(
    request: Request,
    dag_id: str,
    run_id: str,
    task_id: str,
    try_number: int = Query(..., ge=1),
    service: AirflowService = Depends(_service),
) -> EventSourceResponse:
    settings = get_airflow_settings()
    max_duration = settings.log_stream_max_duration_seconds
    heartbeat = settings.log_stream_heartbeat_seconds

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        started_at = time.monotonic()
        seq = 0
        token: str | None = None
        last_heartbeat = started_at
        logger.info(
            "SSE log stream open: dag=%s run=%s task=%s try=%d",
            dag_id,
            run_id,
            task_id,
            try_number,
        )

        try:
            while True:
                if await request.is_disconnected():
                    return

                now = time.monotonic()
                if now - started_at > max_duration:
                    yield {
                        "event": "end",
                        "data": json.dumps({"reason": "max_duration"}),
                    }
                    return

                try:
                    chunk = await service.get_task_logs_page(
                        dag_id,
                        run_id,
                        task_id,
                        try_number=try_number,
                        token=token,
                        seq=seq,
                    )
                except AirflowIntegrationError as exc:
                    logger.warning(
                        "SSE log stream error: dag=%s task=%s seq=%d code=%s msg=%s",
                        dag_id,
                        task_id,
                        seq,
                        exc.code,
                        exc.message,
                    )
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": exc.code, "message": exc.message}),
                    }
                    return

                payload_has_lines = bool(chunk.lines)
                logger.info(
                    "SSE log chunk: dag=%s task=%s seq=%d lines=%d has_more=%s",
                    dag_id,
                    task_id,
                    seq,
                    len(chunk.lines),
                    chunk.has_more,
                )
                if payload_has_lines:
                    yield {
                        "event": "chunk",
                        "data": chunk.model_dump_json(),
                    }
                    seq += 1

                token = chunk.continuation

                if token is None:
                    try:
                        task_instance = await service.get_task_instance(dag_id, run_id, task_id)
                    except AirflowIntegrationError:
                        await asyncio.sleep(2)
                        continue
                    if task_instance.status in {
                        "success",
                        "failed",
                        "skipped",
                    }:
                        yield {
                            "event": "end",
                            "data": json.dumps({"reason": "task_finished"}),
                        }
                        return

                    if (time.monotonic() - last_heartbeat) >= heartbeat:
                        yield {
                            "event": "heartbeat",
                            "data": json.dumps({"ts": _utc_now_iso()}),
                        }
                        last_heartbeat = time.monotonic()

                    await asyncio.sleep(2)
                else:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            yield {
                "event": "end",
                "data": json.dumps({"reason": "user_disconnect"}),
            }
            raise

    return EventSourceResponse(event_stream())


def _utc_now_iso() -> str:
    from datetime import datetime

    return datetime.now(tz=UTC).isoformat()
