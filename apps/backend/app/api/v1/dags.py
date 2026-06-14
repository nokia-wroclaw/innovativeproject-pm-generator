import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Request
from sse_starlette.sse import EventSourceResponse

from app.core.auth import get_user_identity, require_admin
from app.models.auth import TokenPayload
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
    user: TokenPayload = Depends(require_admin),
    service: AirflowService = Depends(_service),
) -> ActionResponse:
    return await service.trigger_dag(
        dag_id,
        body=body or TriggerRequest(),
        triggered_by=get_user_identity(user),
    )


@router.post("/{dag_id}/runs/{run_id}/stop", response_model=ActionResponse)
async def stop_dag_run(
    dag_id: str,
    run_id: str,
    user: TokenPayload = Depends(require_admin),
    service: AirflowService = Depends(_service),
) -> ActionResponse:
    return await service.stop_dag_run(dag_id, run_id, triggered_by=get_user_identity(user))


@router.post("/{dag_id}/runs/{run_id}/clear", response_model=ActionResponse)
async def clear_dag_run(
    dag_id: str,
    run_id: str,
    user: TokenPayload = Depends(require_admin),
    service: AirflowService = Depends(_service),
) -> ActionResponse:
    return await service.clear_dag_run(dag_id, run_id, triggered_by=get_user_identity(user))


@router.post(
    "/{dag_id}/runs/{run_id}/tasks/{task_id}/clear",
    response_model=ActionResponse,
)
async def clear_task_instance(
    dag_id: str,
    run_id: str,
    task_id: str,
    downstream: bool = Query(False),
    user: TokenPayload = Depends(require_admin),
    service: AirflowService = Depends(_service),
) -> ActionResponse:
    return await service.clear_task_instance(
        dag_id,
        run_id,
        task_id,
        downstream=downstream,
        triggered_by=get_user_identity(user),
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
    """Streams task logs via Server-Sent Events (SSE)."""
    settings = get_airflow_settings()
    max_duration = settings.log_stream_max_duration_seconds
    heartbeat_interval = settings.log_stream_heartbeat_seconds

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        started_at = time.monotonic()
        heartbeat = HeartbeatTimer(heartbeat_interval)
        seq = 0
        token: str | None = None

        logger.info(
            f"SSE log stream opened: dag={dag_id} run={run_id} task={task_id} try={try_number}"
        )

        try:
            while True:
                if await request.is_disconnected():
                    return

                now = time.monotonic()
                if now - started_at > max_duration:
                    yield _build_sse_event("end", {"reason": "max_duration"})
                    return

                try:
                    chunk = await service.get_task_logs_page(
                        dag_id, run_id, task_id, try_number=try_number, token=token, seq=seq
                    )
                except AirflowIntegrationError as exc:
                    logger.warning(
                        "SSE log stream error: task=%s seq=%s code=%s msg=%s",
                        task_id,
                        seq,
                        exc.code,
                        exc.message,
                    )
                    yield _build_sse_event("error", {"error": exc.code, "message": exc.message})
                    return

                if chunk.lines:
                    yield _build_sse_event("chunk", chunk.model_dump_json())
                    seq += 1

                token = chunk.continuation

                if token is None:
                    # No more logs available at the moment, check if task is finished
                    try:
                        task_instance = await service.get_task_instance(dag_id, run_id, task_id)
                        if str(task_instance.status) in {"success", "failed", "skipped"}:
                            yield _build_sse_event("end", {"reason": "task_finished"})
                            return
                    except AirflowIntegrationError:
                        # Ignore temporary Airflow issues and retry later
                        pass

                    # Send heartbeat to keep connection alive
                    if heartbeat.should_beat():
                        yield _build_sse_event("heartbeat", {"ts": _utc_now_iso()})

                    await asyncio.sleep(2)
                else:
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            yield _build_sse_event("end", {"reason": "user_disconnect"})
            raise

    return EventSourceResponse(event_stream())


class HeartbeatTimer:
    """Helper to track when to send SSE heartbeats."""

    def __init__(self, interval_seconds: int | float):
        self.interval = interval_seconds
        self.last_beat = time.monotonic()

    def should_beat(self) -> bool:
        if time.monotonic() - self.last_beat >= self.interval:
            self.last_beat = time.monotonic()
            return True
        return False


def _build_sse_event(event_type: str, payload: dict | str) -> dict[str, str]:
    """Helper to construct Server-Sent Events (SSE) messages."""
    data_str = payload if isinstance(payload, str) else json.dumps(payload)
    return {"event": event_type, "data": data_str}


def _utc_now_iso() -> str:
    """Returns current UTC time in ISO format."""
    return datetime.now(tz=UTC).isoformat()
