import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.airflow.client import AirflowClient
from app.services.airflow.config import AirflowSettings
from app.services.airflow.errors import AirflowIntegrationError, AirflowNotFound
from app.services.airflow.mapper import (
    map_dag_details,
    map_dag_graph,
    map_dag_run,
    map_dag_summary,
    map_log_chunk,
    map_task_instance,
    map_task_try,
)
from app.models.dags import (
    ActionResponse,
    DagDetails,
    DagRunStatus,
    DagRunSummary,
    DagStats,
    DagSummary,
    LogChunk,
    TaskInstance,
    TaskTry,
    TriggerRequest,
)

logger = logging.getLogger(__name__)


class AirflowService:
    def __init__(self, *, client: AirflowClient, settings: AirflowSettings) -> None:
        self._client = client
        self._settings = settings
        self._dag_list_cache: tuple[float, list[DagSummary]] | None = None
        self._dag_list_lock = asyncio.Lock()

    async def list_dags(self) -> list[DagSummary]:
        now = time.monotonic()
        cache_ttl = self._settings.dag_list_cache_seconds
        cached = self._dag_list_cache
        if cached and (now - cached[0]) < cache_ttl:
            return cached[1]

        async with self._dag_list_lock:
            now = time.monotonic()
            cached = self._dag_list_cache
            if cached and (now - cached[0]) < cache_ttl:
                return cached[1]
            summaries = await self._fetch_dag_list_uncached()
            self._dag_list_cache = (time.monotonic(), summaries)
            return summaries

    async def get_dag_details(self, dag_id: str) -> DagDetails:
        dag_raw, tasks_raw, runs_raw = await asyncio.gather(
            self._client.get_dag(dag_id),
            self._client.list_dag_tasks(dag_id),
            self._client.list_dag_runs(dag_id, limit=10),
        )

        graph = map_dag_graph(tasks_raw)
        recent_runs = [map_dag_run(r) for r in runs_raw.get("dag_runs", []) or []]
        last_run = recent_runs[0] if recent_runs else None
        stats = _aggregate_stats_from_runs(recent_runs)
        summary = map_dag_summary(dag_raw, last_run=last_run, stats=stats)

        return map_dag_details(summary=summary, graph=graph, recent_runs=recent_runs)

    async def list_dag_runs(
        self, dag_id: str, *, limit: int = 50, offset: int = 0
    ) -> list[DagRunSummary]:
        raw = await self._client.list_dag_runs(dag_id, limit=limit, offset=offset)
        return [map_dag_run(r) for r in raw.get("dag_runs", []) or []]

    async def get_dag_run(self, dag_id: str, run_id: str) -> DagRunSummary:
        raw = await self._fetch_dag_run_raw(
            dag_id,
            run_id,
            genpm_run_id=run_id if run_id.startswith("genpm_") else None,
        )
        return map_dag_run(raw)

    async def list_task_instances(
        self, dag_id: str, run_id: str
    ) -> list[TaskInstance]:
        raw = await self._client.list_task_instances(dag_id, run_id)
        items = raw.get("task_instances", []) or []
        logger.info(
            "list_task_instances dag=%s run=%s -> %d items (keys=%s)",
            dag_id, run_id, len(items), list(raw.keys()),
        )
        return [map_task_instance(ti) for ti in items]

    async def get_task_instance(
        self, dag_id: str, run_id: str, task_id: str
    ) -> TaskInstance:
        raw = await self._client.get_task_instance(dag_id, run_id, task_id)
        return map_task_instance(raw)

    async def list_task_tries(
        self, dag_id: str, run_id: str, task_id: str
    ) -> list[TaskTry]:
        raw = await self._client.list_task_tries(dag_id, run_id, task_id)
        return [
            map_task_try(t)
            for t in (raw.get("task_instances", []) or [])
        ]

    async def get_task_logs_page(
        self,
        dag_id: str,
        run_id: str,
        task_id: str,
        *,
        try_number: int,
        token: str | None = None,
        seq: int = 0,
    ) -> LogChunk:
        raw = await self._client.get_task_logs(
            dag_id, run_id, task_id, try_number, token=token
        )
        return map_log_chunk(raw, try_number=try_number, seq=seq)

    async def trigger_dag(
        self,
        dag_id: str,
        *,
        body: TriggerRequest,
        triggered_by: str | None,
    ) -> ActionResponse:
        await self._ensure_dag_unpaused(dag_id)
        note = _compose_note(body.note, triggered_by)
        logical_date = (
            body.logical_date.isoformat() if body.logical_date else None
        )
        raw = await self._client.trigger_dag(
            dag_id,
            conf=body.conf,
            dag_run_id=body.dag_run_id,
            logical_date=logical_date,
            note=note,
        )
        self._invalidate_dag_list_cache()
        requested_id = str(body.dag_run_id or "")
        resolved_raw = await self._fetch_dag_run_raw(
            dag_id,
            _dag_run_id_from_payload(raw) or requested_id,
            genpm_run_id=requested_id or None,
        )
        run_id = _dag_run_id_from_payload(resolved_raw)
        if not run_id:
            raise AirflowIntegrationError(
                f"Airflow triggered '{dag_id}' but returned no dag_run_id.",
            )
        return ActionResponse(
            run_id=run_id,
            message=f"Triggered DAG '{dag_id}' (run {run_id})",
            airflow_status=200,
        )

    async def _fetch_dag_run_raw(
        self,
        dag_id: str,
        run_id: str,
        *,
        genpm_run_id: str | None = None,
    ) -> dict[str, Any]:
        if not run_id and not genpm_run_id:
            raise AirflowNotFound(
                f"No run id provided for DAG '{dag_id}'",
                details={"dag_id": dag_id},
            )

        lookup_ids = {value for value in (run_id, genpm_run_id) if value}
        if run_id:
            try:
                return await self._client.get_dag_run(dag_id, run_id)
            except AirflowNotFound:
                logger.info(
                    "GET dag run missing for %s/%s; scanning recent runs",
                    dag_id,
                    run_id,
                )

        raw_list = await self._client.list_dag_runs(dag_id, limit=100)
        for item in raw_list.get("dag_runs", []) or []:
            item_id = _dag_run_id_from_payload(item)
            if item_id and item_id in lookup_ids:
                return item
            item_conf = item.get("conf")
            if not isinstance(item_conf, dict):
                continue
            conf_run_id = item_conf.get("genpm_run_id")
            if isinstance(conf_run_id, str) and conf_run_id in lookup_ids:
                return item

        raise AirflowNotFound(
            f"Dag run '{run_id or genpm_run_id}' not found for DAG '{dag_id}'",
            details={"dag_id": dag_id, "run_id": run_id, "genpm_run_id": genpm_run_id},
        )

    async def _ensure_dag_unpaused(self, dag_id: str) -> None:
        dag = await self._client.get_dag(dag_id)
        if not dag.get("is_paused"):
            return
        logger.info("Unpausing DAG %s before manual trigger", dag_id)
        await self._client.patch_dag(dag_id, is_paused=False)
        self._invalidate_dag_list_cache()
        dag = await self._client.get_dag(dag_id)
        if dag.get("is_paused"):
            raise AirflowIntegrationError(
                f"DAG '{dag_id}' is paused in Airflow and could not be unpaused.",
            )

    async def stop_dag_run(
        self, dag_id: str, run_id: str, *, triggered_by: str | None
    ) -> ActionResponse:
        note = _compose_note("Stopped via GenPM", triggered_by)
        await self._client.patch_dag_run_state(
            dag_id, run_id, state="failed", note=note
        )
        self._invalidate_dag_list_cache()
        return ActionResponse(
            run_id=run_id,
            message=f"Stopped DAG run '{run_id}'",
            airflow_status=200,
        )

    async def clear_dag_run(
        self, dag_id: str, run_id: str, *, triggered_by: str | None
    ) -> ActionResponse:
        await self._client.clear_task_instances(
            dag_id, run_id, task_ids=None, reset_dag_runs=True
        )
        self._invalidate_dag_list_cache()
        _ = triggered_by
        return ActionResponse(
            run_id=run_id,
            message=f"Cleared all tasks in run '{run_id}'",
            airflow_status=200,
        )

    async def clear_task_instance(
        self,
        dag_id: str,
        run_id: str,
        task_id: str,
        *,
        downstream: bool = False,
        triggered_by: str | None,
    ) -> ActionResponse:
        await self._client.clear_task_instances(
            dag_id,
            run_id,
            task_ids=[task_id],
            include_downstream=downstream,
            reset_dag_runs=True,
        )
        self._invalidate_dag_list_cache()
        _ = triggered_by
        return ActionResponse(
            run_id=run_id,
            message=f"Cleared task '{task_id}'" + (" + downstream" if downstream else ""),
            airflow_status=200,
        )

    def _invalidate_dag_list_cache(self) -> None:
        self._dag_list_cache = None

    async def _fetch_dag_list_uncached(self) -> list[DagSummary]:
        dags_raw = await self._client.list_dags()
        dag_payloads: list[dict[str, Any]] = list(dags_raw.get("dags", []) or [])
        if not dag_payloads:
            return []

        start_date_gte = (datetime.now(tz=timezone.utc) - timedelta(hours=24)).isoformat()
        results = await asyncio.gather(
            *(
                self._fetch_dag_recent_window(str(d.get("dag_id") or ""), start_date_gte)
                for d in dag_payloads
            ),
            return_exceptions=True,
        )

        summaries: list[DagSummary] = []
        for dag_payload, fetched in zip(dag_payloads, results, strict=True):
            last_run: DagRunSummary | None
            stats: DagStats
            if isinstance(fetched, BaseException):
                logger.warning(
                    "Failed to fetch 24h window for dag_id=%s: %s",
                    dag_payload.get("dag_id"),
                    fetched,
                )
                last_run, stats = None, DagStats()
            else:
                last_run, stats = fetched
            summaries.append(
                map_dag_summary(dag_payload, last_run=last_run, stats=stats)
            )
        return summaries

    async def _fetch_dag_recent_window(
        self, dag_id: str, start_date_gte: str
    ) -> tuple[DagRunSummary | None, DagStats]:
        if not dag_id:
            return None, DagStats()
        try:
            raw = await self._client.list_dag_runs(
                dag_id, limit=200, start_date_gte=start_date_gte
            )
        except AirflowNotFound:
            return None, DagStats()
        runs = [map_dag_run(r) for r in raw.get("dag_runs", []) or []]
        stats = _aggregate_stats_from_runs(runs)
        last_run: DagRunSummary | None = None
        if runs:
            last_run = max(
                runs,
                key=lambda r: r.start_date or r.logical_date or datetime.min.replace(tzinfo=timezone.utc),
            )
        return last_run, stats


def _aggregate_stats_from_runs(runs: list[DagRunSummary]) -> DagStats:
    success = sum(1 for r in runs if r.status == DagRunStatus.SUCCESS)
    failed = sum(1 for r in runs if r.status == DagRunStatus.FAILED)
    running = sum(1 for r in runs if r.status == DagRunStatus.RUNNING)
    return DagStats(success=success, failed=failed, running=running, total=len(runs))


def _dag_run_id_from_payload(raw: dict[str, Any]) -> str:
    for key in ("dag_run_id", "run_id"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _compose_note(note: str | None, triggered_by: str | None) -> str | None:
    parts: list[str] = []
    if note:
        parts.append(note.strip())
    if triggered_by:
        parts.append(f"triggered_by={triggered_by}")
    return " | ".join(parts) if parts else None
