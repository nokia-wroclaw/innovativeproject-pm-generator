import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, cast

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
from app.services.airflow.client import AirflowClient
from app.services.airflow.config import AirflowSettings
from app.services.airflow.errors import AirflowIntegrationError, AirflowNotFound
from app.services.airflow.mapper import (
    map_dag_graph,
    map_log_chunk,
)

logger = logging.getLogger(__name__)


_ACTIVE_RUN_TTL = 5.0
_TERMINAL_RUN_TTL = 3600.0
_TERMINAL_RUN_STATES = frozenset({"success", "failed"})
_TERMINAL_TASK_STATES = frozenset({"success", "failed", "skipped", "upstream_failed"})


class AirflowService:
    def __init__(self, *, client: AirflowClient, settings: AirflowSettings) -> None:
        self._client = client
        self._settings = settings
        self._dag_list_cache: tuple[float, list[DagSummary]] | None = None
        self._dag_list_lock = asyncio.Lock()
        self._run_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._task_list_cache: dict[tuple[str, str], tuple[float, list[TaskInstance]]] = {}
        self._task_cache: dict[tuple[str, str, str], tuple[float, TaskInstance]] = {}

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
            self._client.list_dag_runs(dag_id, limit=10, order_by="-logical_date"),
        )

        graph = map_dag_graph(tasks_raw)
        recent_runs = [DagRunSummary.model_validate(r) for r in runs_raw.get("dag_runs", []) or []]
        stats = _aggregate_stats_from_runs(recent_runs)

        dag_raw["last_run"] = recent_runs[0] if recent_runs else None
        dag_raw["stats_24h"] = stats
        summary = DagSummary.model_validate(dag_raw)

        return DagDetails(summary=summary, graph=graph, recent_runs=recent_runs)

    async def list_dag_runs(
        self, dag_id: str, limit: int = 50, offset: int = 0, order_by: str = "-start_date"
    ) -> list[DagRunSummary]:
        raw = await self._client.list_dag_runs(
            dag_id, limit=limit, offset=offset, order_by=order_by
        )
        return [DagRunSummary.model_validate(r) for r in raw.get("dag_runs", []) or []]

    async def get_dag_run(self, dag_id: str, run_id: str) -> DagRunSummary:
        key = (dag_id, run_id)
        now = time.monotonic()
        cached = self._run_cache.get(key)
        if cached:
            ts, raw = cached
            state = str(raw.get("state") or "")
            ttl = _TERMINAL_RUN_TTL if state in _TERMINAL_RUN_STATES else _ACTIVE_RUN_TTL
            if now - ts < ttl:
                return DagRunSummary.model_validate(raw)

        raw = await self._fetch_dag_run_raw(
            dag_id,
            run_id,
            genpm_run_id=run_id if run_id.startswith("genpm_") else None,
        )
        self._run_cache[key] = (time.monotonic(), raw)
        return DagRunSummary.model_validate(raw)

    async def list_task_instances(self, dag_id: str, run_id: str) -> list[TaskInstance]:
        key = (dag_id, run_id)
        now = time.monotonic()
        cached = self._task_list_cache.get(key)
        if cached:
            ts, items = cached
            all_terminal = all(
                str(getattr(ti, "status", "") or "") in _TERMINAL_TASK_STATES for ti in items
            )
            ttl = _TERMINAL_RUN_TTL if (items and all_terminal) else _ACTIVE_RUN_TTL
            if now - ts < ttl:
                return items

        raw = await self._client.list_task_instances(dag_id, run_id)
        raw_items = raw.get("task_instances", []) or []
        logger.info(
            "list_task_instances dag=%s run=%s -> %d items (keys=%s)",
            dag_id,
            run_id,
            len(raw_items),
            list(raw.keys()),
        )
        items = [TaskInstance.model_validate(ti) for ti in raw_items]
        self._task_list_cache[key] = (time.monotonic(), items)
        return items

    async def get_task_instance(self, dag_id: str, run_id: str, task_id: str) -> TaskInstance:
        key = (dag_id, run_id, task_id)
        now = time.monotonic()
        cached = self._task_cache.get(key)
        if cached:
            ts, ti = cached
            state = str(getattr(ti, "status", "") or "")
            ttl = _TERMINAL_RUN_TTL if state in _TERMINAL_TASK_STATES else _ACTIVE_RUN_TTL
            if now - ts < ttl:
                return ti

        raw = await self._client.get_task_instance(dag_id, run_id, task_id)
        ti = TaskInstance.model_validate(raw)
        self._task_cache[key] = (time.monotonic(), ti)
        return ti

    async def list_task_tries(self, dag_id: str, run_id: str, task_id: str) -> list[TaskTry]:
        raw = await self._client.list_task_tries(dag_id, run_id, task_id)
        return [TaskTry.model_validate(t) for t in (raw.get("task_instances", []))]

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
        raw = await self._client.get_task_logs(dag_id, run_id, task_id, try_number, token=token)
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
        logical_date = body.logical_date.isoformat() if body.logical_date else None
        raw = await self._client.trigger_dag(
            dag_id,
            conf=body.conf,
            run_id=body.run_id,
            logical_date=logical_date,
            note=note,
        )
        self._invalidate_dag_list_cache()
        requested_id = str(body.run_id or "")
        resolved_raw = await self._fetch_dag_run_raw(
            dag_id,
            _run_id_from_payload(raw) or requested_id,
            genpm_run_id=requested_id or None,
        )
        run_id = _run_id_from_payload(resolved_raw)
        if not run_id:
            raise AirflowIntegrationError(
                f"Airflow triggered '{dag_id}' but returned no run_id.",
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
            item_id = _run_id_from_payload(item)
            if item_id and item_id in lookup_ids:
                return cast(dict[str, Any], item)
            item_conf = item.get("conf")
            if not isinstance(item_conf, dict):
                continue
            conf_run_id = item_conf.get("genpm_run_id")
            if isinstance(conf_run_id, str) and conf_run_id in lookup_ids:
                return cast(dict[str, Any], item)

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
        await self._client.patch_dag_run_state(dag_id, run_id, state="failed", note=note)
        self._invalidate_dag_list_cache()
        self._invalidate_run_caches(dag_id, run_id)
        return ActionResponse(
            run_id=run_id,
            message=f"Stopped DAG run '{run_id}'",
            airflow_status=200,
        )

    async def clear_dag_run(
        self, dag_id: str, run_id: str, *, triggered_by: str | None
    ) -> ActionResponse:
        await self._client.clear_task_instances(dag_id, run_id, task_ids=None, reset_dag_runs=True)
        self._invalidate_dag_list_cache()
        self._invalidate_run_caches(dag_id, run_id)
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
        self._invalidate_run_caches(dag_id, run_id)
        _ = triggered_by
        return ActionResponse(
            run_id=run_id,
            message=f"Cleared task '{task_id}'" + (" + downstream" if downstream else ""),
            airflow_status=200,
        )

    def _invalidate_dag_list_cache(self) -> None:
        self._dag_list_cache = None

    def _invalidate_run_caches(self, dag_id: str, run_id: str) -> None:
        self._run_cache.pop((dag_id, run_id), None)
        self._task_list_cache.pop((dag_id, run_id), None)
        keys_to_drop = [k for k in self._task_cache if k[0] == dag_id and k[1] == run_id]
        for k in keys_to_drop:
            self._task_cache.pop(k, None)

    async def _fetch_dag_list_uncached(self) -> list[DagSummary]:
        dags_raw = await self._client.list_dags()
        dag_payloads: list[dict[str, Any]] = list(dags_raw.get("dags", []) or [])
        if not dag_payloads:
            return []

        start_date_gte = (datetime.now(tz=UTC) - timedelta(hours=24)).isoformat()
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
            dag_payload["last_run"] = last_run
            dag_payload["stats_24h"] = stats
            summaries.append(DagSummary.model_validate(dag_payload))
        return summaries

    async def _fetch_dag_recent_window(
        self, dag_id: str, start_date_gte: str
    ) -> tuple[DagRunSummary | None, DagStats]:
        if not dag_id:
            return None, DagStats()
        try:
            raw = await self._client.list_dag_runs(dag_id, limit=200, start_date_gte=start_date_gte)
        except AirflowNotFound:
            return None, DagStats()
        runs = [DagRunSummary.model_validate(r) for r in raw.get("dag_runs", []) or []]
        stats = _aggregate_stats_from_runs(runs)
        last_run: DagRunSummary | None = None
        if runs:
            last_run = max(
                runs,
                key=lambda r: r.start_date or r.logical_date or datetime.min.replace(tzinfo=UTC),
            )
        return last_run, stats


def _aggregate_stats_from_runs(runs: list[DagRunSummary]) -> DagStats:
    success = sum(1 for r in runs if r.status == DagRunStatus.SUCCESS)
    failed = sum(1 for r in runs if r.status == DagRunStatus.FAILED)
    running = sum(1 for r in runs if r.status == DagRunStatus.RUNNING)
    return DagStats(success=success, failed=failed, running=running, total=len(runs))


def _run_id_from_payload(raw: dict[str, Any]) -> str:
    value = raw.get("dag_run_id")
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
