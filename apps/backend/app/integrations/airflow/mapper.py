"""Translates Airflow API v2 payloads into our Pydantic DTOs.

Pure functions, no side effects. Keeping the mapping isolated here means
the rest of the codebase never sees raw Airflow JSON. The functions accept
the *raw* dicts returned by ``AirflowClient`` and return Pydantic models
defined in :mod:`app.models.dags`.

Defensive: Airflow's API is occasionally inconsistent across versions
(field names sometimes change between minor releases). We use ``.get()``
extensively and fall back to safe defaults.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.models.dags import (
    DagDetails,
    DagGraph,
    DagRunStatus,
    DagRunSummary,
    DagStats,
    DagSummary,
    LogChunk,
    LogLine,
    TaskEdge,
    TaskInstance,
    TaskNode,
    TaskStatus,
    TaskTry,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Status normalisation (contract §2)
# ─────────────────────────────────────────────────────────────────────────────

_TASK_STATUS_MAP: dict[str, TaskStatus] = {
    "success": TaskStatus.SUCCESS,
    "running": TaskStatus.RUNNING,
    "failed": TaskStatus.FAILED,
    "upstream_failed": TaskStatus.FAILED,
    "up_for_retry": TaskStatus.UP_FOR_RETRY,
    "up_for_reschedule": TaskStatus.UP_FOR_RETRY,
    "restarting": TaskStatus.UP_FOR_RETRY,
    "queued": TaskStatus.QUEUED,
    "scheduled": TaskStatus.QUEUED,
    "deferred": TaskStatus.QUEUED,
    "skipped": TaskStatus.SKIPPED,
    "removed": TaskStatus.NONE,
    "none": TaskStatus.NONE,
}

_DAG_RUN_STATUS_MAP: dict[str, DagRunStatus] = {
    "success": DagRunStatus.SUCCESS,
    "running": DagRunStatus.RUNNING,
    "failed": DagRunStatus.FAILED,
    "queued": DagRunStatus.QUEUED,
}


def normalize_task_status(raw_state: str | None) -> TaskStatus:
    if raw_state is None:
        return TaskStatus.NONE
    return _TASK_STATUS_MAP.get(raw_state.lower(), TaskStatus.NONE)


def normalize_dag_run_status(raw_state: str | None) -> DagRunStatus:
    if raw_state is None:
        return DagRunStatus.QUEUED
    return _DAG_RUN_STATUS_MAP.get(raw_state.lower(), DagRunStatus.QUEUED)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        cleaned = value.rstrip("Z")
        suffix = "+00:00" if value.endswith("Z") else ""
        try:
            dt = datetime.fromisoformat(cleaned + suffix)
        except ValueError:
            try:
                dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
            except ValueError:
                logger.debug("Could not parse Airflow timestamp: %r", value)
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _duration_ms(start: datetime | None, end: datetime | None, raw: Any = None) -> int | None:
    if isinstance(raw, (int, float)) and raw >= 0:
        return int(raw * 1000)
    if start is not None and end is not None:
        return max(0, int((end - start).total_seconds() * 1000))
    return None


def _schedule_str(value: Any) -> str | None:
    """Airflow returns schedule either as a string or a typed object."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("value") or value.get("expression")
    return str(value)


def _run_type(value: str | None) -> str:
    allowed = {"manual", "scheduled", "backfill", "asset_triggered"}
    if value in allowed:
        return value
    if value == "dataset_triggered":
        return "asset_triggered"
    return "manual"


def _tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for tag in value:
        if isinstance(tag, str):
            out.append(tag)
        elif isinstance(tag, dict) and "name" in tag:
            out.append(str(tag["name"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DAG run
# ─────────────────────────────────────────────────────────────────────────────

def map_dag_run(raw: dict[str, Any]) -> DagRunSummary:
    start = _parse_dt(raw.get("start_date"))
    end = _parse_dt(raw.get("end_date"))
    logical = _parse_dt(raw.get("logical_date") or raw.get("execution_date"))
    state = raw.get("state")
    return DagRunSummary(
        run_id=str(raw.get("dag_run_id") or raw.get("run_id") or ""),
        logical_date=logical or start or datetime.now(tz=timezone.utc),
        start_date=start,
        end_date=end,
        duration_ms=_duration_ms(start, end),
        status=normalize_dag_run_status(state),
        raw_state=str(state or "queued"),
        run_type=_run_type(raw.get("run_type")),
        triggered_by=_triggered_by_from_note(raw.get("note")),
    )


_TRIGGERED_BY_RE = re.compile(r"triggered_by=([\w.\-]+)")


def _triggered_by_from_note(note: Any) -> str | None:
    if not isinstance(note, str):
        return None
    match = _TRIGGERED_BY_RE.search(note)
    return match.group(1) if match else None


# ─────────────────────────────────────────────────────────────────────────────
# DAG summary
# ─────────────────────────────────────────────────────────────────────────────

def map_dag_summary(
    raw: dict[str, Any],
    *,
    last_run: DagRunSummary | None = None,
    stats: DagStats | None = None,
) -> DagSummary:
    dag_id = str(raw.get("dag_id") or "")
    display_name = str(raw.get("dag_display_name") or dag_id)
    return DagSummary(
        dag_id=dag_id,
        display_name=display_name,
        description=raw.get("description") or None,
        owners=[str(o) for o in (raw.get("owners") or [])],
        tags=_tags(raw.get("tags")),
        is_paused=bool(raw.get("is_paused", False)),
        is_active=bool(raw.get("is_active", True)),
        schedule=_schedule_str(
            raw.get("schedule_interval") or raw.get("timetable_summary")
        ),
        next_run_at=_parse_dt(raw.get("next_dagrun") or raw.get("next_dagrun_run_after")),
        last_run=last_run,
        stats_24h=stats or DagStats(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph (tasks + edges)
# ─────────────────────────────────────────────────────────────────────────────

def map_dag_graph(raw: dict[str, Any]) -> DagGraph:
    """Builds a graph from ``GET /api/v2/dags/{id}/tasks``.

    Each task carries ``downstream_task_ids`` which we use to materialise edges.
    """
    nodes: list[TaskNode] = []
    edges: list[TaskEdge] = []

    for task in raw.get("tasks", []) or []:
        task_id = str(task.get("task_id") or "")
        if not task_id:
            continue
        label = str(task.get("task_display_name") or task_id)
        operator = str(
            task.get("operator_name")
            or _operator_from_class_ref(task.get("class_ref"))
            or "Operator"
        )
        nodes.append(
            TaskNode(
                task_id=task_id,
                label=label,
                operator=operator,
                is_group=bool(task.get("is_mapped", False)),
                trigger_rule=str(task.get("trigger_rule") or "all_success"),
                retries_max=int(task.get("retries") or 0),
                depends_on_past=bool(task.get("depends_on_past", False)),
            )
        )
        for downstream in task.get("downstream_task_ids", []) or []:
            edges.append(TaskEdge(source=task_id, target=str(downstream)))

    return DagGraph(nodes=nodes, edges=edges)


def _operator_from_class_ref(class_ref: Any) -> str | None:
    if isinstance(class_ref, dict):
        return class_ref.get("class_name")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DAG details
# ─────────────────────────────────────────────────────────────────────────────

def map_dag_details(
    *, summary: DagSummary, graph: DagGraph, recent_runs: list[DagRunSummary]
) -> DagDetails:
    return DagDetails(summary=summary, graph=graph, recent_runs=recent_runs)


# ─────────────────────────────────────────────────────────────────────────────
# Task instance / tries
# ─────────────────────────────────────────────────────────────────────────────

def map_task_instance(raw: dict[str, Any]) -> TaskInstance:
    start = _parse_dt(raw.get("start_date"))
    end = _parse_dt(raw.get("end_date"))
    state = raw.get("state")
    return TaskInstance(
        task_id=str(raw.get("task_id") or ""),
        run_id=str(raw.get("dag_run_id") or raw.get("run_id") or ""),
        status=normalize_task_status(state),
        raw_state=str(state or "none"),
        try_number=int(raw.get("try_number") or 0),
        max_tries=int(raw.get("max_tries") or 0),
        start_date=start,
        end_date=end,
        duration_ms=_duration_ms(start, end, raw.get("duration")),
        operator=str(raw.get("operator") or "Operator"),
        pool=str(raw.get("pool") or "default_pool"),
        queue=str(raw.get("queue") or "default"),
        executor_config=raw.get("executor_config") or {},
        note=raw.get("note") or None,
    )


def map_task_try(raw: dict[str, Any]) -> TaskTry:
    start = _parse_dt(raw.get("start_date"))
    end = _parse_dt(raw.get("end_date"))
    return TaskTry(
        try_number=int(raw.get("try_number") or 0),
        status=normalize_task_status(raw.get("state")),
        start_date=start,
        end_date=end,
        duration_ms=_duration_ms(start, end, raw.get("duration")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logs
# ─────────────────────────────────────────────────────────────────────────────

# Best-effort parser for the common Airflow log prefix:
#   [2026-05-20T15:00:18.345+0000] {operator.py:174} INFO - message text
_LOG_PREFIX_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s*"
    r"(?:\{(?P<source>[^}]+)\}\s*)?"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*-\s*"
    r"(?P<msg>.*)$"
)


def map_log_chunk(
    raw: dict[str, Any],
    *,
    try_number: int,
    seq: int,
) -> LogChunk:
    """Maps an Airflow ``/logs/{try}`` JSON response into a LogChunk.

    Airflow returns various shapes depending on version & log handler:
      * ``{"content": "string\nwith\nlines", "continuation_token": "..."}``
      * ``{"content": [[["source", "line text"], ...]], "continuation_token": "..."}``
    """
    continuation = raw.get("continuation_token") or None
    content = raw.get("content")

    raw_lines: list[tuple[str | None, str]] = []
    if isinstance(content, str):
        for line in content.splitlines():
            raw_lines.append((None, line))
    elif isinstance(content, list):
        for outer in content:
            if not isinstance(outer, list):
                continue
            for entry in outer:
                if isinstance(entry, list) and len(entry) >= 2:
                    src, text = entry[0], entry[1]
                    for line in str(text).splitlines():
                        raw_lines.append((str(src) if src else None, line))
                elif isinstance(entry, str):
                    for line in entry.splitlines():
                        raw_lines.append((None, line))

    lines = [_parse_log_line(src, text) for src, text in raw_lines if text.strip()]

    return LogChunk(
        try_number=try_number,
        seq=seq,
        lines=lines,
        has_more=continuation is not None,
        continuation=continuation,
    )


def _parse_log_line(source: str | None, text: str) -> LogLine:
    match = _LOG_PREFIX_RE.match(text)
    if not match:
        return LogLine(timestamp=None, level=None, source=source, message=text)
    return LogLine(
        timestamp=_parse_dt(match.group("ts")),
        level=match.group("level"),  # type: ignore[arg-type]
        source=match.group("source") or source,
        message=match.group("msg"),
    )
