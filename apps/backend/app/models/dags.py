"""Pydantic DTOs for the DAG management feature.

These models are the single source of truth for the wire format between
our backend (FastAPI) and our frontend (Vue 3). They mirror the contract
defined in ``docs/architecture/dag-management.md`` §3.

Conventions:
    * Timestamps are timezone-aware ``datetime`` (Pydantic serialises them
      as ISO 8601 strings).
    * Durations are integers in milliseconds.
    * Statuses are project-level enums (see ``TaskStatus`` / ``DagRunStatus``),
      never raw Airflow strings — the raw value is preserved in ``raw_state``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """Project-level normalised task instance status (see contract §2.1)."""

    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"
    UP_FOR_RETRY = "up_for_retry"
    QUEUED = "queued"
    SKIPPED = "skipped"
    NONE = "none"


class DagRunStatus(str, Enum):
    """Project-level normalised DAG run status (see contract §2.2)."""

    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"
    QUEUED = "queued"


RunType = Literal["manual", "scheduled", "backfill", "asset_triggered"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class DagStats(BaseModel):
    """Aggregated DAG run counts over a configurable window (default 24h)."""

    model_config = ConfigDict(extra="forbid")

    success: int = 0
    failed: int = 0
    running: int = 0
    total: int = 0


class DagRunSummary(BaseModel):
    """Compact DAG run row used in listings and dropdowns."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    logical_date: datetime
    start_date: datetime | None = None
    end_date: datetime | None = None
    duration_ms: int | None = None
    status: DagRunStatus
    raw_state: str
    run_type: RunType
    triggered_by: str | None = None


class DagSummary(BaseModel):
    """One row of the dashboard DAG table."""

    model_config = ConfigDict(extra="forbid")

    dag_id: str
    display_name: str
    description: str | None = None
    owners: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_paused: bool
    is_active: bool
    schedule: str | None = None
    next_run_at: datetime | None = None
    last_run: DagRunSummary | None = None
    stats_24h: DagStats = Field(default_factory=DagStats)


class TaskNode(BaseModel):
    """Static task metadata (independent of any DAG run)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    label: str
    operator: str
    is_group: bool = False
    trigger_rule: str = "all_success"
    retries_max: int = 0
    depends_on_past: bool = False


class TaskEdge(BaseModel):
    """Directed dependency between two task nodes."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str


class DagGraph(BaseModel):
    """Graph payload consumed by Vue Flow."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[TaskNode]
    edges: list[TaskEdge]


class DagDetails(BaseModel):
    """Full payload for the DAG detail view."""

    model_config = ConfigDict(extra="forbid")

    summary: DagSummary
    graph: DagGraph
    recent_runs: list[DagRunSummary] = Field(default_factory=list)


class TaskInstance(BaseModel):
    """Stateful task instance within a specific DAG run."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    run_id: str
    status: TaskStatus
    raw_state: str
    try_number: int
    max_tries: int
    start_date: datetime | None = None
    end_date: datetime | None = None
    duration_ms: int | None = None
    operator: str
    pool: str = "default_pool"
    queue: str = "default"
    executor_config: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


class TaskTry(BaseModel):
    """One attempt of a task instance (for the log history dropdown)."""

    model_config = ConfigDict(extra="forbid")

    try_number: int
    status: TaskStatus
    start_date: datetime | None = None
    end_date: datetime | None = None
    duration_ms: int | None = None


class LogLine(BaseModel):
    """Parsed log line (best effort)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime | None = None
    level: LogLevel | None = None
    source: str | None = None
    message: str


class LogChunk(BaseModel):
    """One chunk of task logs (either SSE event payload or HTTP response)."""

    model_config = ConfigDict(extra="forbid")

    try_number: int
    seq: int
    lines: list[LogLine]
    has_more: bool = False
    continuation: str | None = None


class TriggerRequest(BaseModel):
    """POST /dags/{dag_id}/runs body."""

    model_config = ConfigDict(extra="forbid")

    conf: dict[str, Any] | None = None
    logical_date: datetime | None = None
    note: str | None = None


class ActionResponse(BaseModel):
    """Response shape for any mutating endpoint (trigger / clear / stop)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    message: str
    airflow_status: int


class ApiError(BaseModel):
    """Uniform error envelope returned by all /api/v1 endpoints."""

    model_config = ConfigDict(extra="forbid")

    error: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str
