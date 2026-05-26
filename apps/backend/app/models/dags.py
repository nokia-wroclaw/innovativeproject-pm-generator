from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.db.schemas import DagRunStatus, RunType, TaskStatus, LogLevel


class DagStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: int = 0
    failed: int = 0
    running: int = 0
    total: int = 0


class DagRunSummary(BaseModel):

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
    model_config = ConfigDict(extra="forbid")

    task_id: str
    label: str
    operator: str
    is_group: bool = False
    trigger_rule: str = "all_success"
    retries_max: int = 0
    depends_on_past: bool = False


class TaskEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str


class DagGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[TaskNode]
    edges: list[TaskEdge]


class DagDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: DagSummary
    graph: DagGraph
    recent_runs: list[DagRunSummary] = Field(default_factory=list)


class TaskInstance(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    try_number: int
    status: TaskStatus
    start_date: datetime | None = None
    end_date: datetime | None = None
    duration_ms: int | None = None


class LogLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime | None = None
    level: LogLevel | None = None
    source: str | None = None
    message: str


class LogChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    try_number: int
    seq: int
    lines: list[LogLine]
    has_more: bool = False
    continuation: str | None = None


class TriggerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conf: dict[str, Any] | None = None
    dag_run_id: str | None = None
    logical_date: datetime | None = None
    note: str | None = None


class ActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    message: str
    airflow_status: int


class ApiError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str
