import json
import re
from datetime import UTC, datetime
from typing import Annotated, Any, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from app.db.schemas import DagRunStatus, LogLevel, RunType, TaskStatus

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
    "scheduled": DagRunStatus.QUEUED,
    "up_for_retry": DagRunStatus.RUNNING,
    "restarting": DagRunStatus.RUNNING,
}


def parse_task_status(raw_state: Any) -> TaskStatus:
    if isinstance(raw_state, str):
        return _TASK_STATUS_MAP.get(str(raw_state).lower(), TaskStatus.NONE)
    return TaskStatus.NONE


def parse_dag_run_status(raw_state: Any) -> DagRunStatus:
    if isinstance(raw_state, str):
        return _DAG_RUN_STATUS_MAP.get(str(raw_state).lower(), DagRunStatus.QUEUED)
    return DagRunStatus.QUEUED


def parse_duration(v: Any) -> int | None:
    if isinstance(v, (int, float)) and v >= 0:
        return int(v * 1000)
    return None


def parse_run_type(v: Any) -> RunType:
    allowed: frozenset[RunType] = frozenset({"manual", "scheduled", "backfill", "asset_triggered"})
    if isinstance(v, str):
        if v in allowed:
            return cast(RunType, v)
        if v == "dataset_triggered":
            return "asset_triggered"
    return "manual"


def parse_triggered_by(v: Any) -> str | None:
    if not isinstance(v, str):
        return None
    match = re.search(r"triggered_by=([\w.\-]+)", v)
    return match.group(1) if match else None


def parse_owners(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(o) for o in v]
    return []


def parse_tags(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for tag in v:
        if isinstance(tag, str):
            out.append(tag)
        elif isinstance(tag, dict) and "name" in tag:
            out.append(str(tag["name"]))
    return out


def parse_schedule(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("value") or v.get("expression")
    return str(v)


def parse_executor_config(v: Any) -> dict[str, Any]:
    if v is None or v == "":
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return {"raw": v}
        return parsed if isinstance(parsed, dict) else {"raw": v}
    return {}


def parse_raw_state(v: Any) -> str:
    return str(v or "queued")


def parse_task_raw_state(v: Any) -> str:
    return str(v or "none")


AirflowTaskStatus = Annotated[TaskStatus, BeforeValidator(parse_task_status)]
AirflowDagRunStatus = Annotated[DagRunStatus, BeforeValidator(parse_dag_run_status)]
AirflowDuration = Annotated[int | None, BeforeValidator(parse_duration)]
AirflowRunType = Annotated[RunType, BeforeValidator(parse_run_type)]
AirflowTriggeredBy = Annotated[str | None, BeforeValidator(parse_triggered_by)]
AirflowOwners = Annotated[list[str], BeforeValidator(parse_owners)]
AirflowTags = Annotated[list[str], BeforeValidator(parse_tags)]
AirflowSchedule = Annotated[str | None, BeforeValidator(parse_schedule)]
AirflowExecutorConfig = Annotated[dict[str, Any], BeforeValidator(parse_executor_config)]
AirflowRawState = Annotated[str, BeforeValidator(parse_raw_state)]
AirflowTaskRawState = Annotated[str, BeforeValidator(parse_task_raw_state)]


class DagStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: int = 0
    failed: int = 0
    running: int = 0
    total: int = 0


class DagRunSummary(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    run_id: str = Field(validation_alias="dag_run_id")
    logical_date: datetime | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    duration_ms: AirflowDuration = Field(default=None, validation_alias="duration")
    status: AirflowDagRunStatus = Field(default=DagRunStatus.QUEUED, validation_alias="state")
    raw_state: AirflowRawState = Field(default="queued", validation_alias="state")
    run_type: AirflowRunType = Field(default="manual", validation_alias="run_type")
    triggered_by: AirflowTriggeredBy = Field(default=None, validation_alias="note")
    conf: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _compute_after(self) -> "DagRunSummary":
        if self.logical_date is None:
            self.logical_date = self.start_date or datetime.now(UTC)
        if self.duration_ms is None and self.start_date and self.end_date:
            self.duration_ms = max(0, int((self.end_date - self.start_date).total_seconds() * 1000))
        return self


class DagSummary(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    dag_id: str
    display_name: str = Field(validation_alias="dag_display_name")
    description: str | None = None
    owners: AirflowOwners = Field(default_factory=list)
    tags: AirflowTags = Field(default_factory=list)
    is_paused: bool = False
    is_active: bool = True
    schedule: AirflowSchedule = Field(default=None, validation_alias="timetable_summary")
    next_run_at: datetime | None = Field(default=None, validation_alias="next_dagrun_run_after")
    last_run: DagRunSummary | None = None
    stats_24h: DagStats = Field(default_factory=DagStats)

    @model_validator(mode="before")
    @classmethod
    def _pre_parse(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("dag_display_name") and not data.get("display_name"):
                data["display_name"] = str(data.get("dag_id", ""))
        return data


class TaskNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    label: str
    operator: str
    is_group: bool = False
    trigger_rule: str = "all_success"
    retries_max: int = 0
    depends_on_past: bool = False


class TaskEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    target: str


class DagGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nodes: list[TaskNode]
    edges: list[TaskEdge]


class DagDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: DagSummary
    graph: DagGraph
    recent_runs: list[DagRunSummary] = Field(default_factory=list)


class TaskInstance(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    task_id: str
    run_id: str = Field(validation_alias="dag_run_id")
    status: AirflowTaskStatus = Field(default=TaskStatus.NONE, validation_alias="state")
    raw_state: AirflowTaskRawState = Field(default="none", validation_alias="state")
    try_number: int = 0
    max_tries: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None
    duration_ms: AirflowDuration = Field(default=None, validation_alias="duration")
    operator: str = "Operator"
    pool: str = "default_pool"
    queue: str = "default"
    executor_config: AirflowExecutorConfig = Field(default_factory=dict)
    note: str | None = None

    @model_validator(mode="after")
    def _compute_after(self) -> "TaskInstance":
        if self.duration_ms is None and self.start_date and self.end_date:
            self.duration_ms = max(0, int((self.end_date - self.start_date).total_seconds() * 1000))
        return self


class TaskTry(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    try_number: int = 0
    status: AirflowTaskStatus = Field(default=TaskStatus.NONE, validation_alias="state")
    start_date: datetime | None = None
    end_date: datetime | None = None
    duration_ms: AirflowDuration = Field(default=None, validation_alias="duration")

    @model_validator(mode="after")
    def _compute_after(self) -> "TaskTry":
        if self.duration_ms is None and self.start_date and self.end_date:
            self.duration_ms = max(0, int((self.end_date - self.start_date).total_seconds() * 1000))
        return self


class LogLine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime | None = None
    level: LogLevel | None = None
    source: str | None = None
    message: str


class LogChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    try_number: int
    seq: int
    lines: list[LogLine]
    has_more: bool = False
    continuation: str | None = None


class TriggerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    conf: dict[str, Any] | None = None
    run_id: str | None = Field(default=None, validation_alias="dag_run_id")
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
