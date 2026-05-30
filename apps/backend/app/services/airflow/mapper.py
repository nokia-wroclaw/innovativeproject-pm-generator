import ast
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from app.db.schemas import LogLevel, RunType
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


def normalize_task_status(raw_state: str | None) -> TaskStatus:
    if raw_state is None:
        return TaskStatus.NONE
    return _TASK_STATUS_MAP.get(raw_state.lower(), TaskStatus.NONE)


def normalize_dag_run_status(raw_state: str | None) -> DagRunStatus:
    if raw_state is None:
        return DagRunStatus.QUEUED
    return _DAG_RUN_STATUS_MAP.get(raw_state.lower(), DagRunStatus.QUEUED)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
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
            dt = dt.replace(tzinfo=UTC)
        return dt
    return None


def _duration_ms(start: datetime | None, end: datetime | None, raw: Any = None) -> int | None:
    if isinstance(raw, int | float) and raw >= 0:
        return int(raw * 1000)
    if start is not None and end is not None:
        return max(0, int((end - start).total_seconds() * 1000))
    return None


def _schedule_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("value") or value.get("expression")
    return str(value)


def _run_type(value: str | None) -> RunType:
    allowed: frozenset[RunType] = frozenset({"manual", "scheduled", "backfill", "asset_triggered"})
    if value in allowed:
        return cast(RunType, value)
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


def map_dag_run(raw: dict[str, Any]) -> DagRunSummary:
    start = _parse_dt(raw.get("start_date"))
    end = _parse_dt(raw.get("end_date"))
    logical = _parse_dt(raw.get("logical_date"))
    state = raw.get("state")
    return DagRunSummary(
        run_id=str(raw.get("dag_run_id") or ""),
        logical_date=logical or start or datetime.now(tz=UTC),
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
        schedule=_schedule_str(raw.get("timetable_summary")),
        next_run_at=_parse_dt(raw.get("next_dagrun_run_after")),
        last_run=last_run,
        stats_24h=stats or DagStats(),
    )


def map_dag_graph(raw: dict[str, Any], *, dag_id: str | None = None) -> DagGraph:
    nodes: list[TaskNode] = []
    edges: list[TaskEdge] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

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
        _append_node(
            nodes,
            seen_nodes,
            task_id=task_id,
            label=label,
            operator=operator,
            is_group=bool(task.get("is_mapped", False)),
            trigger_rule=str(task.get("trigger_rule") or "all_success"),
            retries_max=int(task.get("retries") or 0),
            depends_on_past=bool(task.get("depends_on_past", False)),
        )
        for downstream in task.get("downstream_task_ids", []) or []:
            _append_edge(edges, seen_edges, task_id, str(downstream))
        for upstream in task.get("upstream_task_ids", []) or []:
            _append_edge(edges, seen_edges, str(upstream), task_id)

    if dag_id:
        _apply_local_dag_file_fallback(dag_id, nodes, edges, seen_nodes, seen_edges)

    return DagGraph(nodes=nodes, edges=edges)


def _append_node(
    nodes: list[TaskNode],
    seen_nodes: set[str],
    *,
    task_id: str,
    label: str | None = None,
    operator: str = "Operator",
    is_group: bool = False,
    trigger_rule: str = "all_success",
    retries_max: int = 0,
    depends_on_past: bool = False,
) -> None:
    if task_id in seen_nodes:
        return
    seen_nodes.add(task_id)
    nodes.append(
        TaskNode(
            task_id=task_id,
            label=label or task_id,
            operator=operator,
            is_group=is_group,
            trigger_rule=trigger_rule,
            retries_max=retries_max,
            depends_on_past=depends_on_past,
        )
    )


def _append_edge(
    edges: list[TaskEdge],
    seen_edges: set[tuple[str, str]],
    source: str,
    target: str,
) -> None:
    if not source or not target:
        return
    key = (source, target)
    if key in seen_edges:
        return
    seen_edges.add(key)
    edges.append(TaskEdge(source=source, target=target))


def _apply_local_dag_file_fallback(
    dag_id: str,
    nodes: list[TaskNode],
    edges: list[TaskEdge],
    seen_nodes: set[str],
    seen_edges: set[tuple[str, str]],
) -> None:
    parsed = _parse_local_dag_file(dag_id)
    if parsed is None:
        return

    tasks, dependencies = parsed
    local_task_ids = {
        str(task.get("task_id") or variable_name) for variable_name, task in tasks.items()
    }
    missing_local_tasks = local_task_ids - seen_nodes

    if missing_local_tasks and dependencies:
        # Airflow can briefly serve an older serialized graph after a DAG file
        # changes. In that case keep Airflow-only edges, but refresh edges among
        # tasks we can read from the local DAG file so newly inserted tasks show.
        edges[:] = [
            edge
            for edge in edges
            if edge.source not in local_task_ids or edge.target not in local_task_ids
        ]
        seen_edges.clear()
        seen_edges.update((edge.source, edge.target) for edge in edges)

    for variable_name, task in tasks.items():
        task_id = str(task.get("task_id") or variable_name)
        _append_node(
            nodes,
            seen_nodes,
            task_id=task_id,
            label=task_id,
            operator=str(task.get("operator") or "Operator"),
        )
    for source_var, target_var in dependencies:
        source = str(tasks.get(source_var, {}).get("task_id") or source_var)
        target = str(tasks.get(target_var, {}).get("task_id") or target_var)
        _append_edge(edges, seen_edges, source, target)


def _parse_local_dag_file(
    dag_id: str,
) -> tuple[dict[str, dict[str, str]], list[tuple[str, str]]] | None:
    dag_file = _find_local_dag_file(dag_id)
    if dag_file is None:
        return None
    try:
        tree = ast.parse(dag_file.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.debug("Could not read local DAG file %s: %s", dag_file, exc)
        return None
    except SyntaxError as exc:
        logger.debug("Could not parse local DAG file %s: %s", dag_file, exc)
        return None

    tasks: dict[str, dict[str, str]] = {}
    dependencies: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            _collect_task_assignment(node, tasks)
        elif isinstance(node, ast.Expr):
            dependencies.extend(_extract_shift_dependencies(node.value))

    if not tasks:
        return None
    return tasks, dependencies


def _find_local_dag_file(dag_id: str) -> Path | None:
    roots = _candidate_dag_roots()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if f'dag_id="{dag_id}"' in text or f"dag_id='{dag_id}'" in text:
                return path
    return None


def _candidate_dag_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(value: str | os.PathLike[str] | None) -> None:
        if not value:
            return
        path = Path(value).expanduser()
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    add(os.getenv("AIRFLOW_DAGS_DIR"))
    add("/opt/airflow/dags")
    add("/app/dags")
    add("/app/apps/airflow/dags")
    add(Path.cwd() / "apps" / "airflow" / "dags")

    # Support local runs from either the repo root or apps/backend.
    for ancestor in Path(__file__).resolve().parents:
        add(ancestor / "dags")
        add(ancestor / "apps" / "airflow" / "dags")

    return roots


def _collect_task_assignment(
    node: ast.Assign,
    tasks: dict[str, dict[str, str]],
) -> None:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return
    if not isinstance(node.value, ast.Call):
        return
    operator = _call_name(node.value.func)
    if not operator or not operator.endswith("Operator"):
        return
    task_id = _keyword_string(node.value, "task_id") or node.targets[0].id
    tasks[node.targets[0].id] = {"task_id": task_id, "operator": operator}


def _extract_shift_dependencies(node: ast.AST) -> list[tuple[str, str]]:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.RShift):
        return []
    left = _dependency_terms(node.left)
    right = _dependency_terms(node.right)
    dependencies = [(source, target) for source in left for target in right]
    dependencies.extend(_extract_shift_dependencies(node.left))
    dependencies.extend(_extract_shift_dependencies(node.right))
    return dependencies


def _dependency_terms(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.List | ast.Tuple):
        terms: list[str] = []
        for item in node.elts:
            terms.extend(_dependency_terms(item))
        return terms
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
        return _dependency_terms(node.right)
    return []


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _keyword_string(node: ast.Call, keyword_name: str) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return keyword.value.value
    return None


def _operator_from_class_ref(class_ref: Any) -> str | None:
    if isinstance(class_ref, dict):
        return class_ref.get("class_name")
    return None


def map_dag_details(
    *, summary: DagSummary, graph: DagGraph, recent_runs: list[DagRunSummary]
) -> DagDetails:
    return DagDetails(summary=summary, graph=graph, recent_runs=recent_runs)


def map_task_instance(raw: dict[str, Any]) -> TaskInstance:
    start = _parse_dt(raw.get("start_date"))
    end = _parse_dt(raw.get("end_date"))
    state = raw.get("state")
    return TaskInstance(
        task_id=str(raw.get("task_id") or ""),
        run_id=str(raw.get("dag_run_id") or ""),
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
        executor_config=_coerce_executor_config(raw.get("executor_config")),
        note=raw.get("note") or None,
    )


def _coerce_executor_config(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"raw": value}
    return {}


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
    continuation = raw.get("continuation_token") or None
    content = raw.get("content")

    lines: list[LogLine] = []

    if isinstance(content, str):
        for text in content.splitlines():
            if not text.strip():
                continue
            lines.append(_parse_log_line(None, text))

    elif isinstance(content, list):
        for entry in content:
            try:
                _consume_log_entry(entry, lines)
            except Exception as exc:
                logger.warning("Skipping malformed log entry: %s (%r)", exc, entry)
                continue

    if not lines and content:
        logger.warning(
            "map_log_chunk: unknown content shape (type=%s); " "emitting raw fallback. Sample=%s",
            type(content).__name__,
            repr(content[0])[:300]
            if isinstance(content, list) and content
            else repr(content)[:300],
        )
        if isinstance(content, list):
            for entry in content:
                text = (
                    entry
                    if isinstance(entry, str)
                    else json.dumps(entry, default=str, ensure_ascii=False)
                )
                for line in str(text).splitlines() or [str(text)]:
                    if line.strip():
                        lines.append(LogLine(message=line))

    if isinstance(content, str | list):
        content_len: int | str = len(content)
    else:
        content_len = "n/a"

    logger.info(
        "map_log_chunk: content_type=%s len=%s -> %d lines",
        type(content).__name__,
        content_len,
        len(lines),
    )

    return LogChunk(
        try_number=try_number,
        seq=seq,
        lines=lines,
        has_more=continuation is not None,
        continuation=continuation,
    )


def _consume_log_entry(entry: Any, lines: list[LogLine]) -> None:
    if isinstance(entry, dict):
        event_text = str(entry.get("event") or "").rstrip("\n")
        if not event_text.strip():
            return
        lines.append(
            LogLine(
                timestamp=_parse_dt(entry.get("timestamp")),
                level=_normalize_log_level(entry.get("level")),
                source=str(entry.get("logger") or entry.get("source") or "") or None,
                message=event_text,
            )
        )
        return
    if isinstance(entry, str):
        for text in entry.splitlines():
            if not text.strip():
                continue
            lines.append(_parse_log_line(None, text))
        return


_ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _normalize_log_level(value: Any) -> LogLevel | None:
    if not value:
        return None
    candidate = str(value).upper().strip()
    if candidate in _ALLOWED_LOG_LEVELS:
        return cast(LogLevel, candidate)
    if candidate == "WARN":
        return "WARNING"
    if candidate == "FATAL":
        return "CRITICAL"
    return None


def _parse_log_line(source: str | None, text: str) -> LogLine:
    match = _LOG_PREFIX_RE.match(text)
    if not match:
        return LogLine(timestamp=None, level=None, source=source, message=text)
    level_raw = match.group("level")
    level: LogLevel | None = cast(LogLevel, level_raw) if level_raw in _ALLOWED_LOG_LEVELS else None
    return LogLine(
        timestamp=_parse_dt(match.group("ts")),
        level=level,
        source=match.group("source") or source,
        message=match.group("msg"),
    )
