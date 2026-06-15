import logging
import re
import uuid
from typing import Any

from fastapi import HTTPException

from app.db.schemas import DagRunStatus, Dataset, DatasetStatus, DatasetType, TaskStatus
from app.models.auth import TokenPayload
from app.models.dags import DagRunSummary, TriggerRequest
from app.models.s3 import DatasetVisualizationResponse, DatasetVisualizationStatusResponse
from app.services.airflow.errors import AirflowIntegrationError, AirflowNotFound
from app.services.airflow.runtime import get_airflow_service
from app.services.airflow.service import AirflowService
from app.services.s3.pm_schema import (
    unsupported_schema_payload,
    validate_pm_columns,
)
from app.services.s3.service import S3Service
from app.services.s3.visualization_artifacts import (
    VisualizationStorageError,
    delete_visualization_error_artifact,
    load_kpi_analysis_artifact,
    load_visualization_artifact,
    persist_unsupported_schema_artifact,
    status_from_artifact,
    visualization_artifact_key,
)
from app.services.spark_dag_conf import build_visualization_dag_conf

DATASET_VISUALIZATION_DAG_ID = "dataset_visualization_spark"
VISUALIZATION_SPARK_TASK_ID = "run_pm_visualization"
GENERATE_PIPELINE_DAG_ID = "generate_pipeline"
SPARK_VERSION_PATTERN = re.compile(r"Spark version:\s*(\S+)", re.IGNORECASE)

logger = logging.getLogger(__name__)


class VisualizationSchemaError(Exception):
    def __init__(self, payload: TokenPayload) -> None:
        self.payload = payload
        msg = str(payload.message) if payload.message else "Unsupported dataset schema"
        super().__init__(msg)


def check_dataset_pm_schema(s3_service: S3Service, dataset: Dataset) -> dict[str, Any] | None:
    """Return unsupported_schema payload, or None when columns match PM schema."""
    try:
        columns = s3_service.read_primary_column_names(dataset)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "Could not read columns for visualization schema check dataset_id=%s: %s",
            dataset.id,
            exc,
        )
        return None

    ok, missing = validate_pm_columns(columns)
    if ok:
        return None
    return unsupported_schema_payload(missing, present_columns=columns)


def _unsupported_status_response(
    *,
    dataset_id: int,
    payload: TokenPayload,
    run_id: str | None = None,
) -> DatasetVisualizationStatusResponse:
    return DatasetVisualizationStatusResponse(
        dataset_id=dataset_id,
        dag_id=DATASET_VISUALIZATION_DAG_ID,
        run_id=run_id,
        status="unsupported_schema",
        message=payload.message,
        summary=payload,
    )


def is_raw_completed(dataset: Dataset) -> bool:
    status = str(dataset.status)
    dataset_type = str(dataset.type)
    return dataset_type == DatasetType.RAW.value and status == DatasetStatus.COMPLETED.value


async def trigger_dataset_visualization(
    dataset: Dataset,
    *,
    airflow: AirflowService | None = None,
    s3_service: S3Service | None = None,
    triggered_by: str | None = None,
) -> DatasetVisualizationResponse:
    if s3_service is not None and str(dataset.type) == DatasetType.RAW.value:
        schema_error = check_dataset_pm_schema(s3_service, dataset)
        if schema_error is not None:
            persist_unsupported_schema_artifact(dataset.s3_key, schema_error, str(dataset.type))
            raise VisualizationSchemaError(schema_error)
        delete_visualization_error_artifact(dataset.s3_key, str(dataset.type))

    service = airflow or get_airflow_service()
    run_id = f"genpm_viz_{dataset.id}_{uuid.uuid4().hex[:8]}"
    conf = build_visualization_dag_conf(
        dataset_id=dataset.id,
        s3_key=dataset.s3_key,
        file_name=dataset.file_name,
    ).to_airflow_conf()

    action = await service.trigger_dag(
        DATASET_VISUALIZATION_DAG_ID,
        body=TriggerRequest(
            conf=conf,
            run_id=run_id,
            note=f"Dashboard calculation for {dataset.file_name}",
        ),
        triggered_by=triggered_by,
    )
    effective_run_id = action.run_id or run_id
    return DatasetVisualizationResponse(
        message=action.message,
        dag_id=DATASET_VISUALIZATION_DAG_ID,
        airflow_run_id=effective_run_id,
    )


async def trigger_dataset_visualization_on_raw_completed(
    dataset: Dataset,
    *,
    s3_service: S3Service | None = None,
    triggered_by: str | None = None,
) -> None:
    if not is_raw_completed(dataset):
        return

    if s3_service is not None:
        schema_error = check_dataset_pm_schema(s3_service, dataset)
        if schema_error is not None:
            try:
                persist_unsupported_schema_artifact(dataset.s3_key, schema_error, str(dataset.type))
            except VisualizationStorageError as exc:
                logger.warning(
                    "Skipped visualization for dataset_id=%s: %s",
                    dataset.id,
                    exc,
                )
                return
            logger.info(
                "Skipped visualization for dataset_id=%s: unsupported PM schema",
                dataset.id,
            )
            return

    try:
        result = await trigger_dataset_visualization(
            dataset,
            s3_service=s3_service,
            triggered_by=triggered_by,
        )
        logger.info(
            "Triggered dataset visualization for raw dataset_id=%s run=%s",
            dataset.id,
            result.airflow_run_id,
        )
    except VisualizationStorageError as exc:
        logger.warning(
            "Skipped visualization for dataset_id=%s: %s",
            dataset.id,
            exc,
        )
    except VisualizationSchemaError:
        return
    except (AirflowNotFound, AirflowIntegrationError):
        logger.exception(
            "Failed to trigger dataset visualization for dataset_id=%s",
            dataset.id,
        )
    except Exception:
        logger.exception(
            "Unexpected error triggering dataset visualization for dataset_id=%s",
            dataset.id,
        )


def _conf_dataset_id(conf: Any) -> int | None:
    if not isinstance(conf, dict):
        return None
    raw = conf.get("dataset_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


_ACTIVE_RUN_STATES = frozenset({"running", "queued", "scheduled", "up_for_retry", "restarting"})


def _run_matches_dataset(run: DagRunSummary, dataset_id: int) -> bool:
    if _conf_dataset_id(run.conf) == dataset_id:
        return True
    run_id_prefix = f"genpm_viz_{dataset_id}_"
    return run.run_id.startswith(run_id_prefix)


def _find_latest_run_for_dataset(
    runs: list[DagRunSummary], dataset_id: int
) -> DagRunSummary | None:
    matched = [run for run in runs if _run_matches_dataset(run, dataset_id)]
    if not matched:
        return None

    for run in matched:
        if str(run.status) in _ACTIVE_RUN_STATES:
            return run

    return matched[0]


async def _collect_task_log_text(airflow: AirflowService, run_id: str, *, task_id: str) -> str:
    messages: list[str] = []
    token: str | None = None
    seq = 0
    try_number = 1

    while True:
        chunk = await airflow.get_task_logs_page(
            DATASET_VISUALIZATION_DAG_ID,
            run_id,
            task_id,
            try_number=try_number,
            token=token,
            seq=seq,
        )
        messages.extend(line.message for line in chunk.lines if line.message)
        if not chunk.has_more:
            break
        token = chunk.continuation
        seq += 1

    return "\n".join(messages)


def _parse_spark_version(log_text: str) -> str | None:
    match = SPARK_VERSION_PATTERN.search(log_text)
    return match.group(1) if match else None


def _extract_error_from_logs(log_text: str, *, max_len: int = 600) -> str | None:
    if not log_text:
        return None
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    for pattern in ("Traceback", "Error:", "Exception", "RuntimeError", "FAILED"):
        for index, line in enumerate(lines):
            if pattern in line:
                snippet = "\n".join(lines[index : index + 12])
                return snippet[:max_len]
    if lines:
        return lines[-1][:max_len]
    return None


async def _get_spark_task(airflow: AirflowService, run_id: str):
    try:
        return await airflow.get_task_instance(
            DATASET_VISUALIZATION_DAG_ID,
            run_id,
            VISUALIZATION_SPARK_TASK_ID,
        )
    except (AirflowNotFound, AirflowIntegrationError):
        return None


async def _failed_run_response(
    *,
    dataset_id: int,
    run_id: str,
    airflow: AirflowService,
    base_message: str,
    artifact: dict[str, Any] | None = None,
) -> DatasetVisualizationStatusResponse:
    message = base_message
    if artifact and artifact.get("message"):
        message = str(artifact["message"])
    else:
        try:
            log_text = await _collect_task_log_text(
                airflow, run_id, task_id=VISUALIZATION_SPARK_TASK_ID
            )
            log_hint = _extract_error_from_logs(log_text)
            if log_hint:
                message = f"{message} {log_hint}"
        except (AirflowNotFound, AirflowIntegrationError):
            pass

    return DatasetVisualizationStatusResponse(
        dataset_id=dataset_id,
        dag_id=DATASET_VISUALIZATION_DAG_ID,
        run_id=run_id,
        status="failed",
        message=message,
        summary=artifact,
    )


def _missing_artifact_response(
    *, dataset_id: int, dataset_s3_key: str, run_id: str
) -> DatasetVisualizationStatusResponse:
    expected_key = visualization_artifact_key(dataset_s3_key, "summary.json")
    return DatasetVisualizationStatusResponse(
        dataset_id=dataset_id,
        dag_id=DATASET_VISUALIZATION_DAG_ID,
        run_id=run_id,
        status="failed",
        message=(
            "The Airflow run finished but no visualization file exists in S3 "
            f"({expected_key}). "
            "Nothing is running now — use Retry to start a new pipeline run."
        ),
    )


async def _response_from_artifact(
    *,
    dataset_id: int,
    run_id: str,
    artifact: dict[str, Any],
    airflow: AirflowService,
    kpi_analysis: dict[str, Any] | None = None,
) -> DatasetVisualizationStatusResponse:
    resolved_status = status_from_artifact(artifact)
    spark_version = artifact.get("spark_version")
    if not spark_version:
        try:
            log_text = await _collect_task_log_text(
                airflow, run_id, task_id=VISUALIZATION_SPARK_TASK_ID
            )
            spark_version = _parse_spark_version(log_text)
        except (AirflowNotFound, AirflowIntegrationError):
            spark_version = None

    if resolved_status == "unsupported_schema":
        return DatasetVisualizationStatusResponse(
            dataset_id=dataset_id,
            dag_id=DATASET_VISUALIZATION_DAG_ID,
            run_id=run_id,
            status="unsupported_schema",
            spark_version=spark_version,
            message=artifact.get("message"),
            summary=artifact,
            kpi_analysis=kpi_analysis,
        )

    return DatasetVisualizationStatusResponse(
        dataset_id=dataset_id,
        dag_id=DATASET_VISUALIZATION_DAG_ID,
        run_id=run_id,
        status="success",
        spark_version=spark_version,
        message=artifact.get("message") or "Visualization completed.",
        summary=artifact,
        kpi_analysis=kpi_analysis,
    )


async def _get_generated_dataset_visualization_status(
    dataset_id: int,
    dataset: Dataset,
) -> DatasetVisualizationStatusResponse:
    """Return data-similarity results for a GENERATED dataset.

    The run_data_similarity Airflow task writes summary.json next to the generated
    parquet files.  We read that artifact directly — no separate visualization DAG is
    involved for generated datasets.
    """
    try:
        artifact = load_visualization_artifact(dataset.s3_key, str(dataset.type))
    except VisualizationStorageError as exc:
        return DatasetVisualizationStatusResponse(
            dataset_id=dataset_id,
            dag_id=GENERATE_PIPELINE_DAG_ID,
            status="unavailable",
            message=str(exc),
        )

    if artifact is not None:
        return DatasetVisualizationStatusResponse(
            dataset_id=dataset_id,
            dag_id=GENERATE_PIPELINE_DAG_ID,
            status="success",
            message="Data similarity analysis completed.",
            summary=artifact,
        )

    return DatasetVisualizationStatusResponse(
        dataset_id=dataset_id,
        dag_id=GENERATE_PIPELINE_DAG_ID,
        status="not_found",
        message=(
            "No data similarity results found yet. "
            "The generation pipeline may still be running or the run_data_similarity "
            "task did not produce a summary."
        ),
    )


async def get_dataset_visualization_status(
    dataset_id: int,
    dataset: Dataset,
    *,
    airflow: AirflowService | None = None,
    s3_service: S3Service | None = None,
) -> DatasetVisualizationStatusResponse:
    if str(dataset.type) == DatasetType.GENERATED.value:
        return await _get_generated_dataset_visualization_status(dataset_id, dataset)

    if s3_service is not None and str(dataset.type) == DatasetType.RAW.value:
        schema_error = check_dataset_pm_schema(s3_service, dataset)
        if schema_error is not None:
            return _unsupported_status_response(
                dataset_id=dataset_id,
                payload=schema_error,
            )

    service = airflow or get_airflow_service()

    try:
        runs = await service.list_dag_runs(
            DATASET_VISUALIZATION_DAG_ID,
            limit=100,
            order_by="-logical_date",
        )
    except (AirflowNotFound, AirflowIntegrationError) as exc:
        logger.warning("Could not list visualization DAG runs: %s", exc)
        return DatasetVisualizationStatusResponse(
            dataset_id=dataset_id,
            dag_id=DATASET_VISUALIZATION_DAG_ID,
            status="unavailable",
            message=(
                "Visualization service is unavailable (Airflow DAG not reachable). "
                "Check that the scheduler is running and the DAG is deployed."
            ),
        )

    run = _find_latest_run_for_dataset(runs, dataset_id)
    if run is None:
        return DatasetVisualizationStatusResponse(
            dataset_id=dataset_id,
            dag_id=DATASET_VISUALIZATION_DAG_ID,
            status="not_found",
            message="No visualization run found for this dataset yet.",
        )

    run_id = run.run_id
    status = str(run.status)

    try:
        artifact = load_visualization_artifact(dataset.s3_key, str(dataset.type))
    except VisualizationStorageError as exc:
        return DatasetVisualizationStatusResponse(
            dataset_id=dataset_id,
            dag_id=DATASET_VISUALIZATION_DAG_ID,
            status="unavailable",
            message=str(exc),
        )

    if artifact is not None and artifact.get("status") == "unsupported_schema":
        logger.info(
            "Ignoring stale unsupported_schema artifact for dataset_id=%s "
            "(dataset columns match PM schema under current rules)",
            dataset_id,
        )
        delete_visualization_error_artifact(dataset.s3_key, str(dataset.type))
        artifact = None

    if artifact is not None:
        kpi_analysis = None
        if artifact.get("status") == "success":
            kpi_analysis = load_kpi_analysis_artifact(dataset.s3_key, str(dataset.type))
        return await _response_from_artifact(
            dataset_id=dataset_id,
            run_id=run_id,
            artifact=artifact,
            airflow=service,
            kpi_analysis=kpi_analysis,
        )

    if status in {DagRunStatus.QUEUED.value, DagRunStatus.RUNNING.value}:
        spark_task = await _get_spark_task(service, run_id)
        if spark_task is not None:
            task_status = str(spark_task.status)
            if task_status == TaskStatus.FAILED.value:
                return await _failed_run_response(
                    dataset_id=dataset_id,
                    run_id=run_id,
                    airflow=service,
                    base_message="Spark visualization task failed in Airflow.",
                )
            if task_status == TaskStatus.SUCCESS.value:
                return _missing_artifact_response(
                    dataset_id=dataset_id,
                    dataset_s3_key=dataset.s3_key,
                    run_id=run_id,
                )

        if run.end_date is not None:
            return _missing_artifact_response(
                dataset_id=dataset_id,
                dataset_s3_key=dataset.s3_key,
                run_id=run_id,
            )

        return DatasetVisualizationStatusResponse(
            dataset_id=dataset_id,
            dag_id=DATASET_VISUALIZATION_DAG_ID,
            run_id=run_id,
            status=status,
            message="Visualization is running in Airflow (Spark job in progress).",
        )

    if status == DagRunStatus.FAILED.value:
        return await _failed_run_response(
            dataset_id=dataset_id,
            run_id=run_id,
            airflow=service,
            base_message="Visualization pipeline failed in Airflow.",
            artifact=artifact,
        )

    if status != DagRunStatus.SUCCESS.value:
        return DatasetVisualizationStatusResponse(
            dataset_id=dataset_id,
            dag_id=DATASET_VISUALIZATION_DAG_ID,
            run_id=run_id,
            status="failed",
            message=f"Visualization pipeline finished with state: {run.raw_state}.",
        )

    return _missing_artifact_response(
        dataset_id=dataset_id,
        dataset_s3_key=dataset.s3_key,
        run_id=run_id,
    )
