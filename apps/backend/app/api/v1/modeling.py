import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.auth import require_admin, require_auth
from app.db.database import db_manager
from app.db.schemas import DagRunStatus, DatasetStatus
from app.models.dags import TriggerRequest
from app.models.modeling import (
    ModelingArtifact,
    ModelingDatasetOption,
    ModelingFormField,
    ModelingFormOption,
    ModelingFormSchema,
    ModelingProcessType,
    ModelingRunCreated,
    ModelingRunRequest,
    ModelingRunStatus,
)
from app.services.airflow.errors import AirflowIntegrationError, AirflowNotFound
from app.services.airflow.runtime import get_airflow_service
from app.services.airflow.service import AirflowService
from app.services.s3.service import S3Service

router = APIRouter(prefix="/modeling", tags=["modeling"])

DAG_ID_MAP: dict[ModelingProcessType, str] = {
    "preprocessing_feature_engineering": "moj_pierwszy_dag",
    "training_dataset": "moj_pierwszy_dag",
}

# Short dag_run_id prefix — long ids (with full process_type) are harder for Airflow to accept.
_RUN_ID_PREFIX: dict[ModelingProcessType, str] = {
    "preprocessing_feature_engineering": "pe",
    "training_dataset": "td",
}

_BASE_FIELDS: list[ModelingFormField] = [
    ModelingFormField(
        name="dataset_id",
        label="Dataset input",
        type="dataset_select",
        required=True,
        help="Dataset with COMPLETED status and visible to the backend.",
    ),
    ModelingFormField(
        name="dataset_type",
        label="Dataset type",
        type="radio",
        required=True,
        default="working_days",
        options=[
            ModelingFormOption(value="working_days", label="Working days"),
            ModelingFormOption(value="weekends", label="Weekends"),
        ],
    ),
    ModelingFormField(
        name="dag_args",
        label="Additional DAG arguments",
        type="json",
        default={},
        help="JSON object passed to conf.dag_args.",
    ),
]

_PROCESS_TITLES: dict[ModelingProcessType, str] = {
    "preprocessing_feature_engineering": "Preprocessing + Feature Engineering",
    "training_dataset": "Training dataset creation",
}


def _get_s3_service(db: Session = Depends(db_manager.get_db)) -> S3Service:
    return S3Service(db=db)


def _airflow_service() -> AirflowService:
    return get_airflow_service()


def _identity(payload: dict[str, Any]) -> str | None:
    return (
        payload.get("preferred_username")
        or payload.get("email")
        or payload.get("sub")
    )


def _mock_logs(status: DagRunStatus) -> list[str]:
    logs = [
        "DAG configuration received from Airflow conf.",
        "Preprocessing started for the selected dataset type.",
    ]
    if status in {DagRunStatus.RUNNING, DagRunStatus.SUCCESS, DagRunStatus.FAILED}:
        logs.append("Feature engineering: generated calendar features and aggregates.")
        logs.append("Model training: saving checkpoints and metrics in progress.")
    if status == DagRunStatus.SUCCESS:
        logs.append("Model saved as a pickle file. Pipeline completed successfully.")
    if status == DagRunStatus.FAILED:
        logs.append("Airflow marked the run as failed. Check task logs in the DAG view.")
    return logs


def _mock_metrics(status: DagRunStatus) -> dict[str, float] | None:
    if status != DagRunStatus.SUCCESS:
        return None
    return {"mae": 0.083, "rmse": 0.127, "mape": 4.82, "validation_loss": 0.018}


def _mock_artifacts(
    status: DagRunStatus, run_id: str, process_type: ModelingProcessType
) -> list[ModelingArtifact]:
    saved: Literal["pending", "saved"] = "saved" if status == DagRunStatus.SUCCESS else "pending"
    base = f"s3://genpm-modeling/{run_id}"
    if process_type == "preprocessing_feature_engineering":
        return [
            ModelingArtifact(
                kind="preprocessed_dataset",
                path=f"{base}/preprocessed_dataset.parquet",
                status=saved,
            ),
            ModelingArtifact(
                kind="featured_dataset",
                path=f"{base}/featured_dataset.parquet",
                status=saved,
            ),
        ]
    return [
        ModelingArtifact(
            kind="training_dataset",
            path=f"{base}/training_dataset.parquet",
            status=saved,
        ),
        ModelingArtifact(
            kind="model_pickle",
            path=f"{base}/model.pkl",
            status=saved,
        ),
    ]


@router.get("/datasets", response_model=list[ModelingDatasetOption])
def list_modeling_datasets(
    _user: dict[str, Any] = Depends(require_auth),
    service: S3Service = Depends(_get_s3_service),
) -> list[ModelingDatasetOption]:
    return [
        ModelingDatasetOption.model_validate(ds)
        for ds in service.get_datasets()
        if ds.status == DatasetStatus.COMPLETED
    ]


@router.get(
    "/processes/{process_type}/form-schema",
    response_model=ModelingFormSchema,
)
def get_form_schema(
    process_type: ModelingProcessType,
    _user: dict[str, Any] = Depends(require_auth),
) -> ModelingFormSchema:
    return ModelingFormSchema(
        process_type=process_type,
        title=_PROCESS_TITLES[process_type],
        fields=_BASE_FIELDS,
    )


@router.post(
    "/processes/{process_type}/runs",
    response_model=ModelingRunCreated,
)
async def trigger_modeling_run(
    process_type: ModelingProcessType,
    body: ModelingRunRequest,
    user: dict[str, Any] = Depends(require_admin),
    service: S3Service = Depends(_get_s3_service),
    airflow: AirflowService = Depends(_airflow_service),
) -> ModelingRunCreated:
    dataset = service.get_dataset(body.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.status != DatasetStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail="Dataset must be COMPLETED before modeling can start",
        )

    dag_id = DAG_ID_MAP[process_type]
    run_id = f"genpm_{_RUN_ID_PREFIX[process_type]}_{uuid.uuid4().hex[:12]}"
    conf = {
        "genpm_run_id": run_id,
        "dataset_id": dataset.id,
        "dataset_name": dataset.file_name,
        "s3_key": dataset.s3_key,
        "dataset_type": body.dataset_type,
        "dag_args": body.dag_args,
        "process_type": process_type,
    }

    try:
        action = await airflow.trigger_dag(
            dag_id,
            body=TriggerRequest(
                conf=conf,
                dag_run_id=run_id,
                note=f"Modeling process {process_type}",
            ),
            triggered_by=_identity(user),
        )
    except AirflowNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"DAG '{dag_id}' is not registered in Airflow. "
                "Ensure apps/airflow/dags is mounted and the scheduler has parsed the DAG."
            ),
        ) from exc
    except AirflowIntegrationError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc

    effective_run_id = action.run_id or run_id

    return ModelingRunCreated(
        process_type=process_type,
        dag_id=dag_id,
        run_id=effective_run_id,
        message=action.message,
        airflow_status=action.airflow_status,
        conf=conf,
    )


@router.get(
    "/processes/{process_type}/runs/{run_id}",
    response_model=ModelingRunStatus,
)
async def get_modeling_run_status(
    process_type: ModelingProcessType,
    run_id: str,
    _user: dict[str, Any] = Depends(require_auth),
    airflow: AirflowService = Depends(_airflow_service),
) -> ModelingRunStatus:
    dag_id = DAG_ID_MAP[process_type]

    try:
        run = await airflow.get_dag_run(dag_id, run_id)
    except AirflowNotFound as exc:
        dag_missing = False
        try:
            await airflow.get_dag_details(dag_id)
        except AirflowNotFound:
            dag_missing = True
        if dag_missing:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"DAG '{dag_id}' is not registered in Airflow. "
                    "Ensure apps/airflow/dags is mounted and the scheduler has parsed the DAG."
                ),
            ) from exc
        raise HTTPException(
            status_code=404,
            detail=(
                f"Run '{run_id}' not found in Airflow for DAG '{dag_id}'. "
                "It may have expired after an Airflow reset — start a new process run."
            ),
        ) from exc
    except AirflowIntegrationError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc

    return ModelingRunStatus(
        process_type=process_type,
        dag_id=dag_id,
        run_id=run.run_id,
        status=run.status,
        raw_state=run.raw_state,
        start_date=run.start_date.isoformat() if run.start_date else None,
        end_date=run.end_date.isoformat() if run.end_date else None,
        duration_ms=run.duration_ms,
        logs=_mock_logs(run.status),
        metrics=_mock_metrics(run.status),
        artifacts=_mock_artifacts(run.status, run.run_id, process_type),
    )
