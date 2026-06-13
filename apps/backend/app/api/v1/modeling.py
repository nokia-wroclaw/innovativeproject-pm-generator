import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.auth import (
    assert_modeling_admin,
    get_user_identity,
    require_auth,
    require_modeling_admin,
)
from app.db.database import db_manager
from app.db.schemas import DagRunStatus, DatasetStatus, DatasetType
from app.models.auth import TokenPayload
from app.models.dags import TriggerRequest
from app.models.modeling import (
    GenerateRunRequest,
    ModelingArtifact,
    ModelingDatasetOption,
    ModelingFormField,
    ModelingFormOption,
    ModelingFormSchema,
    ModelingProcessType,
    ModelingRunCreated,
    ModelingRunRequest,
    ModelingRunStatus,
    ModelingTrainedModelOption,
)
from app.models.spark_jobs import PreprocessingConfigError
from app.services.airflow.errors import AirflowIntegrationError, AirflowNotFound
from app.services.airflow.runtime import get_airflow_service
from app.services.airflow.service import AirflowService
from app.services.preprocessing.conf import (
    PREPROCESSING_DAG_ID,
    preprocessing_artifact_paths,
)
from app.services.s3.service import S3Service
from app.services.spark_dag_conf import build_preprocessing_dag_conf

router = APIRouter(prefix="/modeling", tags=["modeling"])

# Training / generate DAGs are not implemented yet — these ids are intentional placeholders.
# Triggering them returns AirflowNotFound (handled as a clear 404) until the DAGs are added.
TRAINING_DAG_ID = "training_dataset_pipeline"
GENERATE_DAG_ID = "generate_pipeline"

DAG_ID_MAP: dict[ModelingProcessType, str] = {
    "preprocessing_feature_engineering": PREPROCESSING_DAG_ID,
    "training_dataset": TRAINING_DAG_ID,
    "generate": GENERATE_DAG_ID,
}

# Short dag_run_id prefix — long ids (with full process_type) are harder for Airflow to accept.
_RUN_ID_PREFIX: dict[ModelingProcessType, str] = {
    "preprocessing_feature_engineering": "pe",
    "training_dataset": "td",
    "generate": "gen",
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
    "generate": "Synthetic data generation",
}

_MOCK_TRAINED_MODELS: list[ModelingTrainedModelOption] = [
    ModelingTrainedModelOption(
        id="model_001",
        name="PM predictor v1 (working days)",
        source_run_id="genpm_td_a1b2c3d4e5f6",
        path="s3://genpm-modeling/models/model_001.pkl",
        created_at="2026-05-20T14:30:00Z",
    ),
    ModelingTrainedModelOption(
        id="model_002",
        name="PM predictor v1 (weekends)",
        source_run_id="genpm_td_b2c3d4e5f6a1",
        path="s3://genpm-modeling/models/model_002.pkl",
        created_at="2026-05-22T09:15:00Z",
    ),
    ModelingTrainedModelOption(
        id="model_003",
        name="PM predictor v2 (ensemble)",
        source_run_id="genpm_td_c3d4e5f6a1b2",
        path="s3://genpm-modeling/models/model_003.pkl",
        created_at="2026-05-28T16:45:00Z",
    ),
]


def _get_s3_service(db: Session = Depends(db_manager.get_db)) -> S3Service:
    return S3Service(db=db)


def _airflow_service() -> AirflowService:
    return get_airflow_service()


def _preprocessing_logs(status: DagRunStatus) -> list[str]:
    logs = [
        "DAG configuration received from Airflow conf.",
        "Spark preprocessing job submitted (read RAW + auxiliary parquets).",
    ]
    if status in {DagRunStatus.RUNNING, DagRunStatus.SUCCESS, DagRunStatus.FAILED}:
        logs.append("KPI coverage filtering and windowing in progress.")
    if status == DagRunStatus.SUCCESS:
        logs.append("Preprocessed artifacts written to S3. Pipeline completed successfully.")
    if status == DagRunStatus.FAILED:
        logs.append("Airflow marked the run as failed. Check task logs in the DAG view.")
    return logs


def _mock_logs(status: DagRunStatus, process_type: ModelingProcessType) -> list[str]:
    if process_type == "generate":
        logs = [
            "Generation configuration received from Airflow conf.",
            "Loading trained model artifact from S3.",
        ]
        if status in {DagRunStatus.RUNNING, DagRunStatus.SUCCESS, DagRunStatus.FAILED}:
            logs.append("Sampling synthetic traces from the selected model.")
        if status == DagRunStatus.SUCCESS:
            logs.append("Event log and generation report saved successfully.")
        if status == DagRunStatus.FAILED:
            logs.append("Generation failed. Check task logs in the DAG view.")
        return logs

    if process_type == "preprocessing_feature_engineering":
        return _preprocessing_logs(status)

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


def _mock_metrics(
    status: DagRunStatus, process_type: ModelingProcessType
) -> dict[str, float] | None:
    if status != DagRunStatus.SUCCESS:
        return None
    if process_type == "preprocessing_feature_engineering":
        return None
    if process_type == "generate":
        return {"traces": 128.0, "events": 45210.0, "avg_trace_length": 353.2}
    return {"mae": 0.083, "rmse": 0.127, "mape": 4.82, "validation_loss": 0.018}


def _preprocessing_artifacts(
    status: DagRunStatus,
    conf: dict[str, Any] | None,
) -> list[ModelingArtifact]:
    saved: Literal["pending", "saved"] = "saved" if status == DagRunStatus.SUCCESS else "pending"
    dag_args = (conf or {}).get("dag_args") if isinstance(conf, dict) else None
    output_prefix = (
        str(dag_args.get("output_path_prefix")).strip()
        if isinstance(dag_args, dict) and dag_args.get("output_path_prefix")
        else ""
    )
    if output_prefix:
        paths = preprocessing_artifact_paths(output_prefix)
        return [
            ModelingArtifact(
                kind="preprocessed_dataset",
                path=paths["pm_df_long_indexed_winds"],
                status=saved,
            ),
            ModelingArtifact(
                kind="featured_dataset",
                path=paths["scaling_params_df"],
                status="pending",
            ),
        ]
    return [
        ModelingArtifact(kind="preprocessed_dataset", path="", status="pending"),
        ModelingArtifact(kind="featured_dataset", path="", status="pending"),
    ]


def _mock_artifacts(
    status: DagRunStatus,
    run_id: str,
    process_type: ModelingProcessType,
    conf: dict[str, Any] | None = None,
) -> list[ModelingArtifact]:
    saved: Literal["pending", "saved"] = "saved" if status == DagRunStatus.SUCCESS else "pending"
    base = f"s3://genpm-modeling/{run_id}"
    if process_type == "preprocessing_feature_engineering":
        return _preprocessing_artifacts(status, conf)
    if process_type == "generate":
        return [
            ModelingArtifact(
                kind="generated_event_log",
                path=f"{base}/generated_event_log.parquet",
                status=saved,
            ),
            ModelingArtifact(
                kind="generation_report",
                path=f"{base}/generation_report.json",
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


@router.get("/models", response_model=list[ModelingTrainedModelOption])
def list_trained_models(
    _user: TokenPayload = Depends(require_auth),
) -> list[ModelingTrainedModelOption]:
    return _MOCK_TRAINED_MODELS


@router.get("/datasets", response_model=list[ModelingDatasetOption])
def list_modeling_datasets(
    _user: TokenPayload = Depends(require_modeling_admin),
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
    _user: TokenPayload = Depends(require_modeling_admin),
) -> ModelingFormSchema:
    return ModelingFormSchema(
        process_type=process_type,
        title=_PROCESS_TITLES[process_type],
        fields=_BASE_FIELDS,
    )


@router.post(
    "/processes/generate/runs",
    response_model=ModelingRunCreated,
)
async def trigger_generate_run(
    body: GenerateRunRequest,
    user: TokenPayload = Depends(require_auth),
    airflow: AirflowService = Depends(_airflow_service),
) -> ModelingRunCreated:
    process_type: Literal["generate"] = "generate"
    model = next((m for m in _MOCK_TRAINED_MODELS if m.id == body.model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {body.model_id}")

    dag_id = DAG_ID_MAP[process_type]
    run_id = f"genpm_{_RUN_ID_PREFIX[process_type]}_{uuid.uuid4().hex[:12]}"
    conf = {
        "genpm_run_id": run_id,
        "model_id": body.model_id,
        "model_name": model.name,
        "model_path": model.path,
        "prompt": body.prompt,
        "process_type": process_type,
        "dag_args": body.dag_args,
    }

    try:
        action = await airflow.trigger_dag(
            dag_id,
            body=TriggerRequest(
                conf=conf,
                dag_run_id=run_id,
                note="Modeling process generate",
            ),
            triggered_by=get_user_identity(user),
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


@router.post(
    "/processes/{process_type}/runs",
    response_model=ModelingRunCreated,
)
async def trigger_modeling_run(
    process_type: Literal["preprocessing_feature_engineering", "training_dataset"],
    body: ModelingRunRequest,
    user: TokenPayload = Depends(require_modeling_admin),
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
    if process_type == "preprocessing_feature_engineering" and dataset.type != DatasetType.RAW:
        raise HTTPException(
            status_code=409,
            detail="Preprocessing requires a RAW dataset as input",
        )

    dag_id = DAG_ID_MAP[process_type]
    run_id = f"genpm_{_RUN_ID_PREFIX[process_type]}_{uuid.uuid4().hex[:12]}"
    user_dag_args = dict(body.dag_args)
    if process_type == "preprocessing_feature_engineering":
        try:
            conf_model = build_preprocessing_dag_conf(
                genpm_run_id=run_id,
                dataset_id=dataset.id,
                s3_key=dataset.s3_key,
                dataset_name=dataset.file_name,
                user_dag_args=user_dag_args,
                process_type=process_type,
            )
        except PreprocessingConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        conf = conf_model.to_airflow_conf()
    else:
        resolved_dag_args = {**user_dag_args, "dataset_id": dataset.id}
        conf = {
            "genpm_run_id": run_id,
            "dataset_id": dataset.id,
            "dataset_name": dataset.file_name,
            "s3_key": dataset.s3_key,
            "dag_args": resolved_dag_args,
            "process_type": process_type,
        }
    if process_type != "preprocessing_feature_engineering":
        conf["dataset_type"] = body.dataset_type

    try:
        action = await airflow.trigger_dag(
            dag_id,
            body=TriggerRequest(
                conf=conf,
                dag_run_id=run_id,
                note=f"Modeling process {process_type}",
            ),
            triggered_by=get_user_identity(user),
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
    user: TokenPayload = Depends(require_auth),
    airflow: AirflowService = Depends(_airflow_service),
) -> ModelingRunStatus:
    if process_type != "generate":
        assert_modeling_admin(user)

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
        logs=_mock_logs(run.status, process_type),
        metrics=_mock_metrics(run.status, process_type),
        artifacts=_mock_artifacts(run.status, run.run_id, process_type, run.conf),
    )
