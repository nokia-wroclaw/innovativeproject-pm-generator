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
    ModelingRunCreated,
    ModelingRunRequest,
    ModelingRunStatus,
)
from app.services.airflow.runtime import get_airflow_service
from app.services.airflow.service import AirflowService
from app.services.s3.service import S3Service

router = APIRouter(prefix="/modeling", tags=["modeling"])

MODELING_DAG_ID = "mock_modeling_dag"


def _airflow_service() -> AirflowService:
    return get_airflow_service()


def _s3_service(db: Session = Depends(db_manager.get_db)) -> S3Service:
    return S3Service(db=db)


def _identity(payload: dict[str, Any]) -> str | None:
    return (
        payload.get("preferred_username")
        or payload.get("email")
        or payload.get("sub")
    )


@router.get("/datasets", response_model=list[ModelingDatasetOption])
async def list_modeling_datasets(
    _user: dict[str, Any] = Depends(require_auth),
    service: S3Service = Depends(_s3_service),
) -> list[ModelingDatasetOption]:
    return [
        ModelingDatasetOption.model_validate(dataset)
        for dataset in service.get_datasets()
        if dataset.status == DatasetStatus.COMPLETED
    ]


@router.post("/runs", response_model=ModelingRunCreated)
async def trigger_modeling_run(
    body: ModelingRunRequest,
    user: dict[str, Any] = Depends(require_admin),
    datasets: S3Service = Depends(_s3_service),
    airflow: AirflowService = Depends(_airflow_service),
) -> ModelingRunCreated:
    dataset = datasets.get_dataset(body.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.status != DatasetStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail="Dataset must be COMPLETED before modeling can start",
        )

    conf = {
        "dataset_id": dataset.id,
        "dataset_name": dataset.file_name,
        "s3_key": dataset.s3_key,
        "dataset_type": body.dataset_type,
        "training": {
            "epochs": body.epochs,
            "batch_size": body.batch_size,
            "learning_rate": body.learning_rate,
        },
        "steps": {
            "preprocessing": True,
            "feature_engineering": True,
            "training": True,
        },
    }
    action = await airflow.trigger_dag(
        MODELING_DAG_ID,
        body=TriggerRequest(conf=conf, note="Modeling run"),
        triggered_by=_identity(user),
    )
    return ModelingRunCreated(
        dag_id=MODELING_DAG_ID,
        run_id=action.run_id,
        message=action.message,
        airflow_status=action.airflow_status,
        conf=conf,
    )


@router.get("/runs/{run_id}", response_model=ModelingRunStatus)
async def get_modeling_run_status(
    run_id: str,
    _user: dict[str, Any] = Depends(require_auth),
    airflow: AirflowService = Depends(_airflow_service),
) -> ModelingRunStatus:
    run = await airflow.get_dag_run(MODELING_DAG_ID, run_id)
    return ModelingRunStatus(
        dag_id=MODELING_DAG_ID,
        run_id=run.run_id,
        status=run.status,
        raw_state=run.raw_state,
        start_date=run.start_date.isoformat() if run.start_date else None,
        end_date=run.end_date.isoformat() if run.end_date else None,
        duration_ms=run.duration_ms,
        logs=_mock_logs(run.status),
        metrics=_mock_metrics(run.status),
        artifacts=_mock_artifacts(run.status, run.run_id),
    )


def _mock_logs(status: DagRunStatus) -> list[str]:
    logs = [
        "Konfiguracja DAG odebrana z Airflow conf.",
        "Preprocessing uruchomiony dla wybranego typu danych.",
    ]
    if status in {DagRunStatus.RUNNING, DagRunStatus.SUCCESS, DagRunStatus.FAILED}:
        logs.append("Feature engineering: wygenerowano cechy kalendarzowe i agregaty.")
        logs.append("Trening modelu: zapis checkpointow i metryk w toku.")
    if status == DagRunStatus.SUCCESS:
        logs.append("Model zapisany jako plik pickle. Pipeline zakonczony sukcesem.")
    if status == DagRunStatus.FAILED:
        logs.append("Airflow oznaczyl run jako failed. Sprawdz logi taskow w widoku DAG.")
    return logs


def _mock_metrics(status: DagRunStatus) -> dict[str, float] | None:
    if status != DagRunStatus.SUCCESS:
        return None
    return {
        "mae": 0.083,
        "rmse": 0.127,
        "mape": 4.82,
        "validation_loss": 0.018,
    }


def _mock_artifacts(status: DagRunStatus, run_id: str) -> list[ModelingArtifact]:
    saved = status == DagRunStatus.SUCCESS
    state: Literal["pending", "saved"] = "saved" if saved else "pending"
    base = f"s3://genpm-modeling/{run_id}"
    return [
        ModelingArtifact(
            kind="preprocessed_dataset",
            path=f"{base}/preprocessed_dataset.parquet",
            status=state,
        ),
        ModelingArtifact(
            kind="featured_dataset",
            path=f"{base}/featured_dataset.parquet",
            status=state,
        ),
        ModelingArtifact(
            kind="model_pickle",
            path=f"{base}/model.pkl",
            status=state,
        ),
    ]
