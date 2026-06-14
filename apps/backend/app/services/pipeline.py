import datetime
import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.schemas import Dataset, DatasetType, PipelineRun, PipelineRunStatus, PipelineType
from app.models.dags import TriggerRequest
from app.models.spark_jobs import PreprocessingConfigError
from app.services.airflow.errors import AirflowIntegrationError, AirflowNotFound
from app.services.airflow.runtime import get_airflow_service
from app.services.preprocessing.conf import PREPROCESSING_DAG_ID
from app.services.spark_dag_conf import build_preprocessing_dag_conf

DAG_ID_MAP = {
    PipelineType.PREPROCESSING: PREPROCESSING_DAG_ID,
    PipelineType.FEATURE_ENGINEERING: "feature_engineering_pipeline",
    PipelineType.TRAINING: "training_pipeline",
}


class PipelineService:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_runs(self) -> list[dict]:
        rows = (
            self._db.query(PipelineRun, Dataset.file_name)
            .outerjoin(Dataset, PipelineRun.dataset_id == Dataset.id)
            .order_by(PipelineRun.created_at.desc())
            .all()
        )
        return [
            {
                "id": run.id,
                "dataset_id": run.dataset_id,
                "dataset_name": dataset_name,
                "pipeline_type": run.pipeline_type,
                "status": run.status,
                "airflow_run_id": run.airflow_run_id,
                "created_at": run.created_at,
            }
            for run, dataset_name in rows
        ]

    def get_run(self, run_id: int) -> PipelineRun | None:
        return self._db.query(PipelineRun).filter(PipelineRun.id == run_id).first()

    async def create_run(
        self,
        dataset_id: int,
        pipeline_type: PipelineType,
        dag_args: dict[str, Any] | None = None,
    ) -> dict:
        dataset = self._db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if pipeline_type == PipelineType.PREPROCESSING and dataset.type != DatasetType.RAW:
            raise HTTPException(
                status_code=409,
                detail="Preprocessing requires a RAW dataset as input",
            )

        run = PipelineRun(
            dataset_id=dataset_id,
            pipeline_type=pipeline_type,
            status=PipelineRunStatus.PENDING,
            created_at=datetime.datetime.utcnow(),
        )
        self._db.add(run)
        self._db.commit()
        self._db.refresh(run)

        airflow_run_id = await self._trigger_airflow(run, dataset, dag_args or {})
        if airflow_run_id:
            run.airflow_run_id = airflow_run_id
            run.status = PipelineRunStatus.RUNNING
            self._db.commit()
            self._db.refresh(run)

        return {
            "id": run.id,
            "dataset_id": run.dataset_id,
            "dataset_name": dataset.file_name,
            "pipeline_type": run.pipeline_type,
            "status": run.status,
            "airflow_run_id": run.airflow_run_id,
            "created_at": run.created_at,
        }

    async def _trigger_airflow(
        self,
        run: PipelineRun,
        dataset: Dataset,
        dag_args: dict[str, Any],
    ) -> str | None:
        try:
            airflow = get_airflow_service()
        except RuntimeError:
            return None

        dag_id = DAG_ID_MAP.get(run.pipeline_type)
        if not dag_id:
            return None

        logical_run_id = f"genpm_pp_{run.id}_{uuid.uuid4().hex[:8]}"

        try:
            if run.pipeline_type == PipelineType.PREPROCESSING:
                conf = build_preprocessing_dag_conf(
                    genpm_run_id=logical_run_id,
                    dataset_id=run.dataset_id,
                    s3_key=dataset.s3_key,
                    file_name=dataset.file_name,
                    user_dag_args=dag_args,
                ).to_airflow_conf()
            else:
                conf = {
                    "genpm_run_id": logical_run_id,
                    "dataset_id": run.dataset_id,
                    "s3_key": dataset.s3_key,
                    "file_name": dataset.file_name,
                    "dag_args": dag_args,
                }

            action = await airflow.trigger_dag(
                dag_id,
                body=TriggerRequest(
                    conf=conf,
                    run_id=logical_run_id,
                    note=f"Pipeline {run.pipeline_type.value}",
                ),
            )
            return action.run_id or logical_run_id
        except PreprocessingConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except (AirflowNotFound, AirflowIntegrationError):
            return None

    def delete_run(self, run_id: int) -> None:
        run = self.get_run(run_id)
        if run:
            self._db.delete(run)
            self._db.commit()
