import datetime
import os
import uuid
from typing import Any

import requests  # type: ignore[import-untyped]
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.schemas import Dataset, DatasetType, PipelineRun, PipelineRunStatus, PipelineType
from app.services.preprocessing.conf import (
    PREPROCESSING_DAG_ID,
    PreprocessingConfigError,
    build_preprocessing_dag_args,
)

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

    def create_run(
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

        airflow_run_id = self._trigger_airflow(run, dataset, dag_args or {})
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

    def _trigger_airflow(
        self,
        run: PipelineRun,
        dataset: Dataset,
        dag_args: dict[str, Any],
    ) -> str | None:
        airflow_url = os.getenv("AIRFLOW_URL", "").rstrip("/")
        if not airflow_url:
            return None

        dag_id = DAG_ID_MAP.get(run.pipeline_type)
        if not dag_id:
            return None

        logical_run_id = f"genpm_pp_{run.id}_{uuid.uuid4().hex[:8]}"
        conf: dict[str, Any] = {
            "genpm_run_id": logical_run_id,
            "dataset_id": run.dataset_id,
            "s3_key": dataset.s3_key,
            "file_name": dataset.file_name,
        }

        if run.pipeline_type == PipelineType.PREPROCESSING:
            try:
                resolved_dag_args = build_preprocessing_dag_args(
                    genpm_run_id=logical_run_id,
                    raw_s3_key=dataset.s3_key,
                    user_args=dag_args,
                )
            except PreprocessingConfigError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            conf["dag_args"] = resolved_dag_args
            conf["process_type"] = "preprocessing_feature_engineering"

        try:
            response = requests.post(
                f"{airflow_url}/api/v2/dags/{dag_id}/dagRuns",
                json={
                    "dag_run_id": logical_run_id,
                    "conf": conf,
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                run_id = payload.get("dag_run_id", logical_run_id)
                return str(run_id) if run_id is not None else logical_run_id
            return logical_run_id
        except HTTPException:
            raise
        except Exception:
            return None

    def delete_run(self, run_id: int) -> None:
        run = self.get_run(run_id)
        if run:
            self._db.delete(run)
            self._db.commit()
