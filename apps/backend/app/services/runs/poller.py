"""Background poller that finalizes process runs in the user's backend.

When a preprocessing DAG (run by admin's Airflow) finishes, nothing writes back to this (the user's)
database on its own — the Spark job only writes parquet to MinIO. This poller watches the in-flight
``ProcessRun`` rows, and on success registers the output as a ``PREPROCESSED`` dataset in this DB
(owned by the user that triggered the run). It is restart-safe: state lives in the DB, so a backend
restart simply resumes polling the still-RUNNING rows.
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from app.db.database import db_manager
from app.db.schemas import (
    DagRunStatus,
    Dataset,
    DatasetType,
    PipelineRunStatus,
    ProcessRun,
)
from app.services.airflow.errors import AirflowIntegrationError, AirflowNotFound
from app.services.airflow.runtime import get_airflow_service
from app.services.s3.service import S3Service

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 30.0


class ProcessRunPoller:
    def __init__(self, interval_seconds: float | None = None) -> None:
        self._interval = interval_seconds or float(
            os.getenv("PROCESS_RUN_POLL_INTERVAL_SECONDS", _DEFAULT_INTERVAL)
        )
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="process-run-poller")
            logger.info("ProcessRunPoller started (interval=%.0fs)", self._interval)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self._poll_once()
            except Exception:
                logger.exception("ProcessRunPoller tick failed")
            await asyncio.sleep(self._interval)

    async def _poll_once(self) -> None:
        try:
            airflow = get_airflow_service()
        except RuntimeError:
            return  # Airflow runtime not up yet; try again next tick.

        db = db_manager.SessionLocal()
        try:
            runs = db.query(ProcessRun).filter(ProcessRun.status == PipelineRunStatus.RUNNING).all()
            for run in runs:
                await self._finalize(db, airflow, run)
        finally:
            db.close()

    async def _finalize(self, db, airflow, run: ProcessRun) -> None:
        try:
            dag_run = await airflow.get_dag_run(run.dag_id, run.run_id)
        except AirflowNotFound:
            logger.warning("Run %s not found in Airflow — marking FAILED", run.run_id)
            run.status = PipelineRunStatus.FAILED
            db.commit()
            return
        except AirflowIntegrationError:
            return  # transient (Airflow down/unauthorized); retry next tick

        if dag_run.status == DagRunStatus.SUCCESS:
            self._register_output(db, run)
        elif dag_run.status == DagRunStatus.FAILED:
            run.status = PipelineRunStatus.FAILED
            db.commit()
        # RUNNING / QUEUED → leave for a later tick.

    _PROCESS_TYPE_TO_DATASET_TYPE: dict[str, DatasetType] = {
        "preprocessing_feature_engineering": DatasetType.PREPROCESSED,
        "generate": DatasetType.GENERATED,
    }

    def _resolve_dataset_type(self, run: ProcessRun) -> DatasetType:
        return self._PROCESS_TYPE_TO_DATASET_TYPE.get(run.process_type, DatasetType.PREPROCESSED)

    def _derive_pm_metadata_s3_key(self, run: ProcessRun) -> str | None:
        """Return the expected pm_metadata.json S3 key for preprocessing runs, or None."""
        if run.process_type != "preprocessing_feature_engineering":
            return None
        prefix = run.output_s3_key.rstrip("/")
        return f"{prefix}/pm_metadata.json"

    def _register_output(self, db, run: ProcessRun) -> None:
        service = S3Service(db=db)
        dataset_type = self._resolve_dataset_type(run)
        pm_metadata_s3_key = self._derive_pm_metadata_s3_key(run)
        try:
            dataset = service.register_existing_dataset(
                user_uuid=run.user_uuid,
                s3_key=run.output_s3_key,
                file_name=run.output_name,
                type=dataset_type,
                pm_metadata_s3_key=pm_metadata_s3_key,
            )
            run.registered_dataset_id = dataset.id
            run.status = PipelineRunStatus.COMPLETED
            db.commit()
            logger.info("Registered %s dataset %s for run %s", dataset_type, dataset.id, run.run_id)
        except IntegrityError:
            # Already registered (unique s3_key+type) — link the existing row and finish.
            db.rollback()
            existing = (
                db.query(Dataset)
                .filter(
                    Dataset.s3_key == run.output_s3_key,
                    Dataset.type == dataset_type,
                )
                .first()
            )
            run.registered_dataset_id = existing.id if existing else None
            run.status = PipelineRunStatus.COMPLETED
            db.commit()
        except HTTPException as exc:
            # Output prefix not visible in S3 yet (eventual consistency) —
            # keep RUNNING, retry later.
            db.rollback()
            logger.warning("Output not registerable yet for run %s: %s", run.run_id, exc.detail)
