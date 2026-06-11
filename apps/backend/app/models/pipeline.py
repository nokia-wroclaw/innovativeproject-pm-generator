import datetime

from pydantic import BaseModel, ConfigDict

from app.db.schemas import PipelineRunStatus, PipelineType


class PipelineRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    dataset_name: str | None = None
    pipeline_type: PipelineType
    status: PipelineRunStatus
    airflow_run_id: str | None
    created_at: datetime.datetime


class PipelineRunCreate(BaseModel):
    dataset_id: int
    pipeline_type: PipelineType


class PipelineRunDeleteResponse(BaseModel):
    message: str
    run_id: int
