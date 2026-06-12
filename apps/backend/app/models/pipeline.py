import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
    dag_args: dict[str, Any] = Field(default_factory=dict)


class PipelineRunDeleteResponse(BaseModel):
    message: str
    run_id: int
