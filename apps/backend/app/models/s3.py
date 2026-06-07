import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.db.schemas import DatasetStatus, DatasetType


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_uuid: uuid.UUID
    s3_key: str
    file_name: str
    status: DatasetStatus
    type: DatasetType


class DatasetCreate(BaseModel):
    s3_key: str
    file_name: str
    type: DatasetType = DatasetType.RAW


class UploadUrlResponse(BaseModel):
    url: str
    file_name: str


class DatasetStatusUpdate(BaseModel):
    dataset_id: int
    status: DatasetStatus


class PartInfo(BaseModel):
    PartNumber: int
    ETag: str


class DatasetRegisterRequest(BaseModel):
    s3_key: str
    file_name: str | None = None
    type: DatasetType = DatasetType.RAW


class CompleteMultipartRequest(BaseModel):
    upload_id: str
    parts: list[PartInfo]


class AbortMultipartRequest(BaseModel):
    upload_id: str


class MultipartInitiateResponse(BaseModel):
    upload_id: str
    s3_key: str


class PartUrlResponse(BaseModel):
    url: str


class TablePreview(BaseModel):
    name: str
    columns: list[str]
    rows: list[dict[str, object]]


class DatasetPreviewResponse(BaseModel):
    dataset_id: int
    file_name: str
    s3_key: str
    type: DatasetType
    tables: list[TablePreview]


class DatasetVisualizationResponse(BaseModel):
    message: str
    dag_id: str
    airflow_run_id: str


class DatasetVisualizationStatusResponse(BaseModel):
    dataset_id: int
    dag_id: str
    run_id: str | None = None
    status: str
    spark_version: str | None = None
    message: str | None = None
    summary: dict[str, Any] | None = None
    kpi_analysis: dict[str, Any] | None = None
