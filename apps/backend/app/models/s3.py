import uuid

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
    tables: list[TablePreview]
