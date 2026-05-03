import uuid
from pydantic import BaseModel, ConfigDict

from app.db.schemas import DatasetStatus


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_uuid: uuid.UUID
    s3_key: str
    file_name: str
    status: DatasetStatus


class DatasetCreate(BaseModel):
    s3_key: str
    file_name: str


class UploadUrlResponse(BaseModel):
    url: str
    file_name: str
