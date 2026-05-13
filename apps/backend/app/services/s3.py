import os
import unicodedata
import uuid
import re
from pathlib import Path

import boto3
from botocore.client import Config
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.schemas import Dataset, DatasetStatus
from app.core.config import settings


s3_client_internal = boto3.client(
    "s3",
    endpoint_url=str(settings.s3_url),
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)


s3_client_external = boto3.client(
    "s3",
    endpoint_url=str(settings.s3_external_url),
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)

S3_BUCKET = os.getenv("S3_BUCKET")


class S3Service:
    def __init__(self, db: Session):
        self._db = db
        self.s3_client_internal = s3_client_internal
        self.s3_client_external = s3_client_external

    def create_dataset(self, user_uuid: uuid.UUID, file_name: str, s3_key: str) -> Dataset:
        unique_id = str(uuid.uuid4())

        name = Path(file_name).name
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        name = re.sub(r"[^\w\.\-]", "_", name)

        final_s3_key = f"{s3_key}/{unique_id}_{name}"

        dataset = Dataset(user_uuid=user_uuid, file_name=name, s3_key=final_s3_key)
        self._db.add(dataset)
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def register_existing_dataset(
        self, user_uuid: uuid.UUID, s3_key: str, file_name: str | None = None
    ) -> Dataset:
        try:
            self.s3_client_internal.head_object(Bucket=S3_BUCKET, Key=s3_key)
        except Exception as e:
            raise HTTPException(
                status_code=404, detail=f"[S3] Dataset not found on S3 or access denied: {str(e)}"
            )

        if not file_name:
            file_name = Path(s3_key).name

        name = unicodedata.normalize("NFKD", file_name).encode("ascii", "ignore").decode("ascii")
        name = re.sub(r"[^\w\.\-]", "_", name)

        dataset = Dataset(
            user_uuid=user_uuid, file_name=name, s3_key=s3_key, status=DatasetStatus.COMPLETED
        )
        self._db.add(dataset)
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def change_dataset_status(self, dataset_id: int, status: DatasetStatus) -> type[Dataset]:
        dataset = self.get_dataset(dataset_id)
        dataset.status = status
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def initiate_multipart_upload(self, dataset: Dataset) -> dict:
        try:
            response = self.s3_client_internal.create_multipart_upload(
                Bucket=S3_BUCKET, Key=dataset.s3_key
            )
            return {"upload_id": response["UploadId"], "s3_key": dataset.s3_key}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"[S3] Initialization error: {str(e)}")

    def get_presigned_part_url(self, s3_key: str, upload_id: str, part_number: int) -> str:
        try:
            presigned_url = self.s3_client_external.generate_presigned_url(
                ClientMethod="upload_part",
                Params={
                    "Bucket": S3_BUCKET,
                    "Key": s3_key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                },
                ExpiresIn=3600,
            )
            return presigned_url

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"[S3] URL generation error: {str(e)}")

    def complete_multipart_upload(self, s3_key: str, upload_id: str, parts: list[dict]) -> dict:
        try:
            response = self.s3_client_internal.complete_multipart_upload(
                Bucket=S3_BUCKET, Key=s3_key, UploadId=upload_id, MultipartUpload={"Parts": parts}
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"[S3] Merge upload error: {str(e)}")

    def abort_multipart_upload(self, s3_key: str, upload_id: str) -> None:
        try:
            self.s3_client_internal.abort_multipart_upload(
                Bucket=S3_BUCKET, Key=s3_key, UploadId=upload_id
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"[S3] Cancel upload error: {str(e)}")

    def delete_dataset(self, dataset_id: int) -> None:
        dataset = self.get_dataset(dataset_id)

        try:
            self.s3_client_internal.delete_object(Bucket=S3_BUCKET, Key=dataset.s3_key)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"[S3] Error deleting file from S3: {str(e)}"
            )

        self._db.delete(dataset)
        self._db.commit()

    def get_dataset(self, dataset_id: int) -> type[Dataset]:
        dataset = self._db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"[S3] Dataset not found: {dataset_id}")

        return dataset

    def get_datasets(self) -> list[type[Dataset]]:
        return self._db.query(Dataset).all()
