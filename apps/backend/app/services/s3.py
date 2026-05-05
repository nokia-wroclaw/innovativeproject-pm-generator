import os
import uuid

import boto3
from botocore.client import Config
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.schemas import Dataset, DatasetStatus

s3_client_internal = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_URL"),
    aws_access_key_id=os.getenv("S3_USER"),
    aws_secret_access_key=os.getenv("S3_PASSWORD"),
    region_name="us-east-1",
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)

S3_EXTERNAL_URL = os.getenv("S3_EXTERNAL_URL", "http://localhost:9000")

s3_client_external = boto3.client(
    "s3",
    endpoint_url=S3_EXTERNAL_URL,
    aws_access_key_id=os.getenv("S3_USER"),
    aws_secret_access_key=os.getenv("S3_PASSWORD"),
    region_name="us-east-1",
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
        final_s3_key = f"{s3_key}/{unique_id}_{file_name}"

        dataset = Dataset(user_uuid=user_uuid, file_name=file_name, s3_key=final_s3_key)
        self._db.add(dataset)
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def change_dataset_status(self, dataset_id: int, status: DatasetStatus) -> type[Dataset]:
        dataset = self._db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"[S3] Couldn't find dataset with id: {dataset_id}"
            )

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
        dataset = self._db.query(Dataset).filter(Dataset.id == dataset_id).first()
        self._db.delete(dataset)
        self._db.commit()

    def get_dataset(self, dataset_id: int) -> Dataset | None:
        return self._db.query(Dataset).filter(Dataset.id == dataset_id).first()

    def get_datasets(self) -> list[type[Dataset]]:
        return self._db.query(Dataset).all()
