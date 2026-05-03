import os
import uuid

import boto3
from botocore.client import Config
from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.db.schemas import Dataset

s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_URL"),
    aws_access_key_id=os.getenv("S3_USER"),
    aws_secret_access_key=os.getenv("S3_PASSWORD"),
    region_name="us-east-1",
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)
S3_BUCKET = os.getenv("S3_BUCKET")


class S3Service:
    def __init__(self, db: Session):
        self._db = db
        self.s3_client = s3_client

    def create_dataset(self, user_uuid: uuid.UUID, file_name: str, s3_key: str) -> Dataset:
        dataset = Dataset(user_uuid=user_uuid, file_name=file_name, s3_key=s3_key)
        self._db.add(dataset)
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def get_presigned_url(self, dataset: Dataset) -> str:
        unique_id = str(uuid.uuid4())
        s3_key = f"{dataset.s3_key}/{unique_id}_{dataset.file_name}"

        try:
            presigned_url = self.s3_client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": S3_BUCKET,
                    "Key": s3_key,
                },
                ExpiresIn=3600,
            )
            return presigned_url

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def get_dataset(self, dataset_id: int) -> type[Dataset]:
        return self._db.query(Dataset).filter(Dataset.id == dataset_id).first()

    def get_datasets(self) -> list[type[Dataset]]:
        return self._db.query(Dataset).all()
