import os
import uuid

import boto3
from sqlalchemy.orm import Session

from app.db.schemas import Dataset


class S3Service:
    def __init__(self, db: Session):
        self._db = db
        self._s3_client = boto3.client("s3")

    def create_dataset(self, user_uuid: uuid.UUID, file_name: str, s3_key: str, s3_bucket: str) -> Dataset:
        dataset = Dataset(user_uuid=user_uuid, file_name=file_name, s3_key=s3_key, s3_bucket=s3_bucket)
        self._db.add(dataset)
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def get_datasets(self) -> list[type[Dataset]]:
        return self._db.query(Dataset).all()
