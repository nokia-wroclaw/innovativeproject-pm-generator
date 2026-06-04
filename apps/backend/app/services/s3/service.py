import io
import json
import os
import re
import unicodedata
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import boto3
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from botocore.client import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.schemas import Dataset, DatasetStatus, DatasetType

PREVIEW_ROW_LIMIT = 5
CSV_PREVIEW_RANGE_BYTES = 4 * 1024 * 1024
MAX_PREVIEW_TABLES = 50
SUPPORTED_PREVIEW_EXTENSIONS = {".parquet", ".csv"}


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


@lru_cache
def _get_s3_filesystem() -> pafs.S3FileSystem:
    parsed = urlparse(str(settings.s3_url))
    scheme = parsed.scheme or "http"
    endpoint = parsed.netloc or parsed.path.lstrip("/")

    return pafs.S3FileSystem(
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_override=endpoint,
        scheme=scheme,
        region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


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

        dataset = Dataset(
            user_uuid=user_uuid,
            file_name=name,
            s3_key=final_s3_key,
            type=DatasetType.RAW,
        )
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
                status_code=404,
                detail=f"[S3] Dataset not found on S3 or access denied: {str(e)}",
            ) from e

        if not file_name:
            file_name = Path(s3_key).name

        name = unicodedata.normalize("NFKD", file_name).encode("ascii", "ignore").decode("ascii")
        name = re.sub(r"[^\w\.\-]", "_", name)

        dataset = Dataset(
            user_uuid=user_uuid,
            file_name=name,
            s3_key=s3_key,
            status=DatasetStatus.COMPLETED,
            type=DatasetType.RAW,
        )
        self._db.add(dataset)
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def change_dataset_status(self, dataset_id: int, status: DatasetStatus) -> Dataset:
        dataset = self.get_dataset(dataset_id)
        dataset.status = status
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def initiate_multipart_upload(self, dataset: Dataset) -> dict[str, Any]:
        try:
            response = self.s3_client_internal.create_multipart_upload(
                Bucket=S3_BUCKET, Key=dataset.s3_key
            )
            return {"upload_id": response["UploadId"], "s3_key": dataset.s3_key}
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"[S3] Initialization error: {str(e)}"
            ) from e

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
            return str(presigned_url)

        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"[S3] URL generation error: {str(e)}"
            ) from e

    def complete_multipart_upload(
        self, s3_key: str, upload_id: str, parts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            response = self.s3_client_internal.complete_multipart_upload(
                Bucket=S3_BUCKET, Key=s3_key, UploadId=upload_id, MultipartUpload={"Parts": parts}
            )
            return cast(dict[str, Any], response)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"[S3] Merge upload error: {str(e)}") from e

    def abort_multipart_upload(self, s3_key: str, upload_id: str) -> None:
        try:
            self.s3_client_internal.abort_multipart_upload(
                Bucket=S3_BUCKET, Key=s3_key, UploadId=upload_id
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"[S3] Cancel upload error: {str(e)}"
            ) from e

    def delete_dataset(self, dataset_id: int) -> None:
        dataset = self.get_dataset(dataset_id)

        try:
            self.s3_client_internal.delete_object(Bucket=S3_BUCKET, Key=dataset.s3_key)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"[S3] Error deleting file from S3: {str(e)}",
            ) from e

        self._db.delete(dataset)
        self._db.commit()

    def get_dataset(self, dataset_id: int) -> Dataset:
        dataset = self._db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"[S3] Dataset not found: {dataset_id}")

        return dataset

    def get_datasets(self, dataset_type: DatasetType | None = None) -> list[Dataset]:
        query = self._db.query(Dataset)
        if dataset_type is not None:
            query = query.filter(Dataset.type == dataset_type)
        return query.all()

    def preview_dataset(self, dataset_id: int) -> dict:
        dataset = self.get_dataset(dataset_id)

        status = (
            dataset.status.value
            if isinstance(dataset.status, DatasetStatus)
            else str(dataset.status)
        )
        if status != DatasetStatus.COMPLETED.value:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be completed before preview (current status: {status})",
            )

        tables = []
        for table_name, object_key in self._discover_preview_objects(dataset.s3_key):
            columns, rows = self._preview_object(object_key)
            tables.append({"name": table_name, "columns": columns, "rows": rows})

        return {
            "dataset_id": dataset.id,
            "file_name": dataset.file_name,
            "s3_key": dataset.s3_key,
            "tables": tables,
        }

    def _discover_preview_objects(self, s3_key: str) -> list[tuple[str, str]]:
        extension = Path(s3_key).suffix.lower()
        if extension in SUPPORTED_PREVIEW_EXTENSIONS:
            try:
                self.s3_client_internal.head_object(Bucket=S3_BUCKET, Key=s3_key)
                return [(Path(s3_key).name or s3_key, s3_key)]
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code", "")
                if error_code not in {"404", "NoSuchKey", "NotFound"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"[S3] Failed to access dataset: {exc}",
                    ) from exc

        discovered: list[tuple[str, str]] = []
        prefixes = [s3_key, f"{s3_key.rstrip('/')}/"]
        paginator = self.s3_client_internal.get_paginator("list_objects_v2")

        for prefix in prefixes:
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue
                    if Path(key).suffix.lower() not in SUPPORTED_PREVIEW_EXTENSIONS:
                        continue

                    relative_name = key.removeprefix(prefix).strip("/")
                    table_name = relative_name or Path(key).name
                    discovered.append((table_name, key))

            if discovered:
                break

        if not discovered:
            raise HTTPException(
                status_code=404,
                detail="No previewable data files found for this dataset",
            )

        return discovered[:MAX_PREVIEW_TABLES]

    def _s3_object_path(self, s3_key: str) -> str:
        return f"{S3_BUCKET}/{s3_key}"

    def _preview_object(self, s3_key: str) -> tuple[list[str], list[dict[str, object]]]:
        extension = Path(s3_key).suffix.lower()

        try:
            if extension == ".parquet":
                return self._read_parquet_preview_from_s3(s3_key)
            if extension == ".csv":
                return self._read_csv_preview_from_s3(s3_key)

            for reader in (self._read_parquet_preview_from_s3, self._read_csv_preview_from_s3):
                try:
                    return reader(s3_key)
                except Exception:
                    continue

            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Supported formats: .parquet, .csv",
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse dataset preview for '{s3_key}': {exc}",
            ) from exc

    def _read_parquet_preview_from_s3(
        self, s3_key: str
    ) -> tuple[list[str], list[dict[str, object]]]:
        s3_path = self._s3_object_path(s3_key)
        filesystem = _get_s3_filesystem()
        parquet_file = pq.ParquetFile(s3_path, filesystem=filesystem)

        if parquet_file.metadata.num_rows == 0:
            return [], []

        if parquet_file.num_row_groups == 0:
            columns = [field.name for field in parquet_file.schema_arrow]
            return columns, []

        table = parquet_file.read_row_group(0)
        if table.num_rows > PREVIEW_ROW_LIMIT:
            table = table.slice(0, PREVIEW_ROW_LIMIT)
        dataframe = table.to_pandas()
        return list(dataframe.columns), self._dataframe_to_rows(dataframe)

    def _read_csv_preview_from_s3(self, s3_key: str) -> tuple[list[str], list[dict[str, object]]]:
        try:
            head = self.s3_client_internal.head_object(Bucket=S3_BUCKET, Key=s3_key)
            content_length = head.get("ContentLength", CSV_PREVIEW_RANGE_BYTES)
            range_end = max(0, min(content_length, CSV_PREVIEW_RANGE_BYTES) - 1)
            range_header = f"bytes=0-{range_end}"

            response = self.s3_client_internal.get_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Range=range_header,
            )
            body = response["Body"].read()
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"InvalidRange", "416"}:
                response = self.s3_client_internal.get_object(Bucket=S3_BUCKET, Key=s3_key)
                body = response["Body"].read()
            else:
                raise

        dataframe = pd.read_csv(io.BytesIO(body), nrows=PREVIEW_ROW_LIMIT)
        return list(dataframe.columns), self._dataframe_to_rows(dataframe)

    @staticmethod
    def _dataframe_to_rows(dataframe: pd.DataFrame) -> list[dict[str, object]]:
        trimmed = dataframe.head(PREVIEW_ROW_LIMIT)
        rows = json.loads(trimmed.to_json(orient="records", date_format="iso"))
        return cast(list[dict[str, object]], rows)
