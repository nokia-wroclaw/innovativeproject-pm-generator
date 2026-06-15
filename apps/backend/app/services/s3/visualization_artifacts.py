import json
import logging
from pathlib import PurePosixPath
from typing import Any

from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from app.models.auth import TokenPayload
from app.services.s3.service import get_s3_client_internal

logger = logging.getLogger(__name__)


class VisualizationStorageError(RuntimeError):
    """S3 is not configured for visualization artifact storage."""


def _require_s3_bucket() -> str:
    from app.core.config import get_settings

    bucket = get_settings().s3_bucket
    if not bucket:
        raise VisualizationStorageError(
            "S3_BUCKET is not configured; cannot read or write visualization artifacts."
        )
    return bucket


def dataset_visualization_prefix(dataset_s3_key: str, dataset_type: str = "RAW") -> str:
    key = (dataset_s3_key or "").strip().lstrip("/")
    if not key:
        raise ValueError("dataset s3_key is empty")

    if str(dataset_type) in ("GENERATED", "PREPROCESSED"):
        return key

    parent = PurePosixPath(key).parent.as_posix()
    if parent and parent != ".":
        return parent
    return PurePosixPath(key).stem


def visualization_artifact_key(
    dataset_s3_key: str, filename: str, dataset_type: str = "RAW"
) -> str:
    base = dataset_visualization_prefix(dataset_s3_key, dataset_type).strip("/")
    return f"{base}/{filename.lstrip('/')}"


def visualization_artifact_keys(dataset_s3_key: str, dataset_type: str = "RAW") -> tuple[str, str]:
    return (
        visualization_artifact_key(dataset_s3_key, "summary.json", dataset_type),
        visualization_artifact_key(dataset_s3_key, "summary_error.json", dataset_type),
    )


def kpi_analysis_artifact_key(dataset_s3_key: str, dataset_type: str = "RAW") -> str:
    return visualization_artifact_key(dataset_s3_key, "kpi_analysis.json", dataset_type)


def read_s3_json_artifact(key: str) -> dict[str, Any] | None:
    bucket = _require_s3_bucket()
    try:
        response = get_s3_client_internal().get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        payload = json.loads(body.decode("utf-8"))
        if isinstance(payload, dict):
            return payload
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in {"404", "NoSuchKey", "NotFound"}:
            logger.warning("Failed to read visualization artifact %s: %s", key, exc)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in visualization artifact %s", key)
    return None


def load_kpi_analysis_artifact(
    dataset_s3_key: str, dataset_type: str = "RAW"
) -> dict[str, Any] | None:
    return read_s3_json_artifact(kpi_analysis_artifact_key(dataset_s3_key, dataset_type))


def load_visualization_artifact(
    dataset_s3_key: str, dataset_type: str = "RAW"
) -> dict[str, Any] | None:
    summary_key, error_key = visualization_artifact_keys(dataset_s3_key, dataset_type)
    for key in (summary_key, error_key):
        payload = read_s3_json_artifact(key)
        if payload is not None:
            return payload
    return None


def delete_visualization_error_artifact(dataset_s3_key: str, dataset_type: str = "RAW") -> None:
    """Remove a stale summary_error.json written by an older schema check or Spark run."""
    bucket = _require_s3_bucket()
    _, error_key = visualization_artifact_keys(dataset_s3_key, dataset_type)
    try:
        get_s3_client_internal().delete_object(Bucket=bucket, Key=error_key)
    except ClientError as exc:
        logger.warning(
            "Failed to delete visualization error artifact for s3_key=%s: %s",
            dataset_s3_key,
            exc,
        )


def persist_unsupported_schema_artifact(
    dataset_s3_key: str, payload: TokenPayload, dataset_type: str = "RAW"
) -> None:
    bucket = _require_s3_bucket()
    _, error_key = visualization_artifact_keys(dataset_s3_key, dataset_type)
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    try:
        get_s3_client_internal().put_object(
            Bucket=bucket,
            Key=error_key,
            Body=body,
            ContentType="application/json",
        )
    except ClientError as exc:
        logger.warning(
            "Failed to write unsupported_schema artifact for s3_key=%s: %s", dataset_s3_key, exc
        )


def status_from_artifact(artifact: dict[str, Any]) -> str:
    raw_status = artifact.get("status")
    if raw_status == "unsupported_schema":
        return "unsupported_schema"
    if raw_status == "success":
        return "success"
    return "success"
