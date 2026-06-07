import json
import logging
import os
from typing import Any

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

GENPM_VIZ_PREFIX = os.getenv("GENPM_VIZ_PREFIX", "genpm/visualizations")


def visualization_artifact_keys(dataset_id: int) -> tuple[str, str]:
    base = f"{GENPM_VIZ_PREFIX.strip('/')}/{dataset_id}"
    return f"{base}/summary.json", f"{base}/summary_error.json"


def kpi_analysis_artifact_key(dataset_id: int) -> str:
    base = f"{GENPM_VIZ_PREFIX.strip('/')}/{dataset_id}"
    return f"{base}/kpi_analysis.json"


def read_s3_json_artifact(key: str) -> dict[str, Any] | None:
    from app.services.s3.service import S3_BUCKET, s3_client_internal

    if not S3_BUCKET:
        return None
    try:
        response = s3_client_internal.get_object(Bucket=S3_BUCKET, Key=key)
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


def load_kpi_analysis_artifact(dataset_id: int) -> dict[str, Any] | None:
    return read_s3_json_artifact(kpi_analysis_artifact_key(dataset_id))


def load_visualization_artifact(dataset_id: int) -> dict[str, Any] | None:
    summary_key, error_key = visualization_artifact_keys(dataset_id)
    for key in (summary_key, error_key):
        payload = read_s3_json_artifact(key)
        if payload is not None:
            return payload
    return None


def persist_unsupported_schema_artifact(dataset_id: int, payload: dict[str, Any]) -> None:
    from app.services.s3.service import S3_BUCKET, s3_client_internal

    if not S3_BUCKET:
        return

    _, error_key = visualization_artifact_keys(dataset_id)
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    try:
        s3_client_internal.put_object(
            Bucket=S3_BUCKET,
            Key=error_key,
            Body=body,
            ContentType="application/json",
        )
    except ClientError as exc:
        logger.warning(
            "Failed to write unsupported_schema artifact for dataset_id=%s: %s",
            dataset_id,
            exc,
        )


def status_from_artifact(artifact: dict[str, Any]) -> str:
    raw_status = artifact.get("status")
    if raw_status == "unsupported_schema":
        return "unsupported_schema"
    if raw_status == "success":
        return "success"
    return "success"
