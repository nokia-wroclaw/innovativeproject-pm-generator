"""S3 path helpers shared by CLI entrypoints and Airflow DAG arg builders."""

from __future__ import annotations


def s3a_path(bucket: str, key_or_path: str) -> str:
    value = (key_or_path or "").strip()
    if not value:
        return ""
    if value.startswith("s3a://"):
        return value
    return f"s3a://{bucket}/{value.lstrip('/')}"
