from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.client import Config


def write_json_to_s3(payload: dict[str, Any], *, bucket: str, key: str) -> None:
    """Serialize payload as JSON and upload to MinIO/S3 at the given bucket/key."""
    endpoint = os.environ.get("S3_URL", "http://minio:9000")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    print(f"Wrote artifact s3://{bucket}/{key}")
