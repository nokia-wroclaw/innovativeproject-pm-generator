import json
import os
import sys

import boto3
from botocore.client import Config
from pyspark.sql import SparkSession  # noqa: E402

from genpm.raw_vis.data_visualisation import (  # noqa: E402
    make_kpi_analysis,
    make_summary,
    top_kpis_for_analysis,
)
from genpm.raw_vis.s3_layout import visualization_artifact_key  # noqa: E402
from genpm.utils.consts import SPARK_CONFIGS
from genpm.utils.spark_session import SparkDataManager


def _ensure_spark_pythonpath() -> None:
    """PySpark lives under SPARK_HOME, not in genpm-venv."""
    spark_home = os.environ.get("SPARK_HOME", "/opt/spark")
    py4j_zip = f"{spark_home}/python/lib/py4j-0.10.9.7-src.zip"
    spark_python = f"{spark_home}/python"
    prefix = f"{spark_python}:{py4j_zip}"
    current = os.environ.get("PYTHONPATH", "")
    if spark_python not in current:
        os.environ["PYTHONPATH"] = f"{prefix}:{current}" if current else prefix


def _ensure_pyspark_python() -> None:
    """Re-exec driver with Airflow Python when Spark-submit picked a stale interpreter."""
    target = os.environ.get("GENPM_PYSPARK_PYTHON", "/opt/airflow/genpm-venv/bin/python")
    if not os.path.isfile(target):
        return
    if os.environ.get("GENPM_PY_REEXEC") == "1":
        return
    if os.path.realpath(sys.executable) == os.path.realpath(target):
        return
    os.environ["GENPM_PY_REEXEC"] = "1"
    os.environ["PYSPARK_DRIVER_PYTHON"] = target
    os.execv(target, [target, *sys.argv])


_ensure_pyspark_python()
_ensure_spark_pythonpath()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _build_spark() -> SparkSession:
    return SparkDataManager(
        app_name="DatasetVisualizationSparkJob",
        additional_conf=SPARK_CONFIGS["HALF_SAFE"],
    ).spark


def _write_json_to_s3(payload: dict, bucket: str, key: str) -> None:
    endpoint = _env("S3_URL", "http://minio:9000")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    print(f"Wrote visualization artifact s3://{bucket}/{key}")


def main() -> None:
    dataset_id = _env("GENPM_DATASET_ID")
    s3_key = _env("GENPM_S3_KEY")
    bucket = _env("S3_BUCKET", "test-bucket")

    if not dataset_id or not s3_key:
        raise RuntimeError("GENPM_DATASET_ID and GENPM_S3_KEY must be set")

    spark = _build_spark()
    spark_version = spark.version
    print(f"Spark version: {spark_version}")

    read_path = f"s3a://{bucket}/{s3_key.lstrip('/')}"
    print(f"Reading dataset from {read_path}")
    raw_df = spark.read.parquet(read_path)

    summary = make_summary(raw_df, spark_version=spark_version)
    artifact_name = (
        "summary_error.json" if summary.get("status") == "unsupported_schema" else "summary.json"
    )
    out_key = visualization_artifact_key(s3_key, artifact_name)
    _write_json_to_s3(summary, bucket, out_key)

    if summary.get("status") == "success":
        kpi_ids = top_kpis_for_analysis(summary)
        if kpi_ids:
            analysis = make_kpi_analysis(raw_df, kpi_ids)
            analysis_key = visualization_artifact_key(s3_key, "kpi_analysis.json")
            _write_json_to_s3(analysis, bucket, analysis_key)

    spark.stop()
    print("Dataset visualization job finished successfully")


if __name__ == "__main__":
    main()
