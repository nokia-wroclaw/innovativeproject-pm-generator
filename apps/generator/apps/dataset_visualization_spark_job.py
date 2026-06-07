import json
import os
import sys
from pathlib import Path


def _ensure_pyspark_python() -> None:
    """Re-exec driver with Airflow Python when Spark-submit picked a stale interpreter."""
    target = os.environ.get("GENPM_PYSPARK_PYTHON", "/usr/python/bin/python3.13")
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

from pyspark.sql import SparkSession  # noqa: E402


def _resolve_generator_root() -> Path:
    """Locate apps/generator root (contains genpm/raw_vis/) in Airflow or local dev."""
    env_root = os.environ.get("GENPM_GENERATOR_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
        if (root / "genpm" / "raw_vis").is_dir():
            return root

    here = Path(__file__).resolve()
    for candidate in (here.parents[1], Path("/opt/airflow/generator")):
        if (candidate / "genpm" / "raw_vis").is_dir():
            return candidate

    raise RuntimeError(
        "Cannot find generator package root (genpm/raw_vis/). "
        "Mount apps/generator to /opt/airflow/generator or set GENPM_GENERATOR_ROOT."
    )


_GENERATOR_ROOT = _resolve_generator_root()
_gen_root_str = str(_GENERATOR_ROOT)
if _gen_root_str not in sys.path:
    sys.path.insert(0, _gen_root_str)

from genpm.raw_vis.data_visualisation import (  # noqa: E402
    make_kpi_analysis,
    make_summary,
    top_kpis_for_analysis,
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _build_spark() -> SparkSession:
    s3_endpoint = _env("S3_URL", "http://minio:9000")
    access_key = _env("AWS_ACCESS_KEY_ID", "your_default_access_key")
    secret_key = _env("AWS_SECRET_ACCESS_KEY", "your_default_secret_key")

    return (
        SparkSession.builder.appName("DatasetVisualizationSparkJob")
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


def _artifact_s3_key(dataset_id: str, filename: str) -> str:
    prefix = _env("GENPM_VIZ_PREFIX", "genpm/visualizations").strip("/")
    return f"{prefix}/{dataset_id}/{filename}"


def _write_json_to_s3(payload: dict, bucket: str, key: str) -> None:
    import boto3
    from botocore.client import Config

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
    out_key = _artifact_s3_key(dataset_id, artifact_name)
    _write_json_to_s3(summary, bucket, out_key)

    if summary.get("status") == "success":
        kpi_ids = top_kpis_for_analysis(summary)
        if kpi_ids:
            analysis = make_kpi_analysis(raw_df, kpi_ids)
            analysis_key = _artifact_s3_key(dataset_id, "kpi_analysis.json")
            _write_json_to_s3(analysis, bucket, analysis_key)

    spark.stop()
    print("Dataset visualization job finished successfully")


if __name__ == "__main__":
    main()
