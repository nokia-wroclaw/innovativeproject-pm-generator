"""Spark session helpers for Airflow cluster submit and SDM (no yaml/config deps)."""

import os

from pyspark.sql import SparkSession

from genpm.utils.consts import SPARK_CONFIGS

_CLUSTER_PRESET_SKIP_KEYS = frozenset(
    {
        "spark.master",
        "spark.driver.memory",
    }
)


def minio_spark_conf() -> dict[str, str]:
    """Hadoop s3a settings for MinIO / S3-compatible storage (from env)."""
    return {
        "spark.hadoop.fs.s3a.endpoint": os.getenv("S3_URL", "http://minio:9000"),
        "spark.hadoop.fs.s3a.access.key": os.getenv("AWS_ACCESS_KEY_ID", "your_default_access_key"),
        "spark.hadoop.fs.s3a.secret.key": os.getenv(
            "AWS_SECRET_ACCESS_KEY", "your_default_secret_key"
        ),
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    }


def build_cluster_spark_session(
    app_name: str,
    *,
    profile: str | None = None,
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """Spark session for Airflow spark-submit (cluster/client mode).

    Uses SPARK_CONFIGS presets from consts.py for query tuning and SPARK_* env vars
    for executor/driver memory and core limits.
    """
    profile_name = profile or os.getenv("GENPM_SPARK_CONFIG", "AGG_HEAVY")
    preset = dict(SPARK_CONFIGS.get(profile_name, SPARK_CONFIGS["HALF_SAFE"]))
    for key in _CLUSTER_PRESET_SKIP_KEYS:
        preset.pop(key, None)

    cores = os.getenv("SPARK_CORE_NUMBER") or "8"
    executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY") or "10g"
    driver_memory = os.getenv("SPARK_DRIVER_MEMORY") or "6g"

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.executor.cores", cores)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        .config("spark.cores.max", cores)
        .config("spark.default.parallelism", cores)
        .config("spark.log.level", "WARN")
    )

    for conf, val in preset.items():
        builder = builder.config(conf, str(val))

    if extra_conf:
        for conf, val in extra_conf.items():
            builder = builder.config(conf, str(val))

    return builder.getOrCreate()
