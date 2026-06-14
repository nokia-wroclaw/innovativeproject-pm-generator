"""Spark session helpers for Airflow cluster submit and local CLI runs (no yaml/config deps).

One session contract for both execution models:

* Under Airflow ``SparkSubmitOperator`` the master / memory / cores are supplied by
  ``spark-submit --master ... --conf ...``. We must **not** override ``spark.master`` here, or
  compute silently collapses to local mode inside the Airflow worker.
* When run as a bare CLI (``python -m genpm.preprocessing ...``) or a notebook there is no
  spark-submit, so we fall back to ``local[N]`` sized from ``SPARK_CORE_NUMBER``.
"""

from __future__ import annotations

import atexit
import os
import signal
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from .logger import get_logger

logger = get_logger()

# App-level (non-resource) defaults that are safe in every execution mode. Resource sizing
# (master / memory / cores / shuffle partitions) is intentionally NOT set here — it comes from
# spark-submit under Airflow, or from the local[N] fallback below.
_APP_CONF: dict[str, str] = {
    "spark.log.level": "WARN",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    # RAPIDS is enabled by default in the Spark image's spark-defaults.conf, but the genpm
    # pipeline relies on operations the plugin may not accelerate; keep it off for correctness.
    "spark.plugins": "",
    "spark.rapids.sql.enabled": "false",
    "spark.kryo.registrator": "",
}


class SparkDataManager:
    def __init__(self, app_name: str | None = None, additional_conf: dict | None = None) -> None:
        logger.info("\tSPARK DATA MANAGER")

        builder = SparkSession.builder.appName(app_name or "GenPM")  # type: ignore[attr-defined]

        for conf, val in _APP_CONF.items():
            builder = builder.config(conf, val)

        for conf, val in minio_spark_conf().items():
            builder = builder.config(conf, val)

        # Under spark-submit (Airflow) the master / memory / cores come from --master / --conf — we
        # must NOT set them here or compute collapses to local mode in the Airflow worker. Only a
        # genuine bare-CLI / notebook run (no spark-submit) needs the local[N] fallback.
        if _running_under_spark_submit():
            logger.info("Running under spark-submit — using its master/conf (no local override)")
        else:
            cores = os.getenv("SPARK_CORE_NUMBER") or "8"
            logger.info(f"No spark-submit detected — defaulting to local[{cores}]")
            builder = (
                builder.master(f"local[{cores}]")
                .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY") or "6g")
                .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY") or "10g")
                .config(
                    "spark.sql.shuffle.partitions", os.getenv("SPARK_PARALLELISM_COUNT") or "200"
                )
            )

        if additional_conf:
            logger.info(f"Additional Spark config added: {additional_conf}")
            for conf, val in additional_conf.items():
                builder = builder.config(conf, val)

        self.spark: SparkSession = builder.getOrCreate()
        logger.info(f"Spark master: {self.spark.sparkContext.master}")

        # Always release the cluster slot, even on crash / container stop / Ctrl+C.
        atexit.register(self.stop)
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT, lambda *_: self.stop())

    def stop(self) -> None:
        if getattr(self, "spark", None) is not None:
            self.spark.stop()
            self.spark = None  # type: ignore[assignment]

    # Backwards-compatible alias for existing callers.
    _stop_spark = stop

    def __enter__(self) -> SparkDataManager:
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    @staticmethod
    def minio_spark_conf() -> dict[str, str]:
        return minio_spark_conf()

    def read_parquet(self, path: Path | str, **options) -> DataFrame:
        logger.info(f"Reading Dataframe from {str(path)} ...")
        return self.spark.read.parquet(str(path), **options)

    def write_parquet(self, df: DataFrame, path: Path | str, mode: str = "error", **kwargs):
        logger.info(f"Writing DataFrame to {str(path)} ...")
        df.write.parquet(path=str(path), mode=mode, **kwargs)

    def hard_checkpoint_to_parquet(self, df: DataFrame, path: Path | str) -> DataFrame:
        self.write_parquet(df, path, mode="overwrite")
        return self.read_parquet(path)


def _running_under_spark_submit() -> bool:
    """True when this process was launched by spark-submit (so a master is already configured).

    Must be decidable *before* the Spark gateway/JVM exists — a fresh ``SparkConf()`` is empty at
    that point and cannot see the CLI ``--master``. We instead use deterministic env signals:

    * ``GENPM_SPARK_SUBMIT=1`` — set by our Airflow DAG layer (``lib.spark_config.infra_env_vars``).
    * ``PYSPARK_GATEWAY_PORT`` — set by spark-submit's ``PythonRunner`` for any submitted Python app
      (covers a manual ``spark-submit run_*.py`` too).
    """
    return os.environ.get("GENPM_SPARK_SUBMIT") == "1" or "PYSPARK_GATEWAY_PORT" in os.environ


def minio_spark_conf() -> dict[str, str]:
    """Hadoop s3a settings for MinIO / S3-compatible storage (from env)."""
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        logger.warning(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set — s3a reads/writes will fail."
        )
    return {
        "spark.hadoop.fs.s3a.endpoint": os.getenv("S3_URL", "http://minio:9000"),
        "spark.hadoop.fs.s3a.access.key": access_key,
        "spark.hadoop.fs.s3a.secret.key": secret_key,
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    }
