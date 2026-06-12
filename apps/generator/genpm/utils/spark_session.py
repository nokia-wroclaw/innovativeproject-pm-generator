"""Spark session helpers for Airflow cluster submit and SDM (no yaml/config deps)."""

import atexit
import os
import signal
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from .logger import get_logger

logger = get_logger()


class SparkDataManager:
    def __init__(self, app_name: str | None = None, additional_conf: dict | None = None) -> None:
        SPARK_CORE_NUMBER = os.getenv("SPARK_CORE_NUMBER") or "8"
        SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY") or "10g"
        SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY") or "6g"
        SPARK_PARALLELISM_COUNT = os.getenv("SPARK_PARALLELISM_COUNT") or "8"
        logger.info("\tSPARK DATA MANAGER")

        app_name = app_name or "GenPM"

        # spark session builder
        builder = (
            SparkSession.builder.master(f"local[{SPARK_CORE_NUMBER}]")  # type: ignore
            .appName(app_name)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", SPARK_PARALLELISM_COUNT)
            .config("spark.log.level", "WARN")
            # Additional Spark optimizations
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            # Disable RAPIDS
            .config("spark.plugins", "")
            .config("spark.rapids.sql.enabled", "false")
            .config("spark.kryo.registrator", "")
        )

        for conf, val in minio_spark_conf().items():
            builder = builder.config(conf, val)

        if additional_conf is not None:
            logger.info(f"ADDITIONALL SPARK CONFIG ADDED: {additional_conf}")
            for conf, val in additional_conf.items():
                builder = builder.config(conf, val)

        self.spark: SparkSession = builder.getOrCreate()

        # STOPPING FUNCTIONS TO RESERVE SPACE AT ALL TIMES

        # Fires on normal exit and unhandled exceptions
        atexit.register(self._stop_spark)

        # Fires on SIGTERM (e.g. container stop, Airflow kill)
        signal.signal(signal.SIGTERM, lambda sig, frame: self._stop_spark())
        # Fires on SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, lambda sig, frame: self._stop_spark())

    def _stop_spark(self):
        if self.spark:
            self.spark.stop()
            self.spark = None  # type: ignore

    @staticmethod
    def minio_spark_conf() -> dict[str, str]:
        return minio_spark_conf()

    def read_parquet(self, path: Path | str, **options) -> DataFrame:
        # TODO: S3 compatability will be introduced here
        logger.info(f"Reading Dataframe from {str(path)} ...")
        return self.spark.read.parquet(str(path), **options)

    def write_parquet(self, df: DataFrame, path: Path | str, mode: str = "error", **kwargs):
        logger.info(f"Writing DataFrame to {str(path)} ...")
        df.write.parquet(path=str(path), mode=mode, **kwargs)

    def hard_checkpoint_to_parquet(self, df: DataFrame, path: Path | str) -> DataFrame:
        self.write_parquet(df, path, mode="overwrite")
        return self.read_parquet(path)


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
