import os
from math import ceil
from pathlib import Path

from consts import SPARK_CHECKPOINT_PATH
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession

load_dotenv()


class SparkDataManager:
    def __init__(self, spark_conf: dict | None = None, checkpoint_dir: str | None = None) -> None:
        SPARK_CORE_NUMBER = os.getenv("SPARK_CORE_NUMBER") or "*"
        SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY") or "8g"
        SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY") or "8g"

        PARALLELISM_COUNT = str(ceil(2.5 * int(SPARK_CORE_NUMBER)))

        # spark session builder
        builder = (
            SparkSession.builder.master(f"local[{SPARK_CORE_NUMBER}]")
            .appName("GenPM")
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", PARALLELISM_COUNT)
            .config("spark.log.level", "WARNING")
            # Additional Spark optimizations
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
        )

        if spark_conf is not None:
            for conf, val in spark_conf.items():
                builder = builder.config(conf, val)

        self.spark = builder.getOrCreate()

        self.spark.sparkContext.setCheckpointDir(checkpoint_dir or str(SPARK_CHECKPOINT_PATH))

    @staticmethod
    def minio_spark_conf():
        # TODO: MINIO setup for sparksession
        pass

    def read_parquet(
        self,
        path: Path | str,
    ) -> DataFrame:
        # TODO: S3 compatability will be introduced here
        return self.spark.read.parquet(str(path) if isinstance(path, Path) else path)

    def write_parquet(self, df: DataFrame, path: Path | str, mode: str = "error"):
        df.write.parquet(path=str(path) if isinstance(path, Path) else path, mode=mode)
