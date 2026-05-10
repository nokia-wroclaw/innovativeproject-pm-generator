import os
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession

from .consts import SPARK_CHECKPOINT_PATH

load_dotenv()


class SparkDataManager:
    def __init__(self, spark_conf: dict | None = None, checkpoint_dir: str | None = None) -> None:
        SPARK_CORE_NUMBER = os.getenv("SPARK_CORE_NUMBER") or "8"
        SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY") or "10g"
        SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY") or "6g"
        print("\tSPARK DATA MANAGER")
        print(f"\n\tSPARK_CORE_NUMBER = {SPARK_CORE_NUMBER}")
        print(f"\tSPARK_EXECUTOR_MEMORY = {SPARK_EXECUTOR_MEMORY}")
        print(f"\tSPARK_DRIVER_MEMORY = {SPARK_DRIVER_MEMORY}")

        PARALLELISM_COUNT = "8"

        # spark session builder
        builder = (
            SparkSession.builder.master(f"local[{SPARK_CORE_NUMBER}]")
            .appName("GenPM")
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", PARALLELISM_COUNT)
            .config("spark.log.level", "WARN")
            # Additional Spark optimizations
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
        )

        if spark_conf is not None:
            for conf, val in spark_conf.items():
                builder = builder.config(conf, val)

        self.spark = builder.getOrCreate()

        checkpoint_directory = checkpoint_dir or str(SPARK_CHECKPOINT_PATH)

        if os.path.exists(checkpoint_directory):
            shutil.rmtree(checkpoint_directory)
        os.makedirs(checkpoint_directory, exist_ok=True)
        self.spark.sparkContext.setCheckpointDir(checkpoint_directory)

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

    def write_parquet(self, df: DataFrame, path: Path | str, mode: str = "error", **kwargs):
        string_path = str(path) if isinstance(path, Path) else path
        print(f"writing DataFrame to {string_path} ...")
        df.write.parquet(path=string_path, mode=mode, **kwargs)


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            "Config file not found. Copy pipeline_config.yaml.example "
            "to pipeline_config.yaml and fill in your values."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check nothing was left as null
    def check_nulls(d: dict, path: str = ""):
        for k, v in d.items():
            full_key = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                check_nulls(v, full_key)
            elif v is None:
                raise ValueError(
                    f"Config value '{full_key}' is not set — check pipeline_config.yaml"
                )

    check_nulls(config)
    return config
