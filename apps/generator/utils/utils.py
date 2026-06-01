import os
import re
from functools import reduce
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f

from .consts import SPARK_CHECKPOINT_PATH

load_dotenv()


class SparkDataManager:
    def __init__(self, spark_conf: dict | None = None, checkpoint_dir: str | None = None) -> None:
        SPARK_CORE_NUMBER = os.getenv("SPARK_CORE_NUMBER") or "8"
        SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY") or "10g"
        SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY") or "6g"
        print("\tSPARK DATA MANAGER")
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

        self.spark: SparkSession = builder.getOrCreate()

        checkpoint_directory = checkpoint_dir or str(SPARK_CHECKPOINT_PATH)

        self.spark.sparkContext.setCheckpointDir(checkpoint_directory)

    @staticmethod
    def minio_spark_conf():
        # TODO: MINIO setup for sparksession
        pass

    def read_parquet(self, path: Path | str, **options) -> DataFrame:
        # TODO: S3 compatability will be introduced here
        print(f"Reading Dataframe from {str(path)} ...")
        return self.spark.read.parquet(str(path), **options)

    def write_parquet(self, df: DataFrame, path: Path | str, mode: str = "error", **kwargs):
        print(f"Writing DataFrame to {str(path)} ...")
        df.write.parquet(path=str(path), mode=mode, **kwargs)

    def hard_checkpoint_to_parquet(self, df: DataFrame, path: Path | str) -> DataFrame:
        self.write_parquet(df, path, mode="overwrite")
        return self.read_parquet(path)


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


def when_chained(conditions: list[tuple], otherwise=None) -> Column:
    def reducer(acc, pair):
        cond, val = pair
        return acc.when(cond, val)

    first_cond, first_val = conditions[0]
    chain = reduce(reducer, conditions[1:], f.when(first_cond, first_val))
    return chain.otherwise(otherwise)


def make_pattern(words):
    return r"(?i)(^|[^a-z0-9])(" + "|".join(re.escape(w) for w in words) + r")([^a-z0-9]|$)"


def classify_kpis(
    df: DataFrame,
    avg_keywords: list[str],
    max_keywords: list[str],
    min_keywords: list[str],
    mean_like_keywords: list[str],
    ratio_keywords: list[str],
    volume_keywords: list[str],
    mean_like_units: list[str],
    volume_units: list[str],
) -> DataFrame:
    ratio_pattern = make_pattern(ratio_keywords)
    mean_like_pattern = make_pattern(mean_like_keywords)
    volume_pattern = make_pattern(volume_keywords)
    avg_pattern = make_pattern(avg_keywords)
    max_pattern = make_pattern(max_keywords)
    min_pattern = make_pattern(min_keywords)

    kpi_classified = (
        df.withColumn(
            "stat_keyword_match",
            when_chained(
                [
                    (f.col("kpi_name").rlike(avg_pattern), "avg"),
                    (f.col("kpi_name").rlike(max_pattern), "max"),
                    (f.col("kpi_name").rlike(min_pattern), "min"),
                ],
                otherwise="unknown",
            ),
        )
        .withColumn(
            "unit_match",
            when_chained(
                [
                    (f.col("unit").isin(*mean_like_units, "mean_like"), "mean_like"),
                    (f.col("unit").isin(*volume_units, "volume"), "volume"),
                ],
                otherwise="unknown",
            ),
        )
        .withColumn(
            "keyword_match",
            when_chained(
                [
                    (f.col("kpi_name").rlike(ratio_pattern), "ratio"),
                    (f.col("kpi_name").rlike(mean_like_pattern), "mean_like"),
                    (f.col("kpi_name").rlike(volume_pattern), "volume"),
                ],
                otherwise="unknown",
            ),
        )
        .withColumn(
            "kpi_character",
            when_chained(
                [
                    (f.col("stat_keyword_match") != "unknown", f.col("stat_keyword_match")),
                    (f.col("unit_match") != "unknown", f.col("unit_match")),
                    (f.col("keyword_match") != "unknown", f.col("keyword_match")),
                    (f.col("kpi_min") > 0, "mean_like"),
                    ((f.col("kpi_min") >= 0) & (f.col("kpi_max") <= 100), "ratio"),
                ],
                otherwise="unknown",
            ),
        )
        .withColumn(
            "classification_source",
            when_chained(
                [
                    (f.col("stat_keyword_match") != "unknown", "stat_keyword"),
                    (f.col("unit_match").isin("mean_like", "volume"), "unit"),
                    (f.col("keyword_match") != "unknown", "keyword"),
                ],
                otherwise="value_range_fallback",
            ),
        )
        .withColumn(
            "agg_method",
            when_chained(
                [
                    (
                        f.col("kpi_character").isin("mean_like", "ratio", "avg"),
                        "avg",
                    ),
                    (f.col("kpi_character") == "max", "max"),
                    (f.col("kpi_character") == "min", "min"),
                ],
                otherwise="sum",
            ),
        )
        .drop("keyword_match")
    )
    return kpi_classified
