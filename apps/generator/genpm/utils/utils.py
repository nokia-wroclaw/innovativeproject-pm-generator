import atexit
import os
import re
import shutil
import signal
import subprocess
from functools import reduce
from pathlib import Path

import yaml
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f

from .consts import SPARK_CHECKPOINT_PATH
from .logger import get_logger

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = get_logger()

PREPROCESSED_S3_PREFIX = "preprocessed"


def resolve_storage_path(path: Path | str) -> str:
    """Map local paths, S3 keys and s3:// URIs to a Spark-readable location."""
    raw = str(path).strip()
    if not raw:
        raise ValueError("Empty storage path")

    if raw.startswith("s3a://"):
        return raw
    if raw.startswith("s3://"):
        return f"s3a://{raw[5:]}"
    if raw.startswith("file://"):
        return raw.removeprefix("file://")
    if raw.startswith("/"):
        return raw

    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        return raw
    return f"s3a://{bucket}/{raw.lstrip('/')}"


def default_preprocessed_output_prefix(run_id: str) -> str:
    bucket = os.getenv("S3_BUCKET", "datasets").strip() or "datasets"
    return f"s3a://{bucket}/{PREPROCESSED_S3_PREFIX}/{run_id}"


class SparkDataManager:
    def __init__(self, spark_conf: dict | None = None, checkpoint_dir: str | None = None) -> None:
        SPARK_CORE_NUMBER = os.getenv("SPARK_CORE_NUMBER") or "8"
        SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY") or "10g"
        SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY") or "6g"
        logger.info("\tSPARK DATA MANAGER")
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

        for conf, val in self.s3_spark_conf().items():
            builder = builder.config(conf, val)

        if spark_conf is not None:
            for conf, val in spark_conf.items():
                builder = builder.config(conf, val)

        self.spark: SparkSession = builder.getOrCreate()

        checkpoint_directory = checkpoint_dir or str(SPARK_CHECKPOINT_PATH)
        checkpoint_path = Path(checkpoint_directory)

        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        shared_group = os.getenv("USER")
        if shared_group:
            subprocess.run(["chgrp", "-R", shared_group, str(checkpoint_path)], check=False)
            subprocess.run(["chmod", "g+w", str(checkpoint_path)], check=False)

        self.spark.sparkContext.setCheckpointDir(checkpoint_directory)
        logger.warning(f"CHECKPOINT DIR SET TO {checkpoint_directory}")

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
    def s3_spark_conf() -> dict[str, str]:
        endpoint = (os.getenv("S3_URL") or os.getenv("S3_ENDPOINT") or "").rstrip("/")
        access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        if not endpoint or not access_key or not secret_key:
            return {}

        return {
            "spark.hadoop.fs.s3a.endpoint": endpoint,
            "spark.hadoop.fs.s3a.access.key": access_key,
            "spark.hadoop.fs.s3a.secret.key": secret_key,
            "spark.hadoop.fs.s3a.path.style.access": "true",
            "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        }

    def read_parquet(self, path: Path | str, **options) -> DataFrame:
        resolved = resolve_storage_path(path)
        logger.info(f"Reading Dataframe from {resolved} ...")
        return self.spark.read.parquet(resolved, **options)

    def write_parquet(self, df: DataFrame, path: Path | str, mode: str = "error", **kwargs):
        resolved = resolve_storage_path(path)
        logger.info(f"Writing DataFrame to {resolved} ...")
        df.write.parquet(path=resolved, mode=mode, **kwargs)

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
    def _reducer(acc, pair):
        cond, val = pair
        return acc.when(cond, val)

    first_cond, first_val = conditions[0]
    chain = reduce(_reducer, conditions[1:], f.when(first_cond, first_val))
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


def validate_windowed_pm(
    df: DataFrame,
    window_hours: int = 168,
) -> None:
    """
    Validate the output of attach_windows_index_to_pm.
    Checks structural integrity, coverage completeness, and data quality.
    All findings are logged — no exceptions raised unless critical.
    """

    logger.info("=" * 60)
    logger.info("[validate] starting windowed dataframe validation")
    logger.info("=" * 60)

    # ── 1. Basic shape ─────────────────────────────────────────────────────
    total_rows = df.count()
    n_distnames = df.select("distname").distinct().count()
    n_kpis = df.select("kpi_id").distinct().count()
    n_anchors = df.select("distname", "window_anchor").distinct().count()

    logger.info(f"[shape] total rows:      {total_rows:>15,}")
    logger.info(f"[shape] distnames:       {n_distnames:>15,}")
    logger.info(f"[shape] kpis:            {n_kpis:>15,}")
    logger.info(f"[shape] unique windows:  {n_anchors:>15,}")

    expected_rows = n_anchors * n_kpis * window_hours
    logger.info(
        f"[shape] expected rows:   {expected_rows:>15,}  (windows × kpis × {window_hours}h)"
    )
    if total_rows != expected_rows:
        logger.warning(
            f"[shape] FAIL row count mismatch: got {total_rows:,}, expected {expected_rows:,}"
        )
    else:
        logger.info("[shape] OK row count matches expected")

    # ── 2. Null check ──────────────────────────────────────────────────────
    n_null_values = df.filter(f.col("kpi_value").isNull()).count()
    n_null_flags = df.filter(f.col("imputed_flag").isNull()).count()

    if n_null_values > 0:
        logger.warning(f"[nulls] FAIL kpi_value nulls: {n_null_values:,}")
    else:
        logger.info("[nulls] OK no kpi_value nulls")

    if n_null_flags > 0:
        logger.warning(f"[nulls] FAIL imputed_flag nulls: {n_null_flags:,}")
    else:
        logger.info("[nulls] OK no imputed_flag nulls")

    # ── 3. hour_idx range ─────────────────────────────────────────────────
    # Every (distname, window_anchor, kpi_id) must have exactly
    # hour_idx 0 to window_hours-1 — no gaps, no out-of-range values
    hour_stats = df.groupBy("distname", "window_anchor", "kpi_id").agg(
        f.count("*").alias("n_hours"),
        f.min("hour_idx").alias("min_hour"),
        f.max("hour_idx").alias("max_hour"),
    )

    bad_hour_counts = hour_stats.filter(f.col("n_hours") != window_hours)
    bad_hour_range = hour_stats.filter(
        (f.col("min_hour") != 0) | (f.col("max_hour") != window_hours - 1)
    )

    n_bad_counts = bad_hour_counts.count()
    n_bad_range = bad_hour_range.count()

    if n_bad_counts > 0:
        logger.warning(f"[hours] FAIL combos with wrong hour count: {n_bad_counts:,}")
        bad_hour_counts.orderBy("n_hours").show(10, truncate=False)
    else:
        logger.info(f"[hours] OK all combos have exactly {window_hours} hours")

    if n_bad_range > 0:
        logger.warning(f"[hours] FAIL combos with wrong hour_idx range: {n_bad_range:,}")
        bad_hour_range.show(10, truncate=False)
    else:
        logger.info(f"[hours] OK all combos span hour_idx 0 → {window_hours - 1}")

    # ── 4. KPI consistency across windows ─────────────────────────────────
    # Every window_anchor for a given distname should have the same set of KPIs
    # A distname that loses KPIs across windows will produce ragged tensors
    kpis_per_window = df.groupBy("distname", "window_anchor").agg(
        f.countDistinct("kpi_id").alias("n_kpis")
    )

    kpi_count_variance = (
        kpis_per_window.groupBy("distname")
        .agg(
            f.min("n_kpis").alias("min_kpis"),
            f.max("n_kpis").alias("max_kpis"),
        )
        .filter(f.col("min_kpis") != f.col("max_kpis"))
    )

    n_inconsistent = kpi_count_variance.count()
    if n_inconsistent > 0:
        logger.warning(
            f"[kpis] FAIL {n_inconsistent:,} distnames have inconsistent KPI counts across windows"
        )
        kpi_count_variance.show(10, truncate=False)
    else:
        logger.info("[kpis] OK all distnames have consistent KPI count across windows")

    # ── 5. Imputation flag distribution ───────────────────────────────────
    # flag=0: real observed value
    # flag=1: filled by upstream per-KPI imputation
    # flag=2: filled by cross-KPI alignment (should be rare)
    flag_dist = (
        df.groupBy("imputed_flag")
        .agg(f.count("*").alias("n_rows"))
        .withColumn("pct", f.round(f.col("n_rows") / total_rows * 100, 2))
        .orderBy("imputed_flag")
    )

    logger.info("[flags] imputed_flag distribution:")
    flag_dist.show(truncate=False)

    # Warn if cross-KPI imputation (flag=2) is unexpectedly high
    flag2_row = flag_dist.filter(f.col("imputed_flag") == 2).collect()
    if flag2_row:
        pct_flag2 = flag2_row[0]["pct"]
        if pct_flag2 > 1.0:
            logger.warning(
                f"[flags] FAIL {pct_flag2}% of rows are cross-KPI imputed (flag=2) "
                f"— joint range trimming may not have been applied"
            )
        else:
            logger.info(f"[flags] OK cross-KPI imputation (flag=2) is {pct_flag2}% — acceptable")

    # ── 6. Windows per distname ────────────────────────────────────────────
    # Flags distnames that ended up with very few windows after alignment —
    # these will be underrepresented in CVAE training
    windows_per_distname = (
        df.select("distname", "window_anchor")
        .distinct()
        .groupBy("distname")
        .agg(f.count("*").alias("n_windows"))
        .orderBy("n_windows")
    )

    stats = windows_per_distname.agg(
        f.min("n_windows").alias("min"),
        f.max("n_windows").alias("max"),
        f.mean("n_windows").alias("mean"),
        f.percentile_approx("n_windows", 0.10).alias("p10"),
        f.percentile_approx("n_windows", 0.50).alias("p50"),
    ).collect()[0]

    logger.info(
        f"[windows/distname] min={stats['min']} | p10={stats['p10']} | "
        f"p50={stats['p50']} | mean={stats['mean']:.1f} | max={stats['max']}"
    )

    sparse_distnames = windows_per_distname.filter(f.col("n_windows") < 30)
    n_sparse = sparse_distnames.count()
    if n_sparse > 0:
        logger.warning(
            f"[windows/distname] FAIL {n_sparse:,} distnames have fewer than 30 windows "
            f"— may be insufficient for CVAE training"
        )
        sparse_distnames.show(20, truncate=False)
    else:
        logger.info("[windows/distname] OK all distnames have >= 30 windows")

    logger.info("=" * 60)
    logger.info("[validate] done")
    logger.info("=" * 60)
