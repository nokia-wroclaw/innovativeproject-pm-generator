from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from utils.consts import (
    AVG_KEYWORDS,
    MAX_IMPUTABLE_GAP,
    MAX_KEYWORDS,
    MEAN_LIKE_KEYWORDS,
    MEAN_LIKE_UNITS,
    MIN_KEYWORDS,
    RATIO_KEYWORDS,
    SHARED_DIR_PATH,
    SPARK_CONFIGS,
    VOLUME_KEYWORDS,
    VOLUME_UNITS,
)
from utils.utils import classify_kpis, when_chained

cfg = SPARK_CONFIGS["WINDOW_HEAVY"]

spark = (
    SparkSession.builder.appName("GenPM-fe")
    .config(map=cfg)
    .config("spark.log.level", "ERROR")
    .getOrCreate()
)

EDA_DATA_PATH = SHARED_DIR_PATH / "eda_data"
raw_pm_path = EDA_DATA_PATH / "raw_pm_data"
pm_kpi_pivot = EDA_DATA_PATH / "pm_data_pivot"
sample_path = EDA_DATA_PATH / "sample"
pm_stats_path = EDA_DATA_PATH / "stats"
pm_agg_path = EDA_DATA_PATH / "agg"
pm_metadata = EDA_DATA_PATH / "pm_metadata"
PREPROCESSED_DATASET_PATH = SHARED_DIR_PATH / "preprocessed_dataset"
long_path = PREPROCESSED_DATASET_PATH / "pm_data_long"
wide_path = PREPROCESSED_DATASET_PATH / "pm_data_wide"

pm_df_long = spark.read.parquet(str(long_path))

kpi_defs = spark.read.parquet(str(pm_metadata / "kpis_definitions"))


def prepare_data_for_imputing(df: DataFrame) -> DataFrame:
    # clean kpi_ids after versioning
    kpi_defs_clean = kpi_defs.withColumn(
        "kpi_id", f.regexp_replace(f.col("kpi_id" ""), r"[a-zA-Z]$", "")
    )
    # calculate stats required for kpi agg character classification
    kpi_stats = pm_df_long.groupBy("kpi_id").agg(
        f.round(f.min("kpi_value"), 4).alias("kpi_min"),
        f.round(f.max("kpi_value"), 4).alias("kpi_max"),
        f.round(f.sum(f.col("kpi_value").isNull().cast("int")) / f.count("*") * 100, 2).alias(
            "null_pct"
        ),
    )
    # join to have table ready for kpi agg character classification
    pm_df_with_defs = kpi_stats.join(kpi_defs_clean, on="kpi_id", how="left")

    pm_classfied = classify_kpis(
        pm_df_with_defs,
        MEAN_LIKE_UNITS,
        VOLUME_UNITS,
        MIN_KEYWORDS,
        MAX_KEYWORDS,
        AVG_KEYWORDS,
        RATIO_KEYWORDS,
        MEAN_LIKE_KEYWORDS,
        VOLUME_KEYWORDS,
    )

    pm_df = pm_df_long.join(pm_classfied.select("kpi_id", "agg_method"), on="kpi_id", how="left")

    return pm_df


def detect_gap_runs(
    df: DataFrame, group_cols: list[str], order_col: str, value_col: str, max_imputable_gap: int = 6
) -> DataFrame:
    w = Window.partitionBy(*group_cols).orderBy(order_col)

    base = (
        df.withColumn("is_missing", f.col(value_col).isNull().cast("int"))
        .withColumn("prev_missing", f.lag("is_missing").over(w))
        .withColumn(
            "gap_start_flag",
            f.when(
                (f.col("is_missing") == 1)
                & (f.col("prev_missing").isNull() | (f.col("prev_missing") == 0)),
                1,
            ).otherwise(0),
        )
        .withColumn("gap_id", f.sum("gap_start_flag").over(w))
    )

    gap_runs = (
        base.filter(f.col("is_missing") == 1)
        .groupBy(*(group_cols + ["gap_id"]))
        .agg(
            #   f.min(order_col).alias("gap_start_ts"),
            #   f.max(order_col).alias("gap_end_ts"),
            f.count(f.lit(1)).alias("gap_len"),
        )
    )

    output_df = (
        base.join(gap_runs, on=group_cols + ["gap_id"], how="left")
        .withColumn("gap_len", f.coalesce(f.col("gap_len"), f.lit(0)))
        .withColumn("is_imputable", f.col("gap_len") <= max_imputable_gap)
    )
    return output_df


def build_series_stats(
    df: DataFrame, group_cols: list[str], value_col: str, agg_method_col: str
) -> DataFrame:
    df_stats = df.groupBy(*group_cols).agg(
        f.first(agg_method_col, ignorenulls=True).alias("agg_method"),
        f.min(value_col).alias("min_value"),
        f.max(value_col).alias("max_value"),
        f.expr(f"percentile_approx({value_col}, 0.5)").alias("median_value"),
    )
    return df_stats


def impute(
    df: DataFrame,
    group_cols: list[str],
    order_col: str,
    value_col: str,
    agg_method_col: str,
    max_imputable_gap: int = 6,
) -> DataFrame:
    tagged_df = detect_gap_runs(df, group_cols, order_col, value_col, max_imputable_gap).drop(
        agg_method_col
    )
    stats = build_series_stats(df, group_cols, value_col, agg_method_col)
    x = tagged_df.join(f.broadcast(stats), on=group_cols, how="left")

    w_prev = (
        Window.partitionBy(*group_cols).orderBy(order_col).rowsBetween(Window.unboundedPreceding, 0)
    )
    w_next = (
        Window.partitionBy(*group_cols).orderBy(order_col).rowsBetween(0, Window.unboundedFollowing)
    )
    w_local = Window.partitionBy(*group_cols).orderBy(order_col).rowsBetween(-4, 4)

    x = (
        x.withColumn("_orig_null", f.col(value_col).isNull())
        .withColumn("_ts", f.col(order_col).cast("long"))
        .withColumn("_prev_val", f.last(value_col, ignorenulls=True).over(w_prev))
        .withColumn("_next_val", f.first(value_col, ignorenulls=True).over(w_next))
        .withColumn(
            "_prev_ts",
            f.last(f.when(f.col(value_col).isNotNull(), f.col("_ts")), ignorenulls=True).over(
                w_prev
            ),
        )
        .withColumn(
            "_next_ts",
            f.first(f.when(f.col(value_col).isNotNull(), f.col("_ts")), ignorenulls=True).over(
                w_next
            ),
        )
        .withColumn("_local_max", f.max(value_col).over(w_local))
        .withColumn("_local_min", f.min(value_col).over(w_local))
    )
    # linear interpolation
    interp = ((f.col("_ts") - f.col("_prev_ts")) / (f.col("_next_ts") - f.col("_prev_ts"))) * (
        f.col("_next_val") - f.col("_prev_val")
    ) + f.col("_prev_val")

    interp_or_null = f.when(
        f.col("_prev_val").isNotNull()
        & f.col("_next_val").isNotNull()
        & (f.col("_next_ts") > f.col("_prev_ts")),
        interp,
    ).otherwise(f.col(value_col))

    # too many statements in imputed_expr. chained f.when can easily stackup and are costly.
    # A better approach would be to create a fact dataframe, calculate some values,
    # and then f.broadcast join the 2 tables with some rules in join
    # e.g. df1.join(df2, how=inner, on=[f.col("val_calc") < f.lit(2)
    imputed_expr = when_chained(
        [
            (f.col(value_col).isNotNull(), f.col(value_col)),
            (~f.col("is_imputable"), f.col(value_col)),
            (f.col("agg_method") == "avg", interp_or_null),
            ((f.col("agg_method") == "sum") & (f.col("gap_len") <= 2), f.col("_prev_val")),
            ((f.col("agg_method") == "sum") & (f.col("gap_len") > 2), f.col("median_value")),
            (f.col("agg_method") == "max", f.col("_local_max")),
            (f.col("agg_method") == "min", f.col("_local_min")),
        ],
        otherwise=f.col(value_col),
    )

    df_imputed = (
        x.withColumn(value_col, imputed_expr)
        .withColumn(
            "imputed_flag",
            f.when(f.col("_orig_null") & f.col(value_col).isNotNull(), 1).otherwise(0),
        )
        .drop(
            "_orig_null",
            "_prev_val",
            "_next_val",
            "_ts",
            "_prev_ts",
            "_next_ts",
            "_local_max",
            "_local_min",
            "gap_id",
            "is_missing",
            "prev_missing",
            "gap_start_flag",
            "gap_len",
            "is_imputable",
            "agg_method",
            "min_value",
            "max_value",
            "median_value",
        )
    )

    return df_imputed


prepared_df = prepare_data_for_imputing(pm_df_long)

# hope it will help with calculations later
prepared_df = prepared_df.repartition("kpi_id", "bts_id", "distname").sortWithinPartitions(
    "kpi_id", "bts_id", "distname", "start_time"
)

imputed_df = impute(
    df=prepared_df,
    group_cols=["kpi_id", "bts_id", "distname"],
    order_col="start_time",
    value_col="kpi_value",
    max_imputable_gap=MAX_IMPUTABLE_GAP,
)
