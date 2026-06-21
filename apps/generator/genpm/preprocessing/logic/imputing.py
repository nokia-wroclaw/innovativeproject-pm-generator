from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from genpm.preprocessing.logic.kpi_grouping_defs import classify_kpis
from genpm.utils.consts import (
    AVG_KEYWORDS,
    MAX_KEYWORDS,
    MEAN_LIKE_KEYWORDS,
    MEAN_LIKE_UNITS,
    MIN_KEYWORDS,
    RATIO_KEYWORDS,
    VOLUME_KEYWORDS,
    VOLUME_UNITS,
)
from genpm.utils.utils import when_chained


def categorize_kpi_with_definitions(pm_df: DataFrame, kpi_defs_df: DataFrame) -> DataFrame:
    """Classify each KPI's aggregation method by joining stats and definitions, then attach agg_method to pm_df."""
    # calculate stats required for kpi agg character classification
    kpi_stats = pm_df.groupBy("kpi_id").agg(
        f.round(f.min("kpi_value"), 4).alias("kpi_min"),
        f.round(f.max("kpi_value"), 4).alias("kpi_max"),
        f.round(f.sum(f.col("kpi_value").isNull().cast("int")) / f.count("*") * 100, 2).alias(
            "null_pct"
        ),
    )
    # join to have table ready for kpi agg character classification
    pm_stats_with_definitions = kpi_stats.join(kpi_defs_df, on="kpi_id", how="left")

    kpi_classified = classify_kpis(
        pm_stats_with_definitions,
        MEAN_LIKE_UNITS,
        VOLUME_UNITS,
        MIN_KEYWORDS,
        MAX_KEYWORDS,
        AVG_KEYWORDS,
        RATIO_KEYWORDS,
        MEAN_LIKE_KEYWORDS,
        VOLUME_KEYWORDS,
    )

    pm_df = pm_df.join(kpi_classified.select("kpi_id", "agg_method"), on="kpi_id", how="left")

    return pm_df


def detect_gap_runs(
    pm_df: DataFrame,
    group_cols: list[str],
    order_col: str,
    value_col: str,
    max_imputable_gap: int = 6,
) -> DataFrame:
    """Tag each null row with its gap-run length and whether the gap is short enough to impute."""
    w = Window.partitionBy(*group_cols).orderBy(order_col)

    base = (
        pm_df.withColumn("is_missing", f.col(value_col).isNull().cast("int"))
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
    """Compute per-series min/max/median statistics for imputation fallback values."""
    df_stats = df.groupBy(*group_cols).agg(
        f.first(agg_method_col, ignorenulls=True).alias("agg_method"),
        f.min(value_col).alias("min_value"),
        f.max(value_col).alias("max_value"),
        f.expr(f"percentile_approx({value_col}, 0.5)").alias("median_value"),
    )
    return df_stats


def impute(
    pm_df: DataFrame,
    group_cols: list[str],
    order_col: str,
    value_col: str,
    agg_method_col: str,
    max_imputable_gap: int = 6,
) -> DataFrame:
    """Impute null kpi_value entries using agg-method-specific strategies (linear interp, forward fill, local extremes)."""
    tagged_df = detect_gap_runs(pm_df, group_cols, order_col, value_col, max_imputable_gap).drop(
        agg_method_col
    )
    stats = build_series_stats(pm_df, group_cols, value_col, agg_method_col)
    tagged_df_with_stats = tagged_df.join(f.broadcast(stats), on=group_cols, how="left")

    w_prev = (
        Window.partitionBy(*group_cols).orderBy(order_col).rowsBetween(Window.unboundedPreceding, 0)
    )
    w_next = (
        Window.partitionBy(*group_cols).orderBy(order_col).rowsBetween(0, Window.unboundedFollowing)
    )
    w_local = Window.partitionBy(*group_cols).orderBy(order_col).rowsBetween(-4, 4)

    tagged_df_with_stats = (
        tagged_df_with_stats.withColumn("_orig_null", f.col(value_col).isNull())
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
        tagged_df_with_stats.withColumn(value_col, imputed_expr)
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
