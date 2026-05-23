from functools import reduce

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from genpm.utils.consts import SHARED_DIR_PATH
from genpm.utils.logger import get_logger

logger = get_logger()

EDA_DATA_PATH = SHARED_DIR_PATH / "eda_data"
PREPROCESSED_DATASET_PATH = SHARED_DIR_PATH / "preprocessed_dataset"

# HELPER FUNCTIONS


def _pivot_pm_data(pm_df_long: DataFrame) -> DataFrame:
    # PIVOT ATTEMPT
    def __chunk(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    # Batch size - according to how much RAM is available
    BATCH_SIZE = 30

    kpis = [r.kpi_id for r in pm_df_long.select("kpi_id").distinct().collect()]
    batches = list(__chunk(kpis, BATCH_SIZE))
    logger.info(f"{len(batches)=}")

    pm_df_long = pm_df_long.repartition("kpi_id").persist()

    # count for activating evaluation
    pm_df_long.count()

    pm_df_wide = None

    logger.info("PM DATA PIVOTTING")

    for i, batch in enumerate(batches):
        logger.info(f"\tBatch {i}")

        df_batch = pm_df_long.filter(f.col("kpi_id").isin(batch))

        df_pivot = (
            df_batch.groupBy("bts_id", "distname", "start_time")
            .pivot("kpi_id")
            .agg(f.first("kpi_value"))
        )

        if i % 5 == 0 and i != 0 and pm_df_wide is not None:
            pm_df_wide = pm_df_wide.checkpoint()

        if pm_df_wide is None:
            pm_df_wide = df_pivot
        else:
            pm_df_wide = pm_df_wide.join(df_pivot, ["bts_id", "distname", "start_time"], "outer")

    logger.info("PM DATA PIVOTTING COMPLETED")

    return pm_df_wide  # type: ignore


def fill_missing_timestamps(
    df: DataFrame,
    time_col: str,
    group_cols: list[str],
) -> DataFrame:
    """
    Fill missing hourly timestamps per station using each station's
    own min/max time range. Operates in long format — safe for large data.
    """
    logger.info("PREPROCESSING: FILLING MISSING TIMESTAMPS")
    # Per-group time bounds — small aggregation, stays distributed
    station_bounds = df.groupBy(*group_cols).agg(
        f.min(time_col).alias("min_t"), f.max(time_col).alias("max_t")
    )

    # Generate hourly spine per group using sequence + explode
    station_spines = station_bounds.withColumn(
        time_col, f.explode(f.sequence(f.col("min_t"), f.col("max_t"), f.expr("INTERVAL 1 HOUR")))
    ).drop("min_t", "max_t")

    # Left join original data — only fills gaps, no cross-group explosion
    return station_spines.join(df, on=[*group_cols, time_col], how="left")


def coalesce_kpi_version(
    pm_df_long: DataFrame,
    kpi_definitions: DataFrame,
    chunk_size=10,
    overwrite_null_with_older_version_value=False,
    include_kpi_origin=False,
) -> tuple[DataFrame, DataFrame]:
    def _kpi_definition_comparison(kpi_definitions: DataFrame) -> DataFrame:
        KPI_DEFINITION_VARIABLES = [
            col for col in kpi_definitions.columns if col not in ["kpi_id", "base_kpi"]
        ]
        # TODO: Verify if there are any KPI' versions that are reverted!!!
        w = Window.partitionBy("base_kpi").orderBy("kpi_id")

        kpi_definitions_with_lag = kpi_definitions.withColumns(
            {f"prev_{v}": f.lag(v).over(w) for v in KPI_DEFINITION_VARIABLES}
        )

        # get kpi_changed expr
        change_expr = None

        for c in KPI_DEFINITION_VARIABLES:
            expr = f.col(f"prev_{c}").isNotNull() & (f.col(c) != f.col(f"prev_{c}"))

            change_expr = expr if change_expr is None else (change_expr | expr)

        # check if there were any changes per version
        kpi_definitions_signed = kpi_definitions_with_lag.withColumn(
            "kpi_definition_changed",
            f.when(change_expr, f.lit(True)).otherwise(f.lit(False)),  # type: ignore
        )

        kpi_definitions_change_result = kpi_definitions_signed.groupBy(
            "base_kpi", "kpi_definition_changed"
        ).agg(f.sort_array(f.collect_list(f.col("kpi_id"))).alias("kpi_id_list"))

        return kpi_definitions_change_result

    # TODO: ANALYZE HOW KPI STATISTICS CHANGE, WHEN VERSION CHANGES

    logger.info("PREPROCESSING - KPI VERSION COALESCE")
    # get only relevent kpis
    distinct_kpis = pm_df_long.select("kpi_id").distinct()
    kpi_definitions = kpi_definitions.join(f.broadcast(distinct_kpis), on="kpi_id", how="inner")

    BASE_KPI_REGEX = r"^(.*?)[a-z]?$"

    kpi_definitions = kpi_definitions.withColumn(
        "base_kpi", f.regexp_extract("kpi_id", BASE_KPI_REGEX, 1)
    )

    kpi_changed_definition_mapping = _kpi_definition_comparison(kpi_definitions)

    # TODO: add other filters for kpi version mapping (maybe statistical?)

    # take only applicable kpis
    appliccable_kpi_mask = (
        f.col("kpi_definition_changed") == f.lit(False)
        # Add other filters here
    )

    kpi_changed_definition_mapping_posexpl = kpi_changed_definition_mapping.where(
        appliccable_kpi_mask
    ).select("base_kpi", f.posexplode("kpi_id_list").alias("version_rank", "kpi_id"))

    pm_data_joined_mapping = pm_df_long.join(
        kpi_changed_definition_mapping_posexpl, on="kpi_id", how="left"
    )

    kpis_for_coalesce_mask = f.col("base_kpi").isNotNull()

    pm_data_for_kpi_merge = pm_data_joined_mapping.filter(kpis_for_coalesce_mask)
    pm_data_rest = pm_data_joined_mapping.filter(~kpis_for_coalesce_mask).drop(
        "base_kpi", "version_rank"
    )

    pm_data_for_kpi_merge = (
        pm_data_for_kpi_merge.repartition(512, "base_kpi")
        .sortWithinPartitions("base_kpi")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    # Collect for persist
    pm_data_for_kpi_merge.count()

    base_kpis = [
        r.base_kpi for r in kpi_changed_definition_mapping.select("base_kpi").distinct().collect()
    ]
    chunks = [base_kpis[i : i + chunk_size] for i in range(0, len(base_kpis), chunk_size)]

    result_frames = []

    for chunk in chunks:
        logger.info(f"CHUNK CONTAINS: \n{', '.join(chunk)}")
        chunk_df = pm_data_for_kpi_merge.filter(f.col("base_kpi").isin(chunk))

        # best non-null version_rank per group
        best_rows = chunk_df.groupBy("bts_id", "distname", "start_time", "base_kpi").agg(
            f.max(
                f.struct(
                    # prioritize non-null KPI
                    f.when(f.col("kpi_value").isNotNull(), f.lit(1))
                    .otherwise(f.lit(0))
                    .alias("null_overwrite")
                    if overwrite_null_with_older_version_value
                    else f.lit(0).alias("null_overwrite"),
                    # prioritize newest version
                    f.col("version_rank"),
                    # keep entire row
                    *[f.col(c) for c in chunk_df.columns],
                )
            ).alias("best")
        )

        coalesced = best_rows.select("best.*").drop("version_rank", "null_overwrite")

        if include_kpi_origin:
            coalesced = coalesced.withColumnRenamed("kpi_id", "kpi_origin")
        else:
            coalesced = coalesced.drop("kpi_id")

        coalesced = coalesced.withColumnRenamed("base_kpi", "kpi_id")

        result_frames.append(coalesced)

    if include_kpi_origin:
        pm_data_rest = pm_data_rest.withColumn("kpi_origin", f.col("kpi_id"))

    # ── 5. Union all chunks + rest
    df_result: DataFrame = reduce(lambda a, b: a.unionByName(b), result_frames + [pm_data_rest])
    df_result = df_result.coalesce(512)
    # KPI definitions transform
    kpi_changed_definition_mapping = (
        kpi_changed_definition_mapping.withColumn(
            "tmp_kpi_id_arr",
            f.when(
                appliccable_kpi_mask,
                # Take only newest kpi definition
                f.array(f.element_at(f.col("kpi_id_list"), -1)),
            ).otherwise(f.col("kpi_id_list")),
        )
        .withColumn("kpi_id", f.explode("tmp_kpi_id_arr"))
        .drop("base_kpi", "tmp_kpi_id_arr", "kpi_id_list")
    )

    kpi_definitions_final = kpi_definitions.join(
        kpi_changed_definition_mapping, on="kpi_id", how="inner"
    ).withColumn(
        "kpi_id",
        # overwrite kpi_id with base_kpi if appliccable
        f.when(appliccable_kpi_mask, f.col("base_kpi")).otherwise(f.col("kpi_id")),
    )

    return df_result, kpi_definitions_final


# pivot simple reports
def simple_reports_pivot(simple_reports: DataFrame):
    grouping_cols = ("datetime", "bts_id", "distname")
    simple_reports_pivot = (
        simple_reports.groupBy(*grouping_cols).pivot("report_name").agg(f.first("report_result"))
    )

    return simple_reports_pivot


# raw_pm
def raw_pm_preperation(pm_df_long: DataFrame) -> DataFrame:
    pm_df_long = pm_df_long.dropDuplicates()
    pm_df_long = pm_df_long.dropna(subset=("start_time", "bts_id", "distname"))
    return pm_df_long


# TODO: change this function, so it sees low coverage in time, not overall
# Low coverage can misidentified by kpis with a period of no data, and a full coverage period
# Add trimming of low coverage periods to certain kpis / add flags/masks to mark long NO_DATA
# periods (for trimming)


# For now, naive trimming is acceptable
def drop_low_coverage(
    pm_df: DataFrame,
    cell_threshold: float = 0.5,
    kpi_threshold: float = 0.5,
) -> DataFrame:
    """
    Independently compute coverage at cell and KPI level, report bad ones,
    then drop both.

    Coverage definition:
      - cell : non_null(kpi_value) / total rows  per (kpi_id, bts_id, distname)
      - kpi  : non_null(kpi_value) / total rows  per kpi_id  (across ALL cells)

    Dropping is additive — a row is removed if its cell OR its KPI is below
    the respective threshold.
    """

    # ── Cell-level coverage ──────────────────────────────────────────────────
    cell_stats = (
        pm_df.groupBy("kpi_id", "bts_id", "distname")
        .agg(
            f.count("*").alias("total"),
            f.count("kpi_value").alias("non_null"),
        )
        .withColumn("coverage", f.col("non_null") / f.col("total"))
    )

    good_cells = cell_stats.filter(f.col("coverage") >= cell_threshold)
    bad_cells = cell_stats.filter(f.col("coverage") < cell_threshold)

    n_cells = cell_stats.count()
    n_bad_cells = bad_cells.count()
    logger.info(
        f"[coverage] Cells  — dropped: {n_bad_cells:>6} / {n_cells}  "
        f"(threshold={cell_threshold:.0%})"
    )
    # logger.info("[coverage] Worst offending cells:")
    # (
    #     bad_cells.orderBy("coverage")
    #     .select("kpi_id", "bts_id", "distname", "coverage")
    #     .show(10, truncate=False)
    # )

    # ── KPI-level coverage ───────────────────────────────────────────────────
    kpi_stats = (
        pm_df.groupBy("kpi_id")
        .agg(
            f.count("*").alias("total"),
            f.count("kpi_value").alias("non_null"),
        )
        .withColumn("coverage", f.col("non_null") / f.col("total"))
    )

    good_kpis = kpi_stats.filter(f.col("coverage") >= kpi_threshold)
    bad_kpis = kpi_stats.filter(f.col("coverage") < kpi_threshold)

    n_kpis = kpi_stats.count()
    n_bad_kpis = bad_kpis.count()
    logger.info(
        f"[coverage] KPIs   — dropped: {n_bad_kpis:>6} / {n_kpis}  (threshold={kpi_threshold:.0%})"
    )
    # logger.info("[coverage] Dropped KPIs:")
    # (
    #     bad_kpis.orderBy("coverage")
    #     .select("kpi_id", "coverage", "total", "non_null")
    #     .show(20, truncate=False)
    # )

    # ── Drop: remove bad cells, then remove bad KPIs ─────────────────────────
    df_clean = pm_df.join(
        good_cells.select("kpi_id", "bts_id", "distname"),
        on=["kpi_id", "bts_id", "distname"],
        how="inner",
    ).join(good_kpis.select("kpi_id"), on="kpi_id", how="inner")

    return df_clean


def pop_constant_kpis(pm_df_long: DataFrame) -> tuple[DataFrame, DataFrame]:
    # WARNING: those are only per kpi aggregation
    # TODO: analyze, if it should be per distname-kpi_id
    constant_kpis = (
        pm_df_long.groupBy("kpi_id")
        .agg(
            # check what kpis are contant
            f.count_distinct("kpi_value").alias("kpi_value_distinct_count"),
            f.first("kpi_value").alias("kpi_value"),
        )
        .where(f.col("kpi_value_distinct_count") == 1)
    )

    # logger.info(f"CONSTANT KPIs FOUND: {constant_kpis.count()}")
    # drop those kpis from data
    pm_df_long_no_constant_kpis = pm_df_long.join(constant_kpis, on="kpi_id", how="left_anti")

    return pm_df_long_no_constant_kpis, constant_kpis
