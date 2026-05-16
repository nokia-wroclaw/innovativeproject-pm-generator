from functools import reduce

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from utils.consts import SHARED_DIR_PATH, SPARK_CONFIGS
from utils.logger import get_logger
from utils.utils import SparkDataManager

logger = get_logger()
sdm = SparkDataManager(SPARK_CONFIGS["FULL_HEAVY"])


# PREPROCESSING_CONFIG = load_config(
# "/home/sparkuser/app/apps/apps/generator/preprocessing/preprocessing_config.yaml")

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
    print(f"{len(batches)=}")

    pm_df_long = pm_df_long.repartition("kpi_id").persist()

    # count for activating evaluation
    pm_df_long.count()

    pm_df_wide = None

    print("PM DATA PIVOTTING")

    for i, batch in enumerate(batches):
        print(f"\tBatch {i}")

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

    print("PM DATA PIVOTTING COMPLETED")

    return pm_df_wide  # type: ignore


def fill_missing_timestamps(
    df: DataFrame,
    time_col: str,
    station_col: str,
) -> DataFrame:
    """
    Fill missing hourly timestamps per station using each station's
    own min/max time range. Operates in long format — safe for large data.
    """
    print("PREPROCESSING: FILLING MISSING TIMESTAMPS")
    # Per-station time bounds — small aggregation, stays distributed
    station_bounds = df.groupBy(station_col).agg(
        f.min(time_col).alias("min_t"), f.max(time_col).alias("max_t")
    )

    # Generate hourly spine per station using sequence + explode
    station_spines = station_bounds.withColumn(
        time_col, f.explode(f.sequence(f.col("min_t"), f.col("max_t"), f.expr("INTERVAL 1 HOUR")))
    ).drop("min_t", "max_t")

    # Left join original data — only fills gaps, no cross-station explosion
    return station_spines.join(df, on=[station_col, time_col], how="left")


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

    # TODO: Add empirical KPI exclusion for statistical differences

    print("PREPROCESSING: KPI VERSION COALESCE")
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
        print(f"CHUNK CONTAINS: \n{', '.join(chunk)}")
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


# Data download


def load_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    PM_DATA_PATH = [SHARED_DIR_PATH / "raw_data" / f"pm_kpis_part{i}.parquet" for i in range(1, 6)]
    KPI_DEFINITIONS_PATH = SHARED_DIR_PATH / "raw_data" / "kpis_definitions.parquet"
    SIMPLE_REPORTS_PATH = SHARED_DIR_PATH / "raw_data" / "simple_reports.parquet"

    list_of_pm_dfs = [sdm.read_parquet(dp) for dp in PM_DATA_PATH]

    pm_df_long: DataFrame = reduce(lambda df1, df2: df1.unionByName(df2), list_of_pm_dfs)

    kpis_definitions_df = sdm.read_parquet(KPI_DEFINITIONS_PATH)
    simple_reports_df = sdm.read_parquet(SIMPLE_REPORTS_PATH)

    return pm_df_long, kpis_definitions_df, simple_reports_df


# pivot simple reports
def simple_reports_pivot(simple_reports: DataFrame):
    grouping_cols = ("datetime", "bts_id", "distname")
    simple_reports_pivot = (
        simple_reports.groupBy(*grouping_cols).pivot("report_name").agg(f.first("report_result"))
    )

    return simple_reports_pivot


# raw_pm
def raw_pm_preperation(pm_df_long: DataFrame) -> DataFrame:
    pm_df_long = pm_df_long.withColumnsRenamed({"bts_anon": "bts_id", "distname_anon": "distname"})
    pm_df_long = pm_df_long.dropDuplicates()
    pm_df_long = pm_df_long.dropna(subset=("start_time", "bts_id", "distname"))
    return pm_df_long


def drop_low_coverage_kpis(pm_df_long: DataFrame, minimal_coverage: float) -> DataFrame:
    # First drop low coverage cells

    # TODO: Still implement this
    pass


def save_preprocessed_data(
    pm_df_long: DataFrame,
    pm_df_wide: DataFrame,
    kpi_definitions: DataFrame,
    simple_reports: DataFrame,
):
    sdm.write_parquet(
        pm_df_long,
        PREPROCESSED_DATASET_PATH / "pm_data_long",
        mode="overwrite",
        partitionBy=["kpi_id", "bts_id"],
    )

    sdm.write_parquet(
        pm_df_wide,
        PREPROCESSED_DATASET_PATH / "pm_data_wide",
        mode="overwrite",
        partitionBy="bts_id",
    )

    sdm.write_parquet(
        kpi_definitions,
        PREPROCESSED_DATASET_PATH / "kpi_definitions",
        mode="overwrite",
        partitionBy="kpi_id",
    )

    sdm.write_parquet(
        simple_reports,
        PREPROCESSED_DATASET_PATH / "simple_reports",
        mode="overwrite",
        partitionBy="bts_id",
    )


def main():
    print("RAW DATA PROCESSING")
    print("LOADING DATA")

    SAVE_RAW_DATA = False

    pm_df_long_raw, kpi_definitions_df_raw, simple_reports_df_raw = load_data()

    pm_df_long_raw = raw_pm_preperation(pm_df_long_raw)

    if SAVE_RAW_DATA:
        sdm.write_parquet(
            pm_df_long_raw,
            EDA_DATA_PATH / "raw" / "raw_pm_data",
            mode="overwrite",
            partitionBy="kpi_id",
        )
        sdm.write_parquet(
            kpi_definitions_df_raw, EDA_DATA_PATH / "raw" / "raw_kpi_definitions", mode="overwrite"
        )
        sdm.write_parquet(
            simple_reports_df_raw, EDA_DATA_PATH / "raw" / "raw_simple_reports", mode="overwrite"
        )

        # pivot raw data - EDA
        pm_df_wide_raw = _pivot_pm_data(pm_df_long_raw)
        sdm.write_parquet(
            pm_df_wide_raw, EDA_DATA_PATH / "raw" / "raw_pm_data_wide", mode="overwrite"
        )

    # KPI version flattening
    pm_df_long, kpi_definitions = coalesce_kpi_version(pm_df_long_raw, kpi_definitions_df_raw)

    # Timestamp frequency uniformoty (1 hour) and KPI-bts recording range verification
    pm_df_long = fill_missing_timestamps(pm_df_long, "start_time", "bts_id")

    pm_df_long = pm_df_long.cache()
    # Kpi wide format
    pm_df_wide = _pivot_pm_data(pm_df_long)

    simple_reports = simple_reports_df_raw

    save_preprocessed_data(
        pm_df_long,
        pm_df_wide,
        kpi_definitions,
        simple_reports,
    )


if __name__ == "__main__":
    main()
