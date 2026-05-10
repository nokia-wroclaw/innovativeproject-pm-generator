from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from utils.consts import SHARED_DIR_PATH
from utils.utils import SparkDataManager

# custom config for test
spark_config = {
    "spark.log.level": "ERROR",
    # Driver
    "spark.driver.memory": "20g",
    "spark.driver.maxResultSize": "8g",
    # Executor
    "spark.executor.memory": "24g",
    "spark.executor.memoryOverhead": "6g",
    # Memory fractions
    "spark.memory.fraction": "0.8",
    "spark.memory.storageFraction": "0.2",
    # Shuffle
    "spark.sql.shuffle.partitions": "800",
}

sdm = SparkDataManager(spark_config)

PM_DATA_PATH = [SHARED_DIR_PATH / "raw_data" / f"pm_kpis_part{i}.parquet" for i in range(1, 6)]
KPI_DEFINITIONS_PATH = SHARED_DIR_PATH / "raw_data" / "kpis_definitions.parquet"
SIMPLE_REPORTS_PATH = SHARED_DIR_PATH / "raw_data" / "simple_reports.parquet"


# PREPROCESSING_CONFIG = load_config(
# "/home/sparkuser/app/apps/apps/generator/preprocessing/preprocessing_config.yaml")

EDA_DATA_PATH = SHARED_DIR_PATH / "eda_data"


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


def _coalesce_kpi_version(pm_df_long: DataFrame, kpi_definitions: DataFrame):
    def __kpi_definition_comparison(kpi_definitions: DataFrame) -> DataFrame:
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
        ).agg(f.collect_list(f.col("kpi_id")).alias("kpi_id_list"))

        return kpi_definitions_change_result

    # TODO: ANALYZE HOW KPI STATISTICS CHANGE, WHEN VERSION CHANGES

    # def __statistical_pm_version_comparison(pm_df_filtered: DataFrame):

    #     # calculate signinficant statistics
    #     significance_stats = {
    #         "mean_kpi_value": f.mean,
    #         "std_kpi_value": f.std
    #         # Add other statistics
    #     }

    #     pm_df_stats = pm_df_filtered.groupBy("bts_id", "base_kpi", "kpi_id").agg(
    #         *[
    #             func(f.col("kpi_value")).alias(key)
    #             for key, func in significance_stats.items()
    #         ]
    #     )

    #     # TODO: Verify if there are any KPI' versions that are reverted!!!
    #     # If there are any, orderby on kpi_id is WRONG (for some kpis)
    #     w = Window.partitionBy("bts_id", "base_kpi").orderBy("kpi_id")

    #     pm_df_stats_with_lag = pm_df_stats.withColumns(
    #         {f"prev_{v}": f.lag(v).over(w) for v in significance_stats.keys()}
    #     )

    print("PREPROCESSING: KPI VERSION COALESCE")
    # get only relevent kpis
    distinct_kpis = pm_df_long.select("kpi_id").distinct()
    kpi_definitions = kpi_definitions.join(distinct_kpis, on="kpi_id", how="inner")

    BASE_KPI_REGEX = r"^(.*?)[a-z]?$"

    kpi_definitions = kpi_definitions.withColumn(
        "base_kpi", f.regexp_extract("kpi_id", BASE_KPI_REGEX, 1)
    )

    kpi_changed_definition_mapping = __kpi_definition_comparison(kpi_definitions)

    # TODO: add other filters for kpi version mapping (maybe statistical?)

    # take only applicable kpis
    mask = (
        f.col("kpi_definition_changed") == f.lit(False)
        # Add other filters here
    )

    kpi_changed_definition_mapping = kpi_changed_definition_mapping.where(mask).select(
        "base_kpi", f.posexplode(f.sort_array("kpi_id_list")).alias("version_rank", "kpi_id")
    )

    pm_data_joined_mapping = pm_df_long.join(
        kpi_changed_definition_mapping, on="kpi_id", how="left"
    )

    kpis_for_coalesce_mask = f.col("base_kpi").isNotNull()

    pm_data_for_kpi_merge = pm_data_joined_mapping.filter(kpis_for_coalesce_mask)
    pm_data_rest = pm_data_joined_mapping.filter(~kpis_for_coalesce_mask).drop(
        "base_kpi", "version_rank"
    )

    # 1. Find the newest version_rank that has a non-null value per group
    best_rank = (
        pm_data_for_kpi_merge.filter(f.col("kpi_value").isNotNull())
        .groupBy("bts_id", "distname", "start_time", "base_kpi")
        .agg(f.max("version_rank").alias("best_rank"))
    )

    # 2. For groups where ALL values are null, fall back to the newest version (max rank regardless)
    all_ranks = pm_data_for_kpi_merge.groupBy("bts_id", "distname", "start_time", "base_kpi").agg(
        f.max("version_rank").alias("max_rank")
    )

    best_rank = (
        all_ranks.join(best_rank, on=["bts_id", "distname", "start_time", "base_kpi"], how="left")
        .withColumn("best_rank", f.coalesce(f.col("best_rank"), f.col("max_rank")))
        .drop("max_rank")
    )

    # 3. Join back and filter to only the best row per group
    pm_data_kpis_merged = (
        pm_data_for_kpi_merge.join(
            f.broadcast(best_rank), on=["bts_id", "distname", "start_time", "base_kpi"]
        )
        .filter(f.col("version_rank") == f.col("best_rank"))
        .drop("version_rank", "best_rank", "kpi_id")
        .withColumnRenamed("base_kpi", "kpi_id")
    )

    # union back
    df_result = pm_data_rest.unionByName(pm_data_kpis_merged)

    return df_result


# Data download


def load_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    list_of_pm_dfs = [sdm.read_parquet(dp) for dp in PM_DATA_PATH]

    pm_df_long: DataFrame = reduce(lambda df1, df2: df1.unionByName(df2), list_of_pm_dfs)

    kpis_definitions_df = sdm.read_parquet(KPI_DEFINITIONS_PATH)
    simple_reports_df = sdm.read_parquet(SIMPLE_REPORTS_PATH)

    return pm_df_long, kpis_definitions_df, simple_reports_df


# PM DATA


def prepare_pm_data(pm_df_long: DataFrame, kpi_definitions: DataFrame):
    print("PREPROCESSING PM DATA")
    # Simple df screening
    pm_df_long.printSchema()

    print(f"{pm_df_long.count()=}")

    # # pm_df entry cleaning
    # pm_df_long = pm_df_long.withColumnsRenamed({"bts_anon": "bts_id", "distname_anon": "distname"}
    # )
    # pm_df_long = pm_df_long.dropDuplicates()
    # pm_df_long = pm_df_long.dropna(subset=("start_time", "bts_id", "distname"))

    # Timestamp frequency uniformoty (1 hour) and KPI-bts recording range verification
    pm_df_long = fill_missing_timestamps(pm_df_long, "start_time", "bts_id")

    # KPI version flattening
    pm_df_long = _coalesce_kpi_version(pm_df_long, kpi_definitions)

    sdm.write_parquet(pm_df_long, EDA_DATA_PATH / "debug_coalesced_kpi")
    print("COALESCE SUCCESS")
    # at last pivot
    pm_df_wide = _pivot_pm_data(pm_df_long)

    return pm_df_long, pm_df_wide


def prepare_kpi_definitions(kpi_definitions_df: DataFrame) -> DataFrame:
    # TODO: Filtering only needed kpi definitions

    return kpi_definitions_df


def prepare_simple_reports(simple_reports_df: DataFrame) -> DataFrame:
    # TODO: Add simple reports filtering and report pivot?

    return simple_reports_df


def main():
    print("RAW DATA PROCESSING")
    print("LOADING DATA")

    SAVE_RAW_DATA = False

    pm_df_long_raw, kpi_definitions_df_raw, simple_reports_df_raw = load_data()

    # pm_df entry cleaning
    pm_df_long_raw = pm_df_long_raw.withColumnsRenamed(
        {"bts_anon": "bts_id", "distname_anon": "distname"}
    )
    pm_df_long_raw = pm_df_long_raw.dropDuplicates()
    pm_df_long_raw = pm_df_long_raw.dropna(subset=("start_time", "bts_id", "distname"))

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

    pm_df_long_preprocessed, pm_df_wide_preprocessed = prepare_pm_data(
        pm_df_long_raw, kpi_definitions_df_raw
    )

    pm_df_long_preprocessed.count()

    pm_df_long_preprocessed.show()

    print("DONE")


if __name__ == "__main__":
    main()
