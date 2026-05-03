from functools import reduce

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as f


from utils.utils import SparkDataManager, load_config
from utils.consts import SHARED_DIR_PATH

sdm = SparkDataManager()

PM_DATA_PATH = [SHARED_DIR_PATH / "raw_data" / f"pm_kpis_part{i}.parquet" for i in range(1, 6)]
KPI_DEFINITIONS_PATH = SHARED_DIR_PATH / "raw_data" / "kpis_definitions.parquet"
SIMPLE_REPORTS_PATH = SHARED_DIR_PATH / "raw_data" / "simple_reports.parquet"


# PREPROCESSING_CONFIG = load_config("/home/sparkuser/app/apps/apps/generator/preprocessing/preprocessing_config.yaml")

EDA_DATA_PATH = SHARED_DIR_PATH / "eda_data"


def load_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    list_of_pm_dfs = [sdm.read_parquet(dp) for dp in PM_DATA_PATH]

    pm_df_long: DataFrame = reduce(lambda df1, df2: df1.unionByName(df2), list_of_pm_dfs)

    kpis_definitions_df = sdm.read_parquet(KPI_DEFINITIONS_PATH)
    simple_reports_df = sdm.read_parquet(SIMPLE_REPORTS_PATH)

    return pm_df_long, kpis_definitions_df, simple_reports_df


def prepare_pm_data(pm_df_long: DataFrame):

    # Simple df screening
    pm_df_long.printSchema()

    print(f"{pm_df_long.count()=}")

    pm_df_long.describe()

    # pm_df entry cleaning
    pm_df_long = pm_df_long.withColumnsRenamed({"bts_anon": "bts_id", "distname_anon": "distname"})
    pm_df_long = pm_df_long.dropDuplicates()
    pm_df_long = pm_df_long.dropna(subset=("start_time", "bts_id", "distname"))

    # Timestamp frequency uniformoty (1 hour) and KPI-bts recording range verification
    pm_df_long = fill_missing_timestamps(pm_df_long, "start_time", "bts_id")

    # KPI version flattening

    # at last pivot
    pm_df_wide = _pivot_pm_data(pm_df_long)


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

        if i % 5 == 0 and i != 0:
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


def _clean_prepare_pm_data_wide(pm_df_wide: DataFrame) -> DataFrame:
    pass


def prepare_kpi_definitions(kpi_definitions_df: DataFrame) -> DataFrame:
    # TODO: Filtering only needed kpi definitions

    return kpi_definitions_df


def prepare_simple_reports(simple_reports_df: DataFrame) -> DataFrame:
    # TODO: Add simple reports filtering and report pivot?

    return simple_reports_df


def main():
    print("RAW DATA PROCESSING")
    print("LOADING DATA")
    pm_df_long_raw, kpi_definitions_df_raw, simple_reports_df_raw = load_data()

    # pm_df entry cleaning
    pm_df_long_raw = pm_df_long_raw.withColumnsRenamed(
        {"bts_anon": "bts_id", "distname_anon": "distname"}
    )
    pm_df_long_raw = pm_df_long_raw.dropDuplicates()
    pm_df_long_raw = pm_df_long_raw.dropna(subset=("start_time", "bts_id", "distname"))

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
    sdm.write_parquet(pm_df_wide_raw, EDA_DATA_PATH / "raw" / "raw_pm_data_wide", mode="overwrite")


if __name__ == "__main__":
    main()
