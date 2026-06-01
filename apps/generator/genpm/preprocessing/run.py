import argparse

from pyspark.sql import DataFrame

from genpm.preprocessing import changepoint_detection, imputing, kpi_coverage, preprocessing_logic
from genpm.utils.consts import MAX_IMPUTABLE_GAP, SHARED_DIR_PATH, SPARK_CONFIGS
from genpm.utils.logger import get_logger
from genpm.utils.utils import SparkDataManager

logger = get_logger()


EDA_DATA_PATH = SHARED_DIR_PATH / "eda_data"
PREPROCESSED_DATASET_PATH = SHARED_DIR_PATH / "preprocessed_dataset"


# Data download


def _load_data(
    sdm: SparkDataManager, args: argparse.Namespace
) -> tuple[DataFrame, DataFrame, DataFrame]:
    pm_df_long = sdm.read_parquet(args.pm_data_raw_path)
    kpis_definitions_df = sdm.read_parquet(args.kpi_definitions_raw_path)
    simple_reports_df = sdm.read_parquet(args.simple_reports_raw_path)

    return pm_df_long, kpis_definitions_df, simple_reports_df


def run(args: argparse.Namespace) -> None:
    sdm = SparkDataManager(SPARK_CONFIGS["HALF_SAFE"])

    pm_df_long_raw, kpi_definitions_df_raw, simple_reports_df_raw = _load_data(sdm, args)

    pm_df_long_raw = preprocessing_logic.raw_pm_preperation(pm_df_long_raw)

    if args.save_eda:
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
        pm_df_wide_raw = preprocessing_logic._pivot_pm_data(pm_df_long_raw)
        sdm.write_parquet(
            pm_df_wide_raw, EDA_DATA_PATH / "raw" / "raw_pm_data_wide", mode="overwrite"
        )

    # KPI version flattening
    pm_df_long, kpi_definitions = preprocessing_logic.coalesce_kpi_version(
        pm_df_long_raw, kpi_definitions_df_raw
    )

    pm_df_long, pm_df_const_kpi = preprocessing_logic.pop_constant_kpis(pm_df_long)

    # STAGE: COVERAGE ANALYSIS

    # Timestamp frequency uniformoty (1 hour)
    # Perc cell allignment to min max (window coverage handles the rest)
    pm_df_long = kpi_coverage.align_cell_time_ranges(pm_df_long)
    # TODO: Let user add those parameters
    selected_kpis, pm_df_long, pm_df_windows_assigned = kpi_coverage.pm_data_kpi_coverage(
        pm_df_long,
        # windowing + density
        window_hours=168,
        stride_hours=24,
        density_threshold=0.917,
        # max-gap filter
        max_gap_hours=12,
        # temporal stability
        min_weeks_with_good_windows=8,
        min_frac_weeks_covered=0.60,
        # variance
        min_cv=0.01,
        max_zero_frac=0.95,
        # cross-cell consistency
        max_iqr_ratio=5.0,
        # pre-filter
        min_window_coverage_frac=0.50,
        min_frac_contributing_cells=0.20,
        # greedy
        min_joint_coverage_frac=0.90,
        min_joint_windows_abs=10_000,
    )

    logger.info(f"KPI COVERAGE - KPIs SELECTED: {' '.join(selected_kpis)}")

    pm_df_long = imputing.categorize_kpi_with_definitions(pm_df_long, kpi_definitions)

    # pm_df_long = pm_df_long.repartition("kpi_id", "bts_id", "distname").sortWithinPartitions(
    #     "kpi_id", "bts_id", "distname", "start_time"
    # )

    pm_df_long = imputing.impute(
        pm_df=pm_df_long,
        group_cols=["kpi_id", "bts_id", "distname"],
        order_col="start_time",
        value_col="kpi_value",
        agg_method_col="agg_method",
        max_imputable_gap=MAX_IMPUTABLE_GAP,
    )

    # PELT changepoint detection
    pm_df_long_segmented = changepoint_detection.add_regime_ids(pm_df_long)

    # ADD AFTER KPI SEGMANTATION SCALING
    print(pm_df_long_segmented)
    # COMBINE SCALED SEGMENTS TO ONE KPIS AGAIN

    # TODO: SAVING FOR VISUALS DATAFRAMES and DATA OVERALL
    pm_df_long = pm_df_long.cache()
    pm_df_long.count()
    # Kpi wide format
    pm_df_wide = preprocessing_logic._pivot_pm_data(pm_df_long)

    simple_reports = preprocessing_logic.simple_reports_pivot(simple_reports_df_raw)

    # Save preprocessed data
    list_of_dfs = [
        pm_df_long,
        pm_df_wide,
        pm_df_const_kpi,
        kpi_definitions,
        simple_reports,
    ]

    preprocessed_data_filenames = [
        "pm_data_long",
        "pm_data_wide",
        "pm_data_const_kpi",
        "kpi_definitions",
        "simple_reports",
    ]

    # END OF PREPROCESSING

    # SAVE PREPROCESSED DATA

    # TODO: Integrate this with S3 and standardize with BaseModel the Paths?
    for df, df_path in zip(list_of_dfs, preprocessed_data_filenames, strict=True):
        sdm.write_parquet(
            df,
            # TODO: Add getting output path from args
            PREPROCESSED_DATASET_PATH / df_path,
            mode="overwrite",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-data-raw-path", required=False)
    parser.add_argument("--kpi-definitions-raw-path", required=False)
    parser.add_argument("--simple-reports-raw-path", required=False)
    parser.add_argument("--save-eda", action="store_true", default=False, help="Save data for EDA")
    args = parser.parse_args()
    run(args)
