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


INTERMEDIATE_LOAD = False


def run(args: argparse.Namespace) -> None:
    sdm = SparkDataManager(SPARK_CONFIGS["FULL_RESOURCES"])

    # UNCOMMENT LATER FOR FULL PREPROP
    # pm_df_long_raw, kpi_definitions_df_raw, simple_reports_df_raw = _load_data(sdm, args)

    # pm_df_long_raw = preprocessing_logic.raw_pm_preperation(pm_df_long_raw)

    # # KPI version flattening
    # pm_df_long, kpi_definitions = preprocessing_logic.coalesce_kpi_version(
    #     pm_df_long_raw, kpi_definitions_df_raw
    # )

    # pm_df_long, pm_df_const_kpi = preprocessing_logic.pop_constant_kpis(pm_df_long)

    # sdm.write_parquet(pm_df_long, PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long", mode="overwrite")
    # sdm.write_parquet(kpi_definitions, PREPROCESSED_DATASET_PATH / "intermediate" / "kpi_definitions", mode="overwrite")

    pm_df_long = sdm.read_parquet(PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long")
    kpi_definitions = sdm.read_parquet(
        PREPROCESSED_DATASET_PATH / "intermediate" / "kpi_definitions"
    )
    # STAGE: COVERAGE ANALYSIS
    # Timestamp frequency uniformoty (1 hour)
    # Perc cell allignment to min max (window coverage handles the rest)
    print("INTERMIEDATE 1")
    pm_df_long_filled_gaps = kpi_coverage.fill_internal_gaps(pm_df_long, "start_time")
    pm_df_long_filled_gaps.cache()
    pm_df_long_filled_gaps.count()
    logger.info("COVERAGE - gap-filled rows computed")

    # GLOBAL COVERAGE FILTER
    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps.join(
        kpi_coverage.filter_global_value_density(pm_df_long_filled_gaps, min_global_density=0.5),
        on="kpi_id",
        how="inner",
    )

    # GAP PATTERNS FILTERING (consistant NULL patterns detection throuh median)
    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.join(
        kpi_coverage.filter_gap_pattern(
            pm_df_long_filled_gaps_filtered,
            max_imputable_gap=6,
            min_imputable_gap_frac=0.60,
        ),
        on="kpi_id",
        how="inner",
    )

    # Left with around 400 kpis

    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.cache()
    print(pm_df_long_filled_gaps_filtered.count())

    # FILTERING READY FOR IMPUTING

    pm_df_long_outliers_dropped = preprocessing_logic.iqr_kpi_outlier_detection(
        pm_df_long_filled_gaps_filtered, k=1.5
    )

    # UP TO THIS MOMENT WORKS

    pm_df_long_pre_impute = imputing.categorize_kpi_with_definitions(
        pm_df_long_outliers_dropped, kpi_definitions
    )

    pm_df_long_pre_impute = pm_df_long_pre_impute.repartition(
        "kpi_id", "bts_id", "distname"
    ).sortWithinPartitions("kpi_id", "bts_id", "distname", "start_time")

    pm_df_long_imputed = imputing.impute(
        pm_df=pm_df_long_pre_impute,
        group_cols=["kpi_id", "bts_id", "distname"],
        order_col="start_time",
        value_col="kpi_value",
        agg_method_col="agg_method",
        max_imputable_gap=MAX_IMPUTABLE_GAP,
    )

    # NOTE: There is no way, to allign all kpis to the min of cell
    # The crossjoin for this will be added at the end of preprocessing, when all goes through

    pm_training_window_valids = kpi_coverage.compute_window_density_sparse(pm_df_long_filled_gaps)

    print("READ")

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

    # TODO: Integrate this with S3
    for df, df_path in zip(list_of_dfs, preprocessed_data_filenames, strict=True):
        sdm.write_parquet(
            df,
            # TODO: Add getting output path from args
            PREPROCESSED_DATASET_PATH / df_path,
            mode="overwrite",
        )


# and standardize with BaseModel the Paths?

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-data-raw-path", required=False)
    parser.add_argument("--kpi-definitions-raw-path", required=False)
    parser.add_argument("--simple-reports-raw-path", required=False)
    parser.add_argument("--save-eda", action="store_true", default=False, help="Save data for EDA")
    args = parser.parse_args()
    run(args)
