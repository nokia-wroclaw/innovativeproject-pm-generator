import argparse

from pyspark.sql import DataFrame

from genpm.preprocessing import kpi_coverage, preprocessing_logic
from genpm.utils.consts import SHARED_DIR_PATH, SPARK_CONFIGS
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

    # pm_df_long = sdm.read_parquet(PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long")
    kpi_definitions = sdm.read_parquet(
        PREPROCESSED_DATASET_PATH / "intermediate" / "kpi_definitions"
    )
    # STAGE: COVERAGE ANALYSIS
    # Timestamp frequency uniformoty (1 hour)
    # Perc cell allignment to min max (window coverage handles the rest)
    # print("INTERMIEDATE 1")
    # pm_df_long_filled_gaps = kpi_coverage.fill_internal_gaps(pm_df_long, "start_time")
    # pm_df_long_filled_gaps.cache()
    # pm_df_long_filled_gaps.count()
    # logger.info("COVERAGE - gap-filled rows computed")

    # # GLOBAL COVERAGE FILTER
    # pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps.join(
    #     kpi_coverage.filter_global_value_density(pm_df_long_filled_gaps, min_global_density=0.5),
    #     on="kpi_id",
    #     how="inner",
    # )

    # # GAP PATTERNS FILTERING (consistant NULL patterns detection throuh median)
    # pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.join(
    #     kpi_coverage.filter_gap_pattern(
    #         pm_df_long_filled_gaps_filtered,
    #         max_imputable_gap=6,
    #         min_imputable_gap_frac=0.60,
    #     ),
    #     on="kpi_id",
    #     how="inner",
    # )

    # # Left with around 400 kpis

    # pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.cache()
    # print(pm_df_long_filled_gaps_filtered.count())

    # # FILTERING READY FOR IMPUTING

    # # pm_df_long_outliers_dropped = preprocessing_logic.iqr_kpi_outlier_detection(
    # #     pm_df_long_filled_gaps_filtered, k=3.0
    # # )

    # # UP TO THIS MOMENT WORKS

    # pm_df_long_pre_impute = imputing.categorize_kpi_with_definitions(
    #     pm_df_long_filled_gaps_filtered, kpi_definitions
    # )

    # pm_df_long_pre_impute = pm_df_long_pre_impute.repartition(
    #     "kpi_id", "bts_id", "distname"
    # ).sortWithinPartitions("kpi_id", "bts_id", "distname", "start_time")

    # pm_df_long_imputed = imputing.impute(
    #     pm_df=pm_df_long_pre_impute,
    #     group_cols=["kpi_id", "bts_id", "distname"],
    #     order_col="start_time",
    #     value_col="kpi_value",
    #     agg_method_col="agg_method",
    #     max_imputable_gap=MAX_IMPUTABLE_GAP,
    # )

    # # sdm.write_parquet(pm_df_long_imputed, PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed")
    # pm_df_long_imputed = sdm.read_parquet(PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed")

    # # NOTE: There is no way, to allign all kpis to the min of cell
    # # The crossjoin for this will be added at the end of preprocessing, when all goes through

    # pm_training_windows_density = kpi_coverage.compute_window_density_sparse(
    #     pm_df_long_imputed,
    #     window_hours=168,
    #     stride_hours=24,
    #     density_threshold=0.917,
    # )

    # # ------------------------------------------------------------------
    # # Stage 3: discard density-failing windows
    # # ------------------------------------------------------------------
    # logger.info("Stage 3: discarding density-failing windows ...")
    # good_training_windows_density = kpi_coverage.discard_invalid_windows(pm_training_windows_density)

    # # ------------------------------------------------------------------
    # # Stage 2b: max-gap filter with leading-gap awareness
    # # ------------------------------------------------------------------
    # logger.info(
    #     f"Stage 2b: applying max-gap filter "
    #     f"(max_gap_hours={6}, leading-gap aware) ..."
    # )
    # good_windows_all = kpi_coverage.filter_max_gap_sparse(
    #     pm_df_long_imputed,
    #     good_training_windows_density,
    #     window_hours=168,
    #     stride_hours=24,
    #     max_gap_hours=6,
    # )

    # good_windows_all.cache()
    # n_after_gap = good_windows_all.count()
    # logger.info(f"  {n_after_gap:,} windows remain after max-gap filter.")

    # # ------------------------------------------------------------------
    # # Stage 4: theoretical maximum windows per (distname, kpi_id)
    # # ------------------------------------------------------------------
    # logger.info("Stage 4: computing theoretical window maxima ...")
    # theoretical_max = kpi_coverage.compute_theoretical_max_windows(
    #     pm_df_long_imputed,
    #     window_hours=168,
    #     stride_hours=24,
    # )

    # # ------------------------------------------------------------------
    # # Stage 5: per-KPI yield statistics
    # # ------------------------------------------------------------------
    # logger.info("Stage 5: computing per-KPI yield statistics ...")
    # total_distinct_cells = pm_df_long_imputed.select("distname").distinct().count()

    # kpi_stats = kpi_coverage.compute_kpi_yield_stats(
    #     good_windows_all,
    #     theoretical_max,
    #     total_distinct_cells=total_distinct_cells,
    # )

    # # ------------------------------------------------------------------
    # # Stage 5c: variance filter
    # # ------------------------------------------------------------------
    # # TODO: FIX
    # logger.info("Stage 5c: applying variance filter ...")
    # variant_kpis = kpi_coverage.filter_variance(
    #     pm_df_long_imputed,
    #     good_windows_all,
    #     min_std_val=0.01,
    #     max_zero_frac=0.95,
    # )
    # logger.info(f"  {len(variant_kpis)} KPIs pass variance filter.")

    # kpi_stats_filtered = kpi_stats.filter(f.col("kpi_id").isin(list(variant_kpis)))

    # # Filtering KPIs per window coverage
    # candidates = kpi_coverage.prefilter_kpis(
    #     kpi_stats_filtered,
    #     min_window_coverage_frac=0.5,
    #     min_frac_contributing_cells=0.5,
    # )

    # # Build cached greedy-loop DataFrame: candidates only
    # good_windows_candidates = good_windows_all.filter(f.col("kpi_id").isin(candidates)).drop(
    #     "window_valid_frac", "is_good_window"
    # )

    # good_windows_all.unpersist()
    # good_windows_candidates.cache()
    # good_windows_candidates.count()
    # # NR_5487 do sprawdzenia case (mniejszy udział w danych)
    # # ------------------------------------------------------------------
    # # Stage 7: greedy joint KPI selection
    # # ------------------------------------------------------------------
    # logger.info("Stage 7: running greedy joint KPI selection ...")
    # # Pass 1: run without floor, get full curve

    # selected_kpis = kpi_coverage.greedy_joint_kpi_selection(
    #     good_windows_candidates,
    #     candidates,
    #     min_joint_windows_abs=None,
    # )
    # logger.info(f"  Selected {len(selected_kpis)} KPIs from {len(candidates)} candidates.")

    # logger.info(f"KPI COVERAGE - KPIs SELECTED: {' '.join(selected_kpis)}")

    # good_windows_selected = good_windows_candidates.filter(f.col("kpi_id").isin(selected_kpis))
    # pm_df_long_imputed_selected = pm_df_long_imputed.filter(f.col("kpi_id").isin(selected_kpis))

    # # PELT changepoint detection
    # # pm_df_long_segmented = changepoint_detection.add_regime_ids(pm_df_long_imputed_selected)

    # # pm_df_long_segmented = pm_df_long_segmented.cache()

    # # # standardize per given segment

    # # pm_df_long_segmented.count()
    # # ADD AFTER KPI SEGMANTATION SCALING
    # # COMBINE SCALED SEGMENTS TO ONE KPIS AGAIN
    # # TODO: SAVING FOR VISUALS DATAFRAMES and DATA OVERALL

    # sdm.write_parquet(pm_df_long_imputed_selected, PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed_selected", mode="overwrite")
    # sdm.write_parquet(good_windows_selected, PREPROCESSED_DATASET_PATH / "intermediate" / "good_windows_selected", mode="overwrite")

    pm_df_long_imputed_selected = sdm.read_parquet(
        PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed_selected"
    )
    good_windows_selected = sdm.read_parquet(
        PREPROCESSED_DATASET_PATH / "intermediate" / "good_windows_selected"
    )

    print(
        pm_df_long_imputed_selected.select("kpi_id").distinct().rdd.flatMap(lambda x: x).collect()
    )

    pm_df_long_indexed_winds = kpi_coverage.attach_windows_index_to_pm(
        pm_df_long_imputed_selected,
        good_windows_selected,
        window_hours=168,
    )

    pm_df_long_indexed_winds = pm_df_long_indexed_winds.cache()
    pm_df_long_indexed_winds.count()
    pm_df_long_preprocessed = kpi_coverage.extract_valid_pm_windows(
        pm_df_long_imputed_selected,
        good_windows_selected,
        window_hours=168,
    )

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
