from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from genpm.preprocessing.configs import PreprocessingConfig
from genpm.preprocessing.logic import imputing, kpi_coverage, preprocessing_logic
from genpm.preprocessing.logic.scaling import GroupedKPIScaler
from genpm.utils.consts import MAX_IMPUTABLE_GAP, SHARED_DIR_PATH
from genpm.utils.logger import get_logger
from genpm.utils.utils import SparkDataManager

logger = get_logger()

PREPROCESSED_DATASET_PATH = SHARED_DIR_PATH / "preprocessed_dataset"

VERBOSE_DIAGNOSTICS = False


def _log_pm_diag(label: str, df: DataFrame) -> None:
    """Print kpi_id count and kpi×distname count for a long-format PM dataframe."""
    if not VERBOSE_DIAGNOSTICS:
        return
    n_kpi = df.select("kpi_id").distinct().count()
    n_kpi_dist = df.select("kpi_id", "distname").distinct().count()
    logger.info(f"[DIAG] {label} | kpi_ids={n_kpi:,}  kpi×distname={n_kpi_dist:,}")


def _log_window_diag(label: str, df: DataFrame) -> None:
    """Print window counts (total + per-distname stats) for a window metadata dataframe."""
    if not VERBOSE_DIAGNOSTICS:
        return
    n_kpi = df.select("kpi_id").distinct().count()
    per_dist = (
        df.select("distname", "start_time")
        .distinct()
        .groupBy("distname")
        .agg(f.count("*").alias("n_windows"))
    )
    stats = per_dist.agg(
        f.sum("n_windows").alias("total"),
        f.min("n_windows").alias("min_per_dist"),
        f.max("n_windows").alias("max_per_dist"),
        f.mean("n_windows").alias("mean_per_dist"),
    ).collect()[0]
    logger.info(
        f"[DIAG] {label} | kpi_ids={n_kpi:,}  "
        f"total_windows={stats['total']:,}  "
        f"per_distname min={stats['min_per_dist']} max={stats['max_per_dist']} "
        f"mean={stats['mean_per_dist']:.1f}"
    )


def _load_data(
    sdm: SparkDataManager, preprocessing_cfg: PreprocessingConfig
) -> tuple[DataFrame, DataFrame, DataFrame]:
    pm_df_long = sdm.read_parquet(preprocessing_cfg.pm_data_raw_path)
    kpis_definitions_df = sdm.read_parquet(preprocessing_cfg.kpi_definitions_raw_path)
    simple_reports_df = sdm.read_parquet(preprocessing_cfg.simple_reports_raw_path)

    return pm_df_long, kpis_definitions_df, simple_reports_df


def run_preprocessing(sdm: SparkDataManager, preprocessing_cfg: PreprocessingConfig) -> None:
    pm_df_long_raw, kpi_definitions_df_raw, simple_reports_df_raw = _load_data(
        sdm, preprocessing_cfg
    )

    pm_df_long_raw = preprocessing_logic.raw_pm_preperation(pm_df_long_raw)
    # [VERBOSE_DIAGNOSTICS] _log_pm_diag("after raw_pm_preperation", pm_df_long_raw)

    # # KPI version flattening
    pm_df_long, kpi_definitions = preprocessing_logic.coalesce_kpi_version(
        pm_df_long_raw, kpi_definitions_df_raw
    )
    # [VERBOSE_DIAGNOSTICS] _log_pm_diag("after coalesce_kpi_version", pm_df_long)

    pm_df_long, pm_df_const_kpi = preprocessing_logic.pop_constant_kpis(pm_df_long)
    # [VERBOSE_DIAGNOSTICS] _log_pm_diag("after pop_constant_kpis", pm_df_long)

    # # NOTE: INTERMEDIATE SAVES - FOR DEBUGGING AND QUICKER RUNS

    # sdm.write_parquet(pm_df_long, PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long", mode="overwrite")
    # sdm.write_parquet(kpi_definitions, PREPROCESSED_DATASET_PATH / "intermediate" / "kpi_definitions", mode="overwrite")
    # sdm.write_parquet(pm_df_const_kpi, PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_const_kpi", mode="overwrite")

    # pm_df_long = sdm.read_parquet(PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long")
    # kpi_definitions = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "kpi_definitions"
    # )

    # pm_df_const_kpi = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_const_kpi"
    # )

    # STAGE: COVERAGE ANALYSIS
    # Timestamp frequency uniformoty (1 hour)
    # Perc cell allignment to min max (window coverage handles the rest)
    pm_df_long_filled_gaps = kpi_coverage.fill_internal_gaps(pm_df_long, "start_time")
    logger.info("COVERAGE - gap-filled rows computed")
    _log_pm_diag("after fill_internal_gaps", pm_df_long_filled_gaps)

    # GLOBAL COVERAGE FILTER
    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps.join(
        kpi_coverage.filter_global_value_density(
            pm_df_long_filled_gaps,
            min_global_density=preprocessing_cfg.kpi_min_global_density,
            min_frac_cells_passing=preprocessing_cfg.kpi_global_min_frac_cells_passing,
        ),
        on="kpi_id",
        how="inner",
    )
    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.cache()
    _log_pm_diag("after filter_global_value_density", pm_df_long_filled_gaps_filtered)

    # GAP PATTERNS FILTERING (consistant NULL patterns detection throuh median)
    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.join(
        kpi_coverage.filter_gap_pattern(
            pm_df_long_filled_gaps_filtered,
            max_imputable_gap=24,
            min_imputable_gap_frac=preprocessing_cfg.min_imputable_gap_frac,
        ),
        on="kpi_id",
        how="inner",
    )

    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.cache()
    _log_pm_diag("after filter_gap_pattern", pm_df_long_filled_gaps_filtered)

    # pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.cache()
    # print(pm_df_long_filled_gaps_filtered.count())

    # FILTERING READY FOR IMPUTING

    # NOTE: KPI outlier imputting is for now disabled
    # pm_df_long_filled_gaps_filtered = preprocessing_logic.iqr_kpi_outlier_detection(
    #     pm_df_long_filled_gaps_filtered, k=args.iqr_outlier_k_param
    # )
    # [VERBOSE_DIAGNOSTICS] _log_pm_diag("after iqr_kpi_outlier_detection", pm_df_long_filled_gaps_filtered)

    pm_df_long_pre_impute = imputing.categorize_kpi_with_definitions(
        pm_df_long_filled_gaps_filtered, kpi_definitions
    )
    _log_pm_diag(
        "after categorize_kpi_with_definitions (join with kpi_definitions)", pm_df_long_pre_impute
    )

    # repartition for better imputing
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

    # # NOTE: INTERMEDIATE SAVES - FOR DEBUGGING AND QUICKER RUNS

    sdm.write_parquet(
        pm_df_long_imputed,
        PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed",
        mode="overwrite",
    )
    pm_df_long_imputed = sdm.read_parquet(
        PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed"
    )
    _log_pm_diag("after imputing (pm_df_long_imputed)", pm_df_long_imputed)

    # NOTE: There is no way, to allign all kpis to the min of cell
    # The crossjoin for this will be added at the end of preprocessing, when all goes through

    pm_training_windows_density = kpi_coverage.drop_windows_containing_nulls(
        pm_df_long_imputed,
        window_hours=preprocessing_cfg.window_width_hours,
        stride_hours=preprocessing_cfg.stride_hours,
    )
    _log_window_diag("after drop_windows_containing_nulls", pm_training_windows_density)

    # Stage - discard density-failing windows
    good_training_windows_density = kpi_coverage.discard_invalid_windows(
        pm_training_windows_density
    )
    _log_window_diag("after discard_invalid_windows", good_training_windows_density)

    # Stage 2b: max-gap filter with leading-gap awareness
    good_windows_all = kpi_coverage.filter_max_gap_sparse(
        pm_df_long_imputed,
        good_training_windows_density,
        window_hours=preprocessing_cfg.window_width_hours,
        stride_hours=preprocessing_cfg.stride_hours,
        # TODO: Find, if this should be MAX_IMPUTABLE_GAP or it could be higher
        max_gap_hours=MAX_IMPUTABLE_GAP,
    )

    good_windows_all.cache()
    _log_window_diag("after filter_max_gap_sparse", good_windows_all)

    # TODO: The below calculation of max windows doesnt make sense and IS JUST WRONG
    theoretical_max = kpi_coverage.compute_theoretical_max_windows(
        pm_df_long_imputed,
        window_hours=preprocessing_cfg.window_width_hours,
        stride_hours=preprocessing_cfg.stride_hours,
    )

    # THE ABOVE FILTER INFLUENCES THIS ONE
    # Stage 5: per-KPI yield statistics
    kpi_stats = kpi_coverage.compute_kpi_yield_stats(
        good_windows_all,
        theoretical_max,
        total_distinct_cells=pm_df_long_imputed.select("distname").distinct().count(),
    )

    # Stage - variance filter
    # TODO: Add returning those KPIs for naive stale kpi generation
    logger.info("Stage 5c: applying variance filter ...")
    variant_kpis = kpi_coverage.filter_variance(
        pm_df_long_imputed,
        good_windows_all,
        min_std_val=preprocessing_cfg.kpi_min_std_val,
        max_zero_frac=preprocessing_cfg.max_zero_frac,
    )
    logger.info(f"{len(variant_kpis)} KPIs pass variance filter.")
    if VERBOSE_DIAGNOSTICS:
        logger.info(f"[DIAG] after filter_variance | kpi_ids={len(variant_kpis):,}")

    kpi_stats_filtered = kpi_stats.filter(f.col("kpi_id").isin(list(variant_kpis)))
    if VERBOSE_DIAGNOSTICS:
        logger.info(
            f"[DIAG] kpi_stats_filtered (post variance) | kpi_ids={kpi_stats_filtered.count():,}"
        )

    # Filtering KPIs per window coverage
    # TODO: PREFILTER KPIS IS GARBAGE - SHOULD BE AVOIDED (CHECK WHAT IS USEFULL HERE)
    candidates = kpi_coverage.prefilter_kpis(
        kpi_stats_filtered,
        min_window_coverage_frac=0.5,
        min_frac_contributing_cells=0.5,
    )
    if VERBOSE_DIAGNOSTICS:
        logger.info(f"[DIAG] after prefilter_kpis | candidates={len(candidates):,}")

    # Build cached greedy-loop DataFrame: candidates only
    # TODO: There should be a way, to filter after all of the filters from above, so some kpis
    # could be forced through
    good_windows_candidates = good_windows_all.filter(f.col("kpi_id").isin(candidates)).drop(
        "window_valid_frac", "is_good_window"
    )

    good_windows_all.unpersist()
    good_windows_candidates.cache()
    good_windows_candidates.count()
    _log_window_diag("good_windows_candidates (post prefilter)", good_windows_candidates)

    # Greedy joint KPI selection
    selected_kpis = kpi_coverage.greedy_joint_kpi_selection(
        good_windows_candidates,
        # TODO: Add overwriting the candidate list, so some KPIs could be forced in training
        candidates,
        min_joint_windows_abs=None,
    )
    logger.info(f"Selected {len(selected_kpis)} KPIs from {len(candidates)} candidates.")

    good_windows_selected = good_windows_candidates.filter(f.col("kpi_id").isin(selected_kpis))
    pm_df_long_imputed_selected = pm_df_long_imputed.filter(f.col("kpi_id").isin(selected_kpis))
    _log_pm_diag("pm_df_long_imputed_selected (post greedy)", pm_df_long_imputed_selected)
    _log_window_diag("good_windows_selected (post greedy)", good_windows_selected)

    # TODO: PELT changepoint detection WITH STANDARDIZATION PER SEGMENT
    # pm_df_long_segmented = changepoint_detection.add_regime_ids(pm_df_long_imputed_selected)
    # [VERBOSE_DIAGNOSTICS] _log_pm_diag("after add_regime_ids", pm_df_long_segmented)

    # pm_df_long_segmented = pm_df_long_segmented.cache()

    # standardize per given segment

    # ADD AFTER KPI SEGMANTATION SCALING

    # pm_df_long_imputed_selected = pm_df_long_imputed_selected.localCheckpoint()
    # sdm.write_parquet(pm_df_long_imputed_selected, PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed_selected_lol", mode="overwrite")
    pm_df_long_imputed_selected = sdm.read_parquet(
        PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed_selected_lol"
    )
    _log_pm_diag("pm_df_long_imputed_selected (read back)", pm_df_long_imputed_selected)

    print("intermediate")
    # Scaling

    scaler = GroupedKPIScaler(
        value_col="kpi_value",
        group_cols=["kpi_id", "bts_id", "distname"],
        min_valid_points=4,
        percentile_accuracy=10_000,
        broadcast_params=True,
        params_path=str(PREPROCESSED_DATASET_PATH / "final_scaled" / "scaling_params_df"),
    )

    audit_df = scaler.fit(pm_df_long_imputed_selected)
    pm_df_long_scaled = scaler.transform(pm_df_long_imputed_selected)

    # TODO: FIX
    # temporary for excluding SKIP kpi-cells
    clean_audit_df = audit_df.filter(f.col("scaler") != f.lit("SKIP"))

    pm_df_long_scaled = pm_df_long_scaled.join(
        f.broadcast(clean_audit_df.select("kpi_id", "distname").distinct()),
        on=["kpi_id", "distname"],
        how="inner",
    )

    # COMBINE SCALED SEGMENTS TO ONE KPIS AGAIN
    # TODO: SAVING FOR VISUALS DATAFRAMES and DATA OVERALL

    # NOTE: INTERMEDIATE SAVE
    # sdm.write_parquet(
    #     pm_df_long_imputed_selected,
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed_selected",
    #     mode="overwrite",
    # )
    # sdm.write_parquet(
    #     good_windows_selected,
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "good_windows_selected",
    #     mode="overwrite",
    # )

    # pm_df_long_imputed_selected = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed_selected"
    # )
    # good_windows_selected = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "good_windows_selected"
    # )

    # print(
    #     pm_df_long_imputed_selected.select("kpi_id").distinct().rdd.flatMap(lambda x: x).collect()
    # )
    # This list, has to be created here, as it is brought from filtered
    # selected_kpis = [
    #     r["kpi_id"] for r in pm_df_long_imputed_selected.select("kpi_id").distinct().collect()
    # ]
    # INTERMIEDIATE END

    # TODO: Verify, what could be the root cause of all those nulls in data,
    # Maybe The starting points of KPIs are the real issue?
    good_windows_selected = kpi_coverage.filter_joint_complete_windows(
        good_windows_selected,
        selected_kpis,
    )
    _log_window_diag("after filter_joint_complete_windows", good_windows_selected)
    # good_windows_selected.cache()
    # n_joint = good_windows_selected.select("distname", "start_time").distinct().count()
    # logger.info(f"  {n_joint:,} joint-complete (distname, anchor) windows survive.")

    # This function materializes the windows and indexes window start and its width in 2 columns
    pm_df_long_indexed_winds = kpi_coverage.attach_windows_index_to_pm(
        pm_df_long_scaled,
        good_windows_selected,
        window_hours=preprocessing_cfg.window_width_hours,
    )
    _log_pm_diag("after attach_windows_index_to_pm", pm_df_long_indexed_winds)

    simple_reports = preprocessing_logic.simple_reports_pivot(simple_reports_df_raw)

    # TODO: FIX PIVOT NOT THIS:
    wide = (
        pm_df_long_indexed_winds.groupBy("distname", "bts_id", "window_anchor", "hour_idx")
        .pivot("kpi_id")
        .agg(f.first("kpi_value"))
        .orderBy("distname", "window_anchor", "hour_idx")
    ).cache()

    # TODO: Verify, why this function kills the Spark drivers and FIX
    # pm_df_long_indexed_winds = pm_df_long_indexed_winds.localCheckpoint()
    # pm_df_long_indexed_winds_with_simple_reports = pm_df_long_indexed_winds.join(
    #     simple_reports.drop("datetime", "bts_id"), on="distname", how="left"
    # )
    # [VERBOSE_DIAGNOSTICS] _log_pm_diag("after join with simple_reports", pm_df_long_indexed_winds_with_simple_reports)

    # Save preprocessed data
    list_of_dfs = [
        # pm_df_long_indexed_winds_with_simple_reports,
        pm_df_long_indexed_winds,
        wide,
        pm_df_const_kpi,
        kpi_definitions,
        simple_reports,
    ]

    preprocessed_data_filenames = [
        # "pm_data_dataset_preprocessed",
        "pm_df_long_indexed_winds",
        "pm_df_wide_indexed_winds",
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
            "/".join([str(PREPROCESSED_DATASET_PATH / "final_scaled"), df_path]),
            mode="overwrite",
        )
