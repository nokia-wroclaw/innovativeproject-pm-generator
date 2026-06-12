from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from genpm.preprocessing.configs import PreprocessingConfig
from genpm.preprocessing.logic import imputing, kpi_coverage, scaling, simple_logic
from genpm.utils.consts import MAX_IMPUTABLE_GAP, SHARED_DIR_PATH
from genpm.utils.logger import get_logger
from genpm.utils.spark_session import SparkDataManager
from genpm.utils.utils import validate_windowed_pm

logger = get_logger()


PREPROCESSED_DATASET_PATH = SHARED_DIR_PATH / "preprocessed_dataset"

VERBOSE_DIAGNOSTICS = True


def _log_pm_diag(label: str, df: DataFrame, group_cols: tuple[str, ...], verbose: bool) -> None:
    """Print kpi_id count and kpi×group count for a long-format PM dataframe."""
    if not verbose:
        return
    df = df.cache()
    n_kpi = df.select("kpi_id").distinct().count()
    n_kpi_group = df.select("kpi_id", *group_cols).distinct().count()
    logger.info(f"[DIAG] {label} | kpi_ids={n_kpi:,}  kpi×group={n_kpi_group:,}")


def _log_window_diag(label: str, df: DataFrame, group_cols: tuple[str, ...], verbose: bool) -> None:
    """Print window counts (total + per-group stats) for a window metadata dataframe."""
    if not verbose:
        return
    n_kpi = df.select("kpi_id").distinct().count()
    per_group = (
        df.select(*group_cols, "start_time")
        .distinct()
        .groupBy(*group_cols)
        .agg(f.count("*").alias("n_windows"))
    )
    stats = per_group.agg(
        f.sum("n_windows").alias("total"),
        f.min("n_windows").alias("min_per_group"),
        f.max("n_windows").alias("max_per_group"),
        f.mean("n_windows").alias("mean_per_group"),
    ).collect()[0]
    logger.info(
        f"[DIAG] {label} | kpi_ids={n_kpi:,}  "
        f"total_windows={stats['total']:,}  "
        f"per_group min={stats['min_per_group']} max={stats['max_per_group']} "
        f"mean={stats['mean_per_group']:.1f}"
    )


def _load_data(
    sdm: SparkDataManager, preprocessing_cfg: PreprocessingConfig
) -> tuple[DataFrame, DataFrame, DataFrame]:
    logger.info("Loading raw data: PM, KPI definitions, simple reports")
    pm_df_long = sdm.read_parquet(preprocessing_cfg.pm_data_raw_path)
    kpis_definitions_df = sdm.read_parquet(preprocessing_cfg.kpi_definitions_raw_path)
    simple_reports_df = sdm.read_parquet(preprocessing_cfg.simple_reports_raw_path)
    logger.info("Raw data loaded successfully")
    return pm_df_long, kpis_definitions_df, simple_reports_df


def run_preprocessing(sdm: SparkDataManager, preprocessing_cfg: PreprocessingConfig) -> None:
    logger.info("Starting preprocessing pipeline")
    pm_df_long_raw, kpi_definitions_df_raw, simple_reports_df_raw = _load_data(
        sdm, preprocessing_cfg
    )

    pm_df_long_raw = simple_logic.raw_pm_preperation(pm_df_long_raw)
    logger.info("Raw PM preparation done")

    # # KPI version flattening
    pm_df_long, kpi_definitions = simple_logic.coalesce_kpi_version(
        pm_df_long_raw, kpi_definitions_df_raw
    )
    logger.info("KPI version coalescing done")

    pm_df_long, pm_df_const_kpi = simple_logic.pop_constant_kpis(pm_df_long)
    logger.info("Constant KPIs separated")

    simple_reports_pivoted = simple_logic.simple_reports_pivot(simple_reports_df_raw)
    logger.info("Simple reports pivoted")

    simple_report_grouping_cols = ("distname", "bts_id", "datetime")
    cell_config = tuple(
        [c for c in simple_reports_pivoted.columns if c not in simple_report_grouping_cols]
    )

    # GROUPING COLS DEFINITION FOR LATER ARGUMENTS
    _GROUPING_COLS = ("distname", "bts_id", *cell_config)

    pm_cm_df_long = simple_logic.pm_and_reports_data_joined(
        pm_df_long, simple_reports_pivoted, cell_config
    )
    logger.info(f"PM and CM data joined — grouping cols: {_GROUPING_COLS}")

    # Intermediate saves for better performance
    pm_cm_df_long = sdm.hard_checkpoint_to_parquet(
        pm_cm_df_long, "/".join([preprocessing_cfg.intermediate_path, "pm_cm_df_long"])
    )
    kpi_definitions = sdm.hard_checkpoint_to_parquet(
        kpi_definitions, "/".join([preprocessing_cfg.intermediate_path, "kpi_definitions"])
    )
    simple_reports_pivoted = sdm.hard_checkpoint_to_parquet(
        simple_reports_pivoted,
        "/".join([preprocessing_cfg.intermediate_path, "simple_reports_pivoted"]),
    )
    logger.info("Initial checkpoints written")

    # NOTE: INTERMEDIATE SAVES - THOSE ARE DEBUG

    # sdm.write_parquet(pm_cm_df_long, PREPROCESSED_DATASET_PATH / "intermediate" / "pm_cm_df_long", mode="overwrite")
    # sdm.write_parquet(kpi_definitions, PREPROCESSED_DATASET_PATH / "intermediate" / "kpi_definitions", mode="overwrite")
    # sdm.write_parquet(pm_df_const_kpi, PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_const_kpi", mode="overwrite")
    # sdm.write_parquet(simple_reports_pivoted, PREPROCESSED_DATASET_PATH / "intermediate" / "simple_reports_pivoted", mode="overwrite")

    # pm_cm_df_long = sdm.read_parquet(PREPROCESSED_DATASET_PATH / "intermediate" / "pm_cm_df_long")
    # kpi_definitions = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "kpi_definitions"
    # )

    # pm_df_const_kpi = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_const_kpi"
    # )

    # simple_reports_pivoted = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "simple_reports_pivoted"
    # )

    # simple_report_grouping_cols = ("distname", "bts_id", "datetime")
    # cell_config = tuple(
    #     [c for c in simple_reports_pivoted.columns if c not in simple_report_grouping_cols]
    # )

    # _GROUPING_COLS = ("distname", "bts_id", *cell_config)
    # NOTE: INTERMEDIATE SAVES END

    # STAGE: COVERAGE ANALYSIS
    # Timestamp frequency uniformoty (1 hour)
    # Perc cell allignment to min max (window coverage handles the rest)
    logger.info("Stage: coverage analysis — filling internal gaps")
    pm_df_long_filled_gaps = kpi_coverage.fill_internal_gaps(
        pm_cm_df_long, _GROUPING_COLS, "start_time"
    )
    logger.info("COVERAGE - gap-filled rows computed")
    _log_pm_diag(
        "after fill_internal_gaps",
        pm_df_long_filled_gaps,
        _GROUPING_COLS,
        preprocessing_cfg.verbose,
    )

    # Stage 0: series-scoped pre-impute gate
    # Drops (group_cols, kpi_id) series that are too sparse OR dominated by long
    # null runs. Joins on BOTH columns so weak series are removed individually —
    # a KPI that survives in any cell carries on; the whole KPI is never dropped.
    logger.info("Stage: series imputability gate")
    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps.join(
        kpi_coverage.series_imputability_gate(
            pm_df_long_filled_gaps,
            _GROUPING_COLS,
            min_global_density=preprocessing_cfg.kpi_min_global_density,
            max_imputable_gap=preprocessing_cfg.max_gap_hours,  # THIS PARAMETER IS VERY AGGRESIVE,
            min_imputable_gap_frac=preprocessing_cfg.min_imputable_gap_frac,
        ),
        on=["kpi_id", *_GROUPING_COLS],
        how="inner",
    )

    _log_pm_diag(
        "after series_imputability_gate",
        pm_df_long_filled_gaps_filtered,
        _GROUPING_COLS,
        preprocessing_cfg.verbose,
    )

    # # FILTERING READY FOR IMPUTING

    # NOTE: KPI outlier imputting is for now disabled
    # pm_df_long_filled_gaps_filtered = simple_logic.iqr_kpi_outlier_detection(
    #     pm_df_long_filled_gaps_filtered, k=args.iqr_outlier_k_param
    # )
    # [VERBOSE_DIAGNOSTICS] _log_pm_diag("after iqr_kpi_outlier_detection", pm_df_long_filled_gaps_filtered, _GROUPING_COLS)

    pm_df_long_pre_impute = imputing.categorize_kpi_with_definitions(
        pm_df_long_filled_gaps_filtered, kpi_definitions
    )
    _log_pm_diag(
        "after categorize_kpi_with_definitions (join with kpi_definitions)",
        pm_df_long_pre_impute,
        _GROUPING_COLS,
        preprocessing_cfg.verbose,
    )

    # repartition for better imputing
    logger.info("Repartitioning for imputation")
    pm_df_long_pre_impute = pm_df_long_pre_impute.repartition(
        "kpi_id", *_GROUPING_COLS
    ).sortWithinPartitions("kpi_id", *_GROUPING_COLS, "start_time")

    logger.info("Stage: imputing")
    pm_df_long_imputed = imputing.impute(
        pm_df=pm_df_long_pre_impute,
        group_cols=["kpi_id", *_GROUPING_COLS],
        order_col="start_time",
        value_col="kpi_value",
        agg_method_col="agg_method",
        max_imputable_gap=MAX_IMPUTABLE_GAP,
    )

    pm_df_long_imputed = sdm.hard_checkpoint_to_parquet(
        pm_df_long_imputed, "/".join([preprocessing_cfg.intermediate_path, "pm_cm_df_long"])
    )

    # NOTE: INTERMEDIATE SAVES - FOR DEBUGGING AND QUICKER RUNS

    # sdm.write_parquet(
    #     pm_df_long_imputed,
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed",
    #     mode="overwrite",
    # )
    # pm_df_long_imputed = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed"
    # )
    _log_pm_diag(
        "after imputing (pm_df_long_imputed)",
        pm_df_long_imputed,
        _GROUPING_COLS,
        preprocessing_cfg.verbose,
    )

    # NOTE: There is no way, to allign all kpis to the min of cell earlier
    # The crossjoin for this will be at the end only

    pm_df_windows = kpi_coverage.drop_windows_with_nulls(
        pm_df_long_imputed,
        _GROUPING_COLS,
        window_hours=preprocessing_cfg.window_width_hours,
        stride_hours=preprocessing_cfg.stride_hours,
    )
    _log_window_diag(
        "after drop_windows_with_nulls", pm_df_windows, _GROUPING_COLS, preprocessing_cfg.verbose
    )

    # Stage 2: discard density-failing windows; log good/total in one agg pass
    logger.info("Stage: discarding invalid windows and computing KPI yield stats")
    good_windows_cleaned = kpi_coverage.discard_invalid_windows(pm_df_windows)

    n_cells = good_windows_cleaned.select(*_GROUPING_COLS).distinct().count()

    kpi_yield_stats = kpi_coverage.compute_kpi_yield_stats(
        good_windows_cleaned, _GROUPING_COLS, total_distinct_cells=n_cells
    )

    valid_kpi_candidates = kpi_coverage.prefilter_kpis(kpi_yield_stats)

    if preprocessing_cfg.verbose:
        logger.info(f"[DIAG] after prefilter_kpis | candidates={len(valid_kpi_candidates):,}")

    # Stage 5: build cached greedy-loop DataFrame from candidate KPIs only
    # TODO: There should be a way, to filter after all of the filters from above, so some kpis
    # could be forced through
    good_windows_candidates = good_windows_cleaned.filter(
        f.col("kpi_id").isin(valid_kpi_candidates)
    ).drop("window_valid_frac", "is_good_window")

    good_windows_cleaned.unpersist()
    good_windows_candidates.cache()
    good_windows_candidates.count()
    _log_window_diag(
        "good_windows_candidates (post prefilter)",
        good_windows_candidates,
        _GROUPING_COLS,
        preprocessing_cfg.verbose,
    )

    # SCALING HERE

    # --- old GroupedKPIScaler
    # scaler = GroupedKPIScaler(
    #     value_col="kpi_value",
    #     group_cols=["kpi_id", "bts_id", "distname"],
    #     min_valid_points=4,
    #     percentile_accuracy=10_000,
    # )
    # params_df = scaler.fit(pm_df_long_imputed)
    # pm_df_long_scaled = scaler.transform(pm_df_long_imputed)
    # clean_audit_df = scaler.summary().filter(f.col("scaler") != f.lit("SKIP"))
    # pm_df_long_scaled = pm_df_long_scaled.join(
    #     f.broadcast(clean_audit_df.select("kpi_id", "distname").distinct()),
    #     on=["kpi_id", "distname"],
    #     how="inner",
    # )

    logger.info("Stage: fitting and applying MinMax scaler")
    scaler = scaling.SimpleMinMaxScaler(
        value_col="kpi_value", group_cols=["kpi_id", *_GROUPING_COLS]
    )
    params_df = scaler.fit(pm_df_long_imputed)
    pm_df_long_scaled = scaler.transform(pm_df_long_imputed)

    # Greedy joint KPI selection
    logger.info("Stage: greedy joint KPI selection")
    selected_kpis = kpi_coverage.greedy_joint_kpi_selection(
        good_windows_candidates,
        valid_kpi_candidates,
        _GROUPING_COLS,
        min_joint_windows_abs=preprocessing_cfg.min_joint_windows_abs,
        forced_kpis=preprocessing_cfg.forced_kpis,
    )
    logger.info(f"Selected {len(selected_kpis)} KPIs from {len(valid_kpi_candidates)} candidates.")

    good_windows_selected = good_windows_candidates.filter(f.col("kpi_id").isin(selected_kpis))
    pm_df_long_scaled = pm_df_long_scaled.filter(f.col("kpi_id").isin(selected_kpis))
    _log_pm_diag(
        "pm_df_long_imputed_selected (post greedy)",
        pm_df_long_scaled,
        _GROUPING_COLS,
        preprocessing_cfg.verbose,
    )
    _log_window_diag(
        "good_windows_selected (post greedy)",
        good_windows_selected,
        _GROUPING_COLS,
        preprocessing_cfg.verbose,
    )

    good_windows_selected = sdm.hard_checkpoint_to_parquet(
        good_windows_selected,
        "/".join([preprocessing_cfg.intermediate_path, "good_windows_selected"]),
    )
    pm_df_long_scaled = sdm.hard_checkpoint_to_parquet(
        pm_df_long_scaled, "/".join([preprocessing_cfg.intermediate_path, "pm_df_long_scaled"])
    )

    # NOTE: INTERMEDIATE SAVE
    # sdm.write_parquet(
    #     pm_df_long_scaled,
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed_selected",
    #     mode="overwrite",
    # )
    # sdm.write_parquet(
    #     good_windows_selected,
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "good_windows_selected",
    #     mode="overwrite",
    # )

    # pm_df_long_scaled = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "pm_df_long_imputed_selected"
    # )
    # good_windows_selected = sdm.read_parquet(
    #     PREPROCESSED_DATASET_PATH / "intermediate" / "good_windows_selected"
    # )

    # # This list, has to be created here, as it is brought from filtered
    # selected_kpis = [r["kpi_id"] for r in pm_df_long_scaled.select("kpi_id").distinct().collect()]
    # INTERMIEDIATE END

    good_windows_selected = kpi_coverage.filter_joint_complete_windows(
        good_windows_selected,
        selected_kpis,
        _GROUPING_COLS,
    )
    _log_window_diag(
        "after filter_joint_complete_windows",
        good_windows_selected,
        _GROUPING_COLS,
        preprocessing_cfg.verbose,
    )

    # This function indexes window_start in pm data
    logger.info("Stage: emitting window index")
    pm_df_long_indexed_winds = kpi_coverage.emit_window_index(
        pm_df_long_scaled,
        good_windows_selected,
        _GROUPING_COLS,
        window_hours=preprocessing_cfg.window_width_hours,
    )
    if VERBOSE_DIAGNOSTICS:
        validate_windowed_pm(pm_df_long_indexed_winds)

    # PM long pivot to wide
    logger.info("Stage: pivoting long PM to wide format")
    pm_df_wide_indexed_windows = (
        pm_df_long_indexed_winds.groupBy(*_GROUPING_COLS, "window_anchor", "start_time")
        .pivot("kpi_id")
        .agg(f.first("kpi_value"))
        .orderBy("distname", "window_anchor")
    )

    _TIME_COLS = ["window_anchor", "start_time"]

    _KPI_COLS = [
        c
        for c in pm_df_wide_indexed_windows.columns
        if c not in _TIME_COLS and c not in _GROUPING_COLS
    ]

    pm_df_wide_materialized_windows = simple_logic.materialize_windows(
        pm_df_wide_preprocessed=pm_df_wide_indexed_windows,
        identity_cols=_GROUPING_COLS,
        kpi_cols=_KPI_COLS,
        window_width=preprocessing_cfg.window_width_hours,
    )
    logger.info(f"Windows materialized — {len(_KPI_COLS)} KPI columns")

    # HELPER DFS:
    unique_kpis = pm_df_long_indexed_winds.select("kpi_id").distinct()

    # Save preprocessed data

    dataset_paths_and_dfs: dict[str, DataFrame] = {
        "pm_df_wide_materialized_windows": pm_df_wide_materialized_windows,
        "pm_df_long_indexed_winds": pm_df_long_indexed_winds,
        "pm_df_wide_indexed_winds": pm_df_wide_indexed_windows,
        "scaling_params_df": params_df,
        # Visual and forms HELPER dfs
        "HELPER_pm_data_const_kpi": pm_df_const_kpi,
        "HELPER_unique_kpis": unique_kpis,
        # TODO: add other helpers
    }

    # END OF PREPROCESSING

    # SAVE PREPROCESSED DATA
    logger.info(
        f"Saving {len(dataset_paths_and_dfs)} output datasets to {preprocessing_cfg.output_path_prefix}"
    )

    for df_path, df in dataset_paths_and_dfs.items():
        logger.info(f"Writing: {df_path}")
        sdm.write_parquet(
            df,
            "/".join([preprocessing_cfg.output_path_prefix, df_path]),
            mode="overwrite",
        )

    logger.info("Preprocessing pipeline complete")
