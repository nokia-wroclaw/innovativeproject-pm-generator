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


def _log_pm_diag(label: str, df: DataFrame, group_cols: tuple[str, ...]) -> None:
    """Print kpi_id count and kpi×group count for a long-format PM dataframe."""
    if not VERBOSE_DIAGNOSTICS:
        return
    df = df.cache()
    n_kpi = df.select("kpi_id").distinct().count()
    n_kpi_group = df.select("kpi_id", *group_cols).distinct().count()
    logger.info(f"[DIAG] {label} | kpi_ids={n_kpi:,}  kpi×group={n_kpi_group:,}")


def _log_window_diag(label: str, df: DataFrame, group_cols: tuple[str, ...]) -> None:
    """Print window counts (total + per-group stats) for a window metadata dataframe."""
    if not VERBOSE_DIAGNOSTICS:
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
    pm_df_long = sdm.read_parquet(preprocessing_cfg.pm_data_raw_path)
    kpis_definitions_df = sdm.read_parquet(preprocessing_cfg.kpi_definitions_raw_path)
    simple_reports_df = sdm.read_parquet(preprocessing_cfg.simple_reports_raw_path)

    return pm_df_long, kpis_definitions_df, simple_reports_df


def run_preprocessing(sdm: SparkDataManager, preprocessing_cfg: PreprocessingConfig) -> None:
    pm_df_long_raw, kpi_definitions_df_raw, simple_reports_df_raw = _load_data(
        sdm, preprocessing_cfg
    )

    pm_df_long_raw = simple_logic.raw_pm_preperation(pm_df_long_raw)

    # # KPI version flattening
    pm_df_long, kpi_definitions = simple_logic.coalesce_kpi_version(
        pm_df_long_raw, kpi_definitions_df_raw
    )

    pm_df_long, pm_df_const_kpi = simple_logic.pop_constant_kpis(pm_df_long)

    simple_reports_pivoted = simple_logic.simple_reports_pivot(simple_reports_df_raw)

    simple_report_grouping_cols = ("distname", "bts_id", "datetime")
    cell_config = tuple(
        [c for c in simple_reports_pivoted.columns if c not in simple_report_grouping_cols]
    )

    # GROUPING COLS DEFINITION FOR LATER ARGUMENTS
    _GROUPING_COLS = ("distname", "bts_id", *cell_config)

    pm_cm_df_long = simple_logic.pm_and_reports_data_joined(
        pm_df_long, simple_reports_pivoted, cell_config
    )
    # NOTE: INTERMEDIATE SAVES - FOR DEBUGGING AND QUICKER RUNS

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
    pm_df_long_filled_gaps = kpi_coverage.fill_internal_gaps(
        pm_cm_df_long, _GROUPING_COLS, "start_time"
    )
    logger.info("COVERAGE - gap-filled rows computed")
    _log_pm_diag("after fill_internal_gaps", pm_df_long_filled_gaps, _GROUPING_COLS)

    # Stage 0: series-scoped pre-impute gate
    # Drops (group_cols, kpi_id) series that are too sparse OR dominated by long
    # null runs. Joins on BOTH columns so weak series are removed individually —
    # a KPI that survives in any cell carries on; the whole KPI is never dropped.
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
    pm_df_long_filled_gaps_filtered = pm_df_long_filled_gaps_filtered.cache()
    print(pm_df_long_filled_gaps_filtered.count())
    _log_pm_diag("after series_imputability_gate", pm_df_long_filled_gaps_filtered, _GROUPING_COLS)

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
    )

    # repartition for better imputing
    pm_df_long_pre_impute = pm_df_long_pre_impute.repartition(
        "kpi_id", *_GROUPING_COLS
    ).sortWithinPartitions("kpi_id", *_GROUPING_COLS, "start_time")

    pm_df_long_imputed = imputing.impute(
        pm_df=pm_df_long_pre_impute,
        group_cols=["kpi_id", *_GROUPING_COLS],
        order_col="start_time",
        value_col="kpi_value",
        agg_method_col="agg_method",
        max_imputable_gap=MAX_IMPUTABLE_GAP,
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
    _log_pm_diag("after imputing (pm_df_long_imputed)", pm_df_long_imputed, _GROUPING_COLS)

    # NOTE: There is no way, to allign all kpis to the min of cell earlier
    # The crossjoin for this will be at the end only

    pm_df_windows = kpi_coverage.drop_windows_with_nulls(
        pm_df_long_imputed,
        _GROUPING_COLS,
        window_hours=preprocessing_cfg.window_width_hours,
        stride_hours=preprocessing_cfg.stride_hours,
    )
    _log_window_diag("after drop_windows_with_nulls", pm_df_windows, _GROUPING_COLS)

    # Stage 2: discard density-failing windows; log good/total in one agg pass
    good_windows_cleaned = kpi_coverage.discard_invalid_windows(pm_df_windows)
    density_stats = pm_df_windows.agg(
        f.count("*").alias("total_windows"),
        f.sum(f.col("is_good_window").cast("long")).alias("good_windows"),
        f.countDistinct("kpi_id").alias("kpis_assessed"),
        f.countDistinct(*_GROUPING_COLS).alias("groups_assessed"),
    ).first()
    logger.info(
        f"[stage 2] window density: "
        f"{density_stats['good_windows']:,}/{density_stats['total_windows']:,} windows pass "
        f"({100 * density_stats['good_windows'] / density_stats['total_windows']:.1f}%) "
        f"| {density_stats['kpis_assessed']} KPIs | {density_stats['groups_assessed']} groups"
    )

    n_cells = good_windows_cleaned.select(*_GROUPING_COLS).distinct().count()

    kpi_yield_stats = kpi_coverage.compute_kpi_yield_stats(
        good_windows_cleaned, _GROUPING_COLS, total_distinct_cells=n_cells
    )
    kpi_yield_stats.select("frac_contributing_cells").summary().show()

    valid_kpi_candidates = kpi_coverage.prefilter_kpis(kpi_yield_stats)

    if VERBOSE_DIAGNOSTICS:
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
        "good_windows_candidates (post prefilter)", good_windows_candidates, _GROUPING_COLS
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

    scaler = scaling.SimpleMinMaxScaler(
        value_col="kpi_value", group_cols=["kpi_id", *_GROUPING_COLS]
    )
    params_df = scaler.fit(pm_df_long_imputed)
    pm_df_long_scaled = scaler.transform(pm_df_long_imputed)

    # Greedy joint KPI selection
    selected_kpis = kpi_coverage.greedy_joint_kpi_selection(
        good_windows_candidates,
        valid_kpi_candidates,
        _GROUPING_COLS,
        min_joint_windows_abs=preprocessing_cfg.min_joint_windows_abs,
        forced_kpis=preprocessing_cfg.forced_kpis,
    )
    logger.info(f"Selected {len(selected_kpis)} KPIs from {len(valid_kpi_candidates)} candidates.")

    # joint_anchor_pairs holds start_time as STRING — cast back before the range join

    # joint_anchors_df = sdm.spark.createDataFrame(
    #     anchor_rows, ["distname", "start_time"]
    # ).withColumn("start_time", f.col("start_time").cast("timestamp"))
    # joint_anchors_df = joint_anchors_df.cache()
    # print(joint_anchors_df.count())
    # joint_anchors_df.groupby("distname").agg(f.count_distinct("start_time")).show()
    # joint_anchors_df.groupBy("distname").agg(f.count_distinct("start_time").alias("ccc")).agg(
    #     f.mean("ccc")
    # ).show()
    # print(joint_anchors_df.groupby("distname").agg(f.count_distinct("start_time")).count())

    # POST GREEDY good_windows_cleaned (pre-greedy, post-contiguity)
    # joint_anchors_df.select("kpi_id").distinct().count()  # you reported 485
    # joint_anchors_df.select("distname").distinct().count()  # you reported 396
    # joint_anchors_df.select("distname", "start_time").distinct().count()

    # joint_anchors_df has only (distname, start_time).  attach_windows_index_to_pm
    # does .select("distname", "start_time") on its good_windows argument and discards
    # everything else — so this minimal DataFrame is exactly what it needs.
    #
    # good_windows_selected below is still needed for flag_flat_series_pre_pelt, which
    # wants per-(distname, kpi_id) pairs.  Keep it, but pass joint_anchors_df (not
    # good_windows_selected) to attach_windows_index_to_pm and remove the
    # filter_joint_complete_windows call further down.
    good_windows_selected = good_windows_candidates.filter(f.col("kpi_id").isin(selected_kpis))
    pm_df_long_scaled = pm_df_long_scaled.filter(f.col("kpi_id").isin(selected_kpis))
    _log_pm_diag("pm_df_long_imputed_selected (post greedy)", pm_df_long_scaled, _GROUPING_COLS)
    _log_window_diag("good_windows_selected (post greedy)", good_windows_selected, _GROUPING_COLS)

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
    _log_window_diag("after filter_joint_complete_windows", good_windows_selected, _GROUPING_COLS)

    # This function indexes window_start in pm data
    pm_df_long_indexed_winds = kpi_coverage.emit_window_index(
        pm_df_long_scaled,
        good_windows_selected,
        _GROUPING_COLS,
        window_hours=preprocessing_cfg.window_width_hours,
    )
    if VERBOSE_DIAGNOSTICS:
        validate_windowed_pm(pm_df_long_indexed_winds)

    # PM long pivot to wide
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

    # Save preprocessed data

    dataset_paths_and_dfs: dict[str, DataFrame] = {
        "pm_df_wide_materialized_windows": pm_df_wide_materialized_windows,
        "pm_df_long_indexed_winds": pm_df_long_indexed_winds,
        "pm_df_wide_indexed_winds": pm_df_wide_indexed_windows,
        "scaling_params_df": params_df,
        "pm_data_const_kpi": pm_df_const_kpi,
        "kpi_definitions": kpi_definitions,
        "simple_reports": simple_reports_pivoted,
    }

    # END OF PREPROCESSING

    # SAVE PREPROCESSED DATA

    for df_path, df in dataset_paths_and_dfs.items():
        sdm.write_parquet(
            df,
            "/".join([preprocessing_cfg.output_path_prefix, df_path]),
            mode="overwrite",
        )
