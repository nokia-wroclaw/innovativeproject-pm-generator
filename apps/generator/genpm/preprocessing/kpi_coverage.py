from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from genpm.utils.logger import get_logger

logger = get_logger()

"""
kpi_window_selection.py
=======================
Full pipeline for selecting the largest jointly-valid KPI subset for TimeVAE training,
and materialising the filtered long-format training dataset.

Pipeline stages
---------------
1.  align_cell_time_ranges           — define a per-cell canonical hourly time axis as the
                                       UNION of all (distname, kpi_id) series ranges, then
                                       reindex every series onto that axis, null-filling gaps.
2.  compute_window_density           — sliding-window density via Window.rowsBetween; marks
                                       each (distname, kpi_id, window_start) as valid/invalid.
2b. filter_max_gap                   — NEW: rejects windows where any single null run exceeds
                                       max_gap_hours, regardless of overall density.
                                       Runs on the aligned spine before anchor reduction.
3.  discard_invalid_windows          — drops rows where is_good_window == 0.
4.  compute_theoretical_max_windows  — per-(distname, kpi_id) upper bound on window count
                                       derived from the KPI's own active (non-null) range only.
5.  compute_kpi_yield_stats          — aggregates per-KPI statistics including
                                       window_coverage_frac (observed / theoretical max).
5b. filter_temporal_stability        — NEW: rejects KPIs whose good windows are concentrated
                                       in fewer than min_weeks_with_good_windows distinct weeks.
5c. filter_variance                  — NEW: rejects KPIs with near-zero coefficient of variation
                                       or near-constant zero values (no signal for the AE).
5d. filter_cross_cell_consistency    — NEW: rejects KPIs whose per-cell median distribution
                                       spans an implausibly wide range (vendor/config split).
6.  prefilter_kpis                   — drops structurally bad KPIs on coverage and cell-breadth
                                       fraction criteria; sorts by window_coverage_frac.
7.  greedy_joint_kpi_selection       — greedily builds the largest KPI set whose joint window
                                       count stays above a data-relative floor.
8.  extract_valid_training_data      — range semi-join: returns only aligned rows that fall
                                       inside at least one valid window for the selected KPIs.
                                       Window expansion is deferred to the data loader.

Input schema (raw long-format DataFrame)
-----------------------------------------
    start_time  : timestamp  — hourly
    kpi_id      : string
    kpi_value   : double
    bts_id      : string     — parent of distname
    distname    : string     — cell identifier

Output schema (filtered long-format DataFrame)
-----------------------------------------------
    start_time  : timestamp  — every hour inside at least one valid window
    kpi_id      : string     — selected KPIs only
    kpi_value   : double
    bts_id      : string
    distname    : string

Window boundaries are NOT materialised here.  Pass good_windows_selected
(also returned) to the data loader for per-batch anchor expansion.
"""


# ---------------------------------------------------------------------------
# Stage 1 – Cell-level time-range alignment
# ---------------------------------------------------------------------------


def align_cell_time_ranges(
    df: DataFrame,
    *,
    freq_hours: int = 1,
) -> DataFrame:
    """Define a canonical hourly time axis per cell and reindex every KPI onto it.

    Motivation
    ----------
    KPIs within the same cell may have been provisioned at very different dates
    (standard deviation of ~20–40 days is common).  A trimming approach based on
    a consensus range would discard large fractions of valid data on a short
    3-month horizon.

    Instead, we use the UNION of all per-(distname, kpi_id) active ranges as the
    cell's canonical axis — the widest possible span — and null-fill each KPI
    outside its own active period.  Pre-provisioning and post-decommission periods
    are represented as nulls, which the downstream density filter handles correctly.

    Parameters
    ----------
    df : DataFrame
        Raw long-format input: (start_time, kpi_id, kpi_value, bts_id, distname).
    freq_hours : int
        Expected time step in hours (default 1).

    Returns
    -------
    DataFrame
        Same schema as input.  Every (distname, kpi_id) pair has exactly one row
        per hour in [cell_tmin, cell_tmax].  kpi_value is null where no measurement
        existed.
    """
    # Step 1: per-(distname, kpi_id) active range from non-null observations
    series_endpoints = (
        df.filter(f.col("kpi_value").isNotNull())
        .groupBy("distname", "kpi_id")
        .agg(
            f.min("start_time").alias("kpi_tmin"),
            f.max("start_time").alias("kpi_tmax"),
        )
    )

    # Step 2: cell-level union range — widest span across all KPIs in cell
    cell_axis = series_endpoints.groupBy("distname").agg(
        f.min("kpi_tmin").alias("cell_tmin"),
        f.max("kpi_tmax").alias("cell_tmax"),
    )

    # Step 3: generate canonical hourly timestamp sequence per cell
    interval_expr = f.expr(f"INTERVAL {freq_hours} HOURS")
    cell_timestamps = cell_axis.withColumn(
        "ts",
        f.explode(f.sequence(f.col("cell_tmin"), f.col("cell_tmax"), interval_expr)),
    ).select("distname", f.col("ts").alias("start_time"))

    # Step 4: cross-join canonical timestamps × all (distname, kpi_id, bts_id) pairs
    cell_kpi_pairs = df.select("distname", "kpi_id", "bts_id").distinct()
    spine = cell_timestamps.join(cell_kpi_pairs, on="distname", how="inner")

    # Step 5: left-join actual measurements onto the spine
    aligned = spine.join(
        df.select("distname", "kpi_id", "start_time", "kpi_value"),
        on=["distname", "kpi_id", "start_time"],
        how="left",
    ).select("start_time", "kpi_id", "kpi_value", "bts_id", "distname")

    return aligned


# ---------------------------------------------------------------------------
# Stage 2 – Sliding-window density computation
# ---------------------------------------------------------------------------


def compute_window_density(
    df: DataFrame,
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
    density_threshold: float = 0.917,
) -> DataFrame:
    """Compute per-anchor sliding-window density and flag valid windows.

    Parameters
    ----------
    df : DataFrame
        Output of align_cell_time_ranges.
    window_hours : int
        Window width in hours (default 168 = 1 week).
    stride_hours : int
        Stride between window anchors in hours (default 24 = 1 day).
    density_threshold : float
        Minimum non-null fraction for a window to be marked valid (default 0.917,
        allowing at most ~14 consecutive null hours per window).

    Returns
    -------
    DataFrame
        Schema: (bts_id, distname, kpi_id, start_time, window_valid_frac, is_good_window).
        One row per stride-aligned anchor within the KPI's active range.
    """
    cell_origin_spec = Window.partitionBy("distname")
    series_end_spec = Window.partitionBy("distname", "kpi_id")

    with_meta = (
        df.withColumn(
            "cell_origin_epoch",
            f.min(f.unix_timestamp("start_time")).over(cell_origin_spec),
        )
        .withColumn(
            "series_end_epoch",
            f.max(f.unix_timestamp("start_time")).over(series_end_spec),
        )
        .withColumn(
            "hour_offset",
            (f.unix_timestamp("start_time") - f.col("cell_origin_epoch")) / 3600,
        )
        .withColumn(
            "window_end_epoch",
            f.unix_timestamp("start_time") + f.lit((window_hours - 1) * 3600),
        )
    )

    # Rolling non-null count over the full hourly series
    w_spec = (
        Window.partitionBy("distname", "kpi_id")
        .orderBy("start_time")
        .rowsBetween(0, window_hours - 1)
    )

    with_density = (
        with_meta.withColumn(
            "non_null_indicator",
            f.when(f.col("kpi_value").isNotNull(), 1).otherwise(0),
        )
        .withColumn(
            "non_null_count",
            f.sum("non_null_indicator").over(w_spec),
        )
        .withColumn(
            "window_valid_frac",
            f.col("non_null_count") / f.lit(window_hours),
        )
        .withColumn(
            "is_good_window",
            f.when(f.col("window_valid_frac") >= density_threshold, 1).otherwise(0),
        )
    )

    # Apply stride and tail filters after density is computed
    result = (
        with_density.filter((f.col("hour_offset") % stride_hours) == 0)
        .filter(f.col("window_end_epoch") <= f.col("series_end_epoch"))
        .select(
            "bts_id",
            "distname",
            "kpi_id",
            "start_time",
            "window_valid_frac",
            "is_good_window",
        )
    )

    return result


# ---------------------------------------------------------------------------
# Stage 2b – Max-gap filter (NEW)
# ---------------------------------------------------------------------------


def filter_max_gap(
    aligned_df: DataFrame,
    good_windows: DataFrame,
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
    max_gap_hours: int = 12,
) -> DataFrame:
    """Reject windows that contain a single null run longer than max_gap_hours.

    Motivation
    ----------
    density_threshold is blind to gap shape.  Two windows with identical density
    can have very different null structure:

        Window A: ████░░████░░████░░████  (density=0.86, max_gap=2h  — safe)
        Window B: ████████████░░░░░░░░░░  (density=0.86, max_gap=24h — destroys
                                           an entire night period)

    Both pass density_threshold=0.875 but Window B has a structural hole that
    will corrupt the autoencoder's day/night cycle reconstruction.

    Algorithm
    ---------
    For each (distname, kpi_id) series on the aligned spine:
      1. Label each hour as null (1) or not (0).
      2. Detect null-run boundaries using lag comparison.
      3. Assign a monotonic run_id to each null run via cumulative sum.
      4. For every stride-aligned anchor, compute the maximum null-run length
         whose run starts inside the window [anchor, anchor + window_hours - 1].
      5. Mark the anchor as bad if max_null_run > max_gap_hours.

    Note on run attribution
    -----------------------
    A null run is attributed to the window containing its *start* hour.  This
    avoids double-counting runs that straddle two adjacent windows and is
    consistent with how the density rolling window is anchored.

    Parameters
    ----------
    aligned_df : DataFrame
        Output of align_cell_time_ranges — full hourly spine with kpi_value.
    good_windows : DataFrame
        Output of discard_invalid_windows — density-passing anchors only.
        Schema: (bts_id, distname, kpi_id, start_time, window_valid_frac,
                 is_good_window).
    window_hours : int
        Window width in hours (default 168).  Must match compute_window_density.
    stride_hours : int
        Stride in hours (default 24).  Used only to cross-check anchor alignment;
        the actual anchor set comes from good_windows.
    max_gap_hours : int
        Maximum tolerable consecutive null hours per window (default 12).
        Rationale: 12 h keeps any blind spot within a single half-day period,
        safe for linear interpolation and preserving day/night cycles.

    Returns
    -------
    DataFrame
        Same schema as good_windows, with windows whose max null run exceeds
        max_gap_hours removed.
    """
    lag_w = Window.partitionBy("distname", "kpi_id").orderBy("start_time")

    # Step 1–3: assign a monotonic run_id to each consecutive null sequence
    with_run_ids = (
        aligned_df.withColumn("is_null", f.col("kpi_value").isNull().cast("int"))
        .withColumn("prev_is_null", f.lag("is_null", 1, 0).over(lag_w))
        .withColumn(
            # A new null run starts when current row is null and previous was not
            "null_run_start",
            f.when((f.col("is_null") == 1) & (f.col("prev_is_null") == 0), 1).otherwise(0),
        )
        .withColumn(
            "run_id",
            f.sum("null_run_start").over(lag_w.rowsBetween(Window.unboundedPreceding, 0)),
        )
    )

    # Step 4: compute the length of each null run and tag the run-start epoch
    # so we can attribute the run to the window containing its start hour.
    null_runs = (
        with_run_ids.filter(f.col("is_null") == 1)
        .groupBy("distname", "kpi_id", "run_id")
        .agg(
            f.count("*").alias("run_length"),
            f.min(f.unix_timestamp("start_time")).alias("run_start_epoch"),
        )
    )

    # Step 5: for each good-window anchor, find the worst null run that starts
    # inside [anchor_epoch, anchor_epoch + (window_hours - 1) * 3600].
    anchors = good_windows.select(
        "distname",
        "kpi_id",
        f.unix_timestamp("start_time").alias("anchor_epoch"),
        "start_time",  # keep original for final join key
        "bts_id",
        "window_valid_frac",
        "is_good_window",
    )

    window_end_offset = (window_hours - 1) * 3600

    worst_gap_per_anchor = (
        anchors.join(null_runs, on=["distname", "kpi_id"], how="left")
        .filter(
            (f.col("run_start_epoch") >= f.col("anchor_epoch"))
            & (f.col("run_start_epoch") <= f.col("anchor_epoch") + f.lit(window_end_offset))
        )
        .groupBy(
            "distname",
            "kpi_id",
            "anchor_epoch",
            "start_time",
            "bts_id",
            "window_valid_frac",
            "is_good_window",
        )
        .agg(f.max("run_length").alias("max_null_run"))
    )

    # Anchors with no null rows at all won't appear after the null_runs join —
    # recover them by filling max_null_run = 0 via a left join from anchors.
    with_gap = anchors.join(
        worst_gap_per_anchor.select("distname", "kpi_id", "start_time", "max_null_run"),
        on=["distname", "kpi_id", "start_time"],
        how="left",
    ).fillna({"max_null_run": 0})

    return with_gap.filter(f.col("max_null_run") <= max_gap_hours).select(
        "bts_id", "distname", "kpi_id", "start_time", "window_valid_frac", "is_good_window"
    )


# ---------------------------------------------------------------------------
# Stage 3 – Discard invalid windows
# ---------------------------------------------------------------------------


def discard_invalid_windows(
    window_density: DataFrame,
) -> DataFrame:
    """Drop windows that did not meet the density threshold.

    Parameters
    ----------
    window_density : DataFrame
        Output of compute_window_density with column is_good_window.

    Returns
    -------
    DataFrame
        Same schema, is_good_window == 1 rows only.
    """
    return window_density.filter(f.col("is_good_window") == 1)


# ---------------------------------------------------------------------------
# Stage 4 – Theoretical maximum window count per (distname, kpi_id)
# ---------------------------------------------------------------------------


def compute_theoretical_max_windows(
    aligned_df: DataFrame,
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
) -> DataFrame:
    """Compute the upper-bound window count per (distname, kpi_id) at perfect density.

    Uses only the KPI's own active (non-null) range — NOT the padded cell spine —
    to avoid overestimating theoretical max for late-provisioned or early-decommissioned
    KPIs.  Using count("*") on the aligned spine inflates series_hours by including
    null-padded rows, producing observed/theoretical ratios above 100%.

    Formula
    -------
        active_hours    = (kpi_tmax - kpi_tmin) in hours + 1  (both endpoints inclusive)
        theoretical_max = max(0, floor((active_hours - window_hours) / stride_hours) + 1)

    Parameters
    ----------
    aligned_df : DataFrame
        Output of align_cell_time_ranges.
    window_hours : int
        Window width in hours (default 168).
    stride_hours : int
        Stride between anchors in hours (default 24).

    Returns
    -------
    DataFrame
        Schema: (distname, kpi_id, active_hours, theoretical_max_windows).
    """
    return (
        aligned_df.filter(f.col("kpi_value").isNotNull())
        .groupBy("distname", "kpi_id")
        .agg(
            f.min("start_time").alias("kpi_tmin"),
            f.max("start_time").alias("kpi_tmax"),
        )
        .withColumn(
            "active_hours",
            (f.unix_timestamp("kpi_tmax") - f.unix_timestamp("kpi_tmin")) / 3600 + 1,
        )
        .withColumn(
            "theoretical_max_windows",
            f.greatest(
                f.lit(0),
                ((f.col("active_hours") - f.lit(window_hours)) / f.lit(stride_hours)).cast("long")
                + f.lit(1),
            ),
        )
        .select("distname", "kpi_id", "active_hours", "theoretical_max_windows")
    )


# ---------------------------------------------------------------------------
# Stage 5 – Per-KPI yield statistics
# ---------------------------------------------------------------------------


def compute_kpi_yield_stats(
    good_windows: DataFrame,
    theoretical_max: DataFrame,
    *,
    total_distinct_cells: int,
) -> DataFrame:
    """Aggregate per-KPI statistics needed by the pre-filter.

    Parameters
    ----------
    good_windows : DataFrame
        Long-format good windows (is_good_window == 1).
        Schema must contain at minimum (distname, kpi_id, start_time).
    theoretical_max : DataFrame
        Output of compute_theoretical_max_windows.
        Schema: (distname, kpi_id, active_hours, theoretical_max_windows).
    total_distinct_cells : int
        Total number of distinct distnames in the dataset.

    Returns
    -------
    DataFrame
        One row per kpi_id with columns:
        kpi_id, total_windows, theoretical_max_windows, window_coverage_frac,
        n_cells, frac_contributing_cells, mean_windows_per_cell.
    """
    kpi_theoretical_max = theoretical_max.groupBy("kpi_id").agg(
        f.sum("theoretical_max_windows").alias("theoretical_max_windows")
    )

    observed = (
        good_windows.groupBy("kpi_id")
        .agg(
            f.count("*").alias("total_windows"),
            f.countDistinct("distname").alias("n_cells"),
        )
        .withColumn(
            "frac_contributing_cells",
            f.col("n_cells") / f.lit(total_distinct_cells),
        )
        .withColumn(
            "mean_windows_per_cell",
            f.col("total_windows") / f.col("n_cells"),
        )
    )

    stats = observed.join(kpi_theoretical_max, on="kpi_id", how="left").withColumn(
        "window_coverage_frac",
        f.when(
            f.col("theoretical_max_windows") > 0,
            f.col("total_windows") / f.col("theoretical_max_windows"),
        ).otherwise(f.lit(0.0)),
    )

    return stats


# ---------------------------------------------------------------------------
# Stage 5b – Temporal stability filter (NEW)
# ---------------------------------------------------------------------------


def filter_temporal_stability(
    good_windows: DataFrame,
    *,
    min_weeks_with_good_windows: int = 8,
    total_weeks_in_dataset: int,
    min_frac_weeks_covered: float = 0.60,
) -> list[str]:
    """Reject KPIs whose good windows are concentrated in too few distinct weeks.

    Motivation
    ----------
    A KPI may pass coverage checks globally yet have all its valid windows packed
    into a single month.  The autoencoder would then fail to see the full weekly
    periodicity of the dataset and would not generalise across time.

    Both an absolute floor (min_weeks_with_good_windows) and a fractional floor
    (min_frac_weeks_covered × total_weeks_in_dataset) must be satisfied — the
    stricter of the two wins per KPI.

    Parameters
    ----------
    good_windows : DataFrame
        Density-passing (and max-gap-passing) anchor DataFrame.
        Schema must contain at minimum (kpi_id, start_time).
    min_weeks_with_good_windows : int
        Absolute minimum number of distinct ISO weeks with at least one good
        window (default 8 — covers two full monthly cycles on a 3-month horizon).
    total_weeks_in_dataset : int
        Total number of distinct ISO weeks present in the dataset.  Pass as a
        pre-computed scalar to avoid an extra full-scan inside this function.
    min_frac_weeks_covered : float
        Fractional floor: KPI must have good windows in at least this fraction
        of all weeks in the dataset (default 0.60).

    Returns
    -------
    list[str]
        KPI IDs that pass the temporal stability filter.
    """
    effective_week_floor = max(
        min_weeks_with_good_windows,
        int(min_frac_weeks_covered * total_weeks_in_dataset),
    )

    return (
        good_windows.withColumn("week", f.date_trunc("week", "start_time"))
        .groupBy("kpi_id")
        .agg(f.countDistinct("week").alias("n_weeks_with_good_windows"))
        .filter(f.col("n_weeks_with_good_windows") >= effective_week_floor)
        .select("kpi_id")
        .rdd.flatMap(lambda r: [r["kpi_id"]])
        .collect()
    )


# ---------------------------------------------------------------------------
# Stage 5c – Variance filter (NEW)
# ---------------------------------------------------------------------------


def filter_variance(
    aligned_df: DataFrame,
    good_windows: DataFrame,
    *,
    min_cv: float = 0.01,
    max_zero_frac: float = 0.95,
) -> list[str]:
    """Reject KPIs with near-zero variance or near-constant zero values.

    Motivation
    ----------
    A KPI that is always zero or always the same constant is structurally valid
    (passes all density checks) but carries no signal for the autoencoder.  It
    will dominate reconstruction loss without contributing learnable structure,
    and may cause gradient collapse in the bottleneck layer.

    Two complementary criteria:
    - Coefficient of variation (std / |mean|) guards scale-independent flatness.
      min_cv = 0.01 means std must be at least 1% of the mean magnitude.
    - Zero fraction guards constant-zero KPIs where mean ≈ 0 makes CV undefined.
      max_zero_frac = 0.95 rejects KPIs that are zero more than 95% of the time.

    Only values inside valid (density + max-gap passing) windows are considered,
    so the filter reflects signal quality in the actual training set.

    Parameters
    ----------
    aligned_df : DataFrame
        Output of align_cell_time_ranges — full hourly spine with kpi_value.
    good_windows : DataFrame
        Density- and gap-passing anchor DataFrame.
        Schema must contain at minimum (distname, kpi_id).
    min_cv : float
        Minimum coefficient of variation floor (default 0.01).
    max_zero_frac : float
        Maximum fraction of values that may be exactly zero (default 0.95).

    Returns
    -------
    list[str]
        KPI IDs that pass the variance filter.
    """
    # Restrict to values observed inside valid windows only
    valid_values = aligned_df.join(
        good_windows.select("distname", "kpi_id").distinct(),
        on=["distname", "kpi_id"],
        how="inner",
    ).filter(f.col("kpi_value").isNotNull())

    stats = (
        valid_values.groupBy("kpi_id")
        .agg(
            f.mean("kpi_value").alias("mean_val"),
            f.stddev("kpi_value").alias("std_val"),
            (f.sum(f.when(f.col("kpi_value") == 0, 1).otherwise(0)) / f.count("*")).alias(
                "zero_frac"
            ),
        )
        .withColumn(
            "cv",
            f.when(
                f.col("mean_val") != 0,
                f.abs(f.col("std_val") / f.col("mean_val")),
            ).otherwise(f.lit(0.0)),
        )
    )

    return (
        stats.filter(f.col("zero_frac") <= max_zero_frac)
        .filter(f.col("cv") >= min_cv)
        .select("kpi_id")
        .rdd.flatMap(lambda r: [r["kpi_id"]])
        .collect()
    )


# ---------------------------------------------------------------------------
# Stage 5d – Cross-cell consistency filter (NEW)
# ---------------------------------------------------------------------------


def filter_cross_cell_consistency(
    aligned_df: DataFrame,
    good_windows: DataFrame,
    *,
    max_iqr_ratio: float = 5.0,
) -> list[str]:
    """Reject KPIs whose per-cell median distribution spans an implausible range.

    Motivation
    ----------
    A KPI may be healthy on 80% of cells but operate in a completely different
    value range on the remaining 20% (different hardware vendor, software version,
    or misconfiguration).  This creates a multi-modal distribution in the training
    set that the autoencoder cannot reconcile — it will learn a blurred average
    of two incompatible regimes.

    Metric
    ------
    For each KPI, compute the median kpi_value per cell, then measure the spread
    of those per-cell medians via the IQR ratio:

        iqr_ratio = p75(cell_medians) / p25(cell_medians)

    A ratio of 1.0 means all cells have the same median; a ratio of 5.0 means
    the 75th-percentile cell has a median five times larger than the 25th.
    Values above max_iqr_ratio suggest a structural split between cell populations.

    Edge cases
    ----------
    - If p25 == 0 (many cells idle), iqr_ratio is set to 999 to force rejection.
      These KPIs are better handled by the variance filter (zero_frac check).
    - Only values inside valid windows are used, consistent with the variance filter.

    Parameters
    ----------
    aligned_df : DataFrame
        Output of align_cell_time_ranges — full hourly spine with kpi_value.
    good_windows : DataFrame
        Density- and gap-passing anchor DataFrame.
        Schema must contain at minimum (distname, kpi_id).
    max_iqr_ratio : float
        Maximum tolerable ratio of p75 to p25 of per-cell medians (default 5.0).

    Returns
    -------
    list[str]
        KPI IDs that pass the cross-cell consistency filter.
    """
    valid_values = aligned_df.join(
        good_windows.select("distname", "kpi_id").distinct(),
        on=["distname", "kpi_id"],
        how="inner",
    ).filter(f.col("kpi_value").isNotNull())

    cell_medians = valid_values.groupBy("kpi_id", "distname").agg(
        f.expr("percentile(kpi_value, 0.50)").alias("cell_median")
    )

    consistency = (
        cell_medians.groupBy("kpi_id")
        .agg(
            f.expr("percentile(cell_median, 0.25)").alias("p25"),
            f.expr("percentile(cell_median, 0.75)").alias("p75"),
        )
        .withColumn(
            "iqr_ratio",
            f.when(
                f.col("p25") > 0,
                f.col("p75") / f.col("p25"),
            ).otherwise(f.lit(999.0)),
        )
    )

    return (
        consistency.filter(f.col("iqr_ratio") <= max_iqr_ratio)
        .select("kpi_id")
        .rdd.flatMap(lambda r: [r["kpi_id"]])
        .collect()
    )


# ---------------------------------------------------------------------------
# Stage 6 – Pre-filtering
# ---------------------------------------------------------------------------


def prefilter_kpis(
    kpi_yield_stats: DataFrame,
    *,
    min_window_coverage_frac: float = 0.50,
    min_frac_contributing_cells: float = 0.50,
) -> list[str]:
    """Apply structural filters and return surviving KPI list sorted by coverage.

    Filters (a KPI must pass both):
    1. window_coverage_frac >= min_window_coverage_frac
    2. frac_contributing_cells >= min_frac_contributing_cells

    Parameters
    ----------
    kpi_yield_stats : DataFrame
        Output of compute_kpi_yield_stats.
    min_window_coverage_frac : float
        Minimum fraction of theoretical max windows (default 0.50).
    min_frac_contributing_cells : float
        Minimum fraction of cells with >=1 good window (default 0.50).

    Returns
    -------
    list[str]
        KPI IDs passing all filters, sorted by window_coverage_frac descending.
    """
    surviving = (
        kpi_yield_stats.filter(f.col("window_coverage_frac") >= min_window_coverage_frac)
        .filter(f.col("frac_contributing_cells") >= min_frac_contributing_cells)
        .orderBy(f.desc("window_coverage_frac"))
        .select("kpi_id")
        .rdd.flatMap(lambda r: [r["kpi_id"]])
        .collect()
    )

    return surviving


# ---------------------------------------------------------------------------
# Stage 7 – Greedy joint KPI selection
# ---------------------------------------------------------------------------


def greedy_joint_kpi_selection(
    good_windows: DataFrame,
    candidates: list[str],
    theoretical_max_joint: int,
    *,
    min_joint_coverage_frac: float = 0.90,
    min_joint_windows_abs: int = 10_000,
) -> list[str]:
    """Greedily build the largest KPI set whose joint window count stays above the floor.

    The effective floor is max(fraction-based, absolute):
        floor = max(
            int(min_joint_coverage_frac × theoretical_max_joint),
            min_joint_windows_abs,
        )

    Parameters
    ----------
    good_windows : DataFrame
        Cached good-windows DataFrame (candidates only), schema:
        (bts_id, distname, kpi_id, start_time).
    candidates : list[str]
        KPI IDs ordered by window_coverage_frac descending (output of prefilter_kpis).
    theoretical_max_joint : int
        Exact theoretical joint maximum from compute_joint_theoretical_max.
    min_joint_coverage_frac : float
        Fraction of theoretical_max_joint the joint count must meet (default 0.90).
        High value (0.90) means each added KPI may reduce joint windows by at most 10%,
        strongly favouring KPIs with broad, overlapping temporal coverage.
    min_joint_windows_abs : int
        Hard absolute floor regardless of fraction (default 10_000).

    Returns
    -------
    list[str]
        Accepted KPI IDs in the order they were added.
    """
    min_joint_windows = max(
        int(min_joint_coverage_frac * theoretical_max_joint),
        min_joint_windows_abs,
    )
    logger.info(
        f"[greedy] floor = max("
        f"{min_joint_coverage_frac:.0%} × {theoretical_max_joint:,}, "
        f"{min_joint_windows_abs:,}) "
        f"= {min_joint_windows:,}"
    )

    selected: list[str] = []

    for idx, kpi in enumerate(candidates):
        tentative = selected + [kpi]
        n = len(tentative)

        joint_count = (
            good_windows.filter(f.col("kpi_id").isin(tentative))
            .groupBy("distname", "start_time")
            .agg(f.count("*").alias("n_good_kpis"))
            .filter(f.col("n_good_kpis") == n)
            .count()
        )

        if joint_count >= min_joint_windows:
            selected.append(kpi)
            logger.info(
                f"[greedy] step {idx + 1:>4d} | accepted '{kpi}' "
                f"| selected={len(selected):>4d} "
                f"| joint_windows={joint_count:,} "
                f"({joint_count / theoretical_max_joint:.1%} of theoretical max)"
            )
        else:
            logger.info(
                f"[greedy] step {idx + 1:>4d} | SKIPPED  '{kpi}' "
                f"| joint_windows={joint_count:,} < {min_joint_windows:,}"
            )

    return selected


# ---------------------------------------------------------------------------
# Stage 8 – Extract valid training data
# ---------------------------------------------------------------------------


def extract_valid_pm_windows(
    aligned_df: DataFrame,
    good_windows: DataFrame,
    selected_kpis: list[str],
    *,
    window_hours: int = 168,
) -> DataFrame:
    """Return aligned rows that fall inside at least one valid window.

    Uses a range semi-join rather than window explosion to avoid materialising
    the 7× overlap factor (window_hours / stride_hours) at this stage.
    Window boundaries are deferred to the data loader, which receives the
    good_windows_selected DataFrame alongside training_data.

    Parameters
    ----------
    aligned_df : DataFrame
        Output of align_cell_time_ranges — full hourly spine with kpi_value.
    good_windows : DataFrame
        All-filters-passing anchor DataFrame restricted to selected KPIs.
        Schema: (bts_id, distname, kpi_id, start_time).
    selected_kpis : list[str]
        Final KPI list from greedy_joint_kpi_selection.
    window_hours : int
        Window width in hours — must match compute_window_density.

    Returns
    -------
    DataFrame
        Long-format, schema: (start_time, kpi_id, kpi_value, bts_id, distname).
        Every row falls inside at least one density- and gap-passing window for
        a selected KPI.  No window_start / window_end columns — the caller joins
        good_windows_selected for per-batch grouping.
    """
    anchors = good_windows.filter(f.col("kpi_id").isin(selected_kpis)).select(
        "distname",
        "kpi_id",
        f.col("start_time").alias("anchor_start"),
        (f.unix_timestamp("start_time") + f.lit((window_hours - 1) * 3600)).alias(
            "anchor_end_epoch"
        ),
    )

    covered = (
        aligned_df.filter(f.col("kpi_id").isin(selected_kpis))
        .join(anchors, on=["distname", "kpi_id"], how="inner")
        .filter(
            (f.unix_timestamp("start_time") >= f.unix_timestamp("anchor_start"))
            & (f.unix_timestamp("start_time") <= f.col("anchor_end_epoch"))
        )
        .select("start_time", "kpi_id", "kpi_value", "bts_id", "distname")
        .distinct()  # deduplicate hours covered by multiple overlapping windows
    )

    return covered


# ---------------------------------------------------------------------------
# Exact joint theoretical max (replaces bottleneck-KPI heuristic)
# ---------------------------------------------------------------------------


def compute_joint_theoretical_max(
    aligned_df: DataFrame,
    candidates: list[str],
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
) -> int:
    """Exact theoretical max joint windows via per-cell anchor range intersection.

    For a joint window at (distname, t) to be theoretically possible, every
    candidate KPI must have an active range that accommodates a full window at t:
        t >= max over kpi_id of (kpi_tmin)
        t <= min over kpi_id of (kpi_tmax) - (window_hours - 1)h

    The number of stride-aligned anchors in that intersection is computed
    analytically — no row counting, no pivot.

    Parameters
    ----------
    aligned_df : DataFrame
        Output of align_cell_time_ranges.
    candidates : list[str]
        KPI IDs to include in the joint computation.
    window_hours : int
        Window width in hours (default 168).
    stride_hours : int
        Stride in hours (default 24).

    Returns
    -------
    int
        Total exact theoretical joint window count across all cells.
    """
    series_bounds = (
        aligned_df.filter(f.col("kpi_id").isin(candidates))
        .filter(f.col("kpi_value").isNotNull())
        .groupBy("distname", "kpi_id")
        .agg(
            f.min("start_time").alias("kpi_tmin"),
            f.max("start_time").alias("kpi_tmax"),
        )
    )

    cell_joint_range = series_bounds.groupBy("distname").agg(
        f.unix_timestamp(f.max("kpi_tmin")).alias("joint_anchor_start_epoch"),
        (f.unix_timestamp(f.min("kpi_tmax")) - f.lit((window_hours - 1) * 3600)).alias(
            "joint_anchor_end_epoch"
        ),
    )

    with_counts = cell_joint_range.withColumn(
        "span_seconds",
        f.col("joint_anchor_end_epoch") - f.col("joint_anchor_start_epoch"),
    ).withColumn(
        "cell_joint_max",
        f.greatest(
            f.lit(0),
            (f.col("span_seconds") / f.lit(stride_hours * 3600)).cast("long") + f.lit(1),
        ),
    )

    total = with_counts.agg(f.sum("cell_joint_max")).collect()[0][0]
    return int(total or 0)


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------


def pm_data_kpi_coverage(
    pm_df: DataFrame,
    *,
    # Stage 2 – window density
    window_hours: int = 168,
    stride_hours: int = 24,
    density_threshold: float = 0.917,
    # Stage 2b – max-gap filter
    max_gap_hours: int = 12,
    # Stage 5b – temporal stability
    min_weeks_with_good_windows: int = 8,
    min_frac_weeks_covered: float = 0.60,
    # Stage 5c – variance
    min_cv: float = 0.01,
    max_zero_frac: float = 0.95,
    # Stage 5d – cross-cell consistency
    max_iqr_ratio: float = 5.0,
    # Stage 6 – pre-filter
    min_window_coverage_frac: float = 0.50,
    min_frac_contributing_cells: float = 0.50,
    # Stage 7 – greedy selection
    min_joint_coverage_frac: float = 0.90,
    min_joint_windows_abs: int = 10_000,
) -> tuple[list[str], DataFrame, DataFrame]:
    """Execute the full KPI selection pipeline end-to-end.

    Returns the selected KPI list, the filtered long-format training DataFrame,
    and the good-windows anchor DataFrame (needed by the data loader for
    per-batch window grouping).

    Parameters
    ----------
    raw_df : DataFrame
        Raw long-format data: (start_time, kpi_id, kpi_value, bts_id, distname).
    freq_hours : int
        Hourly granularity of the canonical cell time axis (default 1).
    window_hours : int
        Sliding window width in hours (default 168 = 1 week).
    stride_hours : int
        Stride between window anchors in hours (default 24 = 1 day).
    density_threshold : float
        Minimum non-null fraction per window (default 0.917 ≈ max 14h gap).
    max_gap_hours : int
        Maximum consecutive null hours allowed inside a single window (default 12).
    min_weeks_with_good_windows : int
        Temporal stability: absolute minimum distinct weeks with good windows
        (default 8).
    min_frac_weeks_covered : float
        Temporal stability: fractional minimum of dataset weeks covered (default 0.60).
    min_cv : float
        Variance filter: minimum coefficient of variation (default 0.01).
    max_zero_frac : float
        Variance filter: maximum fraction of values that may be zero (default 0.95).
    max_iqr_ratio : float
        Consistency filter: maximum p75/p25 ratio of per-cell medians (default 5.0).
    min_window_coverage_frac : float
        Pre-filter: KPI must achieve this fraction of its theoretical window max
        (default 0.50).
    min_frac_contributing_cells : float
        Pre-filter: minimum fraction of cells contributing >=1 good window
        (default 0.50).
    min_joint_coverage_frac : float
        Greedy floor: minimum fraction of theoretical joint max (default 0.90).
    min_joint_windows_abs : int
        Greedy floor: hard absolute minimum (default 10_000).

    Returns
    -------
    selected_kpis : list[str]
        Final KPI set accepted by the greedy algorithm.
    training_data : DataFrame
        Filtered long-format DataFrame ready for autoencoder training.
        Schema: (start_time, kpi_id, kpi_value, bts_id, distname).
    good_windows_selected : DataFrame
        Anchor DataFrame for the selected KPIs and valid windows.
        Schema: (bts_id, distname, kpi_id, start_time).
        Pass to the data loader for per-batch window grouping.
    """

    pm_df.cache()
    pm_df.count()
    # ------------------------------------------------------------------
    # Stage 2: compute sliding-window density
    # ------------------------------------------------------------------
    logger.info("Stage 2: computing window density ...")
    window_density = compute_window_density(
        pm_df,
        window_hours=window_hours,
        stride_hours=stride_hours,
        density_threshold=density_threshold,
    )

    # ------------------------------------------------------------------
    # Stage 3: discard density-failing windows
    # ------------------------------------------------------------------
    logger.info("Stage 3: discarding density-failing windows ...")
    good_windows_density = discard_invalid_windows(window_density)

    # ------------------------------------------------------------------
    # Stage 2b: max-gap filter — runs after density discard so the gap
    # computation only touches anchors that already passed density.
    # ------------------------------------------------------------------
    logger.info(f"Stage 2b: applying max-gap filter (max_gap_hours={max_gap_hours}) ...")
    good_windows_all = filter_max_gap(
        pm_df,
        good_windows_density,
        window_hours=window_hours,
        stride_hours=stride_hours,
        max_gap_hours=max_gap_hours,
    )
    good_windows_all.cache()
    n_after_gap = good_windows_all.count()
    logger.info(f"  {n_after_gap:,} windows remain after max-gap filter.")

    # ------------------------------------------------------------------
    # Stage 4: compute theoretical maximum windows per (distname, kpi_id)
    # ------------------------------------------------------------------
    logger.info("Stage 4: computing theoretical window maxima ...")
    theoretical_max = compute_theoretical_max_windows(
        pm_df,
        window_hours=window_hours,
        stride_hours=stride_hours,
    )

    # ------------------------------------------------------------------
    # Stage 5: compute per-KPI yield statistics
    # ------------------------------------------------------------------
    logger.info("Stage 5: computing per-KPI yield statistics ...")
    total_distinct_cells = pm_df.select("distname").distinct().count()

    kpi_stats = compute_kpi_yield_stats(
        good_windows_all,
        theoretical_max,
        total_distinct_cells=total_distinct_cells,
    )

    # ------------------------------------------------------------------
    # Stage 5b: temporal stability filter
    # ------------------------------------------------------------------
    logger.info("Stage 5b: applying temporal stability filter ...")
    total_weeks_in_dataset = (
        pm_df.select(f.date_trunc("week", "start_time").alias("week")).distinct().count()
    )
    stable_kpis = filter_temporal_stability(
        good_windows_all,
        min_weeks_with_good_windows=min_weeks_with_good_windows,
        total_weeks_in_dataset=total_weeks_in_dataset,
        min_frac_weeks_covered=min_frac_weeks_covered,
    )
    logger.info(f"  {len(stable_kpis)} KPIs pass temporal stability.")

    # ------------------------------------------------------------------
    # Stage 5c: variance filter
    # ------------------------------------------------------------------
    logger.info("Stage 5c: applying variance filter ...")
    variant_kpis = filter_variance(
        pm_df,
        good_windows_all,
        min_cv=min_cv,
        max_zero_frac=max_zero_frac,
    )
    logger.info(f"  {len(variant_kpis)} KPIs pass variance filter.")

    # ------------------------------------------------------------------
    # Stage 5d: cross-cell consistency filter
    # ------------------------------------------------------------------
    logger.info("Stage 5d: applying cross-cell consistency filter ...")
    consistent_kpis = filter_cross_cell_consistency(
        pm_df,
        good_windows_all,
        max_iqr_ratio=max_iqr_ratio,
    )
    logger.info(f"  {len(consistent_kpis)} KPIs pass consistency filter.")

    # ------------------------------------------------------------------
    # Intersect all Stage 5 filter survivors before pre-filter.
    # kpi_stats carries coverage/cell fractions; restrict it to the KPIs
    # that passed all three quality filters.
    # ------------------------------------------------------------------
    quality_survivors = set(stable_kpis) & set(variant_kpis) & set(consistent_kpis)
    logger.info(f"  {len(quality_survivors)} KPIs survive all Stage-5 quality filters.")

    kpi_stats_filtered = kpi_stats.filter(f.col("kpi_id").isin(list(quality_survivors)))

    # ------------------------------------------------------------------
    # Stage 6: pre-filter on coverage and cell-breadth fractions
    # ------------------------------------------------------------------
    logger.info("Stage 6: pre-filtering KPIs ...")
    candidates = prefilter_kpis(
        kpi_stats_filtered,
        min_window_coverage_frac=min_window_coverage_frac,
        min_frac_contributing_cells=min_frac_contributing_cells,
    )
    logger.info(f"  {len(candidates)} candidates passed pre-filter.")

    # ------------------------------------------------------------------
    # Exact theoretical_max_joint via anchor range intersection
    # ------------------------------------------------------------------
    theoretical_max_joint = compute_joint_theoretical_max(
        pm_df,
        candidates,
        window_hours=window_hours,
        stride_hours=stride_hours,
    )
    logger.info(f"  theoretical_max_joint (exact) = {theoretical_max_joint:,}")

    # Build cached greedy-loop DataFrame: candidates only, density cols dropped
    good_windows_candidates = good_windows_all.filter(f.col("kpi_id").isin(candidates)).drop(
        "window_valid_frac", "is_good_window"
    )

    good_windows_all.unpersist()
    good_windows_candidates.cache()
    good_windows_candidates.count()

    # ------------------------------------------------------------------
    # Stage 7: greedy joint KPI selection
    # ------------------------------------------------------------------
    logger.info("Stage 7: running greedy joint KPI selection ...")
    selected_kpis = greedy_joint_kpi_selection(
        good_windows_candidates,
        candidates,
        theoretical_max_joint,
        min_joint_coverage_frac=min_joint_coverage_frac,
        min_joint_windows_abs=min_joint_windows_abs,
    )
    logger.info(f"  Selected {len(selected_kpis)} KPIs from {len(candidates)} candidates.")

    # ------------------------------------------------------------------
    # Stage 8: extract valid training data (range semi-join, no explosion)
    # ------------------------------------------------------------------
    logger.info("Stage 8: extracting valid training data ...")
    good_windows_selected = good_windows_candidates.filter(f.col("kpi_id").isin(selected_kpis))

    valid_pm_windows_df = extract_valid_pm_windows(
        pm_df,
        good_windows_selected,
        selected_kpis,
        window_hours=window_hours,
    )

    # Release intermediates; caller owns training_data and good_windows_selected
    good_windows_candidates.unpersist()
    pm_df.unpersist()

    logger.info("Done.")
    logger.info(f"  training_data schema : {valid_pm_windows_df.columns}")
    logger.info(f"  good_windows schema  : {good_windows_selected.columns}")

    return selected_kpis, valid_pm_windows_df, good_windows_selected


# ---------------------------------------------------------------------------
# Usage example (not executed on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from genpm.preprocessing.kpi_coverage import pm_data_kpi_coverage
    from genpm.utils.consts import SHARED_DIR_PATH, SPARK_CONFIGS
    from genpm.utils.utils import SparkDataManager

    sdm = SparkDataManager(SPARK_CONFIGS["HALF_SAFE"])

    PREPROCESSED_DATASET_PATH = SHARED_DIR_PATH / "preprocessed_dataset"
    pm_df = sdm.read_parquet(PREPROCESSED_DATASET_PATH / "pm_data_long")

    selected_kpis, training_data, good_windows_selected = pm_data_kpi_coverage(
        pm_df,
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

    logger.info("Selected KPIs:", selected_kpis)

    # Persist and inspect
    training_data.cache()
    logger.info("Training rows :", training_data.count())
    logger.info("Anchor windows:", good_windows_selected.count())
    training_data.show(5)

    # Write training data and anchors for the model training job
    sdm.write_parquet(
        training_data,
        SHARED_DIR_PATH / "training_data" / "kpi_windows_long",
    )
    sdm.write_parquet(
        good_windows_selected,
        SHARED_DIR_PATH / "training_data" / "kpi_window_anchors",
    )
