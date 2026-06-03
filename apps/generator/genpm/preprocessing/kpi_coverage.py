import numpy as np
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f
from pyspark.sql.types import LongType

from genpm.utils.consts import MAX_IMPUTABLE_GAP
from genpm.utils.logger import get_logger

logger = get_logger()

"""
kpi_window_selection.py  (refactored — sparse pipeline)
========================================================
Full pipeline for selecting the largest jointly-valid KPI subset for TimeVAE
training, and materialising the filtered long-format training dataset.

Key change vs. the original
----------------------------
The old ``allign_kpis_in_distname`` extended every (distname, kpi_id) series
to the **distname-wide** time envelope, producing a massive dense grid
(≈ 8 760 h × 25 000 combos per distname) that was mostly NaN.  All downstream
stages (density, max-gap, variance …) then ran on this grid.

The refactored pipeline removes that cross-join:

*  **Stage 1** fills gaps only inside each KPI's *own* [min_t, max_t] range.
   A KPI that lives for 3 000 hours never sees the other 5 760 hours of null
   padding that the old code generated.

*  **Stage 2** computes window density via an *explode-into-anchors* approach:
   every data row is mapped to the ≤ 7 stride-aligned window anchors that
   contain it (W/S = 168/24 = 7), then a simple ``groupBy`` + ``count`` gives
   non-null density.  No ``rowsBetween`` on a pre-densified grid is needed.

*  **Stage 2b** detects the longest null run per window from the per-KPI
   gap-filled spine **plus** an explicit leading-gap term for windows whose
   anchor precedes the KPI's first timestamp.

*  **Stage 8** performs a *deferred densification*: the full hourly grid is
   materialised only for the final selected KPIs and valid windows, which is
   orders of magnitude smaller than the upfront cross-join.

Everything else (stages 3–7) operates on window metadata or on non-null values
only, so it works identically on the sparser input.

Pipeline stages
---------------
1.  fill_internal_gaps               — per-(distname, kpi_id) hourly spine
                                       covering [kpi_min_t, kpi_max_t] only;
                                       left-joins actual values.
2.  compute_window_density_sparse    — explode each row into its ≤ 7 window
                                       memberships; count non-nulls per anchor.
2b. filter_max_gap_sparse            — null-run detection on the per-KPI spine
                                       plus leading-gap awareness.
3.  discard_invalid_windows          — (unchanged) is_good_window == 1.
4.  compute_theoretical_max_windows  — (unchanged) uses non-null values only.
5.  compute_kpi_yield_stats          — (unchanged).
5b. filter_temporal_stability        — (unchanged).
5c. filter_variance                  — (unchanged).
5d. filter_cross_cell_consistency    — (unchanged).
6.  prefilter_kpis                   — (unchanged).
7.  greedy_joint_kpi_selection       — (unchanged).
8.  extract_valid_pm_windows         — deferred densification inside valid
                                       windows only.

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
    kpi_value   : double     — null where data was absent
    bts_id      : string
    distname    : string
"""


# OTHER FILTERS BEFORE IQR OUTLIERS AND IMPUTING
def filter_global_value_density(
    gap_filled_df: DataFrame,
    *,
    min_global_density: float = 0.80,
    min_frac_cells_passing: float = 0.80,
) -> DataFrame:
    """Reject KPIs that are globally sparse across cells.

    Operates per (distname, kpi_id) first, then decides at the KPI level how
    many cells must meet the threshold.  This prevents a KPI that is dense in
    a handful of cells but absent in most from sneaking through.

    Parameters
    ----------
    gap_filled_df : DataFrame
        Output of fill_internal_gaps — per-KPI hourly spine with null rows for
        missing hours.  count(*) over the spine gives the true active-range
        length per series, so no cross-KPI padding inflates the denominator.
    min_global_density : float
        Minimum non-null fraction over a series' own active range for that
        (distname, kpi_id) pair to be considered "dense" (default 0.80).
    min_frac_cells_passing : float
        Fraction of a KPI's contributing cells that must individually meet
        min_global_density for the KPI to pass (default 0.80).

    Returns
    -------
    list[str]
        KPI IDs that pass the global density filter.
    """
    per_series = (
        gap_filled_df.groupBy("kpi_id", "distname")
        .agg(
            f.sum(f.when(f.col("kpi_value").isNotNull(), 1).otherwise(0)).alias("non_null_count"),
            f.count("*").alias("total_hours"),
        )
        .withColumn(
            "global_density",
            f.col("non_null_count") / f.col("total_hours"),
        )
        .withColumn(
            "cell_passes",
            (f.col("global_density") >= min_global_density).cast("int"),
        )
    )

    return (
        per_series.groupBy("kpi_id")
        .agg(f.mean("cell_passes").alias("frac_passing_cells"))
        .filter(f.col("frac_passing_cells") >= min_frac_cells_passing)
        .select("kpi_id")
    )


def filter_gap_pattern(
    gap_filled_df: DataFrame,
    *,
    max_imputable_gap: int = 6,
    min_imputable_gap_frac: float = 0.90,
) -> DataFrame:
    """Reject KPIs whose null-run distribution is dominated by long gaps.

    The original criteria (median gap length + gap frequency) were too strict
    for the imputation-guard role and targeted different failure modes.
    This version asks one question: of all null runs in this KPI, what fraction
    are short enough to impute safely (≤ max_imputable_gap hours)?


    Window A: ████░░████░░████░░████  (density=0.86, max_gap=2h  — safe)
    Window B: ████████████░░░░░░░░░░  (density=0.86, max_gap=24h — destroys
                                        an entire night period)

    A KPI where 95 % of gaps are 1–6 h passes, regardless of how many gaps
    there are or how they are spaced.  A KPI with many long gaps fails even if
    its median is technically short, because imputable_gap_frac penalises any
    run that exceeds the threshold.

    Parameters
    ----------
    gap_filled_df : DataFrame
        Output of fill_internal_gaps.
    max_imputable_gap : int
        Maximum null-run length in hours that is safe to impute (default 6).
    min_imputable_gap_frac : float
        Minimum fraction of all null runs that must be ≤ max_imputable_gap for
        the KPI to pass (default 0.90).

    Returns
    -------
    list[str]
        KPI IDs that pass the gap pattern filter.
    """
    lag_w = Window.partitionBy("distname", "kpi_id").orderBy("start_time")

    with_runs = (
        gap_filled_df.withColumn("is_null", f.col("kpi_value").isNull().cast("int"))
        .withColumn("prev_is_null", f.lag("is_null", 1, 0).over(lag_w))
        .withColumn(
            "null_run_start",
            f.when((f.col("is_null") == 1) & (f.col("prev_is_null") == 0), 1).otherwise(0),
        )
        .withColumn(
            "run_id",
            f.sum("null_run_start").over(lag_w.rowsBetween(Window.unboundedPreceding, 0)),
        )
    )

    null_run_lengths = (
        with_runs.filter(f.col("is_null") == 1)
        .groupBy("kpi_id", "distname", "run_id")
        .agg(f.count("*").alias("run_length"))
    )

    return (
        null_run_lengths.groupBy("kpi_id")
        .agg(
            f.mean(f.when(f.col("run_length") <= max_imputable_gap, 1).otherwise(0)).alias(
                "imputable_gap_frac"
            ),
        )
        .filter(f.col("imputable_gap_frac") >= min_imputable_gap_frac)
        .select("kpi_id")
    )


def flag_periodic_gaps(
    gap_filled_df: DataFrame,
    *,
    max_imputable_gap: int = 6,
    recurrence_threshold: float = 0.50,
    min_occurrences: int = 4,
) -> DataFrame:
    """Detect (distname, kpi_id) pairs where short null runs recur systematically.

    A gap that appears at the same hour-of-week in ≥ recurrence_threshold of
    all weeks is considered periodic — imputing it would fabricate a structural
    pattern rather than filling incidental noise.

    Only null runs of length ≤ max_imputable_gap are examined.  Long gaps are
    already handled by filter_gap_pattern and Stage 2b; the concern here is
    short gaps that look imputable in isolation but are really systematic.

    Note: returns (distname, kpi_id) pairs rather than a KPI-level list.
    The same KPI may be periodic in some cells but clean in others; the caller
    should anti-join this result against the data rather than dropping the
    whole KPI.

    Parameters
    ----------
    gap_filled_df : DataFrame
        Output of fill_internal_gaps.
    max_imputable_gap : int
        Only null runs of this length or shorter are checked for periodicity.
    recurrence_threshold : float
        A gap hour-of-week slot is flagged if it appears in this fraction of
        the series' total weeks (default 0.50 — at least every other week).
    min_occurrences : int
        Minimum number of distinct weeks a periodic gap must appear in before
        it is flagged (guards against short series, default 4).

    Returns
    -------
    DataFrame
        Schema: (distname, kpi_id) — pairs to EXCLUDE from imputation.
        Caller should anti-join this against the data before imputing.
    """
    lag_w = Window.partitionBy("distname", "kpi_id").orderBy("start_time")

    null_run_starts = (
        gap_filled_df.withColumn("is_null", f.col("kpi_value").isNull().cast("int"))
        .withColumn("prev_is_null", f.lag("is_null", 1, 0).over(lag_w))
        .withColumn(
            "run_id",
            f.sum(
                f.when((f.col("is_null") == 1) & (f.col("prev_is_null") == 0), 1).otherwise(0)
            ).over(lag_w.rowsBetween(Window.unboundedPreceding, 0)),
        )
    )

    run_stats = (
        null_run_starts.filter(f.col("is_null") == 1)
        .groupBy("distname", "kpi_id", "run_id")
        .agg(
            f.count("*").alias("run_length"),
            f.min("start_time").alias("run_start"),
        )
        # Only short gaps are candidates for periodic imputation
        .filter(f.col("run_length") <= max_imputable_gap)
        .withColumn(
            "hour_of_week",
            f.dayofweek("run_start") * 24 + f.hour("run_start"),
        )
        .withColumn("week", f.date_trunc("week", "run_start"))
    )

    # Total weeks each series is active — denominator for recurrence fraction
    total_weeks = gap_filled_df.groupBy("distname", "kpi_id").agg(
        f.countDistinct(f.date_trunc("week", "start_time")).alias("total_weeks")
    )

    recurrence = (
        run_stats.groupBy("distname", "kpi_id", "hour_of_week")
        .agg(f.countDistinct("week").alias("weeks_with_gap"))
        .join(total_weeks, on=["distname", "kpi_id"])
        .withColumn(
            "recurrence_frac",
            f.col("weeks_with_gap") / f.col("total_weeks"),
        )
        .filter(f.col("weeks_with_gap") >= min_occurrences)
        .filter(f.col("recurrence_frac") >= recurrence_threshold)
    )

    return recurrence.select("distname", "kpi_id").distinct()


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 – Per-KPI internal gap filling  (replaces allign_kpis_in_distname)
# ═══════════════════════════════════════════════════════════════════════════


def fill_internal_gaps(
    df: DataFrame,
    time_col: str = "start_time",
    freq_hours: int = 1,
) -> DataFrame:
    """Create an hourly spine *per (distname, kpi_id, bts_id)* and left-join values.

    Unlike the old ``allign_kpis_in_distname`` this does **not** extend any
    series to the distname-wide envelope.  Only hours between the KPI's own
    earliest and latest timestamps are generated, so internal gaps are
    null-filled but no cross-KPI padding is created.

    Complexity comparison (example: 500 KPIs, distname span 8 760 h,
    average KPI span 4 000 h):

        Old cross-join:  500 × 8 760  = 4 380 000 rows/distname
        New per-KPI:     500 × 4 000  = 2 000 000 rows/distname  (−54 %)

    For KPIs with heterogeneous lifetimes the savings are even larger.

    Parameters
    ----------
    df : DataFrame
        Raw long-format data.
    time_col : str
        Timestamp column (default ``"start_time"``).
    freq_hours : int
        Grid resolution in hours (default 1).

    Returns
    -------
    DataFrame
        Schema identical to *df* but with null-valued rows inserted for
        every missing hour inside each series' own [min_t, max_t].
    """
    interval_expr = f"INTERVAL {freq_hours} HOUR"

    df = df.repartition("distname", "kpi_id")

    # Per-series bounds — NOT distname-wide
    series_bounds = df.groupBy("distname", "kpi_id", "bts_id").agg(
        f.min(time_col).alias("min_t"),
        f.max(time_col).alias("max_t"),
    )

    # Hourly spine per series
    series_spine = series_bounds.withColumn(
        time_col,
        f.explode(f.sequence(f.col("min_t"), f.col("max_t"), f.expr(interval_expr))),
    ).drop("min_t", "max_t")

    # Left-join actual values onto the per-series spine
    return series_spine.join(df, on=["distname", "kpi_id", "bts_id", time_col], how="left")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 – Sparse sliding-window density  (replaces compute_window_density)
# ═══════════════════════════════════════════════════════════════════════════


def build_pm_windows_anchor_df(
    df: DataFrame,
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
):
    """
    Creates a dataframe of **training windows** for every distname (cell)

    1. Broadcast the distname-level time origin (earliest timestamp across all
       KPIs in the distname).  This is the reference for stride-aligned anchors.

    2. For every data row compute the hourly offset from the distname origin.

    3. Determine the ≤ W/S stride-aligned anchor indices whose window
       [anchor, anchor + W) contains this row, and ``explode`` them.
       With W = 168 and S = 24 each row maps to at most 7 anchors.

    """

    n_overlap = window_hours // stride_hours  # 7 for W=168, S=24
    window_end_offset_s = (window_hours - 1) * 3600  # seconds

    # ── distname-level origin (broadcast-safe) ──────────────────────────
    distname_origin = df.groupBy("distname").agg(
        f.min(f.unix_timestamp("start_time")).alias("dist_origin_epoch"),
    )

    base = (
        df.join(f.broadcast(distname_origin), on="distname")
        .withColumn("row_epoch", f.unix_timestamp("start_time"))
        .withColumn(
            "offset_h",
            ((f.col("row_epoch") - f.col("dist_origin_epoch")) / 3600).cast("long"),
        )
    )

    # ── explode into anchor memberships ────────────────────────────────
    #
    # A row at offset_h belongs to anchors k where
    #     k × stride  ≤  offset_h  <  k × stride + window_hours
    # ⇒   k  ∈  [ max(0, ⌊(offset_h − W + 1)/S⌋ + 1) …  ⌊offset_h/S⌋ ]
    #
    # Simplified: k ∈ [ max(0, max_k − (n_overlap − 1)) …  max_k ]
    with_anchors = (
        base.withColumn("max_k", f.floor(f.col("offset_h") / f.lit(stride_hours)).cast("long"))
        .withColumn(
            "min_k",
            f.greatest(f.lit(0).cast("long"), f.col("max_k") - f.lit(n_overlap - 1)),
        )
        .withColumn("anchor_k", f.explode(f.sequence(f.col("min_k"), f.col("max_k"))))
        .withColumn(
            "anchor_epoch",
            f.col("dist_origin_epoch") + f.col("anchor_k") * f.lit(stride_hours * 3600),
        )
    )

    return with_anchors


def compute_window_density_sparse(
    df: DataFrame,
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
    density_threshold: float = 0.917,
) -> DataFrame:
    """Compute per-anchor window density **without** a distname-wide dense grid.

    Algorithm
    ---------
    1. Broadcast the distname-level time origin (earliest timestamp across all
       KPIs in the distname).  This is the reference for stride-aligned anchors.

    2. For every data row compute the hourly offset from the distname origin.

    3. Determine the ≤ W/S stride-aligned anchor indices whose window
       [anchor, anchor + W) contains this row, and ``explode`` them.
       With W = 168 and S = 24 each row maps to at most 7 anchors.

    4. ``groupBy(distname, kpi_id, anchor)`` → count non-null values.

    5. ``density = non_null_count / window_hours``.

    The 7× row expansion is far cheaper than the full cross-join densification
    (which created one row per hour per *distname* for every KPI).

    Tail filter
    -----------
    Anchors whose window would extend past the **KPI's own** series-end are
    discarded.  This is stricter than the old code (which used the padded
    distname-end) but correct: those trailing windows had density ≈ 0 anyway.

    Parameters
    ----------
    df : DataFrame
        Per-KPI gap-filled data from ``fill_internal_gaps``, **or** raw data
        (both work — nulls from gap-fill are correctly excluded from non-null
        counts, and absent rows are inherently excluded).
    window_hours : int
        Window width in hours (default 168 = 1 week).
    stride_hours : int
        Stride between window anchors in hours (default 24 = 1 day).
    density_threshold : float
        Minimum non-null fraction for a window to be marked valid.

    Returns
    -------
    DataFrame
        Schema: (bts_id, distname, kpi_id, start_time, window_valid_frac,
                 is_good_window).
        One row per stride-aligned anchor that fits inside the KPI's active
        range.
    """
    n_overlap = window_hours // stride_hours  # 7 for W=168, S=24
    window_end_offset_s = (window_hours - 1) * 3600  # seconds

    # ── distname-level origin (broadcast-safe) ──────────────────────────
    distname_origin = df.groupBy("distname").agg(
        f.min(f.unix_timestamp("start_time")).alias("dist_origin_epoch"),
    )

    # ── per-(distname, kpi_id) series end for tail filter ───────────────
    series_end = (
        df.filter(f.col("kpi_value").isNotNull())
        .groupBy("distname", "kpi_id")
        .agg(
            f.max(f.unix_timestamp("start_time")).alias("series_end_epoch"),
        )
    )

    # ── bts_id lookup (one bts_id per distname × kpi_id) ───────────────
    bts_lookup = df.select("distname", "kpi_id", "bts_id").distinct()

    # ── attach origin & compute offset ─────────────────────────────────
    base = (
        df.join(f.broadcast(distname_origin), on="distname")
        .withColumn("row_epoch", f.unix_timestamp("start_time"))
        .withColumn(
            "offset_h",
            ((f.col("row_epoch") - f.col("dist_origin_epoch")) / 3600).cast("long"),
        )
    )

    # ── explode into anchor memberships ────────────────────────────────
    #
    # A row at offset_h belongs to anchors k where
    #     k × stride  ≤  offset_h  <  k × stride + window_hours
    # ⇒   k  ∈  [ max(0, ⌊(offset_h − W + 1)/S⌋ + 1) …  ⌊offset_h/S⌋ ]
    #
    # Simplified: k ∈ [ max(0, max_k − (n_overlap − 1)) …  max_k ]
    with_anchors = (
        base.withColumn("max_k", f.floor(f.col("offset_h") / f.lit(stride_hours)).cast("long"))
        .withColumn(
            "min_k",
            f.greatest(f.lit(0).cast("long"), f.col("max_k") - f.lit(n_overlap - 1)),
        )
        .withColumn("anchor_k", f.explode(f.sequence(f.col("min_k"), f.col("max_k"))))
        .withColumn(
            "anchor_epoch",
            f.col("dist_origin_epoch") + f.col("anchor_k") * f.lit(stride_hours * 3600),
        )
    )

    # ── aggregate density per (distname, kpi_id, anchor) ───────────────
    window_stats = (
        with_anchors.groupBy("distname", "kpi_id", "anchor_epoch")
        .agg(
            f.sum(f.when(f.col("kpi_value").isNotNull(), 1).otherwise(0)).alias("non_null_count"),
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

    # ── tail filter + attach metadata ──────────────────────────────────
    result = (
        window_stats.join(series_end, on=["distname", "kpi_id"])
        .filter(f.col("anchor_epoch") + f.lit(window_end_offset_s) <= f.col("series_end_epoch"))
        .drop("series_end_epoch", "non_null_count")
        .join(bts_lookup, on=["distname", "kpi_id"])
        .withColumn(
            "start_time",
            f.from_unixtime(f.col("anchor_epoch")).cast("timestamp"),
        )
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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2b – Max-gap filter (sparse-aware)
# ═══════════════════════════════════════════════════════════════════════════


def filter_max_gap_sparse(
    gap_filled_df: DataFrame,
    good_windows: DataFrame,
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
    max_gap_hours: int = MAX_IMPUTABLE_GAP,
) -> DataFrame:
    """Reject windows containing a single null run longer than *max_gap_hours*.

    Compared with the original ``filter_max_gap`` this version adds
    **leading-gap awareness**: because the per-KPI spine no longer extends to
    the distname-wide origin, a window whose anchor precedes the KPI's first
    timestamp has an implicit null run at the front that was previously
    materialised as explicit null rows.

        leading_gap_h = max(0, (kpi_start_epoch − anchor_epoch) / 3600)

    The effective max null run for each window is then::

        max(leading_gap_h, worst_internal_null_run)

    The trailing edge does not need special treatment because the tail filter
    in ``compute_window_density_sparse`` already discards anchors whose window
    extends past the KPI's series end.

    Parameters
    ----------
    gap_filled_df : DataFrame
        Output of ``fill_internal_gaps`` — per-KPI hourly spine.
    good_windows : DataFrame
        Density-passing anchors (output of ``discard_invalid_windows``).
    window_hours : int
        Window width in hours (must match density stage).
    stride_hours : int
        Stride in hours (informational only; anchors come from *good_windows*).
    max_gap_hours : int
        Maximum tolerable consecutive null hours per window.

    Returns
    -------
    DataFrame
        Same schema as *good_windows*, with offending windows removed.
    """
    lag_w = Window.partitionBy("distname", "kpi_id").orderBy("start_time")

    # ── step 1-3: null-run detection on the per-KPI spine ──────────────
    with_run_ids = (
        gap_filled_df.withColumn("is_null", f.col("kpi_value").isNull().cast("int"))
        .withColumn("prev_is_null", f.lag("is_null", 1, 0).over(lag_w))
        .withColumn(
            "null_run_start",
            f.when((f.col("is_null") == 1) & (f.col("prev_is_null") == 0), 1).otherwise(0),
        )
        .withColumn(
            "run_id",
            f.sum("null_run_start").over(lag_w.rowsBetween(Window.unboundedPreceding, 0)),
        )
    )

    null_runs = (
        with_run_ids.filter(f.col("is_null") == 1)
        .groupBy("distname", "kpi_id", "run_id")
        .agg(
            f.count("*").alias("run_length"),
            f.min(f.unix_timestamp("start_time")).alias("run_start_epoch"),
        )
    )

    # ── step 4: per-KPI series start for leading-gap computation ───────
    kpi_starts = gap_filled_df.groupBy("distname", "kpi_id").agg(
        f.min(f.unix_timestamp("start_time")).alias("kpi_start_epoch")
    )

    # ── step 5: for each anchor, compute worst internal null run ───────
    anchors = good_windows.select(
        "distname",
        "kpi_id",
        f.unix_timestamp("start_time").alias("anchor_epoch"),
        "start_time",
        "bts_id",
        "window_valid_frac",
        "is_good_window",
    )

    window_end_offset = (window_hours - 1) * 3600

    worst_internal = (
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
        .agg(f.max("run_length").alias("max_internal_gap"))
    )

    # ── step 6: combine internal + leading gap ─────────────────────────
    with_gaps = (
        anchors.join(
            worst_internal.select("distname", "kpi_id", "start_time", "max_internal_gap"),
            on=["distname", "kpi_id", "start_time"],
            how="left",
        )
        .join(kpi_starts, on=["distname", "kpi_id"], how="left")
        .fillna({"max_internal_gap": 0})
        .withColumn(
            "leading_gap",
            f.greatest(
                f.lit(0),
                ((f.col("kpi_start_epoch") - f.col("anchor_epoch")) / 3600).cast("long"),
            ),
        )
        .withColumn(
            "max_null_run",
            f.greatest(f.col("leading_gap"), f.col("max_internal_gap")),
        )
    )

    return with_gaps.filter(f.col("max_null_run") <= max_gap_hours).select(
        "bts_id",
        "distname",
        "kpi_id",
        "start_time",
        "window_valid_frac",
        "is_good_window",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 – Discard invalid windows  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


def discard_invalid_windows(
    window_density: DataFrame,
) -> DataFrame:
    """Drop windows that did not meet the density threshold."""
    return window_density.filter(f.col("is_good_window") == 1)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 – Theoretical maximum window count  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


def compute_theoretical_max_windows(
    df: DataFrame,
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
) -> DataFrame:
    """Compute the upper-bound window count per (distname, kpi_id).

    Uses only non-null rows — works identically on per-KPI gap-filled data
    and the old distname-wide grid (gap-fill rows are null, filtered out here).
    """
    return (
        df.filter(f.col("kpi_value").isNotNull())
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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 – Per-KPI yield statistics  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


def compute_kpi_yield_stats(
    good_windows: DataFrame,
    theoretical_max: DataFrame,
    *,
    total_distinct_cells: int,
) -> DataFrame:
    """Aggregate per-KPI statistics needed by the pre-filter."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5b – Temporal stability filter  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


# DROP THIS SHITE
def filter_temporal_stability(
    good_windows: DataFrame,
    *,
    min_weeks_with_good_windows: int = 8,
    total_weeks_in_dataset: int,
    min_frac_weeks_covered: float = 0.60,
) -> list[str]:
    """Reject KPIs whose good windows are concentrated in too few weeks."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5c – Variance filter  (unchanged — operates on non-null values)
# ═══════════════════════════════════════════════════════════════════════════


def filter_variance(
    aligned_df: DataFrame,
    good_windows: DataFrame,
    *,
    min_std_val: float = 0.01,
    max_zero_frac: float = 0.95,
) -> list[str]:
    """Reject KPIs with near-zero variance or near-constant zero values."""
    valid_values = aligned_df.join(
        good_windows.select("distname", "kpi_id").distinct(),
        on=["distname", "kpi_id"],
        how="inner",
    ).filter(f.col("kpi_value").isNotNull())

    stats = valid_values.groupBy("kpi_id").agg(
        f.mean("kpi_value").alias("mean_val"),
        f.stddev("kpi_value").alias("std_val"),
        (f.sum(f.when(f.col("kpi_value") == 0, 1).otherwise(0)) / f.count("*")).alias("zero_frac"),
    )

    return (
        stats.filter(f.col("zero_frac") <= max_zero_frac)
        .filter(f.col("std_val") >= min_std_val)
        .select("kpi_id")
        .rdd.flatMap(lambda r: [r["kpi_id"]])
        .collect()
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5d – Cross-cell consistency filter  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


def filter_cross_cell_consistency(
    aligned_df: DataFrame,
    good_windows: DataFrame,
    *,
    max_iqr_ratio: float = 5.0,
) -> list[str]:
    """Reject KPIs whose per-cell median distribution spans an implausible range."""
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
            f.when(f.col("p25") > 0, f.col("p75") / f.col("p25")).otherwise(f.lit(999.0)),
        )
    )

    return (
        consistency.filter(f.col("iqr_ratio") <= max_iqr_ratio)
        .select("kpi_id")
        .rdd.flatMap(lambda r: [r["kpi_id"]])
        .collect()
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6 – Pre-filtering  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


def prefilter_kpis(
    kpi_yield_stats: DataFrame,
    *,
    min_window_coverage_frac: float = 0.50,
    min_frac_contributing_cells: float = 0.50,
) -> list[str]:
    """Apply structural filters and return surviving KPI list sorted by coverage."""
    surviving = (
        kpi_yield_stats.filter(f.col("window_coverage_frac") >= min_window_coverage_frac)
        .filter(f.col("frac_contributing_cells") >= min_frac_contributing_cells)
        .orderBy(f.desc("window_coverage_frac"))
        .select("kpi_id")
        .rdd.flatMap(lambda r: [r["kpi_id"]])
        .collect()
    )
    return surviving


# ═══════════════════════════════════════════════════════════════════════════
# Stage 7 – Greedy joint KPI selection  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def greedy_coverage_curve(
    good_windows: DataFrame,
    candidates: list[str],
) -> list[dict]:
    """Run greedy selection with no floor, recording the full coverage curve.
    Returns a list of dicts with step, kpi_id, joint_windows for elbow analysis.
    """
    # Same one-scan upfront collection as before
    rows = (
        good_windows.withColumn("start_time", f.col("start_time").cast("string"))
        .groupBy("kpi_id", "distname")
        .agg(f.collect_set("start_time").alias("anchors"))
        .collect()
    )

    kpi_coverage: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        pairs = {(row["distname"], anchor) for anchor in row["anchors"]}
        if row["kpi_id"] not in kpi_coverage:
            kpi_coverage[row["kpi_id"]] = set()
        kpi_coverage[row["kpi_id"]] |= pairs

    available = [k for k in candidates if k in kpi_coverage]
    best_seed = max(available, key=lambda k: len(kpi_coverage[k]))
    selected = [best_seed]
    current_intersection = kpi_coverage[best_seed].copy()
    available.remove(best_seed)

    curve = [{"step": 1, "kpi_id": best_seed, "joint_windows": len(current_intersection)}]

    step = 2
    while available:
        best_kpi, best_next, best_count = None, None, -1
        for kpi in available:
            tentative = current_intersection & kpi_coverage[kpi]
            count = len(tentative)
            if count > best_count:
                best_count, best_kpi, best_next = count, kpi, tentative

        selected.append(best_kpi)
        current_intersection = best_next
        available.remove(best_kpi)
        curve.append({"step": step, "kpi_id": best_kpi, "joint_windows": best_count})
        step += 1

    return curve


def find_elbow(curve: list[dict]) -> int:
    """Return the step index at the elbow of the coverage curve."""
    counts = np.array([r["joint_windows"] for r in curve], dtype=float)

    # First derivative: coverage loss per step
    d1 = np.diff(counts)  # negative values — each step loses windows

    # Second derivative: rate of change of loss
    # A large negative value means coverage started dropping much faster
    d2 = np.diff(d1)

    # Elbow = where second derivative is most negative
    # +1 offset because diff reduces length by 1 each time
    elbow_idx = int(np.argmin(d2)) + 1

    return elbow_idx


def suggest_threshold(curve: list[dict], elbow_idx: int) -> int:
    """Return the joint_windows value just before the elbow — conservative threshold."""
    # Use the step before the elbow as the threshold — that's where
    # coverage is still healthy before the cliff
    safe_idx = max(0, elbow_idx - 1)
    return curve[safe_idx]["joint_windows"]


def greedy_joint_kpi_selection(
    good_windows: DataFrame,
    candidates: list[str],
    *,
    min_joint_windows_abs: int | None = None,
) -> list[str]:
    """Greedily build the largest KPI set whose joint window count stays above floor.

    A 'joint window' is a unique (distname, start_time) pair where every
    selected KPI has data. Each such pair represents one training sample.
    """
    if min_joint_windows_abs is None:
        logger.info(
            "[elbow] min_joint_windows_abs not selected - defaulting to elbow method of selection"
        )
        curve = greedy_coverage_curve(good_windows, candidates)

        # Find elbow
        elbow_idx = find_elbow(curve)
        suggested_threshold = suggest_threshold(curve, elbow_idx)
        suggested_n_kpis = elbow_idx  # number of KPIs at the elbow

        logger.info(
            f"[elbow] suggested cutoff: {suggested_n_kpis} KPIs "
            f"| joint_windows at elbow: {suggested_threshold:,}"
        )
        min_joint_windows = suggested_threshold
    else:
        min_joint_windows = min_joint_windows_abs

    logger.info(f"{min_joint_windows=}")

    # Build kpi_id → set of (distname, start_time) anchor pairs in one pass.
    # After this, all greedy logic is pure Python set operations
    rows = (
        good_windows.withColumn("start_time", f.col("start_time").cast("string"))  # safe hashing
        .groupBy("kpi_id", "distname")
        .agg(f.collect_set("start_time").alias("anchors"))
        .collect()
    )

    kpi_coverage: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        pairs = {(row["distname"], anchor) for anchor in row["anchors"]}
        if row["kpi_id"] not in kpi_coverage:
            kpi_coverage[row["kpi_id"]] = set()
        kpi_coverage[row["kpi_id"]] |= pairs

    # We explicitly pick the KPI with the most coverage as the seed.
    available = [k for k in candidates if k in kpi_coverage]
    best_seed = max(available, key=lambda k: len(kpi_coverage[k]))
    selected: list[str] = [best_seed]
    current_intersection: set[tuple[str, str]] = kpi_coverage[best_seed].copy()
    available.remove(best_seed)

    logger.info(f"[greedy] seed '{best_seed}' " f"| coverage={len(current_intersection):,}")

    # ── Bug fix 3: true greedy — pick best candidate at each step ──────────
    # Original code iterated candidates in fixed order, accepting each one
    # that passed the threshold. That is not greedy — it is a sequential
    # filter. True greedy means: at each step, among all remaining candidates,
    # pick the one that loses the fewest windows from current_intersection.
    # This produces a better KPI set for the same coverage floor.
    step = 1
    while available:
        best_kpi: str | None = None
        best_next_intersection: set[tuple[str, str]] | None = None
        best_count = -1

        for kpi in available:
            tentative_intersection = current_intersection & kpi_coverage[kpi]
            count = len(tentative_intersection)
            if count > best_count:
                best_count = count
                best_kpi = kpi
                best_next_intersection = tentative_intersection

        # If even the best candidate drops below floor, stop entirely —
        # no remaining candidate can do better than the least-damaging one.
        if best_count < min_joint_windows:
            logger.info(
                f"[greedy] stopping at step {step} — "
                f"best candidate '{best_kpi}' would reduce coverage "
                f"to {best_count:,} < {min_joint_windows:,}"
            )
            break

        selected.append(best_kpi)
        current_intersection = best_next_intersection
        available.remove(best_kpi)

        logger.info(
            f"[greedy] step {step:>4d} | accepted '{best_kpi}' "
            f"| selected={len(selected):>4d} "
            f"| joint_windows={best_count:,} "
        )
        step += 1

    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Stage 8 – Extract valid training data  (with deferred densification)
# ═══════════════════════════════════════════════════════════════════════════
def attach_windows_index_to_pm(
    pm_df_long_imputed_selected: DataFrame,
    good_windows: DataFrame,
    window_hours: int = 168,
):
    # Step 1: add window_end to good_windows for the range join
    good_windows_with_end = good_windows.withColumn(
        "window_end", f.col("start_time") + f.expr(f"INTERVAL {window_hours} HOURS")
    ).withColumnRenamed("start_time", "window_anchor")

    # Step 2: range join — each hourly row maps to all anchors it falls within
    # Join keys: distname + kpi_id (a row only belongs to windows of its own cell/kpi)
    joined = (
        pm_df_long_imputed_selected.alias("p")
        .join(
            good_windows_with_end.alias("g"),
            on=[
                pm_df_long_imputed_selected.distname == good_windows_with_end.distname,
                pm_df_long_imputed_selected.kpi_id == good_windows_with_end.kpi_id,
                pm_df_long_imputed_selected.start_time >= good_windows_with_end.window_anchor,
                pm_df_long_imputed_selected.start_time < good_windows_with_end.window_end,
            ],
            how="inner",
        )
        .select("p.*", "g.window_anchor")
    )

    # Step 3: compute hour_idx — position within the window
    indexed = joined.withColumn(
        "hour_idx",
        (
            (f.col("start_time").cast(LongType()) - f.col("window_anchor").cast(LongType())) / 3600
        ).cast("integer"),
    ).select(
        "distname",
        "bts_id",
        "kpi_id",
        f.col("window_anchor"),
        "hour_idx",
        "kpi_value",
        "imputed_flag",
    )

    return indexed


def extract_valid_pm_windows(
    pm_df: DataFrame,
    training_windows_anchors: DataFrame,
    *,
    window_hours: int = 168,
) -> DataFrame:
    """Return densified rows that fall inside at least one valid window.

    **Deferred densification** — the full hourly grid is built here, but only
    for the final selected KPIs and valid-window ranges.  This is the only
    point in the pipeline where the cross-join (anchor spine × selected KPIs)
    is materialised.  Because only surviving anchors participate, the output is
    orders of magnitude smaller than the old upfront ``allign_kpis_in_distname``.

    For each valid window anchor the function:

    1. Generates the ``window_hours``-long hourly spine.
    2. Crosses the spine with the selected KPIs (plus their bts_id).
    3. Left-joins actual values from the raw data.
    4. Deduplicates hours covered by multiple overlapping windows.

    Parameters
    ----------
    raw_df : DataFrame
        Raw (or per-KPI gap-filled) long-format data.
    good_windows : DataFrame
        All-filters-passing anchor DataFrame restricted to selected KPIs.
    window_hours : int
        Window width in hours.

    Returns
    -------
    DataFrame
        Long-format, schema: (start_time, kpi_id, kpi_value, bts_id, distname).
        Contains null kpi_value rows where data was absent inside a valid window.
    """
    interval_1h = f.expr("INTERVAL 1 HOUR")
    window_end_expr = f.expr(f"INTERVAL {window_hours - 1} HOUR")

    # ── build per-anchor hourly spines ─────────────────────────────────
    #
    # Distinct (distname, start_time) anchors — KPI-agnostic, because the
    # greedy stage already ensured all selected KPIs share these anchors.
    distinct_anchors = training_windows_anchors.select("distname", "start_time").distinct()

    anchor_spines = distinct_anchors.withColumn(
        "window_time",
        f.explode(
            f.sequence(
                f.col("start_time"),
                f.col("start_time") + window_end_expr,
                interval_1h,
            )
        ),
    )

    # ── cross with selected KPIs ───────────────────────────────────────
    kpi_dims = pm_df.select("distname", "kpi_id", "bts_id").distinct()

    full_grid = anchor_spines.join(kpi_dims, on="distname", how="inner")

    # ── left-join actual values ────────────────────────────────────────
    covered = (
        full_grid.join(
            pm_df.select(
                "distname",
                "kpi_id",
                "bts_id",
                f.col("start_time").alias("window_time"),
                "kpi_value",
            ),
            on=["distname", "kpi_id", "bts_id", "window_time"],
            how="left",
        )
        .select(
            f.col("window_time").alias("start_time"),
            "kpi_id",
            "kpi_value",
            "bts_id",
            "distname",
        )
        .distinct()  # deduplicate overlapping-window hours
    )

    return covered


# ═══════════════════════════════════════════════════════════════════════════
# Exact joint theoretical max  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


def compute_joint_theoretical_max(
    aligned_df: DataFrame,
    candidates: list[str],
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
) -> int:
    """Exact theoretical max joint windows via per-cell anchor range intersection."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════


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

    Returns
    -------
    selected_kpis : list[str]
        Final KPI set accepted by the greedy algorithm.
    training_data : DataFrame
        Filtered long-format DataFrame ready for autoencoder training.
        Densified only within valid windows (nulls where data was absent).
    good_windows_selected : DataFrame
        Anchor DataFrame for the selected KPIs and valid windows.
    """

    pm_df.cache()
    pm_df.count()

    # ------------------------------------------------------------------
    # Stage 1: per-KPI internal gap filling (replaces cross-join align)
    # ------------------------------------------------------------------
    logger.info("Stage 1: filling internal gaps (per-KPI spine) ...")
    gap_filled_df = fill_internal_gaps(pm_df)
    gap_filled_df.cache()
    gap_filled_df.count()
    logger.info("  gap-filled rows computed (per-KPI range only, no cross-join).")

    # ------------------------------------------------------------------
    # Stage 2: sparse window density
    # ------------------------------------------------------------------
    logger.info("Stage 2: computing window density (sparse / explode-into-anchors) ...")
    window_density = compute_window_density_sparse(
        gap_filled_df,
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
    # Stage 2b: max-gap filter with leading-gap awareness
    # ------------------------------------------------------------------
    logger.info(
        f"Stage 2b: applying max-gap filter "
        f"(max_gap_hours={max_gap_hours}, leading-gap aware) ..."
    )
    good_windows_all = filter_max_gap_sparse(
        gap_filled_df,
        good_windows_density,
        window_hours=window_hours,
        stride_hours=stride_hours,
        max_gap_hours=max_gap_hours,
    )
    good_windows_all.cache()
    n_after_gap = good_windows_all.count()
    logger.info(f"  {n_after_gap:,} windows remain after max-gap filter.")

    # ------------------------------------------------------------------
    # Stage 4: theoretical maximum windows per (distname, kpi_id)
    # ------------------------------------------------------------------
    logger.info("Stage 4: computing theoretical window maxima ...")
    theoretical_max = compute_theoretical_max_windows(
        gap_filled_df,
        window_hours=window_hours,
        stride_hours=stride_hours,
    )

    # ------------------------------------------------------------------
    # Stage 5: per-KPI yield statistics
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
    # logger.info("Stage 5b: applying temporal stability filter ...")
    # total_weeks_in_dataset = (
    #     pm_df.select(f.date_trunc("week", "start_time").alias("week")).distinct().count()
    # )
    # stable_kpis = filter_temporal_stability(
    #     good_windows_all,
    #     min_weeks_with_good_windows=min_weeks_with_good_windows,
    #     total_weeks_in_dataset=total_weeks_in_dataset,
    #     min_frac_weeks_covered=min_frac_weeks_covered,
    # )
    # logger.info(f"  {len(stable_kpis)} KPIs pass temporal stability.")

    # ------------------------------------------------------------------
    # Stage 5c: variance filter
    # ------------------------------------------------------------------
    # TODO: FIX
    logger.info("Stage 5c: applying variance filter ...")
    variant_kpis = filter_variance(
        gap_filled_df,
        good_windows_all,
        min_std_val=min_cv,
        max_zero_frac=max_zero_frac,
    )
    logger.info(f"  {len(variant_kpis)} KPIs pass variance filter.")

    # ------------------------------------------------------------------
    # Stage 5d: cross-cell consistency filter
    # ------------------------------------------------------------------

    # MAYBE DROP? LOOSE THRESHOLD
    # logger.info("Stage 5d: applying cross-cell consistency filter ...")
    # consistent_kpis = filter_cross_cell_consistency(
    #     gap_filled_df,
    #     good_windows_all,
    #     max_iqr_ratio=max_iqr_ratio,
    # )
    # logger.info(f"  {len(consistent_kpis)} KPIs pass consistency filter.")

    # ------------------------------------------------------------------
    # Intersect all Stage 5 filter survivors
    # ------------------------------------------------------------------
    quality_survivors = set(variant_kpis)
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
    # Exact theoretical_max_joint
    # ------------------------------------------------------------------
    theoretical_max_joint = compute_joint_theoretical_max(
        gap_filled_df,
        candidates,
        window_hours=window_hours,
        stride_hours=stride_hours,
    )
    logger.info(f"  theoretical_max_joint (exact) = {theoretical_max_joint:,}")

    # Build cached greedy-loop DataFrame: candidates only
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
    # Stage 8: extract + densify valid training data
    # ------------------------------------------------------------------
    logger.info("Stage 8: extracting valid training data (deferred densification) ...")
    good_windows_selected = good_windows_candidates.filter(f.col("kpi_id").isin(selected_kpis))

    valid_pm_windows_df = extract_valid_pm_windows(
        pm_df,
        good_windows_selected,
        selected_kpis,
        window_hours=window_hours,
    )

    # Release intermediates; caller owns training_data and good_windows_selected
    good_windows_candidates.unpersist()
    gap_filled_df.unpersist()
    pm_df.unpersist()

    logger.info("Done.")
    logger.info(f"  training_data schema : {valid_pm_windows_df.columns}")
    logger.info(f"  good_windows schema  : {good_windows_selected.columns}")

    return selected_kpis, valid_pm_windows_df, good_windows_selected


# ═══════════════════════════════════════════════════════════════════════════
# Usage example (not executed on import)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
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
