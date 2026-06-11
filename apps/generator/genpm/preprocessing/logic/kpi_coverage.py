import numpy as np
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from genpm.utils.logger import get_logger

logger = get_logger()

"""
kpi_coverage.py  (refactored — sparse pipeline)
================================================
Full pipeline for selecting the largest jointly-valid KPI subset for TimeVAE
training, and materialising the indexed training dataset.

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

*  **Stage 8** indexes each hourly row by window anchor and hour_idx by joining
   the imputed data against the joint-complete anchor set.  Only selected KPIs
   and valid windows are touched, so the output is orders of magnitude smaller
   than the old upfront cross-join.

Everything else (stages 3–7) operates on window metadata or on non-null values
only, so it works identically on the sparser input.

────────────────────────────────────────────────────────────────────────────
CORRECTNESS FIX (windowed validation failures)
────────────────────────────────────────────────────────────────────────────
Validation on the materialised windows revealed three coupled defects, all
rooted in the same cause: window validity was decided **per (distname, kpi_id)
independently** and **as a fraction**, so

  1. A window could pass at a density of 0.917 while missing its first or last
     few hours — a *clipped edge*.  The anchor grid is stride-aligned to the
     distname origin (earliest ts across all KPIs in the cell), but each KPI
     starts at its own time, so KPIs that start late get a window anchored
     before their first row.  6 missing leading hours out of 168 = 3.6 %,
     comfortably under the threshold, so the clipped window was marked good.
     Result: windows with n_hours=162, hour_idx spanning 6..167.

  2. Because the gate is per-KPI, two KPIs in the same cell survive at
     different anchor sets.  When the windows are assembled, each anchor
     inherits whatever subset of KPIs happened to pass for *that* anchor →
     min_kpis=84, max_kpis=134 within one cell.

  3. The Stage-8 indexing join re-joined on kpi_id, so it never enforced a
     joint anchor set either; it faithfully indexed whatever incomplete data
     arrived.

The fix has three parts, applied below:

  (A) Stage 2 now uses **strict edge-contiguity**: a window is good for a KPI
      only if it has all 168 distinct hours with hour_idx spanning exactly
      0..167.  ``window_valid_frac`` is retained for diagnostics but the gate
      is the contiguity flag, not the fraction.

  (B) ``greedy_joint_kpi_selection`` now returns the final joint anchor set
      directly — the running intersection it already computed internally.
      The caller rebuilds a (distname, start_time) DataFrame from that set
      and passes it to Stage 8; no separate ``filter_joint_complete_windows``
      step is needed because the returned set is already the exact intersection.

  (C) Stage 8 ``attach_windows_index_to_pm`` gates window membership on
      **distname + time only** (joint anchors), using kpi_id solely to carry
      values, never to decide window membership.

Pipeline stages (as executed in run.py)
----------------------------------------
0.  series_imputability_gate         — series-scoped pre-impute gate (merges the
                                       former filter_global_value_density +
                                       filter_gap_pattern).  Per (distname, kpi_id)
                                       checks: (a) density over own active range and
                                       (b) gap-run shape (fraction of null runs that
                                       are ≤ MAX_IMPUTABLE_GAP hours).  Drops only
                                       the weak series; a KPI survives if any cell
                                       passes both checks.  Joins on (kpi_id,
                                       distname) — never on kpi_id alone.
 [imputing]                          — forward-fill / interpolation of short gaps
                                       (up to MAX_IMPUTABLE_GAP hours); window
                                       density is computed on the imputed data.
1.  fill_internal_gaps               — per-(distname, kpi_id) hourly spine
                                       covering [kpi_min_t, kpi_max_t] only;
                                       left-joins actual values.
2.  compute_window_density_sparse    — explode imputed rows into their ≤ 7 window
                                       memberships; STRICT edge-contiguity gate.
3.  discard_invalid_windows          — keep is_good_window == 1 only.
2b. filter_max_gap_sparse            — null-run detection on the per-KPI spine
                                       plus leading-gap awareness; runs on the
                                       density-passing windows (after Stage 3).
4.  compute_kpi_yield_stats          — diagnostic per-KPI stats (total_windows,
                                       n_cells, frac_contributing_cells).
                                       theoretical_max_windows removed —
                                       that number is fictional on gappy PM data.
4c. flag_flat_series                 — flag near-constant / near-zero-constant
                                       series (is_flat=True); does NOT drop them.
                                       Flat series are informative signal for the
                                       conditional model; the flag lets the
                                       per-segment scaler avoid div-by-zero.
5.  prefilter_kpis                   — honest structural pre-cut on
                                       frac_contributing_cells and total_windows;
                                       returns unordered candidate list.
6.  greedy_joint_kpi_selection       — greedily build the largest KPI set whose
                                       joint (distname, anchor) count stays above
                                       the elbow-method floor; returns
                                       (selected_kpis, joint_anchor_pairs).
7.  attach_windows_index_to_pm       — join imputed selected-KPI data against joint
                                       anchors on distname + time span; assigns
                                       window_anchor and hour_idx to every row.

NOTE: ``filter_temporal_stability`` and ``filter_cross_cell_consistency`` exist
in this module but are **not** called in the current pipeline.
``extract_valid_pm_windows`` also exists (deferred densification via a cross-join)
but is likewise unused; ``attach_windows_index_to_pm`` is the active Stage 8.

Input schema (raw long-format DataFrame)
-----------------------------------------
    start_time  : timestamp  — hourly
    kpi_id      : string
    kpi_value   : double
    bts_id      : string     — parent of distname
    distname    : string     — cell identifier

Output schema (indexed training DataFrame)
------------------------------------------
    distname      : string
    bts_id        : string
    kpi_id        : string   — selected KPIs only
    window_anchor : timestamp — stride-aligned window start
    hour_idx      : int      — 0-based position within the window (0..167)
    kpi_value     : double   — imputed where data was absent
    imputed_flag  : int/bool — 1 where value was imputed
"""


def series_imputability_gate(
    gap_filled_df: DataFrame,
    *,
    min_global_density: float = 0.80,
    max_imputable_gap: int = 6,
    min_imputable_gap_frac: float = 0.90,
) -> DataFrame:
    """Gate each (distname, kpi_id) series on density AND gap-run shape.

    Merges the former filter_global_value_density + filter_gap_pattern into a
    single series-scoped gate that runs before imputation.  The key design
    change is scope: decisions are made per (distname, kpi_id) pair, NOT
    aggregated to the KPI level.  A KPI that is dense in some cells but absent
    in others is not discarded — only the individual weak series are dropped.

    Two questions are answered for each (distname, kpi_id):

      1. **Density** — is the series too sparse over its own active range to be
         worth imputing?  ``non_null_count / total_hours >= min_global_density``

      2. **Gap-run shape** — are the null runs short enough to impute without
         fabricating large blocks of invented signal?
         ``fraction(run_length <= max_imputable_gap) >= min_imputable_gap_frac``

    Decision table:

        ┌──────────────────┬──────────────────┬──────────────────────────────┐
        │ density_passes   │ gap_shape_passes  │ outcome                      │
        ├──────────────────┼──────────────────┼──────────────────────────────┤
        │ True             │ True              │ KEEP — safe to impute        │
        │ True             │ False             │ DROP — Swiss-cheese gaps     │
        │ False            │ True              │ DROP — too sparse overall    │
        │ False            │ False             │ DROP — both fail             │
        └──────────────────┴──────────────────┴──────────────────────────────┘

    Case 1 — passes (sparse-ish but all short gaps):

        kpi_X @ c1:  v v N v v v N v v v    density=0.80, gaps ≤ MIG
                     → KEEP: two 1-h gaps are honest imputation targets

    Case 2 — fails on gap shape (Swiss-cheese):

        kpi_X @ c2:  v N v N v N v N v N    density=0.50, gap every 2 h
                     → DROP: half the series would be invented signal

    Case 3 — fails on density (mostly empty):

        kpi_X @ c3:  v N N N N N N N v v    density=0.30, long holes
                     → DROP: too little real data to anchor any imputation

    Case 4 — KPI survives even though some series die:

        kpi_X @ c1:  v v N v v v N v v v    → KEEP  (Case 1)
        kpi_X @ c2:  v N v N v N v N v N    → DROP  (Case 2)
        kpi_X @ c3:  v N N N N N N N v v    → DROP  (Case 3)
        kpi_X is NOT discarded — it contributes only c1 downstream.

    Parameters
    ----------
    gap_filled_df : DataFrame
        Output of fill_internal_gaps — per-KPI hourly spine with null rows for
        missing hours.  count(*) over the spine gives the true active-range
        length per series, so no cross-KPI padding inflates the denominator.
    min_global_density : float
        Minimum non-null fraction over a series' own active range for the
        (distname, kpi_id) pair to pass the density check (default 0.80).
    max_imputable_gap : int
        Maximum null-run length in hours considered safely imputable
        (default 6).  Runs longer than this threshold count as "bad gaps".
    min_imputable_gap_frac : float
        Minimum fraction of all null runs that must be ≤ max_imputable_gap
        for the series to pass the gap-shape check (default 0.90).  Series
        with zero null runs pass vacuously — no bad gaps can exist.

    Returns
    -------
    DataFrame
        Columns (kpi_id, distname).  Only series where BOTH density_passes
        AND gap_shape_passes are True are included.  The caller joins on
        both columns to filter gap_filled_df at the series level.
    """
    # ── Step 1: density per (distname, kpi_id) ─────────────────────────────
    # The gap-filled spine has exactly one row per hour inside the KPI's own
    # [min_t, max_t].  count(*) is therefore the total active-range length and
    # needs no correction for cross-KPI padding.
    density_stats = (
        gap_filled_df.groupBy("kpi_id", "distname")
        .agg(
            # count non-null hours: 1 where kpi_value exists, 0 where it is null
            f.sum(f.when(f.col("kpi_value").isNotNull(), 1).otherwise(0)).alias("non_null_count"),
            # denominator = total spine rows = own active-range length
            f.count("*").alias("total_hours"),
        )
        # density = fraction of active-range hours that carry a real value
        .withColumn("global_density", f.col("non_null_count") / f.col("total_hours"))
        # True when the series is at least min_global_density non-null
        .withColumn("density_passes", f.col("global_density") >= f.lit(min_global_density))
        .select("kpi_id", "distname", "density_passes")
    )

    # ── Step 2: null-run detection per (distname, kpi_id) ──────────────────
    # A "null run" is a contiguous streak of null-valued spine rows.  We detect
    # run boundaries with a lag-based finite-difference:
    #
    #   is_null        = 1 if kpi_value is null, else 0
    #   prev_is_null   = is_null of the preceding row in the same series
    #                    (default 0 at the partition boundary so the first row
    #                    is never treated as a run continuation)
    #   null_run_start = 1 only at the 0→1 rising edge (non-null → null),
    #                    i.e., the first null row of each new run
    #   run_id         = cumulative sum of null_run_start from the start of the
    #                    partition; gives each null run a unique integer ID
    #                    within its (distname, kpi_id) partition
    lag_w = Window.partitionBy("distname", "kpi_id").orderBy("start_time")

    with_runs = (
        gap_filled_df
        # flag nulls: 1 = missing hour, 0 = hour with a real value
        .withColumn("is_null", f.col("kpi_value").isNull().cast("int"))
        # bring in the previous row's flag; 0-default at partition start
        .withColumn("prev_is_null", f.lag("is_null", 1, 0).over(lag_w))
        # rising edge: 1 only when transitioning from non-null (0) to null (1)
        .withColumn(
            "null_run_start",
            f.when((f.col("is_null") == 1) & (f.col("prev_is_null") == 0), 1).otherwise(0),
        )
        # cumsum of run_start markers → monotonically increasing run counter
        # rowsBetween(unboundedPreceding, 0) accumulates from partition start to current row
        .withColumn(
            "run_id",
            f.sum("null_run_start").over(lag_w.rowsBetween(Window.unboundedPreceding, 0)),
        )
    )

    # Keep only null rows, then count how many spine hours belong to each run.
    # That count is the run length (consecutive null hours).
    null_run_lengths = (
        with_runs.filter(f.col("is_null") == 1)
        .groupBy("kpi_id", "distname", "run_id")
        # count(*) over a single run = number of consecutive null hours
        .agg(f.count("*").alias("run_length"))
    )

    # ── Step 3: gap-shape score per (distname, kpi_id) ─────────────────────
    # For each series, what fraction of its null runs are "short enough"
    # (≤ max_imputable_gap hours) to be imputed without fabricating signal?
    # mean(short_flag) = count(short runs) / count(all runs).
    #
    # Example — safe series (95 % short gaps):
    #   ████░░████░░████░░████  density=0.86, max_gap=2h  → passes
    #
    # Example — unsafe series (one long nighttime block):
    #   ████████████░░░░░░░░░░  density=0.86, max_gap=24h → fails
    #   A single 24-h run wipes out an entire night — imputing it fabricates
    #   a full overnight period of invented values.
    gap_shape_stats = (
        null_run_lengths.groupBy("kpi_id", "distname")
        .agg(
            # short_flag = 1 if this run can be safely imputed, 0 if too long
            # mean over all runs = fraction of runs that are short enough
            f.mean(f.when(f.col("run_length") <= max_imputable_gap, 1).otherwise(0)).alias(
                "imputable_gap_frac"
            ),
        )
        # True when ≥ min_imputable_gap_frac of the series' runs are short
        .withColumn(
            "gap_shape_passes",
            f.col("imputable_gap_frac") >= f.lit(min_imputable_gap_frac),
        )
        .select("kpi_id", "distname", "gap_shape_passes")
    )

    # ── Step 4: combine both gates ─────────────────────────────────────────
    # Left-join gap_shape_stats onto density_stats.
    # Series with zero null runs have no rows in null_run_lengths and therefore
    # no row in gap_shape_stats.  After the left join their gap_shape_passes is
    # NULL.  coalesce(gap_shape_passes, True) encodes vacuous truth:
    # "no null runs at all" trivially satisfies the gap-shape check.
    return (
        density_stats.join(gap_shape_stats, on=["kpi_id", "distname"], how="left")
        # vacuous truth: zero null runs → no bad gaps → gap shape passes
        .withColumn(
            "gap_shape_passes",
            f.coalesce(f.col("gap_shape_passes"), f.lit(True)),
        )
        # series survives only when BOTH density AND gap shape are acceptable
        .filter(f.col("density_passes") & f.col("gap_shape_passes"))
        .select("kpi_id", "distname")
    )


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

    # Dedup once, up front: at most one row per (distname, kpi_id, bts_id, hour).
    # Downstream window stats use a plain count() of non-null rows as a proxy
    # for the distinct-hour count; that equivalence only holds if the spine has
    # no duplicate timestamps per series. Paying the dedup here (one pass) is
    # far cheaper than forcing a countDistinct on every window in Stage 2.
    df = df.dropDuplicates(["distname", "kpi_id", "bts_id", time_col])

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


def drop_windows_with_nulls(
    pm_df: DataFrame,
    *,
    window_hours: int = 168,
    stride_hours: int = 24,
) -> DataFrame:
    """Compute per-anchor window validity **without** a distname-wide dense grid.

    Algorithm
    ---------
    1. Broadcast the distname-level time origin (earliest timestamp across all
       KPIs in the distname).  This is the reference for stride-aligned anchors.

    2. For every data row compute the hourly offset from the distname origin.

    3. Determine the ≤ W/S stride-aligned anchor indices whose window
       [anchor, anchor + W) contains this row, and ``explode`` them.
       With W = 168 and S = 24 each row maps to at most 7 anchors.

    4. ``groupBy(distname, kpi_id, anchor)`` → count non-null values **and**
       record the min / max within-window hour offset of those non-null rows.

    5. A window is good only under **strict edge-contiguity**:

           non_null_count == window_hours
           AND min_hour_idx == 0
           AND max_hour_idx == window_hours - 1

       ``window_valid_frac = non_null_count / window_hours`` is still emitted
       for diagnostics, but it is NOT the gate.

    Why strict contiguity (vs. the old fractional ``density_threshold``)
    --------------------------------------------------------------------
    The anchor grid is aligned to the distname origin (earliest ts across all
    KPIs in the cell), but each KPI starts at its own time.  A KPI that starts
    6 h after the cell origin gets a window anchored *before* its first row, so
    that window is missing hours 0..5.  At W=168 that is only 3.6 % missing —
    under any reasonable fractional threshold — yet the resulting tensor cannot
    be built (hour_idx 0..5 do not exist).  Requiring a full, contiguous 0..167
    span eliminates these clipped-edge windows at the source.

    The ``density_threshold`` parameter is retained for signature stability /
    diagnostics but no longer gates ``is_good_window``.

    Tail filter
    -----------
    Anchors whose window would extend past the **KPI's own** series-end are
    discarded.  (Strict contiguity already rejects them — a window past the
    series end cannot have hour 167 — but the explicit tail filter keeps the
    intermediate set small.)

    Parameters
    ----------
    df : DataFrame
        Per-KPI gap-filled data from ``fill_internal_gaps``, **or** raw data.
    window_hours : int
        Window width in hours (default 168 = 1 week).
    stride_hours : int
        Stride between window anchors in hours (default 24 = 1 day).
    density_threshold : float
        Retained for diagnostics; no longer gates validity.

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
    distname_origin = pm_df.groupBy("distname").agg(
        f.min(f.unix_timestamp("start_time")).alias("dist_origin_epoch"),
    )

    # ── per-(distname, kpi_id) series end for tail filter ───────────────
    series_end = (
        pm_df.filter(f.col("kpi_value").isNotNull())
        .groupBy("distname", "kpi_id")
        .agg(
            f.max(f.unix_timestamp("start_time")).alias("series_end_epoch"),
        )
    )

    # ── distname lookup (one cell per distname × kpi_id) ───────────────
    distname_lookup = pm_df.select("distname", "kpi_id").distinct()

    # ── attach origin & compute offset ─────────────────────────────────
    base = (
        pm_df.join(f.broadcast(distname_origin), on="distname")
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
        # within-window hour offset of THIS row for the exploded anchor
        .withColumn(
            "hour_idx",
            ((f.col("row_epoch") - f.col("anchor_epoch")) / 3600).cast("long"),
        )
    )

    # ── aggregate validity per (distname, kpi_id, anchor) ──────────────
    #
    # Strict edge-contiguity: count distinct non-null hours AND track the
    # first / last non-null hour offset inside the window.  Only non-null
    # rows count toward coverage.
    non_null = with_anchors.filter(
        (f.col("kpi_value").isNotNull())
        & (f.col("hour_idx") >= 0)
        & (f.col("hour_idx") < window_hours)
    )

    window_stats = (
        non_null.groupBy("distname", "kpi_id", "anchor_epoch")
        .agg(
            # Plain count(), NOT countDistinct: after the Stage-1 dedup there is
            # at most one row per (series, hour), so count of non-null rows in a
            # window equals the distinct-hour count. count/min/max are all
            # partial aggregates (map-side combinable) — countDistinct is not,
            # and forces a heavy second shuffle + per-group set tracking that
            # spills to disk. This is the line that was slow.
            f.count(f.lit(1)).alias("non_null_count"),
            f.min("hour_idx").alias("min_hour_idx"),
            f.max("hour_idx").alias("max_hour_idx"),
        )
        .withColumn(
            "window_valid_frac",
            f.col("non_null_count") / f.lit(window_hours),
        )
        .withColumn(
            "is_good_window",
            f.when(
                (f.col("non_null_count") == f.lit(window_hours))
                & (f.col("min_hour_idx") == f.lit(0))
                & (f.col("max_hour_idx") == f.lit(window_hours - 1)),
                1,
            ).otherwise(0),
        )
    )

    # ── tail filter + attach metadata ──────────────────────────────────
    result = (
        window_stats.join(series_end, on=["distname", "kpi_id"])
        .filter(f.col("anchor_epoch") + f.lit(window_end_offset_s) <= f.col("series_end_epoch"))
        .drop("series_end_epoch", "non_null_count", "min_hour_idx", "max_hour_idx")
        .join(distname_lookup, on=["distname", "kpi_id"])
        .withColumn(
            "start_time",
            f.from_unixtime(f.col("anchor_epoch")).cast("timestamp"),
        )
        .select(
            "distname",
            "kpi_id",
            "start_time",
            "window_valid_frac",
            "is_good_window",
        )
    )

    return result


def discard_invalid_windows(
    pm_windows: DataFrame,
) -> DataFrame:
    """Drop windows that did not meet the validity gate."""
    return pm_windows.filter(f.col("is_good_window") == 1)


def compute_kpi_yield_stats(
    good_windows: DataFrame,
    *,
    total_distinct_cells: int,
) -> DataFrame:
    """Aggregate per-KPI statistics for diagnostics and pre-filtering.

    Returns total_windows, n_cells, frac_contributing_cells, mean_windows_per_cell.
    window_coverage_frac is intentionally absent — its denominator
    (theoretical_max_windows) assumes a gap-free series and is unreliable on
    real PM data.
    """
    return (
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


def prefilter_kpis(
    kpi_yield_stats: DataFrame,
    *,
    breadth_percentile: float = 0.10,
    min_breadth_floor: float = 0.05,
    max_drop_frac: float = 0.25,
) -> list[str]:
    """Distribution-relative structural pre-cut before the O(K^2) greedy loop.

    Instead of a fixed frac_contributing_cells threshold, derive the cut point
    from the data: drop only KPIs in the bottom `breadth_percentile` of cell
    breadth. Guarded so the cut can never become aggressive enough to amputate
    candidates greedy might have kept.

    Guards (in order of application):
      1. Threshold is the breadth_percentile-th percentile of
         frac_contributing_cells, but never below min_breadth_floor (so on
         uniform distributions we don't cut at a meaninglessly high floor).
      2. If the resulting cut would drop more than max_drop_frac of candidates,
         the pre-filter disables itself entirely and returns all KPIs — a sign
         the distribution is too flat for a safe structural cut.
    """
    total_kpis = kpi_yield_stats.count()

    # Empirical threshold: the breadth_percentile quantile of cell breadth.
    # approxQuantile is one cheap pass; relativeError 0.01 is plenty here.
    threshold = kpi_yield_stats.approxQuantile(
        "frac_contributing_cells", [breadth_percentile], 0.01
    )[0]

    # Never cut above the floor: if the 10th-percentile breadth is high
    # (uniform distribution), fall back to a low absolute floor so we only
    # ever remove genuinely thin KPIs.
    threshold = (
        min(threshold, max(threshold, min_breadth_floor))
        if threshold < min_breadth_floor
        else min(threshold, min_breadth_floor)
    )
    # clearer: cut at the SMALLER of (empirical percentile, floor)
    threshold = min(threshold, min_breadth_floor) if threshold > min_breadth_floor else threshold

    survivors = kpi_yield_stats.filter(f.col("frac_contributing_cells") >= f.lit(threshold))
    n_survivors = survivors.count()
    n_dropped = total_kpis - n_survivors

    # Safety guard: if we'd cut too deep, the distribution isn't tail-shaped
    # enough for a safe pre-cut. Disable rather than risk shrinking the K-set.
    if n_dropped / total_kpis > max_drop_frac:
        logger.info(
            f"[prefilter] would drop {n_dropped}/{total_kpis} "
            f"({n_dropped/total_kpis:.0%}) > max_drop_frac={max_drop_frac:.0%} "
            f"— distribution too flat, disabling pre-cut"
        )
        return kpi_yield_stats.select("kpi_id").rdd.flatMap(lambda r: [r["kpi_id"]]).collect()

    logger.info(
        f"[prefilter] breadth threshold={threshold:.3f} "
        f"(p{breadth_percentile*100:.0f}) | "
        f"dropped {n_dropped}/{total_kpis} KPIs"
    )
    return survivors.select("kpi_id").rdd.flatMap(lambda r: [r["kpi_id"]]).collect()


# ═══════════════════════════════════════════════════════════════════════════
#                Greedy joint KPI selection
# ═══════════════════════════════════════════════════════════════════════════
def _collect_kpi_coverage(
    good_windows: DataFrame,
    candidates: list[str],
) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    """Collect KPI coverage once and pack it into a boolean numpy matrix.

    This is the **single** Spark→driver collect for the whole greedy stage
    (the old code collected the same groupBy twice — once for the elbow curve
    and once for the real selection).

    Returns
    -------
    M : np.ndarray[bool], shape (n_kpis, n_anchors)
        ``M[i, j]`` is True when KPI ``kpi_ids[i]`` has a good window at the
        ``(distname, start_time)`` pair ``anchor_pairs[j]``.
    kpi_ids : list[str]
        Row → KPI id.  Restricted to ``candidates`` that actually appear in the
        collected coverage, ordered by descending coverage so the natural seed
        (most-covering KPI) is row 0.
    anchor_pairs : list[tuple[str, str]]
        Column → (distname, start_time) pair.
    """
    # One scan: per (kpi, distname) the set of anchor start_times. start_time is
    # cast to string so the (distname, start_time) tuples hash cleanly.
    rows = (
        good_windows.withColumn("start_time", f.col("start_time").cast("string"))
        .groupBy("kpi_id", "distname")
        .agg(f.collect_set("start_time").alias("anchors"))
        .collect()
    )

    candidate_set = set(candidates)

    # kpi_id → set of (distname, start_time) anchor pairs, candidates only.
    kpi_coverage: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        kpi = row["kpi_id"]
        if kpi not in candidate_set:
            continue
        pairs = {(row["distname"], anchor) for anchor in row["anchors"]}
        kpi_coverage.setdefault(kpi, set()).update(pairs)

    # Order rows by descending coverage so row 0 is the natural seed.
    kpi_ids = sorted(kpi_coverage, key=lambda k: len(kpi_coverage[k]), reverse=True)

    # Global anchor → column index.
    anchor_index: dict[tuple[str, str], int] = {}
    for kpi in kpi_ids:
        for pair in kpi_coverage[kpi]:
            if pair not in anchor_index:
                anchor_index[pair] = len(anchor_index)

    n_kpis = len(kpi_ids)
    n_anchors = len(anchor_index)
    M = np.zeros((n_kpis, n_anchors), dtype=bool)
    for i, kpi in enumerate(kpi_ids):
        cols = [anchor_index[pair] for pair in kpi_coverage[kpi]]
        M[i, cols] = True

    anchor_pairs = [pair for pair, _ in sorted(anchor_index.items(), key=lambda kv: kv[1])]
    return M, kpi_ids, anchor_pairs


def _greedy_order(
    M: np.ndarray,
    forced_rows: list[int],
) -> tuple[list[int], list[int]]:
    """Greedy KPI ordering over the boolean coverage matrix — no floor applied.

    At each step the candidate that retains the most joint windows (loses the
    fewest) is appended.  Forced rows are placed first (ordered greedily among
    themselves) and seed the running intersection; they are kept regardless of
    how much coverage they cost.

    Returns
    -------
    order : list[int]
        Row indices in greedy pick order.
    counts : list[int]
        ``counts[k]`` is the joint-window count after the k-th pick.  This is
        the coverage curve; it is monotonically non-increasing.
    """
    n_kpis = M.shape[0]

    order: list[int] = []
    counts: list[int] = []
    available = set(range(n_kpis))

    # ── seed ───────────────────────────────────────────────────────────────
    # live_cols = anchor column indices still in the running intersection.
    if forced_rows:
        # Highest-coverage forced KPI seeds; remaining forced ones are ordered
        # greedily too, but all are kept unconditionally.
        forced_sorted = sorted(forced_rows, key=lambda r: int(M[r].sum()), reverse=True)
        seed = forced_sorted[0]
        forced_rest = forced_sorted[1:]
    else:
        # argmax(row coverage); M rows are already coverage-sorted so this is 0,
        # but argmax keeps the function correct regardless of row ordering.
        seed = int(M.sum(axis=1).argmax())
        forced_rest = []

    live_cols = np.flatnonzero(M[seed])
    order.append(seed)
    counts.append(int(live_cols.size))
    available.discard(seed)

    def _pick(row: int) -> None:
        nonlocal live_cols
        order.append(row)
        live_cols = live_cols[M[row, live_cols]]
        counts.append(int(live_cols.size))
        available.discard(row)

    # Forced KPIs next — kept no matter the coverage cost.
    for row in forced_rest:
        _pick(row)

    # ── greedy fill over the rest ────────────────────────────────────────────
    while available:
        cand = np.fromiter(available, dtype=np.intp, count=len(available))
        # Joint-window count each candidate would retain. live_cols are all-True
        # in the current intersection, so the AND-count reduces to a column slice.
        cand_counts = M[np.ix_(cand, live_cols)].sum(axis=1)
        best = int(cand[cand_counts.argmax()])
        _pick(best)

    return order, counts


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
    forced_kpis: list[str] | None = None,
) -> list[str]:
    """Greedily build the largest KPI set whose joint window count stays above floor.

    A 'joint window' is a unique (distname, start_time) pair where every
    selected KPI has data. Each such pair represents one training sample.

    The greedy loop runs on a boolean numpy coverage matrix (see
    ``_collect_kpi_coverage`` / ``_greedy_order``) instead of Python set algebra,
    and Spark is collected exactly **once**: the no-floor greedy ordering is the
    same ordering as the floored selection (the floor only decides where to
    stop), and the joint-window count is monotonically non-increasing along it,
    so the final selection is just the prefix of that single ordering that stays
    at or above the floor.

    Parameters
    ----------
    forced_kpis : list[str] | None
        KPIs that must appear in the result.  They are placed first and seed the
        running intersection, and are **exempt from the floor** — kept even if
        they alone drop joint coverage below it (a warning is logged).  Forced
        KPIs absent from the collected coverage are skipped with a warning.
        When ``None``/empty, behavior is unchanged: the elbow method (or
        ``min_joint_windows_abs``) chooses the KPI count.
    """
    # ── single Spark collect → numpy coverage matrix ─────────────────────────
    M, kpi_ids, _anchor_pairs = _collect_kpi_coverage(good_windows, candidates)
    row_of = {kpi: i for i, kpi in enumerate(kpi_ids)}

    # ── resolve forced KPIs to rows; skip any without collected coverage ─────
    forced_rows: list[int] = []
    for kpi in forced_kpis or []:
        if kpi in row_of:
            forced_rows.append(row_of[kpi])
        else:
            logger.warning(
                f"[greedy] forced KPI '{kpi}' has no good-window coverage among "
                f"candidates — skipping it"
            )

    # ── one greedy pass → full ordering + coverage curve ─────────────────────
    order, counts = _greedy_order(M, forced_rows)
    curve = [
        {"step": k + 1, "kpi_id": kpi_ids[row], "joint_windows": counts[k]}
        for k, row in enumerate(order)
    ]

    # ── determine the joint-window floor ─────────────────────────────────────
    if min_joint_windows_abs is None:
        logger.info(
            "[elbow] min_joint_windows_abs not selected - defaulting to elbow method of selection"
        )
        elbow_idx = find_elbow(curve)
        min_joint_windows = suggest_threshold(curve, elbow_idx)
        logger.info(
            f"[elbow] suggested cutoff: {elbow_idx} KPIs "
            f"| joint_windows at elbow: {min_joint_windows:,}"
        )
    else:
        min_joint_windows = min_joint_windows_abs

    logger.info(f"{min_joint_windows=}")

    n_forced = len(forced_rows)
    seed_kpi = kpi_ids[order[0]]
    logger.info(f"[greedy] seed '{seed_kpi}' | coverage={counts[0]:,}")

    if n_forced and counts[n_forced - 1] < min_joint_windows:
        logger.warning(
            f"[greedy] forced KPIs alone drop joint coverage to "
            f"{counts[n_forced - 1]:,} < floor {min_joint_windows:,} — "
            f"keeping them anyway (exempt from floor)"
        )

    # ── select the floor-respecting prefix; forced picks are always kept ─────
    selected: list[str] = []
    for k, row in enumerate(order):
        kpi = kpi_ids[row]
        joint_windows = counts[k]

        # Forced picks (the first n_forced steps) bypass the floor entirely.
        if k >= n_forced and joint_windows < min_joint_windows:
            logger.info(
                f"[greedy] stopping at step {k + 1} — "
                f"best candidate '{kpi}' would reduce coverage "
                f"to {joint_windows:,} < {min_joint_windows:,}"
            )
            break

        selected.append(kpi)
        logger.info(
            f"[greedy] step {k + 1:>4d} | accepted '{kpi}' "
            f"| selected={len(selected):>4d} "
            f"| joint_windows={joint_windows:,} "
        )

    return selected


def filter_joint_complete_windows(
    good_windows: DataFrame,
    selected_kpis: list[str],
) -> DataFrame:
    """Keep only anchors where **every** selected KPI is complete.

    This is the gate that fixes the min_kpis / max_kpis divergence within a
    cell.  ``good_windows`` is already restricted to fully-contiguous per-KPI
    windows (Stage 2 strict gate), but different KPIs still survive at
    different anchor sets.  When the windows are later densified / indexed,
    each (distname, anchor) must carry the *same* set of selected KPIs — every
    one present and complete — or the assembled tensor is ragged.

    An anchor (distname, start_time) survives iff the number of distinct
    selected KPIs that have a good window there equals ``len(selected_kpis)``.

    The returned frame is restricted to ``selected_kpis`` and to surviving
    anchors, so every (distname, start_time) it contains has exactly
    ``len(selected_kpis)`` rows — one complete window per selected KPI.

    Parameters
    ----------
    good_windows : DataFrame
        All-filters-passing per-KPI anchors (Stage 2 / 2b survivors), with at
        least columns (distname, kpi_id, start_time, bts_id).  May contain KPIs
        beyond ``selected_kpis``; they are filtered out here.
    selected_kpis : list[str]
        Final KPI set from greedy selection.

    Returns
    -------
    DataFrame
        Same schema as *good_windows*, restricted to selected KPIs and to
        anchors where all selected KPIs are present.
    """
    n_selected = len(selected_kpis)

    selected_windows = good_windows.filter(f.col("kpi_id").isin(selected_kpis))

    complete_anchors = (
        selected_windows.groupBy("distname", "start_time")
        .agg(f.countDistinct("kpi_id").alias("n_kpis_present"))
        .filter(f.col("n_kpis_present") == f.lit(n_selected))
        .select("distname", "start_time")
    )

    return selected_windows.join(complete_anchors, on=["distname", "start_time"], how="inner")


def emit_window_index(
    pm_df_long_selected: DataFrame,
    good_windows: DataFrame,
    window_hours: int = 168,
) -> DataFrame:
    """Mark window starts on the long PM data WITHOUT materialising windows.

    The old ``attach_windows_index_to_pm`` range-joined the long data against the
    joint anchors on ``start_time IN [anchor, anchor + W)``.  With W=168 and S=24
    each window overlaps its 6 neighbours, so every hourly value was duplicated up
    to 7×: the full window grid was materialised up front.  That is the heavy step
    on real PM data.

    This function does not materialise anything.  It returns the long PM data **as
    is** (one row per (distname, kpi_id, start_time), no duplication) with a single
    extra ``window_anchor`` column that is non-null **only on the rows that begin a
    window** and null everywhere else::

        distname  kpi_id  start_time  window_anchor  kpi_value
        c1        A       date1       date1          0.1   <- anchor start
        c1        A       date2       null           0.1
        c1        A       date3       null           0.1
        ...                           (next stride anchor) ...
        c1        A       date25      date25         0.1   <- anchor start

    Actual K×W windows are reconstructed downstream (see
    ``genpm.modelling.dataset_prep.dataset_prep.generate_window_tensors``), which
    reads the non-null marks as the window starts and slices [anchor, anchor+W) out
    of this same frame — the overlap then costs zero storage here.

    ``good_windows`` is expected to be the JOINT-complete anchor set: every
    (distname, start_time) it contains is valid for *all* selected KPIs.  Its
    ``start_time`` is a stride anchor (not a real row timestamp), so it is renamed
    to ``anchor_start_time`` internally; the emitted column is ``window_anchor``.

    Two restrictions are applied to the emitted rows, both via non-duplicating
    semi-joins:
      (a) only (distname, kpi_id) pairs present in ``good_windows`` survive, and
      (b) only rows that fall inside at least one window span are kept ("only the
          windows that are there").

    Membership is keyed on ``distname + time`` only — never ``kpi_id`` — so the
    joint guarantee is preserved.
    """
    # good_windows.start_time IS the stride anchor — name it accordingly.
    anchors = good_windows.select(
        "distname", f.col("start_time").alias("anchor_start_time")
    ).distinct()
    anchor_spans = anchors.withColumn(
        "anchor_end",
        f.col("anchor_start_time") + f.expr(f"INTERVAL {window_hours} HOURS"),
    )

    # (a) keep only the (distname, kpi_id) pairs that survived into good_windows
    selected_pairs = good_windows.select("distname", "kpi_id").distinct()
    pm = pm_df_long_selected.join(selected_pairs, on=["distname", "kpi_id"], how="left_semi")

    # (b) keep only rows inside at least one window span — left_semi is a FILTER,
    #     not a 7× materialisation: a row survives once if any anchor covers it.
    pm_in_windows = pm.alias("p").join(
        anchor_spans.alias("s"),
        on=[
            f.col("p.distname") == f.col("s.distname"),
            f.col("p.start_time") >= f.col("s.anchor_start_time"),
            f.col("p.start_time") < f.col("s.anchor_end"),
        ],
        how="left_semi",
    )

    # (c) mark ONLY the rows whose timestamp is an anchor start; null elsewhere.
    #     equality left-join adds one column without growing the row count.
    return (
        pm_in_windows.alias("p")
        .join(
            anchors.alias("a"),
            on=[
                f.col("p.distname") == f.col("a.distname"),
                f.col("p.start_time") == f.col("a.anchor_start_time"),
            ],
            how="left",
        )
        .withColumn("window_anchor", f.col("a.anchor_start_time"))  # null = not a start
        .select(
            "p.distname",
            "p.bts_id",
            "p.kpi_id",
            "p.start_time",
            "window_anchor",
            "p.kpi_value",
            "p.imputed_flag",
        )
    )


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

    ``training_windows_anchors`` is expected to be the JOINT-complete anchor
    set (output of ``filter_joint_complete_windows``).  Because every
    (distname, start_time) there is valid for all selected KPIs, the
    KPI-agnostic cross of (distinct anchors) × (selected KPIs) yields exactly
    the dense, ragged-free grid the dataloader needs.

    For each valid window anchor the function:

    1. Generates the ``window_hours``-long hourly spine.
    2. Crosses the spine with the selected KPIs (plus their bts_id).
    3. Left-joins actual values from the raw data.
    4. Deduplicates hours covered by multiple overlapping windows.

    Parameters
    ----------
    pm_df : DataFrame
        Raw (or per-KPI gap-filled) long-format data.
    training_windows_anchors : DataFrame
        Joint-complete anchor DataFrame restricted to selected KPIs.
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

    # Distinct (distname, start_time) anchors — KPI-agnostic, because the
    # joint-completeness gate already ensured all selected KPIs share these
    # anchors.
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

    kpi_dims = pm_df.select("distname", "kpi_id", "bts_id").distinct()

    full_grid = anchor_spines.join(kpi_dims, on="distname", how="inner")

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
