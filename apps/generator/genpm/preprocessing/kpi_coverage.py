from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from genpm.utils.logger import get_logger

logger = get_logger()
# TODO: change this function, so it sees low coverage in time, not overall
# BUT THIS APPROACH SHOULD STAY ASWELL
# THE ABOVE SHOULD BE A PREFILTER for KPI selection
# First we should do a per window (given window of N hours), coverage thresholding
# (this will include kpis with big holes, but tails are valid overall)
# Then, we should apply a greedy approach maximizing per window coverage for all kpis
# And include only a kpi combination, with the highest coverage
#
# THE GOAL
# Find the largest subset of KPIs such that enough (cell, window)
# combinations have joint coverage ≥ threshold.
#

"""
kpi_window_selection.py
=======================
Full pipeline for selecting the largest jointly-valid KPI subset for TimeVAE training.

Pipeline stages
---------------
1. align_cell_time_ranges           — define a per-cell canonical hourly time axis as the
                                      UNION of all (distname, kpi_id) series ranges, then
                                      reindex every series onto that axis, null-filling gaps.
                                      KPIs with different provisioning dates coexist without
                                      data loss; the density filter handles pre-provisioning
                                      nulls automatically downstream.
2. compute_window_density           — sliding-window density via Window.rowsBetween; marks
                                      each (distname, kpi_id, window_start) as valid/invalid.
3. discard_invalid_windows          — drops rows where is_good_window == 0, keeping only the
                                      material needed for downstream counting.
4. compute_theoretical_max_windows  — per-(distname, kpi_id) upper bound on window count
                                      assuming perfect density; used to normalise thresholds
                                      into data-volume-independent fractions.
5. compute_kpi_yield_stats          — aggregates per-KPI statistics including
                                      window_coverage_frac (observed / theoretical max).
6. prefilter_kpis                   — drops structurally bad KPIs on two fraction-based and
                                      one absolute criterion; sorts by window_coverage_frac.
7. greedy_joint_kpi_selection       — greedily builds the largest KPI set whose joint window
                                      count stays above a data-relative floor; long format
                                      only, no pivot.

Input schema (raw long-format DataFrame)
-----------------------------------------
    start_time  : timestamp  — hourly, already null-filled (no missing rows)
    kpi_id      : string
    kpi_value   : double
    bts_id      : string     — parent of distname
    distname    : string     — cell identifier

All public functions expose every tunable knob as a keyword parameter with a documented
default.  Thresholds are expressed as fractions wherever possible so they remain
meaningful as data volume, cell count, or horizon length change.
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
    outside its own active period.  The result is that every KPI in a cell shares
    the same set of timestamps; pre-provisioning and post-decommission periods are
    represented as nulls, which the downstream density filter handles correctly
    (a window sitting in a dead zone fails the density threshold and is discarded).

    This preserves all data, avoids any cross-KPI interference, and makes the
    null-fill semantically correct: a null produced here is indistinguishable from
    a real measurement gap, which is the right treatment.

    Algorithm
    ---------
    For every ``distname``:
      1. Find each KPI's first and last non-null timestamp  → per-(distname, kpi_id)
         active range.
      2. Take the cell-level min of those first timestamps and max of those last
         timestamps  → ``cell_tmin``, ``cell_tmax``  (union range).
      3. Generate a contiguous hourly sequence from ``cell_tmin`` to ``cell_tmax``.
      4. Cross-join that sequence with all (distname, kpi_id) pairs observed in the
         cell  → the full spine.
      5. Left-join actual measurements onto the spine; unmatched spine rows get
         ``kpi_value = null``.

    Performance note
    ----------------
    The spine explosion is ~200 cells × ~5 cells/BTS × 700 KPIs × 2160 hours ≈
    1.5 B rows before the left join, but the computation is embarrassingly parallel
    by ``distname``.  ``cell_kpi_pairs`` is tiny and is broadcast automatically by
    Spark.  Ensure the cluster has sufficient shuffle partitions set for the join.

    Parameters
    ----------
    df : DataFrame
        Raw long-format input with columns
        (start_time, kpi_id, kpi_value, bts_id, distname).
    freq_hours : int
        Expected time step in hours (default 1).  Used to generate the canonical
        axis via ``sequence(cell_tmin, cell_tmax, INTERVAL N HOURS)``.

    Returns
    -------
    DataFrame
        Same schema as input.  Every (distname, kpi_id) pair has exactly one row
        per hour in [cell_tmin, cell_tmax].  ``kpi_value`` is null where no
        measurement existed (pre-provisioning, post-decommission, or genuine gap).
    """
    # ------------------------------------------------------------------
    # Step 1: per-(distname, kpi_id) active range from non-null observations
    # ------------------------------------------------------------------
    series_endpoints = (
        df.filter(f.col("kpi_value").isNotNull())
        .groupBy("distname", "kpi_id")
        .agg(
            f.min("start_time").alias("kpi_tmin"),
            f.max("start_time").alias("kpi_tmax"),
        )
    )

    # ------------------------------------------------------------------
    # Step 2: cell-level union range — widest span across all KPIs in cell
    # ------------------------------------------------------------------
    cell_axis = series_endpoints.groupBy("distname").agg(
        f.min("kpi_tmin").alias("cell_tmin"),
        f.max("kpi_tmax").alias("cell_tmax"),
    )

    # ------------------------------------------------------------------
    # Step 3: generate the canonical hourly timestamp sequence per cell
    # sequence() produces an array; explode() turns it into one row per hour.
    # ------------------------------------------------------------------
    interval_expr = f.expr(f"INTERVAL {freq_hours} HOURS")
    cell_timestamps = cell_axis.withColumn(
        "ts",
        f.explode(f.sequence(f.col("cell_tmin"), f.col("cell_tmax"), interval_expr)),
    ).select("distname", f.col("ts").alias("start_time"))

    # ------------------------------------------------------------------
    # Step 4: cross-join canonical timestamps × all (distname, kpi_id, bts_id)
    # pairs observed in the data  → the full alignment spine.
    # cell_kpi_pairs is small; Spark will broadcast it automatically.
    # ------------------------------------------------------------------
    cell_kpi_pairs = df.select("distname", "kpi_id", "bts_id").distinct()

    spine = cell_timestamps.join(cell_kpi_pairs, on="distname", how="inner")

    # ------------------------------------------------------------------
    # Step 5: left-join actual measurements onto the spine.
    # Rows with no matching measurement receive kpi_value = null, which
    # is the correct representation for any inactive period.
    # ------------------------------------------------------------------
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
    density_threshold: float = 0.875,
) -> DataFrame:
    # ------------------------------------------------------------------
    # Cell origin and series end — computed over the full hourly series.
    # Both window specs operate on all rows, not just anchor rows.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Rolling non-null count over the FULL hourly series.
    # rowsBetween(0, window_hours - 1) correctly spans exactly window_hours
    # consecutive hourly rows because no rows have been removed yet.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # NOW apply stride and tail filters — density is already computed
    # correctly over the full hourly series above.
    # Stride: keep only rows that are valid window anchor points.
    # Tail:   drop anchors whose window would extend past the series end.
    # ------------------------------------------------------------------
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
# Stage 3 – Discard invalid windows
# ---------------------------------------------------------------------------


def discard_invalid_windows(
    window_density: DataFrame,
) -> DataFrame:
    """Drop windows that did not meet the density threshold.

    This materialises the "good-windows only" view that all downstream stages
    operate on, and is intentionally separated from ``compute_window_density``
    so that callers can inspect the raw density distribution before committing
    to a threshold.

    Parameters
    ----------
    window_density : DataFrame
        Output of ``compute_window_density`` with column ``is_good_window``.

    Returns
    -------
    DataFrame
        Same schema, ``is_good_window == 1`` rows only.
        ``window_valid_frac`` is retained for auditability but not used
        downstream — callers may drop it before caching.
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

    This is the number of stride-aligned windows that *fit* in the series' canonical
    time range, regardless of how many of those windows actually pass the density
    threshold.  It is used to normalise ``total_windows`` into a coverage fraction
    so that pre-filter thresholds are expressed as fractions of what each KPI could
    theoretically contribute — making them independent of data volume, cell count,
    and observation horizon.

    Formula
    -------
        series_hours = number of rows in the aligned series (one row per hour)
        theoretical_max = max(0, floor((series_hours - window_hours) / stride_hours) + 1)

    The ``max(0, ...)`` clamp handles series shorter than one window (they can
    never contribute and will be dropped by the coverage-fraction filter).

    Parameters
    ----------
    aligned_df : DataFrame
        Output of ``align_cell_time_ranges`` — every (distname, kpi_id) has a
        contiguous hourly axis.  Only row count is used here; kpi_value is ignored.
    window_hours : int
        Window width in hours (default 168).
    stride_hours : int
        Stride between anchors in hours (default 24).

    Returns
    -------
    DataFrame
        Schema: (distname, kpi_id, series_hours, theoretical_max_windows).
    """
    return (
        aligned_df.groupBy("distname", "kpi_id")
        .agg(f.count("*").alias("series_hours"))
        .withColumn(
            "theoretical_max_windows",
            f.greatest(
                f.lit(0),
                ((f.col("series_hours") - f.lit(window_hours)) / f.lit(stride_hours)).cast("long")
                + f.lit(1),
            ),
        )
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
        Output of ``compute_theoretical_max_windows``.
        Schema: (distname, kpi_id, series_hours, theoretical_max_windows).
    total_distinct_cells : int
        Total number of distinct distnames in the dataset.  Pass as a pre-computed
        scalar to avoid an extra full-scan inside this function.

    Returns
    -------
    DataFrame
        One row per ``kpi_id`` with columns:

        kpi_id                      : string
        total_windows               : long    — observed good (distname, window) pairs
        theoretical_max_windows     : long    — sum of per-cell theoretical maxima
        window_coverage_frac        : double  — total_windows / theoretical_max_windows;
                                               fraction of what the KPI could contribute
                                               at perfect density — data-volume independent
        n_cells                     : long    — distinct distnames with ≥1 good window
        frac_contributing_cells     : double  — n_cells / total_distinct_cells
        mean_windows_per_cell       : double  — total_windows / n_cells
        n_active_months             : long    — distinct yyyy-MM with ≥1 good window
    """
    # Per-KPI theoretical maximum: sum across all cells where the KPI exists.
    kpi_theoretical_max = theoretical_max.groupBy("kpi_id").agg(
        f.sum("theoretical_max_windows").alias("theoretical_max_windows")
    )

    # Observed good-window statistics.
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

    # Join theoretical max and derive the normalised coverage fraction.
    stats = observed.join(kpi_theoretical_max, on="kpi_id", how="left").withColumn(
        "window_coverage_frac",
        f.when(
            f.col("theoretical_max_windows") > 0,
            f.col("total_windows") / f.col("theoretical_max_windows"),
        ).otherwise(f.lit(0.0)),
    )

    return stats


# ---------------------------------------------------------------------------
# Stage 6 – Pre-filtering
# ---------------------------------------------------------------------------


def prefilter_kpis(
    kpi_yield_stats: DataFrame,
    *,
    min_window_coverage_frac: float = 0.20,
    min_frac_contributing_cells: float = 0.30,
) -> list[str]:
    """Apply structural filters using normalised fractions and return surviving KPI list.

    Filters (applied in conjunction — a KPI must pass *all three*):

    1. ``window_coverage_frac >= min_window_coverage_frac``
       The KPI must achieve at least this fraction of its own theoretical window
       maximum (sum of per-cell perfect-density ceilings).  This replaces the
       former opaque absolute ``min_total_windows`` — a value of 0.20 means
       "the KPI achieves at least 20% of what it could if perfectly dense",
       which is meaningful regardless of horizon length, cell count, or data volume.

    2. ``frac_contributing_cells >= min_frac_contributing_cells``
       Guards against KPIs active on only a handful of BTS stations, which would
       cause the model to learn cell-specific quirks rather than general patterns.
       Already a fraction — unchanged from prior version.

    3. ``n_active_months >= min_active_months``
       Minimal guard against temporally degenerate KPIs (e.g. provisioned last
       week with no history).  Set to 2 rather than 3 given the 3-month data
       horizon — 3 would be nearly vacuous and would pass almost everything.

    Sort key
    --------
    Candidates are returned sorted by ``window_coverage_frac`` descending rather
    than ``total_windows`` descending.  Coverage fraction is a fairer ordering for
    the greedy loop: a KPI active on 10 cells with 90% coverage is a better
    candidate than one active on 200 cells with 5% coverage, even if the latter
    has more raw windows.

    Parameters
    ----------
    kpi_yield_stats : DataFrame
        Output of ``compute_kpi_yield_stats``.
    min_window_coverage_frac : float
        Minimum fraction of theoretical max windows a KPI must achieve (default 0.20).
    min_frac_contributing_cells : float
        Minimum fraction of cells with ≥1 good window (default 0.30).
    min_active_months : int
        Minimum distinct calendar months with ≥1 good window (default 2).

    Returns
    -------
    List[str]
        KPI IDs that pass all filters, sorted by ``window_coverage_frac`` descending.
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
    min_joint_coverage_frac: float = 0.50,  # fraction of theoretical max
    min_joint_windows_abs: int = 10_000,  # hard absolute floor
) -> list[str]:
    """
    ...
    The effective floor is the MAX of the two criteria — whichever is more
    restrictive wins:

        floor = max(
            int(min_joint_coverage_frac × theoretical_max_joint),
            min_joint_windows_abs,
        )

    This prevents two failure modes:
    - Too permissive on large datasets (absolute floor alone would let
      coverage collapse to near-zero fraction when theoretical_max is huge)
    - Too restrictive on small datasets (fraction alone could set a floor
      higher than what any reasonable KPI set can achieve on 3 months of data)
    ...
    """
    min_joint_windows = max(
        int(min_joint_coverage_frac * theoretical_max_joint),
        min_joint_windows_abs,
    )
    print(
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
            print(
                f"[greedy] step {idx + 1:>4d} | accepted '{kpi}' "
                f"| selected={len(selected):>4d} "
                f"| joint_windows={joint_count:,} "
                f"({joint_count / theoretical_max_joint:.1%} of theoretical max)"
            )
        else:
            print(
                f"[greedy] step {idx + 1:>4d} | SKIPPED  '{kpi}' "
                f"| joint_windows={joint_count:,} < {min_joint_windows:,}"
            )

    return selected


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------


def run_kpi_selection_pipeline(
    raw_df: DataFrame,
    *,
    # Stage 1 – alignment
    freq_hours: int = 1,
    # Stage 2 – window density
    window_hours: int = 168,
    stride_hours: int = 24,
    density_threshold: float = 0.875,
    # Stage 6 – pre-filter (all fraction-based except min_active_months)
    min_window_coverage_frac: float = 0.20,
    min_frac_contributing_cells: float = 0.30,
    # Stage 7 – greedy selection
    min_joint_coverage_frac: float = 0.10,
) -> tuple[list[str], DataFrame]:
    """Execute the full KPI selection pipeline end-to-end.

    Parameters
    ----------
    spark : SparkSession
    raw_df : DataFrame
        Raw long-format data with schema
        (start_time, kpi_id, kpi_value, bts_id, distname).
    freq_hours : int
        Hourly granularity of the canonical cell time axis (default 1).
    window_hours : int
        Sliding window width in hours (default 168 = 1 week).
    stride_hours : int
        Stride between window anchors in hours (default 24 = 1 day).
    density_threshold : float
        Minimum non-null fraction for a window to be valid (default 0.875).
    min_window_coverage_frac : float
        Pre-filter: KPI must achieve this fraction of its theoretical window
        maximum (default 0.20 = 20%).
    min_frac_contributing_cells : float
        Pre-filter: minimum fraction of cells contributing ≥1 good window
        (default 0.30).
    min_active_months : int
        Pre-filter: minimum distinct calendar months with ≥1 good window
        (default 2; 3 is nearly vacuous on a 3-month horizon).
    min_joint_coverage_frac : float
        Greedy floor: minimum fraction of the theoretical joint maximum that the
        joint window count must meet to accept a new KPI (default 0.10 = 10%).

    Returns
    -------
    selected_kpis : List[str]
        The final KPI set accepted by the greedy algorithm.
    good_windows_cached : DataFrame
        The cached good-windows DataFrame (candidates only, is_good_window == 1,
        window_valid_frac dropped).  Call ``.unpersist()`` when done.
    """

    # ------------------------------------------------------------------
    # Stage 1: align cell time ranges (union range + null-fill)
    # ------------------------------------------------------------------
    print("[pipeline] Stage 1: aligning cell time ranges (union + null-fill) ...")
    aligned = align_cell_time_ranges(raw_df, freq_hours=freq_hours)

    # ------------------------------------------------------------------
    # Stage 2: compute sliding-window density
    # ------------------------------------------------------------------
    print("[pipeline] Stage 2: computing window density ...")
    window_density = compute_window_density(
        aligned,
        window_hours=window_hours,
        stride_hours=stride_hours,
        density_threshold=density_threshold,
    )

    # ------------------------------------------------------------------
    # Stage 3: discard invalid windows
    # ------------------------------------------------------------------
    print("[pipeline] Stage 3: discarding invalid windows ...")
    good_windows_all = discard_invalid_windows(window_density)

    # ------------------------------------------------------------------
    # Stage 4: compute theoretical maximum windows per (distname, kpi_id)
    # Derived from the aligned DataFrame (row count = series length in hours).
    # ------------------------------------------------------------------
    print("[pipeline] Stage 4: computing theoretical window maxima ...")
    theoretical_max = compute_theoretical_max_windows(
        aligned,
        window_hours=window_hours,
        stride_hours=stride_hours,
    )

    # ------------------------------------------------------------------
    # Stage 5: compute per-KPI yield statistics
    # Cache good_windows_all before the aggregation; reused in Stage 7.
    # total_distinct_cells is computed once here and passed as a scalar.
    # ------------------------------------------------------------------
    print("[pipeline] Stage 5: computing per-KPI yield statistics ...")
    total_distinct_cells = raw_df.select("distname").distinct().count()
    good_windows_all.cache()

    kpi_stats = compute_kpi_yield_stats(
        good_windows_all,
        theoretical_max,
        total_distinct_cells=total_distinct_cells,
    )

    # ------------------------------------------------------------------
    # Stage 6: pre-filter
    # ------------------------------------------------------------------
    print("[pipeline] Stage 6: pre-filtering KPIs ...")
    candidates = prefilter_kpis(
        kpi_stats,
        min_window_coverage_frac=min_window_coverage_frac,
        min_frac_contributing_cells=min_frac_contributing_cells,
    )
    print(f"[pipeline]   {len(candidates)} candidates passed pre-filter.")

    # ------------------------------------------------------------------
    # Compute theoretical_max_joint: bottleneck KPI sets the ceiling for
    # joint windows.  Use the minimum per-KPI theoretical max across candidates.
    # ------------------------------------------------------------------
    theoretical_max_joint = (
        theoretical_max.filter(f.col("kpi_id").isin(candidates))
        .groupBy("kpi_id")
        .agg(f.sum("theoretical_max_windows").alias("kpi_max"))
        .agg(f.min("kpi_max").alias("joint_max"))
        .collect()[0]["joint_max"]
    )
    print(f"[pipeline]   theoretical_max_joint = {theoretical_max_joint:,}")

    # ------------------------------------------------------------------
    # Build the cached greedy-loop DataFrame:
    # candidates only, good windows only, density column dropped.
    # ------------------------------------------------------------------
    good_windows_cached = good_windows_all.filter(f.col("kpi_id").isin(candidates)).drop(
        "window_valid_frac", "is_good_window"
    )
    good_windows_all.unpersist()  # release full-KPI cache
    good_windows_cached.cache()
    good_windows_cached.count()  # materialise eagerly before the greedy loop

    # ------------------------------------------------------------------
    # Stage 7: greedy joint KPI selection
    # ------------------------------------------------------------------
    print("[pipeline] Stage 7: running greedy joint KPI selection ...")
    selected_kpis = greedy_joint_kpi_selection(
        good_windows_cached,
        candidates,
        theoretical_max_joint,
        min_joint_coverage_frac=min_joint_coverage_frac,
    )

    print(f"[pipeline] Done. Selected {len(selected_kpis)} KPIs from {len(candidates)} candidates.")

    return selected_kpis, good_windows_cached


# ---------------------------------------------------------------------------
# Usage example (not executed on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from genpm.preprocessing.kpi_coverage import run_kpi_selection_pipeline
    from genpm.utils.consts import SHARED_DIR_PATH, SPARK_CONFIGS
    from genpm.utils.utils import SparkDataManager

    sdm = SparkDataManager(SPARK_CONFIGS["FULL_HEAVY"])

    PREPROCESSED_DATASET_PATH = SHARED_DIR_PATH / "preprocessed_dataset"
    # ── Load your raw long-format DataFrame here ────────────────────────────
    # raw_df = spark.read.parquet("s3a://your-bucket/raw_kpi_data/")
    pm_df = sdm.read_parquet(PREPROCESSED_DATASET_PATH / "pm_data_long")
    # ── Run the pipeline ────────────────────────────────────────────────────
    selected_kpis, cached_df = run_kpi_selection_pipeline(
        pm_df,  # type: ignore[name-defined]
        # alignment
        freq_hours=1,
        # windowing
        window_hours=168,
        stride_hours=24,
        density_threshold=0.875,
        # pre-filter (fraction-based)
        min_window_coverage_frac=0.20,
        min_frac_contributing_cells=0.30,
        # greedy (fraction-based)
        min_joint_coverage_frac=0.10,
    )

    print("Selected KPIs:", selected_kpis)

    # ── Release the cache when downstream training is bootstrapped ──────────
    cached_df.unpersist()

    sdm.write_parquet(cached_df, SHARED_DIR_PATH / "tmp" / "TEST_GREEDY")
