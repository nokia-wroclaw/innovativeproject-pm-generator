from functools import reduce

from dotenv import load_dotenv
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from .logger import get_logger

load_dotenv()

logger = get_logger()


def when_chained(conditions: list[tuple], otherwise=None) -> Column:
    """Build a chained when().when()...otherwise() Column from a list of (condition, value) pairs."""

    def _reducer(acc, pair):
        cond, val = pair
        return acc.when(cond, val)

    first_cond, first_val = conditions[0]
    chain = reduce(_reducer, conditions[1:], f.when(first_cond, first_val))
    return chain.otherwise(otherwise)


def validate_windowed_pm(
    df: DataFrame,
    window_hours: int = 168,
) -> None:
    """Validate the output of ``emit_window_index``.

    The windows are **no longer materialised**.  ``emit_window_index`` returns the
    long PM data — one row per (distname, kpi_id, start_time), no row duplication —
    with a ``window_anchor`` column that is non-null **only on the rows that begin a
    window** and null everywhere else.  There is no ``hour_idx`` column; the K×W
    windows are reconstructed lazily downstream by slicing [anchor, anchor+W).

    This validator therefore checks two things:

      * cheap structural invariants directly on the marked long frame (shape,
        nulls, anchor-mark integrity, KPI consistency, windows per distname), and
      * a **validation-only reconstruction**: it range-joins each anchor back over
        the long data (the very slice the dataloader will do) and confirms every
        window resolves to a full, gap-free, joint-complete W×K block.

    The reconstruction re-materialises windows transiently for the check only — it
    is a diagnostic, not part of the production path.  All findings are logged; no
    exceptions are raised.
    """

    logger.info("=" * 60)
    logger.info("[validate] starting windowed dataframe validation")
    logger.info("=" * 60)

    # ── 1. Basic shape (long, un-materialised frame) ───────────────────────
    total_rows = df.count()
    n_distnames = df.select("distname").distinct().count()
    n_kpis = df.select("kpi_id").distinct().count()
    n_marked = df.filter(f.col("window_anchor").isNotNull()).count()
    n_anchors = (
        df.filter(f.col("window_anchor").isNotNull())
        .select("distname", "window_anchor")
        .distinct()
        .count()
    )

    logger.info(f"[shape] total long rows: {total_rows:>15,}")
    logger.info(f"[shape] distnames:       {n_distnames:>15,}")
    logger.info(f"[shape] kpis:            {n_kpis:>15,}")
    logger.info(f"[shape] unique windows:  {n_anchors:>15,}")
    logger.info(f"[shape] anchor marks:    {n_marked:>15,}  (non-null window_anchor rows)")
    logger.info(
        "[shape] ℹ️ rows are the long data as-is — no windows × kpis × W "
        "materialisation expected here"
    )

    # ── 2. Null check ──────────────────────────────────────────────────────
    # kpi_value / imputed_flag must be populated (data inside windows is real or
    # imputed).  window_anchor is INTENTIONALLY null off the anchor rows, so its
    # nulls are reported as fill-rate, never flagged as an error.
    n_null_values = df.filter(f.col("kpi_value").isNull()).count()
    n_null_flags = df.filter(f.col("imputed_flag").isNull()).count()

    if n_null_values > 0:
        logger.warning(f"[nulls] ❌ kpi_value nulls: {n_null_values:,}")
    else:
        logger.info("[nulls] ✅ no kpi_value nulls")

    if n_null_flags > 0:
        logger.warning(f"[nulls] ❌ imputed_flag nulls: {n_null_flags:,}")
    else:
        logger.info("[nulls] ✅ no imputed_flag nulls")

    logger.info(
        f"[nulls] ℹ️ window_anchor marked on {n_marked:,}/{total_rows:,} rows "
        f"({100 * n_marked / total_rows:.2f}%) — the rest null by design"
    )

    # ── 3. Anchor-mark integrity ──────────────────────────────────────────
    # A mark sits on the row whose timestamp IS the window start, so every
    # non-null window_anchor must equal that row's start_time.
    n_bad_marks = df.filter(
        f.col("window_anchor").isNotNull() & (f.col("window_anchor") != f.col("start_time"))
    ).count()
    if n_bad_marks > 0:
        logger.warning(
            f"[marks] ❌ {n_bad_marks:,} rows have window_anchor != start_time "
            f"— mark is not on the window-start row"
        )
    else:
        logger.info("[marks] ✅ every window_anchor mark sits on its own start_time")

    # ── 4. Window reconstruction completeness (validation-only) ───────────
    # Slice each anchor back over the long data exactly as the dataloader will,
    # compute hour_idx, and confirm each (distname, anchor, kpi_id) resolves to a
    # full, contiguous 0..W-1 span — and that every selected KPI is present
    # (joint-complete), so the rebuilt tensor is never ragged.
    anchors = (
        df.filter(f.col("window_anchor").isNotNull())
        .select("distname", "window_anchor")
        .distinct()
        .withColumn(
            "window_end",
            f.col("window_anchor") + f.expr(f"INTERVAL {window_hours} HOURS"),
        )
    )
    reconstructed = (
        df.alias("p")
        .join(
            anchors.alias("a"),
            on=[
                f.col("p.distname") == f.col("a.distname"),
                f.col("p.start_time") >= f.col("a.window_anchor"),
                f.col("p.start_time") < f.col("a.window_end"),
            ],
            how="inner",
        )
        .withColumn(
            "hour_idx",
            (
                (f.col("p.start_time").cast("long") - f.col("a.window_anchor").cast("long")) / 3600
            ).cast("integer"),
        )
        .select("a.distname", "a.window_anchor", "p.kpi_id", "hour_idx", "p.kpi_value")
    )

    hour_stats = reconstructed.groupBy("distname", "window_anchor", "kpi_id").agg(
        f.count("*").alias("n_hours"),
        f.min("hour_idx").alias("min_hour"),
        f.max("hour_idx").alias("max_hour"),
        f.sum(f.col("kpi_value").isNull().cast("int")).alias("n_null"),
    )

    bad_hour_counts = hour_stats.filter(f.col("n_hours") != window_hours)
    bad_hour_range = hour_stats.filter(
        (f.col("min_hour") != 0) | (f.col("max_hour") != window_hours - 1)
    )
    bad_nulls = hour_stats.filter(f.col("n_null") > 0)

    n_bad_counts = bad_hour_counts.count()
    n_bad_range = bad_hour_range.count()
    n_bad_nulls = bad_nulls.count()

    if n_bad_counts > 0:
        logger.warning(
            f"[recon] ❌ window·kpi blocks with != {window_hours} hours: {n_bad_counts:,}"
        )
        bad_hour_counts.orderBy("n_hours").show(10, truncate=False)
    else:
        logger.info(f"[recon] ✅ every window·kpi block resolves to exactly {window_hours} hours")

    if n_bad_range > 0:
        logger.warning(f"[recon] ❌ window·kpi blocks with wrong hour_idx range: {n_bad_range:,}")
        bad_hour_range.show(10, truncate=False)
    else:
        logger.info(f"[recon] ✅ every window·kpi block spans hour_idx 0 → {window_hours - 1}")

    if n_bad_nulls > 0:
        logger.warning(f"[recon] ❌ window·kpi blocks containing null kpi_value: {n_bad_nulls:,}")
    else:
        logger.info("[recon] ✅ no null kpi_value inside any reconstructed window")

    # ── 5. Joint-completeness: every window carries all the distname's KPIs ─
    # A reconstructed window must contain every KPI present in its cell, or the
    # K×W tensor is ragged across windows.
    kpis_per_distname = df.groupBy("distname").agg(f.countDistinct("kpi_id").alias("distname_kpis"))
    win_kpis = reconstructed.groupBy("distname", "window_anchor").agg(
        f.countDistinct("kpi_id").alias("win_kpis")
    )
    ragged = win_kpis.join(kpis_per_distname, on="distname").filter(
        f.col("win_kpis") != f.col("distname_kpis")
    )
    n_ragged = ragged.count()
    if n_ragged > 0:
        logger.warning(
            f"[kpis] ❌ {n_ragged:,} windows do not carry every KPI of their distname "
            f"— ragged tensors"
        )
        ragged.show(10, truncate=False)
    else:
        logger.info("[kpis] ✅ every window carries the full KPI set of its distname")

    # ── 6. Imputation flag distribution ───────────────────────────────────
    # flag=0: real observed value | flag=1: imputed by upstream per-KPI fill
    flag_dist = (
        df.groupBy("imputed_flag")
        .agg(f.count("*").alias("n_rows"))
        .withColumn("pct", f.round(f.col("n_rows") / total_rows * 100, 2))
        .orderBy("imputed_flag")
    )
    logger.info("[flags] imputed_flag distribution (long rows):")
    flag_dist.show(truncate=False)

    # ── 7. Windows per distname ────────────────────────────────────────────
    # Flags distnames with very few windows — underrepresented in CVAE training.
    windows_per_distname = (
        df.filter(f.col("window_anchor").isNotNull())
        .select("distname", "window_anchor")
        .distinct()
        .groupBy("distname")
        .agg(f.count("*").alias("n_windows"))
        .orderBy("n_windows")
    )

    stats = windows_per_distname.agg(
        f.min("n_windows").alias("min"),
        f.max("n_windows").alias("max"),
        f.mean("n_windows").alias("mean"),
        f.percentile_approx("n_windows", 0.10).alias("p10"),
        f.percentile_approx("n_windows", 0.50).alias("p50"),
    ).collect()[0]

    logger.info(
        f"[windows/distname] min={stats['min']} | p10={stats['p10']} | "
        f"p50={stats['p50']} | mean={stats['mean']:.1f} | max={stats['max']}"
    )

    sparse_distnames = windows_per_distname.filter(f.col("n_windows") < 30)
    n_sparse = sparse_distnames.count()
    if n_sparse > 0:
        logger.warning(
            f"[windows/distname] ❌ {n_sparse:,} distnames have fewer than 30 windows "
            f"— may be insufficient for CVAE training"
        )
        sparse_distnames.show(20, truncate=False)
    else:
        logger.info("[windows/distname] ✅ all distnames have >= 30 windows")

    logger.info("=" * 60)
    logger.info("[validate] done")
    logger.info("=" * 60)
