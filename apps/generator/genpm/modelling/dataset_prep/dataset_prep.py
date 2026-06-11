"""Model-side dataset preparation.

Reconstructs the K×W training windows that preprocessing deliberately leaves
un-materialised.  ``genpm.preprocessing.logic.kpi_coverage.emit_window_index``
returns the long PM data with a ``window_anchor`` column that is non-null only on
the rows that begin a window; the actual overlapping windows are rebuilt here, one
at a time, so the 7× overlap never costs storage upstream.

The reconstruction runs on the driver in polars: the marked frame is pulled across
once, then every window is a cheap columnar slice + pivot.  No Spark work happens
per window.
"""

from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import polars as pl
from pyspark.sql import DataFrame


def generate_window_tensors(
    windowed_pm_df: DataFrame,
    *,
    window_hours: int = 168,
) -> Iterator[dict[str, Any]]:
    """Yield one ``K × W`` window tensor per joint window anchor.

    Parameters
    ----------
    windowed_pm_df : pyspark.sql.DataFrame
        Output of ``emit_window_index`` — long PM data (one row per
        (distname, kpi_id, start_time)) with a ``window_anchor`` column that is
        non-null only on anchor-start rows.  It already carries both the anchor
        marks (the non-null rows) and every hourly value needed to fill a window,
        so it is the only input required.
    window_hours : int
        Window width W in hours (default 168).

    Yields
    ------
    dict
        One dict per (distname, window_anchor):
          ``distname``      — the cell.
          ``window_anchor`` — the window's start timestamp.
          ``kpi_ids``       — the K channel order (matches the tensor columns).
          ``tensor``        — ``np.ndarray`` of shape ``(window_hours, K)``;
                              transpose for channels-first ``K × W``.  Hours with
                              no row for a KPI are NaN (should not occur for a
                              joint-complete anchor, but the spine join makes the
                              W axis explicit rather than implicit).
    """
    # Single Spark -> driver hop; everything after is polars on the driver.
    pdf = pl.from_pandas(windowed_pm_df.toPandas())

    # Window starts are exactly the non-null marks (one row per joint window).
    anchors = (
        pdf.filter(pl.col("window_anchor").is_not_null())
        .select("distname", "window_anchor")
        .unique()
        .sort(["distname", "window_anchor"])
    )

    # Pre-split by cell so each window only scans its own distname's rows.
    by_cell = pdf.partition_by("distname", as_dict=True)

    # Complete 0..W-1 hour axis, reused for every window.
    spine = pl.DataFrame({"hour_idx": pl.arange(0, window_hours, eager=True)})

    for anchor_row in anchors.iter_rows(named=True):
        distname = anchor_row["distname"]
        anchor = anchor_row["window_anchor"]
        end = anchor + timedelta(hours=window_hours)

        cell = by_cell[(distname,)]
        window = cell.filter(
            (pl.col("start_time") >= anchor) & (pl.col("start_time") < end)
        ).with_columns(
            ((pl.col("start_time") - anchor).dt.total_hours()).cast(pl.Int32).alias("hour_idx")
        )

        wide = window.pivot(on="kpi_id", index="hour_idx", values="kpi_value")
        # Force a full, ordered 0..W-1 axis regardless of which hours were present.
        wide = spine.join(wide, on="hour_idx", how="left").sort("hour_idx")

        kpi_cols = [c for c in wide.columns if c != "hour_idx"]
        yield {
            "distname": distname,
            "window_anchor": anchor,
            "kpi_ids": kpi_cols,
            "tensor": wide.select(kpi_cols).to_numpy(),  # (W, K)
        }
