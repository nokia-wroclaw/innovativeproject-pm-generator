from __future__ import annotations

import json
from typing import Any

from pyspark.sql import DataFrame

from genpm.raw_vis.data_vis_utils import (
    basic_info,
    kpi_bts_coverage,
    kpi_catalog,
    plot_kpi_timeline,
    schema,
)
from genpm.raw_vis.pm_schema import (
    normalize_pm_dataframe,
    required_columns,
    unsupported_schema_payload,
    validate_pm_schema,
)

MAX_CATALOG_ROWS = 200
MAX_KPI_TIMELINE_PLOTS = 5


def make_summary(raw_df: DataFrame, *, spark_version: str | None = None) -> dict[str, Any]:
    """Build visualization summary payload for S3 and the API."""
    raw_df = normalize_pm_dataframe(raw_df)
    ok, missing = validate_pm_schema(raw_df)
    if not ok:
        payload = unsupported_schema_payload(missing)
        if spark_version:
            payload["spark_version"] = spark_version
        return payload

    coverage_result = kpi_bts_coverage(raw_df)
    catalog_df = kpi_catalog(raw_df)
    catalog_truncated = len(catalog_df) > MAX_CATALOG_ROWS
    catalog_records = catalog_df.head(MAX_CATALOG_ROWS).to_dict(orient="records")

    summary: dict[str, Any] = {
        "status": "success",
        "schema": schema(raw_df).to_dict(orient="records"),
        "basic_info": basic_info(raw_df).to_dict(orient="records")[0],
        "kpi_catalog": catalog_records,
        "kpi_catalog_meta": {
            "truncated": catalog_truncated,
            "total_kpis": int(len(catalog_df)),
            "max_rows": MAX_CATALOG_ROWS,
        },
        "kpi_bts_coverage": {
            "z": coverage_result["z"],
            "x": coverage_result["x"],
            "y": coverage_result["y"],
        },
        "coverage_meta": {
            "truncated": False,
            "bts_count": coverage_result.get("bts_count"),
            "kpi_count": coverage_result.get("kpi_count"),
        },
        "required_columns": list(required_columns()),
    }
    if spark_version:
        summary["spark_version"] = spark_version
    if catalog_truncated:
        summary["catalog_warning"] = (
            f"KPI catalog truncated to top {MAX_CATALOG_ROWS} KPIs by record count "
            f"(of {len(catalog_df)} total)."
        )
    return summary


def make_summary_json(raw_df: DataFrame, output_path: str = "data_summary.json") -> None:
    """CLI helper: write make_summary() payload to a local JSON file."""
    summary = make_summary(raw_df)
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, ensure_ascii=False, default=str)


def top_kpis_for_analysis(
    summary: dict[str, Any], *, limit: int = MAX_KPI_TIMELINE_PLOTS
) -> list[str]:
    """Top KPIs by record count (same order as kpi_catalog in make_summary)."""
    catalog = summary.get("kpi_catalog") or []
    return [str(row["kpi_id"]) for row in catalog[:limit] if row.get("kpi_id") is not None]


def make_kpi_analysis(
    raw_df: DataFrame,
    kpi_list: list[str],
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Per-KPI timeline + distribution plots (Plotly JSON).
    Optionally writes to *output_path* for local CLI usage.
    """
    kpi_plots: dict[str, Any] = {}
    for kpi in kpi_list:
        try:
            fig = plot_kpi_timeline(raw_df, kpi)
            kpi_plots[kpi] = json.loads(fig.to_json())
        except ValueError as exc:
            kpi_plots[kpi] = {"error": str(exc)}

    analysis = {
        "kpi_list": kpi_list,
        "kpi_plots": kpi_plots,
    }
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(analysis, f_out, ensure_ascii=False, default=str)
    return analysis
