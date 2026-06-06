import json

from pyspark.sql import DataFrame

from genpm.raw_vis.data_vis_utils import (
    basic_info,
    kpi_bts_coverage,
    kpi_catalog,
    plot_kpi_timeline,
    schema,
)


def make_summary_json(raw_df: DataFrame, output_path: str = "data_summary.json") -> None:
    """
    Writes a dataset summary JSON to *output_path*.

    Keys:
    - schema: column types and null percentages
    - basic_info: row/KPI/BTS/distname counts and date range
    - kpi_catalog: per-KPI statistics (count, min/max/mean/std, null %)
    - kpi_bts_coverage_heatmap: JSON-serialised Plotly heatmap of KPI/BTS presence
    """
    summary = {
        "schema": schema(raw_df).to_dict(orient="records"),
        "basic_info": basic_info(raw_df).to_dict(orient="records")[0],
        "kpi_catalog": kpi_catalog(raw_df).to_dict(orient="records"),
        "kpi_bts_coverage_heatmap": kpi_bts_coverage(raw_df),
    }
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, ensure_ascii=False, default=str)


def make_kpi_analysis(
    raw_df: DataFrame,
    kpi_list: list[str],
    output_path: str = "kpi_analysis.json",
) -> None:
    """
    Writes a per-KPI analysis JSON to *output_path*.

    Keys:
    - kpi_list: KPI IDs included in this run
    - kpi_plots: mapping from KPI ID to JSON-serialised Plotly figure
                 (timeline + rolling mean + Pettitt change-point + distribution fits)
    """
    kpi_plots = {}
    for kpi in kpi_list:
        fig = plot_kpi_timeline(raw_df, kpi)
        kpi_plots[kpi] = json.loads(fig.to_json())

    analysis = {
        "kpi_list": kpi_list,
        "kpi_plots": kpi_plots,
    }
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(analysis, f_out, ensure_ascii=False, default=str)
