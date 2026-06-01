import json

from pyspark.sql import DataFrame

from raw_vis.data_vis_utils import (
    basic_info,
    kpi_bts_coverage,
    kpi_catalog,
    plot_kpi_timeline,
    schema,
)
from utils.consts import RAW_DATASET_PATH
from utils.utils import SparkDataManager

sdm = SparkDataManager()

raw_df = sdm.read_parquet(RAW_DATASET_PATH)


def make_summary_json(raw_df: DataFrame):
    """
    makes summary in json:
    - schema (df): dataset's schema and columns null %
    - basic_info (df): rows, kpi, bts and distname count, start and end date
    - kpi_catalog (df): some basic info for every kpi
    - kpi_bts_coverage_heatmap (plotly fig): shows kpi bts coverage
    """
    summary = {
        "schema": schema(raw_df).to_dict(orient="records"),
        "basic_info": basic_info(raw_df).to_dict(orient="records")[0],
        "kpi_catalog": kpi_catalog(raw_df).to_dict(orient="records"),
        "kpi_bts_coverage_heatmap": kpi_bts_coverage(raw_df),
    }
    with open("data_summary.json", "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, ensure_ascii=False, default=str)


def make_kpi_analysis(raw_df: DataFrame, kpi_list: list[str]):
    kpi_plots = {}
    for kpi in kpi_list:
        fig = plot_kpi_timeline(raw_df, kpi)
        kpi_plots[kpi] = json.loads(fig.to_json())

    analysis = {
        "kpi_list": kpi_list,
        "kpi_plots": kpi_plots,
    }
    with open("kpi_analysis.json", "w", encoding="utf-8") as f_out:
        json.dump(analysis, f_out, ensure_ascii=False, default=str)
