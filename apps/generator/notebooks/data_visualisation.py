import json

from notebooks.data_vis_utils import (
    basic_info,
    fig_to_base64,
    kpi_bts_coverage,
    kpi_catalog,
    schema,
)
from notebooks.kpi_distribution_utils import analyze_kpi
from utils.consts import SHARED_DIR_PATH
from utils.utils import SparkDataManager

sdm = SparkDataManager()

RAW_DATASET_PATH = SHARED_DIR_PATH / "eda_data/raw_pm_data"


def make_summary_json(data_path):
    """
    makes summary in json:
    - schema (df): dataset's schema and columns null %
    - basic_info (df): rows, kpi, bts and distname count, start and end date
    - kpi_catalog (df): some basic info for every kpi
    - kpi_bts_coverage_heatmap (plotly fig): shows kpi bts coverage

    this process takes around 3-4 minutes to complete for the nokia raw dataset
    """
    raw_df = sdm.read_parquet(data_path)
    summary = {
        "schema": schema(raw_df).to_dict(orient="records"),
        "basic_info": basic_info(raw_df).to_dict(orient="records")[0],
        "kpi_catalog": kpi_catalog(raw_df).to_dict(orient="records"),
        "kpi_bts_coverage_heatmap": kpi_bts_coverage(raw_df),
    }
    with open("data_summary.json", "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, ensure_ascii=False, default=str)


def make_kpi_analysis(data_path, kpi_list: list[str]):
    """
    Generuje JSON z wykresami (PNG base64) dla wybranych KPI.
    """
    raw_df = sdm.read_parquet(data_path)

    kpi_plots = {}

    for i, kpi_id in enumerate(kpi_list, 1):
        print(f"[{i}/{len(kpi_list)}] {kpi_id}...")
        try:
            result = analyze_kpi(raw_df, kpi_id)

            # konwertuj fig
            kpi_plots[kpi_id] = fig_to_base64(result["figure"])

        except Exception as e:
            print(f"{kpi_id} failed: {e}")

    analysis = {
        "kpi_list": kpi_list,
        "kpi_plots": kpi_plots,
    }
    with open("kpi_analysis.json", "w", encoding="utf-8") as f_out:
        json.dump(analysis, f_out, ensure_ascii=False, default=str)
