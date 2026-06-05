"""Quick-run raw_vis with hardcoded dev values.

Usage:  python scripts/run_raw_vis.py
        (from the repo root, no install needed)
"""

import sys
from pathlib import Path

# make `import genpm` work without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genpm.raw_vis.configs import RawVisConfig
from genpm.raw_vis.data_visualisation import make_kpi_analysis, make_summary_json
from genpm.utils.consts import SHARED_DIR_PATH, SPARK_CONFIGS
from genpm.utils.utils import SparkDataManager


def run_summary(sdm: SparkDataManager) -> None:
    cfg = RawVisConfig(
        raw_pm_data_path=str(SHARED_DIR_PATH / "raw_data" / "pm_data"),
        output_path=str(SHARED_DIR_PATH / "raw_vis_output" / "data_summary.json"),
    )
    raw_df = sdm.read_parquet(cfg.raw_pm_data_path)
    make_summary_json(raw_df, cfg.output_path)


def run_kpi_analysis(sdm: SparkDataManager) -> None:
    cfg = RawVisConfig(
        raw_pm_data_path=str(SHARED_DIR_PATH / "raw_data" / "pm_data"),
        output_path=str(SHARED_DIR_PATH / "raw_vis_output" / "kpi_analysis.json"),
    )
    kpi_list = []  # replace with desired KPI IDs
    raw_df = sdm.read_parquet(cfg.raw_pm_data_path)
    make_kpi_analysis(raw_df, kpi_list, cfg.output_path)


if __name__ == "__main__":
    sdm = SparkDataManager(SPARK_CONFIGS["HALF_SAFE"])
    # Uncomment the needed function
    run_summary(sdm)
    # run_kpi_analysis(sdm)
