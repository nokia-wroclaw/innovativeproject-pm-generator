"""Quick-run data similarity check: real PM-CM data vs generated output (cVAE run).

Real data:  preprocessed_dataset/final_pmcm/pm_df_wide_indexed_winds
Synth data: generated/config_NR2500(B41)_100MHz_1900(B25)_100_TDD_both_2024-01-15.parquet

Usage:  python scripts/run_data_similarity.py
        (from the repo root, no install needed)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402

from genpm.data_similarity.configs import DataSimilarityConfig  # noqa: E402
from genpm.data_similarity.run import run_data_similarity  # noqa: E402
from genpm.utils.consts import SHARED_DIR_PATH  # noqa: E402
from genpm.utils.spark_session import SparkDataManager  # noqa: E402

RUN_DIR = SHARED_DIR_PATH / "model_runs" / "run_6_dummy"
_cmap = joblib.load(RUN_DIR / "cell_config_map.pkl")

REAL_DATA_PATH = str(
    SHARED_DIR_PATH / "preprocessed_dataset" / "final_pmcm" / "pm_df_wide_indexed_winds"
)
SYNTH_DATA_PATH = str(
    SHARED_DIR_PATH
    / "generated"
    / "config_NR2500(B41)_100MHz_1900(B25)_100_TDD_both_2024-01-15.parquet"
)
OUTPUT_PATH = str(SHARED_DIR_PATH / "data_similarity_output")

# Config values in the same order as config_cols from the artifact (must match generation).
_CONFIG_VALUES = {
    "[CELL] 5gCellDeploymentTypeSaNsa": "NR2500(B41)_100MHz",
    "[CELL] CellDuplexMode": "1900(B25)",
    "[CELL] ChannelBandwidth": "100",
    "[CELL] FrequencyBand": "TDD",
    "[CELL] FrequencyBandAndBandwidth": "both",
}
CELL_CONFIG_COLS = _cmap["config_cols"]
CELL_CONFIGS = [_CONFIG_VALUES[col] for col in CELL_CONFIG_COLS]

# Hardcoded for dev; in the real flow these KPI ids come from the frontend request.
KPI_COLS = [
    "NR_6",
    "NR_11",
    "NR_46",
    "NR_47",
    "NR_125",
]

cfg = DataSimilarityConfig(
    output_path_prefix=OUTPUT_PATH,
    real_data_path=REAL_DATA_PATH,
    synth_data_path=SYNTH_DATA_PATH,
    real_ts_col="start_time",
    synth_ts_col="timestamp",
    ts_col="ts",
    single_kpi_cols=KPI_COLS,
    multi_kpi_cols=KPI_COLS,
    cell_config_cols=CELL_CONFIG_COLS,
    cell_configs=CELL_CONFIGS,
    save_summary_json=True,
)

if __name__ == "__main__":
    with SparkDataManager() as sdm:
        summary = run_data_similarity(sdm, cfg)
        print("Done. Single-KPI keys:", list(summary["single_kpi"].keys()))
