"""Quick-run generation with hardcoded dev values.

Usage:  python scripts/run_generation.py
        (from the repo root, no install needed)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from genpm.modelling.configs import GenerateConfig
from genpm.modelling.generate import run_generation
from genpm.utils.consts import SHARED_DIR_PATH

RUN_DIR = SHARED_DIR_PATH / "model_runs" / "run_10"
KPI_COLS = np.load(RUN_DIR / "kpi_columns.npy", allow_pickle=True).tolist()

# The config column order that cell_configs must follow, plus the valid value set
# per column (from the trained one-hot encoder). Print these to pick valid configs.
_cmap = joblib.load(RUN_DIR / "cell_config_map.pkl")
_encoder = joblib.load(RUN_DIR / "config_encoder.pkl")
print("config_cols (order for cell_configs):", _cmap["config_cols"])
print("valid values per column:", [list(c) for c in _encoder.categories_])

cfg = GenerateConfig(
    run_dir_path=str(RUN_DIR),
    weights_path=str(RUN_DIR / "models_weights_debug" / "cvae_lstm_v7_0.weights.h5"),
    output_path=str(SHARED_DIR_PATH / "generated"),
    # Generate by CONFIG (not cell_id): leave cell_id unset and pass values in the
    # same order as config_cols printed above, e.g. ["A", "X"]. Output is labeled
    # "config_<values>" instead of a cell_id.
    # Values keyed by column name — order is enforced by config_cols from the artifact.
    cell_configs=[
        {
            "[CELL] 5gCellDeploymentTypeSaNsa": "NR2500(B41)_100MHz",
            "[CELL] CellDuplexMode": "1900(B25)",
            "[CELL] ChannelBandwidth": "100",
            "[CELL] FrequencyBand": "TDD",
            "[CELL] FrequencyBandAndBandwidth": "both",
        }[col]
        for col in _cmap["config_cols"]
    ],
    anchor_date="2024-01-15",
    n_weeks=3,
    kpi_list=KPI_COLS,
    holiday=0,
)

if __name__ == "__main__":
    run_generation(cfg)
