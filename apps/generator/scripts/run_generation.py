"""Quick-run generation with hardcoded dev values.

Usage:  python scripts/run_generation.py
        (from the repo root, no install needed)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from genpm.modelling.configs import GenerateConfig
from genpm.modelling.generate import run_generation
from genpm.utils.consts import SHARED_DIR_PATH

RUN_DIR = SHARED_DIR_PATH / "artifacts" / "run_4"
KPI_COLS = np.load(RUN_DIR / "kpi_columns.npy", allow_pickle=True).tolist()

cfg = GenerateConfig(
    run_dir_path=str(RUN_DIR),
    weights_path=str(RUN_DIR / "models_weights" / "cvae_lstm_v5_0.weights.h5"),
    output_path=str(SHARED_DIR_PATH / "generated"),
    cell_id="bts_24/cell_5",
    anchor_date="2024-01-15",
    n_weeks=3,
    kpi_list=KPI_COLS,
    holiday=0,
)

if __name__ == "__main__":
    run_generation(cfg)
