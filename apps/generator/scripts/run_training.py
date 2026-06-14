"""Quick-run training with hardcoded dev values.

Usage:  python scripts/run_training.py
        (from the repo root, no install needed)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genpm.modelling.configs import TrainConfig
from genpm.modelling.train import run_training
from genpm.utils.consts import SHARED_DIR_PATH

# ── Paths ─────────────────────────────────────────────────────────────────────

# New runs go under model_runs/ — the shared artifacts/ dir is not user-writable.
TRAINING_DATA_PATH = SHARED_DIR_PATH / "preprocessed_dataset" / "final_pmcm"
RUN_DIR = SHARED_DIR_PATH / "model_runs" / "run_6_dummy"
WEIGHTS_PATH = RUN_DIR / "models_weights_debug" / "cvae_lstm_v6_0.weights.h5"

# ── Config ────────────────────────────────────────────────────────────────────

cfg = TrainConfig(
    training_data_path=str(TRAINING_DATA_PATH / "pm_df_wide_materialized_windows"),
    run_dir_path=str(RUN_DIR),
    weights_path=str(WEIGHTS_PATH),
    # HP_V5 defaults are used unless overridden below
    # epochs=100,
    # target_beta=2e-4,
    # hidden_dim=256,
)

if __name__ == "__main__":
    run_training(cfg)
