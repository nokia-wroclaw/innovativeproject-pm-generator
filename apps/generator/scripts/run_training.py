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

TRAINING_DATA_PATH = SHARED_DIR_PATH / "preprocessed_dataset" / "final_pmcm"
RUN_DIR = SHARED_DIR_PATH / "model_runs" / "run_10"
WEIGHTS_PATH = RUN_DIR / "models_weights_debug" / "cvae_lstm_v7_0.weights.h5"

# ── Config ────────────────────────────────────────────────────────────────────

cfg = TrainConfig(
    training_data_path=str(TRAINING_DATA_PATH / "pm_df_wide_materialized_windows"),
    run_dir_path=str(RUN_DIR),
    weights_path=str(WEIGHTS_PATH),
    # v7 arch: X-only encoder + z tiled + no PE in decoder + cross-KPI correlation
    # + autocorrelation penalty in the ELBO.
    arch_version="v7",
    epochs=200,
    target_beta=1e-3,
    free_bits_global=0.002,
    cycle_epochs=30,
    n_cycles=6,  # 30*6=180 epochs of cycling, last 20 at max beta
    learning_rate=3e-4,
    # v7-specific: autocorrelation penalty weight and number of lags to match.
    # Increase ac_weight (e.g. 0.5) if AC mismatch persists; decrease if
    # reconstruction quality drops.
    ac_weight=0.1,
    ac_max_lag=24,  # one full diurnal cycle
)

if __name__ == "__main__":
    run_training(cfg)
