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
RUN_DIR = SHARED_DIR_PATH / "model_runs" / "run_9"
WEIGHTS_PATH = RUN_DIR / "models_weights_debug" / "cvae_lstm_v6_200.weights.h5"

# ── Config ────────────────────────────────────────────────────────────────────

cfg = TrainConfig(
    training_data_path=str(TRAINING_DATA_PATH / "pm_df_wide_materialized_windows"),
    run_dir_path=str(RUN_DIR),
    weights_path=str(WEIGHTS_PATH),
    # v6 arch: X-only encoder + z tiled at every decoder step — collapse is now structural
    # impossibility, not a tuning problem.  HP_V6 defaults apply unless overridden.
    epochs=200,
    target_beta=1e-3,
    free_bits_global=0.002,  # back to original — z will naturally exceed the floor
    cycle_epochs=30,
    n_cycles=6,  # 30*6=180 epochs of cycling, last 20 at max beta
    learning_rate=3e-4,
)

if __name__ == "__main__":
    run_training(cfg)
