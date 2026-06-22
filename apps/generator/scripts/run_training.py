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
RUN_DIR = SHARED_DIR_PATH / "model_runs" / "run_18"
WEIGHTS_PATH = RUN_DIR / "models_weights_debug" / "cvae_lstm_v7_55.weights.h5"

# ── Config ────────────────────────────────────────────────────────────────────
# run_18 = run_17's exact config, with the now-complete checkpoint-delay fix
# (monitor_start_epoch waits for the full first ramp, not just kl_delay_epochs;
# plus the new unconditional "*_last.weights.h5" fallback checkpoint), shortened
# to 55 epochs while keeping run_17's delay/cycle/hold proportions:
#   run_17 (75 epochs): delay=30 (40%), cycle=30 (40%), extra hold=15 (20%)
#   run_18 (55 epochs): delay=22 (40%), cycle=22 (40%), extra hold=11 (20%)
# i.e. kl_delay_epochs and cycle_epochs both scaled 30 -> 22 (cycle_ratio=0.5
# unchanged), n_cycles=1 (at least one full cycle, as requested). monitor_start_epoch
# will be 22 + ceil(22*0.5) = 33, leaving epochs 33-54 (22 epochs, 40%) for
# "best" checkpoint tracking — same 40% proportion run_17 had (45-74 of 75).
# global_latent_dim=128, corr_l2=5e-5, ac_weight=0.3 unchanged from run_17.
cfg = TrainConfig(
    training_data_path=str(TRAINING_DATA_PATH / "pm_df_wide_materialized_windows"),
    run_dir_path=str(RUN_DIR),
    weights_path=str(WEIGHTS_PATH),
    # v7 arch: X-only encoder + z tiled + no PE in decoder + cross-KPI correlation
    # + autocorrelation penalty in the ELBO.
    arch_version="v7",
    epochs=55,
    target_beta=1e-3,
    free_bits_global=0.002,
    use_cyclical_kl=True,
    kl_delay_epochs=22,  # beta=0 exactly through epoch 21, then cycling begins
    cycle_epochs=22,
    n_cycles=1,  # 1 * 22 = 22 epochs cycling (22-43), then hold target_beta 44-54
    cycle_ratio=0.5,
    learning_rate=3e-4,
    # v7-specific: autocorrelation penalty weight and number of lags to match.
    ac_weight=0.3,
    ac_max_lag=24,  # one full diurnal cycle
    # v7-specific: L2 on the CrossKPICorrelation F×F kernel. Unchanged from run_17.
    corr_l2=5e-5,
    # Capacity increase, unchanged from run_17.
    global_latent_dim=128,
)

if __name__ == "__main__":
    run_training(cfg)
