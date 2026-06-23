"""Quick-run conditional DDPM (diffusion) training with hardcoded dev values.

Usage:  python scripts/run_diffusion_training.py
        (from the repo root, no install needed)

First diffusion iteration on the same windowed dataset the cVAE runs use.
Epsilon-prediction DDPM with a dilated-Conv1D + FiLM denoiser, conditioned on the
config/calendar vector. The single training metric (noise-prediction MSE) is a
genuine quality signal, so <run>/training_losses.png should show it decreasing and
plateauing; the best-by-loss checkpoint is the one generation loads.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genpm.modelling.configs import DiffusionTrainConfig
from genpm.modelling.train import run_diffusion_training
from genpm.utils.consts import SHARED_DIR_PATH

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAINING_DATA_PATH = SHARED_DIR_PATH / "preprocessed_dataset" / "final_pmcm_yj"
# diffusion_run_4: run-3 architecture (cosine + thresholding + EMA, width 384/16 blocks)
# PLUS per-timestep calendar conditioning (day-of-week, weekend, holiday, holiday-eve,
# long-weekend via the `holidays` US calendar). Targets the over-regular-days residual
# (R2): the model can finally tell weekday/weekend/holiday apart within a window.
RUN_DIR = SHARED_DIR_PATH / "model_runs" / "diffusion_run_5_yj"
WEIGHTS_PATH = RUN_DIR / "models_weights_debug" / "ddpm_5_yj.weights.h5"

# ── Config ────────────────────────────────────────────────────────────────────
cfg = DiffusionTrainConfig(
    training_data_path=str(TRAINING_DATA_PATH / "pm_df_wide_materialized_windows"),
    run_dir_path=str(RUN_DIR),
    weights_path=str(WEIGHTS_PATH),
    num_timesteps=1000,
    beta_schedule="cosine",  # keep — the run-2 sampler fix depends on it (+ thresholding)
    width=384,  # proven in run-3 (≥ feat_dim 248)
    n_blocks=16,  # ~17.6M params
    dilation_cycle=(1, 2, 4, 8, 16, 32, 64),
    time_embed_dim=128,
    cond_embed_dim=128,
    learning_rate=2e-4,
    use_ema=True,
    ema_momentum=0.999,
    use_calendar=True,  # NEW: per-timestep calendar conditioning (6 channels)
    calendar_country="US",
    # run-3 loss plateaued by ~ep300 (LR→min by ~512), so 780 was wasteful. 450 epochs
    # (~5.7h @ ~45s/epoch) is past convergence with EMA headroom. Watch lag-24 AC on
    # daily KPIs drop toward real (~0.33) and weekday/weekend split appear.
    epochs=400,
    batch_size=64,
)

if __name__ == "__main__":
    run_diffusion_training(cfg)
