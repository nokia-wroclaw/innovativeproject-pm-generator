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
TRAINING_DATA_PATH = SHARED_DIR_PATH / "preprocessed_dataset" / "final_pmcm"
# diffusion_run_3: scale-up of the working run-2 (cosine + per-step-thresholding
# sampler). Same proven architecture, just bigger + EMA + a long ~10h run.
#   width 256→384 (more capacity), blocks 12→16, + weight EMA (standard quality boost),
#   ~780 epochs (≈10h @ ~45s/epoch). Targets the run-2 polish: slight over-smoothing
#   (gen_ac1 0.72 vs 0.60) and mild under-dispersion (within_div 0.093 vs 0.141).
RUN_DIR = SHARED_DIR_PATH / "model_runs" / "diffusion_run_3"
WEIGHTS_PATH = RUN_DIR / "models_weights_debug" / "ddpm_3.weights.h5"

# ── Config ────────────────────────────────────────────────────────────────────
cfg = DiffusionTrainConfig(
    training_data_path=str(TRAINING_DATA_PATH / "pm_df_wide_materialized_windows"),
    run_dir_path=str(RUN_DIR),
    weights_path=str(WEIGHTS_PATH),
    num_timesteps=1000,
    beta_schedule="cosine",  # keep — the run-2 sampler fix depends on it (+ thresholding)
    width=384,  # 256→384, more capacity (comfortably ≥ feat_dim 248)
    n_blocks=16,  # 12→16 for depth (~17.6M params)
    dilation_cycle=(1, 2, 4, 8, 16, 32, 64),
    time_embed_dim=128,
    cond_embed_dim=128,
    learning_rate=2e-4,
    use_ema=True,  # EMA of weights → saved/generation use the averaged (better-sampling) weights
    ema_momentum=0.999,
    # ~10h: benchmarked ~45s/epoch for this 17.6M model. Watch gen_ac1 settle near
    # real_ac1 (~0.60, not over 0.72) and loss keep dropping below run-2's 0.063.
    epochs=780,
    batch_size=64,
)

if __name__ == "__main__":
    run_diffusion_training(cfg)
