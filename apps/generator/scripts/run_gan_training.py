"""Quick-run conditional WGAN-GP training with hardcoded dev values.

Usage:  python scripts/run_gan_training.py
        (from the repo root, no install needed)

First GAN iteration on the same windowed dataset the cVAE runs use. WGAN-GP +
config conditioning + per-hour feature matching. Inspect <run>/training_losses.png
afterwards: a healthy run shows w_dist settling to a small positive band and gp
hovering near 0 (its target is a unit-norm critic gradient).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genpm.modelling.configs import GANTrainConfig
from genpm.modelling.train import run_gan_training
from genpm.utils.consts import SHARED_DIR_PATH

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAINING_DATA_PATH = SHARED_DIR_PATH / "preprocessed_dataset" / "final_pmcm"
# gan_run_4: keeps run-3 (PE on, per-step noise, n_critic=3, moment anneal) and adds the
# two critic features that FORCE the generator to use its noise — minibatch-stddev
# (anti-collapse) and first-difference (anti-over-smoothing). Long run (~9h).
RUN_DIR = SHARED_DIR_PATH / "model_runs" / "gan_run_4"
WEIGHTS_PATH = RUN_DIR / "models_weights_debug" / "cgan_4.weights.h5"

# ── Config ────────────────────────────────────────────────────────────────────
cfg = GANTrainConfig(
    training_data_path=str(TRAINING_DATA_PATH / "pm_df_wide_materialized_windows"),
    run_dir_path=str(RUN_DIR),
    weights_path=str(WEIGHTS_PATH),
    # See HP_GAN in core/gan.py for the rationale behind each knob.
    latent_dim=64,
    hidden_dim=256,
    n_layers=2,
    use_attention=True,
    gen_use_pe=True,  # PE on in generator — run-2 off collapsed the diurnal cycle (global-only z)
    critic_use_pe=True,  # PE on in critic too (helps it judge position)
    kpi_proj_activation="linear",  # linear (not relu) pre-residual KPI projection
    per_step_noise_dim=16,  # fresh per-step noise gives the generator entropy…
    use_minibatch_stddev=True,  # NEW: …and this makes the critic punish ignoring it (anti-collapse)
    use_first_diff=True,  # NEW: critic sees ΔX → punishes over-smoothing (lag-1 AC too high in run-3)
    corr_l2=1e-5,
    learning_rate=1e-4,
    n_critic=3,  # rebalance so the generator keeps pace with the critic
    gp_weight=10.0,
    moment_weight=1.0,  # start high to lock the diurnal profile…
    moment_weight_final=0.1,  # …then anneal so the adversarial signal drives late training
    # ~9h run: benchmarked ~7.9 min/epoch on the real data (GP forces the slower MATH
    # attention kernel; first-diff doubles the critic input). 64 epochs ≈ 8.5h.
    # Watch gen_diversity in training_history.json climb toward real_diversity (~0.11);
    # snapshots every ~10 epochs let you pick the best checkpoint.
    epochs=64,
    batch_size=64,
)

if __name__ == "__main__":
    run_gan_training(cfg)
