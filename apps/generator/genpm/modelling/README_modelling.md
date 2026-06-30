# GenPM modelling — running training & evaluation

Three generative model families live here, all trained on the same windowed dataset
(`X_scaled`: `(N, 168, F)` hourly KPI windows, conditioned on `y` = one-hot config +
calendar features). See `core/VAE_ARCHITECTURE_HISTORY.md` for the cVAE design
lineage.

> **Status: only the diffusion model is worth further attention.** Both the cVAE-LSTM
> and the WGAN-GP collapse to the per-config *mean* curve — within-config diversity
> (fixed config, varying noise) stalls at ~2/3 of real diversity no matter how the
> knobs are tuned. A single-config diagnostic proved this is intrinsic to the
> VAE/GAN objective itself (posterior/mode collapse is a safe optimum for both), not
> a conditioning or capacity bug, so neither family is going to get materially
> better. Diffusion's denoising objective has no such shortcut and does not collapse.
> Treat the cVAE/GAN code and run scripts below as reference/history; do new work on
> diffusion (`diffusion_run_5_yj` / `ddpm_5_yj.weights.h5` is the current best run).

## Training

Quick-run dev scripts live in `apps/generator/scripts/` (run from the repo root, no
install needed — each script inserts the repo onto `sys.path` itself):

| Family | Script | Config class |
|---|---|---|
| cVAE-LSTM (v7) | `python scripts/run_training.py` | `TrainConfig` |
| Conditional WGAN-GP | `python scripts/run_gan_training.py` | `GANTrainConfig` |
| Conditional DDPM (diffusion) | `python scripts/run_diffusion_training.py` | `DiffusionTrainConfig` |

Each script hardcodes `RUN_DIR`/`WEIGHTS_PATH` under
`genpm/utils/consts.py::SHARED_DIR_PATH / "model_runs" / <run_id>` and a config
instance with inline comments explaining the current run's knobs vs. the previous
one in its lineage — read the comment block before changing values. Outputs per run:
`X_scaled.npy`, `y.npy`, `cell_ids.npy`, `window_anchors.npy`, `kpi_columns.npy`,
`arch_params.json`, `training_history.json`, `training_losses.png`, and the
`.weights.h5` checkpoint(s) under `models_weights_debug/`.

Backend is torch-only (`tsgm` must be imported before `keras` to force
`KERAS_BACKEND=torch`); the GAN's gradient penalty needs torch's double-backward and
raises `NotImplementedError` on tf/jax.

## Generation

Generation is implemented end-to-end in the product, not just here: the frontend
(`apps/frontend/src/features/modeling/views/Generate.vue` +
`GenerateProcessModal.vue`) calls the backend's
`POST /modeling/processes/generate/runs` endpoint
(`apps/backend/app/api/v1/modeling.py`), which triggers the Airflow `generate_pipeline`
DAG. The pieces below are for local/dev use without going through that stack.

**Quick dev run** — `python scripts/run_generation.py` (from `apps/generator`, no
install needed). Hardcodes a `GenerateConfig` pointed at a run dir; prints the valid
`cell_configs` column order/values for that run before generating, so it doubles as a
"what configs exist" lookup. The reload path is architecture-agnostic, so repointing
`RUN_DIR`/`weights_path` at a diffusion (or GAN/cVAE) run works unchanged — edit it to
target `diffusion_run_5_yj` / `ddpm_5_yj.weights.h5`.

**CLI** — same `GenerateConfig` path, explicit flags instead of hardcoded values:

```bash
python -m genpm.modelling generate \
  --run-dir-path <SHARED_DIR_PATH>/model_runs/diffusion_run_5_yj \
  --weights-path <SHARED_DIR_PATH>/model_runs/diffusion_run_5_yj/models_weights_debug/ddpm_5_yj.weights.h5 \
  --output-path /tmp/genpm_out \
  --cell-id <some distname from cell_config_map> \
  --anchor-date 2024-01-15 \
  --n-weeks 4
```

Both write one parquet file of synthetic windows (still in `[0,1]`-scaled space —
apply the saved MinMax/Yeo-Johnson inverse if you need physical KPI units, as the
notebooks below do). `--cell-id`/`cell_id` looks the config up from the training
`cell_config_map`; pass `--cell-configs`/`cell_configs` instead for an explicit,
not-necessarily-observed config. For ad hoc generation + inspection in one step, use
`generate_windows` directly from a notebook (see `run_visual_checks.ipynb`) instead
of round-tripping through parquet.

## Evaluation notebooks (`model_tests/`)

Point `RUN_ID` / `WEIGHTS_FILE` at the top of a notebook to a `model_runs/<run_id>`
directory and re-run top to bottom. All three below share the same
generate-vs-real pipeline (`load_trained_model` → `generate_windows` → inverse-transform
back to physical KPI units via the saved MinMax/Yeo-Johnson scaler params).

- **`run_visual_checks.ipynb`** — for one trained run, generates synthetic windows
  per distinct cell-config and plots them against real data in physical units:
  timeseries overlay, KDE, autocorrelation, cross-KPI correlation heatmap.
- **`run_metrics_checks.ipynb`** — numeric companion to `run_visual_checks.ipynb`.
  Same pipeline, but produces scalar tables instead of plots: per-KPI Wasserstein,
  MMD, Jensen-Shannon, hourly-profile RMSE, ACF distance, Lomb-Scargle spectrum
  distance, plus cross-KPI sliced-Wasserstein/MMD/correlation-distance metrics
  (from `genpm.data_similarity.data_similarity_utils`).
- **`metrics_comparison.ipynb`** — dashboard comparing *multiple* diffusion runs
  side by side (training-history curves, scalar metric table, per-KPI KS/KL
  boxplots, marginal overlays for the hardest KPIs). Tracks three residuals:
  R1 under-dispersion (`div_ratio`/`std_ratio_med` → 1), R2 over-regular days
  (`ac24_gap` ↓), R3 marginal-shape mismatch (`ks_med`/`kl_med` ↓). KS is
  cross-run comparable (invariant to the raw-vs-Yeo-Johnson scaling difference
  between runs); KL is only comparable within a run's own training space.
