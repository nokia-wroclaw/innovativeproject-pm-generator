from pathlib import Path

# tsgm MUST be imported before keras — it patches keras internals on import.
import tsgm  # noqa: E402, F401, isort:skip
import keras
import numpy as np

from genpm.modelling.configs import DiffusionTrainConfig, GANTrainConfig, TrainConfig
from genpm.modelling.core.artifacts import save_training_artifacts
from genpm.modelling.core.data import load_training_windows
from genpm.modelling.core.model import build_cvae_lstm, build_cvae_lstm_v7
from genpm.modelling.core.training import train_cvae
from genpm.utils.logger import get_logger

logger = get_logger()


def run_training(cfg: TrainConfig):
    """Run the full training pipeline: load data → build model → train → save artifacts."""
    logger.info(f"Loading training data from {cfg.training_data_path}")
    data = load_training_windows(
        Path(cfg.training_data_path),
        drop_constant_kpis=cfg.drop_constant_kpis,
    )

    arch_version = getattr(cfg, "arch_version", "v7")
    logger.info(
        f"Building {arch_version} model: global_latent_dim={cfg.global_latent_dim}, "
        f"hidden_dim={cfg.hidden_dim}, n_layers={cfg.n_layers}"
    )

    common_kwargs = dict(
        seq_len=data["seq_len"],
        feat_dim=data["feat_dim"],
        y_dim=data["y_dim"],
        global_latent_dim=cfg.global_latent_dim,
        local_latent_dim=cfg.local_latent_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        use_attention=cfg.use_attention,
        n_heads=cfg.n_heads,
        beta=cfg.beta,
        learning_rate=cfg.learning_rate,
        free_bits_global=cfg.free_bits_global,
        free_bits_local=cfg.free_bits_local,
        output_activation=cfg.output_activation,
    )

    if arch_version == "v7":
        _, model = build_cvae_lstm_v7(
            **common_kwargs,
            ac_weight=cfg.ac_weight,
            ac_max_lag=cfg.ac_max_lag,
            corr_l2=cfg.corr_l2,
        )
    else:
        _, model = build_cvae_lstm(**common_kwargs)

    logger.info(f"Training for up to {cfg.epochs} epochs → {cfg.weights_path}")
    history = train_cvae(
        model,
        data["X_scaled"],
        data["y"],
        weights_path=Path(cfg.weights_path),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        target_beta=cfg.target_beta,
        use_cyclical_kl=cfg.use_cyclical_kl,
        cycle_epochs=cfg.cycle_epochs,
        n_cycles=cfg.n_cycles,
        cycle_ratio=cfg.cycle_ratio,
        anneal_epochs=cfg.anneal_epochs,
        kl_delay_epochs=cfg.kl_delay_epochs,
        collapse_monitor=cfg.collapse_monitor,
        lr_patience=cfg.lr_patience,
        early_stop_patience=cfg.early_stop_patience,
    )

    arch_params = {
        "arch_version": arch_version,
        "tile_z_in_decoder": True,
        "y_dim": data["y_dim"],
        "global_latent_dim": cfg.global_latent_dim,
        "local_latent_dim": cfg.local_latent_dim,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "use_attention": cfg.use_attention,
        "n_heads": cfg.n_heads,
        "learning_rate": cfg.learning_rate,
        "target_beta": cfg.target_beta,
        "output_activation": cfg.output_activation,
    }
    if arch_version == "v7":
        arch_params["ac_weight"] = cfg.ac_weight
        arch_params["ac_max_lag"] = cfg.ac_max_lag
        arch_params["corr_l2"] = cfg.corr_l2

    logger.info(f"Saving artifacts to {cfg.run_dir_path}")
    save_training_artifacts(
        Path(cfg.run_dir_path),
        data,
        history=history,
        arch_params=arch_params,
    )

    return history


def _last_weights_path(weights_path: Path) -> Path:
    """Sibling '*_last.weights.h5' path used as an unconditional fallback checkpoint."""
    name = weights_path.name
    if name.endswith(".weights.h5"):
        base = name[: -len(".weights.h5")]
        return weights_path.with_name(f"{base}_last.weights.h5")
    return weights_path.with_name(weights_path.stem + "_last" + weights_path.suffix)


def _build_diversity_probe(
    X: np.ndarray, y: np.ndarray, max_configs: int = 8
) -> tuple[np.ndarray, float]:
    """Pick up to ``max_configs`` distinct conditioning vectors and measure the real
    within-config diversity for them.

    Returns (y_probe, real_diversity) where real_diversity is the cross-window std
    (averaged over time and features) of the real windows sharing each probe config,
    averaged over the probe configs — the target the generator's per-config
    cross-sample std should approach. Configs with a single real window are skipped.
    """
    uniq = np.unique(y, axis=0)
    if len(uniq) > max_configs:
        idx = np.linspace(0, len(uniq) - 1, max_configs).astype(int)
        uniq = uniq[idx]
    probe_rows, real_stds = [], []
    for row in uniq:
        mask = np.all(y == row, axis=1)
        if mask.sum() < 2:  # need >=2 windows for a meaningful cross-window std
            continue
        probe_rows.append(row)
        real_stds.append(float(X[mask].std(axis=0).mean()))
    if not probe_rows:  # degenerate: every config has one window — fall back to uniq
        return uniq.astype(np.float32), float("nan")
    return np.stack(probe_rows).astype(np.float32), float(np.mean(real_stds))


def run_gan_training(cfg: GANTrainConfig):
    """Train the conditional WGAN-GP: load data → build → fit → save artifacts."""
    from genpm.modelling.core.gan import (
        GANDiversityMonitor,
        MomentWeightScheduler,
        build_gan,
    )

    logger.info(f"Loading training data from {cfg.training_data_path}")
    data = load_training_windows(
        Path(cfg.training_data_path),
        drop_constant_kpis=cfg.drop_constant_kpis,
    )

    logger.info(
        f"Building GAN: latent_dim={cfg.latent_dim}, hidden_dim={cfg.hidden_dim}, "
        f"n_layers={cfg.n_layers}, n_critic={cfg.n_critic}"
    )
    model, _generator, _critic = build_gan(
        seq_len=data["seq_len"],
        feat_dim=data["feat_dim"],
        y_dim=data["y_dim"],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        use_attention=cfg.use_attention,
        n_heads=cfg.n_heads,
        gen_use_pe=cfg.gen_use_pe,
        critic_use_pe=cfg.critic_use_pe,
        kpi_proj_activation=cfg.kpi_proj_activation,
        per_step_noise_dim=cfg.per_step_noise_dim,
        use_minibatch_stddev=cfg.use_minibatch_stddev,
        use_first_diff=cfg.use_first_diff,
        output_activation=cfg.output_activation,
        corr_l2=cfg.corr_l2,
        learning_rate=cfg.learning_rate,
        adam_beta_1=cfg.adam_beta_1,
        adam_beta_2=cfg.adam_beta_2,
        n_critic=cfg.n_critic,
        gp_weight=cfg.gp_weight,
        moment_weight=cfg.moment_weight,
    )

    weights_path = Path(cfg.weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    # GANs have no clean "best" metric, so the latest epoch's weights are the
    # checkpoint of record (overwritten every epoch). Periodic snapshots are also
    # kept so a better earlier epoch can be recovered (GAN quality is non-monotonic).
    snapshot_path = weights_path.parent / (weights_path.stem + "_e{epoch:03d}.weights.h5")
    steps_per_epoch = (len(data["X_scaled"]) + cfg.batch_size - 1) // cfg.batch_size
    # Keep ~12 evenly spaced snapshots regardless of run length (~48MB each), so a
    # better earlier epoch is recoverable on a long run without flooding the dir.
    snapshot_every = max(10, cfg.epochs // 12)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(weights_path), save_best_only=False, save_weights_only=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(snapshot_path),
            save_best_only=False,
            save_weights_only=True,
            save_freq=steps_per_epoch * snapshot_every,
        ),
    ]
    # Per-epoch mode-collapse probe: logs gen_diversity (and a real_diversity
    # reference line) into history so collapse is visible live, not just post-hoc.
    y_probe, real_diversity = _build_diversity_probe(data["X_scaled"], data["y"])
    logger.info(
        f"Diversity probe: {len(y_probe)} configs, real within-config diversity="
        f"{real_diversity:.5f} (target for gen_diversity)"
    )
    callbacks.append(GANDiversityMonitor(y_probe, real_diversity=real_diversity))
    # Anneal the feature-matching weight down so the generator can't satisfy the loss
    # with one near-deterministic curve per config (logs moment_w into history).
    callbacks.append(MomentWeightScheduler(cfg.moment_weight, cfg.moment_weight_final, cfg.epochs))
    logger.info(f"Training GAN for {cfg.epochs} epochs → {weights_path}")
    history = model.fit(
        data["X_scaled"],
        data["y"],
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    arch_params = {
        "arch_version": "gan",
        "y_dim": data["y_dim"],
        "latent_dim": cfg.latent_dim,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "use_attention": cfg.use_attention,
        "n_heads": cfg.n_heads,
        "gen_use_pe": cfg.gen_use_pe,
        "critic_use_pe": cfg.critic_use_pe,
        "kpi_proj_activation": cfg.kpi_proj_activation,
        "per_step_noise_dim": cfg.per_step_noise_dim,
        "use_minibatch_stddev": cfg.use_minibatch_stddev,
        "use_first_diff": cfg.use_first_diff,
        "output_activation": cfg.output_activation,
        "corr_l2": cfg.corr_l2,
        "n_critic": cfg.n_critic,
        "gp_weight": cfg.gp_weight,
        "moment_weight": cfg.moment_weight,
        "moment_weight_final": cfg.moment_weight_final,
    }
    logger.info(f"Saving artifacts to {cfg.run_dir_path}")
    save_training_artifacts(Path(cfg.run_dir_path), data, history=history, arch_params=arch_params)
    return history


def run_diffusion_training(cfg: DiffusionTrainConfig):
    """Train the conditional DDPM: load data → build → fit → save artifacts."""
    from genpm.modelling.core.diffusion import (
        DiffusionEvalCallback,
        EMACallback,
        _lag1_autocorr,
        build_diffusion,
    )

    logger.info(f"Loading training data from {cfg.training_data_path}")
    data = load_training_windows(
        Path(cfg.training_data_path),
        drop_constant_kpis=cfg.drop_constant_kpis,
    )

    logger.info(
        f"Building diffusion: num_timesteps={cfg.num_timesteps}, width={cfg.width}, "
        f"n_blocks={cfg.n_blocks}"
    )
    model, _denoiser = build_diffusion(
        seq_len=data["seq_len"],
        feat_dim=data["feat_dim"],
        y_dim=data["y_dim"],
        num_timesteps=cfg.num_timesteps,
        beta_schedule=cfg.beta_schedule,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        width=cfg.width,
        n_blocks=cfg.n_blocks,
        dilation_cycle=cfg.dilation_cycle,
        time_embed_dim=cfg.time_embed_dim,
        cond_embed_dim=cfg.cond_embed_dim,
        learning_rate=cfg.learning_rate,
        output_clip=cfg.output_clip,
    )

    weights_path = Path(cfg.weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    last_weights_path = _last_weights_path(weights_path)
    # Diffusion's training MSE is a genuine quality signal, so we can keep the
    # best-by-loss checkpoint plus an unconditional last-epoch fallback.
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(weights_path),
            monitor="loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(last_weights_path), save_best_only=False, save_weights_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss", mode="min", factor=0.5, patience=20, min_lr=1e-5
        ),
    ]
    # Periodic structure probe: pick the most common config and log whether generated
    # series develop temporal structure (gen_ac1) and stay diverse (gen_diversity).
    # Sampling is ~1000 steps so it runs sparsely (see DiffusionEvalCallback).
    uy, counts = np.unique(data["y"], axis=0, return_counts=True)
    probe_row = uy[int(np.argmax(counts))]
    probe_real = data["X_scaled"][np.all(data["y"] == probe_row, axis=1)]
    real_div = float(probe_real.std(axis=0).mean())
    _fi = np.random.default_rng(0).choice(
        probe_real.shape[2], min(20, probe_real.shape[2]), replace=False
    )
    real_ac1 = _lag1_autocorr(
        probe_real[:, :, _fi].transpose(0, 2, 1).reshape(-1, probe_real.shape[1])
    )
    logger.info(f"Diffusion eval probe: real_diversity={real_div:.4f} real_ac1={real_ac1:.4f}")
    callbacks.append(
        DiffusionEvalCallback(probe_row[None], real_diversity=real_div, real_ac1=real_ac1)
    )
    ema_cb = EMACallback(model.denoiser, momentum=cfg.ema_momentum) if cfg.use_ema else None
    if ema_cb is not None:
        callbacks.append(ema_cb)
    logger.info(f"Training diffusion for {cfg.epochs} epochs → {weights_path}")
    history = model.fit(
        data["X_scaled"],
        data["y"],
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # With EMA, the per-epoch ModelCheckpoints hold the raw (non-averaged) weights.
    # Write the EMA averages into the model and overwrite the primary checkpoint, so
    # generation loads the EMA weights (the ones that actually sample better). The
    # "_last" checkpoint keeps the raw weights as a fallback.
    if ema_cb is not None:
        logger.info("Writing EMA weights into the model and re-saving primary checkpoint")
        ema_cb.finalize()
        model.save_weights(str(weights_path))

    arch_params = {
        "arch_version": "diffusion",
        "y_dim": data["y_dim"],
        "num_timesteps": cfg.num_timesteps,
        "beta_schedule": cfg.beta_schedule,
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,
        "output_clip": cfg.output_clip,
        "width": cfg.width,
        "n_blocks": cfg.n_blocks,
        "dilation_cycle": list(cfg.dilation_cycle),
        "time_embed_dim": cfg.time_embed_dim,
        "cond_embed_dim": cfg.cond_embed_dim,
        "use_ema": cfg.use_ema,
        "ema_momentum": cfg.ema_momentum,
    }
    logger.info(f"Saving artifacts to {cfg.run_dir_path}")
    save_training_artifacts(Path(cfg.run_dir_path), data, history=history, arch_params=arch_params)
    return history
