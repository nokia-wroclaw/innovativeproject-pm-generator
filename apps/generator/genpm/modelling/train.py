from pathlib import Path

from genpm.modelling.configs import TrainConfig
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

    logger.info(f"Saving artifacts to {cfg.run_dir_path}")
    save_training_artifacts(
        Path(cfg.run_dir_path),
        data,
        history=history,
        arch_params=arch_params,
    )

    return history
