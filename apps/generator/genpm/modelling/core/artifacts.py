import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genpm.modelling.core.data import CONTEXT_DIM
from genpm.modelling.core.model import HP_V5, HP_V7, build_cvae_lstm, build_cvae_lstm_v7
from genpm.utils.logger import get_logger

logger = get_logger()


def _save_history(out_dir: Path, history) -> None:
    """Save Keras History: raw JSON + loss plot PNG."""
    hist_dict = history.history
    (out_dir / "training_history.json").write_text(json.dumps(hist_dict, indent=2))
    logger.info(f"Saved training_history.json — {list(hist_dict.keys())}")

    metrics = [k for k in hist_dict if not k.startswith("val_")]
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        ax.plot(hist_dict[metric], label=metric)
        if f"val_{metric}" in hist_dict:
            ax.plot(hist_dict[f"val_{metric}"], label=f"val_{metric}", linestyle="--")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("epoch")
    fig.suptitle("Training losses", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "training_losses.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training_losses.png")


def save_training_artifacts(
    out_dir: str | Path,
    data: dict,
    arch_params: dict | None = None,
    history=None,
) -> None:
    """Persist all training artifacts needed to reload the model and generate data.

    Model-agnostic: writes the data tensors (X/y), window metadata, the fitted
    config encoder, the distname→config map, optional scaler params, and the loss
    history/plot. ``arch_params`` (with ``arch_version`` and the input dims) is what
    :func:`load_trained_model` later branches on to rebuild the right model family.

    Args:
        out_dir: Destination directory (created if missing).
        data: The dict returned by :func:`load_training_windows`.
        arch_params: Architecture/hyperparameter dict saved to ``arch_params.json``;
            ``arch_version`` and ``seq_len``/``feat_dim`` are filled in if absent.
        history: Optional Keras ``History`` to dump as JSON + a loss-curve PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving artifacts to {out_dir}")

    np.save(out_dir / "X_scaled.npy", data["X_scaled"])
    logger.info(f"Saved X_scaled.npy — shape {data['X_scaled'].shape}")

    y_save = data.get("y")
    np.save(out_dir / "y.npy", y_save)
    logger.info(f"Saved y.npy — shape {y_save.shape}")

    np.save(out_dir / "window_anchors.npy", data["window_anchors"].astype(str))
    np.save(out_dir / "cell_ids.npy", data["cell_ids"])
    np.save(out_dir / "kpi_columns.npy", np.array(data["kpi_columns"]))
    logger.info(f"Saved window_anchors, cell_ids, kpi_columns ({len(data['kpi_columns'])} KPIs)")

    joblib.dump(data["config_encoder"], out_dir / "config_encoder.pkl")
    logger.info(f"Saved config_encoder.pkl — config_dims={data.get('config_dims')}")

    # distname → its config values, so generation-by-cell_id can look up the configs.
    config_cols = data["config_cols"]
    cell_config_map = {
        str(cid): list(cfg) for cid, cfg in zip(data["cell_ids"], data["configs"], strict=False)
    }
    joblib.dump(
        {
            "config_cols": config_cols,
            "map": cell_config_map,
            # Fitted LabelEncoder for the diffusion cell-identity embedding (core/diffusion.py).
            # Stored here (not a separate file) to avoid changing load_trained_model's
            # return signature; unused by cVAE/GAN. None for runs that predate this feature.
            "cell_encoder": data.get("cell_encoder"),
        },
        out_dir / "cell_config_map.pkl",
    )
    logger.info(f"Saved cell_config_map.pkl — {len(cell_config_map)} cells")

    if data.get("params_df") is not None:
        data["params_df"].to_parquet(out_dir / "params_df.parquet", index=False)
        logger.info("Saved params_df.parquet")

    arch_params = arch_params or {}
    if "arch_version" not in arch_params and data.get("arch_version"):
        arch_params["arch_version"] = data["arch_version"]
    # Persist the input dimensions so generation can rebuild the model without
    # hardcoding them (they must match the checkpoint).
    arch_params.setdefault("seq_len", data["seq_len"])
    arch_params.setdefault("feat_dim", data["feat_dim"])
    if arch_params:
        (out_dir / "arch_params.json").write_text(json.dumps(arch_params, indent=2))
        logger.info(f"Saved arch_params.json — {arch_params}")

    if history is not None:
        _save_history(out_dir, history)

    logger.info(f"All artifacts saved to {out_dir}")


def _resolve_input_dims(
    run_id_path: Path,
    seq_len: int | None,
    feat_dim: int | None,
) -> tuple[int, int]:
    """Resolve (seq_len, feat_dim) from saved artifacts, honoring explicit overrides.

    seq_len comes from arch_params.json; feat_dim from the saved kpi_columns.npy
    (cross-checked against arch_params.json when present).
    """
    arch_params = {}
    arch_path = run_id_path / "arch_params.json"
    if arch_path.exists():
        arch_params = json.loads(arch_path.read_text())

    if seq_len is None:
        seq_len = arch_params.get("seq_len")
        if seq_len is None:
            raise ValueError(
                f"seq_len not provided and not found in {arch_path}; pass it explicitly."
            )

    if feat_dim is None:
        kpi_path = run_id_path / "kpi_columns.npy"
        if kpi_path.exists():
            feat_dim = int(len(np.load(kpi_path, allow_pickle=True)))
            saved = arch_params.get("feat_dim")
            if saved is not None and saved != feat_dim:
                logger.warning(
                    f"feat_dim mismatch: arch_params={saved} but kpi_columns.npy={feat_dim}; "
                    f"using kpi_columns.npy"
                )
        else:
            feat_dim = arch_params.get("feat_dim")
        if feat_dim is None:
            raise ValueError(
                f"feat_dim not provided and not derivable from artifacts in {run_id_path}; "
                f"pass it explicitly."
            )

    return int(seq_len), int(feat_dim)


def _load_alt_model(
    arch_version: str,
    arch_params: dict,
    weights_path: str | Path,
    seq_len: int,
    feat_dim: int,
    y_dim: int,
):
    """Rebuild and weight-load a GAN or diffusion model from saved arch_params.

    Both build_* helpers run a dummy forward pass internally so all variables exist
    before load_weights; the returned model exposes .generate(y) like the cVAEs.
    """
    if arch_version == "gan":
        from genpm.modelling.core.gan import HP_GAN, build_gan

        model, _g, _c = build_gan(
            seq_len=seq_len,
            feat_dim=feat_dim,
            y_dim=y_dim,
            latent_dim=arch_params.get("latent_dim", HP_GAN["latent_dim"]),
            hidden_dim=arch_params.get("hidden_dim", HP_GAN["hidden_dim"]),
            n_layers=arch_params.get("n_layers", HP_GAN["n_layers"]),
            use_attention=arch_params.get("use_attention", HP_GAN["use_attention"]),
            n_heads=arch_params.get("n_heads", HP_GAN["n_heads"]),
            # Back-compat: pre-split checkpoints only have the single "use_pe" key
            # and were trained with PE on both nets + a relu KPI projection, so old
            # runs must fall back to that to keep weight shapes/behaviour matching.
            gen_use_pe=arch_params.get("gen_use_pe", arch_params.get("use_pe", True)),
            critic_use_pe=arch_params.get("critic_use_pe", arch_params.get("use_pe", True)),
            kpi_proj_activation=arch_params.get("kpi_proj_activation", "relu"),
            # Pre-noise-injection checkpoints (run-1/run-2) have no per-step noise.
            per_step_noise_dim=arch_params.get("per_step_noise_dim", 0),
            # run-1..3 critics have neither feature; default off so their weight
            # shapes are reproduced exactly for load_weights.
            use_minibatch_stddev=arch_params.get("use_minibatch_stddev", False),
            use_first_diff=arch_params.get("use_first_diff", False),
            output_activation=arch_params.get("output_activation", HP_GAN["output_activation"]),
            corr_l2=arch_params.get("corr_l2", HP_GAN["corr_l2"]),
        )
    elif arch_version == "diffusion":
        from genpm.modelling.core.diffusion import HP_DIFFUSION, build_diffusion

        model, _d = build_diffusion(
            seq_len=seq_len,
            feat_dim=feat_dim,
            y_dim=y_dim,
            num_timesteps=arch_params.get("num_timesteps", HP_DIFFUSION["num_timesteps"]),
            # Back-compat: diffusion_run_1 predates the schedule option → it was linear.
            beta_schedule=arch_params.get("beta_schedule", "linear"),
            beta_start=arch_params.get("beta_start", HP_DIFFUSION["beta_start"]),
            beta_end=arch_params.get("beta_end", HP_DIFFUSION["beta_end"]),
            width=arch_params.get("width", HP_DIFFUSION["width"]),
            n_blocks=arch_params.get("n_blocks", HP_DIFFUSION["n_blocks"]),
            dilation_cycle=tuple(arch_params.get("dilation_cycle", HP_DIFFUSION["dilation_cycle"])),
            time_embed_dim=arch_params.get("time_embed_dim", HP_DIFFUSION["time_embed_dim"]),
            cond_embed_dim=arch_params.get("cond_embed_dim", HP_DIFFUSION["cond_embed_dim"]),
            output_clip=arch_params.get("output_clip", HP_DIFFUSION["output_clip"]),
            # Back-compat: run-1..3 have no per-timestep calendar → cond_dim 0.
            cond_dim=arch_params.get("cond_dim", 0),
            # Back-compat: runs before the cell-identity embedding have no
            # cell_vocab_size → 0 disables it, reproducing the old weight shapes.
            cell_vocab_size=arch_params.get("cell_vocab_size", 0),
            cell_embed_dim=arch_params.get("cell_embed_dim", HP_DIFFUSION["cell_embed_dim"]),
        )
    else:
        raise ValueError(f"Unknown alt arch_version: {arch_version!r}")

    logger.info(f"Loading {arch_version} weights from {weights_path}")
    model.load_weights(str(weights_path))
    logger.info("Weights loaded successfully")
    return model


def load_trained_model(
    run_id_path: str | Path,
    weights_path: str | Path,
    scaling_params_path: str | Path | None = None,
    global_latent_dim: int | None = None,
    local_latent_dim: int | None = None,
    hidden_dim: int | None = None,
    n_layers: int | None = None,
    use_attention: bool | None = None,
    n_heads: int | None = None,
    free_bits_global: float = HP_V5["free_bits_global"],
    free_bits_local: float = HP_V5["free_bits_local"],
    output_activation: str | None = None,
    seq_len: int | None = None,
    feat_dim: int | None = None,
) -> tuple[object, object, dict]:
    """Reload saved artifacts and restore the trained model with weights.

    Branches on ``arch_params.json["arch_version"]``: GAN/diffusion models are rebuilt
    via :func:`_load_alt_model` (the cVAE kwargs below are ignored for them); otherwise
    a cVAE-LSTM v6/v7 is rebuilt, a dummy forward pass builds the graph, and weights
    are loaded.

    Args:
        run_id_path: Run directory holding ``config_encoder.pkl``,
            ``cell_config_map.pkl``, ``arch_params.json``, ``kpi_columns.npy``, etc.
        weights_path: Checkpoint to load into the rebuilt model.
        scaling_params_path: Unused here; accepted for caller symmetry.
        global_latent_dim, local_latent_dim, hidden_dim, n_layers, use_attention,
            n_heads, free_bits_global, free_bits_local, output_activation, seq_len,
            feat_dim: cVAE geometry overrides. Each defaults to the value saved at
            training time (``arch_params.json`` / ``kpi_columns.npy``) so it matches
            the checkpoint exactly — override only deliberately, since a mismatch
            changes layer shapes and breaks weight loading.

    Returns:
        Tuple ``(model, config_encoder, cell_config_map)``, where ``cell_config_map``
        maps each training distname to its config values (used to build ``y``).
    """
    run_id_path = Path(run_id_path)
    logger.info(f"Loading config_encoder from {run_id_path}")
    config_encoder = joblib.load(run_id_path / "config_encoder.pkl")
    cell_config_map = joblib.load(run_id_path / "cell_config_map.pkl")

    seq_len, feat_dim = _resolve_input_dims(run_id_path, seq_len, feat_dim)

    # y_dim is the one-hot config width + holiday + seasonal context.
    y_dim = sum(len(c) for c in config_encoder.categories_) + CONTEXT_DIM
    logger.info(
        f"Config encoder loaded — y_dim={y_dim}, seq_len={seq_len}, feat_dim={feat_dim}, "
        f"{len(cell_config_map['map'])} cells"
    )

    arch_params_path = run_id_path / "arch_params.json"
    arch_params = json.loads(arch_params_path.read_text()) if arch_params_path.exists() else {}
    tile_z = bool(arch_params.get("tile_z_in_decoder", True))
    arch_version = arch_params.get("arch_version", "v6")

    # GAN / diffusion are separate model families — build, restore, and return early.
    # (The VAE-specific kwargs above are ignored; these read their geometry from
    # arch_params.json saved at training time.)
    if arch_version in ("gan", "diffusion"):
        model = _load_alt_model(arch_version, arch_params, weights_path, seq_len, feat_dim, y_dim)
        return model, config_encoder, cell_config_map

    def _resolve(value, key, default):
        return value if value is not None else arch_params.get(key, default)

    global_latent_dim = _resolve(global_latent_dim, "global_latent_dim", HP_V5["global_latent_dim"])
    local_latent_dim = _resolve(local_latent_dim, "local_latent_dim", HP_V5["local_latent_dim"])
    hidden_dim = _resolve(hidden_dim, "hidden_dim", 256)
    n_layers = _resolve(n_layers, "n_layers", 2)
    use_attention = _resolve(use_attention, "use_attention", True)
    n_heads = _resolve(n_heads, "n_heads", 4)
    output_activation = _resolve(output_activation, "output_activation", HP_V5["output_activation"])

    common_kwargs = dict(
        seq_len=seq_len,
        feat_dim=feat_dim,
        y_dim=y_dim,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_attention=use_attention,
        n_heads=n_heads,
        free_bits_global=free_bits_global,
        free_bits_local=free_bits_local,
        output_activation=output_activation,
        tile_z=tile_z,
    )
    if arch_version == "v7":
        _, model = build_cvae_lstm_v7(
            **common_kwargs,
            ac_weight=arch_params.get("ac_weight", HP_V7["ac_weight"]),
            ac_max_lag=arch_params.get("ac_max_lag", HP_V7["ac_max_lag"]),
            corr_l2=arch_params.get("corr_l2", 0.0),
        )
        logger.info(
            f"Loaded v7 model (ac_weight={arch_params.get('ac_weight')}, "
            f"corr_l2={arch_params.get('corr_l2', 0.0)}, hidden_dim={hidden_dim}, "
            f"global_latent_dim={global_latent_dim})"
        )
    else:
        _, model = build_cvae_lstm(**common_kwargs)
        logger.info(f"Loaded {arch_version} model")

    dummy_X = np.zeros((1, seq_len, feat_dim), dtype=np.float32)
    dummy_y = np.zeros((1, y_dim), dtype=np.float32)
    logger.info("Running dummy forward pass to build model graph")
    model([dummy_X, dummy_y], training=False)
    logger.info(f"Loading weights from {weights_path}")
    model.load_weights(str(weights_path))
    logger.info("Weights loaded successfully")

    return model, config_encoder, cell_config_map


def load_saved_windows(out_dir: str | Path) -> dict:
    """Load numpy arrays saved by save_training_artifacts."""
    out_dir = Path(out_dir)
    logger.info(f"Loading training data from {out_dir}")
    y_path = out_dir / "y.npy"
    if not y_path.exists():
        logger.info("y.npy not found, falling back to y_extended.npy")
        y_path = out_dir / "y_extended.npy"
    data = {
        "X_scaled": np.load(out_dir / "X_scaled.npy"),
        "y": np.load(y_path),
        "window_anchors": pd.to_datetime(
            np.load(out_dir / "window_anchors.npy", allow_pickle=True)
        ),
        "cell_ids": np.load(out_dir / "cell_ids.npy", allow_pickle=True),
        "kpi_columns": np.load(out_dir / "kpi_columns.npy", allow_pickle=True).tolist(),
    }
    logger.info(
        f"Training data loaded | X_scaled={data['X_scaled'].shape} "
        f"kpi_columns={len(data['kpi_columns'])} cells={len(set(data['cell_ids']))}"
    )
    return data
