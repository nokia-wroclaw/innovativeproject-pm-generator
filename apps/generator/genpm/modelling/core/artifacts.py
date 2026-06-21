import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genpm.modelling.core.data import CONTEXT_DIM
from genpm.modelling.core.model import HP_V5, build_cvae_lstm
from genpm.utils.logger import get_logger

logger = get_logger()


def save_training_artifacts(
    out_dir: str | Path,
    data: dict,
    arch_params: dict | None = None,
) -> None:
    """Persist all training artifacts needed to reload the model and generate data."""
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
        {"config_cols": config_cols, "map": cell_config_map},
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


def load_trained_model(
    run_id_path: str | Path,
    weights_path: str | Path,
    scaling_params_path: str | Path | None = None,
    global_latent_dim: int = HP_V5["global_latent_dim"],
    local_latent_dim: int = HP_V5["local_latent_dim"],
    hidden_dim: int = 256,
    n_layers: int = 2,
    use_attention: bool = True,
    n_heads: int = 4,
    free_bits_global: float = HP_V5["free_bits_global"],
    free_bits_local: float = HP_V5["free_bits_local"],
    output_activation: str = HP_V5["output_activation"],
    seq_len: int | None = None,
    feat_dim: int | None = None,
) -> tuple[object, object, dict]:
    """Reload saved artifacts and restore the trained model with weights.

    seq_len/feat_dim default to the values saved at training time (arch_params.json /
    kpi_columns.npy) so they always match the checkpoint; pass them only to override.

    Returns (model, config_encoder, cell_config_map) where cell_config_map maps each
    training distname to its config values (used to build the conditioning vector).
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

    _, model = build_cvae_lstm(
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
    )

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
