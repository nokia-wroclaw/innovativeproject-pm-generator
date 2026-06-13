"""Saving and loading trained model artifacts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genpm.modelling.model_utils.data import Y_DIM
from genpm.modelling.model_utils.model import HP_V5, build_cvae_lstm
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

    joblib.dump(data["cell_encoder"], out_dir / "cell_encoder.pkl")
    logger.info(f"Saved cell_encoder.pkl — {len(data['cell_encoder'].classes_)} cells")

    if data.get("params_df") is not None:
        data["params_df"].to_parquet(out_dir / "params_df.parquet", index=False)
        logger.info("Saved params_df.parquet")

    arch_params = arch_params or {}
    if "arch_version" not in arch_params and data.get("arch_version"):
        arch_params["arch_version"] = data["arch_version"]
    if arch_params:
        (out_dir / "arch_params.json").write_text(json.dumps(arch_params, indent=2))
        logger.info(f"Saved arch_params.json — {arch_params}")

    logger.info(f"All artifacts saved to {out_dir}")


def load_trained_model(
    run_id_path: str | Path,
    weights_path: str | Path,
    scaling_params_path: str | Path | None = None,
    global_latent_dim: int = HP_V5["global_latent_dim"],
    local_latent_dim: int = HP_V5["local_latent_dim"],
    cell_embed_dim: int = HP_V5["cell_embed_dim"],
    hidden_dim: int = 256,
    n_layers: int = 2,
    use_attention: bool = True,
    n_heads: int = 4,
    free_bits_global: float = HP_V5["free_bits_global"],
    free_bits_local: float = HP_V5["free_bits_local"],
    output_activation: str = HP_V5["output_activation"],
    seq_len: int = 168,
    feat_dim: int = 235,
    y_dim: int = Y_DIM,
) -> tuple[object, object]:
    """Reload saved artifacts and restore the trained model with weights."""
    run_id_path = Path(run_id_path)
    logger.info(f"Loading cell_encoder from {run_id_path}")
    cell_encoder = joblib.load(run_id_path / "cell_encoder.pkl")
    n_cells = len(cell_encoder.classes_)
    logger.info(f"Cell encoder loaded — {n_cells} cells")

    _, model = build_cvae_lstm(
        seq_len=seq_len,
        feat_dim=feat_dim,
        n_cells=n_cells,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        cell_embed_dim=cell_embed_dim,
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

    return model, cell_encoder


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
