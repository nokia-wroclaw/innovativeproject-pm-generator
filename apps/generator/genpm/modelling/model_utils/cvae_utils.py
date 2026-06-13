"""
cvae_utils.py
-------------
End-to-end utility functions for the cVAE-LSTM synthetic KPI generation pipeline.

Covers four stages, each callable from a notebook:
  1. prepare_data    — load parquet, build X tensor, build extended Y vector
  2. build_model     — instantiate cVAE_LSTMv3Architecture + cBetaVAE
  3. train_model     — fit with KL annealing, LR scheduling, checkpointing
  4. save/load       — persist and restore all training artifacts
  5. generate_*      — generate synthetic KPI data for a cell

Typical notebook usage (v5)
---------------------------
    from cvae_utils import (
        prepare_data, build_model_v5, train_model_v5,
        save_artifacts, load_artifacts_v5,
        generate_timespan, generate_n_weeks, HP_V5,
    )

    data = prepare_data(TRAINING_DATA_PATH / "pm_df_wide_indexed_winds")
    data["params_df"] = pd.read_parquet(TRAINING_DATA_PATH / "scaling_params.parquet")

    arch, model = build_model_v5(
        data["seq_len"], data["feat_dim"], data["n_classes"], **HP_V5
    )
    history = train_model_v5(
        model, data["X_scaled"], data["y"],
        weights_path=MODEL_PATH, target_beta=0.02, anneal_epochs=100,
    )
    save_artifacts(RUN_DIR_PATH, model, data, arch_params={**HP_V5, "arch_version": "v5"})
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
import tsgm  # noqa
from sklearn.preprocessing import LabelEncoder

from genpm.modelling.model_utils.model_utils import (
    cBetaVAE_Hierarchical,
    cVAE_LSTMv5Architecture,
)

SEQ_LEN = 168
Y_DIM = 6
SYNTHETIC_ORIGIN = pd.Timestamp("1970-01-01")

_META_COLS = {
    "distname",
    "bts_id",
    "window_anchor",
    "hour_idx",  # legacy: one row per hour
    "n_hours",  # compact: list length per window
    "imputed_flag",
    "holiday",
    "is_holiday",
}


# =============================================================================
# Stage 1 — Data preparation
# =============================================================================


def seasonal_features(anchor: pd.Timestamp) -> np.ndarray:
    """Four sinusoidal values encoding week-of-year and month-of-year, shape (4,)."""
    week = int(anchor.isocalendar().week)
    month = anchor.month
    return np.array(
        [
            np.sin(2 * np.pi * week / 52),
            np.cos(2 * np.pi * week / 52),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ],
        dtype=np.float32,
    )


def build_y(
    cell_ids: np.ndarray,
    window_anchors: np.ndarray,
    cell_encoder: LabelEncoder,
    holiday_flags: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compact conditioning vector — cell index is embedded inside the model.

    Layout: [cell_idx (1) | holiday (1) | seasonal (4)]
    Returns shape (N, 6).
    """
    n = len(cell_ids)
    cell_idx = cell_encoder.transform(cell_ids).astype(np.float32)
    if holiday_flags is None:
        holiday_flags = np.zeros(n, dtype=np.int32)
    y_holiday = holiday_flags.reshape(-1, 1).astype(np.float32)
    y_seasonal = np.stack([seasonal_features(pd.Timestamp(ts)) for ts in window_anchors])
    return np.concatenate(
        [
            cell_idx[:, None],
            y_holiday,
            y_seasonal,
        ],
        axis=1,
    ).astype(np.float32)


def build_y_extended(
    cell_ids: np.ndarray,
    window_anchors: np.ndarray,
    holiday_flags: np.ndarray,
    cell_encoder: LabelEncoder,
) -> np.ndarray:
    """
    Build extended conditioning vector Y per window.

    Layout: [one-hot cell (n_cells) | holiday (1) | seasonal (4)]
    Returns shape (N, n_cells + 5).
    """
    n = len(cell_ids)
    n_cells = len(cell_encoder.classes_)

    y_onehot = np.zeros((n, n_cells), dtype=np.float32)
    y_onehot[np.arange(n), cell_encoder.transform(cell_ids)] = 1.0

    y_holiday = holiday_flags.reshape(-1, 1).astype(np.float32)
    y_seasonal = np.stack([seasonal_features(pd.Timestamp(ts)) for ts in window_anchors])

    return np.concatenate([y_onehot, y_holiday, y_seasonal], axis=1)


CONST_KPI_STD_THRESHOLD = 0.05


def _stack_list_windows(
    pdf: pd.DataFrame,
    feat_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert list-valued KPI columns to (N, seq_len, D).

    Uses vectorized np.vstack when all lists are uniform; falls back row-wise
    when lengths differ.
    """
    try:
        planes = [np.vstack(pdf[c].to_numpy()) for c in feat_cols]
        X = np.stack(planes, axis=-1).astype(np.float32)
        if X.shape[1] != seq_len:
            raise ValueError("ragged list lengths")
        valid = ~np.isnan(X).any(axis=(1, 2))
        return X, valid
    except (ValueError, TypeError):
        n, d = len(pdf), len(feat_cols)
        X = np.full((n, seq_len, d), np.nan, dtype=np.float32)
        valid = np.ones(n, dtype=bool)
        for j, col in enumerate(feat_cols):
            for i, value in enumerate(pdf[col].values):
                if not valid[i]:
                    continue
                arr = np.asarray(value, dtype=np.float32).ravel()
                if len(arr) != seq_len or np.isnan(arr).any():
                    valid[i] = False
                    continue
                X[i, :, j] = arr
        valid &= ~np.isnan(X).any(axis=(1, 2))
        return X, valid


def _load_windows_from_list_format(
    pdf: pd.DataFrame,
    feat_cols: list[str],
    cell_id_col: str,
    anchor_col: str,
    hour_col: str,
    holiday_col: str | None,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """One row per window — KPI columns contain lists of length seq_len."""
    X, valid = _stack_list_windows(pdf, feat_cols, seq_len)

    if hour_col in pdf.columns:
        valid &= pdf[hour_col].to_numpy() == seq_len

    n_rejected = int((~valid).sum())
    if not valid.any():
        raise ValueError(
            f"No valid windows found — all {len(pdf)} rows were rejected "
            f"(expected list length {seq_len}, no NaNs)."
        )
    if n_rejected:
        print(f"Skipped {n_rejected:,} window(s) with wrong length or NaN values")

    pdf_ok = pdf.loc[valid]
    cell_ids_arr = pdf_ok[cell_id_col].to_numpy()
    window_anchors = pd.DatetimeIndex(pd.to_datetime(pdf_ok[anchor_col]))

    if holiday_col and holiday_col in pdf_ok.columns:
        holiday_flags = pdf_ok[holiday_col].fillna(0).astype(np.int32).to_numpy()
    else:
        holiday_flags = np.zeros(len(pdf_ok), dtype=np.int32)

    return X[valid], cell_ids_arr, window_anchors, holiday_flags


def _load_windows_from_long_format(
    pdf: pd.DataFrame,
    feat_cols: list[str],
    cell_id_col: str,
    anchor_col: str,
    hour_col: str,
    holiday_col: str | None,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Legacy format — group 168 hourly rows into one window tensor."""
    groups, cell_ids_list, anchors_list, holiday_list = [], [], [], []

    for (cell_id, anchor), g in pdf.groupby([cell_id_col, anchor_col], sort=False):
        g_sorted = g.sort_values(hour_col)
        kpi = g_sorted[feat_cols].to_numpy(dtype=np.float32)
        if len(g_sorted) != seq_len or np.isnan(kpi).any():
            continue
        groups.append(kpi)
        cell_ids_list.append(cell_id)
        anchors_list.append(pd.Timestamp(anchor))
        if holiday_col and holiday_col in g_sorted.columns:
            holiday_list.append(int(g_sorted[holiday_col].iloc[0]))
        else:
            holiday_list.append(0)

    if not groups:
        raise ValueError(
            f"No valid windows found in long format — expected {seq_len} rows "
            f"per ({cell_id_col}, {anchor_col}) group."
        )

    return (
        np.stack(groups).astype(np.float32),
        np.array(cell_ids_list),
        pd.DatetimeIndex(anchors_list),
        np.array(holiday_list, dtype=np.int32),
    )


def prepare_data(
    wide_scaled_path: str | Path,
    scaled_params_path: str | Path | None = None,
    cell_id_col: str = "distname",
    anchor_col: str = "window_anchor",
    hour_col: str = "n_hours",
    holiday_col: str | None = None,
    meta_cols: set[str] | None = None,
    drop_constant_kpis: bool = True,
    const_std_threshold: float = CONST_KPI_STD_THRESHOLD,
) -> dict:
    """
    Load windowed parquet, build X and Y tensors.

    Supports two parquet layouts:
      - **Compact (new):** one row per window; each KPI column is a list of
        ``seq_len`` values; optional ``n_hours`` column (must equal 168).
      - **Long (legacy):** one row per hour; ``hour_idx`` 0..167 per window.

    Returns dict with keys:
        X_scaled, y, y_extended, window_anchors, cell_ids, holiday_flags,
        params_df, cell_encoder, kpi_columns, seq_len, feat_dim, n_classes, output_dim
    """
    if meta_cols is None:
        meta_cols = _META_COLS

    pdf = pd.read_parquet(wide_scaled_path)
    feat_cols = sorted([c for c in pdf.columns if c not in meta_cols])
    if not feat_cols:
        raise ValueError(f"No KPI columns found in '{wide_scaled_path}'.")

    scaled_params_df = None
    if scaled_params_path is not None:
        scaled_params_df = pd.read_parquet(scaled_params_path)

    if hour_col == "n_hours":
        X_scaled, cell_ids_arr, window_anchors, holiday_flags = _load_windows_from_list_format(
            pdf,
            feat_cols,
            cell_id_col,
            anchor_col,
            hour_col,
            holiday_col,
            SEQ_LEN,
        )
    else:
        X_scaled, cell_ids_arr, window_anchors, holiday_flags = _load_windows_from_long_format(
            pdf,
            feat_cols,
            cell_id_col,
            anchor_col,
            hour_col,
            holiday_col,
            SEQ_LEN,
        )

    # dropping constant KPIs decide by threshold
    if drop_constant_kpis:
        per_feat_std = X_scaled.std(axis=(0, 1))
        const_mask = per_feat_std < const_std_threshold
        if const_mask.any():
            dropped = [c for c, m in zip(feat_cols, const_mask, strict=False) if m]
            print(
                f"Dropping {const_mask.sum()} near-constant KPI column(s) "
                f"(std < {const_std_threshold}): {dropped[:8]}"
                f"{'...' if len(dropped) > 8 else ''}"
            )
            feat_cols = [c for c, m in zip(feat_cols, const_mask, strict=False) if not m]
            X_scaled = X_scaled[:, :, ~const_mask]

    # encode y
    cell_encoder = LabelEncoder()
    cell_encoder.fit(cell_ids_arr)
    y = build_y(cell_ids_arr, window_anchors, cell_encoder, holiday_flags)
    n_classes = len(cell_encoder.classes_)

    print(
        f"Loaded {len(X_scaled):,} windows  |  "
        f"feat_dim={len(feat_cols)}  |  "
        f"cells={n_classes}  |  "
        f"y width={y.shape[1]}"
    )

    return {
        "X_scaled": X_scaled,
        "y": y,
        "window_anchors": window_anchors,
        "cell_ids": cell_ids_arr,
        "holiday_flags": holiday_flags,
        "params_df": scaled_params_df,
        "cell_encoder": cell_encoder,
        "kpi_columns": feat_cols,
        "seq_len": SEQ_LEN,
        "feat_dim": len(feat_cols),
        "n_classes": n_classes,
        "output_dim": Y_DIM,
        "arch_version": "v5",
    }


# def prepare_data(
#     wide_scaled_path: str | Path,
#     scaled_params_path: str | Path | None = None,
#     cell_id_col: str = "distname",
#     anchor_col: str = "window_anchor",
#     hour_col: str = "hour_idx",
#     holiday_col: str | None = None,
#     meta_cols: set[str] | None = None,
#     drop_constant_kpis: bool = True,
#     const_std_threshold: float = CONST_KPI_STD_THRESHOLD,
# ) -> dict:
#     """
#     Load the wide-format windowed parquet, build X and extended Y tensors.

#     Returns dict with keys:
#         X_scaled, y_extended, window_anchors, cell_ids, scaler,
#         cell_encoder, kpi_columns, seq_len, feat_dim, n_classes, output_dim
#     """
#     if meta_cols is None:
#         meta_cols = _META_COLS

#     pdf = pd.read_parquet(wide_scaled_path)
#     feat_cols = sorted([c for c in pdf.columns if c not in meta_cols])
#     if not feat_cols:
#         raise ValueError(f"No KPI columns found in '{wide_scaled_path}'.")

#     scaled_params_df = None
#     if scaled_params_path is not None:
#         scaled_params_df = pd.read_parquet(scaled_params_path)

#     groups, cell_ids_list, anchors_list, holiday_list = [], [], [], []

#     for (cell_id, anchor), g in pdf.groupby([cell_id_col, anchor_col], sort=False):
#         g_sorted = g.sort_values(hour_col)
#         kpi = g_sorted[feat_cols].to_numpy(dtype=np.float32)
#         if len(g_sorted) != SEQ_LEN or np.isnan(kpi).any():
#             continue
#         # # Drop any KPI column that contains at least one NaN anywhere in the dataset.
#         # nan_cols = [c for c in feat_cols if pdf[c].isna().any()]
#         # if nan_cols:
#         #     print(f"Dropping {len(nan_cols)} KPI column(s) with NaN values: {nan_cols}")
#         # feat_cols = [c for c in feat_cols if c not in nan_cols]
#         groups.append(kpi)
#         cell_ids_list.append(cell_id)
#         anchors_list.append(pd.Timestamp(anchor))
#         if holiday_col and holiday_col in g_sorted.columns:
#             holiday_list.append(int(g_sorted[holiday_col].iloc[0]))
#         else:
#             holiday_list.append(0)

#     X_scaled = np.stack(groups).astype(np.float32)

#     # dropping constant KPIs decide by threshold
#     if drop_constant_kpis:
#         per_feat_std = X_scaled.std(axis=(0, 1))
#         const_mask = per_feat_std < const_std_threshold
#         if const_mask.any():
#             dropped = [c for c, m in zip(feat_cols, const_mask) if m]
#             print(
#                 f"Dropping {const_mask.sum()} near-constant KPI column(s) "
#                 f"(std < {const_std_threshold}): {dropped[:8]}"
#                 f"{'...' if len(dropped) > 8 else ''}"
#             )
#             feat_cols = [c for c, m in zip(feat_cols, const_mask) if not m]
#             X_scaled = X_scaled[:, :, ~const_mask]


#     cell_ids_arr = np.array(cell_ids_list)
#     window_anchors = pd.DatetimeIndex(anchors_list)
#     holiday_flags = np.array(holiday_list, dtype=np.int32)

#     cell_encoder = LabelEncoder()
#     cell_encoder.fit(cell_ids_arr)
#     y = build_y(cell_ids_arr, window_anchors, holiday_flags, cell_encoder)
#     n_classes = len(cell_encoder.classes_)

#     print(
#         f"Loaded {len(X_scaled):,} windows  |  "
#         f"feat_dim={len(feat_cols)}  |  "
#         f"cells={n_classes}  |  "
#         f"Y width={y.shape[1]}"
#     )

#     return {
#         "X_scaled": X_scaled,
#         "y": y,
#         "window_anchors": window_anchors,
#         "cell_ids": cell_ids_arr,
#         "holiday_flags": holiday_flags,
#         "params_df": scaled_params_df,  # caller sets this to the audit/params DataFrame after scaling
#         "cell_encoder": cell_encoder,
#         "kpi_columns": feat_cols,
#         "seq_len": SEQ_LEN,
#         "feat_dim": len(feat_cols),
#         "n_classes": n_classes,
#         "output_dim": y.shape[1],
#         "arch_version": "v5",
#     }


# =============================================================================
# Stage 2 — Model building
# =============================================================================

# v4 defaults — tuned to prevent posterior collapse on 168-step telecom KPI windows.
# Key changes vs the v3 run:
#   latent_dim 32→64 : larger code, but KL is now over 64 dims (not 32×168=5376)
#   target_beta  1.0→0.1 : much softer KL pressure; decoder keeps using z
#   anneal_epochs 20→80  : slow ramp so reconstruction quality is established first
#   free_bits    0→0.5   : each latent dim contributes ≥0.5 nats, prevents full collapse


def build_model(
    seq_len: int,
    feat_dim: int,
    n_cells: int,
    global_latent_dim: int = 64,
    local_latent_dim: int = 0,
    cell_embed_dim: int = 32,
    hidden_dim: int = 256,
    n_layers: int = 2,
    use_attention: bool = True,
    n_heads: int = 4,
    beta: float = 0.0,
    learning_rate: float = 3e-4,
    free_bits_global: float = 0.002,
    free_bits_local: float = 0.0,
    output_activation: str = "sigmoid",
):
    arch = cVAE_LSTMv5Architecture(
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
        output_activation=output_activation,
    )
    model = cBetaVAE_Hierarchical(
        encoder=arch.encoder,
        decoder=arch.decoder,
        cond_layer=arch.cond_layer,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        seq_len=seq_len,
        beta=beta,
        free_bits_global=free_bits_global,
        free_bits_local=free_bits_local,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0))
    return arch, model


# v5 hyperparameters after KL-floor diagnosis (see cvae_diagnosis_and_fixes.md)
HP_V5 = dict(
    epochs=300,
    batch_size=64,
    global_latent_dim=64,
    local_latent_dim=0,
    cell_embed_dim=32,
    hidden_dim=256,
    n_layers=2,
    use_attention=True,
    n_heads=4,
    beta=0.0,
    learning_rate=3e-4,
    free_bits_global=0.002,  # floor ≈ 0.128 nats (was 46.4 at 0.1×128 + 0.05×4×168)
    free_bits_local=0.0,
    output_activation="sigmoid",
    target_beta=2e-4,
    anneal_epochs=150,
    cycle_epochs=40,
    n_cycles=6,
    cycle_ratio=0.5,
)

# =============================================================================
# Stage 3 — Training
# =============================================================================


class _KLAnneal(keras.callbacks.Callback):
    """Linearly ramp model.beta from 0 to target_beta over warmup_epochs."""

    def __init__(self, target_beta: float, warmup_epochs: int) -> None:
        super().__init__()
        self.target_beta = target_beta
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self.model.beta = self.target_beta * min(1.0, (epoch + 1) / self.warmup_epochs)


class CyclicalKLAnneal(keras.callbacks.Callback):
    """
    Cyclical beta annealing (Fu et al. 2019).
    Ramps beta within each cycle, then holds — periodic resets reduce posterior collapse.
    """

    def __init__(
        self,
        target_beta: float,
        cycle_epochs: int = 40,
        ratio: float = 0.5,
        n_cycles: int = 6,
    ) -> None:
        super().__init__()
        self.target_beta = target_beta
        self.cycle_epochs = cycle_epochs
        self.ratio = ratio
        self.n_cycles = n_cycles

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        total = self.cycle_epochs * self.n_cycles
        if epoch >= total:
            self.model.beta = self.target_beta
            return
        pos = (epoch % self.cycle_epochs) / self.cycle_epochs
        if pos < self.ratio:
            self.model.beta = self.target_beta * (pos / self.ratio)
        else:
            self.model.beta = self.target_beta


class CollapseMonitor(keras.callbacks.Callback):
    """Log |z_mean| and mean(z_log_var) each epoch to catch posterior collapse."""

    def __init__(self, X_sample: np.ndarray, y_sample: np.ndarray, n: int = 256) -> None:
        super().__init__()
        self.X = X_sample[:n].astype(np.float32)
        self.y = y_sample[:n].astype(np.float32)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if not hasattr(self.model, "encoder"):
            return
        x = keras.ops.convert_to_tensor(self.X)
        y = keras.ops.convert_to_tensor(self.y)
        enc_out = self.model.encoder([x, y], training=False)
        z_mean = enc_out[0]
        z_log_var = enc_out[1]
        mean_norm = float(keras.ops.mean(keras.ops.abs(z_mean)))
        mean_logvar = float(keras.ops.mean(z_log_var))
        beta = getattr(self.model, "beta", 0.0)
        print(
            f"  [collapse] |z_mean|={mean_norm:.4f}  "
            f"z_logvar={mean_logvar:.4f}  beta={beta:.6f}"
        )


def train_model(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    weights_path: str | Path,
    epochs: int = HP_V5["epochs"],
    batch_size: int = HP_V5["batch_size"],
    target_beta: float = HP_V5["target_beta"],
    use_cyclical_kl: bool = True,
    cycle_epochs: int = HP_V5["cycle_epochs"],
    n_cycles: int = HP_V5["n_cycles"],
    cycle_ratio: float = HP_V5["cycle_ratio"],
    anneal_epochs: int = HP_V5["anneal_epochs"],
    collapse_monitor: bool = True,
    **kwargs,
) -> keras.callbacks.History:
    """Train v5 with cyclical KL annealing and optional collapse monitoring."""
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cyclical_kl:
        kl_cb = CyclicalKLAnneal(
            target_beta,
            cycle_epochs=cycle_epochs,
            ratio=cycle_ratio,
            n_cycles=n_cycles,
        )
    else:
        kl_cb = _KLAnneal(target_beta, anneal_epochs)

    callbacks = [
        kl_cb,
        keras.callbacks.ReduceLROnPlateau(
            monitor="reconstruction_loss",
            mode="min",
            factor=0.5,
            patience=kwargs.pop("lr_patience", 20),
            min_lr=1e-5,
        ),
        keras.callbacks.ModelCheckpoint(
            str(weights_path),
            monitor="reconstruction_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="reconstruction_loss",
            mode="min",
            patience=kwargs.pop("early_stop_patience", 60),
            restore_best_weights=True,
        ),
    ]
    if collapse_monitor and _is_hierarchical_model(model):
        callbacks.append(CollapseMonitor(X_scaled, y))

    return model.fit(
        X_scaled,
        y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        **kwargs,
    )


# =============================================================================
# Stage 4 — Artifact persistence
# =============================================================================


def save_artifacts(
    out_dir: str | Path,
    data: dict,
    arch_params: dict | None = None,
) -> None:
    """Persist training artifacts needed to reload the model and generate data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_scaled.npy", data["X_scaled"])
    y_save = data.get("y", data.get("y"))
    np.save(out_dir / "y.npy", y_save)
    np.save(out_dir / "window_anchors.npy", data["window_anchors"].astype(str))
    np.save(out_dir / "cell_ids.npy", data["cell_ids"])
    np.save(out_dir / "kpi_columns.npy", np.array(data["kpi_columns"]))

    joblib.dump(data["cell_encoder"], out_dir / "cell_encoder.pkl")

    if data.get("params_df") is not None:
        data["params_df"].to_parquet(out_dir / "params_df.parquet", index=False)

    arch_params = arch_params or {}
    if "arch_version" not in arch_params and data.get("arch_version"):
        arch_params["arch_version"] = data["arch_version"]
    if arch_params:
        (out_dir / "arch_params.json").write_text(json.dumps(arch_params, indent=2))

    print(f"Artifacts saved to {out_dir}")


# def load_artifacts(
#     out_dir: str | Path,
#     weights_path: str | Path,
#     scaling_params_path: str | Path | None = None,
#     global_latent_dim: int = HP_V5["global_latent_dim"],
#     local_latent_dim: int = HP_V5["local_latent_dim"],
#     cell_embed_dim: int = HP_V5["cell_embed_dim"],
#     hidden_dim: int = 256,
#     n_layers: int = 2,
#     use_attention: bool = True,
#     n_heads: int = 4,
#     free_bits_global: float = HP_V5["free_bits_global"],
#     free_bits_local: float = HP_V5["free_bits_local"],
#     output_activation: str = HP_V5["output_activation"],
# ) -> tuple[object, dict]:
#     """Reload v5 artifacts and restore trained hierarchical model."""
#     out_dir = Path(out_dir)
#     scaling_params_path = (
#         Path(scaling_params_path)
#         if scaling_params_path is not None
#         else out_dir / "params_df.parquet"
#     )

#     X_scaled = np.load(out_dir / "X_scaled.npy")
#     y_path = out_dir / "y.npy"
#     y = np.load(y_path if y_path.exists() else out_dir / "y_extended.npy")
#     window_anchors = pd.to_datetime(np.load(out_dir / "window_anchors.npy", allow_pickle=True))
#     cell_ids = np.load(out_dir / "cell_ids.npy", allow_pickle=True)
#     kpi_columns = np.load(out_dir / "kpi_columns.npy", allow_pickle=True).tolist()
#     cell_encoder = joblib.load(out_dir / "cell_encoder.pkl")
#     params_df = pd.read_parquet(scaling_params_path) if scaling_params_path.exists() else None

#     seq_len = X_scaled.shape[1]
#     feat_dim = X_scaled.shape[2]
#     n_cells = len(cell_encoder.classes_)

#     _, model = build_model(
#         seq_len=seq_len,
#         feat_dim=feat_dim,
#         n_cells=n_cells,
#         global_latent_dim=global_latent_dim,
#         local_latent_dim=local_latent_dim,
#         cell_embed_dim=cell_embed_dim,
#         hidden_dim=hidden_dim,
#         n_layers=n_layers,
#         use_attention=use_attention,
#         n_heads=n_heads,
#         free_bits_global=free_bits_global,
#         free_bits_local=free_bits_local,
#         output_activation=output_activation,
#     )

#     dummy_X = np.zeros((1, seq_len, feat_dim), dtype=np.float32)
#     dummy_y = np.zeros((1, Y_DIM), dtype=np.float32)
#     model([dummy_X, dummy_y], training=False)
#     model.load_weights(str(weights_path))
#     print(f"Loaded v5 weights from {weights_path}")

#     data = {
#         "X_scaled": X_scaled,
#         "y": y,
#         "window_anchors": window_anchors,
#         "cell_ids": cell_ids,
#         "params_df": params_df,
#         "cell_encoder": cell_encoder,
#         "kpi_columns": kpi_columns,
#         "seq_len": seq_len,
#         "feat_dim": feat_dim,
#         "n_classes": n_cells,
#         "output_dim": Y_DIM,
#         "arch_version": "v5",
#     }
#     return model, data


def load_artifacts(
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
    y_dim: int = 6,
) -> tuple[object, dict]:
    """Reload v5 artifacts and restore trained hierarchical model."""

    cell_encoder = joblib.load(run_id_path / "cell_encoder.pkl")
    n_cells = len(cell_encoder.classes_)

    _, model = build_model(
        seq_len=168,
        feat_dim=235,
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
    model([dummy_X, dummy_y], training=False)
    model.load_weights(str(weights_path))
    print(f"Loaded v5 weights from {weights_path}")

    return model, cell_encoder


# =============================================================================
# Stage 5 — Generation
# =============================================================================


def _is_hierarchical_model(model) -> bool:
    return isinstance(model, cBetaVAE_Hierarchical)


def _holiday_col_index(y_labels: np.ndarray) -> int:
    """Holiday column: index 1 for y (6), last-5 for legacy one-hot Y."""
    return 1 if y_labels.shape[1] == Y_DIM else y_labels.shape[1] - 5


def _to_numpy(tensor) -> np.ndarray:
    """Convert any Keras/TF/PyTorch tensor to numpy, including CUDA tensors."""
    try:
        return np.asarray(tensor)
    except TypeError:
        # PyTorch CUDA tensor — must move to CPU before converting
        return tensor.detach().cpu().numpy()


def _sample_z(z_mean, z_log_var, n_samples, rng):
    # Mirror the [-6, 2] clamp applied to z_log_var during training so that
    # generation uses the same effective variance range the decoder was trained on.
    z_log_var = np.clip(z_log_var, -6.0, 2.0)
    z_std = np.exp(0.5 * z_log_var)
    eps = rng.standard_normal((n_samples, z_mean.shape[0]))
    return (z_mean + z_std * eps).astype(np.float32)


def _encode_windows(model, X_windows, y_windows, batch_size=64):
    z_means, z_log_vars = [], []
    for start in range(0, len(X_windows), batch_size):
        enc_in = model._get_encoder_input(
            X_windows[start : start + batch_size],
            y_windows[start : start + batch_size],
        )
        z_mean, z_log_var, _ = model.encoder(enc_in, training=False)
        z_means.append(_to_numpy(z_mean))
        z_log_vars.append(_to_numpy(z_log_var))
    return np.concatenate(z_means), np.concatenate(z_log_vars)


def _decode_samples(model, z_samples, y_labels, batch_size=64):
    decoded = []
    for start in range(0, len(z_samples), batch_size):
        dec_in = model._get_decoder_input(
            z_samples[start : start + batch_size],
            y_labels[start : start + batch_size],
        )
        decoded.append(_to_numpy(model.decoder(dec_in, training=False)))
    return np.concatenate(decoded)


PARAM_COLUMNS = ["scaler", "param_a", "param_b"]


def _inverse_transform(
    df: pd.DataFrame,
    params_df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    keep_params: bool = False,
) -> pd.DataFrame:
    """
    Restore original-scale value_col using fitted group params.

    Expects params_df to have columns: group_cols + ['scaler', 'param_a', 'param_b'].
    """
    joined = df.merge(params_df, on=group_cols, how="left")

    x = joined[value_col].astype("float64")
    scaler = joined["scaler"]
    param_a = joined["param_a"]
    param_b = joined["param_b"]

    can_restore = scaler.notna() & (scaler != "SKIP") & x.notna()

    restored = x.copy()

    mask_standard = can_restore & (scaler == "standard")
    mask_robust = can_restore & (scaler == "robust")
    mask_minmax = can_restore & (scaler == "minmax")
    mask_log_std = can_restore & (scaler == "log_standard")
    mask_const = can_restore & (scaler == "const_val")

    for mask in (mask_standard, mask_robust, mask_minmax):
        restored[mask] = x[mask] * param_b[mask] + param_a[mask]

    restored[mask_const] = param_a

    restored[mask_log_std] = np.expm1(
        x[mask_log_std] * param_b[mask_log_std] + param_a[mask_log_std]
    )

    joined[value_col] = restored

    if not keep_params:
        joined = joined.drop(columns=PARAM_COLUMNS, errors="ignore")

    return joined


def _inverse_transform_3d(
    x_3d: np.ndarray,
    params_df: pd.DataFrame,
    kpi_columns: list[str],
    cell_id: str,
) -> np.ndarray:
    """
    Inverse-transform decoder output using per-(distname, kpi_id) scaler params.

    params_df columns: distname, kpi_id, scaler, param_a, param_b (+ audit cols).
    kpi_columns must match params_df['kpi_id'] values.
    """
    if params_df is None:
        raise ValueError("params_df is None — set data['params_df'] before generation.")

    cell_params = params_df.loc[params_df["distname"].astype(str) == str(cell_id)]
    if cell_params.empty:
        raise ValueError(f"No scaler params found for distname='{cell_id}'.")

    lookup = (
        cell_params[["kpi_id", "scaler", "param_a", "param_b"]]
        .drop_duplicates(subset=["kpi_id"], keep="first")
        .assign(_kpi_key=lambda d: d["kpi_id"].astype(str))
        .set_index("_kpi_key")
    )

    scalers, a_vals, b_vals, missing = [], [], [], []
    for kpi in kpi_columns:
        key = str(kpi)
        if key not in lookup.index:
            missing.append(kpi)
            continue
        row = lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        scalers.append(row["scaler"])
        a_vals.append(row["param_a"])
        b_vals.append(row["param_b"])

    if missing:
        raise ValueError(
            f"No scaler params for {len(missing)} KPI(s) on '{cell_id}'. "
            f"First few: {missing[:5]}. "
            "Check that kpi_columns match params_df['kpi_id']."
        )

    scalers = np.array(scalers)
    a = np.array(a_vals, dtype=np.float64)
    b = np.array(b_vals, dtype=np.float64)

    x = x_3d.astype(np.float64)  # (N, seq_len, feat_dim)
    result = x.copy()

    linear_mask = np.isin(scalers, ["standard", "robust", "minmax"])
    if linear_mask.any():
        result[:, :, linear_mask] = x[:, :, linear_mask] * b[linear_mask] + a[linear_mask]

    log_mask = scalers == "log_standard"
    if log_mask.any():
        result[:, :, log_mask] = np.expm1(x[:, :, log_mask] * b[log_mask] + a[log_mask])

    const_mask = scalers == "const_val"
    if const_mask.any():
        result[:, :, const_mask] = a[const_mask]

    return result.astype(np.float32)


def _filter_windows(
    X_scaled,
    y,
    window_anchors,
    cell_ids,
    cell_id,
    date_start=None,
    date_end=None,
    holiday=None,
):
    anchors_dt = pd.to_datetime(window_anchors)
    mask = cell_ids == cell_id
    if date_start is not None:
        mask &= anchors_dt >= pd.Timestamp(date_start)
    if date_end is not None:
        mask &= anchors_dt <= pd.Timestamp(date_end)
    if holiday is not None:
        mask &= y[:, _holiday_col_index(y)].astype(int) == int(holiday)
    return X_scaled[mask], y[mask], anchors_dt[mask]


# def _run_pipeline_v4(model, X_windows, y_windows, rng, batch_size=64):
#     """Encode real windows → sample posterior → decode (v4 path)."""
#     z_means, z_log_vars = _encode_windows(model, X_windows, y_windows, batch_size)
#     all_z = np.concatenate(
#         [_sample_z(z_means[i], z_log_vars[i], 1, rng) for i in range(len(X_windows))]
#     )
#     return _decode_samples(model, all_z, y_windows, batch_size)


def _run_pipeline(model, y_windows, batch_size=64):
    """Prior sampling from N(0,I) — true synthetic generation for v5."""
    decoded = []
    for start in range(0, len(y_windows), batch_size):
        yb = y_windows[start : start + batch_size]
        x_syn, _ = model.generate(yb)
        decoded.append(_to_numpy(x_syn))
    return np.concatenate(decoded)


def _generate_windows(model, X_windows, y_windows, rng, batch_size=64):
    del X_windows, rng  # v5 generates from prior; X/rng kept for API compatibility
    return _run_pipeline(model, y_windows, batch_size)


def generate_timespan(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    window_anchors,
    cell_ids: np.ndarray,
    kpi_columns: list[str],
    params_df: pd.DataFrame,
    cell_id: str,
    date_start: str,
    date_end: str,
    holiday: int | None = None,
    seed: int | None = None,
    batch_size: int = 64,
    **_,
) -> pd.DataFrame:
    """Generate synthetic KPIs for a cell within a calendar date range."""
    rng = np.random.default_rng(seed)
    X_m, y_m, anchors_m = _filter_windows(
        X_scaled,
        y,
        window_anchors,
        cell_ids,
        cell_id,
        date_start,
        date_end,
        holiday,
    )
    if len(X_m) == 0:
        raise ValueError(
            f"No windows found for cell_id='{cell_id}' "
            f"date_start={date_start}, date_end={date_end}, holiday={holiday}."
        )
    print(f"Matched {len(X_m)} windows → {len(X_m) * SEQ_LEN:,} rows")

    x_syn = _generate_windows(model, X_m, y_m, rng, batch_size)
    x_inv = x_syn
    # x_inv = _inverse_transform_3d(x_syn, params_df, kpi_columns, cell_id)

    n_windows, seq_len, n_kpis = x_inv.shape
    anchors_arr = anchors_m.to_numpy()
    kpi_flat = x_inv.reshape(n_windows * seq_len, n_kpis)

    df = pd.DataFrame(kpi_flat, columns=kpi_columns)
    df.insert(0, "seed", seed)
    df.insert(
        0,
        "timestamp",
        pd.to_datetime(np.repeat(anchors_arr, seq_len))
        + pd.to_timedelta(np.tile(np.arange(seq_len), n_windows), unit="h"),
    )
    df.insert(0, "window_anchor", np.repeat(anchors_arr, seq_len))
    df.insert(0, "cell_id", cell_id)
    return df


def generate_n_weeks(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    window_anchors,
    cell_ids: np.ndarray,
    params_df: pd.DataFrame,
    kpi_columns: list[str],
    cell_id: str,
    n_weeks: int,
    holiday: int | None = None,
    seed: int | None = None,
    batch_size: int = 64,
    **_,
) -> pd.DataFrame:
    """Generate N continuous synthetic weeks for a given cell."""
    if n_weeks < 1:
        raise ValueError("n_weeks must be >= 1")

    rng = np.random.default_rng(seed)
    X_pool, y_pool, _ = _filter_windows(
        X_scaled,
        y,
        window_anchors,
        cell_ids,
        cell_id,
        holiday=holiday,
    )
    if len(X_pool) == 0:
        raise ValueError(f"No windows found for cell_id='{cell_id}' with holiday={holiday}.")

    pool_size = len(X_pool)
    idx = rng.choice(pool_size, size=n_weeks, replace=(n_weeks > pool_size))
    print(f"Sampled {n_weeks} windows from pool of {pool_size} → {n_weeks * SEQ_LEN:,} rows")

    x_syn = _generate_windows(model, X_pool[idx], y_pool[idx], rng, batch_size)
    # x_inv = _inverse_transform_3d(x_syn, params_df, kpi_columns, cell_id)
    x_inv = x_syn
    n_w, seq_len, n_kpis = x_inv.shape
    week_idx = np.repeat(np.arange(n_w), seq_len)
    hour_in_week = np.tile(np.arange(seq_len), n_w)
    kpi_flat = x_inv.reshape(n_w * seq_len, n_kpis)

    df = pd.DataFrame(kpi_flat, columns=kpi_columns)
    df.insert(0, "seed", seed)
    df.insert(
        0,
        "timestamp",
        SYNTHETIC_ORIGIN + pd.to_timedelta(week_idx * seq_len + hour_in_week, unit="h"),
    )
    df.insert(0, "hour_in_week", hour_in_week)
    df.insert(0, "week_number", week_idx + 1)
    df.insert(0, "cell_id", cell_id)
    return df
