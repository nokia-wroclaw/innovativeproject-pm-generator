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

Typical notebook usage
----------------------
    from scripts.cvae_utils import (
        prepare_data, build_model, train_model,
        save_artifacts, load_artifacts,
        generate_timespan, generate_n_weeks,
    )

    ARTIFACTS_DIR = Path("training_data/greg_tmp")

    data = prepare_data(ARTIFACTS_DIR / "wide_win_idx")
    arch, model = build_model(data["seq_len"], data["feat_dim"], data["output_dim"])
    history = train_model(model, data["X_scaled"], data["y_extended"],
                          weights_path=ARTIFACTS_DIR / "cvae_lstm.weights.h5")
    save_artifacts(ARTIFACTS_DIR, model, data)

    df = generate_timespan(model=model, **data, cell_id="...", date_start="...", date_end="...")
    df = generate_n_weeks(model=model, **data, cell_id="...", n_weeks=3)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from model_utils import cBetaVAE, cVAE_LSTMv4Architecture
from sklearn.preprocessing import LabelEncoder

SEQ_LEN = 168
SYNTHETIC_ORIGIN = pd.Timestamp("1970-01-01")

_META_COLS = {"distname", "bts_id", "window_anchor", "hour_idx"}


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


def prepare_data(
    parquet_path: str | Path,
    cell_id_col: str = "distname",
    anchor_col: str = "window_anchor",
    hour_col: str = "hour_idx",
    holiday_col: str | None = None,
    meta_cols: set[str] | None = None,
) -> dict:
    """
    Load the wide-format windowed parquet, build X and extended Y tensors.

    Returns dict with keys:
        X_scaled, y_extended, window_anchors, cell_ids, scaler,
        cell_encoder, kpi_columns, seq_len, feat_dim, n_classes, output_dim
    """
    if meta_cols is None:
        meta_cols = _META_COLS

    pdf = pd.read_parquet(parquet_path)
    feat_cols = sorted([c for c in pdf.columns if c not in meta_cols])

    groups, cell_ids_list, anchors_list, holiday_list = [], [], [], []

    for (cell_id, anchor), g in pdf.groupby([cell_id_col, anchor_col], sort=False):
        g_sorted = g.sort_values(hour_col)
        kpi = g_sorted[feat_cols].to_numpy(dtype=np.float32)
        if len(g_sorted) != SEQ_LEN or np.isnan(kpi).any():
            continue
        # # Drop any KPI column that contains at least one NaN anywhere in the dataset.
        # nan_cols = [c for c in feat_cols if pdf[c].isna().any()]
        # if nan_cols:
        #     print(f"Dropping {len(nan_cols)} KPI column(s) with NaN values: {nan_cols}")
        # feat_cols = [c for c in feat_cols if c not in nan_cols]
        groups.append(kpi)
        cell_ids_list.append(cell_id)
        anchors_list.append(pd.Timestamp(anchor))
        if holiday_col and holiday_col in g_sorted.columns:
            holiday_list.append(int(g_sorted[holiday_col].iloc[0]))
        else:
            holiday_list.append(0)

    X_scaled = np.stack(groups).astype(np.float32)

    cell_ids_arr = np.array(cell_ids_list)
    window_anchors = pd.DatetimeIndex(anchors_list)
    holiday_flags = np.array(holiday_list, dtype=np.int32)

    cell_encoder = LabelEncoder()
    cell_encoder.fit(cell_ids_arr)
    y_extended = build_y_extended(cell_ids_arr, window_anchors, holiday_flags, cell_encoder)
    n_classes = len(cell_encoder.classes_)

    print(
        f"Loaded {len(X_scaled):,} windows  |  "
        f"feat_dim={len(feat_cols)}  |  "
        f"cells={n_classes}  |  "
        f"Y width={y_extended.shape[1]}"
    )

    return {
        "X_scaled": X_scaled,
        "y_extended": y_extended,
        "window_anchors": window_anchors,
        "cell_ids": cell_ids_arr,
        "params_df": None,  # caller sets this to the audit/params DataFrame after scaling
        "cell_encoder": cell_encoder,
        "kpi_columns": feat_cols,
        "seq_len": SEQ_LEN,
        "feat_dim": len(feat_cols),
        "n_classes": n_classes,
        "output_dim": y_extended.shape[1],
    }


# =============================================================================
# Stage 2 — Model building
# =============================================================================

# v4 defaults — tuned to prevent posterior collapse on 168-step telecom KPI windows.
# Key changes vs the v3 run:
#   latent_dim 32→64 : larger code, but KL is now over 64 dims (not 32×168=5376)
#   target_beta  1.0→0.1 : much softer KL pressure; decoder keeps using z
#   anneal_epochs 20→80  : slow ramp so reconstruction quality is established first
#   free_bits    0→0.5   : each latent dim contributes ≥0.5 nats, prevents full collapse
_DEFAULT_HP = dict(
    latent_dim=64,
    hidden_dim=256,
    n_layers=2,
    use_attention=True,
    n_heads=4,
    beta=0.0,
    learning_rate=3e-4,
    free_bits=0.5,
)


def build_model(
    seq_len: int,
    feat_dim: int,
    output_dim: int,
    latent_dim: int = _DEFAULT_HP["latent_dim"],
    hidden_dim: int = _DEFAULT_HP["hidden_dim"],
    n_layers: int = _DEFAULT_HP["n_layers"],
    use_attention: bool = _DEFAULT_HP["use_attention"],
    n_heads: int = _DEFAULT_HP["n_heads"],
    beta: float = _DEFAULT_HP["beta"],
    learning_rate: float = _DEFAULT_HP["learning_rate"],
    free_bits: float = _DEFAULT_HP["free_bits"],
) -> tuple:
    """Build and compile cVAE_LSTMv4Architecture + cBetaVAE. Returns (arch, model)."""
    arch = cVAE_LSTMv4Architecture(
        seq_len=seq_len,
        feat_dim=feat_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_attention=use_attention,
        n_heads=n_heads,
    )
    model = cBetaVAE(
        arch.encoder,
        arch.decoder,
        latent_dim,
        temporal=False,
        beta=beta,
        global_z=True,
        free_bits=free_bits,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0))
    return arch, model


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


def train_model(
    model,
    X_scaled: np.ndarray,
    y_extended: np.ndarray,
    weights_path: str | Path,
    epochs: int = 300,
    batch_size: int = 64,
    target_beta: float = 0.1,
    anneal_epochs: int = 80,
    lr_patience: int = 20,
    early_stop_patience: int = 60,
    min_lr: float = 1e-5,
) -> keras.callbacks.History:
    """Train cBetaVAE with KL annealing, LR scheduling, checkpointing, early stopping."""
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        _KLAnneal(target_beta, anneal_epochs),
        keras.callbacks.ReduceLROnPlateau(
            monitor="reconstruction_loss",
            mode="min",
            factor=0.5,
            patience=lr_patience,
            min_lr=min_lr,
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
            patience=early_stop_patience,
            restore_best_weights=True,
        ),
    ]

    return model.fit(
        X_scaled,
        y_extended,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )


# =============================================================================
# Stage 4 — Artifact persistence
# =============================================================================


def save_artifacts(
    out_dir: str | Path,
    model,
    data: dict,
    arch_params: dict | None = None,
) -> None:
    """Persist training artifacts needed to reload the model and generate data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_scaled.npy", data["X_scaled"])
    np.save(out_dir / "y_extended.npy", data["y_extended"])
    np.save(out_dir / "window_anchors.npy", data["window_anchors"].astype(str))
    np.save(out_dir / "cell_ids.npy", data["cell_ids"])
    np.save(out_dir / "kpi_columns.npy", np.array(data["kpi_columns"]))

    joblib.dump(data["scaler"], out_dir / "scaler.pkl")
    joblib.dump(data["cell_encoder"], out_dir / "cell_encoder.pkl")

    if "params_df" in data and data["params_df"] is not None:
        data["params_df"].to_parquet(out_dir / "params_df.parquet", index=False)

    if arch_params is not None:
        (out_dir / "arch_params.json").write_text(json.dumps(arch_params, indent=2))

    print(f"Artifacts saved to {out_dir}")


def load_artifacts(
    out_dir: str | Path,
    weights_path: str | Path,
    scaling_params_path: str | Path,
    latent_dim: int = _DEFAULT_HP["latent_dim"],
    hidden_dim: int = _DEFAULT_HP["hidden_dim"],
    n_layers: int = _DEFAULT_HP["n_layers"],
    use_attention: bool = _DEFAULT_HP["use_attention"],
    n_heads: int = _DEFAULT_HP["n_heads"],
    free_bits: float = _DEFAULT_HP["free_bits"],
) -> tuple[object, dict]:
    """Reload artifacts and restore trained model. Returns (model, data)."""
    out_dir = Path(out_dir)

    X_scaled = np.load(out_dir / "X_scaled.npy")
    y_extended = np.load(out_dir / "y_extended.npy")
    window_anchors = pd.to_datetime(np.load(out_dir / "window_anchors.npy"))
    cell_ids = np.load(out_dir / "cell_ids.npy", allow_pickle=True)
    kpi_columns = np.load(out_dir / "kpi_columns.npy", allow_pickle=True).tolist()
    scaler = joblib.load(out_dir / "scaler.pkl")
    cell_encoder = joblib.load(out_dir / "cell_encoder.pkl")

    params_df = pd.read_parquet(scaling_params_path) if scaling_params_path.exists() else None

    seq_len = X_scaled.shape[1]
    feat_dim = X_scaled.shape[2]
    output_dim = y_extended.shape[1]

    _, model = build_model(
        seq_len=seq_len,
        feat_dim=feat_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_attention=use_attention,
        n_heads=n_heads,
        free_bits=free_bits,
    )

    # cBetaVAE (keras.Model subclass) must be built before weights can be
    # restored. A single forward pass on a 1-sample dummy input is sufficient.
    dummy_X = np.zeros((1, seq_len, feat_dim), dtype=np.float32)
    dummy_y = np.zeros((1, output_dim), dtype=np.float32)
    model((dummy_X, dummy_y), training=False)

    model.load_weights(str(weights_path))
    print(f"Loaded weights from {weights_path}")

    data = {
        "X_scaled": X_scaled,
        "y_extended": y_extended,
        "window_anchors": window_anchors,
        "cell_ids": cell_ids,
        "params_df": params_df,
        "scaler": scaler,
        "cell_encoder": cell_encoder,
        "kpi_columns": kpi_columns,
        "seq_len": seq_len,
        "feat_dim": feat_dim,
        "n_classes": output_dim - 5,
        "output_dim": output_dim,
    }
    return model, data


# =============================================================================
# Stage 5 — Generation
# =============================================================================


def _holiday_col_index(y_extended: np.ndarray) -> int:
    return y_extended.shape[1] - 5


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
    y_extended,
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
        mask &= y_extended[:, _holiday_col_index(y_extended)].astype(int) == int(holiday)
    return X_scaled[mask], y_extended[mask], anchors_dt[mask]


def _run_pipeline(model, X_windows, y_windows, params_df, rng, batch_size):
    z_means, z_log_vars = _encode_windows(model, X_windows, y_windows, batch_size)
    all_z = np.concatenate(
        [_sample_z(z_means[i], z_log_vars[i], 1, rng) for i in range(len(X_windows))]
    )
    x_syn = _decode_samples(model, all_z, y_windows, batch_size)
    return x_syn


def generate_timespan(
    model,
    X_scaled: np.ndarray,
    y_extended: np.ndarray,
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
        y_extended,
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

    x_syn = _run_pipeline(model, X_m, y_m, rng, batch_size)
    x_inv = _inverse_transform_3d(x_syn, params_df, kpi_columns, cell_id)

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
    y_extended: np.ndarray,
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
        y_extended,
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

    x_syn = _run_pipeline(model, X_pool[idx], y_pool[idx], rng, batch_size)
    x_inv = _inverse_transform_3d(x_syn, params_df, kpi_columns, cell_id)
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
