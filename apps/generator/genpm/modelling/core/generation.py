"""Synthetic KPI generation: prior sampling, inverse scaling, and output formatting."""

import numpy as np
import pandas as pd

from genpm.modelling.core.data import CONTEXT_DIM, SEQ_LEN, SYNTHETIC_ORIGIN
from genpm.utils.logger import get_logger

logger = get_logger()

PARAM_COLUMNS = ["scaler", "param_a", "param_b"]


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


def _inverse_transform(
    df: pd.DataFrame,
    params_df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    keep_params: bool = False,
) -> pd.DataFrame:
    """Restore original-scale value_col using fitted group params."""
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

    x = x_3d.astype(np.float64)
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


def _holiday_col_index(y_labels: np.ndarray) -> int:
    """Holiday column index — first of the trailing context block [holiday | seasonal(4)]."""
    return y_labels.shape[1] - CONTEXT_DIM


def _select_cell_windows(
    X_scaled,
    y,
    window_anchors,
    cell_ids,
    cell_id,
    date_start=None,
    date_end=None,
    holiday=None,
):
    """Filter the training pool to windows matching cell, date range, and holiday flag."""
    anchors_dt = pd.to_datetime(window_anchors)
    mask = cell_ids == cell_id
    if date_start is not None:
        mask &= anchors_dt >= pd.Timestamp(date_start)
    if date_end is not None:
        mask &= anchors_dt <= pd.Timestamp(date_end)
    if holiday is not None:
        mask &= y[:, _holiday_col_index(y)].astype(int) == int(holiday)
    return X_scaled[mask], y[mask], anchors_dt[mask]


def _decode_from_prior(model, y_windows, batch_size=64):
    """Sample from N(0, I) prior and decode — true synthetic generation (v5)."""
    decoded = []
    for start in range(0, len(y_windows), batch_size):
        yb = y_windows[start : start + batch_size]
        x_syn, _ = model.generate(yb)
        decoded.append(_to_numpy(x_syn))
    return np.concatenate(decoded)


def _generate_window_batch(model, X_windows, y_windows, rng, batch_size=64):
    del X_windows, rng  # v5 generates from prior; X/rng kept for API compatibility
    return _decode_from_prior(model, y_windows, batch_size)


def generate_for_date_range(
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
    X_m, y_m, anchors_m = _select_cell_windows(
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
    logger.info(
        f"Matched {len(X_m)} windows for '{cell_id}' [{date_start} → {date_end}] "
        f"→ {len(X_m) * SEQ_LEN:,} rows"
    )

    x_syn = _generate_window_batch(model, X_m, y_m, rng, batch_size)
    x_inv = x_syn

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


def generate_n_synthetic_weeks(
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
    X_pool, y_pool, _ = _select_cell_windows(
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
    logger.info(f"Pool for '{cell_id}': {pool_size} windows (holiday={holiday})")
    idx = rng.choice(pool_size, size=n_weeks, replace=(n_weeks > pool_size))
    logger.info(f"Sampled {n_weeks} windows → {n_weeks * SEQ_LEN:,} rows")

    x_syn = _generate_window_batch(model, X_pool[idx], y_pool[idx], rng, batch_size)
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


def _run_batched_generation(model, y_windows: np.ndarray, batch_size: int) -> np.ndarray:
    decoded = []
    for start in range(0, len(y_windows), batch_size):
        yb = y_windows[start : start + batch_size]
        x_syn, _ = model.generate(yb)
        decoded.append(_to_numpy(x_syn))
    return np.concatenate(decoded)


def _config_label(cell_configs: list) -> str:
    """A filesystem-friendly output label derived from explicit config values."""
    return "config_" + "_".join(str(c) for c in cell_configs)


def generate_windows(
    model,
    config_encoder,
    cell_config_map: dict,
    cell_id: str | None,
    anchor_date: str,
    n_weeks: int,
    holiday: int,
    batch_size: int,
    seed: int,
    kpi_list: list,
    cell_configs: list | None = None,
) -> pd.DataFrame:
    """Generate synthetic windows conditioned on cell config values.

    Configs come from `cell_configs` if given, else are looked up from
    `cell_config_map` by `cell_id`. The output is keyed by a `cell_id` column when a
    cell_id is provided, otherwise by a `config_id` column holding the joined config
    values. seq_len and n_dim are taken from the generated array (which matches the
    trained checkpoint).
    """
    from genpm.modelling.core.data import encode_seasonal_features

    if cell_configs is None:
        if cell_id is None:
            raise ValueError("Provide either cell_id or cell_configs.")
        configs_map = cell_config_map["map"]
        if str(cell_id) not in configs_map:
            raise ValueError(
                f"cell_id='{cell_id}' not found in cell_config_map; pass cell_configs explicitly."
            )
        cell_configs = configs_map[str(cell_id)]

    # cell_id mode keys output by "cell_id"; config-first mode keys by "config_id"
    # (the joined config values) so the column name reflects its actual contents.
    if cell_id is not None:
        id_col, id_val = "cell_id", cell_id
    else:
        id_col, id_val = "config_id", "|".join(str(c) for c in cell_configs)

    config_onehot = config_encoder.transform([cell_configs])[0].astype(np.float32)

    anchors = []
    y_windows = []
    for week in range(n_weeks):
        anchor = pd.Timestamp(anchor_date) + pd.Timedelta(weeks=week)
        seasonal = encode_seasonal_features(anchor)
        y_windows.append([*config_onehot, holiday, *seasonal])
        anchors.append(anchor)

    y_windows = np.array(y_windows, dtype=np.float32)
    anchors_arr = np.array(anchors)

    kpi_array = _run_batched_generation(model, y_windows, batch_size=batch_size)
    _, seq_len, n_dim = kpi_array.shape
    kpi_flat = kpi_array.reshape(n_weeks * seq_len, n_dim)

    df = pd.DataFrame(kpi_flat, columns=kpi_list)
    df.insert(0, "seed", seed)
    df.insert(
        0,
        "timestamp",
        pd.to_datetime(np.repeat(anchors_arr, seq_len))
        + pd.to_timedelta(np.tile(np.arange(seq_len), n_weeks), unit="h"),
    )
    df.insert(0, "window_anchor", np.repeat(anchors_arr, seq_len))
    df.insert(0, id_col, id_val)

    return df
