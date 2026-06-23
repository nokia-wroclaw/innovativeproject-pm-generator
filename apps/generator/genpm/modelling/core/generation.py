"""Synthetic KPI generation: the generate_windows entrypoint and inverse-scaling utils."""

import numpy as np
import pandas as pd

from genpm.utils.logger import get_logger

logger = get_logger()

PARAM_COLUMNS = ["scaler", "param_a", "param_b"]


def _to_numpy(tensor) -> np.ndarray:
    """Convert any Keras/TF/PyTorch tensor to numpy, including CUDA tensors."""
    try:
        return np.asarray(tensor)
    except (TypeError, RuntimeError):
        # PyTorch tensor with requires_grad=True or CUDA tensor — detach first
        return tensor.detach().cpu().numpy()


def _inverse_transform(
    df: pd.DataFrame,
    params_df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    keep_params: bool = False,
) -> pd.DataFrame:
    """Restore original-scale value_col using fitted group params.

    Currently unused by ``generate_windows`` (which returns scaled output); kept as a
    ready utility for when callers need real-scale values from a long-format frame.
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
    """Inverse-transform decoder output using per-(distname, kpi_id) scaler params.

    params_df columns: distname, kpi_id, scaler, param_a, param_b (+ audit cols).
    kpi_columns must match params_df['kpi_id'] values.

    Currently unused by ``generate_windows`` (which returns scaled output); kept as a
    ready utility for when callers need real-scale values from a (N, T, F) array.
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


def _run_batched_generation(
    model, y_windows: np.ndarray, batch_size: int, c_windows: np.ndarray | None = None
) -> np.ndarray:
    decoded = []
    for start in range(0, len(y_windows), batch_size):
        yb = y_windows[start : start + batch_size]
        if c_windows is not None:
            x_syn, _ = model.generate(yb, calendar=c_windows[start : start + batch_size])
        else:
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

    This is the generation entrypoint shared by every model family — cVAE, GAN, and
    diffusion all expose ``generate(y[, calendar]) -> (x, y)``. Configs come from
    ``cell_configs`` if given, else are looked up from ``cell_config_map`` by
    ``cell_id``. seq_len and n_dim are taken from the generated array (which matches
    the trained checkpoint).

    Args:
        model: A trained generator exposing ``generate``; ``model.cond_dim`` > 0
            signals it needs per-timestep calendar features (rebuilt here).
        config_encoder: Fitted one-hot encoder for config values.
        cell_config_map: ``{"map": {cell_id: config_values}}`` saved at training time.
        cell_id: Cell to generate for; its configs are looked up unless
            ``cell_configs`` is given. Also labels the output (``cell_id`` column).
        anchor_date: Start date of the first synthetic week.
        n_weeks: Number of consecutive weekly windows to generate.
        holiday: Holiday flag written into every window's ``y``.
        batch_size: Generation batch size.
        seed: Currently unused — reserved for reproducibility once a seedable prior
            path is added (diffusion/GAN noise is drawn from the keras RNG today).
        kpi_list: Column names for the output KPIs.
        cell_configs: Explicit config values; when provided, output is keyed by a
            ``config_id`` column (the joined values) instead of ``cell_id``.

    Returns:
        Long-format DataFrame: id column, ``window_anchor``, ``timestamp``, one
        column per KPI; ``n_weeks * seq_len`` rows.

    Raises:
        ValueError: If neither ``cell_id`` nor ``cell_configs`` is provided, or the
            ``cell_id`` is absent from ``cell_config_map``.
    """
    from genpm.modelling.core.data import build_calendar_features, encode_seasonal_features

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

    # Per-timestep calendar conditioning, rebuilt from the target anchors with the
    # same logic used at training time (only if the model was trained with it).
    c_windows = None
    if getattr(model, "cond_dim", 0) > 0:
        c_windows = build_calendar_features(anchors_arr, seq_len=model._seq_len)

    kpi_array = _run_batched_generation(
        model, y_windows, batch_size=batch_size, c_windows=c_windows
    )
    _, seq_len, n_dim = kpi_array.shape
    kpi_flat = kpi_array.reshape(n_weeks * seq_len, n_dim)

    df = pd.DataFrame(kpi_flat, columns=kpi_list)
    df.insert(
        0,
        "timestamp",
        pd.to_datetime(np.repeat(anchors_arr, seq_len))
        + pd.to_timedelta(np.tile(np.arange(seq_len), n_weeks), unit="h"),
    )
    df.insert(0, "window_anchor", np.repeat(anchors_arr, seq_len))
    df.insert(0, id_col, id_val)

    return df
