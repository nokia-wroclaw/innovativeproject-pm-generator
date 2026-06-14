"""Data loading and conditioning vector construction for the cVAE-LSTM pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import OneHotEncoder

from genpm.utils.logger import get_logger

logger = get_logger()

SEQ_LEN = 168
# Conditioning context = holiday (1) + seasonal (4). The config one-hot block is
# data-dependent, so the full y width (CONTEXT_DIM + one-hot width) is computed at load time.
CONTEXT_DIM = 5
SYNTHETIC_ORIGIN = pd.Timestamp("1970-01-01")
CONST_KPI_STD_THRESHOLD = 0.05

# Per-cell categorical configuration columns are detected by this prefix rather than
# hardcoded names. They define the conditioning identity (one-hot encoded) instead of
# the cell's distname.
CELL_CONFIG_PREFIX = "[CELL] "

_META_COLS = {
    "distname",
    "bts_id",
    "window_anchor",
    "hour_idx",
    "n_hours",
    "imputed_flag",
    "holiday",
    "is_holiday",
}


def detect_config_cols(schema_names: list[str]) -> list[str]:
    """Config columns are those carrying the per-cell CELL_CONFIG_PREFIX, sorted."""
    return sorted(c for c in schema_names if c.startswith(CELL_CONFIG_PREFIX))


def encode_seasonal_features(anchor: pd.Timestamp) -> np.ndarray:
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


def build_conditioning_vector(
    configs: np.ndarray,
    window_anchors: np.ndarray,
    config_encoder: OneHotEncoder,
    holiday_flags: np.ndarray | None = None,
) -> np.ndarray:
    """
    Conditioning vector — cell configs are one-hot encoded and concatenated.

    Layout: [one-hot configs (C) | holiday (1) | seasonal (4)]
    Returns shape (N, C + CONTEXT_DIM).
    """
    n = len(configs)
    config_onehot = config_encoder.transform(configs).astype(np.float32)
    if holiday_flags is None:
        holiday_flags = np.zeros(n, dtype=np.int32)
    y_holiday = holiday_flags.reshape(-1, 1).astype(np.float32)
    _anchors = pd.DatetimeIndex(window_anchors)
    _weeks = _anchors.isocalendar().week.to_numpy(dtype=np.float32)
    _months = _anchors.month.to_numpy(dtype=np.float32)
    _2pi = 2.0 * np.pi
    y_seasonal = np.column_stack(
        [
            np.sin(_2pi * _weeks / 52),
            np.cos(_2pi * _weeks / 52),
            np.sin(_2pi * _months / 12),
            np.cos(_2pi * _months / 12),
        ]
    ).astype(np.float32)
    return np.concatenate(
        [config_onehot, y_holiday, y_seasonal],
        axis=1,
    ).astype(np.float32)


def _stack_hourly_parquet_to_windows(
    path: str | Path,
    feat_cols: list[str],
    cell_id_col: str,
    anchor_col: str,
    hour_col: str,
    holiday_col: str | None,
    config_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Read long-format hourly parquet and stack into (N, seq_len, n_kpis) windows.

    Expected schema:
        distname       string    — cell identifier
        window_anchor  timestamp — start of the 168-hour window
        hour_idx       integer   — 0..seq_len-1 within the window
        [CELL] config* string    — per-cell categorical config (constant per cell)
        NR_*           double    — KPI values

    Groups by (cell_id_col, anchor_col), sorts by hour_col. Config columns are
    constant per cell, so they're collapsed with .first() like the holiday flag.
    Windows with wrong row count or any NaN are skipped.

    Note on conditioning: grouping is by (cell_id, anchor) — not by config — because a
    window is ONE physical cell's seq_len-hour sequence; grouping by config would merge
    rows from different cells into a single nonsensical window. Conditioning on config is
    done entirely via the y vector (build_conditioning_vector), where many cell-windows
    that share a config carry the same one-hot, so the model learns P(X | config). The
    cell identity (distname) is never fed to the model; it is only a grouping/output key.

    Returns (X, cell_ids, window_anchors, holiday_flags, configs) where configs has
    shape (N, len(config_cols)).
    """
    path = Path(path)
    parquet_glob = str(path / "**" / "*.parquet")
    schema_names = pl.read_parquet_schema(parquet_glob).names()
    use_holiday = holiday_col is not None and holiday_col in schema_names

    missing_cfg = [c for c in config_cols if c not in schema_names]
    if missing_cfg:
        raise ValueError(f"Config column(s) missing from parquet schema: {missing_cfg}")

    needed_cols = [cell_id_col, anchor_col, hour_col] + feat_cols + config_cols
    if use_holiday:
        needed_cols.append(holiday_col)

    agg_exprs = [pl.col(c).sort_by(hour_col) for c in feat_cols]
    agg_exprs += [pl.col(c).first() for c in config_cols]
    if use_holiday:
        agg_exprs.append(pl.col(holiday_col).first())

    logger.info(
        f"Stacking hourly rows into windows — grouping by ({cell_id_col}, {anchor_col}), "
        f"sorting by {hour_col}"
    )

    agg_df = (
        pl.scan_parquet(parquet_glob)
        .select(needed_cols)
        .group_by([cell_id_col, anchor_col])
        .agg(agg_exprs)
        .collect()
    )

    n_total = len(agg_df)
    right_len_df = agg_df.filter(pl.col(feat_cols[0]).list.len() == seq_len)
    n_skipped_len = n_total - len(right_len_df)

    X = np.stack(
        [np.array(right_len_df[c].to_list(), dtype=np.float32) for c in feat_cols],
        axis=-1,
    )

    nan_mask = np.isnan(X).any(axis=(1, 2))
    n_skipped_nan = int(nan_mask.sum())

    if n_skipped_len:
        logger.warning(
            f"Skipped {n_skipped_len:,} window(s) — wrong row count (expected {seq_len})"
        )
    if n_skipped_nan:
        logger.warning(f"Skipped {n_skipped_nan:,} window(s) — contained NaN values")

    valid_df = right_len_df.filter(~pl.Series(nan_mask))
    X_valid = X[~nan_mask]

    if len(X_valid) == 0:
        raise ValueError(
            f"No valid windows — expected exactly {seq_len} rows "
            f"per ({cell_id_col}, {anchor_col}) group with no NaNs."
        )

    cell_ids = valid_df[cell_id_col].to_numpy()
    window_anchors = pd.DatetimeIndex(valid_df[anchor_col].to_numpy())
    configs = valid_df.select(config_cols).to_numpy().astype(str)

    if use_holiday:
        holiday_flags = valid_df[holiday_col].to_numpy().astype(np.int32)
    else:
        holiday_flags = np.zeros(len(X_valid), dtype=np.int32)

    logger.info(
        f"Stacked {len(X_valid):,} windows — shape ({len(X_valid)}, {seq_len}, {len(feat_cols)})"
    )
    return X_valid, cell_ids, window_anchors, holiday_flags, configs


def load_training_windows(
    wide_scaled_path: str | Path,
    scaled_params_path: str | Path | None = None,
    cell_id_col: str = "distname",
    anchor_col: str = "window_anchor",
    hour_col: str = "hour_idx",
    holiday_col: str | None = None,
    config_cols: list[str] | None = None,
    meta_cols: set[str] | None = None,
    drop_constant_kpis: bool = True,
    const_std_threshold: float = CONST_KPI_STD_THRESHOLD,
) -> dict:
    """
    Load windowed parquet and build X tensor + conditioning vector Y.

    Expected schema (hourly format):
        distname       string    — cell identifier (grouping key)
        window_anchor  timestamp — start of the 168-hour window
        hour_idx       integer   — 0..167 within the window
        [CELL] config* string    — per-cell categorical config (conditioning identity)
        NR_*           double    — one column per KPI

    Conditioning is built from the config columns (one-hot encoded), not the
    distname. distname is kept as cell_ids for output identity / inverse scaling.

    Returns dict with keys:
        X_scaled, y, window_anchors, cell_ids, configs, holiday_flags,
        params_df, config_encoder, kpi_columns, seq_len, feat_dim, y_dim, config_dims
    """
    if meta_cols is None:
        meta_cols = _META_COLS

    logger.info(f"Loading windowed parquet from {wide_scaled_path}")
    schema = pl.read_parquet_schema(str(Path(wide_scaled_path) / "**" / "*.parquet"))
    if config_cols is None:
        config_cols = detect_config_cols(schema.names())
    if not config_cols:
        raise ValueError(
            f"No config columns (prefix '{CELL_CONFIG_PREFIX}') found in '{wide_scaled_path}'."
        )
    logger.info(f"Found {len(config_cols)} config columns: {config_cols}")

    # KPI columns are everything that's neither metadata nor a config column.
    config_set = set(config_cols)
    feat_cols = sorted(c for c in schema.names() if c not in meta_cols and c not in config_set)
    if not feat_cols:
        raise ValueError(f"No KPI columns found in '{wide_scaled_path}'.")
    logger.info(f"Found {len(feat_cols)} KPI columns")

    scaled_params_df = None
    if scaled_params_path is not None:
        logger.info(f"Loading scaling params from {scaled_params_path}")
        scaled_params_df = pd.read_parquet(scaled_params_path)

    (
        X_scaled,
        cell_ids_arr,
        window_anchors,
        holiday_flags,
        configs_arr,
    ) = _stack_hourly_parquet_to_windows(
        wide_scaled_path,
        feat_cols,
        cell_id_col,
        anchor_col,
        hour_col,
        holiday_col,
        config_cols,
        SEQ_LEN,
    )

    logger.info(f"Parsed {len(X_scaled):,} valid windows — shape {X_scaled.shape}")

    if drop_constant_kpis:
        per_feat_std = X_scaled.std(axis=(0, 1))
        const_mask = per_feat_std < const_std_threshold
        if const_mask.any():
            dropped = [c for c, m in zip(feat_cols, const_mask, strict=False) if m]
            logger.warning(
                f"Dropping {const_mask.sum()} near-constant KPI column(s) "
                f"(std < {const_std_threshold}): {dropped[:8]}"
                f"{'...' if len(dropped) > 8 else ''}"
            )
            feat_cols = [c for c, m in zip(feat_cols, const_mask, strict=False) if not m]
            X_scaled = X_scaled[:, :, ~const_mask]
        else:
            logger.info("No near-constant KPI columns found")

    config_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    config_encoder.fit(configs_arr)
    config_dims = [len(c) for c in config_encoder.categories_]
    y = build_conditioning_vector(configs_arr, window_anchors, config_encoder, holiday_flags)
    y_dim = y.shape[1]

    logger.info(
        f"Data ready | windows={len(X_scaled):,}  feat_dim={len(feat_cols)}  "
        f"config_dims={config_dims}  y_dim={y_dim}"
    )

    return {
        "X_scaled": X_scaled,
        "y": y,
        "window_anchors": window_anchors,
        "cell_ids": cell_ids_arr,
        "configs": configs_arr,
        "config_cols": config_cols,
        "holiday_flags": holiday_flags,
        "params_df": scaled_params_df,
        "config_encoder": config_encoder,
        "kpi_columns": feat_cols,
        "seq_len": SEQ_LEN,
        "feat_dim": len(feat_cols),
        "y_dim": y_dim,
        "config_dims": config_dims,
        "arch_version": "v6",
    }
