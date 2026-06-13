"""Data loading and conditioning vector construction for the cVAE-LSTM pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder

from genpm.utils.logger import get_logger

logger = get_logger()

SEQ_LEN = 168
Y_DIM = 6
SYNTHETIC_ORIGIN = pd.Timestamp("1970-01-01")
CONST_KPI_STD_THRESHOLD = 0.05

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
        [cell_idx[:, None], y_holiday, y_seasonal],
        axis=1,
    ).astype(np.float32)


def build_conditioning_vector_onehot(
    cell_ids: np.ndarray,
    window_anchors: np.ndarray,
    holiday_flags: np.ndarray,
    cell_encoder: LabelEncoder,
) -> np.ndarray:
    """
    Extended conditioning vector with one-hot cell encoding (legacy architectures).

    Layout: [one-hot cell (n_cells) | holiday (1) | seasonal (4)]
    Returns shape (N, n_cells + 5).
    """
    n = len(cell_ids)
    n_cells = len(cell_encoder.classes_)

    y_onehot = np.zeros((n, n_cells), dtype=np.float32)
    y_onehot[np.arange(n), cell_encoder.transform(cell_ids)] = 1.0

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

    return np.concatenate([y_onehot, y_holiday, y_seasonal], axis=1)


def _stack_hourly_parquet_to_windows(
    path: str | Path,
    feat_cols: list[str],
    cell_id_col: str,
    anchor_col: str,
    hour_col: str,
    holiday_col: str | None,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    Read long-format hourly parquet and stack into (N, seq_len, n_kpis) windows.

    Expected schema:
        distname       string    — cell identifier
        window_anchor  timestamp — start of the 168-hour window
        hour_idx       integer   — 0..seq_len-1 within the window
        NR_*           double    — KPI values

    Groups by (cell_id_col, anchor_col), sorts by hour_col. Windows with wrong
    row count or any NaN are skipped.
    """
    path = Path(path)
    parquet_glob = str(path / "**" / "*.parquet")
    schema_names = pl.read_parquet_schema(parquet_glob).names()
    use_holiday = holiday_col is not None and holiday_col in schema_names

    needed_cols = [cell_id_col, anchor_col, hour_col] + feat_cols
    if use_holiday:
        needed_cols.append(holiday_col)

    agg_exprs = [pl.col(c).sort_by(hour_col) for c in feat_cols]
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

    if use_holiday:
        holiday_flags = valid_df[holiday_col].to_numpy().astype(np.int32)
    else:
        holiday_flags = np.zeros(len(X_valid), dtype=np.int32)

    logger.info(
        f"Stacked {len(X_valid):,} windows — shape ({len(X_valid)}, {seq_len}, {len(feat_cols)})"
    )
    return X_valid, cell_ids, window_anchors, holiday_flags


def load_training_windows(
    wide_scaled_path: str | Path,
    scaled_params_path: str | Path | None = None,
    cell_id_col: str = "distname",
    anchor_col: str = "window_anchor",
    hour_col: str = "hour_idx",
    holiday_col: str | None = None,
    meta_cols: set[str] | None = None,
    drop_constant_kpis: bool = True,
    const_std_threshold: float = CONST_KPI_STD_THRESHOLD,
) -> dict:
    """
    Load windowed parquet and build X tensor + conditioning vector Y.

    Expected schema (hourly format):
        distname       string    — cell identifier
        window_anchor  timestamp — start of the 168-hour window
        hour_idx       integer   — 0..167 within the window
        NR_*           double    — one column per KPI

    Returns dict with keys:
        X_scaled, y, window_anchors, cell_ids, holiday_flags,
        params_df, cell_encoder, kpi_columns, seq_len, feat_dim, n_classes, output_dim
    """
    if meta_cols is None:
        meta_cols = _META_COLS

    logger.info(f"Loading windowed parquet from {wide_scaled_path}")
    schema = pl.read_parquet_schema(str(Path(wide_scaled_path) / "**" / "*.parquet"))
    feat_cols = sorted([c for c in schema.names() if c not in meta_cols])
    if not feat_cols:
        raise ValueError(f"No KPI columns found in '{wide_scaled_path}'.")
    logger.info(f"Found {len(feat_cols)} KPI columns")

    scaled_params_df = None
    if scaled_params_path is not None:
        logger.info(f"Loading scaling params from {scaled_params_path}")
        scaled_params_df = pd.read_parquet(scaled_params_path)

    X_scaled, cell_ids_arr, window_anchors, holiday_flags = _stack_hourly_parquet_to_windows(
        wide_scaled_path,
        feat_cols,
        cell_id_col,
        anchor_col,
        hour_col,
        holiday_col,
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

    cell_encoder = LabelEncoder()
    cell_encoder.fit(cell_ids_arr)
    y = build_conditioning_vector(cell_ids_arr, window_anchors, cell_encoder, holiday_flags)
    n_classes = len(cell_encoder.classes_)

    logger.info(
        f"Data ready | windows={len(X_scaled):,}  feat_dim={len(feat_cols)}  "
        f"cells={n_classes}  y_dim={y.shape[1]}"
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
