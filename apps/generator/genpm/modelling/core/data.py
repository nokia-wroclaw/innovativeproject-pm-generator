"""Data loading and conditioning construction, shared across all model families.

Reads the windowed hourly parquet into ``(N, 168, F)`` KPI tensors and builds the
conditioning that every generator (cVAE, WGAN-GP, diffusion) consumes:

* ``y`` — one broadcast vector per window: ``[config one-hot | holiday | seasonal(4)]``
  (:func:`build_conditioning_vector`). Conditioning is config-based, pooled across every
  cell sharing a config.
* ``calendar`` — optional per-timestep features (day-of-week, holiday, ...) that VARY
  within a window (:func:`build_calendar_features`); used by the diffusion denoiser.
* ``cell_idx`` — per-window integer index into a cell-identity vocabulary
  (``cell_encoder``, index 0 reserved for "unknown"), for the diffusion denoiser's
  optional learned per-cell embedding (core/diffusion.py). This is a separate, additive
  signal from ``y``: config is a many-to-one function of ``distname`` here, so ``y``
  captures the pooled "typical config behaviour" and ``cell_idx`` lets the embedding
  learn only the residual per-cell deviation on top of it.

The loader is model-agnostic — the chosen architecture is recorded by the training
entrypoint in ``arch_params.json``, not here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
    """Build the broadcast conditioning vector ``y`` for a batch of windows.

    Layout per window: ``[one-hot configs (C) | holiday (1) | seasonal (4)]``.

    Args:
        configs: Per-window config values, shape (N, n_config_cols), to one-hot encode.
        window_anchors: Window start timestamps (N,), source of the seasonal sin/cos.
        config_encoder: Fitted ``OneHotEncoder`` for the config columns.
        holiday_flags: Optional 0/1 holiday flag per window; zeros when None.

    Returns:
        Float32 array of shape ``(N, C + CONTEXT_DIM)`` where C is the total one-hot
        width and ``CONTEXT_DIM`` is 5 (holiday + 4 seasonal).
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


# Per-timestep calendar channels (order matters — generation must match):
#   0 day_of_week sin, 1 day_of_week cos, 2 is_weekend,
#   3 is_holiday, 4 is_holiday_eve, 5 is_long_weekend
CALENDAR_DIM = 6


def build_calendar_features(
    window_anchors: np.ndarray, seq_len: int = SEQ_LEN, country: str = "US"
) -> np.ndarray:
    """Build per-timestep calendar conditioning, shape (N, seq_len, CALENDAR_DIM).

    Unlike the broadcast ``y`` (one vector per window), these features VARY within a
    168h window — day-of-week cycles every 24h and holidays fall on specific days —
    so they are computed for each hour ``anchor + t`` and fed to the denoiser per
    timestep. Same anchor-derived source as ``y``, just evaluated hourly instead of
    collapsed to the anchor's value (which would mis-label the other 6 days).

    Args:
        window_anchors: Window start timestamps (N,).
        seq_len: Window length in hours.
        country: Country code for the ``holidays`` calendar lookup.

    Returns:
        Float32 array (N, seq_len, CALENDAR_DIM) with channels: day-of-week sin/cos,
        is_weekend, is_holiday, is_holiday_eve, is_long_weekend (a day inside a run of
        >=3 consecutive off-days = weekend|holiday).
    """
    import holidays as holidays_lib

    anchors = pd.DatetimeIndex(pd.to_datetime(window_anchors))
    # +2 years of padding so eve/long-weekend lookups never run off the holiday table.
    years = list(range(anchors.min().year, anchors.max().year + 2))
    hol = holidays_lib.country_holidays(country, years=years)
    hol_dates = set(hol.keys())

    # Per-day lookup table over the full span the windows can touch.
    span_end = anchors.max().normalize() + pd.Timedelta(days=seq_len // 24 + 3)
    days = pd.date_range(anchors.min().normalize(), span_end, freq="D")
    day_dates = np.array([d.date() for d in days])
    is_off = np.array([(d.weekday() >= 5) or (d in hol_dates) for d in day_dates])
    # long weekend = membership in a maximal run of >=3 consecutive off-days
    long_wk = np.zeros(len(days), dtype=bool)
    i = 0
    while i < len(is_off):
        if is_off[i]:
            j = i
            while j < len(is_off) and is_off[j]:
                j += 1
            if j - i >= 3:
                long_wk[i:j] = True
            i = j
        else:
            i += 1
    day_lookup = {
        d: (
            d in hol_dates,
            (pd.Timestamp(d) + pd.Timedelta(days=1)).date() in hol_dates,
            bool(lw),
        )
        for d, lw in zip(day_dates, long_wk, strict=True)
    }

    n = len(anchors)
    out = np.zeros((n, seq_len, CALENDAR_DIM), dtype=np.float32)
    hours = np.arange(seq_len)
    for w, anchor in enumerate(anchors):
        ts = anchor + pd.to_timedelta(hours, unit="h")
        dow = ts.dayofweek.to_numpy()
        out[w, :, 0] = np.sin(2 * np.pi * dow / 7)
        out[w, :, 1] = np.cos(2 * np.pi * dow / 7)
        out[w, :, 2] = (dow >= 5).astype(np.float32)
        for h, t in enumerate(ts):
            is_hol, is_eve, is_lw = day_lookup[t.date()]
            out[w, h, 3] = is_hol
            out[w, h, 4] = is_eve
            out[w, h, 5] = is_lw
    return out


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
    add_calendar: bool = False,
    calendar_country: str = "US",
) -> dict:
    """Load windowed parquet into an X tensor plus the conditioning vector ``y``.

    Expected schema (hourly format)::

        distname       string    — cell identifier (grouping key)
        window_anchor  timestamp — start of the 168-hour window
        hour_idx       integer   — 0..167 within the window
        [CELL] config* string    — per-cell categorical config (conditioning identity)
        NR_*           double    — one column per KPI

    Conditioning is built from the config columns (one-hot encoded), not the
    distname. distname is kept as cell_ids for output identity / inverse scaling.

    Args:
        wide_scaled_path: Directory of windowed (already-scaled) parquet files.
        scaled_params_path: Optional parquet of per-(distname, kpi) scaler params,
            attached as ``params_df`` for inverse scaling; None to skip.
        cell_id_col, anchor_col, hour_col: Column names for the grouping key, window
            start, and within-window hour index.
        holiday_col: Optional holiday-flag column; zeros are used when absent/missing.
        config_cols: Config columns to one-hot encode; auto-detected by the
            ``[CELL]`` prefix when None.
        meta_cols: Non-KPI, non-config columns to exclude from the KPI set; defaults
            to ``_META_COLS``.
        drop_constant_kpis: Drop KPI channels with std below ``const_std_threshold``
            (a sigmoid decoder cannot learn a constant ~0/1 channel cleanly).
        const_std_threshold: Std cutoff for the constant-KPI drop.
        add_calendar: Also build per-timestep calendar features (for diffusion).
        calendar_country: Country code for the holiday calendar when ``add_calendar``.

    Returns:
        Dict with keys: ``X_scaled``, ``y``, ``calendar`` (or None), ``cond_dim``,
        ``window_anchors``, ``cell_ids``, ``configs``, ``config_cols``,
        ``holiday_flags``, ``params_df``, ``config_encoder``, ``kpi_columns``,
        ``seq_len``, ``feat_dim``, ``y_dim``, ``config_dims``, ``cell_idx``,
        ``cell_encoder``, ``n_cells``.

    Raises:
        ValueError: If no config columns, no KPI columns, or no valid windows are found.
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

    # Cell-identity vocabulary for the optional learned distname embedding (diffusion
    # only, see core/diffusion.py). Index 0 is reserved for "unknown cell" (config-only
    # generation, or a cell_id the encoder never saw), so known cells map to 1..n_cells-1.
    cell_encoder = LabelEncoder()
    cell_encoder.fit(cell_ids_arr)
    cell_idx = (cell_encoder.transform(cell_ids_arr) + 1).astype(np.int32)
    n_cells = len(cell_encoder.classes_) + 1

    calendar = None
    cond_dim = 0
    if add_calendar:
        calendar = build_calendar_features(window_anchors, SEQ_LEN, country=calendar_country)
        cond_dim = calendar.shape[-1]
        logger.info(f"Built per-timestep calendar features — shape {calendar.shape}")

    logger.info(
        f"Data ready | windows={len(X_scaled):,}  feat_dim={len(feat_cols)}  "
        f"config_dims={config_dims}  y_dim={y_dim}  cond_dim={cond_dim}"
    )

    return {
        "X_scaled": X_scaled,
        "y": y,
        "calendar": calendar,
        "cond_dim": cond_dim,
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
        "cell_idx": cell_idx,
        "cell_encoder": cell_encoder,
        "n_cells": n_cells,
    }
