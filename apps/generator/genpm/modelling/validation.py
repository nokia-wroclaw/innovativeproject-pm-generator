"""Validation utilities: data-format helpers and comparison plots for real vs synthetic KPIs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from genpm.modelling.model_utils.data import SEQ_LEN

# =============================================================================
# Data format helpers
# =============================================================================


def wide_timespan_to_long(df: pd.DataFrame, *, keep_window_anchor: bool = True) -> pd.DataFrame:
    """Convert wide-format synthetic output (one row per hour, KPI columns) to long format."""
    out = df.copy()
    if "distname" not in out.columns and "cell_id" in out.columns:
        out["distname"] = out["cell_id"]
    id_vars = ["distname", "timestamp"]
    if keep_window_anchor and "window_anchor" in out.columns:
        id_vars.append("window_anchor")
    drop_cols = [c for c in ("cell_id", "seed") if c in out.columns]
    value_vars = [c for c in out.columns if c not in set(id_vars) | set(drop_cols)]
    return (
        out.drop(columns=drop_cols, errors="ignore")
        .melt(id_vars=id_vars, value_vars=value_vars, var_name="kpi_id", value_name="kpi_value")
        .reset_index(drop=True)
    )


def load_real_long(
    parquet_path: str | Path,
    distname: str,
    date_start: str,
    date_end: str,
    *,
    keep_window_anchor: bool = True,
) -> pd.DataFrame:
    """Load real PM data in long format filtered to a cell and date range."""
    pdf = pd.read_parquet(parquet_path)
    pdf["timestamp"] = pd.to_datetime(pdf["window_anchor"]) + pd.to_timedelta(
        pdf["hour_idx"], unit="h"
    )
    ts_start, ts_end = pd.Timestamp(date_start), pd.Timestamp(date_end)
    mask = (
        (pdf["distname"] == distname)
        & (pdf["timestamp"] >= ts_start)
        & (pdf["timestamp"] <= ts_end)
    )
    cols = ["distname", "timestamp", "kpi_id", "kpi_value"]
    if keep_window_anchor and "window_anchor" in pdf.columns:
        cols.append("window_anchor")
    return pdf.loc[mask, cols].reset_index(drop=True)


def enrich_with_window_cols(long_df: pd.DataFrame, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    """Add window_anchor and hour_in_window columns to a long-format DataFrame."""
    df = long_df.sort_values(["distname", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "window_anchor" in df.columns:
        anchor = pd.to_datetime(df["window_anchor"])
    else:
        gap_h = df.groupby("distname")["timestamp"].diff().dt.total_seconds().div(3600)
        new_window = gap_h.isna() | (gap_h > 1)
        df["_window_id"] = new_window.groupby(df["distname"]).cumsum()
        anchor = df.groupby(["distname", "_window_id"])["timestamp"].transform("min")
        df = df.drop(columns=["_window_id"])

    df["window_anchor"] = anchor
    df["hour_in_window"] = (
        (df["timestamp"] - df["window_anchor"]).dt.total_seconds() // 3600
    ).astype(int)

    bad = (df["hour_in_window"] < 0) | (df["hour_in_window"] >= seq_len)
    if bad.any():
        df = df.loc[~bad].copy()

    return df.reset_index(drop=True)


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format KPI data back to wide format (one column per KPI)."""
    return long_df.pivot_table(
        index=["distname", "timestamp"],
        columns="kpi_id",
        values="kpi_value",
        aggfunc="first",
    ).reset_index()


def long_to_windows_3d(
    long_df: pd.DataFrame, kpi_columns: list[str], seq_len: int = SEQ_LEN
) -> np.ndarray:
    """Convert long-format data to a 3-D array (n_windows, seq_len, n_kpis)."""
    enriched = enrich_with_window_cols(long_df, seq_len=seq_len)
    windows = []
    for _, g in enriched.groupby(["distname", "window_anchor"], sort=False):
        g = g.sort_values("hour_in_window")
        if len(g) != seq_len:
            continue
        wide = g.pivot_table(
            index="hour_in_window", columns="kpi_id", values="kpi_value", aggfunc="first"
        )
        if not set(kpi_columns).issubset(wide.columns):
            continue
        windows.append(wide.reindex(columns=kpi_columns).to_numpy(dtype=np.float32))
    if not windows:
        raise ValueError("No complete windows found in long data.")
    return np.stack(windows)


def kpi_series(long_df: pd.DataFrame, kpi_id: str) -> pd.Series:
    """Extract the value series for a single KPI from a long-format DataFrame."""
    return long_df.loc[long_df["kpi_id"] == kpi_id, "kpi_value"]


def autocorr_lags(series, max_lag: int = 48) -> np.ndarray:
    """Compute autocorrelation at lags 1..max_lag."""
    s = pd.Series(series)
    return np.array([s.autocorr(lag=lag) for lag in range(1, max_lag + 1)])


# =============================================================================
# Comparison metrics
# =============================================================================


def compute_kpi_stats(
    long_real: pd.DataFrame, long_syn: pd.DataFrame, kpi_list: list[str]
) -> pd.DataFrame:
    """Return a per-KPI statistics table comparing real and synthetic distributions."""
    rows = []
    for kpi in kpi_list:
        r = kpi_series(long_real, kpi)
        s = kpi_series(long_syn, kpi)
        rows.append(
            {
                "KPI": kpi,
                "real_mean": round(r.mean(), 4),
                "syn_mean": round(s.mean(), 4),
                "real_std": round(r.std(), 4),
                "syn_std": round(s.std(), 4),
                "real_min": round(r.min(), 4),
                "syn_min": round(s.min(), 4),
                "real_max": round(r.max(), 4),
                "syn_max": round(s.max(), 4),
                "mean_diff%": round(abs(r.mean() - s.mean()) / (abs(r.mean()) + 1e-9) * 100, 2),
            }
        )
    return pd.DataFrame(rows).set_index("KPI")


# =============================================================================
# Plots
# =============================================================================


def plot_timeseries_overlay(
    long_real: pd.DataFrame,
    long_syn: pd.DataFrame,
    kpi_list: list[str],
    distname: str,
    date_start: str,
    date_end: str,
    seq_len: int = SEQ_LEN,
) -> None:
    """Plot mean ± 1σ band per KPI for real vs synthetic windows."""
    real_enriched = enrich_with_window_cols(long_real, seq_len)
    syn_enriched = enrich_with_window_cols(long_syn, seq_len)

    fig, axes = plt.subplots(len(kpi_list), 1, figsize=(14, 3 * len(kpi_list)), sharex=False)
    if len(kpi_list) == 1:
        axes = [axes]

    for ax, kpi in zip(axes, kpi_list, strict=False):
        real_windows, syn_windows = [], []
        for _, g in real_enriched.loc[real_enriched["kpi_id"] == kpi].groupby(
            ["distname", "window_anchor"], sort=False
        ):
            g = g.sort_values("hour_in_window")
            if len(g) == seq_len:
                real_windows.append(g["kpi_value"].to_numpy())
        for _, g in syn_enriched.loc[syn_enriched["kpi_id"] == kpi].groupby(
            ["distname", "window_anchor"], sort=False
        ):
            g = g.sort_values("hour_in_window")
            if len(g) == seq_len:
                syn_windows.append(g["kpi_value"].to_numpy())

        if not real_windows or not syn_windows:
            ax.set_title(f"{kpi} (insufficient complete windows)", fontsize=9)
            continue

        real_arr, syn_arr = np.stack(real_windows), np.stack(syn_windows)
        r_mean, r_std = real_arr.mean(axis=0), real_arr.std(axis=0)
        s_mean, s_std = syn_arr.mean(axis=0), syn_arr.std(axis=0)
        hours = np.arange(seq_len)

        ax.fill_between(
            hours, r_mean - r_std, r_mean + r_std, alpha=0.25, color="steelblue", label="real ±1σ"
        )
        ax.plot(hours, r_mean, color="steelblue", linewidth=1.5, label="real mean")
        ax.fill_between(
            hours, s_mean - s_std, s_mean + s_std, alpha=0.25, color="tomato", label="synthetic ±1σ"
        )
        ax.plot(hours, s_mean, color="tomato", linewidth=1.5, label="synthetic mean")
        ax.set_title(kpi, fontsize=9)
        ax.set_xlabel("hour in window")
        ax.legend(fontsize=7, loc="upper right")

    plt.suptitle(f"{distname} | {date_start} → {date_end}", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_kde(long_real: pd.DataFrame, long_syn: pd.DataFrame, kpi_list: list[str]) -> None:
    """Plot KDE marginal distributions for real vs synthetic per KPI."""
    n_cols = min(3, len(kpi_list))
    n_rows = (len(kpi_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).flatten()

    for ax, kpi in zip(axes, kpi_list, strict=False):
        r_vals = kpi_series(long_real, kpi).dropna().to_numpy()
        s_vals = kpi_series(long_syn, kpi).dropna().to_numpy()
        lo = min(r_vals.min(), s_vals.min())
        hi = max(r_vals.max(), s_vals.max())
        xs = np.linspace(lo, hi, 300)
        if r_vals.std() > 0:
            ax.plot(xs, gaussian_kde(r_vals)(xs), color="steelblue", label="real")
        if s_vals.std() > 0:
            ax.plot(xs, gaussian_kde(s_vals)(xs), color="tomato", label="synthetic")
        ax.set_title(kpi, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("value")
        ax.set_ylabel("density")

    for ax in axes[len(kpi_list) :]:
        ax.set_visible(False)

    plt.suptitle("Marginal distributions — real vs synthetic", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_autocorr(
    long_real: pd.DataFrame,
    long_syn: pd.DataFrame,
    kpi_list: list[str],
    max_lag: int = 48,
) -> None:
    """Plot autocorrelation at lags 1..max_lag for real vs synthetic per KPI."""
    n_cols = min(3, len(kpi_list))
    n_rows = (len(kpi_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).flatten()

    for ax, kpi in zip(axes, kpi_list, strict=False):
        real_ac = autocorr_lags(
            long_real.loc[long_real["kpi_id"] == kpi]
            .sort_values("timestamp")["kpi_value"]
            .dropna(),
            max_lag,
        )
        syn_ac = autocorr_lags(
            long_syn.loc[long_syn["kpi_id"] == kpi].sort_values("timestamp")["kpi_value"].dropna(),
            max_lag,
        )
        lags = np.arange(1, max_lag + 1)
        ax.plot(lags, real_ac, color="steelblue", label="real")
        ax.plot(lags, syn_ac, color="tomato", label="synthetic")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(24, color="gray", linewidth=0.5, linestyle=":", label="lag=24h")
        ax.set_title(kpi, fontsize=9)
        ax.set_xlabel("lag (hours)")
        ax.set_ylabel("autocorrelation")
        ax.legend(fontsize=7)

    for ax in axes[len(kpi_list) :]:
        ax.set_visible(False)

    plt.suptitle("Autocorrelation — real vs synthetic", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_kpi_interactive(df: pd.DataFrame, kpi_id: str, distname: str) -> None:
    """Plotly line chart for a single KPI — overlays multiple distname traces."""
    import plotly.express as px

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_kpi = df[df["kpi_id"] == kpi_id].sort_values("timestamp")
    fig = px.line(
        df_kpi,
        x="timestamp",
        y="kpi_value",
        color="distname",
        title=f"KPI {kpi_id} — {distname}",
    )
    fig.update_layout(template="plotly_white")
    fig.show()
