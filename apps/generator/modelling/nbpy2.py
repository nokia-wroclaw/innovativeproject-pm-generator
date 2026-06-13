# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# import keras
import pandas as pd
from cvae_utils import (
    SEQ_LEN,
    _to_numpy,
    generate_timespan,
    load_artifacts,
    seasonal_features,
)
from IPython.display import display
from scipy.stats import gaussian_kde

# %%
USER = "user"

# %%
SHARED_DIR_PATH = Path(f"/home/{USER}/app/apps/apps/generator/data/shared_dir")

# %%
TRAINING_DATA_PATH = SHARED_DIR_PATH / "preprocessed_dataset" / "final_scaled_only_minmax"

ARTIFACTS_DIR_PATH = SHARED_DIR_PATH / "artifacts"
RUN_DIR_PATH = ARTIFACTS_DIR_PATH / "run_4"
WEIGHTS_PATH = RUN_DIR_PATH / "models_weights"

# %%
MODEL_PATH = WEIGHTS_PATH / "cvae_lstm_v5_0.weights.h5"

# %%
model, data = load_artifacts(
    out_dir=RUN_DIR_PATH,
    weights_path=MODEL_PATH,
)

# %%
# arch, model = build_model(
#     seq_len=data["seq_len"],
#     feat_dim=data["feat_dim"],
#     n_cells=data["n_classes"],
# )

# %%
# model.load_weights("cvae_lstm_v5_0.weights.h5")

# %%
# model.export(WEIGHTS_PATH / "cvae_v5_minmax")

# %%
data["params_df"]

# %%
params_df = data["params_df"]
params_df["param_a"] = params_df["mm_min"]
params_df["param_b"] = params_df["mm_max"] - params_df["mm_min"]
params_df = params_df.drop(["mm_min", "mm_max"], axis=1)
params_df["scaler"] = "minmax"

# %%
DISTNAME = "bts_24/cell_5"

# %%
cell_id = DISTNAME
anchor = pd.Timestamp("2024-01-15")
holiday = 0
cell_idx = data["cell_encoder"].transform([cell_id])[0]
seasonal = seasonal_features(anchor)
y_one = np.array([[cell_idx, holiday, *seasonal]], dtype=np.float32)  # shape (1, 6)
# Prior sampling — correct v5 path
x_scaled, _ = model.generate(y_one)
x_scaled = np.asarray(_to_numpy(x_scaled))  # (1, 168, n_kpis)
# Back to raw KPI scale
# x_raw = _inverse_transform_3d(
#     x_scaled,
#     params_df,
#     data["kpi_columns"],
#     cell_id,
# )
x_raw = x_scaled

# %%
# ── Validation config (edit to match your run) ────────────────────────────────
DATE_START = "2023-12-12"
DATE_END = "2024-03-12"
N_WEEKS = 3
SEED = 42

# Long-format real PM parquet for comparison plots
REAL_PM_PARQUET = (
    SHARED_DIR_PATH
    / "preprocessed_dataset"
    / "final_scaled_only_minmax"
    / "pm_df_long_indexed_winds"
)

# PLOT_KPIS = data["kpi_columns"][:10]
PLOT_KPIS = ["NR_5096", "NR_5244", "NR_5076", "NR_630", "NR_5193", "NR_5324"]
KPI_COLS = data["kpi_columns"]

# %%
# ── Validation helpers (long-format PM) ───────────────────────────────────────


def wide_timespan_to_long(df: pd.DataFrame, *, keep_window_anchor: bool = True) -> pd.DataFrame:
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
    df = long_df.sort_values(["distname", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "window_anchor" in df.columns:
        anchor = pd.to_datetime(df["window_anchor"])
    else:
        gap_h = df.groupby("distname")["timestamp"].diff().dt.total_seconds().div(3600)
        new_window = gap_h.isna() | (gap_h > 1)
        window_id = new_window.groupby(df["distname"]).cumsum()
        anchor = df.groupby(["distname", window_id])["timestamp"].transform("min")
    df["window_anchor"] = anchor
    df["hour_in_window"] = (
        (df["timestamp"] - df["window_anchor"]).dt.total_seconds() // 3600
    ).astype(int)
    bad = (df["hour_in_window"] < 0) | (df["hour_in_window"] >= seq_len)
    if bad.any():
        df = df.loc[~bad].copy()
    return df.reset_index(drop=True)


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.pivot_table(
        index=["distname", "timestamp"], columns="kpi_id", values="kpi_value", aggfunc="first"
    ).reset_index()


def long_to_windows_3d(
    long_df: pd.DataFrame, kpi_columns: list[str], seq_len: int = SEQ_LEN
) -> np.ndarray:
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
    return long_df.loc[long_df["kpi_id"] == kpi_id, "kpi_value"]


def autocorr_lags(series, max_lag=48):
    s = pd.Series(series)
    return np.array([s.autocorr(lag=l) for l in range(1, max_lag + 1)])  # noqa


# %%
# CELL 0 — Generate synthetic timespan → long PM format
df_timespan = generate_timespan(
    model=model,
    X_scaled=data["X_scaled"],
    y=data["y"],
    window_anchors=data["window_anchors"],
    cell_ids=data["cell_ids"],
    kpi_columns=data["kpi_columns"],
    params_df=params_df,
    cell_id=DISTNAME,
    date_start=DATE_START,
    date_end=DATE_END,
    seed=SEED,
)
df_timespan["distname"] = df_timespan["cell_id"]

long_syn = wide_timespan_to_long(df_timespan)
long_syn_user = long_syn.drop(columns=["window_anchor"], errors="ignore")

print(f"Synthetic rows (long): {len(long_syn_user):,}")
display(long_syn_user.head())

# %%
# CELL 1 — Load matching real PM rows
long_real = load_real_long(
    REAL_PM_PARQUET,
    distname=DISTNAME,
    date_start=DATE_START,
    date_end=DATE_END,
)
long_real_user = long_real.drop(columns=["window_anchor"], errors="ignore")

syn_enriched = enrich_with_window_cols(long_syn)
real_enriched = enrich_with_window_cols(long_real)

n_real_windows = real_enriched.groupby(["distname", "window_anchor"]).ngroups
n_syn_windows = syn_enriched.groupby(["distname", "window_anchor"]).ngroups

print(f"Real rows (long)       : {len(long_real_user):,}  ({n_real_windows} windows)")
print(f"Synthetic rows (long)  : {len(long_syn_user):,}  ({n_syn_windows} windows)")

# %%
# CELL 2 — Per-KPI statistics table
stats_rows = []
for kpi in PLOT_KPIS:
    r = kpi_series(long_real, kpi)
    s = kpi_series(long_syn, kpi)
    stats_rows.append(
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

df_stats = pd.DataFrame(stats_rows).set_index("KPI")
display(df_stats)

# %%
# CELL 2 — Per-KPI statistics table
stats_rows = []
for kpi in PLOT_KPIS:
    r = kpi_series(long_real, kpi)
    s = kpi_series(long_syn, kpi)
    stats_rows.append(
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

df_stats = pd.DataFrame(stats_rows).set_index("KPI")
display(df_stats)


# %%
# CELL 3 — Time-series overlay: mean ± std band per KPI
n_kpis_plot = len(PLOT_KPIS)
fig, axes = plt.subplots(n_kpis_plot, 1, figsize=(14, 3 * n_kpis_plot), sharex=False)
if n_kpis_plot == 1:
    axes = [axes]

for ax, kpi in zip(axes, PLOT_KPIS, strict=False):
    real_windows, syn_windows = [], []
    for _, g in real_enriched.loc[real_enriched["kpi_id"] == kpi].groupby(
        ["distname", "window_anchor"], sort=False
    ):
        g = g.sort_values("hour_in_window")
        if len(g) == SEQ_LEN:
            real_windows.append(g["kpi_value"].to_numpy())
    for _, g in syn_enriched.loc[syn_enriched["kpi_id"] == kpi].groupby(
        ["distname", "window_anchor"], sort=False
    ):
        g = g.sort_values("hour_in_window")
        if len(g) == SEQ_LEN:
            syn_windows.append(g["kpi_value"].to_numpy())

    if not real_windows or not syn_windows:
        ax.set_title(f"{kpi} (insufficient complete windows)", fontsize=9)
        continue

    real_arr, syn_arr = np.stack(real_windows), np.stack(syn_windows)
    r_mean, r_std = real_arr.mean(axis=0), real_arr.std(axis=0)
    s_mean, s_std = syn_arr.mean(axis=0), syn_arr.std(axis=0)
    hours = np.arange(SEQ_LEN)

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

plt.suptitle(f"{DISTNAME} | timespan {DATE_START} → {DATE_END}", fontsize=11)
plt.tight_layout()
plt.show()

# %%
# CELL 4 — Marginal distributions (KDE)
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
axes = axes.flatten()

for ax, kpi in zip(axes, PLOT_KPIS, strict=False):
    r_vals = kpi_series(long_real, kpi).dropna().to_numpy()
    s_vals = kpi_series(long_syn, kpi).dropna().to_numpy()
    lo, hi = min(r_vals.min(), s_vals.min()), max(r_vals.max(), s_vals.max())
    xs = np.linspace(lo, hi, 300)
    if r_vals.std() > 0:
        ax.plot(xs, gaussian_kde(r_vals)(xs), color="steelblue", label="real")
    if s_vals.std() > 0:
        ax.plot(xs, gaussian_kde(s_vals)(xs), color="tomato", label="synthetic")
    ax.set_title(kpi, fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlabel("value")
    ax.set_ylabel("density")

for ax in axes[len(PLOT_KPIS) :]:
    ax.set_visible(False)

plt.suptitle("Marginal distributions — real vs synthetic", fontsize=11)
plt.tight_layout()
plt.show()

# %%
# CELL 5 — Autocorrelation comparison (lag 1–48)
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
axes = axes.flatten()

for ax, kpi in zip(axes, PLOT_KPIS, strict=False):
    real_ac = autocorr_lags(
        long_real.loc[long_real["kpi_id"] == kpi].sort_values("timestamp")["kpi_value"].dropna()
    )
    syn_ac = autocorr_lags(
        long_syn.loc[long_syn["kpi_id"] == kpi].sort_values("timestamp")["kpi_value"].dropna()
    )
    lags = np.arange(1, 49)
    ax.plot(lags, real_ac, color="steelblue", label="real")
    ax.plot(lags, syn_ac, color="tomato", label="synthetic")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(24, color="gray", linewidth=0.5, linestyle=":", label="lag=24h")
    ax.set_title(kpi, fontsize=9)
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation")
    ax.legend(fontsize=7)

for ax in axes[len(PLOT_KPIS) :]:
    ax.set_visible(False)

plt.suptitle("Autocorrelation — real vs synthetic", fontsize=11)
plt.tight_layout()
plt.show()

# %%


# %%


# %%
