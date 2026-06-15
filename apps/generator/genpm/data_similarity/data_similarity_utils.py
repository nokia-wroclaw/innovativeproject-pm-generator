"""
Synthetic time series validation demo.

Assumptions:
- real data can cover a longer period, for example 3 months,
- synthetic data can cover a shorter period, for example 2 weeks,
- missing values are intentional and should be represented as Spark NULL,
- long missing periods are not imputed,
- metrics work on observed points or use pairwise-complete logic.

"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.timeseries import LombScargle
from plotly.subplots import make_subplots
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from scipy.spatial.distance import jensenshannon, pdist
from scipy.stats import gaussian_kde, wasserstein_distance

from scripts.fake_timeseries import (
    make_multi,
    make_series,
    multi_schema,
    pdf_to_spark_with_schema,
    single_schema,
    to_numeric_array_with_nan,
)

rng = np.random.default_rng(42)
REAL_COLOR = "#1f77b4"
SYNTH_COLOR = "#ff7f0e"

# =============================================================================
# Test data generation
# =============================================================================


def create_demo_dataframes(spark: Any) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Creates demo real and synthetic Spark DataFrames for single-KPI and multi-KPI validation.
    """
    real_pdf = make_series(
        n_days=90,
        period_hours=24,
        noise_std=0.3,
        gap_prob=0.05,
        seed=1,
    )

    synth_pdf = make_series(
        n_days=14,
        period_hours=24,
        noise_std=1.2,
        gap_prob=0.08,
        seed=2,
    )

    real_multi_pdf = make_multi(
        start_date="2025-01-01",
        n_days=90,
        noise_std=0.3,
        seed=10,
    )

    synth_multi_pdf = make_multi(
        start_date="2025-04-01",
        n_days=14,
        noise_std=1.0,
        seed=20,
    )

    real_sdf = pdf_to_spark_with_schema(real_pdf, single_schema)
    synth_sdf = pdf_to_spark_with_schema(synth_pdf, single_schema)

    real_multi_sdf = pdf_to_spark_with_schema(real_multi_pdf, multi_schema)
    synth_multi_sdf = pdf_to_spark_with_schema(synth_multi_pdf, multi_schema)

    return real_sdf, synth_sdf, real_multi_sdf, synth_multi_sdf


# =============================================================================
# Data collection and missing-value helpers
# =============================================================================


def _not_missing(col_name: str):
    """
    Returns a Spark expression that filters out both NULL and NaN values.
    """
    c = F.col(col_name)
    return c.isNotNull() & ~F.isnan(c)


def collect_clean(
    sdf: DataFrame,
    value_col: str,
    ts_col: str = "ts",
    max_n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collects observed time-value pairs from Spark.

    Missing values are not imputed.
    The returned time axis is relative time in hours from the start of each series.
    """
    df = sdf.select(ts_col, value_col).filter(_not_missing(value_col))

    if max_n is not None:
        total = df.count()
        if total > max_n:
            df = df.sample(fraction=max_n / total, seed=42)

    pdf = df.toPandas().sort_values(ts_col)

    if pdf.empty:
        return np.array([]), np.array([])

    ts = pd.to_datetime(pdf[ts_col])
    ts_hours = (ts - ts.min()).dt.total_seconds().to_numpy() / 3600.0

    values = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy(dtype=float)

    return ts_hours, values


def collect_full_pdf(
    sdf: DataFrame,
    value_col: str,
    ts_col: str = "ts",
) -> pd.DataFrame:
    """
    Collects the full time series into pandas.

    Spark NULL values will usually appear as NaN in pandas numeric columns.
    """
    return sdf.select(ts_col, value_col).toPandas().sort_values(ts_col)


def collect_clean_multi(
    sdf: DataFrame,
    value_cols: list[str],
    ts_col: str = "ts",
    max_n: int | None = None,
) -> np.ndarray:
    """
    Collects complete-case multi-KPI observations from Spark.

    A row is kept only if all selected KPI values are present.
    """
    df = sdf.select(ts_col, *value_cols)

    for c in value_cols:
        df = df.filter(_not_missing(c))

    if max_n is not None:
        total = df.count()
        if total > max_n:
            df = df.sample(fraction=max_n / total, seed=42)

    pdf = df.toPandas().sort_values(ts_col)

    if pdf.empty:
        return np.empty((0, len(value_cols)))

    return pdf[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def missing_gap_lengths(values: pd.Series | np.ndarray | list) -> list[int]:
    """
    Returns lengths of consecutive missing-value blocks.
    """
    arr = pd.Series(values)
    missing_mask = arr.isna().to_numpy()

    if not missing_mask.any():
        return []

    diff = np.diff(missing_mask.astype(int))

    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if missing_mask[0]:
        starts = np.insert(starts, 0, 0)

    if missing_mask[-1]:
        ends = np.append(ends, len(missing_mask))

    return [int(e - s) for s, e in zip(starts, ends, strict=False)]


def missing_summary(pdf: pd.DataFrame, value_col: str) -> dict[str, float]:
    """
    Summarizes missing-value structure without imputation.
    """
    values = pdf[value_col]
    gaps = missing_gap_lengths(values)

    missing_rate = float(values.isna().mean())
    n_gaps = int(len(gaps))
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    max_gap = float(np.max(gaps)) if gaps else 0.0

    return {
        "missing_rate": missing_rate,
        "n_gaps": n_gaps,
        "mean_gap_h": mean_gap,
        "max_gap_h": max_gap,
    }


def hourly_profile(
    pdf: pd.DataFrame,
    value_col: str,
    ts_col: str = "ts",
) -> pd.Series:
    """
    Computes the average daily profile by hour of day.
    """
    tmp = pdf.copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col])
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp["hour"] = tmp[ts_col].dt.hour

    return tmp.groupby("hour")[value_col].mean().reindex(range(24))


# =============================================================================
# Single-KPI metrics
# =============================================================================


def metric_wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes 1D Wasserstein distance between value distributions.
    """
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    return float(wasserstein_distance(x, y))


def _median_heuristic_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    max_sample: int = 1000,
) -> float:
    """
    Computes RBF kernel bandwidth using the median heuristic.
    """
    pool = np.concatenate([x, y])

    if len(pool) > max_sample:
        pool = rng.choice(pool, size=max_sample, replace=False)

    d = pdist(pool.reshape(-1, 1))
    d = d[d > 0]

    if len(d) == 0:
        return 1.0

    med = float(np.median(d))

    return med if med > 0 else 1.0


def metric_mmd(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float | None = None,
    max_sample: int = 2000,
) -> float:
    """
    Computes unbiased MMD squared with an RBF kernel.
    """
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    if len(x) > max_sample:
        x = rng.choice(x, size=max_sample, replace=False)

    if len(y) > max_sample:
        y = rng.choice(y, size=max_sample, replace=False)

    bw = bandwidth or _median_heuristic_bandwidth(x, y)
    gamma = 1.0 / (2 * bw**2)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = (a[:, None] - b[None, :]) ** 2
        return np.exp(-gamma * d2)

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)

    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)

    nx = len(x)
    ny = len(y)

    mmd2 = kxx.sum() / (nx * (nx - 1)) + kyy.sum() / (ny * (ny - 1)) - 2 * kxy.mean()

    return float(max(mmd2, 0.0))


def lomb_scargle_spectrum(
    ts_hours: np.ndarray,
    values: np.ndarray,
    min_period_h: float = 2.0,
    max_period_h: float = 24 * 14,
    n_freq: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Lomb-Scargle periodogram for observed points.
    """
    if len(values) < 5:
        return np.array([]), np.array([])

    freq_min = 1.0 / max_period_h
    freq_max = 1.0 / min_period_h

    frequency = np.linspace(freq_min, freq_max, n_freq)
    power = LombScargle(ts_hours, values).power(frequency)
    periods = 1.0 / frequency

    return periods, power


def spectrum_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Computes L2 distance between normalized spectra.
    """
    if len(p1) == 0 or len(p2) == 0:
        return float("nan")

    p1n = p1 / (p1.sum() + 1e-12)
    p2n = p2 / (p2.sum() + 1e-12)

    return float(np.linalg.norm(p1n - p2n))


def acf_nan_aware(values: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Computes pairwise-complete autocorrelation function.
    """
    x = values.astype(float)

    if np.all(np.isnan(x)):
        return np.full(max_lag + 1, np.nan)

    mu = np.nanmean(x)

    out = np.full(max_lag + 1, np.nan)
    out[0] = 1.0

    for k in range(1, max_lag + 1):
        a = x[:-k]
        b = x[k:]

        mask = ~(np.isnan(a) | np.isnan(b))

        if mask.sum() < 10:
            continue

        a_ = a[mask] - mu
        b_ = b[mask] - mu

        num = np.sum(a_ * b_)
        den = np.sqrt(np.sum(a_**2) * np.sum(b_**2))

        out[k] = num / den if den > 0 else np.nan

    return out


def acf_distance(acf_1: np.ndarray, acf_2: np.ndarray) -> float:
    """
    Computes normalized L2 distance between two ACF curves.
    """
    common_mask = ~(np.isnan(acf_1) | np.isnan(acf_2))

    if common_mask.sum() == 0:
        return float("nan")

    return float(
        np.linalg.norm(acf_1[common_mask] - acf_2[common_mask]) / np.sqrt(common_mask.sum())
    )


def kde_curves(
    real_values: np.ndarray,
    synth_values: np.ndarray,
    n_grid: int = 300,
    pad_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates gaussian KDE for both series on a shared x grid.

    Returns (x_grid, density_real, density_synth).
    The grid spans the joint range of both series with a small padding.
    """
    if len(real_values) < 2 or len(synth_values) < 2:
        return np.array([]), np.array([]), np.array([])

    lo = float(min(real_values.min(), synth_values.min()))
    hi = float(max(real_values.max(), synth_values.max()))
    pad = (hi - lo) * pad_frac

    x_grid = np.linspace(lo - pad, hi + pad, n_grid)

    def _sample_for_kde(arr: np.ndarray, max_n: int = 20000) -> np.ndarray:
        if len(arr) > max_n:
            return arr[rng.choice(len(arr), size=max_n, replace=False)]
        return arr

    density_real = gaussian_kde(_sample_for_kde(real_values))(x_grid)
    density_synth = gaussian_kde(_sample_for_kde(synth_values))(x_grid)

    return x_grid, density_real, density_synth


def metric_jensen_shannon(
    density_real: np.ndarray,
    density_synth: np.ndarray,
    base: float = 2.0,
) -> float:
    """
    Computes Jensen-Shannon divergence between two density curves.

    The divergence is the squared Jensen-Shannon distance.
    With base=2 the result is bounded in [0, 1].
    With base=e (natural log) the result is bounded in [0, ln(2)].

    Both densities are expected to be evaluated on the same x grid
    (scipy internally normalizes them to sum to 1).
    """
    if len(density_real) == 0 or len(density_synth) == 0:
        return float("nan")

    distance = jensenshannon(density_real, density_synth, base=base)

    if np.isnan(distance):
        return float("nan")

    return float(distance**2)


def hourly_profile_rmse(
    profile_1: pd.Series,
    profile_2: pd.Series,
) -> float:
    """
    Computes RMSE between two average hourly profiles.
    """
    a = profile_1.to_numpy(dtype=float)
    b = profile_2.to_numpy(dtype=float)

    mask = ~(np.isnan(a) | np.isnan(b))

    if mask.sum() == 0:
        return float("nan")

    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def compute_single_metrics(
    real_sdf: DataFrame,
    synth_sdf: DataFrame,
    value_col: str = "value",
    ts_col: str = "ts",
    acf_max_lag: int = 24 * 8,
    ls_min_period_h: float = 2.0,
    ls_max_period_h: float = 24 * 14,
    ls_n_freq: int = 2000,
    kde_n_grid: int = 300,
) -> dict[str, Any]:
    """
    Computes all single-KPI metrics and returns a dictionary with values and plot data.
    """
    ts_r, val_r = collect_clean(real_sdf, value_col=value_col, ts_col=ts_col)
    ts_s, val_s = collect_clean(synth_sdf, value_col=value_col, ts_col=ts_col)

    real_full_pdf = collect_full_pdf(real_sdf, value_col=value_col, ts_col=ts_col)
    synth_full_pdf = collect_full_pdf(synth_sdf, value_col=value_col, ts_col=ts_col)

    real_values_full = to_numeric_array_with_nan(real_full_pdf[value_col])
    synth_values_full = to_numeric_array_with_nan(synth_full_pdf[value_col])

    w1 = metric_wasserstein_1d(val_r, val_s)
    mmd1 = metric_mmd(val_r, val_s)

    per_r, pow_r = lomb_scargle_spectrum(
        ts_hours=ts_r,
        values=val_r,
        min_period_h=ls_min_period_h,
        max_period_h=ls_max_period_h,
        n_freq=ls_n_freq,
    )

    per_s, pow_s = lomb_scargle_spectrum(
        ts_hours=ts_s,
        values=val_s,
        min_period_h=ls_min_period_h,
        max_period_h=ls_max_period_h,
        n_freq=ls_n_freq,
    )

    spec_d = spectrum_distance(pow_r, pow_s)

    acf_r = acf_nan_aware(real_values_full, max_lag=acf_max_lag)
    acf_s = acf_nan_aware(synth_values_full, max_lag=acf_max_lag)
    acf_d = acf_distance(acf_r, acf_s)

    real_missing = missing_summary(real_full_pdf, value_col=value_col)
    synth_missing = missing_summary(synth_full_pdf, value_col=value_col)

    real_hourly = hourly_profile(real_full_pdf, value_col=value_col, ts_col=ts_col)
    synth_hourly = hourly_profile(synth_full_pdf, value_col=value_col, ts_col=ts_col)
    hourly_rmse = hourly_profile_rmse(real_hourly, synth_hourly)

    kde_grid, kde_real, kde_synth = kde_curves(
        real_values=val_r,
        synth_values=val_s,
        n_grid=kde_n_grid,
    )

    js_div = metric_jensen_shannon(kde_real, kde_synth, base=2.0)

    return {
        "value_col": value_col,
        "ts_col": ts_col,
        "real_full_pdf": real_full_pdf,
        "synth_full_pdf": synth_full_pdf,
        "ts_real_hours": ts_r,
        "ts_synth_hours": ts_s,
        "real_values_observed": val_r,
        "synth_values_observed": val_s,
        "real_values_full": real_values_full,
        "synth_values_full": synth_values_full,
        "wasserstein_1d": w1,
        "mmd_rbf": mmd1,
        "ls_periods_real": per_r,
        "ls_power_real": pow_r,
        "ls_periods_synth": per_s,
        "ls_power_synth": pow_s,
        "ls_spectrum_distance": spec_d,
        "acf_real": acf_r,
        "acf_synth": acf_s,
        "acf_distance": acf_d,
        "real_missing": real_missing,
        "synth_missing": synth_missing,
        "real_hourly_profile": real_hourly,
        "synth_hourly_profile": synth_hourly,
        "hourly_profile_rmse": hourly_rmse,
        "kde_grid": kde_grid,
        "kde_real": kde_real,
        "kde_synth": kde_synth,
        "jensen_shannon": js_div,
        "acf_max_lag": acf_max_lag,
        "ls_min_period_h": ls_min_period_h,
        "ls_max_period_h": ls_max_period_h,
    }


# =============================================================================
# Multi-KPI metrics
# =============================================================================


def metric_sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 200,
    max_sample: int = 5000,
) -> float:
    """
    Computes sliced Wasserstein distance for multivariate data.
    """
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    if len(x) > max_sample:
        x = x[rng.choice(len(x), size=max_sample, replace=False)]

    if len(y) > max_sample:
        y = y[rng.choice(len(y), size=max_sample, replace=False)]

    d = x.shape[1]

    dirs = rng.normal(size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    dists = []

    for v in dirs:
        dists.append(wasserstein_distance(x @ v, y @ v))

    return float(np.mean(dists))


def metric_mmd_multivariate(
    x: np.ndarray,
    y: np.ndarray,
    max_sample: int = 1500,
) -> float:
    """
    Computes multivariate MMD squared with an RBF kernel.
    """
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    if len(x) > max_sample:
        x = x[rng.choice(len(x), size=max_sample, replace=False)]

    if len(y) > max_sample:
        y = y[rng.choice(len(y), size=max_sample, replace=False)]

    pool = np.vstack([x, y])
    n_pool = min(1000, len(pool))
    sample = pool[rng.choice(len(pool), size=n_pool, replace=False)]

    d = pdist(sample)
    d = d[d > 0]

    bw = float(np.median(d)) if len(d) else 1.0
    gamma = 1.0 / (2 * bw**2)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq = np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2 * a @ b.T
        return np.exp(-gamma * sq)

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)

    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)

    nx = len(x)
    ny = len(y)

    mmd2 = kxx.sum() / (nx * (nx - 1)) + kyy.sum() / (ny * (ny - 1)) - 2 * kxy.mean()

    return float(max(mmd2, 0.0))


def partial_correlations(x: np.ndarray) -> np.ndarray:
    """
    Computes a partial correlation matrix from complete-case observations.
    """
    if x.shape[0] < 3:
        return np.full((x.shape[1], x.shape[1]), np.nan)

    r = np.corrcoef(x, rowvar=False)

    if np.isnan(r).any():
        return np.full_like(r, np.nan)

    r_reg = r + 1e-8 * np.eye(r.shape[0])
    r_inv = np.linalg.inv(r_reg)

    d = np.sqrt(np.diag(r_inv))
    p = -r_inv / np.outer(d, d)

    np.fill_diagonal(p, 1.0)

    return p


def partial_corr_distance(
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """
    Computes Frobenius distance between two partial correlation matrices.
    """
    if np.isnan(p1).any() or np.isnan(p2).any():
        return float("nan")

    return float(np.linalg.norm(p1 - p2, ord="fro"))


def pairwise_corr_matrix_from_pdf(
    pdf: pd.DataFrame,
    value_cols: list[str],
) -> np.ndarray:
    """
    Computes a pairwise-complete Pearson correlation matrix.
    """
    d = len(value_cols)
    corr = np.eye(d)

    for i, c1 in enumerate(value_cols):
        for j, c2 in enumerate(value_cols):
            if i >= j:
                continue

            x = pd.to_numeric(pdf[c1], errors="coerce")
            y = pd.to_numeric(pdf[c2], errors="coerce")

            mask = ~(x.isna() | y.isna())

            if mask.sum() < 3:
                val = np.nan
            else:
                val = float(np.corrcoef(x[mask], y[mask])[0, 1])

            corr[i, j] = val
            corr[j, i] = val

    return corr


def corr_matrix_distance(
    c1: np.ndarray,
    c2: np.ndarray,
) -> float:
    """
    Computes Frobenius distance between two pairwise correlation matrices.
    """
    mask = ~(np.isnan(c1) | np.isnan(c2))

    if mask.sum() == 0:
        return float("nan")

    return float(np.linalg.norm((c1 - c2)[mask]))


def compute_multi_metrics(
    real_sdf: DataFrame,
    synth_sdf: DataFrame,
    value_cols: list[str],
    ts_col: str = "ts",
    n_projections: int = 200,
) -> dict[str, Any]:
    """
    Computes all multi-KPI metrics and returns a dictionary with values and plot data.
    """
    x_r = collect_clean_multi(real_sdf, value_cols=value_cols, ts_col=ts_col)
    x_s = collect_clean_multi(synth_sdf, value_cols=value_cols, ts_col=ts_col)

    real_full_pdf = real_sdf.select(ts_col, *value_cols).toPandas().sort_values(ts_col)
    synth_full_pdf = synth_sdf.select(ts_col, *value_cols).toPandas().sort_values(ts_col)

    sw = metric_sliced_wasserstein(
        x=x_r,
        y=x_s,
        n_projections=n_projections,
    )

    mmd_mv = metric_mmd_multivariate(x_r, x_s)

    p_r = partial_correlations(x_r)
    p_s = partial_correlations(x_s)
    pcorr_diff = p_r - p_s
    pcorr_d = partial_corr_distance(p_r, p_s)

    corr_r = pairwise_corr_matrix_from_pdf(real_full_pdf, value_cols)
    corr_s = pairwise_corr_matrix_from_pdf(synth_full_pdf, value_cols)
    corr_diff = corr_r - corr_s
    corr_d = corr_matrix_distance(corr_r, corr_s)

    return {
        "value_cols": value_cols,
        "ts_col": ts_col,
        "real_full_pdf": real_full_pdf,
        "synth_full_pdf": synth_full_pdf,
        "real_complete_values": x_r,
        "synth_complete_values": x_s,
        "sliced_wasserstein": sw,
        "mmd_multivariate": mmd_mv,
        "partial_corr_real": p_r,
        "partial_corr_synth": p_s,
        "partial_corr_diff": pcorr_diff,
        "partial_corr_distance": pcorr_d,
        "pairwise_corr_real": corr_r,
        "pairwise_corr_synth": corr_s,
        "pairwise_corr_diff": corr_diff,
        "pairwise_corr_distance": corr_d,
        "n_projections": n_projections,
    }


# =============================================================================
# Plotly helpers
# =============================================================================


def fmt_float(value: float, digits: int = 4) -> str:
    """
    Formats a float for display in tables and titles.
    """
    if value is None:
        return "NaN"

    try:
        if np.isnan(value):
            return "NaN"
    except TypeError:
        pass

    return f"{float(value):.{digits}f}"


def fmt_percent(value: float) -> str:
    """
    Formats a fraction as percentage.
    """
    if value is None:
        return "NaN"

    try:
        if np.isnan(value):
            return "NaN"
    except TypeError:
        pass

    return f"{100 * float(value):.2f}%"


def heatmap_text(matrix: np.ndarray, digits: int = 2) -> list[list[str]]:
    """
    Creates text annotations for heatmaps.
    """
    out = []

    for row in matrix:
        out_row = []

        for value in row:
            if np.isnan(value):
                out_row.append("NaN")
            else:
                out_row.append(f"{value:.{digits}f}")

        out.append(out_row)

    return out


# =============================================================================
# Figure creation
# =============================================================================


def create_single_kpi_figure(metrics: dict[str, Any]) -> go.Figure:
    """
    Creates a Plotly dashboard for single-KPI validation.

    Layout: 2 rows x 3 cols, with the summary table spanning both rows
    on the right.
    """
    val_r = metrics["real_values_observed"]
    val_s = metrics["synth_values_observed"]
    kde_grid = metrics["kde_grid"]
    kde_real = metrics["kde_real"]
    kde_synth = metrics["kde_synth"]
    js_div = metrics["jensen_shannon"]
    real_hourly = metrics["real_hourly_profile"]
    synth_hourly = metrics["synth_hourly_profile"]
    per_r = metrics["ls_periods_real"]
    pow_r = metrics["ls_power_real"]
    per_s = metrics["ls_periods_synth"]
    pow_s = metrics["ls_power_synth"]
    acf_r = metrics["acf_real"]
    acf_s = metrics["acf_synth"]
    lags = np.arange(len(acf_r))
    real_full_pdf = metrics["real_full_pdf"]
    synth_full_pdf = metrics["synth_full_pdf"]
    real_missing = metrics["real_missing"]
    synth_missing = metrics["synth_missing"]
    w1 = metrics["wasserstein_1d"]
    mmd1 = metrics["mmd_rbf"]
    spec_d = metrics["ls_spectrum_distance"]
    acf_d = metrics["acf_distance"]
    hourly_rmse = metrics["hourly_profile_rmse"]
    ls_max_period_h = metrics["ls_max_period_h"]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "KDE — value distribution",
            "Hourly profile",
            "Summary table",
            "Lomb-Scargle",
            "Pairwise-complete ACF",
            "",
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "table", "rowspan": 2}],
            [{"type": "xy"}, {"type": "xy"}, None],
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
        column_widths=[0.30, 0.30, 0.40],
    )

    # KDE
    if len(kde_grid) > 0:
        fig.add_trace(
            go.Scatter(
                x=kde_grid,
                y=kde_real,
                mode="lines",
                name="real",
                legendgroup="real",
                line=dict(color=REAL_COLOR, width=2),
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.18)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=kde_grid,
                y=kde_synth,
                mode="lines",
                name="synth",
                legendgroup="synth",
                line=dict(color=SYNTH_COLOR, width=2),
                fill="tozeroy",
                fillcolor="rgba(255,127,14,0.18)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="density", row=1, col=1)

    # Hourly profile
    fig.add_trace(
        go.Scatter(
            x=real_hourly.index,
            y=real_hourly.values,
            mode="lines+markers",
            name="real",
            legendgroup="real",
            line=dict(color=REAL_COLOR),
            marker=dict(color=REAL_COLOR),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=synth_hourly.index,
            y=synth_hourly.values,
            mode="lines+markers",
            name="synth",
            legendgroup="synth",
            line=dict(color=SYNTH_COLOR),
            marker=dict(color=SYNTH_COLOR),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text="hour of day",
        tickmode="array",
        tickvals=list(range(0, 24, 3)),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="mean value", row=1, col=2)

    # Lomb-Scargle
    if len(per_r) > 0:
        fig.add_trace(
            go.Scatter(
                x=per_r,
                y=pow_r,
                mode="lines",
                name="real",
                legendgroup="real",
                line=dict(color=REAL_COLOR),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    if len(per_s) > 0:
        fig.add_trace(
            go.Scatter(
                x=per_s,
                y=pow_s,
                mode="lines",
                name="synth",
                legendgroup="synth",
                line=dict(color=SYNTH_COLOR),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    ls_tickvals = [v for v in [24, 168] if v <= ls_max_period_h]
    ls_ticktext = [f"{v}h" for v in ls_tickvals]

    fig.add_vline(
        x=24,
        line_dash="dot",
        line_width=1,
        line_color="gray",
        annotation_text="24h",
        annotation_position="bottom right",
        annotation_font_size=10,
        row=2,
        col=1,
    )
    if 24 * 7 <= ls_max_period_h:
        fig.add_vline(
            x=24 * 7,
            line_dash="dot",
            line_width=1,
            line_color="gray",
            annotation_text="168h",
            annotation_position="bottom right",
            annotation_font_size=10,
            row=2,
            col=1,
        )

    fig.update_xaxes(
        title_text="period [h]",
        range=[0, ls_max_period_h],
        tickmode="array",
        tickvals=ls_tickvals,
        ticktext=ls_ticktext,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="power", rangemode="tozero", row=2, col=1)

    # ACF
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=acf_r,
            mode="lines",
            name="real",
            legendgroup="real",
            line=dict(color=REAL_COLOR),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=acf_s,
            mode="lines",
            name="synth",
            legendgroup="synth",
            line=dict(color=SYNTH_COLOR),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_hline(y=0, line_width=1, line_color="gray", row=2, col=2)

    for lag in [24, 48, 24 * 7]:
        if lag <= metrics["acf_max_lag"]:
            fig.add_vline(
                x=lag,
                line_dash="dot",
                line_width=1,
                line_color="gray",
                annotation_text=f"{lag}h",
                annotation_position="bottom right",
                annotation_font_size=10,
                row=2,
                col=2,
            )
    fig.update_xaxes(title_text="lag [h]", row=2, col=2)
    fig.update_yaxes(title_text="ACF", row=2, col=2)

    # Summary table
    table_rows = [
        ["Real rows", f"{len(real_full_pdf)}", "all points"],
        ["Synth rows", f"{len(synth_full_pdf)}", "all points"],
        ["Real observed", f"{len(val_r)}", "non-missing"],
        ["Synth observed", f"{len(val_s)}", "non-missing"],
        ["1D Wasserstein", fmt_float(w1), "value distribution"],
        ["MMD² RBF", fmt_float(mmd1), "tails / nonlinear"],
        ["Jensen-Shannon", fmt_float(js_div), "divergence in [0, 1]"],
        ["LS spectrum dist", fmt_float(spec_d), "periodic structure"],
        ["ACF L2 dist", fmt_float(acf_d), "temporal dependence"],
        ["Hourly RMSE", fmt_float(hourly_rmse), "daily profile"],
        ["NULL rate real", fmt_percent(real_missing["missing_rate"]), "missingness"],
        ["NULL rate synth", fmt_percent(synth_missing["missing_rate"]), "missingness"],
        [
            "Gaps real/synth",
            f"{real_missing['n_gaps']}/{synth_missing['n_gaps']}",
            "n gaps",
        ],
        [
            "Mean gap real/synth [h]",
            f"{real_missing['mean_gap_h']:.1f}/{synth_missing['mean_gap_h']:.1f}",
            "avg length",
        ],
        [
            "Max gap real/synth [h]",
            f"{real_missing['max_gap_h']:.0f}/{synth_missing['max_gap_h']:.0f}",
            "longest",
        ],
    ]

    fig.add_trace(
        go.Table(
            columnwidth=[0.34, 0.20, 0.46],
            header=dict(
                values=["Metric", "Value", "Interpretation"],
                align="left",
                font=dict(size=13),
                height=34,
                fill_color="#e8e8e8",
            ),
            cells=dict(
                values=[
                    [r[0] for r in table_rows],
                    [r[1] for r in table_rows],
                    [r[2] for r in table_rows],
                ],
                align="left",
                font=dict(size=12),
                height=38,
            ),
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Synthetic time series validation — Single KPI<br>"
                f"<sup>W1={fmt_float(w1)} · MMD²={fmt_float(mmd1)} · "
                f"LS dist={fmt_float(spec_d)} · ACF dist={fmt_float(acf_d)} · "
                f"Hourly RMSE={fmt_float(hourly_rmse)}</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        height=850,
        width=1500,
        template="plotly_white",
        margin=dict(l=60, r=30, t=120, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=12),
        ),
    )

    return fig


def create_multi_kpi_figure(
    metrics: dict[str, Any],
) -> go.Figure:
    """
    Creates a Plotly dashboard for multi-KPI validation.
    """
    labels = metrics["value_cols"]

    corr_r = metrics["pairwise_corr_real"]
    corr_s = metrics["pairwise_corr_synth"]
    corr_diff = metrics["pairwise_corr_diff"]

    sw = metrics["sliced_wasserstein"]
    mmd_mv = metrics["mmd_multivariate"]
    corr_d = metrics["pairwise_corr_distance"]
    pcorr_d = metrics["partial_corr_distance"]

    real_full_pdf = metrics["real_full_pdf"]
    synth_full_pdf = metrics["synth_full_pdf"]

    x_r = metrics["real_complete_values"]
    x_s = metrics["synth_complete_values"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Pairwise correlation — real",
            "Pairwise correlation — synth",
            "Pairwise correlation difference: real - synth",
            "Summary table",
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "table"}],
        ],
        column_widths=[0.48, 0.52],
        row_heights=[0.52, 0.48],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    fig.add_trace(
        go.Heatmap(
            z=corr_r,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=heatmap_text(corr_r),
            texttemplate="%{text}",
            hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.4f}<extra></extra>",
            colorbar=dict(
                len=0.36,
                thickness=14,
                x=0.45,
                y=0.80,
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=corr_s,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=heatmap_text(corr_s),
            texttemplate="%{text}",
            hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.4f}<extra></extra>",
            colorbar=dict(
                len=0.36,
                thickness=14,
                x=1.02,
                y=0.80,
            ),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=corr_diff,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=heatmap_text(corr_diff),
            texttemplate="%{text}",
            hovertemplate="x=%{x}<br>y=%{y}<br>diff=%{z:.4f}<extra></extra>",
            colorbar=dict(
                len=0.36,
                thickness=14,
                x=0.47,
                y=0.22,
            ),
        ),
        row=2,
        col=1,
    )

    table_rows = [
        ["Real rows", f"{len(real_full_pdf)}", "all rows"],
        ["Synth rows", f"{len(synth_full_pdf)}", "all rows"],
        ["Real complete rows", f"{x_r.shape[0]}", "rows without NULL for all selected KPI"],
        ["Synth complete rows", f"{x_s.shape[0]}", "rows without NULL for all selected KPI"],
        ["Sliced-Wasserstein", fmt_float(sw), "multivariate distribution"],
        ["MMD² multivariate", fmt_float(mmd_mv), "joint distribution"],
        ["Pairwise corr dist", fmt_float(corr_d), "pairwise correlation matrix difference"],
        ["Partial corr dist", fmt_float(pcorr_d), "partial correlation matrix difference"],
    ]

    fig.add_trace(
        go.Table(
            columnwidth=[0.35, 0.18, 0.47],
            header=dict(
                values=["Metric", "Value", "Interpretation"],
                align="left",
                font=dict(size=12),
                height=28,
            ),
            cells=dict(
                values=[
                    [row[0] for row in table_rows],
                    [row[1] for row in table_rows],
                    [row[2] for row in table_rows],
                ],
                align="left",
                font=dict(size=11),
                height=34,
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title=(
            "Synthetic time series validation — Multi-KPI<br>"
            f"<sup>Sliced-Wasserstein={fmt_float(sw)}, "
            f"MMD²={fmt_float(mmd_mv)}, "
            f"Pairwise corr dist={fmt_float(corr_d)}, "
            f"Partial corr dist={fmt_float(pcorr_d)}</sup>"
        ),
        height=750,
        width=1100,
        template="plotly_white",
        margin=dict(l=60, r=90, t=120, b=50),
    )

    return fig
