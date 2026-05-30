"""
KPI Distribution Analysis
=========================
- analyze_kpi()     : visual + statistical analysis for a single KPI
- analyze_all_kpis(): batch statistical tests across every KPI in the DataFrame
"""

from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _pull_kpi(df: DataFrame, kpi: str) -> pd.DataFrame:
    """Filter to one KPI and aggregate mean over all bts/distname per timestamp."""
    return (
        df.filter(F.col("kpi_id") == kpi)
        .groupBy("start_time")
        .agg(
            F.mean("kpi_value").alias("mean_value"),
            F.stddev("kpi_value").alias("std_value"),
            F.count("kpi_value").alias("n"),
        )
        .orderBy("start_time")
        .toPandas()
        .assign(start_time=lambda d: pd.to_datetime(d["start_time"]))
    )


def _fit_distributions(
    values: np.ndarray,
) -> list[dict]:
    """Fit a set of candidate distributions and return sorted goodness-of-fit."""
    candidates = {
        "norm": stats.norm,
        "lognorm": stats.lognorm,
        "gamma": stats.gamma,
        "expon": stats.expon,
        "weibull_min": stats.weibull_min,
        "beta": stats.beta,
        "cauchy": stats.cauchy,
        "laplace": stats.laplace,
    }
    results = []
    for name, dist in candidates.items():
        try:
            params = dist.fit(values)
            D, p = stats.kstest(values, name, args=params)
            # AIC proxy: 2k - 2*logL
            log_l = np.sum(dist.logpdf(values, *params))
            aic = 2 * len(params) - 2 * log_l
            results.append(dict(distribution=name, ks_stat=D, ks_pvalue=p, aic=aic, params=params))
        except Exception:
            pass
    return sorted(results, key=lambda r: r["aic"])


def _normality_tests(values: np.ndarray) -> dict:
    sw_stat, sw_p = stats.shapiro(values[:5000])  # Shapiro-Wilk (max 5 k)
    ag_stat, ag_p = stats.normaltest(values)  # D'Agostino-Pearson
    jb_stat, jb_p = stats.jarque_bera(values)
    ks_stat, ks_p = stats.kstest((values - values.mean()) / values.std(), "norm")
    return dict(
        shapiro_wilk=dict(stat=sw_stat, pvalue=sw_p),
        dagostino=dict(stat=ag_stat, pvalue=ag_p),
        jarque_bera=dict(stat=jb_stat, pvalue=jb_p),
        ks_normal=dict(stat=ks_stat, pvalue=ks_p),
    )


def _pettitt_test(series: np.ndarray) -> tuple[int, float]:
    """
    Non-parametric Pettitt change-point test.
    Uses the recurrence U_t = U_{t-1} + sum(sign(series[t] - series[j]), j=0..t-1)
    which avoids the shape-mismatch from naively splitting at t=0.
    Returns (change_point_index, approx_p_value).
    """
    n = len(series)
    # U[t] is built incrementally: each step adds sign(x[t] - x[j]) for all j < t
    U = np.zeros(n, dtype=float)
    for t in range(1, n):
        U[t] = U[t - 1] + np.sum(np.sign(series[t] - series[:t]))
    K = int(np.argmax(np.abs(U)))
    T = np.max(np.abs(U))
    p = 2.0 * np.exp(-6.0 * T**2 / (n**3 + n**2))
    return K, float(p)


# ──────────────────────────────────────────────────────────────────────────────
# Single-KPI visual analysis
# ──────────────────────────────────────────────────────────────────────────────


def analyze_kpi(
    df: DataFrame,
    kpi_id: str,
    rolling_window: int = 12,
    figsize: tuple[int, int] = (18, 14),
    color: str = "#2563EB",
) -> dict:
    """
    Full visual + statistical analysis for a single KPI.

    Parameters
    ----------
    df             : PySpark DataFrame with schema described above.
    kpi_id         : KPI identifier string to analyse.
    rolling_window : Rolling-mean window (number of time steps).
    figsize        : Matplotlib figure size.
    color          : Base accent colour.

    Returns
    -------
    dict with keys:
        ``timeseries``  – aggregated pandas DataFrame
        ``normality``   – dict of normality test results
        ``best_fits``   – list of distribution fits sorted by AIC
        ``change_point``– dict with index, timestamp, p_value
    """
    pdf = _pull_kpi(df, kpi_id)
    if pdf.empty:
        raise ValueError(f"KPI '{kpi_id}' not found in DataFrame.")

    values = pdf["mean_value"].dropna().values
    n = len(values)

    # ── Statistical work ────────────────────────────────────────────────────
    norm_tests = _normality_tests(values)
    best_fits = _fit_distributions(values)
    cp_idx, cp_p = _pettitt_test(values)
    cp_time = pdf["start_time"].iloc[cp_idx] if cp_idx < len(pdf) else None

    # ── Figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="#0F172A")
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    ax_ts = fig.add_subplot(gs[0, :])  # full-width timeline
    ax_hist = fig.add_subplot(gs[1, :2])  # histogram + PDF fits
    ax_qq = fig.add_subplot(gs[1, 2])  # Q-Q plot
    ax_box = fig.add_subplot(gs[2, 0])  # box-plot per period
    ax_roll = fig.add_subplot(gs[2, 1])  # rolling stats
    ax_tab = fig.add_subplot(gs[2, 2])  # test-result table

    _style_axes([ax_ts, ax_hist, ax_qq, ax_box, ax_roll, ax_tab])

    DARK = "#0F172A"
    MID = "#1E293B"
    TEXT = "#F1F5F9"
    MUTED = "#94A3B8"
    ACC = color
    WARN = "#F59E0B"
    ERR = "#EF4444"

    # ── 1. Timeline ──────────────────────────────────────────────────────────
    t = pdf["start_time"]
    v = pdf["mean_value"]
    std = pdf["std_value"].fillna(0)

    ax_ts.fill_between(t, v - std, v + std, alpha=0.18, color=ACC, linewidth=0)
    ax_ts.plot(t, v, color=ACC, linewidth=1.4, label="mean across stations")

    roll = v.rolling(rolling_window, center=True)
    ax_ts.plot(
        t,
        roll.mean(),
        color=WARN,
        linewidth=2.2,
        linestyle="--",
        label=f"rolling mean (w={rolling_window})",
    )

    # change-point marker
    if cp_p < 0.10:
        ax_ts.axvline(
            cp_time, color=ERR, linewidth=1.8, linestyle=":", label=f"change point p≈{cp_p:.3f}"
        )

    ax_ts.set_title(f"Timeline — {kpi_id}", color=TEXT, fontsize=13, pad=8)
    ax_ts.set_ylabel("mean value", color=MUTED, fontsize=9)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_ts.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right", color=MUTED, fontsize=8)
    ax_ts.legend(fontsize=8, facecolor=MID, edgecolor="none", labelcolor=TEXT)

    # ── 2. Histogram + top-2 PDF fits ────────────────────────────────────────
    ax_hist.hist(values, bins=60, color=ACC, alpha=0.55, density=True, label="empirical")
    x_pdf = np.linspace(values.min(), values.max(), 400)
    palette = [WARN, "#22D3EE", "#A78BFA"]
    for i, fit in enumerate(best_fits[:3]):
        dist = getattr(stats, fit["distribution"])
        y = dist.pdf(x_pdf, *fit["params"])
        ax_hist.plot(
            x_pdf,
            y,
            color=palette[i],
            linewidth=1.6,
            label=f"{fit['distribution']} (AIC={fit['aic']:.0f})",
        )
    ax_hist.set_title("Distribution fits", color=TEXT, fontsize=11, pad=6)
    ax_hist.set_ylabel("density", color=MUTED, fontsize=9)
    ax_hist.legend(fontsize=7.5, facecolor=MID, edgecolor="none", labelcolor=TEXT)

    # ── 3. Q-Q plot ───────────────────────────────────────────────────────────
    (osm, osr), (slope, intercept, _) = stats.probplot(values, dist="norm")
    ax_qq.scatter(osm, osr, s=8, alpha=0.5, color=ACC)
    ax_qq.plot(
        [osm.min(), osm.max()],
        [slope * osm.min() + intercept, slope * osm.max() + intercept],
        color=WARN,
        linewidth=1.8,
    )
    ax_qq.set_title("Q-Q (normal)", color=TEXT, fontsize=11, pad=6)
    ax_qq.set_xlabel("theoretical quantiles", color=MUTED, fontsize=8)
    ax_qq.set_ylabel("sample quantiles", color=MUTED, fontsize=8)

    # ── 4. Boxplot by temporal quartile ──────────────────────────────────────
    # Use the non-null subset so qcut length always matches the DataFrame length
    pdf2 = pdf.dropna(subset=["mean_value"]).copy().reset_index(drop=True)
    pdf2["period"] = pd.qcut(np.arange(len(pdf2)), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    groups = [pdf2.loc[pdf2["period"] == q, "mean_value"].values for q in ["Q1", "Q2", "Q3", "Q4"]]
    bp = ax_box.boxplot(
        groups, patch_artist=True, notch=False, medianprops=dict(color=WARN, linewidth=2)
    )
    for patch, clr in zip(bp["boxes"], [ACC, "#22D3EE", "#A78BFA", ERR], strict=False):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax_box.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], color=MUTED, fontsize=9)
    ax_box.set_title("Value dist. by time quartile", color=TEXT, fontsize=10, pad=6)

    # ── 5. Rolling mean ± std ─────────────────────────────────────────────────
    rm = v.rolling(rolling_window, center=True).mean()
    rs = v.rolling(rolling_window, center=True).std()
    ax_roll.fill_between(t, rm - rs, rm + rs, alpha=0.22, color=ACC)
    ax_roll.plot(t, rm, color=ACC, linewidth=1.4)
    ax_roll.set_title(f"Rolling μ ± σ  (w={rolling_window})", color=TEXT, fontsize=10, pad=6)
    ax_roll.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_roll.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_roll.xaxis.get_majorticklabels(), rotation=30, ha="right", color=MUTED, fontsize=7)

    # ── 6. Stats table ────────────────────────────────────────────────────────
    ax_tab.axis("off")
    rows = [
        ["Test", "Stat", "p-val", "Normal?"],
        [
            "Shapiro-Wilk",
            f"{norm_tests['shapiro_wilk']['stat']:.4f}",
            f"{norm_tests['shapiro_wilk']['pvalue']:.4f}",
            "✓" if norm_tests["shapiro_wilk"]["pvalue"] > 0.05 else "✗",
        ],
        [
            "D'Agostino",
            f"{norm_tests['dagostino']['stat']:.4f}",
            f"{norm_tests['dagostino']['pvalue']:.4f}",
            "✓" if norm_tests["dagostino"]["pvalue"] > 0.05 else "✗",
        ],
        [
            "Jarque-Bera",
            f"{norm_tests['jarque_bera']['stat']:.4f}",
            f"{norm_tests['jarque_bera']['pvalue']:.4f}",
            "✓" if norm_tests["jarque_bera"]["pvalue"] > 0.05 else "✗",
        ],
        [
            "KS (normal)",
            f"{norm_tests['ks_normal']['stat']:.4f}",
            f"{norm_tests['ks_normal']['pvalue']:.4f}",
            "✓" if norm_tests["ks_normal"]["pvalue"] > 0.05 else "✗",
        ],
        ["─────", "─────", "──────", "──────"],
        ["Best fit", best_fits[0]["distribution"], f"AIC={best_fits[0]['aic']:.0f}", ""],
        [
            "Change pt",
            str(cp_time.date()) if cp_time else "—",
            f"p≈{cp_p:.3f}",
            "✓" if cp_p < 0.05 else "✗",
        ],
    ]
    tbl = ax_tab.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(MID if r % 2 == 0 else DARK)
        cell.set_edgecolor("#334155")
        cell.set_text_props(color=TEXT)
        if r == 0:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color=WARN, weight="bold")
        if c == 3 and r > 0:
            txt = cell.get_text().get_text()
            cell.set_text_props(
                color="#22C55E" if txt == "✓" else ("#EF4444" if txt == "✗" else TEXT)
            )

    fig.suptitle(
        f"KPI Analysis  ·  {kpi_id}  ·  n={n:,} time steps",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )

    return dict(
        figure=fig,
        timeseries=pdf,
        normality=norm_tests,
        best_fits=best_fits,
        change_point=dict(index=cp_idx, timestamp=cp_time, pvalue=cp_p),
    )


def _style_axes(axes):
    # DARK = "#0F172A"
    MID = "#1E293B"
    TEXT = "#F1F5F9"
    MUTED = "#94A3B8"
    for ax in axes:
        ax.set_facecolor(MID)
        ax.spines[:].set_color("#334155")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)


# ──────────────────────────────────────────────────────────────────────────────
# Batch analysis across ALL KPIs
# ──────────────────────────────────────────────────────────────────────────────


def analyze_all_kpis(
    df: DataFrame,
    normality_alpha: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run statistical tests for every KPI in the DataFrame (no plots).

    For each KPI this computes:
      - basic descriptive stats (n, mean, std, skew, kurtosis)
      - Shapiro-Wilk, D'Agostino-Pearson, Jarque-Bera, KS-normal
      - best-fitting distribution (by AIC)
      - Pettitt change-point (index, timestamp, p-value)

    Parameters
    ----------
    df              : PySpark DataFrame.
    normality_alpha : Significance level for normality verdict.
    verbose         : Print progress.

    Returns
    -------
    pandas DataFrame with one row per KPI, all test results as columns.
    """
    kpi_list = [r["kpi_id"] for r in df.select("kpi_id").distinct().collect()]
    if verbose:
        print(f"Found {len(kpi_list)} KPIs — running tests …\n")

    records = []
    for i, kpi in enumerate(kpi_list, 1):
        if verbose:
            print(f"  [{i:>4}/{len(kpi_list)}]  {kpi}", end="\r", flush=True)
        try:
            pdf = _pull_kpi(df, kpi)
            values = pdf["mean_value"].dropna().values
            if len(values) < 8:
                continue  # too few points

            nt = _normality_tests(values)
            bf = _fit_distributions(values)
            cp_idx, cp_p = _pettitt_test(values)
            cp_time = pdf["start_time"].iloc[cp_idx] if cp_idx < len(pdf) else None

            is_normal = all(
                nt[t]["pvalue"] > normality_alpha
                for t in ["shapiro_wilk", "dagostino", "jarque_bera", "ks_normal"]
            )

            records.append(
                dict(
                    kpi_id=kpi,
                    n=len(values),
                    mean=float(np.mean(values)),
                    std=float(np.std(values)),
                    skewness=float(stats.skew(values)),
                    kurtosis=float(stats.kurtosis(values)),
                    # normality tests
                    sw_stat=nt["shapiro_wilk"]["stat"],
                    sw_pvalue=nt["shapiro_wilk"]["pvalue"],
                    dag_stat=nt["dagostino"]["stat"],
                    dag_pvalue=nt["dagostino"]["pvalue"],
                    jb_stat=nt["jarque_bera"]["stat"],
                    jb_pvalue=nt["jarque_bera"]["pvalue"],
                    ks_stat=nt["ks_normal"]["stat"],
                    ks_pvalue=nt["ks_normal"]["pvalue"],
                    is_normal=is_normal,
                    # best distribution
                    best_dist=bf[0]["distribution"],
                    best_dist_aic=bf[0]["aic"],
                    second_dist=bf[1]["distribution"] if len(bf) > 1 else None,
                    # change-point
                    cp_idx=cp_idx,
                    cp_timestamp=cp_time,
                    cp_pvalue=cp_p,
                    cp_significant=cp_p < 0.05,
                )
            )
        except Exception as exc:
            if verbose:
                print(f"\n  ⚠  {kpi} failed: {exc}")

    if verbose:
        print(f"\nDone. {len(records)} KPIs analysed.")

    result = pd.DataFrame(records).sort_values("kpi_id").reset_index(drop=True)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Quick summary helper
# ──────────────────────────────────────────────────────────────────────────────


def summarize_batch(results: pd.DataFrame) -> None:
    """Print a human-readable summary of analyze_all_kpis() output."""
    n = len(results)
    normal = results["is_normal"].sum()
    cp_sig = results["cp_significant"].sum()
    top_dists = results["best_dist"].value_counts().head(5)

    print("=" * 56)
    print(f"  KPIs analysed       : {n}")
    print(f"  Normal (all 4 tests): {normal} ({100*normal/n:.1f}%)")
    print(f"  Change-point (p<.05): {cp_sig} ({100*cp_sig/n:.1f}%)")
    print()
    print("  Most common best-fitting distributions:")
    for dist, cnt in top_dists.items():
        print(f"    {dist:<14}  {cnt:>4}  ({100*cnt/n:.1f}%)")
    print("=" * 56)
