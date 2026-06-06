import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.subplots import make_subplots
from pyspark.sql import DataFrame
from pyspark.sql import functions as f


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


def _pull_kpi(df: DataFrame, kpi: str) -> pd.DataFrame:
    """Filter to one KPI and aggregate mean over all bts/distname per timestamp."""
    return (
        df.filter(f.col("kpi_id") == kpi)
        .groupBy("start_time")
        .agg(
            f.mean("kpi_value").alias("mean_value"),
            f.stddev("kpi_value").alias("std_value"),
            f.count("kpi_value").alias("n"),
        )
        .orderBy("start_time")
        .toPandas()
        .assign(start_time=lambda d: pd.to_datetime(d["start_time"]))
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


def plot_kpi_timeline(
    df: DataFrame,
    kpi_id: str,
    rolling_window: int = 12,
    color: str = "#2563EB",
) -> go.Figure:
    """
    Interactive timeline + distribution for a single KPI.
    Returns Plotly figure.
    """
    WARN = "#F59E0B"
    ERR = "#EF4444"
    palette = [WARN, "#22D3EE", "#A78BFA"]

    pdf = _pull_kpi(df, kpi_id)
    if pdf.empty:
        raise ValueError(f"KPI '{kpi_id}' not found in DataFrame.")

    t = pdf["start_time"]
    v = pdf["mean_value"]
    std = pdf["std_value"].fillna(0)
    values: np.ndarray = v.dropna().to_numpy()

    cp_idx, cp_p = _pettitt_test(values)
    cp_time = pdf["start_time"].iloc[cp_idx] if cp_idx < len(pdf) else None
    roll_mean = v.rolling(rolling_window, center=True).mean()
    best_fits = _fit_distributions(values)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        subplot_titles=[f"Timeline — {kpi_id}", "Distribution fits"],
    )

    # 1. Timeline
    fig.add_trace(
        go.Scatter(
            x=pd.concat([t, t[::-1]]),
            y=pd.concat([v + std, (v - std)[::-1]]),
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±std",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=v,
            mode="lines",
            line=dict(color=color, width=1.4),
            name="mean",
            legend="legend",
            hovertemplate="%{x}<br>mean: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=roll_mean,
            mode="lines",
            line=dict(color=WARN, width=2.2, dash="dash"),
            name=f"rolling mean (w={rolling_window})",
            legend="legend",
            hovertemplate="%{x}<br>rolling: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if cp_p < 0.10 and cp_time is not None:
        fig.add_vline(  # type: ignore[call-arg]
            x=cp_time.value // 10**6,
            line_color=ERR,
            line_dash="dot",
            line_width=1.8,
            annotation_text=f"change point p≈{cp_p:.3f}",
            annotation_font_color=ERR,
            annotation_position="top right",
            row=1,
            col=1,
        )

    # 2. Histogram + PDF fits
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=60,
            histnorm="probability density",
            marker_color=color,
            opacity=0.55,
            name="empirical",
            legend="legend2",
            hovertemplate="value: %{x:.4f}<br>density: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    x_pdf = np.linspace(float(values.min()), float(values.max()), 400)
    for i, fit in enumerate(best_fits[:3]):
        dist = getattr(stats, fit["distribution"])
        y_pdf = dist.pdf(x_pdf, *fit["params"])
        fig.add_trace(
            go.Scatter(
                x=x_pdf,
                y=y_pdf,
                mode="lines",
                line=dict(color=palette[i], width=2),
                name=f"{fit['distribution']} (AIC={fit['aic']:.0f})",
                legend="legend2",
            ),
            row=2,
            col=1,
        )

    # Layout
    fig.update_layout(
        height=650,
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(30,41,59,0.7)",
            bordercolor="#334155",
            font=dict(color="#F1F5F9", size=10),
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
        ),
        legend2=dict(
            bgcolor="rgba(30,41,59,0.7)",
            bordercolor="#334155",
            font=dict(color="#F1F5F9", size=10),
            orientation="h",
            yanchor="top",
            y=0.36,
            xanchor="left",
            x=0.01,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")

    return fig


def schema(df: DataFrame) -> pd.DataFrame:
    """Return a DataFrame with each column's Spark type, nullable flag, and null percentage."""
    total = df.count()

    null_exprs = [
        f.round(f.sum(f.col(c).isNull().cast("int")) / total * 100, 2).alias(c) for c in df.columns
    ]
    nulls_row = df.select(null_exprs).toPandas()

    schema_rows = []
    for field in df.schema.fields:
        schema_rows.append(
            {
                "column": field.name,
                "type": str(field.dataType),
                "nullable?": field.nullable,
                "null_pct": float(nulls_row[field.name].iloc[0]),
            }
        )
    schema_df = pd.DataFrame(schema_rows)
    return schema_df


def basic_info(df: DataFrame) -> pd.DataFrame:
    """Return a single-row DataFrame with row count, distinct KPI/BTS/distname counts, and date range."""
    counts = df.agg(
        f.count("*").alias("rows_count"),
        f.countDistinct("kpi_id").alias("kpi_count"),
        f.countDistinct("bts_id").alias("bts_count"),
        f.countDistinct("distname").alias("distname_count"),
        f.min("start_date").alias("start_date"),
        f.max("start_date").alias("end_date"),
    ).toPandas()
    return counts


def kpi_bts_coverage(df: DataFrame):
    """
    Interactive heatmap KPI vs BTS (Plotly)

    Green = KPI present on BTS
    Red = KPI not present on BTS

    BTS sorted by KPI count
    """

    # 1. KPI presence on BTS
    presence = df.select("bts_id", "kpi_id").distinct().withColumn("value", f.lit(1))

    # 2. All combinations
    all_bts = df.select("bts_id").distinct()
    all_kpi = df.select("kpi_id").distinct()

    full = all_bts.crossJoin(all_kpi)

    # 3. Join → no presence = 0
    full_presence = full.join(presence, on=["bts_id", "kpi_id"], how="left").fillna(
        0, subset=["value"]
    )

    # 4. KPI count on BTS (for sorting)
    kpi_counts = full_presence.groupBy("bts_id").agg(f.sum("value").alias("kpi_count"))

    # 5. Sorting BTS
    full_presence = full_presence.join(kpi_counts, on="bts_id").orderBy(f.desc("kpi_count"))

    # 6. Spark pivot
    kpi_columns = [r["kpi_id"] for r in all_kpi.collect()]

    pdf = (
        full_presence.groupBy("bts_id", "kpi_count")
        .pivot("kpi_id", kpi_columns)
        .agg(f.first("value"))
        .orderBy(f.desc("kpi_count"))
        .drop("kpi_count")
        .toPandas()
        .set_index("bts_id")
    )

    # 7. Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pdf.values,
            x=pdf.columns.tolist(),
            y=pdf.index.tolist(),
            colorscale=[[0.0, "red"], [1.0, "green"]],
            showscale=False,
            hovertemplate="BTS: %{y}<br>KPI: %{x}<br>present: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title="KPI Coverage per BTS (green = present, red = missing)",
        xaxis_title="KPI ID",
        yaxis_title="BTS ID",
        height=800,
        template="plotly_dark",
        paper_bgcolor="#0F172A",
    )

    return fig.to_json()


def kpi_catalog(df: DataFrame) -> pd.DataFrame:
    """Return per-KPI statistics: record count, BTS/day counts, time range, value mean/std/min/max, and null %."""
    kpi_catalog = (
        df.groupBy("kpi_id")
        .agg(
            f.count("*").alias("records_count"),
            f.countDistinct("bts_id").alias("bts_count"),
            f.countDistinct("start_date").alias("day_count"),
            f.min("start_time").alias("start_time"),
            f.max("start_time").alias("end_time"),
            f.round(f.avg("kpi_value"), 4).alias("kpi_value_mean"),
            f.round(f.stddev("kpi_value"), 4).alias("kpi_value_std"),
            f.round(f.min("kpi_value"), 4).alias("kpi_value_min"),
            f.round(f.max("kpi_value"), 4).alias("kpi_value_max"),
            f.round(f.sum(f.col("kpi_value").isNull().cast("int")) / f.count("*") * 100, 2).alias(
                "null_pct"
            ),
        )
        .orderBy(f.desc("records_count"))
    ).toPandas()

    return kpi_catalog
