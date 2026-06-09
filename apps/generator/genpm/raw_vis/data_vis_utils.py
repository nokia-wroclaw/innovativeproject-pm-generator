import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.subplots import make_subplots
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from genpm.raw_vis.kpi_stats import fit_distributions, pettitt_test, pull_kpi


def plot_kpi_timeline(
    df: DataFrame,
    kpi_id: str,
    rolling_window: int = 12,
    color: str = "#2563EB",
) -> go.Figure:
    WARN = "#F59E0B"
    ERR = "#EF4444"
    palette = [WARN, "#22D3EE", "#A78BFA"]

    pdf = pull_kpi(df, kpi_id)
    if pdf.empty:
        raise ValueError(f"KPI '{kpi_id}' not found in DataFrame.")

    t = pdf["start_time"]
    v = pdf["mean_value"]
    std = pdf["std_value"].fillna(0)
    values: np.ndarray = v.dropna().to_numpy()

    cp_idx, cp_p = pettitt_test(values)
    cp_time = pdf["start_time"].iloc[cp_idx] if cp_idx < len(pdf) else None
    roll_mean = v.rolling(rolling_window, center=True).mean()
    best_fits = fit_distributions(values)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        subplot_titles=[f"Timeline — {kpi_id}", "Distribution fits"],
    )

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


def schema(df: DataFrame):
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
    return pd.DataFrame(schema_rows)


def basic_info(df: DataFrame) -> pd.DataFrame:
    """Return a single-row DataFrame with row count, distinct KPI/BTS/distname counts, and date range."""
    return df.agg(
        f.count("*").alias("rows_count"),
        f.countDistinct("kpi_id").alias("kpi_count"),
        f.countDistinct("bts_id").alias("bts_count"),
        f.countDistinct("distname").alias("distname_count"),
        f.min("start_date").alias("start_date"),
        f.max("start_date").alias("end_date"),
    ).toPandas()


def _matrix_to_json_lists(matrix) -> list[list[float]]:
    return [[float(v) for v in row] for row in matrix.tolist()]


def kpi_bts_coverage(df: DataFrame) -> dict:
    """Full KPI vs BTS presence matrix; BTS rows sorted by KPI count (desc)."""
    presence = df.select("bts_id", "kpi_id").distinct().withColumn("value", f.lit(1))

    all_bts = df.select("bts_id").distinct()
    all_kpi = df.select("kpi_id").distinct()

    full = all_bts.crossJoin(all_kpi)
    full_presence = full.join(presence, on=["bts_id", "kpi_id"], how="left").fillna(
        0, subset=["value"]
    )

    kpi_counts = full_presence.groupBy("bts_id").agg(f.sum("value").alias("kpi_count"))
    full_presence = full_presence.join(kpi_counts, on="bts_id").orderBy(f.desc("kpi_count"))

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

    return {
        "z": _matrix_to_json_lists(pdf.values),
        "x": [str(c) for c in pdf.columns.tolist()],
        "y": [str(i) for i in pdf.index.tolist()],
        "truncated": False,
        "bts_count": len(pdf.index),
        "kpi_count": len(pdf.columns),
    }


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
            f.round(
                f.sum(f.col("kpi_value").isNull().cast("int")) / f.count("*") * 100,
                2,
            ).alias("null_pct"),
        )
        .orderBy(f.desc("records_count"))
    ).toPandas()

    return kpi_catalog
