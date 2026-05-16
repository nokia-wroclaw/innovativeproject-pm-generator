import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import functions as f
from pyspark.sql.window import Window


def compute_kpi_coverage(df):
    bounds = df.groupBy("distname", "kpi_id").agg(
        f.min("start_time").alias("min_time"), f.max("start_time").alias("max_time")
    )

    # 2. Generate expected hourly timestamps per group
    expected = bounds.withColumn(
        "start_time", f.explode(f.sequence("min_time", "max_time", f.expr("INTERVAL 1 HOUR")))
    ).select("distname", "kpi_id", "start_time")

    # 3. Get actual timestamps (deduplicated)
    actual = df.select("distname", "kpi_id", "start_time").dropDuplicates()

    # 4. Left join to detect missing timestamps
    joined = expected.join(
        actual.withColumn("present", f.lit(1)), on=["distname", "kpi_id", "start_time"], how="left"
    )

    # 5. Mark missing
    result = joined.withColumn("is_missing", f.when(f.col("present").isNull(), 1).otherwise(0))

    # 6. Aggregate coverage
    coverage = (
        result.groupBy("distname", "kpi_id")
        .agg(
            f.count("*").alias("expected_points"),
            f.sum(1 - f.col("is_missing")).alias("actual_points"),
            f.sum("is_missing").alias("missing_points"),
        )
        .withColumn("coverage_ratio", f.col("actual_points") / f.col("expected_points"))
    )

    return coverage


def visualize_kpi_bts_coverage(df):
    """
    Interaktywna heatmapa KPI vs BTS (Plotly)

    Zielony = KPI istnieje na BTS
    Czerwony = brak KPI

    BTS posortowane malejąco po liczbie KPI
    """

    # 1. Obecność KPI na BTS
    presence = df.select("bts_id", "kpi_id").distinct().withColumn("value", f.lit(1))

    # 2. Wszystkie kombinacje
    all_bts = df.select("bts_id").distinct()
    all_kpi = df.select("kpi_id").distinct()

    full = all_bts.crossJoin(all_kpi)

    # 3. Join → brak = 0
    full_presence = full.join(presence, on=["bts_id", "kpi_id"], how="left").fillna(
        0, subset=["value"]
    )

    # 4. Liczba KPI na BTS (do sortowania)
    kpi_counts = full_presence.groupBy("bts_id").agg(f.sum("value").alias("kpi_count"))

    # 5. Sortowanie BTS
    full_presence = full_presence.join(kpi_counts, on="bts_id").orderBy(f.desc("kpi_count"))

    # 6. Pandas pivot
    pdf = full_presence.select("bts_id", "kpi_id", "value").toPandas()

    pivot = pdf.pivot(index="bts_id", columns="kpi_id", values="value")

    # 7. Plotly heatmap
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[[0.0, "red"], [1.0, "green"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title="KPI Coverage per BTS (green = present, red = missing)",
        xaxis_title="KPI ID",
        yaxis_title="BTS ID",
        height=800,
    )

    fig.show()

    return pivot


def visualize_kpi_overlap(df):
    """
    Computes pairwise KPI time coverage:
    coverage(A -> B) = % of timestamps where A exists and B also exists

    Returns:
    - pandas DataFrame (matrix)
    - interactive Plotly heatmap
    """
    # 1. Unique (bts_id, time, kpi)
    base = df.select("bts_id", "start_time", "kpi_id").distinct()

    # 2. Count occurrences per KPI (denominator)
    kpi_counts = base.groupBy("kpi_id").agg(f.count("*").alias("cnt_A"))

    # 3. Self join to find overlaps at same (bts_id, time)
    pairs = (
        base.alias("a")
        .join(base.alias("b"), on=["bts_id", "start_time"], how="inner")
        .select(f.col("a.kpi_id").alias("kpi_A"), f.col("b.kpi_id").alias("kpi_B"))
    )

    # 4. Count overlaps |A ∩ B|
    overlap_counts = pairs.groupBy("kpi_A", "kpi_B").agg(f.count("*").alias("cnt_AB"))

    # 5. Join with counts of A
    coverage = overlap_counts.join(
        kpi_counts.withColumnRenamed("kpi_id", "kpi_A"), on="kpi_A"
    ).withColumn("coverage", f.col("cnt_AB") / f.col("cnt_A"))

    # 6. Convert to pandas pivot
    pdf = coverage.select("kpi_A", "kpi_B", "coverage").toPandas()

    pivot = pdf.pivot(index="kpi_A", columns="kpi_B", values="coverage").fillna(0)

    # 7. Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Coverage %"),
        )
    )

    fig.update_layout(
        title="KPI Coverage Matrix (A → B)",
        xaxis_title="KPI B (covered)",
        yaxis_title="KPI A (base)",
        height=800,
    )

    fig.show()

    return pivot


def profiling_regularity(pm):
    partition_cols = ["kpi_id", "bts_id", "distname"]
    print("--- Profiling Time Series Regularity ---")

    chrono_window = Window.partitionBy(*partition_cols).orderBy("start_time")

    # Calculate the gap in MINUTES between the current row and the previous row
    gap_df = pm.withColumn("prev_ts", f.lag("start_time").over(chrono_window)).withColumn(
        "gap_minutes",
        f.round((f.unix_timestamp("start_time") - f.unix_timestamp("prev_ts")) / 60, 0),
    )

    freq_counts = (
        gap_df.filter(f.col("gap_minutes").isNotNull())
        .groupBy(*partition_cols, "gap_minutes")
        .count()
    )

    window_freq = Window.partitionBy(*partition_cols).orderBy(f.desc("count"))

    natural_freq_df = (
        freq_counts.withColumn("rank", f.row_number().over(window_freq))
        .filter(f.col("rank") == 1)
        .select(*partition_cols, f.col("gap_minutes").alias("natural_freq_minutes"))
    )
    return natural_freq_df


def analyze_multicollinearity(df, group_cols, kpi_ids, max_lag):
    df_filtered = (
        df.select("start_time", "kpi_id", "kpi_value", *group_cols)
        .filter(f.col("kpi_id").isin(kpi_ids))
        .withColumn("kpi_value", f.col("kpi_value").cast("double"))
    )

    pivot_df = (
        df_filtered.groupBy("start_time")
        .pivot("kpi_id", kpi_ids)
        .agg(f.avg("kpi_value"))
        .orderBy("start_time")
    )

    value_cols = [c for c in pivot_df.columns if c != "start_time"]
    w = Window.orderBy("start_time")

    lagged = pivot_df
    for c in value_cols:
        for lag in range(1, max_lag + 1):
            lagged = lagged.withColumn(f"{c}_l{lag}", f.lag(f.col(c), lag).over(w))

    lagged = lagged.dropna()

    feature_cols = [c for c in lagged.columns if c != "start_time"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    vec_df = assembler.transform(lagged).select("features")

    corr_matrix = Correlation.corr(vec_df, "features", method="pearson").collect()[0][0]
    corr_array = corr_matrix.toArray()

    corr_df = pd.DataFrame(corr_array, index=feature_cols, columns=feature_cols)

    figures = {}

    for kpi in kpi_ids:
        row_features = [c for c in corr_df.index if c == kpi or c.startswith(f"{kpi}_l")]
        col_features = [c for c in corr_df.columns if c not in row_features]

        if not row_features or not col_features:
            continue

        sub_df = corr_df.loc[row_features, col_features]

        fig = go.Figure(
            data=go.Heatmap(
                z=sub_df.values,
                x=sub_df.columns,
                y=sub_df.index,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                zmid=0,
                colorbar=dict(title="corr"),
            )
        )

        fig.update_layout(
            title=f"Correlation heatmap: {kpi} vs other features",
            xaxis_title="Other features",
            yaxis_title=f"{kpi} features",
            height=max(100, 20 * len(row_features)),
            width=max(400, 20 * len(col_features)),
        )

        figures[kpi] = fig
        fig.show()

    return corr_df, figures


def plot_acf(df: pd.DataFrame, freq_map):
    # Get the unique KPI IDs from your pandas dataframe
    kpi_ids = df["kpi_id"].unique()

    fig = go.Figure()
    dropdown_buttons = []

    for i, kpi in enumerate(kpi_ids):
        df_kpi = df[df["kpi_id"] == kpi].sort_values("Lag")
        lags = df_kpi["Lag"].tolist()
        corrs = df_kpi["Correlation"].tolist()

        natural_freq = freq_map.get(kpi, "N/A")

        # stems
        x_stems = []
        y_stems = []
        for x, y in zip(lags, corrs, strict=False):
            x_stems.extend([x, x, None])
            y_stems.extend([0, y, None])

        fig.add_trace(
            go.Scatter(
                x=x_stems,
                y=y_stems,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                showlegend=False,
                hoverinfo="skip",
                visible=(i == 0),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=lags,
                y=corrs,
                mode="markers",
                marker=dict(color="#1f77b4", size=8),
                name="ACF",
                hovertemplate=(
                    "KPI ID: "
                    + str(kpi)
                    + "<br>Lag: %{x}"
                    + "<br>Correlation: %{y:.3f}"
                    + f"<br>Natural frequency (min): {natural_freq}"
                    "<extra></extra>"
                ),
                visible=(i == 0),
            )
        )

        visibility = [False] * (len(kpi_ids) * 2)
        visibility[i * 2] = True
        visibility[i * 2 + 1] = True

        dropdown_buttons.append(
            dict(
                label=str(kpi),
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": f"Autocorrelation (ACF) Plot for KPI: {kpi}",
                        "annotations": [
                            dict(
                                text="Select KPI:",
                                x=0,
                                y=1.13,
                                xref="paper",
                                yref="paper",
                                showarrow=False,
                                font=dict(size=14),
                            ),
                            dict(
                                text=f"Natural frequency: <b>{natural_freq}</b> min",
                                x=0,
                                y=1.06,
                                xref="paper",
                                yref="paper",
                                showarrow=False,
                                align="left",
                                font=dict(size=13, color="#444"),
                                bgcolor="rgba(240,240,240,0.8)",
                                bordercolor="lightgray",
                                borderwidth=1,
                                borderpad=4,
                            ),
                        ],
                    },
                ],
            )
        )

    fig.add_hline(y=0, line_width=1, line_color="black")

    first_kpi = kpi_ids[0]
    first_freq = freq_map.get(first_kpi, "N/A")

    fig.update_layout(
        title=f"Autocorrelation (ACF) Plot for KPI: {first_kpi}",
        xaxis_title="Lag (Hours)",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.1, 1.1]),
        xaxis=dict(tickmode="linear", dtick=1),
        plot_bgcolor="white",
        height=600,
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        annotations=[
            dict(
                text="Select KPI:",
                x=0,
                y=1.13,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            ),
            dict(
                text=f"Natural frequency: <b>{first_freq}</b> min",
                x=0,
                y=1.06,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font=dict(size=13, color="#444"),
                bgcolor="rgba(240,240,240,0.8)",
                bordercolor="lightgray",
                borderwidth=1,
                borderpad=4,
            ),
        ],
    )

    fig.show()


def check_even_timestamps(df, time_col):
    w = Window.orderBy(time_col)

    df_diff = (
        df.withColumn("prev_ts", f.lag(time_col).over(w))
        .withColumn("diff_sec", f.col(time_col).cast("long") - f.col("prev_ts").cast("long"))
        .filter(f.col("diff_sec").isNotNull())
    )

    stats = df_diff.agg(
        f.min("diff_sec").alias("min_diff"),
        f.max("diff_sec").alias("max_diff"),
        f.countDistinct("diff_sec").alias("unique_diffs"),
    ).collect()[0]

    is_even = stats["min_diff"] == stats["max_diff"]

    return {
        "is_even": is_even,
        "min_diff": stats["min_diff"],
        "max_diff": stats["max_diff"],
        "unique_diffs": stats["unique_diffs"],
    }


def multi_agg_plots(
    df, group_col, value_col, time_col=None, rolling_window=3, title="Aggregations"
):
    # Base aggregation
    agg_df = df.groupby(group_col)[value_col].agg(["mean", "max", "min", "count"]).reset_index()

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Mean",
            "Range (Min-Max)",
            "Count",
            f"Rolling Mean (window={rolling_window})",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    fig.add_trace(go.Bar(x=agg_df[group_col], y=agg_df["mean"], name="Mean"), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=agg_df[group_col], y=agg_df["max"], mode="lines", line=dict(width=0), showlegend=False
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=agg_df[group_col],
            y=agg_df["min"],
            mode="lines",
            fill="tonexty",  # fill between min and max
            name="Range (min-max)",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(go.Bar(x=agg_df[group_col], y=agg_df["count"], name="Count"), row=2, col=1)

    if time_col:
        ts_df = df.sort_values(time_col)

        ts_df["rolling_mean"] = ts_df[value_col].rolling(window=rolling_window).mean()

        fig.add_trace(
            go.Scatter(
                x=ts_df[time_col],
                y=ts_df["rolling_mean"],
                mode="lines",
                name=f"Rolling Mean ({rolling_window})",
            ),
            row=2,
            col=2,
        )
    else:
        fig.add_trace(
            go.Scatter(x=agg_df[group_col], y=agg_df["mean"], mode="lines", name="Mean (fallback)"),
            row=2,
            col=2,
        )

    fig.update_layout(title=title, height=700, showlegend=False)

    return fig


def ts_with_distribution(df, time_col, value_col, title="Time Series with Distribution"):
    df = df.sort_values(time_col)

    values = df[value_col]

    mean = values.mean()
    std = values.std()

    upper_3std = mean + 3 * std
    # lower_3std = mean - 3 * std

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("Time Series", "Distribution"),
        horizontal_spacing=0.08,
    )

    # --- TIME SERIES ---
    fig.add_trace(go.Scatter(x=df[time_col], y=values, mode="lines", name="Value"), row=1, col=1)

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=df[time_col], y=[mean] * len(df), mode="lines", name="Mean", line=dict(dash="dash")
        ),
        row=1,
        col=1,
    )

    # +3 std
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=[upper_3std] * len(df),
            mode="lines",
            name="+3σ",
            line=dict(dash="dot"),
        ),
        row=1,
        col=1,
    )

    # # -3 std
    # fig.add_trace(
    #     go.Scatter(
    #         x=df[time_col],
    #         y=[lower_3std] * len(df),
    #         mode='lines',
    #         name='-3σ',
    #         line=dict(dash='dot')
    #     ),
    #     row=1, col=1
    # )

    # --- DISTRIBUTION (HISTOGRAM) ---
    fig.add_trace(
        go.Histogram(y=values, nbinsy=30, name="Distribution", showlegend=False), row=1, col=2
    )

    # Layout tweaks
    fig.update_layout(title=title, height=500, bargap=0.05)

    # Align histogram axis
    fig.update_yaxes(title_text=value_col, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    return fig


def plot_kpi_versions_selector(spark_df, group_col="bts_id", all_kpis=False):
    """
    Interactive KPI version timeline explorer.

    Workflow:
      1. A base_kpi is extracted from kpi_id (strips trailing letter).
      2. An ipywidgets dropdown lets you pick a base_kpi without loading all data.
      3. On selection, Plotly renders Gantt-style bars for every version (kpi_id)
         of that base_kpi — blue spans, red gaps (>1 h holes in data).

    Parameters
    ----------
    spark_df : pyspark.sql.DataFrame
        Schema: start_time (timestamp), kpi_id (string), kpi_value (double),
                bts_id (string), distname (string)

    Returns
    -------
    None  (displays widgets inline in a Jupyter notebook)
    """
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from IPython.display import display

    # ── Step 1: derive base_kpi & pre-compute spans + gaps for ALL base KPIs ──
    # We do the heavy Spark work once, then filter in pandas per selection.

    df = spark_df.withColumn("base_kpi", f.regexp_extract("kpi_id", r"^(.*?)[a-z]?$", 1))

    if not all_kpis:
        # base_kpis that have more than one distinct kpi_id
        multi_version_bases = (
            df.groupBy("base_kpi")
            .agg(f.countDistinct("kpi_id").alias("version_count"))
            .filter(f.col("version_count") > 1)
            .select("base_kpi")
        )

        df = df.join(multi_version_bases, on="base_kpi", how="inner")

    group_cols = ["base_kpi", "kpi_id", group_col]

    # Availability spans
    span_df = (
        df.groupBy(*group_cols)
        .agg(
            f.min("start_time").alias("start_time"),
            f.max("start_time").alias("end_time"),
        )
        .withColumn("type", f.lit("span"))
    )

    # Gaps: consecutive timestamps more than 1 h apart
    w = Window.partitionBy(*group_cols).orderBy("start_time")
    gap_df = (
        df.withColumn("next_time", f.lead("start_time").over(w))
        .filter(
            f.col("next_time").isNotNull()
            & (f.col("next_time") > f.expr("start_time + interval 1 hour"))
        )
        .select(
            *group_cols,
            f.col("start_time"),
            f.col("next_time").alias("end_time"),
        )
        .withColumn("type", f.lit("gap"))
    )

    pdf = span_df.unionByName(gap_df).toPandas()
    pdf["start_time"] = pd.to_datetime(pdf["start_time"])
    pdf["end_time"] = pd.to_datetime(pdf["end_time"])
    pdf["duration"] = pdf["end_time"] - pdf["start_time"]
    pdf["y_label"] = pdf["kpi_id"] + "  │  " + pdf[group_col]

    base_kpis = sorted(pdf["base_kpi"].unique())

    # ── Step 2: widget ────────────────────────────────────────────────────────

    dropdown = widgets.Dropdown(
        options=base_kpis,
        description="Base KPI:",
        layout=widgets.Layout(width="350px"),
        style={"description_width": "80px"},
    )

    out = widgets.Output()

    def render(base):
        subset = pdf[pdf["base_kpi"] == base].copy()
        kpi_versions = sorted(subset["kpi_id"].unique())

        COLOR = {"span": "#4C9BE8", "gap": "#E8524C"}

        fig = go.Figure()

        for kpi_id in kpi_versions:
            rows = subset[subset["kpi_id"] == kpi_id]
            for _, row in rows.iterrows():
                duration_ms = row["duration"].total_seconds() * 1000
                fig.add_trace(
                    go.Bar(
                        x=[duration_ms],
                        y=[row["y_label"]],
                        base=row["start_time"].timestamp() * 1000,
                        orientation="h",
                        marker=dict(
                            color=COLOR[row["type"]],
                            opacity=0.85,
                            line=dict(width=0),
                        ),
                        name=row["type"].capitalize(),
                        legendgroup=row["type"],
                        showlegend=False,  # legend added manually below
                        hovertemplate=(
                            f"<b>{row['kpi_id']}</b><br>"
                            f"{group_col}: {row[group_col]}<br>"
                            f"Type: {row['type']}<br>"
                            f"From: {row['start_time'].strftime('%Y-%m-%d %H:%M')}<br>"
                            f"To:   {row['end_time'].strftime('%Y-%m-%d %H:%M')}<br>"
                            f"Duration: {str(row['duration'])}"
                            "<extra></extra>"
                        ),
                    )
                )

        # Manual legend entries
        for label, color in COLOR.items():
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    orientation="h",
                    marker=dict(color=color),
                    name=label.capitalize(),
                    legendgroup=label,
                    showlegend=True,
                )
            )

        # X axis: plotly needs epoch-ms for timestamps on bar base
        fig.update_layout(
            title=dict(
                text=f"KPI Version Timeline — <b>{base}</b>",
                font=dict(size=16),
            ),
            xaxis=dict(
                type="date",
                title="Time",
                tickformat="%Y-%m-%d",
            ),
            yaxis=dict(
                title=f"KPI Version  │  {group_col}",
                autorange="reversed",
                categoryorder="array",
                categoryarray=sorted(
                    subset["y_label"].unique(), key=lambda s: (s.split("│")[0], s.split("│")[1])
                ),
                tickfont=dict(family="monospace", size=11),
            ),
            barmode="overlay",
            bargap=0.25,
            height=max(1400, 60 + 45 * len(kpi_versions)),
            width=1400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=120, r=40, t=80, b=60),
            plot_bgcolor="#F7F9FC",
            paper_bgcolor="#FFFFFF",
        )

        with out:
            out.clear_output(wait=True)
            fig.show()

    # Render immediately for the default selection
    render(dropdown.value)
    dropdown.observe(lambda change: render(change["new"]), names="value")

    display(widgets.VBox([dropdown, out]))


def analyze_kpi_vs_metadata(df_kpi, df_meta):
    """
    Checks how metadata (report_value) impacts KPI values.

    Returns:
    - correlation table
    - Plotly boxplots per KPI vs metadata
    """

    # --- KPI stats ---
    kpi_stats = df_kpi.groupBy("distname", "kpi_id").agg(f.avg("kpi_value").alias("kpi_mean"))

    # --- metadata pivot ---
    meta = df_meta.groupBy("distname").pivot("report_name").agg(f.first("report_result"))

    # --- join ---
    data = kpi_stats.join(meta, on="distname")

    pdf = data.toPandas()

    import pandas as pd
    import plotly.express as px

    results = []

    report_cols = [c for c in pdf.columns if c not in ["distname", "kpi_id", "kpi_mean"]]

    # --- correlation analysis ---
    for report in report_cols:
        for kpi in pdf["kpi_id"].unique():
            subset = pdf[pdf["kpi_id"] == kpi]

            if subset[report].nunique() > 1:
                corr = subset[[report, "kpi_mean"]].corr().iloc[0, 1]
            else:
                corr = None

            results.append({"kpi_id": kpi, "report_name": report, "correlation": corr})

            # --- visualization ---
            fig = px.box(subset, x=report, y="kpi_mean", title=f"{kpi} vs {report}")
            fig.show()

    results_df = pd.DataFrame(results)

    return results_df
