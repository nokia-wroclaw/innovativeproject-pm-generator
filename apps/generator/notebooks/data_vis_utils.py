import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from pyspark.sql import functions as f

from utils.utils import SparkDataManager

# use spark session if needed
sdm = SparkDataManager()


matplotlib.use("Agg")  # backend bez GUI — kluczowe dla batch processingu


def fig_to_base64(fig) -> str:
    """Konwertuje matplotlib figure do base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def schema(df):
    # SCHEMAT + NULL % per kolumna
    total = df.count()

    null_exprs = [
        f.round(f.sum(f.col(c).isNull().cast("int")) / total * 100, 2).alias(c) for c in df.columns
    ]
    nulls_row = df.select(null_exprs).toPandas()

    schema_rows = []
    for field in df.schema.fields:
        schema_rows.append(
            {
                "kolumna": field.name,
                "typ": str(field.dataType),
                "nullable": field.nullable,
                "null_pct": float(nulls_row[field.name].iloc[0]),
            }
        )
    schema_df = pd.DataFrame(schema_rows)
    return schema_df


def basic_info(df):
    # PODSTAWOWE INFORMACJE O DATASECIE
    counts = df.agg(
        f.count("*").alias("n_wierszy"),
        f.countDistinct("kpi_id").alias("n_kpi"),
        f.countDistinct("bts_id").alias("n_btsow"),
        f.countDistinct("distname").alias("n_distname"),
        f.min("start_date").alias("data_od"),
        f.max("start_date").alias("data_do"),
    ).toPandas()
    return counts


def kpi_bts_coverage(df):
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

    return fig.to_json()


def kpi_catalog(df):
    # Katalog KPI
    kpi_catalog = (
        df.groupBy("kpi_id")
        .agg(
            f.count("*").alias("n_rekordow"),
            f.countDistinct("bts_id").alias("n_stacji"),
            f.countDistinct("start_date").alias("n_dni"),
            f.min("start_time").alias("od"),
            f.max("start_time").alias("do"),
            f.round(f.avg("kpi_value"), 4).alias("srednia"),
            f.round(f.stddev("kpi_value"), 4).alias("std"),
            f.round(f.min("kpi_value"), 4).alias("min"),
            f.round(f.max("kpi_value"), 4).alias("max"),
            f.round(f.sum(f.col("kpi_value").isNull().cast("int")) / f.count("*") * 100, 2).alias(
                "null_pct"
            ),
        )
        .orderBy(f.desc("n_rekordow"))
    ).toPandas()

    return kpi_catalog
