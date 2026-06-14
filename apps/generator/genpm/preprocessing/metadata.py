"""Dataset metadata extraction for the wide windowed PM parquet."""

import datetime
import json
import os
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.client import Config
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from genpm.utils.logger import get_logger

logger = get_logger()

_META_COLS = {"distname", "bts_id", "window_anchor", "hour_idx"}


def get_kpi_list(df: DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c not in _META_COLS)


def get_metadata(df: DataFrame) -> dict[str, Any]:
    kpi_cols = get_kpi_list(df)

    agg_row = df.agg(
        f.count("*").alias("n_rows"),
        f.countDistinct("distname").alias("n_cells"),
        f.min("window_anchor").alias("window_min"),
        f.max("window_anchor").alias("window_max"),
        f.countDistinct("window_anchor").alias("n_windows"),
        f.min("hour_idx").alias("hour_min"),
        f.max("hour_idx").alias("hour_max"),
        f.countDistinct("hour_idx").alias("hours_per_window"),
    ).collect()[0]

    spatial_rows = (
        df.groupBy("bts_id")
        .agg(f.countDistinct("distname").alias("n_cells_in_bts"))
        .orderBy("bts_id")
        .collect()
    )
    cells_per_bts = {r["bts_id"]: r["n_cells_in_bts"] for r in spatial_rows}

    return {
        "n_rows": int(agg_row["n_rows"]),
        "temporal": {
            "window_anchor_min": str(agg_row["window_min"]),
            "window_anchor_max": str(agg_row["window_max"]),
            "n_windows": int(agg_row["n_windows"]),
            "hour_idx_range": [int(agg_row["hour_min"]), int(agg_row["hour_max"])],
            "hours_per_window": int(agg_row["hours_per_window"]),
        },
        "spatial": {
            "n_bts": len(cells_per_bts),
            "bts_ids": sorted(cells_per_bts.keys()),
            "n_cells": int(agg_row["n_cells"]),
            "cells_per_bts": {k: int(v) for k, v in cells_per_bts.items()},
        },
        "kpis": {
            "count": len(kpi_cols),
            "names": kpi_cols,
        },
    }


def get_summary_df(df: DataFrame) -> pd.DataFrame:
    kpi_cols = get_kpi_list(df)

    agg_exprs = []
    for kpi in kpi_cols:
        agg_exprs += [
            f.mean(f.col(kpi)).alias(f"{kpi}__mean"),
            f.stddev(f.col(kpi)).alias(f"{kpi}__std"),
            f.min(f.col(kpi)).alias(f"{kpi}__min"),
            f.percentile_approx(f.col(kpi), 0.5).alias(f"{kpi}__median"),
            f.max(f.col(kpi)).alias(f"{kpi}__max"),
            f.count(f.when(f.col(kpi).isNull(), 1)).alias(f"{kpi}__null_count"),
        ]

    row = df.agg(*agg_exprs).collect()[0].asDict()

    records = [
        {
            "kpi": kpi,
            "mean": round(row[f"{kpi}__mean"] or 0.0, 6),
            "std": round(row[f"{kpi}__std"] or 0.0, 6),
            "min": row[f"{kpi}__min"],
            "median": row[f"{kpi}__median"],
            "max": row[f"{kpi}__max"],
            "null_count": int(row[f"{kpi}__null_count"]),
        }
        for kpi in kpi_cols
    ]

    return pd.DataFrame(records)


def build_metadata_json(df: DataFrame, dataset_path: str | None = None) -> dict[str, Any]:
    kpi_cols = get_kpi_list(df)
    metadata = get_metadata(df)
    n_rows = metadata.pop("n_rows")
    summary = get_summary_df(df)

    kpi_stats = {
        row["kpi"]: {k: row[k] for k in ("mean", "std", "min", "median", "max", "null_count")}
        for _, row in summary.iterrows()
    }

    return {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "dataset": {
            **({"path": dataset_path} if dataset_path else {}),
            "n_rows": n_rows,
            "n_columns": len(df.columns),
            "meta_columns": sorted(_META_COLS),
            "kpi_count": len(kpi_cols),
        },
        **metadata,
        "kpi_stats": kpi_stats,
    }


def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("s3a://")


def _parse_s3_path(path: str) -> tuple[str, str]:
    """Return (bucket, key) from s3://bucket/key or s3a://bucket/key."""
    stripped = path.split("://", 1)[1]
    bucket, _, key = stripped.partition("/")
    return bucket, key


def _write_to_s3(payload: dict, bucket: str, key: str) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_URL", "http://minio:9000"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    body = json.dumps(payload, ensure_ascii=False, default=str, indent=2).encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info(f"Metadata JSON written to s3://{bucket}/{key}")


def _write_to_local(payload: dict, path: str) -> None:
    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info(f"Metadata JSON written to {local_path}")


def dump_metadata_json(
    df: DataFrame,
    output_path: str | Path,
    dataset_path: str | None = None,
) -> None:
    output_path = str(output_path)
    payload = build_metadata_json(df, dataset_path=dataset_path)

    if _is_s3_path(output_path):
        bucket, key = _parse_s3_path(output_path)
        _write_to_s3(payload, bucket, key)
    else:
        _write_to_local(payload, output_path)
