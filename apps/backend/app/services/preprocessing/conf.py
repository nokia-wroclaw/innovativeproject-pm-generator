"""Airflow conf helpers for the preprocessing_pipeline DAG."""

from __future__ import annotations

import os
from typing import Any

from app.services.s3.visualization_artifacts import dataset_visualization_prefix

PREPROCESSING_DAG_ID = "preprocessing_pipeline"

REQUIRED_DAG_ARG_KEYS = ("kpi_definitions_raw_path", "simple_reports_raw_path")

DEFAULT_PREPROCESSING_DAG_ARGS: dict[str, Any] = {
    "kpi_min_global_density": 0.5,
    "kpi_global_min_frac_cells_passing": 0.8,
    "min_imputable_gap_frac": 0.8,
    "kpi_min_std_val": 0.01,
    "max_zero_frac": 0.95,
    "window_width_hours": 168,
    "stride_hours": 24,
    "max_gap_hours": 24,
    "impute": True,
}

PREPROCESSING_OUTPUT_OBJECTS = (
    "pm_df_long_indexed_winds",
    "scaling_params_df",
    "pm_data_const_kpi",
    "kpi_definitions",
    "simple_reports",
)


class PreprocessingConfigError(ValueError):
    """Invalid or incomplete preprocessing DAG configuration."""


def preprocessing_output_prefix(genpm_run_id: str, raw_s3_key: str) -> str:
    """S3 key prefix for final preprocessed artifacts (no bucket, no s3a://)."""
    base = dataset_visualization_prefix(raw_s3_key).strip("/")
    run_id = genpm_run_id.strip("/")
    return f"{base}/preprocessed/{run_id}/final"


def build_preprocessing_dag_args(
    *,
    genpm_run_id: str,
    raw_s3_key: str,
    user_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {**DEFAULT_PREPROCESSING_DAG_ARGS, **(user_args or {})}
    output_prefix = str(merged.get("output_path_prefix") or "").strip()
    if not output_prefix:
        merged["output_path_prefix"] = preprocessing_output_prefix(genpm_run_id, raw_s3_key)

    missing = [key for key in REQUIRED_DAG_ARG_KEYS if not str(merged.get(key) or "").strip()]
    if missing:
        raise PreprocessingConfigError(
            "Missing required preprocessing dag_args: "
            + ", ".join(missing)
            + ". Provide KPI definitions and simple reports S3 keys."
        )
    return merged


def s3_uri_for_output_prefix(output_prefix: str) -> str:
    bucket = os.getenv("S3_BUCKET") or "datasets"
    key = output_prefix.strip().lstrip("/")
    return f"s3://{bucket}/{key}"


def preprocessing_artifact_paths(output_prefix: str) -> dict[str, str]:
    base = s3_uri_for_output_prefix(output_prefix).rstrip("/")
    return {name: f"{base}/{name}" for name in PREPROCESSING_OUTPUT_OBJECTS}
