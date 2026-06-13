"""Airflow conf helpers for the preprocessing_pipeline DAG."""

from __future__ import annotations

import os
from typing import Any

from app.models.spark_jobs import DEFAULT_PREPROCESSING_DAG_ARGS, PreprocessingConfigError
from app.services.spark_dag_conf import preprocessing_output_prefix, resolve_preprocessing_dag_args

PREPROCESSING_DAG_ID = "preprocessing_pipeline"

REQUIRED_DAG_ARG_KEYS = ("kpi_definitions_raw_path", "simple_reports_raw_path")

PREPROCESSING_OUTPUT_OBJECTS = (
    "pm_df_long_indexed_winds",
    "scaling_params_df",
    "pm_data_const_kpi",
    "kpi_definitions",
    "simple_reports",
)


def build_preprocessing_dag_args(
    *,
    genpm_run_id: str,
    raw_s3_key: str,
    user_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return resolve_preprocessing_dag_args(
        genpm_run_id=genpm_run_id,
        raw_s3_key=raw_s3_key,
        user_args=user_args,
    ).to_conf_dict()


def s3_uri_for_output_prefix(output_prefix: str) -> str:
    bucket = os.getenv("S3_BUCKET") or "datasets"
    key = output_prefix.strip().lstrip("/")
    return f"s3://{bucket}/{key}"


def preprocessing_artifact_paths(output_prefix: str) -> dict[str, str]:
    base = s3_uri_for_output_prefix(output_prefix).rstrip("/")
    return {name: f"{base}/{name}" for name in PREPROCESSING_OUTPUT_OBJECTS}


__all__ = [
    "DEFAULT_PREPROCESSING_DAG_ARGS",
    "PREPROCESSING_DAG_ID",
    "PREPROCESSING_OUTPUT_OBJECTS",
    "PreprocessingConfigError",
    "REQUIRED_DAG_ARG_KEYS",
    "build_preprocessing_dag_args",
    "preprocessing_artifact_paths",
    "preprocessing_output_prefix",
]
