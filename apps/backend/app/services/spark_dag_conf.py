"""Shared builders for Airflow dag_run.conf payloads."""

from __future__ import annotations

from typing import Any

from app.models.spark_jobs import (
    PreprocessingDagArgs,
    PreprocessingDagConf,
    VisualizationDagConf,
)
from app.services.s3.visualization_artifacts import dataset_visualization_prefix


def preprocessing_output_prefix(genpm_run_id: str, raw_s3_key: str) -> str:
    """S3 key prefix for final preprocessed artifacts (no bucket, no s3a://)."""
    base = dataset_visualization_prefix(raw_s3_key).strip("/")
    run_id = genpm_run_id.strip("/")
    return f"{base}/preprocessed/{run_id}/final"


def resolve_preprocessing_dag_args(
    *,
    genpm_run_id: str,
    raw_s3_key: str,
    user_args: dict[str, Any] | None = None,
) -> PreprocessingDagArgs:
    args = PreprocessingDagArgs.from_user(user_args)
    if not str(args.output_path_prefix or "").strip():
        args = args.model_copy(
            update={
                "output_path_prefix": preprocessing_output_prefix(genpm_run_id, raw_s3_key),
            }
        )
    args.require_paths()
    return args


def build_preprocessing_dag_conf(
    *,
    genpm_run_id: str,
    dataset_id: int,
    s3_key: str,
    user_dag_args: dict[str, Any] | None = None,
    process_type: str = "preprocessing_feature_engineering",
    file_name: str | None = None,
    dataset_name: str | None = None,
) -> PreprocessingDagConf:
    dag_args = resolve_preprocessing_dag_args(
        genpm_run_id=genpm_run_id,
        raw_s3_key=s3_key,
        user_args=user_dag_args,
    )
    return PreprocessingDagConf(
        genpm_run_id=genpm_run_id,
        dataset_id=dataset_id,
        s3_key=s3_key,
        dag_args=dag_args,
        process_type=process_type,
        file_name=file_name,
        dataset_name=dataset_name,
    )


def build_visualization_dag_conf(
    *,
    dataset_id: int,
    s3_key: str,
    file_name: str | None = None,
) -> VisualizationDagConf:
    return VisualizationDagConf(
        dataset_id=dataset_id,
        s3_key=s3_key,
        file_name=file_name,
    )
