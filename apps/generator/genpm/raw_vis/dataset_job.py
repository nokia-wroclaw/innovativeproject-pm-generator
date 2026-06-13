"""Airflow dataset visualization: read RAW parquet from S3, write summary JSON artifacts."""

from __future__ import annotations

import os

from genpm.raw_vis.data_visualisation import (
    make_kpi_analysis,
    make_summary,
    top_kpis_for_analysis,
)
from genpm.raw_vis.pm_schema import normalize_pm_dataframe
from genpm.raw_vis.s3_layout import visualization_artifact_key
from genpm.utils.s3_io import write_json_to_s3
from genpm.utils.s3_paths import s3a_path
from genpm.utils.spark_session import SparkDataManager


def run_dataset_visualization(*, dataset_id: str, s3_key: str, bucket: str | None = None) -> None:
    bucket = bucket or os.environ.get("S3_BUCKET", "datasets")
    print(f"Dataset visualization starting (dataset_id={dataset_id})")

    with SparkDataManager(app_name="DatasetVisualizationSparkJob") as sdm:
        spark = sdm.spark
        spark_version = spark.version
        print(f"Spark version: {spark_version}")

        read_path = s3a_path(bucket, s3_key)
        print(f"Reading dataset from {read_path}")
        raw_df = spark.read.parquet(read_path)
        raw_df = normalize_pm_dataframe(raw_df)

        summary = make_summary(raw_df, spark_version=spark_version)
        artifact_name = (
            "summary_error.json"
            if summary.get("status") == "unsupported_schema"
            else "summary.json"
        )
        out_key = visualization_artifact_key(s3_key, artifact_name)
        write_json_to_s3(summary, bucket=bucket, key=out_key)

        if summary.get("status") == "success":
            kpi_ids = top_kpis_for_analysis(summary)
            if kpi_ids:
                analysis = make_kpi_analysis(raw_df, kpi_ids)
                analysis_key = visualization_artifact_key(s3_key, "kpi_analysis.json")
                write_json_to_s3(analysis, bucket=bucket, key=analysis_key)

    print("Dataset visualization job finished successfully")
