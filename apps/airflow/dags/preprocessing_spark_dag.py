"""Trigger PM preprocessing: Spark reads RAW + auxiliary parquets, writes preprocessed artifacts to S3."""

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
}

_SPARK_EXECUTOR_PYTHON = os.environ.get(
    "GENPM_SPARK_EXECUTOR_PYTHON",
    "/home/hostuser/app/.venv/bin/python3",
)

_SPARK_HOME = os.environ.get("SPARK_HOME", "/opt/spark")
_SPARK_PYTHONPATH = (
    f"{_SPARK_HOME}/python:{_SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip"
)

with DAG(
    dag_id="preprocessing_pipeline",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["spark", "preprocessing"],
) as dag:
    SparkSubmitOperator(
        task_id="run_preprocessing",
        conn_id="spark_default",
        application="/opt/airflow/generator/apps/preprocessing_spark_job.py",
        name="PreprocessingSparkApp",
        verbose=True,
        conf={
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
            "spark.pyspark.driver.python": os.environ.get(
                "GENPM_PYSPARK_PYTHON", "/opt/airflow/genpm-venv/bin/python"
            ),
            "spark.pyspark.python": _SPARK_EXECUTOR_PYTHON,
        },
        env_vars={
            "GENPM_PYSPARK_PYTHON": os.environ.get(
                "GENPM_PYSPARK_PYTHON", "/opt/airflow/genpm-venv/bin/python"
            ),
            "GENPM_SPARK_EXECUTOR_PYTHON": _SPARK_EXECUTOR_PYTHON,
            "GENPM_SPARK_CONFIG": os.environ.get("GENPM_SPARK_CONFIG", "AGG_HEAVY"),
            "SPARK_CORE_NUMBER": os.environ.get("SPARK_CORE_NUMBER", "8"),
            "SPARK_EXECUTOR_MEMORY": os.environ.get("SPARK_EXECUTOR_MEMORY", "8g"),
            "SPARK_DRIVER_MEMORY": os.environ.get("SPARK_DRIVER_MEMORY", "8g"),
            "GENPM_DATASET_ID": "{{ dag_run.conf.get('dataset_id', '') | string }}",
            "GENPM_S3_KEY": "{{ dag_run.conf.get('s3_key', '') | string }}",
            "GENPM_KPI_DEFINITIONS_S3_KEY": (
                "{{ dag_run.conf.get('dag_args', {}).get('kpi_definitions_raw_path', '') }}"
            ),
            "GENPM_SIMPLE_REPORTS_S3_KEY": (
                "{{ dag_run.conf.get('dag_args', {}).get('simple_reports_raw_path', '') }}"
            ),
            "GENPM_OUTPUT_PREFIX": (
                "{{ dag_run.conf.get('dag_args', {}).get('output_path_prefix', '') }}"
            ),
            "GENPM_KPI_MIN_GLOBAL_DENSITY": (
                "{{ dag_run.conf.get('dag_args', {}).get('kpi_min_global_density', 0.5) }}"
            ),
            "GENPM_KPI_GLOBAL_MIN_FRAC_CELLS_PASSING": (
                "{{ dag_run.conf.get('dag_args', {}).get('kpi_global_min_frac_cells_passing', 0.8) }}"
            ),
            "GENPM_MIN_IMPUTABLE_GAP_FRAC": (
                "{{ dag_run.conf.get('dag_args', {}).get('min_imputable_gap_frac', 0.8) }}"
            ),
            "GENPM_KPI_MIN_STD_VAL": (
                "{{ dag_run.conf.get('dag_args', {}).get('kpi_min_std_val', 0.01) }}"
            ),
            "GENPM_MAX_ZERO_FRAC": (
                "{{ dag_run.conf.get('dag_args', {}).get('max_zero_frac', 0.95) }}"
            ),
            "GENPM_WINDOW_WIDTH_HOURS": (
                "{{ dag_run.conf.get('dag_args', {}).get('window_width_hours', 168) }}"
            ),
            "GENPM_STRIDE_HOURS": (
                "{{ dag_run.conf.get('dag_args', {}).get('stride_hours', 24) }}"
            ),
            "GENPM_MAX_GAP_HOURS": (
                "{{ dag_run.conf.get('dag_args', {}).get('max_gap_hours', 24) }}"
            ),
            "GENPM_MIN_JOINT_WINDOWS_ABS": (
                "{{ dag_run.conf.get('dag_args', {}).get('min_joint_windows_abs', '') }}"
            ),
            "GENPM_IMPUTE": (
                "{{ dag_run.conf.get('dag_args', {}).get('impute', true) | string }}"
            ),
            "S3_URL": os.environ.get("S3_URL", "http://minio:9000"),
            "S3_BUCKET": os.environ.get("S3_BUCKET", "datasets"),
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "PYSPARK_DRIVER_PYTHON": os.environ.get(
                "GENPM_PYSPARK_PYTHON", "/opt/airflow/genpm-venv/bin/python"
            ),
            "PYTHONPATH": _SPARK_PYTHONPATH,
        },
    )
