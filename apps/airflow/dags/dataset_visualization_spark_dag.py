"""Trigger PM dataset visualization: Spark reads RAW parquet, writes summary JSON to S3."""

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

with DAG(
    dag_id="dataset_visualization_spark",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["spark", "visualization"],
) as dag:
    SparkSubmitOperator(
        task_id="run_pm_visualization",
        conn_id="spark_default",
        application="/opt/airflow/generator/apps/dataset_visualization_spark_job.py",
        name="DatasetVisualizationSparkApp",
        verbose=True,
        conf={
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
            "spark.pyspark.driver.python": "/usr/python/bin/python3.13",
            "spark.pyspark.python": _SPARK_EXECUTOR_PYTHON,
        },
        env_vars={
            "GENPM_GENERATOR_ROOT": "/opt/airflow/generator",
            "GENPM_SCHEMA_PATH": "/opt/airflow/shared/pm_schema_columns.json",
            "GENPM_PYSPARK_PYTHON": "/usr/python/bin/python3.13",
            "GENPM_SPARK_EXECUTOR_PYTHON": _SPARK_EXECUTOR_PYTHON,
            "GENPM_DATASET_ID": "{{ dag_run.conf.get('dataset_id', '') | string }}",
            "GENPM_S3_KEY": "{{ dag_run.conf.get('s3_key', '') | string }}",
            "PYSPARK_DRIVER_PYTHON": "/usr/python/bin/python3.13",
        },
    )
