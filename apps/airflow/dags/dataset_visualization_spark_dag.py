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

_SPARK_HOME = os.environ.get("SPARK_HOME", "/opt/spark")
_SPARK_PYTHONPATH = (
    f"{_SPARK_HOME}/python:{_SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip"
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
            "spark.pyspark.driver.python": os.environ.get(
                "GENPM_PYSPARK_PYTHON", "/opt/airflow/genpm-venv/bin/python"
            ),
            "spark.pyspark.python": _SPARK_EXECUTOR_PYTHON,
        },
        env_vars={
            "GENPM_SCHEMA_PATH": "/opt/airflow/shared/pm_schema_columns.json",
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
