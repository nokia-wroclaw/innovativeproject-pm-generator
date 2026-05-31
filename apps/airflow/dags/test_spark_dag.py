from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
}

with DAG(
    dag_id='test_spark_connection',
    default_args=default_args,
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['spark', 'test'],
) as dag:

    submit_job = SparkSubmitOperator(
        task_id='run_pyspark_test',
        conn_id='spark_default',
        application='/opt/airflow/spark-apps/test_spark_job.py',
        name='TestSparkAppFromAirflow',
        verbose=True,
        conf={
            'spark.driver.host': 'airflow-worker',
            'spark.driver.bindAddress': '0.0.0.0',
        },
    )
    submit_job