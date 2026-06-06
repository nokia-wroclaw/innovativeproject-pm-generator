import os
from datetime import datetime, timedelta
from pprint import pformat
from typing import Any

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import get_current_context

PREPROCESSING_APP = "/opt/airflow/spark-apps/run_preprocessing_job.py"
GENPM_GENERATOR_ROOT = "/opt/genpm/generator"
SPARK_SUBMIT = os.getenv("SPARK_SUBMIT", "/opt/spark/bin/spark-submit")
SPARK_LOCAL_MASTER = os.getenv("SPARK_LOCAL_MASTER", "local[4]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "8g")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "8g")

_REQUIRED_DAG_ARGS = (
    "kpi_definitions_raw_path",
    "simple_reports_raw_path",
)


def _conf(context: dict[str, Any]) -> dict[str, Any]:
    return context["dag_run"].conf or {}


def _dag_args(conf: dict[str, Any]) -> dict[str, Any]:
    raw = conf.get("dag_args")
    return raw if isinstance(raw, dict) else {}


def _spark_submit_conf() -> dict[str, str]:
    return {
        "spark.master": SPARK_LOCAL_MASTER,
        "spark.executorEnv.PYTHONPATH": GENPM_GENERATOR_ROOT,
        "spark.driverEnv.PYTHONPATH": GENPM_GENERATOR_ROOT,
        "spark.memory.fraction": os.getenv("SPARK_MEMORY_FRACTION", "0.7"),
        "spark.memory.storageFraction": os.getenv("SPARK_STORAGE_FRACTION", "0.2"),
        "spark.sql.shuffle.partitions": os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "200"),
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        **_s3_spark_conf(),
    }


def _s3_spark_conf() -> dict[str, str]:
    endpoint = (os.getenv("S3_URL") or os.getenv("S3_ENDPOINT") or "").rstrip("/")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    if not endpoint or not access_key or not secret_key:
        return {}

    return {
        "spark.hadoop.fs.s3a.endpoint": endpoint,
        "spark.hadoop.fs.s3a.access.key": access_key,
        "spark.hadoop.fs.s3a.secret.key": secret_key,
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    }


def _resolve_storage_path(path: str) -> str:
    raw = path.strip()
    if not raw:
        raise ValueError("Empty storage path")
    if raw.startswith("s3a://"):
        return raw
    if raw.startswith("s3://"):
        return f"s3a://{raw[5:]}"
    if raw.startswith("file://"):
        return raw.removeprefix("file://")
    if raw.startswith("/"):
        return raw

    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        return raw
    return f"s3a://{bucket}/{raw.lstrip('/')}"


def _resolve_output_prefix(conf: dict[str, Any], dag_args: dict[str, Any]) -> str:
    explicit = dag_args.get("output_path_prefix") or conf.get("output_s3_prefix")
    if explicit:
        return _resolve_storage_path(str(explicit))

    run_id = conf.get("genpm_run_id")
    if not run_id:
        raise ValueError("output_path_prefix or genpm_run_id is required for S3 output")

    bucket = os.getenv("S3_BUCKET", "datasets").strip() or "datasets"
    return f"s3a://{bucket}/preprocessed/{run_id}"


def _build_preprocessing_argv(conf: dict[str, Any]) -> list[str]:
    dag_args = _dag_args(conf)
    pm_data_path = dag_args.get("pm_data_raw_path") or conf.get("s3_key")
    if not pm_data_path:
        raise ValueError("pm_data_raw_path or s3_key is required")

    output_prefix = _resolve_output_prefix(conf, dag_args)

    args = [
        "--pm-data-raw-path",
        _resolve_storage_path(str(pm_data_path)),
        "--kpi-definitions-raw-path",
        _resolve_storage_path(str(dag_args["kpi_definitions_raw_path"])),
        "--simple-reports-raw-path",
        _resolve_storage_path(str(dag_args["simple_reports_raw_path"])),
        "--output-path-prefix",
        output_prefix,
        "--kpi-min-global-density",
        str(dag_args.get("kpi_min_global_density", 0.5)),
        "--kpi-global-min-frac-cells-passing",
        str(dag_args.get("kpi_global_min_frac_cells_passing", 0.8)),
        "--kpi-window-coverage-frac",
        str(dag_args.get("kpi_window_coverage_frac", 0.917)),
        "--min-imputable-gap-frac",
        str(dag_args.get("min_imputable_gap_frac", 0.8)),
        "--kpi-min-std-val",
        str(dag_args.get("kpi_min_std_val", 0.01)),
        "--max-zero-frac",
        str(dag_args.get("max_zero_frac", 0.95)),
        "--window-width-hours",
        str(dag_args.get("window_width_hours", 168)),
        "--stride-hours",
        str(dag_args.get("stride_hours", 24)),
        "--max-gap-hours",
        str(dag_args.get("max_gap_hours", 6)),
    ]

    min_joint_windows = dag_args.get("min_joint_windows_abs")
    if min_joint_windows is not None and min_joint_windows != "":
        args.extend(["--min-joint-windows-abs", str(min_joint_windows)])

    if dag_args.get("impute", True):
        args.append("--impute")

    return args


def validate_preprocessing_conf() -> list[str]:
    context = get_current_context()
    conf = _conf(context)
    dag_args = _dag_args(conf)

    missing = [key for key in _REQUIRED_DAG_ARGS if not dag_args.get(key)]
    pm_data_path = dag_args.get("pm_data_raw_path") or conf.get("s3_key")
    if not pm_data_path:
        missing.append("pm_data_raw_path or s3_key")

    if missing:
        raise ValueError(f"Missing preprocessing configuration: {', '.join(missing)}")

    output_prefix = _resolve_output_prefix(conf, dag_args)

    print("Preprocessing DAG configuration:")
    print(pformat(conf))
    print(f"process_type={conf.get('process_type')}")
    print(f"genpm_run_id={conf.get('genpm_run_id')}")
    print(f"dataset_id={conf.get('dataset_id')}")
    print(f"dataset_name={conf.get('dataset_name')}")
    print(f"pm_data_raw_path={_resolve_storage_path(str(pm_data_path))}")
    print(f"output_path_prefix={output_prefix}")

    return _build_preprocessing_argv(conf)


def log_preprocessing_completion() -> None:
    context = get_current_context()
    conf = _conf(context)
    dag_args = _dag_args(conf)
    output_prefix = _resolve_output_prefix(conf, dag_args)
    print("Preprocessing pipeline finished successfully.")
    print(f"genpm_run_id={conf.get('genpm_run_id')}")
    print(f"output_path_prefix={output_prefix}")


default_args = {
    "owner": "genpm",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="preprocessing_pipeline",
    default_args=default_args,
    description="GenPM preprocessing pipeline triggered from modeling or pipeline API",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["feature-engineering", "preprocessing"],
) as dag:
    validate_conf = PythonOperator(
        task_id="validate_configuration",
        python_callable=validate_preprocessing_conf,
    )

    run_preprocessing = SparkSubmitOperator(
        task_id="run_preprocessing",
        application=PREPROCESSING_APP,
        name="GenPM_Preprocessing",
        conn_id="spark_default",
        spark_binary=SPARK_SUBMIT,
        deploy_mode="client",
        driver_memory=SPARK_DRIVER_MEMORY,
        executor_memory=SPARK_EXECUTOR_MEMORY,
        verbose=True,
        conf=_spark_submit_conf(),
        application_args="{{ ti.xcom_pull(task_ids='validate_configuration') }}",
        env_vars={"PYTHONPATH": GENPM_GENERATOR_ROOT},
    )

    finalize = PythonOperator(
        task_id="finalize_run",
        python_callable=log_preprocessing_completion,
    )

    validate_conf >> run_preprocessing >> finalize
