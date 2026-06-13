"""Reusable factory for GenPM Spark DAGs.

Every Spark DAG follows the same shape:

    prepare_conf (PythonOperator)  >>  run_<job> (SparkSubmitOperator)

``prepare_conf`` rebuilds the genpm py_files zip from the live source and finalizes the
``dag_run.conf`` (validation + defaults), pushing it to XCom as a JSON string. The Spark task then
receives it via a single ``--conf-json`` argument. Adding a new DAG is a few lines — see
``apps/airflow/dags/README.md``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Make `lib` importable from DAG files regardless of how Airflow loads them.
_DAGS_ROOT = Path(__file__).resolve().parent.parent
if str(_DAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DAGS_ROOT))

from airflow import DAG  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

from lib.spark_submit import genpm_spark_submit, rebuild_py_files  # noqa: E402

ConfFinalizer = Callable[[dict[str, Any]], dict[str, Any]]

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _prepare_conf_callable(conf_finalizer: ConfFinalizer | None) -> Callable[..., str]:
    def _prepare(**context) -> str:
        conf = context["dag_run"].conf or {}
        rebuild_py_files()
        finalized = conf_finalizer(conf) if conf_finalizer else conf
        return json.dumps(finalized)

    return _prepare


def build_spark_job_dag(
    *,
    dag_id: str,
    application: str,
    app_name: str,
    command: list[str] | None = None,
    conf_finalizer: ConfFinalizer | None = None,
    spark_preset: str | None = None,
    env_vars_extra: dict[str, str] | None = None,
    execution_timeout: timedelta | None = None,
    tags: list[str] | None = None,
    default_args: dict[str, Any] | None = None,
) -> DAG:
    with DAG(
        dag_id=dag_id,
        default_args={**DEFAULT_ARGS, **(default_args or {})},
        schedule=None,
        start_date=datetime(2023, 1, 1),
        catchup=False,
        tags=tags or ["spark"],
    ) as dag:
        prepare_conf = PythonOperator(
            task_id="prepare_conf",
            python_callable=_prepare_conf_callable(conf_finalizer),
        )

        conf_arg = "{{ ti.xcom_pull(task_ids='prepare_conf') }}"
        application_args = [*(command or []), "--conf-json", conf_arg]

        run_job = genpm_spark_submit(
            task_id=f"run_{dag_id}",
            application=application,
            application_args=application_args,
            app_name=app_name,
            spark_preset=spark_preset,
            env_vars_extra=env_vars_extra,
            execution_timeout=execution_timeout,
        )

        prepare_conf >> run_job

    return dag
