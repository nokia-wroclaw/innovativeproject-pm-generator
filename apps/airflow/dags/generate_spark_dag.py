"""Synthetic dataset generation: downloads trained model from S3, runs CVAE-LSTM inference,
uploads generated parquet back to S3."""

import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

_DAGS_ROOT = Path(__file__).resolve().parent
for _p in (
    str(_DAGS_ROOT),
    os.environ.get("GENPM_GENERATOR_ROOT", "/opt/airflow/generator"),
):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from genpm.modelling.defaults import finalize_generate_dag_args  # noqa: E402

from lib.spark_config import GENERATION_APPLICATION, DATA_SIMILARITY_APPLICATION  # noqa: E402
from lib.spark_dag import build_spark_job_dag  # noqa: E402
from lib.spark_submit import genpm_spark_submit  # noqa: E402


def _finalize(conf: dict[str, Any]) -> dict[str, Any]:
    """Validate + apply defaults + auto-resolve output_path_prefix (idempotent if backend filled it)."""
    return {**conf, "dag_args": finalize_generate_dag_args(conf=conf)}


dag = build_spark_job_dag(
    dag_id="generate_pipeline",
    application=GENERATION_APPLICATION,
    app_name="GenerationSparkApp",
    command=["generate"],
    conf_finalizer=_finalize,
    spark_preset="HALF_SAFE",
    execution_timeout=timedelta(hours=2),
    tags=["spark", "generate"],
)

with dag:
    run_data_similarity = genpm_spark_submit(
        task_id="run_data_similarity",
        application=DATA_SIMILARITY_APPLICATION,
        application_args=["--conf-json", "{{ ti.xcom_pull(task_ids='prepare_conf') }}"],
        app_name="DataSimilaritySparkApp",
        spark_preset="HALF_SAFE",
    )

    dag.get_task("run_generate_pipeline") >> run_data_similarity
