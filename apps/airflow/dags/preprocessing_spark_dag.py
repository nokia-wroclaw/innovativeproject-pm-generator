"""PM preprocessing: Spark reads RAW + auxiliary parquets, writes preprocessed artifacts to S3."""

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

from genpm.preprocessing.defaults import finalize_dag_args  # noqa: E402

from lib.spark_config import PREPROCESSING_APPLICATION  # noqa: E402
from lib.spark_dag import build_spark_job_dag  # noqa: E402


def _finalize(conf: dict[str, Any]) -> dict[str, Any]:
    """Validate + apply defaults + resolve output_path_prefix (idempotent if backend filled it)."""
    return {**conf, "dag_args": finalize_dag_args(conf=conf)}


dag = build_spark_job_dag(
    dag_id="preprocessing_pipeline",
    application=PREPROCESSING_APPLICATION,
    app_name="PreprocessingSparkApp",
    conf_finalizer=_finalize,
    spark_preset="FULL_RESOURCES",
    execution_timeout=timedelta(hours=3),
    tags=["spark", "preprocessing"],
)
