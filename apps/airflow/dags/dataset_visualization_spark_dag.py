"""PM dataset visualization: Spark reads RAW parquet, writes summary JSON artifacts to S3."""

import os
import sys
from pathlib import Path
from typing import Any

_DAGS_ROOT = Path(__file__).resolve().parent
for _p in (
    str(_DAGS_ROOT),
    os.environ.get("GENPM_GENERATOR_ROOT", "/opt/airflow/generator"),
):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from lib.spark_config import VISUALIZATION_APPLICATION, schema_path  # noqa: E402
from lib.spark_dag import build_spark_job_dag  # noqa: E402


def _finalize(conf: dict[str, Any]) -> dict[str, Any]:
    if not str(conf.get("s3_key") or "").strip():
        raise ValueError("dag_run.conf missing required key: s3_key")
    return conf


dag = build_spark_job_dag(
    dag_id="dataset_visualization_spark",
    application=VISUALIZATION_APPLICATION,
    app_name="DatasetVisualizationSparkApp",
    command=["dataset"],
    conf_finalizer=_finalize,
    env_vars_extra={"GENPM_SCHEMA_PATH": schema_path()},
    tags=["spark", "visualization"],
)
