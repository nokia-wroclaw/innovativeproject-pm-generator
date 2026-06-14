"""Shared SparkSubmit paths and config for GenPM Airflow DAGs.

Resource presets are sourced once from ``genpm.utils.consts.SPARK_CONFIGS`` (single source of
truth). The ``spark.master`` key is stripped here because the master is supplied by the Airflow
Spark connection (``conn_id`` -> ``spark://<user>-genpm-spark:7077``); the genpm job respects it.

Infrastructure values come from the environment. Required ones fail loudly rather than falling back
to a misleading hardcoded path, so a misconfigured (e.g. wrong-user) stack surfaces immediately.
"""

from __future__ import annotations

import os

from genpm.utils.consts import SPARK_CONFIGS
from genpm.utils.spark_bootstrap import spark_pythonpath

SPARK_CONN_ID = "spark_default"

GENPM_GENERATOR_ROOT = os.environ.get("GENPM_GENERATOR_ROOT", "/opt/airflow/generator")
# Dev-only override: when set, genpm is shipped to executors via --py-files (live code). In prod it
# is left unset and executors use the pinned wheel installed in the image. See lib/spark_submit.py.
GENPM_PY_FILES = os.environ.get("GENPM_PY_FILES")

# SparkSubmitOperator.application must be a file path. We use stable launcher scripts that delegate
# to the installed genpm package (wheel in prod, live mount in dev), so the path is independent of
# where genpm itself is installed. The launchers live under <generator>/apps, which exists both in
# the image (COPY) and via the dev mount; spark-submit ships the file to the cluster in cluster mode.
GENPM_SPARK_APPS_DIR = os.environ.get(
    "GENPM_SPARK_APPS_DIR", f"{GENPM_GENERATOR_ROOT}/apps"
)

PREPROCESSING_APPLICATION = f"{GENPM_SPARK_APPS_DIR}/run_preprocessing.py"
VISUALIZATION_APPLICATION = f"{GENPM_SPARK_APPS_DIR}/run_visualization.py"

DEFAULT_SPARK_PRESET = "HALF_SAFE"


def _required_env(name: str, *, hint: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required infra env var {name} is not set. {hint}")
    return value


def spark_driver_python() -> str:
    """Python for the Spark driver — runs in the Airflow worker (genpm-venv)."""
    return os.environ.get("GENPM_PYSPARK_PYTHON", "/opt/airflow/genpm-venv/bin/python")


def spark_executor_python() -> str:
    """Python for Spark executors — runs in the <user>-genpm-spark container's venv."""
    return _required_env(
        "GENPM_SPARK_EXECUTOR_PYTHON",
        hint="Set it to the python in the Spark container venv, e.g. /home/<user>/app/.venv/bin/python3.",
    )


def schema_path() -> str:
    """Path to the shared PM schema JSON, mounted into the Airflow + Spark containers."""
    return _required_env(
        "GENPM_SCHEMA_PATH",
        hint="Mount ../shared and set GENPM_SCHEMA_PATH (e.g. /opt/airflow/shared/pm_schema_columns.json).",
    )


def spark_submit_conf(preset: str = DEFAULT_SPARK_PRESET) -> dict[str, str]:
    """Resource + driver/executor config for ``SparkSubmitOperator(conf=...)``."""
    if preset not in SPARK_CONFIGS:
        raise KeyError(
            f"Unknown Spark preset {preset!r}; known: {sorted(SPARK_CONFIGS)}"
        )
    preset_conf = {
        k: v for k, v in SPARK_CONFIGS[preset].items() if k != "spark.master"
    }
    return {
        **preset_conf,
        "spark.driver.host": os.environ.get("SPARK_DRIVER_HOST", "airflow-worker"),
        "spark.driver.bindAddress": "0.0.0.0",
        "spark.pyspark.driver.python": spark_driver_python(),
        "spark.pyspark.python": spark_executor_python(),
        # MinIO rejects PUT of 0-byte objects (IncompleteBody). Disabling fake
        # directory markers in S3A avoids createEmptyObject calls during task commit.
        "spark.hadoop.fs.s3a.create.performance": "true",
        # Use the S3A committer to avoid rename-based commit (incompatible with MinIO).
        "spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a": (
            "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory"
        ),
        "spark.hadoop.fs.s3a.committer.name": "magic",
        "spark.hadoop.fs.s3a.committer.magic.enabled": "true",
    }


def _driver_pythonpath() -> str:
    """PYTHONPATH for the Spark driver process.

    Prod: just SPARK_HOME/python + py4j — the driver imports the pinned genpm wheel from its venv.
    Dev (GENPM_PY_FILES set): prepend GENPM_GENERATOR_ROOT so the live-mounted source shadows the
    wheel and driver-side changes are picked up without an image rebuild.
    """
    base = spark_pythonpath()
    if GENPM_PY_FILES:
        return f"{GENPM_GENERATOR_ROOT}:{base}"
    return base


def infra_env_vars(*, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Infrastructure env vars forwarded to the Spark job — no dag_run business parameters."""
    env_vars = {
        # Tells SparkDataManager it's a cluster submit so it leaves spark-submit's --master alone
        # (otherwise it would override it with local[N] inside the Airflow worker).
        "GENPM_SPARK_SUBMIT": "1",
        "GENPM_PYSPARK_PYTHON": spark_driver_python(),
        "GENPM_SPARK_EXECUTOR_PYTHON": spark_executor_python(),
        "S3_URL": os.environ.get("S3_URL", "http://minio:9000"),
        "S3_BUCKET": os.environ.get("S3_BUCKET", "datasets"),
        "AWS_ACCESS_KEY_ID": _required_env(
            "AWS_ACCESS_KEY_ID",
            hint="MinIO/S3 access key is required for the Spark job.",
        ),
        "AWS_SECRET_ACCESS_KEY": _required_env(
            "AWS_SECRET_ACCESS_KEY",
            hint="MinIO/S3 secret key is required for the Spark job.",
        ),
        "PYSPARK_DRIVER_PYTHON": spark_driver_python(),
        "PYTHONPATH": _driver_pythonpath(),
    }
    if extra:
        env_vars.update(extra)
    return env_vars
