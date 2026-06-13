"""Driver bootstrap for Spark jobs launched via Airflow SparkSubmit."""

from __future__ import annotations

import glob
import os
import sys


def spark_home() -> str:
    return os.environ.get("SPARK_HOME", "/opt/spark")


def spark_pythonpath() -> str:
    """`$SPARK_HOME/python` + the bundled py4j zip, discovered by glob (no pinned version)."""
    home = spark_home()
    spark_python = f"{home}/python"
    matches = sorted(glob.glob(f"{spark_python}/lib/py4j-*-src.zip"))
    if matches:
        return f"{spark_python}:{matches[-1]}"
    return spark_python


def ensure_spark_pythonpath() -> None:
    """PySpark lives under SPARK_HOME, not in genpm-venv."""
    spark_python = f"{spark_home()}/python"
    current = os.environ.get("PYTHONPATH", "")
    if spark_python not in current:
        prefix = spark_pythonpath()
        os.environ["PYTHONPATH"] = f"{prefix}:{current}" if current else prefix


def ensure_pyspark_python() -> None:
    """Re-exec driver with Airflow Python when Spark-submit picked a stale interpreter."""
    target = os.environ.get("GENPM_PYSPARK_PYTHON", "/opt/airflow/genpm-venv/bin/python")
    if not os.path.isfile(target):
        return
    if os.environ.get("GENPM_PY_REEXEC") == "1":
        return
    if os.path.realpath(sys.executable) == os.path.realpath(target):
        return
    os.environ["GENPM_PY_REEXEC"] = "1"
    os.environ["PYSPARK_DRIVER_PYTHON"] = target
    os.execv(target, [target, *sys.argv])


def bootstrap_spark_submit_driver() -> None:
    ensure_pyspark_python()
    ensure_spark_pythonpath()
