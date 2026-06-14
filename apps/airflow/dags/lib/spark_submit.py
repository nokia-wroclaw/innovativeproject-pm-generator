"""Factory helpers for GenPM Spark Airflow tasks."""

from __future__ import annotations

import os
import shutil
from datetime import timedelta
from typing import Any

from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from lib.spark_config import (
    GENPM_GENERATOR_ROOT,
    GENPM_PY_FILES,
    SPARK_CONN_ID,
    infra_env_vars,
    spark_submit_conf,
)


def rebuild_py_files() -> str | None:
    """Dev-only: zip the *live* genpm package into GENPM_PY_FILES at submit time.

    In prod, GENPM_PY_FILES is unset and executors use the pinned genpm wheel installed in the Spark
    image — this returns None and no --py-files is attached. In dev (compose sets GENPM_PY_FILES and
    mounts the source), we rebuild the zip from the live mount so executors run the edited code
    without an image rebuild; the zip is prepended to the executor sys.path, shadowing the wheel.
    """
    if not GENPM_PY_FILES:
        return None
    package_dir = os.path.join(GENPM_GENERATOR_ROOT, "genpm")
    if not os.path.isdir(package_dir):
        raise RuntimeError(
            f"genpm package not found at {package_dir}; cannot build py_files. "
            "Check GENPM_GENERATOR_ROOT and the generator volume mount."
        )
    base, _ = os.path.splitext(GENPM_PY_FILES)
    os.makedirs(os.path.dirname(GENPM_PY_FILES) or ".", exist_ok=True)
    # make_archive zips the `genpm` dir (root_dir=generator root, base_dir=genpm).
    shutil.make_archive(base, "zip", root_dir=GENPM_GENERATOR_ROOT, base_dir="genpm")
    print(f"Rebuilt py_files: {GENPM_PY_FILES}")
    return GENPM_PY_FILES


def genpm_spark_submit(
    *,
    task_id: str,
    application: str,
    application_args: str | list[str],
    app_name: str,
    spark_preset: str | None = None,
    env_vars_extra: dict[str, str] | None = None,
    execution_timeout: timedelta | None = None,
    verbose: bool = True,
) -> SparkSubmitOperator:
    kwargs: dict[str, Any] = {
        "task_id": task_id,
        "conn_id": SPARK_CONN_ID,
        "application": application,
        "application_args": application_args,
        "name": app_name,
        "verbose": verbose,
        "execution_timeout": execution_timeout,
        "conf": spark_submit_conf(spark_preset)
        if spark_preset
        else spark_submit_conf(),
        "env_vars": infra_env_vars(extra=env_vars_extra),
    }
    if GENPM_PY_FILES:
        # Path is static; the upstream prepare task rebuilds the zip before this task runs.
        kwargs["py_files"] = GENPM_PY_FILES
    return SparkSubmitOperator(**kwargs)
