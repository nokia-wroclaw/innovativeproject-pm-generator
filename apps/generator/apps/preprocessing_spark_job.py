import os
import sys


def _ensure_spark_pythonpath() -> None:
    """PySpark lives under SPARK_HOME, not in genpm-venv."""
    spark_home = os.environ.get("SPARK_HOME", "/opt/spark")
    py4j_zip = f"{spark_home}/python/lib/py4j-0.10.9.7-src.zip"
    spark_python = f"{spark_home}/python"
    prefix = f"{spark_python}:{py4j_zip}"
    current = os.environ.get("PYTHONPATH", "")
    if spark_python not in current:
        os.environ["PYTHONPATH"] = f"{prefix}:{current}" if current else prefix


def _ensure_pyspark_python() -> None:
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


_ensure_pyspark_python()
_ensure_spark_pythonpath()

from genpm.preprocessing.configs import PreprocessingConfig  # noqa: E402
from genpm.preprocessing.run import run_preprocessing  # noqa: E402
from genpm.utils.spark_session import (  # noqa: E402
    build_cluster_spark_session,
    minio_spark_conf,
)
from genpm.utils.utils import SparkDataManager  # noqa: E402


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_optional_int(name: str) -> int | None:
    raw = _env(name)
    if not raw or raw.lower() == "none":
        return None
    return int(raw)


def _s3a_path(key_or_path: str) -> str:
    value = (key_or_path or "").strip()
    if not value:
        return ""
    if value.startswith("s3a://"):
        return value
    bucket = _env("S3_BUCKET", "datasets")
    return f"s3a://{bucket}/{value.lstrip('/')}"


def _build_config() -> PreprocessingConfig:
    pm_key = _env("GENPM_S3_KEY")
    kpi_defs_key = _env("GENPM_KPI_DEFINITIONS_S3_KEY")
    simple_reports_key = _env("GENPM_SIMPLE_REPORTS_S3_KEY")
    output_prefix = _env("GENPM_OUTPUT_PREFIX")

    missing = [
        label
        for label, value in (
            ("GENPM_S3_KEY", pm_key),
            ("GENPM_KPI_DEFINITIONS_S3_KEY", kpi_defs_key),
            ("GENPM_SIMPLE_REPORTS_S3_KEY", simple_reports_key),
            ("GENPM_OUTPUT_PREFIX", output_prefix),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required preprocessing env vars: "
            + ", ".join(missing)
            + ". Pass them via dag_run.conf / dag_args."
        )

    return PreprocessingConfig(
        pm_data_raw_path=_s3a_path(pm_key),
        kpi_definitions_raw_path=_s3a_path(kpi_defs_key),
        simple_reports_raw_path=_s3a_path(simple_reports_key),
        output_path_prefix=_s3a_path(output_prefix),
        kpi_min_global_density=_env_float("GENPM_KPI_MIN_GLOBAL_DENSITY", 0.5),
        min_frac_contributing_cells=_env_float("GENPM_KPI_GLOBAL_MIN_FRAC_CELLS_PASSING", 0.5),
        min_imputable_gap_frac=_env_float("GENPM_MIN_IMPUTABLE_GAP_FRAC", 0.8),
        kpi_min_std_val=_env_float("GENPM_KPI_MIN_STD_VAL", 0.01),
        max_zero_frac=_env_float("GENPM_MAX_ZERO_FRAC", 0.95),
        window_width_hours=_env_int("GENPM_WINDOW_WIDTH_HOURS", 168),
        stride_hours=_env_int("GENPM_STRIDE_HOURS", 24),
        max_gap_hours=_env_int("GENPM_MAX_GAP_HOURS", 24),
        min_joint_windows_abs=_env_optional_int("GENPM_MIN_JOINT_WINDOWS_ABS"),
        impute=_env("GENPM_IMPUTE", "true").lower() in {"1", "true", "yes"},
    )


def main() -> None:
    dataset_id = _env("GENPM_DATASET_ID")
    print(f"Preprocessing Spark job starting (dataset_id={dataset_id or 'n/a'})")

    cfg = _build_config()
    print(f"PM data: {cfg.pm_data_raw_path}")
    print(f"KPI definitions: {cfg.kpi_definitions_raw_path}")
    print(f"Simple reports: {cfg.simple_reports_raw_path}")
    print(f"Output prefix: {cfg.output_path_prefix}")

    spark = build_cluster_spark_session(
        "PreprocessingSparkJob",
        extra_conf=minio_spark_conf(),
    )
    print(f"Spark version: {spark.version}")

    sdm = SparkDataManager(spark=spark)
    run_preprocessing(sdm, cfg)

    spark.stop()
    print("Preprocessing job finished successfully")


if __name__ == "__main__":
    main()
