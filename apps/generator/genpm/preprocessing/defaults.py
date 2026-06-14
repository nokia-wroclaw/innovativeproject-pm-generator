"""Single source of truth for the preprocessing ``dag_run.conf.dag_args`` contract.

The backend keeps a parallel Pydantic model (``app.models.spark_jobs.PreprocessingDagArgs``) for
API validation because it cannot import genpm (separate container). A contract test asserts the two
stay in sync — keep this dict authoritative and update the test if you add a parameter here.
"""

from __future__ import annotations

from typing import Any

from genpm.raw_vis.s3_layout import dataset_visualization_prefix

# Canonical defaults for the dag_args namespace exchanged over the Airflow REST API.
DEFAULT_PREPROCESSING_DAG_ARGS: dict[str, Any] = {
    "kpi_definitions_raw_path": "",
    "simple_reports_raw_path": "",
    "output_path_prefix": "",
    "kpi_min_global_density": 0.5,
    "kpi_global_min_frac_cells_passing": 0.8,
    "min_imputable_gap_frac": 0.8,
    "kpi_min_std_val": 0.01,
    "max_zero_frac": 0.95,
    "window_width_hours": 168,
    "stride_hours": 24,
    "max_gap_hours": 24,
    "min_joint_windows_abs": None,
    "impute": True,
}

# Paths a caller must provide; output_path_prefix is auto-resolved when omitted.
REQUIRED_USER_PATH_KEYS = (
    "kpi_definitions_raw_path",
    "simple_reports_raw_path",
)


def resolve_output_path_prefix(genpm_run_id: str, raw_s3_key: str) -> str:
    """S3 key prefix (no bucket / no s3a://) for final preprocessed artifacts."""
    base = dataset_visualization_prefix(raw_s3_key).strip("/")
    run_id = (genpm_run_id or "").strip("/")
    return f"{base}/preprocessed/{run_id}/final"


def finalize_dag_args(
    *,
    conf: dict[str, Any],
    genpm_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate required keys, apply defaults and auto-resolve ``output_path_prefix``.

    Idempotent: when the backend already filled ``output_path_prefix`` this is a no-op for it.
    Raises ``ValueError`` with a clear message when a required input is missing.
    """
    s3_key = str(conf.get("s3_key") or "").strip()
    if not s3_key:
        raise ValueError("dag_run.conf missing required key: s3_key")

    dag_args: dict[str, Any] = {**DEFAULT_PREPROCESSING_DAG_ARGS, **(conf.get("dag_args") or {})}

    missing = [k for k in REQUIRED_USER_PATH_KEYS if not str(dag_args.get(k) or "").strip()]
    if missing:
        raise ValueError("dag_run.conf.dag_args missing required keys: " + ", ".join(missing))

    if not str(dag_args.get("output_path_prefix") or "").strip():
        run_id = genpm_run_id or conf.get("genpm_run_id") or ""
        dag_args["output_path_prefix"] = resolve_output_path_prefix(str(run_id), s3_key)

    return dag_args
