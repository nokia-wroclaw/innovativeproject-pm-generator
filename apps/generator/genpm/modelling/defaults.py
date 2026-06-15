"""Single source of truth for the generate ``dag_run.conf.dag_args`` contract.

The backend keeps a parallel Pydantic model (``app.models.spark_jobs.GenerateDagArgs``) for API
validation. A contract test asserts the two stay in sync — keep this dict authoritative and update
the test when adding a parameter here.
"""

from __future__ import annotations

from typing import Any

DEFAULT_GENERATE_DAG_ARGS: dict[str, Any] = {
    "cell_id": "",
    "anchor_date": "",
    "n_weeks": 4,
    "holiday": 0,
    "seed": 42,
    "batch_size": 64,
    "kpi_list": [],
    "output_path_prefix": "",
}

REQUIRED_GENERATE_KEYS = ("anchor_date", "n_weeks")


def finalize_generate_dag_args(
    *,
    conf: dict[str, Any],
    genpm_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate required keys, apply defaults, auto-resolve ``output_path_prefix``.

    Idempotent: when the backend already filled ``output_path_prefix`` this is a no-op for it.
    Raises ``ValueError`` when a required field is missing or empty.
    """
    dag_args: dict[str, Any] = {**DEFAULT_GENERATE_DAG_ARGS, **(conf.get("dag_args") or {})}

    missing = [k for k in REQUIRED_GENERATE_KEYS if not str(dag_args.get(k) or "").strip()]
    if missing:
        raise ValueError("dag_run.conf.dag_args missing required keys: " + ", ".join(missing))

    if not str(dag_args.get("output_path_prefix") or "").strip():
        run_id = genpm_run_id or conf.get("genpm_run_id") or ""
        dag_args["output_path_prefix"] = f"generated/{run_id.strip('/')}"

    return dag_args
