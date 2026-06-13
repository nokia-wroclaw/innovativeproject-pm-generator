"""Contract test: backend dag_args model must stay in sync with the genpm job schema.

The backend (FastAPI container) and genpm (Spark container) are separate packages that cannot
import each other at runtime, so each keeps its own copy of the preprocessing dag_args contract.
This test fails loudly if they drift — update both sides (and this test) together.

Run with `uv run pytest apps/backend/tests/test_spark_jobs_contract.py`, or directly:
`uv run python apps/backend/tests/test_spark_jobs_contract.py`.
"""

from __future__ import annotations

from genpm.preprocessing.defaults import (
    DEFAULT_PREPROCESSING_DAG_ARGS as GENPM_DEFAULTS,
)
from genpm.preprocessing.defaults import (
    REQUIRED_USER_PATH_KEYS,
)

from app.models.spark_jobs import DEFAULT_PREPROCESSING_DAG_ARGS as BACKEND_DEFAULTS


def test_default_dag_args_keys_match() -> None:
    assert set(BACKEND_DEFAULTS) == set(GENPM_DEFAULTS), (
        "Backend and genpm preprocessing dag_args keys drifted: "
        f"backend-only={set(BACKEND_DEFAULTS) - set(GENPM_DEFAULTS)}, "
        f"genpm-only={set(GENPM_DEFAULTS) - set(BACKEND_DEFAULTS)}"
    )


def test_default_dag_args_values_match() -> None:
    assert BACKEND_DEFAULTS == GENPM_DEFAULTS, "Backend and genpm dag_args default values drifted."


def test_required_keys_are_known() -> None:
    assert set(REQUIRED_USER_PATH_KEYS) <= set(GENPM_DEFAULTS)


if __name__ == "__main__":
    test_default_dag_args_keys_match()
    test_default_dag_args_values_match()
    test_required_keys_are_known()
    print("contract OK")
