"""Contract test: backend dag_args models must stay in sync with the genpm job schemas.

The backend (FastAPI container) and genpm (Spark container) are separate packages that cannot
import each other at runtime, so each keeps its own copy of the dag_args contracts.
This test fails loudly if they drift — update both sides (and this test) together.

Run with `uv run pytest apps/backend/tests/test_spark_jobs_contract.py`, or directly:
`uv run python apps/backend/tests/test_spark_jobs_contract.py`.
"""

from __future__ import annotations

from genpm.modelling.defaults import (
    DEFAULT_GENERATE_DAG_ARGS as GENPM_GENERATE_DEFAULTS,
)
from genpm.modelling.defaults import REQUIRED_GENERATE_KEYS
from genpm.preprocessing.defaults import (
    DEFAULT_PREPROCESSING_DAG_ARGS as GENPM_PREPROCESSING_DEFAULTS,
)
from genpm.preprocessing.defaults import REQUIRED_USER_PATH_KEYS

from app.models.spark_jobs import (
    DEFAULT_GENERATE_DAG_ARGS as BACKEND_GENERATE_DEFAULTS,
)
from app.models.spark_jobs import (
    DEFAULT_PREPROCESSING_DAG_ARGS as BACKEND_PREPROCESSING_DEFAULTS,
)

# ── Preprocessing contract ────────────────────────────────────────────────────


def test_preprocessing_dag_args_keys_match() -> None:
    assert set(BACKEND_PREPROCESSING_DEFAULTS) == set(GENPM_PREPROCESSING_DEFAULTS), (
        "Backend and genpm preprocessing dag_args keys drifted: "
        f"backend-only={set(BACKEND_PREPROCESSING_DEFAULTS) - set(GENPM_PREPROCESSING_DEFAULTS)}, "
        f"genpm-only={set(GENPM_PREPROCESSING_DEFAULTS) - set(BACKEND_PREPROCESSING_DEFAULTS)}"
    )


def test_preprocessing_dag_args_values_match() -> None:
    assert (
        BACKEND_PREPROCESSING_DEFAULTS == GENPM_PREPROCESSING_DEFAULTS
    ), "Backend and genpm preprocessing dag_args default values drifted."


def test_preprocessing_required_keys_are_known() -> None:
    assert set(REQUIRED_USER_PATH_KEYS) <= set(GENPM_PREPROCESSING_DEFAULTS)


# ── Generate contract ─────────────────────────────────────────────────────────


def test_generate_dag_args_keys_match() -> None:
    assert set(BACKEND_GENERATE_DEFAULTS) == set(GENPM_GENERATE_DEFAULTS), (
        "Backend and genpm generate dag_args keys drifted: "
        f"backend-only={set(BACKEND_GENERATE_DEFAULTS) - set(GENPM_GENERATE_DEFAULTS)}, "
        f"genpm-only={set(GENPM_GENERATE_DEFAULTS) - set(BACKEND_GENERATE_DEFAULTS)}"
    )


def test_generate_dag_args_values_match() -> None:
    assert (
        BACKEND_GENERATE_DEFAULTS == GENPM_GENERATE_DEFAULTS
    ), "Backend and genpm generate dag_args default values drifted."


def test_generate_required_keys_are_known() -> None:
    assert set(REQUIRED_GENERATE_KEYS) <= set(GENPM_GENERATE_DEFAULTS)


if __name__ == "__main__":
    test_preprocessing_dag_args_keys_match()
    test_preprocessing_dag_args_values_match()
    test_preprocessing_required_keys_are_known()
    test_generate_dag_args_keys_match()
    test_generate_dag_args_values_match()
    test_generate_required_keys_are_known()
    print("all contracts OK")
