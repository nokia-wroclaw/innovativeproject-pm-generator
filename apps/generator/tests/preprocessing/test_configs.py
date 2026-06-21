"""Unit tests for PreprocessingConfig and defaults — no Spark required."""

import pytest

from genpm.preprocessing.configs import PreprocessingConfig, _derive_intermediate_path
from genpm.preprocessing.defaults import DEFAULT_PREPROCESSING_DAG_ARGS, finalize_dag_args

# ---------------------------------------------------------------------------
# _derive_intermediate_path
# ---------------------------------------------------------------------------


def test_derive_intermediate_path_with_final_suffix():
    result = _derive_intermediate_path("s3a://bucket/runs/run1/final")
    assert result == "s3a://bucket/runs/run1/intermediate"


def test_derive_intermediate_path_without_final_suffix():
    result = _derive_intermediate_path("s3a://bucket/runs/run1")
    assert result == "s3a://bucket/runs/run1/intermediate"


def test_derive_intermediate_path_trailing_slash():
    result = _derive_intermediate_path("s3a://bucket/runs/run1/final/")
    assert result == "s3a://bucket/runs/run1/intermediate"


# ---------------------------------------------------------------------------
# finalize_dag_args
# ---------------------------------------------------------------------------

VALID_CONF = {
    "s3_key": "data/raw/pm.parquet",
    "dag_args": {
        "kpi_definitions_raw_path": "data/kpis/kpis.parquet",
        "simple_reports_raw_path": "data/reports/reports.parquet",
        "output_path_prefix": "data/preprocessed/run1/final",
    },
}


def test_finalize_dag_args_returns_dict_with_all_defaults():
    result = finalize_dag_args(conf=VALID_CONF)
    for key in DEFAULT_PREPROCESSING_DAG_ARGS:
        assert key in result, f"key '{key}' missing from finalize_dag_args output"


def test_finalize_dag_args_missing_s3_key_raises():
    with pytest.raises(ValueError, match="s3_key"):
        finalize_dag_args(conf={"dag_args": {}})


def test_finalize_dag_args_missing_required_paths_raises():
    with pytest.raises(ValueError, match="dag_args missing required keys"):
        finalize_dag_args(
            conf={
                "s3_key": "data/raw/pm.parquet",
                "dag_args": {
                    "output_path_prefix": "data/preprocessed/run1/final",
                    # kpi_definitions_raw_path and simple_reports_raw_path both absent
                },
            }
        )


def test_finalize_dag_args_output_prefix_auto_resolved_when_absent():
    conf = {
        "s3_key": "data/raw/pm.parquet",
        "genpm_run_id": "run5",
        "dag_args": {
            "kpi_definitions_raw_path": "data/kpis/kpis.parquet",
            "simple_reports_raw_path": "data/reports/reports.parquet",
            # output_path_prefix deliberately omitted
        },
    }
    result = finalize_dag_args(conf=conf)
    assert result["output_path_prefix"], "output_path_prefix should be auto-resolved"
    assert "run5" in result["output_path_prefix"]


def test_finalize_dag_args_user_values_override_defaults():
    conf = {
        "s3_key": "data/raw/pm.parquet",
        "dag_args": {
            "kpi_definitions_raw_path": "data/kpis/kpis.parquet",
            "simple_reports_raw_path": "data/reports/reports.parquet",
            "output_path_prefix": "data/preprocessed/run1/final",
            "window_width_hours": 336,
            "kpi_min_global_density": 0.9,
        },
    }
    result = finalize_dag_args(conf=conf)
    assert result["window_width_hours"] == 336
    assert result["kpi_min_global_density"] == 0.9


# ---------------------------------------------------------------------------
# PreprocessingConfig.from_conf
# ---------------------------------------------------------------------------

FULL_CONF = {
    "s3_key": "data/raw/pm.parquet",
    "dag_args": {
        "kpi_definitions_raw_path": "data/kpis/kpis.parquet",
        "simple_reports_raw_path": "data/reports/reports.parquet",
        "output_path_prefix": "data/preprocessed/run1/final",
        "kpi_min_global_density": "0.6",
        "kpi_global_min_frac_cells_passing": "0.7",
        "min_imputable_gap_frac": "0.85",
        "kpi_min_std_val": "0.02",
        "max_zero_frac": "0.9",
        "window_width_hours": "168",
        "stride_hours": "24",
        "max_gap_hours": "24",
        "min_joint_windows_abs": None,
        "impute": True,
    },
}


def test_from_conf_produces_correct_types():
    cfg = PreprocessingConfig.from_conf(FULL_CONF, bucket="my-bucket")
    assert isinstance(cfg.kpi_min_global_density, float)
    assert isinstance(cfg.window_width_hours, int)
    assert isinstance(cfg.impute, bool)
    assert isinstance(cfg.min_joint_windows_abs, type(None))


def test_from_conf_s3a_paths_prefixed():
    cfg = PreprocessingConfig.from_conf(FULL_CONF, bucket="my-bucket")
    assert cfg.pm_data_raw_path.startswith("s3a://my-bucket/")
    assert cfg.kpi_definitions_raw_path.startswith("s3a://my-bucket/")
    assert cfg.output_path_prefix.startswith("s3a://my-bucket/")


def test_from_conf_intermediate_path_derived():
    cfg = PreprocessingConfig.from_conf(FULL_CONF, bucket="my-bucket")
    assert cfg.intermediate_path.endswith("/intermediate")
    assert "final" not in cfg.intermediate_path


def test_from_conf_min_joint_windows_abs_none_stays_none():
    cfg = PreprocessingConfig.from_conf(FULL_CONF, bucket="my-bucket")
    assert cfg.min_joint_windows_abs is None


def test_from_conf_min_joint_windows_abs_parsed_as_int():
    conf = {**FULL_CONF, "dag_args": {**FULL_CONF["dag_args"], "min_joint_windows_abs": "50"}}
    cfg = PreprocessingConfig.from_conf(conf, bucket="my-bucket")
    assert cfg.min_joint_windows_abs == 50
