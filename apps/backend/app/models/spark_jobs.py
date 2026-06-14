"""Pydantic schemas for Airflow dag_run.conf passed to Spark DAGs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class PreprocessingConfigError(ValueError):
    """Invalid or incomplete preprocessing DAG configuration."""


class PreprocessingDagArgs(BaseModel):
    """dag_run.conf.dag_args for preprocessing_pipeline.

    Single source of truth on the backend side; kept in sync with
    ``genpm.preprocessing.defaults.DEFAULT_PREPROCESSING_DAG_ARGS`` by a contract test. Unknown
    user keys are dropped (``extra="ignore"``) so we never forward arbitrary fields to Airflow.
    """

    model_config = ConfigDict(extra="ignore")

    kpi_definitions_raw_path: str = ""
    simple_reports_raw_path: str = ""
    output_path_prefix: str = ""
    kpi_min_global_density: float = 0.5
    kpi_global_min_frac_cells_passing: float = 0.8
    min_imputable_gap_frac: float = 0.8
    kpi_min_std_val: float = 0.01
    max_zero_frac: float = 0.95
    window_width_hours: int = 168
    stride_hours: int = 24
    max_gap_hours: int = 24
    min_joint_windows_abs: int | None = None
    impute: bool = True

    @classmethod
    def from_user(cls, user_args: dict[str, Any] | None) -> PreprocessingDagArgs:
        return cls.model_validate(user_args or {})

    def require_paths(self) -> None:
        missing = [
            key
            for key, value in (
                ("kpi_definitions_raw_path", self.kpi_definitions_raw_path),
                ("simple_reports_raw_path", self.simple_reports_raw_path),
            )
            if not str(value or "").strip()
        ]
        if missing:
            raise PreprocessingConfigError(
                "Missing required preprocessing dag_args: "
                + ", ".join(missing)
                + ". Provide KPI definitions and simple reports S3 keys."
            )

    def to_conf_dict(self) -> dict[str, Any]:
        return self.model_dump()


# Derived from the model above — no hand-maintained second copy.
DEFAULT_PREPROCESSING_DAG_ARGS: dict[str, Any] = PreprocessingDagArgs().model_dump()


class PreprocessingDagConf(BaseModel):
    """Full dag_run.conf for preprocessing_pipeline / modeling preprocessing."""

    model_config = ConfigDict(extra="ignore")

    genpm_run_id: str
    dataset_id: int
    s3_key: str
    dag_args: PreprocessingDagArgs
    process_type: str = "preprocessing_feature_engineering"
    file_name: str | None = None
    dataset_name: str | None = None

    def to_airflow_conf(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class VisualizationDagConf(BaseModel):
    """dag_run.conf for dataset_visualization_spark."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: int
    s3_key: str
    file_name: str | None = None

    def to_airflow_conf(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
