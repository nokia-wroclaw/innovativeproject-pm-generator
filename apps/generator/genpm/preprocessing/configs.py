from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genpm.preprocessing.defaults import finalize_dag_args
from genpm.utils.s3_paths import s3a_path


def _derive_intermediate_path(output_path_prefix: str) -> str:
    """`<base>/intermediate` sibling of the `<base>/final` output prefix."""
    prefix = output_path_prefix.rstrip("/")
    base = prefix[: -len("/final")] if prefix.endswith("/final") else prefix
    return f"{base}/intermediate"


@dataclass
class PreprocessingConfig:
    # RAW paths - input
    pm_data_raw_path: str
    kpi_definitions_raw_path: str
    simple_reports_raw_path: str
    # Output paths
    output_path_prefix: str
    intermediate_path: str
    # KPI coverage - filter thresholds
    # Minimum non-null fraction over a series' own active range (Stage 0 density check)
    kpi_min_global_density: float
    # Max gap filtering
    min_imputable_gap_frac: float
    # Stale kpis filtering
    kpi_min_std_val: float
    max_zero_frac: float
    # Stage 5 prefilter thresholds (honest cuts — no fictional theoretical max)
    min_frac_contributing_cells: float = 0.50
    min_total_windows: int = 1
    # Training data windows
    # NOTE: As training will be done in windows of K x N x F
    # (
    #   K - kpi_count
    #   N - Window length
    #   F - Feature Count
    # )
    # Windows must be divided into a combined (most covered) subset with available data per
    # every timestamp in that window
    window_width_hours: int = 168
    stride_hours: int = 24

    # Max gap - threshold for filtering the biggest windows
    max_gap_hours: int = 24

    # Greedy approach - minimal joint windows available in data
    min_joint_windows_abs: int | None = None
    forced_kpis: list | None = None
    # TODO: introcude preference for kpi selection in greedy approach
    # Floor relaxation: effective floor = min(elbow_floor, seed_coverage × frac).
    # 1.0 = use elbow as-is.  Lower values trade joint-window count for more KPIs.
    # Start at 0.3–0.5 when only a handful of KPIs are selected.
    # kpi_window_preference: float = 0.5

    # Impute bool - TODO: include this option
    impute: bool = True
    # Verbose
    verbose: bool = False

    @classmethod
    def from_conf(cls, conf: dict[str, Any], *, bucket: str) -> PreprocessingConfig:
        """Build the internal job config from a finalized `dag_run.conf`.

        Resolves bucket-relative S3 keys to `s3a://` URIs and maps the dag_args namespace onto the
        dataclass fields. Defaults + required-key validation live in
        `genpm.preprocessing.defaults.finalize_dag_args`.
        """

        dag_args = finalize_dag_args(conf=conf)
        output_prefix = s3a_path(bucket, str(dag_args["output_path_prefix"]))

        min_joint = dag_args.get("min_joint_windows_abs")
        min_joint_val = int(min_joint) if min_joint not in (None, "", "None", "none") else None

        return cls(
            pm_data_raw_path=s3a_path(bucket, str(conf["s3_key"])),
            kpi_definitions_raw_path=s3a_path(bucket, str(dag_args["kpi_definitions_raw_path"])),
            simple_reports_raw_path=s3a_path(bucket, str(dag_args["simple_reports_raw_path"])),
            intermediate_path=_derive_intermediate_path(output_prefix),
            output_path_prefix=output_prefix,
            kpi_min_global_density=float(dag_args["kpi_min_global_density"]),
            min_frac_contributing_cells=float(dag_args["kpi_global_min_frac_cells_passing"]),
            min_imputable_gap_frac=float(dag_args["min_imputable_gap_frac"]),
            kpi_min_std_val=float(dag_args["kpi_min_std_val"]),
            max_zero_frac=float(dag_args["max_zero_frac"]),
            window_width_hours=int(dag_args["window_width_hours"]),
            stride_hours=int(dag_args["stride_hours"]),
            max_gap_hours=int(dag_args["max_gap_hours"]),
            min_joint_windows_abs=min_joint_val,
            impute=bool(dag_args["impute"]),
        )
