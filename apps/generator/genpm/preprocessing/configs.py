from dataclasses import dataclass

from genpm.utils.consts import MAX_IMPUTABLE_GAP


@dataclass
class PreprocessingConfig:
    # RAW paths - input
    pm_data_raw_path: str
    kpi_definitions_raw_path: str
    simple_reports_raw_path: str
    # Path prefix - output paths
    output_path_prefix: str
    # KPI coverage - filter thresholds
    # KPI global density threshold
    kpi_min_global_density: float
    kpi_global_min_frac_cells_passing: float
    # KPI in Window coverage threshold
    kpi_window_coverage_frac: float
    # Max gap filtering
    min_imputable_gap_frac: float
    # Stale kpis filtering
    kpi_min_std_val: float
    max_zero_frac: float
    # prefilter kpis
    # SKIP PREFILTER KPIS
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
    max_gap_hours: int = MAX_IMPUTABLE_GAP

    # Greedy approach - minimal joint windows available in data
    min_joint_windows_abs: int | None = None

    # Impute bool - TODO: include this option
    impute: bool = True
