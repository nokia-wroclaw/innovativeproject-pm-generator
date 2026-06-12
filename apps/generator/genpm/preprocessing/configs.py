from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    # RAW paths - input
    pm_data_raw_path: str
    kpi_definitions_raw_path: str
    simple_reports_raw_path: str
    # Path prefix - output paths
    output_path_prefix: str
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
