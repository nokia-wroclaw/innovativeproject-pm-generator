from dataclasses import dataclass, field


@dataclass
class DataSimilarityConfig:
    # Path prefix - output (required)
    output_path_prefix: str

    # RAW paths - input (required)
    real_data_path: str
    synth_data_path: str

    # KPI columns to validate
    single_kpi_cols: list[str] = field(default_factory=list)
    multi_kpi_cols: list[str] = field(default_factory=list)

    # Time column name used internally by metric functions
    ts_col: str = "ts"

    # Raw ts column names in the source parquets (if different from ts_col, renamed on load)
    real_ts_col: str | None = None
    synth_ts_col: str | None = None

    # Cell config filtering: combine these columns (in order) from the real data into
    # config_id and keep only rows whose config_id matches "|".join(cell_configs).
    # Both must be set together or both left as None.
    cell_config_cols: list[str] | None = None  # ordered [CELL] column names in real data
    cell_configs: list[str] | None = None  # target config values in the same column order

    # Single-KPI metric parameters
    acf_max_lag: int = 24 * 8
    ls_min_period_h: float = 2.0
    ls_max_period_h: float = 24 * 14
    ls_n_freq: int = 2000
    kde_n_grid: int = 300

    # Multi-KPI metric parameters
    n_projections: int = 200

    # Output
    save_summary_json: bool = True
