from dataclasses import dataclass, field


@dataclass
class DataSimilarityConfig:
    # Path prefix - output (required)
    output_path_prefix: str

    # RAW paths - input (optional if fake flags are set)
    real_data_path: str | None = None
    synth_data_path: str | None = None

    # KPI columns to validate
    single_kpi_cols: list[str] = field(default_factory=list)
    multi_kpi_cols: list[str] = field(default_factory=list)

    # Time column name
    ts_col: str = "ts"

    # Single-KPI metric parameters
    acf_max_lag: int = 24 * 8
    ls_min_period_h: float = 2.0
    ls_max_period_h: float = 24 * 14
    ls_n_freq: int = 2000
    kde_n_grid: int = 300

    # Multi-KPI metric parameters
    n_projections: int = 200

    # Fake real series params
    fake_real_start_date: str = "2025-01-01"
    fake_real_n_days: int = 90
    fake_real_noise_std: float = 0.3
    fake_real_seed: int = 10

    # Fake synth series params
    fake_synth_start_date: str = "2025-04-01"
    fake_synth_n_days: int = 14
    fake_synth_noise_std: float = 1.0
    fake_synth_seed: int = 20
