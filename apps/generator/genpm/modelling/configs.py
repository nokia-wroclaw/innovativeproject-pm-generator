from dataclasses import dataclass, field


@dataclass
class GenerateConfig:
    # Paths
    run_dir_path: str
    weights_path: str
    output_path: str
    # Generation parameters
    cell_id: str
    anchor_date: str
    n_weeks: int
    kpi_list: list
    holiday: int = 0
    batch_size: int = 64
    seed: int = 42
    # Conditioning: explicit config values; if None, looked up by cell_id from the
    # training cell_config_map.
    cell_configs: list | None = None
    # Model architecture (must match the trained checkpoint)
    seq_len: int = 168
    n_dim: int = 235
    global_latent_dim: int = 64
    local_latent_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    use_attention: bool = True
    n_heads: int = 4
    free_bits_global: float = 0.002
    free_bits_local: float = 0.0
    output_activation: str = "sigmoid"


@dataclass
class TrainConfig:
    # Paths
    training_data_path: str
    run_dir_path: str
    weights_path: str
    # Model architecture (v6)
    global_latent_dim: int = 64
    local_latent_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    use_attention: bool = True
    n_heads: int = 4
    beta: float = 0.0
    learning_rate: float = 3e-4
    free_bits_global: float = 0.002
    free_bits_local: float = 0.0
    output_activation: str = "sigmoid"
    # Training schedule
    epochs: int = 300
    batch_size: int = 64
    target_beta: float = 2e-4
    use_cyclical_kl: bool = True
    cycle_epochs: int = 40
    n_cycles: int = 6
    cycle_ratio: float = 0.5
    anneal_epochs: int = 150
    lr_patience: int = 20
    early_stop_patience: int = 60
    collapse_monitor: bool = True
    drop_constant_kpis: bool = True


@dataclass
class ValidateConfig:
    # Paths
    run_dir_path: str
    weights_path: str
    real_data_path: str
    # Comparison parameters
    cell_id: str
    date_start: str
    date_end: str
    kpi_list: list = field(default_factory=list)
    seed: int = 42
    # Model architecture (must match trained checkpoint)
    seq_len: int = 168
    n_dim: int = 235
    global_latent_dim: int = 64
    local_latent_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    use_attention: bool = True
    n_heads: int = 4
    free_bits_global: float = 0.002
    free_bits_local: float = 0.0
    output_activation: str = "sigmoid"
