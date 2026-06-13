from dataclasses import dataclass


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
    # Model architecture (must match the trained checkpoint)
    seq_len: int = 168
    n_dim: int = 235
    global_latent_dim: int = 64
    local_latent_dim: int = 0
    cell_embed_dim: int = 32
    hidden_dim: int = 256
    n_layers: int = 2
    use_attention: bool = True
    n_heads: int = 4
    free_bits_global: float = 0.002
    free_bits_local: float = 0.0
    output_activation: str = "sigmoid"
