from dataclasses import dataclass, field


@dataclass
class GenerateConfig:
    """Generation request for a trained model (cVAE / GAN / diffusion).

    Conditioning identity comes from either ``cell_id`` (configs looked up from the
    training ``cell_config_map``) or explicit ``cell_configs``. The architecture
    fields default to the values saved at training time and should be overridden only
    deliberately, since they must match the checkpoint.
    """

    # Paths
    run_dir_path: str
    weights_path: str
    output_path: str
    anchor_date: str
    n_weeks: int
    kpi_list: list
    # Conditioning identity. Provide either cell_id (config looked up from the training
    # cell_config_map) or cell_configs (explicit config values). cell_id also labels the
    # output; when omitted, a label is derived from the config values.
    cell_id: str | None = None
    cell_configs: list | None = None
    holiday: int = 0
    batch_size: int = 64
    seed: int = 42
    # Model architecture. seq_len/n_dim default to the values saved at training time
    # (arch_params.json / kpi_columns.npy); set them only to override.
    seq_len: int | None = None
    n_dim: int | None = None
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
    """cVAE-LSTM training config (core/model.py + core/training.py).

    Drives :func:`run_training`. ``arch_version`` selects v7 (default; adds the
    autocorrelation penalty and cross-KPI correlation layer) or v6. The KL-annealing
    schedule fields (cyclical KL, delay, anneal) are the main levers against posterior
    collapse — see core/training.py.
    """

    # Paths
    training_data_path: str
    run_dir_path: str
    weights_path: str
    # Architecture version — "v7" (default) or "v6"
    arch_version: str = "v7"
    # Model architecture
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
    # v7-specific: autocorrelation penalty (ignored when arch_version="v6")
    ac_weight: float = 0.1
    ac_max_lag: int = 24
    # v7-specific: L2 weight on the CrossKPICorrelation F×F kernel, keeps it
    # sparse/near-identity instead of learning a dense KPI mix (ignored when
    # arch_version="v6"). 0.0 = no regularisation (run-11 default/behaviour).
    # L1 was tried first (run_12-14) but its constant per-entry gradient crushed
    # the kernel to ~0 at every magnitude tested (1e-2 down to 1e-4); L2's
    # gradient scales with the weight's own value so it shrinks without zeroing
    # out genuinely useful entries.
    corr_l2: float = 0.0
    # Training schedule
    epochs: int = 200
    batch_size: int = 64
    target_beta: float = 1e-3
    use_cyclical_kl: bool = True
    cycle_epochs: int = 30
    n_cycles: int = 6
    cycle_ratio: float = 0.5
    anneal_epochs: int = 150
    # Hold beta at exactly 0 for this many epochs before any ramp/cycling starts.
    # Not just a slower ramp — a hard zero-KL-cost window. Useful alongside corr_l2:
    # removing the decoder's cross-KPI shortcut can make even a tiny beta enough to
    # tip the encoder into collapsing z before it discovers the slower "use z" payoff.
    kl_delay_epochs: int = 0
    lr_patience: int = 20
    early_stop_patience: int = 60
    collapse_monitor: bool = True
    drop_constant_kpis: bool = True


@dataclass
class GANTrainConfig:
    """Conditional WGAN-GP training config (core/gan.py).

    Shares the data/path layout of TrainConfig but carries the adversarial knobs.
    arch_version is fixed to "gan" and written into arch_params.json so generation
    reloads the right model family.
    """

    # Paths
    training_data_path: str
    run_dir_path: str
    weights_path: str
    arch_version: str = "gan"
    # Generator / critic architecture
    latent_dim: int = 64
    hidden_dim: int = 256
    n_layers: int = 2
    use_attention: bool = True
    n_heads: int = 4
    gen_use_pe: bool = True  # PE ON in generator: with global-only z it is the only
    # per-step positional signal (run-2 off → diurnal cycle collapsed); see core/gan.py
    critic_use_pe: bool = True  # PE on in critic too
    kpi_proj_activation: str = "linear"  # pre-residual KPI projection (linear, not relu)
    per_step_noise_dim: int = 16  # fresh N(0,1) per timestep — gives the generator entropy…
    use_minibatch_stddev: bool = (
        True  # …and this forces it to use it (anti-collapse, see core/gan.py)
    )
    use_first_diff: bool = True  # critic also sees ΔX so it can punish over-smoothing
    output_activation: str = "sigmoid"
    corr_l2: float = 1e-5
    # Adversarial optimisation
    learning_rate: float = 1e-4
    adam_beta_1: float = 0.5
    adam_beta_2: float = 0.9
    n_critic: int = 3  # 5→3: g_loss climbed in run-1/2, critic dominated
    gp_weight: float = 10.0
    moment_weight: float = 1.0  # feature-matching START weight (annealed toward final)
    moment_weight_final: float = 0.1  # anneal target so the generator can't ride moments forever
    # Training schedule
    epochs: int = 300
    batch_size: int = 64
    drop_constant_kpis: bool = True


@dataclass
class DiffusionTrainConfig:
    """Conditional DDPM training config (core/diffusion.py).

    arch_version is fixed to "diffusion" and written into arch_params.json so
    generation reloads the right model family.
    """

    # Paths
    training_data_path: str
    run_dir_path: str
    weights_path: str
    arch_version: str = "diffusion"
    # Diffusion process
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"  # run-1 used "linear" → near-noise output; see core/diffusion.py
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    output_clip: bool = True
    # Denoiser architecture
    width: int = 256  # must be >= feat_dim (248); run-1's 128 was an under-fitting bottleneck
    n_blocks: int = 12
    # Conv dilations cycled across blocks. Includes 64 so the receptive field covers
    # the full 168-hour window (see HP_DIFFUSION in core/diffusion.py for the RF math).
    # Recorded in arch_params.json because it changes the forward pass but not weight
    # shapes — a train/reload mismatch would otherwise pass load_weights silently.
    dilation_cycle: tuple = (1, 2, 4, 8, 16, 32, 64)
    time_embed_dim: int = 128
    cond_embed_dim: int = 128
    # Per-timestep calendar conditioning (day-of-week, holiday, ...). When on, a
    # (B, 168, 6) tensor is fed to the denoiser to fix the over-regular-days residual
    # (R2). See build_calendar_features in core/data.py.
    use_calendar: bool = True
    calendar_country: str = "US"
    # Optimisation / schedule
    learning_rate: float = 2e-4
    use_ema: bool = (
        True  # EMA of weights — standard diffusion quality boost (see core/diffusion.py)
    )
    ema_momentum: float = 0.999
    epochs: int = 300
    batch_size: int = 64
    drop_constant_kpis: bool = True


@dataclass
class ValidateConfig:
    """Real-vs-synthetic validation config (validation.py).

    Compares generated output against real data for one cell over a date range
    (per-hour profiles, marginals, autocorrelation, cross-KPI correlation). The
    architecture fields must match the trained checkpoint being validated.
    """

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
