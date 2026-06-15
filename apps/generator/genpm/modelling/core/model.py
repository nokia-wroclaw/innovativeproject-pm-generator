# tsgm MUST be imported before keras — it patches keras internals on import.
import tsgm  # noqa: E402, F401, isort:skip
import keras

from genpm.modelling.core.architectures import (
    cBetaVAE_Hierarchical,
    cBetaVAE_Hierarchical_v7,
    cVAE_LSTMv6Architecture,
    cVAE_LSTMv7Architecture,
)
from genpm.utils.logger import get_logger

logger = get_logger()

HP_V5 = dict(
    epochs=300,
    batch_size=64,
    global_latent_dim=64,
    local_latent_dim=0,
    hidden_dim=256,
    n_layers=2,
    use_attention=True,
    n_heads=4,
    beta=0.0,
    learning_rate=3e-4,
    free_bits_global=0.002,
    free_bits_local=0.0,
    output_activation="sigmoid",
    target_beta=2e-4,
    anneal_epochs=150,
    cycle_epochs=40,
    n_cycles=6,
    cycle_ratio=0.5,
)

# v6: X-only encoder + z tiled at every decoder step — eliminates posterior collapse
HP_V6 = dict(
    epochs=200,
    batch_size=64,
    global_latent_dim=64,
    local_latent_dim=0,
    hidden_dim=256,
    n_layers=2,
    use_attention=True,
    n_heads=4,
    beta=0.0,
    learning_rate=3e-4,
    free_bits_global=0.002,  # back to original — z will naturally exceed this floor
    free_bits_local=0.0,
    output_activation="sigmoid",
    target_beta=1e-3,
    anneal_epochs=150,
    cycle_epochs=30,
    n_cycles=6,
    cycle_ratio=0.5,
)

# v7: no PE in decoder + cross-KPI correlation layer + autocorrelation penalty
HP_V7 = dict(
    epochs=200,
    batch_size=64,
    global_latent_dim=64,
    local_latent_dim=0,
    hidden_dim=256,
    n_layers=2,
    use_attention=True,
    n_heads=4,
    beta=0.0,
    learning_rate=3e-4,
    free_bits_global=0.002,
    free_bits_local=0.0,
    output_activation="sigmoid",
    target_beta=1e-3,
    anneal_epochs=150,
    cycle_epochs=30,
    n_cycles=6,
    cycle_ratio=0.5,
    # v7-specific
    ac_weight=0.1,  # scale of autocorrelation penalty relative to reconstruction loss
    ac_max_lag=24,  # number of lags (hours) to match; 24 covers one full diurnal cycle
)


def build_cvae_lstm(
    seq_len: int,
    feat_dim: int,
    y_dim: int,
    global_latent_dim: int = 64,
    local_latent_dim: int = 0,
    hidden_dim: int = 256,
    n_layers: int = 2,
    use_attention: bool = True,
    n_heads: int = 4,
    beta: float = 0.0,
    learning_rate: float = 3e-4,
    free_bits_global: float = 0.002,
    free_bits_local: float = 0.0,
    output_activation: str = "sigmoid",
    tile_z: bool = True,
):
    """Instantiate and compile the cVAE-LSTM v6 architecture."""
    logger.info(
        f"Building model | seq_len={seq_len} feat_dim={feat_dim} y_dim={y_dim} "
        f"global_latent_dim={global_latent_dim} hidden_dim={hidden_dim} "
        f"n_layers={n_layers} use_attention={use_attention} tile_z={tile_z}"
    )
    arch = cVAE_LSTMv6Architecture(
        seq_len=seq_len,
        feat_dim=feat_dim,
        y_dim=y_dim,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_attention=use_attention,
        n_heads=n_heads,
        output_activation=output_activation,
        tile_z=tile_z,
    )
    model = cBetaVAE_Hierarchical(
        encoder=arch.encoder,
        decoder=arch.decoder,
        cond_layer=arch.cond_layer,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        seq_len=seq_len,
        beta=beta,
        free_bits_global=free_bits_global,
        free_bits_local=free_bits_local,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0))
    logger.info("Model built and compiled")
    return arch, model


def build_cvae_lstm_v7(
    seq_len: int,
    feat_dim: int,
    y_dim: int,
    global_latent_dim: int = 64,
    local_latent_dim: int = 0,
    hidden_dim: int = 256,
    n_layers: int = 2,
    use_attention: bool = True,
    n_heads: int = 4,
    beta: float = 0.0,
    learning_rate: float = 3e-4,
    free_bits_global: float = 0.002,
    free_bits_local: float = 0.0,
    output_activation: str = "sigmoid",
    tile_z: bool = True,
    ac_weight: float = 0.1,
    ac_max_lag: int = 24,
):
    """Instantiate and compile the cVAE-LSTM v7 architecture.

    v7 changes over v6:
      - HourlyPositionalEncoding removed from decoder (keeps in encoder).
      - CrossKPICorrelation residual layer in the decoder output stage.
      - cBetaVAE_Hierarchical_v7 adds autocorrelation penalty to the ELBO.

    ac_weight: weight of the AC penalty relative to reconstruction loss.
               Start at 0.1; increase if AC mismatch persists after training.
    ac_max_lag: number of hourly lags to penalise (default 24 = one diurnal cycle).
    """
    logger.info(
        f"Building v7 model | seq_len={seq_len} feat_dim={feat_dim} y_dim={y_dim} "
        f"global_latent_dim={global_latent_dim} hidden_dim={hidden_dim} "
        f"n_layers={n_layers} use_attention={use_attention} tile_z={tile_z} "
        f"ac_weight={ac_weight} ac_max_lag={ac_max_lag}"
    )
    arch = cVAE_LSTMv7Architecture(
        seq_len=seq_len,
        feat_dim=feat_dim,
        y_dim=y_dim,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_attention=use_attention,
        n_heads=n_heads,
        output_activation=output_activation,
        tile_z=tile_z,
    )
    model = cBetaVAE_Hierarchical_v7(
        encoder=arch.encoder,
        decoder=arch.decoder,
        cond_layer=arch.cond_layer,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        seq_len=seq_len,
        beta=beta,
        free_bits_global=free_bits_global,
        free_bits_local=free_bits_local,
        ac_weight=ac_weight,
        ac_max_lag=ac_max_lag,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0))
    logger.info("Model built and compiled")
    return arch, model
