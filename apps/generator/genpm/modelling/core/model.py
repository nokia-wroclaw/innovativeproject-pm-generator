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

# v8: same architecture as v7 (cVAE_LSTMv7Architecture / cBetaVAE_Hierarchical_v7),
# tuned hyperparameters following the run-11 post-mortem:
#   - corr_l2 added: the learned CrossKPICorrelation kernel was unregularised and
#     went dense (off-diagonal energy 4.6x diagonal, mean diagonal weight -0.67),
#     which both overstated cross-KPI correlation and leaked 24h periodicity from
#     genuinely-periodic KPIs into KPIs with flat real autocorrelation.
#   - ac_weight raised 0.1 -> 0.3: at run-11's settled loss split (recon ~70%,
#     beta*KL ~20%, ac ~10%) the AC term was too small a fraction of the gradient
#     to override the shared decoder's tendency toward a single oscillatory mode.
#   - hidden_dim / global_latent_dim raised: run-11's losses (recon, kl, ac) had
#     all plateaued by ~epoch 150 with cosine-decayed LR, i.e. converged given
#     current capacity, not undertrained. More capacity is a secondary lever to
#     try after corr_l2 + ac_weight, to see if remaining AC/correlation error is
#     a capacity ceiling rather than just the unregularised-mixing artifact.
HP_V8 = dict(
    epochs=200,
    batch_size=64,
    global_latent_dim=128,
    local_latent_dim=0,
    hidden_dim=384,
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
    ac_weight=0.3,
    ac_max_lag=24,
    corr_l2=1e-5,
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
    corr_l2: float = 0.0,
):
    """Instantiate and compile the cVAE-LSTM v7 architecture.

    v7 changes over v6:
      - HourlyPositionalEncoding removed from decoder (keeps in encoder).
      - CrossKPICorrelation residual layer in the decoder output stage.
      - cBetaVAE_Hierarchical_v7 adds autocorrelation penalty to the ELBO.

    ac_weight: weight of the AC penalty relative to reconstruction loss.
               Start at 0.1; increase if AC mismatch persists after training.
    ac_max_lag: number of hourly lags to penalise (default 24 = one diurnal cycle).
    corr_l2: L2 weight on the CrossKPICorrelation F×F kernel. Without it nothing
             penalises the kernel for going dense — on the run-11 checkpoint it
             learned a mean diagonal of -0.67 with off-diagonal energy 4.6x the
             diagonal, i.e. every KPI was reconstructed mostly from a dense blend
             of other KPIs rather than its own projection. L1 at 1e-2..1e-4 over-
             corrected this to ~0 (constant per-entry gradient regardless of
             weight magnitude crushed nearly everything); L2's gradient scales
             with the weight's own value, so it shrinks without forcing genuinely
             useful entries to exactly zero. Start at 1e-5.
    """
    logger.info(
        f"Building v7 model | seq_len={seq_len} feat_dim={feat_dim} y_dim={y_dim} "
        f"global_latent_dim={global_latent_dim} hidden_dim={hidden_dim} "
        f"n_layers={n_layers} use_attention={use_attention} tile_z={tile_z} "
        f"ac_weight={ac_weight} ac_max_lag={ac_max_lag} corr_l2={corr_l2}"
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
        corr_l2=corr_l2,
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
