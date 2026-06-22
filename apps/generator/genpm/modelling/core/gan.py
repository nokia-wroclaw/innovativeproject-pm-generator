"""Conditional WGAN-GP for telecom KPI windows (first iteration).

Why a WGAN-GP, and why this shape
---------------------------------
This is the GAN counterpart to the cVAE-LSTM family.  It reuses every lesson the
VAE runs taught us, recast for an adversarial objective:

* Conditioning is the config one-hot + calendar context vector ``y`` broadcast to
  every timestep (``CellConditioning``) — exactly like v5/v6/v7.  The cell
  identity (distname) is never an input.  Both the generator and the critic are
  conditioned on ``y`` so the critic judges *config-conditional* realism, not just
  marginal realism.
* Wasserstein loss + gradient penalty (Gulrajani et al. 2017) instead of the
  vanilla GAN log-loss.  Mode collapse is the GAN analogue of the posterior
  collapse that dominated the VAE work; WGAN-GP is the most robust first-line
  defence against it and trains without the BatchNorm/log-loss fragility of DCGAN.
  (Critic uses LayerNorm, never BatchNorm — required for a correct GP.)
* Generator structure mirrors the proven v6/v7 decoder: a global noise vector
  ``z`` seeds the LSTM initial state *and* is tiled to every step so the LSTM
  cannot route around it, with a ``CrossKPICorrelation`` residual at the output
  so KPIs are not emitted independently.
* ``HourlyPositionalEncoding`` is now OFF in the generator (``gen_use_pe=False``)
  and ON in the critic (``critic_use_pe=True``).  The first GAN run showed the same
  failure the v7 VAE work diagnosed: with generator PE on, the synthetic series
  came out *more* periodic and smoother than reality (fake lag-24 autocorrelation
  0.97 vs real 0.90, fake lag-1 0.95 vs real 0.83) — the generator latches onto
  the sin/cos channels.  The hoped-for "the critic will penalise spurious
  periodicity" did not happen.  Keeping PE only in the critic lets it locate
  position to judge realism without feeding the generator a periodicity crutch;
  this matches the v7 stance (PE in encoder, not decoder).  Flip ``gen_use_pe=True``
  to restore the original behaviour.

Deliberately a *first* version: single-scale LSTM G/critic, one global noise
vector, WGAN-GP + an optional feature-matching (per-hour moment) term.  Natural
next steps once this runs: multi-scale / dilated-conv critic, a temporal
discriminator on first differences, spectral-norm, or a hierarchical z.

Backend note: the real training path is torch (tsgm forces KERAS_BACKEND=torch).
The torch ``train_step`` is implemented in full; tf/jax raise NotImplementedError
because the gradient penalty needs a backend-specific double-backward that has not
been validated outside torch.
"""

import os

# tsgm MUST be imported before keras — it patches keras internals on import.
import tsgm  # noqa: E402, F401, isort:skip
import keras
import numpy as np
from keras import layers, ops

from genpm.modelling.core.architectures import (
    CellConditioning,
    CrossKPICorrelation,
    HourlyPositionalEncoding,
    _funnel_schedule,
)
from genpm.utils.logger import get_logger

logger = get_logger()


# Recommended starting hyperparameters for a 235-KPI, 168-step telecom dataset.
HP_GAN = dict(
    epochs=300,
    batch_size=64,
    latent_dim=64,
    hidden_dim=256,
    n_layers=2,
    use_attention=True,
    n_heads=4,
    gen_use_pe=True,  # PE ON in generator: with a global-only z it is the only per-step
    # positional signal; run-2 turned it off and the diurnal cycle collapsed (see docstring)
    critic_use_pe=True,  # PE on in critic too — lets it locate position to judge realism
    kpi_proj_activation="linear",  # pre-residual KPI projection: linear, not relu (see TimeGANGenerator)
    per_step_noise_dim=16,  # fresh N(0,1) injected at EVERY step — gives the generator entropy…
    use_minibatch_stddev=True,  # …and this FORCES it to use that entropy (anti-collapse, see layer)
    use_first_diff=True,  # critic also sees ΔX so it can punish over-smoothing (lag-1 AC too high)
    output_activation="sigmoid",
    corr_l2=1e-5,
    learning_rate=1e-4,  # WGAN-GP likes a small Adam LR with low beta_1
    adam_beta_1=0.5,
    adam_beta_2=0.9,
    n_critic=3,  # critic updates per generator update (5→3: g_loss climbed, critic dominated)
    gp_weight=10.0,  # gradient-penalty coefficient (Gulrajani default)
    moment_weight=1.0,  # feature-matching START weight: match per-hour mean/std of real vs fake
    moment_weight_final=0.1,  # anneal toward this so the generator can't ride moments forever
)


class PerStepNoiseInjection(keras.layers.Layer):
    """Concatenate fresh i.i.d. N(0,1) noise to every timestep.

    The generator otherwise sees an *identical* input vector at every step — ``y``
    is broadcast by ``CellConditioning`` and the global ``z`` is tiled — so it has
    no per-step entropy and collapses to a near-deterministic curve per config
    (run-1/run-2 within-config diversity ~0.06 vs real ~0.11). This layer draws a
    fresh ``noise_dim`` vector at *each* timestep on *every* forward pass (training
    and generation), giving the LSTM a moving stochastic target it cannot fold into
    its hidden state once and then ignore. Sampled with the same
    ``keras.random.normal(shape=ops.shape(...))`` idiom as the VAE ``Sampling``
    layer, so it works under the functional API and the torch backend.
    """

    def __init__(self, noise_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.noise_dim = noise_dim

    def call(self, x):
        shape = ops.shape(x)
        noise = keras.random.normal((shape[0], shape[1], self.noise_dim))
        return ops.concatenate([x, noise], axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] + self.noise_dim)

    def get_config(self):
        return {**super().get_config(), "noise_dim": self.noise_dim}


class FirstDifference(keras.layers.Layer):
    """Append per-timestep first differences ΔX_t = X_t − X_{t-1} to the features.

    A critic that pools over time washes out hour-to-hour jitter, so it cannot see
    that the fakes are too smooth (run-1..3 lag-1 autocorr ~0.95 vs real ~0.88).
    Feeding ΔX exposes local volatility directly: an over-smoothed generator has
    systematically smaller |ΔX|, which the critic can then penalise. The first step
    is zero-padded so the sequence length is unchanged.
    """

    def call(self, x):  # (B, T, F)
        dx = x[:, 1:, :] - x[:, :-1, :]
        pad = ops.zeros_like(x[:, :1, :])
        dx = ops.concatenate([pad, dx], axis=1)
        return ops.concatenate([x, dx], axis=-1)  # (B, T, 2F)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] * 2)


class MinibatchStddev(keras.layers.Layer):
    """Append the minibatch standard deviation as one extra feature (PGGAN/StyleGAN).

    This is the targeted fix for the run-1..3 collapse where the generator ignored
    the per-step noise it was given (within-config diversity stuck at ~0.074 vs real
    ~0.113). It hands the critic a statistic no single fake sample can fake: the
    spread of activations *across the batch*. Because ``train_step`` builds the fake
    batch from the *same* conditioning ``y`` as the real batch, the between-config
    variance is identical on both sides and cancels — so the real-vs-fake difference
    in this statistic is exactly the within-config dispersion the generator dropped.
    The critic learns to penalise under-dispersed fakes, forcing the generator to use
    its noise.

    Per-sample independence note: like BatchNorm, this makes the critic output for a
    sample depend on its batch-mates, which the WGAN-GP gradient penalty assumes away.
    We append only a *single shared scalar*, so the cross-sample term in ∇D is tiny;
    this is the same compromise PGGAN/StyleGAN make when combining minibatch-stddev
    with WGAN-GP, and it is well established.
    """

    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, x):  # (B, P)
        mean = ops.mean(x, axis=0, keepdims=True)
        std = ops.sqrt(ops.mean(ops.square(x - mean), axis=0, keepdims=True) + self.epsilon)
        mb = ops.mean(std)  # single scalar over all features
        batch = ops.shape(x)[0]
        feat = ops.broadcast_to(ops.reshape(mb, (1, 1)), (batch, 1))
        return ops.concatenate([x, feat], axis=-1)  # (B, P + 1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] + 1)

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


class TimeGANGenerator:
    """Builds the conditional LSTM generator G(z, y) -> X_hat (B, T, F)."""

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        y_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_pe: bool = True,
        kpi_proj_activation: str = "linear",
        per_step_noise_dim: int = 16,
        output_activation: str = "sigmoid",
        corr_l2: float = 0.0,
        min_units: int = 32,
        max_dropout: float = 0.3,
    ):
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._y_dim = y_dim
        self._latent_dim = latent_dim
        self._use_pe = use_pe
        self._kpi_proj_activation = kpi_proj_activation
        self._per_step_noise_dim = per_step_noise_dim
        self._output_activation = output_activation
        self._corr_l2 = corr_l2
        self._cond_layer = CellConditioning(y_dim, seq_len)
        self._layer_units, self._layer_dropouts = _funnel_schedule(
            hidden_dim, n_layers, min_units, max_dropout
        )

    def build(self) -> keras.Model:
        z_in = keras.Input(shape=(self._latent_dim,), name="z")
        y_in = keras.Input(shape=(self._y_dim,), name="y")

        cond_rep = self._cond_layer(y_in)  # (B, T, y_dim)
        x = HourlyPositionalEncoding(self._seq_len)(cond_rep) if self._use_pe else cond_rep
        # Tile z to every step so the LSTM cannot forget the global noise.
        z_tiled = layers.RepeatVector(self._seq_len)(z_in)
        x = ops.concatenate([x, z_tiled], axis=-1)
        # Fresh per-step noise on top of the (constant) tiled z and broadcast y —
        # the only source of within-config, step-to-step stochasticity. See
        # PerStepNoiseInjection for why the global z alone collapses.
        if self._per_step_noise_dim > 0:
            x = PerStepNoiseInjection(self._per_step_noise_dim)(x)

        init_units = self._layer_units[0]
        h0 = layers.Dense(init_units, activation="tanh", name="gen_h0")(z_in)
        c0 = layers.Dense(init_units, activation="tanh", name="gen_c0")(z_in)

        for i, (units, drop) in enumerate(
            zip(self._layer_units, self._layer_dropouts, strict=False)
        ):
            if i == 0:
                x = layers.LSTM(units, return_sequences=True, name="gen_lstm_0")(
                    x, initial_state=[h0, c0]
                )
            else:
                x = layers.LSTM(units, return_sequences=True, name=f"gen_lstm_{i}")(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)

        # Cross-KPI correlation residual (same component as v7) so output KPIs are
        # not produced independently of one another. The projection is LINEAR, not
        # relu: a relu here forces the pre-sigmoid signal >= 0, so to emit any KPI
        # below 0.5 the CrossKPICorrelation residual must supply a large negative
        # offset — burning its capacity as a level-shifter instead of modelling
        # correlations. Linear lets the residual stay a pure correction.
        x_kpi = layers.TimeDistributed(
            layers.Dense(self._feat_dim, activation=self._kpi_proj_activation),
            name="gen_kpi_proj",
        )(x)
        x_corr = CrossKPICorrelation(self._feat_dim, corr_l2=self._corr_l2)(x_kpi)
        out = layers.Activation(self._output_activation)(x_kpi + x_corr)
        return keras.Model([z_in, y_in], out, name="generator")


class TimeGANCritic:
    """Builds the conditional BiLSTM critic D(X, y) -> scalar score (B, 1).

    No BatchNorm anywhere — a correct WGAN gradient penalty requires the critic to
    be a function of each sample independently, which BatchNorm breaks.  LayerNorm
    is fine and is what we use.
    """

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        y_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_attention: bool = True,
        n_heads: int = 4,
        use_pe: bool = True,
        use_minibatch_stddev: bool = True,
        use_first_diff: bool = True,
        min_units: int = 32,
        # No dropout in a WGAN-GP critic: dropout makes the critic a *different*
        # (stochastic) function on every forward pass, so the gradient penalty
        # E[(||∇D(x̂)||-1)²] is computed on a different network each time and the
        # 1-Lipschitz constraint is never cleanly enforced (run-1 gp plateaued at
        # ~0.62, not ~0). Defaults are 0 here; the generator keeps its dropout.
        max_dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._y_dim = y_dim
        self._use_attention = use_attention
        self._n_heads = n_heads
        self._use_pe = use_pe
        self._use_minibatch_stddev = use_minibatch_stddev
        self._use_first_diff = use_first_diff
        self._attn_dropout = attn_dropout
        self._cond_layer = CellConditioning(y_dim, seq_len)
        self._layer_units, self._layer_dropouts = _funnel_schedule(
            hidden_dim, n_layers, min_units, max_dropout
        )
        self._attn_key_dim = max((2 * self._layer_units[-1]) // n_heads, 1)

    def build(self) -> keras.Model:
        x_in = keras.Input(shape=(self._seq_len, self._feat_dim), name="x")
        y_in = keras.Input(shape=(self._y_dim,), name="y")

        cond_rep = self._cond_layer(y_in)  # (B, T, y_dim)
        # Expose local volatility (ΔX) so the critic can punish over-smoothing.
        x_feat = FirstDifference()(x_in) if self._use_first_diff else x_in
        h = ops.concatenate([x_feat, cond_rep], axis=-1)
        if self._use_pe:
            h = HourlyPositionalEncoding(self._seq_len)(h)

        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            h = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(h)
            h = layers.LayerNormalization()(h)
            h = layers.Dropout(rate=drop)(h)

        if self._use_attention:
            attn_out = layers.MultiHeadAttention(
                num_heads=self._n_heads, key_dim=self._attn_key_dim, dropout=self._attn_dropout
            )(h, h)
            h = layers.LayerNormalization()(h + attn_out)

        h = layers.GlobalAveragePooling1D()(h)
        # Minibatch stddev: lets the critic detect under-dispersed (collapsed) fakes.
        if self._use_minibatch_stddev:
            h = MinibatchStddev()(h)
        h = layers.Dense(self._layer_units[-1], activation="leaky_relu")(h)
        # Linear output — Wasserstein critic, no sigmoid.
        score = layers.Dense(1, name="critic_score")(h)
        return keras.Model([x_in, y_in], score, name="critic")


class ConditionalWGANGP(keras.Model):
    """Conditional Wasserstein GAN with gradient penalty for KPI windows.

    Trained via ``model.fit(X_scaled, y, ...)`` — ``train_step`` receives ``(x, y)``.
    Generation uses ``generate(y_compact) -> (X_hat, y)`` to match the cVAE API so
    ``core.generation`` works unchanged.
    """

    def __init__(
        self,
        generator: keras.Model,
        critic: keras.Model,
        latent_dim: int,
        seq_len: int,
        n_critic: int = 5,
        gp_weight: float = 10.0,
        moment_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self._seq_len = seq_len
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.moment_weight = moment_weight
        self._step = 0  # gates generator updates to 1-in-n_critic

        self.c_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.gp_tracker = keras.metrics.Mean(name="gp")
        self.w_dist_tracker = keras.metrics.Mean(name="w_dist")

    @property
    def metrics(self) -> list:
        return [
            self.c_loss_tracker,
            self.g_loss_tracker,
            self.gp_tracker,
            self.w_dist_tracker,
        ]

    def compile(self, g_optimizer=None, c_optimizer=None, **kwargs):
        # Keras requires *an* optimizer; we drive both networks manually with our
        # own g/c optimizers, so pass the generator optimizer up to satisfy keras.
        self.g_optimizer = g_optimizer or keras.optimizers.Adam(
            HP_GAN["learning_rate"], beta_1=HP_GAN["adam_beta_1"], beta_2=HP_GAN["adam_beta_2"]
        )
        self.c_optimizer = c_optimizer or keras.optimizers.Adam(
            HP_GAN["learning_rate"], beta_1=HP_GAN["adam_beta_1"], beta_2=HP_GAN["adam_beta_2"]
        )
        super().compile(optimizer=self.g_optimizer, **kwargs)

    def call(self, data):
        # Convenience only (e.g. building the graph): sample and decode.
        if isinstance(data, (list, tuple)) and len(data) == 2:  # noqa
            x, y = data
            z = keras.random.normal((ops.shape(x)[0], self.latent_dim))
            return self.generator([z, y], training=False)
        return data

    def generate(self, y_compact) -> tuple:
        """Sample z ~ N(0, I) and decode conditioned on y. Returns (X_hat, y)."""
        y_compact = ops.convert_to_tensor(y_compact, dtype="float32")
        batch = ops.shape(y_compact)[0]
        z = keras.random.normal((batch, self.latent_dim))
        x_hat = self.generator([z, y_compact], training=False)
        return x_hat, y_compact

    # --- losses -----------------------------------------------------------------
    def _moment_loss(self, x_real, x_fake):
        """Feature matching: match per-hour mean and std across the batch.

        Directly targets the per-hour mean±σ profile we evaluate on, and gives the
        generator a dense gradient early on while the critic is still weak.
        """
        if self.moment_weight <= 0.0:
            return 0.0
        real_mean = ops.mean(x_real, axis=0)
        fake_mean = ops.mean(x_fake, axis=0)
        real_std = ops.std(x_real, axis=0)
        fake_std = ops.std(x_fake, axis=0)
        return ops.mean(ops.square(real_mean - fake_mean)) + ops.mean(
            ops.square(real_std - fake_std)
        )

    # --- torch training ---------------------------------------------------------
    def _gradient_penalty_torch(self, torch, x_real, x_fake, y):
        """E[(||∇_x̂ D(x̂, y)||₂ - 1)²] on interpolates x̂ between real and fake."""
        batch = ops.shape(x_real)[0]
        alpha = keras.random.uniform((batch, 1, 1))
        interp = (alpha * x_real + (1.0 - alpha) * x_fake).detach()
        interp.requires_grad_(True)
        # The GP needs a double-backward (create_graph=True). When the critic uses
        # attention this runs through scaled_dot_product_attention, and the flash /
        # mem-efficient SDPA kernels have no double-backward implementation — only
        # the MATH kernel does. Torch's kernel choice depends on shape/dtype/dropout
        # (e.g. dropout=0 makes it prefer mem-efficient), so force MATH here to keep
        # the GP working regardless of those. Single-backward critic calls elsewhere
        # are unaffected and can use the fast kernels.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            d_interp = self.critic([interp, y], training=True)
            grads = torch.autograd.grad(
                outputs=d_interp.sum(),
                inputs=interp,
                create_graph=True,
                retain_graph=True,
            )[0]
        grad_norm = ops.sqrt(ops.sum(ops.square(grads), axis=[1, 2]) + 1e-12)
        return ops.mean(ops.square(grad_norm - 1.0))

    def _apply_torch(self, torch, loss, variables, optimizer):
        self.zero_grad()
        loss.backward()
        grads = [v.value.grad for v in variables]
        with torch.no_grad():
            optimizer.apply_gradients(list(zip(grads, variables, strict=False)))

    def train_step_torch(self, torch, data) -> dict:
        x_real, y = data
        if hasattr(y, "dtype"):
            y = ops.cast(y, "float32")
        x_real = ops.cast(x_real, "float32")
        batch = ops.shape(x_real)[0]

        # ---- critic update -----------------------------------------------------
        z = keras.random.normal((batch, self.latent_dim))
        with torch.no_grad():
            x_fake = self.generator([z, y], training=True)
        d_real = self.critic([x_real, y], training=True)
        d_fake = self.critic([x_fake, y], training=True)
        w_dist = ops.mean(d_real) - ops.mean(d_fake)  # critic's Wasserstein estimate
        gp = self._gradient_penalty_torch(torch, x_real, x_fake, y)
        c_loss = -w_dist + self.gp_weight * gp
        self._apply_torch(torch, c_loss, self.critic.trainable_weights, self.c_optimizer)

        # ---- generator update (1 in n_critic) ----------------------------------
        self._step += 1
        if self._step % self.n_critic == 0:
            z = keras.random.normal((batch, self.latent_dim))
            x_fake = self.generator([z, y], training=True)
            d_fake = self.critic([x_fake, y], training=True)
            g_loss = -ops.mean(d_fake) + self.moment_weight * self._moment_loss(x_real, x_fake)
            self._apply_torch(torch, g_loss, self.generator.trainable_weights, self.g_optimizer)
            self.g_loss_tracker.update_state(g_loss)

        self.c_loss_tracker.update_state(c_loss)
        self.gp_tracker.update_state(gp)
        self.w_dist_tracker.update_state(w_dist)
        return {
            "c_loss": self.c_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "gp": self.gp_tracker.result(),
            "w_dist": self.w_dist_tracker.result(),
        }

    def train_step(self, data) -> dict:
        backend = os.environ.get("KERAS_BACKEND")
        if backend == "torch":
            from tsgm.backend import get_backend

            return self.train_step_torch(get_backend(), data)
        raise NotImplementedError(
            f"ConditionalWGANGP train_step is implemented for the torch backend only "
            f"(KERAS_BACKEND={backend!r}). The gradient penalty needs a validated "
            f"double-backward; add a tf/jax path if you switch backends."
        )


def build_gan(
    seq_len: int,
    feat_dim: int,
    y_dim: int,
    latent_dim: int = HP_GAN["latent_dim"],
    hidden_dim: int = HP_GAN["hidden_dim"],
    n_layers: int = HP_GAN["n_layers"],
    use_attention: bool = HP_GAN["use_attention"],
    n_heads: int = HP_GAN["n_heads"],
    gen_use_pe: bool = HP_GAN["gen_use_pe"],
    critic_use_pe: bool = HP_GAN["critic_use_pe"],
    kpi_proj_activation: str = HP_GAN["kpi_proj_activation"],
    per_step_noise_dim: int = HP_GAN["per_step_noise_dim"],
    use_minibatch_stddev: bool = HP_GAN["use_minibatch_stddev"],
    use_first_diff: bool = HP_GAN["use_first_diff"],
    output_activation: str = HP_GAN["output_activation"],
    corr_l2: float = HP_GAN["corr_l2"],
    learning_rate: float = HP_GAN["learning_rate"],
    adam_beta_1: float = HP_GAN["adam_beta_1"],
    adam_beta_2: float = HP_GAN["adam_beta_2"],
    n_critic: int = HP_GAN["n_critic"],
    gp_weight: float = HP_GAN["gp_weight"],
    moment_weight: float = HP_GAN["moment_weight"],
) -> tuple[ConditionalWGANGP, keras.Model, keras.Model]:
    """Instantiate and compile the conditional WGAN-GP.

    Returns (model, generator, critic).
    """
    logger.info(
        f"Building GAN | seq_len={seq_len} feat_dim={feat_dim} y_dim={y_dim} "
        f"latent_dim={latent_dim} hidden_dim={hidden_dim} n_layers={n_layers} "
        f"use_attention={use_attention} gen_use_pe={gen_use_pe} "
        f"critic_use_pe={critic_use_pe} kpi_proj_activation={kpi_proj_activation} "
        f"per_step_noise_dim={per_step_noise_dim} "
        f"use_minibatch_stddev={use_minibatch_stddev} use_first_diff={use_first_diff} "
        f"n_critic={n_critic} gp_weight={gp_weight} moment_weight={moment_weight}"
    )
    generator = TimeGANGenerator(
        seq_len=seq_len,
        feat_dim=feat_dim,
        y_dim=y_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_pe=gen_use_pe,
        kpi_proj_activation=kpi_proj_activation,
        per_step_noise_dim=per_step_noise_dim,
        output_activation=output_activation,
        corr_l2=corr_l2,
    ).build()
    critic = TimeGANCritic(
        seq_len=seq_len,
        feat_dim=feat_dim,
        y_dim=y_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_attention=use_attention,
        n_heads=n_heads,
        use_pe=critic_use_pe,
        use_minibatch_stddev=use_minibatch_stddev,
        use_first_diff=use_first_diff,
    ).build()
    model = ConditionalWGANGP(
        generator=generator,
        critic=critic,
        latent_dim=latent_dim,
        seq_len=seq_len,
        n_critic=n_critic,
        gp_weight=gp_weight,
        moment_weight=moment_weight,
    )
    model.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2),
        c_optimizer=keras.optimizers.Adam(learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2),
    )
    # Build all variables AND mark the outer model built so weights can be
    # saved/loaded immediately. Calling the wrapper builds the generator; the
    # critic is not on the wrapper's forward path so it is built separately.
    dummy_x = np.zeros((1, seq_len, feat_dim), dtype=np.float32)
    dummy_y = np.zeros((1, y_dim), dtype=np.float32)
    model([dummy_x, dummy_y], training=False)
    critic([dummy_x, dummy_y], training=False)
    logger.info("GAN built and compiled")
    return model, generator, critic


def _gan_to_numpy(tensor) -> np.ndarray:
    """numpy from any backend tensor, including torch CUDA (mirrors generation._to_numpy)."""
    try:
        return np.asarray(tensor)
    except TypeError:
        return tensor.detach().cpu().numpy()


class GANDiversityMonitor(keras.callbacks.Callback):
    """Log per-epoch sample diversity so mode collapse is visible *during* training.

    The first GAN run looked healthy on its training losses (w_dist/gp flat and
    settled) yet had collapsed on the stochastic axis: holding the config ``y``
    fixed and varying the noise ``z`` produced near-identical windows.  That was
    only caught by a separate post-hoc probe.  This callback runs that probe every
    epoch: for each of a fixed set of probe configs it generates ``n_samples``
    windows (same ``y``, different ``z``) and records the cross-sample std averaged
    over time and features.  Healthy training => ``gen_diversity`` climbs toward
    ``real_diversity``; a flat/low ``gen_diversity`` is collapse.

    The values are written into the epoch ``logs`` dict so they land in
    ``training_history.json`` and the training-loss PNG automatically.
    """

    def __init__(
        self,
        y_probe: np.ndarray,
        n_samples: int = 32,
        real_diversity: float | None = None,
        key: str = "gen_diversity",
    ):
        super().__init__()
        self._y_probe = np.asarray(y_probe, dtype=np.float32)
        self._n_samples = n_samples
        self._real_diversity = real_diversity
        self._key = key

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}
        per_config = []
        for i in range(len(self._y_probe)):
            y_rep = np.repeat(self._y_probe[i : i + 1], self._n_samples, axis=0)
            x_fake, _ = self.model.generate(y_rep)
            x_fake = _gan_to_numpy(x_fake)
            per_config.append(float(x_fake.std(axis=0).mean()))
        logs[self._key] = float(np.mean(per_config))
        # Constant reference line so the PNG shows the gap to the real target.
        if self._real_diversity is not None:
            logs["real_diversity"] = float(self._real_diversity)


class MomentWeightScheduler(keras.callbacks.Callback):
    """Anneal the generator's feature-matching weight over training.

    The per-hour moment-matching term locks in the diurnal mean/std fast, but if
    held high it lets the generator satisfy the loss with one near-deterministic
    curve per config and never learn the harder within-config variability (run-1/2
    collapse). So hold it at ``w_start`` for the first ``hold_frac`` of training to
    pin the shape, then linearly decay to ``w_final`` so the adversarial term plus
    the new per-step noise drive the remaining loss reduction toward diversity.

    Mutates ``model.moment_weight`` (a plain attr read fresh each train_step) and
    logs the current value as ``moment_w`` for visibility.
    """

    def __init__(self, w_start: float, w_final: float, total_epochs: int, hold_frac: float = 0.25):
        super().__init__()
        self._w_start = w_start
        self._w_final = w_final
        self._total = max(1, total_epochs)
        self._hold = int(hold_frac * total_epochs)

    def _weight_at(self, epoch: int) -> float:
        if epoch < self._hold:
            return self._w_start
        t = (epoch - self._hold) / max(1, self._total - self._hold)
        return self._w_start + (self._w_final - self._w_start) * min(1.0, t)

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self.model.moment_weight = float(self._weight_at(epoch))

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is not None:
            logs["moment_w"] = float(self.model.moment_weight)
