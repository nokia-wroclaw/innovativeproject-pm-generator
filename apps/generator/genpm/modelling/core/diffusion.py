"""Conditional DDPM (denoising diffusion) for telecom KPI windows (first iteration).

Why diffusion, and why this shape
---------------------------------
Diffusion is arguably the best structural fit for the failure modes the VAE runs
surfaced:

* No collapse. DDPM has no latent KL term and no adversarial game, so neither
  posterior collapse (the VAE's nemesis) nor mode collapse (the GAN's) can occur.
  Training is a plain per-pixel/per-step MSE regression onto the sampled noise —
  about the most stable objective available.
* Cross-KPI correlation and autocorrelation come for free. The denoiser sees the
  whole (T, F) window jointly at every step, so it models the joint distribution
  over all KPIs and all hours directly, rather than emitting KPIs independently
  the way the VAE decoder did (which is exactly why v7 had to bolt on
  ``CrossKPICorrelation`` and an autocorrelation penalty).


FORWARD (training data prep) — pure math, instant, no network involved

 t=0          t≈250         t≈500         t≈750         t=999
┌──────┐    ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐
│╱╲_╱╲_│ →  │╱╲_╱▒░│  →   │░▒▓░▒▓│  →   │▓▓▒░▓▒│  →   │▓▓▓▓▓▓│
└──────┘    └──────┘      └──────┘      └──────┘      └──────┘
 real KPI    +touch of      half          mostly        pure
  curve       noise         noise         noise         noise
signal: 100%    86%           49%           16%          0.02%   ← √ᾱ_t (cosine schedule)

REVERSE (generation) — 1000 real calls to the trained network, one per step

 i=999        i≈750         i≈500         i≈250         i=0
┌──────┐    ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐
│▓▓▓▓▓▓│ →  │▓▒░▓▒▓│  →   │░▒▓░▒▓│  →   │╱▒_╱▒_│  →   │╱╲_╱╲_│
└──────┘    └──────┘      └──────┘      └──────┘      └──────┘
 start:      denoiser       shape         almost        final
 pure noise  nudges it      emerging      clean         KPI window


Denoiser architecture: a stack of residual **dilated Conv1D** blocks over the time
axis with FiLM conditioning (WaveNet/SSSD-lite).  Dilations cycle 1,2,4,8,16,32 so
a few blocks cover the full 168-hour receptive field — capturing both the 24h and
168h cycles.  Conditioning (``y`` config/calendar vector + the sinusoidal diffusion
timestep embedding) modulates every block via FiLM (per-channel scale/shift).
``HourlyPositionalEncoding`` is concatenated to the input so the denoiser always
knows where each step sits in the day/week.  Operating on the full window jointly,
PE here does not cause the spurious-sinusoid problem the VAE decoder had — the MSE
objective only rewards periodicity the data actually contains.

Data is min-max-scaled to ~[0,1]; internally we map to [-1,1] for the noise
process (DDPM assumes roughly zero-centred data) and map back at sample time.

Deliberately a *first* version: epsilon-prediction, linear beta schedule, full
ancestral sampling.  Natural next steps: DDIM / fewer sampling steps for speed,
a v-prediction or cosine schedule, self-conditioning, or a 1D U-Net denoiser.

Backend note: torch is the real path (tsgm forces KERAS_BACKEND=torch) and is
implemented in full; a tf mirror is provided, jax raises NotImplementedError.
"""

import math
import os

# tsgm MUST be imported before keras — it patches keras internals on import.
import tsgm  # noqa: E402, F401, isort:skip
import keras
import numpy as np
from keras import layers, ops

from genpm.modelling.core.architectures import CellConditioning, HourlyPositionalEncoding
from genpm.utils.logger import get_logger

logger = get_logger()


# Recommended starting hyperparameters for a 235-KPI, 168-step telecom dataset.
HP_DIFFUSION = dict(
    epochs=300,
    batch_size=64,
    num_timesteps=1000,
    beta_schedule="cosine",  # "cosine" (run-2+) or "linear" (run-1). Linear destroyed
    # signal too fast for this structured low-D data → samples were near-noise; cosine
    # keeps more steps at useful SNR. beta_start/beta_end are only used by "linear".
    beta_start=1e-4,
    beta_end=2e-2,
    width=256,  # channel width inside the denoiser. MUST be >= feat_dim (248): run-1 used
    # 128 < 248, an in_proj bottleneck that left the denoiser under-fit (loss plateaued
    # at ~0.50, i.e. it explained only ~half the noise variance).
    n_blocks=12,  # residual blocks (dilations cycle through dilation_cycle)
    # Receptive field (kernel=3, per block = conv1@dilation d + conv2@dilation 1):
    #   RF = 1 + Σ_blocks (2d + 2).
    # The 64 closes the full 168-hour window: dilations [1,2,4,8,16,32,64,1] over
    # 8 blocks give RF = 1 + 2·128 + 16 = 273 ≥ 168, so every output hour can attend
    # to the whole week (incl. same-hour-next-week, 168h apart). Without the 64 the
    # cycle only reached ~149h, just short of the weekly span. Self-attention (a
    # cheaper way to get full global reach) is the planned follow-up after run 1.
    dilation_cycle=(1, 2, 4, 8, 16, 32, 64),
    time_embed_dim=128,
    cond_embed_dim=128,
    learning_rate=2e-4,
    use_ema=True,  # exponential moving average of weights — standard diffusion quality boost
    ema_momentum=0.999,  # EMA decay; generation/saving use the averaged weights
    output_clip=True,  # clip samples to [-1,1] before mapping back to [0,1]
)


class SinusoidalTimeEmbedding(layers.Layer):
    """
    Transformer-style sinusoidal embedding of the (scalar) diffusion timestep.

    e.g.
    freq  0 (slowest):  ╲________________________________________________╱
    freq 16 (medium):   ╲__╱‾‾╲__╱‾‾╲__╱‾‾╲__╱‾‾╲__╱‾‾╲__╱‾‾╲__╱‾‾╲__╱‾‾
    freq 63 (fastest):  ╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱
                            ▲                              ▲
                            t=250                          t=700

    t_emb(250) = the vertical slice through all 128 curves at x=250
    t_emb(700) = a totally different slice → a different "fingerprint"

    Slow waves say roughly "early vs. late in the 1000 steps";
    fast waves pin down the exact step. Together they give every
    layer a precise, smoothly-varying readout of
    "how noisy is this input right now" —
    nearby t get similar fingerprints, distant t get very different ones.

    """

    def __init__(self, dim: int, max_period: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_period = max_period

    def call(self, t):
        # t: (B,) float timestep index
        half = self.dim // 2
        freqs = ops.exp(
            -math.log(self.max_period) * ops.arange(half, dtype="float32") / float(half)
        )
        args = ops.cast(t[:, None], "float32") * freqs[None, :]  # (B, half)
        emb = ops.concatenate([ops.sin(args), ops.cos(args)], axis=-1)  # (B, 2*half)
        if self.dim % 2 == 1:  # zero-pad if odd
            emb = ops.concatenate([emb, ops.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "max_period": self.max_period}


def _film(h, cond, width: int, name: str):
    """Feature-wise linear modulation: h * (1 + scale) + shift, scale/shift from cond."""
    scale = layers.Dense(width, name=f"{name}_scale")(cond)  # (B, width)
    shift = layers.Dense(width, name=f"{name}_shift")(cond)
    scale = ops.expand_dims(scale, axis=1)  # (B, 1, width) → broadcasts over time
    shift = ops.expand_dims(shift, axis=1)
    return h * (1.0 + scale) + shift


def _res_block(h, cond, width: int, dilation: int, idx: int):
    """Pre-norm residual block: LN → dilated Conv → FiLM → swish → Conv → +residual."""
    res = h
    h = layers.LayerNormalization(name=f"block{idx}_ln")(h)
    h = layers.Conv1D(
        width, kernel_size=3, padding="same", dilation_rate=dilation, name=f"block{idx}_conv1"
    )(h)
    h = _film(h, cond, width, name=f"block{idx}_film")
    h = layers.Activation("swish")(h)
    h = layers.Conv1D(width, kernel_size=3, padding="same", name=f"block{idx}_conv2")(h)
    return layers.Add(name=f"block{idx}_add")([res, h])


def build_denoiser(
    seq_len: int,
    feat_dim: int,
    y_dim: int,
    width: int = HP_DIFFUSION["width"],
    n_blocks: int = HP_DIFFUSION["n_blocks"],
    dilation_cycle: tuple = HP_DIFFUSION["dilation_cycle"],
    time_embed_dim: int = HP_DIFFUSION["time_embed_dim"],
    cond_embed_dim: int = HP_DIFFUSION["cond_embed_dim"],
) -> keras.Model:
    """Build epsilon-prediction denoiser: (x_noisy, t, y) -> predicted noise (B,T,F)."""
    x_in = keras.Input(shape=(seq_len, feat_dim), name="x_noisy")
    t_in = keras.Input(shape=(), name="t")
    y_in = keras.Input(shape=(y_dim,), name="y")

    # Conditioning embedding = timestep embedding ⊕ config/calendar embedding.
    t_emb = SinusoidalTimeEmbedding(time_embed_dim)(t_in)
    t_emb = layers.Dense(cond_embed_dim, activation="swish", name="t_mlp1")(t_emb)
    t_emb = layers.Dense(cond_embed_dim, activation="swish", name="t_mlp2")(t_emb)
    y_emb = layers.Dense(cond_embed_dim, activation="swish", name="y_mlp")(y_in)
    cond = layers.Concatenate(name="cond")([t_emb, y_emb])  # (B, 2*cond_embed_dim)

    # Input: noisy KPIs ⊕ broadcast conditioning ⊕ hourly positional encoding.
    cond_rep = CellConditioning(y_dim, seq_len)(y_in)  # (B, T, y_dim)
    h = ops.concatenate([x_in, cond_rep], axis=-1)
    h = HourlyPositionalEncoding(seq_len)(h)
    h = layers.Conv1D(width, kernel_size=1, name="in_proj")(h)

    for i in range(n_blocks):
        dilation = dilation_cycle[i % len(dilation_cycle)]
        h = _res_block(h, cond, width, dilation, idx=i)

    h = layers.LayerNormalization(name="out_ln")(h)
    h = layers.Activation("swish")(h)
    # Zero-init the final conv so the denoiser starts as a no-op (standard DDPM trick).
    eps_out = layers.Conv1D(
        feat_dim,
        kernel_size=1,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="eps_out",
    )(h)
    return keras.Model([x_in, t_in, y_in], eps_out, name="denoiser")


def _make_beta_schedule(
    schedule: str, num_timesteps: int, beta_start: float, beta_end: float
) -> np.ndarray:
    """Return the (num_timesteps,) beta schedule.

    "linear": the original DDPM schedule (Ho et al. 2020), beta_start→beta_end.
    "cosine": Nichol & Dhariwal 2021 — define alphas_cumprod via a cosine and derive
        betas from it. Spends far more steps at moderate SNR instead of collapsing
        the signal early, which is what run-1's linear schedule did to this
        structured, low-dimensional data (output came out as near-noise).
    """
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    if schedule == "cosine":
        s = 0.008
        t = np.linspace(0, num_timesteps, num_timesteps + 1, dtype=np.float64)
        f = np.cos(((t / num_timesteps + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
        acp = f / f[0]
        betas = 1.0 - acp[1:] / acp[:-1]
        return np.clip(betas, 1e-8, 0.999)
    raise ValueError(f"Unknown beta_schedule: {schedule!r} (use 'linear' or 'cosine')")


class ConditionalDDPM(keras.Model):
    """Conditional denoising diffusion model for KPI windows.

    Trained via ``model.fit(X_scaled, y, ...)`` — ``train_step`` receives ``(x, y)``.
    Generation uses ``generate(y_compact) -> (X_hat, y)`` to match the cVAE API so
    ``core.generation`` works unchanged.
    """

    def __init__(
        self,
        denoiser: keras.Model,
        seq_len: int,
        feat_dim: int,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        output_clip: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.denoiser = denoiser
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.output_clip = output_clip
        # Default sampling-diversity knobs (overridable per generate() call). "small"
        # + 1.0 reproduce the standard DDPM sampler.
        self._sample_var_type = "small"
        self._sample_noise_scale = 1.0

        # Beta schedule and derived constants (numpy — used as python floats in the
        # sampling loop and as constant tensors during training).
        betas = _make_beta_schedule(beta_schedule, num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self._betas_np = betas.astype(np.float32)
        self._alphas_np = alphas.astype(np.float32)
        self._acp_np = alphas_cumprod.astype(np.float32)
        self._acp_prev_np = np.concatenate([[1.0], alphas_cumprod[:-1]]).astype(np.float32)
        # Constant tensors for the (batched, varying-t) training step.
        self._sqrt_acp = ops.convert_to_tensor(np.sqrt(self._acp_np))
        self._sqrt_one_minus_acp = ops.convert_to_tensor(np.sqrt(1.0 - self._acp_np))

        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self) -> list:
        return [self.loss_tracker]

    def call(self, data):
        # Convenience: one denoising prediction at t=0 (used only for graph building).
        if isinstance(data, (list, tuple)) and len(data) == 2:  # noqa
            x, y = data
            t = ops.zeros((ops.shape(x)[0],), dtype="float32")
            return self.denoiser([x, t, y], training=False)
        return data

    @staticmethod
    def _to_pm1(x):
        """[0,1] → [-1,1]."""
        return 2.0 * x - 1.0

    @staticmethod
    def _to_01(x):
        """[-1,1] → [0,1]."""
        return (x + 1.0) / 2.0

    def _compute_loss(self, x0, y):
        """Sample t and noise, build x_t, predict noise, return MSE(eps, eps_hat)."""
        batch = ops.shape(x0)[0]
        t = keras.random.randint((batch,), 0, self.num_timesteps)  # (B,) int
        eps = keras.random.normal(ops.shape(x0))
        sqrt_acp = ops.reshape(ops.take(self._sqrt_acp, t), (-1, 1, 1))
        sqrt_om = ops.reshape(ops.take(self._sqrt_one_minus_acp, t), (-1, 1, 1))
        x_t = sqrt_acp * x0 + sqrt_om * eps
        eps_hat = self.denoiser([x_t, ops.cast(t, "float32"), y], training=True)
        return ops.mean(ops.square(eps - eps_hat))

    def train_step_torch(self, torch, data) -> dict:
        x0, y = data
        if hasattr(y, "dtype"):
            y = ops.cast(y, "float32")
        x0 = self._to_pm1(ops.cast(x0, "float32"))
        loss = self._compute_loss(x0, y)
        self.zero_grad()
        loss.backward()
        variables = self.trainable_weights
        grads = [v.value.grad for v in variables]
        with torch.no_grad():
            self.optimizer.apply_gradients(list(zip(grads, variables, strict=False)))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def train_step_tf(self, tf, data) -> dict:
        x0, y = data
        x0 = self._to_pm1(ops.cast(x0, "float32"))
        with tf.GradientTape() as tape:
            loss = self._compute_loss(x0, y)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights, strict=False))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def train_step(self, data) -> dict:
        backend = os.environ.get("KERAS_BACKEND")
        if backend == "torch":
            from tsgm.backend import get_backend

            return self.train_step_torch(get_backend(), data)
        elif backend == "tensorflow":
            from tsgm.backend import get_backend

            return self.train_step_tf(get_backend(), data)
        raise NotImplementedError(
            f"ConditionalDDPM train_step is implemented for torch/tensorflow "
            f"(KERAS_BACKEND={backend!r})."
        )

    def generate(
        self,
        y_compact,
        num_steps: int | None = None,
        var_type: str | None = None,
        noise_scale: float | None = None,
    ) -> tuple:
        """Ancestral DDPM sampling conditioned on y. Returns (X_hat in [0,1], y).

        num_steps is accepted for API symmetry but full ancestral sampling always
        runs over all num_timesteps; subsampling (DDIM) is a planned follow-up.

        Sampling-diversity knobs (default to the model's stored values, which default
        to the standard sampler so existing callers are unchanged):
          var_type    — "small" = posterior variance β̃ (DDPM fixedsmall, default);
                        "large" = β (DDPM fixedlarge), which samples more diversely.
          noise_scale — multiplier on the injected per-step noise std (>1 = more
                        diverse). A cheap continuous dial complementing var_type.
        """
        del num_steps
        var_type = var_type if var_type is not None else self._sample_var_type
        noise_scale = noise_scale if noise_scale is not None else self._sample_noise_scale
        backend = os.environ.get("KERAS_BACKEND")
        if backend == "torch":
            from tsgm.backend import get_backend

            torch = get_backend()
            with torch.no_grad():
                return self._generate_loop(y_compact, var_type, noise_scale)
        return self._generate_loop(y_compact, var_type, noise_scale)

    def _generate_loop(self, y_compact, var_type="small", noise_scale=1.0) -> tuple:
        # Must run grad-free (see generate()): 1000 sequential forward passes
        # through the denoiser would otherwise chain into one giant autograd graph
        # and OOM the GPU well before sampling finishes.
        y_compact = ops.convert_to_tensor(y_compact, dtype="float32")
        batch = int(ops.shape(y_compact)[0])
        x = keras.random.normal((batch, self._seq_len, self._feat_dim))

        for i in range(self.num_timesteps - 1, -1, -1):
            t = ops.full((batch,), float(i), dtype="float32")
            eps_hat = self.denoiser([x, t, y_compact], training=False)
            beta_i = float(self._betas_np[i])
            alpha_i = float(self._alphas_np[i])
            acp_i = float(self._acp_np[i])
            acp_prev = float(self._acp_prev_np[i])

            # Predict x0 and THRESHOLD it to [-1,1] *every* step (Imagen/DDPM static
            # thresholding), then form the posterior mean from the clamped x0. This is
            # the x0-parameterised equivalent of mean=(x-coef·eps)/√α, but clamping is
            # essential with the cosine schedule: at high t, √ᾱ→~5e-5, so the implied
            # x0=(x-√(1-ᾱ)·epŝ)/√ᾱ amplifies a tiny epŝ residual to std ~1e3 and blows
            # up the whole trajectory from step 1 (data x0 std is ~0.6). Clamping keeps
            # every step on the data manifold. (Run-2 model is fine; this fixes sampling.)
            x0_hat = (x - math.sqrt(1.0 - acp_i) * eps_hat) / math.sqrt(acp_i)
            if self.output_clip:
                x0_hat = ops.clip(x0_hat, -1.0, 1.0)
            coef_x0 = math.sqrt(acp_prev) * beta_i / (1.0 - acp_i)
            coef_xt = math.sqrt(alpha_i) * (1.0 - acp_prev) / (1.0 - acp_i)
            mean = coef_x0 * x0_hat + coef_xt * x
            if i > 0:
                # "small" = posterior variance β̃ (less diverse); "large" = β (more diverse).
                var = beta_i if var_type == "large" else beta_i * (1.0 - acp_prev) / (1.0 - acp_i)
                x = mean + noise_scale * math.sqrt(var) * keras.random.normal(ops.shape(x))
            else:
                x = mean

        if self.output_clip:
            x = ops.clip(x, -1.0, 1.0)
        return self._to_01(x), y_compact


def build_diffusion(
    seq_len: int,
    feat_dim: int,
    y_dim: int,
    num_timesteps: int = HP_DIFFUSION["num_timesteps"],
    beta_schedule: str = HP_DIFFUSION["beta_schedule"],
    beta_start: float = HP_DIFFUSION["beta_start"],
    beta_end: float = HP_DIFFUSION["beta_end"],
    width: int = HP_DIFFUSION["width"],
    n_blocks: int = HP_DIFFUSION["n_blocks"],
    dilation_cycle: tuple = HP_DIFFUSION["dilation_cycle"],
    time_embed_dim: int = HP_DIFFUSION["time_embed_dim"],
    cond_embed_dim: int = HP_DIFFUSION["cond_embed_dim"],
    learning_rate: float = HP_DIFFUSION["learning_rate"],
    output_clip: bool = HP_DIFFUSION["output_clip"],
) -> tuple[ConditionalDDPM, keras.Model]:
    """Instantiate and compile the conditional DDPM. Returns (model, denoiser)."""
    logger.info(
        f"Building diffusion | seq_len={seq_len} feat_dim={feat_dim} y_dim={y_dim} "
        f"num_timesteps={num_timesteps} beta_schedule={beta_schedule} width={width} "
        f"n_blocks={n_blocks} time_embed_dim={time_embed_dim} cond_embed_dim={cond_embed_dim}"
    )
    denoiser = build_denoiser(
        seq_len=seq_len,
        feat_dim=feat_dim,
        y_dim=y_dim,
        width=width,
        n_blocks=n_blocks,
        dilation_cycle=dilation_cycle,
        time_embed_dim=time_embed_dim,
        cond_embed_dim=cond_embed_dim,
    )
    model = ConditionalDDPM(
        denoiser=denoiser,
        seq_len=seq_len,
        feat_dim=feat_dim,
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        output_clip=output_clip,
    )
    # EMA is applied via EMACallback (Keras optimizer use_ema does not update through
    # this model's custom torch train_step), so the optimizer here is plain Adam.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0))
    # Build variables AND mark the outer model built (calling the wrapper runs the
    # denoiser at t=0) so weights can be saved/loaded immediately.
    dummy_x = np.zeros((1, seq_len, feat_dim), dtype=np.float32)
    dummy_y = np.zeros((1, y_dim), dtype=np.float32)
    model([dummy_x, dummy_y], training=False)
    logger.info("Diffusion model built and compiled")
    return model, denoiser


def _lag1_autocorr(arr: np.ndarray) -> float:
    """Mean lag-1 autocorrelation over a batch of series (arr: (n, T)). ~0 = white noise."""
    a, b = arr[:, :-1], arr[:, 1:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)) + 1e-8
    return float((num / den).mean())


class DiffusionEvalCallback(keras.callbacks.Callback):
    """Periodically sample and log whether real *structure* is emerging.

    Diffusion's failure mode is the opposite of the GAN's: run-1 produced plenty of
    diversity but pure noise (zero autocorrelation, flat diurnal profile). The loss
    tracks fit, but the decisive quality signal is whether generated series develop
    temporal structure. Full ancestral sampling is ~1000 steps, far too slow to run
    every epoch, so this samples only every ``every_n_epochs`` (on a few configs,
    small batch) and caches the result so the logged series stay full-length:

      * ``gen_ac1``  — mean lag-1 autocorrelation of generated series. ~0 means still
        noise; should climb toward ``real_ac1``. THE key run-2 success signal.
      * ``gen_diversity`` — cross-sample std at fixed y; should stay near
        ``real_diversity`` (diffusion shouldn't collapse, but watch it doesn't over-correct).
    """

    def __init__(
        self,
        y_probe: np.ndarray,
        real_diversity: float,
        real_ac1: float,
        every_n_epochs: int = 25,
        n_samples: int = 16,
        n_feat_sample: int = 20,
    ):
        super().__init__()
        self._y_probe = np.asarray(y_probe, dtype=np.float32)
        self._real_diversity = real_diversity
        self._real_ac1 = real_ac1
        self._every = max(1, every_n_epochs)
        self._n_samples = n_samples
        self._n_feat = n_feat_sample
        self._cached = {"gen_diversity": float("nan"), "gen_ac1": float("nan")}

    def _evaluate(self) -> dict:
        row = self._y_probe[0]
        yb = np.repeat(row[None], self._n_samples, axis=0).astype(np.float32)
        xf = self.model.generate(yb)[0]
        xf = xf.detach().cpu().numpy() if hasattr(xf, "detach") else np.asarray(xf)
        rng = np.random.default_rng(0)
        fi = rng.choice(xf.shape[2], min(self._n_feat, xf.shape[2]), replace=False)
        # lag-1 autocorr averaged over sampled (sample, feature) series
        series = xf[:, :, fi].transpose(0, 2, 1).reshape(-1, xf.shape[1])
        return {
            "gen_diversity": float(xf.std(axis=0).mean()),
            "gen_ac1": _lag1_autocorr(series),
        }

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}
        if epoch == 0 or (epoch + 1) % self._every == 0:
            self._cached = self._evaluate()
        logs.update(self._cached)
        logs["real_diversity"] = float(self._real_diversity)
        logs["real_ac1"] = float(self._real_ac1)


class EMACallback(keras.callbacks.Callback):
    """Exponential moving average of the denoiser's trainable weights.

    Keras' optimizer ``use_ema`` does not update through this model's custom torch
    ``train_step`` (verified: the shadow vars stay equal to the raw weights), so EMA
    is maintained here instead. Shadow tensors are kept on-device and updated in
    place every batch (negligible cost); ``finalize()`` writes the averaged weights
    into the model after training so the saved checkpoint / generation use them.
    Torch-only (matches the project's backend).
    """

    def __init__(self, denoiser: keras.Model, momentum: float = 0.999):
        super().__init__()
        self._denoiser = denoiser
        self._m = momentum
        self._shadow: list | None = None

    def on_train_begin(self, logs: dict | None = None) -> None:
        self._shadow = [w.value.detach().clone() for w in self._denoiser.trainable_weights]

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        m = self._m
        for s, w in zip(self._shadow, self._denoiser.trainable_weights, strict=True):
            s.mul_(m).add_(w.value.detach(), alpha=1.0 - m)

    def finalize(self) -> None:
        """Assign the EMA weights into the denoiser (call after fit, before saving)."""
        if self._shadow is None:
            return
        for w, s in zip(self._denoiser.trainable_weights, self._shadow, strict=True):
            w.assign(s)
