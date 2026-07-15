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
axis with FiLM conditioning (WaveNet/SSSD-lite).  Dilations cycle 1,2,4,8,16,32,64 so
a few blocks cover the full 168-hour receptive field — capturing both the 24h and
168h cycles.  Per-sample conditioning (``y`` config/calendar vector + the sinusoidal
diffusion timestep embedding) modulates every block via FiLM (per-channel scale/shift);
an optional per-timestep calendar tensor (day-of-week, holiday, ... — features that
*vary within* the window) is concatenated to the input stream instead, since FiLM is
per-sample.  ``HourlyPositionalEncoding`` is also concatenated so the denoiser always
knows where each step sits in the day/week.  Operating on the full window jointly,
PE here does not cause the spurious-sinusoid problem the VAE decoder had — the MSE
objective only rewards periodicity the data actually contains.

Data is min-max-scaled to ~[0,1]; internally we map to [-1,1] for the noise
process (DDPM assumes roughly zero-centred data) and map back at sample time.

Current shape: epsilon-prediction, **cosine** beta schedule (default; linear retained
for reloading older runs), full ancestral sampling with **per-step x0 thresholding**
(clamp the predicted x0 to [-1,1] every step — load-bearing under the cosine schedule,
see ``_generate_loop``).  Natural next steps: DDIM / fewer sampling steps for speed,
v-prediction, self-conditioning, learned variance, or a 1D U-Net denoiser.

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
    cell_embed_dim=16,  # learned per-cell (distname) embedding width. Vocab is small
    # (~250 cells in run-6), so this stays modest relative to cond_embed_dim=128 — it
    # only has to carry the residual per-cell deviation on top of the pooled config
    # signal in y, not re-derive the whole distribution (see core/data.py's docstring
    # and the "why separate, not merged into y" note in build_denoiser below).
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
    cond_dim: int = 0,
    cell_vocab_size: int = 0,
    cell_embed_dim: int = HP_DIFFUSION["cell_embed_dim"],
) -> keras.Model:
    """Build the epsilon-prediction denoiser network.

    Maps ``(x_noisy, t, y[, c][, cell_idx]) -> predicted noise`` of shape
    ``(B, T, feat_dim)``. ``cond_dim`` > 0 adds a per-timestep conditioning input ``c``
    (B, T, cond_dim) — calendar features (day-of-week, holiday, ...) that VARY within
    the window, so they cannot ride the broadcast ``y``/FiLM (which is per-sample).
    They join the spatial input stream and are processed by the conv stack alongside
    the KPIs. ``cell_vocab_size`` > 0 adds a per-sample cell-identity input ``cell_idx``
    (B,) — a learned embedding of *which specific cell* this window belongs to.

    ``cell_idx`` is deliberately an independent signal from ``y``, not folded into the
    same one-hot/vocab: config (in ``y``) is a many-to-one function of cell identity
    here (every cell has exactly one fixed config), so ``y`` captures the *pooled*
    "typical behaviour for this config" (gradient signal from every cell sharing it),
    while the cell embedding only has to learn the *residual* per-cell deviation on
    top of that (geography, real traffic mix, install quirks) — the standard
    "category embedding + entity embedding" pattern. Folding cell identity into the
    same one-hot as config would lose that pooling (each cell's own ~dozens of windows
    would have to relearn everything from scratch) and would break the existing
    config-only generation path (no cell_id — see core/generation.py), which has
    nothing to fall back on if identity subsumes config. The cell embedding gets the
    same dual treatment as ``y``: folded into the FiLM ``cond`` AND broadcast raw into
    the spatial stream (via the same ``CellConditioning`` repeat-layer used for ``y``).

    Args:
        seq_len: Window length T (168 hours).
        feat_dim: Number of KPI channels F.
        y_dim: Width of the broadcast conditioning vector ``y`` (config one-hot +
            holiday + seasonal).
        width: Channel width inside the conv stack. Must be >= feat_dim (an in_proj
            narrower than the data under-fits — see ``HP_DIFFUSION``).
        n_blocks: Number of residual dilated-conv blocks.
        dilation_cycle: Dilation rates cycled across blocks; sized so the receptive
            field spans the full 168h week.
        time_embed_dim: Width of the sinusoidal timestep embedding.
        cond_embed_dim: Width of the per-sample conditioning MLP outputs (FiLM source).
        cond_dim: Per-timestep calendar channels; 0 disables the ``c`` input.
        cell_vocab_size: Size of the cell-identity vocabulary (including the reserved
            "unknown" index 0); 0 disables the ``cell_idx`` input/embedding entirely.
        cell_embed_dim: Width of the learned per-cell embedding.

    Returns:
        A ``keras.Model`` named ``"denoiser"`` whose inputs are
        ``[x_noisy, t, y]`` plus ``c`` when ``cond_dim > 0`` plus ``cell_idx`` when
        ``cell_vocab_size > 0`` (in that order), and whose output is the predicted noise.
    """
    x_in = keras.Input(shape=(seq_len, feat_dim), name="x_noisy")
    t_in = keras.Input(shape=(), name="t")
    y_in = keras.Input(shape=(y_dim,), name="y")
    c_in = keras.Input(shape=(seq_len, cond_dim), name="c") if cond_dim > 0 else None
    cell_in = keras.Input(shape=(), dtype="int32", name="cell_idx") if cell_vocab_size > 0 else None

    # Conditioning embedding = timestep embedding ⊕ config/calendar embedding ⊕
    # (optional) cell-identity embedding.
    t_emb = SinusoidalTimeEmbedding(time_embed_dim)(t_in)
    t_emb = layers.Dense(cond_embed_dim, activation="swish", name="t_mlp1")(t_emb)
    t_emb = layers.Dense(cond_embed_dim, activation="swish", name="t_mlp2")(t_emb)
    y_emb = layers.Dense(cond_embed_dim, activation="swish", name="y_mlp")(y_in)
    cond_parts = [t_emb, y_emb]

    # Input: noisy KPIs ⊕ broadcast conditioning ⊕ (per-timestep calendar) ⊕
    # (broadcast cell embedding) ⊕ PE.
    cond_rep = CellConditioning(y_dim, seq_len)(y_in)  # (B, T, y_dim)
    spatial_parts = [x_in, cond_rep]
    if c_in is not None:
        spatial_parts.append(c_in)

    if cell_in is not None:
        cell_emb_raw = layers.Embedding(cell_vocab_size, cell_embed_dim, name="cell_embedding")(
            cell_in
        )  # (B, cell_embed_dim)
        cell_emb = layers.Dense(cond_embed_dim, activation="swish", name="cell_mlp")(cell_emb_raw)
        cond_parts.append(cell_emb)
        # Reuse CellConditioning (a generic per-sample-vector → per-timestep broadcast
        # repeat layer despite the "y_dim"-named ctor arg) for the cell embedding too.
        cell_rep = CellConditioning(cell_embed_dim, seq_len)(cell_emb_raw)  # (B, T, cell_embed_dim)
        spatial_parts.append(cell_rep)

    cond = layers.Concatenate(name="cond")(cond_parts)  # (B, (2 or 3)*cond_embed_dim)

    h = ops.concatenate(spatial_parts, axis=-1)
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
    inputs = [x_in, t_in, y_in]
    if c_in is not None:
        inputs.append(c_in)
    if cell_in is not None:
        inputs.append(cell_in)
    return keras.Model(inputs, eps_out, name="denoiser")


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
        cond_dim: int = 0,
        cell_vocab_size: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.denoiser = denoiser
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self.cond_dim = cond_dim  # per-timestep calendar channels (0 = none)
        self.cell_vocab_size = cell_vocab_size  # cell-identity embedding vocab (0 = none)
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
        # Convenience: one denoising prediction at t=0 (used only for graph building —
        # see build_diffusion's dummy forward pass). data is a flat [x, y, ...] list
        # (NOT the nested (inputs, y) structure train_step/_split_inputs use), with any
        # extra per-sample/spatial tensors (c, cell_idx) appended in denoiser-input order.
        if isinstance(data, (list, tuple)) and len(data) >= 2:  # noqa
            x, y, *rest = data
            t = ops.zeros((ops.shape(x)[0],), dtype="float32")
            return self.denoiser([x, t, y, *rest], training=False)
        return data

    def _split_inputs(self, data):
        """Unpack model.fit data into (x0, y, c, cell_idx). With calendar and/or the
        cell embedding on, fit is called with x=(X, C, cell_idx) (some subset), so
        data = ((x0, c, cell_idx), y); otherwise data = (x0, y)."""
        inputs, y = data
        if self.cond_dim > 0 and self.cell_vocab_size > 0:
            x0, c, cell_idx = inputs
        elif self.cond_dim > 0:
            x0, c = inputs
            cell_idx = None
        elif self.cell_vocab_size > 0:
            x0, cell_idx = inputs
            c = None
        else:
            x0, c, cell_idx = inputs, None, None
        return x0, y, c, cell_idx

    @staticmethod
    def _to_pm1(x):
        """[0,1] → [-1,1]."""
        return 2.0 * x - 1.0

    @staticmethod
    def _to_01(x):
        """[-1,1] → [0,1]."""
        return (x + 1.0) / 2.0

    def _compute_loss(self, x0, y, c=None, cell_idx=None):
        """Sample t and noise, build x_t, predict noise, return MSE(eps, eps_hat)."""
        batch = ops.shape(x0)[0]
        t = keras.random.randint((batch,), 0, self.num_timesteps)  # (B,) int
        eps = keras.random.normal(ops.shape(x0))
        sqrt_acp = ops.reshape(ops.take(self._sqrt_acp, t), (-1, 1, 1))
        sqrt_om = ops.reshape(ops.take(self._sqrt_one_minus_acp, t), (-1, 1, 1))
        x_t = sqrt_acp * x0 + sqrt_om * eps
        t_f = ops.cast(t, "float32")
        inputs = [x_t, t_f, y]
        if c is not None:
            inputs.append(c)
        if cell_idx is not None:
            inputs.append(cell_idx)
        eps_hat = self.denoiser(inputs, training=True)
        return ops.mean(ops.square(eps - eps_hat))

    def train_step_torch(self, torch, data) -> dict:
        x0, y, c, cell_idx = self._split_inputs(data)
        if hasattr(y, "dtype"):
            y = ops.cast(y, "float32")
        x0 = self._to_pm1(ops.cast(x0, "float32"))
        if c is not None:
            c = ops.cast(c, "float32")
        loss = self._compute_loss(x0, y, c, cell_idx)
        self.zero_grad()
        loss.backward()
        variables = self.trainable_weights
        grads = [v.value.grad for v in variables]
        with torch.no_grad():
            self.optimizer.apply_gradients(list(zip(grads, variables, strict=False)))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def train_step_tf(self, tf, data) -> dict:
        x0, y, c, cell_idx = self._split_inputs(data)
        x0 = self._to_pm1(ops.cast(x0, "float32"))
        if c is not None:
            c = ops.cast(c, "float32")
        with tf.GradientTape() as tape:
            loss = self._compute_loss(x0, y, c, cell_idx)
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
        calendar=None,
        cell_idx=None,
        num_steps: int | None = None,
        var_type: str | None = None,
        noise_scale: float | None = None,
    ) -> tuple:
        """Ancestral DDPM sampling conditioned on ``y``.

        Matches the cVAE ``generate`` API so ``core.generation`` works unchanged.

        Args:
            y_compact: Broadcast conditioning, shape (B, y_dim).
            calendar: Per-timestep calendar features (B, T, cond_dim). Required when
                the model was trained with them (cond_dim > 0); ignored otherwise.
            cell_idx: Cell-identity index per sample, shape (B,) int. Required when
                the model was trained with the cell embedding (cell_vocab_size > 0);
                pass 0 for "unknown cell" (e.g. config-only generation). Ignored
                otherwise.
            num_steps: Accepted for API symmetry only — full ancestral sampling always
                runs over all ``num_timesteps`` (DDIM subsampling is a planned follow-up).
            var_type: Posterior-variance choice. ``"small"`` = β̃ (DDPM fixedsmall,
                default, less diverse); ``"large"`` = β (DDPM fixedlarge, more diverse).
                Defaults to the model's stored value (the standard sampler).
            noise_scale: Multiplier on the injected per-step noise std (>1 = more
                diverse) — a cheap continuous dial complementing ``var_type``. Defaults
                to the model's stored value (1.0).

        Returns:
            Tuple ``(X_hat, y_compact)`` with ``X_hat`` in [0, 1], shape (B, T, F).

        Raises:
            ValueError: If the model was trained with calendar conditioning but
                ``calendar`` is not supplied, or with the cell embedding but
                ``cell_idx`` is not supplied.
        """
        del num_steps
        if self.cond_dim > 0 and calendar is None:
            raise ValueError(
                "This model was trained with per-timestep calendar conditioning "
                f"(cond_dim={self.cond_dim}); pass calendar=(B, {self._seq_len}, {self.cond_dim})."
            )
        if self.cell_vocab_size > 0 and cell_idx is None:
            raise ValueError(
                "This model was trained with the cell-identity embedding "
                f"(cell_vocab_size={self.cell_vocab_size}); pass cell_idx=(B,) int "
                "(use 0 for 'unknown cell' if generating config-only)."
            )
        var_type = var_type if var_type is not None else self._sample_var_type
        noise_scale = noise_scale if noise_scale is not None else self._sample_noise_scale
        backend = os.environ.get("KERAS_BACKEND")
        if backend == "torch":
            from tsgm.backend import get_backend

            torch = get_backend()
            with torch.no_grad():
                return self._generate_loop(y_compact, var_type, noise_scale, calendar, cell_idx)
        return self._generate_loop(y_compact, var_type, noise_scale, calendar, cell_idx)

    def _generate_loop(
        self, y_compact, var_type="small", noise_scale=1.0, calendar=None, cell_idx=None
    ) -> tuple:
        # Must run grad-free (see generate()): 1000 sequential forward passes
        # through the denoiser would otherwise chain into one giant autograd graph
        # and OOM the GPU well before sampling finishes.
        y_compact = ops.convert_to_tensor(y_compact, dtype="float32")
        if calendar is not None:
            calendar = ops.convert_to_tensor(calendar, dtype="float32")
        if cell_idx is not None:
            cell_idx = ops.convert_to_tensor(cell_idx, dtype="int32")
        batch = int(ops.shape(y_compact)[0])
        x = keras.random.normal((batch, self._seq_len, self._feat_dim))

        for i in range(self.num_timesteps - 1, -1, -1):
            t = ops.full((batch,), float(i), dtype="float32")
            den_in = [x, t, y_compact]
            if calendar is not None:
                den_in.append(calendar)
            if cell_idx is not None:
                den_in.append(cell_idx)
            eps_hat = self.denoiser(den_in, training=False)
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
    cond_dim: int = 0,
    cell_vocab_size: int = 0,
    cell_embed_dim: int = HP_DIFFUSION["cell_embed_dim"],
) -> tuple[ConditionalDDPM, keras.Model]:
    """Instantiate and compile the conditional DDPM.

    Builds the denoiser, wraps it in a ``ConditionalDDPM``, compiles with Adam, and
    runs a dummy forward pass so ``model.built`` is True and weights can be
    saved/loaded immediately (calling the wrapper also builds the denoiser).

    Args:
        seq_len: Window length T.
        feat_dim: Number of KPI channels F.
        y_dim: Width of the broadcast conditioning vector.
        num_timesteps: Number of diffusion steps.
        beta_schedule: ``"cosine"`` (default) or ``"linear"`` (back-compat).
        beta_start, beta_end: Endpoints used only by the linear schedule.
        width, n_blocks, dilation_cycle, time_embed_dim, cond_embed_dim: Denoiser
            geometry — forwarded to :func:`build_denoiser`.
        learning_rate: Adam learning rate (clipnorm=1.0).
        output_clip: Clip samples to [-1, 1] before mapping back to [0, 1].
        cond_dim: Per-timestep calendar channels; 0 disables calendar conditioning.
        cell_vocab_size: Size of the cell-identity vocabulary (incl. the reserved
            "unknown" index 0); 0 disables the learned per-cell embedding.
        cell_embed_dim: Width of the learned per-cell embedding.

    Returns:
        Tuple ``(model, denoiser)`` — the compiled ``ConditionalDDPM`` and the inner
        denoiser ``keras.Model`` (the latter is what EMA tracks).
    """
    logger.info(
        f"Building diffusion | seq_len={seq_len} feat_dim={feat_dim} y_dim={y_dim} "
        f"num_timesteps={num_timesteps} beta_schedule={beta_schedule} width={width} "
        f"n_blocks={n_blocks} time_embed_dim={time_embed_dim} cond_embed_dim={cond_embed_dim} "
        f"cond_dim={cond_dim} cell_vocab_size={cell_vocab_size} cell_embed_dim={cell_embed_dim}"
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
        cond_dim=cond_dim,
        cell_vocab_size=cell_vocab_size,
        cell_embed_dim=cell_embed_dim,
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
        cond_dim=cond_dim,
        cell_vocab_size=cell_vocab_size,
    )
    # EMA is applied via EMACallback (Keras optimizer use_ema does not update through
    # this model's custom torch train_step), so the optimizer here is plain Adam.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0))
    # Build variables AND mark the outer model built (calling the wrapper runs the
    # denoiser at t=0) so weights can be saved/loaded immediately.
    dummy_x = np.zeros((1, seq_len, feat_dim), dtype=np.float32)
    dummy_y = np.zeros((1, y_dim), dtype=np.float32)
    dummy_args = [dummy_x, dummy_y]
    if cond_dim > 0:
        dummy_args.append(np.zeros((1, seq_len, cond_dim), dtype=np.float32))
    if cell_vocab_size > 0:
        dummy_args.append(np.zeros((1,), dtype=np.int32))
    model(dummy_args, training=False)
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
        calendar_probe: np.ndarray | None = None,
        cell_idx_probe: int | None = None,
    ):
        super().__init__()
        self._y_probe = np.asarray(y_probe, dtype=np.float32)
        self._real_diversity = real_diversity
        self._real_ac1 = real_ac1
        self._every = max(1, every_n_epochs)
        self._n_samples = n_samples
        self._n_feat = n_feat_sample
        # per-timestep calendar for the probe window (1, T, cond_dim) or None
        self._calendar_probe = (
            np.asarray(calendar_probe, dtype=np.float32) if calendar_probe is not None else None
        )
        # cell-identity index for the probe window, or None when the embedding is off
        self._cell_idx_probe = cell_idx_probe
        self._cached = {"gen_diversity": float("nan"), "gen_ac1": float("nan")}

    def _evaluate(self) -> dict:
        row = self._y_probe[0]
        yb = np.repeat(row[None], self._n_samples, axis=0).astype(np.float32)
        cb = None
        if self._calendar_probe is not None:
            cb = np.repeat(self._calendar_probe[:1], self._n_samples, axis=0)
        cell_idx_b = None
        if self._cell_idx_probe is not None:
            cell_idx_b = np.full((self._n_samples,), self._cell_idx_probe, dtype=np.int32)
        xf = self.model.generate(yb, calendar=cb, cell_idx=cell_idx_b)[0]
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
