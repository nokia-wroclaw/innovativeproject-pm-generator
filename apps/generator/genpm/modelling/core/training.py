"""Training loop, KL annealing callbacks, and collapse monitoring."""

import math
from pathlib import Path

# tsgm MUST be imported before keras — it patches keras internals on import.
import tsgm  # noqa: E402, F401, isort:skip
import keras
import numpy as np

from genpm.modelling.core.architectures import cBetaVAE_Hierarchical
from genpm.modelling.core.model import HP_V5
from genpm.utils.logger import get_logger

logger = get_logger()


def _is_hierarchical_model(model) -> bool:
    """True when model is cBetaVAE_Hierarchical (used to gate collapse monitoring)."""
    return isinstance(model, cBetaVAE_Hierarchical)


class _LinearKLAnnealer(keras.callbacks.Callback):
    """Linearly ramp model.beta from 0 to target_beta over warmup_epochs.

    delay_epochs holds beta at exactly 0 for that many epochs first — zero KL
    cost, not just a small one — before the ramp starts. Useful when a
    competing penalty (e.g. corr_l2) removes an easy reconstruction shortcut:
    even a tiny beta can be enough to tip the encoder/decoder into collapsing z
    before the slower "actually use z" payoff has had time to appear.
    """

    def __init__(self, target_beta: float, warmup_epochs: int, delay_epochs: int = 0) -> None:
        super().__init__()
        self.target_beta = target_beta
        self.warmup_epochs = warmup_epochs
        self.delay_epochs = delay_epochs

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        if epoch < self.delay_epochs:
            self.model.beta = 0.0
            return
        progress = (epoch - self.delay_epochs + 1) / self.warmup_epochs
        self.model.beta = self.target_beta * min(1.0, progress)


class CyclicalKLAnnealer(keras.callbacks.Callback):
    """
    Cyclical beta annealing (Fu et al. 2019).
    Ramps beta within each cycle, then holds — periodic resets reduce posterior collapse.

    delay_epochs holds beta at exactly 0 before cycling begins at all (see
    _LinearKLAnnealer for why a hard zero-cost delay can matter, not just a slow ramp).
    """

    def __init__(
        self,
        target_beta: float,
        cycle_epochs: int = 40,
        ratio: float = 0.5,
        n_cycles: int = 6,
        delay_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.target_beta = target_beta
        self.cycle_epochs = cycle_epochs
        self.ratio = ratio
        self.n_cycles = n_cycles
        self.delay_epochs = delay_epochs

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        if epoch < self.delay_epochs:
            self.model.beta = 0.0
            return
        epoch -= self.delay_epochs
        total = self.cycle_epochs * self.n_cycles
        if epoch >= total:
            self.model.beta = self.target_beta
            return
        pos = (epoch % self.cycle_epochs) / self.cycle_epochs
        if pos < self.ratio:
            self.model.beta = self.target_beta * (pos / self.ratio)
        else:
            self.model.beta = self.target_beta


class _DelayedReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
    """ReduceLROnPlateau that does nothing — including not advancing its wait
    counter — until start_from_epoch.

    Without this, a kl_delay_epochs warmup (beta=0, so reconstruction_loss is
    artificially as good as it will ever get) looks like a permanent plateau to
    any post-warmup epoch, since none can beat that pre-regularisation value by
    raw reconstruction_loss. ReduceLROnPlateau has no native start_from_epoch
    (unlike EarlyStopping), so masking the metric isn't enough — its wait
    counter would still tick up every delay epoch and could fire mid-delay.
    Skipping on_epoch_end entirely until start_from_epoch avoids that.
    """

    def __init__(self, *args, start_from_epoch: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_from_epoch = start_from_epoch

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if epoch < self.start_from_epoch:
            return
        super().on_epoch_end(epoch, logs)


class _DelayedModelCheckpoint(keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint that does nothing until start_from_epoch — see
    _DelayedReduceLROnPlateau. Same root cause: save_best_only would otherwise
    permanently freeze the saved weights at the end of the kl_delay_epochs
    warmup, since no later (properly KL-regularised) epoch can beat its
    artificially-good raw reconstruction_loss. That checkpoint is unusable for
    generation: z was never pulled toward N(0,1) at that point (kl_loss in the
    hundreds), so sampling z~N(0,1) at generation time feeds the decoder values
    it never saw during training.
    """

    def __init__(self, *args, start_from_epoch: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_from_epoch = start_from_epoch

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if epoch < self.start_from_epoch:
            return
        super().on_epoch_end(epoch, logs)


class CollapseMonitor(keras.callbacks.Callback):
    """Log |z_mean| and mean(z_log_var) each epoch to catch posterior collapse."""

    def __init__(self, X_sample: np.ndarray, y_sample: np.ndarray, n: int = 256) -> None:
        super().__init__()
        self.X = X_sample[:n].astype(np.float32)
        self.y = y_sample[:n].astype(np.float32)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """Print |z_mean| and mean z_log_var each epoch to diagnose posterior collapse."""
        if not hasattr(self.model, "encoder"):
            return
        x = keras.ops.convert_to_tensor(self.X)
        y = keras.ops.convert_to_tensor(self.y)
        # v5 encoder takes [x, y]; v6 encoder takes x only
        enc_input = [x, y] if len(self.model.encoder.inputs) == 2 else x
        enc_out = self.model.encoder(enc_input, training=False)
        z_mean = enc_out[0]
        z_log_var = enc_out[1]
        mean_norm = float(keras.ops.mean(keras.ops.abs(z_mean)))
        mean_logvar = float(keras.ops.mean(z_log_var))
        beta = getattr(self.model, "beta", 0.0)
        print(f"  [collapse] |z_mean|={mean_norm:.4f}  z_logvar={mean_logvar:.4f}  beta={beta:.6f}")


def train_cvae(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    weights_path: str | Path,
    epochs: int = HP_V5["epochs"],
    batch_size: int = HP_V5["batch_size"],
    target_beta: float = HP_V5["target_beta"],
    use_cyclical_kl: bool = True,
    cycle_epochs: int = HP_V5["cycle_epochs"],
    n_cycles: int = HP_V5["n_cycles"],
    cycle_ratio: float = HP_V5["cycle_ratio"],
    anneal_epochs: int = HP_V5["anneal_epochs"],
    kl_delay_epochs: int = 0,
    collapse_monitor: bool = True,
    **kwargs,
) -> keras.callbacks.History:
    """Train the cVAE with cyclical KL annealing and optional collapse monitoring.

    kl_delay_epochs: hold beta at exactly 0 for this many epochs before any ramp/
    cycling starts. See _LinearKLAnnealer / CyclicalKLAnnealer docstrings.

    Two checkpoint files are written: weights_path tracks the best
    reconstruction_loss seen *after* beta has finished its first ramp (see
    monitor_start_epoch below), and a sibling "*_last.weights.h5" file is
    overwritten unconditionally every epoch regardless of any metric — a
    guaranteed-valid fallback that doesn't depend on the "best" heuristic being
    correct, since that heuristic has already needed two rounds of fixes
    (kl_delay_epochs's beta=0 window, then the ramp that follows it also having
    an unfair advantage over later, fully-regularised epochs).
    """
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    # weights_path always ends in the literal double-suffix ".weights.h5" (required
    # by save_weights_only=True) — Path.stem/.suffix only strip the last ".h5", so
    # insert "_last" before the full ".weights.h5", not after a partial strip.
    last_weights_path = weights_path.with_name(
        weights_path.name.replace(".weights.h5", "_last.weights.h5")
    )

    # reconstruction_loss is artificially cheap not just during kl_delay_epochs but
    # for as long as beta hasn't finished ramping to target_beta — CyclicalKLAnnealer
    # recomputes pos=0 (beta=0 again) at the start of every cycle, so the epoch right
    # after the delay still has beta==0 exactly and is *not* a valid "best" candidate.
    # Wait until beta has fully ramped once before letting any callback compare
    # against it: delay + the first cycle's ramp length (cyclical) or the full
    # warmup (linear).
    ramp_len = math.ceil(cycle_epochs * cycle_ratio) if use_cyclical_kl else anneal_epochs
    monitor_start_epoch = kl_delay_epochs + ramp_len if kl_delay_epochs > 0 else 0

    logger.info(
        f"Starting training | epochs={epochs} batch_size={batch_size} "
        f"target_beta={target_beta} use_cyclical_kl={use_cyclical_kl} "
        f"kl_delay_epochs={kl_delay_epochs} monitor_start_epoch={monitor_start_epoch}"
    )
    logger.info(f"Best weights will be saved to {weights_path}")
    logger.info(f"Last-epoch weights (unconditional fallback) will be saved to {last_weights_path}")

    if use_cyclical_kl:
        kl_cb = CyclicalKLAnnealer(
            target_beta,
            cycle_epochs=cycle_epochs,
            ratio=cycle_ratio,
            n_cycles=n_cycles,
            delay_epochs=kl_delay_epochs,
        )
    else:
        kl_cb = _LinearKLAnnealer(target_beta, anneal_epochs, delay_epochs=kl_delay_epochs)

    callbacks = [
        kl_cb,
        _DelayedReduceLROnPlateau(
            monitor="reconstruction_loss",
            mode="min",
            factor=0.5,
            patience=kwargs.pop("lr_patience", 20),
            min_lr=1e-5,
            start_from_epoch=monitor_start_epoch,
        ),
        _DelayedModelCheckpoint(
            str(weights_path),
            monitor="reconstruction_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            start_from_epoch=monitor_start_epoch,
        ),
        # Unconditional fallback: always overwrite with the latest epoch's weights,
        # no metric/monitor involved at all. Guaranteed valid regardless of any
        # remaining issue in the "best" heuristic above.
        keras.callbacks.ModelCheckpoint(
            str(last_weights_path),
            save_best_only=False,
            save_weights_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="reconstruction_loss",
            mode="min",
            patience=kwargs.pop("early_stop_patience", 60),
            restore_best_weights=True,
            start_from_epoch=monitor_start_epoch,
        ),
    ]
    if collapse_monitor and _is_hierarchical_model(model):
        callbacks.append(CollapseMonitor(X_scaled, y))
        logger.info("CollapseMonitor enabled")

    logger.info(f"Fitting on {len(X_scaled):,} windows")
    return model.fit(
        X_scaled,
        y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
        **kwargs,
    )
