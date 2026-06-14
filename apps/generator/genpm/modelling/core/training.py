"""Training loop, KL annealing callbacks, and collapse monitoring."""

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
    return isinstance(model, cBetaVAE_Hierarchical)


class _LinearKLAnnealer(keras.callbacks.Callback):
    """Linearly ramp model.beta from 0 to target_beta over warmup_epochs."""

    def __init__(self, target_beta: float, warmup_epochs: int) -> None:
        super().__init__()
        self.target_beta = target_beta
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self.model.beta = self.target_beta * min(1.0, (epoch + 1) / self.warmup_epochs)


class CyclicalKLAnnealer(keras.callbacks.Callback):
    """
    Cyclical beta annealing (Fu et al. 2019).
    Ramps beta within each cycle, then holds — periodic resets reduce posterior collapse.
    """

    def __init__(
        self,
        target_beta: float,
        cycle_epochs: int = 40,
        ratio: float = 0.5,
        n_cycles: int = 6,
    ) -> None:
        super().__init__()
        self.target_beta = target_beta
        self.cycle_epochs = cycle_epochs
        self.ratio = ratio
        self.n_cycles = n_cycles

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        total = self.cycle_epochs * self.n_cycles
        if epoch >= total:
            self.model.beta = self.target_beta
            return
        pos = (epoch % self.cycle_epochs) / self.cycle_epochs
        if pos < self.ratio:
            self.model.beta = self.target_beta * (pos / self.ratio)
        else:
            self.model.beta = self.target_beta


class CollapseMonitor(keras.callbacks.Callback):
    """Log |z_mean| and mean(z_log_var) each epoch to catch posterior collapse."""

    def __init__(self, X_sample: np.ndarray, y_sample: np.ndarray, n: int = 256) -> None:
        super().__init__()
        self.X = X_sample[:n].astype(np.float32)
        self.y = y_sample[:n].astype(np.float32)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if not hasattr(self.model, "encoder"):
            return
        x = keras.ops.convert_to_tensor(self.X)
        y = keras.ops.convert_to_tensor(self.y)
        enc_out = self.model.encoder([x, y], training=False)
        z_mean = enc_out[0]
        z_log_var = enc_out[1]
        mean_norm = float(keras.ops.mean(keras.ops.abs(z_mean)))
        mean_logvar = float(keras.ops.mean(z_log_var))
        beta = getattr(self.model, "beta", 0.0)
        print(
            f"  [collapse] |z_mean|={mean_norm:.4f}  "
            f"z_logvar={mean_logvar:.4f}  beta={beta:.6f}"
        )


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
    collapse_monitor: bool = True,
    **kwargs,
) -> keras.callbacks.History:
    """Train the cVAE with cyclical KL annealing and optional collapse monitoring."""
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Starting training | epochs={epochs} batch_size={batch_size} "
        f"target_beta={target_beta} use_cyclical_kl={use_cyclical_kl}"
    )
    logger.info(f"Best weights will be saved to {weights_path}")

    if use_cyclical_kl:
        kl_cb = CyclicalKLAnnealer(
            target_beta,
            cycle_epochs=cycle_epochs,
            ratio=cycle_ratio,
            n_cycles=n_cycles,
        )
    else:
        kl_cb = _LinearKLAnnealer(target_beta, anneal_epochs)

    callbacks = [
        kl_cb,
        keras.callbacks.ReduceLROnPlateau(
            monitor="reconstruction_loss",
            mode="min",
            factor=0.5,
            patience=kwargs.pop("lr_patience", 20),
            min_lr=1e-5,
        ),
        keras.callbacks.ModelCheckpoint(
            str(weights_path),
            monitor="reconstruction_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="reconstruction_loss",
            mode="min",
            patience=kwargs.pop("early_stop_patience", 60),
            restore_best_weights=True,
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
        verbose=1,
        **kwargs,
    )
