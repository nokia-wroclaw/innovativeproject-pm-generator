import abc
import math
import os

import keras
import tsgm
import tsgm.utils
from keras import layers, ops
from tsgm.backend import get_backend


class CellConditioning(keras.layers.Layer):
    """
    Turns compact Y (B, 6) into a per-timestep conditioning tensor (B, T, E+5).
    Input y_compact columns:
        0: cell_idx (integer, stored as float)
        1: holiday
        2-5: seasonal features
    """

    def __init__(self, n_cells: int, embed_dim: int = 32, seq_len: int = 168, **kwargs):
        super().__init__(**kwargs)
        self.n_cells = n_cells
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.cell_embedding = layers.Embedding(n_cells, embed_dim)

    def call(self, y_compact):
        # y_compact: (B, 6)
        cell_idx = ops.cast(y_compact[:, 0], "int32")  # (B,)
        ctx = y_compact[:, 1:]  # (B, 5) holiday + seasonal
        cell_emb = self.cell_embedding(cell_idx)  # (B, E)
        cond = ops.concatenate([cell_emb, ctx], axis=-1)  # (B, E+5) = (B, 37)
        # Broadcast to every hour
        cond_rep = ops.repeat(cond[:, None, :], [self.seq_len], axis=1)  # (B, T, 37)
        return cond_rep

    def get_config(self):
        return {
            **super().get_config(),
            "n_cells": self.n_cells,
            "embed_dim": self.embed_dim,
            "seq_len": self.seq_len,
        }


class BetaVAE(keras.Model):
    """
    beta-VAE implementation for unlabeled time series.
    """

    def __init__(
        self, encoder: keras.Model, decoder: keras.Model, beta: float = 1.0, **kwargs
    ) -> None:
        """
        :param encoder: An encoder model which takes a time series as input.
        :type encoder: keras.Model
        :param decoder: Takes as input a random noise vector and returns a simulated time-series.
        :type decoder: keras.Model
        :param beta: The weight of the KL divergence term. Default is 1.0.
        :type beta: float
        """
        super(BetaVAE, self).__init__(**kwargs)  # noqa
        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self._seq_len = self.decoder.output_shape[1]
        self.latent_dim = self.decoder.input_shape[1]

    @property
    def metrics(self) -> list:
        """
        :returns: A list of metrics trackers (total loss, reconstruction loss, and KL loss).
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, X: tsgm.types.Tensor) -> tsgm.types.Tensor:
        """
        Encodes and decodes time series dataset X.

        :param X: The input time series tensor.
        :type X: tsgm.types.Tensor

        :returns: Generated samples
        :rtype: tsgm.types.Tensor
        """
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def _get_reconstruction_loss(self, X: tsgm.types.Tensor, Xr: tsgm.types.Tensor) -> float:
        reconst_loss = (
            tsgm.utils.reconstruction_loss_by_axis(X, Xr, axis=0)
            + tsgm.utils.reconstruction_loss_by_axis(X, Xr, axis=1)
            + tsgm.utils.reconstruction_loss_by_axis(X, Xr, axis=2)
        )
        return reconst_loss

    def train_step_tf(self, tf, data: tsgm.types.Tensor) -> dict:
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = self._get_reconstruction_loss(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        #  I am not sure if this should be self.optimizer.apply(grads, model.trainable_weights)
        #  see https://keras.io/guides/writing_a_custom_training_loop_in_tensorflow/
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights, strict=False))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step_torch(self, torch, data: tsgm.types.Tensor) -> dict:
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(data, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta * kl_loss
        # Ensure total_loss is a scalar for PyTorch backward()
        if hasattr(total_loss, "shape") and len(total_loss.shape) > 0:
            total_loss = ops.mean(total_loss)
        self.zero_grad()
        total_loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step_jax(self, jax, data: tsgm.types.Tensor) -> dict:
        # JAX backend uses Keras 3.0 automatic differentiation
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(data, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step(self, data: tsgm.types.Tensor) -> dict:
        """
        Performs a training step using a batch of data, stored in data.

        :param data: A batch of data in a format batch_size x seq_len x feat_dim
        :type data: tsgm.types.Tensor

        :returns: A dict with losses
        :rtype: T.Dict
        """
        backend = get_backend()
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            return self.train_step_tf(backend, data)
        elif os.environ.get("KERAS_BACKEND") == "torch":
            return self.train_step_torch(backend, data)
        elif os.environ.get("KERAS_BACKEND") == "jax":
            return self.train_step_jax(backend, data)

    def generate(self, n: int) -> tsgm.types.Tensor:
        """
        Generates new data from the model.

        :param n: the number of samples to be generated.
        :type n: int

        :returns: A tensor with generated samples.
        :rtype: tsgm.types.Tensor
        """
        #  keras 3.0 support
        z = keras.random.normal((n, self.latent_dim))
        return self.decoder(z)


class cBetaVAE(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        latent_dim: int,
        temporal: bool,
        beta: float = 1.0,
        global_z: bool = False,
        free_bits: float = 0.0,
        **kwargs,
    ) -> None:
        """
        :param global_z: When True, z has shape (batch, latent_dim) — the
            encoder compresses the full sequence to a single global code.
            The decoder input is constructed by tiling z along seq_len.
            Required for cVAE_LSTMv4Architecture.  When False (default), z
            has shape (batch, latent_dim * seq_len) — legacy per-timestep mode
            used by v1/v2/v3 architectures.
        :param free_bits: Minimum KL per latent dimension (nats).  Prevents
            posterior collapse by ensuring each dimension contributes at least
            ``free_bits`` nats regardless of beta.  Typical value: 0.5.
            0.0 disables the constraint (backward-compatible default).
        """
        super(cBetaVAE, self).__init__(**kwargs)  # noqa
        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder
        self._global_z = global_z
        self.free_bits = free_bits

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self._temporal = temporal
        self._seq_len = self.decoder.output_shape[1]
        self.latent_dim = latent_dim

    @property
    def metrics(self) -> list:
        """
        Returns the list of loss tracker:  `[loss, reconstruction_loss, kl_loss]`.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def generate(self, labels: tsgm.types.Tensor) -> tuple[tsgm.types.Tensor, tsgm.types.Tensor]:
        """
        Generates new data from the model.

        :param labels: The labels for which to generate conditional samples.
        :type labels: tsgm.types.Tensor

        :returns: A tuple of synthetically generated data and labels.
        :rtype: T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor]
        """
        batch_size = ops.shape(labels)[0]
        dtype = "float32" if os.environ.get("KERAS_BACKEND") == "torch" else labels.dtype
        if self._global_z:
            z = keras.random.normal((batch_size, self.latent_dim), dtype=dtype)
        else:
            z = keras.random.normal((batch_size, self._seq_len, self.latent_dim), dtype=dtype)
        decoder_input = self._get_decoder_input(z, labels)
        return (self.decoder(decoder_input), labels)

    def call(self, data: tsgm.types.Tensor) -> tsgm.types.Tensor:
        """
        Encodes and decodes time series dataset.

        :param data: The input data, either a tensor or a tuple of (X, labels).
        :type data: tsgm.types.Tensor

        :returns: Generated samples.
        :rtype: tsgm.types.Tensor
        """
        # Handle both single tensor and tuple of (X, labels)
        if isinstance(data, (list, tuple)) and len(data) == 2:  # noqa
            X, labels = data
        else:
            # During model building, just return the input
            return data
        encoder_input = self._get_encoder_input(X, labels)
        z_mean, _, _ = self.encoder(encoder_input)
        decoder_input = self._get_decoder_input(z_mean, labels)
        x_decoded = self.decoder(decoder_input)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def _get_reconstruction_loss(self, X: tsgm.types.Tensor, Xr: tsgm.types.Tensor) -> float:
        #  keras 3.0 support
        reconst_loss = (
            ops.mean(ops.square(X - Xr))
            + ops.mean(ops.square(ops.mean(X, axis=1) - ops.mean(Xr, axis=1)))
            + ops.mean(ops.square(ops.mean(X, axis=2) - ops.mean(Xr, axis=2)))
        )
        return reconst_loss

    def _get_encoder_input(
        self, X: tsgm.types.Tensor, labels: tsgm.types.Tensor
    ) -> tsgm.types.Tensor:
        #  keras 3.0 support
        if os.environ.get("KERAS_BACKEND") == "torch" and hasattr(labels, "dtype"):
            labels = ops.cast(labels, "float32")
        if self._temporal:
            return ops.concatenate([X, labels[:, :, None]], axis=2)
        else:
            rep_labels = ops.repeat(labels[:, None, :], [self._seq_len], axis=1)
            return ops.concatenate([X, rep_labels], axis=2)

    def _get_decoder_input(
        self, z: tsgm.types.Tensor, labels: tsgm.types.Tensor
    ) -> tsgm.types.Tensor:
        if os.environ.get("KERAS_BACKEND") == "torch" and hasattr(labels, "dtype"):
            labels = ops.cast(labels, "float32")
        if self._temporal:
            rep_labels = labels[:, :, None]
        else:
            rep_labels = ops.repeat(labels[:, None, :], [self._seq_len], axis=1)
        if self._global_z:
            # z shape: (batch, latent_dim) → tile to (batch, seq_len, latent_dim)
            z_tiled = ops.repeat(z[:, None, :], [self._seq_len], axis=1)
        else:
            z_tiled = ops.reshape(z, [-1, self._seq_len, self.latent_dim])
        return ops.concatenate([z_tiled, rep_labels], axis=2)

    def _compute_kl_loss(self, z_mean: tsgm.types.Tensor, z_log_var: tsgm.types.Tensor) -> float:
        kl_per_dim = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        if self.free_bits > 0.0:
            kl_per_dim = ops.maximum(kl_per_dim, self.free_bits)
        return ops.mean(ops.sum(kl_per_dim, axis=1))

    def train_step_tf(self, tf, data: tsgm.types.Tensor) -> dict[str, float]:
        X, labels = data
        with tf.GradientTape() as tape:
            encoder_input = self._get_encoder_input(X, labels)
            z_mean, z_log_var, z = self.encoder(encoder_input)

            z_log_var = ops.clip(z_log_var, -6.0, 2.0)
            decoder_input = self._get_decoder_input(z, labels)
            reconstruction = self.decoder(decoder_input)
            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)
            kl_loss = self._compute_kl_loss(z_mean, z_log_var)
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights, strict=False))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step_torch(self, torch, data: tsgm.types.Tensor) -> dict[str, float]:
        X, labels = data
        encoder_input = self._get_encoder_input(X, labels)
        z_mean, z_log_var, z = self.encoder(encoder_input)

        z_log_var = ops.clip(z_log_var, -6.0, 2.0)
        # Use the reparameterised z (not z_mean) so the decoder is trained on
        # the same stochastic inputs it receives during generation via _sample_z.
        # Using z_mean here (as was the case before) makes the decoder
        # out-of-distribution at generation time, producing random spikes.
        decoder_input = self._get_decoder_input(z, labels)
        reconstruction = self.decoder(decoder_input)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)
        kl_loss = self._compute_kl_loss(z_mean, z_log_var)
        total_loss = reconstruction_loss + self.beta * kl_loss
        # Ensure total_loss is a scalar for PyTorch backward()
        if hasattr(total_loss, "shape") and len(total_loss.shape) > 0:
            total_loss = ops.mean(total_loss)
        self.zero_grad()
        total_loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            # Keras 3 expects (gradient, variable) pairs
            grads_and_vars = list(zip(gradients, trainable_weights, strict=False))
            self.optimizer.apply_gradients(grads_and_vars)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step_jax(self, jax, data: tsgm.types.Tensor) -> dict[str, float]:
        X, labels = data
        encoder_input = self._get_encoder_input(X, labels)
        z_mean, z_log_var, z = self.encoder(encoder_input)
        z_log_var = ops.clip(z_log_var, -6.0, 2.0)
        decoder_input = self._get_decoder_input(z, labels)
        reconstruction = self.decoder(decoder_input)

        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)
        kl_loss = self._compute_kl_loss(z_mean, z_log_var)
        total_loss = reconstruction_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step(self, data: tsgm.types.Tensor) -> dict[str, float]:
        """
        Performs a training step using a batch of data, stored in data.

        :param data: A batch of data in a format batch_size x seq_len x feat_dim
        :type data: tsgm.types.Tensor

        :returns: A dict with losses
        :rtype: T.Dict[str, float]
        """
        backend = get_backend()
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            return self.train_step_tf(backend, data)
        elif os.environ.get("KERAS_BACKEND") == "torch":
            return self.train_step_torch(backend, data)
        elif os.environ.get("KERAS_BACKEND") == "jax":
            return self.train_step_jax(backend, data)


class cBetaVAE_Hierarchical(keras.Model):
    """
    Conditional Beta-VAE with hierarchical latent codes.

    z_g : (batch, global_latent_dim) week-level summary
    z_l : (batch, seq_len, local_latent_dim) hour-level residuals

    Training data: (X, y_compact) where y_compact is (batch, 6).
    """

    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        cond_layer: CellConditioning,
        global_latent_dim: int,
        local_latent_dim: int,
        seq_len: int,
        beta: float = 0.0,
        free_bits_global: float = 0.1,
        free_bits_local: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.cond_layer = cond_layer
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self._seq_len = seq_len
        self.beta = beta
        self.free_bits_global = free_bits_global
        self.free_bits_local = free_bits_local

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> list:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def _encode(self, x: tsgm.types.Tensor, y: tsgm.types.Tensor, training: bool) -> tuple:
        out = self.encoder([x, y], training=training)
        if self.local_latent_dim == 0:
            z_g_mean, z_g_log_var, z_g = out
            return z_g_mean, z_g_log_var, None, None, z_g, None
        z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l = out
        return z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l

    def call(self, data: tsgm.types.Tensor) -> tsgm.types.Tensor:
        if not isinstance(data, (list, tuple)) or len(data) != 2:  # noqa
            return data
        x, y = data
        *_, z_g, z_l = self._encode(x, y, training=False)
        return self.decoder(self._build_decoder_input(z_g, z_l, y), training=False)

    def generate(self, y_compact: tsgm.types.Tensor) -> tuple[tsgm.types.Tensor, tsgm.types.Tensor]:
        batch_size = ops.shape(y_compact)[0]
        dtype = "float32" if os.environ.get("KERAS_BACKEND") == "torch" else y_compact.dtype
        z_g = keras.random.normal((batch_size, self.global_latent_dim), dtype=dtype)
        z_l = None
        if self.local_latent_dim > 0:
            z_l = keras.random.normal(
                (batch_size, self._seq_len, self.local_latent_dim), dtype=dtype
            )
        dec_in = self._build_decoder_input(z_g, z_l, y_compact)
        return self.decoder(dec_in, training=False), y_compact

    def _build_decoder_input(
        self,
        z_g: tsgm.types.Tensor,
        z_l: tsgm.types.Tensor | None,
        y_compact: tsgm.types.Tensor,
    ) -> list[tsgm.types.Tensor]:
        """z_g seeds the LSTM; per-step cond carries cell embed + calendar context."""
        cond_rep = self.cond_layer(y_compact)
        if self.local_latent_dim > 0 and z_l is not None:
            cond_rep = ops.concatenate([cond_rep, z_l], axis=-1)
        return [z_g, cond_rep]

    def _get_reconstruction_loss(self, x: tsgm.types.Tensor, x_hat: tsgm.types.Tensor) -> float:
        return ops.mean(ops.square(x - x_hat))

    def _kl_loss(
        self,
        mean: tsgm.types.Tensor,
        log_var: tsgm.types.Tensor,
        free_bits: float,
    ) -> float:
        if len(mean.shape) > 2:
            mean = ops.reshape(mean, (ops.shape(mean)[0], -1))
            log_var = ops.reshape(log_var, (ops.shape(log_var)[0], -1))
        log_var = ops.clip(log_var, -6.0, 2.0)
        kl_per_dim = -0.5 * (1 + log_var - ops.square(mean) - ops.exp(log_var))
        if free_bits > 0.0:
            kl_per_dim = ops.maximum(kl_per_dim, free_bits)
        return ops.mean(ops.sum(kl_per_dim, axis=-1))

    def _compute_losses(
        self,
        x: tsgm.types.Tensor,
        y: tsgm.types.Tensor,
        z_g_mean: tsgm.types.Tensor,
        z_g_log_var: tsgm.types.Tensor,
        z_l_mean: tsgm.types.Tensor,
        z_l_log_var: tsgm.types.Tensor,
        z_g: tsgm.types.Tensor,
        z_l: tsgm.types.Tensor,
    ) -> tuple[float, float, float]:
        dec_in = self._build_decoder_input(z_g, z_l, y)
        x_hat = self.decoder(dec_in, training=True)
        reconstruction_loss = self._get_reconstruction_loss(x, x_hat)
        kl_g = self._kl_loss(z_g_mean, z_g_log_var, self.free_bits_global)
        kl_l = 0.0
        if self.local_latent_dim > 0 and z_l_mean is not None:
            kl_l = self._kl_loss(z_l_mean, z_l_log_var, self.free_bits_local)
        kl_loss = kl_g + kl_l
        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step_tf(self, tf, data: tsgm.types.Tensor) -> dict[str, float]:
        x, y = data
        with tf.GradientTape() as tape:
            z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l = self._encode(
                x, y, training=True
            )
            total_loss, reconstruction_loss, kl_loss = self._compute_losses(
                x, y, z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l
            )
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights, strict=False))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step_torch(self, torch, data: tsgm.types.Tensor) -> dict[str, float]:
        x, y = data
        z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l = self._encode(x, y, training=True)
        total_loss, reconstruction_loss, kl_loss = self._compute_losses(
            x, y, z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l
        )
        if hasattr(total_loss, "shape") and len(total_loss.shape) > 0:
            total_loss = ops.mean(total_loss)
        self.zero_grad()
        total_loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply_gradients(list(zip(gradients, trainable_weights, strict=False)))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step_jax(self, jax, data: tsgm.types.Tensor) -> dict[str, float]:
        x, y = data
        z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l = self._encode(x, y, training=True)
        total_loss, reconstruction_loss, kl_loss = self._compute_losses(
            x, y, z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l
        )
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step(self, data: tsgm.types.Tensor) -> dict[str, float]:
        backend = get_backend()
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            return self.train_step_tf(backend, data)
        elif os.environ.get("KERAS_BACKEND") == "torch":
            return self.train_step_torch(backend, data)
        elif os.environ.get("KERAS_BACKEND") == "jax":
            return self.train_step_jax(backend, data)


class Architecture(abc.ABC):
    @abc.abstractproperty
    def arch_type(self):
        raise NotImplementedError


class Sampling(keras.layers.Layer):
    """
    Custom Keras layer for sampling from a latent space.

    This layer samples from a latent space using the reparameterization trick during training.
    It takes as input the mean and log variance of the latent distribution and generates
    samples by adding random noise scaled by the standard deviation to the mean.
    """

    def call(self, inputs: tuple[tsgm.types.Tensor, tsgm.types.Tensor]) -> tsgm.types.Tensor:
        """
        Generates samples from a latent space.

        :param inputs: Tuple containing mean and log variance tensors of the latent distribution.
        :type inputs: tuple[tsgm.types.Tensor, tsgm.types.Tensor]

        :returns: Sampled latent vector.
        :rtype: tsgm.types.Tensor
        """
        z_mean, z_log_var = inputs
        # Clamp log-variance to [-6, 2] (std in ~0.05–2.7) before sampling.
        # Without this, unconstrained z_log_var can grow large when beta is
        # low/zero during annealing, causing exp(0.5*z_log_var) to overflow and
        # the decoder to receive extreme out-of-distribution z inputs at
        # generation time, which produces random spikes in the output.
        z_log_var = ops.clip(z_log_var, -6.0, 2.0)
        #  random noise for keras3.0
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        #  ops for keras3.0
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class DualSampling(keras.layers.Layer):
    """
    Samples global z_g and local z_l from their respective mean/log_var heads.
    Inputs:
        z_g_mean, z_g_log_var : (B, G)
        z_l_mean, z_l_log_var : (B, T, L)
    Outputs:
        z_g : (B, G)
        z_l : (B, T, L)
    """

    def call(self, inputs):
        z_g_mean, z_g_log_var, z_l_mean, z_l_log_var = inputs
        z_g_log_var = ops.clip(z_g_log_var, -6.0, 2.0)
        z_l_log_var = ops.clip(z_l_log_var, -6.0, 2.0)
        eps_g = keras.random.normal(ops.shape(z_g_mean))
        eps_l = keras.random.normal(ops.shape(z_l_mean))
        z_g = z_g_mean + ops.exp(0.5 * z_g_log_var) * eps_g
        z_l = z_l_mean + ops.exp(0.5 * z_l_log_var) * eps_l
        return z_g, z_l


class BaseVAEArchitecture(Architecture):
    """
    Base class for defining architectures of Variational Autoencoders (VAEs).
    """

    @property
    def encoder(self) -> keras.models.Model:
        """
        Property for accessing the encoder model.

        :return: The encoder model.
        :rtype: keras.models.Model
        :raises NotImplementedError: If the encoder model is not implemented.
        """
        if hasattr(self, "_encoder"):
            return self._encoder
        else:
            raise NotImplementedError

    @property
    def decoder(self) -> keras.models.Model:
        """
        Property for accessing the decoder model.

        :return: The decoder model.
        :rtype: keras.models.Model
        :raises NotImplementedError: If the decoder model is not implemented.
        """
        if hasattr(self, "_decoder"):
            return self._decoder
        else:
            raise NotImplementedError

    def get(self) -> dict:
        """
        Retrieves both encoder and decoder models as a dictionary.

        :return: A dictionary containing encoder and decoder models.
        :rtype: dict
        :raises NotImplementedError: If either encoder or decoder models are not implemented.
        """
        if hasattr(self, "_encoder") and hasattr(self, "_decoder"):
            return {"encoder": self._encoder, "decoder": self._decoder}
        else:
            raise NotImplementedError


class VAE_LSTMArchitecture(BaseVAEArchitecture):
    """
    Variational Autoencoder with bidirectional LSTM encoder and LSTM decoder.

    Better suited than Conv1D for long sequences (e.g. 168-hour telecom KPI windows)
    where temporal dependencies span the full sequence length.
    """

    arch_type = "vae:unconditional"

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder(self) -> keras.models.Model:
        encoder_inputs = keras.Input(shape=(self._seq_len, self._feat_dim))
        x = encoder_inputs
        for _ in range(self._n_layers - 1):
            x = layers.Bidirectional(layers.LSTM(self._hidden_dim, return_sequences=True))(x)
            x = layers.Dropout(rate=0.2)(x)
        x = layers.Bidirectional(layers.LSTM(self._hidden_dim, return_sequences=False))(x)
        z_mean = layers.Dense(self._latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self._latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self) -> keras.models.Model:
        latent_inputs = keras.Input(shape=(self._latent_dim,))
        x = layers.RepeatVector(self._seq_len)(latent_inputs)
        for _ in range(self._n_layers):
            x = layers.LSTM(self._hidden_dim, return_sequences=True)(x)
            x = layers.Dropout(rate=0.2)(x)
        decoder_outputs = layers.TimeDistributed(
            layers.Dense(self._feat_dim, activation="sigmoid")
        )(x)
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")


def _funnel_schedule(
    hidden_dim: int,
    n_layers: int,
    min_units: int = 32,
    max_dropout: float = 0.3,
) -> tuple[list[int], list[float]]:
    """
    Per-layer LSTM units and dropout rates for a funnel encoder/decoder.

    Units shrink each layer (hidden_dim // depth), floored at min_units.
    Dropout rises by 0.1 per layer, capped at max_dropout.

    Example (hidden_dim=256, n_layers=2):
        units = [256, 128], dropouts = [0.1, 0.2]
    """
    units = [max(hidden_dim // (i + 1), min_units) for i in range(n_layers)]
    dropouts = [min((i + 1) * 0.1, max_dropout) for i in range(n_layers)]
    return units, dropouts


class cVAE_LSTMArchitecture(BaseVAEArchitecture):
    """
    Conditional VAE with bidirectional LSTM encoder and LSTM decoder.

    Compatible with cBetaVAE. Per-timestep latent of size latent_dim;
    total latent vector size is latent_dim * seq_len.

    Uses a funnel: LSTM units shrink each layer, dropout rises each layer.
    """

    arch_type = "vae:conditional"

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 2,
        min_units: int = 32,
        max_dropout: float = 0.3,
    ) -> None:
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._layer_units, self._layer_dropouts = _funnel_schedule(
            hidden_dim, n_layers, min_units, max_dropout
        )
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder(self) -> keras.models.Model:
        encoder_inputs = keras.Input(shape=(self._seq_len, self._feat_dim + self._output_dim))
        x = encoder_inputs
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
            x = layers.Dropout(rate=drop)(x)
        x = layers.TimeDistributed(layers.Dense(self._latent_dim, activation="relu"))(x)
        x = layers.Flatten()(x)
        latent_size = self._latent_dim * self._seq_len
        z_mean = layers.Dense(latent_size, name="z_mean")(x)
        z_log_var = layers.Dense(latent_size, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self) -> keras.models.Model:
        inputs = keras.Input(shape=(self._seq_len, self._latent_dim + self._output_dim))
        x = inputs
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.Dropout(rate=drop)(x)
        d_output = layers.TimeDistributed(layers.Dense(self._feat_dim, activation="sigmoid"))(x)
        return keras.Model(inputs, d_output, name="decoder")


class HourlyPositionalEncoding(keras.layers.Layer):
    """
    Sinusoidal positional encoding tuned to hourly windows.

    Concatenates 4 fixed channels to the last dimension of the input:
        sin(2π·t/24),  cos(2π·t/24)   — hour-of-day cycle (period = 24 h)
        sin(2π·t/168), cos(2π·t/168)  — day-of-week cycle (period = 168 h)

    This gives the LSTM and attention head explicit knowledge of where each
    timestep sits within the daily and weekly cycle, which is the dominant
    structure in cellular network KPIs.  Works regardless of the window stride
    used to build training samples (e.g. stride=24 h).

    Input shape:  (batch, seq_len, features)
    Output shape: (batch, seq_len, features + 4)
    """

    def __init__(self, seq_len: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seq_len = seq_len
        t = ops.arange(seq_len, dtype="float32")
        sin_24 = ops.sin(2.0 * math.pi * t / 24.0)
        cos_24 = ops.cos(2.0 * math.pi * t / 24.0)
        sin_168 = ops.sin(2.0 * math.pi * t / 168.0)
        cos_168 = ops.cos(2.0 * math.pi * t / 168.0)
        # shape: (seq_len, 4) — constant, no trainable weights
        self._pe = ops.stack([sin_24, cos_24, sin_168, cos_168], axis=-1)

    def call(self, x: tsgm.types.Tensor) -> tsgm.types.Tensor:
        batch = ops.shape(x)[0]
        pe = ops.broadcast_to(self._pe[None, :, :], (batch, self._seq_len, 4))
        return ops.concatenate([x, pe], axis=-1)

    def get_config(self) -> dict:
        return {**super().get_config(), "seq_len": self._seq_len}


class cVAE_LSTMv2Architecture(BaseVAEArchitecture):
    """
    Improved conditional VAE with bidirectional LSTM encoder and LSTM decoder.

    Enhancements over cVAE_LSTMArchitecture:
    - LayerNormalization after each BiLSTM/LSTM layer.
    - Optional temporal self-attention block in the encoder after the LSTM stack.
    - Funnel hidden units (hidden_dim // depth) with rising dropout per layer.
    - Attention key_dim computed from actual last BiLSTM output width.

    Recommended defaults for a 239-KPI, 168-step, multi-class telecom dataset:
        latent_dim=32, hidden_dim=256, n_layers=2, use_attention=True
    """

    arch_type = "vae:conditional_v2"

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_attention: bool = True,
        n_heads: int = 4,
        min_units: int = 32,
        max_dropout: float = 0.3,
    ) -> None:
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._use_attention = use_attention
        self._n_heads = n_heads
        self._layer_units, self._layer_dropouts = _funnel_schedule(
            hidden_dim, n_layers, min_units, max_dropout
        )
        self._attn_key_dim = max((2 * self._layer_units[-1]) // n_heads, 1)
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder(self) -> keras.models.Model:
        encoder_inputs = keras.Input(shape=(self._seq_len, self._feat_dim + self._output_dim))
        x = encoder_inputs
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)

        if self._use_attention:
            attn_out = layers.MultiHeadAttention(
                num_heads=self._n_heads,
                key_dim=self._attn_key_dim,
                dropout=0.1,
            )(x, x)
            x = layers.Add()([x, attn_out])
            x = layers.LayerNormalization()(x)

        x = layers.TimeDistributed(layers.Dense(self._latent_dim, activation="relu"))(x)
        x = layers.Flatten()(x)
        latent_size = self._latent_dim * self._seq_len
        z_mean = layers.Dense(latent_size, name="z_mean")(x)
        z_log_var = layers.Dense(latent_size, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self) -> keras.models.Model:
        inputs = keras.Input(shape=(self._seq_len, self._latent_dim + self._output_dim))
        x = inputs
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)
        d_output = layers.TimeDistributed(layers.Dense(self._feat_dim, activation="sigmoid"))(x)
        return keras.Model(inputs, d_output, name="decoder")


class cVAE_LSTMv3Architecture(BaseVAEArchitecture):
    """
    Improved conditional VAE with bidirectional LSTM encoder and LSTM decoder.

    Enhancements over cVAE_LSTMv2Architecture:
    - HourlyPositionalEncoding prepended to both encoder and decoder inputs.
    - Funnel hidden units with rising dropout per layer.
    - Attention key_dim computed from actual last BiLSTM output width.

    Recommended defaults for a 239-KPI, 168-step, multi-class telecom dataset
    with stride=24 windowing:
        latent_dim=32, hidden_dim=256, n_layers=2, use_attention=True, n_heads=4
        → layer units: [256, 128], dropouts: [0.1, 0.2], key_dim: 64
    """

    arch_type = "vae:conditional_v3"

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_attention: bool = True,
        n_heads: int = 4,
        min_units: int = 32,
        max_dropout: float = 0.3,
    ) -> None:
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._use_attention = use_attention
        self._n_heads = n_heads
        self._layer_units, self._layer_dropouts = _funnel_schedule(
            hidden_dim, n_layers, min_units, max_dropout
        )
        self._attn_key_dim = max((2 * self._layer_units[-1]) // n_heads, 1)
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder(self) -> keras.models.Model:
        encoder_inputs = keras.Input(shape=(self._seq_len, self._feat_dim + self._output_dim))
        x = HourlyPositionalEncoding(self._seq_len)(encoder_inputs)
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)

        if self._use_attention:
            attn_out = layers.MultiHeadAttention(
                num_heads=self._n_heads,
                key_dim=self._attn_key_dim,
                dropout=0.1,
            )(x, x)
            x = layers.Add()([x, attn_out])
            x = layers.LayerNormalization()(x)

        x = layers.TimeDistributed(layers.Dense(self._latent_dim, activation="relu"))(x)
        x = layers.Flatten()(x)
        latent_size = self._latent_dim * self._seq_len
        z_mean = layers.Dense(latent_size, name="z_mean")(x)
        z_log_var = layers.Dense(latent_size, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self) -> keras.models.Model:
        inputs = keras.Input(shape=(self._seq_len, self._latent_dim + self._output_dim))
        x = HourlyPositionalEncoding(self._seq_len)(inputs)
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)
        d_output = layers.TimeDistributed(layers.Dense(self._feat_dim, activation="sigmoid"))(x)
        return keras.Model(inputs, d_output, name="decoder")


class cVAE_LSTMv4Architecture(BaseVAEArchitecture):
    """
    Global-z conditional VAE for telecom 168-step KPI windows.

    Core change over v3: the encoder compresses the full sequence into a single
    global latent vector z ∈ R^latent_dim via GlobalAveragePooling1D, rather
    than a per-timestep vector z ∈ R^(latent_dim × seq_len).

    Why this matters
    ----------------
    v3 KL is summed over latent_dim × seq_len = 32 × 168 = 5 376 dimensions.
    Even at beta = 1.0, the KL pressure is enormous and drives posterior
    collapse within the first 20 epochs.  v4 KL is summed over latent_dim = 64
    dimensions — an 84× reduction — so the encoder can maintain an informative
    posterior.  Combine this with slow KL annealing (target_beta ≤ 0.1,
    anneal_epochs ≥ 80) and free-bits (free_bits = 0.5 nats per dim) in
    cBetaVAE and the model will actually use z.

    Must be used with ``cBetaVAE(global_z=True)``.  The cBetaVAE tiles z from
    (batch, latent_dim) → (batch, seq_len, latent_dim) before concatenating
    with the per-step label broadcast and feeding the decoder.

    HourlyPositionalEncoding (24h + 168h sinusoids) is applied to both
    encoder and decoder inputs so the LSTM always knows its position within
    the daily/weekly telecom cycle.

    Recommended defaults for a 239-KPI, 168-step, multi-class telecom dataset:
        latent_dim=64, hidden_dim=256, n_layers=2, use_attention=True, n_heads=4
        cBetaVAE(global_z=True, free_bits=0.5)
        target_beta=0.1, anneal_epochs=80
    """

    arch_type = "vae:conditional_v4"

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_attention: bool = True,
        n_heads: int = 4,
        min_units: int = 32,
        max_dropout: float = 0.3,
    ) -> None:
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._use_attention = use_attention
        self._n_heads = n_heads
        self._layer_units, self._layer_dropouts = _funnel_schedule(
            hidden_dim, n_layers, min_units, max_dropout
        )
        self._attn_key_dim = max((2 * self._layer_units[-1]) // n_heads, 1)
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder(self) -> keras.models.Model:
        encoder_inputs = keras.Input(shape=(self._seq_len, self._feat_dim + self._output_dim))
        x = HourlyPositionalEncoding(self._seq_len)(encoder_inputs)
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)

        if self._use_attention:
            attn_out = layers.MultiHeadAttention(
                num_heads=self._n_heads,
                key_dim=self._attn_key_dim,
                dropout=0.1,
            )(x, x)
            x = layers.Add()([x, attn_out])
            x = layers.LayerNormalization()(x)

        # GlobalAveragePooling1D → single latent vector (not per-timestep).
        # This is the key change: z ∈ R^latent_dim instead of R^(latent_dim*seq_len).
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self._latent_dim * 2, activation="relu")(x)
        z_mean = layers.Dense(self._latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self._latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self) -> keras.models.Model:
        # cBetaVAE(global_z=True) tiles z from (batch, latent_dim) to
        # (batch, seq_len, latent_dim) before concatenating with labels.
        inputs = keras.Input(shape=(self._seq_len, self._latent_dim + self._output_dim))
        x = HourlyPositionalEncoding(self._seq_len)(inputs)
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)
        d_output = layers.TimeDistributed(layers.Dense(self._feat_dim, activation="sigmoid"))(x)
        return keras.Model(inputs, d_output, name="decoder")


class cVAE_LSTMv5Architecture(BaseVAEArchitecture):
    """
    v5: cell embedding + global latent (optional per-hour local latent).

    Decoder uses z_g as LSTM initial state (not tiled).  Per-step decoder input
    is conditioning only (+ optional z_l), so the LSTM can unroll temporally.

    Recommended start: local_latent_dim=0, global_latent_dim=64,
    free_bits_global=0.002, target_beta=2e-4, cyclical KL annealing.
    """

    arch_type = "vae:conditional_v5"

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        n_cells: int,
        global_latent_dim: int = 64,
        local_latent_dim: int = 0,
        cell_embed_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_attention: bool = True,
        n_heads: int = 4,
        min_units: int = 32,
        max_dropout: float = 0.3,
        output_activation: str = "sigmoid",  # use "sigmoid" for min-max [0,1] data
    ):
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._n_cells = n_cells
        self._global_latent_dim = global_latent_dim
        self._local_latent_dim = local_latent_dim
        self._cell_embed_dim = cell_embed_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._use_attention = use_attention
        self._n_heads = n_heads
        self._output_activation = output_activation
        self._layer_units, self._layer_dropouts = _funnel_schedule(
            hidden_dim, n_layers, min_units, max_dropout
        )
        self._attn_key_dim = max((2 * self._layer_units[-1]) // n_heads, 1)
        self._cond_layer = CellConditioning(n_cells, cell_embed_dim, seq_len)
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    @property
    def cond_layer(self) -> CellConditioning:
        return self._cond_layer

    def _build_encoder(self) -> keras.Model:
        # Inputs
        x_in = keras.Input(shape=(self._seq_len, self._feat_dim), name="x")  # (B,T,F)
        y_in = keras.Input(shape=(6,), name="y_compact")  # (B,6)
        # Conditioning → (B, T, E+5)
        cond_rep = self._cond_layer(y_in)  # (B,T,37)
        # Concat KPIs + conditioning, add positional encoding
        enc_in = ops.concatenate([x_in, cond_rep], axis=-1)  # (B,T,276)
        x = HourlyPositionalEncoding(self._seq_len)(enc_in)  # (B,T,280)
        # BiLSTM stack
        for units, drop in zip(self._layer_units, self._layer_dropouts, strict=False):
            x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)
        h_seq = x  # (B, T, 2*units_last) e.g. (B, 128, 512)
        if self._use_attention:
            attn_out = layers.MultiHeadAttention(
                num_heads=self._n_heads,
                key_dim=self._attn_key_dim,
                dropout=0.1,
            )(h_seq, h_seq)
            h_seq = layers.LayerNormalization()(h_seq + attn_out)
        h_global = layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(h_seq)
        h_global = layers.Dense(self._global_latent_dim * 2, activation="relu")(h_global)
        z_g_mean = layers.Dense(self._global_latent_dim, name="z_g_mean")(h_global)
        z_g_log_var = layers.Dense(self._global_latent_dim, name="z_g_log_var")(h_global)

        if self._local_latent_dim > 0:
            z_l_mean = layers.TimeDistributed(
                layers.Dense(self._local_latent_dim), name="z_l_mean"
            )(h_seq)
            z_l_log_var = layers.TimeDistributed(
                layers.Dense(self._local_latent_dim), name="z_l_log_var"
            )(h_seq)
            z_g, z_l = DualSampling()([z_g_mean, z_g_log_var, z_l_mean, z_l_log_var])
            return keras.Model(
                inputs=[x_in, y_in],
                outputs=[z_g_mean, z_g_log_var, z_l_mean, z_l_log_var, z_g, z_l],
                name="encoder_v5",
            )

        z_g = Sampling()([z_g_mean, z_g_log_var])
        return keras.Model(
            inputs=[x_in, y_in],
            outputs=[z_g_mean, z_g_log_var, z_g],
            name="encoder_v5",
        )

    def _build_decoder(self) -> keras.Model:
        z_in = keras.Input(shape=(self._global_latent_dim,), name="z_g")
        cond_dim = self._cell_embed_dim + 5 + self._local_latent_dim
        cond_in = keras.Input(shape=(self._seq_len, cond_dim), name="cond_seq")

        init_units = self._layer_units[0]
        h0 = layers.Dense(init_units, activation="tanh", name="dec_h0")(z_in)
        c0 = layers.Dense(init_units, activation="tanh", name="dec_c0")(z_in)

        x = HourlyPositionalEncoding(self._seq_len)(cond_in)
        for i, (units, drop) in enumerate(
            zip(self._layer_units, self._layer_dropouts, strict=False)
        ):
            if i == 0:
                x = layers.LSTM(units, return_sequences=True, name="dec_lstm_0")(
                    x, initial_state=[h0, c0]
                )
            else:
                x = layers.LSTM(units, return_sequences=True, name=f"dec_lstm_{i}")(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(rate=drop)(x)

        d_output = layers.TimeDistributed(
            layers.Dense(self._feat_dim, activation=self._output_activation)
        )(x)
        return keras.Model([z_in, cond_in], d_output, name="decoder_v5")
