import abc
import math
import os

import keras
import tsgm
import tsgm.utils
from keras import layers, ops
from tsgm.backend import get_backend


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
    Sinusoidal positional encoding tuned to hourly telecom PM windows.

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
