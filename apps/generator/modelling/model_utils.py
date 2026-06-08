import abc
import math

import keras
import tsgm
from keras import layers, ops


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
