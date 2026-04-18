# -*- coding: utf-8 -*-
"""Custom tf layers."""

import logging

import numpy as np
import tensorflow as tf

from phygnn.utilities.tf_utilities import idw_fill, mean_fill

logger = logging.getLogger(__name__)


def get_custom_layer_objects():
    """Get local custom layer classes for Keras deserialization."""
    return {
        name: obj
        for name, obj in globals().items()
        if isinstance(obj, type)
        and issubclass(obj, tf.keras.layers.Layer)
        and obj.__module__ == __name__
    }


def _register_custom_layer_objects():
    """Register local custom layers in Keras' global object registry."""
    registry = tf.keras.utils.get_custom_objects()
    register = getattr(tf.keras.utils, 'register_keras_serializable', None)

    for name, obj in get_custom_layer_objects().items():
        if register is not None:
            register(package='phygnn', name=name)(obj)
        registry[name] = obj
        registry[f'phygnn>{name}'] = obj


def _dot_product_attention(*args, **kwargs):
    """Call the Keras 3 fused dot-product attention op."""
    return tf.keras.ops.dot_product_attention(*args, **kwargs)


class FlexiblePadding(tf.keras.layers.Layer):
    """Class to perform padding on tensors"""

    def __init__(self, paddings, mode='REFLECT', option='tf', **kwargs):
        """
        Parameters
        ----------
        paddings : int array
            Integer array with shape [n,2] where n is the
            rank of the tensor and elements give the number
            of leading and trailing pads
        mode : str
            tf.pad() / np.pad() padding mode. Can be REFLECT, CONSTANT,
            or SYMMETRIC
        option : str
            Option for TensorFlow padding ("tf") or numpy ("np"). Default is tf
            for tensorflow training. We have observed silent failures of
            tf.pad() with larger array sizes, so "np" might be preferable at
            inference time on large chunks, but it is much slower when it has
            to convert tensors to numpy arrays. See the tensorflow issue
            https://github.com/tensorflow/tensorflow/issues/91027
        """
        super().__init__(**kwargs)
        self._paddings = tuple(
            tuple(int(value) for value in pad) for pad in paddings
        )
        self._mode = mode
        self._option = option
        self.paddings = tf.constant(self._paddings)
        self.rank = len(self._paddings)
        self.mode = mode.lower()
        self.option = option.lower()

        if self.option == 'tf':
            self._pad_fun = tf.pad
        elif self.option == 'np':
            self._pad_fun = np.pad
        else:
            msg = (
                'FlexiblePadding option must be "tf" or "np" but '
                f'received: {self.option}'
            )
            logger.error(msg)
            raise KeyError(msg)

    def compute_output_shape(self, input_shape):
        """Computes output shape after padding

        Parameters
        ----------
        input_shape : tuple
            shape of input tensor

        Returns
        -------
        output_shape : tf.TensorShape
            shape of padded tensor
        """
        output_shape = [0] * self.rank
        for d in range(self.rank):
            output_shape[d] = (
                None
                if input_shape[d] is None
                else sum(self._paddings[d]) + input_shape[d]
            )
        return tf.TensorShape(output_shape)

    def call(self, x):
        """Calls the padding routine

        Parameters
        ----------
        x : tf.Tensor
            tensor on which to perform padding

        Returns
        -------
        x : tf.Tensor
            padded tensor with shape given
            by compute_output_shape

        """
        return self._pad_fun(x, self.paddings, mode=self.mode)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update(
            {
                'paddings': [list(pad) for pad in self._paddings],
                'mode': self._mode,
                'option': self._option,
            }
        )
        return config


class PatchLayer(tf.keras.layers.Layer):
    """Layer with patchification functionality."""

    def __init__(self, name=None, patch_size=1):
        """Initialize the PatchLayer layer.

        Parameters
        ----------
        name : str | None
            Name of layer.
        patch_size : int
            Height, width, and depth of tokens. Default is 1 for pixel-wise
            tokenization.
        """
        super().__init__(name=name)
        self.patch_size = patch_size
        self.rank = None

    def _mask(self, x, out):
        """Helper function to mask output tokens based on NaN values in the
        input tensor.
        """
        if self.patch_size == 1:
            nan_any = tf.math.reduce_any(tf.math.is_nan(x), axis=-1)
            valid_mask = tf.math.logical_not(nan_any)
            out = tf.boolean_mask(out, valid_mask)
        tf.debugging.assert_all_finite(
            out, message='Masked output contains NaN or Inf values.'
        )
        return out

    def build(self, input_shape):
        """Build the PatchLayer layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        self.rank = len(input_shape)

        if self.rank not in {4, 5}:
            msg = (
                f'Input tensor must be 4D or 5D, but got {self.rank}D tensor.'
            )
            logger.error(msg)
            raise ValueError(msg)

    @classmethod
    def _get_padding(cls, x_shape, patch_size=1):
        """Helper function to get the padding for the input tensor based on the
        patch size. This is necessary to ensure that the spatial dimensions of
        the input tensor are divisible by the patch size for tokenization.

        Parameters
        ----------
        x_shape : tf.TensorShape
            Shape of the unpadded 4D or 5D input tensor.
        patch_size : int
            Height, width, and depth of patches. This is used to determine the
            amount of padding needed for each spatial dimension. Default is 1
            for pixel-wise tokenization.
        """
        pads = [[0, 0]]  # batch
        for i in range(1, len(x_shape) - 1):
            rem = x_shape[i] % patch_size
            pad = (patch_size - rem) % patch_size
            pads.append([pad // 2, pad - pad // 2])
        pads.append([0, 0])  # features
        return pads

    @classmethod
    def pad(cls, x, patch_size=1):
        """Pad spatial dimensions of ``x`` so they are evenly
        divisible by ``patch_size``.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor.
        patch_size : int
            Height, width, and depth of patches.

        Returns
        -------
        x_padded : tf.Tensor
            Tensor with each grid axis reflection padded so that they are
            evenly divisible by ``patch_size``.
        """
        if patch_size == 1:
            return x
        pads = cls._get_padding(tf.shape(x), patch_size=patch_size)
        return tf.pad(x, pads, mode='reflect')

    @classmethod
    def crop(cls, x_shape, out, patch_size=1):
        """Remove the padding added by :meth:`pad` so
        the output matches the original (unpadded) spatial shape.

        Parameters
        ----------
        x_shape : tf.TensorShape
            Shape of the original (unpadded) input tensor. This is used to get
            the original spatial shape to crop to.
        out : tf.Tensor
            Tensor whose spatial dims may be larger than target.
        patch_size : int
            Height, width, and depth of patches. This is used to determine the
            amount of padding that was added to the input tensor.

        Returns
        -------
        out : tf.Tensor
            Tensor cropped to x_shape.
        """
        if patch_size == 1:
            return out
        pads = cls._get_padding(x_shape, patch_size=patch_size)
        # Convert pads to a tensor so cropping works with dynamic shapes.
        pads_tensor = tf.convert_to_tensor(pads, dtype=tf.int32)
        # pads_tensor has shape [rank, 2]: [pad_before, pad_after] per axis.
        begin = pads_tensor[:, 0]
        end = pads_tensor[:, 1]
        input_shape = tf.shape(out)
        size = input_shape - begin - end
        return tf.slice(out, begin, size)


class Embedder(PatchLayer):
    """Embedding layer."""

    def __init__(self, name=None, patch_size=1, embed_dim=64):
        """Initialize the Embedding layer.

        Parameters
        ----------
        name : str | None
            Name of layer.
        patch_size : int
            Height, width, and depth of tokens. Default is 1 for pixel-wise
            tokenization.
        embed_dim : int
            Dimension of the embedding. This determines the size of the output
            tokens after tokenization. Default is 64.
        """
        super().__init__(name=name, patch_size=patch_size)
        self.embed_layer = None
        self.embed_dim = embed_dim
        self.rank = None

    def build(self, input_shape):
        """Build the Embedding layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        super().build(input_shape)
        if self.patch_size > 1:
            kwargs = {
                'kernel_size': [self.patch_size] * (self.rank - 2),
                'strides': [self.patch_size] * (self.rank - 2),
                'filters': self.embed_dim,
                'padding': 'valid',
            }
            self.embed_layer = (
                tf.keras.layers.Conv2D(**kwargs)
                if self.rank == 4
                else tf.keras.layers.Conv3D(**kwargs)
            )
        else:
            self.embed_layer = tf.keras.layers.Dense(
                self.embed_dim, use_bias=False
            )

    def call(self, x):
        """Embed inputs for attention blocks.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor. This can be sparse with some NaN values,
            or gapless.

        Returns
        -------
        x_emb : tf.Tensor
            Embedded tensor with shape (batch_size, n_tokens, embed_dim)
        """
        x_emb = self._mask(x, x)
        x_emb = self.embed_layer(x_emb)

        # batch members can have different NaN patterns in general so this
        # reshape could fail if batch_size > 1
        return tf.reshape(x_emb, (tf.shape(x)[0], -1, self.embed_dim))


class PositionEncoder(PatchLayer):
    """Positional encoding layer."""

    def __init__(
        self,
        name=None,
        patch_size=1,
        embed_dim=64,
        min_period_spatial=1e-4,
        max_period_spatial=2,
        min_period_temporal=1,
        max_period_temporal=864000,
    ):
        """Initialize the PositionEncoder layer.

        Parameters
        ----------
        name : str | None
            Name of layer.
        patch_size : int
            Height, width, and depth of patches. This is used to pool the
            positional encoding into the same patch shape as tokens. Default is
            1 for pixel-wise tokenization and encoding.
        embed_dim : int
            Dimension of the embedding. This determines the size of the output
            tokens after encoding. Default is 64.
        min_period_spatial : float
            Minimum period in degrees for the positional encoding. This is
            typically set to a value like 1e-5 to ensure that the positional
            encoding captures high frequency information.
        max_period_spatial : float
            Maximum period in degrees for the positional encoding. This is
            typically set to a value like 5 to ensure that the positional
            encoding captures low frequency information.
        min_period_temporal : float
            Minimum period in seconds for the positional encoding. This is
            typically set to a value like 1 to ensure that the positional
            encoding captures high frequency information.
        max_period_temporal : float
            Maximum period in seconds for the positional encoding. This is
            typically set to a value like 864000 to ensure that the positional
            encoding captures low frequency information.
        """
        super().__init__(name=name, patch_size=patch_size)
        self._pool_layer = None
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.min_period_spatial = min_period_spatial
        self.max_period_spatial = max_period_spatial
        self.min_period_temporal = min_period_temporal
        self.max_period_temporal = max_period_temporal
        self.rank = None

    @classmethod
    def _freq_encode(cls, k, min_period, max_period, d=64):
        """Helper function to create a frequency specified positional encoding
        for attention blocks.

        Parameters
        ----------
        k : tf.Tensor
            Tensor of positions to encode. This can be indices, latitudes,
            longitudes, times, etc. The "units" of k must be the same as
            min/max periods. Must have 4D or 5D shape (..., 1).
        min_period : float
            Minimum period (spatial or temporal) for the positional encoding.
        max_period : float
            Maximum period (spatial or temporal) for the positional encoding.
        d : int
            Dimension of the positional encoding. This should match the
            embed_dim of the attention block.
        """
        assert d % 2 == 0, (
            'Embedding dimension must be even for sin/cos encoding.'
        )
        min_freq = 2 * np.pi / max_period
        max_freq = 2 * np.pi / min_period
        freqs = tf.linspace(min_freq, max_freq, d // 2)
        theta = tf.cast(freqs, k.dtype) * k
        return tf.stack([tf.sin(theta), tf.cos(theta)], axis=-1)

    @staticmethod
    def _compute_doy_soy(time):
        """Compute day of year and second of year from unix timestamps.

        Parameters
        ----------
        time : np.ndarray
            Array of unix timestamps (seconds since epoch).

        Returns
        -------
        doy : np.ndarray
            Day of year as float32.
        soy : np.ndarray
            Second of year as float32.
        """
        dt = time.astype(np.int64).view('datetime64[s]')
        year_start = dt.astype('datetime64[Y]')
        doy = dt.astype('datetime64[D]') - year_start.astype('datetime64[D]')
        soy = dt - year_start.astype('datetime64[s]')
        return (
            (doy / np.timedelta64(1, 'D')).astype(np.float32),
            (soy / np.timedelta64(1, 's')).astype(np.float32),
        )

    def encode_lat_lon(self, x, lat, lon, min_period, max_period):
        """Sinusoidal positional encoding for latitude and longitude.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor to encode. Must have 4D or 5D shape (..., 1). This is
            only used to get the batch and spatial dimensions for the output
            encoding.
        lat : tf.Tensor
            Tensor of latitudes to encode. Must have 4D or 5D shape (..., 1).
            Latitude should be in degrees from -90 to 90.
        lon : tf.Tensor
            Tensor of longitudes to encode. Must have 4D or 5D shape (..., 1).
            Longitude should be in degrees from -180 to 180.
        min_period : float
            Minimum period in degrees for the positional encoding.
        max_period : float
            Maximum period in degrees for the positional encoding.

        Returns
        -------
        lat_lon_enc : tf.Tensor
            Positional encoding tensor for latitude and longitude with shape
            (..., d)
        """
        assert self.embed_dim % 4 == 0, (
            'Embedding dimension must be divisible by 4 for latitude and '
            'longitude encoding.'
        )
        lat_enc = self._freq_encode(
            lat,
            d=self.embed_dim // 2,
            min_period=min_period,
            max_period=max_period,
        )
        lon_enc = self._freq_encode(
            lon,
            d=self.embed_dim // 2,
            min_period=min_period,
            max_period=max_period,
        )
        out = tf.concat([lat_enc, lon_enc], axis=-1)
        out = self._mask(x, out)
        return tf.reshape(out, (tf.shape(x)[0], -1, self.embed_dim))

    def encode_time(self, x, time, min_period, max_period):
        """Sinusoidal positional encoding for time.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor to encode. Must have 4D or 5D shape (..., 1). This is
            only used to get the batch and spatial dimensions for the output
            encoding.
        time : tf.Tensor
            Tensor of datetime values to encode. Must have 4D or 5D shape (...,
            1).
        min_period : float
            Minimum period in seconds for the positional encoding.
        max_period : float
            Maximum period in seconds for the positional encoding.

        Returns
        -------
        time_enc : tf.Tensor
            Positional encoding tensor for time with shape (..., d)
        """
        assert self.embed_dim % 4 == 0, (
            'Embedding dimension must be divisible by 4 for time encoding.'
        )
        doy, soy = tf.numpy_function(
            self._compute_doy_soy, [time], [tf.float32, tf.float32]
        )
        doy = tf.reshape(doy, tf.shape(time))
        soy = tf.reshape(soy, tf.shape(time))
        min_period_doy = min_period / 86400  # convert seconds to days
        max_period_doy = max_period / 86400  # convert seconds to days
        doy_enc = self._freq_encode(
            doy, min_period_doy, max_period_doy, d=self.embed_dim // 2
        )
        soy_enc = self._freq_encode(
            soy, min_period, max_period, d=self.embed_dim // 2
        )
        out = tf.concat([doy_enc, soy_enc], axis=-1)
        out = self._mask(x, out)
        return tf.reshape(out, (tf.shape(x)[0], -1, self.embed_dim))

    def build(self, input_shape):
        """Build the Positional Encoding layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        super().build(input_shape)
        kwargs = {
            'pool_size': self.patch_size,
            'strides': self.patch_size,
            'padding': 'valid',
        }
        self._pool_layer = (
            tf.keras.layers.AveragePooling2D(**kwargs)
            if self.rank == 4
            else tf.keras.layers.AveragePooling3D(**kwargs)
        )

    def call(self, x, lat, lon, time=None):
        """Get positional encoding for attention blocks.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor. This can be sparse with some NaN values,
            or gapless.
        lat : tf.Tensor
            Tensor of latitudes for positional encoding. Must have 4D or 5D
            shape (..., 1).
        lon : tf.Tensor
            Tensor of longitudes for positional encoding. Must have 4D or 5D
            shape (..., 1).
        time : tf.Tensor | None
            Tensor of datetime values for positional encoding. Must have 4D or
            5D shape (..., 1). If None, time encoding will not be included.

        Returns
        -------
        x_enc : tf.Tensor
            Positional encoding tensor with shape (batch_size, n_tokens,
            embed_dim)
        """
        x_enc = self.encode_lat_lon(
            x, lat, lon, self.min_period_spatial, self.max_period_spatial
        )
        if self.rank == 5 and time is not None:
            x_enc += self.encode_time(
                x, time, self.min_period_temporal, self.max_period_temporal
            )
        return x_enc


class MultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """MultiHeadAttention that accepts an additive pre-softmax bias.

    This layer uses the same constructor arguments as
    ``keras.layers.MultiHeadAttention``. The only API extension is that
    ``call()`` accepts a ``bias`` keyword argument. The bias is added to the
    scaled QK^T logits before softmax and must broadcast onto
    ``(B, num_heads, T, S)``.

    Flash attention is used through ``tf.keras.ops.dot_product_attention()``
    when dropout is inactive for the current call and attention scores are not
    requested. The bias is forwarded to the fused op so ALiBi and other
    additive pre-softmax bias terms keep the same behavior.

    Example::
        layer = MultiHeadAttention(num_heads=8, key_dim=64)
        output = layer(query, value, bias=my_bias)
    """

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        bias=None,
    ):
        """Call multi-head attention with optional bias."""
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        # RaggedTensor handling (unchanged from base class)
        query_is_ragged = isinstance(query, tf.RaggedTensor)
        if query_is_ragged:
            query_lengths = query.nested_row_lengths()
            query = query.to_tensor()
        key_is_ragged = isinstance(key, tf.RaggedTensor)
        value_is_ragged = isinstance(value, tf.RaggedTensor)
        if key_is_ragged and value_is_ragged:
            bounding_shape = tf.math.maximum(
                key.bounding_shape(), value.bounding_shape()
            )
            key = key.to_tensor(shape=bounding_shape)
            value = value.to_tensor(shape=bounding_shape)
        elif key_is_ragged:
            key = key.to_tensor(shape=tf.shape(value))
        elif value_is_ragged:
            value = value.to_tensor(shape=tf.shape(key))

        attention_mask = self._compute_attention_mask(
            query,
            value,
            key=key,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            training=training,
            bias=bias,
            return_attention_scores=return_attention_scores,
        )
        attention_output = self._output_dense(attention_output)

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(
                attention_output, lengths=query_lengths
            )

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
        bias=None,
        return_attention_scores=False,
    ):
        use_fused_attention = (
            not return_attention_scores
            and (self._dropout == 0.0 or training is False)
        )

        if use_fused_attention:
            if attention_mask is not None:
                mask_expansion_axis = -len(self._attention_axes) * 2 - 1
                target_rank = len(query.shape)
                for _ in range(target_rank - len(attention_mask.shape)):
                    attention_mask = tf.expand_dims(
                        attention_mask, axis=mask_expansion_axis
                    )

            attention_output = _dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=None if bias is None else tf.cast(bias, query.dtype),
                mask=attention_mask,
                flash_attention=None,
            )
            return attention_output, None

        query = tf.multiply(query, 1.0 / tf.math.sqrt(float(self._key_dim)))

        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        if bias is not None:
            attention_scores = tf.add(
                attention_scores, tf.cast(bias, attention_scores.dtype)
            )

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


class TransformerLayer(tf.keras.layers.Layer):
    """Custom transformer layer with multi-head attention layer that allows
    for additive bias pre-softmax."""

    def __init__(self, num_heads, key_dim, attn_kwargs=None, **kwargs):
        """Initialize the transformer layer.

        Parameters
        ----------
        num_heads : int
            Number of attention heads.
        key_dim : int
            Size of each attention head.
        attn_kwargs : dict | None
            Additional keyword arguments forwarded to the internal
            :class:`MultiHeadAttention` layer.
        **kwargs
            Additional keyword arguments passed to ``tf.keras.layers.Layer``.
        """
        super().__init__(**kwargs)

        self.attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, **(attn_kwargs or {})
        )
        self.lq = tf.keras.layers.LayerNormalization()
        self.lk = tf.keras.layers.LayerNormalization()
        self.lv = tf.keras.layers.LayerNormalization()
        self.lo = tf.keras.layers.LayerNormalization()

        self._mlp_layers = [
            tf.keras.layers.Dense(key_dim, activation='relu'),
            tf.keras.layers.Dense(key_dim),
        ]

    def mlp(self, x):
        """Pass input through MLP layers."""
        for layer in self._mlp_layers:
            x = layer(x)
        return x

    def call(self, query, key, value, bias=None):
        """Call transformer layer with multi-head attention output.

        Parameters
        ----------
        query : tf.Tensor
            Query tensor with shape (batch_size, seq_q, features)
        key : tf.Tensor
            Key tensor with shape (batch_size, seq_k, features)
        value : tf.Tensor
            Value tensor with shape (batch_size, seq_v, features)
        bias : tf.Tensor | None
            Optional bias tensor to add to the attention scores before softmax.
            Must be broadcastable to shape (batch_size, num_heads, seq_q,
            seq_k).
        """
        q = self.lq(query)
        k = self.lk(key)
        v = self.lv(value)
        attn = self.attn(query=q, key=k, value=v, bias=bias)
        out = self.lo(query + attn)
        return query + self.mlp(out)


class Sup3rTransformerLayer(tf.keras.layers.Layer):
    """Custom layer to implement transformer layer with cross attention with
    tokenization and positional encoding. This is typically used for sparse
    observation data assimilation, but can also be used to attend to gapless
    data like topography. Queries are typically the latent space of the model
    and keys/values are the high-resolution features.

    Note: This layer assumes that any sparse input data with NaN values has
    NaNs for the same tokens across all features. If you want to attend to
    sparse data with different NaN patterns across features, you should
    use different attention layers for each feature or group of features with
    the same NaN pattern.
    """

    def __init__(
        self,
        name=None,
        features=None,
        exo_features=None,
        num_heads=1,
        key_dim=64,
        embed_dim=64,
        min_period_spatial=1e-4,
        max_period_spatial=2,
        min_period_temporal=1,
        max_period_temporal=864000,
        attn_kwargs=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str | None
            Name of layer.
        features : list[str] | None
            List of hi-resolution feature names.
        exo_features : list[str] | None
            List of exogenous feature names. These are features that will be
            used for positional encoding like latitude, longitude, and time.
        embed_dim : int
            Dimension of the tokenized inputs.
        num_heads : int
            Number of attention heads
        key_dim : int
            Size of each attention head
        min_period_spatial : float
            Minimum period for the spatial positional encoding.
        max_period_spatial : float
            Maximum period for the spatial positional encoding.
        min_period_temporal : float
            Minimum period for the temporal positional encoding.
        max_period_temporal : float
            Maximum period for the temporal positional encoding.
        attn_kwargs : dict | None
            Additional keyword arguments forwarded to the internal
            :class:`MultiHeadAttention` layer used by ``self.transformer``.
        **kwargs
             Additional keyword arguments to pass to the parent class. This can
             include arguments like trainable and dtype.
        """

        super().__init__(name=name, **kwargs)
        self.features = features or []
        self.exo_features = exo_features or []
        self.rank = None
        self.final_proj = None
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.eq = Embedder(embed_dim=embed_dim)
        self.ek = Embedder(embed_dim=embed_dim)
        self.ev = Embedder(embed_dim=embed_dim)
        self.pe = PositionEncoder(
            embed_dim=embed_dim,
            min_period_spatial=min_period_spatial,
            max_period_spatial=max_period_spatial,
            min_period_temporal=min_period_temporal,
            max_period_temporal=max_period_temporal,
        )
        self.transformer = TransformerLayer(
            key_dim=key_dim,
            num_heads=num_heads,
            attn_kwargs=attn_kwargs,
        )

    def build(self, input_shape):
        """Build the CrossAttentionBlock layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor.
        """
        self.rank = len(input_shape)
        msg = (
            'CrossAttentionBlock input must be 4D or 5D, but received input '
            f'shape: {input_shape}'
        )
        if self.rank not in {4, 5}:
            logger.error(msg)
            raise ValueError(msg)

        self.final_proj = tf.keras.layers.Dense(input_shape[-1])

    def _transformer_layer(self, x_in, hr_in, lat, lon, time=None):
        # embed query, key, and value inputs
        q = self.eq(x_in)
        k = self.ek(hr_in)
        v = self.ev(hr_in)

        # add positional encodings for query, key, and value inputs
        q += self.pe(x_in, lat=lat, lon=lon, time=time)
        k += self.pe(hr_in, lat=lat, lon=lon, time=time)
        v += self.pe(hr_in, lat=lat, lon=lon, time=time)

        out = self.transformer(query=q, key=k, value=v)
        out = self.final_proj(out)
        return tf.reshape(out, tf.shape(x_in))

    def _call(self, x, hi_res_feature, idx, lat, lon, time=None):
        """Call attention layer for a single batch member. This is necessary to
        handle different NaN patterns across batch members in the case of
        sparse observation data.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor. Typically this is the latent space tensor
            being updated by the attention block.
        hi_res_feature : tf.Tensor
            4D or 5D high resolution feature tensor. This will be used as the
            value input. This can be sparse observation data, possibly with
            some NaN values, or high-resolution gapless data like topography.
        idx : int
            Index of the batch member being processed.
        lat : tf.Tensor
            Latitude tensor for positional encoding
        lon : tf.Tensor
            Longitude tensor for positional encoding
        time : tf.Tensor, optional
            Time tensor for positional encoding. Default is None.

        Returns
        -------
        x : tf.Tensor
            Output tensor of the attention block.
        """
        x_in = x[idx : idx + 1]
        hr_in = hi_res_feature[idx : idx + 1]
        lat = lat[idx : idx + 1]
        lon = lon[idx : idx + 1]
        time = None if time is None else time[idx : idx + 1]

        if tf.math.reduce_all(tf.math.is_nan(hr_in)):
            return tf.squeeze(x_in, axis=0)

        out = self._transformer_layer(x_in, hr_in, lat=lat, lon=lon, time=time)

        tf.debugging.assert_all_finite(
            out, message='Attention output contains NaN or Inf values.'
        )

        return tf.squeeze(out, axis=0)

    def call(self, x, hi_res_feature=None, exo_data=None):
        """Call attention layer across batch dimension to handle different NaN
        patterns in the case of sparse observation data.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor. Typically this is the latent space tensor
            being updated by the attention block.
        hi_res_feature : tf.Tensor, optional
            4D or 5D high resolution feature tensor. This will be used as the
            value input. This can be sparse observation data, possibly with
            some NaN values, or high-resolution gapless data like topography.
        exo_data: tf.Tensor, optional
            4D or 5D tensor of features to use for positional encoding. If
            hi_res_feature is provided, this should must include latitude and
            longitude, and optionally time, in that order.  Latitude and
            longitude should be in degrees and time should be in a datetime
            format that can be parsed by
            tf.experimental.numpy.datetime_as_string.

        Returns
        -------
        x : tf.Tensor
            Output tensor of the attention block.
        """
        if hi_res_feature is None:
            return x

        lat = exo_data[..., 0:1]
        lon = exo_data[..., 1:2]
        time = (
            None
            if exo_data is None or exo_data.shape[-1] < 3
            else exo_data[..., 2:3]
        )

        out_spec = tf.TensorSpec(shape=x.shape[1:], dtype=x.dtype)
        return tf.map_fn(
            lambda i: self._call(
                x,
                hi_res_feature,
                lat=lat,
                lon=lon,
                time=time,
                idx=i,
            ),
            tf.range(tf.shape(x)[0]),
            fn_output_signature=out_spec,
        )


class Sup3rTransformerLayerAlibi(Sup3rTransformerLayer):
    """Transformer layer with attention layer with linear biases (ALiBi)
    instead of positional encoding. This adds a distance-based bias to the
    attention scores before softmax.

    References
    ----------
    Press, O., Smith, N. A., & Lewis, M. (2022). Train Short, Test Long:
    Attention with Linear Biases Enables Input Length Extrapolation.
    arXiv:2108.12409. https://arxiv.org/abs/2108.12409
    """

    def __init__(self, *args, sigma=0.01, trainable=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.trainable = trainable

        # Compute head slopes for the ALiBi bias based on the number of
        # attention heads. This follows the approach from the ALiBi paper to
        # give each head a different slope for the distance-based bias.
        x = 2 ** (8 / self.num_heads)
        slopes = np.array(
            [1 / (x ** (i + 1)) for i in range(self.num_heads)],
            dtype=np.float32,
        ).reshape(1, self.num_heads, 1, 1)
        self.head_slopes = tf.constant(slopes, dtype=tf.float32)

    def get_config(self):
        """Implementation of get_config method from tf.keras.layers.Layer for
        saving/loading as part of keras sequential model.

        Returns
        -------
        config : dict
        """
        config = super().get_config().copy()
        config.update({
            'trainable': self.trainable,
            'sigma': float(self.sigma),
        })
        return config

    def build(self, input_shape):
        """Build the Sup3rCrossAlibi layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor.
        """
        super().build(input_shape)
        self.sigma = self.add_weight(
            name='sigma',
            shape=[1],
            trainable=self.trainable,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self.sigma),
        )

    def get_locality_bias(self, x, hi_res_feature, lat, lon, time=None):
        """Helper function to compute a locality bias for the attention based
        on the haversine distance between the query and value tokens.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor. Typically this is the latent space tensor
            being updated by the attention block.
        hi_res_feature : tf.Tensor
            4D or 5D high resolution feature tensor. This will be used as the
            value input. This can be sparse observation data, possibly with
            some NaN values, or high-resolution gapless data like topography.
        lat : tf.Tensor
            Latitude tensor for positional encoding.
        lon : tf.Tensor
            Longitude tensor for positional encoding.
        time : tf.Tensor, optional
            Time tensor for positional encoding. Default is None.

        Returns
        -------
        locality_bias : tf.Tensor
            Tensor representing the locality bias for the attention mechanism.
            (batch_size, n_query_tokens, n_value_tokens, key_dim)
        """
        # Compute pairwise distances between query and value tokens based on
        # lat/lon. This assumes that the tokens are ordered in the same way as
        # the spatial dimensions of the input tensors.

        lat_q = tf.reshape(lat, (tf.shape(lat)[0], -1, 1))
        lon_q = tf.reshape(lon, (tf.shape(lon)[0], -1, 1))

        lat_v = self.ev._mask(hi_res_feature, lat)
        lon_v = self.ev._mask(hi_res_feature, lon)
        lat_v = tf.reshape(lat_v, (tf.shape(lat)[0], 1, -1))
        lon_v = tf.reshape(lon_v, (tf.shape(lon)[0], 1, -1))

        lat_q_rad = lat_q * (np.pi / 180.0)
        lon_q_rad = lon_q * (np.pi / 180.0)
        lat_v_rad = lat_v * (np.pi / 180.0)
        lon_v_rad = lon_v * (np.pi / 180.0)

        dlat = lat_q_rad - lat_v_rad
        dlon = lon_q_rad - lon_v_rad
        a = (
            tf.sin(dlat / 2) ** 2
            + tf.cos(lat_q_rad) * tf.cos(lat_v_rad) * tf.sin(dlon / 2) ** 2
        )
        distance = 2 * tf.asin(tf.sqrt(a))
        bias = -(distance**2) / (2 * self.sigma**2)

        bias = tf.expand_dims(bias, axis=1)
        bias = tf.repeat(bias, repeats=self.num_heads, axis=1)
        bias *= self.head_slopes
        return bias

    def _transformer_layer(self, x_in, hr_in, lat, lon, time=None):
        # embed query, key, and value inputs
        q = self.eq(x_in)
        k = self.ek(hr_in)
        v = self.ev(hr_in)

        # use locality bias instead of positional encodings
        bias = self.get_locality_bias(x_in, hr_in, lat=lat, lon=lon, time=time)
        out = self.transformer(query=q, key=k, value=v, bias=bias)
        out = self.final_proj(out)
        return tf.reshape(out, tf.shape(x_in))


class Sup3rTransformerBlock(tf.keras.layers.Layer):
    """Custom layer to implement a block of Sup3rTransformerLayer layers."""

    def __init__(
        self,
        name=None,
        features=None,
        exo_features=None,
        num_heads=1,
        key_dim=64,
        embed_dim=64,
        use_alibi=False,
        transformer_kwargs=None,
        attn_kwargs=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str | None
            Name of layer.
        features : list[str] | None
            List of hi-resolution feature names. The length of this list
            determines the number of Sup3rTransformerLayer layers in the block.
            Each layer will attend to the corresponding feature in the list.
            For example, if features=['obs', 'topography'] then the first layer
            will attend to the 'obs' feature and the second layer will attend
            to the 'topography' feature. If None, no layers will be created.
        exo_features : list[str] | None
            List of exogenous feature names. These are features that will be
            used for positional encoding like latitude, longitude, and time.
            This will be used for all layers in the block. If None, no
            exogenous features will be used for positional encoding.
        num_heads : int
            Number of attention heads for each transformer layer in the block.
        key_dim : int
            Size of each attention head for each transformer layer in the
            block.
        embed_dim : int
            Dimension of the tokenized inputs for each transformer layer in the
            block. This matches the embed_dim used for the positional encoding.
        use_alibi : bool
            Whether to use ALiBi (Attention with Linear Biases) instead of
            positional encoding. If True, the Sup3rTransformerLayerAlibi class
            will be used for the layers in the block. If False, the standard
            Sup3rTransformerLayer with positional encoding will be used.
            Default is False.
        transformer_kwargs : dict | None
            Keyword arguments forwarded to each transformer layer in the
            block. This is the place to set transformer-layer options like
            ``embed_dim``, ``key_dim``, or positional encoding periods.
        attn_kwargs : dict | None
            Additional keyword arguments forwarded to the internal
            :class:`MultiHeadAttention` layer for each transformer layer.
        **kwargs
             Additional keyword arguments to pass to the block itself. This can
             include arguments like trainable and dtype.
        """
        super().__init__(**kwargs)
        self.features = features or []
        self.exo_features = exo_features or []
        transformer_kwargs = dict(transformer_kwargs or {})
        transformer_kwargs.setdefault('num_heads', num_heads)
        transformer_kwargs.setdefault('key_dim', key_dim)
        transformer_kwargs.setdefault('embed_dim', embed_dim)

        transformer_cls = (
            Sup3rTransformerLayerAlibi if use_alibi else Sup3rTransformerLayer
        )
        self.layers = [
            transformer_cls(
                **transformer_kwargs,
                attn_kwargs=attn_kwargs,
            )
            for _ in self.features
        ]

    def call(self, x, hi_res_features=None, exo_data=None):
        """Call the stack of transformer layers.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor. Typically this is the latent space tensor
            being updated by the attention block.
        hi_res_features : tf.Tensor, optional
            4D or 5D high resolution feature tensor. This will be used as the
            value input. This can be sparse observation data, possibly with
            some NaN values, or high-resolution gapless data like topography.
        exo_data: tf.Tensor, optional
            4D or 5D tensor of features to use for positional encoding. If
            hi_res_feature is provided, this should must include latitude and
            longitude, and optionally time, in that order.  Latitude and
            longitude should be in degrees and time should be in a datetime
            format that can be parsed by
            tf.experimental.numpy.datetime_as_string.

        Returns
        -------
        x : tf.Tensor
            Output tensor of the attention block after passing through all
            layers in the stack.
        """
        x_in = x
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                hi_res_feature=hi_res_features[..., i : i + 1],
                exo_data=exo_data,
            )
        return x_in + x


class ExpandDims(tf.keras.layers.Layer):
    """Layer to add an extra dimension to a tensor."""

    def __init__(self, axis=3, **kwargs):
        """
        Parameters
        ----------
        axis : int
            Target axis at which to expand the shape of the input. Default is
            axis 3 based on creating a new temporal axis of the default
            spatiotemporal shape of: (n_observations, n_spatial_0, n_spatial_1,
            n_temporal, n_features)
        """
        super().__init__(**kwargs)
        self._axis = axis

    def call(self, x):
        """Calls the expand dims operation

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor with an extra dimension based on the init axes arg
        """
        return tf.expand_dims(x, axis=self._axis)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({'axis': self._axis})
        return config


class TileLayer(tf.keras.layers.Layer):
    """Layer to tile (repeat) data across a given axis."""

    def __init__(self, multiples, **kwargs):
        """
        Parameters
        ----------
        multiples : list
            This is a list with the same length as number of dimensions in the
            input tensor. Each entry in the list determines how many times to
            tile each axis in the tensor.
        """
        super().__init__(**kwargs)
        self._multiples = tuple(int(value) for value in multiples)
        self._mult = tf.constant(self._multiples, tf.int32)

    def call(self, x):
        """Calls the tile operation

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor with the specified axes tiled into larger shapes
            based on the multiples initialization argument.
        """
        return tf.tile(x, self._mult)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({'multiples': list(self._multiples)})
        return config


class GaussianAveragePooling2D(tf.keras.layers.Layer):
    """Custom layer to implement tensorflow average pooling layer but with a
    gaussian kernel. This is basically a gaussian smoothing layer with a fixed
    convolution window that limits the area of effect"""

    def __init__(
        self,
        pool_size,
        strides=None,
        padding='valid',
        sigma=1,
        trainable=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        pool_size: integer
            Pooling window size. This sets the number of pixels in each
            dimension that will be averaged into an output pixel. Only one
            integer is specified, the same window length will be used for both
            dimensions. For example, if ``pool_size=2`` and ``strides=2`` then
            the output dimension will be half of the input.
        strides: Integer, tuple of 2 integers, or None.
            Strides values. If None, it will default to `pool_size`.
        padding: One of `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the
            same height/width dimension as the input.
        sigma : float
            Sigma parameter for gaussian distribution
        trainable : bool
            Flag for whether sigma is trainable weight or not.
        kwargs : dict
            Extra kwargs for tf.keras.layers.Layer
        """

        super().__init__(**kwargs)
        assert isinstance(pool_size, int), 'pool_size must be int!'
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.upper()
        self.trainable = trainable
        self.sigma = sigma

    def build(self, input_shape):  # noqa: ARG002
        """Custom implementation of the tf layer build method.

        Initializes the trainable sigma variable

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        if not any(self.weights):
            init = tf.keras.initializers.Constant(value=self.sigma)
            self.sigma = self.add_weight(
                name='sigma',
                shape=[],
                trainable=self.trainable,
                dtype=tf.float32,
                initializer=init,
            )

    def make_kernel(self):
        """Creates 2D gaussian kernel with side length `self.pool_size` and a
        sigma of `sigma`

        Returns
        -------
        kernel : np.ndarray
            2D kernel with shape (self.pool_size, self.pool_size)
        """
        ax = tf.linspace(
            -(self.pool_size - 1) / 2.0,
            (self.pool_size - 1) / 2.0,
            self.pool_size,
        )
        gauss = tf.math.exp(
            -0.5 * tf.math.square(ax) / tf.math.square(self.sigma)
        )
        kernel = tf.expand_dims(gauss, 0) * tf.expand_dims(gauss, -1)
        kernel /= tf.math.reduce_sum(kernel)
        kernel = tf.expand_dims(kernel, -1)
        kernel = tf.expand_dims(kernel, -1)
        return kernel

    def get_config(self):
        """Implementation of get_config method from tf.keras.layers.Layer for
        saving/loading as part of keras sequential model.

        Returns
        -------
        config : dict
        """
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
            'trainable': self.trainable,
            'sigma': float(self.sigma),
        })
        return config

    def call(self, x):
        """Operates on x with the specified function

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor operated on by the specified function
        """

        kernel = self.make_kernel()

        out = []
        for idf in range(x.shape[-1]):
            fslice = slice(idf, idf + 1)
            iout = tf.nn.convolution(
                x[..., fslice],
                kernel,
                strides=self.strides,
                padding=self.padding,
            )
            out.append(iout)
        out = tf.concat(out, -1, name='concat')
        return out


class GaussianNoiseAxis(tf.keras.layers.Layer):
    """Layer to apply random noise along a given axis."""

    def __init__(self, axis, mean=1, stddev=0.1, **kwargs):
        """
        Parameters
        ----------
        axis : int | list | tuple
            Axes to apply random noise across. All other axes will have the
            same noise. For example, for a 5D spatiotemporal tensor with
            axis=(1, 2, 3) (both spatial axes and the temporal axis), this
            layer will apply a single random number to every unique index of
            axis=(1, 2, 3).
        mean : float
            The mean of the normal distribution.
        stddev : float
            The standard deviation of the normal distribution.
        """

        super().__init__(**kwargs)
        self.rank = None
        self._axis = axis if isinstance(axis, (tuple, list)) else [axis]
        self._mean = float(mean)
        self._stddev = float(stddev)

    def _get_rand_shape(self, x):
        """Get shape of random noise along the specified axes."""
        shape = np.ones(len(x.shape), dtype=np.int32)
        for ax in self._axis:
            shape[ax] = x.shape[ax]
        return tf.constant(shape, dtype=tf.dtypes.int32)

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Sets the shape of the random noise along the specified axis

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self.rank = len(input_shape)

    def call(self, x):
        """Calls the tile operation

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor with noise applied to the requested axis.
        """

        rand_tensor = tf.random.normal(
            self._get_rand_shape(x),
            mean=self._mean,
            stddev=self._stddev,
            dtype=tf.dtypes.float32,
        )
        return x + rand_tensor

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update(
            {
                'axis': list(self._axis),
                'mean': self._mean,
                'stddev': self._stddev,
            }
        )
        return config


class FlattenAxis(tf.keras.layers.Layer):
    """Layer to flatten an axis from a 5D spatiotemporal Tensor into axis-0
    observations."""

    def __init__(self, axis=3, **kwargs):
        """
        Parameters
        ----------
        axis : int
            Target axis that holds the dimension to be flattened into the
            axis-0 dimension. Default is axis 3 based on flatteneing the
            temporal axis of the default spatiotemporal shape of:
            (n_observations, n_spatial_0, n_spatial_1, n_temporal, n_features)
        """
        super().__init__(**kwargs)
        self._axis = axis

    @staticmethod
    def _check_shape(input_shape):
        """Assert that the shape of the input tensor is the expected 5D
        spatiotemporal shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        msg = (
            'Input to FlattenAxis must be 5D with dimensions: '
            '(n_observations, n_spatial_0, n_spatial_1, n_temporal, '
            'n_features), but received shape: {}'.format(input_shape)
        )
        assert len(input_shape) == 5, msg

    def call(self, x):
        """Calls the flatten axis operation

        Parameters
        ----------
        x : tf.Tensor
            5D spatiotemporal tensor with dimensions:
            (n_observations, n_spatial_0, n_spatial_1, n_temporal, n_features)

        Returns
        -------
        x : tf.Tensor
            4D spatiotemporal tensor with target axis flattened into axis 0
        """
        self._check_shape(x.shape)
        return tf.concat(tf.unstack(x, axis=self._axis), axis=0)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({'axis': self._axis})
        return config


class SpatialExpansion(tf.keras.layers.Layer):
    """Class to expand the spatial dimensions of tensors with shape:
    (n_observations, n_spatial_0, n_spatial_1, n_features)
    """

    def __init__(
        self, spatial_mult=1, spatial_method='depth_to_space', **kwargs
    ):
        """
        Parameters
        ----------
        spatial_mult : int
            Number of times to multiply the spatial dimensions. Note that the
            spatial expansion is an un-packing of the feature dimension. For
            example, if the input layer has shape (123, 5, 5, 16) with
            multiplier=2 the output shape will be (123, 10, 10, 4). The
            input feature dimension must be divisible by the spatial multiplier
            squared.
        spatial_method : str
            Either "depth_to_space" or an interpolation method for
            tf.image.resize().
        """
        super().__init__(**kwargs)
        self._spatial_mult = int(spatial_mult)
        self._spatial_meth = spatial_method

    @staticmethod
    def _check_shape(input_shape):
        """Assert that the shape of the input tensor is the expected 4D
        spatiotemporal shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        msg = (
            'Input to SpatialExpansion must be 4D with dimensions: '
            '(n_observations, n_spatial_0, n_spatial_1, n_features), '
            'but received shape: {}'.format(input_shape)
        )
        assert len(input_shape) == 4, msg

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self._check_shape(input_shape)

    def _spatial_expand(self, x):
        """Expand the two spatial dimensions (axis=1,2) of a 4D tensor using
        data from the last axes"""

        if self._spatial_meth == 'depth_to_space':
            check_shape = x.shape[-1] % self._spatial_mult**2
            if check_shape != 0:
                msg = (
                    'Spatial expansion of factor {} is being attempted on '
                    'input tensor of shape {}, but the last dimension of the '
                    'input tensor ({}) must be divisible by the spatial '
                    'factor squared ({}).'.format(
                        self._spatial_mult,
                        x.shape,
                        x.shape[-1],
                        self._spatial_mult**2,
                    )
                )
                logger.error(msg)
                raise RuntimeError(msg)

            out = tf.nn.depth_to_space(x, self._spatial_mult)

        else:
            s_expand_shape = tf.stack([
                x.shape[1] * self._spatial_mult,
                x.shape[2] * self._spatial_mult,
            ])
            out = tf.image.resize(x, s_expand_shape, method=self._spatial_meth)

        return out

    def call(self, x):
        """Call the custom SpatialExpansion layer

        Parameters
        ----------
        x : tf.Tensor
            4D spatial tensor
            (n_observations, n_spatial_0, n_spatial_1, n_features)

        Returns
        -------
        x : tf.Tensor
            4D spatiotemporal tensor with axes 1,2 expanded (if spatial_mult>1)
        """
        self._check_shape(x.shape)

        if self._spatial_mult > 1:
            x = self._spatial_expand(x)

        return x

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update(
            {
                'spatial_mult': self._spatial_mult,
                'spatial_method': self._spatial_meth,
            }
        )
        return config


class SpatioTemporalExpansion(tf.keras.layers.Layer):
    """Class to expand the spatiotemporal dimensions of tensors with shape:
    (n_observations, n_spatial_0, n_spatial_1, n_temporal, n_features)
    """

    def __init__(
        self,
        spatial_mult=1,
        temporal_mult=1,
        spatial_method='depth_to_space',
        temporal_method='nearest',
        t_roll=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        spatial_mult : int
            Number of times to multiply the spatial dimensions. Note that the
            spatial expansion is an un-packing of the feature dimension. For
            example, if the input layer has shape (123, 5, 5, 24, 16) with
            multiplier=2 the output shape will be (123, 10, 10, 24, 4). The
            input feature dimension must be divisible by the spatial multiplier
            squared.
        temporal_mult : int
            Number of times to multiply the temporal dimension. For example,
            if the input layer has shape (123, 5, 5, 24, 2) with multiplier=2
            the output shape will be (123, 5, 5, 48, 2).
        spatial_method : str
            Either "depth_to_space" or an interpolation method for
            tf.image.resize().
        temporal_method : str
            Interpolation method for tf.image.resize(). Can also be
            "depth_to_time" for an operation similar to tf.nn.depth_to_space
            where the feature axis is unpacked into the temporal axis.
        t_roll : int
            Option to roll the temporal axis after expanding. When using
            temporal_method="depth_to_time", the default (t_roll=0) will add
            temporal steps after the input steps such that if input temporal
            shape is 3 and the temporal_mult is 24x, the output will have the
            index-0 timesteps at idt=0,24,48 but if t_roll=12, the output will
            have the original timesteps at idt=12,36,60. This is no longer
            recommended, as a positive roll will move the features of timestep
            -1 from the end of the series to the beginning.
        """

        super().__init__(**kwargs)
        self._spatial_mult = int(spatial_mult)
        self._temporal_mult = int(temporal_mult)
        self._temporal_meth = temporal_method
        self._spatial_meth = spatial_method
        self._t_roll = t_roll

    @staticmethod
    def _check_shape(input_shape):
        """Assert that the shape of the input tensor is the expected 5D
        spatiotemporal shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        msg = (
            'Input to SpatioTemporalExpansion must be 5D with dimensions: '
            '(n_observations, n_spatial_0, n_spatial_1, n_temporal, '
            'n_features), but received shape: {}'.format(input_shape)
        )
        assert len(input_shape) == 5, msg

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self._check_shape(input_shape)

    def _temporal_expand(self, x):
        """Expand the temporal dimension (axis=3) of a 5D tensor"""

        if self._temporal_meth == 'depth_to_time':
            check_shape = x.shape[-1] % self._temporal_mult
            if check_shape != 0:
                msg = (
                    'Temporal expansion of factor {} is being attempted on '
                    'input tensor of shape {}, but the last dimension of '
                    'the input tensor ({}) must be divisible by the '
                    'temporal factor ({}).'.format(
                        self._temporal_mult,
                        x.shape,
                        x.shape[-1],
                        self._temporal_mult,
                    )
                )
                logger.error(msg)
                raise RuntimeError(msg)

            shape = (
                x.shape[0],
                x.shape[1],
                x.shape[2],
                x.shape[3] * self._temporal_mult,
                x.shape[4] // self._temporal_mult,
            )
            out = tf.reshape(x, shape)
            out = tf.roll(out, self._t_roll, axis=3)

        else:
            t_expand_shape = tf.stack([
                x.shape[2],
                x.shape[3] * self._temporal_mult,
            ])
            out = []
            for x_unstack in tf.unstack(x, axis=1):
                out.append(
                    tf.image.resize(
                        x_unstack,
                        t_expand_shape,
                        method=self._temporal_meth,
                    )
                )
            out = tf.stack(out, axis=1)

        return out

    def _spatial_expand(self, x):
        """Expand the two spatial dimensions (axis=1,2) of a 5D tensor using
        data from the last axes"""

        if self._spatial_meth == 'depth_to_space':
            check_shape = x.shape[-1] % self._spatial_mult**2
            if check_shape != 0:
                msg = (
                    'Spatial expansion of factor {} is being attempted on '
                    'input tensor of shape {}, but the last dimension of the '
                    'input tensor ({}) must be divisible by the spatial '
                    'factor squared ({}).'.format(
                        self._spatial_mult,
                        x.shape,
                        x.shape[-1],
                        self._spatial_mult**2,
                    )
                )
                logger.error(msg)
                raise RuntimeError(msg)

            out = [
                tf.nn.depth_to_space(x_unstack, self._spatial_mult)
                for x_unstack in tf.unstack(x, axis=3)
            ]

        else:
            s_expand_shape = tf.stack([
                x.shape[1] * self._spatial_mult,
                x.shape[2] * self._spatial_mult,
            ])
            out = []
            for x_unstack in tf.unstack(x, axis=3):
                out.append(
                    tf.image.resize(
                        x_unstack,
                        s_expand_shape,
                        method=self._spatial_meth,
                    )
                )

        return tf.stack(out, axis=3)

    def call(self, x):
        """Call the custom SpatioTemporalExpansion layer

        Parameters
        ----------
        x : tf.Tensor
            5D spatiotemporal tensor.

        Returns
        -------
        x : tf.Tensor
            5D spatiotemporal tensor with axes 1,2 expanded (if spatial_mult>1)
            and axes 3 expanded (if temporal_mult>1).
        """
        self._check_shape(x.shape)

        if self._temporal_mult > 1:
            x = self._temporal_expand(x)

        if self._spatial_mult > 1:
            x = self._spatial_expand(x)

        return x

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update(
            {
                'spatial_mult': self._spatial_mult,
                'temporal_mult': self._temporal_mult,
                'spatial_method': self._spatial_meth,
                'temporal_method': self._temporal_meth,
                't_roll': self._t_roll,
            }
        )
        return config


class SkipConnection(tf.keras.layers.Layer):
    """Custom layer to implement a skip connection. This layer should be
    initialized and referenced in a layer list by the same name as both the
    skip start and skip end.
    """

    def __init__(self, name, method='add', **kwargs):
        """
        Parameters
        ----------
        name : str
            Unique string identifier of the skip connection. The skip endpoint
            should have the same name.
        method : str
            Method to use for combining the skip start data and skip end data.
            Defaults to 'add'. If 'concat' this is applied along the trailing
            axis
        """
        super().__init__(name=name, **kwargs)
        self._cache = None
        self._method = method

    def call(self, x):
        """Call the custom SkipConnection layer

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor. If this is the skip start, the input will be cached
            and returned without manipulation. If this is the skip endpoint,
            the output will be the input x combined with the tensor cached at
            the skip start. The tensors will be combined according to the
            method given at initialization.
        """
        if self._cache is None:
            self._cache = x
            return x
        try:
            if self._method == 'concat':
                out = tf.concat((x, self._cache), axis=-1)
            else:
                out = getattr(tf, self._method)(x, self._cache)
        except Exception as e:
            msg = (
                'Could not {} SkipConnection "{}" data cache of '
                'shape {} to input of shape {}.'.format(
                    self._method, self.name, self._cache.shape, x.shape
                )
            )
            logger.error(msg)
            raise RuntimeError(msg) from e
        else:
            self._cache = None
            return out

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({'method': self._method})
        return config


class SqueezeAndExcitation(tf.keras.layers.Layer):
    """Custom layer for squeeze and excitation block for convolutional networks

    Note that this is only set up to take a channels-last conv output

    References
    ----------
    1. Hu, Jie, et al. Squeeze-and-Excitation Networks. arXiv:1709.01507,
       arXiv, 16 May 2019, http://arxiv.org/abs/1709.01507.
    2. Pröve, Paul-Louis. “Squeeze-and-Excitation Networks.” Medium, 18 Oct.
       2017,
    https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
    """

    def __init__(self, ratio=16, **kwargs):
        """
        Parameters
        ----------
        ratio : int
            Number of convolutional channels/filters divided by the number of
            dense connections in the SE block.
        """

        super().__init__(**kwargs)
        self._ratio = ratio
        self._n_channels = None
        self._dense_units = None
        self._hidden_layers = None

    def build(self, input_shape):
        """Build the SqueezeAndExcitation layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """

        self._n_channels = input_shape[-1]
        self._dense_units = int(np.ceil(self._n_channels / self._ratio))

        if len(input_shape) == 4:
            pool_layer = tf.keras.layers.GlobalAveragePooling2D()
        elif len(input_shape) == 5:
            pool_layer = tf.keras.layers.GlobalAveragePooling3D()
        else:
            msg = (
                'SqueezeAndExcitation layer can only accept 4D or 5D data '
                'for image or video input but received input shape: {}'.format(
                    input_shape
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        self._hidden_layers = [
            pool_layer,
            tf.keras.layers.Dense(self._dense_units, activation='relu'),
            tf.keras.layers.Dense(self._n_channels, activation='sigmoid'),
            tf.keras.layers.Multiply(),
        ]

    def call(self, x):
        """Call the custom SqueezeAndExcitation layer

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor, this is the squeeze-and-excitation weights
            multiplied by the original input tensor x
        """

        t_in = x
        for layer in self._hidden_layers[:-1]:
            x = layer(x)

        # multiply layer
        x = self._hidden_layers[-1]([t_in, x])

        return x

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({'ratio': self._ratio})
        return config


class MaskedSqueezeAndExcitation(tf.keras.layers.Layer):
    """Custom layer for masked squeeze and excitation block for convolutional
    networks

    Note that this is only set up to take a channels-last conv output"""

    def __init__(self, ratio=16, name=None, **kwargs):
        """
        Parameters
        ----------
        ratio : int
            Number of convolutional channels/filters divided by the number of
            dense connections in the SE block.
        name : str
            Name of layer
        """

        super().__init__(name=name, **kwargs)
        self._ratio = ratio
        self._n_channels = None
        self._dense_units = None
        self._hidden_layers = None

    def build(self, input_shape):
        """Build the SqueezeAndExcitation layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """

        self._n_channels = input_shape[-1]
        self._dense_units = int(np.ceil(self._n_channels / self._ratio))

        if len(input_shape) == 4:
            pool_layer = tf.keras.layers.GlobalAveragePooling2D()
        elif len(input_shape) == 5:
            pool_layer = tf.keras.layers.GlobalAveragePooling3D()
        else:
            msg = (
                'SqueezeAndExcitation layer can only accept 4D or 5D data '
                'for image or video input but received input shape: {}'.format(
                    input_shape
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        self._hidden_layers = [
            pool_layer,
            tf.keras.layers.Dense(self._dense_units, activation='relu'),
            tf.keras.layers.Dense(self._n_channels, activation='sigmoid'),
            tf.keras.layers.Multiply(),
        ]

    def call(self, x, y):
        """Call the custom SqueezeAndExcitation layer

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.
        y : tf.Tensor
            Sparse input tensor used to mask ``x``

        Returns
        -------
        x : tf.Tensor
            Output tensor, this is the squeeze-and-excitation weights
            multiplied by the original input tensor x
        """

        t_in = x
        mask = tf.math.is_nan(y[..., 0])
        x = tf.ragged.boolean_mask(x, mask)
        for layer in self._hidden_layers[:-1]:
            x = layer(x)

        # multiply layer
        x = self._hidden_layers[-1]([t_in, x])

        return x

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({'ratio': self._ratio})
        return config


class CBAM(tf.keras.layers.Layer):
    """Convolutional Block Attention Module

    Note that this is only set up to take a channels-last conv output

    References
    ----------
    1. Woo, Sanghyun, et al. "Cbam: Convolutional block attention module."
       Proceedings of the European conference on computer vision (ECCV). 2018.
    2. Ma, Bing, et al. "CBAM-GAN: generative adversarial networks based on
       convolutional block attention module." Artificial Intelligence and
       Security: 5th International Conference, ICAIS 2019, New York, NY, USA,
       July 26-28, 2019, Proceedings, Part I 5. Springer International
       Publishing, 2019.
    """

    def __init__(self, ratio=8, **kwargs):
        """
        Parameters
        ----------
        ratio : int
            Number of convolutional channels/filters divided by the number of
            dense connections in the CBAM block.
        """

        super().__init__(**kwargs)
        self._ratio = ratio
        self._n_channels = None
        self._dense_units = None
        self._ch_avg = None
        self._ch_max = None
        self._ch_scale = None
        self._st_scale = None

    def build(self, input_shape):
        """Build the CBAM layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """

        self._n_channels = input_shape[-1]
        self._dense_units = int(np.ceil(self._n_channels / self._ratio))

        if len(input_shape) == 4:
            avg_pool_layer = tf.keras.layers.GlobalAveragePooling2D()
            max_pool_layer = tf.keras.layers.GlobalMaxPooling2D()
            conv_layer = tf.keras.layers.Conv2D(
                1, kernel_size=7, padding='same', activation='sigmoid'
            )
            reshape_layer = tf.keras.layers.Reshape((1, 1, self._n_channels))
        elif len(input_shape) == 5:
            avg_pool_layer = tf.keras.layers.GlobalAveragePooling3D()
            max_pool_layer = tf.keras.layers.GlobalMaxPooling3D()
            conv_layer = tf.keras.layers.Conv3D(
                1, kernel_size=7, padding='same', activation='sigmoid'
            )
            reshape_layer = tf.keras.layers.Reshape((
                1,
                1,
                1,
                self._n_channels,
            ))
        else:
            msg = (
                'CBAM layer can only accept 4D or 5D data for image or video '
                'input but received input shape: {}'.format(input_shape)
            )
            logger.error(msg)
            raise RuntimeError(msg)

        self._ch_avg = [
            avg_pool_layer,
            tf.keras.layers.Dense(self._dense_units, activation='relu'),
            tf.keras.layers.Dense(self._n_channels, activation='sigmoid'),
        ]
        self._ch_max = [
            max_pool_layer,
            tf.keras.layers.Dense(self._dense_units, activation='relu'),
            tf.keras.layers.Dense(self._n_channels, activation='sigmoid'),
        ]
        self._ch_scale = [
            tf.keras.layers.Add(),
            tf.keras.layers.Activation('sigmoid'),
            reshape_layer,
            tf.keras.layers.Multiply(),
        ]

        self._st_scale = [
            tf.keras.layers.Concatenate(axis=-1),
            conv_layer,
            tf.keras.layers.Multiply(),
        ]

    def channel_attention(self, x):
        """Call the channel attention block

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor, this is the channel attention weights
            multiplied by the original input tensor x
        """

        t_in = x
        avg_pool = x
        max_pool = x
        for layer in self._ch_avg:
            avg_pool = layer(avg_pool)

        for layer in self._ch_max:
            max_pool = layer(max_pool)

        x = [avg_pool, max_pool]
        for layer in self._ch_scale[:-1]:
            x = layer(x)

        # multiply layer
        x = self._ch_scale[-1]([t_in, x])

        return x

    def spatiotemporal_attention(self, x):
        """Call the spatiotemporal attention block

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor, this is the spatiotemporal attention weights
            multiplied by the original input tensor x
        """

        t_in = x
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        x = [avg_pool, max_pool]

        for layer in self._st_scale[:-1]:
            x = layer(x)

        # multiply layer
        x = self._st_scale[-1]([t_in, x])

        return x

    def call(self, x):
        """Call the full CBAM block

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor, this is channel attention followed by spatiotemporal
            attention
        """

        x = self.channel_attention(x)
        x = self.spatiotemporal_attention(x)
        return x

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({'ratio': self._ratio})
        return config


class FNO(tf.keras.layers.Layer):
    """Custom layer for fourier neural operator block

    Note that this is only set up to take a channels-last input

    References
    ----------
    1. FourCastNet: A Global Data-driven High-resolution Weather Model using
    Adaptive Fourier Neural Operators. http://arxiv.org/abs/2202.11214
    2. Adaptive Fourier Neural Operators: Efficient Token Mixers for
    Transformers. http://arxiv.org/abs/2111.13587
    """

    def __init__(self, filters, sparsity_threshold=0.5, activation='relu'):
        """
        Parameters
        ----------
        filters : int
            Number of dense connections in the FNO block.
        sparsity_threshold : float
            Parameter to control sparsity and shrinkage in the softshrink
            activation function following the MLP layers.
        activation : str
            Activation function used in MLP layers.
        """

        super().__init__()
        self._filters = filters
        self._fft_layer = None
        self._ifft_layer = None
        self._mlp_layers = None
        self._activation = activation
        self._n_channels = None
        self._perms_in = None
        self._perms_out = None
        self._lambd = sparsity_threshold

    def _softshrink(self, x):
        """Softshrink activation function

        https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
        """
        values_below_lower = tf.where(x < -self._lambd, x + self._lambd, 0)
        values_above_upper = tf.where(self._lambd < x, x - self._lambd, 0)
        return values_below_lower + values_above_upper

    def _fft(self, x):
        """Apply needed transpositions and fft operation."""
        x = tf.transpose(x, perm=self._perms_in)
        x = self._fft_layer(tf.cast(x, tf.complex64))
        x = tf.transpose(x, perm=self._perms_out)
        return x

    def _ifft(self, x):
        """Apply needed transpositions and ifft operation."""
        x = tf.transpose(x, perm=self._perms_in)
        x = self._ifft_layer(tf.cast(x, tf.complex64))
        x = tf.transpose(x, perm=self._perms_out)
        return x

    def build(self, input_shape):
        """Build the FNO layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        self._n_channels = input_shape[-1]
        dims = list(range(len(input_shape)))
        self._perms_in = [dims[-1], *dims[:-1]]
        self._perms_out = [*dims[1:], dims[0]]

        if len(input_shape) == 4:
            self._fft_layer = tf.signal.fft2d
            self._ifft_layer = tf.signal.ifft2d
        elif len(input_shape) == 5:
            self._fft_layer = tf.signal.fft3d
            self._ifft_layer = tf.signal.ifft3d
        else:
            msg = (
                'FNO layer can only accept 4D or 5D data for image or video '
                'input but received input shape: {}'.format(input_shape)
            )
            logger.error(msg)
            raise RuntimeError(msg)

        self._mlp_layers = [
            tf.keras.layers.Dense(self._filters, activation=self._activation),
            tf.keras.layers.Dense(self._n_channels),
        ]

    def _mlp(self, x):
        """Run mlp layers on input"""
        for layer in self._mlp_layers:
            x = layer(x)
        return x

    def call(self, x):
        """Call the custom FourierNeuralOperator layer

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor, this is the FNO weights added to the original input
            tensor.
        """
        t_in = x
        x = self._fft(x)
        x = self._mlp(x)
        x = self._softshrink(x)
        x = self._ifft(x)
        x = tf.cast(x, dtype=t_in.dtype)

        return x + t_in


class Sup3rAdder(tf.keras.layers.Layer):
    """Layer to add high-resolution data to a sup3r model in the middle of a
    super resolution forward pass."""

    def __init__(self, name=None, **kwargs):
        """
        Parameters
        ----------
        name : str | None
            Unique str identifier of the adder layer. Usually the name of the
            hi-resolution feature used in the addition.
        """
        super().__init__(name=name, **kwargs)

    @staticmethod
    def call(x, hi_res_adder):
        """Adds hi-resolution data to the input tensor x in the middle of a
        sup3r resolution network.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor
        hi_res_adder : tf.Tensor | np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features) that can be added to x.

        Returns
        -------
        x : tf.Tensor
            Output tensor with the hi_res_adder added to x.
        """
        return x + hi_res_adder


class Sup3rConcatObs(tf.keras.layers.Layer):
    """Layer to concatenate sparse data in the middle of a super resolution
    forward pass. This is used to condition models on sparse observation data.
    If no fill_method is provided, this uses the first channel of the input
    tensor as a background for the provided values and then concatenates with
    the input tensor. Other options for fill_method are 'mean' and 'idw'.
    Additionally, there is an option to include a mask of where there are valid
    observation data in the concatenation."""

    def __init__(
        self, name=None, fill_method=None, include_mask=False, **kwargs
    ):
        """
        Parameters
        ----------
        name : str | None
            Unique str identifier of the layer. Usually the name of the
            hi-resolution feature used in the concatenation.
        fill_method : str | None
            Method to use for filling the NaN values in the hi_res_feature.
            If this is None then the first channel of x will be used.
            Otherwise, accepted values are 'mean' and 'idw'.
        include_mask : bool
            If True, the mask of the hi_res_feature showing where there is
            valid observation data will be included in the concatenation.
        """
        super().__init__(name=name, **kwargs)
        self._fill_method_name = fill_method
        if fill_method == 'mean':
            self.fill_method = mean_fill
        elif fill_method == 'idw':
            self.fill_method = idw_fill
        else:
            self.fill_method = None
        self.include_mask = include_mask

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update(
            {
                'fill_method': self._fill_method_name,
                'include_mask': self.include_mask,
            }
        )
        return config

    def call(self, x, hi_res_feature=None):
        """Combine the first channel of x and the non-nan data in
        hi_res_feature and concatenate with x.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor
        hi_res_feature : tf.Tensor | np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), 1). This is NaN where there are no observations and
            real values where observations exist.

        Returns
        -------
        x : tf.Tensor
            Output tensor with the hi_res_feature used to fix values of x.
        """
        if hi_res_feature is None:
            hi_res_feature = tf.constant(
                np.nan, shape=x[..., :1].shape, dtype=x.dtype
            )

        if self.fill_method is None:
            mask = tf.math.is_nan(hi_res_feature)
            fixed = tf.where(mask, x[..., :1], hi_res_feature)
        else:
            fixed, mask = self.fill_method(hi_res_feature)

        if self.include_mask:
            mask = tf.cast(mask, dtype=fixed.dtype)
            fixed = tf.concat((fixed, mask), axis=-1)

        return tf.concat((x, fixed), axis=-1)


class Sup3rObsModel(tf.keras.layers.Layer):
    """Layer to concatenate sparse data in the middle of a super
    resolution forward pass, with a learned embedding. Mutiple observation
    features and multiple continuous exogenous features can be provided.
    The embedding network is defined with a list of hidden layers. If no
    hidden layers are provided, this layer will simply concatenate the
    hi_res_feature, exogenous data (if provided), and mask (if
    ``include_mask`` is True), to the input tensor after filling the
    NaNs."""

    def __init__(
        self,
        name=None,
        features=None,
        exo_features=None,
        hidden_layers=None,
        fill_method='mean',
        include_mask=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str | None
            Unique str identifier of the layer. Usually the name of the
            hi-resolution feature used in the concatenation.
        features : list | None
            The names of the observation features to be included in the
            embedding input.
        exo_features : list | None
            The names of exogenous features to be included in the embedding
            input
        hidden_layers : list | None
            The list of layers used to create the embedding network.
        fill_method : str
            The method used to fill in the NaN values in the hi_res_feature
            before embedding. Options are 'mean', 'idw', or None. If None then
            the first channel of x will be used to fill the NaN values.
        include_mask : bool
            Whether to include the mask for where there is valid observation
            data in the embedding. If False, the mask will not be included in
            the embedding.
        """
        super().__init__(name=name, **kwargs)
        self._hidden_layers = hidden_layers or []
        self.features = features or []
        self.exo_features = exo_features or []
        self.include_mask = include_mask
        self.rank = None
        self.fill_method = None
        self._fill_method_name = fill_method

        if fill_method == 'mean':
            self.fill_method = mean_fill
        elif fill_method == 'idw':
            self.fill_method = idw_fill

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update(
            {
                'features': self.features,
                'exo_features': self.exo_features,
                'hidden_layers': [
                    tf.keras.layers.serialize(layer)
                    for layer in self._hidden_layers
                ],
                'fill_method': self._fill_method_name,
                'include_mask': self.include_mask,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Deserialize nested hidden layers for Keras loading."""
        hidden_layers = config.pop('hidden_layers', [])
        hidden_layers = [
            tf.keras.layers.deserialize(
                layer_config, custom_objects=get_custom_layer_objects()
            )
            for layer_config in hidden_layers
        ]
        config['hidden_layers'] = hidden_layers
        return cls(**config)

    def build(self, input_shape):
        """Build the weight net layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        self.rank = len(input_shape)

    def call(self, x, hi_res_feature=None, exo_data=None):
        """Apply the embed net to hi_res_feature, exogenous data, and the
        mask representing where hi_res_feature is not nan. Concatenate the
        output with x. ``hi_res_feature`` and ``exo_data`` are allowed to be
        None so that models can be trained with hi_res_feature and exogenous
        data and then run with various sets of inputs.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor
        hi_res_feature : tf.Tensor | np.ndarray | None
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features). This is NaN where there are no observations
            and real values where observations exist.
        exo_data : tf.Tensor | np.ndarray | None
            This is an array of exogenous data used to imform the embedding,
            like topography

        Returns
        -------
        x : tf.Tensor
            Output tensor with embedding concatenated to input.
        """
        if hi_res_feature is None:
            hr_shape = (*x[..., 0].shape, len(self.features))
            hi_res_feature = tf.constant(np.nan, shape=hr_shape, dtype=x.dtype)

        if exo_data is None and len(self.exo_features) > 0:
            exo_shape = (*x[..., 0].shape, len(self.exo_features))
            exo_data = tf.constant(0, shape=exo_shape, dtype=x.dtype)

        if self.fill_method is None:
            mask = tf.math.is_nan(hi_res_feature)
            hr_feat = tf.where(
                mask, x[..., : len(self.features)], hi_res_feature
            )
        else:
            hr_feat, mask = self.fill_method(hi_res_feature)

        if not self.include_mask:
            embed = hr_feat
        else:
            embed = tf.concat([hr_feat, mask], axis=-1)

        if exo_data is not None:
            embed = tf.concat([exo_data, embed], axis=-1)

        for layer in self._hidden_layers:
            embed = layer(embed)

        return tf.concat([x, embed], axis=-1)


class Sup3rConcat(tf.keras.layers.Layer):
    """Layer to concatenate a high-resolution feature to a sup3r model in the
    middle of a super resolution forward pass."""

    def __init__(self, name=None, **kwargs):
        """
        Parameters
        ----------
        name : str | None
            Unique str identifier for the concat layer. Usually the name of the
            hi-resolution feature used in the concatenation.
        """
        super().__init__(name=name, **kwargs)

    @staticmethod
    def call(x, hi_res_feature):
        """Concatenates a hi-resolution feature to the input tensor x in the
        middle of a sup3r resolution network.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor
        hi_res_feature : tf.Tensor | np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features) that can be concatenated to x.

        Returns
        -------
        x : tf.Tensor
            Output tensor with the hi_res_feature added to x.
        """
        return tf.concat((x, hi_res_feature), axis=-1)


class FunctionalLayer(tf.keras.layers.Layer):
    """Custom layer to implement the tensorflow layer functions (e.g., add,
    subtract, multiply, maximum, and minimum) with a constant value. These
    cannot be implemented in phygnn as normal layers because they need to
    operate on two tensors of equal shape."""

    def __init__(self, name, value, **kwargs):
        """
        Parameters
        ----------
        name : str
            Name of the tensorflow layer function to be implemented, options
            are (all lower-case): add, subtract, multiply, maximum, and minimum
        value : float
            Constant value to use in the function operation
        """

        options = ('add', 'subtract', 'multiply', 'maximum', 'minimum')
        msg = (
            f'FunctionalLayer input `name` must be one of "{options}" '
            f'but received "{name}"'
        )
        assert name in options, msg

        super().__init__(name=name, **kwargs)
        self._function_name = name
        self.value = value
        self.fun = getattr(tf.keras.layers, self._function_name)

    def call(self, x):
        """Operates on x with the specified function

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor operated on by the specified function
        """
        const = tf.constant(value=self.value, shape=x.shape, dtype=x.dtype)
        return self.fun((x, const))

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({'value': self.value})
        return config


class SigLin(tf.keras.layers.Layer):
    """Sigmoid linear unit. This can be used to set a soft minimum on a range.

    y = 1/(1+exp(-x)) where x<0.5
    y = x + 0.5 where x>=0.5
    """

    @staticmethod
    def call(x):
        """Operates on x with SigLin

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor with same shape as input x operated on by SigLin
        """

        return tf.math.maximum(tf.math.sigmoid(x), x + 0.5)


class LogTransform(tf.keras.layers.Layer):
    """Log transform or inverse transform of data

    ``y = log(x + adder) * scalar`` or
    ``y = exp(x / scalar) - adder`` for the inverse
    """

    def __init__(
        self,
        name=None,
        adder=0,
        scalar=1,
        inverse=False,
        idf=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str | None
            Name of the tensorflow layer
        adder : float
            Adder term for ``y = log(x + adder) * scalar``
        scalar : float
            Scalar term for ``y = log(x + adder) * scalar``
        inverse : bool
            Option to perform the inverse operation e.g.
            ``y = exp(x / scalar) - adder``
        idf : int | list | None
            One or more feature channel indices to perform log transform on.
            None will perform transform on all feature channels.
        """

        super().__init__(name=name, **kwargs)
        self.adder = adder
        self.scalar = scalar
        self.inverse = inverse
        self.rank = None
        self.idf = [idf] if isinstance(idf, int) else idf

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self.rank = len(input_shape)

    def _logt(self, x):
        if not self.inverse:
            return tf.math.log(x + self.adder) * self.scalar
        return tf.math.exp(x / self.scalar) - self.adder

    def call(self, x):
        """Operates on x with (inverse) log transform

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        y : tf.Tensor
            Log-transformed x tensor
        """

        if self.idf is None:
            return self._logt(x)
        out = []
        for idf in range(x.shape[-1]):
            if idf in self.idf:
                out.append(self._logt(x[..., idf : idf + 1]))
            else:
                out.append(x[..., idf : idf + 1])

        out = tf.concat(out, -1, name='concat')
        return out

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update(
            {
                'adder': self.adder,
                'scalar': self.scalar,
                'inverse': self.inverse,
                'idf': self.idf,
            }
        )
        return config


class UnitConversion(tf.keras.layers.Layer):
    """Layer to convert units per feature channel using the linear transform:
    ``y = x * scalar + adder``

    Be sure to check how this will interact with normalization factors.
    """

    def __init__(self, name=None, adder=0, scalar=1, **kwargs):
        """
        Parameters
        ----------
        name : str | None
            Name of the tensorflow layer
        adder : float | list
            Adder term for ``y = x * scalar + adder``. If this is a float, the
            same value will be used for all feature channels. If this is a
            list, each value will be used for the corresponding feature channel
            and the length must match the number of feature channels
        scalar : float | list
            Scalar term for ``y = x * scalar + adder``. If this is a float, the
            same value will be used for all feature channels. If this is a
            list, each value will be used for the corresponding feature channel
            and the length must match the number of feature channels
        """

        super().__init__(name=name, **kwargs)
        self._adder_config = adder
        self._scalar_config = scalar
        self.adder = adder
        self.scalar = scalar
        self.rank = None

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self.rank = len(input_shape)
        nfeat = input_shape[-1]

        dtypes = (int, np.int64, np.int32, float, np.float32, np.float64)

        if isinstance(self.adder, dtypes):
            self.adder = np.ones(nfeat) * self.adder
        else:
            msg = (
                f'UnitConversion layer `adder` array has length '
                f'{len(self.adder)} but input shape has last dimension '
                f'as {input_shape[-1]}'
            )
            assert len(self.adder) == input_shape[-1], msg

        self.adder = tf.convert_to_tensor(self.adder, dtype=tf.float32)

        if isinstance(self.scalar, dtypes):
            self.scalar = np.ones(nfeat) * self.scalar
        else:
            msg = (
                f'UnitConversion layer `scalar` array has length '
                f'{len(self.scalar)} but input shape has last dimension '
                f'as {input_shape[-1]}'
            )
            assert len(self.scalar) == input_shape[-1], msg

        self.scalar = tf.convert_to_tensor(self.scalar, dtype=tf.float32)

        super().build(input_shape)

    def call(self, x):
        """Convert units

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        y : tf.Tensor
            Unit-converted x tensor
        """

        if self.rank is None:
            self.build(x.shape)

        adder = tf.cast(self.adder, dtype=x.dtype)
        scalar = tf.cast(self.scalar, dtype=x.dtype)
        return x * scalar + adder

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update(
            {
                'adder': self._adder_config,
                'scalar': self._scalar_config,
            }
        )
        return config


_register_custom_layer_objects()
