# -*- coding: utf-8 -*-
"""Custom tf layers."""

import logging
from dataclasses import dataclass
from typing import Union

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


@dataclass(frozen=True)
class WindowGeometry:
    """Computed layout for one windowed-attention call."""

    batch_size: tf.Tensor
    query_height: Union[int, tf.Tensor]
    query_width: Union[int, tf.Tensor]
    kv_height: Union[int, tf.Tensor]
    kv_width: Union[int, tf.Tensor]
    window_size: Union[int, tf.Tensor]
    window_shift: Union[int, tf.Tensor]
    tile_size: Union[int, tf.Tensor]
    radius: Union[int, tf.Tensor]
    query_top_padding: Union[int, tf.Tensor]
    query_left_padding: Union[int, tf.Tensor]
    query_height_padding: Union[int, tf.Tensor]
    query_width_padding: Union[int, tf.Tensor]
    padded_query_height: Union[int, tf.Tensor]
    padded_query_width: Union[int, tf.Tensor]
    n_window_rows: Union[int, tf.Tensor]
    n_window_cols: Union[int, tf.Tensor]
    n_windows: Union[int, tf.Tensor]
    tile_tokens: Union[int, tf.Tensor]
    pad_spec: tf.Tensor


def _get_keras_mask(x):
    """Return the attached Keras mask across TF/Keras versions."""
    get_mask = getattr(tf.keras.backend, 'get_keras_mask', None)
    if get_mask is not None:
        return get_mask(x)
    return getattr(x, '_keras_mask', None)


def _set_keras_mask(x, mask):
    """Attach a Keras mask across TF/Keras versions."""
    set_mask = getattr(tf.keras.backend, 'set_keras_mask', None)
    if set_mask is not None:
        set_mask(x, mask)
    elif hasattr(x, '_keras_mask') or mask is not None:
        x._keras_mask = mask


class SwiGLU(tf.keras.layers.Layer):
    """SwiGLU activation function."""

    @tf.function
    def call(self, x):
        """Apply the SwiGLU activation function to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor with shape (..., 2 * d)

        Returns
        -------
        tf.Tensor
            Output tensor with shape (..., d) after applying SwiGLU activation.
        """
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        return x1 * tf.nn.silu(x2)


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
        config.update({
            'paddings': [list(pad) for pad in self._paddings],
            'mode': self._mode,
            'option': self._option,
        })
        return config


class PatchEncoder(tf.keras.layers.Layer):
    """Project spatial inputs into token features."""

    def __init__(self, name=None, patch_size=1, embed_dim=64, **kwargs):
        """Initialize the PatchEncoder layer.

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
        **kwargs : dict
            Additional keyword arguments passed to
            ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name, **kwargs)
        self.proj_layer = None
        self.avg_pool = None
        self.valid_pool = None
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.rank = None

    def build(self, input_shape):
        """Build the PatchEncoder layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        self.rank = len(input_shape)
        if self.patch_size > 1:
            kwargs = {
                'kernel_size': [self.patch_size] * (self.rank - 2),
                'strides': [self.patch_size] * (self.rank - 2),
                'filters': self.embed_dim,
                'padding': 'valid',
            }
            pool_kwargs = {
                'pool_size': self.patch_size,
                'strides': self.patch_size,
                'padding': 'valid',
            }
            self.proj_layer = (
                tf.keras.layers.Conv2D(**kwargs)
                if self.rank == 4
                else tf.keras.layers.Conv3D(**kwargs)
            )
            self.avg_pool = (
                tf.keras.layers.AveragePooling2D(**pool_kwargs)
                if self.rank == 4
                else tf.keras.layers.AveragePooling3D(**pool_kwargs)
            )
            self.valid_pool = (
                tf.keras.layers.MaxPooling2D(**pool_kwargs)
                if self.rank == 4
                else tf.keras.layers.MaxPooling3D(**pool_kwargs)
            )
        else:
            self.proj_layer = tf.keras.layers.Dense(
                self.embed_dim, use_bias=False
            )
        self.proj_layer.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        """Embed inputs for attention blocks.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor. NaN values should be replaced with 0
            before calling this layer.

        Returns
        -------
        x_emb : tf.Tensor
            Embedded tensor with same spatial dimensions as input but
            last dimension = embed_dim.
        """
        return self.proj_layer(x)

    def prepare_sparse_tensor(self, x, lat=None, lon=None):
        """Prepare sparse spatial inputs for patch encoding.

        NaNs are filled patchwise so partially observed patches remain usable.
        The returned mask marks token positions that are fully NaN.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D sparse input tensor.
        lat : tf.Tensor | None
            Optional latitude grid to pool to token resolution.
        lon : tf.Tensor | None
            Optional longitude grid to pool to token resolution.

        Returns
        -------
        x_clean : tf.Tensor
            Sparse input with NaNs filled patchwise.
        nan_mask : tf.Tensor
            Boolean token mask where True marks fully invalid patches.
        pooled_lat : tf.Tensor | None
            Latitude pooled to token resolution when patching is active.
        pooled_lon : tf.Tensor | None
            Longitude pooled to token resolution when patching is active.
        """
        pixel_valid = ~tf.math.reduce_any(tf.math.is_nan(x), axis=-1)
        x_clean = self._fill_patchwise_nans(x)
        nan_mask = ~pixel_valid

        if self.patch_size == 1:
            return x_clean, nan_mask, lat, lon

        token_valid = tf.cast(pixel_valid[..., tf.newaxis], tf.float32)
        token_valid = self.valid_pool(token_valid) > 0
        nan_mask = ~tf.squeeze(token_valid, axis=-1)
        pooled_lat = None if lat is None else self.avg_pool(lat)
        pooled_lon = None if lon is None else self.avg_pool(lon)
        return x_clean, nan_mask, pooled_lat, pooled_lon

    def _fill_patchwise_nans(self, x):
        """Fill NaNs with the mean of valid values inside each patch."""
        x_zero = tf.where(tf.math.is_nan(x), 0.0, x)
        if self.patch_size == 1:
            return x_zero

        valid = tf.cast(~tf.math.is_nan(x), x.dtype)
        patch_mean = self.avg_pool(x_zero)
        patch_valid = self.avg_pool(valid)
        patch_mean = tf.math.divide_no_nan(patch_mean, patch_valid)

        spatial_axes = range(1, self.rank - 1)
        upsampled = patch_mean
        for axis in spatial_axes:
            upsampled = tf.repeat(upsampled, self.patch_size, axis=axis)

        target_shape = tf.shape(x)
        current_shape = tf.shape(upsampled)
        paddings = [[0, 0]]
        for axis in spatial_axes:
            paddings.append([0, target_shape[axis] - current_shape[axis]])
        paddings.append([0, 0])
        upsampled = tf.pad(upsampled, paddings)

        return tf.where(tf.math.is_nan(x), upsampled, x)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
        })
        return config


class PatchDecoder(tf.keras.layers.Layer):
    """Project token features back to the query feature grid."""

    def __init__(
        self,
        name=None,
        patch_size=1,
        output_dim=None,
        **kwargs,
    ):
        """Initialize the PatchDecoder layer.

        Parameters
        ----------
        name : str | None
            Name of layer.
        patch_size : int
            Height, width, and depth of attention patches.
        output_dim : int | None
            Number of output features after decoding.
        **kwargs : dict
            Additional keyword arguments passed to
            ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.rank = None
        self.proj_layer = None

    def build(self, input_shape):
        """Build the PatchDecoder layer based on an input shape."""
        self.rank = len(input_shape)
        if self.patch_size > 1:
            kwargs = {
                'filters': self.output_dim,
                'kernel_size': [self.patch_size] * (self.rank - 2),
                'strides': [self.patch_size] * (self.rank - 2),
                'padding': 'valid',
            }
            self.proj_layer = (
                tf.keras.layers.Conv2DTranspose(**kwargs)
                if self.rank == 4
                else tf.keras.layers.Conv3DTranspose(**kwargs)
            )
        else:
            self.proj_layer = tf.keras.layers.Dense(self.output_dim)
        self.proj_layer.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        """Decode token features back into the query feature space."""
        return self.proj_layer(x)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'output_dim': self.output_dim,
        })
        return config


class PositionEncoder(tf.keras.layers.Layer):
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
        **kwargs,
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
            Minimum period in degrees for the positional encoding.
        max_period_spatial : float
            Maximum period in degrees for the positional encoding.
        min_period_temporal : float
            Minimum period in seconds for the positional encoding.
        max_period_temporal : float
            Maximum period in seconds for the positional encoding.
        **kwargs : dict
            Additional keyword arguments passed to
            ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name, **kwargs)
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
            Tensor of positions to encode.
        min_period : float
            Minimum period for the positional encoding.
        max_period : float
            Maximum period for the positional encoding.
        d : int
            Dimension of the positional encoding.
        """
        assert d % 2 == 0, (
            'Embedding dimension must be even for sin/cos encoding.'
        )
        min_freq = 2 * np.pi / max_period
        max_freq = 2 * np.pi / min_period
        freqs = tf.linspace(min_freq, max_freq, d // 2)
        theta = tf.cast(freqs, k.dtype) * k
        return tf.concat([tf.sin(theta), tf.cos(theta)], axis=-1)

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
            Input tensor used for shape reference.
        lat : tf.Tensor
            Latitude tensor (..., 1) in degrees.
        lon : tf.Tensor
            Longitude tensor (..., 1) in degrees.
        min_period : float
            Minimum period in degrees.
        max_period : float
            Maximum period in degrees.

        Returns
        -------
        lat_lon_enc : tf.Tensor
            Positional encoding with shape (batch, n_tokens, embed_dim)
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
        return tf.concat([lat_enc, lon_enc], axis=-1)

    def encode_time(self, x, time, min_period, max_period):
        """Sinusoidal positional encoding for time.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor used for shape reference.
        time : tf.Tensor
            Tensor of datetime values (..., 1).
        min_period : float
            Minimum period in seconds.
        max_period : float
            Maximum period in seconds.

        Returns
        -------
        time_enc : tf.Tensor
            Positional encoding with shape (batch, n_tokens, embed_dim)
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
        return tf.concat([doy_enc, soy_enc], axis=-1)

    def build(self, input_shape):
        """Build the Positional Encoding layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        self.rank = len(input_shape)
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
        super().build(input_shape)

    @tf.function
    def call(self, x, lat, lon, time=None):
        """Get positional encoding for attention blocks.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor used for shape reference.
        lat : tf.Tensor
            Latitude tensor (..., 1) in degrees.
        lon : tf.Tensor
            Longitude tensor (..., 1) in degrees.
        time : tf.Tensor | None
            Time tensor (..., 1). If None, time encoding is skipped.

        Returns
        -------
        x_enc : tf.Tensor
            Positional encoding tensor (batch, n_tokens, embed_dim)
        """
        if self.patch_size > 1:
            x = self._pool_layer(x)
            lat = self._pool_layer(lat)
            lon = self._pool_layer(lon)
            if time is not None:
                time = self._pool_layer(time)

        x_enc = self.encode_lat_lon(
            x, lat, lon, self.min_period_spatial, self.max_period_spatial
        )
        if self.rank == 5 and time is not None:
            x_enc += self.encode_time(
                x, time, self.min_period_temporal, self.max_period_temporal
            )
        return x_enc

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'min_period_spatial': self.min_period_spatial,
            'max_period_spatial': self.max_period_spatial,
            'min_period_temporal': self.min_period_temporal,
            'max_period_temporal': self.max_period_temporal,
        })
        return config


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

    @tf.function
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
        if not self.built:
            self.build(
                query.shape,
                value.shape,
                None if key is None else key.shape,
            )
        if key is None:
            key = value

        query_mask = _get_keras_mask(query)
        value_mask = _get_keras_mask(value)
        key_mask = _get_keras_mask(key)
        _set_keras_mask(query, None)
        _set_keras_mask(value, None)
        _set_keras_mask(key, None)

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
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
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

        if query_mask is not None:
            _set_keras_mask(attention_output, query_mask)

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
        use_fused_attention = not return_attention_scores and (
            self._dropout == 0.0 or training is False
        )

        if use_fused_attention:
            if attention_mask is not None:
                mask_expansion_axis = -len(self._attention_axes) * 2 - 1
                target_rank = len(query.shape)
                for _ in range(target_rank - len(attention_mask.shape)):
                    attention_mask = tf.expand_dims(
                        attention_mask, axis=mask_expansion_axis
                    )

            attention_output = tf.keras.ops.dot_product_attention(
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


class WindowedMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention with overlapping spatial windowing.

    Partitions query tokens into non-overlapping spatial windows, expands
    each execution block into a larger key/value tile, runs attention per
    block, and reassembles the output.

    This reduces peak memory from O(n_q * n_v) to
    O(n_q * (window_size + 2 * radius)^2) while preserving locality
    information from ALiBi bias (or any additive pre-softmax bias).

    Parameters
    ----------
    window_size : int
        Side length of the non-overlapping query execution block in token
        units.
        When patch encoding is active upstream, each token represents a
        ``patch_size x patch_size`` spatial region.
        ``window_size=1`` is supported but operationally discouraged because
        it creates one halo tile per query token, which is typically poor for
        runtime and memory efficiency.
    radius : int
        Symmetric halo radius, in token units, added on each side of the
        query window when reading the key/value tile. The effective tile side
        length is ``window_size + 2 * radius``.
    num_heads : int
        Number of attention heads.
    key_dim : int
        Dimension of each attention head.
    **kwargs
        Additional keyword arguments forwarded to ``MultiHeadAttention``.

    Example::
        layer = WindowedMultiHeadAttention(
            window_size=8, radius=20, num_heads=4, key_dim=64
        )
        output = layer(
            query, value,
            bias=alibi_bias,
            query_spatial_shape=(32, 32),
            kv_spatial_shape=(64, 64),
        )
    """

    def __init__(
        self,
        window_size=None,
        radius=None,
        window_shift=0,
        num_heads=1,
        key_dim=64,
        alibi_scale=0.0,
        **kwargs,
    ):
        super().__init__(num_heads=num_heads, key_dim=key_dim, **kwargs)
        self.window_size = window_size
        self.radius = radius
        self.window_shift = int(window_shift)
        self.alibi_scale = float(alibi_scale)
        self.use_alibi = self.alibi_scale > 0
        self.head_slopes = None

        if self.radius is not None and self.radius < 0:
            msg = 'radius must be >= 0.'
            logger.error(msg)
            raise ValueError(msg)
        if self.window_shift < 0:
            msg = 'window_shift must be >= 0.'
            logger.error(msg)
            raise ValueError(msg)
        if self.window_size is not None:
            if self.radius is None:
                msg = 'radius is required when window_size is set.'
                logger.error(msg)
                raise ValueError(msg)
            if self.window_shift >= self.window_size:
                msg = 'window_shift must be < window_size.'
                logger.error(msg)
                raise ValueError(msg)
        elif self.radius is not None:
            msg = 'radius must be None when window_size is None.'
            logger.error(msg)
            raise ValueError(msg)

    def build(self, query_shape, value_shape, key_shape=None):
        """Build projection layers with 3D shapes.

        ``super().call()`` always receives 3D windowed tensors
        ``(B*n_win, tokens, C)`` regardless of input rank, so the
        internal projection layers must be built for ndim=3.
        """
        feat = query_shape[-1]
        shape_3d = (None, None, feat)
        super().build(shape_3d, shape_3d, key_shape=shape_3d)

        if self.use_alibi:
            x = 2 ** (8 / self._num_heads)
            slopes = np.array(
                [1 / (x ** (i + 1)) for i in range(self._num_heads)],
                dtype=np.float32,
            ).reshape(1, self._num_heads, 1, 1)
            self.head_slopes = self.add_weight(
                name='head_slopes',
                shape=slopes.shape,
                trainable=False,
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(slopes),
            )

    @staticmethod
    def _get_spatial_shape(tensor):
        """Get spatial height and width from static or dynamic shape."""
        height = tensor.shape[1]
        width = tensor.shape[2]

        if None in {height, width}:
            return tf.shape(tensor)[1], tf.shape(tensor)[2], True

        return int(height), int(width), False

    def _resolve_windowing(self, query, key):
        """Resolve the active window size and routing for this call."""
        if self.window_size is None:
            return None, True

        query_height, query_width, query_dynamic = self._get_spatial_shape(
            query
        )
        kv_height, kv_width, kv_dynamic = self._get_spatial_shape(key)

        tile_size = self.window_size + 2 * self.radius

        if query_dynamic or kv_dynamic:
            use_full_attention = tf.logical_and(
                tf.logical_and(
                    tile_size >= query_height,
                    tile_size >= query_width,
                ),
                tf.logical_and(
                    tile_size >= kv_height,
                    tile_size >= kv_width,
                ),
            )
            return self.window_size, use_full_attention

        window_size = min(
            self.window_size,
            int(query_height),
            int(query_width),
            int(kv_height),
            int(kv_width),
        )
        use_full_attention = (
            tile_size >= query_height
            and tile_size >= query_width
            and tile_size >= kv_height
            and tile_size >= kv_width
        )
        return window_size, use_full_attention

    def _get_window_geometry(self, query, key, window_size):
        """Get the full geometry for one windowed-attention call.

        This applies shifted padded-window geometry for the multi-window
        execution path.
        """

        batch_size = tf.shape(query)[0]
        query_height, query_width, q_dyn = self._get_spatial_shape(query)
        kv_height, kv_width, kv_dyn = self._get_spatial_shape(key)

        window_shift = min(self.window_shift, window_size - 1)
        if q_dyn or kv_dyn:
            max_tile = tf.reduce_min(
                [query_height, query_width, kv_height, kv_width]
            )
            radius = tf.maximum(
                tf.minimum(self.radius, (max_tile - window_size) // 2), 0
            )
            tile_size = window_size + 2 * radius
        else:
            max_tile = min(query_height, query_width, kv_height, kv_width)
            radius = max(min(self.radius, (max_tile - window_size) // 2), 0)
            tile_size = window_size + 2 * radius

        query_top_padding = window_shift
        query_left_padding = window_shift
        query_height_padding = (
            window_size - (query_height + query_top_padding) % window_size
        ) % window_size
        query_width_padding = (
            window_size - (query_width + query_left_padding) % window_size
        ) % window_size
        padded_query_height = (
            query_height + query_top_padding + query_height_padding
        )
        padded_query_width = (
            query_width + query_left_padding + query_width_padding
        )
        n_window_rows = padded_query_height // window_size
        n_window_cols = padded_query_width // window_size
        n_windows = n_window_rows * n_window_cols

        tile_tokens = tile_size * tile_size

        total_height = window_size * (n_window_rows - 1) + tile_size
        total_width = window_size * (n_window_cols - 1) + tile_size
        kv_top_padding = radius + query_top_padding
        kv_left_padding = radius + query_left_padding
        extra_height = tf.maximum(
            0, total_height - (kv_height + kv_top_padding)
        )
        extra_width = tf.maximum(0, total_width - (kv_width + kv_left_padding))
        pad_spec = tf.stack([
            tf.stack([0, 0]),
            tf.stack([kv_top_padding, extra_height]),
            tf.stack([kv_left_padding, extra_width]),
            tf.stack([0, 0]),
        ])

        return WindowGeometry(
            batch_size=batch_size,
            query_height=query_height,
            query_width=query_width,
            kv_height=kv_height,
            kv_width=kv_width,
            window_size=window_size,
            window_shift=window_shift,
            tile_size=tile_size,
            radius=radius,
            query_top_padding=query_top_padding,
            query_left_padding=query_left_padding,
            query_height_padding=query_height_padding,
            query_width_padding=query_width_padding,
            padded_query_height=padded_query_height,
            padded_query_width=padded_query_width,
            n_window_rows=n_window_rows,
            n_window_cols=n_window_cols,
            n_windows=n_windows,
            tile_tokens=tile_tokens,
            pad_spec=pad_spec,
        )

    @staticmethod
    def _partition_windows(tensor_4d, geometry):
        """Reshape ``(B, H, W, C)`` into ``(B*n_win, ws*ws, C)`` windows.

        Pads spatial dimensions to a multiple of ``window_size``, then
        reshapes into non-overlapping tiles.
        """
        c = tf.shape(tensor_4d)[-1]
        padding = [
            [0, 0],
            [geometry.query_top_padding, geometry.query_height_padding],
            [geometry.query_left_padding, geometry.query_width_padding],
            [0, 0],
        ]
        t = tf.pad(tensor_4d, padding)
        dims = [
            geometry.batch_size,
            geometry.n_window_rows,
            geometry.window_size,
            geometry.n_window_cols,
            geometry.window_size,
            c,
        ]
        t = tf.transpose(tf.reshape(t, dims), [0, 1, 3, 2, 4, 5])
        dims = [
            geometry.batch_size * geometry.n_windows,
            geometry.window_size * geometry.window_size,
            c,
        ]
        return tf.reshape(t, dims)

    @staticmethod
    def _extract_overlap_patches(tensor_4d, geometry):
        """Pad ``(B, H, W, C)`` and extract overlapping patches.

        Returns
        -------
        tf.Tensor
            ``(B * n_windows, tile_tokens, C)``
        """
        c = tf.shape(tensor_4d)[-1]
        padded = tf.pad(tensor_4d, geometry.pad_spec)
        patches = tf.image.extract_patches(
            padded,
            sizes=[1, geometry.tile_size, geometry.tile_size, 1],
            strides=[1, geometry.window_size, geometry.window_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        dims = [
            geometry.batch_size * geometry.n_windows,
            geometry.tile_tokens,
            c,
        ]
        return tf.reshape(patches, dims)

    @staticmethod
    def _build_window_mask(kv_nan_mask, dtype, geometry):
        """Build the full per-window boolean attention mask.

        Combines padding validity, optional NaN positions from
        *kv_nan_mask*, and the exact local neighborhood inside each K/V tile.

        Returns
        -------
        tf.Tensor
            Bool ``(B * n_windows, ws*ws, tile_size*tile_size)``.
        """
        if kv_nan_mask is not None:
            kv_valid = tf.expand_dims(tf.cast(~kv_nan_mask, dtype), -1)
        else:
            kv_valid = tf.ones(
                tf.stack([1, geometry.kv_height, geometry.kv_width, 1]),
                dtype=dtype,
            )

        kv_valid_padded = tf.pad(kv_valid, geometry.pad_spec)
        valid_patches = tf.image.extract_patches(
            kv_valid_padded,
            sizes=[1, geometry.tile_size, geometry.tile_size, 1],
            strides=[1, geometry.window_size, geometry.window_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        if kv_nan_mask is None:
            valid_patches = tf.broadcast_to(
                valid_patches,
                [
                    geometry.batch_size,
                    geometry.n_window_rows,
                    geometry.n_window_cols,
                    geometry.tile_tokens,
                ],
            )
        dims = [
            geometry.batch_size * geometry.n_windows,
            1,
            geometry.tile_tokens,
        ]
        kv_mask = tf.reshape(valid_patches, dims)
        window_rows = tf.repeat(
            tf.range(geometry.window_size), geometry.window_size
        )
        window_cols = tf.tile(
            tf.range(geometry.window_size), [geometry.window_size]
        )
        tile_rows = tf.repeat(tf.range(geometry.tile_size), geometry.tile_size)
        tile_cols = tf.tile(tf.range(geometry.tile_size), [geometry.tile_size])

        row_mask = tf.logical_and(
            tile_rows[None, :] >= window_rows[:, None],
            tile_rows[None, :] <= (window_rows[:, None] + 2 * geometry.radius),
        )
        col_mask = tf.logical_and(
            tile_cols[None, :] >= window_cols[:, None],
            tile_cols[None, :] <= (window_cols[:, None] + 2 * geometry.radius),
        )
        local_mask = tf.logical_and(row_mask, col_mask)[None, :, :]
        return tf.logical_and(tf.cast(kv_mask, tf.bool), local_mask)

    def _haversine_bias(self, lat_q, lon_q, lat_v, lon_v):
        """Compute scaled haversine ALiBi bias.

        All inputs should be broadcastable tensors of lat/lon in
        **degrees**. The last two dims represent (n_q, 1) and (1, n_v)
        or equivalent shapes that broadcast to (n_q, n_v).

        Returns
        -------
        tf.Tensor
            ``(..., num_heads, n_q, n_v)`` bias tensor.
        """
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
        distance = 2 * 6.371e6 * tf.asin(tf.sqrt(a))
        bias = -(distance**2) * self.alibi_scale
        bias = tf.expand_dims(bias, axis=1)
        return bias * self.head_slopes

    def _compute_window_alibi(
        self,
        lat,
        lon,
        geometry,
    ):
        """Compute per-window ALiBi bias from lat/lon coordinates.

        Partitions lat/lon into Q windows, extracts KV lat/lon patches,
        computes haversine distance, and scales by head slopes.

        Returns
        -------
        tf.Tensor
            ``(B * n_windows, num_heads, ws*ws, tile_tokens)``
        """
        # Q lat/lon windows
        q_lat_win = self._partition_windows(lat, geometry)
        q_lon_win = self._partition_windows(lon, geometry)

        # KV lat/lon patches - (B*n_win, tile_tokens, 1)
        kv_lat_win = self._extract_overlap_patches(lat, geometry)
        kv_lon_win = self._extract_overlap_patches(lon, geometry)

        # Transpose KV to (B*n_win, 1, tile_tokens) for broadcasting
        kv_lat_win = tf.transpose(kv_lat_win, [0, 2, 1])
        kv_lon_win = tf.transpose(kv_lon_win, [0, 2, 1])

        return self._haversine_bias(
            q_lat_win, q_lon_win, kv_lat_win, kv_lon_win
        )

    def _compute_full_alibi(self, lat, lon, batch_size):
        """Compute full-attention ALiBi bias from lat/lon coordinates.

        Returns
        -------
        tf.Tensor
            ``(B, num_heads, n_tokens, n_tokens)``
        """
        dims = [batch_size, -1, 1]
        lat_q = tf.reshape(lat, dims)
        lon_q = tf.reshape(lon, dims)
        dims = [batch_size, 1, -1]
        lat_v = tf.reshape(lat, dims)
        lon_v = tf.reshape(lon, dims)

        return self._haversine_bias(lat_q, lon_q, lat_v, lon_v)

    @staticmethod
    def _reassemble_windows(output, query, geometry):
        """Reshape windowed output back to ``(B, n_q, C)`` sequence."""
        feat_out = tf.shape(output)[-1]
        dims = [
            geometry.batch_size,
            geometry.n_window_rows,
            geometry.n_window_cols,
            geometry.window_size,
            geometry.window_size,
            feat_out,
        ]
        output = tf.transpose(tf.reshape(output, dims), [0, 1, 3, 2, 4, 5])
        dims = [
            geometry.batch_size,
            geometry.padded_query_height,
            geometry.padded_query_width,
            feat_out,
        ]
        output = tf.reshape(output, dims)
        output = output[
            :,
            geometry.query_top_padding : geometry.query_top_padding
            + geometry.query_height,
            geometry.query_left_padding : geometry.query_left_padding
            + geometry.query_width,
            :,
        ]
        return tf.reshape(output, tf.shape(query))

    def _full_attention_call(
        self,
        query,
        key,
        value,
        training,
        kv_nan_mask,
        lat,
        lon,
    ):
        """Full attention path when the current call does not need windowing.

        Flattens spatial dims to 3-D, computes ALiBi bias and NaN
        masking, runs standard MHA, and reshapes back.
        """
        batch_size = tf.shape(query)[0]
        feat = tf.shape(query)[-1]
        dims = [batch_size, -1, feat]
        q_flat = tf.reshape(query, dims)
        k_flat = tf.reshape(key, dims)
        v_flat = tf.reshape(value, dims)

        bias = None
        if self.use_alibi and lat is not None:
            bias = self._compute_full_alibi(lat, lon, batch_size)

        if kv_nan_mask is not None:
            nan_flat = tf.reshape(kv_nan_mask, [batch_size, 1, 1, -1])
            if bias is not None:
                bias = tf.where(nan_flat, tf.cast(-1e9, bias.dtype), bias)
            else:
                bias = tf.where(
                    nan_flat,
                    tf.constant(-1e9, dtype=query.dtype),
                    0.0,
                )

        output = super().call(
            query=q_flat,
            value=v_flat,
            key=k_flat,
            training=training,
            bias=bias,
        )
        return tf.reshape(output, tf.shape(query))

    def _window_attention_call(
        self,
        query,
        key,
        value,
        training,
        kv_nan_mask,
        lat,
        lon,
        window_size,
    ):
        """Execute the full windowed attention path."""
        geometry = self._get_window_geometry(query, key, window_size)

        q_win = self._partition_windows(query, geometry)
        k_win = self._extract_overlap_patches(key, geometry)
        v_win = self._extract_overlap_patches(value, geometry)
        kv_mask = self._build_window_mask(
            kv_nan_mask,
            query.dtype,
            geometry,
        )

        bias_win = None
        if self.use_alibi and lat is not None:
            bias_win = self._compute_window_alibi(lat, lon, geometry)

        output = super().call(
            query=q_win,
            value=v_win,
            key=k_win,
            attention_mask=kv_mask,
            training=training,
            bias=bias_win,
        )

        return self._reassemble_windows(output, query, geometry)

    @tf.function
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
        kv_nan_mask=None,
        lat=None,
        lon=None,
    ):
        """Run windowed attention.

        Parameters
        ----------
        query : tf.Tensor
            ``(batch, H_q, W_q, features)``
        value : tf.Tensor
            ``(batch, H_v, W_v, features)``
        key : tf.Tensor | None
            ``(batch, H_v, W_v, features)``. Defaults to *value*.
        bias : tf.Tensor | None
            Additive pre-softmax bias ``(batch, heads, n_q, n_v)`` or
            None.
        kv_nan_mask : tf.Tensor | None
            Boolean ``(B, H_v, W_v)`` where True marks NaN KV positions.
        lat, lon : tf.Tensor | None
            ``(B, H, W, 1)`` grids for per-window ALiBi bias.
        attention_mask, return_attention_scores, use_causal_mask
            Unused; kept for Keras API compatibility.
        training : bool | None
            Training flag forwarded to dropout.

        Returns
        -------
        output : tf.Tensor
            ``(batch, H_q, W_q, features)``
        """
        if key is None:
            key = value

        window_size, use_full_attention = self._resolve_windowing(query, key)
        if use_full_attention:
            return self._full_attention_call(
                query,
                key,
                value,
                training,
                kv_nan_mask,
                lat,
                lon,
            )
        return self._window_attention_call(
            query,
            key,
            value,
            training,
            kv_nan_mask,
            lat,
            lon,
            window_size,
        )

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'window_size': self.window_size,
            'radius': self.radius,
            'window_shift': self.window_shift,
            'alibi_scale': self.alibi_scale,
        })
        return config


class TransformerLayer(tf.keras.layers.Layer):
    """Custom transformer layer with multi-head attention layer that allows
    for additive bias pre-softmax."""

    def __init__(
        self,
        num_heads,
        key_dim,
        alibi_scale=0.0,
        window_size=None,
        radius=None,
        window_shift=0,
        dropout=0.0,
        **kwargs,
    ):
        """Initialize the transformer layer.

        Parameters
        ----------
        num_heads : int
            Number of attention heads.
        key_dim : int
            Size of each attention head.
        alibi_scale : float
            Positive values enable ALiBi distance-based attention bias and
            set its scaling factor. Non-positive values disable ALiBi.
        window_size : int | None
            Side length of the non-overlapping query execution block in token
            units.
            Patch encoding is applied before windowed attention, so this is
            measured on the token grid. ``None`` uses full attention over the
            entire token grid.
        radius : int | None
            Symmetric halo radius, in token units, added around each query
            window when reading key/value tokens.
        window_shift : int
            Shift of the query-window start on the token grid. This is only
            active when the current call uses multiple windows; otherwise it
            is ignored because the layer routes to full attention.
        dropout : float
            Dropout rate for attention weights.
        **kwargs
            Additional keyword arguments passed to ``tf.keras.layers.Layer``.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.window_size = window_size
        self.radius = radius
        self.window_shift = window_shift
        self.dropout = dropout

        self.attn = WindowedMultiHeadAttention(
            window_size=window_size,
            radius=radius,
            window_shift=window_shift,
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            alibi_scale=alibi_scale,
            dropout=self.dropout,
        )
        self.lo = tf.keras.layers.RMSNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * self.key_dim),
            SwiGLU(),
            tf.keras.layers.Dense(self.key_dim),
        ])

    def build(self, query_shape, key_shape, value_shape):
        """Build all sub-layers."""
        feat = query_shape[-1]
        self.attn.build(query_shape, value_shape, key_shape)
        generic_shape = (None, None, feat)
        self.lo.build(generic_shape)
        self.mlp.build(generic_shape)
        super().build(query_shape)

    @tf.function
    def call(
        self,
        query,
        key,
        value,
        kv_nan_mask=None,
        lat=None,
        lon=None,
    ):
        """Call transformer layer with multi-head attention output.

        Parameters
        ----------
        query : tf.Tensor
            ``(B, H, W, C)`` query tensor.
        key : tf.Tensor
            ``(B, H, W, C)`` key tensor.
        value : tf.Tensor
            ``(B, H, W, C)`` value tensor.
        kv_nan_mask : tf.Tensor | None
            Boolean mask for NaN KV positions.
        lat, lon : tf.Tensor | None
            Latitude / longitude grids for ALiBi bias.
        """
        attn = self.attn(
            query=query,
            key=key,
            value=value,
            kv_nan_mask=kv_nan_mask,
            lat=lat,
            lon=lon,
        )
        out = self.lo(query + attn)
        out_shape = tf.shape(out)
        batch = out_shape[0]
        feat = tf.shape(out)[-1]
        out_flat = tf.reshape(out, [batch, -1, feat])
        mlp_out = self.mlp(out_flat)
        mlp_out = tf.reshape(mlp_out, out_shape)
        return query + mlp_out

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'alibi_scale': self.attn.alibi_scale,
            'window_size': self.window_size,
            'radius': self.radius,
            'window_shift': self.window_shift,
            'dropout': self.dropout,
        })
        return config


class Sup3rTransformerLayer(tf.keras.layers.Layer):
    """Transformer layer with cross attention, tokenization, and optional
    ALiBi positional bias.  Queries are typically the latent space of the
    model; keys/values are high-resolution features (observations,
    topography, etc.).

    When ``alibi_scale > 0`` a distance-based bias replaces explicit
    positional encodings (ALiBi - Press et al., 2022). When
    ``alibi_scale <= 0``, sinusoidal positional encodings are added to Q
    and K.

    Note: This layer assumes that any sparse input data with NaN values has
    NaNs for the same tokens across all features.
    """

    def __init__(
        self,
        name=None,
        features=None,
        exo_features=None,
        patch_size=1,
        num_heads=1,
        key_dim=64,
        embed_dim=64,
        min_period_spatial=1e-4,
        max_period_spatial=2,
        min_period_temporal=1,
        max_period_temporal=864000,
        alibi_scale=0.0,
        window_size=None,
        radius=None,
        window_shift=0,
        dropout=0.0,
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
            List of exogenous feature names (latitude, longitude, time).
        patch_size : int
            Height, width, and optional depth of attention patches.
        embed_dim : int
            Dimension of the tokenized inputs.
        num_heads : int
            Number of attention heads.
        key_dim : int
            Size of each attention head.
        min_period_spatial : float
            Minimum period for the spatial positional encoding.
        max_period_spatial : float
            Maximum period for the spatial positional encoding.
        min_period_temporal : float
            Minimum period for the temporal positional encoding.
        max_period_temporal : float
            Maximum period for the temporal positional encoding.
        alibi_scale : float
            Positive values use ALiBi distance-based bias instead of
            positional encoding and set its scaling factor. Non-positive
            values disable ALiBi.
        window_size : int | None
            Side length of the non-overlapping query execution block in token
            units.
            Patch encoding is applied before attention, so this is measured on
            the token grid. ``None`` uses full attention.
        radius : int | None
            Symmetric halo radius, in token units, added around each query
            window when reading key/value tokens.
        window_shift : int
            Shift of the query-window start on the token grid. This is only
            active when the current call uses multiple windows; otherwise it
            is ignored because the layer routes to full attention.
        dropout : float
            Dropout rate for attention weights.
        **kwargs
            Additional keyword arguments for the parent class.
        """

        super().__init__(name=name, **kwargs)
        self.features = features or []
        self.exo_features = exo_features or []
        self.rank = None
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.min_period_spatial = min_period_spatial
        self.max_period_spatial = max_period_spatial
        self.min_period_temporal = min_period_temporal
        self.max_period_temporal = max_period_temporal
        self.window_size = window_size
        self.radius = radius
        self.window_shift = window_shift
        self.alibi_scale = float(alibi_scale)
        self.use_alibi = self.alibi_scale > 0
        self.dropout = dropout
        self.eq = PatchEncoder(
            patch_size=self.patch_size, embed_dim=self.embed_dim
        )
        self.ek = PatchEncoder(
            patch_size=self.patch_size, embed_dim=self.embed_dim
        )
        self.ev = PatchEncoder(
            patch_size=self.patch_size, embed_dim=self.embed_dim
        )
        self.pe = (
            None
            if self.use_alibi
            else PositionEncoder(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                min_period_spatial=self.min_period_spatial,
                max_period_spatial=self.max_period_spatial,
                min_period_temporal=self.min_period_temporal,
                max_period_temporal=self.max_period_temporal,
            )
        )
        self.tl = TransformerLayer(
            key_dim=self.key_dim,
            num_heads=self.num_heads,
            alibi_scale=self.alibi_scale,
            window_size=self.window_size,
            radius=self.radius,
            window_shift=self.window_shift,
            dropout=self.dropout,
        )
        self.decoder = None

    def _validate_build_shapes(self, x_shape, exo_data_shape):
        """Validate query and exogenous tensor shapes for build()."""
        self.rank = len(x_shape)
        msg = (
            'Sup3rTransformerLayer input must be 4D or 5D, but received '
            f'input shape: {x_shape}'
        )
        if self.rank not in {4, 5}:
            logger.error(msg)
            raise ValueError(msg)

        if exo_data_shape is None:
            return

        exo_rank = len(exo_data_shape)
        if exo_rank != self.rank:
            msg = (
                'Sup3rTransformerLayer exo_data rank must match the '
                f'query rank. Received x shape {x_shape} and exo_data '
                f'shape {exo_data_shape}.'
            )
            logger.error(msg)
            raise ValueError(msg)

        mismatched_dims = [
            (x_dim, exo_dim)
            for x_dim, exo_dim in zip(x_shape[:-1], exo_data_shape[:-1])
            if x_dim is not None and exo_dim is not None and x_dim != exo_dim
        ]
        if mismatched_dims:
            msg = (
                'Sup3rTransformerLayer exo_data spatial dimensions must '
                'match the query tensor. Received x shape '
                f'{x_shape} and exo_data shape {exo_data_shape}.'
            )
            logger.error(msg)
            raise ValueError(msg)

        exo_features = exo_data_shape[-1]
        if exo_features is not None and exo_features < 2:
            msg = (
                'Sup3rTransformerLayer exo_data must contain at least '
                'latitude and longitude channels. Received exo_data '
                f'shape {exo_data_shape}.'
            )
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _split_exo_inputs(exo_data):
        """Split exogenous inputs into lat, lon, and optional time."""
        lat = None if exo_data is None else exo_data[..., 0:1]
        lon = None if exo_data is None else exo_data[..., 1:2]
        time = (
            None
            if exo_data is None or exo_data.shape[-1] < 3
            else exo_data[..., 2:3]
        )
        return lat, lon, time

    @staticmethod
    def _merge_time_tensor(tensor, batch_time):
        """Merge a 5D ``(B, H, W, T, C)`` tensor into ``(B*T, H, W, C)``."""
        dims = tf.concat(
            [
                tf.reshape(batch_time, [1]),
                tf.shape(tensor)[1:3],
                tf.shape(tensor)[4:5],
            ],
            axis=0,
        )
        return tf.reshape(tf.transpose(tensor, [0, 3, 1, 2, 4]), dims)

    def _pad_to_patch_multiple(self, tensor):
        """Pad patch axes so tokenization preserves the full query domain."""
        if tensor is None or self.patch_size == 1:
            return tensor

        paddings = [[0, 0]]
        for axis in range(1, self.rank - 1):
            size = tf.shape(tensor)[axis]
            pad = (
                self.patch_size - tf.math.floormod(size, self.patch_size)
            ) % self.patch_size
            paddings.append([0, pad])
        paddings.append([0, 0])
        return tf.pad(tensor, paddings)

    def _prepare_attention_inputs(self, x, hi_res_feature, lat, lon, time):
        """Prepare encoded Q/K/V inputs and optional position features."""
        hr_clean, nan_mask, attn_lat, attn_lon = (
            self.ek.prepare_sparse_tensor(
                hi_res_feature, lat=lat, lon=lon
            )
        )

        q = self.eq(x)
        k = self.ek(hr_clean)
        v = self.ev(hr_clean)

        if not self.use_alibi and self.pe is not None:
            q = q + self.pe(x, lat=lat, lon=lon, time=time)  # noqa: PLR6104
            k = k + self.pe(hr_clean, lat=lat, lon=lon, time=time)  # noqa: PLR6104

        return q, k, v, nan_mask, attn_lat, attn_lon

    def build(self, x_shape, hi_res_feature_shape=None, exo_data_shape=None):
        """Build the layer based on an input shape.

        Parameters
        ----------
        x_shape : tuple
            Shape tuple of the query tensor.
        hi_res_feature_shape : tuple | None
            Shape tuple of the high resolution feature tensor.
        exo_data_shape : tuple | None
            Shape tuple of the exogenous data tensor.
        """
        self._validate_build_shapes(x_shape, exo_data_shape)
        value_shape = hi_res_feature_shape or x_shape
        embed_shape = (None, None, None, self.embed_dim)
        self.decoder = PatchDecoder(
            patch_size=self.patch_size, output_dim=x_shape[-1]
        )
        self.eq.build(x_shape)
        self.ek.build(value_shape)
        self.ev.build(value_shape)
        if self.pe is not None:
            self.pe.build(x_shape)
        self.tl.build(embed_shape, embed_shape, embed_shape)
        self.decoder.build(embed_shape)
        super().build(x_shape)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'features': self.features,
            'exo_features': self.exo_features,
            'patch_size': self.patch_size,
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'embed_dim': self.embed_dim,
            'min_period_spatial': self.min_period_spatial,
            'max_period_spatial': self.max_period_spatial,
            'min_period_temporal': self.min_period_temporal,
            'max_period_temporal': self.max_period_temporal,
            'alibi_scale': self.alibi_scale,
            'window_size': self.window_size,
            'radius': self.radius,
            'window_shift': self.window_shift,
            'dropout': self.dropout,
        })
        return config

    @staticmethod
    def _merge_time_into_batch(q, k, v, nan_mask, lat=None, lon=None):
        """Merge the time axis into the batch axis for 5D attention.

        Converts 5D tensors from ``(B, H, W, T, C)`` to ``(B*T, H, W, C)``
        and applies the same transformation to the NaN mask and optional
        latitude / longitude tensors.
        """
        batch_size = tf.shape(q)[0]
        time_steps = tf.shape(q)[3]
        bt = batch_size * time_steps
        q = Sup3rTransformerLayer._merge_time_tensor(q, bt)
        k = Sup3rTransformerLayer._merge_time_tensor(k, bt)
        v = Sup3rTransformerLayer._merge_time_tensor(v, bt)
        nan_mask = tf.squeeze(
            Sup3rTransformerLayer._merge_time_tensor(
                nan_mask[..., tf.newaxis], bt
            ),
            axis=-1,
        )
        if lat is not None:
            lat = Sup3rTransformerLayer._merge_time_tensor(lat, bt)
            lon = Sup3rTransformerLayer._merge_time_tensor(lon, bt)

        return q, k, v, nan_mask, lat, lon, batch_size, time_steps

    @tf.function
    def call(self, x, hi_res_feature=None, exo_data=None):
        """Call transformer layer on the full batch.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor (latent space being updated).
        hi_res_feature : tf.Tensor, optional
            4D or 5D high-resolution feature tensor used as key/value
            input.  May contain NaNs for sparse observations.
        exo_data : tf.Tensor, optional
            Exogenous data with latitude (channel 0), longitude (channel 1),
            and optionally time (channel 2).

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as *x*.
        """
        if hi_res_feature is None:
            return x

        original_shape = tf.shape(x)
        x = self._pad_to_patch_multiple(x)
        hi_res_feature = self._pad_to_patch_multiple(hi_res_feature)
        exo_data = self._pad_to_patch_multiple(exo_data)

        lat, lon, time = self._split_exo_inputs(exo_data)
        q, k, v, nan_mask, lat, lon = self._prepare_attention_inputs(
            x, hi_res_feature, lat, lon, time
        )

        batch_size = None
        time_steps = None
        if self.rank == 5:
            out = self._merge_time_into_batch(q, k, v, nan_mask, lat, lon)
            q, k, v, nan_mask, lat, lon, batch_size, time_steps = out

        out = self.tl(
            query=q, key=k, value=v, kv_nan_mask=nan_mask, lat=lat, lon=lon
        )

        if self.rank == 5:
            # Unmerge: (B*T, H, W, C) → (B, T, H, W, C) → (B, H, W, T, C)
            dims = tf.concat(
                [tf.stack([batch_size, time_steps]), tf.shape(out)[1:]], axis=0
            )
            out = tf.reshape(out, dims)
            out = tf.transpose(out, [0, 2, 3, 1, 4])

        out = self.decoder(out)

        return tf.slice(
            out, tf.zeros(self.rank, dtype=tf.int32), original_shape
        )


class Sup3rTransformerBlock(tf.keras.layers.Layer):
    """Custom layer to implement a block of Sup3rTransformerLayer layers."""

    def __init__(
        self,
        name=None,
        features=None,
        exo_features=None,
        patch_size=1,
        num_heads=1,
        key_dim=64,
        embed_dim=64,
        alibi_scale=0.0,
        window_size=None,
        radius=None,
        window_shift=0,
        dropout=0.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str | None
            Name of layer.
        features : list[str] | None
            List of hi-resolution feature names. The length of this list
            determines the number of Sup3rTransformerLayer layers in the
            block.
        exo_features : list[str] | None
            List of exogenous feature names (latitude, longitude, time).
        patch_size : int
            Height, width, and optional depth of attention patches.
        num_heads : int
            Number of attention heads for each transformer layer.
        key_dim : int
            Size of each attention head.
        embed_dim : int
            Dimension of the tokenized inputs.
        alibi_scale : float
            Positive values enable ALiBi and set its distance scaling
            factor. Non-positive values disable ALiBi.
        window_size : int | None
            Side length of the non-overlapping query execution block.
            ``None`` uses full attention.
        radius : int | None
            Symmetric halo radius, in token units, added around each query
            window when reading key/value tokens.
        window_shift : int
            Shift of the query-window start on the token grid. This is only
            active when the current call uses multiple windows; otherwise it
            is ignored because the layer routes to full attention.
        dropout : float
            Dropout rate for attention weights.
        **kwargs
            Additional keyword arguments for the block.
        """
        super().__init__(**kwargs)
        self.features = features or []
        self.exo_features = exo_features or []
        self.alibi_scale = float(alibi_scale)
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.radius = radius
        self.window_shift = window_shift
        self.dropout = dropout
        self.layers = [
            Sup3rTransformerLayer(
                features=[feat],
                patch_size=self.patch_size,
                num_heads=self.num_heads,
                key_dim=self.key_dim,
                embed_dim=self.embed_dim,
                alibi_scale=self.alibi_scale,
                window_size=self.window_size,
                radius=self.radius,
                window_shift=self.window_shift,
                dropout=self.dropout,
            )
            for feat in self.features
        ]

    @tf.function
    def call(self, x, hi_res_features=None, exo_data=None):
        """Call the stack of transformer layers.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor (latent space).
        hi_res_features : tf.Tensor, optional
            4D or 5D high-resolution feature tensor stack.
        exo_data : tf.Tensor, optional
            Exogenous data (latitude, longitude, optional time).

        Returns
        -------
        tf.Tensor
            Output tensor after all layers plus skip connection.
        """
        if hi_res_features is None:
            return x

        x_in = x
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                hi_res_feature=hi_res_features[..., i : i + 1],
                exo_data=exo_data,
            )
        return x_in + x

    def build(
        self,
        x_shape,
        hi_res_features_shape=None,
        exo_data_shape=None,
    ):
        """Build the block based on an input shape.

        Parameters
        ----------
        x_shape : tuple
            Shape tuple of the query tensor.
        hi_res_features_shape : tuple | None
            Shape tuple of the high resolution feature tensor stack.
        exo_data_shape : tuple | None
            Shape tuple of the exogenous data tensor.
        """
        layer_hi_res_shape = None
        if hi_res_features_shape is not None:
            layer_hi_res_shape = (*hi_res_features_shape[:-1], 1)

        for layer in self.layers:
            layer.build(x_shape, layer_hi_res_shape, exo_data_shape)
        super().build(x_shape)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'features': self.features,
            'exo_features': self.exo_features,
            'patch_size': self.patch_size,
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'embed_dim': self.embed_dim,
            'alibi_scale': self.alibi_scale,
            'window_size': self.window_size,
            'radius': self.radius,
            'window_shift': self.window_shift,
            'dropout': self.dropout,
        })
        return config


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

    @tf.function
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

    @tf.function
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
        config.update({
            'axis': list(self._axis),
            'mean': self._mean,
            'stddev': self._stddev,
        })
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

    @tf.function
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
        config.update({
            'spatial_mult': self._spatial_mult,
            'spatial_method': self._spatial_meth,
        })
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
        config.update({
            'spatial_mult': self._spatial_mult,
            'temporal_mult': self._temporal_mult,
            'spatial_method': self._spatial_meth,
            'temporal_method': self._temporal_meth,
            't_roll': self._t_roll,
        })
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

    @tf.function
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

    @tf.function
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

    @tf.function
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
    @tf.function
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
        config.update({
            'fill_method': self._fill_method_name,
            'include_mask': self.include_mask,
        })
        return config

    @tf.function
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
        config.update({
            'features': self.features,
            'exo_features': self.exo_features,
            'hidden_layers': [
                tf.keras.layers.serialize(layer)
                for layer in self._hidden_layers
            ],
            'fill_method': self._fill_method_name,
            'include_mask': self.include_mask,
        })
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

    @tf.function
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
    @tf.function
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

    @tf.function
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
    @tf.function
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

    @tf.function
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
        config.update({
            'adder': self.adder,
            'scalar': self.scalar,
            'inverse': self.inverse,
            'idf': self.idf,
        })
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

    @tf.function
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
        config.update({
            'adder': self._adder_config,
            'scalar': self._scalar_config,
        })
        return config


_register_custom_layer_objects()
