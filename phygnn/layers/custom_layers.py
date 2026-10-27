# -*- coding: utf-8 -*-
"""Custom tf layers."""

import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import tensorflow as tf

from phygnn.utilities.tf_utilities import idw_fill, mean_fill

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0


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
    window_size: Union[int, tf.Tensor]
    window_shift: Union[int, tf.Tensor]
    query_height_padding: Union[int, tf.Tensor]
    query_width_padding: Union[int, tf.Tensor]
    padded_query_height: Union[int, tf.Tensor]
    padded_query_width: Union[int, tf.Tensor]
    n_window_rows: Union[int, tf.Tensor]
    n_window_cols: Union[int, tf.Tensor]
    n_windows: Union[int, tf.Tensor]
    time_steps: Union[int, tf.Tensor] = 1


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


def _parse_patch_size(patch_size, rank):
    """Normalize a scalar or per-axis patch size for an input rank."""
    n_patch_axes = rank - 2
    if isinstance(patch_size, (int, np.integer)):
        patch_shape = (int(patch_size),) * n_patch_axes
    else:
        patch_shape = tuple(int(size) for size in patch_size)

    invalid_size = len(patch_shape) != n_patch_axes or any(
        size < 1 for size in patch_shape
    )
    if invalid_size:
        msg = (
            f'patch_size must contain {n_patch_axes} positive integers for '
            f'a rank-{rank} input, but received {patch_size}.'
        )
        logger.error(msg)
        raise ValueError(msg)
    return patch_shape


def _serialize_patch_size(patch_size):
    """Return a JSON-compatible patch size without changing scalar configs."""
    return (
        int(patch_size)
        if isinstance(patch_size, (int, np.integer))
        else list(patch_size)
    )


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

    @tf.function(reduce_retracing=True)
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
        patch_size : int | tuple[int, ...]
            Patch size for each non-batch, non-feature axis. A scalar applies
            the same size to every axis. For 5D inputs, ``(y, x, 1)`` applies
            spatial patching while preserving each temporal step.
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
        self.patch_shape = None
        self.rank = None

    def build(self, input_shape):
        """Build the PatchEncoder layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        self.rank = len(input_shape)
        self.patch_shape = _parse_patch_size(self.patch_size, self.rank)
        kwargs = {
            'kernel_size': self.patch_shape,
            'strides': self.patch_shape,
            'filters': self.embed_dim,
            'padding': 'valid',
        }
        pool_kwargs = {
            'pool_size': self.patch_shape,
            'strides': self.patch_shape,
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
        self.proj_layer.build(input_shape)
        super().build(input_shape)

    @tf.function(reduce_retracing=True)
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

    def prepare_sparse_tensor(self, x):
        """Prepare sparse spatial inputs for patch encoding.

        NaNs are filled patchwise so partially observed patches remain usable.
        The returned mask marks token positions that are fully NaN.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D sparse input tensor.

        Returns
        -------
        x_clean : tf.Tensor
            Sparse input with NaNs filled patchwise.
        nan_mask : tf.Tensor
            Boolean token mask where True marks fully invalid patches.
        """
        pixel_valid = ~tf.math.reduce_all(tf.math.is_nan(x), axis=-1)
        x_clean = self._fill_patchwise_nans(x)
        nan_mask = ~pixel_valid

        if all(size == 1 for size in self.patch_shape):
            return x_clean, nan_mask

        token_valid = tf.cast(pixel_valid[..., tf.newaxis], tf.float32)
        token_valid = self.valid_pool(token_valid) > 0
        nan_mask = ~tf.squeeze(token_valid, axis=-1)
        return x_clean, nan_mask

    def pool_sparse_tensor(self, x, weights=None, return_weights=False):
        """Pool sparse inputs while preserving fully invalid tokens.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D sparse input tensor.
        weights : tf.Tensor | None
            Valid observation weights aligned with ``x``. If None, each
            finite value receives unit weight.
        return_weights : bool
            Whether to return the pooled weights with the pooled values.

        Returns
        -------
        tf.Tensor | tuple[tf.Tensor, tf.Tensor]
            Weighted patchwise means with fully invalid feature channels
            represented by NaN, optionally followed by pooled weights.
        """
        valid = ~tf.math.is_nan(x)
        x_clean = tf.where(valid, x, 0.0)
        weights = (
            tf.cast(valid, x.dtype)
            if weights is None
            else tf.where(valid, tf.cast(weights, x.dtype), 0.0)
        )
        weighted = x_clean * weights
        if not all(size == 1 for size in self.patch_shape):
            weighted = self.avg_pool(weighted)
            weights = self.avg_pool(weights)
        pooled = tf.math.divide_no_nan(weighted, weights)
        pooled = tf.where(weights > 0, pooled, float('nan'))
        return (pooled, weights) if return_weights else pooled

    def _fill_patchwise_nans(self, x):
        """Fill NaNs with the mean of valid values inside each patch."""
        x_zero = tf.where(tf.math.is_nan(x), 0.0, x)
        if all(size == 1 for size in self.patch_shape):
            return x_zero

        valid = tf.cast(~tf.math.is_nan(x), x.dtype)
        patch_mean = self.avg_pool(x_zero)
        patch_valid = self.avg_pool(valid)
        patch_mean = tf.math.divide_no_nan(patch_mean, patch_valid)

        spatial_axes = range(1, self.rank - 1)
        upsampled = patch_mean
        for axis, size in zip(spatial_axes, self.patch_shape):
            upsampled = tf.repeat(upsampled, size, axis=axis)

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
            'patch_size': _serialize_patch_size(self.patch_size),
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
        patch_size : int | tuple[int, ...]
            Patch size for each non-batch, non-feature axis. A scalar applies
            the same size to every axis.
        output_dim : int | None
            Number of output features after decoding.
        **kwargs : dict
            Additional keyword arguments passed to
            ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size
        self.patch_shape = None
        self.output_dim = output_dim
        self.rank = None
        self.proj_layer = None

    def build(self, input_shape):
        """Build the PatchDecoder layer based on an input shape."""
        self.rank = len(input_shape)
        self.patch_shape = _parse_patch_size(self.patch_size, self.rank)
        kwargs = {
            'filters': self.output_dim,
            'kernel_size': self.patch_shape,
            'strides': self.patch_shape,
            'padding': 'valid',
        }
        self.proj_layer = (
            tf.keras.layers.Conv2DTranspose(**kwargs)
            if self.rank == 4
            else tf.keras.layers.Conv3DTranspose(**kwargs)
        )
        self.proj_layer.build(input_shape)
        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x):
        """Decode token features back into the query feature space."""
        return self.proj_layer(x)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'patch_size': _serialize_patch_size(self.patch_size),
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
        patch_size : int | tuple[int, ...]
            Patch size for each non-batch, non-feature axis. A scalar applies
            the same size to every axis. This pools positional encodings to
            the same patch shape as tokens.
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
        self.patch_shape = None
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
        theta = tf.cast(freqs, k.dtype)[:, tf.newaxis] * k[..., tf.newaxis, :]
        return tf.concat([tf.sin(theta), tf.cos(theta)], axis=-2)

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

    def encode_lat_lon(self, lat_lon):
        """Sinusoidal positional encoding for latitude and longitude.

        Parameters
        ----------
        lat_lon : tf.Tensor
            Coordinate tensor (..., 2) in degrees with latitude in channel 0
            and longitude in channel 1.

        Returns
        -------
        lat_lon_enc : tf.Tensor
            Positional encoding with the same leading dimensions as ``lat``
            and final dimension ``embed_dim``. The spatial or token grid is
            not flattened.
        """
        assert self.embed_dim % 4 == 0, (
            'Embedding dimension must be divisible by 4 for latitude and '
            'longitude encoding.'
        )
        lat_lon_enc = self._freq_encode(
            lat_lon,
            d=self.embed_dim // 2,
            min_period=self.min_period_spatial,
            max_period=self.max_period_spatial,
        )
        return tf.reshape(
            lat_lon_enc, tf.concat([tf.shape(lat_lon_enc)[:-2], [-1]], axis=0)
        )

    def encode_time(self, time):
        """Sinusoidal positional encoding for time.

        Parameters
        ----------
        time : tf.Tensor
            Tensor of Unix timestamps in seconds since epoch with shape
            ``(..., 1)``.

        Returns
        -------
        time_enc : tf.Tensor
            Positional encoding with the same leading dimensions as ``time``
            and final dimension ``embed_dim``. The spatial or token grid is
            not flattened.
        """
        assert self.embed_dim % 4 == 0, (
            'Embedding dimension must be divisible by 4 for time encoding.'
        )
        doy, soy = tf.numpy_function(
            self._compute_doy_soy, [time], [tf.float32, tf.float32]
        )
        doy = tf.reshape(doy, tf.shape(time))
        soy = tf.reshape(soy, tf.shape(time))
        min_period_doy = self.min_period_temporal / 86400
        max_period_doy = self.max_period_temporal / 86400
        doy_enc = self._freq_encode(
            doy, min_period_doy, max_period_doy, d=self.embed_dim // 2
        )
        soy_enc = self._freq_encode(
            soy,
            self.min_period_temporal,
            self.max_period_temporal,
            d=self.embed_dim // 2,
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
        self.patch_shape = _parse_patch_size(self.patch_size, self.rank)
        kwargs = {
            'pool_size': self.patch_shape,
            'strides': self.patch_shape,
            'padding': 'valid',
        }
        self._pool_layer = (
            tf.keras.layers.AveragePooling2D(**kwargs)
            if self.rank == 4
            else tf.keras.layers.AveragePooling3D(**kwargs)
        )
        super().build(input_shape)

    @tf.function
    def call(self, lat_lon, time=None):
        """Get positional encoding for attention blocks.

        Parameters
        ----------
        lat_lon : tf.Tensor
            Coordinate tensor (..., 2) in degrees with latitude in channel 0
            and longitude in channel 1.
        time : tf.Tensor | None
            Optional Unix timestamp tensor in seconds since epoch with shape
            ``(..., 1)``. If ``None``, time encoding is skipped.

        Returns
        -------
        x_enc : tf.Tensor
            Positional encoding tensor with the same leading dimensions as the
            active spatial or token grid and final dimension ``embed_dim``.
            When any patch dimension is greater than one, this corresponds to
            the pooled token grid rather than the unpooled coordinate grid.
        """
        lat_lon = self._pool_layer(lat_lon)
        time = self._pool_layer(time) if time is not None else None

        x_enc = self.encode_lat_lon(lat_lon)
        if self.rank == 5 and time is not None:
            x_enc += self.encode_time(time)
        return x_enc

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'patch_size': _serialize_patch_size(self.patch_size),
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

        attention_output = self._compute_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            training=training,
            bias=bias,
        )
        attention_output = self._output_dense(attention_output)

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(
                attention_output, lengths=query_lengths
            )

        if query_mask is not None:
            _set_keras_mask(attention_output, query_mask)

        return attention_output

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
        bias=None,
    ):
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
        return attention_output


class WindowedMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention with overlapping spatial windowing.

    Partitions query tokens into non-overlapping spatial windows, runs
    attention per window, and reassembles the output.

    This reduces peak memory from O(n_q * n_v) to
    O(n_q * window_size^2) while preserving locality information from
    ALiBi bias (or any additive pre-softmax bias).

    Parameters
    ----------
    window_size : int
        Side length of the non-overlapping execution block in token units.
        When patch encoding is active upstream, each token represents a
        ``patch_size x patch_size`` spatial region.
    num_heads : int
        Number of attention heads.
    key_dim : int
        Dimension of each attention head.
    distance_scale : float
        Length scale (in meters) over which the ALiBi distance bias
        decays. Larger values produce a gentler decay, allowing
        attention to reach farther spatially.
    temporal_bias_scale : float
        Positive values enable temporal ALiBi and set its scaling factor.
    temporal_distance_scale : float
        Length scale in seconds over which temporal ALiBi decays.
    **kwargs
        Additional keyword arguments forwarded to ``MultiHeadAttention``.

    Example::
        layer = WindowedMultiHeadAttention(
            window_size=8, num_heads=4, key_dim=64
        )
        output = layer(
            query, value,
            lat_lon=lat_lon,
        )
    """

    def __init__(
        self,
        window_size=None,
        window_shift=0,
        num_heads=1,
        key_dim=64,
        bias_scale=0.0,
        distance_scale=10000.0,
        temporal_bias_scale=0.0,
        temporal_distance_scale=86400.0,
        **kwargs,
    ):
        super().__init__(num_heads=num_heads, key_dim=key_dim, **kwargs)
        self.window_size = window_size
        self.window_shift = int(window_shift)
        self.bias_scale = float(bias_scale)
        self.distance_scale = float(distance_scale)
        self.temporal_bias_scale = float(temporal_bias_scale)
        self.temporal_distance_scale = float(temporal_distance_scale)
        self.use_spatial_bias = self.bias_scale > 0
        self.use_temporal_bias = self.temporal_bias_scale > 0
        self.use_pos_bias = self.use_spatial_bias or self.use_temporal_bias
        self.head_slopes = None

        if self.window_shift < 0:
            msg = 'window_shift must be >= 0.'
            logger.error(msg)
            raise ValueError(msg)
        if (
            self.window_size is not None
            and self.window_shift >= self.window_size
        ):
            msg = 'window_shift must be < window_size.'
            logger.error(msg)
            raise ValueError(msg)
        if self.use_temporal_bias and self.temporal_distance_scale <= 0:
            msg = (
                'temporal_distance_scale must be > 0 when temporal ALiBi '
                'is enabled.'
            )
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

        if self.use_pos_bias:
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

    def _get_window_geometry(self, query, window_size, time_steps=1):
        """Get the full geometry for one windowed-attention call.

        This applies shifted padded-window geometry for the multi-window
        execution path.
        """

        batch_size = tf.shape(query)[0]
        query_height = tf.shape(query)[1]
        query_width = tf.shape(query)[2]

        window_shift = min(self.window_shift, window_size - 1)

        query_height_padding = (
            window_size - (query_height + window_shift) % window_size
        ) % window_size
        query_width_padding = (
            window_size - (query_width + window_shift) % window_size
        ) % window_size
        padded_query_height = (
            query_height + window_shift + query_height_padding
        )
        padded_query_width = query_width + window_shift + query_width_padding
        n_window_rows = padded_query_height // window_size
        n_window_cols = padded_query_width // window_size
        n_windows = n_window_rows * n_window_cols

        return WindowGeometry(
            batch_size=batch_size,
            query_height=query_height,
            query_width=query_width,
            window_size=window_size,
            window_shift=window_shift,
            query_height_padding=query_height_padding,
            query_width_padding=query_width_padding,
            padded_query_height=padded_query_height,
            padded_query_width=padded_query_width,
            n_window_rows=n_window_rows,
            n_window_cols=n_window_cols,
            n_windows=n_windows,
            time_steps=time_steps,
        )

    @staticmethod
    def _partition_windows(tensor, geometry):
        """Reshape ``(B, H, W, C)`` or ``(B, H, W, T, C)`` into windows.

        For 4D input produces ``(B*n_win, ws*ws, C)``. For 5D produces
        ``(B*n_win, ws*ws*T, C)`` with spatial positions as the outer index
        (token ordering: spatial outer, time inner).  Pads spatial dimensions
        to a multiple of ``window_size``.
        """
        if len(tensor.shape) == 4:
            tensor = tensor[:, :, :, tf.newaxis, :]  # treat as T=1
        t_steps = tf.shape(tensor)[3]
        c = tf.shape(tensor)[-1]
        ws = geometry.window_size
        padding = [
            [0, 0],
            [geometry.window_shift, geometry.query_height_padding],
            [geometry.window_shift, geometry.query_width_padding],
            [0, 0],
            [0, 0],
        ]
        t = tf.pad(tensor, padding)
        win_shape = [
            geometry.batch_size,
            geometry.n_window_rows,
            ws,
            geometry.n_window_cols,
            ws,
            t_steps,
            c,
        ]
        t = tf.transpose(tf.reshape(t, win_shape), [0, 1, 3, 2, 4, 5, 6])
        flat = geometry.batch_size * geometry.n_windows
        return tf.reshape(t, [flat, ws * ws * t_steps, c])

    @classmethod
    def _build_window_mask(cls, kv_nan_mask, geometry):
        """Build the full per-window boolean attention mask.

        Combines padding validity and optional NaN positions from
        *kv_nan_mask*.

        For 4D Q/KV, *kv_nan_mask* is ``(B, H, W)`` and the returned
        mask has shape ``(B*n_windows, 1, ws*ws)``.
        For 5D Q/KV, *kv_nan_mask* is ``(B, H, W, T)`` and the
        returned mask has shape ``(B*n_windows, 1, ws*ws*T)``.

        Returns
        -------
        tf.Tensor
            Bool attention mask.
        """
        T = geometry.time_steps
        ws = geometry.window_size

        # --- NaN / padding validity -----------------------------------------
        if kv_nan_mask is not None:
            kv_valid = ~kv_nan_mask
            if len(kv_valid.shape) == 3:  # 4D: (B, H, W) -> add channel dim
                kv_valid = kv_valid[..., tf.newaxis]
        else:
            kv_valid = tf.ones(
                tf.stack([
                    geometry.batch_size,
                    geometry.query_height,
                    geometry.query_width,
                    T,
                ]),
                dtype=tf.bool,
            )

        # _partition_windows zero-pads, so padded positions become 0.0
        # Output shape: (B*n_windows, ws*ws*T, 1)
        valid_win = cls._partition_windows(kv_valid, geometry)

        # Keras broadcasts this key-validity vector over all query tokens.
        flat = geometry.batch_size * geometry.n_windows
        window_t = ws * ws * T
        kv_mask = tf.reshape(valid_win, [flat, window_t, 1])
        return tf.transpose(kv_mask, [0, 2, 1])

    @staticmethod
    def _compact_kv(tensors, valid):
        """Gather valid K/V tokens and aligned metadata in each group.

        Parameters
        ----------
        tensors : tuple[tf.Tensor | None, ...]
            Grouped K/V tensors and metadata with token axis 1.
        valid : tf.Tensor
            Boolean mask with shape ``(groups, tokens)`` where True marks
            valid K/V positions.

        Returns
        -------
        compact_tensors : tuple[tf.Tensor | None, ...]
            Input tensors gathered to the largest valid count per group.
        compact_mask : tf.Tensor
            Boolean mask for valid entries in the compact tensors.
        """
        max_valid = tf.maximum(
            tf.reduce_max(tf.reduce_sum(tf.cast(valid, tf.int32), axis=1)),
            1,
        )
        indices = tf.argsort(
            tf.cast(valid, tf.int32), axis=1, direction='DESCENDING'
        )[:, :max_valid]
        compact_tensors = tuple(
            None
            if tensor is None
            else tf.gather(tensor, indices, batch_dims=1)
            for tensor in tensors
        )
        return compact_tensors, tf.gather(valid, indices, batch_dims=1)

    def _haversine_bias(
        self, lat_lon_q, lat_lon_v, apply_head_slopes=True
    ):
        """Compute scaled haversine ALiBi bias.

        Inputs are grouped coordinate tensors in **degrees** with trailing
        latitude/longitude channels. The method computes pairwise distances
        between query and key/value coordinates.

        Parameters
        ----------
        lat_lon_q : tf.Tensor
            Query coordinates with shape ``(..., n_q, 2)``.
        lat_lon_v : tf.Tensor
            Key/value coordinates with shape ``(..., n_v, 2)``.
        apply_head_slopes : bool
            Whether to expand the bias across attention heads and apply the
            per-head slopes.

        Returns
        -------
        tf.Tensor
            ``(..., num_heads, n_q, n_v)`` bias tensor.
        """
        lat_lon_q = tf.expand_dims(lat_lon_q, axis=-2)
        lat_lon_v = tf.expand_dims(lat_lon_v, axis=-3)

        lat_q_rad, lon_q_rad = tf.unstack(lat_lon_q * (np.pi / 180.0), axis=-1)
        lat_v_rad, lon_v_rad = tf.unstack(lat_lon_v * (np.pi / 180.0), axis=-1)

        dlat = lat_q_rad - lat_v_rad
        dlon = lon_q_rad - lon_v_rad
        a = (
            tf.sin(dlat / 2) ** 2
            + tf.cos(lat_q_rad) * tf.cos(lat_v_rad) * tf.sin(dlon / 2) ** 2
        )
        a = tf.clip_by_value(a, 0.0, 1.0)
        distance = 2 * EARTH_RADIUS_M * tf.asin(tf.sqrt(a))
        bias = self.bias_scale * tf.math.tanh(-distance / self.distance_scale)
        return (
            tf.expand_dims(bias, axis=1) * self.head_slopes
            if apply_head_slopes
            else bias
        )

    def _temporal_bias(self, time_q, time_v, apply_head_slopes=True):
        """Compute temporal ALiBi bias from relative time offsets.

        Parameters
        ----------
        time_q : tf.Tensor
            Query time offsets in seconds with shape ``(..., n_q, 1)``.
        time_v : tf.Tensor
            Key/value time offsets in seconds with shape ``(..., n_v, 1)``.
        apply_head_slopes : bool
            Whether to expand the bias across attention heads and apply the
            per-head slopes.

        Returns
        -------
        tf.Tensor
            ``(..., num_heads, n_q, n_v)`` bias tensor.
        """
        time_q = tf.expand_dims(time_q[..., 0], axis=-1)
        time_v = tf.expand_dims(time_v[..., 0], axis=-2)
        distance = tf.abs(time_q - time_v)
        bias = self.temporal_bias_scale * tf.math.tanh(
            -distance / self.temporal_distance_scale
        )
        return (
            tf.expand_dims(bias, axis=1) * self.head_slopes
            if apply_head_slopes
            else bias
        )

    def _position_bias(
        self, query_lat_lon, query_time, kv_lat_lon=None, kv_time=None
    ):
        """Combine enabled spatial and temporal ALiBi components.

        Parameters
        ----------
        query_lat_lon : tf.Tensor | None
            Query latitude and longitude coordinates.
        query_time : tf.Tensor | None
            Query timestamps.
        kv_lat_lon : tf.Tensor | None
            K/V coordinates. Defaults to the query coordinates.
        kv_time : tf.Tensor | None
            K/V timestamps. Defaults to the query timestamps.
        """
        kv_lat_lon = query_lat_lon if kv_lat_lon is None else kv_lat_lon
        kv_time = query_time if kv_time is None else kv_time
        bias = None
        if self.use_spatial_bias and query_lat_lon is not None:
            bias = self._haversine_bias(
                query_lat_lon, kv_lat_lon, apply_head_slopes=False
            )
        if self.use_temporal_bias and query_time is not None:
            temporal_bias = self._temporal_bias(
                query_time, kv_time, apply_head_slopes=False
            )
            bias = temporal_bias if bias is None else bias + temporal_bias
        return (
            None
            if bias is None
            else tf.expand_dims(bias, axis=1) * self.head_slopes
        )

    @staticmethod
    def _reassemble_windows(output, query, geometry):
        """Reshape windowed output back to the original spatial layout.

        Handles 4D ``(B, H, W, C)`` and 5D ``(B, H, W, T, C)`` query shapes.
        For 4D queries, T is treated as 1 and squeezed via the final reshape to
        ``tf.shape(query)``.
        """
        feat_out = tf.shape(output)[-1]
        t_steps = tf.shape(query)[3] if len(query.shape) == 5 else 1
        ws = geometry.window_size
        win_shape = [
            geometry.batch_size,
            geometry.n_window_rows,
            geometry.n_window_cols,
            ws,
            ws,
            t_steps,
            feat_out,
        ]
        output = tf.transpose(
            tf.reshape(output, win_shape), [0, 1, 3, 2, 4, 5, 6]
        )
        pad_shape = [
            geometry.batch_size,
            geometry.padded_query_height,
            geometry.padded_query_width,
            t_steps,
            feat_out,
        ]
        output = tf.reshape(output, pad_shape)
        ws = geometry.window_shift
        h, w = geometry.query_height, geometry.query_width
        output = output[:, ws : ws + h, ws : ws + w, :, :]
        return tf.reshape(output, tf.shape(query))

    def _full_attention_call(
        self,
        query,
        key,
        value,
        training,
        kv_nan_mask,
        lat_lon,
        time=None,
        use_causal_mask=False,
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

        flat_lat_lon = (
            None
            if lat_lon is None
            else tf.reshape(lat_lon, [batch_size, -1, 2])
        )
        flat_time = (
            None if time is None else tf.reshape(time, [batch_size, -1, 1])
        )

        attention_mask = None
        if kv_nan_mask is not None:
            valid = ~tf.reshape(kv_nan_mask, [batch_size, -1])
            (k_flat, v_flat, kv_lat_lon, kv_time), valid = self._compact_kv(
                (k_flat, v_flat, flat_lat_lon, flat_time), valid
            )
            attention_mask = valid[:, tf.newaxis, :]
        else:
            kv_lat_lon = flat_lat_lon
            kv_time = flat_time
        bias = self._position_bias(
            flat_lat_lon, flat_time, kv_lat_lon, kv_time
        )

        output = super().call(
            query=q_flat,
            value=v_flat,
            key=k_flat,
            attention_mask=attention_mask,
            training=training,
            use_causal_mask=use_causal_mask,
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
        lat_lon,
        window_size,
        time=None,
        use_causal_mask=False,
    ):
        """Execute the full windowed attention path."""
        time_steps = tf.shape(query)[3] if len(query.shape) == 5 else 1
        geometry = self._get_window_geometry(
            query, window_size, time_steps=time_steps
        )

        q_win = self._partition_windows(query, geometry)
        k_win = self._partition_windows(key, geometry)
        v_win = self._partition_windows(value, geometry)

        coords_win = (
            None
            if lat_lon is None
            else self._partition_windows(lat_lon, geometry)
        )
        time_win = (
            None if time is None else self._partition_windows(time, geometry)
        )
        if kv_nan_mask is None:
            kv_mask = self._build_window_mask(None, geometry)
            kv_coords_win = coords_win
            kv_time_win = time_win
        else:
            valid = tf.cast(~kv_nan_mask, tf.bool)[..., tf.newaxis]
            valid = tf.squeeze(
                self._partition_windows(valid, geometry), axis=-1
            )
            (k_win, v_win, kv_coords_win, kv_time_win), kv_valid = (
                self._compact_kv(
                    (k_win, v_win, coords_win, time_win), valid
                )
            )
            kv_mask = kv_valid[:, tf.newaxis, :]
        bias_win = self._position_bias(
            coords_win, time_win, kv_coords_win, kv_time_win
        )

        output = super().call(
            query=q_win,
            value=v_win,
            key=k_win,
            attention_mask=kv_mask,
            training=training,
            use_causal_mask=use_causal_mask,
            bias=bias_win,
        )

        return self._reassemble_windows(output, query, geometry)

    @tf.function
    def call(
        self,
        query,
        value,
        key=None,
        time=None,
        training=None,
        use_causal_mask=False,
        kv_nan_mask=None,
        lat_lon=None,
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
        time : tf.Tensor | None
            Relative seconds with shape ``(B, H, W, T, 1)`` for
            temporal ALiBi bias.
        kv_nan_mask : tf.Tensor | None
            Boolean ``(B, H_v, W_v)`` where True marks NaN KV positions.
            Both the full and windowed paths convert this internally to the
            appropriate attention mask format.
        lat_lon : tf.Tensor | None
            ``(B, H, W, 2)`` coordinate grid for per-window ALiBi bias.
        use_causal_mask : bool
            Forwarded to the underlying ``MultiHeadAttention`` call.
        training : bool | None
            Training flag forwarded to dropout.

        Returns
        -------
        output : tf.Tensor
            ``(batch, H_q, W_q, features)``
        """
        if key is None:
            key = value

        kwargs = dict(
            query=query,
            key=key,
            value=value,
            time=time,
            training=training,
            kv_nan_mask=kv_nan_mask,
            lat_lon=lat_lon,
            use_causal_mask=use_causal_mask,
        )
        return (
            self._full_attention_call(**kwargs)
            if self.window_size is None
            else self._window_attention_call(
                **kwargs, window_size=self.window_size
            )
        )

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'window_size': self.window_size,
            'window_shift': self.window_shift,
            'bias_scale': self.bias_scale,
            'distance_scale': self.distance_scale,
            'temporal_bias_scale': self.temporal_bias_scale,
            'temporal_distance_scale': self.temporal_distance_scale,
        })
        return config


class Sup3rTransformerLayer(tf.keras.layers.Layer):
    """Transformer layer with shared Q encoding, joint K/V attention,
    and a shared post-attention pathway.

    Queries are typically the latent space of the model; keys/values are
    high-resolution features (observations, topography, etc.).  All
    feature channels are processed jointly by a single K/V encoder and
    attention head.

    Spatial and temporal ALiBi independently replace the corresponding
    sinusoidal positional encoding when their bias scale is positive.
    """

    def __init__(
        self,
        name=None,
        features=None,
        exo_features=None,
        patch_size=1,
        num_heads=1,
        embed_dim=64,
        key_dim=None,
        min_period_spatial=1e-4,
        max_period_spatial=2,
        min_period_temporal=1,
        max_period_temporal=864000,
        bias_scale=0.0,
        distance_scale=10000.0,
        temporal_bias_scale=0.0,
        temporal_distance_scale=86400.0,
        window_size=None,
        window_shift=0,
        dropout=0.0,
        encode_query=True,
        decode=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str | None
            Name of layer.
        features : list[str] | None
            List of hi-resolution feature names used for exogenous-data
            dispatch by the calling framework.
        exo_features : list[str] | None
            List of exogenous feature names (latitude, longitude, and time).
        patch_size : int | tuple[int, ...]
            Patch size for each non-batch, non-feature axis. A scalar applies
            the same size to every axis.
        num_heads : int
            Number of attention heads.
        key_dim : int | None
            Size of each attention head. If None, defaults to embed_dim
        embed_dim : int
            Dimension of the tokenized inputs.
        min_period_spatial : float
            Minimum period for the spatial positional encoding.
        max_period_spatial : float
            Maximum period for the spatial positional encoding.
        min_period_temporal : float
            Minimum period for the temporal positional encoding.
        max_period_temporal : float
            Maximum period for the temporal positional encoding.
        bias_scale : float
            Positive values enable ALiBi and set its distance scaling
            factor.  Non-positive values disable ALiBi.
        distance_scale : float
            Length scale (in meters) over which the ALiBi distance bias
            decays. Larger values produce a gentler decay, allowing
            attention to reach farther spatially.
        temporal_bias_scale : float
            Positive values enable temporal ALiBi and set its scaling factor.
            Non-positive values use sinusoidal temporal encoding.
        temporal_distance_scale : float
            Length scale in seconds over which temporal ALiBi decays.
        window_size : int | None
            Side length of the non-overlapping query execution block.
            ``None`` uses full attention.
        window_shift : int
            Shift of the query-window start on the token grid. This is
            only active when the current call uses multiple windows;
            otherwise it is ignored because the layer routes to full
            attention.
        dropout : float
            Dropout rate for attention weights.
        encode_query : bool
            Whether to patch encode the query before attention. Hierarchical
            latent stages disable this because the block owns query encoding.
        decode : bool
            Whether to decode tokens back to the input grid and add the input
            residual. If False, return the encoded latent tokens.
        **kwargs
            Additional keyword arguments for the layer.
        """
        kwargs.setdefault('autocast', False)
        super().__init__(name=name, **kwargs)
        self.features = features or []
        self.exo_features = exo_features or []
        self.rank = None
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.key_dim = key_dim or embed_dim
        self.min_period_spatial = min_period_spatial
        self.max_period_spatial = max_period_spatial
        self.min_period_temporal = min_period_temporal
        self.max_period_temporal = max_period_temporal
        self.window_size = window_size
        self.window_shift = window_shift
        self.bias_scale = float(bias_scale)
        self.distance_scale = float(distance_scale)
        self.temporal_bias_scale = float(temporal_bias_scale)
        self.temporal_distance_scale = float(temporal_distance_scale)
        self.use_spatial_bias = self.bias_scale > 0
        self.use_temporal_bias = self.temporal_bias_scale > 0
        self.use_pos_bias = self.use_spatial_bias or self.use_temporal_bias
        self.dropout = dropout
        self.encode_query = encode_query
        self.decode = decode

        # -- shared layers --
        self.eq = PatchEncoder(
            patch_size=self.patch_size, embed_dim=self.embed_dim
        )
        self.pe = PositionEncoder(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            min_period_spatial=self.min_period_spatial,
            max_period_spatial=self.max_period_spatial,
            min_period_temporal=self.min_period_temporal,
            max_period_temporal=self.max_period_temporal,
        )
        self.lo = tf.keras.layers.RMSNormalization()
        self.lq = tf.keras.layers.RMSNormalization()
        self.lk = tf.keras.layers.RMSNormalization()
        self.mlp = None  # built in build()
        self.decoder = None  # built in build()

        # -- single K/V encoder and attention head (joint across features) --
        self.ek = PatchEncoder(
            patch_size=self.patch_size, embed_dim=self.embed_dim
        )
        self.ev = PatchEncoder(
            patch_size=self.patch_size, embed_dim=self.embed_dim
        )
        self.attn = WindowedMultiHeadAttention(
            window_size=self.window_size,
            window_shift=self.window_shift,
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            bias_scale=self.bias_scale,
            distance_scale=self.distance_scale,
            temporal_bias_scale=self.temporal_bias_scale,
            temporal_distance_scale=self.temporal_distance_scale,
            dropout=self.dropout,
        )

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
            msg = (
                'Sup3rTransformerLayer requires exo_data with latitude and '
                'longitude channels.'
            )
            logger.error(msg)
            raise ValueError(msg)

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
        """Extract coordinates and optional time coordinates."""
        lat_lon = tf.cast(exo_data[..., :2], tf.float32)
        time = (
            None
            if exo_data.shape[-1] < 3
            else exo_data[..., 2:3]
        )
        return lat_lon, time

    def _pad_to_patch_multiple(
        self, tensor, mode='CONSTANT', constant_values=0
    ):
        """Pad patch axes so tokenization preserves the full query domain.

        Parameters
        ----------
        tensor : tf.Tensor | None
            Tensor to pad. Returned unchanged when ``None`` or when every
            patch dimension is one.
        mode : str
            Padding mode forwarded to ``tf.pad``.
        constant_values : float
            Fill value used only when ``mode='CONSTANT'``.
        """
        patch_shape = _parse_patch_size(self.patch_size, self.rank)
        if tensor is None or all(size == 1 for size in patch_shape):
            return tensor

        paddings = [[0, 0]]
        for axis, patch_size in zip(
            range(1, self.rank - 1), patch_shape
        ):
            size = tf.shape(tensor)[axis]
            pad = (
                patch_size - tf.math.floormod(size, patch_size)
            ) % patch_size
            paddings.append([0, pad])
        paddings.append([0, 0])
        if mode == 'CONSTANT':
            return tf.pad(tensor, paddings, constant_values=constant_values)
        return tf.pad(tensor, paddings, mode=mode)

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
        patch_shape = _parse_patch_size(self.patch_size, self.rank)
        if not self.encode_query and (
            any(size != 1 for size in patch_shape)
            or x_shape[-1] != self.embed_dim
        ):
            raise ValueError(
                'Sup3rTransformerLayer with encode_query=False requires '
                'patch_size=1 and input features equal to embed_dim. '
                f'Received patch_size={self.patch_size}, input features='
                f'{x_shape[-1]}, and embed_dim={self.embed_dim}.'
            )
        hr_shape = (
            hi_res_feature_shape
            if hi_res_feature_shape is not None
            else x_shape
        )
        embed_shape = (None,) * (self.rank - 1) + (self.embed_dim,)

        # shared layers
        if self.encode_query:
            self.eq.build(x_shape)
        self.pe.build(x_shape)
        self.lo.build((None, None, self.embed_dim))
        self.lq.build((None, None, self.embed_dim))
        self.lk.build((None, None, self.embed_dim))
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * self.embed_dim),
            SwiGLU(),
            tf.keras.layers.Dense(self.embed_dim),
        ])
        self.mlp.build((None, None, self.embed_dim))
        if self.decode:
            self.decoder = PatchDecoder(
                patch_size=self.patch_size, output_dim=x_shape[-1]
            )
            self.decoder.build(embed_shape)

        # Include per-feature validity so missing and observed zeros differ.
        kv_shape = (*hr_shape[:-1], 2 * hr_shape[-1])
        self.ek.build(kv_shape)
        self.ev.build(kv_shape)
        self.attn.build(embed_shape, embed_shape, embed_shape)

        super().build(x_shape)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'features': self.features,
            'exo_features': self.exo_features,
            'patch_size': _serialize_patch_size(self.patch_size),
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'embed_dim': self.embed_dim,
            'min_period_spatial': self.min_period_spatial,
            'max_period_spatial': self.max_period_spatial,
            'min_period_temporal': self.min_period_temporal,
            'max_period_temporal': self.max_period_temporal,
            'bias_scale': self.bias_scale,
            'distance_scale': self.distance_scale,
            'temporal_bias_scale': self.temporal_bias_scale,
            'temporal_distance_scale': self.temporal_distance_scale,
            'window_size': self.window_size,
            'window_shift': self.window_shift,
            'dropout': self.dropout,
            'encode_query': self.encode_query,
            'decode': self.decode,
        })
        return config

    @tf.function
    def call(
        self,
        x,
        hi_res_feature=None,
        exo_data=None,
        training=None,
    ):
        """Call the transformer layer.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D input tensor (latent space being updated).
        hi_res_feature : tf.Tensor, optional
            4D or 5D high-resolution feature tensor used as key/value
            input.  All channels are processed jointly.
            May contain NaNs for sparse observations.
        exo_data : tf.Tensor
            Exogenous data with latitude/longitude in channels 0:2 and
            optional time coordinates in seconds in channel 2. Required when
            *hi_res_feature* is provided.
        training : bool | None
            Training flag forwarded to dropout layers.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as *x* when decoding, otherwise
            encoded latent tokens with ``embed_dim`` features.
        """
        if hi_res_feature is None:
            return x

        if exo_data is None:
            msg = (
                'Sup3rTransformerLayer requires exo_data with latitude and '
                'longitude channels.'
            )
            logger.error(msg)
            raise ValueError(msg)

        x = tf.cast(x, self.compute_dtype)
        hi_res_feature = tf.cast(hi_res_feature, self.compute_dtype)
        has_observations = tf.reduce_any(
            ~tf.math.is_nan(hi_res_feature),
            axis=tf.range(1, tf.rank(hi_res_feature)),
        )
        has_observations = tf.reshape(
            has_observations,
            tf.concat([
                tf.shape(has_observations),
                tf.ones(self.rank - 1, dtype=tf.int32),
            ], axis=0),
        )

        x_in = x
        original_shape = tf.shape(x)

        x = self._pad_to_patch_multiple(x, mode='REFLECT')
        hi_res_feature = self._pad_to_patch_multiple(
            hi_res_feature, constant_values=float('nan')
        )
        exo_data = self._pad_to_patch_multiple(exo_data, mode='REFLECT')

        lat_lon, time = self._split_exo_inputs(exo_data)

        encoded_query = self.eq(x) if self.encode_query else x
        q = encoded_query
        pool = self.eq.avg_pool if self.encode_query else None
        if pool is not None:
            lat_lon = pool(lat_lon)
            time = pool(time) if time is not None else None

        pos_enc = None
        if not self.use_spatial_bias:
            pos_enc = self.pe.encode_lat_lon(lat_lon)
        if (
            not self.use_temporal_bias
            and self.rank == 5
            and time is not None
        ):
            temporal_enc = self.pe.encode_time(time)
            pos_enc = (
                temporal_enc if pos_enc is None else pos_enc + temporal_enc
            )
        if pos_enc is not None:
            q = q + pos_enc  # noqa: PLR6104

        # joint K/V encoding across all features
        hr_clean, nan_mask = self.ek.prepare_sparse_tensor(hi_res_feature)
        feature_valid = tf.cast(
            ~tf.math.is_nan(hi_res_feature), hr_clean.dtype
        )
        kv_input = tf.concat((hr_clean, feature_valid), axis=-1)
        k = self.ek(kv_input)
        v = self.ev(kv_input)

        if pos_enc is not None:
            k = k + pos_enc  # noqa: PLR6104

        projected = self.attn(
            query=self.lq(q),
            key=self.lk(k),
            value=v,
            kv_nan_mask=nan_mask,
            lat_lon=lat_lon,
            time=time,
            training=training,
        )

        # shared post-attention: residual -> norm -> MLP -> residual
        attn_res = q + projected
        out = self.lo(attn_res)
        out_shape = tf.shape(out)
        out = tf.reshape(out, [out_shape[0], -1, self.embed_dim])
        out = attn_res + tf.reshape(
            self.mlp(out, training=training), out_shape
        )

        if not self.decode:
            return tf.where(has_observations, out, encoded_query)

        decoded = self.decoder(out)
        decoded = tf.slice(
            decoded, tf.zeros(self.rank, dtype=tf.int32), original_shape
        )
        return tf.where(has_observations, x_in + decoded, x_in)


class Sup3rTransformerBlock(tf.keras.layers.Layer):
    """Stack of ``Sup3rTransformerLayer`` instances with optional
    window shifting on odd layers in the stack.

    When ``window_shift`` is non-zero, the window_shift for the i-th layer
    is computed as ``i * window_shift % window_size``."""

    def __init__(
        self,
        name=None,
        features=None,
        exo_features=None,
        n_layers=2,
        window_shift=0,
        **layer_kwargs,
    ):
        """
        Parameters
        ----------
        features : list[str] | None
            List of hi-resolution feature names used for exogenous-data
            dispatch by the calling framework.
        exo_features : list[str] | None
            List of exogenous feature names (latitude, longitude, and time).
        n_layers : int
            Number of ``Sup3rTransformerLayer`` instances in the stack.
        window_shift : int
            Shift increment between layers. The shift for layer ``i`` is
            ``i * window_shift % window_size``. Zero disables shifting.
        name : str | None
            Name for this block.
        **layer_kwargs
            Forwarded to every ``Sup3rTransformerLayer`` constructor
            (e.g. ``embed_dim``, ``window_size``, ``patch_size``, etc.).
            ``window_shift`` in *layer_kwargs* is silently replaced by
            the per-layer value computed here.
        """
        super().__init__(name=name, autocast=False)
        self.features = features
        self.exo_features = exo_features
        self.n_layers = n_layers
        self.window_shift = window_shift
        self.window_size = layer_kwargs.get('window_size')
        self._layer_kwargs = {
            k: v for k, v in layer_kwargs.items() if k != 'window_shift'
        }
        self.transformer_layers = []
        for i in range(n_layers):
            shift = (
                i * window_shift % self.window_size
                if self.window_size is not None
                else 0
            )
            self.transformer_layers.append(
                Sup3rTransformerLayer(
                    features=features,
                    exo_features=exo_features,
                    window_shift=shift,
                    **self._layer_kwargs,
                )
            )

    def build(self, x_shape, hi_res_feature_shape=None, exo_data_shape=None):
        """Build every child transformer layer.

        Parameters
        ----------
        x_shape : tuple
            Shape tuple of the query tensor.
        hi_res_feature_shape : tuple | None
            Shape tuple of the high resolution feature tensor.
        exo_data_shape : tuple | None
            Shape tuple of the exogenous data tensor.
        """
        for layer in self.transformer_layers:
            layer.build(
                x_shape,
                hi_res_feature_shape=hi_res_feature_shape,
                exo_data_shape=exo_data_shape,
            )
        super().build(x_shape)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'features': self.features,
            'exo_features': self.exo_features,
            'n_layers': self.n_layers,
            'window_shift': self.window_shift,
            **self._layer_kwargs,
        })
        return config

    def call(
        self,
        x,
        hi_res_feature=None,
        exo_data=None,
        training=None,
    ):
        """Run *x* through each transformer layer in sequence.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D latent tensor.
        hi_res_feature : tf.Tensor | None
            High-resolution features (passed to every layer).
        exo_data : tf.Tensor | None
            Exogenous latitude/longitude/time data passed to every layer.
        training : bool | None
            Training flag forwarded to dropout layers.

        Returns
        -------
        tf.Tensor
            Same shape as *x*.
        """
        for layer in self.transformer_layers:
            x = layer(
                x,
                hi_res_feature=hi_res_feature,
                exo_data=exo_data,
                training=training,
            )
        return x


class Sup3rHierarchicalTransformerBlock(tf.keras.layers.Layer):
    """Hierarchical sparse-observation cross-attention with spatial skips."""

    def __init__(
        self,
        name=None,
        features=None,
        exo_features=None,
        patch_size=1,
        stage_dims=(64, 128, 256),
        stage_depths=2,
        num_heads=4,
        key_dims=None,
        window_sizes=None,
        window_shifts=0,
        bias_scales=0.0,
        distance_scales=10000.0,
        temporal_bias_scales=0.0,
        temporal_distance_scales=86400.0,
        dropout=0.0,
        min_period_spatial=1e-4,
        max_period_spatial=2,
        min_period_temporal=1,
        max_period_temporal=864000,
        **kwargs,
    ):
        """Initialize the hierarchical transformer block.

        Parameters
        ----------
        name : str | None
            Name of layer.
        features : list[str] | None
            High-resolution observation features used for data dispatch.
        exo_features : list[str] | None
            Latitude, longitude, and optional time features used for dispatch.
        patch_size : int | tuple[int, ...]
            Finest-level patch size. Each deeper level doubles only the two
            spatial patch dimensions.
        stage_dims : sequence[int]
            Embedding width at each unique spatial scale.
        stage_depths : int | sequence[int]
            Number of cross-attention layers at each scale. Decoder stages
            mirror the corresponding encoder depth.
        num_heads : int | sequence[int]
            Number of attention heads at each scale.
        key_dims : int | sequence[int] | None
            Per-head key dimension at each scale. If None, each value is the
            stage embedding width divided by its number of heads.
        window_sizes : int | sequence[int] | None
            Spatial attention-window side length at each scale.
        window_shifts : int | sequence[int]
            Shift increment between attention layers at each scale.
        bias_scales : float | sequence[float]
            Spatial ALiBi scale at each scale.
        distance_scales : float | sequence[float]
            Spatial ALiBi distance scale in meters at each scale.
        temporal_bias_scales : float | sequence[float]
            Temporal ALiBi scale at each scale.
        temporal_distance_scales : float | sequence[float]
            Temporal ALiBi distance scale in seconds at each scale.
        dropout : float | sequence[float]
            Attention dropout rate at each scale.
        min_period_spatial : float
            Minimum period for spatial positional encoding.
        max_period_spatial : float
            Maximum period for spatial positional encoding.
        min_period_temporal : float
            Minimum period for temporal positional encoding.
        max_period_temporal : float
            Maximum period for temporal positional encoding.
        **kwargs
            Additional keyword arguments for the layer.
        """
        kwargs.setdefault('autocast', False)
        super().__init__(name=name, **kwargs)
        self.features = features or []
        self.exo_features = exo_features or []
        self.patch_size = patch_size
        self.stage_dims = list(stage_dims)
        if not self.stage_dims:
            raise ValueError('stage_dims must contain at least one value.')

        n_stages = len(self.stage_dims)
        self.stage_depths = self._stage_values(
            stage_depths, n_stages, 'stage_depths'
        )
        self.num_heads = self._stage_values(
            num_heads, n_stages, 'num_heads'
        )
        self.key_dims = (
            [dim // heads for dim, heads in zip(
                self.stage_dims, self.num_heads
            )]
            if key_dims is None
            else self._stage_values(key_dims, n_stages, 'key_dims')
        )
        self.window_sizes = self._stage_values(
            window_sizes, n_stages, 'window_sizes'
        )
        self.window_shifts = self._stage_values(
            window_shifts, n_stages, 'window_shifts'
        )
        self.bias_scales = self._stage_values(
            bias_scales, n_stages, 'bias_scales'
        )
        self.distance_scales = self._stage_values(
            distance_scales, n_stages, 'distance_scales'
        )
        self.temporal_bias_scales = self._stage_values(
            temporal_bias_scales, n_stages, 'temporal_bias_scales'
        )
        self.temporal_distance_scales = self._stage_values(
            temporal_distance_scales,
            n_stages,
            'temporal_distance_scales',
        )
        self.dropout = self._stage_values(dropout, n_stages, 'dropout')
        self.min_period_spatial = min_period_spatial
        self.max_period_spatial = max_period_spatial
        self.min_period_temporal = min_period_temporal
        self.max_period_temporal = max_period_temporal
        self.rank = None
        self.initial_encoder = PatchEncoder(
            patch_size=patch_size, embed_dim=self.stage_dims[0]
        )
        self.merge_encoders = []
        self.expand_decoders = []
        self.final_decoder = None
        self.encoder_stages = [
            self._make_stage(i, f'encoder_stage_{i}')
            for i in range(n_stages)
        ]
        self.decoder_stages = [
            self._make_stage(i, f'decoder_stage_{i}')
            for i in range(n_stages - 1)
        ]

    @staticmethod
    def _stage_values(value, n_stages, name):
        """Expand a scalar stage setting or validate a stage sequence."""
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(values) == 1:
            values *= n_stages
        if len(values) != n_stages:
            raise ValueError(
                f'{name} must contain one or {n_stages} values, but '
                f'received {len(values)}.'
            )
        return values

    def _make_stage(self, stage_index, name):
        """Create one same-resolution latent cross-attention stage."""
        layers = []
        window_size = self.window_sizes[stage_index]
        for layer_index in range(self.stage_depths[stage_index]):
            shift = (
                layer_index * self.window_shifts[stage_index] % window_size
                if window_size is not None
                else 0
            )
            layer = Sup3rTransformerLayer(
                name=f'{name}_layer_{layer_index}',
                patch_size=1,
                num_heads=self.num_heads[stage_index],
                embed_dim=self.stage_dims[stage_index],
                key_dim=self.key_dims[stage_index],
                min_period_spatial=self.min_period_spatial,
                max_period_spatial=self.max_period_spatial,
                min_period_temporal=self.min_period_temporal,
                max_period_temporal=self.max_period_temporal,
                bias_scale=self.bias_scales[stage_index],
                distance_scale=self.distance_scales[stage_index],
                temporal_bias_scale=self.temporal_bias_scales[stage_index],
                temporal_distance_scale=(
                    self.temporal_distance_scales[stage_index]
                ),
                window_size=window_size,
                window_shift=shift,
                dropout=self.dropout[stage_index],
                encode_query=False,
                decode=False,
            )
            setattr(self, layer.name, layer)
            layers.append(layer)
        return layers

    def build(self, x_shape, hi_res_feature_shape=None, exo_data_shape=None):
        """Build spatial transitions and the final decoder.

        Parameters
        ----------
        x_shape : tuple
            Shape tuple of the query tensor.
        hi_res_feature_shape : tuple | None
            Shape tuple of the sparse observation tensor.
        exo_data_shape : tuple | None
            Shape tuple of the coordinate tensor.
        """
        self.rank = len(x_shape)
        patch_shape = _parse_patch_size(self.patch_size, self.rank)
        if self.rank == 5 and patch_shape[-1] != 1:
            raise ValueError(
                'Sup3rHierarchicalTransformerBlock requires temporal patch '
                'size 1.'
            )
        merge_patch = (2, 2) if self.rank == 4 else (2, 2, 1)
        self.merge_encoders = []
        self.expand_decoders = []
        for stage_index, embed_dim in enumerate(self.stage_dims[1:], start=1):
            encoder = PatchEncoder(
                name=f'merge_encoder_{stage_index}',
                patch_size=merge_patch,
                embed_dim=embed_dim,
            )
            setattr(self, encoder.name, encoder)
            self.merge_encoders.append(encoder)
        for stage_index in range(len(self.stage_dims) - 1):
            decoder = PatchDecoder(
                name=f'expand_decoder_{stage_index}',
                patch_size=merge_patch,
                output_dim=self.stage_dims[stage_index],
            )
            setattr(self, decoder.name, decoder)
            self.expand_decoders.append(decoder)
        self.final_decoder = PatchDecoder(
            patch_size=self.patch_size, output_dim=x_shape[-1]
        )

        query_shape = tuple(x_shape[:-1]) + (self.stage_dims[0],)
        observation_shape = tuple(hi_res_feature_shape)
        coordinate_shape = tuple(exo_data_shape)
        self.initial_encoder.build(x_shape)
        for layer in self.encoder_stages[0]:
            layer.build(query_shape, observation_shape, coordinate_shape)

        query_shapes = [query_shape]
        for stage_index, encoder in enumerate(self.merge_encoders, start=1):
            encoder.build(query_shape)
            query_shape = tuple(query_shape[:-1]) + (
                self.stage_dims[stage_index],
            )
            query_shapes.append(query_shape)
            for layer in self.encoder_stages[stage_index]:
                layer.build(query_shape, observation_shape, coordinate_shape)

        for stage_index in range(len(self.stage_dims) - 2, -1, -1):
            self.expand_decoders[stage_index].build(query_shape)
            query_shape = query_shapes[stage_index]
            for layer in self.decoder_stages[stage_index]:
                layer.build(query_shape, observation_shape, coordinate_shape)

        self.final_decoder.build(query_shape)
        super().build(x_shape)

    def _pad_to_multiple(
        self, tensor, patch_size, mode='CONSTANT', constant_values=0
    ):
        """Pad non-feature axes to a patch multiple."""
        patch_shape = _parse_patch_size(patch_size, self.rank)
        paddings = [[0, 0]]
        for axis, size in zip(range(1, self.rank - 1), patch_shape):
            remainder = tf.math.floormod(tf.shape(tensor)[axis], size)
            paddings.append([0, tf.math.floormod(size - remainder, size)])
        paddings.append([0, 0])
        if mode == 'CONSTANT':
            return tf.pad(tensor, paddings, constant_values=constant_values)
        return tf.pad(tensor, paddings, mode=mode)

    def _encode_level(
        self, x, observations, observation_weights, exo_data, encoder
    ):
        """Patch encode a query and pool its aligned observation inputs.

        Parameters
        ----------
        x : tf.Tensor
            Query tensor at the current hierarchy level.
        observations : tf.Tensor
            Sparse observations aligned with ``x``.
        observation_weights : tf.Tensor | None
            Valid counts represented by each observation value.
        exo_data : tf.Tensor
            Coordinates aligned with ``x``.
        encoder : PatchEncoder
            Encoder defining the next hierarchy resolution.

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            Encoded query, pooled observations, pooled observation weights,
            and pooled coordinates.
        """
        x = self._pad_to_multiple(x, encoder.patch_size, mode='REFLECT')
        observations = self._pad_to_multiple(
            observations, encoder.patch_size, constant_values=float('nan')
        )
        if observation_weights is not None:
            observation_weights = self._pad_to_multiple(
                observation_weights, encoder.patch_size
            )
        exo_data = self._pad_to_multiple(
            exo_data, encoder.patch_size, mode='REFLECT'
        )
        x = encoder(x)
        observations, observation_weights = encoder.pool_sparse_tensor(
            observations,
            weights=observation_weights,
            return_weights=True,
        )
        if not all(size == 1 for size in encoder.patch_shape):
            exo_data = encoder.avg_pool(exo_data)
        return x, observations, observation_weights, exo_data

    @staticmethod
    def _crop_like(x, reference):
        """Crop a decoded tensor to the non-feature shape of a reference."""
        size = tf.concat([tf.shape(reference)[:-1], [tf.shape(x)[-1]]], axis=0)
        return tf.slice(x, tf.zeros(tf.rank(x), dtype=tf.int32), size)

    def _run_stage(self, layers, x, observations, exo_data, training):
        """Run a latent tensor through one cross-attention stage."""
        for layer in layers:
            x = layer(
                x,
                hi_res_feature=observations,
                exo_data=exo_data,
                training=training,
            )
        return x

    def call(
        self,
        x,
        hi_res_feature=None,
        exo_data=None,
        training=None,
    ):
        """Run hierarchical sparse-observation assimilation.

        Parameters
        ----------
        x : tf.Tensor
            4D or 5D query tensor.
        hi_res_feature : tf.Tensor | None
            Sparse observations aligned with the query grid.
        exo_data : tf.Tensor | None
            Latitude, longitude, and optional time coordinates.
        training : bool | None
            Training flag forwarded to attention layers.

        Returns
        -------
        tf.Tensor
            Assimilated tensor with the same shape as ``x``.
        """
        if hi_res_feature is None:
            return x
        if exo_data is None:
            raise ValueError(
                'Sup3rHierarchicalTransformerBlock requires exo_data.'
            )

        x = tf.cast(x, self.compute_dtype)
        observations = tf.cast(hi_res_feature, self.compute_dtype)
        exo_data = tf.cast(exo_data, self.compute_dtype)
        x_in = x
        has_observations = tf.reduce_any(
            ~tf.math.is_nan(observations),
            axis=tf.range(1, tf.rank(observations)),
        )
        observation_weights = None
        x, observations, observation_weights, exo_data = self._encode_level(
            x,
            observations,
            observation_weights,
            exo_data,
            self.initial_encoder,
        )
        x = self._run_stage(
            self.encoder_stages[0], x, observations, exo_data, training
        )
        skips = [x]
        observation_levels = [observations]
        exo_levels = [exo_data]

        for stage_index, encoder in enumerate(self.merge_encoders, start=1):
            x, observations, observation_weights, exo_data = (
                self._encode_level(
                    x,
                    observations,
                    observation_weights,
                    exo_data,
                    encoder,
                )
            )
            x = self._run_stage(
                self.encoder_stages[stage_index],
                x,
                observations,
                exo_data,
                training,
            )
            skips.append(x)
            observation_levels.append(observations)
            exo_levels.append(exo_data)

        for stage_index in range(len(self.stage_dims) - 2, -1, -1):
            x = self.expand_decoders[stage_index](x)
            x = self._crop_like(x, skips[stage_index]) + skips[stage_index]
            x = self._run_stage(
                self.decoder_stages[stage_index],
                x,
                observation_levels[stage_index],
                exo_levels[stage_index],
                training,
            )

        x = self.final_decoder(x)
        output = x_in + self._crop_like(x, x_in)
        has_observations = tf.reshape(
            has_observations,
            tf.concat([
                tf.shape(has_observations),
                tf.ones(self.rank - 1, dtype=tf.int32),
            ], axis=0),
        )
        return tf.where(has_observations, output, x_in)

    def get_config(self):
        """Get config for Keras serialization."""
        config = super().get_config()
        config.update({
            'features': self.features,
            'exo_features': self.exo_features,
            'patch_size': _serialize_patch_size(self.patch_size),
            'stage_dims': self.stage_dims,
            'stage_depths': self.stage_depths,
            'num_heads': self.num_heads,
            'key_dims': self.key_dims,
            'window_sizes': self.window_sizes,
            'window_shifts': self.window_shifts,
            'bias_scales': self.bias_scales,
            'distance_scales': self.distance_scales,
            'temporal_bias_scales': self.temporal_bias_scales,
            'temporal_distance_scales': self.temporal_distance_scales,
            'dropout': self.dropout,
            'min_period_spatial': self.min_period_spatial,
            'max_period_spatial': self.max_period_spatial,
            'min_period_temporal': self.min_period_temporal,
            'max_period_temporal': self.max_period_temporal,
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

    @tf.function(reduce_retracing=True)
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

    @tf.function(reduce_retracing=True)
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
                tf.shape(x)[1] * self._spatial_mult,
                tf.shape(x)[2] * self._spatial_mult,
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
                tf.shape(x)[2],
                tf.shape(x)[3] * self._temporal_mult,
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
                tf.shape(x)[1] * self._spatial_mult,
                tf.shape(x)[2] * self._spatial_mult,
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
