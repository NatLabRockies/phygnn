# -*- coding: utf-8 -*-
"""
Tensorflow utilities
"""

import tensorflow as tf


def tf_isin(a, b):
    """Check whether a is in b"""
    assert isinstance(b, (list, tuple)), 'Second arg must be a list or tuple!'
    x = [tf.equal(a, i) for i in b]
    out = tf.reduce_any(tf.stack(x), axis=0)
    assert out.shape == a.shape
    return out


def tf_log10(x):
    """Compute log base 10 of a tensor x"""
    num = tf.math.log(x)
    den = tf.math.log(tf.constant(10, dtype=num.dtype))
    return num / den


def _mean_fill(x, mask):
    """Fill NaN values in the input tensor with the mean of the non-NaN.
    If all values are NaN, fill with 0. This is assumed to be either a 3D
    or 4D tensor, with the trailing feature channel removed.

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (n_obs, spatial_1, spatial_2) or
        (n_obs, spatial_1, spatial_2, n_temporal)
    mask : tf.Tensor
        Boolean mask of the non-NaN locations in the original x

    Returns
    -------
    x : tf.Tensor
        Input tensor with NaN values filled with the mean of the non-NaN
        values or 0 if all values are NaN.
    mask : tf.Tensor
        Mask of the input tensor where True is not NaN and False is NaN.
    """
    if tf.reduce_all(tf.math.logical_not(mask)):
        return tf.zeros_like(x)
    mean = tf.reduce_mean(x[mask])
    return tf.where(mask, x, mean)


def mean_fill(x):
    """Fill NaN values in the input tensor with the mean of the non-NaN.
    If all values are NaN, fill with 0. This is assumed to be either a 4D
    or 5D tensor, with the trailing feature channel included.

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (n_obs, spatial_1, spatial_2, n_features) or
        (n_obs, spatial_1, spatial_2, n_temporal, n_features)

    Returns
    -------
    x_filled : tf.Tensor
        Input tensor with NaN values filled with the mean of the non-NaN
        values or 0 if all values are NaN.
    mask : tf.Tensor
        Mask of the input tensor where 1 is not NaN and 0 is NaN.
    """
    mask = tf.math.logical_not(tf.math.is_nan(x))
    hr_feat = [_mean_fill(x[..., i], mask[..., i]) for i in range(x.shape[-1])]
    hr_feat = tf.stack(hr_feat, axis=-1)
    return hr_feat, tf.cast(mask, x.dtype)


def _idw_fill(x, coords, mask):
    """IDW fill for flattened tensors without depth dimension. Assumes input
    shape is (N), where N is the number of points (H * W).

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (N)
    coords: tf.Tensor
        Coordinates of the points in the image, shape (N, 2) where N is the
        number of points (H * W)
    mask : tf.Tensor
        Mask of the input tensor where True is not NaN and False is NaN.

    Returns
    -------
    x_filled: tf.Tensor
        Flattened tensor with NaNs filled
    """
    nan_mask = tf.math.logical_not(mask)
    valid_coords = tf.boolean_mask(coords, mask)
    nan_coords = tf.boolean_mask(coords, nan_mask)
    valid_vals = tf.boolean_mask(x, mask)

    diffs = tf.expand_dims(nan_coords, 1) - tf.expand_dims(valid_coords, 0)
    dists = tf.norm(diffs, axis=-1)

    weights = 1 / dists
    weights /= tf.reduce_sum(weights, axis=1, keepdims=True)

    vals = tf.reduce_sum(weights * tf.expand_dims(valid_vals, 0), axis=1)

    filled = tf.where(mask, x, tf.zeros_like(x))
    return tf.tensor_scatter_nd_update(filled, tf.where(nan_mask), vals)


def idw_fill(x, low_mem=True):
    """IDW fill for tensors with channel dimension. Assumes input shape
    is (n_obs, spatial_1, spatial_2, n_features) or (n_obs, spatial_1,
    spatial_2, n_temporal, n_features).

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (n_obs, spatial_1, spatial_2, n_features) or
        (n_obs, spatial_1, spatial_2, n_temporal, n_features)
    low_mem : bool
        If True, use a low memory implementation that loops over time and
        channel dimensions. This is slower but uses less memory.

    Returns
    -------
    x_filled: tf.Tensor
        Tensor with NaNs filled
    mask : tf.Tensor
        Mask of the input tensor where 1 is not NaN and 0 is NaN.
    """
    rank = len(x.shape)
    assert rank in [4, 5], 'Input tensor must be 4D or 5D'
    x = tf.expand_dims(x, axis=-2) if rank == 4 else x
    mask = tf.math.logical_not(tf.math.is_nan(x))
    B, H, W, D, C = x.shape

    if low_mem:
        N = H * W
        coords = tf.meshgrid(
            tf.cast(tf.range(H), x.dtype),
            tf.cast(tf.range(W), x.dtype),
            indexing='ij',
        )
        coords = tf.reshape(tf.stack(coords, axis=-1), [N, 2])
        x_flat = tf.reshape(x, [B, N, D, C])
        mask_flat = tf.reshape(mask, [B, N, D, C])

        filled = []
        for b in range(B):
            x_c = []
            for c in range(C):
                x_d = [
                    _idw_fill(
                        x_flat[b, :, d, c], coords, mask_flat[b, :, d, c]
                    )
                    for d in range(D)
                ]
                x_c.append(tf.stack(x_d, axis=-1))
            filled.append(tf.stack(x_c, axis=-1))
    else:
        N = H * W * D
        # weigh time (D) dimension higher than spatial dimensions. hacky, but
        # efficient, way of ensuring spatial weights are computed only with
        # distances from the same time while vectorizing across time
        coords = tf.meshgrid(
            tf.range(H),
            tf.range(W),
            int(1e6) * tf.range(D),
            indexing='ij',
        )
        coords = tf.reshape(tf.stack(coords, axis=-1), [N, 3])
        coords = tf.cast(coords, x.dtype)
        x_flat = tf.reshape(x, [B, N, C])
        mask_flat = tf.reshape(mask, [B, N, C])
        filled = []
        for b in range(B):
            x_c = [
                _idw_fill(x_flat[b, ..., c], coords, mask_flat[b, ..., c])
                for c in range(C)
            ]
            filled.append(tf.stack(x_c, axis=-1))
    filled = tf.stack(filled, axis=0)
    filled = tf.reshape(filled, [B, H, W, D, C])
    filled = tf.squeeze(filled, axis=-2) if rank == 4 else filled
    mask = tf.squeeze(mask, axis=-2) if rank == 4 else mask
    return tf.cast(filled, x.dtype), tf.cast(mask, x.dtype)
