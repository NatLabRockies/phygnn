"""
Test the custom tensorflow utilities
"""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import tensorflow as tf

from phygnn import TfModel
from phygnn.layers.custom_layers import (
    Embedder,
    ExpandDims,
    FlattenAxis,
    FunctionalLayer,
    GaussianAveragePooling2D,
    GaussianNoiseAxis,
    LogTransform,
    MaskedSqueezeAndExcitation,
    PatchLayer,
    PositionEncoder,
    SigLin,
    SkipConnection,
    SpatioTemporalExpansion,
    Sup3rConcatObs,
    Sup3rObsModel,
    Sup3rTransformerLayer,
    Sup3rTransformerLayerAlibi,
    TileLayer,
    UnitConversion,
)
from phygnn.layers.handlers import HiddenLayers, Layers


@pytest.mark.parametrize(
    'hidden_layers',
    [None, [{'units': 64, 'name': 'relu1'}, {'units': 64, 'name': 'relu2'}]],
)
def test_layers(hidden_layers):
    """Test Layers handler"""
    n_features = 1
    n_labels = 1
    layers = Layers(n_features, n_labels=n_labels, hidden_layers=hidden_layers)
    n_layers = len(hidden_layers) + 2 if hidden_layers is not None else 2
    assert len(layers) == n_layers


def test_dropouts():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers = [
        {'units': 64, 'name': 'relu1', 'dropout': 0.1},
        {'units': 64, 'name': 'relu2', 'dropout': 0.1},
    ]
    layers = HiddenLayers(hidden_layers)

    assert len(layers) == 4


def test_activate():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers = [
        {'units': 64, 'activation': 'relu', 'name': 'relu1'},
        {'units': 64, 'activation': 'relu', 'name': 'relu2'},
    ]
    layers = HiddenLayers(hidden_layers)

    assert len(layers) == 4


def test_batch_norm():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers = [
        {'units': 64, 'name': 'relu1', 'batch_normalization': {'axis': -1}},
        {'units': 64, 'name': 'relu2', 'batch_normalization': {'axis': -1}},
    ]
    layers = HiddenLayers(hidden_layers)

    assert len(layers) == 4


def test_complex_layers():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers = [
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'units': 64},
        {'batch_normalization': {'axis': -1}},
        {'activation': 'relu'},
        {'dropout': 0.01},
    ]
    layers = HiddenLayers(hidden_layers)

    assert len(layers) == 7


def test_repeat_layers():
    """Test repeat argument to duplicate layers"""
    hidden_layers = [
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {
            'n': 3,
            'repeat': [
                {'units': 64},
                {'activation': 'relu'},
                {'dropout': 0.01},
            ],
        },
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers) == 12

    hidden_layers = [
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'n': 3, 'repeat': {'units': 64}},
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers) == 6

    hidden_layers = [
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'repeat': {'units': 64}},
    ]
    with pytest.raises(KeyError):
        layers = HiddenLayers(hidden_layers)


def test_skip_concat_connection():
    """Test a functional skip connection with concatenation"""
    hidden_layers = [
        {
            'class': 'Conv2D',
            'filters': 4,
            'kernel_size': 3,
            'activation': 'relu',
            'padding': 'same',
        },
        {'class': 'SkipConnection', 'name': 'a', 'method': 'concat'},
        {
            'class': 'Conv2D',
            'filters': 4,
            'kernel_size': 3,
            'activation': 'relu',
            'padding': 'same',
        },
        {'class': 'SkipConnection', 'name': 'a'},
        {
            'class': 'Conv2D',
            'filters': 4,
            'kernel_size': 3,
            'activation': 'relu',
            'padding': 'same',
        },
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 5

    skip_layers = [x for x in layers.layers if isinstance(x, SkipConnection)]
    assert len(skip_layers) == 2
    assert id(skip_layers[0]) == id(skip_layers[1])

    x = np.ones((5, 10, 10, 4))
    cache = None
    x_input = None

    for i, layer in enumerate(layers):
        if i == 1:  # skip start
            cache = tf.identity(x)
            assert id(cache) != id(x)
        elif i == 3:  # skip end
            x_input = tf.identity(x)
            assert id(x_input) != id(x)

        x = layer(x)

        if i == 1:  # skip start
            assert layer._cache is not None
        elif i == 2:
            assert np.allclose(cache.numpy(), layers[3]._cache.numpy())
        elif i == 3:  # skip end
            assert layer._cache is None
            tf.assert_equal(x, tf.concat((x_input, cache), axis=-1))


def test_skip_connection():
    """Test a functional skip connection"""
    hidden_layers = [
        {
            'class': 'Conv2D',
            'filters': 4,
            'kernel_size': 3,
            'activation': 'relu',
            'padding': 'same',
        },
        {'class': 'SkipConnection', 'name': 'a'},
        {
            'class': 'Conv2D',
            'filters': 4,
            'kernel_size': 3,
            'activation': 'relu',
            'padding': 'same',
        },
        {'class': 'SkipConnection', 'name': 'a'},
        {
            'class': 'Conv2D',
            'filters': 4,
            'kernel_size': 3,
            'activation': 'relu',
            'padding': 'same',
        },
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 5

    skip_layers = [x for x in layers.layers if isinstance(x, SkipConnection)]
    assert len(skip_layers) == 2
    assert id(skip_layers[0]) == id(skip_layers[1])

    x = np.ones((5, 10, 10, 4))
    cache = None
    x_input = None

    for i, layer in enumerate(layers):
        if i == 1:  # skip start
            cache = tf.identity(x)
            assert id(cache) != id(x)
        elif i == 3:  # skip end
            x_input = tf.identity(x)
            assert id(x_input) != id(x)

        x = layer(x)

        if i == 1:  # skip start
            assert layer._cache is not None
        elif i == 2:
            assert np.allclose(cache.numpy(), layers[3]._cache.numpy())
        elif i == 3:  # skip end
            assert layer._cache is None
            tf.assert_equal(x, tf.add(x_input, cache))


def test_double_skip():
    """Test two skip connections (4 layers total) with the same name. Gotta
    make sure the 1st skip data != 2nd skip data."""
    hidden_layers = [
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'class': 'SkipConnection', 'name': 'a'},
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'class': 'SkipConnection', 'name': 'a'},
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'class': 'SkipConnection', 'name': 'a'},
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'class': 'SkipConnection', 'name': 'a'},
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 16

    skip_layers = [x for x in layers.layers if isinstance(x, SkipConnection)]
    assert len(skip_layers) == 4
    assert len({id(layer) for layer in skip_layers}) == 1

    x = np.ones((5, 3))
    cache = None
    x_input = None

    for i, layer in enumerate(layers):
        if i in {3, 11}:  # skip start
            cache = tf.identity(x)
            assert id(cache) != id(x)
        elif i in {7, 15}:  # skip end
            x_input = tf.identity(x)
            assert id(x_input) != id(x)

        x = layer(x)

        if i in {3, 11}:  # skip start
            assert layer._cache is not None
        elif i in {7, 15}:  # skip end
            assert layer._cache is None
            tf.assert_equal(x, tf.add(x_input, cache))


@pytest.mark.parametrize(
    ('t_mult', 's_mult'), ((1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (5, 3))
)
def test_st_expansion(t_mult, s_mult):
    """Test the spatiotemporal expansion layer."""
    layer = SpatioTemporalExpansion(spatial_mult=s_mult, temporal_mult=t_mult)
    n_filters = 2 * s_mult**2
    x = np.ones((123, 10, 10, 24, n_filters))
    y = layer(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == s_mult * x.shape[1]
    assert y.shape[2] == s_mult * x.shape[2]
    assert y.shape[3] == t_mult * x.shape[3]
    assert y.shape[4] == x.shape[4] / (s_mult**2)


@pytest.mark.parametrize(
    ('spatial_method'), ('depth_to_space', 'bilinear', 'nearest')
)
def test_st_expansion_with_spatial_meth(spatial_method):
    """Test the spatiotemporal expansion layer with different spatial resize
    methods."""
    s_mult = 3
    t_mult = 5
    layer = SpatioTemporalExpansion(
        spatial_mult=s_mult,
        temporal_mult=t_mult,
        spatial_method=spatial_method,
    )
    n_filters = 2 * s_mult**2
    x = np.ones((123, 10, 10, 24, n_filters))
    y = layer(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == s_mult * x.shape[1]
    assert y.shape[2] == s_mult * x.shape[2]
    assert y.shape[3] == t_mult * x.shape[3]

    if spatial_method == 'depth_to_space':
        assert y.shape[4] == x.shape[4] / (s_mult**2)


@pytest.mark.parametrize(
    ('t_mult', 's_mult', 't_roll'),
    (
        (2, 1, 0),
        (2, 1, 1),
        (1, 2, 0),
        (2, 2, 0),
        (2, 2, 1),
        (5, 3, 0),
        (5, 1, 0),
        (5, 1, 2),
        (5, 1, 3),
        (5, 2, 3),
        (24, 1, 12),
    ),
)
def test_temporal_depth_to_time(t_mult, s_mult, t_roll):
    """Test the spatiotemporal expansion layer."""
    layer = SpatioTemporalExpansion(
        spatial_mult=s_mult,
        temporal_mult=t_mult,
        temporal_method='depth_to_time',
        t_roll=t_roll,
    )
    n_filters = 2 * s_mult**2 * t_mult
    shape = (1, 4, 4, 3, n_filters)
    n = np.prod(shape)
    x = np.arange(n).reshape(shape)
    y = layer(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == s_mult * x.shape[1]
    assert y.shape[2] == s_mult * x.shape[2]
    assert y.shape[3] == t_mult * x.shape[3]
    assert y.shape[4] == x.shape[4] / (t_mult * s_mult**2)
    if s_mult == 1:
        for idy in range(y.shape[3]):
            idx = np.maximum(0, idy - t_roll) // t_mult
            even = ((idy - t_roll) % t_mult) == 0
            x1, y1 = x[0, :, :, idx, 0], y[0, :, :, idy, 0]
            if even:
                assert np.allclose(x1, y1)
            else:
                assert not np.allclose(x1, y1)


def test_st_expansion_new_shape():
    """Test that the spatiotemporal expansion layer can expand multiple shapes
    and is not bound to the shape it was built on (bug found on 3/16/2022.)"""
    s_mult = 3
    t_mult = 6
    layer = SpatioTemporalExpansion(spatial_mult=s_mult, temporal_mult=t_mult)
    n_filters = 2 * s_mult**2
    x = np.ones((32, 10, 10, 24, n_filters))
    y = layer(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == s_mult * x.shape[1]
    assert y.shape[2] == s_mult * x.shape[2]
    assert y.shape[3] == t_mult * x.shape[3]
    assert y.shape[4] == x.shape[4] / (s_mult**2)

    x = np.ones((32, 11, 11, 36, n_filters))
    y = layer(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == s_mult * x.shape[1]
    assert y.shape[2] == s_mult * x.shape[2]
    assert y.shape[3] == t_mult * x.shape[3]
    assert y.shape[4] == x.shape[4] / (s_mult**2)


def test_st_expansion_bad():
    """Test an illegal spatial expansion request due to number of channels not
    able to unpack into spatiotemporal dimensions."""
    layer = SpatioTemporalExpansion(spatial_mult=2, temporal_mult=2)
    x = np.ones((123, 10, 10, 24, 3))
    with pytest.raises(RuntimeError):
        _ = layer(x)


@pytest.mark.parametrize(
    ('hidden_layers'),
    (
        ([
            {
                'class': 'FlexiblePadding',
                'paddings': [[1, 1], [2, 2]],
                'mode': 'REFLECT',
            }
        ]),
        ([
            {
                'class': 'FlexiblePadding',
                'paddings': [[1, 1], [2, 2]],
                'mode': 'CONSTANT',
            }
        ]),
        ([
            {
                'class': 'FlexiblePadding',
                'paddings': [[1, 1], [2, 2]],
                'mode': 'SYMMETRIC',
            }
        ]),
    ),
)
def test_flexible_padding(hidden_layers):
    """Test flexible padding routine"""
    layer = HiddenLayers(hidden_layers).layers[0]
    t = tf.constant([[1, 2, 3], [4, 5, 6]])
    if layer.mode.upper() == 'CONSTANT':
        t_check = tf.constant([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0, 0],
            [0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
    elif layer.mode.upper() == 'REFLECT':
        t_check = tf.constant([
            [6, 5, 4, 5, 6, 5, 4],
            [3, 2, 1, 2, 3, 2, 1],
            [6, 5, 4, 5, 6, 5, 4],
            [3, 2, 1, 2, 3, 2, 1],
        ])
    elif layer.mode.upper() == 'SYMMETRIC':
        t_check = tf.constant([
            [2, 1, 1, 2, 3, 3, 2],
            [2, 1, 1, 2, 3, 3, 2],
            [5, 4, 4, 5, 6, 6, 5],
            [5, 4, 4, 5, 6, 6, 5],
        ])
    tf.assert_equal(layer(t), t_check)


def test_flatten_axis():
    """Test the layer to flatten the temporal dimension into the axis-0
    observation dimension.
    """
    layer = FlattenAxis(axis=3)
    x = np.ones((5, 10, 10, 4, 2))
    y = layer(x)
    assert len(y.shape) == 4
    assert y.shape[0] == 5 * 4
    assert y.shape[1] == 10
    assert y.shape[2] == 10
    assert y.shape[3] == 2


def test_expand_dims():
    """Test the layer to expand a new dimension"""
    layer = ExpandDims(axis=3)
    x = np.ones((5, 10, 10, 2))
    y = layer(x)
    assert len(y.shape) == 5
    assert y.shape[0] == 5
    assert y.shape[1] == 10
    assert y.shape[2] == 10
    assert y.shape[3] == 1
    assert y.shape[4] == 2


def test_tile():
    """Test the layer to tile (repeat) an existing dimension"""
    layer = TileLayer(multiples=[1, 0, 2, 3])
    x = np.ones((5, 10, 10, 2))
    y = layer(x)
    assert len(y.shape) == 4
    assert y.shape[0] == 5
    assert y.shape[1] == 0
    assert y.shape[2] == 20
    assert y.shape[3] == 6


def test_noise_axis():
    """Test the custom noise layer on a single axis"""

    # apply random noise along axis=3 (temporal axis)
    layer = GaussianNoiseAxis(axis=3)
    x = np.ones((16, 4, 4, 12, 8), dtype=np.float32)
    y = layer(x)

    # axis=3 should all have unique random values
    rand_axis = y[0, 0, 0, :, 0].numpy()
    assert len(set(rand_axis)) == len(rand_axis)

    # slices along other axis should be the same random number
    for i in range(4):
        for axis in (0, 1, 2, 4):
            slice_tuple = [i] * 5
            slice_tuple[axis] = slice(None)
            slice_tuple = tuple(slice_tuple)

            assert all(y[slice_tuple] == rand_axis[i])


def test_squeeze_excite_2d():
    """Test the SqueezeAndExcitation layer with 2D data (4D tensor input)"""
    hidden_layers = [
        {'class': 'Conv2D', 'filters': 8, 'kernel_size': 3},
        {'activation': 'relu'},
        {'class': 'SqueezeAndExcitation'},
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 3

    x = np.random.normal(0, 1, size=(1, 4, 4, 3))

    for layer in layers:
        x_in = x
        x = layer(x)
        with pytest.raises(tf.errors.InvalidArgumentError):
            tf.assert_equal(x_in, x)


def test_squeeze_excite_3d():
    """Test the SqueezeAndExcitation layer with 3D data (5D tensor input)"""
    hidden_layers = [
        {'class': 'Conv3D', 'filters': 8, 'kernel_size': 3},
        {'activation': 'relu'},
        {'class': 'SqueezeAndExcitation'},
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 3

    x = np.random.normal(0, 1, size=(1, 4, 4, 6, 3))

    for layer in layers:
        x_in = x
        x = layer(x)
        with pytest.raises(tf.errors.InvalidArgumentError):
            tf.assert_equal(x_in, x)


def test_cbam_2d():
    """Test the CBAM layer with 2D data (4D tensor input)"""
    hidden_layers = [{'class': 'CBAM'}]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 1

    x = np.random.normal(0, 1, size=(1, 4, 4, 3))

    for layer in layers:
        x_in = x
        x = layer(x)
        assert x.shape == x_in.shape
        with pytest.raises(tf.errors.InvalidArgumentError):
            tf.assert_equal(x_in, x)


def test_cbam_3d():
    """Test the CBAM layer with 3D data (5D tensor input)"""
    hidden_layers = [{'class': 'CBAM'}]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 1

    x = np.random.normal(0, 1, size=(1, 10, 10, 6, 3))

    for layer in layers:
        x_in = x
        x = layer(x)
        assert x.shape == x_in.shape
        with pytest.raises(tf.errors.InvalidArgumentError):
            tf.assert_equal(x_in, x)


def test_transformer_layer_2d():
    """Test the transformer layer with 2D data (4D tensor input)"""
    hidden_layers = [
        {
            'class': 'Sup3rTransformerLayer',
            'embed_dim': 8,
            'key_dim': 8,
        }
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 1

    x = np.random.normal(0, 1, size=(4, 4, 4, 3))
    y = np.random.uniform(0, 1, size=(4, 4, 4, 1))
    mask = np.random.choice([False, True], (1, 4, 4), p=[0.1, 0.9])
    mask = np.repeat(mask, 4, axis=0)  # shape (4, 4, 4)
    y[mask] = np.nan

    lat = np.linspace(30, 40, 4).reshape(1, 4, 1, 1) * np.ones((4, 1, 4, 1))
    lon = np.linspace(-100, -90, 4).reshape(1, 1, 4, 1) * np.ones((4, 4, 1, 1))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)

    for layer in layers:
        x_in = x
        x = layer(
            x.astype(np.float32),
            y.astype(np.float32),
            exo_data=exo_data,
        )
        assert x.shape == x_in.shape
        with pytest.raises(tf.errors.InvalidArgumentError):
            tf.assert_equal(x_in, x)
    assert not any(np.isnan(x.numpy().flatten()))


def test_transformer_layer_3d():
    """Test the transformer layer with 3D data (5D tensor input)"""
    hidden_layers = [
        {
            'class': 'Sup3rTransformerLayer',
            'features': ['a', 'b', 'c'],
            'exo_features': ['d', 'e', 'f'],
            'embed_dim': 8,
            'key_dim': 8,
        }
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 1

    x = np.random.normal(0, 1, size=(1, 10, 10, 6, 3)).astype(np.float32)
    y = np.random.uniform(0, 1, size=(1, 10, 10, 6, 1)).astype(np.float32)
    mask = np.random.choice([False, True], (1, 10, 10, 6), p=[0.1, 0.9])
    y[mask] = np.nan

    lat = np.linspace(30, 40, 10).reshape(1, 10, 1, 1, 1) * np.ones((
        1,
        1,
        10,
        6,
        1,
    ))
    lon = np.linspace(-100, -90, 10).reshape(1, 1, 10, 1, 1) * np.ones((
        1,
        10,
        1,
        6,
        1,
    ))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)

    for layer in layers:
        x_in = x
        x = layer(x, y, exo_data=exo_data)
        assert x.shape == x_in.shape
        with pytest.raises(tf.errors.InvalidArgumentError):
            tf.assert_equal(x_in, x)
    assert not any(np.isnan(x.numpy().flatten()))


def test_pos_encoding_patch_size_gt1_2d():
    """Test 2D positional encoding with patch_size > 1.

    Verify:
    1. Encoding output has the correct token shape
    2. Different spatial positions get distinct encodings
    """
    embed_dim = 8
    patch_size = 2
    n_rows, n_cols = 4, 4
    x = tf.zeros((1, n_rows, n_cols, 1), dtype=tf.float32)

    pos_enc = PositionEncoder(patch_size=patch_size, embed_dim=embed_dim)
    pos_enc.build(x.shape)

    n_tokens = n_rows * n_cols

    lat = np.linspace(30, 40, n_rows).reshape(1, n_rows, 1, 1) * np.ones((
        1,
        1,
        n_cols,
        1,
    ))
    lon = np.linspace(-100, -90, n_cols).reshape(1, 1, n_cols, 1) * np.ones((
        1,
        n_rows,
        1,
        1,
    ))
    lat = tf.constant(lat, dtype=tf.float32)
    lon = tf.constant(lon, dtype=tf.float32)

    enc = pos_enc(x, lat=lat, lon=lon)
    assert enc.shape == (1, n_tokens, embed_dim)

    # Different rows must produce different encodings
    enc_spatial = tf.reshape(enc, (1, n_rows, n_cols, embed_dim))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            enc_spatial[0, 0, 0].numpy(),
            enc_spatial[0, 1, 0].numpy(),
        )


def test_pos_encoding_patch_size_gt1_3d():
    """Test 3D positional encoding with patch_size > 1.

    Same logic as the 2D case but for a 5D tensor.
    """
    embed_dim = 8
    patch_size = 2
    n_rows, n_cols, n_times = 4, 4, 4
    x = tf.zeros((1, n_rows, n_cols, n_times, 1), dtype=tf.float32)

    pos_enc = PositionEncoder(patch_size=patch_size, embed_dim=embed_dim)
    pos_enc.build(x.shape)

    n_tokens = n_rows * n_cols * n_times

    lat = np.linspace(30, 40, n_rows).reshape(1, n_rows, 1, 1, 1)
    lat *= np.ones((1, 1, n_cols, n_times, 1))
    lon = np.linspace(-100, -90, n_cols).reshape(1, 1, n_cols, 1, 1)
    lon *= np.ones((1, n_rows, 1, n_times, 1))
    lat = tf.constant(lat, dtype=tf.float32)
    lon = tf.constant(lon, dtype=tf.float32)

    enc = pos_enc(x, lat=lat, lon=lon)
    assert enc.shape == (1, n_tokens, embed_dim)

    # Different spatial positions must produce different encodings
    enc_spatial = tf.reshape(enc, (1, n_rows, n_cols, n_times, embed_dim))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            enc_spatial[0, 0, 0, 0].numpy(),
            enc_spatial[0, 1, 0, 0].numpy(),
        )


def test_tokenize_encode_lat_lon_encoding_values():
    """Test PositionEncoder lat/lon encoding matches freq encoding."""
    embed_dim = 8
    x = tf.zeros((1, 2, 2, 1), dtype=tf.float32)
    lat = tf.constant([[[[0.0], [1.0]], [[2.0], [3.0]]]], dtype=tf.float32)
    lon = tf.constant([[[[10.0], [10.0]], [[20.0], [20.0]]]], dtype=tf.float32)

    pos_enc = PositionEncoder(patch_size=1, embed_dim=embed_dim)
    pos_enc.build(x.shape)

    min_period = pos_enc.min_period_spatial
    max_period = pos_enc.max_period_spatial

    enc = pos_enc.encode_lat_lon(
        x, lat, lon, min_period=min_period, max_period=max_period
    )
    expected = tf.concat(
        [
            PositionEncoder._freq_encode(
                lat,
                d=embed_dim // 2,
                min_period=min_period,
                max_period=max_period,
            ),
            PositionEncoder._freq_encode(
                lon,
                d=embed_dim // 2,
                min_period=min_period,
                max_period=max_period,
            ),
        ],
        axis=-1,
    )
    expected = tf.reshape(expected, (1, -1, embed_dim))
    np.testing.assert_allclose(enc.numpy(), expected.numpy(), atol=1e-6)

    x_enc = pos_enc(x, lat=lat, lon=lon)
    np.testing.assert_allclose(x_enc.numpy(), enc.numpy(), atol=1e-6)


def test_tokenize_encode_call_adds_time_encoding(monkeypatch):
    """Test 5D call() adds time encoding when lat/lon/time are provided."""
    embed_dim = 8
    x = tf.zeros((1, 2, 2, 2, 1), dtype=tf.float32)
    lat = tf.ones_like(x)
    lon = 2.0 * tf.ones_like(x)
    time = 3.0 * tf.ones_like(x)

    pos_enc = PositionEncoder(patch_size=1, embed_dim=embed_dim)
    pos_enc.build(x.shape)

    n_tokens = int(np.prod(x.shape[1:-1]))
    lat_lon_out = tf.ones((1, n_tokens, embed_dim), dtype=tf.float32)
    time_out = 2.0 * tf.ones((1, n_tokens, embed_dim), dtype=tf.float32)

    calls = {'lat_lon': 0, 'time': 0}

    def _fake_lat_lon(*args, **kwargs):  # noqa: ARG001
        calls['lat_lon'] += 1
        return lat_lon_out

    def _fake_time(*args, **kwargs):  # noqa: ARG001
        calls['time'] += 1
        return time_out

    monkeypatch.setattr(pos_enc, 'encode_lat_lon', _fake_lat_lon)
    monkeypatch.setattr(pos_enc, 'encode_time', _fake_time)

    x_enc = pos_enc(x, lat=lat, lon=lon, time=time)
    expected = lat_lon_out + time_out

    np.testing.assert_allclose(x_enc.numpy(), expected.numpy(), atol=1e-6)
    assert calls['lat_lon'] == 1
    assert calls['time'] == 1


def test_tokenize_encode_time_encoding_values():
    """Test PositionEncoder time encoding against known unix timestamps."""
    embed_dim = 8
    x = tf.zeros((1, 1, 1, 2, 1), dtype=tf.float32)
    time = tf.constant(
        [[[[[1704067200.0], [1704153600.0]]]]],
        dtype=tf.float32,
    )

    pos_enc = PositionEncoder(patch_size=1, embed_dim=embed_dim)
    pos_enc.build(x.shape)

    doy = tf.constant([[[[[0.0], [1.0]]]]], dtype=tf.float32)
    soy = tf.constant([[[[[0.0], [86400.0]]]]], dtype=tf.float32)
    expected = tf.concat(
        [
            PositionEncoder._freq_encode(
                doy,
                min_period=pos_enc.min_period_temporal / 86400,
                max_period=pos_enc.max_period_temporal / 86400,
                d=embed_dim // 2,
            ),
            PositionEncoder._freq_encode(
                soy,
                min_period=pos_enc.min_period_temporal,
                max_period=pos_enc.max_period_temporal,
                d=embed_dim // 2,
            ),
        ],
        axis=-1,
    )
    expected = tf.reshape(expected, (1, -1, embed_dim))

    enc = pos_enc.encode_time(
        x,
        time,
        pos_enc.min_period_temporal,
        pos_enc.max_period_temporal,
    )
    np.testing.assert_allclose(enc.numpy(), expected.numpy(), atol=1e-6)


def test_transformer_patch_size_gt1_shapes():
    """Test Tokenizer and PositionEncoder with patch_size > 1, including
    non-divisible spatial dimensions (5x7). Inputs are padded via
    PatchLayer.pad() before tokenize/encode.
    """
    patch_size = 2
    embed_dim = 8

    for h, w in [(6, 8), (5, 7)]:
        x = tf.random.normal((1, h, w, 3), dtype=tf.float32)
        y = tf.random.normal((1, h, w, 1), dtype=tf.float32)

        x_padded = PatchLayer.pad(x, patch_size=patch_size)
        y_padded = PatchLayer.pad(y, patch_size=patch_size)

        # Padded dims must be divisible by patch_size.
        for i in range(1, len(x_padded.shape) - 1):
            assert x_padded.shape[i] % patch_size == 0
            assert y_padded.shape[i] % patch_size == 0

        ph = int(np.ceil(h / patch_size)) * patch_size
        pw = int(np.ceil(w / patch_size)) * patch_size
        n_tokens = (ph // patch_size) * (pw // patch_size)

        tokenizer_q = Embedder(patch_size=patch_size, embed_dim=embed_dim)
        tokenizer_v = Embedder(patch_size=patch_size, embed_dim=embed_dim)
        pos_enc = PositionEncoder(patch_size=patch_size, embed_dim=embed_dim)

        lat_pad = tf.constant(
            np.linspace(30, 40, ph).reshape(1, ph, 1, 1)
            * np.ones((1, 1, pw, 1)),
            dtype=tf.float32,
        )
        lon_pad = tf.constant(
            np.linspace(-100, -90, pw).reshape(1, 1, pw, 1)
            * np.ones((1, ph, 1, 1)),
            dtype=tf.float32,
        )

        q = tokenizer_q(x_padded)
        v = tokenizer_v(y_padded)
        q_enc = pos_enc(x_padded, lat=lat_pad, lon=lon_pad)
        v_enc = pos_enc(y_padded, lat=lat_pad, lon=lon_pad)

        n_spatial_tokens = ph * pw
        assert q.shape == (1, n_tokens, embed_dim)
        assert v.shape == (1, n_tokens, embed_dim)
        assert q_enc.shape == (1, n_spatial_tokens, embed_dim)
        assert v_enc.shape == (1, n_spatial_tokens, embed_dim)

        # Full Sup3rTransformerLayer forward pass must return original shape
        lat = tf.constant(
            np.linspace(30, 40, h).reshape(1, h, 1, 1) * np.ones((1, 1, w, 1)),
            dtype=tf.float32,
        )
        lon = tf.constant(
            np.linspace(-100, -90, w).reshape(1, 1, w, 1)
            * np.ones((1, h, 1, 1)),
            dtype=tf.float32,
        )
        exo_data = tf.concat([lat, lon], axis=-1)
        layer = Sup3rTransformerLayer(embed_dim=embed_dim, key_dim=embed_dim)
        out = layer(x, y, exo_data=exo_data)
        assert out.shape == x.shape


def test_transformer_exo_data_time_forwarding(monkeypatch):
    """Test Sup3rTransformerLayer forwards lat/lon/time from exo_data."""
    layer = Sup3rTransformerLayer()

    x = tf.random.normal((1, 3, 4, 2), dtype=tf.float32)
    y = tf.random.normal((1, 3, 4, 1), dtype=tf.float32)

    calls = []

    def _fake_call(x, hi_res_feature, idx, lat=None, lon=None, time=None):
        del hi_res_feature
        calls.append((lat is not None, lon is not None, time is not None))
        return x[idx]

    monkeypatch.setattr(layer, '_call', _fake_call)

    exo_data = tf.random.normal((1, 3, 4, 3), dtype=tf.float32)
    out = layer(x, y, exo_data=exo_data)
    np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-6)
    assert calls == [(True, True, True)]

    calls.clear()
    exo_data = tf.random.normal((1, 3, 4, 2), dtype=tf.float32)
    out = layer(x, y, exo_data=exo_data)
    np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-6)
    assert calls == [(True, True, False)]


def test_transformer_alibi_bfloat16_nan_branch_consistency():
    """Sup3rTransformerLayerAlibi should keep bfloat16 dtype across both
    the all-NaN fast path and the attention path.
    """
    layer = Sup3rTransformerLayerAlibi(
        embed_dim=8,
        key_dim=8,
        num_heads=2,
        features=['obs'],
        dtype='bfloat16',
    )

    x = tf.cast(tf.random.normal((2, 4, 4, 8)), tf.bfloat16)
    y = tf.cast(tf.random.normal((2, 4, 4, 1)), tf.bfloat16)
    y = tf.concat(
        [
            tf.fill((1, 4, 4, 1), tf.cast(np.nan, tf.bfloat16)),
            y[1:2],
        ],
        axis=0,
    )

    lat = tf.cast(
        np.linspace(30, 40, 4).reshape(1, 4, 1, 1) * np.ones((2, 1, 4, 1)),
        tf.bfloat16,
    )
    lon = tf.cast(
        np.linspace(-110, -100, 4).reshape(1, 1, 4, 1)
        * np.ones((2, 4, 1, 1)),
        tf.bfloat16,
    )
    exo_data = tf.concat([lat, lon], axis=-1)

    out = layer(x, y, exo_data=exo_data)

    assert out.shape == x.shape
    assert out.dtype == tf.bfloat16
    tf.debugging.assert_equal(out[0], x[0])
    tf.debugging.assert_all_finite(tf.cast(out[1], tf.float32), 'bad output')


def test_transformer_alibi_bias_stays_finite_with_zero_alpha():
    """ALiBi locality bias should stay finite even if alpha is driven to 0."""
    layer = Sup3rTransformerLayerAlibi(
        embed_dim=8,
        key_dim=8,
        num_heads=2,
        features=['obs'],
        dtype='bfloat16',
    )

    x = tf.cast(tf.random.normal((1, 4, 4, 8)), tf.bfloat16)
    y = tf.cast(tf.random.normal((1, 4, 4, 1)), tf.bfloat16)
    lat = tf.cast(
        np.linspace(30, 40, 4).reshape(1, 4, 1, 1),
        tf.bfloat16,
    )
    lon = tf.cast(
        np.linspace(-110, -100, 4).reshape(1, 1, 4, 1),
        tf.bfloat16,
    )
    lat = tf.repeat(lat, repeats=4, axis=2)
    lon = tf.repeat(lon, repeats=4, axis=1)

    layer.build((None, 4, 4, 8))
    layer.alpha.assign(0.0)

    bias = layer.get_locality_bias(x, y, lat=lat, lon=lon)

    assert bias.dtype == tf.bfloat16
    tf.debugging.assert_all_finite(tf.cast(bias, tf.float32), 'bad bias')


def test_functional_layer():
    """Test the generic functional layer"""

    layer = FunctionalLayer('maximum', 1)
    x = np.random.normal(0.5, 3, size=(1, 4, 4, 6, 3))
    assert layer(x).numpy().min() == 1.0

    # make sure layer works with input of arbitrary shape
    x = np.random.normal(0.5, 3, size=(2, 8, 8, 4, 1))
    assert layer(x).numpy().min() == 1.0

    layer = FunctionalLayer('multiply', 1.5)
    x = np.random.normal(0.5, 3, size=(1, 4, 4, 6, 3))
    assert np.allclose(layer(x).numpy(), x * 1.5)

    with pytest.raises(AssertionError) as excinfo:
        FunctionalLayer('bad_arg', 0)
    assert 'must be one of' in str(excinfo.value)


def test_gaussian_pooling():
    """Test the gaussian average pooling layer"""

    kernels = []
    for stdev in [1, 2]:
        layer = GaussianAveragePooling2D(pool_size=5, strides=1, sigma=stdev)
        _ = layer(np.ones((24, 100, 100, 35)))
        kernel = layer.make_kernel().numpy()
        kernels.append(kernel)

        assert np.allclose(kernel[:, :, 0, 0].sum(), 1, rtol=1e-4)
        assert kernel[2, 2, 0, 0] == kernel.max()
        assert kernel[0, 0, 0, 0] == kernel.min()
        assert kernel[-1, -1, 0, 0] == kernel.min()

    assert kernels[1].max() < kernels[0].max()
    assert kernels[1].min() > kernels[0].min()

    layers = [
        {'class': 'GaussianAveragePooling2D', 'pool_size': 12, 'strides': 1}
    ]
    model1 = TfModel.build(
        ['a', 'b', 'c'],
        ['d'],
        hidden_layers=layers,
        input_layer=False,
        output_layer=False,
        normalize=False,
    )
    x_in = np.random.uniform(0, 1, (1, 12, 12, 3))
    out1 = model1.predict(x_in)
    kernel1 = model1.layers[0].make_kernel()[:, :, 0, 0].numpy()

    for idf in range(out1.shape[-1]):
        test = (x_in[0, :, :, idf] * kernel1).sum()
        assert np.allclose(test, out1[..., idf])

    assert out1.shape[1] == out1.shape[2] == 1
    assert out1[0, 0, 0, 0] != out1[0, 0, 0, 1] != out1[0, 0, 0, 2]

    with TemporaryDirectory() as td:
        model_path = os.path.join(td, 'test_model')
        model1.save_model(model_path)
        model2 = TfModel.load(model_path)

        kernel2 = model2.layers[0].make_kernel()[:, :, 0, 0].numpy()
        out2 = model2.predict(x_in)
        assert np.allclose(kernel1, kernel2)
        assert np.allclose(out1, out2)

        layer = model2.layers[0]
        x_in = np.random.uniform(0, 1, (10, 24, 24, 3))
        _ = model2.predict(x_in)


def test_gaussian_pooling_train():
    """Test the trainable sigma functionality of the gaussian average pool"""
    pool_size = 5
    xtrain = np.random.uniform(0, 1, (10, pool_size, pool_size, 1))
    ytrain = np.random.uniform(0, 1, (10, 1, 1, 1))
    hidden_layers = [
        {
            'class': 'GaussianAveragePooling2D',
            'pool_size': pool_size,
            'trainable': False,
            'strides': 1,
            'padding': 'valid',
            'sigma': 2,
        }
    ]

    model = TfModel.build(
        ['x'],
        ['y'],
        hidden_layers=hidden_layers,
        input_layer=False,
        output_layer=False,
        learning_rate=1e-3,
        normalize=(True, True),
    )
    model.layers[0].build(xtrain.shape)
    assert len(model.layers[0].trainable_weights) == 0

    hidden_layers[0]['trainable'] = True
    model = TfModel.build(
        ['x'],
        ['y'],
        hidden_layers=hidden_layers,
        input_layer=False,
        output_layer=False,
        learning_rate=1e-3,
        normalize=(True, True),
    )
    model.layers[0].build(xtrain.shape)
    assert len(model.layers[0].trainable_weights) == 1

    layer = model.layers[0]
    sigma1 = float(layer.sigma)
    kernel1 = layer.make_kernel().numpy().copy()
    model.train_model(xtrain, ytrain, epochs=10)
    sigma2 = float(layer.sigma)
    kernel2 = layer.make_kernel().numpy().copy()

    assert not np.allclose(sigma1, sigma2)
    assert not np.allclose(kernel1, kernel2)

    with TemporaryDirectory() as td:
        model_path = os.path.join(td, 'test_model')
        model.save_model(model_path)
        model2 = TfModel.load(model_path)

    assert np.allclose(model.predict(xtrain), model2.predict(xtrain))


def test_siglin():
    """Test the sigmoid linear layer"""
    n_points = 1000
    mid = n_points // 2
    sl = SigLin()
    x = np.linspace(-10, 10, n_points + 1)
    y = sl(x).numpy()
    assert x.shape == y.shape
    assert (y > 0).all()
    assert np.allclose(y[mid:], x[mid:] + 0.5)


def test_logtransform():
    """Test the log transform layer"""
    n_points = 1000
    lt = LogTransform(adder=0)
    x = np.linspace(0, 10, n_points + 1)
    y = lt(x).numpy()
    assert x.shape == y.shape
    assert y[0] == -np.inf

    lt = LogTransform(adder=1)
    ilt = LogTransform(adder=1, inverse=True)
    x = np.random.uniform(0.01, 10, (n_points + 1, 2))
    y = lt(x).numpy()
    xinv = ilt(y).numpy()
    assert not np.isnan(y).any()
    assert np.allclose(y, np.log(x + 1))
    assert np.allclose(x, xinv)

    lt = LogTransform(adder=1, idf=1)
    ilt = LogTransform(adder=1, inverse=True, idf=1)
    x = np.random.uniform(0.01, 10, (n_points + 1, 2))
    y = lt(x).numpy()
    xinv = ilt(y).numpy()
    assert np.allclose(x[:, 0], y[:, 0])
    assert not np.allclose(x[:, 1], y[:, 1])
    assert not np.isnan(y).any()
    assert np.allclose(y[:, 1], np.log(x[:, 1] + 1))
    assert np.allclose(x, xinv)


def test_unit_conversion():
    """Test the custom unit conversion layer"""
    x = np.random.uniform(0, 1, (1, 10, 10, 4))  # 4 features

    layer = UnitConversion(adder=0, scalar=1)
    y = layer(x).numpy()
    assert np.allclose(x, y)

    layer = UnitConversion(adder=1, scalar=1)
    y = layer(x).numpy()
    assert (y >= 1).all() and (y <= 2).all()

    layer = UnitConversion(adder=1, scalar=100)
    y = layer(x).numpy()
    assert (y >= 1).all() and (y > 90).any() and (y <= 101).all()

    layer = UnitConversion(adder=0, scalar=[100, 1, 1, 1])
    y = layer(x).numpy()
    assert (y[..., 0] > 90).any() and (y[..., 0] <= 100).all()
    assert (y[..., 1:] >= 0).all() and (y[..., 1:] <= 1).all()

    with pytest.raises(AssertionError):
        # bad number of scalar values
        layer = UnitConversion(adder=0, scalar=[100, 1, 1])
        y = layer(x)


def test_masked_squeeze_excite():
    """Make sure ``MaskedSqueezeAndExcite`` layer works properly"""
    x = np.random.normal(0, 1, size=(1, 10, 10, 6, 3))
    y = np.random.uniform(0, 1, size=(1, 10, 10, 6, 1))
    mask = np.random.choice([False, True], (1, 10, 10), p=[0.1, 0.9])
    y[mask] = np.nan

    layer = MaskedSqueezeAndExcitation()
    out = layer(x, y).numpy()

    assert not tf.reduce_any(tf.math.is_nan(out))


def test_sup3r_obs_model():
    """Make sure ``Sup3rObsModel`` layer works properly"""
    x = np.random.normal(0, 1, size=(1, 10, 10, 6, 3))
    y = np.random.uniform(0, 1, size=(1, 10, 10, 6, 1))
    mask = np.random.choice([False, True], (1, 10, 10), p=[0.1, 0.9])
    y[mask] = np.nan

    layer = Sup3rObsModel(
        fill_method='idw',
        features=['a'],
        hidden_layers=[],
        include_mask=True,
    )
    out = layer(x.astype(np.float32), y.astype(np.float32)).numpy()

    assert tf.reduce_any(tf.math.is_nan(y))
    assert not tf.reduce_any(tf.math.is_nan(out))


def test_sup3r_obs_model_2d():
    """Make sure ``Sup3rObsModel`` layer works properly"""
    x = np.random.normal(0, 1, size=(1, 10, 10, 3))
    y = np.random.uniform(0, 1, size=(1, 10, 10, 1))
    mask = np.random.choice([False, True], (1, 10, 10), p=[0.1, 0.9])
    y[mask] = np.nan

    layer = Sup3rObsModel(
        fill_method='idw',
        features=['a'],
        hidden_layers=[],
        include_mask=True,
    )
    out = layer(x.astype(np.float32), y.astype(np.float32)).numpy()

    assert tf.reduce_any(tf.math.is_nan(y))
    assert not tf.reduce_any(tf.math.is_nan(out))


def test_concat_obs_layer():
    """Make sure ``Sup3rConcatObs`` layer works properly"""
    x = np.random.normal(0, 1, size=(1, 10, 10, 6, 3))
    y = np.random.uniform(0, 1, size=(1, 10, 10, 6, 1))
    mask = np.random.choice([False, True], (1, 10, 10), p=[0.1, 0.9])
    y[mask] = np.nan

    layer = Sup3rConcatObs()
    out = layer(x, y).numpy()

    assert tf.reduce_any(tf.math.is_nan(y))
    assert np.allclose(out[..., -1][~mask], y[..., 0][~mask])
    assert np.allclose(out[..., -1][mask], x[..., 0][mask])
    assert x.shape[:-1] == out.shape[:-1]
    assert not tf.reduce_any(tf.math.is_nan(out))


def test_recursive_hidden_layers_init():
    """Make sure initializing a layer with a hidden_layer argument works
    properly. Include test of IDW inpterpolation."""

    config = [
        {
            'class': 'Sup3rObsModel',
            'name': 'test',
            'fill_method': 'idw',
            'hidden_layers': [
                {
                    'class': 'Conv3D',
                    'padding': 'same',
                    'filters': 8,
                    'kernel_size': 3,
                }
            ],
        }
    ]
    layer = HiddenLayers(config)._layers[0]

    x = np.random.normal(0, 1, size=(1, 10, 10, 6, 3))
    y = np.random.uniform(0, 1, size=(1, 10, 10, 6, 1))
    mask = np.random.choice([False, True], (1, 10, 10), p=[0.1, 0.9])
    y[mask] = np.nan

    out = layer(x, y).numpy()

    assert tf.reduce_any(tf.math.is_nan(y))
    assert not tf.reduce_any(tf.math.is_nan(out))
