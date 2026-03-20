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
    ExpandDims,
    FlattenAxis,
    FunctionalLayer,
    GaussianAveragePooling2D,
    GaussianNoiseAxis,
    LogTransform,
    MaskedSqueezeAndExcitation,
    SigLin,
    SkipConnection,
    SpatioTemporalExpansion,
    Sup3rConcatObs,
    Sup3rCrossAttention,
    Sup3rObsModel,
    TileLayer,
    TokenizeEncodeBase,
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


@pytest.mark.parametrize('patch_size', (1, 2, 3))
def test_cross_attn_2d(patch_size):
    """Test the cross attention layer with 2D data (4D tensor input)"""
    hidden_layers = [
        {
            'class': 'Sup3rCrossAttention',
            'query_patch_size': patch_size,
            'value_patch_size': 1,
        }
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 1

    x = np.random.normal(0, 1, size=(1, 4, 4, 3))
    y = np.random.uniform(0, 1, size=(1, 4, 4, 1))
    mask = np.random.choice([False, True], (1, 4, 4), p=[0.1, 0.9])
    y[mask] = np.nan

    for layer in layers:
        x_in = x
        x = layer(x.astype(np.float32), y.astype(np.float32))
        assert x.shape == x_in.shape
        with pytest.raises(tf.errors.InvalidArgumentError):
            tf.assert_equal(x_in, x)
    assert not any(np.isnan(x.numpy().flatten()))


@pytest.mark.parametrize('patch_size', (1, 2, 3))
def test_cross_attn_3d(patch_size):
    """Test the cross attention layer with 3D data (5D tensor input)"""
    hidden_layers = [
        {
            'class': 'Sup3rCrossAttention',
            'query_patch_size': patch_size,
            'value_patch_size': 1,
        }
    ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 1

    x = np.random.normal(0, 1, size=(1, 10, 10, 6, 3)).astype(np.float32)
    y = np.random.uniform(0, 1, size=(1, 10, 10, 6, 1)).astype(np.float32)
    mask = np.random.choice([False, True], (1, 10, 10, 6), p=[0.1, 0.9])
    y[mask] = np.nan

    for layer in layers:
        x_in = x
        x = layer(x, y)
        assert x.shape == x_in.shape
        with pytest.raises(tf.errors.InvalidArgumentError):
            tf.assert_equal(x_in, x)
    assert not any(np.isnan(x.numpy().flatten()))


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
