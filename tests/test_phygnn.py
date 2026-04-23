"""
Tests for basic phygnn functionality and execution.
"""

# pylint: disable=W0613
import os
import tempfile
import types

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Conv1D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
)
from tensorflow.keras.optimizers import Adam

from phygnn import PhysicsGuidedNeuralNetwork
from phygnn.layers.custom_layers import SkipConnection

N = 100
A = np.linspace(-1, 1, N)
B = np.linspace(-1, 1, N)
A, B = np.meshgrid(A, B)
A = np.expand_dims(A.flatten(), axis=1)
B = np.expand_dims(B.flatten(), axis=1)

Y = np.sqrt(A**2 + B**2)
X = np.hstack((A, B))
P = X.copy()
Y_NOISE = Y * (1 + (np.random.random(Y.shape) - 0.5) * 0.5) + 0.1


HIDDEN_LAYERS = [
    {'units': 64, 'activation': 'relu', 'name': 'relu1'},
    {'units': 64, 'activation': 'relu', 'name': 'relu2'},
]


def p_fun_pythag(model, y_true, y_predicted, p):  # noqa : ARG001
    """Example function for loss calculation using physical relationships.

    Parameters
    ----------
    model : PhysicsGuidedNeuralNetwork
        Instance of the phygnn model at the current point in training.
    y_true : np.ndarray
        Known y values that were given to the PhyGNN fit method.
    y_predicted : tf.Tensor
        Predicted y values in a 2D tensor based on x values in this batch.
    p : np.ndarray
        Supplemental physical feature data that can be used to calculate a
        y_physical value to compare against y_predicted. The rows in this
        array have been carried through the batching process alongside y_true
        and the features used to create y_predicted and so can be used 1-to-1
        with the rows in y_predicted and y_true.

    Returns
    -------
    p_loss : tf.Tensor
        A 0D tensor physical loss value.
    """

    p = tf.convert_to_tensor(p, dtype=tf.float32)
    y_physical = tf.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    y_physical = tf.expand_dims(y_physical, 1)

    p_loss = tf.math.reduce_mean(tf.math.abs(y_predicted - y_physical))

    return p_loss


def p_fun_bad(model, y_true, y_predicted, p):  # noqa : ARG001
    """Example of a poorly formulated p_fun() that use numpy operations."""

    y_physical = p[:, 0] ** 2 + p[:, 1] ** 2
    p_loss = np.mean(np.abs(y_predicted.numpy() - y_physical))
    p_loss = tf.convert_to_tensor(p_loss, dtype=tf.float32)

    return p_loss


def test_nn():
    """Test the basic NN operation of the PGNN without weighting pfun."""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=1,
        feature_names=['a', 'b'],
        output_names=['c'],
    )
    model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)

    test_mae = np.mean(np.abs(model.predict(X) - Y))

    assert len(model.layers) == 6
    assert len(model.weights) == 6
    assert len(model.history) == 20
    assert model.history.validation_loss.values[-1] < 0.15
    assert test_mae < 0.15


def test_phygnn():
    """Test the operation of the PGNN with weighting pfun."""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
    )
    model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)

    test_mae = np.mean(np.abs(model.predict(X) - Y))

    assert len(model.layers) == 6
    assert len(model.weights) == 6
    assert len(model.history) == 20
    assert isinstance(model.layers[0], InputLayer)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], Activation)
    assert isinstance(model.layers[3], Dense)
    assert isinstance(model.layers[4], Activation)
    assert isinstance(model.layers[5], Dense)
    assert model.history.validation_loss.values[-1] < 0.05
    assert test_mae < 0.05


def test_df_input():
    """Test the operation of the PGNN with labeled input dataframes."""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
    )
    x_df = pd.DataFrame(X, columns=('a', 'b'))
    y_df = pd.DataFrame(Y_NOISE, columns=('c',))
    p_df = pd.DataFrame(P, columns=('a', 'b'))
    model.fit(x_df, y_df, p_df, n_batch=1, n_epoch=2)

    assert model.feature_names == ['a', 'b']
    assert model.output_names == ['c']

    x_df_bad = pd.DataFrame(X, columns=('x1', 'x2'))
    y_df_bad = pd.DataFrame(Y_NOISE, columns=('y',))

    try:
        model.fit(x_df_bad, y_df_bad, p_df, n_batch=1, n_epoch=2)
    except AssertionError as e:
        assert "Cannot work with input x columns: ['x1', 'x2']" in str(e)

    try:
        model.fit(x_df, y_df_bad, p_df, n_batch=1, n_epoch=2)
    except AssertionError as e:
        assert "Cannot work with input y columns: ['y']" in str(e)


def test_kernel_regularization():
    """Test the kernel regularization of phygnn."""
    base = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=1,
    )

    model_l1 = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=1,
        kernel_reg_rate=0.01,
        kernel_reg_power=1,
    )

    model_l2 = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=1,
        kernel_reg_rate=0.01,
        kernel_reg_power=2,
    )

    base.seed(0)
    base.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)
    model_l1.seed(0)
    model_l1.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)
    model_l2.seed(0)
    model_l2.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)

    assert base.kernel_reg_term > model_l1.kernel_reg_term
    assert model_l1.kernel_reg_term > model_l2.kernel_reg_term


def test_bias_regularization():
    """Test the bias regularization of phygnn."""
    base = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=1,
    )

    model_l1 = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=1,
        bias_reg_rate=0.01,
        bias_reg_power=1,
    )

    model_l2 = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=1,
        bias_reg_rate=0.01,
        bias_reg_power=2,
    )

    base.seed(0)
    base.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)
    model_l1.seed(0)
    model_l1.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)
    model_l2.seed(0)
    model_l2.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)

    assert base.bias_reg_term > model_l1.bias_reg_term
    assert model_l1.bias_reg_term > model_l2.bias_reg_term
    assert np.abs(base.bias_reg_term - 5) < 5
    assert np.abs(model_l1.bias_reg_term - 4) < 5
    assert np.abs(model_l2.bias_reg_term - 1) < 5


def test_calc_loss_uses_float32_accumulation_for_bfloat16():
    """Scalar phygnn losses should accumulate in float32."""
    model = PhysicsGuidedNeuralNetwork.__new__(PhysicsGuidedNeuralNetwork)
    model._loss_weights = (1.0, 1.0)
    model._metric_fun = (
        lambda y_true, y_predicted: tf.reduce_mean(
            tf.cast(tf.abs(y_true - y_predicted), tf.bfloat16)
        )
    )
    model._p_fun = (
        lambda _model, _y_true, y_predicted, _p, **_kwargs: tf.reduce_mean(
            tf.cast(y_predicted + 2.0, tf.bfloat16)
        )
    )
    model.kernel_reg_rate = 0.0
    model.bias_reg_rate = 0.0

    y_true = tf.ones((2, 1), dtype=tf.bfloat16)
    y_predicted = tf.zeros((2, 1), dtype=tf.bfloat16)

    loss, nn_loss, p_loss = model.calc_loss(y_true, y_predicted, p=None)

    assert loss.dtype == tf.float32
    assert nn_loss.dtype == tf.float32
    assert p_loss.dtype == tf.float32
    assert loss.numpy() == pytest.approx(3.0)


def test_save_load():
    """Test the save/load operations of PGNN"""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
        feature_names=['a', 'b'],
        output_names=['c'],
    )

    model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)
    y_pred = model.predict(X)

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'tempfile.pkl')
        model.save(fpath)
        loaded = PhysicsGuidedNeuralNetwork.load(fpath)

    assert len(model.layers) == len(loaded.layers)
    for layer0, layer1 in zip(model.layers, loaded.layers):
        for i, weights0 in enumerate(layer0.weights):
            assert weights0.shape == layer1.weights[i].shape
            assert np.allclose(weights0, layer1.weights[i])

    y_pred_loaded = loaded.predict(X)
    assert np.allclose(y_pred, y_pred_loaded)
    assert loaded.feature_names == ['a', 'b']
    assert loaded.output_names == ['c']
    assert isinstance(model._optimizer, Adam)
    assert isinstance(loaded._optimizer, Adam)
    assert model._optimizer.get_config() == loaded._optimizer.get_config()


def test_compute_dtype_save_load():
    """Test compute_dtype persistence across phygnn save/load."""
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
        compute_dtype='float16',
    )

    model.predict(X[:4])

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'tempfile.pkl')
        model.save(fpath)
        loaded = PhysicsGuidedNeuralNetwork.load(fpath)

    dense_layers = [
        layer for layer in loaded.layers if isinstance(layer, Dense)
    ]
    assert loaded.compute_dtype == 'float16'
    assert dense_layers
    assert dense_layers[0].compute_dtype == 'float16'


def test_save_load_skip_connections():
    """Test save/load preserves repeated shared skip-connection execution."""
    hidden_layers = [
        {'units': 64, 'activation': 'relu', 'name': 'relu1'},
        {'class': 'SkipConnection', 'name': 'skip_a'},
        {'units': 64, 'activation': 'relu', 'name': 'relu2'},
        {'class': 'SkipConnection', 'name': 'skip_a'},
        {'units': 64, 'activation': 'relu', 'name': 'relu3'},
    ]

    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=hidden_layers,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
        feature_names=['a', 'b'],
        output_names=['c'],
    )

    model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=10)
    y_pred = model.predict(X)

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'tempfile.pkl')
        model.save(fpath)
        loaded = PhysicsGuidedNeuralNetwork.load(fpath)

    y_pred_loaded = loaded.predict(X)
    skip_layers = [
        layer for layer in loaded.layers if isinstance(layer, SkipConnection)
    ]

    assert len(model.layers) == len(loaded.layers)
    assert len(skip_layers) == 2
    assert id(skip_layers[0]) == id(skip_layers[1])
    assert np.allclose(y_pred, y_pred_loaded)


def test_dummy_p_fun():
    """Test the phygnn model with dummy pfun that is just MAE"""
    PhysicsGuidedNeuralNetwork.seed(0)
    model_0 = PhysicsGuidedNeuralNetwork(
        p_fun=None,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        metric='mae',
        n_features=2,
        n_labels=1,
        learning_rate=5e-4,
    )
    model_0.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20, shuffle=False)
    pred_0 = model_0.predict(X, to_numpy=True)

    PhysicsGuidedNeuralNetwork.seed(0)
    model_1 = PhysicsGuidedNeuralNetwork(
        p_fun=None,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(1.0, 0.0),
        metric='mae',
        n_features=2,
        n_labels=1,
        learning_rate=5e-4,
    )
    model_1.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20, shuffle=False)
    pred_1 = model_1.predict(X, to_numpy=True)

    loss_0 = model_0.history.training_loss.values.astype(float)
    loss_1 = model_1.history.training_loss.values.astype(float)

    assert np.allclose(pred_0, pred_1)
    assert np.allclose(loss_0, loss_1)


def test_bad_pfun():
    """Test the preflight check with a non-differentiable p_fun"""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_bad,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
    )
    with pytest.raises(RuntimeError) as e:
        model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)

    assert 'not differentiable' in str(e.value)


def test_dropouts():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers_1 = [
        {'units': 64},
        {'activation': 'relu'},
        {'units': 64, 'activation': 'relu', 'name': 'relu2'},
    ]
    hidden_layers_2 = [
        {'units': 64},
        {'activation': 'relu'},
        {'dropout': 0.1},
        {'units': 64, 'activation': 'relu', 'name': 'relu2', 'dropout': 0.1},
    ]
    PhysicsGuidedNeuralNetwork.seed()
    model_1 = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=hidden_layers_1,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
    )
    PhysicsGuidedNeuralNetwork.seed()
    model_2 = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=hidden_layers_2,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
    )

    assert len(model_1.layers) == 6
    assert len(model_2.layers) == 8, 'dropout layers did not get added!'
    assert isinstance(model_2.layers[0], InputLayer)
    assert isinstance(model_2.layers[1], Dense)
    assert isinstance(model_2.layers[2], Activation)
    assert isinstance(model_2.layers[3], Dropout)
    assert isinstance(model_2.layers[4], Dense)
    assert isinstance(model_2.layers[5], Activation)
    assert isinstance(model_2.layers[6], Dropout)

    PhysicsGuidedNeuralNetwork.seed()
    model_1.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)

    PhysicsGuidedNeuralNetwork.seed()
    model_2.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)

    y_pred_1 = model_1.predict(X)
    y_pred_2 = model_2.predict(X)

    # make sure dropouts dont predict the same as non-dropout
    diff = np.abs(y_pred_1 - y_pred_2)
    assert not np.allclose(y_pred_1, y_pred_2)
    assert np.max(diff) > 0.05

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'tempfile.pkl')
        model_2.save(fpath)
        loaded = PhysicsGuidedNeuralNetwork.load(fpath)

    y_pred_loaded = loaded.predict(X)
    assert np.allclose(y_pred_2, y_pred_loaded)
    assert len(model_2.layers) == len(loaded.layers)


def test_batch_norm():
    """Test the addition of BatchNormalization layers"""
    HIDDEN_LAYERS = [
        {'units': 64},
        {'batch_normalization': {'axis': 1}},
        {'activation': 'relu'},
        {
            'units': 64,
            'activation': 'relu',
            'batch_normalization': {'axis': 1},
        },
    ]
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=HIDDEN_LAYERS,
        loss_weights=(0.0, 1.0),
        n_features=2,
        n_labels=1,
    )

    assert len(model.layers) == 8, 'Batch norm layers did not get added!'
    assert isinstance(model.layers[0], InputLayer)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], BatchNormalization)
    assert isinstance(model.layers[3], Activation)
    assert isinstance(model.layers[4], Dense)
    assert isinstance(model.layers[5], BatchNormalization)
    assert isinstance(model.layers[6], Activation)

    model.fit(X, Y_NOISE, P, n_batch=1, n_epoch=10)
    y_pred = model.predict(X)

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'tempfile.pkl')
        model.save(fpath)
        loaded = PhysicsGuidedNeuralNetwork.load(fpath)

    y_pred_loaded = loaded.predict(X)

    assert np.allclose(y_pred, y_pred_loaded)
    assert len(model.layers) == len(loaded.layers)


def test_conv1d():
    """Test a phygnn model with a conv1d layer. The data in this test is
    garbage, just a test on shapes and save/load functionality"""

    input_layer = {
        'class': 'Conv1D',
        'filters': 12,
        'kernel_size': (4,),
        'activation': 'relu',
    }
    hidden_layers = [{'units': 64, 'activation': 'relu'}, {'class': 'Flatten'}]
    output_layer = {'units': 24}
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=hidden_layers,
        input_layer=input_layer,
        output_layer=output_layer,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=24,
    )

    train_x = np.random.uniform(-1, 1, (50, 12, 2))
    train_y = np.random.uniform(-1, 1, (50, 24))

    assert len(model.layers) == 5, 'conv layers did not get added!'
    assert isinstance(model.layers[0], Conv1D)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], Activation)
    assert isinstance(model.layers[3], Flatten)
    assert isinstance(model.layers[4], Dense)

    model.fit(train_x, train_y, train_x, n_batch=1, n_epoch=10)
    y_pred = model.predict(train_x)
    assert y_pred.shape == (50, 24)

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'tempfile.pkl')
        model.save(fpath)
        loaded = PhysicsGuidedNeuralNetwork.load(fpath)

    assert len(model.layers) == len(loaded.layers)
    for layer0, layer1 in zip(model.layers, loaded.layers):
        for i, weights0 in enumerate(layer0.weights):
            assert weights0.shape == layer1.weights[i].shape
            assert np.allclose(weights0, layer1.weights[i])

    y_pred_loaded = loaded.predict(train_x)

    assert np.allclose(y_pred, y_pred_loaded)
    assert len(model.layers) == len(loaded.layers)


def test_conv3d():
    """Test a phygnn model with a conv3d layer. The data in this test is
    garbage, just a test on shapes and save/load functionality"""

    input_layer = {
        'class': 'Conv3D',
        'filters': 2,
        'kernel_size': 3,
        'activation': 'relu',
    }
    hidden_layers = [{'units': 64, 'activation': 'relu'}, {'class': 'Flatten'}]
    output_layer = {'units': 24}
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=hidden_layers,
        input_layer=input_layer,
        output_layer=output_layer,
        loss_weights=(1.0, 0.0),
        n_features=1,
        n_labels=24,
    )

    train_x_bad = np.random.uniform(-1, 1, (50, 12, 7, 7, 2))
    train_x = np.random.uniform(-1, 1, (50, 12, 7, 7, 1))
    train_y = np.random.uniform(-1, 1, (50, 24))

    assert len(model.layers) == 5, 'conv layers did not get added!'
    assert isinstance(model.layers[0], Conv3D)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], Activation)
    assert isinstance(model.layers[3], Flatten)
    assert isinstance(model.layers[4], Dense)

    # test raise on bad feature channel dimension
    with pytest.raises(AssertionError):
        model.fit(train_x_bad, train_y, train_x, n_batch=1, n_epoch=10)

    model.fit(train_x, train_y, train_x, n_batch=1, n_epoch=10)
    y_pred = model.predict(train_x)
    assert y_pred.shape == (50, 24)

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'tempfile.pkl')
        model.save(fpath)
        loaded = PhysicsGuidedNeuralNetwork.load(fpath)

    assert len(model.layers) == len(loaded.layers)
    for layer0, layer1 in zip(model.layers, loaded.layers):
        for i, weights0 in enumerate(layer0.weights):
            assert weights0.shape == layer1.weights[i].shape
            assert np.allclose(weights0, layer1.weights[i])

    y_pred_loaded = loaded.predict(train_x)

    assert np.allclose(y_pred, y_pred_loaded)
    assert len(model.layers) == len(loaded.layers)


def test_lstm():
    """Test a phygnn model with a conv1d layer. The data in this test is
    garbage, just a test on shapes and creation. Save/load doesnt work yet
    for lstm"""

    input_layer = {'class': 'LSTM', 'units': 24, 'return_sequences': True}
    hidden_layers = [{'units': 64, 'activation': 'relu'}]
    output_layer = {'units': 24}
    model = PhysicsGuidedNeuralNetwork(
        p_fun=p_fun_pythag,
        hidden_layers=hidden_layers,
        input_layer=input_layer,
        output_layer=output_layer,
        loss_weights=(1.0, 0.0),
        n_features=2,
        n_labels=24,
    )

    train_x = np.random.uniform(-1, 1, (50, 12, 2))
    train_y = np.random.uniform(-1, 1, (50, 12, 24))

    assert len(model.layers) == 4, 'lstm layers did not get added!'
    assert isinstance(model.layers[0], LSTM)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], Activation)
    assert isinstance(model.layers[3], Dense)

    model.fit(train_x, train_y, train_x, n_batch=1, n_epoch=10)
    y_pred = model.predict(train_x)
    assert y_pred.shape == (50, 12, 24)

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'tempfile.pkl')
        model.save(fpath)
        loaded = PhysicsGuidedNeuralNetwork.load(fpath)

    assert len(model.layers) == len(loaded.layers)
    for layer0, layer1 in zip(model.layers, loaded.layers):
        for i, weights0 in enumerate(layer0.weights):
            assert weights0.shape == layer1.weights[i].shape
            assert np.allclose(weights0, layer1.weights[i])

    y_pred_loaded = loaded.predict(train_x)

    assert np.allclose(y_pred, y_pred_loaded)
    assert len(model.layers) == len(loaded.layers)


def test_validation_split_shuffle():
    """Test the validation split operation with shuffling"""
    out = PhysicsGuidedNeuralNetwork.get_val_split(
        X, Y, P, shuffle=True, validation_split=0.3
    )
    x, x_val = out[0]
    y, y_val = out[1]
    p, p_val = out[2]

    assert (x_val == p_val).all()
    assert (x == p).all()

    assert id(x) != id(X)
    assert x.shape[1] == x.shape[1]
    assert len(x) == int(0.7 * len(X))
    assert len(x_val) == int(0.3 * len(X))

    assert id(y) != id(Y)
    assert y.shape[1] == y.shape[1]
    assert len(y) == int(0.7 * len(Y))
    assert len(y_val) == int(0.3 * len(Y))

    assert id(p) != id(P)
    assert p.shape[1] == p.shape[1]
    assert len(p) == int(0.7 * len(P))
    assert len(p_val) == int(0.3 * len(P))

    for i in range(len(x_val)):
        row = x_val[i, :]
        assert ~np.any(np.all((row == x), axis=1))

    for i in range(len(p_val)):
        row = p_val[i, :]
        assert ~np.any(np.all((row == p), axis=1))


def test_validation_split_no_shuffle():
    """Test the validation split operation without shuffling"""
    out = PhysicsGuidedNeuralNetwork.get_val_split(
        X, Y, P, shuffle=False, validation_split=0.3
    )
    x, x_val = out[0]
    y, y_val = out[1]
    p, p_val = out[2]
    assert (x_val == p_val).all()
    assert (x == p).all()
    assert all(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2).reshape((len(x), 1)) == y)
    assert (x_val == X[0 : len(x_val)]).all()
    assert (y_val == Y[0 : len(y_val)]).all()
    assert (p_val == P[0 : len(p_val)]).all()


def test_validation_split_5D():
    """Test the validation split with high dimensional data (5D) with only two
    dataset arguments"""
    x0 = np.random.uniform(0, 1, (50, 4, 4, 4, 2))
    y0 = np.random.uniform(0, 1, (50, 4, 1, 1, 1))
    out = PhysicsGuidedNeuralNetwork.get_val_split(
        x0, y0, shuffle=False, validation_split=0.3
    )
    x, x_val = out[0]
    y, y_val = out[1]
    assert len(x0.shape) == 5
    assert len(y0.shape) == 5
    assert len(x.shape) == 5
    assert len(y.shape) == 5
    assert len(x_val.shape) == 5
    assert len(y_val.shape) == 5
    assert (x == x0[-len(x) :]).all()
    assert (y == y0[-len(y) :]).all()
    assert (x_val == x0[: len(x_val)]).all()
    assert (y_val == y0[: len(y_val)]).all()


def test_batching():
    """Test basic batching operation"""
    batch_iter = PhysicsGuidedNeuralNetwork.make_batches(
        X, Y, P, n_batch=4, shuffle=True
    )

    assert isinstance(batch_iter, types.GeneratorType)

    # unpack generator
    batch_iter = list(batch_iter)
    x_batches = [b[0] for b in batch_iter]
    y_batches = [b[1] for b in batch_iter]
    p_batches = [b[2] for b in batch_iter]

    assert len(x_batches) == 4
    assert len(y_batches) == 4
    assert len(p_batches) == 4

    # test on only two batching datasets with batch size
    batch_iter = PhysicsGuidedNeuralNetwork.make_batches(
        X, Y, n_batch=None, batch_size=100, shuffle=False
    )

    assert isinstance(batch_iter, types.GeneratorType)

    batch_iter = list(batch_iter)
    for x, y in batch_iter:
        assert len(x) == 100
        assert len(y) == 100


def test_batching_shuffle():
    """Test the batching operation with shuffling"""
    batch_iter = PhysicsGuidedNeuralNetwork.make_batches(
        X, Y, P, n_batch=4, shuffle=True
    )

    # unpack generator
    batch_iter = list(batch_iter)
    x_batches = [b[0] for b in batch_iter]
    y_batches = [b[1] for b in batch_iter]
    p_batches = [b[2] for b in batch_iter]

    assert len(x_batches) == 4
    assert len(y_batches) == 4
    assert len(p_batches) == 4

    assert ~(x_batches[0] == X[0 : len(x_batches[0]), :]).all()
    assert ~(y_batches[0] == Y[0 : len(y_batches[0]), :]).all()
    assert ~(p_batches[0] == P[0 : len(p_batches[0]), :]).all()

    for i, x_b in enumerate(x_batches):
        assert (x_b == p_batches[i]).all()
        truth = np.sqrt(x_b[:, 0] ** 2 + x_b[:, 1] ** 2).reshape((len(x_b), 1))
        y_check = y_batches[i]
        assert np.allclose(truth, y_check)


def test_batching_no_shuffle():
    """Test the batching operation without shuffling"""
    batch_iter = PhysicsGuidedNeuralNetwork.make_batches(
        X, Y, P, n_batch=6, shuffle=False
    )

    # unpack generator
    batch_iter = list(batch_iter)
    x_batches = [b[0] for b in batch_iter]
    y_batches = [b[1] for b in batch_iter]
    p_batches = [b[2] for b in batch_iter]
    assert len(x_batches) == 6
    assert len(y_batches) == 6
    assert len(p_batches) == 6

    assert (x_batches[0] == X[0 : len(x_batches[0]), :]).all()
    assert (y_batches[0] == Y[0 : len(y_batches[0]), :]).all()
    assert (p_batches[0] == P[0 : len(p_batches[0]), :]).all()

    assert (x_batches[-1] == X[-(len(x_batches[0]) - 1) :, :]).all()
    assert (y_batches[-1] == Y[-(len(y_batches[0]) - 1) :, :]).all()
    assert (p_batches[-1] == P[-(len(p_batches[0]) - 1) :, :]).all()

    for i, x_b in enumerate(x_batches):
        assert (x_b == p_batches[i]).all()
        truth = np.sqrt(x_b[:, 0] ** 2 + x_b[:, 1] ** 2).reshape((len(x_b), 1))
        y_check = y_batches[i]
        assert np.allclose(truth, y_check)


def test_batching_5D():
    """Test the batching with high dimensional data (5D)"""
    x0 = np.random.uniform(0, 1, (50, 4, 4, 4, 2))
    y0 = np.random.uniform(0, 1, (50, 4, 1, 1, 1))
    p0 = x0.copy()

    batch_iter = PhysicsGuidedNeuralNetwork.make_batches(
        x0, y0, p0, n_batch=6, shuffle=False
    )

    # unpack generator
    batch_iter = list(batch_iter)
    x_batches = [b[0] for b in batch_iter]
    y_batches = [b[1] for b in batch_iter]
    p_batches = [b[2] for b in batch_iter]

    assert len(x_batches) == 6
    assert len(y_batches) == 6
    assert len(p_batches) == 6
    assert len(x0.shape) == 5
    assert len(y0.shape) == 5
    assert len(p0.shape) == 5
    assert len(x_batches[0].shape) == 5
    assert len(y_batches[0].shape) == 5
    assert len(p_batches[0].shape) == 5

    assert (x_batches[0] == x0[: len(x_batches[0])]).all()
    assert (y_batches[0] == y0[: len(y_batches[0])]).all()
    assert (p_batches[0] == p0[: len(p_batches[0])]).all()

    assert (x_batches[-1] == x0[-(len(x_batches[0]) - 1) :]).all()
    assert (y_batches[-1] == y0[-(len(y_batches[0]) - 1) :]).all()
    assert (p_batches[-1] == p0[-(len(p_batches[0]) - 1) :]).all()
