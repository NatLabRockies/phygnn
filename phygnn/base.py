# -*- coding: utf-8 -*-
"""
Custom Neural Network Infrastructure.
"""

import logging
import os
import pickle
import pprint
import random
import tempfile
from abc import ABC, abstractmethod
from inspect import signature
from warnings import warn

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout

from phygnn.layers.custom_layers import (
    SkipConnection,
    get_custom_layer_objects,
)
from phygnn.layers.handlers import Layers
from phygnn.utilities import VERSION_RECORD

logger = logging.getLogger(__name__)


class CustomNetwork(ABC):
    """Custom infrastructure for feed forward neural networks.

    Note that the phygnn model requires TensorFlow 2.x
    """

    def __init__(
        self,
        n_features=None,
        n_labels=None,
        hidden_layers=None,
        input_layer=False,
        output_layer=False,
        layers_obj=None,
        feature_names=None,
        output_names=None,
        name=None,
    ):
        """
        Parameters
        ----------
        n_features : int, optional
            Number of input features. This should match the last dimension
            of the feature training data.
        n_labels : int, optional
            Number of output labels. This should match the last dimension
            of the label training data.
        hidden_layers : list, optional
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 8 hidden layers (10 layers including input+output):
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01},
                 {'class': 'Flatten'},
                 ]
        input_layer : None | bool | dict
            Input layer. specification. Can be a dictionary similar to
            hidden_layers specifying a dense / conv / lstm layer.
            Defaults to False so the input layer will be included in the
            hidden_layers input.
        output_layer : None | bool | list | dict
            Output layer specification. Can be a list/dict similar to
            hidden_layers input specifying a dense layer with activation.
            For example, for a classfication problem with a single output,
            output_layer should be [{'units': 1}, {'activation': 'sigmoid'}].
            Default is False so the output layer will be included in the
            hidden_layers input.
        layers_obj : None | phygnn.utilities.tf_layers.Layers
            Optional initialized Layers object to set as the model layers
            including pre-set weights. This option will override the
            hidden_layers, input_layer, and output_layer arguments.
        feature_names : list | tuple | None, optional
            Training feature names (strings). Mostly a convenience so that a
            loaded-from-disk model will have declared feature names, making it
            easier to feed in features for prediction. This will also get set
            if phygnn is trained on a DataFrame.
        output_names : list | tuple | None, optional
            Prediction output names (strings). Mostly a convenience so that a
            loaded-from-disk model will have declared output names, making it
            easier to understand prediction output. This will also get set
            if phygnn is trained on a DataFrame.
        name : None | str
            Optional model name for debugging.
        """

        self._n_features = n_features
        self._n_labels = n_labels
        self.feature_names = feature_names
        self.output_names = output_names
        self.name = name if isinstance(name, str) else 'CustomNetwork'

        self._version_record = VERSION_RECORD
        logger.info(
            'Active python environment versions: \n{}'.format(
                pprint.pformat(self._version_record, indent=4)
            )
        )

        # iterator counter
        self._i = 0

        self._layers = layers_obj
        if layers_obj is None:
            self._layers = Layers(
                n_features,
                n_labels=n_labels,
                hidden_layers=hidden_layers,
                input_layer=input_layer,
                output_layer=output_layer,
            )
        elif not isinstance(layers_obj, Layers):
            msg = (
                'phygnn received layers_obj input of type "{}" but must be '
                'a phygnn Layers object'.format(type(layers_obj))
            )
            logger.error(msg)
            raise TypeError(msg)

        logger.info(
            'Successfully initialized model with {} layers'.format(
                len(self.layers)
            )
        )

    def __iter__(self):
        """Iterate through the layers in this CustomNetwork object."""
        return self

    def __next__(self):
        """Iterate through the layers in this CustomNetwork object."""
        if self._i >= len(self.layers):
            self._i = 0
            raise StopIteration

        layer = self.layers[self._i]
        self._i += 1

        return layer

    @staticmethod
    def _check_shapes(x, y):
        """Check the shape of two input arrays for usage in this NN."""
        msg = (
            "Number of input observations don't match! Received arrays of "
            'shapes {} and {} where the 0-axis should match and be the '
            'number of observations'.format(x.shape, y.shape)
        )
        assert x.shape[0] == y.shape[0], msg
        return True

    @property
    def version_record(self):
        """A record of important versions that this model was built with.

        Returns
        -------
        dict
        """
        return self._version_record

    @property
    def layers(self):
        """
        Ordered list of TensorFlow keras layers that make up this model
        including input and output layers

        Returns
        -------
        list
        """
        return self._layers.layers

    @property
    def layers_obj(self):
        """
        Phygnn layers handler object

        Returns
        -------
        phygnn.utilities.tf_layers.Layers
        """
        return self._layers

    @property
    def weights(self):
        """
        Get a list of layer weights and bias terms for gradient calculations.

        Returns
        -------
        list
        """
        return self._layers.weights

    @property
    def kernel_weights(self):
        """
        Get a list of the NN kernel weights (tensors)

        (can be used for kernel regularization).

        Does not include input layer or dropout layers.
        Does include the output layer.

        Returns
        -------
        list
        """
        return self._layers.kernel_weights

    @property
    def bias_weights(self):
        """
        Get a list of the NN bias weights (tensors)

        (can be used for bias regularization).

        Does not include input layer or dropout layers.
        Does include the output layer.

        Returns
        -------
        list
        """
        return self._layers.bias_weights

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """

        model_params = {
            'hidden_layers': self._layers.hidden_layer_kwargs,
            'input_layer': self._layers.input_layer_kwargs,
            'output_layer': self._layers.output_layer_kwargs,
            'n_features': self._n_features,
            'n_labels': self._n_labels,
            'layers_obj': self.layers_obj,
            'feature_names': self.feature_names,
            'output_names': self.output_names,
            'name': self.name,
            'version_record': self.version_record,
        }

        return model_params

    @staticmethod
    def seed(s=0):
        """
        Set the random seed for reproducable results.

        Parameters
        ----------
        s : int
            Random seed
        """
        random.seed(s)
        np.random.seed(s)
        tf.random.set_seed(s)

    @classmethod
    def get_val_split(cls, *args, shuffle=True, validation_split=0.2):
        """Get a validation split and remove from from the training data.
        This applies the split along the 1st data dimension.

        Parameters
        ----------
        args : np.ndarray
            This is one or more positional arguments that are numpy arrays
            to be split. They must have the same length.
        shuffle : bool
            Flag to randomly subset the validation data from x and y.
            shuffle=False will take the first entries in x and y.
        validation_split : float
            Fraction of x and y to put in the validation set.

        Returns
        -------
        out : list
            List with the same length as the number of positional input
            arguments. Each list entry is itself a list with two entries.
            For example, the first entry in the output is of the format:
            [the training split, and the validation split] and corresponds to
            the first positional input argument.
        """

        L = args[0].shape[0]
        n = int(L * validation_split)

        # get the validation dataset indices, vi
        if shuffle:
            vi = np.random.choice(L, replace=False, size=(n,))
        else:
            vi = np.arange(n)

        # get the training dataset indices, ti
        ti = np.array(list(set(range(L)) - set(vi)))

        assert len(set(vi)) == len(vi)
        assert len(set(list(vi) + list(ti))) == L

        out = [[arg[ti], arg[vi]] for arg in args]

        for out_sub in out[1:]:
            cls._check_shapes(out[0][0], out_sub[0])
            cls._check_shapes(out[0][1], out_sub[1])

        logger.debug(
            'Validation feature data has shape {} and training '
            'feature data has shape {} (split of {})'.format(
                out[0][1].shape, out[0][0].shape, validation_split
            )
        )

        return out

    @staticmethod
    def make_batches(*args, n_batch=16, batch_size=None, shuffle=True):
        """Make lists of unique data batches by splitting x and y along the
        1st data dimension.

        Parameters
        ----------
        args : np.ndarray
            This is one or more positional arguments that are numpy arrays
            to be batched. They must have the same length.
        n_batch : int | None
            Number of times to update the NN weights per epoch. The training
            data will be split into this many batches and the NN will train on
            each batch, update weights, then move onto the next batch.
        batch_size : int | None
            Number of training samples per batch. This input is redundant to
            n_batch and will not be used if n_batch is not None.
        shuffle : bool
            Flag to randomly subset the validation data from x and y.

        Returns
        -------
        batches : GeneratorType
            Generator of batches, each iteration of the generator has as many
            entries as are input in the positional arguments. Each entry in the
            iteration is an ND array with the same original dimensions as the
            input just with a subset batch of the 0 axis
        """

        L = args[0].shape[0]
        if shuffle:
            i = np.random.choice(L, replace=False, size=(L,))
            assert len(set(i)) == L
        else:
            i = np.arange(L)

        for arg in args:
            msg = 'Received arrays to be batched of multiple lengths: {} {}'
            msg = msg.format(L, len(arg))
            assert len(arg) == L, msg

        if n_batch is None and isinstance(batch_size, int):
            n_batch = int(np.ceil(L / batch_size))

        batch_indexes = np.array_split(i, n_batch)

        for batch_index in batch_indexes:
            yield [arg[batch_index] for arg in args]

    def preflight_features(self, x):
        """Run preflight checks and data conversions on feature data.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Feature data in a >=2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm. Generally speaking, the data should
            always have the number of observations in the first axis and the
            number of features/channels in the last axis. Spatial and temporal
            dimensions can be used in intermediate axes.

        Returns
        -------
        x : np.ndarray
            Feature data in a >=2D array
        """

        if self._n_features is None:
            self._n_features = x.shape[-1]

        x_msg = 'x data has {} features but expected {}'.format(
            x.shape[-1], self._n_features
        )
        assert x.shape[-1] == self._n_features, x_msg

        if isinstance(x, pd.DataFrame):
            x_cols = x.columns.values.tolist()
            if self.feature_names is None:
                self.feature_names = x_cols
            else:
                msg = (
                    'Cannot work with input x columns: {}, previously set '
                    'feature names are: {}'.format(x_cols, self.feature_names)
                )
                assert self.feature_names == x_cols, msg
            x = x.values

        return x

    def predict(
        self,
        x,
        to_numpy=True,
        training=False,
        training_layers=(BatchNormalization, Dropout, LSTM),
    ):
        """Run a prediction on input features.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Feature data in a >=2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm. Generally speaking, the data should
            always have the number of observations in the first axis and the
            number of features/channels in the last axis. Spatial and temporal
            dimensions can be used in intermediate axes.
        to_numpy : bool
            Flag to convert output from tensor to numpy array
        training : bool
            Flag for predict() used in the training routine. This is used
            to freeze the BatchNormalization and Dropout layers.
        training_layers : list | tuple
            List of tensorflow.keras.layers classes that training=bool should
            be passed to. By default this is (BatchNormalization, Dropout,
            LSTM)

        Returns
        -------
        y : tf.Tensor | np.ndarray
            Predicted output data.
        """

        x = self.preflight_features(x)

        # Keras 3 InputLayer stores input metadata but is not meant to be
        # called directly during eager execution.
        if self.layers and isinstance(
            self.layers[0], tf.keras.layers.InputLayer
        ):
            y = x
        else:
            y = self.layers[0](x)

        layers = self.layers[1:]

        for i, layer in enumerate(layers, start=1):
            try:
                if isinstance(layer, training_layers):
                    y = layer(y, training=training)
                else:
                    y = layer(y)
            except Exception as e:
                msg = 'Could not run layer #{} "{}" on tensor of shape {}'
                msg = msg.format(i, layer, y.shape)
                logger.error(msg)
                raise RuntimeError(msg) from e

        if to_numpy:
            y = y.numpy()

        return y

    def save(self, fpath):
        """Save phygnn model to a pickle file containing Keras-native model
        bytes and framework metadata.

        Parameters
        ----------
        fpath : str
            File path to `.pkl` file to save model to.
        """

        if not fpath.endswith('.pkl'):
            e = 'Can only save model to .pkl file!'
            logger.error(e)
            raise ValueError(e)

        path = os.path.abspath(fpath)
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        model_params = self._history_to_dict(self.model_params)
        model_params.pop('layers_obj', None)

        keras_model = self._get_keras_wrapper_model(self)
        payload = {
            'model_params': model_params,
            'keras_model_bytes': self._serialize_keras_model(keras_model),
        }

        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info('Saved model to: {}'.format(path))

    @classmethod
    def load(cls, fpath):
        """Load a phygnn model from a pickle file.

        Parameters
        ----------
        fpath : str
            File path to `.pkl` model file.

        Returns
        -------
        model : PhysicsGuidedNeuralNetwork
            Instantiated phygnn model
        """

        logger.info('Loading saved model: {}'.format(fpath))

        path = os.path.abspath(fpath)
        if not os.path.exists(path):
            e = 'Could not load file, does not exist: {}'.format(path)
            logger.error(e)
            raise FileNotFoundError(e)
        if not os.path.isfile(path) or not path.endswith('.pkl'):
            e = 'Can only load model from .pkl file: {}'.format(path)
            logger.error(e)
            raise ValueError(e)

        with open(path, 'rb') as f:
            payload = pickle.load(f)
        model = cls._load_keras_pickle_model(payload)

        logger.info(
            'Successfully initialized model from file: {}'.format(fpath)
        )

        return model

    @classmethod
    def _normalize_model_params(cls, model_params):
        """Normalize serialized model parameters before initialization."""
        model_params = cls._history_to_df(model_params)
        if 'version_record' in model_params:
            version_record = model_params.pop('version_record')
            logger.info(
                'Loading model from disk that was created with the '
                'following package versions: \n{}'.format(
                    pprint.pformat(version_record, indent=4)
                )
            )
        return model_params

    @classmethod
    def _init_model_from_params(cls, model_params):
        """Initialize a model instance from serialized constructor args."""
        sig = signature(cls)
        model_params = {
            k: v for k, v in model_params.items() if k in sig.parameters
        }
        return cls(**model_params)

    @staticmethod
    def _serialize_keras_model(keras_model):
        """Serialize a Keras model to bytes using the native Keras format."""
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.keras')
            keras_model.save(model_path)
            with open(model_path, 'rb') as f:
                return f.read()

    @staticmethod
    def _load_keras_model_from_bytes(model_bytes):
        """Load a Keras model from serialized bytes."""
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.keras')
            with open(model_path, 'wb') as f:
                f.write(model_bytes)
            return tf.keras.models.load_model(
                model_path, custom_objects=get_custom_layer_objects()
            )

    @classmethod
    def _check_layer_count(cls, ordered_layers, model_params):
        """Warn if the loaded layer count does not match what the saved
        hidden_layers config would build.

        Parameters
        ----------
        ordered_layers : list
            Non-input layers extracted from the loaded Keras model.
        model_params : dict
            Deserialized model parameters from the pickle payload.
        """
        layers_kwargs = {
            k: v
            for k, v in model_params.items()
            if k in signature(Layers).parameters
        }
        reference = Layers(**layers_kwargs)
        ref_non_input = [
            layer
            for layer in reference.layers
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]
        if len(ref_non_input) != len(ordered_layers):
            e = (
                'Loaded model has {} layers but the saved hidden_layers '
                'config builds {} layers. Model did not load '
                'correctly.'.format(len(ordered_layers), len(ref_non_input))
            )
            logger.error(e)
            raise RuntimeError(e)

    @classmethod
    def _load_keras_pickle_model(cls, payload):
        """Load a pickle payload with embedded Keras model bytes."""
        model_params = cls._normalize_model_params(payload['model_params'])
        loaded_keras_model = cls._load_keras_model_from_bytes(
            payload['keras_model_bytes']
        )

        ordered_layers = cls._get_non_input_layers(loaded_keras_model)
        cls._check_layer_count(ordered_layers, model_params)

        layers_kwargs = {
            k: v
            for k, v in model_params.items()
            if k in signature(Layers.from_layers).parameters
        }
        layers_obj = Layers.from_layers(ordered_layers, **layers_kwargs)
        model_params['layers_obj'] = layers_obj

        return cls._init_model_from_params(model_params)

    @classmethod
    def _get_non_input_layers(cls, model):
        """Get model layers excluding any stored InputLayer wrapper."""
        return [
            layer
            for layer in model.layers
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]

    @staticmethod
    def _skip_layer_serialization_name(base_name, occurrence):
        """Return a unique name for the *occurrence*-th appearance of a
        SkipConnection named *base_name* in the serialized layer list.

        Even occurrences → ``_start`` (the caching half).
        Odd  occurrences → ``_end``   (the combining half).
        Pairs beyond the first get a numeric suffix: ``_start_1``, ``_end_1``.
        """
        suffix = '_start' if occurrence % 2 == 0 else '_end'
        pair_idx = occurrence // 2
        if pair_idx == 0:
            return f'{base_name}{suffix}'
        return f'{base_name}{suffix}_{pair_idx}'

    @classmethod
    def _get_keras_wrapper_model(cls, model):
        """Build a Keras Sequential wrapper around the model's layers.

        When the same ``SkipConnection`` instance appears more than once (the
        normal runtime representation), Keras deduplicates them to a single
        layer on ``save()`` / ``load()``.  To prevent that, each occurrence is
        replaced with a new SkipConnection whose name has a ``_start`` /
        ``_end`` suffix.  After loading,
        :meth:`~phygnn.layers.handlers.Layers.relink_skip_connections`
        strips the suffixes and restores the shared-instance invariant.

        ``model._layers.skip_layers`` guarantees that each unique instance has
        a unique name, so layer names are used as keys instead of object ids.
        """
        wrapper_model = tf.keras.Sequential(name=model.name)
        skip_counts = {}  # base name -> occurrence count
        for layer in cls._get_non_input_layers(model):
            if isinstance(layer, SkipConnection):
                count = skip_counts.get(layer.name, 0)
                layer_name = cls._skip_layer_serialization_name(
                    layer.name, count
                )
                skip_counts[layer.name] = count + 1
                wrapper_model.add(
                    SkipConnection(layer_name, method=layer._method)
                )
            else:
                wrapper_model.add(layer)
        # The constituent layers are already built from training (they
        # have weights) even though the Sequential wrapper itself has
        # never been called as a unit.  Mark it built so Keras does not
        # emit a spurious "not yet built" warning when serializing.
        wrapper_model.built = bool(wrapper_model.weights)
        if not wrapper_model.built:
            msg = 'Saving model "{}" with unbuilt Keras wrapper.'.format(
                model.name
            )
            logger.warning(msg)
            warn(msg)
        return wrapper_model

    @classmethod
    def _history_to_dict(cls, model_params):
        """Make sure history is a dictionary prior to saving"""
        if isinstance(model_params.get('history', None), pd.DataFrame):
            model_params['history'] = model_params['history'].to_dict()
        return model_params

    @classmethod
    def _history_to_df(cls, model_params):
        """Convert history to pandas dataframe after model initialization"""
        if isinstance(model_params.get('history', None), dict):
            model_params['history'] = pd.DataFrame(model_params['history'])
        return model_params


class GradientUtils(ABC):
    """TF 2.0 gradient descent utilities."""

    def __init__(self):
        # placeholders attributes for concrete class
        self._layers = []
        self.weights = None
        self._optimizer = None

    @abstractmethod
    def predict(self, x):
        """Placeholder for loss function

        Parameters
        ----------
        x : np.ndarray
            Input feature data to predict on in a >=2D array.

        Returns
        -------
        y_predicted : tf.Tensor
            Model-predicted output data in a >=2D tensor.
        """

    @abstractmethod
    def calc_loss(self, y_true, y_predicted):
        """Placeholder for loss function

        Parameters
        ----------
        y_true : np.ndarray
            Known output data in a >=2D array.
        y_predicted : tf.Tensor
            Model-predicted output data in a >=2D tensor.

        Returns
        -------
        loss : tf.tensor
            Loss function output comparing the y_predicted against y_true.
        """

    def _get_grad(self, x, y_true):
        """Get the gradient based on a mini-batch of x and y_true data.

        Parameters
        ----------
        x : np.ndarray
            Feature data in a >=2D array. Generally speaking, the data should
            always have the number of observations in the first axis and the
            number of features/channels in the last axis. Spatial and temporal
            dimensions can be used in intermediate axes.
        y_true : np.ndarray
            Known y values.

        Returns
        -------
        grad : tf.Tensor
            Gradient data relating the change in model weights to the change in
            loss value
        loss :
            Loss function output comparing the y_predicted against y_true.
        """
        with tf.GradientTape() as tape:
            y_predicted = self.predict(x, to_numpy=False, training=True)
            loss = self.calc_loss(y_true, y_predicted)
            grad = tape.gradient(loss, self.weights)

        return grad, loss

    def run_gradient_descent(self, x, y_true):
        """Run gradient descent for one mini-batch of (x, y_true)
        and adjust NN weights

        Parameters
        ----------
        x : np.ndarray
            Feature data in a >=2D array. Generally speaking, the data should
            always have the number of observations in the first axis and the
            number of features/channels in the last axis. Spatial and temporal
            dimensions can be used in intermediate axes.
        y_true : np.ndarray
            Known y values.

        Returns
        -------
        loss : tf.Tensor
            Loss function output comparing the y_predicted against y_true.
        """
        grad, loss = self._get_grad(x, y_true)
        self._optimizer.apply_gradients(zip(grad, self.weights))
        return loss
