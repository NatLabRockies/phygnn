# -*- coding: utf-8 -*-
"""
Tensorflow Layers Handlers
"""
import copy
import logging

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    InputLayer,
)

import phygnn.layers.custom_layers
from phygnn.layers.custom_layers import SkipConnection

logger = logging.getLogger(__name__)


class HiddenLayers:
    """
    Class to handle TensorFlow hidden layers
    """

    def __init__(self, hidden_layers):
        """
        Parameters
        ----------
        hidden_layers : list
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 7 hidden layers:
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01}]
        """
        self._i = 0
        self._layers = []
        self._hidden_layers_kwargs = copy.deepcopy(hidden_layers)

        if self._hidden_layers_kwargs is not None:
            self._hidden_layers_kwargs = self.parse_repeats(
                self._hidden_layers_kwargs)

        for layer in self._hidden_layers_kwargs:
            self.add_layer(layer)

    def __repr__(self):
        msg = "{} with {} hidden layers".format(self.__class__.__name__,
                                                len(self))

        return msg

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        """
        Get layer at given index

        Parameters
        ----------
        idx : int
            index of layer to extract

        Returns
        -------
        tensorflow.keras.layers
        """
        return self.layers[idx]

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration

        layer = self[self._i]
        self._i += 1

        return layer

    @staticmethod
    def parse_repeats(hidden_layers):
        """Parse repeat layers. Must have "repeat" and "n" to repeat one
        or more layers.

        Parameters
        ----------
        hidden_layers : list
            Hidden layer kwargs including possibly entries with
            {'n': 2, 'repeat': [{...}, {...}]} that will duplicate the list sub
            entry n times.

        Returns
        -------
        hidden_layers : list
            Hidden layer kwargs exploded for 'repeat' entries.
        """

        out = []
        for layer in hidden_layers:
            if 'repeat' in layer and 'n' in layer:
                repeat = layer.pop('repeat')
                repeat = [repeat] if isinstance(repeat, dict) else repeat
                n = layer.pop('n')
                for _ in range(n):
                    out += repeat

            elif 'repeat' in layer and 'n' not in layer:
                msg = ('Keyword "repeat" was found in layer but "n" was not: '
                       '{}'.format(layer))
                raise KeyError(msg)

            else:
                out.append(layer)

        return out

    @property
    def layers(self):
        """
        TensorFlow keras layers

        Returns
        -------
        list
        """
        return self._layers

    @property
    def skip_layers(self):
        """
        Get a dictionary of unique SkipConnection objects in the layers list
        keyed by SkipConnection name.

        Returns
        -------
        list
        """
        out = {}
        for layer in self.layers:
            if isinstance(layer, SkipConnection):
                out[layer.name] = layer
        return out

    @property
    def hidden_layer_kwargs(self):
        """
        List of dictionaries of key word arguments for each hidden
        layer in the NN. This is a copy of the hidden_layers input arg
        that can be used to reconstruct the network.

        Returns
        -------
        list
        """
        return self._hidden_layers_kwargs

    @property
    def weights(self):
        """
        Get a list of layer weights for gradient calculations.

        Returns
        -------
        list
        """
        weights = []
        for layer in self:
            # Include gamma and beta weights for BatchNormalization
            # but do not include moving mean/stdev
            if isinstance(layer, BatchNormalization):
                weights += layer.variables[:2]

            elif layer.trainable:
                weights += layer.trainable_weights

        return weights

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
        weights = []
        for layer in self:
            if isinstance(layer, Dense):
                weights.append(layer.variables[0])

        return weights

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
        weights = []
        for layer in self:
            if isinstance(layer, Dense):
                weights.append(layer.variables[1])

        return weights

    def add_skip_layer(self, name, method='add'):
        """Add a skip layer, looking for a prior skip connection start point if
        already in the layer list.

        Parameters
        ----------
        name : str
            Unique string identifier of the skip connection. The skip endpoint
            should have the same name.
        method : str
            Method to use for combining skip start data and skip end data.
        """
        if name in self.skip_layers:
            self._layers.append(self.skip_layers[name])
        else:
            self._layers.append(SkipConnection(name, method))

    def add_layer_by_class(self, class_name, **kwargs):
        """Add a new layer by the class name, either from
        phygnn.layers.custom_layers or tf.keras.layers

        Parameters
        ----------
        class_name : str
            Class name from phygnn.layers.custom_layers or tf.keras.layers
        kwargs : dict
            Key word arguments to initialize the class.
        """
        layer_class = None
        msg = ('Need layer "class" definition as string to retrieve '
               'from phygnn.layers.custom_layers or from '
               'tensorflow.keras.layers, but received: {} {}'
               .format(type(class_name), class_name))
        assert isinstance(class_name, str), msg

        # prioritize phygnn custom classes
        layer_class = getattr(phygnn.layers.custom_layers, class_name, None)

        if layer_class is None:
            layer_class = getattr(tf.keras.layers, class_name, None)

        if layer_class is None:
            msg = ('Could not retrieve layer class "{}" from '
                   'phygnn.layers.custom_layers or from '
                   'tensorflow.keras.layers'
                   .format(class_name))
            logger.error(msg)
            raise KeyError(msg)

        # If a layer has a "hidden_layers" argument it's assumed that
        # "layer_class" holds a nested set of hidden layers
        if 'hidden_layers' in kwargs:
            hl = HiddenLayers(hidden_layers=kwargs.pop('hidden_layers'))
            layer = layer_class(hidden_layers=hl._layers, **kwargs)
            self._layers.append(layer)
        elif layer_class == SkipConnection:
            self.add_skip_layer(**kwargs)
        else:
            self._layers.append(layer_class(**kwargs))

    def add_layer(self, layer_kwargs):
        """Add a hidden layer to the DNN.

        Parameters
        ----------
        layer_kwargs : dict
            Dictionary of key word arguments for list layer. For example,
            any of the following are valid inputs:
                {'units': 64, 'activation': 'relu', 'dropout': 0.05}
                {'units': 64, 'name': 'relu1'}
                {'activation': 'relu'}
                {'batch_normalization': {'axis': -1}}
                {'dropout': 0.1}
        """

        layer_kws = copy.deepcopy(layer_kwargs)

        class_name = layer_kws.pop('class', None)
        if class_name is not None:
            self.add_layer_by_class(class_name, **layer_kws)

        else:
            activation_arg = layer_kws.pop('activation', None)
            dropout_rate = layer_kws.pop('dropout', None)
            batch_norm_kwargs = layer_kws.pop('batch_normalization', None)
            dense_units = layer_kws.pop('units', None)

            dense_layer = None
            bn_layer = None
            a_layer = None
            drop_layer = None

            if dense_units is not None:
                dense_layer = Dense(dense_units)
            if batch_norm_kwargs is not None:
                bn_layer = BatchNormalization(**batch_norm_kwargs)
            if activation_arg is not None:
                a_layer = Activation(activation_arg)
            if dropout_rate is not None:
                drop_layer = Dropout(dropout_rate)

            # This ensures proper default ordering of layers if requested
            # together.
            for layer in [dense_layer, bn_layer, a_layer, drop_layer]:
                if layer is not None:
                    self._layers.append(layer)

    @classmethod
    def compile(cls, model, hidden_layers):
        """
        Add hidden layers to model

        Parameters
        ----------
        model : tensorflow.keras
            Model to add hidden layers too
        hidden_layers : list
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 7 hidden layers:
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01}]

        Returns
        -------
        model : tensorflow.keras
            Model with layers added
        """
        hidden_layers = cls(hidden_layers)
        for layer in hidden_layers:
            model.add(layer)

        return model


class Layers(HiddenLayers):
    """
    Class to handle TensorFlow layers
    """

    def __init__(self, n_features, n_labels=1, hidden_layers=None,
                 input_layer=None, output_layer=None):
        """
        Parameters
        ----------
        n_features : int
            Number of features (inputs) to train the model on
        n_labels : int, optional
            Number of labels (outputs) to the model, by default 1
        hidden_layers : list | None, optional
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
            by default None which will lead to a single linear layer
        input_layer : None | bool | dict
            Input layer. specification. Can be a dictionary similar to
            hidden_layers specifying a dense / conv / lstm layer.  Will
            default to a keras InputLayer with input shape = n_features.
            Can be False if the input layer will be included in the
            hidden_layers input.
        output_layer : None | bool | list | dict
            Output layer specification. Can be a list/dict similar to
            hidden_layers input specifying a dense layer with activation.
            For example, for a classfication problem with a single output,
            output_layer should be [{'units': 1}, {'activation': 'sigmoid'}].
            This defaults to a single dense layer with no activation
            (best for regression problems).  Can be False if the output layer
            will be included in the hidden_layers input.
        """

        self._i = 0
        self._layers = []
        self._n_features = n_features
        self._n_labels = n_labels
        self._input_layer_kwargs = copy.deepcopy(input_layer)
        self._output_layer_kwargs = copy.deepcopy(output_layer)
        self._hidden_layers_kwargs = copy.deepcopy(hidden_layers)

        if self._hidden_layers_kwargs is not None:
            self._hidden_layers_kwargs = self.parse_repeats(
                self._hidden_layers_kwargs)

        self._add_input_layer()

        if hidden_layers:
            for layer in self._hidden_layers_kwargs:
                self.add_layer(layer)

        self._add_output_layer()

    @classmethod
    def from_layers(
        cls,
        layers,
        n_features=None,
        n_labels=1,
        hidden_layers=None,
        input_layer=None,
        output_layer=None,
        input_shape=None,
    ):
        """Wrap an existing ordered list of Keras layers.

        Parameters
        ----------
        layers : list | tuple
            Ordered Keras layers including any input/output layers.
        n_features : int | None
            Number of input features used to initialize the original model.
        n_labels : int
            Number of output labels used to initialize the original model.
        hidden_layers : list | None
            Serialized hidden layer kwargs metadata.
        input_layer : None | bool | dict
            Serialized input layer kwargs metadata.
        output_layer : None | bool | list | dict
            Serialized output layer kwargs metadata.
        input_shape : tuple | list | None
            Explicit input shape metadata used to restore the default
            InputLayer wrapper when needed.

        Returns
        -------
        Layers
            Layers wrapper around the existing Keras layer objects.
        """

        obj = cls.__new__(cls)
        obj._i = 0
        obj._layers = list(layers)
        obj._n_features = n_features
        obj._n_labels = n_labels
        obj._input_layer_kwargs = copy.deepcopy(input_layer)
        obj._output_layer_kwargs = copy.deepcopy(output_layer)
        obj._hidden_layers_kwargs = copy.deepcopy(hidden_layers)

        if (
            obj._input_layer_kwargs is None
            and obj._layers
            and not isinstance(obj._layers[0], InputLayer)
        ):
            if input_shape is None and n_features is not None:
                input_shape = (n_features,)
            if input_shape is not None:
                obj._layers.insert(0, InputLayer(shape=input_shape))

        if obj._hidden_layers_kwargs is not None:
            obj._hidden_layers_kwargs = cls.parse_repeats(
                obj._hidden_layers_kwargs
            )

        return obj

    def _add_input_layer(self):
        """Add an input layer, defaults to tf.layers.InputLayer"""

        if self.input_layer_kwargs is None:
            self._layers = [InputLayer(shape=(self._n_features,))]

        elif self.input_layer_kwargs:
            if not isinstance(self.input_layer_kwargs, dict):
                msg = ('Input layer spec needs to be a dict but received: {}'
                       .format(type(self.input_layer_kwargs)))
                raise TypeError(msg)
            self.add_layer(self.input_layer_kwargs)

    def _add_output_layer(self):
        """Add an output layer, defaults to tf.layers.Dense without activation
        """

        if self._output_layer_kwargs is None:
            self._layers.append(Dense(self._n_labels))
        elif self._output_layer_kwargs:
            if isinstance(self._output_layer_kwargs, dict):
                self._output_layer_kwargs = [self._output_layer_kwargs]
            if not isinstance(self._output_layer_kwargs, list):
                msg = ('Output layer spec needs to be a dict or list but '
                       'received: {}'.format(type(self._output_layer_kwargs)))
                raise TypeError(msg)
            for layer in self._output_layer_kwargs:
                self.add_layer(layer)

    @property
    def input_layer_kwargs(self):
        """
        Dictionary of key word arguments for the input layer.
        This is a copy of the input_layer input arg
        that can be used to reconstruct the network.

        Returns
        -------
        list
        """
        return self._input_layer_kwargs

    @property
    def output_layer_kwargs(self):
        """
        Dictionary of key word arguments for the output layer.
        This is a copy of the output_layer input arg
        that can be used to reconstruct the network.

        Returns
        -------
        list
        """
        return self._output_layer_kwargs

    @classmethod
    def compile(cls, model, n_features, n_labels=1, hidden_layers=None,
                input_layer=None, output_layer=None):
        """
        Build all layers needed for model

        Parameters
        ----------
        model : tensorflow.keras.Sequential
            Model to add layers too
        n_features : int
            Number of features (inputs) to train the model on
        n_labels : int, optional
            Number of labels (outputs) to the model, by default 1
        hidden_layers : list | None, optional
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 7 hidden layers (9 layers total):
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01}]
            by default None which will lead to a single linear layer
        input_layer : None | bool | InputLayer
            Keras input layer. Will default to an InputLayer with
            input shape = n_features.
            Can be False if the input layer will be included in the
            hidden_layers input.
        output_layer : None | bool | list | dict
            Output layer specification. Can be a list/dict similar to
            hidden_layers input specifying a dense layer with activation.
            For example, for a classfication problem with a single output,
            output_layer should be [{'units': 1}, {'activation': 'sigmoid'}]
            This defaults to a single dense layer with no activation
            (best for regression problems).  Can be False if the output layer
            will be included in the hidden_layers input.

        Returns
        -------
        model : tensorflow.keras
            Model with layers added
        """
        layers = cls(n_features, n_labels=n_labels,
                     hidden_layers=hidden_layers,
                     input_layer=input_layer,
                     output_layer=output_layer)
        for layer in layers:
            model.add(layer)

        return model
