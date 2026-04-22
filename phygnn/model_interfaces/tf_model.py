# -*- coding: utf-8 -*-
"""
TensorFlow Model
"""
import json
import logging
import os
import pprint
from warnings import warn

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras.optimizers import Adam

from phygnn.layers.handlers import Layers
from phygnn.model_interfaces.base_model import ModelBase
from phygnn.utilities import TF2
from phygnn.utilities.pre_processing import PreProcess

logger = logging.getLogger(__name__)


class TfModel(ModelBase):
    """
    TensorFlow Keras Sequential Model interface
    """

    def __init__(self, model, feature_names=None, label_names=None,
                 norm_params=None, normalize=(True, False),
                 one_hot_categories=None):
        """
        Parameters
        ----------
        model : tensorflow.keras.models.Sequential
            Tensorflow Keras Model
        feature_names : list
            Ordered list of feature names.
        label_names : list
            Ordered list of label (output) names.
        norm_params : dict, optional
            Dictionary mapping feature and label names (keys) to normalization
            parameters (mean, stdev), by default None
        normalize : bool | tuple, optional
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)
            by default True
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories, if None do
            not run one-hot encoding, by default None
        """
        super().__init__(model, feature_names=feature_names,
                         label_names=label_names, norm_params=norm_params,
                         normalize=normalize,
                         one_hot_categories=one_hot_categories)

        self._history = None

    @property
    def layers(self):
        """
        Model layers

        Returns
        -------
        list
        """
        return self.model.layers

    @property
    def weights(self):
        """
        Get a list of layer weights for gradient calculations.

        Returns
        -------
        list
        """
        weights = []
        for layer in self.layers:
            weights += layer.get_weights()

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
        for layer in self.layers:
            weights.append(layer.get_weights()[0])

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
        for layer in self.layers:
            weights.append(layer.get_weights()[1])

        return weights

    @property
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        if self._history is None:
            msg = 'Model has not been trained yet!'
            logger.warning(msg)
            warn(msg)
            history = None
        else:
            history = pd.DataFrame(self._history.history)
            history['epoch'] = self._history.epoch

        return history

    @staticmethod
    def _clean_name(name):
        """
        Make feature | label name compatible with TensorFlow

        Parameters
        ----------
        name : str
            Feature |label name from GOOML

        Returns
        -------
        name : str
            Feature | label name compatible with TensorFlow
        """
        name = name.replace(' ', '_')
        name = name.replace('*', '-x-')
        name = name.replace('+', '-plus-')
        name = name.replace('**', '-exp-')
        name = name.replace(')', '')
        name = name.replace('log(', 'log-')

        return name

    @staticmethod
    def _generate_feature_columns(features):
        """
        Generate feature layer from features table

        Parameters
        ----------
        features : dict
            model features

        Returns
        -------
        feature_columns : list
            List of tensorFlow.feature_column objects
        """
        feature_columns = []
        for name, data in features.items():
            name = TfModel._clean_name(name)
            if np.issubdtype(data.dtype.name, np.number):
                f_col = feature_column.numeric_column(name)
            else:
                f_col = TfModel._generate_cat_column(name, data)

            feature_columns.append(f_col)

        return feature_columns

    @staticmethod
    def _generate_cat_column(name, data, vocab_threshold=50, bucket_size=100):
        """Generate a feature column from a categorical string data set

        Parameters
        ----------
        name : str
            Name of categorical columns
        data : np.ndarray | list
            String data array
        vocab_threshold : int
            Number of unique entries in the data array below which this
            will use a vocabulary list, above which a hash bucket will be used.
        bucket_size : int
            Hash bucket size.

        Returns
        -------
        f_col : IndicatorColumn
            Categorical feature column.
        """

        n_unique = len(set(data))

        if n_unique < vocab_threshold:
            f_col = feature_column.categorical_column_with_vocabulary_list(
                name, list(set(data)))
        else:
            f_col = feature_column.categorical_column_with_hash_bucket(
                name, bucket_size)

        f_col = feature_column.indicator_column(f_col)

        return f_col

    @staticmethod
    def _build_feature_columns(feature_columns):
        """
        Build the feature layer from given feature column descriptions

        Parameters
        ----------
        feature_columns : list
            list of feature column descriptions (dictionaries)

        Returns
        -------
        tf_columns : list
            List of tensorFlow.feature_column objects
        """
        tf_columns = {}
        col_map = {}  # TODO: build map to tf.feature_column functions
        # TODO: what feature_columns need to be wrapped
        indicators = [feature_column.categorical_column_with_hash_bucket,
                      feature_column.categorical_column_with_identity,
                      feature_column.categorical_column_with_vocabulary_file,
                      feature_column.categorical_column_with_vocabulary_list,
                      feature_column.crossed_column]
        for col in feature_columns:
            name = col['name']
            f_type = col_map.get(col['type'], col['type'])
            kwargs = col.get('kwargs', {})

            if f_type == feature_column.crossed_column:
                cross_cols = [tf_columns[name]
                              for name in col['cross_columns']]
                f_col = f_type(cross_cols, **kwargs)
            elif f_type == feature_column.embedding_column:
                embedded_type = col_map[col['embedded_col']]
                f_col = embedded_type(name, **kwargs)
                f_col = f_type(f_col, **kwargs)
            else:
                f_col = f_type(name, **kwargs)

            if f_type in indicators:
                f_col = feature_column.indicator_column(f_col)

            tf_columns[name] = f_col

        return tf_columns

    @staticmethod
    def compile_model(n_features, n_labels=1, hidden_layers=None,
                      input_layer=None, output_layer=None,
                      learning_rate=0.001, loss="mean_squared_error",
                      metrics=('mae', 'mse'), optimizer_class=Adam, **kwargs):
        """
        Build tensorflow sequential model from given layers and kwargs

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
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        optimizer_class : tf.keras.optimizers, optional
            Optional explicit request of optimizer. This should be a class
            that will be instantated in the TfModel.compile_model() method
            The default is the Adam optimizer
        kwargs : dict
            kwargs for tensorflow.keras.models.compile

        Returns
        -------
        tensorflow.keras.models.Sequential
            Compiled tensorflow Sequential model
        """
        model = tf.keras.models.Sequential()
        model = Layers.compile(model, n_features, n_labels=n_labels,
                               hidden_layers=hidden_layers,
                               input_layer=input_layer,
                               output_layer=output_layer)

        if isinstance(metrics, tuple):
            metrics = list(metrics)
        elif not isinstance(metrics, list):
            metrics = [metrics]

        optimizer = optimizer_class(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                      **kwargs)

        return model

    def train_model(self, features, labels, epochs=100, shuffle=True,
                    validation_split=0.2, early_stop=True, stop_kwargs=None,
                    parse_kwargs=None, fit_kwargs=None):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        features : dict | pandas.DataFrame
            Input features to train on
        labels : dict | pandas.DataFrame
            label to train on
        norm_labels : bool, optional
            Flag to normalize label, by default True
        epochs : int, optional
            Number of epochs to train the model, by default 100
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from features and labels.
        validation_split : float, optional
            Fraction of the training data to be used as validation data,
            by default 0.2
        early_stop : bool
            Flag to stop training when it stops improving
        stop_kwargs : dict | None
            kwargs for tf.keras.callbacks.EarlyStopping() if early_stop
            default is: {'monitor': 'val_loss', 'patience': 10}
        parse_kwargs : dict | None
            kwargs for cls.parse_features
        fit_kwargs : dict | None
            kwargs for tensorflow.keras.models.fit
        """

        parse_kwargs = parse_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        stop_kwargs = stop_kwargs or {'monitor': 'val_loss', 'patience': 10}

        if (isinstance(features, np.ndarray)
                and features.shape[-1] == self.feature_dims):
            parse_kwargs['names'] = self.feature_names

        label_names = None
        if (isinstance(labels, np.ndarray)
                and labels.shape[-1] == self.label_dims):
            label_names = self.label_names

        features = self.parse_features(features, **parse_kwargs)
        labels = self.parse_labels(labels, names=label_names)

        if self._history is not None:
            msg = 'Model has already been trained and will be re-fit!'
            logger.warning(msg)
            warn(msg)

        if early_stop:
            early_stop = tf.keras.callbacks.EarlyStopping(**stop_kwargs)
            callbacks = fit_kwargs.pop('callbacks', None)
            if callbacks is None:
                callbacks = [early_stop]
            else:
                callbacks.append(early_stop)

            fit_kwargs['callbacks'] = callbacks

        if shuffle:
            L = len(features)
            i = np.random.choice(L, size=L, replace=False)
            features = features[i]
            labels = labels[i]

        if validation_split > 0:
            split = int(len(features) * validation_split)
            validate_features = features[-split:]
            validate_labels = labels[-split:]
            validation_data = (validate_features, validate_labels)

            features = features[:-split]
            labels = labels[:-split]
        else:
            validation_data = None

        self._history = self._model.fit(x=features, y=labels, epochs=epochs,
                                        validation_data=validation_data,
                                        **fit_kwargs)

    def save_model(self, path):
        """
        Save TfModel to path.

        Parameters
        ----------
        path : str
            Directory path to save model to. The tensorflow model will be
            saved to the directory while the framework parameters will be
            saved in json.
        """
        if path.endswith('.json'):
            path = path.replace('.json', '/')

        if not path.endswith('/'):
            path += '/'

        if not os.path.exists(path):
            os.makedirs(path)

        if TF2:
            self.model.save(path)
        if not TF2:
            tf.saved_model.save(self.model, path)

        model_params = {'feature_names': self.feature_names,
                        'label_names': self.label_names,
                        'norm_params': self.normalization_parameters,
                        'normalize': (self.normalize_features,
                                      self.normalize_labels),
                        'one_hot_categories': self.one_hot_categories,
                        'version_record': self.version_record,
                        }

        json_path = path + 'model.json'
        model_params = self.dict_json_convert(model_params)
        with open(json_path, 'w') as f:
            json.dump(model_params, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        """
        Load model from model path.

        Parameters
        ----------
        path : str
            Directory path for TfModel to load model from. There should be a
            saved model directory with json and pickle files for the TfModel
            framework.

        Returns
        -------
        model : TfModel
            Loaded TfModel from disk.
        """
        if path.endswith('.json'):
            path = path.replace('.json', '/')

        if not path.endswith('/'):
            path += '/'

        if not os.path.isdir(path):
            e = ('Can only load directory path but target is not '
                 'directory: {}'.format(path))
            logger.error(e)
            raise OSError(e)

        loaded = tf.keras.models.load_model(path)

        json_path = path + 'model.json'
        with open(json_path) as f:
            model_params = json.load(f)

        if 'version_record' in model_params:
            version_record = model_params.pop('version_record')
            logger.info('Loading model from disk that was created with the '
                        'following package versions: \n{}'
                        .format(pprint.pformat(version_record, indent=4)))

        model = cls(loaded, **model_params)

        return model

    @classmethod
    def build(cls, feature_names, label_names,
              normalize=(True, False),
              one_hot_categories=None,
              hidden_layers=None,
              input_layer=None,
              output_layer=None,
              learning_rate=0.001,
              loss="mean_squared_error",
              metrics=('mae', 'mse'),
              optimizer_class=Adam,
              **kwargs):
        """
        Build tensorflow sequential model from given features, layers and
        kwargs

        Parameters
        ----------
        feature_names : list
            Ordered list of feature names.
        label_names : list
            Ordered list of label (output) names.
        normalize : bool | tuple, optional
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)
            by default True
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories, if None do
            not run one-hot encoding, by default None
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
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        optimizer_class : tf.keras.optimizers, optional
            Optional explicit request of optimizer. This should be a class
            that will be instantated in the TfModel.compile_model() method
            The default is the Adam optimizer
        kwargs : dict
            kwargs for tensorflow.keras.models.compile

        Returns
        -------
        model : TfModel
            Initialized TfModel obj
        """
        if isinstance(label_names, str):
            label_names = [label_names]

        if one_hot_categories is not None:
            check_names = feature_names + label_names
            PreProcess.check_one_hot_categories(one_hot_categories,
                                                feature_names=check_names)
            feature_names = cls.make_one_hot_feature_names(feature_names,
                                                           one_hot_categories)

        model = cls.compile_model(len(feature_names),
                                  n_labels=len(label_names),
                                  hidden_layers=hidden_layers,
                                  input_layer=input_layer,
                                  output_layer=output_layer,
                                  learning_rate=learning_rate, loss=loss,
                                  metrics=metrics,
                                  optimizer_class=optimizer_class,
                                  **kwargs)

        model = cls(model, feature_names=feature_names,
                    label_names=label_names, normalize=normalize,
                    one_hot_categories=one_hot_categories)

        return model

    @classmethod
    def build_trained(cls, features, labels, normalize=(True, False),
                      one_hot_categories=None, hidden_layers=None,
                      input_layer=None, output_layer=None,
                      learning_rate=0.001, loss="mean_squared_error",
                      metrics=('mae', 'mse'), optimizer_class=Adam, epochs=100,
                      shuffle=True, validation_split=0.2,
                      early_stop=True, stop_kwargs=None,
                      save_path=None, compile_kwargs=None, parse_kwargs=None,
                      fit_kwargs=None):
        """
        Build tensorflow sequential model from given features, layers and
        kwargs and then train with given label and kwargs

        Parameters
        ----------
        features : dict | pandas.DataFrame
            Model features
        labels : dict | pandas.DataFrame
            label to train on
        normalize : bool | tuple, optional
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)
            by default True
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories, if None do
            not run one-hot encoding, by default None
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
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        optimizer_class : tf.keras.optimizers, optional
            Optional explicit request of optimizer. This should be a class
            that will be instantated in the TfModel.compile_model() method
            The default is the Adam optimizer
        epochs : int, optional
            Number of epochs to train the model, by default 100
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from features and labels.
        validation_split : float, optional
            Fraction of the training data to be used as validation data,
            by default 0.2
        early_stop : bool
            Flag to stop training when it stops improving
        stop_kwargs : dict | None
            kwargs for tf.keras.callbacks.EarlyStopping() if early_stop
            default is: {'monitor': 'val_loss', 'patience': 10}
        save_path : str
            Directory path to save model to. The tensorflow model will be
            saved to the directory while the framework parameters will be
            saved in json.
        compile_kwargs : dict
            kwargs for tensorflow.keras.models.compile
        parse_kwargs : dict
            kwargs for cls.parse_features
        fit_kwargs : dict
            kwargs for tensorflow.keras.models.fit

        Returns
        -------
        model : TfModel
            Initialized and trained TfModel obj
        """
        if compile_kwargs is None:
            compile_kwargs = {}

        _, feature_names = cls._parse_data_names(features, fallback_prefix='F')
        _, label_names = cls._parse_data_names(labels, fallback_prefix='L')

        model = cls.build(feature_names, label_names,
                          normalize=normalize,
                          one_hot_categories=one_hot_categories,
                          hidden_layers=hidden_layers,
                          input_layer=input_layer,
                          output_layer=output_layer,
                          learning_rate=learning_rate,
                          loss=loss,
                          metrics=metrics,
                          optimizer_class=optimizer_class,
                          **compile_kwargs)

        model.train_model(features, labels,
                          epochs=epochs,
                          shuffle=shuffle,
                          validation_split=validation_split,
                          early_stop=early_stop,
                          stop_kwargs=stop_kwargs,
                          parse_kwargs=parse_kwargs,
                          fit_kwargs=fit_kwargs)

        if save_path is not None:
            model.save_model(save_path)

        return model
