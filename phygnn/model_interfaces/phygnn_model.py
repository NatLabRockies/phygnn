# -*- coding: utf-8 -*-
"""
TensorFlow Model
"""
import json
import logging
import os
import pprint

import numpy as np

from phygnn.model_interfaces.base_model import ModelBase
from phygnn.phygnn import PhysicsGuidedNeuralNetwork
from phygnn.utilities.pre_processing import PreProcess

logger = logging.getLogger(__name__)


class PhygnnModel(ModelBase):
    """
    Phygnn Model interface
    """

    # Underlying model interface class. Used for loading models from disk
    MODEL_CLASS = PhysicsGuidedNeuralNetwork

    def __init__(self, model, feature_names=None, label_names=None,
                 norm_params=None, normalize=(True, False),
                 one_hot_categories=None):
        """
        Parameters
        ----------
        model : PhysicsGuidedNeuralNetwork
            PhysicsGuidedNeuralNetwork Model instance
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
        return self.model.weights

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
        return self.model.kernel_weights

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
        return self.bias_weights

    @property
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        return self.model.history

    @property
    def version_record(self):
        """A record of important versions that this model was built with.

        Returns
        -------
        dict
        """
        return self.model.version_record

    def train_model(self, features, labels, p,
                    n_batch=16, batch_size=None, n_epoch=10,
                    shuffle=True, validation_split=0.2, run_preflight=True,
                    return_diagnostics=False, p_kwargs=None,
                    parse_kwargs=None):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        features : np.ndarray | pd.DataFrame
            Feature data in a >=2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm. A 2D input should have the shape:
            (n_observations, n_features). A 3D input should have the shape:
            (n_observations, n_timesteps, n_features). 4D inputs have not been
            tested and should be used with caution.
        labels : np.ndarray | pd.DataFrame
            Known output data in a 2D array or DataFrame.
            Same dimension rules as features.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in 2D array
            or DataFrame. Same dimension rules as features.
        n_batch : int
            Number of times to update the NN weights per epoch (number of
            mini-batches). The training data will be split into this many
            mini-batches and the NN will train on each mini-batch, update
            weights, then move onto the next mini-batch.
        batch_size : int | None
            Number of training samples per batch. This input is redundant to
            n_batch and will not be used if n_batch is not None.
        n_epoch : int
            Number of times to iterate on the training data.
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from features, labels, and p.
        validation_split : float
            Fraction of features and labels to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function p_fun.
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.
        parse_kwargs : dict
            kwargs for cls.parse_features
        norm_labels : bool, optional
            Flag to normalize label, by default True

        Returns
        -------
        diagnostics : dict, optional
            Namespace of training parameters that can be used for diagnostics.
        """
        if parse_kwargs is None:
            parse_kwargs = {}

        if (isinstance(features, np.ndarray)
                and features.shape[-1] == self.feature_dims):
            parse_kwargs['names'] = self.feature_names

        label_names = None
        if (isinstance(labels, np.ndarray)
                and labels.shape[-1] == self.label_dims):
            label_names = self.label_names

        x = self.parse_features(features, **parse_kwargs)
        y = self.parse_labels(labels, names=label_names)

        diagnostics = self.model.fit(x, y, p,
                                     n_batch=n_batch,
                                     batch_size=batch_size,
                                     n_epoch=n_epoch,
                                     shuffle=shuffle,
                                     validation_split=validation_split,
                                     p_kwargs=p_kwargs,
                                     run_preflight=run_preflight,
                                     return_diagnostics=return_diagnostics)

        return diagnostics

    def save_model(self, path):
        """
        Save phygnn model to path.

        Parameters
        ----------
        path : str
            Target model save path. Can be a target .json, .pkl, or a directory
            that will be created+populated with a json parameters file and a
            `.pkl` file for the underlying PhysicsGuidedNeuralNetwork.
        """

        json_path = os.path.abspath(path)
        if json_path.endswith(('.json', '.pkl')):
            dir_path = os.path.dirname(json_path)
            if json_path.endswith('.pkl'):
                json_path = json_path.replace('.pkl', '.json')
        else:
            dir_path = json_path
            fn = os.path.basename(json_path) + '.json'
            json_path = os.path.join(dir_path, fn)

        pkl_path = json_path.replace('.json', '.pkl')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model_params = {'feature_names': self.feature_names,
                        'label_names': self.label_names,
                        'norm_params': self.normalization_parameters,
                        'normalize': (self.normalize_features,
                                      self.normalize_labels),
                        'version_record': self.version_record,
                        'one_hot_categories': self.one_hot_categories,
                        }

        model_params = self.dict_json_convert(model_params)
        with open(json_path, 'w') as f:
            json.dump(model_params, f, indent=2, sort_keys=True)

        self.model.save(pkl_path)

    def set_loss_weights(self, loss_weights):
        """Set new loss weights

        Parameters
        ----------
        loss_weights : tuple
            Loss weights for the neural network y_true vs y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
        """
        self.model._loss_weights = loss_weights

    @classmethod
    def build(cls, p_fun, feature_names, label_names,
              normalize=(True, False),
              one_hot_categories=None,
              loss_weights=(0.5, 0.5),
              hidden_layers=None,
              input_layer=None,
              output_layer=None,
              layers_obj=None,
              metric='mae',
              optimizer=None,
              learning_rate=0.01,
              history=None,
              kernel_reg_rate=0.0,
              kernel_reg_power=1,
              bias_reg_rate=0.0,
              bias_reg_power=1,
              name=None):
        """
        Build phygnn model from given features, layers and kwargs

        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This fun must take (phygnn, y_true, y_predicted, p, **p_kwargs)
            as arguments with datatypes (PhysicsGuidedNeuralNetwork, tf.Tensor,
            np.ndarray, np.ndarray). The function must return a tf.Tensor
            object with a single numeric loss value (output.ndim == 0).
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
        loss_weights : tuple, optional
            Loss weights for the neural network y_true vs y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
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
        layers_obj : None | phygnn.utilities.tf_layers.Layers
            Optional initialized Layers object to set as the model layers
            including pre-set weights. This option will override the
            hidden_layers, input_layer, and output_layer arguments.
        metric : str, optional
            Loss metric option for the NN loss function (not the physical
            loss function). Must be a valid key in phygnn.loss_metrics.METRICS
        optimizer : tensorflow.keras.optimizers | dict | None
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        history : None | pd.DataFrame, optional
            Learning history if continuing a training session.
        kernel_reg_rate : float, optional
            Kernel regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer weights and should reduce
            model complexity. Setting this to 0.0 will disable kernel
            regularization.
        kernel_reg_power : int, optional
            Kernel regularization power. kernel_reg_power=1 is L1
            regularization (lasso regression), and kernel_reg_power=2 is L2
            regularization (ridge regression).
        bias_reg_rate : float, optional
            Bias regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer biases and should reduce
            model complexity. Setting this to 0.0 will disable bias
            regularization.
        bias_reg_power : int, optional
            Bias regularization power. bias_reg_power=1 is L1
            regularization (lasso regression), and bias_reg_power=2 is L2
            regularization (ridge regression).
        name : None | str
            Optional model name for debugging.

        Returns
        -------
        model : PhygnnModel
            Initialized PhygnnModel instance
        """
        if isinstance(label_names, str):
            label_names = [label_names]

        if one_hot_categories is not None:
            check_names = feature_names + label_names
            PreProcess.check_one_hot_categories(one_hot_categories,
                                                feature_names=check_names)
            feature_names = cls.make_one_hot_feature_names(feature_names,
                                                           one_hot_categories)

        n_features = None if feature_names is None else len(feature_names)
        n_labels = None if label_names is None else len(label_names)

        model = PhysicsGuidedNeuralNetwork(p_fun,
                                           loss_weights=loss_weights,
                                           n_features=n_features,
                                           n_labels=n_labels,
                                           hidden_layers=hidden_layers,
                                           input_layer=input_layer,
                                           output_layer=output_layer,
                                           layers_obj=layers_obj,
                                           metric=metric,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           history=history,
                                           kernel_reg_rate=kernel_reg_rate,
                                           kernel_reg_power=kernel_reg_power,
                                           bias_reg_rate=bias_reg_rate,
                                           bias_reg_power=bias_reg_power,
                                           feature_names=feature_names,
                                           output_names=label_names,
                                           name=name)

        model = cls(model, feature_names=feature_names,
                    label_names=label_names, normalize=normalize,
                    one_hot_categories=one_hot_categories)

        return model

    @classmethod
    def build_trained(cls, p_fun, features, labels, p,
                      normalize=(True, False),
                      one_hot_categories=None,
                      loss_weights=(0.5, 0.5),
                      hidden_layers=None,
                      input_layer=None,
                      output_layer=None,
                      layers_obj=None,
                      metric='mae',
                      optimizer=None,
                      learning_rate=0.01,
                      history=None,
                      kernel_reg_rate=0.0,
                      kernel_reg_power=1,
                      bias_reg_rate=0.0,
                      bias_reg_power=1,
                      n_batch=16,
                      batch_size=None,
                      n_epoch=10,
                      shuffle=True,
                      validation_split=0.2,
                      run_preflight=True,
                      return_diagnostics=False,
                      p_kwargs=None,
                      parse_kwargs=None,
                      save_path=None,
                      name=None):
        """
        Build phygnn model from given features, layers and
        kwargs and then train with given labels and kwargs

        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This fun must take (phygnn, y_true, y_predicted, p, **p_kwargs)
            as arguments with datatypes (PhysicsGuidedNeuralNetwork, tf.Tensor,
            np.ndarray, np.ndarray). The function must return a tf.Tensor
            object with a single numeric loss value (output.ndim == 0).
        features : np.ndarray | pd.DataFrame
            Feature data in a >=2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm. A 2D input should have the shape:
            (n_observations, n_features). A 3D input should have the shape:
            (n_observations, n_timesteps, n_features). 4D inputs have not been
            tested and should be used with caution.
        labels : np.ndarray | pd.DataFrame
            Known output data in a 2D array or DataFrame.
            Same dimension rules as features.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in 2D array
            or DataFrame. Same dimension rules as features.
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
        loss_weights : tuple, optional
            Loss weights for the neural network y_true vs y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
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
            hidden_layers specifying a dense / conv / lstm layer.  Will
            default to a keras InputLayer with input shape = n_features.
            Can be False if the input layer will be included in the
            hidden_layers input.
        output_layer : None } bool | list | dict
            Output layer specification. Can be a list/dict similar to
            hidden_layers input specifying a dense layer with activation.
            For example, for a classfication problem with a single output,
            output_layer should be [{'units': 1}, {'activation': 'sigmoid'}].
            This defaults to a single dense layer with no activation
            (best for regression problems).  Can be False if the output layer
            will be included in the hidden_layers input.
        layers_obj : None | phygnn.utilities.tf_layers.Layers
            Optional initialized Layers object to set as the model layers
            including pre-set weights. This option will override the
            hidden_layers, input_layer, and output_layer arguments.
        metric : str, optional
            Loss metric option for the NN loss function (not the physical
            loss function). Must be a valid key in phygnn.loss_metrics.METRICS
        optimizer : tensorflow.keras.optimizers | dict | None
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        history : None | pd.DataFrame, optional
            Learning history if continuing a training session.
        kernel_reg_rate : float, optional
            Kernel regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer weights and should reduce
            model complexity. Setting this to 0.0 will disable kernel
            regularization.
        kernel_reg_power : int, optional
            Kernel regularization power. kernel_reg_power=1 is L1
            regularization (lasso regression), and kernel_reg_power=2 is L2
            regularization (ridge regression).
        bias_reg_rate : float, optional
            Bias regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer biases and should reduce
            model complexity. Setting this to 0.0 will disable bias
            regularization.
        bias_reg_power : int, optional
            Bias regularization power. bias_reg_power=1 is L1
            regularization (lasso regression), and bias_reg_power=2 is L2
            regularization (ridge regression).
        n_batch : int
            Number of times to update the NN weights per epoch (number of
            mini-batches). The training data will be split into this many
            mini-batches and the NN will train on each mini-batch, update
            weights, then move onto the next mini-batch.
        batch_size : int | None
            Number of training samples per batch. This input is redundant to
            n_batch and will not be used if n_batch is not None.
        n_epoch : int
            Number of times to iterate on the training data.
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from features and labels.
        validation_split : float
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.
            Fraction of features and labels to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function p_fun.
        parse_kwargs : dict
            kwargs for cls.parse_features
        norm_labels : bool, optional
            Flag to normalize label, by default True
        save_path : str, optional
            Directory path to save model to. The tensorflow model will be
            saved to the directory while the framework parameters will be
            saved in json, by default None
        name : None | str
            Optional model name for debugging.

        Returns
        -------
        model : TfModel
            Initialized and trained TfModel obj
        diagnostics : dict, optional
            Namespace of training parameters that can be used for diagnostics.
        """

        _, feature_names = cls._parse_data_names(features, fallback_prefix='F')
        _, label_names = cls._parse_data_names(labels, fallback_prefix='L')

        model = cls.build(p_fun, feature_names, label_names,
                          normalize=normalize,
                          one_hot_categories=one_hot_categories,
                          loss_weights=loss_weights,
                          hidden_layers=hidden_layers,
                          input_layer=input_layer,
                          output_layer=output_layer,
                          layers_obj=layers_obj,
                          metric=metric,
                          optimizer=optimizer,
                          learning_rate=learning_rate,
                          history=history,
                          kernel_reg_rate=kernel_reg_rate,
                          kernel_reg_power=kernel_reg_power,
                          bias_reg_rate=bias_reg_rate,
                          bias_reg_power=bias_reg_power,
                          name=name)

        diagnostics = model.train_model(features, labels, p,
                                        n_batch=n_batch,
                                        batch_size=batch_size,
                                        n_epoch=n_epoch,
                                        shuffle=shuffle,
                                        validation_split=validation_split,
                                        run_preflight=run_preflight,
                                        return_diagnostics=return_diagnostics,
                                        p_kwargs=p_kwargs,
                                        parse_kwargs=parse_kwargs)

        if save_path is not None:
            model.save_model(save_path)

        if diagnostics:
            return model, diagnostics
        return model

    @classmethod
    def load(cls, path):
        """
        Load model from model path.

        Parameters
        ----------
        path : str
            Directory path for PhygnnModel to load model from. There should be
            a saved model directory with a json file for the PhygnnModel
            framework and a `.pkl` file for the underlying
            PhysicsGuidedNeuralNetwork.

        Returns
        -------
        model : PhygnnModel
            Loaded PhygnnModel from disk.
        """
        path = os.path.abspath(path)
        if not path.endswith(('.json', '.pkl')):
            pkl_path = os.path.join(path, os.path.basename(path) + '.pkl')
        elif path.endswith('.json'):
            pkl_path = path.replace('.json', '.pkl')
        elif path.endswith('.pkl'):
            pkl_path = path

        if not os.path.exists(pkl_path):
            e = ('{} does not exist'.format(pkl_path))
            logger.error(e)
            raise OSError(e)

        loaded = cls.MODEL_CLASS.load(pkl_path)

        json_path = pkl_path.replace('.pkl', '.json')
        if not os.path.exists(json_path):
            e = ('{} does not exist'.format(json_path))
            logger.error(e)
            raise OSError(e)

        with open(json_path) as f:
            model_params = json.load(f)

        if 'version_record' in model_params:
            version_record = model_params.pop('version_record')
            logger.info('Loading model from disk that was created with the '
                        'following package versions: \n{}'
                        .format(pprint.pformat(version_record, indent=4)))

        model = cls(loaded, **model_params)

        return model
