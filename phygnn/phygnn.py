# -*- coding: utf-8 -*-
"""
Physics Guided Neural Network
"""

import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers

from phygnn.base import CustomNetwork
from phygnn.utilities.loss_metrics import METRICS

logger = logging.getLogger(__name__)


class PhysicsGuidedNeuralNetwork(CustomNetwork):
    """Simple Deep Neural Network with custom physical loss function.

    Note that the phygnn model requires TensorFlow 2.x
    """

    def __init__(
        self,
        p_fun,
        loss_weights=(0.5, 0.5),
        n_features=1,
        n_labels=1,
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
        feature_names=None,
        output_names=None,
        name=None,
        compute_dtype='float32',
    ):
        """
        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This fun must take (phygnn, y_true, y_predicted, p, **p_kwargs)
            as arguments with datatypes (PhysicsGuidedNeuralNetwork, tf.Tensor,
            np.ndarray, np.ndarray). The function must return a tf.Tensor
            object with a single numeric loss value (output.ndim == 0).
        loss_weights : tuple, optional
            Loss weights for the neural network y_true vs. y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
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
            or a method in tensorflow.keras.losses that takes
            (y_true, y_predicted) as arguments.
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
        compute_dtype : str, optional
            Per-model compute dtype used to construct TensorFlow layers.
            Defaults to ``'float32'``.
        """

        super().__init__(
            n_features=n_features,
            n_labels=n_labels,
            hidden_layers=hidden_layers,
            input_layer=input_layer,
            output_layer=output_layer,
            layers_obj=layers_obj,
            feature_names=feature_names,
            output_names=output_names,
            compute_dtype=compute_dtype,
        )

        self._p_fun = p_fun if p_fun is not None else self.p_fun_dummy
        self._loss_weights = None
        self._metric = metric
        self._optimizer = None
        self._history = history
        self._learning_rate = learning_rate
        self.kernel_reg_rate = kernel_reg_rate
        self.kernel_reg_power = kernel_reg_power
        self.bias_reg_rate = bias_reg_rate
        self.bias_reg_power = bias_reg_power
        self.name = name if isinstance(name, str) else 'phygnn'

        self.set_loss_weights(loss_weights)

        if self._metric.lower() in METRICS:
            self._metric_fun = METRICS[self._metric.lower()]
        else:
            try:
                self._metric_fun = getattr(tf.keras.losses, self._metric)
            except Exception as e:
                msg = (
                    'Could not recognize error metric "{}". The following '
                    'error metrics are available: {}'.format(
                        self._metric, list(METRICS.keys())
                    )
                )
                logger.error(msg)
                raise KeyError(msg) from e

        self._optimizer = optimizer
        if isinstance(optimizer, dict):
            self._optimizer = optimizers.deserialize({
                'class_name': optimizer['name'],
                'config': optimizer,
            })
        elif optimizer is None:
            self._optimizer = optimizers.Adam(learning_rate=learning_rate)

    @staticmethod
    def p_fun_dummy(model, y_true, y_predicted, p):  # noqa : ARG004
        """Example dummy function for p loss calculation.

        This dummy function does not do a real physics calculation, it just
        shows the required p_fun interface and calculates a normal MAE loss
        based on y_predicted and y_true.

        Parameters
        ----------
        model : PhysicsGuidedNeuralNetwork
            Instance of the phygnn model at the current point in training.
        y_true : np.ndarray
            Known y values that were given to the phygnn.fit() method.
        y_predicted : tf.Tensor
            Predicted y values in a >=2D tensor based on x values in the
            current batch.
        p : np.ndarray
            Supplemental physical feature data that can be used to calculate a
            y_physical value to compare against y_predicted. The rows in this
            array have been carried through the batching process alongside
            y_true and the x-features used to create y_predicted and so can be
            used 1-to-1 with the rows in y_predicted and y_true.

        Returns
        -------
        p_loss : tf.Tensor
            A 0D tensor physical loss value.
        """
        # pylint: disable=W0613
        return tf.math.reduce_mean(tf.math.abs(y_predicted - y_true))

    def preflight_data(self, x, y, p):
        """Run simple preflight checks on data shapes and data types.

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
        y : np.ndarray | pd.DataFrame
            Known output data in a >=2D array or DataFrame.
            Same dimension rules as x.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in >=2D
            array or DataFrame. Same dimension rules as x.

        Returns
        -------
        x : np.ndarray
            Feature data
        y : np.ndarray
            Known output data
        p : np.ndarray
            Supplemental feature data
        """

        self._check_shapes(x, y)
        self._check_shapes(x, p)

        if self._n_features is None:
            self._n_features = x.shape[-1]
        if self._n_labels is None:
            self._n_labels = y.shape[-1]

        x_msg = 'x data has {} features but expected {}'.format(
            x.shape[-1], self._n_features
        )
        y_msg = 'y data has {} features but expected {}'.format(
            y.shape[-1], self._n_labels
        )
        assert x.shape[-1] == self._n_features, x_msg
        assert y.shape[-1] == self._n_labels, y_msg

        x = self.preflight_features(x)

        if isinstance(y, pd.DataFrame):
            y_cols = y.columns.values.tolist()
            if self.output_names is None:
                self.output_names = y_cols
            else:
                msg = (
                    'Cannot work with input y columns: {}, previously set '
                    'output names are: {}'.format(y_cols, self.output_names)
                )
                assert self.output_names == y_cols, msg
            y = y.values

        if isinstance(p, pd.DataFrame):
            p = p.values

        return x, y, p

    @property
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        return self._history

    @property
    def kernel_reg_term(self):
        """Get the regularization term for the kernel regularization without
        the regularization rate applied."""
        loss_k_reg = [tf.math.abs(x) for x in self.kernel_weights]
        loss_k_reg = [
            tf.math.pow(x, self.kernel_reg_power) for x in loss_k_reg
        ]
        loss_k_reg = tf.math.reduce_sum([
            tf.math.reduce_sum(x) for x in loss_k_reg
        ])

        return loss_k_reg

    @property
    def bias_reg_term(self):
        """Get the regularization term for the bias regularization without
        the regularization rate applied."""
        loss_b_reg = [tf.math.abs(x) for x in self.bias_weights]
        loss_b_reg = [tf.math.pow(x, self.bias_reg_power) for x in loss_b_reg]
        loss_b_reg = tf.math.reduce_sum([
            tf.math.reduce_sum(x) for x in loss_b_reg
        ])

        return loss_b_reg

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """

        model_params = super().model_params
        model_params.update({
            'p_fun': self._p_fun,
            'loss_weights': self._loss_weights,
            'metric': self._metric,
            'optimizer': self._optimizer.get_config(),
            'learning_rate': self._learning_rate,
            'layers_obj': self.layers_obj,
            'history': self.history,
            'kernel_reg_rate': self.kernel_reg_rate,
            'kernel_reg_power': self.kernel_reg_power,
            'bias_reg_rate': self.bias_reg_rate,
            'bias_reg_power': self.bias_reg_power,
        })

        return model_params

    def preflight_p_fun(self, x, y_true, p, p_kwargs):
        """Run a pre-flight check making sure the p_fun is differentiable."""

        if p_kwargs is None:
            p_kwargs = {}

        with tf.GradientTape() as tape:
            y_predicted = self.predict(x, to_numpy=False)
            p_loss = self._p_fun(self, y_true, y_predicted, p, **p_kwargs)
            grad = tape.gradient(p_loss, self.weights)

            if not tf.is_tensor(p_loss):
                emsg = 'Loss output from p_fun() must be a tensor!'
                logger.error(emsg)
                raise TypeError(emsg)

            if p_loss.ndim > 1:
                emsg = (
                    'Loss output from p_fun() should be a scalar tensor '
                    'but received a tensor with shape {}'.format(p_loss.shape)
                )
                logger.error(emsg)
                raise ValueError(emsg)

            assert isinstance(grad, list)
            if grad[0] is None:
                emsg = (
                    'The input p_fun was not differentiable! '
                    'Please use only tensor math in the p_fun.'
                )
                logger.error(emsg)
                raise RuntimeError(emsg)

        logger.debug('p_fun passed preflight check.')

    def reset_history(self):
        """Erase previous training history without resetting trained weights"""
        self._history = None

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
        assert np.sum(loss_weights) > 0, 'Sum of loss_weights must be > 0!'
        assert len(loss_weights) == 2, 'loss_weights can only have two values!'
        self._loss_weights = loss_weights

    def calc_loss(self, y_true, y_predicted, p, p_kwargs):
        """Calculate the loss function by comparing y_true to model-predicted y

        Parameters
        ----------
        y_true : np.ndarray
            Known output data in a >=2D array.
        y_predicted : tf.Tensor
            Model-predicted output data in a >=2D tensor.
        p : np.ndarray
            Supplemental feature data for the physics loss function in >=2D
            array
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.

        Returns
        -------
        loss : tf.tensor
            Sum of the NN loss function comparing the y_predicted against
            y_true and the physical loss function (self._p_fun) with
            respective weights applied.
        nn_loss : tf.tensor
            Standard NN training loss comparing y to y_predicted.
        p_loss : tf.tensor
            Physics loss from p_fun.
        """

        if p_kwargs is None:
            p_kwargs = {}

        loss = tf.zeros((), dtype=tf.float32)
        nn_loss = tf.zeros((), dtype=tf.float32)
        p_loss = tf.zeros((), dtype=tf.float32)

        if self._loss_weights[0] != 0:
            nn_loss = tf.cast(
                self._metric_fun(y_true, y_predicted),
                tf.float32,
            )
            msg = 'Bad shape from nn_loss fun! Must be 0D but received: {}'
            msg = msg.format(nn_loss)
            assert nn_loss.ndim == 0, msg
            loss += self._loss_weights[0] * nn_loss

        if self._loss_weights[1] != 0:
            p_loss = tf.cast(
                self._p_fun(self, y_true, y_predicted, p, **p_kwargs),
                tf.float32,
            )
            msg = 'Bad shape from p_loss fun! Must be 0D but received: {}'
            msg = msg.format(p_loss)
            assert p_loss.ndim == 0, msg
            loss += self._loss_weights[1] * p_loss

        logger.debug(
            'NN Loss: {:.2e}, P Loss: {:.2e}, Total Loss: {:.2e}'.format(
                nn_loss, p_loss, loss
            )
        )

        if self.kernel_reg_rate != 0:
            loss_kernel_reg = tf.cast(self.kernel_reg_term, tf.float32)
            loss_kernel_reg *= self.kernel_reg_rate
            loss += loss_kernel_reg
            logger.debug(
                'Kernel regularization loss: {:.2e}, '
                'Total Loss: {:.2e}'.format(loss_kernel_reg, loss)
            )

        if self.bias_reg_rate != 0:
            loss_bias_reg = tf.cast(self.bias_reg_term, tf.float32)
            loss_bias_reg *= self.bias_reg_rate
            loss += loss_bias_reg
            logger.debug(
                'Bias regularization loss: {:.2e}, Total Loss: {:.2e}'.format(
                    loss_bias_reg, loss
                )
            )

        if tf.math.is_nan(loss):
            msg = 'phygnn calculated a NaN loss value!'
            logger.error(msg)
            raise ArithmeticError(msg)

        return loss, nn_loss, p_loss

    def _get_grad(self, x, y_true, p, p_kwargs):
        """Get the gradient based on a mini-batch of x and y_true data."""
        with tf.GradientTape() as tape:
            y_predicted = self.predict(x, to_numpy=False, training=True)
            loss, nn_loss, p_loss = self.calc_loss(
                y_true, y_predicted, p, p_kwargs
            )
            grad = tape.gradient(loss, self.weights)

        return grad, loss, nn_loss, p_loss

    def run_gradient_descent(self, x, y_true, p, p_kwargs):
        """Run gradient descent for one mini-batch of (x, y_true)
        and adjust NN weights."""
        grad, loss, nn_loss, p_loss = self._get_grad(x, y_true, p, p_kwargs)
        self._optimizer.apply_gradients(zip(grad, self.weights))
        return loss, nn_loss, p_loss

    def fit(
        self,
        x,
        y,
        p,
        n_batch=16,
        batch_size=None,
        n_epoch=10,
        shuffle=True,
        validation_split=0.2,
        p_kwargs=None,
        run_preflight=True,
        return_diagnostics=False,
    ):
        """Fit the neural network to data from x and y.

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
        y : np.ndarray | pd.DataFrame
            Known output data in a >=2D array or DataFrame.
            Same dimension rules as x.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in >=2D
            array or DataFrame. Same dimension rules as x.
        n_batch : int | None
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
            from x, y, and p.
        validation_split : float
            Fraction of x, y, and p to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.

        Returns
        -------
        diagnostics : dict
            Namespace of training parameters that can be used for diagnostics.
        """

        x, y, p = self.preflight_data(x, y, p)

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(
                columns=[
                    'elapsed_time',
                    'training_loss',
                    'training_nn_loss',
                    'training_p_loss',
                    'validation_loss',
                    'validation_nn_loss',
                    'validation_p_loss',
                ]
            )
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        val_splits = self.get_val_split(
            x, y, p, shuffle=shuffle, validation_split=validation_split
        )
        x, x_val = val_splits[0]
        y, y_val = val_splits[1]
        p, p_val = val_splits[2]

        if self._loss_weights[1] > 0 and run_preflight:
            self.preflight_p_fun(x_val, y_val, p_val, p_kwargs)

        t0 = time.time()
        for epoch in epochs:
            t_batch_iter = self.make_batches(
                x,
                y,
                p,
                n_batch=n_batch,
                batch_size=batch_size,
                shuffle=shuffle,
            )

            v_batch_iter = self.make_batches(
                x_val,
                y_val,
                p_val,
                n_batch=n_batch,
                batch_size=batch_size,
                shuffle=False,
            )

            e_tr_loss = []
            e_tr_nn_loss = []
            e_tr_p_loss = []

            e_val_loss = []
            e_val_nn_loss = []
            e_val_p_loss = []

            for b, (x_batch, y_batch, p_batch) in enumerate(t_batch_iter):
                b_out = self.run_gradient_descent(
                    x_batch, y_batch, p_batch, p_kwargs
                )
                b_tr_loss, b_tr_nn_loss, b_tr_p_loss = b_out
                e_tr_loss.append(b_tr_loss.numpy())
                e_tr_nn_loss.append(b_tr_nn_loss.numpy())
                e_tr_p_loss.append(b_tr_p_loss.numpy())
                logger.debug(
                    'Epoch {} batch {} train loss: {:.2e} for "{}"'.format(
                        epoch, b, b_tr_loss, self.name
                    )
                )

            for x_batch, y_batch, p_batch in v_batch_iter:
                y_val_pred = self.predict(x_batch, to_numpy=False)
                out = self.calc_loss(y_batch, y_val_pred, p_batch, p_kwargs)
                b_val_loss, b_val_nn_loss, b_val_p_loss = out
                e_val_loss.append(b_val_loss.numpy())
                e_val_nn_loss.append(b_val_nn_loss.numpy())
                e_val_p_loss.append(b_val_p_loss.numpy())

            e_tr_loss = np.mean(e_tr_loss)
            e_tr_nn_loss = np.mean(e_tr_nn_loss)
            e_tr_p_loss = np.mean(e_tr_p_loss)
            e_val_loss = np.mean(e_val_loss)
            e_val_nn_loss = np.mean(e_val_nn_loss)
            e_val_p_loss = np.mean(e_val_p_loss)

            logger.info(
                'Epoch {} train loss: {:.2e} val loss: {:.2e} for "{}"'.format(
                    epoch, e_tr_loss, e_val_loss, self.name
                )
            )

            self._history.at[epoch, 'elapsed_time'] = time.time() - t0
            self._history.at[epoch, 'training_loss'] = e_tr_loss
            self._history.at[epoch, 'training_nn_loss'] = e_tr_nn_loss
            self._history.at[epoch, 'training_p_loss'] = e_tr_p_loss
            self._history.at[epoch, 'validation_loss'] = e_val_loss
            self._history.at[epoch, 'validation_nn_loss'] = e_val_nn_loss
            self._history.at[epoch, 'validation_p_loss'] = e_val_p_loss

        diagnostics = {
            'x': x,
            'y': y,
            'p': p,
            'x_val': x_val,
            'y_val': y_val,
            'p_val': p_val,
            'history': self.history,
        }

        if return_diagnostics:
            return diagnostics
        return None
