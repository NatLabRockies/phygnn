# -*- coding: utf-8 -*-
"""
Base Model Interface
"""
import copy
import logging
import pprint
import random
from abc import ABC
from warnings import warn

import numpy as np
import pandas as pd
import tensorflow as tf

from phygnn.utilities import TF2, VERSION_RECORD
from phygnn.utilities.pre_processing import PreProcess

logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """
    Base Model Interface
    """

    def __init__(self, model, feature_names=None, label_names=None,
                 norm_params=None, normalize=(True, False),
                 one_hot_categories=None):
        """
        Parameters
        ----------
        model : OBJ
            Initialized model object
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
        self._model = model

        if isinstance(feature_names, str):
            feature_names = [feature_names]
        elif isinstance(feature_names, (np.ndarray, pd.Index)):
            feature_names = feature_names.tolist()

        self._feature_names = feature_names

        if isinstance(label_names, str):
            label_names = [label_names]
        elif isinstance(label_names, (np.ndarray, pd.Index)):
            label_names = label_names.tolist()

        self._label_names = label_names
        if norm_params is None:
            norm_params = {}

        self._norm_params = norm_params
        self._normalize = self._parse_normalize(normalize)
        if one_hot_categories is not None:
            PreProcess.check_one_hot_categories(one_hot_categories)

        self._one_hot_categories = one_hot_categories

        self._version_record = VERSION_RECORD
        logger.info('Active python environment versions: \n{}'
                    .format(pprint.pformat(self._version_record, indent=4)))

    def __repr__(self):
        msg = "{}:\n{}".format(self.__class__.__name__, self.model_summary)

        return msg

    def __getitem__(self, features):
        """
        Use model to predict label from given features

        Parameters
        ----------
        features : pandas.DataFrame
            features to predict from

        Returns
        -------
        pandas.DataFrame
            label prediction
        """
        return self.predict(features)

    @property
    def version_record(self):
        """A record of important versions that this model was built with.

        Returns
        -------
        dict
        """
        return self._version_record

    @property
    def model_summary(self):
        """
        Tensorflow model summary

        Returns
        -------
        str
        """
        try:
            summary = self._model.summary()
        except ValueError:
            summary = None

        return summary

    @property
    def normalize_features(self):
        """
        Flag to normalize features

        Returns
        -------
        bool
        """
        return self._normalize[0]

    @property
    def feature_names(self):
        """
        List of the feature variable names.

        Returns
        -------
        list
        """
        return self._feature_names

    @property
    def feature_dims(self):
        """
        Number of features

        Returns
        -------
        int
        """
        n_features = (len(self.feature_names)
                      if self.feature_names is not None else None)

        return n_features

    @property
    def normalize_labels(self):
        """
        Flag to normalize labels

        Returns
        -------
        bool
        """
        return self._normalize[1]

    @property
    def label_names(self):
        """
        Label variable names

        Returns
        -------
        list
        """
        return self._label_names

    @property
    def label_dims(self):
        """
        Number of labels

        Returns
        -------
        int
        """
        n_labels = (len(self.label_names)
                    if self.label_names is not None else None)

        return n_labels

    @property
    def normalization_parameters(self):
        """
        Features and label (un)normalization parameters

        Returns
        -------
        dict
        """
        return self._norm_params

    @property
    def means(self):
        """
        Mapping feature/label names to the mean values for
        (un)normalization

        Returns
        -------
        dict
        """
        means = {k: v['mean'] for k, v in self._norm_params.items()}

        return means

    @property
    def stdevs(self):
        """
        Mapping feature/label names to the stdev values for
        (un)normalization

        Returns
        -------
        dict
        """
        stdevs = {k: v['stdev'] for k, v in self._norm_params.items()}

        return stdevs

    @property
    def model(self):
        """
        Trained model

        Returns
        -------
        tensorflow.keras.models
        """
        return self._model

    @property
    def feature_means(self):
        """
        Feature means, used for (un)normalization

        Returns
        -------
        list
        """
        means = None
        if self._feature_names is not None:
            means = []
            for f in self._feature_names:
                means.append(self.get_mean(f))

        return means

    @property
    def feature_stdevs(self):
        """
        Feature stdevs, used for (un)normalization

        Returns
        -------
        list
        """
        stdevs = None
        if self._feature_names is not None:
            stdevs = []
            for f in self._feature_names:
                stdevs.append(self.get_stdev(f))

        return stdevs

    @property
    def label_means(self):
        """
        Label means, used for (un)normalization

        Returns
        -------
        list
        """
        means = None
        if self.label_names is not None:
            means = []
            for l_n in self.label_names:
                means.append(self.get_mean(l_n))

        return means

    @property
    def label_stdevs(self):
        """
        Label stdevs, used for (un)normalization

        Returns
        -------
        list
        """
        stdevs = None
        if self.label_names is not None:
            stdevs = []
            for l_n in self.label_names:
                stdevs.append(self.get_stdev(l_n))

        return stdevs

    @property
    def input_feature_names(self):
        """
        Input feature names

        Return
        ------
        list
        """
        if self._one_hot_categories is None:
            input_feature_names = self.feature_names
        else:
            input_feature_names = list(set(self.feature_names)
                                       - set(self.one_hot_feature_names)
                                       | set(self.one_hot_input_feature_names))

        return input_feature_names

    @property
    def one_hot_input_feature_names(self):
        """
        Input feature names to be one-hot encoded

        Return
        ------
        list
        """
        return list(self.one_hot_categories.keys())

    @property
    def one_hot_feature_names(self):
        """
        One-hot encoded feature names

        Return
        ------
        list
        """
        return [i for sub in self.one_hot_categories.values() for i in sub]

    @property
    def one_hot_categories(self):
        """
        Categories to use for one-hot encoding

        Returns
        -------
        dict
        """
        if self._one_hot_categories is None:
            return {}
        return self._one_hot_categories

    @staticmethod
    def _parse_normalize(normalize):
        """
        Parse normalize flag(s)

        Parameters
        ----------
        normalize : bool | tuple
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)

        Returns
        -------
        normalize : tuple
            Boolean flags (normalize_feature, normalize_label)
        """
        if isinstance(normalize, bool):
            normalize = (normalize, normalize)
        elif isinstance(normalize, (tuple, list)):
            if len(normalize) != 2:
                msg = ('Expecting only 2 values: '
                       '(normalize_feature, normalize_label), but {} values '
                       'were provided!: {}'.format(len(normalize), normalize))
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = ('"normalize" must be a boolean flag or a tuple/list: '
                   '(normalize_feature, normalize_label), but {} was '
                   'provided!: {}'
                   .format(type(normalize), normalize))
            logger.error(msg)
            raise TypeError(msg)

        return tuple(normalize)

    @staticmethod
    def dict_json_convert(inp):
        """Recursively convert numeric values in dict to work with json dump

        Parameters
        ----------
        inp : dict
            Dictionary to convert.

        Returns
        -------
        out : dict
            Copy of dict input with all nested numeric values converted to
            base python int or float and all arrays converted to lists.
        """

        if isinstance(inp, dict):
            out = {k: ModelBase.dict_json_convert(v) for k, v in inp.items()}
        elif isinstance(inp, (list, tuple)):
            out = [ModelBase.dict_json_convert(i) for i in inp]
        elif np.issubdtype(type(inp), np.floating):
            out = float(inp)
        elif np.issubdtype(type(inp), np.integer):
            out = int(inp)
        elif isinstance(inp, np.ndarray):
            out = inp.tolist()
        else:
            out = inp

        return out

    @staticmethod
    def seed(s=0):
        """
        Set the random seed for reproducible results.

        Parameters
        ----------
        s : int
            Random number generator seed
        """
        random.seed(s)
        np.random.seed(s)
        if TF2:
            tf.random.set_seed(s)
        else:
            tf.random.set_random_seed(s)

    @staticmethod
    def _parse_data_names(data, names=None, fallback_prefix=''):
        """
        Parse data array and names from input data

        Parameters
        ----------
        data : pandas.DataFrame | dict | ndarray
            Features/labels to parse
        names : list, optional
            List of data item names, by default None
        fallback_prefix : str
            If data is provided as a numpy array without associated labels, the
            fallback_prefix will be used to name the enumerated feature
            dimension

        Returns
        -------
        data : ndarray
            Data array
        names: list
            List of data item names
        """
        if isinstance(data, pd.DataFrame):
            names = data.columns.tolist()
            data = data.values
        elif isinstance(data, dict):
            names = list(data.keys())
            data = np.dstack(list(data.values()))[0]
        elif isinstance(data, np.ndarray) and names is None:
            names = [fallback_prefix + str(i) for i in range(data.shape[-1])]

        return data, names

    @staticmethod
    def _get_item_number(arr):
        """
        Get number of items in array (labels or features)

        Parameters
        ----------
        arr : ndarray
            1 or 2D array

        Returns
        -------
        n : int
            Number of items
        """
        n = 1 if len(arr.shape) == 1 else arr.shape[-1]

        return n

    @staticmethod
    def make_one_hot_feature_names(feature_names, one_hot_categories):
        """
        Update feature_names after one-hot encoding

        Parameters
        ----------
        feature_names : list
            Input feature names
        one_hot_categories : dict
            Features to one-hot encode using given categories

        Returns
        -------
        one_hot_feature_names : list
            Updated list of feature names with one_hot categories
        """
        one_hot_feature_names = feature_names.copy()
        for name, categories in one_hot_categories.items():
            if name in one_hot_feature_names:
                one_hot_feature_names.remove(name)

            for c in categories:
                if c not in one_hot_feature_names:
                    one_hot_feature_names.append(c)

        return one_hot_feature_names

    def get_norm_params(self, names):
        """
        Get means and stdevs for given feature/label names

        Parameters
        ----------
        names : list
            list of feature/label names to get normalization params for

        Returns
        -------
        means : list
            List of means to use for (un)normalization
        stdevs : list
            List of stdevs to use for (un)normalization
        """
        means = []
        stdevs = []
        for name in names:
            means.append(self.get_mean(name))
            stdevs.append(self.get_stdev(name))

        if None in means:
            means = None

        if None in stdevs:
            stdevs = None

        return means, stdevs

    def get_mean(self, name):
        """
        Get feature | label mean

        Parameters
        ----------
        name : str
            feature | label name

        Returns
        -------
        mean : float
            Mean value used for normalization
        """
        mean = self._norm_params.get(name, None)
        if mean is not None:
            mean = mean.get('mean', None)

        return mean

    def get_stdev(self, name):
        """
        Get feature | label stdev

        Parameters
        ----------
        name : str
            feature | label name

        Returns
        -------
        stdev : float
            Stdev value used for normalization
        """
        stdev = self._norm_params.get(name, None)
        if stdev is not None:
            stdev = stdev.get('stdev', None)

        return stdev

    def _normalize_dict(self, items):
        """
        Normalize given dictionary of items (features | labels)

        Parameters
        ----------
        items : dict
            mapping of names to vectors

        Returns
        -------
        norm_items : dict
            mapping of names to normalized-feature vectors
        """
        norm_items = {}
        for key, value in items.items():
            if key not in self.one_hot_feature_names:
                mean = self.get_mean(key)
                stdev = self.get_stdev(key)
                update = mean is None or stdev is None
                try:
                    value, mean, stdev = PreProcess.normalize(value,
                                                              mean=mean,
                                                              stdev=stdev)
                    if update:
                        norm_params = {key: {'mean': mean, 'stdev': stdev}}
                        self._norm_params.update(norm_params)
                except Exception as ex:
                    msg = "Could not normalize {}:\n{}".format(key, ex)
                    logger.warning(msg)
                    warn(msg)

            norm_items[key] = value

        return norm_items

    def _normalize_arr(self, arr, names):
        """
        Normalize array and save normalization parameters to given names

        Parameters
        ----------
        arr : ndarray
            Array of features/label to normalize
        names : list
            List of feature/label names

        Returns
        -------
        norm_arr : ndarray
            Normalized features/label
        """
        n_names = self._get_item_number(arr)
        if len(names) != n_names:
            msg = ("Number of item names ({}) does not match number of items "
                   "({})".format(len(names), arr.shape[1]))
            logger.error(msg)
            raise RuntimeError(msg)

        means, stdevs = self.get_norm_params(names)
        update = means is None or stdevs is None

        norm_arr, means, stdevs = PreProcess.normalize(arr, mean=means,
                                                       stdev=stdevs)
        if update:
            for i, n in enumerate(names):
                norm_params = {n: {'mean': means[i], 'stdev': stdevs[i]}}
                self._norm_params.update(norm_params)

        return norm_arr

    def normalize(self, data, names=None):
        """
        Normalize given data

        Parameters
        ----------
        data : dict | pandas.DataFrame | ndarray
            Data to normalize
        names : list, optional
            List of data item names, needed to normalized ndarrays,
            by default None

        Returns
        -------
        data : dict | pandas.DataFrame | ndarray
            Normalized data in same format as input
        """
        data = copy.deepcopy(data)
        if isinstance(data, dict):
            data = self._normalize_dict(data)
        elif isinstance(data, pd.DataFrame):
            if self.one_hot_feature_names:
                cols = [c for c in data if c not in self.one_hot_feature_names]
                data.loc[:, cols] = self._normalize_arr(
                    data.loc[:, cols].values, cols)
            else:
                data.loc[:] = self._normalize_arr(data.values, data.columns)
        elif isinstance(data, (list, np.ndarray)):
            if names is None:
                msg = ('Names of items must be supplied to nomralize data '
                       'arrays')
                logger.error(msg)
                raise RuntimeError(msg)
            if self.one_hot_feature_names:
                idx = [i for i, f in enumerate(names)
                       if f not in self.one_hot_feature_names]
                norm_names = np.array(names)[idx]
                data[:, idx] = self._normalize_arr(data[:, idx],
                                                   norm_names)
            else:
                data = self._normalize_arr(data, names)
        else:
            msg = "Cannot normalize data of type: {}".format(type(data))
            logger.error(msg)
            raise RuntimeError(msg)

        return data

    def _unnormalize_dict(self, items):
        """
        Un-normalize given dictionary of items (features | labels)

        Parameters
        ----------
        items : dict
            mapping of names to vectors

        Returns
        -------
        native_items : dict
            mapping of names to native vectors
        """
        native_items = {}
        for key, value in items.items():
            norm_params = self.normalization_parameters[key]
            if norm_params is not None:
                value = PreProcess.unnormalize(value, norm_params['mean'],
                                               norm_params['stdev'])
            else:
                msg = ("Normalization Parameters unavailable, {} will not be "
                       "un-normalized!".format(key))
                logger.warning(msg)
                warn(msg)

            native_items[key] = value

        return native_items

    def _unnormalize_df(self, df):
        """
        Un-normalize DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of features/label to un-normalize

        Returns
        -------
        df : pandas.DataFrame
            Native features/label df if norm params are not None
        """
        means, stdevs = self.get_norm_params(df.columns)

        if means is not None and stdevs is not None:
            df = PreProcess.unnormalize(df.copy(), means, stdevs)
        else:
            msg = ("Normalization parameters are unavailable, df will not be "
                   "un-normalized!")
            logger.warning(msg)
            warn(msg)

        return df

    def _unnormalize_arr(self, arr, names):
        """
        Un-normalize array using given names

        Parameters
        ----------
        arr : ndarray
            Array of features/label to un-normalize
        names : list
            List of feature/label names

        Returns
        -------
        arr : ndarray
            Native features/label array if norm params are not None
        """
        n_names = self._get_item_number(arr)
        if len(names) != n_names:
            msg = ("Number of item names ({}) does not match number of items "
                   "({})".format(len(names), arr.shape[1]))
            logger.error(msg)
            raise RuntimeError(msg)

        means, stdevs = self.get_norm_params(names)

        if means is not None and stdevs is not None:
            arr = PreProcess.unnormalize(arr.copy(), means, stdevs)
        else:
            msg = ("Normalization parameters are unavailable, arr will not be "
                   "un-normalized!")
            logger.warning(msg)
            warn(msg)

        return arr

    def unnormalize(self, data, names=None):
        """
        Un-normalize given data

        Parameters
        ----------
        data : dict | pandas.DataFrame | ndarray
            Data to un-normalize
        names : list, optional
            List of data item names, needed to un-normalized ndarrays,
            by default None

        Returns
        -------
        data : dict | pandas.DataFrame | ndarray
            Native data in same format as input
        """
        if isinstance(data, dict):
            data = self._unnormalize_dict(data)
        elif isinstance(data, pd.DataFrame):
            data = self._unnormalize_df(data)
        elif isinstance(data, (list, np.ndarray)):
            if names is None:
                msg = ('Names of items must be supplied to un-nomralize data '
                       'arrays')
                logger.error(msg)
                raise RuntimeError(msg)
            data = self._unnormalize_arr(data, names)
        else:
            msg = "Cannot un-normalize data of type: {}".format(type(data))
            logger.error(msg)
            raise RuntimeError(msg)

        return data

    def _check_one_hot_feature_names(self, feature_names):
        """
        Check one_hot_feature_names, update feature_names to remove features
        that were one-hot encoded and add in new one-hot features if needed

        Parameters
        ----------
        feature_names : list
            Input feature names
        """
        one_hot_feature_names = self.make_one_hot_feature_names(
            feature_names, self.one_hot_categories)
        if one_hot_feature_names != self.feature_names:
            check_names = feature_names.copy()
            if self.label_names is not None:
                check_names += self.label_names

            PreProcess.check_one_hot_categories(self.one_hot_categories,
                                                feature_names=check_names)
            self._feature_names = one_hot_feature_names

    def parse_features(self, features, names=None, **kwargs):
        """Parse features - preprocessing of feature data before training or
        prediction. This will do one-hot encoding based on
        self.one_hot_categories, and feature normalization based on
        self.normalize_features

        Parameters
        ----------
        features : pandas.DataFrame | dict | ndarray
            Features to train on or predict from
        names : list, optional
            List of feature names, by default None
        kwargs : dict, optional
            kwargs for PreProcess.one_hot

        Returns
        -------
        features : ndarray
            Parsed features array normalized and with str columns converted
            to one hot vectors if desired
        """

        features, feature_names = self._parse_data_names(features, names=names,
                                                         fallback_prefix='F')

        if self.feature_names is None:
            self._feature_names = feature_names

        check = (bool(self.one_hot_categories)
                 and feature_names is not None
                 and all(np.isin(feature_names, self.input_feature_names)))
        if check:
            self._check_one_hot_feature_names(feature_names)
            kwargs.update({'feature_names': feature_names,
                           'categories': self.one_hot_categories})
            features = PreProcess.one_hot(features, **kwargs)
        elif (not isinstance(features, np.ndarray)
                and self.feature_names != feature_names):
            msg = ('Expecting features with names: {}, but was provided with: '
                   '{}!'.format(self.feature_names, feature_names))
            logger.error(msg)
            raise RuntimeError(msg)

        if self.normalize_features:
            features = self.normalize(features, names=self.feature_names)

        if (self.feature_dims is not None
                and features.shape[-1] != self.feature_dims):
            msg = ('data has {} features but expected {}'
                   .format(features.shape[1], self.feature_dims))
            logger.error(msg)
            raise RuntimeError(msg)

        return features

    def parse_labels(self, labels, names=None):
        """
        Parse labels and normalize if desired

        Parameters
        ----------
        labels : pandas.DataFrame | dict | ndarray
            Features to train on or predict from
        names : list, optional
            List of label names, by default None

        Returns
        -------
        labels : ndarray
            Parsed labels array, normalized if desired
        """

        labels, label_names = self._parse_data_names(labels, names=names,
                                                     fallback_prefix='L')

        if self.label_names is not None:
            n_labels = self._get_item_number(labels)
            if n_labels != len(self.label_names):
                msg = ('data has {} labels but expected {}'
                       .format(labels.shape[1], self.label_dims))
                logger.error(msg)
                raise RuntimeError(msg)

        if self._label_names is None:
            self._label_names = label_names
        elif (not isinstance(labels, np.ndarray)
                and self.label_names != label_names):
            msg = ('Expecting labels with names: {}, but was provided with: '
                   '{}!'.format(self.label_names, label_names))
            logger.error(msg)
            raise RuntimeError(msg)

        if self.normalize_labels:
            labels = self.normalize(labels, names=label_names)

        return labels

    def predict(self, features, table=True, parse_kwargs=None,
                predict_kwargs=None):
        """
        Use model to predict label from given features

        Parameters
        ----------
        features : dict | pandas.DataFrame
            features to predict from
        table : bool, optional
            Return pandas DataFrame
        parse_kwargs : dict
            kwargs for cls.parse_features
        predict_kwargs : dict
            kwargs for tensorflow.*.predict

        Returns
        -------
        prediction : ndarray | pandas.DataFrame
            label prediction
        """

        if parse_kwargs is None:
            parse_kwargs = {}

        if isinstance(features, np.ndarray):
            n_features = features.shape[-1]
            if n_features == self.feature_dims:
                kwargs = {"names": self.feature_names}
                logger.debug('Parsing features with feature_names: {}'
                             .format(self.feature_names))
            elif n_features == len(self.input_feature_names):
                kwargs = {"names": self.input_feature_names}
                logger.debug('Parsing features with input_feature_names: {}'
                             .format(self.input_feature_names))
            else:
                msg = ('Number of features provided ({}) does not match number'
                       ' of model features ({}) or number of input features '
                       '({})'.format(n_features, self.feature_dims,
                                     len(self.input_feature_names)))
                logger.error(msg)
                raise RuntimeError(msg)

            parse_kwargs.update(kwargs)

        features = self.parse_features(features, **parse_kwargs)

        if predict_kwargs is None:
            predict_kwargs = {}

        prediction = self._model.predict(features, **predict_kwargs)
        if self.normalize_labels:
            prediction = self.unnormalize(prediction, names=self.label_names)

        if table and len(prediction.shape) in {1, 2}:
            prediction = pd.DataFrame(prediction, columns=self.label_names)

        return prediction
