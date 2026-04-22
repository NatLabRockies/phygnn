# -*- coding: utf-8 -*-
"""Physics Guided Neural Network python library."""

import os

from tensorflow.keras.utils import get_custom_objects

from ._version import __version__
from .base import CustomNetwork, GradientUtils
from .layers import HiddenLayers, Layers
from .layers.custom_layers import (
    GaussianAveragePooling2D,
    get_custom_layer_objects,
)
from .model_interfaces import PhygnnModel, TfModel
from .phygnn import PhysicsGuidedNeuralNetwork
from .utilities import PreProcess, tf_isin, tf_log10

get_custom_objects().update(get_custom_layer_objects())

__author__ = """Grant Buster"""
__email__ = 'grant.buster@nlr.gov'

PHYGNNDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(PHYGNNDIR), 'tests', 'data')
