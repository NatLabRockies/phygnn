"""Pytest configuration for CPU-only TensorFlow test execution."""

import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
