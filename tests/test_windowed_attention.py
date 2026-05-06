"""Tests for windowed attention implementation."""

import numpy as np
import pytest
import tensorflow as tf

from phygnn.layers.custom_layers import (
    MultiHeadAttention,
    Sup3rTransformerBlock,
    Sup3rTransformerLayer,
    Sup3rTransformerLayerAlibi,
    TransformerLayer,
    WindowedMultiHeadAttention,
)


class TestWindowedMultiHeadAttention:
    """Tests for the WindowedMultiHeadAttention layer."""

    def test_output_shape_self_attention(self):
        """Output shape should match query shape for self-attention."""
        layer = WindowedMultiHeadAttention(
            num_heads=2, key_dim=4, window_size=4, overlap=2
        )
        query = tf.random.normal((2, 16, 8))
        output = layer(
            query,
            query,
            query_spatial_shape=(4, 4),
            kv_spatial_shape=(4, 4),
        )
        assert output.shape == (2, 16, 8)

    def test_output_shape_cross_attention(self):
        """Output should work with different query and KV spatial sizes."""
        layer = WindowedMultiHeadAttention(
            num_heads=2, key_dim=4, window_size=3, overlap=1
        )
        # Query is 4x4=16 tokens, KV is 8x8=64 tokens
        query = tf.random.normal((1, 16, 8))
        kv = tf.random.normal((1, 64, 8))
        output = layer(
            query,
            kv,
            query_spatial_shape=(4, 4),
            kv_spatial_shape=(8, 8),
        )
        assert output.shape == (1, 16, 8)

    def test_full_window_matches_full_attention(self):
        """Windowed attention with window >= grid should match full."""
        tf.random.set_seed(99)
        num_heads, key_dim = 2, 4
        query = tf.random.normal((1, 16, 8))
        kv = tf.random.normal((1, 16, 8))

        full_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        win_layer = WindowedMultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, window_size=10, overlap=2
        )
        full_layer.build((None, 16, 8), (None, 16, 8))
        win_layer.build((None, 16, 8), (None, 16, 8))
        for fw, ww in zip(full_layer.weights, win_layer.weights):
            ww.assign(fw)

        full_out = full_layer(query, kv)
        win_out = win_layer(
            query,
            kv,
            query_spatial_shape=(4, 4),
            kv_spatial_shape=(4, 4),
        )
        np.testing.assert_allclose(
            full_out.numpy(), win_out.numpy(), atol=1e-5
        )

    def test_with_bias(self):
        """Windowed attention should correctly slice bias."""
        layer = WindowedMultiHeadAttention(
            num_heads=2, key_dim=4, window_size=2, overlap=1
        )
        query = tf.random.normal((1, 16, 8))
        kv = tf.random.normal((1, 16, 8))
        bias = tf.random.normal((1, 2, 16, 16))
        output = layer(
            query,
            kv,
            bias=bias,
            query_spatial_shape=(4, 4),
            kv_spatial_shape=(4, 4),
        )
        assert output.shape == (1, 16, 8)

    def test_without_bias(self):
        """Windowed attention should work with bias=None."""
        layer = WindowedMultiHeadAttention(
            num_heads=2, key_dim=4, window_size=2, overlap=1
        )
        query = tf.random.normal((1, 16, 8))
        kv = tf.random.normal((1, 16, 8))
        output = layer(
            query,
            kv,
            bias=None,
            query_spatial_shape=(4, 4),
            kv_spatial_shape=(4, 4),
        )
        assert output.shape == (1, 16, 8)

    def test_consistency_with_full_attention(self):
        """Windowed attention with large window should match full attention."""
        tf.random.set_seed(42)
        num_heads, key_dim = 2, 4
        query = tf.random.normal((1, 9, 8))
        kv = tf.random.normal((1, 9, 8))

        # Build both layers first, then copy weights
        full_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        win_layer = WindowedMultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, window_size=10, overlap=0
        )
        full_layer.build((None, 9, 8), (None, 9, 8))
        win_layer.build((None, 9, 8), (None, 9, 8))
        for fw, ww in zip(full_layer.weights, win_layer.weights):
            ww.assign(fw)

        # Call both with same input
        full_out = full_layer(query, kv)
        win_out = win_layer(
            query, kv, query_spatial_shape=(3, 3), kv_spatial_shape=(3, 3)
        )
        np.testing.assert_allclose(
            full_out.numpy(), win_out.numpy(), atol=1e-5
        )

    def test_get_config(self):
        """Config should include window_size and overlap."""
        layer = WindowedMultiHeadAttention(
            num_heads=2, key_dim=4, window_size=5, overlap=3
        )
        config = layer.get_config()
        assert config['window_size'] == 5
        assert config['overlap'] == 3
        assert config['num_heads'] == 2

    def test_non_square_grid(self):
        """Should handle non-square spatial grids."""
        layer = WindowedMultiHeadAttention(
            num_heads=2, key_dim=4, window_size=3, overlap=1
        )
        # Query is 6x4=24 tokens, KV is 12x8=96 tokens
        query = tf.random.normal((1, 24, 8))
        kv = tf.random.normal((1, 96, 8))
        output = layer(
            query,
            kv,
            query_spatial_shape=(6, 4),
            kv_spatial_shape=(12, 8),
        )
        assert output.shape == (1, 24, 8)


class TestTransformerLayerWindowed:
    """Tests for TransformerLayer with attention_type='windowed'."""

    def test_windowed_attention_type(self):
        """TransformerLayer should use WindowedMultiHeadAttention."""
        layer = TransformerLayer(
            num_heads=2,
            key_dim=8,
            attention_type='windowed',
            attention_kwargs={'window_size': 4, 'overlap': 2},
        )
        assert isinstance(layer.attn, WindowedMultiHeadAttention)
        assert layer.attn.window_size == 4
        assert layer.attn.overlap == 2

    def test_windowed_forward_pass(self):
        """TransformerLayer should produce correct output with windowed."""
        layer = TransformerLayer(
            num_heads=2,
            key_dim=8,
            attention_type='windowed',
            attention_kwargs={'window_size': 3, 'overlap': 1},
        )
        query = tf.random.normal((1, 16, 8))
        key = tf.random.normal((1, 64, 8))
        value = tf.random.normal((1, 64, 8))
        output = layer(
            query,
            key,
            value,
            query_spatial_shape=(4, 4),
            kv_spatial_shape=(8, 8),
        )
        assert output.shape == (1, 16, 8)

    def test_full_attention_type_default(self):
        """Default attention_type should be 'full'."""
        layer = TransformerLayer(num_heads=2, key_dim=8)
        assert isinstance(layer.attn, MultiHeadAttention)
        assert not isinstance(layer.attn, WindowedMultiHeadAttention)

    def test_get_config_includes_attention_type(self):
        """get_config should include attention_type."""
        layer = TransformerLayer(
            num_heads=2, key_dim=8, attention_type='windowed',
            attention_kwargs={'window_size': 4, 'overlap': 2},
        )
        config = layer.get_config()
        assert config['attention_type'] == 'windowed'


class TestSup3rTransformerLayerWindowed:
    """Tests for Sup3rTransformerLayer with windowed attention."""

    def test_windowed_4d_input(self):
        """Sup3rTransformerLayer should work with 4D input and windowed."""
        layer = Sup3rTransformerLayer(
            features=['obs'],
            num_heads=2,
            key_dim=8,
            embed_dim=8,
            attention_type='windowed',
            attention_kwargs={'window_size': 4, 'overlap': 2},
        )
        x = tf.random.normal((1, 8, 8, 16))
        hr = tf.random.normal((1, 8, 8, 1))
        lat = np.linspace(30, 40, 8).reshape(1, 8, 1, 1) * np.ones(
            (1, 1, 8, 1)
        )
        lon = np.linspace(-100, -90, 8).reshape(1, 1, 8, 1) * np.ones(
            (1, 8, 1, 1)
        )
        exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)
        output = layer(x, hi_res_feature=hr, exo_data=exo_data)
        assert output.shape == (1, 8, 8, 16)

    def test_windowed_cross_attention_4d(self):
        """Should work with hi_res_feature having different spatial dims."""
        layer = Sup3rTransformerLayer(
            features=['obs'],
            num_heads=2,
            key_dim=8,
            embed_dim=8,
            attention_type='windowed',
            attention_kwargs={'window_size': 3, 'overlap': 1},
        )
        x = tf.random.normal((1, 8, 8, 16))
        hr = tf.random.normal((1, 8, 8, 1))
        lat = np.linspace(30, 40, 8).reshape(1, 8, 1, 1) * np.ones(
            (1, 1, 8, 1)
        )
        lon = np.linspace(-100, -90, 8).reshape(1, 1, 8, 1) * np.ones(
            (1, 8, 1, 1)
        )
        exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)
        output = layer(x, hi_res_feature=hr, exo_data=exo_data)
        assert output.shape == (1, 8, 8, 16)

    def test_get_config_includes_attention_type(self):
        """get_config should include attention_type."""
        layer = Sup3rTransformerLayer(
            features=['obs'],
            num_heads=2,
            key_dim=8,
            embed_dim=8,
            attention_type='windowed',
            attention_kwargs={'window_size': 4, 'overlap': 2},
        )
        config = layer.get_config()
        assert config['attention_type'] == 'windowed'


class TestSup3rTransformerBlockWindowed:
    """Tests for Sup3rTransformerBlock with windowed attention."""

    def test_block_windowed_construction(self):
        """Block should create windowed layers when specified."""
        block = Sup3rTransformerBlock(
            features=['obs', 'topography'],
            num_heads=2,
            key_dim=8,
            embed_dim=8,
            attention_type='windowed',
            attention_kwargs={'window_size': 4, 'overlap': 2},
        )
        assert len(block.layers) == 2
        for layer in block.layers:
            assert isinstance(layer.tl.attn, WindowedMultiHeadAttention)
            assert layer.tl.attn.window_size == 4
            assert layer.tl.attn.overlap == 2

    def test_block_windowed_forward_pass(self):
        """Block should produce correct output with windowed attention."""
        block = Sup3rTransformerBlock(
            features=['obs'],
            num_heads=2,
            key_dim=8,
            embed_dim=8,
            attention_type='windowed',
            attention_kwargs={'window_size': 3, 'overlap': 1},
        )
        x = tf.random.normal((1, 8, 8, 16))
        hr = tf.random.normal((1, 8, 8, 1))
        lat = np.linspace(30, 40, 8).reshape(1, 8, 1, 1) * np.ones(
            (1, 1, 8, 1)
        )
        lon = np.linspace(-100, -90, 8).reshape(1, 1, 8, 1) * np.ones(
            (1, 8, 1, 1)
        )
        exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)
        output = block(x, hi_res_features=hr, exo_data=exo_data)
        assert output.shape == (1, 8, 8, 16)

    def test_block_windowed_alibi(self):
        """Block should work with windowed + ALiBi."""
        block = Sup3rTransformerBlock(
            features=['obs'],
            num_heads=2,
            key_dim=8,
            embed_dim=8,
            use_alibi=True,
            attention_type='windowed',
            attention_kwargs={'window_size': 4, 'overlap': 2},
        )
        assert len(block.layers) == 1
        assert isinstance(block.layers[0], Sup3rTransformerLayerAlibi)
        assert isinstance(
            block.layers[0].tl.attn, WindowedMultiHeadAttention
        )

    def test_block_get_config(self):
        """Block get_config should include attention_type."""
        block = Sup3rTransformerBlock(
            features=['obs'],
            num_heads=2,
            key_dim=8,
            embed_dim=8,
            attention_type='windowed',
            attention_kwargs={'window_size': 4, 'overlap': 2},
        )
        config = block.get_config()
        assert config['attention_type'] == 'windowed'
        assert config['attention_kwargs'] == {'window_size': 4, 'overlap': 2}

    def test_block_default_full_attention(self):
        """Block should default to full attention."""
        block = Sup3rTransformerBlock(
            features=['obs'],
            num_heads=2,
            key_dim=8,
            embed_dim=8,
        )
        assert block.attention_type == 'full'
        assert isinstance(block.layers[0].tl.attn, MultiHeadAttention)
        assert not isinstance(
            block.layers[0].tl.attn, WindowedMultiHeadAttention
        )
