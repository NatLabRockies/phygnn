"""Regression tests for transformer and attention kwarg forwarding."""

import importlib

import pytest
import tensorflow as tf

from phygnn.layers import custom_layers as custom_layers_module

custom_layers_module = importlib.reload(custom_layers_module)
MultiHeadAttention = custom_layers_module.MultiHeadAttention
Sup3rTransformerBlock = custom_layers_module.Sup3rTransformerBlock
Sup3rTransformerLayer = custom_layers_module.Sup3rTransformerLayer
Sup3rTransformerLayerAlibi = custom_layers_module.Sup3rTransformerLayerAlibi
TransformerLayer = custom_layers_module.TransformerLayer


def _patch_fused_attention(monkeypatch, dot_product_attention):
    """Patch the fused attention helper used by the custom layer."""
    monkeypatch.setattr(
        custom_layers_module,
        '_dot_product_attention',
        dot_product_attention,
    )


def test_transformer_layer_forwards_attn_kwargs():
    """TransformerLayer should pass attention kwargs to MultiHeadAttention."""
    layer = TransformerLayer(
        num_heads=2,
        key_dim=8,
        attn_kwargs={'dropout': 0.25, 'use_bias': False},
    )

    assert layer.attn._dropout == pytest.approx(0.25)
    assert layer.attn._use_bias is False


def test_sup3r_transformer_layer_forwards_attn_kwargs():
    """Sup3rTransformerLayer should preserve attention kwargs."""
    layer = Sup3rTransformerLayer(
        embed_dim=16,
        num_heads=2,
        key_dim=8,
        attn_kwargs={'dropout': 0.15},
    )

    assert layer.transformer.attn._dropout == pytest.approx(0.15)


def test_sup3r_transformer_block_forwards_layer_and_attention_kwargs():
    """Sup3rTransformerBlock should keep kwargs separated by layer type."""
    block = Sup3rTransformerBlock(
        features=['obs', 'topography'],
        num_heads=4,
        key_dim=12,
        embed_dim=24,
        attn_kwargs={'dropout': 0.05},
    )

    assert len(block.layers) == 2
    assert all(
        isinstance(layer, Sup3rTransformerLayer) for layer in block.layers
    )
    assert all(layer.embed_dim == 24 for layer in block.layers)
    assert all(layer.key_dim == 12 for layer in block.layers)
    assert all(layer.num_heads == 4 for layer in block.layers)
    assert all(
        layer.transformer.attn._dropout == pytest.approx(0.05)
        for layer in block.layers
    )


def test_sup3r_transformer_block_uses_alibi_variant():
    """Sup3rTransformerBlock should build the ALiBi transformer variant."""
    block = Sup3rTransformerBlock(
        features=['obs'],
        use_alibi=True,
        num_heads=2,
        key_dim=8,
        transformer_kwargs={'embed_dim': 16},
        attn_kwargs={'dropout': 0.1},
    )

    assert len(block.layers) == 1
    assert isinstance(block.layers[0], Sup3rTransformerLayerAlibi)
    assert block.layers[0].transformer.attn._dropout == pytest.approx(0.1)


def test_multi_head_attention_uses_fused_path_when_available(monkeypatch):
    """MultiHeadAttention should call the fused op when it is eligible."""
    calls = {}

    def fake_dot_product_attention(
        query,
        key,
        value,
        *,
        bias=None,
        mask=None,
        scale=None,
        is_causal=False,
        flash_attention=None,
        attn_logits_soft_cap=None,
    ):
        calls['query_shape'] = tuple(query.shape)
        calls['key_shape'] = tuple(key.shape)
        calls['value_shape'] = tuple(value.shape)
        calls['bias_shape'] = None if bias is None else tuple(bias.shape)
        calls['mask_shape'] = None if mask is None else tuple(mask.shape)
        calls['scale'] = scale
        calls['is_causal'] = is_causal
        calls['flash_attention'] = flash_attention
        calls['attn_logits_soft_cap'] = attn_logits_soft_cap
        return query

    _patch_fused_attention(monkeypatch, fake_dot_product_attention)

    layer = MultiHeadAttention(num_heads=2, key_dim=4)
    query = tf.random.normal((1, 3, 8))
    value = tf.random.normal((1, 5, 8))
    bias = tf.random.normal((1, 2, 3, 5))
    attention_mask = tf.ones((1, 3, 5), dtype=tf.bool)

    output = layer(
        query,
        value,
        attention_mask=attention_mask,
        bias=bias,
    )

    assert output.shape == (1, 3, 8)
    assert calls['query_shape'] == (1, 3, 2, 4)
    assert calls['key_shape'] == (1, 5, 2, 4)
    assert calls['value_shape'] == (1, 5, 2, 4)
    assert calls['bias_shape'] == (1, 2, 3, 5)
    assert calls['mask_shape'] == (1, 1, 3, 5)
    assert calls['scale'] is None
    assert calls['is_causal'] is False
    assert calls['flash_attention'] is None
    assert calls['attn_logits_soft_cap'] is None


def test_multi_head_attention_falls_back_for_scores(monkeypatch):
    """Requesting attention scores should disable the fused path."""

    def fail_dot_product_attention(*_args, **_kwargs):
        raise AssertionError('fused attention should not be used')

    _patch_fused_attention(monkeypatch, fail_dot_product_attention)

    layer = MultiHeadAttention(num_heads=2, key_dim=4)
    query = tf.random.normal((1, 3, 8))
    value = tf.random.normal((1, 5, 8))

    output, scores = layer(query, value, return_attention_scores=True)

    assert output.shape == (1, 3, 8)
    assert scores.shape == (1, 2, 3, 5)


def test_multi_head_attention_falls_back_when_dropout_is_active(monkeypatch):
    """Training-time dropout should keep using the explicit attention path."""

    def fail_dot_product_attention(*_args, **_kwargs):
        raise AssertionError('fused attention should not be used')

    _patch_fused_attention(monkeypatch, fail_dot_product_attention)

    layer = MultiHeadAttention(num_heads=2, key_dim=4, dropout=0.1)
    query = tf.random.normal((1, 3, 8))
    value = tf.random.normal((1, 5, 8))

    output = layer(query, value, training=True)

    assert output.shape == (1, 3, 8)
