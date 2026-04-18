"""Regression tests for transformer and attention kwarg forwarding."""

import pytest

from phygnn.layers.custom_layers import (
    Sup3rTransformerBlock,
    Sup3rTransformerLayer,
    Sup3rTransformerLayerAlibi,
    TransformerLayer,
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
        embed_dim=24,
        num_heads=4,
        key_dim=12,
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
        embed_dim=16,
        num_heads=2,
        key_dim=8,
        attn_kwargs={'dropout': 0.1},
    )

    assert len(block.layers) == 1
    assert isinstance(block.layers[0], Sup3rTransformerLayerAlibi)
    assert block.layers[0].transformer.attn._dropout == pytest.approx(0.1)
