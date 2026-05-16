"""Tests for transformer and windowed attention layers."""

import numpy as np
import pytest
import tensorflow as tf

from phygnn.layers import custom_layers as custom_layers_module

MultiHeadAttention = custom_layers_module.MultiHeadAttention
Sup3rTransformerBlock = custom_layers_module.Sup3rTransformerBlock
Sup3rTransformerLayer = custom_layers_module.Sup3rTransformerLayer
TransformerLayer = custom_layers_module.TransformerLayer
WindowedMultiHeadAttention = custom_layers_module.WindowedMultiHeadAttention


# --- WindowedMultiHeadAttention ---


def test_wmha_output_shape():
    """WMHA should produce correct output shapes."""
    layer = WindowedMultiHeadAttention(
        num_heads=2, key_dim=4, window_size=4, radius=2
    )
    query = tf.random.normal((2, 4, 4, 8))
    output = layer(query, query)
    assert output.shape == (2, 4, 4, 8)


def test_wmha_with_and_without_bias():
    """WMHA should work with explicit bias and with bias=None."""
    layer = WindowedMultiHeadAttention(
        num_heads=2, key_dim=4, window_size=2, radius=1
    )
    query = tf.random.normal((1, 4, 4, 8))
    kv = tf.random.normal((1, 4, 4, 8))

    out_with = layer(query, kv)
    out_without = layer(query, kv, bias=None)
    assert out_with.shape == (1, 4, 4, 8)
    assert out_without.shape == (1, 4, 4, 8)


def test_wmha_full_window_matches_standard_mha():
    """WMHA with window >= grid should match standard MHA exactly."""
    tf.random.set_seed(99)
    num_heads, key_dim = 2, 4
    query_4d = tf.random.normal((1, 4, 4, 8))
    kv_4d = tf.random.normal((1, 4, 4, 8))
    query_3d = tf.reshape(query_4d, (1, 16, 8))
    kv_3d = tf.reshape(kv_4d, (1, 16, 8))

    full_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    win_layer = WindowedMultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        window_size=10,
        radius=2,
    )
    full_layer.build((None, 16, 8), (None, 16, 8))
    win_layer.build((None, None, None, 8), (None, None, None, 8))
    for fw, ww in zip(full_layer.weights, win_layer.weights):
        ww.assign(fw)

    full_out = full_layer(query_3d, kv_3d)
    win_out = win_layer(query_4d, kv_4d)
    np.testing.assert_allclose(
        full_out.numpy(),
        tf.reshape(win_out, (1, 16, 8)).numpy(),
        atol=1e-5,
    )


def test_wmha_caps_window_size_to_tensor_shape(monkeypatch):
    """Configured window sizes larger than the grid should be capped."""
    captured = {}

    def fake_call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        bias=None,
    ):
        captured['query_shape'] = tuple(query.shape)
        captured['value_shape'] = tuple(value.shape)
        return query

    monkeypatch.setattr(MultiHeadAttention, 'call', fake_call)

    layer = WindowedMultiHeadAttention(
        num_heads=2, key_dim=4, window_size=10, radius=0
    )
    query = tf.random.normal((1, 4, 4, 8))
    value = tf.random.normal((1, 4, 4, 8))

    output = layer(query, value)

    assert output.shape == (1, 4, 4, 8)
    assert captured['query_shape'] == (1, 16, 8)
    assert captured['value_shape'] == (1, 16, 8)


def test_wmha_ignores_shift_when_window_covers_grid(monkeypatch):
    """Single-window calls should route to full attention and ignore shift."""
    captured = {}

    def fake_call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        bias=None,
    ):
        captured['query_shape'] = tuple(query.shape)
        captured['value_shape'] = tuple(value.shape)
        return query

    monkeypatch.setattr(MultiHeadAttention, 'call', fake_call)

    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        window_size=10,
        radius=1,
        window_shift=3,
    )
    query = tf.random.normal((1, 4, 4, 8))
    value = tf.random.normal((1, 4, 4, 8))

    output = layer(query, value)

    assert output.shape == (1, 4, 4, 8)
    assert captured['query_shape'] == (1, 16, 8)
    assert captured['value_shape'] == (1, 16, 8)


def test_wmha_get_config():
    """Config should include window size, radius, and num_heads."""
    layer = WindowedMultiHeadAttention(
        num_heads=2, key_dim=4, window_size=5, radius=3, window_shift=1
    )
    config = layer.get_config()
    assert config['window_size'] == 5
    assert config['radius'] == 3
    assert config['window_shift'] == 1
    assert config['num_heads'] == 2


def test_wmha_non_square_grid():
    """Should handle non-square spatial grids."""
    layer = WindowedMultiHeadAttention(
        num_heads=2, key_dim=4, window_size=3, radius=1
    )
    query = tf.random.normal((1, 6, 4, 8))
    output = layer(query, query)
    assert output.shape == (1, 6, 4, 8)


def test_wmha_shifted_window_path(monkeypatch):
    """Shifted multi-window calls should use the windowed path."""
    captured = {}

    def fake_call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        bias=None,
    ):
        captured['query_shape'] = tuple(query.shape)
        captured['value_shape'] = tuple(value.shape)
        return query

    monkeypatch.setattr(MultiHeadAttention, 'call', fake_call)

    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        window_size=2,
        radius=1,
        window_shift=1,
    )
    query = tf.random.normal((1, 6, 6, 8))
    value = tf.random.normal((1, 6, 6, 8))

    output = layer(query, value)

    assert output.shape == (1, 6, 6, 8)
    assert captured['query_shape'] == (16, 4, 8)
    assert captured['value_shape'] == (16, 16, 8)


def test_wmha_shifted_window_alibi():
    """Shifted window attention should work with ALiBi enabled."""
    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        window_size=2,
        radius=1,
        window_shift=1,
        bias_scale=1.0,
    )
    query = tf.random.normal((1, 4, 4, 8))
    lat = np.linspace(30, 40, 4).reshape(1, 4, 1, 1) * np.ones((1, 1, 4, 1))
    lon = np.linspace(-100, -90, 4).reshape(1, 1, 4, 1) * np.ones((1, 4, 1, 1))

    output = layer(
        query,
        query,
        lat=tf.constant(lat, dtype=tf.float32),
        lon=tf.constant(lon, dtype=tf.float32),
    )

    assert output.shape == (1, 4, 4, 8)


def test_wmha_masks_padded_kv_positions():
    """Halo padding should be masked out of attention for boundary windows."""
    layer = WindowedMultiHeadAttention(
        num_heads=1, key_dim=2, window_size=2, radius=1
    )
    # Use 4x4 spatial so radius=1 isn't capped (max_tile=4 > ws+2*r=4).
    query = tf.random.normal((1, 4, 4, 4))
    geometry = layer._get_window_geometry(query, query, window_size=2)
    attention_mask = layer._build_window_mask(
        None, query.dtype, geometry
    ).numpy()

    # 4 windows, each Q has 4 tokens, each KV tile has 16 tokens
    assert attention_mask.shape == (4, 4, 16)
    # Top-left window's halo extends beyond the boundary on top and left.
    # Row 0 and col 0 of the tile are padding → should be masked False.
    top_left_mask = attention_mask[0]
    pad_row_indices = [0, 1, 2, 3]  # tile row 0, all cols
    pad_col_indices = [0, 4, 8, 12]  # tile col 0, all rows
    pad_indices = list(set(pad_row_indices + pad_col_indices))
    # Every Q token should mask out padding positions that fall in its
    # local neighborhood.
    for q_idx in range(4):
        for p_idx in pad_indices:
            assert not top_left_mask[q_idx, p_idx], (
                f'Q token {q_idx} should mask padded KV index {p_idx}'
            )


# --- TransformerLayer ---


def test_transformer_layer_windowed():
    """TransformerLayer should use WMHA and forward window params."""
    layer = TransformerLayer(
        num_heads=2, key_dim=8, window_size=4, radius=2, window_shift=1
    )
    assert isinstance(layer.attn, WindowedMultiHeadAttention)
    assert layer.attn.window_size == 4
    assert layer.attn.radius == 2
    assert layer.attn.window_shift == 1

    config = layer.get_config()
    assert config['window_size'] == 4
    assert config['radius'] == 2
    assert config['window_shift'] == 1


def test_transformer_layer_default_full_attention():
    """Default window_size=None should use full attention via WMHA."""
    layer = TransformerLayer(num_heads=2, key_dim=8)
    assert isinstance(layer.attn, WindowedMultiHeadAttention)
    assert layer.attn.window_size is None


def test_transformer_layer_dropout():
    """TransformerLayer should forward dropout to the attention layer."""
    layer = TransformerLayer(num_heads=2, key_dim=8, dropout=0.25)
    assert layer.attn._dropout == pytest.approx(0.25)


def test_transformer_layer_forward_pass():
    """TransformerLayer should produce correct output shape."""
    layer = TransformerLayer(num_heads=2, key_dim=8, window_size=3, radius=1)
    query = tf.random.normal((1, 4, 4, 8))
    key = tf.random.normal((1, 4, 4, 8))
    value = tf.random.normal((1, 4, 4, 8))
    output = layer(query, key, value)
    assert output.shape == (1, 4, 4, 8)


def test_transformer_layer_linear_attention():
    """Linear attention should run without extra feature-map options."""
    layer = TransformerLayer(
        num_heads=2,
        key_dim=4,
        embed_dim=8,
        linear_attention=True,
    )

    query = tf.random.normal((1, 4, 4, 8))
    key = tf.random.normal((1, 4, 4, 8))
    value = tf.random.normal((1, 4, 4, 8))

    output = layer(query, key, value)

    assert output.shape == (1, 4, 4, 8)
    assert isinstance(
        layer.attn, custom_layers_module.LinearMultiHeadAttention
    )

    config = layer.get_config()
    assert config['linear_attention'] is True


# --- Sup3rTransformerLayer ---


def test_sup3r_transformer_layer_windowed():
    """Sup3rTransformerLayer should forward window params and work."""
    layer = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=2,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=4,
        radius=2,
        window_shift=1,
    )
    assert isinstance(layer.tl.attn, WindowedMultiHeadAttention)
    assert layer.tl.attn.window_size == 4
    assert layer.tl.attn.window_shift == 1

    config = layer.get_config()
    assert config['patch_size'] == 2
    assert config['window_size'] == 4
    assert config['radius'] == 2
    assert config['window_shift'] == 1

    x = tf.random.normal((1, 8, 8, 16))
    hr = tf.random.normal((1, 8, 8, 1))
    lat = np.linspace(30, 40, 8).reshape(1, 8, 1, 1) * np.ones((1, 1, 8, 1))
    lon = np.linspace(-100, -90, 8).reshape(1, 1, 8, 1) * np.ones((1, 8, 1, 1))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)
    output = layer(x, hi_res_feature=hr, exo_data=exo_data)
    assert output.shape == (1, 8, 8, 16)


def test_sup3r_transformer_layer_dropout():
    """Sup3rTransformerLayer should forward dropout through."""
    layer = Sup3rTransformerLayer(
        embed_dim=16, num_heads=2, key_dim=8, dropout=0.3
    )
    assert layer.dropout == pytest.approx(0.3)
    assert layer.tl.attn._dropout == pytest.approx(0.3)


def test_sup3r_transformer_layer_learned_pos_encoding():
    """Sup3rTransformerLayer should pass learned PE config to PositionEncoder.
    """
    layer = Sup3rTransformerLayer(
        embed_dim=16,
        num_heads=2,
        key_dim=8,
        learned_pos_encoding=True,
    )

    assert layer.pe.learned_pos_encoding is True
    assert layer.get_config()['learned_pos_encoding'] is True


# --- Sup3rTransformerBlock ---


def test_block_windowed_construction_and_config():
    """Block should create windowed layers and expose config."""
    block = Sup3rTransformerBlock(
        features=['obs', 'topography'],
        patch_size=2,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=4,
        radius=2,
        window_shift=1,
    )
    assert len(block.layers) == 2
    for layer in block.layers:
        assert isinstance(layer.tl.attn, WindowedMultiHeadAttention)
        assert layer.patch_size == 2
        assert layer.tl.attn.window_size == 4
        assert layer.tl.attn.radius == 2
        assert layer.tl.attn.window_shift == 1

    config = block.get_config()
    assert config['patch_size'] == 2
    assert config['window_size'] == 4
    assert config['radius'] == 2
    assert config['window_shift'] == 1


def test_sup3r_transformer_layer_patch_size_restores_shape():
    """Patchified attention should project back to the unpatched grid."""
    layer = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=2,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=2,
        radius=1,
    )

    x = tf.random.normal((1, 8, 8, 16))
    hr = tf.random.normal((1, 8, 8, 1))
    lat = np.linspace(30, 40, 8).reshape(1, 8, 1, 1) * np.ones((1, 1, 8, 1))
    lon = np.linspace(-100, -90, 8).reshape(1, 1, 8, 1) * np.ones((1, 8, 1, 1))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)

    output = layer(x, hi_res_feature=hr, exo_data=exo_data)

    assert isinstance(
        layer.decoder.proj_layer, tf.keras.layers.Conv2DTranspose
    )
    assert output.shape == x.shape


def test_sup3r_transformer_layer_patch_size_odd_restores_shape():
    """Patchified attention should preserve non-divisible query shapes."""
    layer = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=3,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=2,
        radius=1,
    )

    x = tf.random.normal((1, 10, 11, 16))
    hr = tf.random.normal((1, 10, 11, 1))
    lat = np.linspace(30, 40, 10).reshape(1, 10, 1, 1) * np.ones((1, 1, 11, 1))
    lon = np.linspace(-100, -90, 11).reshape(1, 1, 11, 1) * np.ones((
        1,
        10,
        1,
        1,
    ))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)

    output = layer(x, hi_res_feature=hr, exo_data=exo_data)

    assert isinstance(
        layer.decoder.proj_layer, tf.keras.layers.Conv2DTranspose
    )
    assert output.shape == x.shape


def test_windowed_attention_uses_patch_token_grid():
    """Windowed attention should operate on the patch-token grid."""
    x = tf.random.normal((1, 32, 32, 16))
    hr = tf.random.normal((1, 32, 32, 1))
    lat = np.linspace(30, 40, 32).reshape(1, 32, 1, 1) * np.ones((1, 1, 32, 1))
    lon = np.linspace(-100, -90, 32).reshape(1, 1, 32, 1) * np.ones((
        1,
        32,
        1,
        1,
    ))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)

    layer_patch_1 = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=1,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=4,
        radius=1,
        bias_scale=1.0,
    )
    layer_patch_4 = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=4,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=4,
        radius=1,
        bias_scale=1.0,
    )

    layer_patch_1.build(x.shape, hr.shape, exo_data.shape)
    layer_patch_4.build(x.shape, hr.shape, exo_data.shape)

    assert layer_patch_1.eq(x).shape[1:3] == (32, 32)
    assert layer_patch_4.eq(x).shape[1:3] == (8, 8)
    assert layer_patch_1.tl.attn.window_size == 4
    assert layer_patch_4.tl.attn.window_size == 4


def test_sup3r_transformer_layer_partial_patch_kept():
    """A patch with any valid hi-res values should remain unmasked."""
    layer = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=2,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        bias_scale=1.0,
    )

    x = tf.zeros((1, 4, 4, 2), dtype=tf.float32)
    hr = np.full((1, 4, 4, 1), np.nan, dtype=np.float32)
    hr[0, 0, 0, 0] = 5.0
    hr[0, 2, 2, 0] = 7.0
    hr = tf.constant(hr)
    lat = np.linspace(30, 40, 4).reshape(1, 4, 1, 1) * np.ones((1, 1, 4, 1))
    lon = np.linspace(-100, -90, 4).reshape(1, 1, 4, 1) * np.ones((1, 4, 1, 1))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)

    layer.build(x.shape, hr.shape, exo_data.shape)
    hr_clean, nan_mask, _, _ = layer.ek.prepare_sparse_tensor(hr)
    k = layer.ek(hr_clean)

    assert k.shape == (1, 2, 2, 8)
    np.testing.assert_array_equal(
        nan_mask.numpy(),
        np.array([[[False, True], [True, False]]]),
    )


def test_sup3r_transformer_layer_requires_exo_data():
    """Sup3rTransformerLayer should reject missing exogenous inputs."""
    layer = Sup3rTransformerLayer(
        features=['obs'],
        num_heads=2,
        key_dim=8,
        embed_dim=8,
    )
    x = tf.random.normal((1, 8, 8, 16))
    hr = tf.random.normal((1, 8, 8, 1))

    with pytest.raises(ValueError, match='requires exo_data'):
        layer.build(x.shape, hr.shape, None)

    with pytest.raises(ValueError, match='requires exo_data'):
        layer(x, hi_res_feature=hr, exo_data=None)


def test_block_windowed_forward_pass():
    """Block should produce correct output with windowed attention."""
    block = Sup3rTransformerBlock(
        features=['obs'],
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=3,
        radius=1,
    )
    x = tf.random.normal((1, 8, 8, 16))
    hr = tf.random.normal((1, 8, 8, 1))
    lat = np.linspace(30, 40, 8).reshape(1, 8, 1, 1) * np.ones((1, 1, 8, 1))
    lon = np.linspace(-100, -90, 8).reshape(1, 1, 8, 1) * np.ones((1, 8, 1, 1))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)
    output = block(x, hi_res_features=hr, exo_data=exo_data)
    assert output.shape == (1, 8, 8, 16)


def test_block_alibi_windowed():
    """Block should work with ALiBi + windowed attention."""
    block = Sup3rTransformerBlock(
        features=['obs'],
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        bias_scale=1.0,
        window_size=4,
        radius=2,
    )
    assert len(block.layers) == 1
    assert isinstance(block.layers[0], Sup3rTransformerLayer)
    assert isinstance(block.layers[0].tl.attn, WindowedMultiHeadAttention)


def test_block_default_full_attention():
    """Block should default to full attention (window_size=None)."""
    block = Sup3rTransformerBlock(
        features=['obs'],
        num_heads=2,
        key_dim=8,
        embed_dim=8,
    )
    assert block.window_size is None
    assert isinstance(block.layers[0].tl.attn, WindowedMultiHeadAttention)
    assert block.layers[0].tl.attn.window_size is None


def test_block_dropout():
    """Block should forward dropout to all sub-layers."""
    block = Sup3rTransformerBlock(
        features=['obs', 'topography'],
        num_heads=2,
        key_dim=8,
        embed_dim=16,
        dropout=0.2,
    )
    assert block.dropout == pytest.approx(0.2)
    for layer in block.layers:
        assert layer.dropout == pytest.approx(0.2)
        assert layer.tl.attn._dropout == pytest.approx(0.2)


# --- Fused attention path ---


def _patch_fused_attention(monkeypatch, dot_product_attention):
    """Patch the fused attention helper used by the custom layer."""
    monkeypatch.setattr(
        tf.keras.ops,
        'dot_product_attention',
        dot_product_attention,
    )


def test_mha_uses_fused_path_when_available(monkeypatch):
    """MHA should call the fused op when eligible."""
    calls = {}

    def fake_dot_product_attention(
        query,
        key,
        value,
        *,
        bias=None,
        mask=None,
        **_kwargs,
    ):
        calls['query_shape'] = tuple(query.shape)
        calls['key_shape'] = tuple(key.shape)
        calls['value_shape'] = tuple(value.shape)
        calls['bias_shape'] = None if bias is None else tuple(bias.shape)
        calls['mask_shape'] = None if mask is None else tuple(mask.shape)
        return query

    _patch_fused_attention(monkeypatch, fake_dot_product_attention)

    layer = MultiHeadAttention(num_heads=2, key_dim=4)
    query = tf.random.normal((1, 3, 8))
    value = tf.random.normal((1, 5, 8))
    bias = tf.random.normal((1, 2, 3, 5))
    attention_mask = tf.ones((1, 3, 5), dtype=tf.bool)

    output = layer(query, value, attention_mask=attention_mask, bias=bias)

    assert output.shape == (1, 3, 8)
    assert calls['query_shape'] == (1, 3, 2, 4)
    assert calls['key_shape'] == (1, 5, 2, 4)
    assert calls['value_shape'] == (1, 5, 2, 4)
    assert calls['bias_shape'] == (1, 2, 3, 5)
    assert calls['mask_shape'] == (1, 1, 3, 5)


def test_mha_falls_back_for_scores(monkeypatch):
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


def test_mha_falls_back_when_dropout_active(monkeypatch):
    """Training-time dropout should use the explicit attention path."""

    def fail_dot_product_attention(*_args, **_kwargs):
        raise AssertionError('fused attention should not be used')

    _patch_fused_attention(monkeypatch, fail_dot_product_attention)

    layer = MultiHeadAttention(num_heads=2, key_dim=4, dropout=0.1)
    query = tf.random.normal((1, 3, 8))
    value = tf.random.normal((1, 5, 8))

    output = layer(query, value, training=True)

    assert output.shape == (1, 3, 8)
