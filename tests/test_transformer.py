"""Tests for transformer and windowed attention layers."""

import numpy as np
import pytest
import tensorflow as tf

from phygnn.layers import custom_layers as custom_layers_module

MultiHeadAttention = custom_layers_module.MultiHeadAttention
Sup3rTransformerLayer = custom_layers_module.Sup3rTransformerLayer
TransformerLayer = custom_layers_module.TransformerLayer
WindowedMultiHeadAttention = custom_layers_module.WindowedMultiHeadAttention


# --- WindowedMultiHeadAttention ---


def test_wmha_output_shape():
    """WMHA should produce correct output shapes."""
    layer = WindowedMultiHeadAttention(num_heads=2, key_dim=4, window_size=4)
    query = tf.random.normal((2, 4, 4, 8))
    output = layer(query, query)
    assert output.shape == (2, 4, 4, 8)


def test_wmha_with_and_without_bias():
    """WMHA should work with explicit bias and with bias=None."""
    layer = WindowedMultiHeadAttention(num_heads=2, key_dim=4, window_size=2)
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
        window_size=4,
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


def test_wmha_get_config():
    """Config should include window size and num_heads."""
    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        window_size=5,
        window_shift=1,
        distance_scale=20_000.0,
        temporal_bias_scale=2.0,
        temporal_distance_scale=3_600.0,
    )
    config = layer.get_config()
    assert config['window_size'] == 5
    assert config['window_shift'] == 1
    assert config['num_heads'] == 2
    assert config['distance_scale'] == pytest.approx(20_000.0)
    assert config['temporal_bias_scale'] == pytest.approx(2.0)
    assert config['temporal_distance_scale'] == pytest.approx(3_600.0)


def test_wmha_temporal_bias():
    """Temporal ALiBi should be symmetric and decay with time separation."""
    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        temporal_bias_scale=2.0,
        temporal_distance_scale=3_600.0,
    )
    layer.build((None, None, None, 8), (None, None, None, 8))
    time = tf.constant(
        [[[1_700_000_000], [1_700_003_600], [1_700_007_200]]],
        dtype=tf.float32,
    )

    bias = layer._temporal_bias(time, time).numpy()

    assert bias.shape == (1, 2, 3, 3)
    np.testing.assert_allclose(np.diagonal(bias, axis1=-2, axis2=-1), 0)
    np.testing.assert_allclose(bias, np.swapaxes(bias, -1, -2))
    assert bias[0, 0, 0, 2] < bias[0, 0, 0, 1] < 0
    assert abs(bias[0, 0, 0, 1]) > abs(bias[0, 1, 0, 1])


def test_wmha_combines_spatial_and_temporal_bias():
    """Combined positional bias should sum its enabled components."""
    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        bias_scale=1.0,
        temporal_bias_scale=2.0,
    )
    layer.build((None, None, None, 8), (None, None, None, 8))
    lat_lon = tf.constant([[[30.0, -100.0], [31.0, -99.0]]])
    time = tf.constant([[[1_700_000_000.0], [1_700_003_600.0]]])

    expected = layer._haversine_bias(lat_lon, lat_lon)
    expected += layer._temporal_bias(time, time)

    np.testing.assert_allclose(
        layer._position_bias(lat_lon, time).numpy(), expected.numpy()
    )


@pytest.mark.parametrize('window_size', [None, 2])
def test_wmha_temporal_bias_5d(window_size):
    """Temporal ALiBi should support full and windowed 5D attention."""
    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        window_size=window_size,
        temporal_bias_scale=1.0,
    )
    query = tf.random.normal((1, 4, 4, 2, 8))
    hours = tf.reshape(
        tf.range(2, dtype=tf.float32) * 3_600 + 1_700_000_000,
        (1, 1, 1, 2, 1),
    )
    time = tf.broadcast_to(hours, (1, 4, 4, 2, 1))

    output = layer(query, query, time=time)

    assert output.shape == query.shape


def test_wmha_non_square_grid():
    """Should handle non-square spatial grids."""
    layer = WindowedMultiHeadAttention(num_heads=2, key_dim=4, window_size=3)
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
        window_shift=1,
    )
    query = tf.random.normal((1, 6, 6, 8))
    value = tf.random.normal((1, 6, 6, 8))

    output = layer(query, value)

    assert output.shape == (1, 6, 6, 8)
    assert captured['query_shape'] == (16, 4, 8)
    assert captured['value_shape'] == (16, 4, 8)


def test_wmha_shifted_window_alibi():
    """Shifted window attention should work with ALiBi enabled."""
    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        window_size=2,
        window_shift=1,
        bias_scale=1.0,
    )
    query = tf.random.normal((1, 4, 4, 8))
    lat = np.linspace(30, 40, 4).reshape(1, 4, 1, 1) * np.ones((1, 1, 4, 1))
    lon = np.linspace(-100, -90, 4).reshape(1, 1, 4, 1) * np.ones((1, 4, 1, 1))
    lat_lon = np.concatenate([lat, lon], axis=-1).astype(np.float32)

    output = layer(
        query,
        query,
        lat_lon=tf.constant(lat_lon, dtype=tf.float32),
    )

    assert output.shape == (1, 4, 4, 8)


def test_wmha_masks_padded_kv_positions():
    """Halo padding should be masked out of attention for boundary windows."""
    layer = WindowedMultiHeadAttention(num_heads=1, key_dim=2, window_size=2)
    # Use 4x4 spatial grid.
    query = tf.random.normal((1, 4, 4, 4))
    geometry = layer._get_window_geometry(query, window_size=2)
    attention_mask = layer._build_window_mask(
        None, query.dtype, geometry
    ).numpy()

    # 4 windows, each Q has 4 tokens, each KV tile has 4 tokens
    assert attention_mask.shape == (4, 4, 4)
    # All positions within each window should be valid (no padding needed
    # for a perfectly-divisible grid with no shift)
    assert attention_mask.all()


# --- TransformerLayer ---


def test_transformer_layer_windowed():
    """TransformerLayer should use WMHA and forward window params."""
    layer = TransformerLayer(
        num_heads=2,
        key_dim=8,
        window_size=4,
        window_shift=1,
        distance_scale=20_000.0,
    )
    assert isinstance(layer.attn, WindowedMultiHeadAttention)
    assert layer.attn.window_size == 4
    assert layer.attn.window_shift == 1
    assert layer.attn.distance_scale == pytest.approx(20_000.0)

    config = layer.get_config()
    assert config['window_size'] == 4
    assert config['window_shift'] == 1
    assert config['distance_scale'] == pytest.approx(20_000.0)


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
    layer = TransformerLayer(num_heads=2, key_dim=8, window_size=3)
    query = tf.random.normal((1, 4, 4, 8))
    key = tf.random.normal((1, 4, 4, 8))
    value = tf.random.normal((1, 4, 4, 8))
    output = layer(query, key, value)
    assert output.shape == (1, 4, 4, 8)


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
        window_shift=1,
        distance_scale=20_000.0,
    )
    assert isinstance(layer.attn, WindowedMultiHeadAttention)
    assert layer.attn.window_size == 4
    assert layer.attn.window_shift == 1
    assert layer.attn.distance_scale == pytest.approx(20_000.0)

    config = layer.get_config()
    assert config['patch_size'] == 2
    assert config['window_size'] == 4
    assert config['window_shift'] == 1
    assert config['distance_scale'] == pytest.approx(20_000.0)

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
    assert layer.attn._dropout == pytest.approx(0.3)


# --- Sup3rTransformerLayer construction and config ---


def test_block_windowed_construction_and_config():
    """Layer should create windowed layers and expose config."""
    block = Sup3rTransformerLayer(
        features=['obs', 'topography'],
        patch_size=2,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=4,
        window_shift=1,
    )
    assert isinstance(block.attn, WindowedMultiHeadAttention)
    assert block.attn.window_size == 4
    assert block.attn.window_shift == 1
    assert block.patch_size == 2

    config = block.get_config()
    assert config['patch_size'] == 2
    assert config['window_size'] == 4
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
        bias_scale=1.0,
    )
    layer_patch_4 = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=4,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=4,
        bias_scale=1.0,
    )

    layer_patch_1.build(x.shape, hr.shape, exo_data.shape)
    layer_patch_4.build(x.shape, hr.shape, exo_data.shape)

    assert layer_patch_1.eq(x).shape[1:3] == (32, 32)
    assert layer_patch_4.eq(x).shape[1:3] == (8, 8)
    assert layer_patch_1.attn.window_size == 4
    assert layer_patch_4.attn.window_size == 4


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
    hr_clean, nan_mask, _ = layer.ek.prepare_sparse_tensor(hr)
    k = layer.ek(hr_clean)

    assert k.shape == (1, 2, 2, 8)
    np.testing.assert_array_equal(
        nan_mask.numpy(),
        np.array([[[False, True], [True, False]]]),
    )


def test_multichannel_partial_nan_kept():
    """Pixels valid in some channels but NaN in others should be unmasked."""
    layer = Sup3rTransformerLayer(
        features=['obs_a', 'obs_b'],
        patch_size=1,
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        bias_scale=1.0,
    )

    x = tf.zeros((1, 4, 4, 2), dtype=tf.float32)
    # 2-channel hi-res: channel 0 valid everywhere, channel 1 NaN at (0,0)
    hr = np.ones((1, 4, 4, 2), dtype=np.float32)
    hr[0, 0, 0, 1] = np.nan  # one channel NaN
    hr[0, 3, 3, :] = np.nan  # both channels NaN
    hr = tf.constant(hr)
    lat = np.linspace(30, 40, 4).reshape(1, 4, 1, 1) * np.ones((1, 1, 4, 1))
    lon = np.linspace(-100, -90, 4).reshape(1, 1, 4, 1) * np.ones((1, 4, 1, 1))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)

    layer.build(x.shape, hr.shape, exo_data.shape)
    hr_clean, nan_mask, _ = layer.ek.prepare_sparse_tensor(hr)

    # (0,0): one channel valid -> should be unmasked (False)
    # (3,3): both channels NaN -> should be masked (True)
    assert not nan_mask[0, 0, 0].numpy(), 'partially-valid pixel was masked'
    assert nan_mask[0, 3, 3].numpy(), 'fully-NaN pixel was not masked'
    # cleaned tensor should have no NaNs
    assert not tf.reduce_any(tf.math.is_nan(hr_clean)).numpy()
    # valid channel should be preserved
    np.testing.assert_allclose(hr_clean[0, 0, 0, 0].numpy(), 1.0)
    # NaN channel should be filled with 0 (patch_size=1)
    np.testing.assert_allclose(hr_clean[0, 0, 0, 1].numpy(), 0.0)


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
    """Layer should produce correct output with windowed attention."""
    block = Sup3rTransformerLayer(
        features=['obs'],
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=3,
    )
    x = tf.random.normal((1, 8, 8, 16))
    hr = tf.random.normal((1, 8, 8, 1))
    lat = np.linspace(30, 40, 8).reshape(1, 8, 1, 1) * np.ones((1, 1, 8, 1))
    lon = np.linspace(-100, -90, 8).reshape(1, 1, 8, 1) * np.ones((1, 8, 1, 1))
    exo_data = np.concatenate([lat, lon], axis=-1).astype(np.float32)
    output = block(x, hi_res_feature=hr, exo_data=exo_data)
    assert output.shape == (1, 8, 8, 16)


def test_block_alibi_windowed():
    """Layer should work with ALiBi + windowed attention."""
    block = Sup3rTransformerLayer(
        features=['obs'],
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        bias_scale=1.0,
        window_size=4,
    )
    assert isinstance(block.attn, WindowedMultiHeadAttention)


def test_block_default_full_attention():
    """Layer should default to full attention (window_size=None)."""
    block = Sup3rTransformerLayer(
        features=['obs'],
        num_heads=2,
        key_dim=8,
        embed_dim=8,
    )
    assert block.window_size is None
    assert isinstance(block.attn, WindowedMultiHeadAttention)
    assert block.attn.window_size is None


def test_block_dropout():
    """Layer should forward dropout to all sub-layers."""
    block = Sup3rTransformerLayer(
        features=['obs', 'topography'],
        num_heads=2,
        key_dim=8,
        embed_dim=16,
        dropout=0.2,
    )
    assert block.dropout == pytest.approx(0.2)
    assert block.attn._dropout == pytest.approx(0.2)
