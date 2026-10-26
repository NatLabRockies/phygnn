"""Tests for transformer and windowed attention layers."""

import numpy as np
import pytest
import tensorflow as tf

from phygnn.layers import custom_layers as custom_layers_module

MultiHeadAttention = custom_layers_module.MultiHeadAttention
Sup3rTransformerBlock = custom_layers_module.Sup3rTransformerBlock
Sup3rHierarchicalTransformerBlock = (
    custom_layers_module.Sup3rHierarchicalTransformerBlock
)
Sup3rTransformerLayer = custom_layers_module.Sup3rTransformerLayer
WindowedMultiHeadAttention = custom_layers_module.WindowedMultiHeadAttention


# --- WindowedMultiHeadAttention ---


def test_wmha_output_shape():
    """WMHA should produce correct output shapes."""
    layer = WindowedMultiHeadAttention(num_heads=2, key_dim=4, window_size=4)
    query = tf.random.normal((2, 4, 4, 8))
    output = layer(query, query)
    assert output.shape == (2, 4, 4, 8)


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


@pytest.mark.parametrize('temporal_distance_scale', [0, -1])
def test_wmha_rejects_invalid_temporal_distance_scale(
    temporal_distance_scale,
):
    """Enabled temporal ALiBi requires a positive distance scale."""
    with pytest.raises(
        ValueError, match='temporal_distance_scale must be > 0'
    ):
        WindowedMultiHeadAttention(
            temporal_bias_scale=1.0,
            temporal_distance_scale=temporal_distance_scale,
        )


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
        [[[0], [3_600], [7_200]]],
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
    time = tf.constant([[[0.0], [3_600.0]]])

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
        tf.range(2, dtype=tf.float32) * 3_600,
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


def test_wmha_shifted_window_compacts_sparse_kv(monkeypatch):
    """Shifted windows should compact sparse K/V before attention."""

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
        checks = [
            tf.debugging.assert_equal(tf.shape(query), [9, 8, 8]),
            tf.debugging.assert_equal(tf.shape(value), [9, 2, 8]),
            tf.debugging.assert_equal(tf.shape(attention_mask), [9, 1, 2]),
            tf.debugging.assert_equal(tf.shape(bias), [9, 2, 8, 2]),
        ]
        with tf.control_dependencies(checks):
            return tf.identity(query)

    monkeypatch.setattr(MultiHeadAttention, 'call', fake_call)

    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        window_size=2,
        window_shift=1,
        bias_scale=1.0,
        temporal_bias_scale=1.0,
    )
    query = tf.random.normal((1, 4, 4, 2, 8))
    nan_mask = np.ones((1, 4, 4, 2), dtype=bool)
    nan_mask[:, 1, 1, :] = False
    nan_mask[:, 3, 3, 0] = False
    lat_lon = tf.random.normal((1, 4, 4, 2, 2))
    time = tf.random.normal((1, 4, 4, 2, 1))

    output = layer(
        query,
        query,
        kv_nan_mask=tf.constant(nan_mask),
        lat_lon=lat_lon,
        time=time,
    )

    assert output.shape == query.shape


def test_wmha_full_attention_compacts_sparse_kv(monkeypatch):
    """Full attention should use the shared sparse K/V compaction."""

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
        checks = [
            tf.debugging.assert_equal(tf.shape(query), [2, 32, 8]),
            tf.debugging.assert_equal(tf.shape(value), [2, 3, 8]),
            tf.debugging.assert_equal(tf.shape(attention_mask), [2, 1, 3]),
            tf.debugging.assert_equal(tf.shape(bias), [2, 2, 32, 3]),
        ]
        with tf.control_dependencies(checks):
            return tf.identity(query)

    monkeypatch.setattr(MultiHeadAttention, 'call', fake_call)

    layer = WindowedMultiHeadAttention(
        num_heads=2,
        key_dim=4,
        bias_scale=1.0,
        temporal_bias_scale=1.0,
    )
    query = tf.random.normal((2, 4, 4, 2, 8))
    nan_mask = np.ones((2, 4, 4, 2), dtype=bool)
    nan_mask[0, 0, 0, :2] = False
    nan_mask[1, 0, 0, :2] = False
    nan_mask[1, 1, 1, 0] = False
    lat_lon = tf.random.normal((2, 4, 4, 2, 2))
    time = tf.random.normal((2, 4, 4, 2, 1))

    output = layer(
        query,
        query,
        kv_nan_mask=tf.constant(nan_mask),
        lat_lon=lat_lon,
        time=time,
    )

    assert output.shape == query.shape


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
    attention_mask = layer._build_window_mask(None, geometry).numpy()

    # 4 windows with a broadcastable validity mask for 4 KV tokens
    assert attention_mask.shape == (4, 1, 4)
    # All positions within each window should be valid (no padding needed
    # for a perfectly-divisible grid with no shift)
    assert attention_mask.all()


# --- Sup3rTransformerLayer ---


def test_sup3r_transformer_layer_without_decode():
    """Transformer layer should optionally return trainable latent tokens."""
    layer = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=(2, 2, 1),
        embed_dim=8,
        num_heads=2,
        key_dim=4,
        window_size=2,
        window_shift=1,
        bias_scale=1.0,
        temporal_bias_scale=1.0,
        decode=False,
    )
    x = tf.Variable(tf.random.normal((1, 4, 4, 2, 3)))
    observations = tf.random.normal((1, 4, 4, 2, 1))
    exo_data = tf.random.normal((1, 4, 4, 2, 3))

    with tf.GradientTape() as tape:
        output = layer(
            x,
            hi_res_feature=observations,
            exo_data=exo_data,
        )
        loss = tf.reduce_sum(output)
    gradients = tape.gradient(loss, [x, *layer.trainable_variables])

    assert output.shape == (1, 2, 2, 2, 8)
    assert layer.eq.built
    assert layer.decoder is None
    assert all(gradient is not None for gradient in gradients)
    restored = Sup3rTransformerLayer.from_config(layer.get_config())
    assert restored.embed_dim == 8
    assert restored.window_shift == 1
    assert not restored.decode


@pytest.mark.parametrize(
    ('patch_size', 'embed_dim'),
    [(1, 8), ((2, 2, 1), 8), (1, 16)],
)
def test_sup3r_transformer_layer_without_query_encoding(patch_size, embed_dim):
    """Pre-encoded latent queries should retain their identity path."""
    layer = Sup3rTransformerLayer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        bias_scale=1.0,
        temporal_bias_scale=1.0,
        encode_query=False,
        decode=False,
    )
    if patch_size != 1 or embed_dim != 8:
        with pytest.raises(ValueError, match='encode_query=False requires'):
            layer.build((1, 4, 4, 2, 8), exo_data_shape=(1, 4, 4, 2, 2))
        return

    x = tf.random.normal((1, 4, 4, 2, 8))
    observations = tf.random.normal((1, 4, 4, 2, 1))
    exo_data = tf.random.normal((1, 4, 4, 2, 3))
    layer.build(
        x.shape,
        hi_res_feature_shape=observations.shape,
        exo_data_shape=exo_data.shape,
    )
    variables = (
        *layer.attn.trainable_variables,
        *layer.mlp.trainable_variables,
    )
    for variable in variables:
        variable.assign(tf.zeros_like(variable))
    output = layer(x, hi_res_feature=observations, exo_data=exo_data)

    assert not layer.eq.built
    np.testing.assert_allclose(output, x, atol=1e-6)


def test_hierarchical_transformer_shape_gradients_and_config():
    """Hierarchical assimilation should round-trip odd spatial shapes."""
    layer = Sup3rHierarchicalTransformerBlock(
        features=['obs'],
        exo_features=['latitude', 'longitude', 'time'],
        patch_size=(1, 1, 1),
        stage_dims=(8, 16, 32, 64),
        stage_depths=(1, 1, 1, 1),
        num_heads=(2, 2, 4, 4),
        key_dims=(4, 8, 8, 16),
        window_sizes=(4, 4, 2, 2),
        window_shifts=(2, 2, 1, 1),
        bias_scales=(1.0, 1.0, 1.0, 1.0),
        distance_scales=(100.0, 100.0, 100.0, 100.0),
        temporal_bias_scales=(1.0, 1.0, 1.0, 1.0),
        temporal_distance_scales=(3600.0, 3600.0, 3600.0, 3600.0),
    )
    x = tf.Variable(tf.random.normal((1, 9, 11, 2, 3)))
    observations = np.full((1, 9, 11, 2, 1), np.nan, np.float32)
    observations[:, ::2, ::2, :, :] = 1.0
    exo_data = tf.random.normal((1, 9, 11, 2, 3))

    with tf.GradientTape() as tape:
        output = layer(x, tf.constant(observations), exo_data)
        loss = tf.reduce_sum(output)
    gradients = tape.gradient(loss, [x, *layer.trainable_variables])

    assert output.shape == x.shape
    assert not tf.reduce_any(tf.math.is_nan(output))
    assert all(gradient is not None for gradient in gradients)
    stages = (*layer.encoder_stages, *layer.decoder_stages)
    latent_layers = [transformer for stage in stages for transformer in stage]
    assert all(not transformer.encode_query for transformer in latent_layers)
    assert all(not transformer.eq.built for transformer in latent_layers)
    restored = Sup3rHierarchicalTransformerBlock.from_config(
        layer.get_config()
    )
    assert restored.stage_dims == [8, 16, 32, 64]
    assert restored.patch_size == [1, 1, 1]


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


@pytest.mark.parametrize(
    ('window_size', 'expected_shifts'),
    [(5, [0, 2, 4, 1, 3]), (None, [0, 0, 0, 0, 0])],
)
def test_sup3r_transformer_block_shifts(window_size, expected_shifts):
    """Transformer blocks should cycle shifts across layers."""
    block = Sup3rTransformerBlock(
        n_layers=5,
        window_size=window_size,
        window_shift=2,
    )
    shifts = [layer.window_shift for layer in block.transformer_layers]
    assert shifts == expected_shifts


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


def test_spatial_only_patch_size_preserves_time():
    """A per-axis patch size should compress space but preserve time."""
    layer = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=(2, 2, 1),
        num_heads=2,
        key_dim=8,
        embed_dim=8,
        window_size=2,
    )
    x = tf.random.normal((1, 4, 4, 3, 16))
    hr = tf.random.normal((1, 4, 4, 3, 1))
    lat_lon = tf.random.normal((1, 4, 4, 3, 2))

    output = layer(x, hi_res_feature=hr, exo_data=lat_lon)

    assert layer.eq(x).shape == (1, 2, 2, 3, 8)
    assert output.shape == x.shape
    assert layer.get_config()['patch_size'] == [2, 2, 1]


def test_spatial_only_patch_does_not_fill_future_time():
    """Spatial patch filling should not copy observations into later times."""
    layer = Sup3rTransformerLayer(
        features=['obs'],
        patch_size=(2, 2, 1),
        num_heads=2,
        key_dim=8,
        embed_dim=8,
    )
    x = tf.zeros((1, 2, 2, 2, 4), dtype=tf.float32)
    hr = np.full((1, 2, 2, 2, 1), np.nan, dtype=np.float32)
    hr[:, :, :, 0, :] = 5.0
    lat_lon = tf.zeros((1, 2, 2, 2, 2), dtype=tf.float32)
    layer.build(x.shape, hr.shape, lat_lon.shape)

    hr_clean, nan_mask = layer.ek.prepare_sparse_tensor(tf.constant(hr))

    np.testing.assert_allclose(hr_clean[..., 0, :], 5.0)
    np.testing.assert_allclose(hr_clean[..., 1, :], 0.0)
    np.testing.assert_array_equal(
        nan_mask.numpy(), np.array([[[[False, True]]]])
    )


def test_sparse_patch_pool_preserves_validity_and_time():
    """Sparse pooling should average valid values without temporal mixing."""
    encoder = custom_layers_module.PatchEncoder(
        patch_size=(2, 2, 1), embed_dim=4
    )
    values = np.full((1, 4, 4, 2, 1), np.nan, dtype=np.float32)
    values[0, 0, 0, 0, 0] = 2.0
    values[0, 0, 1, 0, 0] = 4.0
    values[0, 2, 2, 1, 0] = 8.0
    encoder.build(values.shape)

    pooled = encoder.pool_sparse_tensor(tf.constant(values)).numpy()

    assert pooled.shape == (1, 2, 2, 2, 1)
    assert pooled[0, 0, 0, 0, 0] == pytest.approx(3.0)
    assert np.isnan(pooled[0, 0, 0, 1, 0])
    assert np.isnan(pooled[0, 1, 1, 0, 0])
    assert pooled[0, 1, 1, 1, 0] == pytest.approx(8.0)


def test_tuple_patch_size_config_round_trip():
    """Tuple patch sizes should survive Keras config serialization."""
    layer = Sup3rTransformerLayer(patch_size=(4, 4, 1))
    restored = Sup3rTransformerLayer.from_config(layer.get_config())

    assert restored.patch_size == [4, 4, 1]


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
    hr_clean, nan_mask = layer.ek.prepare_sparse_tensor(hr)
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
    hr_clean, nan_mask = layer.ek.prepare_sparse_tensor(hr)

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
