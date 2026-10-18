"""Independent oracles for key-side priors in SERIAL temporal/spatial attention."""

import copy

import pytest
import torch

from trokens.models.attention import Attention
from trokens.models.label_conditioned_routing import _prepare, _encode
from test_label_pre_temporal import _new_model, _episode, _loss
from trokens.models.label_conditioned_routing import label_conditioned_episode


def _oracle_attention(module, x, mask=None, bias=None):
    """Do not call Attention.prepare/aggregate: test projection, mask and AV."""
    b, length, d = x.shape
    if module.with_qkv:
        q, k, v = module.qkv(x).reshape(b, length, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
    else:
        q = k = v = x.reshape(b, length, module.num_heads, -1).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) * module.scale
    if bias is not None:
        scores = scores + bias[:, None, None, :].to(scores.dtype)
    valid = None
    if module.use_mask:
        visible = mask[:, None, None, :] if mask.ndim == 2 else mask[:, None]
        valid = visible.any(-1, keepdim=True)
        scores = scores.masked_fill(~visible, -torch.inf)
        scores = torch.where(valid, scores, torch.zeros_like(scores))
    weights = scores.softmax(-1)
    if module.use_mask:
        weights = weights.masked_fill(~visible, 0.)
    result = (module.attn_drop(weights) @ v).transpose(1, 2).reshape(b, length, d)
    if module.with_qkv:
        result = module.proj_drop(module.proj(result))
    if valid is not None:
        result = torch.where(valid.squeeze(1), result, torch.zeros_like(result))
    return result


@pytest.mark.parametrize("mask_kind", ["none", "keys", "pairs"])
@pytest.mark.parametrize("with_qkv", [False, True])
def test_key_log_prior_matches_explicit_softmax_outputs_and_gradients(mask_kind, with_qkv):
    torch.manual_seed(31)
    module = Attention(8, 2, qkv_bias=True, with_qkv=with_qkv, use_mask=mask_kind != "none")
    x = torch.randn(3, 4, 8, requires_grad=True)
    gate = (.8 + .4 * torch.rand(3, 4)).requires_grad_()
    bias = 1.7 * (gate + 1e-6).log()
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[0] = False
    mask[1, 2] = False
    if mask_kind == "pairs":
        mask = mask[:, None].expand(-1, 4, -1).clone()
        mask[2, 1] = False
    actual = module(x, mask, key_bias=bias)
    expected = _oracle_attention(module, x, mask, bias)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    params = (x, gate, *module.parameters())
    upstream = torch.randn_like(actual)
    a = torch.autograd.grad((actual * upstream).sum(), params, retain_graph=True)
    b = torch.autograd.grad((expected * upstream).sum(), params)
    for p, q in zip(a, b):
        assert torch.isfinite(p).all()
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-6)


def test_bias_redistributes_attention_without_scaling_values():
    module = Attention(4, 2, with_qkv=False)
    logits = torch.zeros(1, 2, 3, 3)
    v = torch.ones(1, 2, 3, 2)
    gate = torch.tensor([[.81, 1., 1.25]], requires_grad=True)
    weights, _ = module._weights(logits, None, gate.log())
    expected = (gate / gate.sum(-1, keepdim=True))[:, None, None].expand_as(weights)
    torch.testing.assert_close(weights, expected)
    output = module.aggregate_logits(logits, v, key_bias=gate.log())
    torch.testing.assert_close(output, torch.ones_like(output))
    # Uniform source-key bias cancels, unlike multiplying V by a uniform gate.
    original = module.aggregate_logits(logits, v)
    uniform = module.aggregate_logits(logits, v, key_bias=torch.full((1, 3), .2))
    torch.testing.assert_close(original, uniform, atol=0, rtol=0)


def _oracle_serial(model, features, mask, gate, temporal_scale, spatial_scale):
    b, t, n, d = features.shape
    block = model.blocks[0]
    temporal = features.transpose(1, 2).reshape(b * n, t, d)
    temporal_mask = mask.transpose(1, 2).reshape(b * n, t)
    # Use literal log(g+eps), without production's neutral centering.
    prior = (gate + 1e-6).log()
    temporal_bias = temporal_scale * prior.transpose(1, 2).reshape(b * n, t)
    temporal = _oracle_attention(block.temporal_attn, block.temporal_norm1(temporal),
                                 temporal_mask, temporal_bias)
    temporal = block.temporal_fc(temporal.reshape(b, n * t, d))
    after_temporal = features.transpose(1, 2).reshape(b, n * t, d) + temporal
    spatial = after_temporal.reshape(b, n, t, d).transpose(1, 2).reshape(b * t, n, d)
    spatial_mask = mask.reshape(b * t, n)
    spatial_gate = gate.reshape(b * t, n)
    cls = model.cfg.MODEL.USE_CLS_TOKEN
    if cls:
        spatial = torch.cat((model.cls_token.expand(b * t, -1, -1), spatial), dim=1)
        spatial_mask = torch.cat((mask.new_ones(b * t, 1), spatial_mask), dim=1)
        spatial_gate = torch.cat((gate.new_ones(b * t, 1), spatial_gate), dim=1)
    spatial = _oracle_attention(block.attn, block.norm1(spatial), spatial_mask,
                                spatial_scale * (spatial_gate + 1e-6).log())
    if cls:
        cls_residual = spatial[:, 0].reshape(b, t, d).mean(1, keepdim=True)
        spatial = spatial[:, 1:]
    spatial = spatial.reshape(b, t, n, d).transpose(1, 2).reshape(b, n * t, d)
    encoded = after_temporal + spatial
    if cls:
        encoded = torch.cat((model.cls_token.expand(b, -1, -1) + cls_residual, encoded), dim=1)
    encoded = model.norm(encoded + block.mlp(block.norm2(encoded)))
    if cls:
        encoded = encoded[:, 1:]
    return encoded.reshape(b, n, t, d).transpose(1, 2)


@pytest.mark.parametrize("cls", [False, True])
@pytest.mark.parametrize("visibility", [False, True])
@pytest.mark.parametrize("scales", [(1., 0.), (0., 1.), (.7, 1.3)])
def test_serial_temporal_then_spatial_matches_independent_formula(cls, visibility, scales):
    model = _new_model(cls=cls).eval()
    block = model.blocks[0]
    block.use_pt_visibility = block.temporal_attn.use_mask = block.attn.use_mask = visibility
    router = model.label_pre_temporal
    router.temporal_bias_scale, router.spatial_bias_scale = scales
    features, _, meta = _episode()
    features.requires_grad_()
    mask = meta["pred_query_mask"].clone()
    mask[0] = False
    mask[1, :, 0] = False
    mask[2, 1] = False
    gate = (.81 + .44 * torch.rand_like(mask, dtype=torch.float32)).requires_grad_()
    actual = _encode(model, *_prepare(model, features, mask), gate)
    expected = _oracle_serial(model, features, mask, gate, *scales)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    params = (features, gate, *block.parameters())
    upstream = torch.randn_like(actual)
    a = torch.autograd.grad((actual * upstream).sum(), params, retain_graph=True)
    b = torch.autograd.grad((expected * upstream).sum(), params)
    for p, q in zip(a, b):
        assert torch.isfinite(p).all()
        torch.testing.assert_close(p, q, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("cls", [False, True])
def test_zero_scales_restore_original_encoder_even_with_nonuniform_gates(cls):
    model = _new_model(cls=cls).eval()
    router = model.label_pre_temporal
    router.temporal_bias_scale = router.spatial_bias_scale = 0.
    features, raw, meta = _episode()
    mask = meta["pred_query_mask"]
    gate = router(raw, torch.eye(4)[:2], mask)[:, 0]
    actual = _encode(model, *_prepare(model, features, mask), gate)
    expected = model.pt_forward(features, meta)[1]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("scales", [(1., 0.), (0., 1.)])
def test_each_bias_independently_changes_episode_and_trains_router(scales):
    model = _new_model(classification=True).eval()
    model.label_pre_temporal.temporal_bias_scale, model.label_pre_temporal.spatial_bias_scale = scales
    neutral = copy.deepcopy(model)
    neutral.label_pre_temporal.temporal_bias_scale = neutral.label_pre_temporal.spatial_bias_scale = 0.
    features, raw, meta = _episode()
    actual = label_conditioned_episode(model, features, raw, meta)
    baseline = label_conditioned_episode(neutral, features, raw, meta)
    assert not torch.allclose(actual["query_partial_q2s_base_logits"], baseline["query_partial_q2s_base_logits"])
    _loss(model, actual, meta).backward()
    for p in model.label_pre_temporal.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["TEMPORAL_BIAS_SCALE", "SPATIAL_BIAS_SCALE"])
@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_bad_bias_scales_rejected(name, value):
    model = _new_model()
    model.cfg.FEW_SHOT.LABEL_PRE_TEMPORAL[name] = value
    with pytest.raises(ValueError, match=name):
        model._init_label_pre_temporal()
