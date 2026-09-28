"""Empty observations must be safe in both attention and routed prototypes."""
from types import SimpleNamespace

import pytest
import torch

from trokens.config.defaults import get_cfg
from trokens.datasets.base_ds import _build_point_query_mask, _load_point_queries
from trokens.models.attention import Attention
from trokens.models.pointformer import Pointformer
from trokens.utils.amp import checked_amp_scale


@pytest.mark.parametrize("scale", [7.62939453125e-6, 0.0, float('nan'), float('inf')])
def test_amp_guard_rejects_collapsed_or_invalid_restored_scale(scale):
    scaler = SimpleNamespace(is_enabled=lambda: True, get_scale=lambda: scale)
    with pytest.raises(RuntimeError, match="Do not resume"):
        checked_amp_scale(scaler, min_scale=1e-4)


def test_amp_guard_preserves_valid_scales_and_fp32():
    scaler = SimpleNamespace(is_enabled=lambda: True, get_scale=lambda: 65536.)
    assert checked_amp_scale(scaler, 1e-4) == 65536.
    scaler.is_enabled = lambda: False
    assert checked_amp_scale(scaler, 1e-4) == 1.


def test_visibility_keeps_backward_observations_but_not_invisible_points():
    seeds = [2, 2, 3]
    visible = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 1, 0], [1, 1, 1]]).bool()
    for key in ["point_queries", "per_point_queries"]:
        queries = _load_point_queries({key: seeds}, 3)
        mask = _build_point_query_mask(queries, visible, "visibility")
        assert torch.equal(mask, visible)
        assert mask[0, 0] and not mask[1].any()
    seed_mask = _build_point_query_mask(seeds, visible, "seed")
    assert torch.equal(seed_mask, torch.tensor(
        [[0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 1]]).bool())
    assert _build_point_query_mask(seeds, visible, "all").all()
    with pytest.raises(ValueError, match="QUERY_MASK_MODE"):
        _build_point_query_mask(seeds, visible, "guess_from_key")


def test_datasets_explicitly_share_visibility_policy():
    for ds in ["sav", "tinyvirat"]:
        cfg = get_cfg()
        cfg.merge_from_file(f"configs/trokens/{ds}.yaml")
        assert cfg.POINT_INFO.USE_PT_QUERY_MASK
        assert cfg.POINT_INFO.QUERY_MASK_MODE == "visibility"
        assert not cfg.TRAIN.AUTO_RESUME
        assert cfg.TRAIN.CHECKPOINT_FILE_PATH == ""


@pytest.mark.parametrize("pairwise", [False, True])
def test_attention_empty_keys_forward_backward_and_nonempty_equivalence(pairwise):
    torch.manual_seed(8)
    attn = Attention(8, num_heads=2, qkv_bias=True, use_mask=True)
    x = torch.randn(2, 3, 8, requires_grad=True)
    mask = torch.tensor([[False, False, False], [True, False, True]])
    if pairwise:
        mask = mask[:, None, :].expand(-1, 3, -1).clone()
        mask[1, 1] = False
    y = attn(x, mask)
    valid = mask.any(-1)
    if not pairwise:
        valid = valid[:, None].expand(-1, 3)
    assert torch.isfinite(y).all()
    assert torch.equal(y[~valid], torch.zeros_like(y[~valid]))
    # Compare old mathematics only where a row has at least one key.
    q, k, v = attn.qkv(x).reshape(2, 3, 3, 2, 4).permute(2, 0, 3, 1, 4)
    scores = (q @ k.transpose(-2, -1)) * attn.scale
    m = mask[:, None, :, :] if pairwise else mask[:, None, None, :]
    # The reference graph is not differentiated: its empty rows are unsafe.
    ref = (scores.masked_fill(~m, float('-inf')).softmax(-1) @ v)
    ref = attn.proj(ref.transpose(1, 2).reshape(2, 3, 8))
    torch.testing.assert_close(y[valid], ref[valid])
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0].eq(0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in attn.parameters())


def test_weighted_pool_preserves_subunit_weight_normalization():
    model = Pointformer.__new__(Pointformer)
    feat = torch.tensor([[[2., 0.], [0., 4.]], [[1., 2.], [3., 4.]]], requires_grad=True)
    mask = torch.tensor([[True, True], [False, False]])
    weights = torch.tensor([[.1, .3], [.2, .1]], requires_grad=True)
    out = model._aggregate_weighted_st_support_tokens(feat, mask, weights)
    torch.testing.assert_close(out[0], torch.tensor([.5, 3.]))
    assert out[1].eq(0).all()
    (out.sum() * 65536).backward()
    assert feat.grad[1].eq(0).all() and weights.grad[1].eq(0).all()
    assert torch.isfinite(feat.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batched", [False, True])
def test_autocast_empty_prototypes_scaled_backward(dtype, batched):
    model = Pointformer.__new__(Pointformer)
    model.pot_route_cfg = SimpleNamespace(FRAME_SOFTMAX_TAU=.04)
    torch.manual_seed(1)
    # Actual model patch values are FP32 inside an active autocast region.
    tokens = torch.randn(2, 3, 6, 8, device="cuda", requires_grad=True)
    scores = torch.randn(2, 2, 3, 6, device="cuda", requires_grad=True)
    mask = torch.ones(2, 3, 6, device="cuda", dtype=torch.bool)
    mask[0, 0] = False
    mask[1] = False
    with torch.autocast("cuda", dtype=dtype):
        if batched:
            proto, weights = model._compute_batched_frame_softmax_prototypes_from_similarity(
                tokens, mask, scores)
        else:
            pairs = [model._compute_frame_softmax_prototypes_from_similarity(t, m, s)
                     for t, m, s in zip(tokens, mask, scores)]
            proto, weights = (torch.stack([p[i] for p in pairs]) for i in [0, 1])
        # Zero-vector normalization recreates the large upstream gradient that
        # formerly overflowed on the fallback einsum's FP16 backward path.
        loss = model._safe_l2_normalize(proto).sum()
    assert torch.isfinite(proto).all()
    assert proto[0, :, 0].eq(0).all() and proto[1].eq(0).all()
    assert weights[1].eq(0).all()
    (loss * 65536).backward()
    assert torch.isfinite(tokens.grad).all() and torch.isfinite(scores.grad).all()
    assert tokens.grad[~mask].eq(0).all()
    assert tokens.grad[mask].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_attention_autocast_completely_invisible_trajectory():
    attn = Attention(8, num_heads=2, qkv_bias=True, use_mask=True).cuda()
    x = torch.randn(2, 8, 8, device="cuda", requires_grad=True)
    mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    mask[0] = False
    with torch.autocast("cuda", dtype=torch.float16):
        y = attn(x, mask)
        loss = y.float().square().mean()
    (loss * 1024).backward()
    assert y[0].eq(0).all() and torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all() and x.grad[0].eq(0).all()
    assert all(torch.isfinite(p.grad).all() for p in attn.parameters())
