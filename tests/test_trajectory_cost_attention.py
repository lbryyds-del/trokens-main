"""Unit tests for visual-guided trajectory class-response attention."""

import math

import pytest
import torch
from torch.nn import functional as F

from trokens.models.temporal_similarity_refiner import (
    TrajectoryCostAttentionRefiner,
)


def _inputs():
    generator = torch.Generator().manual_seed(2026)
    similarity = torch.rand(2, 3, 8, 4, generator=generator) * 0.8 - 0.4
    tokens = torch.randn(2, 8, 4, 12, generator=generator)
    mask = torch.rand(2, 8, 4, generator=generator) > 0.15
    return similarity, tokens, mask


def _activated_model(**kwargs):
    model = TrajectoryCostAttentionRefiner(12, **kwargs)
    with torch.no_grad():
        model.out_proj.weight.normal_(0.0, 0.3)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_zero_start_shape_dtype_bound_and_parameter_count(dtype):
    similarity, tokens, mask = _inputs()
    model = TrajectoryCostAttentionRefiner(12)
    output, auxiliary = model(
        similarity.to(dtype),
        tokens.to(dtype),
        mask,
        return_aux=True,
    )
    assert output.shape == similarity.shape
    assert output.dtype == torch.float32
    assert torch.equal(output, torch.zeros_like(output))
    assert auxiliary["attention"].shape == (2, 3, 4, 8, 8)
    assert torch.isfinite(auxiliary["attention"]).all()
    assert sum(
        parameter.numel()
        for parameter in TrajectoryCostAttentionRefiner(1024).parameters()
    ) == 17765


def test_active_output_is_finite_and_bounded():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    output = model(similarity, tokens, mask)
    assert torch.isfinite(output).all()
    assert output.abs().max() <= model.max_logit_delta


def test_batch_matches_individual_videos():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    batched = model(similarity, tokens, mask)
    stacked = torch.stack(
        [
            model(similarity[index], tokens[index], mask[index])
            for index in range(similarity.shape[0])
        ]
    )
    torch.testing.assert_close(batched, stacked, atol=1e-7, rtol=1e-5)


def test_class_and_track_permutations_are_equivariant():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    output = model(similarity, tokens, mask)
    class_order = torch.tensor([2, 0, 1])
    track_order = torch.tensor([3, 1, 0, 2])
    permuted = model(
        similarity[:, class_order, :, :][:, :, :, track_order],
        tokens[:, :, track_order],
        mask[:, :, track_order],
    )
    torch.testing.assert_close(
        permuted,
        output[:, class_order, :, :][:, :, :, track_order],
    )


def test_evidence_never_mixes_across_video_class_or_track():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    original = model(similarity, tokens, mask)
    changed = similarity.clone()
    changed[0, 0, :, 0] += 0.3
    perturbed = model(changed, tokens, mask)
    torch.testing.assert_close(original[1], perturbed[1], rtol=0, atol=0)
    torch.testing.assert_close(
        original[0, 1:],
        perturbed[0, 1:],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original[0, 0, :, 1:],
        perturbed[0, 0, :, 1:],
        rtol=0,
        atol=0,
    )


def test_invalid_observation_cannot_pollute_valid_neighbors():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    mask = mask.clone()
    mask[:, 3, 0] = False
    original = model(similarity, tokens, mask)

    changed_similarity = similarity.clone()
    changed_similarity[:, :, 3, 0] = 1.0
    changed_tokens = tokens.clone()
    changed_tokens[:, 3, 0] = 100.0
    perturbed = model(changed_similarity, changed_tokens, mask)
    torch.testing.assert_close(original, perturbed, rtol=0, atol=0)
    assert torch.equal(
        perturbed[:, :, 3, 0],
        torch.zeros_like(perturbed[:, :, 3, 0]),
    )


def test_all_invalid_single_valid_and_isolated_valid_rows_are_exact_noops():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    invalid = torch.zeros_like(mask)
    output, auxiliary = model(
        similarity,
        tokens,
        invalid,
        return_aux=True,
    )
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(
        auxiliary["attention"],
        torch.zeros_like(auxiliary["attention"]),
    )

    isolated = torch.zeros_like(mask)
    isolated[:, 0] = True
    isolated[:, 4] = True
    assert torch.equal(
        model(similarity, tokens, isolated),
        torch.zeros_like(output),
    )


def test_attention_is_local_with_radius_two():
    similarity, tokens, mask = _inputs()
    mask = torch.ones_like(mask)
    model = _activated_model(radius=2)
    original, auxiliary = model(
        similarity,
        tokens,
        mask,
        return_aux=True,
    )
    attention = auxiliary["attention"]
    time = torch.arange(similarity.shape[2])
    nonlocal_pairs = (time[None, :] - time[:, None]).abs() > 2
    assert torch.equal(
        attention[..., nonlocal_pairs],
        torch.zeros_like(attention[..., nonlocal_pairs]),
    )
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
    )

    changed_similarity = similarity.clone()
    changed_similarity[:, :, 7] += 0.3
    changed_tokens = tokens.clone()
    changed_tokens[:, 7] *= -2
    perturbed = model(changed_similarity, changed_tokens, mask)
    torch.testing.assert_close(
        original[:, :, :5],
        perturbed[:, :, :5],
        rtol=0,
        atol=0,
    )


def test_sliced_local_implementation_matches_dense_attention_formula():
    similarity, tokens, mask = _inputs()
    mask = mask.clone()
    mask[:, 3, 0] = False
    model = _activated_model()
    output, auxiliary = model(
        similarity,
        tokens,
        mask,
        return_aux=True,
    )

    scores = similarity.detach().float().permute(0, 1, 3, 2)
    valid_bnt = mask.permute(0, 2, 1)
    cost = model.cost_embed(scores.unsqueeze(-1))
    cost = F.layer_norm(cost.float(), (model.cost_dim,))
    cost = cost * valid_bnt[:, None, :, :, None]
    visual = F.normalize(tokens.detach().float(), dim=-1, eps=1e-12)
    guidance = model.visual_norm(model.visual_proj(visual)).float()
    guidance = guidance.permute(0, 2, 1, 3)
    guidance = guidance[:, None].expand(-1, scores.shape[1], -1, -1, -1)
    qk = torch.cat((cost, guidance), dim=-1)
    query = F.layer_norm(model.q_proj(qk).float(), (model.attention_dim,))
    key = F.layer_norm(model.k_proj(qk).float(), (model.attention_dim,))
    logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
        model.attention_dim
    )
    time = torch.arange(similarity.shape[2])
    offset = time[None, :] - time[:, None]
    local = offset.abs() <= model.radius
    bias_index = offset.clamp(-model.radius, model.radius) + model.radius
    logits = logits + model.relative_bias[bias_index]
    allowed = (
        valid_bnt[..., :, None]
        & valid_bnt[..., None, :]
        & local[None, None]
    )
    weights = torch.softmax(logits.masked_fill(~allowed[:, None], -1e4), dim=-1)
    weights = weights * allowed[:, None]
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    context = torch.matmul(weights, cost)
    has_neighbor = (
        allowed
        & ~torch.eye(similarity.shape[2], dtype=torch.bool)
    ).any(dim=-1)
    dense_delta = model.max_logit_delta * torch.tanh(
        model.out_proj(context - cost).squeeze(-1)
    )
    dense_delta = dense_delta * has_neighbor[:, None]
    dense_delta = dense_delta.permute(0, 1, 3, 2).contiguous()

    torch.testing.assert_close(auxiliary["attention"], weights)
    torch.testing.assert_close(output, dense_delta)


def test_neighbor_and_visual_guidance_affect_attention_after_activation():
    similarity, tokens, mask = _inputs()
    mask = torch.ones_like(mask)
    model = _activated_model()
    output, first_aux = model(
        similarity,
        tokens,
        mask,
        return_aux=True,
    )

    changed_similarity = similarity.clone()
    changed_similarity[:, :, 2] = 0.95
    neighbor_changed = model(changed_similarity, tokens, mask)
    assert (output[:, :, 3] - neighbor_changed[:, :, 3]).abs().max() > 1e-6

    changed_tokens = tokens.clone()
    changed_tokens[:, 2] *= -1
    _, second_aux = model(
        similarity,
        changed_tokens,
        mask,
        return_aux=True,
    )
    assert (
        first_aux["attention"] - second_aux["attention"]
    ).abs().max() > 1e-7


def test_changing_only_cross_frame_track_correspondence_changes_correction():
    similarity, tokens, mask = _inputs()
    mask = torch.ones_like(mask)
    model = _activated_model()
    original = model(similarity, tokens, mask)

    routed_similarity = similarity.clone()
    routed_tokens = tokens.clone()
    routed_mask = mask.clone()
    permutations = []
    generator = torch.Generator().manual_seed(91)
    for frame in range(similarity.shape[2]):
        permutation = torch.randperm(similarity.shape[3], generator=generator)
        permutations.append(permutation)
        routed_similarity[:, :, frame] = similarity[:, :, frame, permutation]
        routed_tokens[:, frame] = tokens[:, frame, permutation]
        routed_mask[:, frame] = mask[:, frame, permutation]

    routed_delta = model(routed_similarity, routed_tokens, routed_mask)
    restored_delta = torch.empty_like(routed_delta)
    for frame, permutation in enumerate(permutations):
        restored_delta[:, :, frame, permutation] = routed_delta[:, :, frame]

    assert (original - restored_delta).abs().max() > 1e-6


def test_constant_class_response_does_not_create_temporal_evidence():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    output = model(torch.full_like(similarity, 0.27), tokens, torch.ones_like(mask))
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=1e-6)


def test_inputs_are_detached_from_the_additional_branch():
    similarity, tokens, mask = _inputs()
    similarity.requires_grad_()
    tokens.requires_grad_()
    model = _activated_model()
    output = model(similarity, tokens, mask)
    (output * torch.randn_like(output)).sum().backward()
    assert similarity.grad is None
    assert tokens.grad is None
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_zero_start_first_update_reaches_only_output_projection():
    similarity, tokens, mask = _inputs()
    model = TrajectoryCostAttentionRefiner(12)
    output = model(similarity, tokens, torch.ones_like(mask))
    (output * torch.randn_like(output)).sum().backward()
    assert model.out_proj.weight.grad is not None
    assert model.out_proj.weight.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if name == "out_proj.weight":
            continue
        assert parameter.grad is not None, name
        assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name


def test_all_parameters_train_after_output_projection_activates():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    output = model(similarity, tokens, torch.ones_like(mask))
    (output * torch.randn_like(output)).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize(
    "shape",
    [(0, 3, 8, 4), (2, 0, 8, 4), (2, 3, 0, 4), (2, 3, 8, 0)],
)
def test_empty_axes_are_finite(shape):
    batch, classes, frames, tracks = shape
    model = TrajectoryCostAttentionRefiner(12)
    output = model(
        torch.zeros(shape),
        torch.zeros(batch, frames, tracks, 12),
        torch.ones(batch, frames, tracks, dtype=torch.bool),
    )
    assert output.shape == shape
    assert torch.isfinite(output).all()


def test_relative_bias_initialization_and_radius_zero_behavior():
    model = TrajectoryCostAttentionRefiner(12, radius=2)
    torch.testing.assert_close(
        model.relative_bias.detach(),
        torch.tensor([-1.0, -0.5, 0.0, -0.5, -1.0]),
    )
    similarity, tokens, mask = _inputs()
    radius_zero = _activated_model(radius=0)
    assert torch.equal(
        radius_zero(similarity, tokens, torch.ones_like(mask)),
        torch.zeros_like(similarity),
    )


@pytest.mark.parametrize("frames,radius", [(1, 2), (1, 4), (2, 2), (2, 4), (3, 4)])
def test_short_clips_and_radius_beyond_clip_length(frames, radius):
    model = _activated_model(radius=radius)
    similarity = torch.randn(2, 3, frames, 4).clamp(-1, 1)
    tokens = torch.randn(2, frames, 4, 12)
    mask = torch.ones(2, frames, 4, dtype=torch.bool)
    delta, aux = model(similarity, tokens, mask, return_aux=True)
    assert delta.shape == similarity.shape
    assert aux["attention"].shape == (2, 3, 4, frames, frames)
    assert torch.isfinite(delta).all()
    torch.testing.assert_close(
        aux["attention"].sum(dim=-1),
        torch.ones(2, 3, 4, frames),
    )
    if frames == 1:
        assert torch.equal(delta, torch.zeros_like(delta))
    delta.square().sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


def test_nonfinite_and_zero_input_stay_finite_under_amp():
    model = _activated_model()
    similarity, tokens, mask = _inputs()
    similarity[:, :, 0, 0] = float("nan")
    similarity[:, :, 1, 0] = float("inf")
    similarity[:, :, 2, 0] = -float("inf")
    tokens[:, 0, 0] = float("nan")
    tokens[:, 1, 0] = float("inf")
    tokens[:, 2, 0] = 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        delta = model(similarity, tokens.bfloat16(), mask)
    assert torch.isfinite(delta).all()
    delta.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 0},
        {"cost_dim": 0},
        {"guidance_dim": 0},
        {"attention_dim": 0},
        {"radius": -1},
        {"radius": 1.5},
        {"max_logit_delta": -1},
        {"max_logit_delta": float("nan")},
    ],
)
def test_invalid_hyperparameters_fail(kwargs):
    kwargs = kwargs.copy()
    feature_dim = kwargs.pop("feature_dim", 12)
    with pytest.raises(ValueError):
        TrajectoryCostAttentionRefiner(feature_dim, **kwargs)


def test_shape_and_parameter_dtype_errors_are_explicit():
    model = TrajectoryCostAttentionRefiner(12)
    with pytest.raises(ValueError, match="K,T,N"):
        model(torch.zeros(2, 3), torch.zeros(2, 3, 12), torch.ones(2, 3))
    with pytest.raises(ValueError, match="feature_dim"):
        model(
            torch.zeros(2, 3, 4),
            torch.zeros(3, 5, 12),
            torch.ones(3, 4),
        )
    with pytest.raises(ValueError, match="parameters FP32"):
        model.half()(
            torch.zeros(2, 3, 4),
            torch.zeros(3, 4, 12),
            torch.ones(3, 4),
        )


def test_cpu_autocast_forward_backward_is_finite():
    similarity, tokens, mask = _inputs()
    model = _activated_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(similarity, tokens, mask)
        loss = output.square().sum()
    loss.backward()
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_forward_backward_is_finite():
    similarity, tokens, mask = _inputs()
    similarity = similarity.cuda()
    tokens = tokens.cuda()
    mask = mask.cuda()
    model = _activated_model().cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(similarity, tokens, mask)
        loss = output.square().sum()
    loss.backward()
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
