"""Three-channel temporal residual routing and deterministic fast path."""

from pathlib import Path

import pytest
import torch
from torch import nn

from trokens.config.defaults import get_cfg
from trokens.models.pointformer import Pointformer
from trokens.models.optimizer import construct_optimizer
from trokens.models.temporal_similarity_refiner import (
    TrajectoryTemporalSimilarityRefiner,
)


def _sav_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / "configs/trokens/sav.yaml"))
    return cfg


def _model(enable=True, learned=False, matchability=True):
    model = Pointformer.__new__(Pointformer)
    nn.Module.__init__(model)
    model.cfg = _sav_cfg()
    model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE = matchability
    model.pot_route_cfg = model.cfg.FEW_SHOT.POT_ROUTE
    model.pot_route_cfg.TEMPORAL_REFINEMENT.ENABLE = enable
    model.use_frame_softmax_route = True
    model.use_query_partial_q2s = True
    model.use_cat_cost_aggregation = False
    model.use_query_null_route = False
    model.use_support_text_fusion = True
    model.support_text_fusion_cfg = model.cfg.FEW_SHOT.SUPPORT_TEXT_FUSION
    model._get_pot_label_text_features = lambda ids, dtype: torch.eye(2, dtype=dtype)
    model._init_temporal_similarity_refinement()
    if learned and model.temporal_similarity_refiner is not None:
        with torch.no_grad():
            model.temporal_similarity_refiner.out_proj.weight.fill_(0.1)
    return model


def _episode():
    generator = torch.Generator().manual_seed(77)
    post = torch.randn(3, 4, 3, 2, generator=generator)
    raw = torch.randn(3, 4, 3, 2, generator=generator)
    mask = torch.ones(3, 4, 3, dtype=torch.bool)
    mask[0, 0, 2] = False
    visibility = torch.ones_like(mask)
    visibility[0, 1, 0] = False
    visibility[1, 2, 1] = False
    visibility[2, 1, 2] = False
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": mask,
        "pred_visibility": visibility,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor([[1, 0], [0, 1], [1, 1]]).bool(),
    }
    return post, raw, metadata


def _run(model, post, raw, metadata):
    return model._build_frame_softmax_q2s_aux(
        post, metadata, matchability_evidence_tokens=raw,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_shape_dtype_range_and_bounded_correction(dtype):
    model = TrajectoryTemporalSimilarityRefiner()
    x = torch.linspace(-1, 1, 60).reshape(3, 5, 4).to(dtype)
    y = model(x, torch.ones(5, 4, dtype=torch.bool))
    assert y.shape == x.shape and y.dtype == torch.float32
    assert torch.isfinite(y).all()
    assert y.min() >= -model.max_logit_delta
    assert y.max() <= model.max_logit_delta
    assert sum(p.numel() for p in model.parameters()) == 960


@pytest.mark.parametrize("shape", [(0, 4, 3), (2, 0, 3), (2, 4, 0), (2, 1, 3)])
def test_empty_axes_and_single_frame(shape):
    x = torch.zeros(shape)
    y = TrajectoryTemporalSimilarityRefiner()(x, torch.ones(shape[1:], dtype=torch.bool))
    assert y.shape == shape and torch.isfinite(y).all()


def test_zero_strength_returns_zero_residual():
    x = torch.linspace(-1, 1, 30).reshape(2, 5, 3)
    y = TrajectoryTemporalSimilarityRefiner(max_logit_delta=0.0)(
        x, torch.ones(5, 3).bool(),
    )
    assert torch.equal(y, torch.zeros_like(y))


def test_zero_initialized_projection_returns_zero_residual():
    x = torch.linspace(-1, 1, 30).reshape(2, 5, 3)
    y = TrajectoryTemporalSimilarityRefiner()(x, torch.ones(5, 3).bool())
    assert torch.equal(y, torch.zeros_like(y))


def test_no_class_or_trajectory_mixing_and_permutation_equivariance():
    torch.manual_seed(11)
    model = TrajectoryTemporalSimilarityRefiner()
    with torch.no_grad():
        model.out_proj.weight.fill_(0.1)
    x = torch.rand(2, 5, 3)
    mask = torch.rand(5, 3) > 0.2
    y = model(x, mask)
    changed = x.clone()
    changed[0, :, 0] = -0.5
    other = model(changed, mask)
    assert torch.equal(y[1], other[1])
    assert torch.equal(y[0, :, 1:], other[0, :, 1:])
    order = torch.tensor([2, 0, 1])
    assert torch.equal(model(x[:, :, order], mask[:, order]), y[:, :, order])


def test_batched_forward_matches_stacked_sample_forward():
    torch.manual_seed(13)
    model = TrajectoryTemporalSimilarityRefiner()
    with torch.no_grad():
        model.out_proj.weight.fill_(0.1)
    x = torch.rand(4, 2, 5, 3)
    mask = torch.rand(4, 5, 3) > 0.2
    batched = model(x, mask)
    stacked = torch.stack([model(x[idx], mask[idx]) for idx in range(4)])
    assert torch.allclose(batched, stacked, atol=1e-6, rtol=1e-5)


def test_linear_conv_matches_native_conv():
    torch.manual_seed(15)
    model = TrajectoryTemporalSimilarityRefiner()
    value = torch.randn(7, 16, 8)
    expected = model.long_temporal_conv(value)
    actual = model._linear_conv1d(value, model.long_temporal_conv)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_neighbors_really_affect_current_similarity_with_local_receptive_field():
    model = TrajectoryTemporalSimilarityRefiner(hidden_dim=1)
    with torch.no_grad():
        model.temporal_conv.weight.zero_()
        model.temporal_conv.weight[0, 0, 0] = 1.0  # Previous frame only.
        model.temporal_conv.bias.zero_()
        model.long_temporal_conv.weight.zero_()
        model.long_temporal_conv.weight[0, 0, 1] = 1.0
        model.long_temporal_conv.bias.zero_()
        model.out_proj.weight.fill_(1.0)
    x = torch.zeros(1, 6, 2)
    mask = torch.ones(6, 2).bool()
    original = model(x, mask)
    x[0, 2, 0] = 0.8
    changed = model(x, mask)
    assert changed[0, 3, 0] > original[0, 3, 0]
    assert torch.equal(changed[:, 4:], original[:, 4:])
    assert torch.equal(changed[:, :, 1], original[:, :, 1])


@pytest.mark.parametrize("use_visibility", [False, True])
def test_invalid_observations_cannot_pollute_neighbors_even_without_mask_channel(use_visibility):
    torch.manual_seed(17)
    model = TrajectoryTemporalSimilarityRefiner(use_visibility=use_visibility)
    with torch.no_grad():
        model.out_proj.weight.fill_(0.1)
    x = torch.rand(2, 5, 3)
    mask = torch.ones(5, 3).bool()
    mask[2, 0] = False
    y = x.clone()
    x[:, 2, 0], y[:, 2, 0] = -1, 1
    out_x, out_y = model(x, mask), model(y, mask)
    valid = mask.unsqueeze(0).expand_as(x)
    assert torch.equal(out_x[valid], out_y[valid])
    assert torch.equal(out_x[:, 2, 0], torch.zeros_like(out_x[:, 2, 0]))
    assert torch.equal(out_y[:, 2, 0], torch.zeros_like(out_y[:, 2, 0]))
    assert torch.equal(
        model(x, torch.zeros_like(mask)),
        torch.zeros_like(out_x),
    )


def test_nonfinite_input_is_sanitized():
    x = torch.tensor([[[float("nan"), float("inf"), -float("inf")]]])
    y = TrajectoryTemporalSimilarityRefiner()(x, torch.ones(1, 3).bool())
    assert torch.isfinite(y).all() and y.min() >= -1 and y.max() <= 1


@pytest.mark.parametrize("kwargs", [
    {"kernel_size": 2}, {"kernel_size": 0}, {"kernel_size": 1.5},
    {"hidden_dim": 0}, {"long_dilation": 0},
    {"max_logit_delta": -1}, {"max_logit_delta": float("nan")},
])
def test_invalid_hyperparameters_fail(kwargs):
    with pytest.raises(ValueError):
        TrajectoryTemporalSimilarityRefiner(**kwargs)


def test_shape_errors_are_explicit():
    model = TrajectoryTemporalSimilarityRefiner()
    with pytest.raises(ValueError, match="K,T,N"):
        model(torch.zeros(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="T,N"):
        model(torch.zeros(2, 3, 4), torch.ones(3, 5))


def test_all_temporal_parameters_receive_gradients_after_projection_activates():
    torch.manual_seed(19)
    model = TrajectoryTemporalSimilarityRefiner()
    with torch.no_grad():
        model.out_proj.weight.fill_(0.1)
    x = torch.rand(2, 5, 3, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(x, torch.ones(5, 3).bool())
        loss = output.square().sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


@pytest.mark.parametrize("matchability", [False, True])
def test_zero_projection_and_disabled_branch_preserve_complete_legacy_outputs(matchability):
    post, raw, metadata = _episode()
    disabled = _model(enable=False, matchability=matchability)
    enabled = _model(matchability=matchability)
    old = _run(disabled, post, raw, metadata)
    new = _run(enabled, post, raw, metadata)
    assert old.keys() == new.keys()
    for key in old:
        assert torch.equal(old[key], new[key]), key
    enabled.use_temporal_similarity_refinement = False
    enabled.temporal_similarity_refiner.forward = lambda *args: pytest.fail("disabled branch called")
    off = _run(enabled, post, raw, metadata)
    for key in old:
        assert torch.equal(old[key], off[key]), key


def test_shared_support_query_and_reference_router_uses_real_visibility():
    model = _model()
    post, raw, metadata = _episode()
    calls = []
    hook = model.temporal_similarity_refiner.register_forward_pre_hook(
        lambda module, args: calls.append((args[0].detach().clone(), args[1].clone()))
    )
    result = _run(model, post, raw, metadata)
    hook.remove()
    expected_mask = metadata["pred_query_mask"] & metadata["pred_visibility"]
    assert len(calls) == 2
    assert calls[0][0].shape == (2, 2, 4, 3)
    assert calls[1][0].shape == (1, 2, 4, 3)
    assert torch.equal(calls[0][1], expected_mask[:2])
    assert torch.equal(calls[1][1], expected_mask[2:])

    fused = result["support_text_fusion_query_features"]
    expected_proto, expected_weights = model._compute_frame_softmax_text_prototypes(
        post[2], metadata["pred_query_mask"][2], fused,
        temporal_point_mask=expected_mask[2],
    )
    assert torch.allclose(result["query_partial_query_prototypes"][0], expected_proto)
    # Temporal invisibility does not silently change the original spatial mask.
    assert expected_weights[:, 1, 2].gt(0).all()
    assert "query_evidence_patch_weights" not in result


def test_learned_route_changes_prototypes_without_query_label_leakage():
    torch.manual_seed(29)
    model = _model(learned=True)
    post, raw, metadata = _episode()
    new = _run(model, post, raw, metadata)
    old = _run(_model(enable=False), post, raw, metadata)
    assert not torch.equal(new["query_partial_query_prototypes"], old["query_partial_query_prototypes"])
    assert "query_frame_absolute_patch_mass" not in new
    changed = dict(metadata)
    changed["episode_positive_labels"] = metadata["episode_positive_labels"].clone()
    changed["episode_positive_labels"][2] = False
    other = _run(model, post, raw, changed)
    for key in new:
        assert torch.equal(new[key], other[key]), key


def test_spatial_all_invalid_rows_remain_zero_and_precomputed_costs_bypass_refiner():
    model = _model()
    tokens = torch.randn(3, 4, 2)
    mask = torch.ones(3, 4).bool()
    mask[1] = False
    proto, weights = model._compute_frame_softmax_text_prototypes(tokens, mask, torch.eye(2))
    assert torch.equal(proto[:, 1], torch.zeros_like(proto[:, 1]))
    assert torch.equal(weights[:, 1], torch.zeros_like(weights[:, 1]))
    scores = torch.randn(2, 3, 4)
    model.temporal_similarity_refiner.forward = lambda *args: pytest.fail("precomputed route refined")
    expected = _model(enable=False)._compute_frame_softmax_prototypes_from_similarity(tokens, mask, scores)
    actual = model._compute_frame_softmax_prototypes_from_similarity(tokens, mask, scores)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_enabled_full_route_backward_reaches_shared_refiner():
    torch.manual_seed(37)
    model = _model(learned=True)
    post, raw, metadata = _episode()
    post.requires_grad_()
    result = _run(model, post, raw, metadata)
    target = torch.tensor([[1.0, 0.0]])
    loss = nn.functional.binary_cross_entropy_with_logits(result["query_partial_q2s_logits"], target)
    loss.backward()
    assert post.grad is not None and torch.isfinite(post.grad).all()
    for name, param in model.temporal_similarity_refiner.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_default_off_sav_on_registration_rng_and_checkpoint_roundtrip():
    assert not get_cfg().FEW_SHOT.POT_ROUTE.TEMPORAL_REFINEMENT.ENABLE
    assert _sav_cfg().FEW_SHOT.POT_ROUTE.TEMPORAL_REFINEMENT.ENABLE
    model = _model(enable=False)
    assert model.temporal_similarity_refiner is None and not model.state_dict()
    model.pot_route_cfg.TEMPORAL_REFINEMENT.ENABLE = True
    before = torch.random.get_rng_state().clone()
    model._init_temporal_similarity_refinement()
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(key.startswith("temporal_similarity_refiner.") for key in model.state_dict())
    copy = _model()
    copy.load_state_dict(model.state_dict(), strict=True)
    x = torch.rand(2, 4, 3)
    mask = torch.ones(4, 3).bool()
    assert torch.equal(model._refine_trajectory_similarity(x, mask), copy._refine_trajectory_similarity(x, mask))


@pytest.mark.parametrize("route", ["cat", "null", "local", "no_points", "no_q2s"])
def test_incompatible_routes_fail_instead_of_silently_bypassing_temporal_refinement(route):
    model = _model(enable=False)
    model.pot_route_cfg.TEMPORAL_REFINEMENT.ENABLE = True
    if route == "cat":
        model.use_cat_cost_aggregation = True
    elif route == "null":
        model.use_query_null_route = True
    elif route == "local":
        model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LOCAL_REFINEMENT_ENABLE = True
    elif route == "no_points":
        model.cfg.POINT_INFO.ENABLE = False
    else:
        model.use_query_partial_q2s = False
    with pytest.raises(ValueError, match="TEMPORAL_REFINEMENT"):
        model._init_temporal_similarity_refinement()


def test_enabled_episode_requires_real_visibility_but_legacy_does_not():
    post, raw, metadata = _episode()
    del metadata["pred_visibility"]
    with pytest.raises(ValueError, match="pred_visibility"):
        _run(_model(), post, raw, metadata)
    assert _run(_model(enable=False), post, raw, metadata) is not None


def test_real_constructor_preserves_old_weights_and_registers_optimizer_parameters(monkeypatch):
    # Exercise the real Pointformer constructor without loading external
    # DinoTxt checkpoints. All existing trainable branches initialize normally.
    monkeypatch.setattr(Pointformer, "_load_dinotxt_visual_model", lambda self: nn.Identity())
    monkeypatch.setattr(Pointformer, "_load_dinotxt_text_model", lambda self: nn.Identity())
    monkeypatch.setattr(Pointformer, "_load_dinotxt_tokenizer", lambda self: None)
    cfg = _sav_cfg()
    cfg.FEW_SHOT.POT_ROUTE.TEMPORAL_REFINEMENT.ENABLE = False
    torch.manual_seed(51)
    baseline = Pointformer(cfg)
    baseline_rng = torch.random.get_rng_state().clone()
    cfg = cfg.clone()
    cfg.FEW_SHOT.POT_ROUTE.TEMPORAL_REFINEMENT.ENABLE = True
    torch.manual_seed(51)
    temporal = Pointformer(cfg)
    assert torch.equal(baseline_rng, torch.random.get_rng_state())
    base_state, temporal_state = baseline.state_dict(), temporal.state_dict()
    for key in base_state:
        assert torch.equal(base_state[key], temporal_state[key]), key
    extra_keys = set(temporal_state) - set(base_state)
    assert len(extra_keys) == 5
    assert all(key.startswith("temporal_similarity_refiner.") for key in extra_keys)
    old_checkpoint = temporal.load_state_dict(base_state, strict=False)
    assert set(old_checkpoint.missing_keys) == extra_keys
    assert not old_checkpoint.unexpected_keys
    optimizer = construct_optimizer(temporal, cfg)
    optimized_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(
        id(p) in optimized_ids
        for p in temporal.temporal_similarity_refiner.parameters()
    )
