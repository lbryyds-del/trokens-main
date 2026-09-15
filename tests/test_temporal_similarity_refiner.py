"""Integration tests for trajectory cost attention in Pointformer routing."""

from pathlib import Path

import pytest
import torch
from torch import nn

from trokens.config.defaults import get_cfg
from trokens.models.optimizer import construct_optimizer
from trokens.models.pointformer import Pointformer


def _sav_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        str(Path(__file__).resolve().parents[1] / "configs/trokens/sav.yaml")
    )
    return cfg


def _model(enable=True, learned=False, matchability=True):
    model = Pointformer.__new__(Pointformer)
    nn.Module.__init__(model)
    model.cfg = _sav_cfg()
    model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE = matchability
    model.pot_route_cfg = model.cfg.FEW_SHOT.POT_ROUTE
    model.pot_route_cfg.TEMPORAL_REFINEMENT.ENABLE = enable
    model.embed_dim = 2
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
            model.temporal_similarity_refiner.out_proj.weight.copy_(
                torch.linspace(-0.2, 0.2, 16).unsqueeze(0)
            )
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
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [1, 1]]
        ).bool(),
    }
    return post, raw, metadata


def _run(model, post, raw, metadata):
    return model._build_frame_softmax_q2s_aux(
        post,
        metadata,
        matchability_evidence_tokens=raw,
    )


@pytest.mark.parametrize("matchability", [False, True])
def test_zero_projection_and_disabled_branch_preserve_complete_outputs(matchability):
    post, raw, metadata = _episode()
    disabled = _model(enable=False, matchability=matchability)
    enabled = _model(matchability=matchability)
    baseline = _run(disabled, post, raw, metadata)
    zero_start = _run(enabled, post, raw, metadata)
    assert baseline.keys() == zero_start.keys()
    for key in baseline:
        assert torch.equal(baseline[key], zero_start[key]), key

    enabled.use_temporal_similarity_refinement = False
    enabled.temporal_similarity_refiner.forward = (
        lambda *args: pytest.fail("disabled branch called")
    )
    switched_off = _run(enabled, post, raw, metadata)
    for key in baseline:
        assert torch.equal(baseline[key], switched_off[key]), key


def test_shared_support_query_refiner_receives_visual_tokens_and_real_visibility():
    model = _model()
    post, raw, metadata = _episode()
    calls = []

    def record_call(module, args):
        calls.append(tuple(value.detach().clone() for value in args))

    hook = model.temporal_similarity_refiner.register_forward_pre_hook(record_call)
    result = _run(model, post, raw, metadata)
    hook.remove()

    expected_mask = metadata["pred_query_mask"] & metadata["pred_visibility"]
    assert len(calls) == 2
    support_call, query_call = calls
    assert support_call[0].shape == (2, 2, 4, 3)
    assert query_call[0].shape == (1, 2, 4, 3)
    assert torch.equal(support_call[1], post[:2])
    assert torch.equal(query_call[1], post[2:])
    assert torch.equal(support_call[2], expected_mask[:2])
    assert torch.equal(query_call[2], expected_mask[2:])

    fused = result["support_text_fusion_query_features"]
    expected_proto, expected_weights = model._compute_frame_softmax_text_prototypes(
        post[2],
        metadata["pred_query_mask"][2],
        fused,
        temporal_point_mask=expected_mask[2],
    )
    assert torch.allclose(
        result["query_partial_query_prototypes"][0],
        expected_proto,
    )
    # Temporal invisibility remains separate from the spatial aggregation mask.
    assert expected_weights[:, 1, 2].gt(0).all()
    assert "query_evidence_patch_weights" not in result


def test_learned_route_changes_prototypes_without_query_label_leakage():
    torch.manual_seed(29)
    model = _model(learned=True)
    post, raw, metadata = _episode()
    refined = _run(model, post, raw, metadata)
    baseline = _run(_model(enable=False), post, raw, metadata)
    assert not torch.equal(
        refined["query_partial_query_prototypes"],
        baseline["query_partial_query_prototypes"],
    )
    assert "query_frame_absolute_patch_mass" not in refined

    changed = dict(metadata)
    changed["episode_positive_labels"] = metadata[
        "episode_positive_labels"
    ].clone()
    changed["episode_positive_labels"][2] = False
    label_changed = _run(model, post, raw, changed)
    for key in refined:
        assert torch.equal(refined[key], label_changed[key]), key


def test_spatial_invalid_rows_stay_zero_and_precomputed_costs_bypass_refiner():
    model = _model()
    tokens = torch.randn(3, 4, 2)
    mask = torch.ones(3, 4).bool()
    mask[1] = False
    prototypes, weights = model._compute_frame_softmax_text_prototypes(
        tokens,
        mask,
        torch.eye(2),
    )
    assert torch.equal(prototypes[:, 1], torch.zeros_like(prototypes[:, 1]))
    assert torch.equal(weights[:, 1], torch.zeros_like(weights[:, 1]))

    scores = torch.randn(2, 3, 4)
    model.temporal_similarity_refiner.forward = (
        lambda *args: pytest.fail("precomputed route refined")
    )
    expected = _model(
        enable=False
    )._compute_frame_softmax_prototypes_from_similarity(tokens, mask, scores)
    actual = model._compute_frame_softmax_prototypes_from_similarity(
        tokens,
        mask,
        scores,
    )
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_full_route_backward_reaches_refiner_and_keeps_main_visual_gradient():
    torch.manual_seed(37)
    model = _model(learned=True)
    post, raw, metadata = _episode()
    post.requires_grad_()
    result = _run(model, post, raw, metadata)
    target = torch.tensor([[1.0, 0.0]])
    loss = nn.functional.binary_cross_entropy_with_logits(
        result["query_partial_q2s_logits"],
        target,
    )
    loss.backward()
    assert post.grad is not None and torch.isfinite(post.grad).all()
    assert post.grad.abs().sum() > 0
    for name, parameter in model.temporal_similarity_refiner.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_logit_residual_is_multiplied_by_tau_and_raw_similarity_keeps_gradient():
    model = _model()
    similarity = torch.rand(2, 4, 3, requires_grad=True)
    tokens = torch.rand(4, 3, 2, requires_grad=True)
    mask = torch.ones(4, 3, dtype=torch.bool)
    delta = torch.linspace(-0.5, 0.5, similarity.numel()).reshape_as(similarity)
    model.temporal_similarity_refiner.forward = lambda *args: delta
    refined = model._refine_trajectory_similarity(
        similarity, tokens, mask, softmax_tau=0.04,
    )
    torch.testing.assert_close(refined, similarity + 0.04 * delta)
    torch.testing.assert_close(refined / 0.04, similarity / 0.04 + delta)

    # With the actual refiner, raw similarity remains attached but the new
    # residual itself does not add a second gradient path into either input.
    model = _model(learned=True)
    model._refine_trajectory_similarity(similarity, tokens, mask).sum().backward()
    assert torch.equal(similarity.grad, torch.ones_like(similarity))
    assert tokens.grad is None


def test_reenabled_verifier_reference_path_uses_same_visual_guided_refiner():
    model = _model(learned=True)
    match_cfg = model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    match_cfg.MODE = "positive_confuser_margin"
    match_cfg.EVIDENCE_VERIFICATION_ENABLE = True
    match_cfg.ABSOLUTE_MASS_ENABLE = True
    post, raw, metadata = _episode()
    expected_mask = metadata["pred_query_mask"] & metadata["pred_visibility"]
    calls = []
    hook = model.temporal_similarity_refiner.register_forward_pre_hook(
        lambda module, args: calls.append(tuple(x.detach().clone() for x in args))
    )
    result = _run(model, post, raw, metadata)
    hook.remove()
    # Existing verifier and global confuser diagnostics each build two
    # references. Preserve that behavior rather than changing those branches.
    assert len(calls) == 6
    assert torch.equal(calls[0][1], post[:2])
    for sample in range(2):
        assert torch.equal(calls[sample + 1][1], post[sample])
        assert torch.equal(calls[sample + 1][2], expected_mask[sample])
        assert torch.equal(calls[sample + 4][1], post[sample])
        assert torch.equal(calls[sample + 4][2], expected_mask[sample])
    assert torch.equal(calls[3][1], post[2:])
    assert "query_evidence_patch_weights" in result
    assert "query_frame_absolute_patch_mass" in result
    assert torch.isfinite(result["query_partial_q2s_logits"]).all()

    changed = dict(metadata)
    changed["episode_positive_labels"] = metadata["episode_positive_labels"].clone()
    changed["episode_positive_labels"][2] = False
    other = _run(model, post, raw, changed)
    for key in result:
        assert torch.equal(result[key], other[key]), key


def test_default_off_sav_on_registration_rng_and_checkpoint_roundtrip():
    assert not get_cfg().FEW_SHOT.POT_ROUTE.TEMPORAL_REFINEMENT.ENABLE
    sav_cfg = _sav_cfg()
    assert sav_cfg.FEW_SHOT.POT_ROUTE.TEMPORAL_REFINEMENT.ENABLE
    assert "TYPE" not in sav_cfg.FEW_SHOT.POT_ROUTE.TEMPORAL_REFINEMENT

    model = _model(enable=False)
    assert model.temporal_similarity_refiner is None and not model.state_dict()
    model.pot_route_cfg.TEMPORAL_REFINEMENT.ENABLE = True
    before = torch.random.get_rng_state().clone()
    model._init_temporal_similarity_refinement()
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(
        key.startswith("temporal_similarity_refiner.")
        for key in model.state_dict()
    )

    copy = _model()
    copy.load_state_dict(model.state_dict(), strict=True)
    similarity = torch.rand(2, 4, 3)
    tokens = torch.rand(4, 3, 2)
    mask = torch.ones(4, 3).bool()
    assert torch.equal(
        model._refine_trajectory_similarity(similarity, tokens, mask),
        copy._refine_trajectory_similarity(similarity, tokens, mask),
    )


@pytest.mark.parametrize("route", ["cat", "null", "local", "no_points", "no_q2s"])
def test_incompatible_routes_fail_instead_of_silently_bypassing_refinement(route):
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


def test_enabled_episode_requires_visibility_but_disabled_route_does_not():
    post, raw, metadata = _episode()
    del metadata["pred_visibility"]
    with pytest.raises(ValueError, match="pred_visibility"):
        _run(_model(), post, raw, metadata)
    assert _run(_model(enable=False), post, raw, metadata) is not None


def test_real_constructor_preserves_baseline_rng_and_optimizer_contract(monkeypatch):
    monkeypatch.setattr(
        Pointformer,
        "_load_dinotxt_visual_model",
        lambda self: nn.Identity(),
    )
    monkeypatch.setattr(
        Pointformer,
        "_load_dinotxt_text_model",
        lambda self: nn.Identity(),
    )
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
    baseline_state = baseline.state_dict()
    temporal_state = temporal.state_dict()
    for key in baseline_state:
        assert torch.equal(baseline_state[key], temporal_state[key]), key
    extra_keys = set(temporal_state) - set(baseline_state)
    assert len(extra_keys) == 11
    assert all(key.startswith("temporal_similarity_refiner.") for key in extra_keys)

    old_checkpoint = temporal.load_state_dict(baseline_state, strict=False)
    assert set(old_checkpoint.missing_keys) == extra_keys
    assert not old_checkpoint.unexpected_keys

    optimizer = construct_optimizer(temporal, cfg)
    group_by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for name, parameter in temporal.temporal_similarity_refiner.named_parameters():
        group = group_by_parameter[id(parameter)]
        expected_decay = 0.0 if parameter.ndim == 1 else 0.01
        assert group["weight_decay"] == expected_decay, name
        assert group["lr"] == cfg.SOLVER.BASE_LR, name


def test_old_tcn_state_has_no_shape_compatible_temporal_parameters():
    model = _model()
    old_tcn_state = {
        "temporal_similarity_refiner.temporal_conv.weight": torch.randn(16, 3, 3),
        "temporal_similarity_refiner.temporal_conv.bias": torch.randn(16),
        "temporal_similarity_refiner.long_temporal_conv.weight": torch.randn(
            16, 16, 3
        ),
        "temporal_similarity_refiner.long_temporal_conv.bias": torch.randn(16),
        "temporal_similarity_refiner.out_proj.weight": torch.randn(1, 16, 1),
    }
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in old_tcn_state.items()
        if key in current and value.shape == current[key].shape
    }
    assert not compatible
    # A direct strict=False load must not reinterpret the old Conv1D readout
    # as the new Linear readout (the normal loader filters incompatible shapes).
    with pytest.raises(RuntimeError, match="size mismatch.*out_proj.weight"):
        model.load_state_dict(old_tcn_state, strict=False)


def test_training_schedule_and_episode_sampling_controls_are_unchanged():
    cfg = _sav_cfg()
    assert cfg.FEW_SHOT.TRAIN_EPISODES == 2000
    assert not cfg.FEW_SHOT.TRAIN_OG_EPISODES
    assert cfg.RNG_SEED == 1
    assert cfg.SOLVER.BASE_LR == pytest.approx(1e-4)
    assert cfg.SOLVER.LR_POLICY == "cosine"
    assert cfg.SOLVER.COSINE_AFTER_WARMUP
    assert cfg.SOLVER.COSINE_END_LR == pytest.approx(1e-6)
    assert cfg.SOLVER.WARMUP_EPOCHS == 5
    assert cfg.SOLVER.MAX_EPOCH == 20
