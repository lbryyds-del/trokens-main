"""Tests for Query-region specificity and absolute transport mass."""

from types import SimpleNamespace

import pytest
import torch

from trokens.models.pointformer import Pointformer
from trokens.models.query_class_matchability import (
    classwise_frame_similarity,
    compute_evidence_conditioned_frame_matchability,
    compute_support_calibrated_frame_transport_mass,
    factorize_frame_hypothesis_states,
    hypothesis_mass_bimhm_logits,
    normalize_patch_region_weights,
)


def _pointformer(tau=1.0):
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    model.pot_route_cfg = SimpleNamespace(FRAME_SOFTMAX_TAU=float(tau))
    return model


def _evidence_cfg(**overrides):
    values = {
        "ENABLE": True,
        "MODE": "positive_confuser_margin",
        "EVIDENCE_SOURCE": "post",
        "MARGIN_TEMPERATURE": 0.10,
        "MARGIN_BIAS": 0.0,
        "NEGATIVE_AGGREGATION": "max",
        "NEGATIVE_TOPK": 2,
        "NEGATIVE_TEMPERATURE": 0.10,
        "DETACH_CONFUSER_SUPPORT": True,
        "LOCAL_REFINEMENT_ENABLE": False,
        "EVIDENCE_VERIFICATION_ENABLE": True,
        "EVIDENCE_USE_VISIBILITY": True,
        "EVIDENCE_POSITIVE_AGGREGATION": "topk_mean",
        "EVIDENCE_POSITIVE_TOPK": 2,
        "EVIDENCE_CONFUSER_AGGREGATION": "topk_mean",
        "EVIDENCE_CONFUSER_TOPK": 2,
        "EVIDENCE_AGGREGATION_TEMPERATURE": 0.10,
        "EVIDENCE_DETACH_REFERENCES": True,
        "FRAME_MARGIN_TEMPERATURE": 0.10,
        "FRAME_MARGIN_BIAS": 0.0,
        "EVIDENCE_VIDEO_TOPK_FRAMES": 3,
        "EVIDENCE_MIL_TEMPERATURE": 0.10,
        "EVIDENCE_MIL_LOSS_WEIGHT": 0.10,
        "ABSOLUTE_MASS_ENABLE": False,
        "ABSOLUTE_MASS_SOURCE": "raw",
        "ABSOLUTE_MASS_USE_VISIBILITY": True,
        "ABSOLUTE_MASS_PATCH_TOPK": 1,
        "ABSOLUTE_MASS_SUPPORT_TOPK_FRAMES": 1,
        "ABSOLUTE_MASS_CALIBRATION_BETA": 0.50,
        "ABSOLUTE_MASS_TEMPERATURE": 0.05,
        "ABSOLUTE_MASS_MIN_SUPPORT_GAP": 0.0,
        "ABSOLUTE_MASS_DETACH_SUPPORT_STATS": True,
        "ABSOLUTE_MASS_RELIABILITY_FALLBACK": True,
        "ABSOLUTE_MASS_TRANSPORT_STRENGTH": 1.0,
        "ABSOLUTE_MASS_UNMATCHED_COST": 0.0,
        "ABSOLUTE_MASS_ONE_SIDED": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_query_region_weights_respect_visibility_and_renormalize():
    weights = torch.tensor(
        [[[[0.2, 0.3, 0.5]], [[0.0, 0.5, 0.5]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[True, False, True]]])
    normalized = normalize_patch_region_weights(weights, mask)

    assert normalized[0, 0, 0].tolist() == pytest.approx(
        [2.0 / 7.0, 0.0, 5.0 / 7.0]
    )
    assert normalized[0, 1, 0].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert torch.allclose(normalized.sum(dim=-1), torch.ones(1, 2, 1))


def test_frame_matchability_compares_identical_evidence_regions():
    weights = torch.tensor([[[[0.75, 0.25]]]])
    positive = torch.tensor([[[[0.8, 0.0]]]])
    confuser = torch.tensor([[[[0.0, 0.8]]]])
    result = compute_evidence_conditioned_frame_matchability(
        weights,
        positive,
        confuser,
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([1]),
        torch.tensor([1]),
        temperature=0.10,
    )

    assert result["positive_evidence"].item() == pytest.approx(0.6)
    assert result["confuser_evidence"].item() == pytest.approx(0.2)
    assert result["margin"].item() == pytest.approx(0.4)
    assert result["matchability"].item() > 0.98


def test_frame_hypothesis_states_factorize_mass_and_specificity():
    result = factorize_frame_hypothesis_states(
        torch.tensor([[[0.8, 1.0, 0.2]]]),
        torch.tensor([[[0.6, 0.25, 1.0]]]),
    )

    assert result["target_mass"].flatten().tolist() == pytest.approx(
        [0.48, 0.25, 0.20]
    )
    assert result["confuser_mass"].flatten().tolist() == pytest.approx(
        [0.32, 0.75, 0.00]
    )
    assert result["null_mass"].flatten().tolist() == pytest.approx(
        [0.20, 0.00, 0.80]
    )
    total = (
        result["target_mass"]
        + result["confuser_mass"]
        + result["null_mass"]
    )
    assert torch.allclose(total, torch.ones_like(total))


def test_frame_hypothesis_states_reduce_to_each_single_factor():
    value = torch.tensor([[[0.2, 0.7]]])
    quality_only = factorize_frame_hypothesis_states(
        value,
        torch.ones_like(value),
    )
    specificity_only = factorize_frame_hypothesis_states(
        torch.ones_like(value),
        value,
    )

    assert torch.equal(quality_only["target_mass"], value)
    assert torch.equal(specificity_only["target_mass"], value)


def test_support_calibrated_absolute_mass_can_abstain_per_frame():
    # Support row 0 is positive for class 0 and row 1 for class 1.  Query
    # evidence alternates between the two classes across its two frames.
    similarity = torch.tensor(
        [
            [[[0.90], [0.80]], [[0.10], [0.20]]],
            [[[0.10], [0.20]], [[0.90], [0.80]]],
            [[[0.75], [0.10]], [[0.15], [0.70]]],
        ]
    )
    point_mask = torch.ones(3, 2, 1, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    labels = torch.tensor(
        [[1, 0], [0, 1], [0, 1]],
        dtype=torch.bool,
    )
    result = compute_support_calibrated_frame_transport_mass(
        similarity,
        point_mask,
        support_mask,
        labels,
        _evidence_cfg(),
    )

    assert result["threshold"].tolist() == pytest.approx([0.55, 0.55])
    assert result["support_reliable"].tolist() == [True, True]
    assert result["patch_mass"][0, 0, 0] > 0.98
    assert result["patch_mass"][0, 0, 1] < 0.001
    assert result["patch_mass"][0, 1, 0] < 0.001
    assert result["patch_mass"][0, 1, 1] > 0.95
    assert torch.allclose(
        result["patch_mass"] + result["unmatched_mass"],
        torch.ones_like(result["patch_mass"]),
    )

    changed = labels.clone()
    changed[-1] = torch.tensor([1, 0])
    changed_result = compute_support_calibrated_frame_transport_mass(
        similarity,
        point_mask,
        support_mask,
        changed,
        _evidence_cfg(),
    )
    assert torch.equal(result["patch_mass"], changed_result["patch_mass"])


def test_unreliable_absolute_mass_calibration_is_a_strict_fallback():
    similarity = torch.tensor([[[[0.8]]], [[[0.1]]]])
    point_mask = torch.ones(2, 1, 1, dtype=torch.bool)
    support_mask = torch.tensor([True, False])
    labels = torch.ones(2, 1, dtype=torch.bool)
    result = compute_support_calibrated_frame_transport_mass(
        similarity,
        point_mask,
        support_mask,
        labels,
        _evidence_cfg(),
    )
    assert not result["support_reliable"].item()
    assert result["patch_mass"].item() == 1.0
    assert result["unmatched_mass"].item() == 0.0


def test_missing_confuser_or_frame_is_a_strict_noop():
    shape = (1, 1, 2, 2)
    result = compute_evidence_conditioned_frame_matchability(
        torch.full(shape, 0.5),
        torch.ones(shape),
        torch.zeros(shape),
        torch.tensor([[True, False]]),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    assert torch.equal(result["matchability"], torch.ones(1, 1, 2))
    assert torch.count_nonzero(result["margin"]) == 0
    assert not result["class_valid"].item()


def test_zero_transport_strength_exactly_recovers_base_logits():
    similarity = torch.tensor([[[[0.9], [0.8]]]])
    target_mass = torch.tensor([[[0.01, 1.0]]])
    base = torch.tensor([[7.25]])
    result = hypothesis_mass_bimhm_logits(
        similarity,
        target_mass,
        base,
        alpha=10.0,
        transport_strength=0.0,
    )
    assert torch.equal(result["logits"], base)


def test_patch_and_unmatched_mass_change_bimhm_before_frame_max():
    similarity = torch.tensor([[[[0.90], [0.80]]]])
    base = torch.tensor([[8.75]])
    result = hypothesis_mass_bimhm_logits(
        similarity,
        torch.tensor([[[0.0, 1.0]]]),
        base,
        alpha=10.0,
        transport_strength=1.0,
        unmatched_cost=0.0,
    )

    assert result["effective_patch_mass"].tolist() == [[[0.0, 1.0]]]
    assert result["unmatched_mass"].tolist() == [[[1.0, 0.0]]]
    assert result["transported_similarity"].flatten().tolist() == pytest.approx(
        [0.0, 0.8]
    )
    # Base BiMHM similarity is .875; transported BiMHM is .60.
    assert result["logits"].item() == pytest.approx(6.0, abs=1e-6)
    assert result["verified_winner"].item() == 1


def test_one_sided_unmatched_mass_never_rewards_negative_similarity():
    similarity = torch.tensor([[[[-0.5]]]])
    result = hypothesis_mass_bimhm_logits(
        similarity,
        torch.zeros(1, 1, 1),
        torch.tensor([[-5.0]]),
        alpha=10.0,
        unmatched_cost=0.0,
        one_sided_transport=True,
    )
    assert result["transported_similarity"].item() == pytest.approx(-0.5)
    assert result["logits"].item() == pytest.approx(-5.0)


def test_frame_similarity_matches_manual_cosine_matrix():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    support = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    result = classwise_frame_similarity(query, support)
    assert result.shape == (1, 1, 2, 2)
    assert torch.equal(result, torch.eye(2).reshape(1, 1, 2, 2))


def test_wrapper_keeps_construction_route_and_ignores_query_targets():
    model = _pointformer(tau=1.0)
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(
            QUERY_CLASS_MATCHABILITY=_evidence_cfg(),
        ),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=1.0,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = False
    model._get_pot_label_text_features = (
        lambda class_ids, dtype: torch.eye(2, dtype=dtype)
    )
    post = torch.tensor(
        [
            [[[1.0, 0.0], [0.8, 0.2]]],
            [[[0.0, 1.0], [0.2, 0.8]]],
            [[[0.9, 0.1], [0.1, 0.9]]],
        ],
        requires_grad=True,
    )
    raw = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
        ]
    )
    point_mask = torch.ones(3, 1, 2, dtype=torch.bool)
    visibility = point_mask.clone()
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": visibility,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.bool,
        ),
    }
    changed = dict(metadata)
    changed["episode_positive_labels"] = metadata[
        "episode_positive_labels"
    ].clone()
    changed["episode_positive_labels"][-1] = torch.tensor([0, 1])

    # Exercise the actual meta-training path; Query targets remain loss-only
    # and cannot change routing outputs.
    model.train()
    first = model._build_frame_softmax_q2s_aux(
        post,
        metadata,
        matchability_evidence_tokens=raw,
    )
    second = model._build_frame_softmax_q2s_aux(
        post,
        changed,
        matchability_evidence_tokens=raw,
    )

    _, expected_query_weights = model._compute_frame_softmax_text_prototypes(
        post[-1],
        point_mask[-1],
        torch.eye(2),
    )
    assert first["query_evidence_patch_weights"].shape == (1, 2, 1, 2)
    assert torch.equal(
        first["query_evidence_patch_weights"],
        expected_query_weights.unsqueeze(0),
    )
    assert first["query_frame_matchability"].shape == (1, 2, 1)
    assert first["query_partial_q2s_temporal_logits"].shape == (1, 2)
    assert torch.equal(
        first["query_partial_query_prototypes"],
        first["query_partial_query_prototypes_before_refinement"],
    )
    for key in (
        "query_evidence_patch_weights",
        "query_frame_relative_margin",
        "query_partial_q2s_logits",
    ):
        assert torch.equal(first[key], second[key])


def test_wrapper_preserves_explicit_patch_plus_unmatched_mass():
    model = _pointformer(tau=1.0)
    cfg = _evidence_cfg(
        EVIDENCE_VERIFICATION_ENABLE=False,
        ABSOLUTE_MASS_ENABLE=True,
    )
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(QUERY_CLASS_MATCHABILITY=cfg),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=1.0,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = False
    model._get_pot_label_text_features = (
        lambda class_ids, dtype: torch.eye(2, dtype=dtype)
    )
    post = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
            [[[1.0, 1.0]]],
        ],
        requires_grad=True,
    )
    raw = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
            [[[1.0, 0.0]]],
        ]
    )
    mask = torch.ones(3, 1, 1, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": mask,
        "pred_visibility": mask,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [0, 1]],
            dtype=torch.bool,
        ),
    }
    model.train()
    result = model._build_frame_softmax_q2s_aux(
        post,
        metadata,
        matchability_evidence_tokens=raw,
    )

    conditional_sum = result["query_patch_conditional_weights"].sum(dim=-1)
    transported_sum = result["query_patch_transport_weights"].sum(dim=-1)
    patch_mass = result["query_frame_patch_mass"]
    unmatched_mass = result["query_frame_unmatched_mass"]
    assert torch.allclose(conditional_sum, torch.ones_like(conditional_sum))
    assert torch.allclose(transported_sum, patch_mass, atol=1e-6)
    assert torch.allclose(
        transported_sum + unmatched_mass,
        torch.ones_like(transported_sum),
        atol=1e-6,
    )
    assert patch_mass[0, 0, 0] > 0.99
    assert patch_mass[0, 1, 0] < 0.001
    assert result["query_frame_transport_logit_delta"][0, 1] < 0.0
    assert torch.allclose(
        result["query_partial_query_transported_prototypes"],
        result["query_partial_query_prototypes"] * patch_mass.unsqueeze(-1),
        atol=1e-6,
    )
    result["query_partial_q2s_logits"].sum().backward()
    assert post.grad is not None
    assert torch.isfinite(post.grad).all()


def test_wrapper_uses_only_target_state_mass_in_bimhm():
    model = _pointformer(tau=1.0)
    cfg = _evidence_cfg(
        ABSOLUTE_MASS_ENABLE=True,
        ABSOLUTE_MASS_PATCH_TOPK=1,
        ABSOLUTE_MASS_SUPPORT_TOPK_FRAMES=1,
        ABSOLUTE_MASS_CALIBRATION_BETA=0.50,
    )
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(QUERY_CLASS_MATCHABILITY=cfg),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=1.0,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = False
    model._get_pot_label_text_features = (
        lambda class_ids, dtype: torch.eye(2, dtype=dtype)
    )
    values = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
            [[[0.8, 0.2]]],
        ],
        requires_grad=True,
    )
    mask = torch.ones(3, 1, 1, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": mask,
        "pred_visibility": mask,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.bool,
        ),
    }
    result = model._build_frame_softmax_q2s_aux(
        values,
        metadata,
        matchability_evidence_tokens=values.detach(),
    )

    target = result["query_frame_target_mass"]
    confuser = result["query_frame_confuser_mass"]
    null = result["query_frame_null_mass"]
    expected = (
        result["query_frame_absolute_patch_mass"]
        * result["query_frame_matchability"]
    )
    assert torch.allclose(target, expected)
    assert torch.allclose(result["query_frame_patch_mass"], target)
    assert torch.allclose(target + confuser + null, torch.ones_like(target))


def test_evidence_and_local_refinement_are_mutually_exclusive():
    model = _pointformer(tau=1.0)
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(
            QUERY_CLASS_MATCHABILITY=_evidence_cfg(
                LOCAL_REFINEMENT_ENABLE=True,
            ),
        ),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=1.0,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = False
    model._get_pot_label_text_features = (
        lambda class_ids, dtype: torch.eye(2, dtype=dtype)
    )
    values = torch.randn(3, 1, 2, 2)
    mask = torch.ones(3, 1, 2, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": mask,
        "pred_visibility": mask,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.bool,
        ),
    }
    with pytest.raises(ValueError, match="controlled alternatives"):
        model._build_frame_softmax_q2s_aux(
            values,
            metadata,
            matchability_evidence_tokens=values,
        )
