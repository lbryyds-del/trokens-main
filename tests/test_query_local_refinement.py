"""Tests for confuser-aware local Query patch refinement."""

from types import SimpleNamespace

import pytest
import torch

from trokens.models.pointformer import Pointformer
from trokens.models.query_class_matchability import (
    build_class_local_support_references,
    build_confuser_refined_query_prototypes,
    compute_local_positive_confuser_margin,
    pairwise_bimhm,
)


def _pointformer(tau=1.0):
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    model.pot_route_cfg = SimpleNamespace(FRAME_SOFTMAX_TAU=float(tau))
    return model


def _local_cfg(**overrides):
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
        "LOCAL_REFINEMENT_ENABLE": True,
        "LOCAL_MARGIN_TEMPERATURE": 0.10,
        "LOCAL_LOGIT_STRENGTH": 0.50,
        "LOCAL_POSITIVE_AGGREGATION": "topk_mean",
        "LOCAL_POSITIVE_TOPK": 2,
        "LOCAL_NEGATIVE_AGGREGATION": "topk_mean",
        "LOCAL_NEGATIVE_TOPK": 2,
        "LOCAL_NEGATIVE_TEMPERATURE": 0.10,
        "LOCAL_DETACH_REFERENCES": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_zero_strength_exactly_recovers_base_route():
    model = _pointformer(tau=0.4)
    query = torch.tensor(
        [[[[0.8, 0.6], [0.8, -0.6], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    text = torch.tensor([[1.0, 0.0]])
    margin = torch.tensor([[[[0.9, -0.7, 0.2]]]])

    result = build_confuser_refined_query_prototypes(
        model,
        query,
        mask,
        text,
        margin,
        patch_tau=0.4,
        local_tau=0.1,
        local_strength=0.0,
    )

    assert torch.equal(result["base_weights"], result["refined_weights"])
    assert torch.equal(result["base_prototypes"], result["refined_prototypes"])
    assert torch.count_nonzero(result["weight_shift"]) == 0


def test_constant_margin_keeps_relative_patch_weights_unchanged():
    model = _pointformer(tau=0.4)
    query = torch.tensor(
        [[[[0.8, 0.6], [0.8, -0.6], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    text = torch.tensor([[1.0, 0.0]])
    margin = torch.full((1, 1, 1, 3), 0.3)

    result = build_confuser_refined_query_prototypes(
        model,
        query,
        mask,
        text,
        margin,
        patch_tau=0.4,
        local_tau=0.1,
        local_strength=0.5,
    )

    assert torch.allclose(
        result["base_weights"],
        result["refined_weights"],
        atol=1e-7,
        rtol=1e-6,
    )


def test_positive_and_negative_margins_move_patch_weights_in_opposite_directions():
    model = _pointformer(tau=1.0)
    # Every patch has identical text cosine, isolating the local residual.
    query = torch.tensor(
        [[[[0.8, 0.6], [0.8, -0.6], [0.8, 0.6]]]],
        dtype=torch.float32,
    )
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    text = torch.tensor([[1.0, 0.0]])
    margin = torch.tensor([[[[1.0, 0.0, -1.0]]]])

    result = build_confuser_refined_query_prototypes(
        model,
        query,
        mask,
        text,
        margin,
        patch_tau=1.0,
        local_tau=0.1,
        local_strength=0.5,
    )
    before = result["base_weights"][0, 0, 0]
    after = result["refined_weights"][0, 0, 0]

    assert after[0] > before[0]
    assert after[2] < before[2]
    assert after.sum().item() == pytest.approx(1.0)


def test_missing_confuser_produces_neutral_local_margin():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    references = torch.tensor([[[[1.0, 0.0]]]])
    frame_valid = torch.ones(1, 1, 1, dtype=torch.bool)
    positive_valid = torch.ones(1, 1, dtype=torch.bool)
    confuser_valid = torch.zeros(1, 1, dtype=torch.bool)
    query_mask = torch.ones(1, 1, 2, dtype=torch.bool)

    margin, _, _, positive_count, confuser_count = (
        compute_local_positive_confuser_margin(
            query,
            references,
            frame_valid,
            positive_valid,
            confuser_valid,
            query_mask,
        )
    )

    assert torch.count_nonzero(margin) == 0
    assert positive_count.tolist() == [1]
    assert confuser_count.tolist() == [0]


def test_detached_local_references_keep_query_gradient_only():
    query = torch.tensor([[[[0.6, 0.4]]]], requires_grad=True)
    references = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ],
        requires_grad=True,
    )
    frame_valid = torch.ones(2, 1, 1, dtype=torch.bool)
    positive_valid = torch.tensor([[True], [False]])
    confuser_valid = torch.tensor([[False], [True]])
    query_mask = torch.ones(1, 1, 1, dtype=torch.bool)

    margin, _, _, _, _ = compute_local_positive_confuser_margin(
        query,
        references,
        frame_valid,
        positive_valid,
        confuser_valid,
        query_mask,
        detach_references=True,
    )
    margin.sum().backward()

    assert query.grad is not None
    assert query.grad.abs().sum().item() > 0.0
    assert references.grad is None


def test_invalid_support_frames_cannot_win_temporal_max():
    query = torch.tensor([[[[1.0, 0.0]]]])
    references = torch.tensor(
        [
            # The aligned frame is invalid; only the orthogonal frame counts.
            [[[1.0, 0.0], [0.0, 1.0]]],
            # The aligned frame is again invalid; valid confuser is opposite.
            [[[1.0, 0.0], [-1.0, 0.0]]],
        ]
    )
    frame_valid = torch.tensor(
        [[[False, True]], [[False, True]]],
        dtype=torch.bool,
    )
    positive_valid = torch.tensor([[True], [False]])
    confuser_valid = torch.tensor([[False], [True]])
    query_mask = torch.ones(1, 1, 1, dtype=torch.bool)

    margin, positive, confuser, _, _ = compute_local_positive_confuser_margin(
        query,
        references,
        frame_valid,
        positive_valid,
        confuser_valid,
        query_mask,
    )

    assert positive.item() == pytest.approx(0.0)
    assert confuser.item() == pytest.approx(-1.0)
    assert margin.item() == pytest.approx(1.0)


def test_empty_query_mask_stays_finite_and_has_zero_route_mass():
    model = _pointformer(tau=0.4)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    references = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]
    )
    frame_valid = torch.ones(2, 1, 1, dtype=torch.bool)
    positive_valid = torch.tensor([[True], [False]])
    confuser_valid = torch.tensor([[False], [True]])
    query_mask = torch.zeros(1, 1, 2, dtype=torch.bool)

    margin, positive, confuser, _, _ = compute_local_positive_confuser_margin(
        query,
        references,
        frame_valid,
        positive_valid,
        confuser_valid,
        query_mask,
    )
    result = build_confuser_refined_query_prototypes(
        model,
        query,
        query_mask,
        torch.tensor([[1.0, 0.0]]),
        margin,
        patch_tau=0.4,
    )

    for value in (margin, positive, confuser, *result.values()):
        assert torch.isfinite(value).all()
    assert torch.count_nonzero(result["refined_weights"]) == 0
    assert torch.count_nonzero(result["refined_prototypes"]) == 0


def test_local_support_references_use_support_labels_and_frame_visibility():
    model = _pointformer(tau=1.0)
    values = torch.tensor(
        [
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[0.0, 1.0]], [[0.0, 1.0]]],
            [[[1.0, 1.0]], [[1.0, 1.0]]],
        ]
    )
    route_mask = torch.ones(3, 2, 1, dtype=torch.bool)
    frame_mask = route_mask.clone()
    frame_mask[0, 1] = False
    support_mask = torch.tensor([True, True, False])
    labels = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bool)

    references, valid, positive, confuser, indices = (
        build_class_local_support_references(
            model,
            values,
            route_mask,
            frame_mask,
            support_mask,
            labels,
            torch.eye(2),
        )
    )

    assert references.shape == (2, 2, 2, 2)
    assert valid.shape == (2, 2, 2)
    assert valid[0, :, 1].tolist() == [False, False]
    assert positive.tolist() == [[True, False], [False, True]]
    assert confuser.tolist() == [[False, True], [True, False]]
    assert indices.tolist() == [0, 1]


def test_wrapper_outputs_local_diagnostics_and_ignores_query_targets():
    model = _pointformer(tau=1.0)
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(QUERY_CLASS_MATCHABILITY=_local_cfg()),
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
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
        ],
        requires_grad=True,
    )
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
    changed = dict(metadata)
    changed["episode_positive_labels"] = metadata[
        "episode_positive_labels"
    ].clone()
    changed["episode_positive_labels"][-1] = torch.tensor([0, 1])

    model.eval()
    first = model._build_frame_softmax_q2s_aux(values, metadata)
    second = model._build_frame_softmax_q2s_aux(values, changed)

    assert first["query_local_relative_margin"].shape == (1, 2, 1, 2)
    assert first["query_patch_weights_before_refinement"].shape == (1, 2, 1, 2)
    assert first["query_patch_weights_after_refinement"].shape == (1, 2, 1, 2)
    assert first["query_local_weight_shift_mean"].shape == (1, 2)
    assert first["query_local_weight_shift_mean"].min().item() > 0.0
    assert not torch.equal(
        first["query_partial_q2s_base_logits"],
        first["query_partial_q2s_pre_refinement_logits"],
    )
    for key in (
        "query_local_relative_margin",
        "query_patch_weights_after_refinement",
        "query_partial_q2s_logits",
    ):
        assert torch.equal(first[key], second[key])

    (
        first["query_partial_q2s_base_logits"].mean()
        + first["query_partial_q2s_logits"].mean()
    ).backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().sum().item() > 0.0


def test_pairwise_bimhm_accepts_an_empty_support_set():
    query = torch.randn(2, 3, 4, 5)
    support = torch.empty(0, 3, 4, 5)
    result = pairwise_bimhm(query, support)
    assert result.shape == (2, 3, 0)
    assert torch.isfinite(result).all()
