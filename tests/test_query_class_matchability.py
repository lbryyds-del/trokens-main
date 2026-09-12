"""Tests for Support-calibrated Query-class matchability."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trokens.config.defaults import get_cfg
from trokens.models.pointformer import Pointformer
from trokens.models.query_class_matchability import (
    build_class_confuser_prototypes,
    compute_relative_matchability,
    compute_matchability_from_similarity,
    masked_topk_mean,
    pairwise_bimhm,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg(**overrides):
    values = {
        "TOPK_PATCHES": 1,
        "TOPK_FRAMES": 1,
        "EVIDENCE_SOURCE": "post",
        "MODE": "threshold",
        "CALIBRATION_BETA": 0.25,
        "TEMPERATURE": 0.05,
        "DETACH_SUPPORT_STATS": True,
        "MARGIN_TEMPERATURE": 0.10,
        "MARGIN_BIAS": 0.0,
        "NEGATIVE_AGGREGATION": "max",
        "NEGATIVE_TOPK": 2,
        "NEGATIVE_TEMPERATURE": 0.10,
        "DETACH_CONFUSER_SUPPORT": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_enables_matchability_and_disables_learned_null():
    cfg = get_cfg()
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE is False
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.ENABLE is False

    cfg.merge_from_file(str(REPO_ROOT / "configs/trokens/sav.yaml"))
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE is True
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.ENABLE is False
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.TOPK_PATCHES == 8
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.TOPK_FRAMES == 3
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.EVIDENCE_SOURCE == "post"
    assert (
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.MODE
        == "threshold"
    )
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LOCAL_REFINEMENT_ENABLE is False
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LOCAL_LOGIT_STRENGTH == 0.50
    assert (
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LOCAL_NEGATIVE_AGGREGATION
        == "topk_mean"
    )
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.DUAL_LOGIT_LOSS_ENABLE is True
    assert (
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.EVIDENCE_VERIFICATION_ENABLE
        is False
    )
    assert not hasattr(
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY,
        "EVIDENCE_MAP_SOURCE",
    )
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.EVIDENCE_USE_VISIBILITY is True
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.EVIDENCE_VIDEO_TOPK_FRAMES == 3
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.EVIDENCE_MIL_TEMPERATURE == 0.10
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.EVIDENCE_MIL_LOSS_WEIGHT == 0.0
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_ENABLE is False
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_SOURCE == "raw"
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_PATCH_TOPK == 8
    assert (
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_TRANSPORT_STRENGTH
        == 1.0
    )
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_ONE_SIDED is True
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.BASE_LOGIT_LOSS_WEIGHT == 0.60
    assert (
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.VERIFIED_LOGIT_LOSS_WEIGHT
        == 0.40
    )
def test_masked_topk_mean_ignores_invalid_entries():
    scores = torch.tensor([[0.9, 0.7, 100.0, -100.0]])
    mask = torch.tensor([[True, True, False, False]])
    value = masked_topk_mean(scores, mask, k=3)
    assert value.item() == pytest.approx(0.8)

    empty = masked_topk_mean(scores, torch.zeros_like(mask), k=2)
    assert empty.item() == 0.0


def test_support_calibration_separates_high_and_low_query_evidence():
    # First two rows are Support: class-0 positive/class-1 negative, then vice
    # versa. The third Query has strong class-0 and weak class-1 evidence.
    similarity = torch.tensor(
        [
            [[[[0.8]]], [[[0.2]]]],
            [[[[0.2]]], [[[0.8]]]],
            [[[[0.7]]], [[[0.1]]]],
        ]
    ).reshape(3, 2, 1, 1)
    point_mask = torch.ones(3, 1, 1, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    labels = torch.tensor(
        [
            [1, 0],
            [0, 1],
            # Deliberately arbitrary Query targets; they are not consumed.
            [0, 1],
        ],
        dtype=torch.bool,
    )

    result = compute_matchability_from_similarity(
        similarity,
        point_mask,
        support_mask,
        labels,
        _cfg(),
    )

    assert result["support_positive_evidence_mean"].tolist() == pytest.approx(
        [0.8, 0.8]
    )
    assert result["support_negative_evidence_mean"].tolist() == pytest.approx(
        [0.2, 0.2]
    )
    assert result["threshold"].tolist() == pytest.approx([0.35, 0.35])
    assert result["support_reliable"].tolist() == [True, True]
    assert result["matchability"][0, 0].item() > 0.99
    assert result["matchability"][0, 1].item() < 0.01


def test_query_targets_do_not_change_matchability():
    similarity = torch.tensor(
        [
            [[[[0.8]]], [[[0.2]]]],
            [[[[0.2]]], [[[0.8]]]],
            [[[[0.6]]], [[[0.3]]]],
        ]
    ).reshape(3, 2, 1, 1)
    point_mask = torch.ones(3, 1, 1, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    labels_a = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.bool)
    labels_b = labels_a.clone()
    labels_b[-1] = torch.tensor([0, 1])

    result_a = compute_matchability_from_similarity(
        similarity,
        point_mask,
        support_mask,
        labels_a,
        _cfg(),
    )
    result_b = compute_matchability_from_similarity(
        similarity,
        point_mask,
        support_mask,
        labels_b,
        _cfg(),
    )
    assert torch.equal(result_a["matchability"], result_b["matchability"])
    assert torch.equal(result_a["threshold"], result_b["threshold"])


def test_support_statistics_can_be_detached_without_blocking_query_gradient():
    similarity = torch.tensor(
        [
            [[[[0.8]]]],
            [[[[0.2]]]],
            [[[[0.5]]]],
        ]
    ).reshape(3, 1, 1, 1).clone().detach().requires_grad_(True)
    point_mask = torch.ones(3, 1, 1, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    labels = torch.tensor([[1], [0], [0]], dtype=torch.bool)

    result = compute_matchability_from_similarity(
        similarity,
        point_mask,
        support_mask,
        labels,
        _cfg(DETACH_SUPPORT_STATS=True),
    )
    result["matchability"].sum().backward()

    assert similarity.grad is not None
    # Support calibration is detached; only the Query row receives gradient.
    assert torch.equal(similarity.grad[:2], torch.zeros_like(similarity.grad[:2]))
    assert similarity.grad[2].abs().sum().item() > 0.0


def test_pairwise_bimhm_matches_single_class_frame_reduction():
    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    support = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    scores = pairwise_bimhm(query, support)
    assert scores.shape == (1, 2)
    assert scores[0, 0].item() == pytest.approx(1.0)
    # BiMHM uses a frame-wise max, so opposite two-frame sequences retain the
    # orthogonal cross-frame similarity instead of producing -1 everywhere.
    assert scores[0, 1].item() == pytest.approx(0.0)


def test_relative_matchability_masks_missing_confusers_and_supports_topk():
    positive = torch.tensor([[0.8, 0.5]])
    negative = torch.tensor([[[0.2, 0.7, 0.1], [0.4, 0.3, 0.9]]])
    valid = torch.tensor([[True, True, False], [True, False, False]])

    rho, margin, negative_aggregate = compute_relative_matchability(
        positive,
        negative,
        valid,
        temperature=0.1,
        aggregation="topk_mean",
        topk=2,
    )

    # Class 0 uses (0.7 + 0.2) / 2; class 1 has only its 0.4 confuser.
    assert negative_aggregate[0].tolist() == pytest.approx([0.45, 0.4])
    assert margin[0].tolist() == pytest.approx([0.35, 0.1])
    assert rho[0, 0].item() > rho[0, 1].item()

    no_negative = compute_relative_matchability(
        positive,
        negative,
        torch.zeros_like(valid),
    )
    assert no_negative[0][0].tolist() == pytest.approx([1.0, 1.0])
    assert no_negative[1][0].tolist() == pytest.approx([0.0, 0.0])


def test_build_confuser_prototypes_uses_support_labels_only():
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    # The real method is sufficient for this shape-only construction test.
    model.pot_route_cfg = SimpleNamespace(FRAME_SOFTMAX_TAU=1.0)
    values = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    point_mask = torch.ones(3, 1, 2, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    labels = torch.tensor(
        [[1, 0], [0, 1], [1, 1]],
        dtype=torch.bool,
    )
    text = torch.eye(2)

    prototypes, valid, indices = build_class_confuser_prototypes(
        model,
        values,
        point_mask,
        support_mask,
        labels,
        text,
    )
    assert prototypes.shape == (2, 2, 1, 2)
    assert valid.tolist() == [[False, True], [True, False]]
    assert indices.tolist() == [0, 1]


def test_relative_margin_is_diagnostic_only_during_training():
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(
            QUERY_CLASS_MATCHABILITY=_cfg(
                ENABLE=True,
                MODE="positive_confuser_margin",
            )
        ),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
        FRAME_SOFTMAX_TAU=1.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = True
    model.support_text_fusion_cfg = SimpleNamespace(
        TEXT_WEIGHT=1.0,
        VISUAL_WEIGHT=1.0,
        VISUAL_DETACH=True,
    )
    text = torch.eye(2)
    model._get_pot_label_text_features = lambda class_ids, dtype: text.to(dtype)

    value_tokens = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[0.6, 0.4], [0.4, 0.6]]],
        ],
        requires_grad=True,
    )
    point_mask = torch.ones(3, 1, 2, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": point_mask,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [0, 1]],
            dtype=torch.bool,
        ),
    }

    model.train()
    aux = model._build_frame_softmax_q2s_aux(value_tokens, metadata)
    assert torch.equal(
        aux["query_partial_q2s_logits"],
        aux["query_partial_q2s_temporal_logits"],
    )
    assert torch.equal(
        aux["query_partial_q2s_logits"],
        aux["query_partial_q2s_base_logits"],
    )
    aux["query_partial_q2s_logits"].sum().backward()
    assert value_tokens.grad is not None
    assert torch.isfinite(value_tokens.grad).all()


def test_relative_margin_ignores_query_label_rows():
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    cfg = _cfg(
        ENABLE=True,
        MODE="positive_confuser_margin",
    )
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(QUERY_CLASS_MATCHABILITY=cfg),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
        FRAME_SOFTMAX_TAU=1.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = False
    text = torch.eye(2)
    model._get_pot_label_text_features = lambda class_ids, dtype: text.to(dtype)
    values = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[0.6, 0.4], [0.4, 0.6]]],
        ]
    )
    mask = torch.ones(3, 1, 2, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": mask,
        "pred_visibility": mask,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [0, 1]], dtype=torch.bool
        ),
    }
    metadata_changed = dict(metadata)
    metadata_changed["episode_positive_labels"] = torch.tensor(
        [[1, 0], [0, 1], [1, 0]], dtype=torch.bool
    )
    model.eval()
    out_a = model._build_frame_softmax_q2s_aux(values, metadata)
    out_b = model._build_frame_softmax_q2s_aux(values, metadata_changed)
    assert torch.equal(
        out_a["query_class_relative_margin"],
        out_b["query_class_relative_margin"],
    )
    assert torch.equal(
        out_a["query_partial_q2s_logits"],
        out_b["query_partial_q2s_logits"],
    )


def test_global_matchability_does_not_change_final_logits():
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(QUERY_CLASS_MATCHABILITY=_cfg(ENABLE=True)),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
        FRAME_SOFTMAX_TAU=1.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = False
    text = torch.eye(2)
    model._get_pot_label_text_features = lambda class_ids, dtype: text.to(dtype)

    value_tokens = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[0.6, 0.4], [0.4, 0.6]]],
        ]
    )
    point_mask = torch.ones(3, 1, 2, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": point_mask,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [0, 1]],
            dtype=torch.bool,
        ),
    }

    for training in (True, False):
        model.train(training)
        aux = model._build_frame_softmax_q2s_aux(value_tokens, metadata)
        assert torch.equal(
            aux["query_partial_q2s_logits"],
            aux["query_partial_q2s_temporal_logits"],
        )
        assert torch.equal(
            aux["query_partial_q2s_logits"],
            aux["query_partial_q2s_base_logits"],
        )
        assert torch.isfinite(aux["query_class_matchability"]).all()
