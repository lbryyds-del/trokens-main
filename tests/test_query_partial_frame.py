"""Unit tests for frame-level query-to-support matching and direct text routing."""

from types import SimpleNamespace

import torch

from trokens.models.pointformer import Pointformer


def _stub_model(
    frame_softmax_tau=1.0,
    use_support_text_fusion=False,
    visual_detach=True,
):
    model = Pointformer.__new__(Pointformer)
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=frame_softmax_tau,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.cfg = SimpleNamespace(
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.use_support_text_fusion = use_support_text_fusion
    model.support_text_fusion_cfg = SimpleNamespace(
        TEXT_WEIGHT=1.0,
        VISUAL_WEIGHT=1.0,
        VISUAL_DETACH=visual_detach,
    )
    return model


def test_frame_softmax_is_per_text_per_frame_and_respects_mask():
    model = _stub_model(frame_softmax_tau=1.0)
    patch_tokens = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        ]
    )
    point_mask = torch.tensor(
        [[True, True, False], [False, False, False]]
    )
    text_features = torch.eye(2)

    prototypes, weights = model._compute_frame_softmax_text_prototypes(
        patch_tokens,
        point_mask,
        text_features,
    )

    assert prototypes.shape == (2, 2, 2)
    assert weights.shape == (2, 2, 3)
    assert torch.allclose(weights[:, 0].sum(dim=-1), torch.ones(2))
    assert torch.equal(weights[:, 0, 2], torch.zeros(2))
    assert torch.equal(weights[:, 1], torch.zeros(2, 3))
    assert torch.equal(prototypes[:, 1], torch.zeros(2, 2))

    expected_first = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    expected_second = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    assert torch.allclose(weights[0, 0, :2], expected_first)
    assert torch.allclose(weights[1, 0, :2], expected_second)

    similarity = model._compute_bidirectional_frame_similarity(
        prototypes.unsqueeze(0),
        prototypes,
    )
    assert torch.isfinite(similarity).all()


def test_frame_softmax_support_uses_independent_true_labels_and_averages():
    model = _stub_model(frame_softmax_tau=0.5)
    value_tokens = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 1.0], [1.0, -1.0]]],
        ]
    )
    point_mask = torch.ones(3, 1, 2, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    episode_positive_labels = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [0, 0, 1]],
        dtype=torch.bool,
    )
    episode_text = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )

    support_prototypes = model._build_frame_softmax_support_prototypes(
        value_tokens,
        point_mask,
        support_mask,
        episode_positive_labels,
        episode_text,
    )
    sample0, _ = model._compute_frame_softmax_text_prototypes(
        value_tokens[0], point_mask[0], episode_text[[0, 1]]
    )
    sample1, _ = model._compute_frame_softmax_text_prototypes(
        value_tokens[1], point_mask[1], episode_text[[0]]
    )

    assert support_prototypes.shape == (3, 1, 2)
    assert torch.allclose(
        support_prototypes[0],
        torch.stack([sample0[0], sample1[0]], dim=0).mean(dim=0),
    )
    assert torch.allclose(support_prototypes[1], sample0[1])
    assert torch.equal(support_prototypes[2], torch.zeros(1, 2))
    assert not torch.allclose(sample0[0], sample0[1])


def test_support_text_fusion_is_fixed_one_to_one_with_missing_class_fallback():
    model = _stub_model(use_support_text_fusion=True)
    episode_text = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
    )
    support_prototypes = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 3.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )

    fused, support_visual, valid_classes = (
        model._fuse_episode_text_with_support_visual(
            episode_text,
            support_prototypes,
        )
    )

    expected_visual = torch.nn.functional.normalize(
        torch.tensor([1.0, 3.0]),
        dim=0,
    )
    expected_fused = torch.nn.functional.normalize(
        episode_text[0] + expected_visual,
        dim=0,
    )
    assert torch.equal(valid_classes, torch.tensor([True, False]))
    assert torch.allclose(support_visual[0], expected_visual)
    assert torch.equal(support_visual[1], torch.zeros(2))
    assert torch.allclose(fused[0], expected_fused)
    assert torch.equal(fused[1], episode_text[1])


def test_support_text_fusion_detaches_support_conditioning_branch():
    model = _stub_model(
        use_support_text_fusion=True,
        visual_detach=True,
    )
    episode_text = torch.tensor(
        [[1.0, 0.0]],
        requires_grad=True,
    )
    support_prototypes = torch.tensor(
        [[[0.0, 1.0], [0.0, 2.0]]],
        requires_grad=True,
    )

    fused, _, _ = model._fuse_episode_text_with_support_visual(
        episode_text,
        support_prototypes,
    )
    fused[0, 1].backward()

    assert episode_text.grad is not None
    assert torch.isfinite(episode_text.grad).all()
    assert support_prototypes.grad is None


def test_frame_softmax_fusion_conditions_query_only_and_ignores_query_targets():
    model = _stub_model(
        frame_softmax_tau=0.5,
        use_support_text_fusion=True,
    )
    value_tokens = torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
        ]
    )
    point_mask = torch.ones(3, 1, 2, dtype=torch.bool)
    episode_text = torch.eye(2)
    model._get_pot_label_text_features = lambda class_ids, dtype: (
        episode_text.index_select(0, class_ids - 4).to(dtype=dtype)
    )
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": point_mask,
        "episode_class_ids": torch.tensor([4, 5]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.bool,
        ),
    }

    routed_text = []
    original_compute = model._compute_batched_frame_softmax_text_prototypes

    def record_routed_text(
        patch_tokens,
        sample_mask,
        label_text_features,
        temporal_point_mask=None,
    ):
        routed_text.append(label_text_features.detach().clone())
        return original_compute(
            patch_tokens,
            sample_mask,
            label_text_features,
            temporal_point_mask=temporal_point_mask,
        )

    model._compute_batched_frame_softmax_text_prototypes = record_routed_text
    first = model._build_frame_softmax_q2s_aux(value_tokens, metadata)

    assert len(routed_text) == 2
    assert torch.equal(routed_text[0], episode_text)
    assert torch.allclose(
        routed_text[1],
        first["support_text_fusion_query_features"],
    )
    expected_fused = torch.full((2, 2), 2 ** -0.5)
    assert torch.allclose(
        first["support_text_fusion_query_features"],
        expected_fused,
    )
    assert torch.equal(
        first["support_text_fusion_valid_classes"],
        torch.tensor([True, True]),
    )

    changed_metadata = dict(metadata)
    changed_metadata["episode_positive_labels"] = metadata[
        "episode_positive_labels"
    ].clone()
    changed_metadata["episode_positive_labels"][2] = torch.tensor([0, 1])
    routed_text.clear()
    second = model._build_frame_softmax_q2s_aux(
        value_tokens,
        changed_metadata,
    )
    assert torch.allclose(
        first["support_text_fusion_query_features"],
        second["support_text_fusion_query_features"],
    )
    assert torch.allclose(
        first["query_partial_query_prototypes"],
        second["query_partial_query_prototypes"],
    )


def test_frame_softmax_q2s_uses_episode_axis_without_mass_or_query_labels():
    model = _stub_model(frame_softmax_tau=0.5)
    value_tokens = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.9, 0.1], [0.1, 0.9]],
            ],
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.1, 0.9], [0.9, 0.1]],
            ],
        ]
    )
    point_mask = torch.ones(3, 2, 2, dtype=torch.bool)
    episode_class_ids = torch.tensor([4, 7])
    text_by_class = {
        4: torch.tensor([1.0, 0.0]),
        7: torch.tensor([0.0, 1.0]),
    }
    encoded_ids = []

    def encode_text(class_ids, dtype):
        encoded_ids.append(class_ids.detach().cpu().tolist())
        return torch.stack(
            [text_by_class[int(class_id)] for class_id in class_ids.tolist()]
        ).to(dtype=dtype)

    model._get_pot_label_text_features = encode_text
    metadata = {
        "support_mask": torch.tensor([True, False, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": point_mask,
        "episode_class_ids": episode_class_ids,
        "episode_positive_labels": torch.tensor(
            [[1, 0], [1, 0], [0, 1]], dtype=torch.bool
        ),
        "raw_positive_labels": torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool
        ),
    }

    first = model._build_frame_softmax_q2s_aux(value_tokens, metadata)
    changed_metadata = dict(metadata)
    changed_metadata["episode_positive_labels"] = metadata[
        "episode_positive_labels"
    ].clone()
    changed_metadata["episode_positive_labels"][1:] = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.bool
    )
    changed_metadata["raw_positive_labels"] = ~metadata["raw_positive_labels"]
    second = model._build_frame_softmax_q2s_aux(value_tokens, changed_metadata)

    assert encoded_ids == [[4, 7], [4, 7]]
    assert first["query_partial_query_prototypes"].shape == (2, 2, 2, 2)
    assert first["query_partial_support_prototypes"].shape == (2, 2, 2)
    assert first["query_partial_q2s_logits"].shape == (2, 2)
    assert first["query_partial_query_sample_indices"].tolist() == [1, 2]
    assert first["query_partial_label_axis_global_labels"].tolist() == [4, 7]
    assert "query_partial_transport_mass" not in first
    assert "query_partial_beta_mass_term" not in first
    assert torch.allclose(
        first["query_partial_q2s_logits"],
        10.0 * first["query_partial_diag_similarity"] - 2.0,
    )
    assert torch.allclose(
        first["query_partial_query_prototypes"],
        second["query_partial_query_prototypes"],
    )
    assert torch.allclose(
        first["query_partial_q2s_logits"],
        second["query_partial_q2s_logits"],
    )
