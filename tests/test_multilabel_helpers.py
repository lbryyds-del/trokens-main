"""Tests for multi-label few-shot helpers shared across routing modes."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tools.few_shot_multilabel import (
    compute_evidence_mil_loss,
    compute_query_partial_q2s_loss,
    q2s_cos_sim_fp32,
    support_query_split_multilabel_conditioned,
)


def test_q2s_cosine_forces_fp32_for_large_tokens_and_keeps_gradients_finite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    x = (torch.rand(12, 1024, device=device, dtype=torch.float16) * 80).requires_grad_()
    y = (torch.rand(8, 1024, device=device, dtype=torch.float16) * 80).requires_grad_()
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        similarity = q2s_cos_sim_fp32(x, y)

    reference_numerator = x.float() @ y.float().transpose(-1, -2)
    reference_denominator = (
        torch.norm(x.float(), dim=-1, keepdim=True)
        @ torch.norm(y.float(), dim=-1, keepdim=True).transpose(-1, -2)
        + 0.01
    )
    reference = reference_numerator / reference_denominator
    assert similarity.dtype == torch.float32
    assert torch.isfinite(similarity).all()
    assert torch.allclose(similarity, reference, atol=1e-6, rtol=1e-5)

    similarity.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()


def test_conditioned_split_replaces_support_and_query():
    base = {
        "support_preds": torch.randn(3, 2, 4, 8),
        "query_preds": torch.randn(1, 2, 4, 8),
        "query_condition": torch.tensor([False, False, True]),
    }
    query = torch.randn(1, 2, 1, 8)
    aux = {
        "support_conditioned_patch_tokens": torch.randn(2, 2, 1, 8),
        "support_branch_class_indices": torch.tensor([0, 1]),
        "query_conditioned_patch_tokens": query,
        "query_conditioned_sample_indices": torch.tensor([2]),
    }

    result = support_query_split_multilabel_conditioned(base, aux)

    assert result["support_preds"].shape == (3, 2, 1, 8)
    assert torch.equal(result["query_preds"], query)


def _dual_loss_cfg(
    enabled=True,
    base_weight=0.5,
    verified_weight=0.5,
    evidence_mil_weight=0.0,
):
    return SimpleNamespace(
        SOLVER=SimpleNamespace(TEMPRATURE=2.0),
        FEW_SHOT=SimpleNamespace(
            QUERY_CLASS_MATCHABILITY=SimpleNamespace(
                DUAL_LOGIT_LOSS_ENABLE=enabled,
                BASE_LOGIT_LOSS_WEIGHT=base_weight,
                VERIFIED_LOGIT_LOSS_WEIGHT=verified_weight,
                EVIDENCE_VIDEO_TOPK_FRAMES=3,
                EVIDENCE_MIL_TEMPERATURE=0.10,
                EVIDENCE_MIL_LOSS_WEIGHT=evidence_mil_weight,
                EVIDENCE_VERIFICATION_ENABLE=True,
            )
        ),
    )


def test_dual_q2s_loss_supervises_base_and_verified_logits_with_same_temperature():
    verified = torch.tensor([[2.0, -2.0]], requires_grad=True)
    base = torch.tensor([[0.5, -0.5]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    cfg = _dual_loss_cfg()

    loss, scaled_verified, diagnostics = compute_query_partial_q2s_loss(
        verified,
        labels,
        {"query_partial_q2s_base_logits": base},
        cfg,
    )
    expected_verified = F.binary_cross_entropy_with_logits(
        verified / 2.0,
        labels,
    )
    expected_base = F.binary_cross_entropy_with_logits(base / 2.0, labels)

    assert torch.equal(scaled_verified, verified / 2.0)
    assert torch.allclose(loss, 0.5 * expected_base + 0.5 * expected_verified)
    assert torch.allclose(diagnostics["q2s_base_loss"], expected_base)
    assert torch.allclose(diagnostics["q2s_verified_loss"], expected_verified)

    loss.backward()
    assert verified.grad is not None and verified.grad.abs().sum() > 0
    assert base.grad is not None and base.grad.abs().sum() > 0


def test_evidence_mil_uses_top_frames_and_excludes_invalid_pairs():
    margins = torch.tensor(
        [[[0.10, 0.40, 0.30, 0.20], [-0.10, -0.40, -0.30, -0.20]]],
        requires_grad=True,
    )
    valid = torch.tensor(
        [[[True, True, True, True], [False, False, False, False]]]
    )
    labels = torch.tensor([[1.0, 0.0]])
    result = compute_evidence_mil_loss(
        margins,
        valid,
        labels,
        topk_frames=3,
        temperature=0.10,
    )

    assert result["video_margin"][0, 0].item() == pytest.approx(0.30)
    assert result["logits"][0, 0].item() == pytest.approx(3.0)
    assert torch.equal(result["valid"], torch.tensor([[True, False]]))
    expected = F.binary_cross_entropy_with_logits(
        torch.tensor([3.0]),
        torch.tensor([1.0]),
    )
    assert torch.allclose(result["loss"], expected)
    result["loss"].backward()
    assert margins.grad is not None
    assert margins.grad[0, 0].abs().sum() > 0
    assert margins.grad[0, 1].abs().sum() == 0


def test_full_document_q2s_loss_adds_weighted_evidence_mil():
    verified = torch.tensor([[2.0, -2.0]], requires_grad=True)
    base = torch.tensor([[0.5, -0.5]], requires_grad=True)
    margins = torch.tensor(
        [[[0.10, 0.40, 0.30, 0.20], [-0.10, -0.40, -0.30, -0.20]]],
        requires_grad=True,
    )
    valid = torch.ones_like(margins, dtype=torch.bool)
    labels = torch.tensor([[1.0, 0.0]])
    cfg = _dual_loss_cfg(
        base_weight=0.60,
        verified_weight=0.40,
        evidence_mil_weight=0.10,
    )
    loss, _, diagnostics = compute_query_partial_q2s_loss(
        verified,
        labels,
        {
            "query_partial_q2s_base_logits": base,
            "query_frame_relative_margin": margins,
            "query_frame_valid": valid,
        },
        cfg,
    )

    expected_verified = F.binary_cross_entropy_with_logits(
        verified / 2.0,
        labels,
    )
    expected_base = F.binary_cross_entropy_with_logits(base / 2.0, labels)
    expected_mil = F.binary_cross_entropy_with_logits(
        torch.tensor([[3.0, -2.0]]),
        labels,
    )
    expected = 0.60 * expected_base + 0.40 * expected_verified + 0.10 * expected_mil
    assert torch.allclose(loss, expected)
    assert torch.allclose(diagnostics["q2s_evidence_mil_loss"], expected_mil)
    loss.backward()
    assert verified.grad is not None and verified.grad.abs().sum() > 0
    assert base.grad is not None and base.grad.abs().sum() > 0
    assert margins.grad is not None and margins.grad.abs().sum() > 0


def test_dual_q2s_loss_fails_loudly_when_base_logits_are_missing():
    with pytest.raises(ValueError, match="base logits"):
        compute_query_partial_q2s_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            {},
            _dual_loss_cfg(enabled=True),
        )


def test_single_q2s_loss_keeps_previous_verified_only_behavior():
    verified = torch.tensor([[2.0, -2.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    loss, scaled, diagnostics = compute_query_partial_q2s_loss(
        verified,
        labels,
        None,
        _dual_loss_cfg(enabled=False),
    )

    expected = F.binary_cross_entropy_with_logits(verified / 2.0, labels)
    assert torch.allclose(loss, expected)
    assert torch.equal(scaled, verified / 2.0)
    assert set(diagnostics) == {"q2s_verified_loss"}
