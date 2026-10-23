"""Tests for multi-label few-shot helpers shared across routing modes."""

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tools.few_shot_multilabel import (
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
):
    return SimpleNamespace(
        SOLVER=SimpleNamespace(TEMPRATURE=2.0),
        FEW_SHOT=SimpleNamespace(
            QUERY_CLASS_MATCHABILITY=SimpleNamespace(
                DUAL_LOGIT_LOSS_ENABLE=enabled,
                BASE_LOGIT_LOSS_WEIGHT=base_weight,
                VERIFIED_LOGIT_LOSS_WEIGHT=verified_weight,
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


@pytest.mark.parametrize('all_unknown', [False, True])
def test_base_only_loss_skips_verified_and_mil_and_preserves_unknown_mask(all_unknown, monkeypatch):
    from trokens.config.defaults import get_cfg
    import tools.few_shot_multilabel as helper
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs/trokens/sav.yaml'))
    cfg.SOLVER.TEMPRATURE = .73
    base = torch.randn(2, 3, requires_grad=True)
    verified = torch.randn(2, 3, requires_grad=True)
    quality = torch.randn(2, 3, 2, requires_grad=True)
    relation = torch.randn(2, 3, 2, requires_grad=True)
    labels = torch.tensor([[1., 0., -1.], [float('nan'), 1., 0.]])
    if all_unknown:
        labels.fill_(-1.)
    def forbidden(*args, **kwargs):
        raise AssertionError('Neither verifier MIL should be evaluated')
    monkeypatch.setattr(helper, 'compute_learned_evidence_mil_loss', forbidden)
    calls = []
    original_bce = F.binary_cross_entropy_with_logits
    def counted(*args, **kwargs):
        calls.append(1)
        return original_bce(*args, **kwargs)
    monkeypatch.setattr(helper.F, 'binary_cross_entropy_with_logits', counted)
    aux = dict(query_partial_q2s_base_logits=base, query_quality_logits=quality,
               query_relation_logits=relation)
    loss, scaled, diagnostics = compute_query_partial_q2s_loss(verified, labels, aux, cfg)
    known = torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
    expected = (original_bce(base[known] / .73, labels[known]) if known.any()
                else base.sum() * 0.)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(scaled, base / .73)
    assert len(calls) == (0 if all_unknown else 1)
    assert diagnostics['q2s_verified_loss'].item() == 0.
    assert not any('mil' in key for key in diagnostics)
    loss.backward()
    assert verified.grad is quality.grad is relation.grad is None
    assert torch.isfinite(base.grad).all() and base.grad[~known].eq(0).all()


def test_base_only_rejects_missing_base_and_contradictory_dual_mode():
    from trokens.config.defaults import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs/trokens/sav.yaml'))
    with pytest.raises(ValueError, match='requires base logits'):
        compute_query_partial_q2s_loss(torch.zeros(1, 2), torch.zeros(1, 2), {}, cfg)
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.DUAL_LOGIT_LOSS_ENABLE = True
    with pytest.raises(ValueError, match='dual-logit loss OFF'):
        compute_query_partial_q2s_loss(torch.zeros(1, 2), torch.zeros(1, 2), {}, cfg)


def test_base_only_equals_previous_mixed_loss_when_base_is_final():
    from trokens.config.defaults import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs/trokens/sav.yaml'))
    logits = torch.randn(2, 3, requires_grad=True)
    labels = torch.tensor([[1., 0., -1.], [0., 1., 0.]])
    aux = dict(query_partial_q2s_base_logits=logits)
    clean = compute_query_partial_q2s_loss(logits, labels, aux, cfg)[0]
    old = cfg.clone()
    match = old.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    match.BASE_ONLY_LOSS_ENABLE = False
    match.DUAL_LOGIT_LOSS_ENABLE = True
    match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT = .6, .4
    mixed = compute_query_partial_q2s_loss(logits, labels, aux, old)[0]
    torch.testing.assert_close(clean, mixed)
    torch.testing.assert_close(torch.autograd.grad(clean, logits, retain_graph=True)[0],
                               torch.autograd.grad(mixed, logits)[0])
