"""Shared frame matching, masked routing weights and hypothesis transport."""

import pytest
import torch

from trokens.models.query_matching import (
    classwise_frame_similarity, rho_frame_hypothesis_states,
    hypothesis_mass_bimhm_logits, normalize_patch_region_weights,
)


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


def test_rho_states_have_no_retired_m_factor():
    value = torch.tensor([[[0.6, 0.25, 1.0]]], requires_grad=True)
    states = rho_frame_hypothesis_states(value)
    torch.testing.assert_close(states["target_mass"], value, rtol=0, atol=0)
    torch.testing.assert_close(states["confuser_mass"], 1 - value, rtol=0, atol=0)
    assert not states["null_mass"].any()
    states["target_mass"].sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_rho_states_sanitize_nonfinite_values():
    states = rho_frame_hypothesis_states(torch.tensor([float("nan"), float("inf"), -float("inf"), -.1, 2.]))
    assert states["target_mass"].tolist() == [1., 1., 0., 0., 1.]
    assert torch.equal(sum(states.values()), torch.ones(5))


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
