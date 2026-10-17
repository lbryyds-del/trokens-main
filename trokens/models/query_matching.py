"""Support references, frame hypothesis mass and bidirectional matching."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def _safe_unit(value: torch.Tensor) -> torch.Tensor:
    """Return finite FP32 unit vectors without creating NaNs for zero rows."""
    value = torch.nan_to_num(
        value.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def normalize_patch_region_weights(
    region_weights: torch.Tensor,
    point_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a frame mask to class-conditioned ``[B,K,T,N]`` weights.

    Query construction already produces the hypothesis distribution ``pi``.
    Frame specificity reuses its relative patch weights, only removing patches
    that are invalid for evidence verification and renormalizing the remainder.
    """
    if region_weights.ndim != 4:
        raise ValueError(
            "region_weights must have shape [B,K,T,N]; got "
            f"{tuple(region_weights.shape)}."
        )
    batch, _, temporal_dim, num_points = region_weights.shape
    if tuple(point_mask.shape) != (batch, temporal_dim, num_points):
        raise ValueError(
            "point_mask must match region_weights B,T,N; got "
            f"{tuple(point_mask.shape)}, expected "
            f"{(batch, temporal_dim, num_points)}."
        )

    valid = point_mask.to(device=region_weights.device).bool().unsqueeze(1)
    weights = torch.nan_to_num(
        region_weights.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    weights = weights * valid.to(dtype=weights.dtype)
    denominator = weights.sum(dim=-1, keepdim=True)
    normalized = weights / denominator.clamp_min(1e-12)
    return torch.where(
        denominator > 0.0,
        normalized,
        torch.zeros_like(normalized),
    )


def build_class_support_references(
    pointformer: Any,
    value_tokens: torch.Tensor,
    route_point_mask: torch.Tensor,
    support_frame_mask: torch.Tensor,
    support_mask: torch.Tensor,
    episode_positive_labels: torch.Tensor,
    query_label_features: torch.Tensor,
    precomputed_prototypes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one routed reference for every Support/class pair.

    The metric verifier compares positive and confuser Supports in exactly
    the same feature/routing space.  ``route_point_mask`` controls the text
    Softmax, while ``support_frame_mask`` is retained separately so invalid
    Support frames cannot win a later temporal max merely because their zero
    prototype is larger than a negative cosine.

    ``precomputed_prototypes`` may provide the already routed full episode as
    ``[B,K,T,D]`` so the Support references and Query prototypes can share one
    batched temporal-routing call. Returns ``(references, frame_valid,
    positive_valid, confuser_valid, support_indices)`` where references are
    ``[S,K,T,D]``, frame validity is ``[S,K,T]`` and the two class masks are
    ``[S,K]``.
    """
    if value_tokens.ndim != 4:
        raise ValueError(
            "value_tokens must have shape [B,T,N,D]; got "
            f"{tuple(value_tokens.shape)}."
        )
    batch, temporal_dim, _, feat_dim = value_tokens.shape
    expected_point_shape = (batch, temporal_dim, value_tokens.shape[2])
    if tuple(route_point_mask.shape) != expected_point_shape:
        raise ValueError(
            "route_point_mask must match value_tokens B,T,N; got "
            f"{tuple(route_point_mask.shape)} and {expected_point_shape}."
        )
    if tuple(support_frame_mask.shape) != expected_point_shape:
        raise ValueError(
            "support_frame_mask must match value_tokens B,T,N; got "
            f"{tuple(support_frame_mask.shape)} and {expected_point_shape}."
        )
    support_mask = support_mask.to(device=value_tokens.device).bool().flatten()
    if tuple(support_mask.shape) != (batch,):
        raise ValueError(
            "support_mask must have shape [B]; got "
            f"{tuple(support_mask.shape)}."
        )
    labels = episode_positive_labels.to(device=value_tokens.device).bool()
    num_classes = query_label_features.shape[0]
    if tuple(labels.shape) != (batch, num_classes):
        raise ValueError(
            "episode_positive_labels must have shape [B,K]; got "
            f"{tuple(labels.shape)}, expected {(batch, num_classes)}."
        )
    if query_label_features.ndim != 2 or query_label_features.shape[-1] != feat_dim:
        raise ValueError(
            "query_label_features must have shape [K,D] matching values; got "
            f"{tuple(query_label_features.shape)} and D={feat_dim}."
        )
    if precomputed_prototypes is not None:
        expected_prototype_shape = (
            batch,
            num_classes,
            temporal_dim,
            feat_dim,
        )
        if tuple(precomputed_prototypes.shape) != expected_prototype_shape:
            raise ValueError(
                "precomputed_prototypes must have shape [B,K,T,D]; got "
                f"{tuple(precomputed_prototypes.shape)}, expected "
                f"{expected_prototype_shape}."
            )

    route_point_mask = route_point_mask.to(device=value_tokens.device).bool()
    support_frame_mask = support_frame_mask.to(device=value_tokens.device).bool()
    support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
    if support_indices.numel() == 0:
        return (
            value_tokens.new_zeros(0, num_classes, temporal_dim, feat_dim),
            torch.zeros(
                0,
                num_classes,
                temporal_dim,
                device=value_tokens.device,
                dtype=torch.bool,
            ),
            torch.zeros(
                0,
                num_classes,
                device=value_tokens.device,
                dtype=torch.bool,
            ),
            torch.zeros(
                0,
                num_classes,
                device=value_tokens.device,
                dtype=torch.bool,
            ),
            support_indices,
        )

    references = []
    frame_valid_rows = []
    positive_rows = []
    confuser_rows = []
    for sample_idx in support_indices.tolist():
        if precomputed_prototypes is None:
            sample_proto, _ = pointformer._compute_frame_softmax_text_prototypes(
                value_tokens[sample_idx],
                route_point_mask[sample_idx],
                query_label_features,
            )
        else:
            sample_proto = precomputed_prototypes[sample_idx]
        sample_proto = torch.nan_to_num(
            sample_proto,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # A frame must have both a valid Support observation and a finite,
        # non-zero routed prototype to participate in the local temporal max.
        present = support_frame_mask[sample_idx].any(dim=-1)
        frame_valid = present.unsqueeze(0) & sample_proto.float().norm(
            dim=-1,
        ).gt(1e-12)
        class_valid = frame_valid.any(dim=-1)
        sample_labels = labels[sample_idx]
        references.append(sample_proto)
        frame_valid_rows.append(frame_valid)
        positive_rows.append(sample_labels & class_valid)
        confuser_rows.append((~sample_labels) & class_valid)

    return (
        torch.stack(references, dim=0),
        torch.stack(frame_valid_rows, dim=0).bool(),
        torch.stack(positive_rows, dim=0).bool(),
        torch.stack(confuser_rows, dim=0).bool(),
        support_indices,
    )


def factorize_frame_hypothesis_states(
    absolute_mass: torch.Tensor,
    relative_matchability: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Factor one candidate frame into target, confuser, and null states.

    ``absolute_mass`` is the non-null probability ``m`` and
    ``relative_matchability`` is the conditional target probability ``rho``.
    The returned states are ``p+=m*rho``, ``p-=m*(1-rho)`` and ``p0=1-m``.
    They are non-negative and sum to one for every Query/class/frame row.
    """
    if tuple(absolute_mass.shape) != tuple(relative_matchability.shape):
        raise ValueError(
            "absolute_mass and relative_matchability must have the same "
            f"shape; got {tuple(absolute_mass.shape)} and "
            f"{tuple(relative_matchability.shape)}."
        )
    mass = torch.nan_to_num(
        absolute_mass.float(), nan=1.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    specificity = torch.nan_to_num(
        relative_matchability.float(), nan=1.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    target_mass = mass * specificity
    confuser_mass = mass * (1.0 - specificity)
    null_mass = 1.0 - mass
    return {
        "target_mass": target_mass,
        "confuser_mass": confuser_mass,
        "null_mass": null_mass,
    }


def classwise_frame_similarity(
    query_prototypes: torch.Tensor,
    support_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Return the full positive frame matrix ``[Q,K,Tq,Ts]``."""
    if query_prototypes.ndim != 4:
        raise ValueError(
            "query_prototypes must have shape [Q,K,T,D]; got "
            f"{tuple(query_prototypes.shape)}."
        )
    if support_prototypes.ndim != 3:
        raise ValueError(
            "support_prototypes must have shape [K,T,D]; got "
            f"{tuple(support_prototypes.shape)}."
        )
    if (
        query_prototypes.shape[1] != support_prototypes.shape[0]
        or query_prototypes.shape[-1] != support_prototypes.shape[-1]
    ):
        raise ValueError(
            "Query/Support class and feature axes must agree; got "
            f"{tuple(query_prototypes.shape)} and "
            f"{tuple(support_prototypes.shape)}."
        )
    query_unit = _safe_unit(query_prototypes)
    support_unit = _safe_unit(support_prototypes)
    similarity = torch.einsum(
        "qktd,ksd->qkts",
        query_unit,
        support_unit,
    )
    return torch.nan_to_num(
        similarity,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)


def hypothesis_mass_bimhm_logits(
    frame_similarity: torch.Tensor,
    target_mass: torch.Tensor,
    base_logits: torch.Tensor,
    alpha: float = 10.0,
    transport_strength: float = 1.0,
    unmatched_cost: float = 0.0,
    one_sided_transport: bool = True,
) -> Dict[str, torch.Tensor]:
    """Apply target hypothesis mass before both BiMHM reductions.

    Each Query/class/frame candidate transports ``target_mass`` and rejects
    the remainder. Its cosine contribution is
    ``g*C - unmatched_cost*(1-g)`` before the temporal maxima. The optional
    one-sided rule prevents rejection from increasing a negative similarity.
    """
    if frame_similarity.ndim != 4:
        raise ValueError(
            "frame_similarity must have shape [Q,K,Tq,Ts]; got "
            f"{tuple(frame_similarity.shape)}."
        )
    expected_mass = frame_similarity.shape[:3]
    if tuple(target_mass.shape) != tuple(expected_mass):
        raise ValueError(
            "target_mass must have shape [Q,K,Tq]; got "
            f"{tuple(target_mass.shape)}, expected {tuple(expected_mass)}."
        )
    if tuple(base_logits.shape) != tuple(frame_similarity.shape[:2]):
        raise ValueError(
            "base_logits must have shape [Q,K]; got "
            f"{tuple(base_logits.shape)}."
        )
    alpha = float(alpha)
    transport_strength = float(transport_strength)
    unmatched_cost = float(unmatched_cost)
    finite = bool(torch.isfinite(torch.tensor([
        alpha,
        transport_strength,
        unmatched_cost,
    ])).all())
    if not finite or alpha <= 0.0:
        raise ValueError("alpha must be positive and all scalars finite.")
    if not 0.0 <= transport_strength <= 1.0:
        raise ValueError("transport_strength must be in [0, 1].")
    if unmatched_cost < 0.0:
        raise ValueError("unmatched_cost must be non-negative.")
    frame_similarity_fp32 = torch.nan_to_num(
        frame_similarity.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    similarity_logits = alpha * frame_similarity_fp32
    raw_mass = torch.nan_to_num(
        target_mass.float(),
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    effective_patch_mass = 1.0 - transport_strength * (1.0 - raw_mass)
    unmatched_mass = 1.0 - effective_patch_mass
    transported_similarity = (
        effective_patch_mass.unsqueeze(-1) * frame_similarity_fp32
        - unmatched_cost * unmatched_mass.unsqueeze(-1)
    )
    if one_sided_transport:
        transported_similarity = torch.minimum(
            transported_similarity,
            frame_similarity_fp32,
        )

    base_q_to_s = similarity_logits.max(dim=-1).values.mean(dim=-1)
    base_s_to_q, base_winner = similarity_logits.max(dim=-2)
    transported_pair_logits = alpha * transported_similarity
    transported_q_to_s = transported_pair_logits.max(
        dim=-1
    ).values.mean(dim=-1)
    transported_s_to_q, transported_winner = transported_pair_logits.max(
        dim=-2
    )
    q_to_s_delta = transported_q_to_s - base_q_to_s
    s_to_q_delta = (
        transported_s_to_q.mean(dim=-1) - base_s_to_q.mean(dim=-1)
    )
    transport_delta = 0.5 * (q_to_s_delta + s_to_q_delta)
    temporal_logits = base_logits.float() + transport_delta
    if transport_strength == 0.0:
        # Avoid tiny recomputation drift in strict no-op regression tests.
        temporal_logits = base_logits.float()
    winner_switch_fraction = (
        transported_winner.ne(base_winner).float().mean(dim=-1)
    )
    return {
        "logits": torch.nan_to_num(
            temporal_logits,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        ),
        "pair_logits": transported_pair_logits,
        "transported_similarity": transported_similarity,
        "effective_patch_mass": effective_patch_mass,
        "unmatched_mass": unmatched_mass,
        "transport_delta": transport_delta,
        "q_to_s_delta": q_to_s_delta,
        "s_to_q_delta": s_to_q_delta,
        "base_winner": base_winner,
        "verified_winner": transported_winner,
        "winner_switch_fraction": winner_switch_fraction,
    }
