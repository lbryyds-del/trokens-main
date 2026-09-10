"""Query-class matchability for multi-label few-shot action recognition.

This extension separates two questions that were coupled by the previous
Query Null token:

1. ``where``: the text/support-routed Softmax constructs the class-conditioned
   Query frame prototype, optionally refined by a bounded per-patch
   positive-versus-confuser Support margin;
2. ``whether``: either the legacy Support-calibrated text evidence or a global
   positive-versus-confuser Support margin estimates whether the Query-class
   hypothesis is matchable at all;
3. ``how much``: an optional Support-calibrated absolute evidence branch splits
   every Query frame into transported patch mass ``m`` and explicit unmatched
   mass ``1-m``.  The unmatched mass is retained by BiMHM instead of being
   normalized back onto arbitrary patches.

The matchability is calibrated from labeled Support videos in the current
episode and enters the final q2s logit as a non-positive log-probability
penalty. No Query target is consumed by this module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


_PATCH_MARKER = "_query_class_matchability_original_builder"


def _safe_unit(value: torch.Tensor) -> torch.Tensor:
    """Return finite FP32 unit vectors without creating NaNs for zero rows."""
    value = torch.nan_to_num(
        value.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    """Read a config attribute while also supporting lightweight test stubs."""
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def masked_topk_mean(
    scores: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    dim: int = -1,
) -> torch.Tensor:
    """Return a masked Top-K mean and zero for fully invalid rows.

    Args:
        scores: Arbitrary floating tensor.
        mask: Boolean tensor broadcastable to ``scores``.
        k: Maximum number of valid values retained along ``dim``.
        dim: Reduction dimension.
    """
    if scores.numel() == 0:
        output_shape = list(scores.shape)
        output_shape.pop(dim % scores.ndim)
        return scores.new_zeros(output_shape, dtype=torch.float32)

    scores_fp32 = torch.nan_to_num(
        scores.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    mask = mask.to(device=scores.device).bool()
    if mask.shape != scores.shape:
        mask = mask.expand_as(scores)

    dim = dim % scores.ndim
    topk = max(1, min(int(k), scores.shape[dim]))
    masked_scores = scores_fp32.masked_fill(~mask, float("-inf"))
    values, indices = torch.topk(masked_scores, k=topk, dim=dim)
    selected_valid = torch.gather(mask, dim, indices)
    values = torch.where(selected_valid, values, torch.zeros_like(values))
    denominator = selected_valid.sum(dim=dim).clamp_min(1).to(values.dtype)
    result = values.sum(dim=dim) / denominator
    any_valid = mask.any(dim=dim)
    result = torch.where(any_valid, result, torch.zeros_like(result))
    return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)


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


def pairwise_bimhm(
    query_prototypes: torch.Tensor,
    support_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Compute the same bidirectional frame matcher for every pair.

    The normal Pointformer route compares one ``[Q,K,T,D]`` Query tensor with
    one ``[K,T,D]`` positive Support tensor.  The confuser branch needs the
    corresponding score against every negative Support, so this helper
    accepts ``[S,K,T,D]`` and returns ``[Q,K,S]``.  For convenience, the
    single-class form ``[Q,T,D]``/``[S,T,D]`` returns ``[Q,S]``.

    The reductions intentionally mirror ``Pointformer._compute_bidirectional_
    frame_similarity``: a max over the other video's frames followed by a
    mean over the source video's frames in each direction.  Keeping this
    definition identical is important because the positive term is the
    existing q2s score.
    """
    if query_prototypes.ndim not in (3, 4):
        raise ValueError(
            "query_prototypes must have shape [Q,T,D] or [Q,K,T,D]; got "
            f"{tuple(query_prototypes.shape)}."
        )
    if support_prototypes.ndim not in (3, 4):
        raise ValueError(
            "support_prototypes must have shape [S,T,D] or [S,K,T,D]; got "
            f"{tuple(support_prototypes.shape)}."
        )

    squeeze_class = query_prototypes.ndim == 3
    if query_prototypes.ndim == 3:
        query_prototypes = query_prototypes.unsqueeze(1)
    if support_prototypes.ndim == 3:
        support_prototypes = support_prototypes.unsqueeze(1)
    if query_prototypes.shape[1] != support_prototypes.shape[1]:
        raise ValueError(
            "Query and Support class axes must agree; got "
            f"{query_prototypes.shape[1]} and {support_prototypes.shape[1]}."
        )
    if query_prototypes.shape[-1] != support_prototypes.shape[-1]:
        raise ValueError(
            "Query and Support feature dimensions must agree; got "
            f"{query_prototypes.shape[-1]} and {support_prototypes.shape[-1]}."
        )

    # ``torch.max`` is undefined over an empty Support axis.  An episode may
    # legitimately contain no confuser for a class (for example when every
    # sampled Support carries that label), in which case the caller's masked
    # reducer should receive an empty/neutral score tensor instead.
    if support_prototypes.shape[0] == 0:
        empty = query_prototypes.new_zeros(
            query_prototypes.shape[0],
            query_prototypes.shape[1],
            0,
            dtype=torch.float32,
        )
        return empty[:, 0] if squeeze_class else empty
    if query_prototypes.shape[2] == 0 or support_prototypes.shape[2] == 0:
        empty = query_prototypes.new_zeros(
            query_prototypes.shape[0],
            query_prototypes.shape[1],
            support_prototypes.shape[0],
            dtype=torch.float32,
        )
        return empty[:, 0] if squeeze_class else empty

    query_norm = _safe_unit(query_prototypes)
    support_norm = _safe_unit(support_prototypes)
    # [Q,K,Tq,D] x [S,K,Ts,D] -> [Q,K,Tq,S,Ts].
    sim = torch.einsum("qktd,skud->qktsu", query_norm, support_norm)
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
        -1.0,
        1.0,
    )
    # After the Support-time max the dimensions are [Q,K,Tq,S]; average over
    # Query time (dim=2), while retaining the Support-example axis.
    query_to_support = sim.max(dim=-1).values.mean(dim=2)
    support_to_query = sim.max(dim=2).values.mean(dim=-1)
    result = 0.5 * (query_to_support + support_to_query)
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
        -1.0,
        1.0,
    )
    return result[:, 0] if squeeze_class else result


def _aggregate_negative_similarity(
    negative_similarity: torch.Tensor,
    negative_valid: torch.Tensor,
    aggregation: str = "max",
    topk: int = 2,
    temperature: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate ``[Q,K,S]`` confuser scores with a ``[K,S]`` validity mask.

    Returns the aggregate, the index of the hardest individual confuser (for
    diagnostics), and the number of valid confusers per class.  Invalid-only
    classes receive an aggregate of zero and index ``-1``.
    """
    if negative_similarity.ndim != 3:
        raise ValueError(
            "negative_similarity must have shape [Q,K,S]; got "
            f"{tuple(negative_similarity.shape)}."
        )
    if negative_valid.ndim != 2 or tuple(negative_valid.shape) != tuple(
        negative_similarity.shape[1:]
    ):
        raise ValueError(
            "negative_valid must have shape [K,S] matching negative scores; "
            f"got {tuple(negative_valid.shape)}, expected "
            f"{tuple(negative_similarity.shape[1:])}."
        )
    if negative_similarity.shape[-1] == 0:
        shape = negative_similarity.shape[:2]
        zero = negative_similarity.new_zeros(shape, dtype=torch.float32)
        index = torch.full(shape, -1, device=negative_similarity.device, dtype=torch.long)
        count = torch.zeros(
            negative_similarity.shape[1],
            device=negative_similarity.device,
            dtype=torch.long,
        )
        return zero, index, count

    scores = torch.nan_to_num(
        negative_similarity.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    valid = negative_valid.to(device=scores.device).bool()
    valid_broadcast = valid.unsqueeze(0)
    counts = valid.sum(dim=-1)
    masked = scores.masked_fill(~valid_broadcast, float("-inf"))
    hardest, hardest_index = masked.max(dim=-1)
    has_valid = counts.unsqueeze(0) > 0
    hardest = torch.where(has_valid, hardest, torch.zeros_like(hardest))
    hardest_index = torch.where(
        has_valid,
        hardest_index,
        torch.full_like(hardest_index, -1),
    )

    mode = str(aggregation).lower()
    if mode in {"max", "hard_max", "hardest"}:
        aggregate = hardest
    elif mode in {"topk_mean", "topk", "top_k_mean"}:
        k = max(1, min(int(topk), scores.shape[-1]))
        values, indices = torch.topk(masked, k=k, dim=-1)
        selected_valid = torch.gather(valid_broadcast.expand_as(scores), -1, indices)
        values = torch.where(selected_valid, values, torch.zeros_like(values))
        denom = selected_valid.sum(dim=-1).clamp_min(1).to(values.dtype)
        aggregate = values.sum(dim=-1) / denom
        aggregate = torch.where(has_valid, aggregate, torch.zeros_like(aggregate))
    elif mode in {"logmeanexp", "log_mean_exp", "lme"}:
        if temperature <= 0.0:
            raise ValueError("negative aggregation temperature must be positive.")
        # Subtract log(M) so the scale does not depend on how many negatives
        # happen to be present in a multi-label episode.
        lse = torch.logsumexp(masked / float(temperature), dim=-1)
        count_log = counts.clamp_min(1).to(lse.dtype).log().unsqueeze(0)
        aggregate = float(temperature) * (lse - count_log)
        aggregate = torch.where(has_valid, aggregate, torch.zeros_like(aggregate))
    else:
        raise ValueError(
            "NEGATIVE_AGGREGATION must be 'max', 'topk_mean' or "
            f"'logmeanexp'; got {aggregation!r}."
        )

    return (
        torch.nan_to_num(aggregate, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
            -1.0,
            1.0,
        ),
        hardest_index,
        counts,
    )


def _compute_relative_matchability_with_diagnostics(
    positive_similarity: torch.Tensor,
    negative_similarity: torch.Tensor,
    negative_valid: torch.Tensor,
    temperature: float = 0.10,
    bias: float = 0.0,
    aggregation: str = "max",
    topk: int = 2,
    negative_temperature: float = 0.10,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Internal relative-matchability computation with diagnostics."""
    if positive_similarity.ndim != 2:
        raise ValueError(
            "positive_similarity must have shape [Q,K]; got "
            f"{tuple(positive_similarity.shape)}."
        )
    if negative_similarity.ndim != 3:
        raise ValueError(
            "negative_similarity must have shape [Q,K,S]; got "
            f"{tuple(negative_similarity.shape)}."
        )
    if tuple(negative_similarity.shape[:2]) != tuple(positive_similarity.shape):
        raise ValueError(
            "Positive and negative similarities must share [Q,K]; got "
            f"{tuple(positive_similarity.shape)} and "
            f"{tuple(negative_similarity.shape)}."
        )
    if temperature <= 0.0:
        raise ValueError("MARGIN_TEMPERATURE must be positive.")

    negative_aggregate, hardest_index, valid_count = _aggregate_negative_similarity(
        negative_similarity,
        negative_valid,
        aggregation=aggregation,
        topk=topk,
        temperature=negative_temperature,
    )
    positive = torch.nan_to_num(
        positive_similarity.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    margin = positive - negative_aggregate
    rho = torch.sigmoid((margin - float(bias)) / float(temperature))
    has_negative = valid_count.unsqueeze(0).to(device=rho.device) > 0
    # No comparison object means the branch is undefined; leave the original
    # q2s score unchanged by assigning neutral rho=1.
    rho = torch.where(has_negative, rho, torch.ones_like(rho))
    margin = torch.where(has_negative, margin, torch.zeros_like(margin))
    negative_aggregate = torch.where(
        has_negative,
        negative_aggregate,
        torch.zeros_like(negative_aggregate),
    )
    hardest_index = torch.where(
        has_negative,
        hardest_index,
        torch.full_like(hardest_index, -1),
    )
    return rho.clamp(0.0, 1.0), margin, negative_aggregate, hardest_index, valid_count


def compute_relative_matchability(
    positive_similarity: torch.Tensor,
    negative_similarity: torch.Tensor,
    negative_valid: torch.Tensor,
    temperature: float = 0.10,
    bias: float = 0.0,
    aggregation: str = "max",
    topk: int = 2,
    negative_temperature: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert positive-vs-confuser similarity into ``rho``.

    ``negative_valid`` uses the class-major shape ``[K,S]``.  The public
    return is ``(rho, margin, aggregated_negative_similarity)``; the internal
    builder additionally records the hardest Support index and valid count.
    """
    rho, margin, negative_aggregate, _, _ = (
        _compute_relative_matchability_with_diagnostics(
            positive_similarity,
            negative_similarity,
            negative_valid,
            temperature=temperature,
            bias=bias,
            aggregation=aggregation,
            topk=topk,
            negative_temperature=negative_temperature,
        )
    )
    return rho, margin, negative_aggregate


def build_class_confuser_prototypes(
    pointformer: Any,
    value_tokens: torch.Tensor,
    point_mask: torch.Tensor,
    support_mask: torch.Tensor,
    episode_positive_labels: torch.Tensor,
    query_label_features: torch.Tensor,
    detach_support: bool = False,
    temporal_point_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-Support, per-class confuser prototypes.

    A Support is a valid confuser for class ``k`` exactly when its labeled
    episode vector does not contain ``k``.  The route is still class-guided by
    the same Query-side feature ``query_label_features``; no Query labels are
    read.  Returned tensors are ``[S,K,T,D]``, ``[S,K]`` and global Support
    indices ``[S]``.
    """
    if value_tokens.ndim != 4:
        raise ValueError(
            "value_tokens must have shape [B,T,N,D]; got "
            f"{tuple(value_tokens.shape)}."
        )
    batch, temporal_dim, _, feat_dim = value_tokens.shape
    if tuple(point_mask.shape) != tuple(value_tokens.shape[:3]):
        raise ValueError(
            "point_mask must match value_tokens B,T,N; got "
            f"{tuple(point_mask.shape)} and {tuple(value_tokens.shape[:3])}."
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

    support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
    if support_indices.numel() == 0:
        empty_proto = value_tokens.new_zeros(
            0,
            num_classes,
            temporal_dim,
            feat_dim,
        )
        empty_valid = torch.zeros(
            0,
            num_classes,
            device=value_tokens.device,
            dtype=torch.bool,
        )
        return empty_proto, empty_valid, support_indices

    prototypes = []
    valid_rows = []
    for sample_idx in support_indices.tolist():
        sample_proto, _ = pointformer._compute_frame_softmax_text_prototypes(
            value_tokens[sample_idx],
            point_mask[sample_idx],
            query_label_features,
            **({} if temporal_point_mask is None else {
                "temporal_point_mask": temporal_point_mask[sample_idx],
            }),
        )
        if detach_support:
            sample_proto = sample_proto.detach()
        sample_proto = torch.nan_to_num(
            sample_proto,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        frame_valid = sample_proto.float().norm(dim=-1).gt(1e-12).any(dim=-1)
        valid_rows.append((~labels[sample_idx]) & frame_valid)
        prototypes.append(sample_proto)

    return (
        torch.stack(prototypes, dim=0),
        torch.stack(valid_rows, dim=0).bool(),
        support_indices,
    )


def build_class_local_support_references(
    pointformer: Any,
    value_tokens: torch.Tensor,
    route_point_mask: torch.Tensor,
    support_frame_mask: torch.Tensor,
    support_mask: torch.Tensor,
    episode_positive_labels: torch.Tensor,
    query_label_features: torch.Tensor,
    temporal_point_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one routed reference for every Support/class pair.

    The local refinement compares positive and confuser Supports in exactly
    the same feature/routing space.  ``route_point_mask`` controls the text
    Softmax, while ``support_frame_mask`` is retained separately so invalid
    Support frames cannot win a later temporal max merely because their zero
    prototype is larger than a negative cosine.

    Returns ``(references, frame_valid, positive_valid, confuser_valid,
    support_indices)`` where references are ``[S,K,T,D]``, frame validity is
    ``[S,K,T]`` and the two class masks are ``[S,K]``.
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
        sample_proto, _ = pointformer._compute_frame_softmax_text_prototypes(
            value_tokens[sample_idx],
            route_point_mask[sample_idx],
            query_label_features,
            **({} if temporal_point_mask is None else {
                "temporal_point_mask": temporal_point_mask[sample_idx],
            }),
        )
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


def compute_local_positive_confuser_margin(
    query_tokens: torch.Tensor,
    support_references: torch.Tensor,
    support_frame_valid: torch.Tensor,
    positive_valid: torch.Tensor,
    confuser_valid: torch.Tensor,
    point_mask: torch.Tensor,
    positive_aggregation: str = "topk_mean",
    positive_topk: int = 2,
    negative_aggregation: str = "topk_mean",
    negative_topk: int = 2,
    positive_temperature: float = 0.10,
    negative_temperature: float = 0.10,
    detach_references: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a per-patch positive-minus-confuser Support margin.

    ``query_tokens`` is ``[Q,T,N,D]`` and ``support_references`` is
    ``[S,K,T_s,D]``.  Positive and confuser Supports are aggregated with the
    same masked reducer, which avoids treating an averaged positive prototype
    and an individual hard negative as if they had identical statistics.
    The function never consumes Query labels.
    """
    if query_tokens.ndim != 4:
        raise ValueError(
            "query_tokens must have shape [Q,T,N,D]; got "
            f"{tuple(query_tokens.shape)}."
        )
    if support_references.ndim != 4:
        raise ValueError(
            "support_references must have shape [S,K,T,D]; got "
            f"{tuple(support_references.shape)}."
        )
    q, query_time, num_points, feat_dim = query_tokens.shape
    support_count, num_classes, support_time, support_dim = support_references.shape
    if feat_dim != support_dim:
        raise ValueError(
            "Query and Support feature dimensions differ: "
            f"{feat_dim} versus {support_dim}."
        )
    if tuple(support_frame_valid.shape) != (
        support_count,
        num_classes,
        support_time,
    ):
        raise ValueError(
            "support_frame_valid must have shape [S,K,T_s]; got "
            f"{tuple(support_frame_valid.shape)}."
        )
    if tuple(positive_valid.shape) != (support_count, num_classes):
        raise ValueError(
            "positive_valid must have shape [S,K]; got "
            f"{tuple(positive_valid.shape)}."
        )
    if tuple(confuser_valid.shape) != (support_count, num_classes):
        raise ValueError(
            "confuser_valid must have shape [S,K]; got "
            f"{tuple(confuser_valid.shape)}."
        )
    if tuple(point_mask.shape) != (q, query_time, num_points):
        raise ValueError(
            "point_mask must match query_tokens B,T,N; got "
            f"{tuple(point_mask.shape)}."
        )
    temperatures_finite = bool(torch.isfinite(torch.tensor([
        float(positive_temperature),
        float(negative_temperature),
    ])).all())
    if (
        not temperatures_finite
        or positive_temperature <= 0.0
        or negative_temperature <= 0.0
    ):
        raise ValueError("Local Support aggregation temperatures must be positive.")

    output_shape = (q, num_classes, query_time, num_points)
    if support_count == 0 or q == 0 or num_classes == 0:
        zero = query_tokens.new_zeros(output_shape, dtype=torch.float32)
        counts = torch.zeros(
            num_classes,
            device=query_tokens.device,
            dtype=torch.long,
        )
        return zero, zero.clone(), zero.clone(), counts, counts.clone()

    query_unit = _safe_unit(query_tokens)
    support_unit = _safe_unit(support_references)
    if detach_references:
        support_unit = support_unit.detach()
    support_frame_valid = support_frame_valid.to(
        device=query_tokens.device,
        dtype=torch.bool,
    )

    # Reduce Support time immediately for each Support sample.  This keeps
    # the peak tensor at [Q,K,T_query,N,T_support] instead of materializing a
    # [Q,K,T_query,N,S,T_support] tensor.
    per_support_scores = []
    for support_idx in range(support_count):
        similarity = torch.einsum(
            "qtnd,kud->qktnu",
            query_unit,
            support_unit[support_idx],
        )
        frame_mask = support_frame_valid[support_idx].view(
            1,
            num_classes,
            1,
            1,
            support_time,
        )
        masked_similarity = similarity.masked_fill(
            ~frame_mask,
            float("-inf"),
        )
        score = masked_similarity.max(dim=-1).values
        has_frame = frame_mask.any(dim=-1)
        score = torch.where(has_frame, score, torch.zeros_like(score))
        per_support_scores.append(
            torch.nan_to_num(
                score,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(-1.0, 1.0)
        )
    support_scores = torch.stack(per_support_scores, dim=-1)
    # [Q,K,T,N,S] -> [Q*T*N,K,S], matching the global reducer's API.
    flat_scores = support_scores.permute(0, 2, 3, 1, 4).reshape(
        q * query_time * num_points,
        num_classes,
        support_count,
    )
    positive_scores, _, positive_counts = _aggregate_negative_similarity(
        flat_scores,
        positive_valid.to(device=query_tokens.device).bool().transpose(0, 1),
        aggregation=positive_aggregation,
        topk=positive_topk,
        temperature=positive_temperature,
    )
    negative_scores, _, negative_counts = _aggregate_negative_similarity(
        flat_scores,
        confuser_valid.to(device=query_tokens.device).bool().transpose(0, 1),
        aggregation=negative_aggregation,
        topk=negative_topk,
        temperature=negative_temperature,
    )

    def _restore(values):
        return values.reshape(q, query_time, num_points, num_classes).permute(
            0,
            3,
            1,
            2,
        )

    positive_score = _restore(positive_scores)
    negative_score = _restore(negative_scores)
    has_both = (positive_counts > 0) & (negative_counts > 0)
    margin = positive_score - negative_score
    margin = torch.where(
        has_both.view(1, num_classes, 1, 1),
        margin,
        torch.zeros_like(margin),
    )
    query_valid = point_mask.to(
        device=query_tokens.device,
        dtype=margin.dtype,
    ).unsqueeze(1)
    margin = margin * query_valid
    positive_score = positive_score * query_valid
    negative_score = negative_score * query_valid
    margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        margin,
        torch.nan_to_num(positive_score, nan=0.0, posinf=1.0, neginf=-1.0),
        torch.nan_to_num(negative_score, nan=0.0, posinf=1.0, neginf=-1.0),
        positive_counts,
        negative_counts,
    )


def compute_evidence_conditioned_frame_matchability(
    evidence_weights: torch.Tensor,
    positive_response: torch.Tensor,
    confuser_response: torch.Tensor,
    frame_valid: torch.Tensor,
    positive_count: torch.Tensor,
    confuser_count: torch.Tensor,
    temperature: float = 0.10,
    bias: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compare Positive and Confuser explanations on one hypothesis region."""
    if evidence_weights.ndim != 4:
        raise ValueError(
            "evidence_weights must have shape [Q,K,T,N]; got "
            f"{tuple(evidence_weights.shape)}."
        )
    if tuple(positive_response.shape) != tuple(evidence_weights.shape):
        raise ValueError(
            "positive_response must match evidence_weights; got "
            f"{tuple(positive_response.shape)} and "
            f"{tuple(evidence_weights.shape)}."
        )
    if tuple(confuser_response.shape) != tuple(evidence_weights.shape):
        raise ValueError(
            "confuser_response must match evidence_weights; got "
            f"{tuple(confuser_response.shape)} and "
            f"{tuple(evidence_weights.shape)}."
        )
    q, num_classes, temporal_dim, _ = evidence_weights.shape
    if tuple(frame_valid.shape) != (q, temporal_dim):
        raise ValueError(
            "frame_valid must have shape [Q,T]; got "
            f"{tuple(frame_valid.shape)}."
        )
    if tuple(positive_count.shape) != (num_classes,):
        raise ValueError(
            "positive_count must have shape [K]; got "
            f"{tuple(positive_count.shape)}."
        )
    if tuple(confuser_count.shape) != (num_classes,):
        raise ValueError(
            "confuser_count must have shape [K]; got "
            f"{tuple(confuser_count.shape)}."
        )
    temperature = float(temperature)
    bias = float(bias)
    finite = bool(torch.isfinite(torch.tensor([temperature, bias])).all())
    if not finite or temperature <= 0.0:
        raise ValueError(
            "Frame margin temperature must be positive and bias finite."
        )

    weights = torch.nan_to_num(
        evidence_weights.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    positive_evidence = (
        weights * torch.nan_to_num(positive_response.float(), nan=0.0)
    ).sum(dim=-1)
    confuser_evidence = (
        weights * torch.nan_to_num(confuser_response.float(), nan=0.0)
    ).sum(dim=-1)
    class_valid = (
        positive_count.to(device=weights.device).gt(0)
        & confuser_count.to(device=weights.device).gt(0)
    )
    valid = (
        frame_valid.to(device=weights.device).bool().unsqueeze(1)
        & class_valid.view(1, num_classes, 1)
    )
    margin = positive_evidence - confuser_evidence
    margin = torch.where(valid, margin, torch.zeros_like(margin))
    rho = torch.sigmoid((margin - bias) / temperature)
    # Missing frames or comparison classes must be a strict no-op.
    rho = torch.where(valid, rho, torch.ones_like(rho))
    rho = torch.nan_to_num(rho, nan=1.0, posinf=1.0, neginf=0.0).clamp(
        0.0,
        1.0,
    )
    positive_evidence = torch.where(
        valid,
        positive_evidence,
        torch.zeros_like(positive_evidence),
    )
    confuser_evidence = torch.where(
        valid,
        confuser_evidence,
        torch.zeros_like(confuser_evidence),
    )
    return {
        "positive_evidence": positive_evidence,
        "confuser_evidence": confuser_evidence,
        "margin": margin,
        "matchability": rho,
        "valid": valid,
        "class_valid": class_valid,
    }


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


def compute_support_calibrated_frame_transport_mass(
    similarity: torch.Tensor,
    point_mask: torch.Tensor,
    support_mask: torch.Tensor,
    episode_positive_labels: torch.Tensor,
    cfg: Any,
) -> Dict[str, torch.Tensor]:
    """Split Query-frame quality into patch and unmatched mass.

    ``similarity`` is pure-text cosine evidence in a common semantic space,
    normally Raw DinoTxt, with shape ``[B,K,T,N]``.  A Support video first
    produces a robust absolute score by taking the Top-K patches per frame and
    the Top-L frames per video.  Labeled positive and negative Support videos
    then calibrate one threshold per episode class.  Query *frame* evidence is
    compared with that threshold to obtain ``m[q,k,t]``.

    Query labels are deliberately never indexed.  For a class whose Support
    calibration is missing or inverted, the default reliability fallback sets
    valid Query-frame patch mass to one, exactly disabling this branch.  Frames
    with no valid trajectory have zero patch mass and unit unmatched mass.
    """
    if similarity.ndim != 4:
        raise ValueError(
            "similarity must have shape [B,K,T,N]; got "
            f"{tuple(similarity.shape)}."
        )
    batch, num_classes, temporal_dim, num_points = similarity.shape
    if tuple(point_mask.shape) != (batch, temporal_dim, num_points):
        raise ValueError(
            "point_mask must match similarity B,T,N; got "
            f"{tuple(point_mask.shape)}."
        )
    support_mask = support_mask.to(device=similarity.device).bool().flatten()
    if tuple(support_mask.shape) != (batch,):
        raise ValueError(
            "support_mask must have shape [B]; got "
            f"{tuple(support_mask.shape)}."
        )
    labels = episode_positive_labels.to(device=similarity.device).bool()
    if tuple(labels.shape) != (batch, num_classes):
        raise ValueError(
            "episode_positive_labels must have shape [B,K]; got "
            f"{tuple(labels.shape)}, expected {(batch, num_classes)}."
        )
    if not support_mask.any():
        raise ValueError("At least one labeled Support sample is required.")

    patch_topk = int(_cfg_value(cfg, "ABSOLUTE_MASS_PATCH_TOPK", 8))
    frame_topk = int(
        _cfg_value(cfg, "ABSOLUTE_MASS_SUPPORT_TOPK_FRAMES", 3)
    )
    beta = float(_cfg_value(cfg, "ABSOLUTE_MASS_CALIBRATION_BETA", 0.25))
    temperature = float(_cfg_value(cfg, "ABSOLUTE_MASS_TEMPERATURE", 0.05))
    min_gap = float(_cfg_value(cfg, "ABSOLUTE_MASS_MIN_SUPPORT_GAP", 0.0))
    finite = bool(
        torch.isfinite(torch.tensor([beta, temperature, min_gap])).all()
    )
    if patch_topk <= 0 or frame_topk <= 0:
        raise ValueError(
            "ABSOLUTE_MASS_PATCH_TOPK and "
            "ABSOLUTE_MASS_SUPPORT_TOPK_FRAMES must be positive."
        )
    if not finite or temperature <= 0.0 or min_gap < 0.0:
        raise ValueError(
            "Absolute-mass temperature must be positive and min gap finite/non-negative."
        )
    if not 0.0 <= beta <= 1.0:
        raise ValueError("ABSOLUTE_MASS_CALIBRATION_BETA must be in [0, 1].")

    similarity = torch.nan_to_num(
        similarity.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    point_mask = point_mask.to(device=similarity.device).bool()
    patch_valid = point_mask.unsqueeze(1).expand_as(similarity)
    frame_evidence = masked_topk_mean(
        similarity,
        patch_valid,
        patch_topk,
        dim=-1,
    )
    frame_valid = point_mask.any(dim=-1)
    class_frame_valid = frame_valid.unsqueeze(1).expand(
        batch,
        num_classes,
        temporal_dim,
    )
    video_evidence = masked_topk_mean(
        frame_evidence,
        class_frame_valid,
        frame_topk,
        dim=-1,
    )

    support_evidence = video_evidence[support_mask]
    support_targets = labels[support_mask]
    positive_mean, positive_count = _classwise_masked_mean(
        support_evidence,
        support_targets,
    )
    negative_targets = ~support_targets
    negative_mean, negative_count = _classwise_masked_mean(
        support_evidence,
        negative_targets,
    )
    all_negative = support_evidence[negative_targets]
    global_negative = (
        all_negative.mean()
        if all_negative.numel() > 0
        else support_evidence.mean()
    )
    negative_mean = torch.where(
        negative_count > 0,
        negative_mean,
        global_negative.expand_as(negative_mean),
    )
    positive_mean = torch.where(
        positive_count > 0,
        positive_mean,
        negative_mean,
    )
    support_gap = positive_mean - negative_mean
    support_reliable = (
        (positive_count > 0)
        & (negative_count > 0)
        & (support_gap > min_gap)
    )
    threshold = negative_mean + beta * support_gap.clamp_min(0.0)
    if bool(_cfg_value(cfg, "ABSOLUTE_MASS_DETACH_SUPPORT_STATS", True)):
        positive_mean = positive_mean.detach()
        negative_mean = negative_mean.detach()
        support_gap = support_gap.detach()
        threshold = threshold.detach()

    query_frame_evidence = frame_evidence[~support_mask]
    query_frame_valid = frame_valid[~support_mask]
    raw_patch_mass = torch.sigmoid(
        (query_frame_evidence - threshold.view(1, num_classes, 1))
        / temperature
    )
    raw_patch_mass = torch.nan_to_num(
        raw_patch_mass,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    valid = query_frame_valid.unsqueeze(1).expand_as(raw_patch_mass)
    if bool(_cfg_value(cfg, "ABSOLUTE_MASS_RELIABILITY_FALLBACK", True)):
        reliable = support_reliable.view(1, num_classes, 1)
        patch_mass = torch.where(reliable, raw_patch_mass, torch.ones_like(raw_patch_mass))
    else:
        patch_mass = raw_patch_mass
    patch_mass = torch.where(valid, patch_mass, torch.zeros_like(patch_mass))
    unmatched_mass = 1.0 - patch_mass

    valid_float = valid.float()
    valid_count = valid_float.sum(dim=-1)
    mean_patch_mass = (patch_mass * valid_float).sum(dim=-1) / valid_count.clamp_min(1.0)
    mean_patch_mass = torch.where(
        valid_count > 0,
        mean_patch_mass,
        torch.zeros_like(mean_patch_mass),
    )
    top_patch_mass = masked_topk_mean(
        patch_mass,
        valid,
        frame_topk,
        dim=-1,
    )
    mean_frame_evidence = (
        query_frame_evidence * valid_float
    ).sum(dim=-1) / valid_count.clamp_min(1.0)
    mean_frame_evidence = torch.where(
        valid_count > 0,
        mean_frame_evidence,
        torch.zeros_like(mean_frame_evidence),
    )
    return {
        "frame_evidence": frame_evidence,
        "video_evidence": video_evidence,
        "query_frame_evidence": query_frame_evidence,
        "query_frame_valid": query_frame_valid,
        "raw_patch_mass": raw_patch_mass,
        "patch_mass": patch_mass,
        "unmatched_mass": unmatched_mass,
        "mean_patch_mass": mean_patch_mass,
        "top_patch_mass": top_patch_mass,
        "mean_frame_evidence": mean_frame_evidence,
        "support_positive_evidence_mean": positive_mean,
        "support_negative_evidence_mean": negative_mean,
        "support_positive_count": positive_count,
        "support_negative_count": negative_count,
        "support_gap": support_gap,
        "support_reliable": support_reliable,
        "threshold": threshold,
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


def build_confuser_refined_query_prototypes(
    pointformer: Any,
    query_tokens: torch.Tensor,
    point_mask: torch.Tensor,
    query_label_features: torch.Tensor,
    local_margin: torch.Tensor,
    patch_tau: Optional[float] = None,
    local_tau: float = 0.10,
    local_strength: float = 0.50,
) -> Dict[str, torch.Tensor]:
    """Re-route Query patches with a bounded local margin residual.

    The residual is expressed in Softmax-logit units, then converted to cosine
    units before calling Pointformer's canonical masked Softmax builder.  This
    makes ``local_strength=0`` use exactly the same route implementation as
    the pre-refinement path.
    """
    if query_tokens.ndim != 4:
        raise ValueError(
            "query_tokens must have shape [Q,T,N,D]; got "
            f"{tuple(query_tokens.shape)}."
        )
    q, temporal_dim, num_points, feat_dim = query_tokens.shape
    if point_mask.shape != (q, temporal_dim, num_points):
        raise ValueError(
            "point_mask must match query_tokens B,T,N; got "
            f"{tuple(point_mask.shape)}."
        )
    if query_label_features.ndim != 2 or query_label_features.shape[-1] != feat_dim:
        raise ValueError(
            "query_label_features must have shape [K,D] matching query tokens; got "
            f"{tuple(query_label_features.shape)} and D={feat_dim}."
        )
    if local_margin.shape != (
        q,
        query_label_features.shape[0],
        temporal_dim,
        num_points,
    ):
        raise ValueError(
            "local_margin must have shape [Q,K,T,N]; got "
            f"{tuple(local_margin.shape)}."
        )
    if patch_tau is None:
        patch_tau = getattr(
            getattr(pointformer, "pot_route_cfg", None),
            "FRAME_SOFTMAX_TAU",
            0.07,
        )
    patch_tau = float(patch_tau)
    local_tau = float(local_tau)
    local_strength = float(local_strength)
    route_values_finite = bool(torch.isfinite(torch.tensor([
        patch_tau,
        local_tau,
        local_strength,
    ])).all())
    if not route_values_finite:
        raise ValueError("patch_tau, local_tau and local_strength must be finite.")
    if patch_tau <= 0.0 or local_tau <= 0.0:
        raise ValueError("patch_tau and local_tau must be positive.")
    if local_strength < 0.0:
        raise ValueError("local_strength must be non-negative.")

    query_unit = _safe_unit(query_tokens)
    text_unit = _safe_unit(query_label_features.to(device=query_tokens.device))
    base_similarity = torch.einsum(
        "kd,qtnd->qktn",
        text_unit,
        query_unit,
    )
    base_similarity = torch.nan_to_num(
        base_similarity,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    local_residual = local_strength * torch.tanh(
        local_margin.float() / local_tau,
    )
    valid = point_mask.to(device=query_tokens.device).bool().unsqueeze(1)
    local_residual = local_residual * valid.to(local_residual.dtype)
    # The canonical builder divides by patch_tau internally, so adding this
    # amount in cosine units is equivalent to adding local_residual logits.
    refined_similarity = base_similarity + patch_tau * local_residual

    base_prototypes = []
    refined_prototypes = []
    base_weights = []
    refined_weights = []
    for query_idx in range(q):
        base_proto, base_weight = (
            pointformer._compute_frame_softmax_prototypes_from_similarity(
                query_tokens[query_idx],
                point_mask[query_idx],
                base_similarity[query_idx],
                softmax_tau=patch_tau,
            )
        )
        refined_proto, refined_weight = (
            pointformer._compute_frame_softmax_prototypes_from_similarity(
                query_tokens[query_idx],
                point_mask[query_idx],
                refined_similarity[query_idx],
                softmax_tau=patch_tau,
            )
        )
        base_prototypes.append(base_proto)
        refined_prototypes.append(refined_proto)
        base_weights.append(base_weight)
        refined_weights.append(refined_weight)

    if q:
        base_prototypes = torch.stack(base_prototypes, dim=0)
        refined_prototypes = torch.stack(refined_prototypes, dim=0)
        base_weights = torch.stack(base_weights, dim=0)
        refined_weights = torch.stack(refined_weights, dim=0)
    else:
        num_classes = query_label_features.shape[0]
        base_prototypes = query_tokens.new_zeros(
            0,
            num_classes,
            temporal_dim,
            feat_dim,
        )
        refined_prototypes = base_prototypes.clone()
        base_weights = query_tokens.new_zeros(
            0,
            num_classes,
            temporal_dim,
            num_points,
        )
        refined_weights = base_weights.clone()

    shift = (refined_weights.float() - base_weights.float()).abs()
    valid_float = valid.to(shift.dtype)
    valid_count = valid_float.sum(dim=(-2, -1)).clamp_min(1.0)
    shift_mean = (shift * valid_float).sum(dim=(-2, -1)) / valid_count
    return {
        "base_prototypes": torch.nan_to_num(base_prototypes, nan=0.0),
        "refined_prototypes": torch.nan_to_num(refined_prototypes, nan=0.0),
        "base_weights": torch.nan_to_num(base_weights, nan=0.0),
        "refined_weights": torch.nan_to_num(refined_weights, nan=0.0),
        "base_similarity": base_similarity,
        "refined_similarity": torch.nan_to_num(refined_similarity, nan=0.0),
        "local_logit_residual": torch.nan_to_num(local_residual, nan=0.0),
        "weight_shift": torch.nan_to_num(shift, nan=0.0),
        "weight_shift_mean": torch.nan_to_num(shift_mean, nan=0.0),
    }


def _classwise_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-class means and counts for ``values`` shaped ``[S,K]``."""
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError(
            "values and mask must share shape [num_support, num_classes]; got "
            f"{tuple(values.shape)} and {tuple(mask.shape)}."
        )
    weights = mask.to(dtype=values.dtype)
    counts = weights.sum(dim=0)
    means = (values * weights).sum(dim=0) / counts.clamp_min(1.0)
    return means, counts


def compute_matchability_from_similarity(
    similarity: torch.Tensor,
    point_mask: torch.Tensor,
    support_mask: torch.Tensor,
    episode_positive_labels: torch.Tensor,
    cfg: Any,
) -> Dict[str, torch.Tensor]:
    """Compute Support-calibrated Query-class matchability.

    Args:
        similarity: Pure text-to-patch cosine with shape ``[B,K,T,N]``.
        point_mask: Valid trajectory mask with shape ``[B,T,N]``.
        support_mask: Boolean sample split with shape ``[B]``.
        episode_positive_labels: Multi-hot episode labels with shape ``[B,K]``.
            Only Support rows are read; Query rows are deliberately ignored.
        cfg: ``QUERY_CLASS_MATCHABILITY`` config node or a compatible stub.

    Returns:
        Evidence, Support calibration statistics, thresholds and Query
        matchability. Matchability has shape ``[Q,K]``.
    """
    if similarity.ndim != 4:
        raise ValueError(
            "similarity must have shape [B,K,T,N]; got "
            f"{tuple(similarity.shape)}."
        )
    batch, num_classes, temporal_dim, num_points = similarity.shape
    expected_point_shape = (batch, temporal_dim, num_points)
    if tuple(point_mask.shape) != expected_point_shape:
        raise ValueError(
            "point_mask must match similarity B,T,N; got "
            f"{tuple(point_mask.shape)}, expected {expected_point_shape}."
        )
    if tuple(support_mask.shape) != (batch,):
        raise ValueError(
            "support_mask must have shape [B]; got "
            f"{tuple(support_mask.shape)}."
        )
    if tuple(episode_positive_labels.shape) != (batch, num_classes):
        raise ValueError(
            "episode_positive_labels must have shape [B,K]; got "
            f"{tuple(episode_positive_labels.shape)}, expected "
            f"{(batch, num_classes)}."
        )

    point_mask = point_mask.to(device=similarity.device).bool()
    support_mask = support_mask.to(device=similarity.device).bool()
    labels = episode_positive_labels.to(device=similarity.device).bool()
    similarity = torch.nan_to_num(
        similarity.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)

    patch_topk = int(_cfg_value(cfg, "TOPK_PATCHES", 8))
    frame_topk = int(_cfg_value(cfg, "TOPK_FRAMES", 3))
    if patch_topk <= 0 or frame_topk <= 0:
        raise ValueError("TOPK_PATCHES and TOPK_FRAMES must be positive.")

    expanded_point_mask = point_mask.unsqueeze(1).expand_as(similarity)
    frame_evidence = masked_topk_mean(
        similarity,
        expanded_point_mask,
        patch_topk,
        dim=-1,
    )  # [B,K,T]

    frame_valid = point_mask.any(dim=-1).unsqueeze(1).expand(
        batch,
        num_classes,
        temporal_dim,
    )
    video_evidence = masked_topk_mean(
        frame_evidence,
        frame_valid,
        frame_topk,
        dim=-1,
    )  # [B,K]

    support_evidence = video_evidence[support_mask]
    support_targets = labels[support_mask]
    query_evidence = video_evidence[~support_mask]
    if support_evidence.shape[0] == 0:
        raise ValueError("At least one labeled Support sample is required.")

    positive_mean, positive_count = _classwise_masked_mean(
        support_evidence,
        support_targets,
    )
    negative_mask = ~support_targets
    negative_mean, negative_count = _classwise_masked_mean(
        support_evidence,
        negative_mask,
    )

    all_negative = support_evidence[negative_mask]
    if all_negative.numel() > 0:
        global_negative = all_negative.mean()
    else:
        global_negative = support_evidence.mean()
    negative_mean = torch.where(
        negative_count > 0,
        negative_mean,
        global_negative.expand_as(negative_mean),
    )
    # An episode class should have a positive Support example. The fallback
    # keeps malformed/debug episodes finite without reading Query labels.
    positive_mean = torch.where(
        positive_count > 0,
        positive_mean,
        negative_mean,
    )

    support_gap = positive_mean - negative_mean
    support_reliable = (
        (positive_count > 0)
        & (negative_count > 0)
        & (support_gap > 0.0)
    )
    beta = float(_cfg_value(cfg, "CALIBRATION_BETA", 0.25))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("CALIBRATION_BETA must be in [0, 1].")
    positive_gap = support_gap.clamp_min(0.0)
    threshold = negative_mean + beta * positive_gap

    if bool(_cfg_value(cfg, "DETACH_SUPPORT_STATS", True)):
        positive_mean = positive_mean.detach()
        negative_mean = negative_mean.detach()
        support_gap = support_gap.detach()
        threshold = threshold.detach()

    temperature = float(_cfg_value(cfg, "TEMPERATURE", 0.10))
    if temperature <= 0.0:
        raise ValueError("QUERY_CLASS_MATCHABILITY.TEMPERATURE must be positive.")
    matchability = torch.sigmoid(
        (query_evidence - threshold.unsqueeze(0)) / temperature
    )
    matchability = torch.nan_to_num(
        matchability,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    return {
        "frame_evidence": frame_evidence,
        "video_evidence": video_evidence,
        "query_evidence": query_evidence,
        "support_positive_evidence_mean": positive_mean,
        "support_negative_evidence_mean": negative_mean,
        "support_positive_count": positive_count,
        "support_negative_count": negative_count,
        "support_gap": support_gap,
        "support_reliable": support_reliable,
        "threshold": threshold,
        "matchability": matchability,
    }


def _build_frame_softmax_q2s_with_matchability(
    self: Any,
    value_tokens: torch.Tensor,
    metadata: Dict[str, Any],
    pred_tracks: Optional[torch.Tensor] = None,
    matchability_evidence_tokens: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build normal routed prototypes and add Query-class matchability."""
    few_shot_cfg = getattr(getattr(self, "cfg", None), "FEW_SHOT", None)
    cfg = getattr(few_shot_cfg, "QUERY_CLASS_MATCHABILITY", None)
    if not bool(_cfg_value(cfg, "ENABLE", False)):
        original = getattr(self.__class__, _PATCH_MARKER)
        return original(self, value_tokens, metadata, pred_tracks=pred_tracks)
    if bool(getattr(self, "use_query_null_route", False)):
        raise RuntimeError(
            "QUERY_CLASS_MATCHABILITY and QUERY_NULL_ROUTE are mutually "
            "exclusive. Disable the learned Query Null token."
        )

    support_mask = metadata["support_mask"].to(device=value_tokens.device).bool()
    query_mask = ~support_mask
    if not query_mask.any():
        return None

    point_mask = (
        metadata["pred_query_mask"]
        if self.cfg.POINT_INFO.USE_PT_QUERY_MASK
        else metadata["pred_visibility"]
    ).to(device=value_tokens.device).bool()
    # Only the new temporal router consumes this mask. Spatial routing and
    # all quality/specificity masks and formulas below remain unchanged.
    temporal_point_mask = self._get_temporal_similarity_mask(point_mask, metadata)
    episode_positive_labels = metadata["episode_positive_labels"].to(
        device=value_tokens.device,
    ).bool()
    episode_class_ids = metadata["episode_class_ids"].to(
        device=value_tokens.device,
    ).long()
    episode_class_ids = (
        episode_class_ids[0]
        if episode_class_ids.ndim == 2
        else episode_class_ids
    ).flatten()
    if episode_class_ids.numel() == 0:
        return None

    # Pure text is retained for the independent ``whether`` branch.
    episode_label_text = self._get_pot_label_text_features(
        episode_class_ids,
        value_tokens.dtype,
    )

    evidence_source = str(_cfg_value(cfg, "EVIDENCE_SOURCE", "post")).lower()
    if evidence_source == "post":
        evidence_tokens = value_tokens
    elif evidence_source == "raw":
        if matchability_evidence_tokens is None:
            raise ValueError(
                "Raw matchability evidence was requested, but pre-Pointformer "
                "DinoTxt tokens were not provided."
            )
        evidence_tokens = matchability_evidence_tokens
    else:
        raise ValueError(
            "QUERY_CLASS_MATCHABILITY.EVIDENCE_SOURCE must be 'post' or 'raw'; "
            f"got {evidence_source!r}."
        )
    if tuple(evidence_tokens.shape[:3]) != tuple(value_tokens.shape[:3]):
        raise ValueError(
            "Matchability evidence tokens must align with value token B/T/N; got "
            f"{tuple(evidence_tokens.shape)} and {tuple(value_tokens.shape)}."
        )

    refined_similarity = None
    if bool(getattr(self, "use_cat_cost_aggregation", False)):
        refined_similarity = self._compute_split_cat_refined_point_similarity(
            value_tokens,
            point_mask,
            pred_tracks,
            support_mask,
            episode_positive_labels,
            episode_class_ids,
            episode_label_text,
            raw_positive_labels=metadata.get("raw_positive_labels"),
        )

    support_prototypes = self._build_frame_softmax_support_prototypes(
        value_tokens,
        point_mask,
        support_mask,
        episode_positive_labels,
        episode_label_text,
        precomputed_similarity=refined_similarity,
        temporal_point_mask=temporal_point_mask,
    )

    # The existing text+Support feature remains responsible for ``where``.
    query_label_features = episode_label_text
    support_visual = None
    support_visual_valid = None
    if bool(getattr(self, "use_support_text_fusion", False)):
        if refined_similarity is not None:
            raise RuntimeError(
                "SUPPORT_TEXT_FUSION cannot consume CAT precomputed query costs."
            )
        (
            query_label_features,
            support_visual,
            support_visual_valid,
        ) = self._fuse_episode_text_with_support_visual(
            episode_label_text,
            support_prototypes,
        )

    matchability_mode = str(
        _cfg_value(cfg, "MODE", "threshold")
    ).lower()
    relative_modes = {
        "positive_confuser_margin",
        "positive_confuser",
        "confuser_margin",
    }
    legacy_modes = {"threshold", "legacy", "support_threshold"}
    if (
        matchability_mode not in relative_modes
        and matchability_mode not in legacy_modes
    ):
        raise ValueError(
            "QUERY_CLASS_MATCHABILITY.MODE must be 'threshold' or "
            f"'positive_confuser_margin'; got {matchability_mode!r}."
        )

    query_indices = torch.nonzero(query_mask, as_tuple=False).flatten()
    query_tokens = value_tokens.index_select(0, query_indices)
    query_point_mask = point_mask.index_select(0, query_indices)
    local_refinement = None
    local_margin = None
    local_positive_similarity = None
    local_confuser_similarity = None
    local_positive_counts = None
    local_confuser_counts = None
    routed_support_references = None
    routed_support_frame_valid = None
    routed_positive_valid = None
    routed_confuser_valid = None
    routed_support_indices = None

    local_refinement_enable = bool(
        _cfg_value(cfg, "LOCAL_REFINEMENT_ENABLE", False)
    )
    evidence_verification_enable = bool(
        _cfg_value(cfg, "EVIDENCE_VERIFICATION_ENABLE", False)
    )
    absolute_mass_enable = bool(
        _cfg_value(cfg, "ABSOLUTE_MASS_ENABLE", False)
    )
    absolute_mass_source = str(
        _cfg_value(cfg, "ABSOLUTE_MASS_SOURCE", "raw")
    ).lower()
    if absolute_mass_enable and absolute_mass_source not in {"raw", "post"}:
        raise ValueError(
            "ABSOLUTE_MASS_SOURCE must be 'raw' or 'post'; got "
            f"{absolute_mass_source!r}."
        )
    if (
        absolute_mass_enable
        and absolute_mass_source == "raw"
        and matchability_evidence_tokens is None
    ):
        raise ValueError(
            "Raw absolute-mass evidence was enabled, but pre-Pointformer "
            "DinoTxt tokens were not provided."
        )
    if local_refinement_enable and evidence_verification_enable:
        raise ValueError(
            "LOCAL_REFINEMENT_ENABLE and EVIDENCE_VERIFICATION_ENABLE are "
            "controlled alternatives and cannot be enabled together."
        )
    if local_refinement_enable or evidence_verification_enable:
        if matchability_mode not in relative_modes:
            raise ValueError(
                "Local Positive/Confuser evidence requires "
                "MODE='positive_confuser_margin'."
            )
        if local_refinement_enable and evidence_source != "post":
            raise ValueError(
                "LOCAL_REFINEMENT_ENABLE currently requires EVIDENCE_SOURCE='post'."
            )
        if refined_similarity is not None:
            raise RuntimeError(
                "Local Positive/Confuser evidence currently requires "
                "COST_AGG.ENABLE=False."
            )
        support_frame_mask = metadata.get("pred_visibility", point_mask).to(
            device=value_tokens.device,
        ).bool()
        (
            routed_support_references,
            routed_support_frame_valid,
            routed_positive_valid,
            routed_confuser_valid,
            routed_support_indices,
        ) = build_class_local_support_references(
            self,
            value_tokens,
            point_mask,
            support_frame_mask,
            support_mask,
            episode_positive_labels,
            query_label_features,
            temporal_point_mask=temporal_point_mask,
        )
        (
            local_margin,
            local_positive_similarity,
            local_confuser_similarity,
            local_positive_counts,
            local_confuser_counts,
        ) = compute_local_positive_confuser_margin(
            query_tokens,
            routed_support_references,
            routed_support_frame_valid,
            routed_positive_valid,
            routed_confuser_valid,
            query_point_mask,
            positive_aggregation=str(
                _cfg_value(
                    cfg,
                    "LOCAL_POSITIVE_AGGREGATION"
                    if local_refinement_enable
                    else "EVIDENCE_POSITIVE_AGGREGATION",
                    "topk_mean",
                )
            ),
            positive_topk=int(_cfg_value(
                cfg,
                "LOCAL_POSITIVE_TOPK"
                if local_refinement_enable
                else "EVIDENCE_POSITIVE_TOPK",
                2,
            )),
            negative_aggregation=str(
                _cfg_value(cfg, "LOCAL_NEGATIVE_AGGREGATION", "topk_mean")
                if local_refinement_enable
                else _cfg_value(
                    cfg,
                    "EVIDENCE_CONFUSER_AGGREGATION",
                    "topk_mean",
                )
            ),
            negative_topk=int(_cfg_value(
                cfg,
                "LOCAL_NEGATIVE_TOPK"
                if local_refinement_enable
                else "EVIDENCE_CONFUSER_TOPK",
                2,
            )),
            positive_temperature=float(
                _cfg_value(
                    cfg,
                    "LOCAL_MARGIN_TEMPERATURE"
                    if local_refinement_enable
                    else "EVIDENCE_AGGREGATION_TEMPERATURE",
                    0.10,
                )
            ),
            negative_temperature=float(
                _cfg_value(
                    cfg,
                    "LOCAL_NEGATIVE_TEMPERATURE"
                    if local_refinement_enable
                    else "EVIDENCE_AGGREGATION_TEMPERATURE",
                    0.10,
                )
            ),
            detach_references=bool(
                _cfg_value(
                    cfg,
                    "LOCAL_DETACH_REFERENCES"
                    if local_refinement_enable
                    else "EVIDENCE_DETACH_REFERENCES",
                    True,
                )
            ),
        )

    if local_refinement_enable:
        patch_tau = float(
            getattr(self.pot_route_cfg, "FRAME_SOFTMAX_TAU", 0.07)
        )
        local_refinement = build_confuser_refined_query_prototypes(
            self,
            query_tokens,
            query_point_mask,
            query_label_features,
            local_margin,
            patch_tau=patch_tau,
            local_tau=float(
                _cfg_value(cfg, "LOCAL_MARGIN_TEMPERATURE", 0.10)
            ),
            local_strength=float(
                _cfg_value(cfg, "LOCAL_LOGIT_STRENGTH", 0.50)
            ),
        )
        # The refined prototypes are the Query values used by the base q2s
        # score; global matchability is retained only as a diagnostic.
        query_prototypes = local_refinement["refined_prototypes"]
        base_query_prototypes = local_refinement["base_prototypes"]
        query_patch_weights = local_refinement["refined_weights"]
    else:
        query_prototypes = []
        query_patch_weights = []
        for sample_idx in query_indices.tolist():
            if refined_similarity is None:
                sample_prototypes, sample_patch_weights = (
                    self._compute_frame_softmax_text_prototypes(
                        value_tokens[sample_idx],
                        point_mask[sample_idx],
                        query_label_features,
                        **({} if temporal_point_mask is None else {
                            "temporal_point_mask": temporal_point_mask[sample_idx],
                        }),
                    )
                )
            else:
                sample_prototypes, sample_patch_weights = (
                    self._compute_frame_softmax_prototypes_from_similarity(
                        value_tokens[sample_idx],
                        point_mask[sample_idx],
                        refined_similarity[sample_idx],
                    )
                )
            query_prototypes.append(sample_prototypes.unsqueeze(0))
            query_patch_weights.append(sample_patch_weights.unsqueeze(0))
        if not query_prototypes:
            return None
        query_prototypes = torch.cat(query_prototypes, dim=0)
        query_patch_weights = torch.cat(query_patch_weights, dim=0)
        base_query_prototypes = query_prototypes

    diag_similarity = self._compute_bidirectional_frame_similarity(
        query_prototypes,
        support_prototypes,
    )
    alpha = float(getattr(
        self.pot_route_cfg,
        "QUERY_PARTIAL_LOGIT_ALPHA",
        10.0,
    ))
    bias = float(getattr(
        self.pot_route_cfg,
        "QUERY_PARTIAL_LOGIT_BIAS",
        -2.0,
    ))
    base_logits = alpha * diag_similarity + bias
    if local_refinement is None:
        pre_refinement_diag_similarity = diag_similarity
        pre_refinement_logits = base_logits
    else:
        pre_refinement_diag_similarity = (
            self._compute_bidirectional_frame_similarity(
                base_query_prototypes,
                support_prototypes,
            )
        )
        pre_refinement_logits = alpha * pre_refinement_diag_similarity + bias

    temporal_logits = base_logits.float()
    hypothesis_region_aux = None
    frame_matchability_aux = None
    temporal_match_aux = None
    absolute_mass_aux = None
    frame_hypothesis_states = None
    if absolute_mass_enable:
        absolute_tokens_all = (
            matchability_evidence_tokens
            if absolute_mass_source == "raw"
            else value_tokens
        )
        if tuple(absolute_tokens_all.shape[:3]) != tuple(value_tokens.shape[:3]):
            raise ValueError(
                "Absolute-mass tokens must align with post values in B/T/N; got "
                f"{tuple(absolute_tokens_all.shape)} and "
                f"{tuple(value_tokens.shape)}."
            )
        absolute_point_mask = point_mask
        if bool(_cfg_value(cfg, "ABSOLUTE_MASS_USE_VISIBILITY", True)):
            absolute_visibility = metadata.get(
                "pred_visibility",
                point_mask,
            ).to(device=value_tokens.device).bool()
            absolute_point_mask = absolute_point_mask & absolute_visibility
        absolute_similarity = self._compute_batched_point_text_similarity(
            absolute_tokens_all,
            episode_label_text,
        )
        absolute_mass_aux = compute_support_calibrated_frame_transport_mass(
            absolute_similarity,
            absolute_point_mask,
            support_mask,
            episode_positive_labels,
            cfg,
        )

    if evidence_verification_enable:
        evidence_point_mask = query_point_mask
        if bool(_cfg_value(cfg, "EVIDENCE_USE_VISIBILITY", True)):
            query_visibility = metadata.get(
                "pred_visibility",
                point_mask,
            ).to(device=value_tokens.device).bool().index_select(
                0,
                query_indices,
            )
            evidence_point_mask = evidence_point_mask & query_visibility
        # Reuse the exact class-conditioned distribution pi that constructed
        # Q_k. Specificity only reapplies its evidence-validity mask; it does
        # not compute another text-to-patch similarity map.
        hypothesis_weights = normalize_patch_region_weights(
            query_patch_weights,
            evidence_point_mask,
        )
        hypothesis_frame_valid = evidence_point_mask.any(dim=-1)
        hypothesis_effective_patches = (
            1.0
            / hypothesis_weights.square().sum(dim=-1).clamp_min(1e-12)
        )
        hypothesis_top1_mass = hypothesis_weights.max(dim=-1).values
        hypothesis_effective_patches = torch.where(
            hypothesis_frame_valid.unsqueeze(1),
            hypothesis_effective_patches,
            torch.zeros_like(hypothesis_effective_patches),
        )
        hypothesis_top1_mass = torch.where(
            hypothesis_frame_valid.unsqueeze(1),
            hypothesis_top1_mass,
            torch.zeros_like(hypothesis_top1_mass),
        )
        hypothesis_region_aux = {
            "weights": hypothesis_weights,
            "frame_valid": hypothesis_frame_valid,
            "effective_patches": hypothesis_effective_patches,
            "top1_mass": hypothesis_top1_mass,
        }
        frame_matchability_aux = (
            compute_evidence_conditioned_frame_matchability(
                hypothesis_region_aux["weights"],
                local_positive_similarity,
                local_confuser_similarity,
                hypothesis_region_aux["frame_valid"],
                local_positive_counts,
                local_confuser_counts,
                temperature=float(
                    _cfg_value(cfg, "FRAME_MARGIN_TEMPERATURE", 0.10)
                ),
                bias=float(_cfg_value(cfg, "FRAME_MARGIN_BIAS", 0.0)),
            )
        )
    if evidence_verification_enable or absolute_mass_enable:
        positive_frame_similarity = classwise_frame_similarity(
            query_prototypes,
            support_prototypes,
        )
        if frame_matchability_aux is None:
            frame_matchability = torch.ones(
                positive_frame_similarity.shape[:3],
                device=value_tokens.device,
                dtype=torch.float32,
            )
        else:
            frame_matchability = frame_matchability_aux["matchability"]
        absolute_patch_mass = (
            absolute_mass_aux["patch_mass"]
            if absolute_mass_aux is not None
            else torch.ones_like(frame_matchability)
        )
        # This is now the only frame-level verification path.  With one
        # factor disabled its fallback is exactly one, giving quality-only
        # g=m or specificity-only g=rho without a separate implementation.
        frame_hypothesis_states = factorize_frame_hypothesis_states(
            absolute_patch_mass,
            frame_matchability,
        )
        temporal_match_aux = hypothesis_mass_bimhm_logits(
            positive_frame_similarity,
            frame_hypothesis_states["target_mass"],
            base_logits,
            alpha=alpha,
            transport_strength=float(
                _cfg_value(cfg, "ABSOLUTE_MASS_TRANSPORT_STRENGTH", 1.0)
            ),
            unmatched_cost=float(
                _cfg_value(cfg, "ABSOLUTE_MASS_UNMATCHED_COST", 0.0)
            ),
            one_sided_transport=bool(
                _cfg_value(cfg, "ABSOLUTE_MASS_ONE_SIDED", True)
            ),
        )
        temporal_logits = temporal_match_aux["logits"]

    if matchability_mode in relative_modes:
        if evidence_source != "post":
            raise ValueError(
                "positive_confuser_margin currently compares Post-Pointformer "
                "prototypes; set EVIDENCE_SOURCE='post'."
            )
        # The current CAT path intentionally computes Support costs only for
        # known positive labels.  Reusing those zero-filled non-positive slots
        # as confusers would make the two sides incomparable, so fail loudly
        # until a CAT-specific negative route is defined.
        if refined_similarity is not None:
            raise RuntimeError(
                "positive_confuser_margin currently requires COST_AGG.ENABLE=False."
            )
        if routed_support_references is None or not local_refinement_enable:
            confuser_prototypes, confuser_valid, confuser_support_indices = (
                build_class_confuser_prototypes(
                    self,
                    value_tokens,
                    point_mask,
                    support_mask,
                    episode_positive_labels,
                    query_label_features,
                    detach_support=bool(
                        _cfg_value(cfg, "DETACH_CONFUSER_SUPPORT", False)
                    ),
                    temporal_point_mask=temporal_point_mask,
                )
            )
        else:
            # The local and global comparisons must use the same class-routed
            # Support hypotheses.  Reusing them also avoids a second complete
            # Support routing pass when local refinement is enabled.
            confuser_prototypes = routed_support_references
            if bool(_cfg_value(cfg, "DETACH_CONFUSER_SUPPORT", False)):
                confuser_prototypes = confuser_prototypes.detach()
            confuser_valid = routed_confuser_valid
            confuser_support_indices = routed_support_indices
        negative_similarity = pairwise_bimhm(
            query_prototypes,
            confuser_prototypes,
        )
        margin_temperature = float(
            _cfg_value(cfg, "MARGIN_TEMPERATURE", 0.10)
        )
        margin_bias = float(_cfg_value(cfg, "MARGIN_BIAS", 0.0))
        negative_aggregation = str(
            _cfg_value(cfg, "NEGATIVE_AGGREGATION", "max")
        )
        negative_topk = int(_cfg_value(cfg, "NEGATIVE_TOPK", 2))
        negative_temperature = float(
            _cfg_value(cfg, "NEGATIVE_TEMPERATURE", margin_temperature)
        )
        (
            matchability,
            relative_margin,
            negative_aggregate,
            hardest_confuser_local,
            confuser_valid_count,
        ) = _compute_relative_matchability_with_diagnostics(
            diag_similarity,
            negative_similarity,
            confuser_valid.transpose(0, 1),
            temperature=margin_temperature,
            bias=margin_bias,
            aggregation=negative_aggregation,
            topk=negative_topk,
            negative_temperature=negative_temperature,
        )

        if confuser_support_indices.numel() > 0:
            safe_local = hardest_confuser_local.clamp_min(0)
            hardest_confuser_global = confuser_support_indices.index_select(
                0,
                safe_local.reshape(-1),
            ).reshape_as(hardest_confuser_local)
            hardest_confuser_global = torch.where(
                hardest_confuser_local.ge(0),
                hardest_confuser_global,
                torch.full_like(hardest_confuser_global, -1),
            )
        else:
            hardest_confuser_global = torch.full(
                hardest_confuser_local.shape,
                -1,
                device=value_tokens.device,
                dtype=torch.long,
            )

        # Keep the generic evidence/threshold names populated so existing
        # consumers can still read a matchability result.  In this mode the
        # evidence is the relative margin and the threshold is the configured
        # margin bias; the richer diagnostics below expose both raw terms.
        matchability_aux = {
            "matchability": matchability,
            "query_evidence": relative_margin,
            "threshold": torch.full(
                (episode_class_ids.numel(),),
                fill_value=margin_bias,
                device=value_tokens.device,
                dtype=torch.float32,
            ),
            "support_reliable": confuser_valid_count > 0,
            "relative_margin": relative_margin,
            "positive_similarity": diag_similarity,
            "negative_similarity": negative_aggregate,
            "hardest_confuser_index": hardest_confuser_global,
            "confuser_valid_count": confuser_valid_count,
            "negative_similarity_all": negative_similarity,
            "confuser_support_indices": confuser_support_indices,
        }
    else:
        pure_text_similarity = self._compute_batched_point_text_similarity(
            evidence_tokens,
            episode_label_text,
        )
        matchability_aux = compute_matchability_from_similarity(
            pure_text_similarity,
            point_mask,
            support_mask,
            episode_positive_labels,
            cfg,
        )
        matchability_aux.update({
            "relative_margin": matchability_aux["query_evidence"],
            "positive_similarity": diag_similarity,
            "negative_similarity": torch.zeros_like(diag_similarity),
            "hardest_confuser_index": torch.full(
                diag_similarity.shape,
                -1,
                device=value_tokens.device,
                dtype=torch.long,
            ),
            "confuser_valid_count": matchability_aux["support_negative_count"],
        })
    # The unified three-state target mass is the only frame/class correction.
    # Global Positive/Confuser statistics remain available for diagnostics, but
    # are deliberately not applied as a second post-hoc log penalty.
    final_logits = temporal_logits.float()

    if absolute_mass_aux is None:
        query_transport_mass_summary = matchability_aux["matchability"]
    else:
        transport_frame_valid = absolute_mass_aux["query_frame_valid"].bool()
        transport_valid = transport_frame_valid.unsqueeze(1).expand_as(
            temporal_match_aux["effective_patch_mass"]
        )
        transport_valid_float = transport_valid.float()
        transport_valid_count = transport_valid_float.sum(dim=-1)
        query_transport_mass_summary = (
            temporal_match_aux["effective_patch_mass"] * transport_valid_float
        ).sum(dim=-1) / transport_valid_count.clamp_min(1.0)
        query_transport_mass_summary = torch.where(
            transport_valid_count > 0,
            query_transport_mass_summary,
            torch.zeros_like(query_transport_mass_summary),
        )

    target_label_indices = torch.arange(
        episode_class_ids.numel(),
        device=value_tokens.device,
        dtype=torch.long,
    )
    result: Dict[str, torch.Tensor] = {
        "query_partial_q2s_logits": final_logits.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_q2s_base_logits": base_logits.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_q2s_temporal_logits": temporal_logits.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_q2s_pre_refinement_logits": pre_refinement_logits.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_query_prototypes": query_prototypes.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_query_prototypes_before_refinement": (
            base_query_prototypes.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            )
        ),
        "query_partial_support_prototypes": support_prototypes.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_diag_similarity": diag_similarity.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_diag_similarity_before_refinement": (
            pre_refinement_diag_similarity.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            )
        ),
        "query_partial_alpha_sim_term": (alpha * diag_similarity).to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_bias_term": torch.full_like(
            base_logits,
            fill_value=bias,
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_query_sample_indices": query_indices.to(
            device=value_tokens.device,
            dtype=torch.long,
        ),
        "query_partial_label_axis_global_labels": episode_class_ids.to(
            device=value_tokens.device,
            dtype=torch.long,
        ),
        "query_partial_target_label_indices": target_label_indices,
        "query_class_matchability": matchability_aux["matchability"],
        "query_class_transport_mass": query_transport_mass_summary,
        "query_class_evidence": matchability_aux["query_evidence"],
        "query_class_threshold": matchability_aux["threshold"],
        "query_class_positive_similarity": matchability_aux[
            "positive_similarity"
        ],
        "query_class_hardest_confuser_similarity": matchability_aux[
            "negative_similarity"
        ],
        "query_class_relative_margin": matchability_aux["relative_margin"],
        "query_class_hardest_confuser_support_index": matchability_aux[
            "hardest_confuser_index"
        ],
        "query_class_confuser_valid_count": matchability_aux[
            "confuser_valid_count"
        ],
    }
    if frame_hypothesis_states is not None:
        result.update({
            "query_frame_target_mass": frame_hypothesis_states[
                "target_mass"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_confuser_mass": frame_hypothesis_states[
                "confuser_mass"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_null_mass": frame_hypothesis_states[
                "null_mass"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
        })
    if absolute_mass_aux is not None:
        effective_patch_mass = temporal_match_aux["effective_patch_mass"]
        unmatched_mass = temporal_match_aux["unmatched_mass"]
        transported_patch_weights = (
            query_patch_weights.float() * effective_patch_mass.unsqueeze(-1)
        )
        transported_query_prototypes = (
            query_prototypes.float() * effective_patch_mass.unsqueeze(-1)
        )
        transport_valid = absolute_mass_aux["query_frame_valid"].bool().unsqueeze(1)
        transport_valid = transport_valid.expand_as(effective_patch_mass)
        transport_valid_float = transport_valid.float()
        transport_valid_count = transport_valid_float.sum(dim=-1)

        def _transport_frame_mean(value):
            mean = (value.float() * transport_valid_float).sum(
                dim=-1
            ) / transport_valid_count.clamp_min(1.0)
            return torch.where(
                transport_valid_count > 0,
                mean,
                torch.zeros_like(mean),
            )

        result.update({
            "query_patch_conditional_weights": query_patch_weights.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_patch_transport_weights": transported_patch_weights.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_partial_query_transported_prototypes": (
                transported_query_prototypes.to(
                    device=value_tokens.device,
                    dtype=value_tokens.dtype,
                )
            ),
            "query_frame_absolute_evidence": absolute_mass_aux[
                "query_frame_evidence"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_absolute_mass_raw": absolute_mass_aux[
                "raw_patch_mass"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_absolute_patch_mass": absolute_mass_aux[
                "patch_mass"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_patch_mass": effective_patch_mass.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_frame_unmatched_mass": unmatched_mass.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_frame_transport_valid": transport_valid,
            "query_frame_absolute_evidence_mean": _transport_frame_mean(
                absolute_mass_aux["query_frame_evidence"]
            ),
            "query_frame_patch_mass_mean": _transport_frame_mean(
                effective_patch_mass
            ),
            "query_frame_unmatched_mass_mean": _transport_frame_mean(
                unmatched_mass
            ),
            "query_frame_transport_logit_delta": temporal_match_aux[
                "transport_delta"
            ].to(device=value_tokens.device, dtype=torch.float32),
            "query_frame_transported_similarity": temporal_match_aux[
                "transported_similarity"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_absolute_mass_threshold": absolute_mass_aux["threshold"],
            "support_absolute_positive_evidence_mean": absolute_mass_aux[
                "support_positive_evidence_mean"
            ],
            "support_absolute_negative_evidence_mean": absolute_mass_aux[
                "support_negative_evidence_mean"
            ],
            "support_absolute_evidence_gap": absolute_mass_aux["support_gap"],
            "support_absolute_calibration_reliable": absolute_mass_aux[
                "support_reliable"
            ],
            "support_absolute_positive_count": absolute_mass_aux[
                "support_positive_count"
            ],
            "support_absolute_negative_count": absolute_mass_aux[
                "support_negative_count"
            ],
        })
    if hypothesis_region_aux is not None:
        frame_valid = frame_matchability_aux["valid"].to(
            device=value_tokens.device,
        ).bool()
        frame_valid_float = frame_valid.float()
        frame_valid_count = frame_valid_float.sum(dim=-1)

        def _evidence_frame_mean(value, fallback=0.0):
            mean = (
                value.float() * frame_valid_float
            ).sum(dim=-1) / frame_valid_count.clamp_min(1.0)
            return torch.where(
                frame_valid_count.gt(0),
                mean,
                torch.full_like(mean, float(fallback)),
            )

        evidence_effective = hypothesis_region_aux[
            "effective_patches"
        ].expand_as(frame_matchability_aux["matchability"])
        evidence_top1 = hypothesis_region_aux["top1_mass"].expand_as(
            frame_matchability_aux["matchability"]
        )
        result.update({
            "query_evidence_patch_weights": hypothesis_region_aux["weights"].to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_evidence_patch_positive_response": (
                local_positive_similarity.to(
                    device=value_tokens.device,
                    dtype=value_tokens.dtype,
                )
            ),
            "query_evidence_patch_confuser_response": (
                local_confuser_similarity.to(
                    device=value_tokens.device,
                    dtype=value_tokens.dtype,
                )
            ),
            "query_frame_positive_evidence": frame_matchability_aux[
                "positive_evidence"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_confuser_evidence": frame_matchability_aux[
                "confuser_evidence"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_relative_margin": frame_matchability_aux[
                "margin"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_matchability": frame_matchability_aux[
                "matchability"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_frame_valid": frame_valid,
            "query_frame_confuser_available": frame_matchability_aux[
                "class_valid"
            ].to(device=value_tokens.device, dtype=torch.bool),
            "query_frame_positive_evidence_mean": _evidence_frame_mean(
                frame_matchability_aux["positive_evidence"]
            ),
            "query_frame_confuser_evidence_mean": _evidence_frame_mean(
                frame_matchability_aux["confuser_evidence"]
            ),
            "query_frame_relative_margin_mean": _evidence_frame_mean(
                frame_matchability_aux["margin"]
            ),
            "query_frame_matchability_mean": _evidence_frame_mean(
                frame_matchability_aux["matchability"],
                fallback=1.0,
            ),
            "query_evidence_effective_patches_mean": _evidence_frame_mean(
                evidence_effective
            ),
            "query_evidence_top1_mass_mean": _evidence_frame_mean(
                evidence_top1
            ),
            "query_frame_s_to_q_delta": temporal_match_aux[
                "s_to_q_delta"
            ].to(device=value_tokens.device, dtype=torch.float32),
            "query_frame_temporal_logit_delta": (
                temporal_logits.float() - base_logits.float()
            ),
            "query_frame_max_t_switch_fraction": temporal_match_aux[
                "winner_switch_fraction"
            ].to(device=value_tokens.device, dtype=torch.float32),
            "query_frame_base_winner": temporal_match_aux[
                "base_winner"
            ].to(device=value_tokens.device, dtype=torch.long),
            "query_frame_verified_winner": temporal_match_aux[
                "verified_winner"
            ].to(device=value_tokens.device, dtype=torch.long),
            "query_evidence_positive_support_count": local_positive_counts.to(
                device=value_tokens.device,
                dtype=torch.long,
            ),
            "query_evidence_confuser_support_count": local_confuser_counts.to(
                device=value_tokens.device,
                dtype=torch.long,
            ),
        })
    if local_refinement is not None:
        local_valid = query_point_mask.to(
            device=value_tokens.device,
            dtype=torch.float32,
        ).unsqueeze(1)
        local_valid_count = local_valid.sum(dim=(-2, -1)).clamp_min(1.0)

        def _local_pair_mean(value):
            return (
                value.float() * local_valid
            ).sum(dim=(-2, -1)) / local_valid_count

        result.update({
            "query_local_positive_similarity": local_positive_similarity.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_local_confuser_similarity": local_confuser_similarity.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_local_relative_margin": local_margin.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_local_positive_similarity_mean": _local_pair_mean(
                local_positive_similarity
            ),
            "query_local_confuser_similarity_mean": _local_pair_mean(
                local_confuser_similarity
            ),
            "query_local_relative_margin_mean": _local_pair_mean(local_margin),
            "query_patch_weights_before_refinement": local_refinement[
                "base_weights"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_patch_weights_after_refinement": local_refinement[
                "refined_weights"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_local_logit_residual": local_refinement[
                "local_logit_residual"
            ].to(device=value_tokens.device, dtype=value_tokens.dtype),
            "query_local_weight_shift": local_refinement["weight_shift"].to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_local_weight_shift_mean": local_refinement[
                "weight_shift_mean"
            ].to(device=value_tokens.device, dtype=torch.float32),
            "query_local_positive_support_count": local_positive_counts.to(
                device=value_tokens.device,
                dtype=torch.long,
            ),
            "query_local_confuser_support_count": local_confuser_counts.to(
                device=value_tokens.device,
                dtype=torch.long,
            ),
            "query_local_reference_support_indices": routed_support_indices.to(
                device=value_tokens.device,
                dtype=torch.long,
            ),
        })
    if "support_positive_evidence_mean" in matchability_aux:
        result.update({
            "support_positive_evidence_mean": matchability_aux[
                "support_positive_evidence_mean"
            ],
            "support_negative_evidence_mean": matchability_aux[
                "support_negative_evidence_mean"
            ],
            "support_positive_evidence_count": matchability_aux[
                "support_positive_count"
            ],
            "support_negative_evidence_count": matchability_aux[
                "support_negative_count"
            ],
            "support_evidence_gap": matchability_aux["support_gap"],
            "support_calibration_reliable": matchability_aux[
                "support_reliable"
            ],
        })
    else:
        result["query_class_confuser_available"] = matchability_aux[
            "support_reliable"
        ]
    if "negative_similarity_all" in matchability_aux:
        # This tensor is useful for offline diagnostics, but is intentionally
        # not consumed by the q2s loss.  Keep it out of CSV rows to avoid
        # materializing one column per Support.
        result["query_class_confuser_similarities"] = matchability_aux[
            "negative_similarity_all"
        ]
        result["query_class_confuser_support_indices"] = matchability_aux[
            "confuser_support_indices"
        ]
    if support_visual is not None:
        result.update({
            "support_text_fusion_query_features": query_label_features.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "support_text_fusion_visual_prototypes": support_visual.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "support_text_fusion_valid_classes": support_visual_valid.to(
                device=value_tokens.device,
                dtype=torch.bool,
            ),
        })
    return result


def install_query_class_matchability(pointformer_cls: Any) -> None:
    """Install the extension once without editing the large Pointformer file."""
    if hasattr(pointformer_cls, _PATCH_MARKER):
        return
    setattr(
        pointformer_cls,
        _PATCH_MARKER,
        pointformer_cls._build_frame_softmax_q2s_aux,
    )
    pointformer_cls._build_frame_softmax_q2s_aux = (
        _build_frame_softmax_q2s_with_matchability
    )
