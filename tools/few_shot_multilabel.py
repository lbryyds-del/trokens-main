"""Utilities for SAV multi-label few-shot episodes."""

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import trokens.utils.distributed as du


def q2s_cos_sim_fp32(x, y, epsilon=0.01):
    """Compute q2s cosine in FP32 even inside an AMP autocast region."""
    device_type = x.device.type
    if hasattr(torch, "autocast") and device_type in {"cpu", "cuda"}:
        autocast_disabled = torch.autocast(device_type=device_type, enabled=False)
    elif x.is_cuda:
        autocast_disabled = torch.cuda.amp.autocast(enabled=False)
    else:
        autocast_disabled = nullcontext()

    with autocast_disabled:
        x_float = x.float()
        y_float = y.float()
        numerator = torch.matmul(x_float, y_float.transpose(-1, -2))
        xnorm = torch.norm(x_float, dim=-1).unsqueeze(-1)
        ynorm = torch.norm(y_float, dim=-1).unsqueeze(-1)
        denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + float(epsilon)
        similarity = torch.div(numerator, denominator)

    if not torch.isfinite(similarity).all():
        raise FloatingPointError(
            "Non-finite q2s cosine similarity after the FP32 stability path."
        )
    return similarity


def is_multilabel_episode(cfg, labels, metadata):
    """Return True for multi-label episode batches."""
    episode_class_ids = metadata.get("episode_class_ids")
    return (
        cfg.DATA.MULTI_LABEL
        and isinstance(labels, torch.Tensor)
        and labels.ndim == 2
        and isinstance(episode_class_ids, torch.Tensor)
    )


def get_episode_class_ids(metadata, device):
    """Return the current episode's global class ids."""
    episode_class_ids = metadata["episode_class_ids"].to(device).long()
    if episode_class_ids.ndim == 2:
        episode_class_ids = episode_class_ids[0]
    return episode_class_ids


def episode_labels_from_global(labels, episode_class_ids):
    """Filter global multi-hot labels to the current episode classes."""
    return labels.float().index_select(1, episode_class_ids)


def multilabel_classification_loss(preds, labels, cfg, loss_fun):
    """Auxiliary BCE over seen classes to avoid negative-only novel supervision."""
    labels = labels.float()
    if cfg.TEST.SEEN_LABELS:
        class_ids = torch.as_tensor(
            cfg.TEST.SEEN_LABELS, device=preds.device, dtype=torch.long
        )
        preds = preds.index_select(1, class_ids)
        labels = labels.index_select(1, class_ids)
    return loss_fun(preds, labels)


def support_query_split_multilabel(preds, labels, metadata):
    """Split multi-label episode tokens and build one prototype per episode class."""
    device = preds.device
    sample_info = np.array(metadata["sample_type"])
    support_condition = torch.as_tensor(
        sample_info == "support", device=device, dtype=torch.bool
    )
    query_condition = torch.logical_not(support_condition)

    episode_class_ids = get_episode_class_ids(metadata, device)
    batch_labels = episode_labels_from_global(labels.to(device), episode_class_ids)
    support_preds = preds[support_condition]
    support_batch_labels = batch_labels[support_condition]
    query_preds = preds[query_condition]
    query_batch_labels = batch_labels[query_condition]

    support_to_take = []
    zero_proto = torch.zeros_like(support_preds[:1])
    for class_idx in range(support_batch_labels.shape[1]):
        class_condition = support_batch_labels[:, class_idx] > 0.5
        if class_condition.any():
            support_to_take.append(
                support_preds[class_condition].mean(dim=0, keepdim=True)
            )
        else:
            support_to_take.append(zero_proto)

    return {
        "query_labels": labels[query_condition],
        "query_batch_labels": query_batch_labels,
        "support_labels": episode_class_ids,
        "support_batch_labels": torch.eye(
            support_batch_labels.shape[1], device=device
        ),
        "support_preds": torch.cat(support_to_take, dim=0),
        "query_preds": query_preds,
        "episode_class_ids": episode_class_ids,
        "query_condition": query_condition,
    }


def support_query_split_multilabel_conditioned(base_split, few_shot_aux):
    """Replace support prototypes with text-conditioned support branches."""
    support_preds = base_split["support_preds"]
    branch_tokens = few_shot_aux["support_conditioned_patch_tokens"]
    branch_class_indices = few_shot_aux["support_branch_class_indices"].long()

    conditioned_support = []
    num_episode_classes = support_preds.shape[0]
    for class_idx in range(num_episode_classes):
        class_mask = branch_class_indices == class_idx
        if class_mask.any():
            class_tokens = branch_tokens[class_mask]
            conditioned_support.append(class_tokens.mean(dim=0, keepdim=True))
        else:
            conditioned_support.append(
                support_preds[class_idx:class_idx + 1].mean(dim=2, keepdim=True)
            )

    query_preds = base_split["query_preds"]
    if (
        "query_conditioned_patch_tokens" in few_shot_aux
        and "query_conditioned_sample_indices" in few_shot_aux
    ):
        expected_indices = torch.nonzero(
            base_split["query_condition"], as_tuple=False
        ).flatten().to(few_shot_aux["query_conditioned_sample_indices"].device)
        actual_indices = few_shot_aux["query_conditioned_sample_indices"].long()
        if not torch.equal(expected_indices, actual_indices):
            raise ValueError(
                "LGA query sample indices do not match the episode query ordering."
            )
        query_preds = few_shot_aux["query_conditioned_patch_tokens"]

    return {
        **base_split,
        "support_preds": torch.cat(conditioned_support, dim=0),
        "query_preds": query_preds,
    }


def few_shot_aux_has_support_tokens(few_shot_aux):
    """Return True when few_shot_aux carries support replacement tokens."""
    return (
        isinstance(few_shot_aux, dict)
        and "support_conditioned_patch_tokens" in few_shot_aux
        and "support_branch_class_indices" in few_shot_aux
    )


def few_shot_aux_has_query_partial_logits(few_shot_aux):
    """Return True when few_shot_aux carries query-side q2s logits."""
    return (
        isinstance(few_shot_aux, dict)
        and "query_partial_q2s_logits" in few_shot_aux
        and isinstance(few_shot_aux["query_partial_q2s_logits"], torch.Tensor)
    )


def compute_evidence_mil_loss(
    frame_margins,
    frame_valid,
    labels,
    topk_frames=3,
    temperature=0.10,
):
    """Supervise the strongest valid frame margins with video labels.

    This is deliberately a video-level MIL objective: it never consumes a
    patch or frame target.  Positive and Confuser responses have already been
    compared on the same Evidence-map region, and only the strongest valid
    frame margins are averaged for each Query/class pair.  Pairs without a
    valid Positive/Confuser comparison are excluded rather than trained toward
    an arbitrary zero margin.
    """
    if not isinstance(frame_margins, torch.Tensor):
        raise TypeError("frame_margins must be a Tensor.")
    if not isinstance(frame_valid, torch.Tensor):
        raise TypeError("frame_valid must be a Tensor.")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a Tensor.")
    if frame_margins.ndim != 3:
        raise ValueError(
            "frame_margins must have shape [Q,K,T]; got "
            f"{tuple(frame_margins.shape)}."
        )
    if tuple(frame_valid.shape) != tuple(frame_margins.shape):
        raise ValueError(
            "frame_valid must match frame_margins; got "
            f"{tuple(frame_valid.shape)} and {tuple(frame_margins.shape)}."
        )
    if tuple(labels.shape) != tuple(frame_margins.shape[:2]):
        raise ValueError(
            "labels must match the Query/class axes of frame_margins; got "
            f"{tuple(labels.shape)} and {tuple(frame_margins.shape[:2])}."
        )
    topk_frames = int(topk_frames)
    temperature = float(temperature)
    if topk_frames <= 0:
        raise ValueError("EVIDENCE_VIDEO_TOPK_FRAMES must be positive.")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("EVIDENCE_MIL_TEMPERATURE must be finite and positive.")

    margins = torch.nan_to_num(
        frame_margins.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    valid = frame_valid.to(device=margins.device).bool()
    topk = min(topk_frames, margins.shape[-1])
    masked = margins.masked_fill(~valid, float("-inf"))
    values, indices = torch.topk(masked, k=topk, dim=-1)
    selected_valid = torch.gather(valid, -1, indices)
    values = torch.where(selected_valid, values, torch.zeros_like(values))
    counts = selected_valid.sum(dim=-1)
    video_margin = values.sum(dim=-1) / counts.clamp_min(1).to(values.dtype)
    pair_valid = counts.gt(0)
    mil_logits = torch.nan_to_num(
        video_margin / temperature,
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    )
    if pair_valid.any():
        loss = F.binary_cross_entropy_with_logits(
            mil_logits[pair_valid],
            labels.float().to(device=margins.device)[pair_valid],
        )
    else:
        # Preserve a differentiable scalar on the correct device without
        # manufacturing supervision for an episode that has no Confuser.
        loss = margins.sum() * 0.0
    return {
        "loss": loss,
        "video_margin": video_margin,
        "logits": mil_logits,
        "valid": pair_valid,
    }


def compute_query_partial_q2s_loss(
    verified_logits,
    q2s_labels,
    few_shot_aux,
    cfg,
):
    """Compute the training BCE for the Query partial matcher.

    The verified logits are the public/final route (including the optional
    matchability penalty).  When ``DUAL_LOGIT_LOSS_ENABLE`` is set, the
    unpenalized ``query_partial_q2s_base_logits`` are supervised alongside
    them.  Both branches receive exactly the same solver-temperature scaling
    before BCE, so enabling the auxiliary branch does not silently change the
    logit calibration of the existing objective.

    Returns ``(loss, verified_logits_scaled, diagnostics)``.  Diagnostics are
    detached scalar tensors suitable for the existing all-reduce/logging path.
    This helper is intentionally training-only; evaluation should continue to
    use the verified logits for its metric definition.
    """
    if not isinstance(verified_logits, torch.Tensor):
        raise TypeError("verified_logits must be a Tensor.")
    if not isinstance(q2s_labels, torch.Tensor):
        raise TypeError("q2s_labels must be a Tensor.")
    if tuple(verified_logits.shape) != tuple(q2s_labels.shape):
        raise ValueError(
            "verified_logits and q2s_labels must share shape; got "
            f"{tuple(verified_logits.shape)} and {tuple(q2s_labels.shape)}."
        )

    solver_temperature = float(getattr(cfg.SOLVER, "TEMPRATURE", 1.0))
    if not np.isfinite(solver_temperature) or solver_temperature <= 0.0:
        raise ValueError("SOLVER.TEMPRATURE must be finite and positive.")

    def _sanitize(logits):
        scaled = logits / solver_temperature
        return torch.nan_to_num(
            scaled,
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )

    labels = q2s_labels.float()
    verified_scaled = _sanitize(verified_logits)
    verified_loss = F.binary_cross_entropy_with_logits(
        verified_scaled.float(),
        labels,
    )
    diagnostics = {
        "q2s_verified_loss": verified_loss.detach(),
    }

    match_cfg = getattr(
        getattr(cfg, "FEW_SHOT", None),
        "QUERY_CLASS_MATCHABILITY",
        None,
    )
    dual_enabled = bool(getattr(match_cfg, "DUAL_LOGIT_LOSS_ENABLE", False))
    evidence_mil_weight = float(
        getattr(match_cfg, "EVIDENCE_MIL_LOSS_WEIGHT", 0.0)
    )
    if not np.isfinite(evidence_mil_weight) or evidence_mil_weight < 0.0:
        raise ValueError(
            "EVIDENCE_MIL_LOSS_WEIGHT must be finite and non-negative."
        )

    evidence_mil = None
    if evidence_mil_weight > 0.0:
        if not isinstance(few_shot_aux, dict):
            raise ValueError(
                "EVIDENCE_MIL_LOSS_WEIGHT > 0 requires few_shot_aux."
            )
        frame_margins = few_shot_aux.get("query_frame_relative_margin")
        frame_valid = few_shot_aux.get("query_frame_valid")
        if not isinstance(frame_margins, torch.Tensor) or not isinstance(
            frame_valid, torch.Tensor
        ):
            raise ValueError(
                "Evidence-MIL requires query_frame_relative_margin and "
                "query_frame_valid Tensors."
            )
        evidence_mil = compute_evidence_mil_loss(
            frame_margins,
            frame_valid,
            labels,
            topk_frames=int(
                getattr(match_cfg, "EVIDENCE_VIDEO_TOPK_FRAMES", 3)
            ),
            temperature=float(
                getattr(match_cfg, "EVIDENCE_MIL_TEMPERATURE", 0.10)
            ),
        )
        diagnostics.update({
            "q2s_evidence_mil_loss": evidence_mil["loss"].detach(),
            "q2s_evidence_mil_weighted_loss": (
                evidence_mil_weight * evidence_mil["loss"]
            ).detach(),
            "q2s_evidence_mil_valid_fraction": (
                evidence_mil["valid"].float().mean().detach()
            ),
        })

    if not dual_enabled:
        loss = verified_loss
        if evidence_mil is not None:
            loss = loss + evidence_mil_weight * evidence_mil["loss"]
        return loss, verified_scaled, diagnostics

    if not isinstance(few_shot_aux, dict):
        raise ValueError(
            "DUAL_LOGIT_LOSS_ENABLE requires few_shot_aux with base logits."
        )
    base_logits = few_shot_aux.get("query_partial_q2s_base_logits")
    if not isinstance(base_logits, torch.Tensor):
        raise ValueError(
            "DUAL_LOGIT_LOSS_ENABLE requires base logits Tensor "
            "'query_partial_q2s_base_logits'."
        )
    if tuple(base_logits.shape) != tuple(q2s_labels.shape):
        raise ValueError(
            "query_partial_q2s_base_logits must match q2s_labels; got "
            f"{tuple(base_logits.shape)} and {tuple(q2s_labels.shape)}."
        )

    base_weight = float(getattr(match_cfg, "BASE_LOGIT_LOSS_WEIGHT", 0.50))
    verified_weight = float(
        getattr(match_cfg, "VERIFIED_LOGIT_LOSS_WEIGHT", 0.50)
    )
    if (
        not np.isfinite(base_weight)
        or not np.isfinite(verified_weight)
        or base_weight < 0.0
        or verified_weight < 0.0
        or base_weight + verified_weight <= 0.0
    ):
        raise ValueError(
            "BASE_LOGIT_LOSS_WEIGHT and VERIFIED_LOGIT_LOSS_WEIGHT must be "
            "finite, non-negative, and not both zero."
        )

    base_scaled = _sanitize(base_logits)
    base_loss = F.binary_cross_entropy_with_logits(
        base_scaled.float(),
        labels,
    )
    loss = base_weight * base_loss + verified_weight * verified_loss
    if evidence_mil is not None:
        loss = loss + evidence_mil_weight * evidence_mil["loss"]
    diagnostics.update({
        "q2s_base_loss": base_loss.detach(),
        "q2s_verified_loss": verified_loss.detach(),
    })
    return loss, verified_scaled, diagnostics


def get_text_align_loss(few_shot_aux, ref_tensor):
    """Return the text alignment loss scalar or a zero scalar on the right device."""
    if isinstance(few_shot_aux, dict) and "text_align_loss" in few_shot_aux:
        align_loss = few_shot_aux["text_align_loss"]
        if not isinstance(align_loss, torch.Tensor):
            align_loss = ref_tensor.new_tensor(float(align_loss))
        return torch.nan_to_num(align_loss, nan=0.0, posinf=1e4, neginf=0.0)
    return ref_tensor.new_zeros(())


def get_query_null_orthogonal_loss(few_shot_aux, ref_tensor):
    """Return the A2 Null orthogonal loss or an FP32 zero scalar."""
    if (
        isinstance(few_shot_aux, dict)
        and "query_null_orthogonal_loss" in few_shot_aux
    ):
        null_ortho_loss = few_shot_aux["query_null_orthogonal_loss"]
        if not isinstance(null_ortho_loss, torch.Tensor):
            null_ortho_loss = ref_tensor.new_tensor(
                float(null_ortho_loss),
                dtype=torch.float32,
            )
        return torch.nan_to_num(
            null_ortho_loss.float(),
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )
    return ref_tensor.new_zeros((), dtype=torch.float32)


def _query_null_aux_scalar(few_shot_aux, key, reference):
    """Read one finite scalar diagnostic on ``reference``'s device."""
    value = few_shot_aux.get(key, 0.0)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{key} must be a scalar; got shape {tuple(value.shape)}."
            )
        value = value.to(device=reference.device).float().reshape(())
    else:
        value = reference.new_tensor(float(value))
    return torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)


def get_query_null_route_metrics(few_shot_aux, q2s_labels):
    """Compute label-stratified A2 mechanism diagnostics for one episode batch."""
    if (
        not isinstance(few_shot_aux, dict)
        or "query_null_weights" not in few_shot_aux
    ):
        return {}

    null_weights = few_shot_aux["query_null_weights"]
    if not isinstance(null_weights, torch.Tensor) or null_weights.ndim != 3:
        shape = (
            tuple(null_weights.shape)
            if isinstance(null_weights, torch.Tensor)
            else type(null_weights).__name__
        )
        raise ValueError(
            "query_null_weights must be a tensor with shape [Q,K,T]; "
            f"got {shape}."
        )
    video_null = torch.nan_to_num(
        null_weights.float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0).mean(dim=-1)
    labels = q2s_labels.to(device=video_null.device).float()
    if tuple(labels.shape) != tuple(video_null.shape):
        raise ValueError(
            "q2s_labels must match Query Null video weights [Q,K]; got "
            f"{tuple(labels.shape)} versus {tuple(video_null.shape)}."
        )
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask

    def masked_mean(value, mask):
        selected = value[mask]
        if selected.numel() == 0:
            return value.new_zeros(())
        return selected.mean()

    positive_null_mean = masked_mean(video_null, positive_mask)
    negative_null_mean = masked_mean(video_null, negative_mask)
    diag_similarity = few_shot_aux.get("query_partial_diag_similarity")
    if isinstance(diag_similarity, torch.Tensor):
        diag_similarity = torch.nan_to_num(
            diag_similarity.to(device=video_null.device).float(),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        if tuple(diag_similarity.shape) != tuple(video_null.shape):
            raise ValueError(
                "query_partial_diag_similarity must match [Q,K]; got "
                f"{tuple(diag_similarity.shape)} versus "
                f"{tuple(video_null.shape)}."
            )
    else:
        diag_similarity = torch.zeros_like(video_null)

    metrics = {
        "query_null_score": _query_null_aux_scalar(
            few_shot_aux,
            "query_null_score",
            video_null,
        ),
        "positive_null_mean": positive_null_mean,
        "negative_null_mean": negative_null_mean,
        "null_gap": negative_null_mean - positive_null_mean,
        "null_support_mean_abs_cosine": _query_null_aux_scalar(
            few_shot_aux,
            "query_null_support_mean_abs_cosine",
            video_null,
        ),
        "null_support_max_abs_cosine": _query_null_aux_scalar(
            few_shot_aux,
            "query_null_support_max_abs_cosine",
            video_null,
        ),
        "positive_diag_similarity": masked_mean(
            diag_similarity,
            positive_mask,
        ),
        "negative_diag_similarity": masked_mean(
            diag_similarity,
            negative_mask,
        ),
    }
    return {key: value.detach() for key, value in metrics.items()}


def get_query_matchability_metrics(few_shot_aux, q2s_labels):
    """Return label-stratified diagnostics for the Query-class matcher.

    These metrics are computed after the forward pass using the evaluation
    labels only for reporting.  The model-side route receives Support labels
    only; changing Query labels must not change its logits.
    """
    if (
        not isinstance(few_shot_aux, dict)
        or "query_class_matchability" not in few_shot_aux
    ):
        return {}

    matchability = few_shot_aux["query_class_matchability"]
    if not isinstance(matchability, torch.Tensor) or matchability.ndim != 2:
        shape = (
            tuple(matchability.shape)
            if isinstance(matchability, torch.Tensor)
            else type(matchability).__name__
        )
        raise ValueError(
            "query_class_matchability must have shape [Q,K]; got "
            f"{shape}."
        )
    labels = q2s_labels.to(device=matchability.device).float()
    if tuple(labels.shape) != tuple(matchability.shape):
        raise ValueError(
            "q2s_labels must match Query matchability [Q,K]; got "
            f"{tuple(labels.shape)} versus {tuple(matchability.shape)}."
        )

    def _pair_tensor(key, default=None):
        value = few_shot_aux.get(key, default)
        if value is None:
            return None
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
            matchability.shape
        ):
            raise ValueError(
                f"{key} must match [Q,K]; got "
                f"{tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__}."
            )
        return torch.nan_to_num(
            value.to(device=matchability.device).float(),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

    positive_similarity = _pair_tensor(
        "query_class_positive_similarity",
        few_shot_aux.get(
            "query_partial_diag_similarity",
            torch.zeros_like(matchability),
        ),
    )
    negative_similarity = _pair_tensor(
        "query_class_hardest_confuser_similarity",
        torch.zeros_like(matchability),
    )
    relative_margin = _pair_tensor(
        "query_class_relative_margin",
        few_shot_aux.get("query_class_evidence", torch.zeros_like(matchability)),
    )
    penalty = _pair_tensor(
        "query_class_log_penalty",
        torch.zeros_like(matchability),
    )

    positive_mask = labels > 0.5
    negative_mask = ~positive_mask

    def masked_mean(value, mask):
        selected = value[mask]
        if selected.numel() == 0:
            return value.new_zeros(())
        return selected.mean()

    metrics = {
        "matchability_positive_similarity": masked_mean(
            positive_similarity,
            positive_mask,
        ),
        "matchability_negative_similarity": masked_mean(
            positive_similarity,
            negative_mask,
        ),
        "matchability_positive_confuser_similarity": masked_mean(
            negative_similarity,
            positive_mask,
        ),
        "matchability_negative_confuser_similarity": masked_mean(
            negative_similarity,
            negative_mask,
        ),
        "matchability_positive_margin": masked_mean(
            relative_margin,
            positive_mask,
        ),
        "matchability_negative_margin": masked_mean(
            relative_margin,
            negative_mask,
        ),
        "matchability_positive_rho": masked_mean(
            matchability.float(),
            positive_mask,
        ),
        "matchability_negative_rho": masked_mean(
            matchability.float(),
            negative_mask,
        ),
        "matchability_penalty_mean": penalty.mean(),
    }
    local_positive = _pair_tensor("query_local_positive_similarity_mean")
    local_confuser = _pair_tensor("query_local_confuser_similarity_mean")
    local_margin = _pair_tensor("query_local_relative_margin_mean")
    local_weight_shift = _pair_tensor("query_local_weight_shift_mean")
    if local_positive is not None:
        metrics.update({
            "local_positive_similarity_positive": masked_mean(
                local_positive,
                positive_mask,
            ),
            "local_positive_similarity_negative": masked_mean(
                local_positive,
                negative_mask,
            ),
        })
    if local_confuser is not None:
        metrics.update({
            "local_confuser_similarity_positive": masked_mean(
                local_confuser,
                positive_mask,
            ),
            "local_confuser_similarity_negative": masked_mean(
                local_confuser,
                negative_mask,
            ),
        })
    if local_margin is not None:
        metrics.update({
            "local_margin_positive": masked_mean(local_margin, positive_mask),
            "local_margin_negative": masked_mean(local_margin, negative_mask),
        })
    if local_weight_shift is not None:
        metrics["local_weight_shift_mean"] = local_weight_shift.mean()
    frame_pairs = {
        "frame_positive_evidence": _pair_tensor(
            "query_frame_positive_evidence_mean"
        ),
        "frame_confuser_evidence": _pair_tensor(
            "query_frame_confuser_evidence_mean"
        ),
        "frame_margin": _pair_tensor("query_frame_relative_margin_mean"),
        "frame_rho": _pair_tensor("query_frame_matchability_mean"),
        "frame_temporal_delta": _pair_tensor(
            "query_frame_temporal_logit_delta"
        ),
        "frame_max_t_switch": _pair_tensor(
            "query_frame_max_t_switch_fraction"
        ),
        "evidence_effective_patches": _pair_tensor(
            "query_evidence_effective_patches_mean"
        ),
        "evidence_top1_mass": _pair_tensor(
            "query_evidence_top1_mass_mean"
        ),
        "absolute_evidence": _pair_tensor(
            "query_frame_absolute_evidence_mean"
        ),
        "transport_patch_mass": _pair_tensor(
            "query_frame_patch_mass_mean"
        ),
        "transport_unmatched_mass": _pair_tensor(
            "query_frame_unmatched_mass_mean"
        ),
        "transport_logit_delta": _pair_tensor(
            "query_frame_transport_logit_delta"
        ),
    }
    for name, value in frame_pairs.items():
        if value is None:
            continue
        metrics[f"{name}_positive"] = masked_mean(value, positive_mask)
        metrics[f"{name}_negative"] = masked_mean(value, negative_mask)
    valid_count = few_shot_aux.get("query_class_confuser_valid_count")
    if isinstance(valid_count, torch.Tensor):
        metrics["matchability_confuser_valid_count"] = (
            torch.nan_to_num(valid_count.float(), nan=0.0).mean()
        )
    absolute_reliable = few_shot_aux.get(
        "support_absolute_calibration_reliable"
    )
    if isinstance(absolute_reliable, torch.Tensor):
        metrics["transport_support_reliable_fraction"] = (
            absolute_reliable.float().mean()
        )
    return {key: value.detach() for key, value in metrics.items()}


def multilabel_top1_accuracy(logits, labels):
    """Top-1 is correct if the highest scoring class is one of the positive labels."""
    pred_idx = logits.argmax(dim=1, keepdim=True)
    correct = labels.gather(1, pred_idx).sum()
    return correct * 100.0 / max(logits.shape[0], 1)


def empty_ap_storage(num_classes):
    """Create storage for episode-level and per-class AP statistics."""
    return {
        "episodes": [],
        "per_class": {
            class_id: {"scores": [], "labels": []}
            for class_id in range(num_classes)
        },
    }


def update_ap_storage(storage, logits, labels, episode_class_ids):
    """Store the current episode for doc-style AP and per-class logging."""
    scores = logits.detach().float().cpu().numpy()
    targets = labels.detach().float().cpu().numpy()
    class_ids = episode_class_ids.detach().long().cpu().numpy().tolist()
    storage["episodes"].append(
        {
            "scores": scores.tolist(),
            "labels": targets.tolist(),
            "episode_class_ids": class_ids,
        }
    )
    for episode_idx, class_id in enumerate(class_ids):
        storage["per_class"][int(class_id)]["scores"].extend(
            scores[:, episode_idx].tolist()
        )
        storage["per_class"][int(class_id)]["labels"].extend(
            targets[:, episode_idx].tolist()
        )


def merge_ap_storage(storage):
    """Merge per-rank AP storage."""
    gathered = du.all_gather_unaligned(storage)
    merged = empty_ap_storage(len(storage["per_class"]))
    for rank_storage in gathered:
        merged["episodes"].extend(rank_storage["episodes"])
        for class_id, values in rank_storage["per_class"].items():
            merged["per_class"][int(class_id)]["scores"].extend(values["scores"])
            merged["per_class"][int(class_id)]["labels"].extend(values["labels"])
    return merged


def compute_ap_for_classes(storage, class_ids):
    """Compute macro AP over classes with at least one positive label."""
    aps = {}
    for class_id in class_ids:
        values = storage[int(class_id)]
        if not values["labels"]:
            continue
        labels = np.array(values["labels"], dtype=np.float32)
        if not np.any(labels):
            continue
        scores = np.array(values["scores"], dtype=np.float32)
        aps[int(class_id)] = float(average_precision_score(labels, scores))
    if not aps:
        return 0.0, aps
    return float(np.mean(list(aps.values())) * 100.0), aps


def compute_episode_ap(labels, scores, episode_class_ids, target_class_ids):
    """Compute macro AP for one episode over seen or novel classes only."""
    class_mask = np.array(
        [class_id in set(target_class_ids) for class_id in episode_class_ids],
        dtype=bool,
    )
    if not np.any(class_mask):
        return None

    filtered_labels = labels[:, class_mask]
    filtered_scores = scores[:, class_mask]
    valid_labels = np.any(filtered_labels, axis=0)
    if not np.any(valid_labels):
        return None

    filtered_labels = filtered_labels[:, valid_labels]
    filtered_scores = filtered_scores[:, valid_labels]
    return float(
        average_precision_score(filtered_labels, filtered_scores, average="macro")
    )


def compute_base_novel_hm(storage, cfg):
    """Compute doc-style seen/novel AP: per episode first, then mean over episodes."""
    seen_episode_aps = []
    novel_episode_aps = []
    for episode in storage["episodes"]:
        labels = np.array(episode["labels"], dtype=np.float32)
        scores = np.array(episode["scores"], dtype=np.float32)
        episode_class_ids = [int(class_id) for class_id in episode["episode_class_ids"]]

        ap_seen = compute_episode_ap(
            labels, scores, episode_class_ids, cfg.TEST.SEEN_LABELS
        )
        if ap_seen is not None:
            seen_episode_aps.append(ap_seen)

        ap_novel = compute_episode_ap(
            labels, scores, episode_class_ids, cfg.TEST.NOVEL_LABELS
        )
        if ap_novel is not None:
            novel_episode_aps.append(ap_novel)

    base_map = float(np.mean(seen_episode_aps) * 100.0) if seen_episode_aps else 0.0
    novel_map = (
        float(np.mean(novel_episode_aps) * 100.0) if novel_episode_aps else 0.0
    )
    if base_map + novel_map > 0:
        hm_map = 2.0 * base_map * novel_map / (base_map + novel_map)
    else:
        hm_map = 0.0
    _, base_aps = compute_ap_for_classes(storage["per_class"], cfg.TEST.SEEN_LABELS)
    _, novel_aps = compute_ap_for_classes(storage["per_class"], cfg.TEST.NOVEL_LABELS)
    return base_map, novel_map, hm_map, {**base_aps, **novel_aps}


def mean_or_nan(values):
    """Return a float mean or NaN for empty metric lists."""
    return float(np.mean(values)) if values else float("nan")
