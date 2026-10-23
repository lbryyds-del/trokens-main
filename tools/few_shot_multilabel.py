"""Utilities for SAV multi-label few-shot episodes."""

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import trokens.utils.distributed as du
from trokens.config.query_judgments import resolve_query_judgments


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


def unpack_episode_output(model_out):
    """Single-path outputs may include a global head, but no shared Patch tensor."""
    if isinstance(model_out, dict) and model_out.get("single_path_episode", False):
        return model_out.get("global_classification_logits"), None, model_out
    if isinstance(model_out, tuple) and len(model_out) == 3:
        return model_out
    preds, patches = model_out
    return preds, patches, None


def single_path_episode_targets(output, labels, metadata):
    """Align supervision AFTER forward; reject silent class/Query permutations."""
    ids = get_episode_class_ids(metadata, labels.device)
    actual_ids = output["query_partial_label_axis_global_labels"].to(labels.device)
    support = torch.as_tensor(np.array(metadata["sample_type"]) == "support",
                              device=labels.device, dtype=torch.bool)
    indices = (~support).nonzero(as_tuple=False).flatten()
    actual_indices = output["query_partial_query_sample_indices"].to(labels.device)
    if not torch.equal(ids, actual_ids) or not torch.equal(indices, actual_indices):
        raise ValueError("Single-path output does not match episode Query/class order.")
    targets = episode_labels_from_global(labels, ids).index_select(0, indices)
    if "episode_label_known_mask" in metadata:
        known = metadata["episode_label_known_mask"].to(labels.device).bool().index_select(0, indices)
        targets = targets.masked_fill(~known, -1)
    if targets.shape != output["query_partial_q2s_logits"].shape:
        raise ValueError("Single-path logits and supervised episode targets must match.")
    return dict(query_batch_labels=targets, episode_class_ids=ids,
                query_condition=~support, query_labels=labels[~support], support_labels=ids)


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


def compute_learned_evidence_mil_loss(frame_logits, frame_valid, labels):
    """Video-label supervision without a Top-K or calibrated temperature.

    A negative label value denotes unknown and is not supervised. Empty frame
    bags remain connected to autograd with exactly zero gradient.
    """
    if frame_logits.ndim != 3 or frame_valid.shape != frame_logits.shape:
        raise ValueError("Learned MIL requires logits and valid masks shaped [Q,K,T].")
    if labels.shape != frame_logits.shape[:2]:
        raise ValueError("Learned MIL labels must have shape [Q,K].")
    with torch.autocast(device_type=frame_logits.device.type, enabled=False):
        logits = frame_logits.float()
        valid = frame_valid.bool()
        count = valid.sum(-1)
        masked = logits.masked_fill(~valid, -torch.inf)
        masked = torch.where(count.unsqueeze(-1) > 0, masked, torch.zeros_like(masked))
        bag_logits = torch.logsumexp(masked, dim=-1) - count.clamp_min(1).float().log()
        bag_logits = torch.where(count > 0, bag_logits, torch.zeros_like(bag_logits))
        labels = labels.to(logits.device).float()
        pair_valid = (count > 0) & torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
        loss = (F.binary_cross_entropy_with_logits(bag_logits[pair_valid], labels[pair_valid])
                if pair_valid.any() else bag_logits.sum() * 0.0)
    return {"loss": loss, "logits": bag_logits, "valid": pair_valid}


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

    ``BASE_ONLY_LOSS_ENABLE`` instead supervises only the unpenalized base
    logits, skipping final BCE and both MIL losses. It returns scaled base
    logits for the training metric; evaluation output is unchanged.

    Returns ``(loss, training_logits_scaled, diagnostics)``.  Diagnostics are
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

    match_cfg = getattr(getattr(cfg, "FEW_SHOT", None), "QUERY_CLASS_MATCHABILITY", None)
    learned = bool(getattr(match_cfg, "ENABLE", False))
    labels = q2s_labels.float()
    def supervised_bce(logits):
        if not learned:
            return F.binary_cross_entropy_with_logits(logits.float(), labels)
        known = torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
        return (F.binary_cross_entropy_with_logits(logits.float()[known], labels[known])
                if known.any() else logits.sum() * 0.0)

    if bool(getattr(match_cfg, "BASE_ONLY_LOSS_ENABLE", False)):
        judgments = resolve_query_judgments(match_cfg)
        if (judgments.quality or judgments.uniqueness
                or bool(getattr(match_cfg, "DUAL_LOGIT_LOSS_ENABLE", False))):
            raise ValueError("BASE_ONLY_LOSS_ENABLE requires both judgments and dual-logit loss OFF.")
        if (float(match_cfg.BASE_LOGIT_LOSS_WEIGHT) != 1.0
                or float(match_cfg.VERIFIED_LOGIT_LOSS_WEIGHT) != 0.0):
            raise ValueError("Base-only loss requires base/verified weights 1/0.")
        base = (few_shot_aux.get("query_partial_q2s_base_logits")
                if isinstance(few_shot_aux, dict) else None)
        if not isinstance(base, torch.Tensor) or base.shape != q2s_labels.shape:
            raise ValueError("Base-only loss requires base logits matching q2s_labels.")
        base_scaled = _sanitize(base)
        base_loss = supervised_bce(base_scaled)
        # Do not evaluate a duplicate final BCE or either verifier MIL loss.
        return base_loss, base_scaled, {
            "q2s_base_loss": base_loss.detach(),
            "q2s_verified_loss": base_loss.new_zeros(()),
        }

    verified_scaled = _sanitize(verified_logits)
    verified_loss = supervised_bce(verified_scaled)
    diagnostics = {
        "q2s_verified_loss": verified_loss.detach(),
    }

    dual_enabled = bool(getattr(match_cfg, "DUAL_LOGIT_LOSS_ENABLE", False))
    learned_aux_loss = verified_loss.new_zeros(())
    if learned:
        judgments = resolve_query_judgments(match_cfg)
        verifier_cfg = getattr(match_cfg, "LEARNED_VERIFIER", None)
        for name, enabled, weight_name in (
            ("quality", judgments.quality, "SEMANTIC_MIL_WEIGHT"),
            ("relation", judgments.uniqueness, "RELATION_MIL_WEIGHT"),
        ):
            if not enabled:
                continue
            weight = float(getattr(verifier_cfg, weight_name, 0.05))
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"{weight_name} must be finite and non-negative.")
            if not isinstance(few_shot_aux, dict) or f"query_{name}_logits" not in few_shot_aux:
                raise ValueError(f"Learned {name} MIL requires verifier logits.")
            mil = compute_learned_evidence_mil_loss(
                few_shot_aux[f"query_{name}_logits"],
                few_shot_aux[f"query_{name}_valid"], labels,
            )
            learned_aux_loss = learned_aux_loss + weight * mil["loss"]
            diagnostics[f"q2s_{name}_mil_loss"] = mil["loss"].detach()
            diagnostics[f"q2s_{name}_mil_valid_fraction"] = mil["valid"].float().mean().detach()
        if isinstance(few_shot_aux, dict) and "query_verifier_transport_strength" in few_shot_aux:
            diagnostics["q2s_verifier_transport_strength"] = few_shot_aux["query_verifier_transport_strength"].detach()

    if not dual_enabled:
        loss = verified_loss + learned_aux_loss
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
    base_loss = supervised_bce(base_scaled)
    loss = base_weight * base_loss + verified_weight * verified_loss + learned_aux_loss
    diagnostics.update({
        "q2s_base_loss": base_loss.detach(),
        "q2s_verified_loss": verified_loss.detach(),
    })
    return loss, verified_scaled, diagnostics


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

    if few_shot_aux.get("query_evidence_enabled", False):
        metrics = {}
        pairs = {
            "verifier_quality": "query_quality_mass_mean",
            "verifier_rho": "query_frame_matchability_mean",
            "verifier_quality_logit": "query_quality_logit_mean",
            "verifier_relation_logit": "query_relation_logit_mean",
            "verifier_target_mass": "query_frame_target_mass_mean",
            "verifier_confuser_mass": "query_frame_confuser_mass_mean",
            "verifier_null_mass": "query_frame_null_mass_mean",
            "verifier_quality_effective_patches": "query_quality_effective_patches_mean",
            "verifier_quality_top1_mass": "query_quality_top1_mass_mean",
            "verifier_quality_anchor": "query_quality_anchor_mean",
            "verifier_quality_correction": "query_quality_correction_mean",
            "verifier_quality_correction_abs": "query_quality_correction_abs_mean",
            "verifier_quality_correction_saturation": "query_quality_correction_saturation_mean",
            "verifier_quality_pool_shift": "query_quality_pool_shift_mean",
            "verifier_quality_semantic_evidence": "query_quality_semantic_evidence_mean",
            "verifier_quality_relation_feature_norm": "query_quality_relation_feature_norm_mean",
            "verifier_route_effective_patches": "query_evidence_effective_patches_mean",
            "verifier_route_top1_mass": "query_evidence_top1_mass_mean",
            "verifier_logit_delta": "query_frame_transport_logit_delta",
        }
        for name, key in pairs.items():
            value = few_shot_aux.get(key)
            if value is None:
                continue
            if value.shape != labels.shape:
                raise ValueError(f"{key} must match Query labels [Q,K].")
            for suffix, mask in (("positive", labels > 0.5), ("negative", labels == 0)):
                selected = value[mask]
                metrics[f"{name}_{suffix}"] = selected.float().mean() if selected.numel() else value.new_zeros(())
        for branch in ("quality", "relation"):
            available = few_shot_aux.get(f"query_{branch}_reference_available")
            if available is not None:
                metrics[f"verifier_{branch}_reference_fraction"] = available.float().mean()
        for name in ("metric_temperature", "metric_scale", "quality_scale", "quality_bias",
                     "quality_relation_strength", "quality_max_evidence_delta"):
            value = few_shot_aux.get("query_" + name)
            if value is not None:
                metrics["verifier_" + name] = value.float()
        return {name: value.detach() for name, value in metrics.items()}

    return {}


def get_query_evidence_csv_fields(few_shot_aux, labels):
    """Return available per-Query/class diagnostics with the CSV column names."""
    if not isinstance(few_shot_aux, dict):
        return {}
    fields = {
        "query_class_base_score": "query_partial_q2s_base_logits",
        "query_class_temporal_score": "query_partial_q2s_temporal_logits",
        "query_class_matchability": "query_class_matchability",
        "query_quality_mass": "query_quality_mass_mean",
        "query_quality_logit": "query_quality_logit_mean",
        "query_quality_anchor": "query_quality_anchor_mean",
        "query_quality_correction": "query_quality_correction_mean",
        "query_quality_correction_abs": "query_quality_correction_abs_mean",
        "query_quality_correction_saturation": "query_quality_correction_saturation_mean",
        "query_quality_semantic_evidence": "query_quality_semantic_evidence_mean",
        "query_quality_relation_feature_norm": "query_quality_relation_feature_norm_mean",
        "query_relation_logit": "query_relation_logit_mean",
        "query_target_mass": "query_frame_target_mass_mean",
        "query_confuser_mass": "query_frame_confuser_mass_mean",
        "query_null_mass": "query_frame_null_mass_mean",
        "query_frame_matchability": "query_frame_matchability_mean",
        "query_evidence_effective_patches": "query_evidence_effective_patches_mean",
        "query_evidence_top1_mass": "query_evidence_top1_mass_mean",
        "query_frame_temporal_logit_delta": "query_frame_temporal_logit_delta",
        "query_frame_max_t_switch_fraction": "query_frame_max_t_switch_fraction",
        "query_frame_patch_mass": "query_frame_patch_mass_mean",
        "query_frame_unmatched_mass": "query_frame_unmatched_mass_mean",
        "query_frame_transport_logit_delta": "query_frame_transport_logit_delta",
    }
    result = {}
    for name, key in fields.items():
        value = few_shot_aux.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.shape != labels.shape:
            raise ValueError(f"{key} must align with Query/class labels {tuple(labels.shape)}.")
        result[name] = value.detach().float().cpu().numpy()
    return result


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


def compute_episode_ap(
    labels,
    scores,
    episode_class_ids,
    target_class_ids,
    drop_empty_query_rows=False,
):
    """Compute macro AP for one episode over a Base/Novel class subset.

    TinyVIRAT's protocol excludes Query rows with no positive label in the
    evaluated subset. This flag is deliberately evaluation-only: it does not
    alter episode sampling, Support/Query tensors, or training losses.
    """
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
    if drop_empty_query_rows:
        non_zero_mask = ~np.all(filtered_labels == 0, axis=1)
        filtered_labels = filtered_labels[non_zero_mask]
        filtered_scores = filtered_scores[non_zero_mask]
        if filtered_labels.shape[0] == 0:
            return None
    return float(
        average_precision_score(filtered_labels, filtered_scores, average="macro")
    )


def compute_base_novel_hm(storage, cfg):
    """Compute doc-style seen/novel AP: per episode first, then mean over episodes."""
    seen_episode_aps = []
    novel_episode_aps = []
    drop_empty_query_rows = bool(
        getattr(cfg.TEST, "DROP_EMPTY_QUERY_ROWS", False)
    )
    for episode in storage["episodes"]:
        labels = np.array(episode["labels"], dtype=np.float32)
        scores = np.array(episode["scores"], dtype=np.float32)
        episode_class_ids = [int(class_id) for class_id in episode["episode_class_ids"]]

        ap_seen = compute_episode_ap(
            labels,
            scores,
            episode_class_ids,
            cfg.TEST.SEEN_LABELS,
            drop_empty_query_rows=drop_empty_query_rows,
        )
        if ap_seen is not None:
            seen_episode_aps.append(ap_seen)

        ap_novel = compute_episode_ap(
            labels,
            scores,
            episode_class_ids,
            cfg.TEST.NOVEL_LABELS,
            drop_empty_query_rows=drop_empty_query_rows,
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
