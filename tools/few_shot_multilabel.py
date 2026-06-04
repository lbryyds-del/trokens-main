"""Utilities for SAV multi-label few-shot episodes."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import trokens.utils.distributed as du


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
            conditioned_support.append(support_preds[class_idx:class_idx + 1])

    return {
        **base_split,
        "support_preds": torch.cat(conditioned_support, dim=0),
    }


def few_shot_aux_has_support_tokens(few_shot_aux):
    """Return True when few_shot_aux carries support replacement tokens."""
    return (
        isinstance(few_shot_aux, dict)
        and "support_conditioned_patch_tokens" in few_shot_aux
        and "support_branch_class_indices" in few_shot_aux
    )


def get_text_align_loss(few_shot_aux, ref_tensor):
    """Return the text alignment loss scalar or a zero scalar on the right device."""
    if isinstance(few_shot_aux, dict) and "text_align_loss" in few_shot_aux:
        align_loss = few_shot_aux["text_align_loss"]
        if not isinstance(align_loss, torch.Tensor):
            align_loss = ref_tensor.new_tensor(float(align_loss))
        return torch.nan_to_num(align_loss, nan=0.0, posinf=1e4, neginf=0.0)
    return ref_tensor.new_zeros(())


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
