#!/usr/bin/env python3

"""Test a few shot classification model."""
# pylint: disable=wrong-import-position,import-error,wrong-import-order
import json
import os
import sys
import pprint
from datetime import datetime
import logging as pylogging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from einops import rearrange
from few_shot_multilabel import (
    compute_base_novel_hm,
    empty_ap_storage,
    episode_labels_from_global,
    few_shot_aux_has_query_partial_logits,
    few_shot_aux_has_support_tokens,
    get_text_align_loss,
    get_episode_class_ids,
    is_multilabel_episode,
    mean_or_nan,
    merge_ap_storage,
    multilabel_top1_accuracy,
    q2s_cos_sim_fp32,
    support_query_split_multilabel,
    support_query_split_multilabel_conditioned,
    update_ap_storage,
)
import trokens.utils.checkpoint as cu
import trokens.utils.distributed as du
import trokens.utils.logging as logging
import trokens.utils.metrics as metrics
import trokens.utils.misc as misc
from trokens.datasets import loader
from trokens.utils.meters import ValMeter
from trokens.models import build_model
from fvcore.common.config import CfgNode
from fvcore.nn.precise_bn import update_bn_stats

try:
    import wandb
except ImportError:
    wandb = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

def autocast_context(enabled):
    """Create a CUDA autocast context that works across torch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def wandb_init_dict(cfg_node):
    """Convert a config node to dictionary.
    """
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = wandb_init_dict(v)
        return cfg_dict
# pylint: disable=line-too-long
def process_patch_tokens(cfg, support_tokens, query_tokens):
    """
    Process the patch tokens for few shot learning.
    Ref: https://github.com/alibaba-mmai-research/MoLo/blob/f7f73b6dd8cba446b414b1c47652ab26033bc88e/models/base/few_shot.py#L2552
    args:
        cfg: config
        support_tokens: (num_support, temp_len, num_patches, embed_dim)
        query_tokens: (num_query, temp_len, num_patches, embed_dim)
    """
    #Putting an activation here, may be not needed
    support_tokens = torch.nan_to_num(support_tokens, nan=0.0, posinf=0.0, neginf=0.0)
    query_tokens = torch.nan_to_num(query_tokens, nan=0.0, posinf=0.0, neginf=0.0)
    support_tokens = F.relu(support_tokens)
    query_tokens = F.relu(query_tokens)
    num_supports = support_tokens.shape[0]
    num_querries = query_tokens.shape[0]
    if not cfg.MODEL.USE_EXTRA_ENCODER:
        if cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'temporal':
            support_tokens = support_tokens.mean(dim=1)
            query_tokens = query_tokens.mean(dim=1)
        elif cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'spatial':
            support_tokens = support_tokens.mean(dim=2)
            query_tokens = query_tokens.mean(dim=2)
        elif cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'no_agg':
            support_tokens = rearrange(support_tokens, 'b t p e -> b (t p) e')
            query_tokens = rearrange(query_tokens, 'b t p e -> b (t p) e')
        else:
            raise NotImplementedError(
                f"Aggregation method {cfg.FEW_SHOT.PATCH_TOKENS_AGG} not implemented")

    support_tokens = rearrange(support_tokens, 'b p e -> (b p) e')
    query_tokens = rearrange(query_tokens, 'b p e -> (b p) e')
    sim_matrix = cos_sim(query_tokens, support_tokens)
    sim_matrix = torch.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    dist_matrix = 1 - sim_matrix

    dist_rearranged = rearrange(dist_matrix, '(q qt) (s st) -> q s qt st',
                                q=num_querries, s=num_supports)
    # Take the minimum distance for each query token
    dist_logits = dist_rearranged.min(3)[0].sum(2) + dist_rearranged.min(2)[0].sum(2)
    if cfg.FEW_SHOT.DIST_NORM == 'max_div':
        max_dist = dist_logits.max(dim=1, keepdim=True)[0]
        dist_logits = dist_logits / max_dist
    elif cfg.FEW_SHOT.DIST_NORM == 'max_sub':
        max_dist = dist_logits.max(dim=1, keepdim=True)[0]
        dist_logits = max_dist - dist_logits
    dist_logits = torch.nan_to_num(dist_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    return - dist_logits

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    return q2s_cos_sim_fp32(x, y, epsilon=epsilon)


def support_query_split(preds, labels, metadata):
    """
    Split the preds and labels into support and query.
    """

    device = preds.device
    sample_info = np.array(metadata['sample_type'])
    batch_labels = metadata['batch_label']
    support_condition = sample_info=='support'
    support_labels = labels[support_condition]
    support_preds = preds[support_condition]
    support_batch_labels = batch_labels[support_condition]

    # average the support preds for each class
    support_to_take = []
    support_main_label_to_take = []
    support_batch_label_to_take = []
    for label in np.unique(support_batch_labels.cpu().numpy()):
        label_condition = support_batch_labels==label
        label_mean_support = support_preds[label_condition].mean(dim=0,
                                                                keepdims=True)
        support_main_label = support_labels[label_condition][0]
        support_main_label_to_take.append(support_main_label)
        support_batch_label_to_take.append(label)
        support_to_take.append(label_mean_support)
    support_labels = torch.tensor(support_main_label_to_take, device=device)
    support_batch_labels = torch.tensor(support_batch_label_to_take, device=device)
    support_preds = torch.cat(support_to_take, dim=0)


    query_labels = labels[~support_condition]
    query_preds = preds[~support_condition]
    query_batch_labels = batch_labels[~support_condition]
    return_dict = {
        'query_labels':query_labels,
        'query_batch_labels':query_batch_labels,
        'support_labels':support_labels,
        'support_batch_labels':support_batch_labels,
        'support_preds':support_preds,
        'query_preds':query_preds
    }
    return return_dict

logger = logging.get_logger(__name__)


def attach_test_log_file(output_dir):
    """Attach a dedicated per-test log file alongside stdout.log."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"test_{timestamp}.log")
    formatter = pylogging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    handler = pylogging.FileHandler(filename, mode="a", encoding="utf-8")
    handler.setLevel(pylogging.DEBUG)
    handler.setFormatter(formatter)
    pylogging.getLogger().addHandler(handler)
    return filename, handler


def append_test_history(output_dir, payload):
    """Append one JSON line per finished test run."""
    history_path = os.path.join(output_dir, "test_history.jsonl")
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

def conv_fp16(var):
    """Convert to float16.
    """
    return np.float16(np.around(var, 4))


def create_progress_bar(total, description):
    """Create a progress bar only for the master process in interactive terminals."""
    if tqdm is None or not du.is_master_proc() or not sys.stderr.isatty():
        return None
    return tqdm(total=total, desc=description, dynamic_ncols=True, leave=False)


def should_log_iter_stats(progress_bar):
    """Use tqdm in interactive runs and keep per-iter logger output for non-interactive runs."""
    return progress_bar is None


def shot_metric_name(cfg):
    """Display name for the current few-shot setting."""
    return f"shot_{cfg.FEW_SHOT.K_SHOT}_acc"






@torch.no_grad()
def test_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    epoch_top_1_acc_few_shot = []
    epoch_q2s_loss = []
    epoch_align_loss = []
    multi_label = cfg.DATA.MULTI_LABEL
    if multi_label:
        ap_storage = empty_ap_storage(cfg.MODEL.NUM_CLASSES)
    else:
        num_test_classes = len(val_loader.batch_sampler.class_ids)
        if cfg.TRAIN.DATASET == 'FINEGYM':
            num_test_classes = 100
        confusion_matrix = np.zeros((num_test_classes, num_test_classes))
    all_df = []
    shot_acc_name = shot_metric_name(cfg)
    progress_bar = create_progress_bar(
        len(val_loader),
        f"Test ({len(val_loader)} episodes)",
    )

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cur_iter > len(val_loader):
            break
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        val_meter.data_toc()
        if is_multilabel_episode(cfg, labels, meta):
            meta['support_mask'] = torch.as_tensor(
                np.array(meta['sample_type']) == 'support',
                device=labels.device,
                dtype=torch.bool,
            )
            episode_class_ids = get_episode_class_ids(meta, labels.device)
            raw_positive_labels = labels.to(labels.device).float()
            meta['raw_positive_labels'] = raw_positive_labels
            meta['episode_positive_labels'] = episode_labels_from_global(
                raw_positive_labels,
                episode_class_ids,
            )
        with autocast_context(cfg.TRAIN.MIXED_PRECISION):
            input_dict = {'video':inputs, 'metadata':meta}
            # for few shot, patch tokens are also returning
            model_out = model(input_dict)
            if isinstance(model_out, tuple) and len(model_out) == 3:
                preds, patch_tokens, few_shot_aux = model_out
            else:
                preds, patch_tokens = model_out
                few_shot_aux = None
            if isinstance(preds, tuple):
                preds, _ = preds
            align_loss = get_text_align_loss(few_shot_aux, patch_tokens)

            multilabel_episode = is_multilabel_episode(cfg, labels, meta)
            if multilabel_episode:
                base_support_query_dict = support_query_split_multilabel(
                    patch_tokens, labels, meta)
                if few_shot_aux_has_support_tokens(few_shot_aux):
                    patch_support_query_dict = support_query_split_multilabel_conditioned(
                        base_support_query_dict,
                        few_shot_aux,
                    )
                else:
                    patch_support_query_dict = base_support_query_dict
            else:
                patch_support_query_dict = support_query_split(patch_tokens, labels, meta)
            q2s_labels = patch_support_query_dict['query_batch_labels']
            if multilabel_episode and few_shot_aux_has_query_partial_logits(few_shot_aux):
                patch_q2s_logits = few_shot_aux["query_partial_q2s_logits"]
            else:
                patch_q2s_logits = process_patch_tokens(
                                            cfg,
                                            patch_support_query_dict['support_preds'],
                                            patch_support_query_dict['query_preds'])
            if multilabel_episode:
                q2s_loss = F.binary_cross_entropy_with_logits(
                    patch_q2s_logits, q2s_labels.float())
                few_shot_top1_acc = multilabel_top1_accuracy(patch_q2s_logits, q2s_labels)
            else:
                q2s_loss = F.cross_entropy(patch_q2s_logits, q2s_labels)

                # Explicitly declare reduction to mean.
                few_shotk_correct = metrics.topks_correct(patch_q2s_logits,
                                                            q2s_labels, (1, 5))
                few_shot_top1_acc, _ = [
                    (x / patch_q2s_logits.size(0)) * 100.0 for x in few_shotk_correct
                ]
        if multilabel_episode:
            patch_q2s_logits = torch.nan_to_num(
                patch_q2s_logits,
                nan=0.0,
                posinf=30.0,
                neginf=-30.0,
            )
            update_ap_storage(
                ap_storage,
                patch_q2s_logits,
                q2s_labels,
                patch_support_query_dict['episode_class_ids'],
            )

            query_mask = patch_support_query_dict['query_condition'].detach().cpu().numpy()
            video_names = np.array(meta['video_name'])[query_mask]
            episode_class_ids = (
                patch_support_query_dict['episode_class_ids']
                .detach()
                .long()
                .cpu()
                .numpy()
                .tolist()
            )
            scores = patch_q2s_logits.detach().float().cpu().numpy()
            targets = q2s_labels.detach().float().cpu().numpy()
            query_null_rows = None
            query_matchability_rows = None
            if (
                isinstance(few_shot_aux, dict)
                and "query_null_weights" in few_shot_aux
            ):
                frame_null = (
                    few_shot_aux["query_null_weights"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                expected_shape = (*targets.shape, frame_null.shape[-1])
                if frame_null.shape != expected_shape:
                    raise ValueError(
                        "query_null_weights must align with q2s labels; got "
                        f"{frame_null.shape}, expected {expected_shape}."
                    )
                diag_similarity = (
                    few_shot_aux["query_partial_diag_similarity"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                null_score = float(
                    few_shot_aux["query_null_score"].detach().float().item()
                )
                query_null_rows = {
                    "mean": frame_null.mean(axis=-1),
                    "min": frame_null.min(axis=-1),
                    "max": frame_null.max(axis=-1),
                    "frame_fraction_gt_0p5": (frame_null > 0.5).mean(axis=-1),
                    "diag_similarity": diag_similarity,
                    "score": null_score,
                    "support_mean_abs_cosine": float(
                        few_shot_aux[
                            "query_null_support_mean_abs_cosine"
                        ].detach().float().item()
                    ),
                    "support_max_abs_cosine": float(
                        few_shot_aux[
                            "query_null_support_max_abs_cosine"
                        ].detach().float().item()
                    ),
                }
                if bool(getattr(
                    cfg.FEW_SHOT.QUERY_NULL_ROUTE,
                    "CARDINALITY_CORRECTION",
                    True,
                )):
                    clipped_null = np.clip(frame_null, 1e-6, 1.0 - 1e-6)
                    frame_evidence_lme = null_score - float(
                        cfg.FEW_SHOT.POT_ROUTE.FRAME_SOFTMAX_TAU
                    ) * np.log(clipped_null / (1.0 - clipped_null))
                    query_null_rows.update({
                        "evidence_lme_mean": frame_evidence_lme.mean(axis=-1),
                        "evidence_lme_min": frame_evidence_lme.min(axis=-1),
                        "evidence_lme_max": frame_evidence_lme.max(axis=-1),
                    })
            if (
                isinstance(few_shot_aux, dict)
                and "query_class_matchability" in few_shot_aux
            ):
                def _aux_numpy(key, dtype=torch.float32):
                    value = few_shot_aux.get(key)
                    if value is None:
                        return None
                    value = value.detach().cpu()
                    if dtype is torch.bool:
                        return value.bool().numpy()
                    return value.float().numpy()

                query_matchability_rows = {
                    "base_score": _aux_numpy(
                        "query_partial_q2s_base_logits"
                    ),
                    "temporal_score": _aux_numpy(
                        "query_partial_q2s_temporal_logits"
                    ),
                    "matchability": _aux_numpy(
                        "query_class_matchability"
                    ),
                    "evidence": _aux_numpy("query_class_evidence"),
                    "threshold": _aux_numpy("query_class_threshold"),
                    "support_positive": _aux_numpy(
                        "support_positive_evidence_mean"
                    ),
                    "support_negative": _aux_numpy(
                        "support_negative_evidence_mean"
                    ),
                    "support_gap": _aux_numpy("support_evidence_gap"),
                    "support_reliable": _aux_numpy(
                        "support_calibration_reliable", dtype=torch.bool
                    ),
                    "confuser_available": _aux_numpy(
                        "query_class_confuser_available", dtype=torch.bool
                    ),
                    "positive_similarity": _aux_numpy(
                        "query_class_positive_similarity"
                    ),
                    "hardest_confuser_similarity": _aux_numpy(
                        "query_class_hardest_confuser_similarity"
                    ),
                    "relative_margin": _aux_numpy(
                        "query_class_relative_margin"
                    ),
                    "hardest_confuser_index": _aux_numpy(
                        "query_class_hardest_confuser_support_index",
                        dtype=torch.long,
                    ),
                    "confuser_valid_count": _aux_numpy(
                        "query_class_confuser_valid_count",
                        dtype=torch.long,
                    ),
                    "frame_positive_evidence": _aux_numpy(
                        "query_frame_positive_evidence_mean"
                    ),
                    "frame_confuser_evidence": _aux_numpy(
                        "query_frame_confuser_evidence_mean"
                    ),
                    "frame_margin": _aux_numpy(
                        "query_frame_relative_margin_mean"
                    ),
                    "frame_matchability": _aux_numpy(
                        "query_frame_matchability_mean"
                    ),
                    "frame_effective_patches": _aux_numpy(
                        "query_evidence_effective_patches_mean"
                    ),
                    "frame_top1_mass": _aux_numpy(
                        "query_evidence_top1_mass_mean"
                    ),
                    "frame_temporal_delta": _aux_numpy(
                        "query_frame_temporal_logit_delta"
                    ),
                    "frame_max_t_switch": _aux_numpy(
                        "query_frame_max_t_switch_fraction"
                    ),
                    "absolute_evidence": _aux_numpy(
                        "query_frame_absolute_evidence_mean"
                    ),
                    "transport_patch_mass": _aux_numpy(
                        "query_frame_patch_mass_mean"
                    ),
                    "transport_unmatched_mass": _aux_numpy(
                        "query_frame_unmatched_mass_mean"
                    ),
                    "transport_logit_delta": _aux_numpy(
                        "query_frame_transport_logit_delta"
                    ),
                    "absolute_threshold": _aux_numpy(
                        "query_absolute_mass_threshold"
                    ),
                    "support_absolute_positive": _aux_numpy(
                        "support_absolute_positive_evidence_mean"
                    ),
                    "support_absolute_negative": _aux_numpy(
                        "support_absolute_negative_evidence_mean"
                    ),
                    "support_absolute_gap": _aux_numpy(
                        "support_absolute_evidence_gap"
                    ),
                    "support_absolute_reliable": _aux_numpy(
                        "support_absolute_calibration_reliable",
                        dtype=torch.bool,
                    ),
                }
                for name in (
                    "base_score",
                    "temporal_score",
                    "matchability",
                    "evidence",
                    "positive_similarity",
                    "hardest_confuser_similarity",
                    "relative_margin",
                    "hardest_confuser_index",
                    "frame_positive_evidence",
                    "frame_confuser_evidence",
                    "frame_margin",
                    "frame_matchability",
                    "frame_effective_patches",
                    "frame_top1_mass",
                    "frame_temporal_delta",
                    "frame_max_t_switch",
                    "absolute_evidence",
                    "transport_patch_mass",
                    "transport_unmatched_mass",
                    "transport_logit_delta",
                ):
                    if query_matchability_rows[name] is None:
                        continue
                    if query_matchability_rows[name].shape != targets.shape:
                        raise ValueError(
                            f"{name} must align with q2s labels; got "
                            f"{query_matchability_rows[name].shape}, expected "
                            f"{targets.shape}."
                        )
                for name in (
                    "threshold",
                    "support_positive",
                    "support_negative",
                    "support_gap",
                    "support_reliable",
                    "confuser_available",
                    "confuser_valid_count",
                    "absolute_threshold",
                    "support_absolute_positive",
                    "support_absolute_negative",
                    "support_absolute_gap",
                    "support_absolute_reliable",
                ):
                    if query_matchability_rows[name] is not None and (
                        query_matchability_rows[name].shape[-1:]
                        != targets.shape[-1:]
                    ):
                        raise ValueError(
                            f"{name} must align with the episode class axis; got "
                            f"{query_matchability_rows[name].shape}, expected "
                            f"[..., {targets.shape[-1]}]."
                        )
            for query_idx, video_name in enumerate(video_names):
                for episode_idx, class_id in enumerate(episode_class_ids):
                    row = {
                        'video_name': video_name,
                        'class_id': int(class_id),
                        'score': float(scores[query_idx, episode_idx]),
                        'label': float(targets[query_idx, episode_idx]),
                    }
                    if query_null_rows is not None:
                        row.update({
                            "query_null_mean": float(
                                query_null_rows["mean"][query_idx, episode_idx]
                            ),
                            "query_null_min": float(
                                query_null_rows["min"][query_idx, episode_idx]
                            ),
                            "query_null_max": float(
                                query_null_rows["max"][query_idx, episode_idx]
                            ),
                            "query_null_frame_fraction_gt_0p5": float(
                                query_null_rows[
                                    "frame_fraction_gt_0p5"
                                ][query_idx, episode_idx]
                            ),
                            "query_null_score": query_null_rows["score"],
                            "query_null_diag_similarity": float(
                                query_null_rows[
                                    "diag_similarity"
                                ][query_idx, episode_idx]
                            ),
                            "query_null_support_mean_abs_cosine": (
                                query_null_rows["support_mean_abs_cosine"]
                            ),
                            "query_null_support_max_abs_cosine": (
                                query_null_rows["support_max_abs_cosine"]
                            ),
                        })
                        if "evidence_lme_mean" in query_null_rows:
                            row.update({
                                "query_null_evidence_lme_mean": float(
                                    query_null_rows[
                                        "evidence_lme_mean"
                                    ][query_idx, episode_idx]
                                ),
                                "query_null_evidence_lme_min": float(
                                    query_null_rows[
                                        "evidence_lme_min"
                                    ][query_idx, episode_idx]
                                ),
                                "query_null_evidence_lme_max": float(
                                    query_null_rows[
                                        "evidence_lme_max"
                                    ][query_idx, episode_idx]
                                ),
                            })
                    if (
                        query_matchability_rows is not None
                        and query_matchability_rows["matchability"].shape
                        == targets.shape
                    ):
                        row.update({
                            "query_class_base_score": float(
                                query_matchability_rows["base_score"][
                                    query_idx, episode_idx
                                ]
                            ),
                            "query_class_matchability": float(
                                query_matchability_rows["matchability"][
                                    query_idx, episode_idx
                                ]
                            ),
                            "query_class_evidence": float(
                                query_matchability_rows["evidence"][
                                    query_idx, episode_idx
                                ]
                            ),
                        })
                        optional_pair_fields = {
                            "query_class_temporal_score": "temporal_score",
                            "query_class_positive_similarity": (
                                "positive_similarity"
                            ),
                            "query_class_hardest_confuser_similarity": (
                                "hardest_confuser_similarity"
                            ),
                            "query_class_relative_margin": "relative_margin",
                            "query_class_hardest_confuser_support_index": (
                                "hardest_confuser_index"
                            ),
                            "query_frame_positive_evidence": (
                                "frame_positive_evidence"
                            ),
                            "query_frame_confuser_evidence": (
                                "frame_confuser_evidence"
                            ),
                            "query_frame_relative_margin": "frame_margin",
                            "query_frame_matchability": "frame_matchability",
                            "query_evidence_effective_patches": (
                                "frame_effective_patches"
                            ),
                            "query_evidence_top1_mass": "frame_top1_mass",
                            "query_frame_temporal_logit_delta": (
                                "frame_temporal_delta"
                            ),
                            "query_frame_max_t_switch_fraction": (
                                "frame_max_t_switch"
                            ),
                            "query_frame_absolute_evidence": (
                                "absolute_evidence"
                            ),
                            "query_frame_patch_mass": "transport_patch_mass",
                            "query_frame_unmatched_mass": (
                                "transport_unmatched_mass"
                            ),
                            "query_frame_transport_logit_delta": (
                                "transport_logit_delta"
                            ),
                        }
                        for output_name, source_name in optional_pair_fields.items():
                            values = query_matchability_rows[source_name]
                            if values is not None:
                                row[output_name] = float(
                                    values[query_idx, episode_idx]
                                )
                        optional_class_fields = {
                            "query_class_threshold": "threshold",
                            "support_positive_evidence_mean": "support_positive",
                            "support_negative_evidence_mean": "support_negative",
                            "support_evidence_gap": "support_gap",
                            "support_calibration_reliable": "support_reliable",
                            "query_class_confuser_available": "confuser_available",
                            "query_class_confuser_valid_count": "confuser_valid_count",
                            "query_absolute_mass_threshold": "absolute_threshold",
                            "support_absolute_positive_evidence_mean": (
                                "support_absolute_positive"
                            ),
                            "support_absolute_negative_evidence_mean": (
                                "support_absolute_negative"
                            ),
                            "support_absolute_evidence_gap": (
                                "support_absolute_gap"
                            ),
                            "support_absolute_calibration_reliable": (
                                "support_absolute_reliable"
                            ),
                        }
                        for output_name, source_name in optional_class_fields.items():
                            values = query_matchability_rows[source_name]
                            if values is None:
                                continue
                            value = values[episode_idx]
                            row[output_name] = (
                                bool(value)
                                if source_name in {
                                    "support_reliable",
                                    "support_absolute_reliable",
                                }
                                else float(value)
                            )
                    all_df.append(row)
        if cfg['wandb']:
            cfg['wandb'].log({
                'iteration': cur_iter,
                'iter_top_1_acc': few_shot_top1_acc.item(),
            })

        if cfg.NUM_GPUS > 1:
            few_shot_top1_acc, q2s_loss, align_loss = du.all_reduce(
                [few_shot_top1_acc, q2s_loss, align_loss]
            )

        # Copy the errors from GPU to CPU (sync point).
        few_shot_top1_acc = few_shot_top1_acc.item()
        q2s_loss = q2s_loss.item()
        align_loss = align_loss.item()
        epoch_q2s_loss.append(q2s_loss)
        epoch_align_loss.append(align_loss)
        epoch_top_1_acc_few_shot.append(few_shot_top1_acc)

        if not multilabel_episode:
            support_labels = patch_support_query_dict['support_labels']
            query_labels = patch_support_query_dict['query_labels']
            # pylint: disable=unbalanced-tuple-unpacking
            if cfg.NUM_GPUS > 1:
                patch_q2s_logits, support_labels, query_labels = du.all_gather(
                    [patch_q2s_logits, support_labels, query_labels]
                )
            patch_q2s_logits = patch_q2s_logits.cpu().numpy()
            support_labels = support_labels.cpu().numpy()
            query_labels = query_labels.cpu().numpy()
            pred_query_batch_labels = patch_q2s_logits.argmax(axis=1)
            pred_query_labels = support_labels[pred_query_batch_labels]
            confusion_matrix[query_labels, pred_query_labels] += 1
            batch_df = pd.DataFrame({'y_true':query_labels, 'y_preds':pred_query_labels})
            all_df.append(batch_df)

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            q2s_loss,
            few_shot_top1_acc,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            extra_metrices={"align_loss": align_loss},
        )


        val_meter.update_predictions(preds, labels)
        if should_log_iter_stats(progress_bar):
            val_meter.log_iter_stats(cur_epoch, cur_iter)
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    shot_acc_name: f"{few_shot_top1_acc:.2f}",
                    "q2s_loss": f"{q2s_loss:.3f}",
                    "align_loss": f"{align_loss:.3f}",
                },
                refresh=False,
            )
        val_meter.iter_tic()

    # Log epoch stats.
    if progress_bar is not None:
        progress_bar.close()
    val_meter.log_epoch_stats(cur_epoch)
    log_dict = {
        'test_q2s_loss': mean_or_nan(epoch_q2s_loss),
        'test_align_loss': mean_or_nan(epoch_align_loss),
        'test_top1_acc_few_shot': mean_or_nan(epoch_top_1_acc_few_shot),
        'epoch': cur_epoch}
    if multi_label:
        ap_storage = merge_ap_storage(ap_storage)
        base_map, novel_map, hm_map, class_aps = compute_base_novel_hm(ap_storage, cfg)
        logger.info(
            "test base mAP: %.2f, novel mAP: %.2f, hm mAP: %.2f",
            base_map,
            novel_map,
            hm_map,
        )
        for class_id, class_ap in sorted(class_aps.items()):
            logger.info("test class %s AP: %.2f", class_id, class_ap * 100.0)
        log_dict.update({
            'test_base_map': base_map,
            'test_novel_map': novel_map,
            'test_hm_map': hm_map,
        })
        if du.is_master_proc():
            append_test_history(
                cfg.OUTPUT_DIR,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "test_log_file": cfg.get("test_log_file", ""),
                    "checkpoint_path": cfg.TEST.CHECKPOINT_FILE_PATH,
                    "epoch": int(cur_epoch),
                    "base_map": float(base_map),
                    "novel_map": float(novel_map),
                    "hm_map": float(hm_map),
                    "class_aps": {
                        str(class_id): float(class_ap * 100.0)
                        for class_id, class_ap in sorted(class_aps.items())
                    },
                },
            )
    if cfg['wandb']:
        cfg['wandb'].log(log_dict)
    if multi_label:
        if cfg.NUM_GPUS > 1:
            gathered_rows = du.all_gather_unaligned(all_df)
            all_df = [row for rank_rows in gathered_rows for row in rank_rows]
        all_df = pd.DataFrame(all_df)
    else:
        all_df = pd.concat(all_df)
    all_df.to_csv(os.path.join(cfg.OUTPUT_DIR,cfg['csv_dump_name']))

    val_meter.reset()

# pylint: disable=redefined-outer-name
def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i, _ in enumerate(inputs):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)



def test_few_shot(cfg, args, wandb_run=None):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
    """
    # Set up environment.
    if not args.new_dist_init:
        du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    test_log_file = ""
    test_log_handler = None
    if du.is_master_proc():
        test_log_file, test_log_handler = attach_test_log_file(cfg.OUTPUT_DIR)
        logger.info("Dedicated test log file: %s", test_log_file)
    cfg["test_log_file"] = test_log_file

    if wandb_run is not None:
        wandb_instance = wandb_run
        wandb_instance.define_metric("test*", step_metric="epoch")
        wandb_instance.define_metric("test_top1_acc_few_shot", summary="max")
        if cfg.DATA.MULTI_LABEL:
            wandb_instance.define_metric("test_base_map", summary="max")
            wandb_instance.define_metric("test_novel_map", summary="max")
            wandb_instance.define_metric("test_hm_map", summary="max")
    else:
        if du.get_rank() == 0 and wandb is not None:
            wandb_config_dict = wandb_init_dict(cfg)
            wandb_instance = wandb.init(project=cfg.WANDB.PROJECT,config=wandb_config_dict,
                                        entity=cfg.WANDB.ENTITY)
            wandb_instance.define_metric("epoch")
            wandb_instance.define_metric("iteration")

            wandb_instance.define_metric("iter*", step_metric="iteration")

            wandb_instance.define_metric("train*", step_metric="epoch")
            wandb_instance.define_metric("val*", step_metric="epoch")
            wandb_instance.define_metric("test*", step_metric="epoch")

            wandb_instance.define_metric("train_loss", summary="min")
            wandb_instance.define_metric("val_loss", summary="min")
            wandb_instance.define_metric("test_loss", summary="min")
            wandb_instance.define_metric("val_top5_acc", summary="max")
            wandb_instance.define_metric("val_top1_acc", summary="max")
            wandb_instance.define_metric("test_top1_acc_few_shot", summary="max")
            if cfg.DATA.MULTI_LABEL:
                wandb_instance.define_metric("test_base_map", summary="max")
                wandb_instance.define_metric("test_novel_map", summary="max")
                wandb_instance.define_metric("test_hm_map", summary="max")
        else:
            wandb_instance = None
    cfg['wandb'] = wandb_instance
    cfg['csv_dump_name'] = (
        'multilabel_predictions.csv' if cfg.DATA.MULTI_LABEL else 'confusion_matrix.csv'
    )

    # Init multigrid.
    logger.info("Test with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    cur_epoch = cu.load_test_checkpoint(cfg, model)
    val_loader = loader.construct_loader(cfg, "test")
    val_meter = ValMeter(
        len(val_loader),
        cfg,
        metric1_name="q2s_loss",
        metric2_name=shot_metric_name(cfg),
        metric1_mode="min",
        metric2_mode="max",
        extra_metrics_prefix=None,
    )

    test_epoch(val_loader, model, val_meter, cur_epoch, cfg)
    if test_log_handler is not None:
        pylogging.getLogger().removeHandler(test_log_handler)
        test_log_handler.close()
    # Close wandb logging
    if wandb_instance is not None:
        wandb_instance.finish()

    # Exit
    sys.exit()
