#!/usr/bin/env python3

"""Train a few shot video classification model."""
import os
import pprint
import sys
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from fvcore.common.config import CfgNode
from few_shot_multilabel import (
    compute_base_novel_hm,
    compute_query_partial_q2s_loss,
    empty_ap_storage,
    episode_labels_from_global,
    few_shot_aux_has_query_partial_logits,
    few_shot_aux_has_support_tokens,
    get_query_null_orthogonal_loss,
    get_query_matchability_metrics,
    get_query_null_route_metrics,
    get_text_align_loss,
    get_episode_class_ids,
    is_multilabel_episode,
    mean_or_nan,
    merge_ap_storage,
    multilabel_classification_loss,
    multilabel_top1_accuracy,
    q2s_cos_sim_fp32,
    support_query_split_multilabel,
    support_query_split_multilabel_conditioned,
    update_ap_storage,
)
import trokens.models.losses as losses
import trokens.models.optimizer as optim
import trokens.utils.checkpoint as cu
import trokens.utils.distributed as du
import trokens.utils.logging as logging
import trokens.utils.metrics as metrics
import trokens.utils.misc as misc
from trokens.datasets import loader
from trokens.datasets.mixup import MixUp
from trokens.models import build_model
from trokens.utils.meters import EpochTimer, TrainMeter, ValMeter
from trokens.utils.multigrid import MultigridSchedule

try:
    import wandb
except ImportError:
    wandb = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

warnings.filterwarnings('ignore')


def autocast_context(enabled):
    """Create a CUDA autocast context that works across torch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled):
    """Create a GradScaler that works across torch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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

logger = logging.get_logger(__name__)

def count_parameters(model):
    """Count the number of parameters in the model.

    Args:
        model (torch.nn.Module): The model

    """
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'dino' in name or 'clip' in name or 'resnet' in name:
                print(name)
                continue
            count += param.numel()
    return count


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
    """Calculates the cosine similarity between the last dimension of two tensors.
    """
    return q2s_cos_sim_fp32(x, y, epsilon=epsilon)


def support_query_split(preds, labels, metadata):
    """Split the preds and labels into support and query
    Args:
        preds (torch.Tensor): The predictions
        labels (torch.Tensor): The labels
        metadata (dict): The metadata
    Returns:
        dict: The dictionary containing the support and query labels and preds
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

def conv_fp16(inputs):
    """Convert to float 16
    """
    return np.float16(np.around(inputs, 4))



def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    wandb_run=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        wandb_run (wandb.run): wandb run object
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    epoch_top_1_err = []
    epoch_top_5_err = []
    epoch_top_1_acc_few_shot = []
    epoch_cls_loss = []
    epoch_q2s_loss = []
    epoch_align_loss = []
    epoch_null_ortho_loss = []
    epoch_query_null_metrics = {}

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=misc.get_num_classes(cfg)
        )
    lr = optim.get_epoch_lr(cur_epoch, cfg)
    optim.set_lr(optimizer, lr, log=True)
    shot_acc_name = shot_metric_name(cfg)
    progress_bar = create_progress_bar(
        data_size,
        f"Train {cur_epoch + 1}/{cfg.SOLVER.MAX_EPOCH} ({data_size} episodes)",
    )
    for cur_iter, (inputs, labels, _vid_idx, meta) in enumerate(train_loader):
        if cur_iter > len(train_loader):
            break
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])
        # Update the learning rate.


        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples


        query_null_metrics = {}
        with autocast_context(cfg.TRAIN.MIXED_PRECISION):
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
            input_dict = {'video':inputs, 'metadata':meta}

            model_out = model(input_dict)
            if isinstance(model_out, tuple) and len(model_out) == 3:
                preds, patch_tokens, few_shot_aux = model_out
            else:
                preds, patch_tokens = model_out
                few_shot_aux = None

            if isinstance(preds, tuple):
                preds, _ = preds
            preds = preds / cfg.SOLVER.TEMPRATURE
            preds = torch.nan_to_num(preds, nan=0.0, posinf=30.0, neginf=-30.0)
            align_loss = get_text_align_loss(few_shot_aux, patch_tokens)
            null_ortho_loss = get_query_null_orthogonal_loss(
                few_shot_aux,
                patch_tokens,
            )

            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg)(
                reduction="mean"
            )

            multilabel_episode = is_multilabel_episode(cfg, labels, meta)
            if multilabel_episode:
                classfication_loss = multilabel_classification_loss(
                    preds, labels, cfg, loss_fun)
            else:
                classfication_loss = loss_fun(preds, labels)
            loss_dict = {'classfication_loss':classfication_loss}
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
                (
                    q2s_loss,
                    patch_q2s_logits,
                    q2s_objective_metrics,
                ) = compute_query_partial_q2s_loss(
                    patch_q2s_logits,
                    q2s_labels,
                    few_shot_aux,
                    cfg,
                )
                query_null_metrics = get_query_null_route_metrics(
                    few_shot_aux,
                    q2s_labels,
                )
                query_null_metrics.update(
                    get_query_matchability_metrics(
                        few_shot_aux,
                        q2s_labels,
                    )
                )
                query_null_metrics.update(q2s_objective_metrics)
            else:
                patch_q2s_logits = patch_q2s_logits / cfg.SOLVER.TEMPRATURE
                patch_q2s_logits = torch.nan_to_num(
                    patch_q2s_logits,
                    nan=0.0,
                    posinf=30.0,
                    neginf=-30.0,
                )
                q2s_loss = F.cross_entropy(patch_q2s_logits, q2s_labels)
            loss_dict['q2s_loss'] = q2s_loss
            loss_dict['align_loss'] = align_loss
            loss_dict['null_ortho_loss'] = null_ortho_loss
        loss = (cfg.FEW_SHOT.CLASS_LOSS_LAMBDA * classfication_loss +
                cfg.FEW_SHOT.Q2S_LOSS_LAMBDA * q2s_loss +
                cfg.FEW_SHOT.TEXT_ALIGN.LOSS_WEIGHT * align_loss +
                cfg.FEW_SHOT.QUERY_NULL_ROUTE.ORTHO_WEIGHT * null_ortho_loss)

        if not torch.isfinite(loss):
            finite_report = {
                "preds": bool(torch.isfinite(preds).all()),
                "patch_tokens": bool(torch.isfinite(patch_tokens).all()),
                "support_preds": bool(torch.isfinite(patch_support_query_dict["support_preds"]).all()),
                "query_preds": bool(torch.isfinite(patch_support_query_dict["query_preds"]).all()),
                "patch_q2s_logits": bool(torch.isfinite(patch_q2s_logits).all()),
                "classification_loss": bool(torch.isfinite(classfication_loss)),
                "q2s_loss": bool(torch.isfinite(q2s_loss)),
                "align_loss": bool(torch.isfinite(align_loss)),
                "null_ortho_loss": bool(torch.isfinite(null_ortho_loss)),
            }
            logger.warning(
                "Skip non-finite train batch at epoch %d iter %d: %s",
                cur_epoch + 1,
                cur_iter + 1,
                finite_report,
            )
            optimizer.zero_grad(set_to_none=True)
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({"skip": "nan"}, refresh=False)
            train_meter.iter_tic()
            continue
        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )

        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()
        top1_err, top5_err = None, None
        classification_loss = loss_dict['classfication_loss']
        q2s_loss = loss_dict['q2s_loss']
        align_loss = loss_dict['align_loss']
        null_ortho_loss = loss_dict['null_ortho_loss']
        query_null_metric_values = {}

        if multilabel_episode:
            few_shot_top1_acc = multilabel_top1_accuracy(patch_q2s_logits, q2s_labels)
            if cfg.NUM_GPUS > 1:
                (
                    loss,
                    classification_loss,
                    q2s_loss,
                    align_loss,
                    null_ortho_loss,
                    few_shot_top1_acc,
                ) = du.all_reduce(
                    [
                        loss,
                        classification_loss,
                        q2s_loss,
                        align_loss,
                        null_ortho_loss,
                        few_shot_top1_acc,
                    ]
                )
                if query_null_metrics:
                    reduced_metrics = du.all_reduce(
                        list(query_null_metrics.values())
                    )
                    query_null_metrics = dict(zip(
                        query_null_metrics.keys(),
                        reduced_metrics,
                    ))
            loss = loss.item()
            classification_loss = classification_loss.item()
            q2s_loss = q2s_loss.item()
            align_loss = align_loss.item()
            null_ortho_loss = null_ortho_loss.item()
            few_shot_top1_acc = few_shot_top1_acc.item()
            query_null_metric_values = {
                key: value.item()
                for key, value in query_null_metrics.items()
            }

            epoch_cls_loss.append(classification_loss)
            epoch_q2s_loss.append(q2s_loss)
            epoch_align_loss.append(align_loss)
            epoch_null_ortho_loss.append(null_ortho_loss)
            epoch_top_1_acc_few_shot.append(few_shot_top1_acc)
            for key, value in query_null_metric_values.items():
                epoch_query_null_metrics.setdefault(key, []).append(value)
            global_iter = data_size * cur_epoch + cur_iter
            wandb_iter_dict = {
                'iter_cls_loss': classification_loss,
                'iter_q2s_loss': q2s_loss,
                'iter_align_loss': align_loss,
                'iter_null_ortho_loss': null_ortho_loss,
                'iteration': global_iter,
                'iter_top1_acc_few_shot': few_shot_top1_acc,
            }
            wandb_iter_dict.update({
                f'iter_{key}': value
                for key, value in query_null_metric_values.items()
            })
            if wandb_run:
                wandb_run.log(wandb_iter_dict)

        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            # for few shot
            few_shotk_correct = metrics.topks_correct(patch_q2s_logits, q2s_labels, (1, 5))
            few_shot_top1_acc, _ = [
                (x / patch_q2s_logits.size(0)) * 100.0 for x in few_shotk_correct
            ]

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                (
                    classification_loss,
                    top1_err,
                    top5_err,
                    few_shot_top1_acc,
                    q2s_loss,
                    align_loss,
                    null_ortho_loss,
                    loss,
                ) = du.all_reduce(
                    [
                        classification_loss,
                        top1_err,
                        top5_err,
                        few_shot_top1_acc,
                        q2s_loss,
                        align_loss,
                        null_ortho_loss,
                        loss,
                    ]
                )

            # Copy the stats from GPU to CPU (sync point).
            classification_loss, top1_err, top5_err = (
                classification_loss.item(),
                top1_err.item(),
                top5_err.item(),
            )
            q2s_loss = q2s_loss.item()
            align_loss = align_loss.item()
            null_ortho_loss = null_ortho_loss.item()

            few_shot_top1_acc = few_shot_top1_acc.item()
            loss = loss.item()

            epoch_cls_loss.append(classification_loss)
            epoch_q2s_loss.append(q2s_loss)
            epoch_align_loss.append(align_loss)
            epoch_null_ortho_loss.append(null_ortho_loss)

            epoch_top_1_err.append(top1_err)
            epoch_top_5_err.append(top5_err)
            epoch_top_1_acc_few_shot.append(few_shot_top1_acc)
            global_iter = data_size * cur_epoch + cur_iter
            wandb_iter_dict = {'iter_cls_loss':classification_loss,
                                'iter_q2s_loss':q2s_loss,
                                'iter_align_loss':align_loss,
                                'iter_null_ortho_loss':null_ortho_loss,
                                'iter_top1_err':top1_err,
                                'iter_top5_err':top5_err,
                            'iteration':global_iter,
                            'iter_top1_acc_few_shot':few_shot_top1_acc}
            if wandb_run:
                wandb_run.log(wandb_iter_dict)

        # Update and log stats.
        train_extra_metrics = {
            "q2s_loss": q2s_loss,
            "align_loss": align_loss,
            "null_ortho_loss": null_ortho_loss,
            **query_null_metric_values,
        }
        train_meter.update_stats(
            few_shot_top1_acc,
            None,
            loss,
            lr,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            extra_metrics=train_extra_metrics,
        )
        train_meter.iter_toc()  # measure allreduce for this meter
        if should_log_iter_stats(progress_bar):
            train_meter.log_iter_stats(cur_epoch, cur_iter)
        if progress_bar is not None:
            progress_bar.update(1)
            progress_metrics = {
                shot_acc_name: f"{few_shot_top1_acc:.2f}",
                "q2s_loss": f"{q2s_loss:.3f}",
                "align_loss": f"{align_loss:.3f}",
            }
            if "null_gap" in query_null_metric_values:
                progress_metrics["null_gap"] = (
                    f"{query_null_metric_values['null_gap']:.3f}"
                )
            progress_bar.set_postfix(progress_metrics, refresh=False)
        train_meter.iter_tic()

    # Log epoch stats.
    if progress_bar is not None:
        progress_bar.close()
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    wandb_iter_dict = {
        'train_cls_loss': mean_or_nan(epoch_cls_loss),
        'train_q2s_loss': mean_or_nan(epoch_q2s_loss),
        'train_align_loss': mean_or_nan(epoch_align_loss),
        'train_null_ortho_loss': mean_or_nan(epoch_null_ortho_loss),
        'train_top1_acc_few_shot': mean_or_nan(epoch_top_1_acc_few_shot),
        'epoch': cur_epoch,
    }
    wandb_iter_dict.update({
        f'train_{key}': mean_or_nan(values)
        for key, values in epoch_query_null_metrics.items()
    })
    if epoch_top_1_err:
        wandb_iter_dict.update({
            'train_top1_err': np.mean(epoch_top_1_err),
            'train_top5_err': np.mean(epoch_top_5_err),
            'train_top5_acc': 100 - np.mean(epoch_top_5_err),
            'train_top1_acc': 100 - np.mean(epoch_top_1_err),
        })

    if wandb_run:
        wandb_run.log(wandb_iter_dict)


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_run=None):
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
        wandb_run (wandb.run): wandb run object
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    epoch_top_1_acc_few_shot = []
    epoch_q2s_loss = []
    epoch_align_loss = []
    epoch_null_ortho_loss = []
    epoch_query_null_metrics = {}
    ap_storage = empty_ap_storage(cfg.MODEL.NUM_CLASSES) if cfg.DATA.MULTI_LABEL else None
    shot_acc_name = shot_metric_name(cfg)
    progress_bar = create_progress_bar(
        len(val_loader),
        f"Val {cur_epoch + 1}/{cfg.SOLVER.MAX_EPOCH} ({len(val_loader)} episodes)",
    )

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cur_iter > len(val_loader):
            break
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
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
        query_null_metrics = {}
        with autocast_context(cfg.TRAIN.MIXED_PRECISION):
            input_dict = {'video':inputs, 'metadata':meta}
            model_out = model(input_dict)
            if isinstance(model_out, tuple) and len(model_out) == 3:
                preds, patch_tokens, few_shot_aux = model_out
            else:
                preds, patch_tokens = model_out
                few_shot_aux = None
            if isinstance(preds, tuple):
                preds, _ = preds
            align_loss = get_text_align_loss(few_shot_aux, patch_tokens)
            null_ortho_loss = get_query_null_orthogonal_loss(
                few_shot_aux,
                patch_tokens,
            )

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
                query_null_metrics = get_query_null_route_metrics(
                    few_shot_aux,
                    q2s_labels,
                )
                query_null_metrics.update(
                    get_query_matchability_metrics(
                        few_shot_aux,
                        q2s_labels,
                    )
                )
            else:
                q2s_loss = F.cross_entropy(patch_q2s_logits, q2s_labels)

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


        if cfg.NUM_GPUS > 1:
            (
                few_shot_top1_acc,
                q2s_loss,
                align_loss,
                null_ortho_loss,
            ) = du.all_reduce(
                [
                    few_shot_top1_acc,
                    q2s_loss,
                    align_loss,
                    null_ortho_loss,
                ]
            )
            if query_null_metrics:
                reduced_metrics = du.all_reduce(
                    list(query_null_metrics.values())
                )
                query_null_metrics = dict(zip(
                    query_null_metrics.keys(),
                    reduced_metrics,
                ))

        # Copy the errors from GPU to CPU (sync point).
        few_shot_top1_acc = few_shot_top1_acc.item()
        q2s_loss = q2s_loss.item()
        align_loss = align_loss.item()
        null_ortho_loss = null_ortho_loss.item()
        query_null_metric_values = {
            key: value.item()
            for key, value in query_null_metrics.items()
        }
        epoch_q2s_loss.append(q2s_loss)
        epoch_align_loss.append(align_loss)
        epoch_null_ortho_loss.append(null_ortho_loss)
        epoch_top_1_acc_few_shot.append(few_shot_top1_acc)
        for key, value in query_null_metric_values.items():
            epoch_query_null_metrics.setdefault(key, []).append(value)

        val_meter.iter_toc()
        # Update and log stats.
        val_extra_metrics = {
            "align_loss": align_loss,
            "null_ortho_loss": null_ortho_loss,
            **query_null_metric_values,
        }
        val_meter.update_stats(
            q2s_loss,
            few_shot_top1_acc,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            extra_metrices=val_extra_metrics,
        )
        # write to tensorboard format if available.


        val_meter.update_predictions(preds, labels)
        if should_log_iter_stats(progress_bar):
            val_meter.log_iter_stats(cur_epoch, cur_iter)
        if progress_bar is not None:
            progress_bar.update(1)
            progress_metrics = {
                shot_acc_name: f"{few_shot_top1_acc:.2f}",
                "q2s_loss": f"{q2s_loss:.3f}",
                "align_loss": f"{align_loss:.3f}",
            }
            if "null_gap" in query_null_metric_values:
                progress_metrics["null_gap"] = (
                    f"{query_null_metric_values['null_gap']:.3f}"
                )
            progress_bar.set_postfix(progress_metrics, refresh=False)
        val_meter.iter_tic()

    # Log epoch stats.
    if progress_bar is not None:
        progress_bar.close()
    val_meter.log_epoch_stats(cur_epoch)

    log_dict = {
        'val_q2s_loss': mean_or_nan(epoch_q2s_loss),
        'val_align_loss': mean_or_nan(epoch_align_loss),
        'val_null_ortho_loss': mean_or_nan(epoch_null_ortho_loss),
        'val_top1_acc_few_shot': mean_or_nan(epoch_top_1_acc_few_shot),
        'epoch': cur_epoch}
    log_dict.update({
        f'val_{key}': mean_or_nan(values)
        for key, values in epoch_query_null_metrics.items()
    })
    epoch_mean_acc = mean_or_nan(epoch_top_1_acc_few_shot)
    if cfg.DATA.MULTI_LABEL:
        ap_storage = merge_ap_storage(ap_storage)
        base_map, novel_map, hm_map, _ = compute_base_novel_hm(ap_storage, cfg)
        logger.info(
            "base mAP: %.2f, novel mAP: %.2f, hm mAP: %.2f",
            base_map,
            novel_map,
            hm_map,
        )
        log_dict.update({
            'val_base_map': base_map,
            'val_novel_map': novel_map,
            'val_hm_map': hm_map,
        })
        epoch_mean_acc = hm_map
    if wandb_run:
        wandb_run.log(log_dict)
    val_meter.reset()
    return epoch_mean_acc



def train_few_shot(cfg, args, wandb_run=None):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        args (argparse.Namespace): arguments
        wandb_run (wandb.run): wandb run object
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

    if cfg.TRAIN.CHECKPOINT_ARCHIVE_BEST and cfg.TRAIN.CHECKPOINT_RUN_ID == "":
        cfg.TRAIN.CHECKPOINT_RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if cfg.TRAIN.CHECKPOINT_ARCHIVE_BEST:
        logger.info("Checkpoint run id: %s", cfg.TRAIN.CHECKPOINT_RUN_ID)

    if du.get_rank() == 0 and wandb is not None:
        wandb_config_dict = wandb_init_dict(cfg)
        wandb_config_dict['slurm_id'] = os.environ.get('SLURM_JOB_ID')
        wandb_run = wandb.init(project=cfg.WANDB.PROJECT,config=wandb_config_dict,
                                    entity=cfg.WANDB.ENTITY, name=cfg.WANDB.EXP_NAME)
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("iteration")

        wandb_run.define_metric("iter*", step_metric="iteration")

        wandb_run.define_metric("train*", step_metric="epoch")
        wandb_run.define_metric("val*", step_metric="epoch")
        wandb_run.define_metric("train_loss", summary="min")
        wandb_run.define_metric("val_loss", summary="min")
        wandb_run.define_metric("val_top5_acc", summary="max")
        wandb_run.define_metric("val_top1_acc", summary="max")
    else:
        wandb_run = None


    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.NUM_GPUS>1:
        cfg['num_patches'] = model.module.num_patches
    else:
        cfg['num_patches'] = model.num_patches

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = make_grad_scaler(cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )
    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    # Keep the original training-time evaluation behavior on the test split.
    val_loader = loader.construct_loader(cfg, "test", less_iters=True)
    if cfg.LOG_MODEL_INFO and du.is_master_proc():
        model_info = misc.log_model_info(model, cfg, train_loader)
        # log in wandb as a summary
        if wandb_run is not None:
            wandb_run.summary.update(model_info)

    # Create meters.


    train_meter = TrainMeter(
        len(train_loader),
        cfg,
        metric1_name=shot_metric_name(cfg),
        metric2_name=None,
        extra_metrics_prefix=None,
    )
    val_meter = ValMeter(
        len(val_loader),
        cfg,
        metric1_name="q2s_loss",
        metric2_name=shot_metric_name(cfg),
        metric1_mode="min",
        metric2_mode="max",
        extra_metrics_prefix=None,
    )



    # Perform the training loop.
    logger.info("Start epoch: %s", start_epoch + 1)

    epoch_timer = EpochTimer()
    best_val_acc = 0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Keep per-rank episode counts fixed while changing training episodes.
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        if not cfg.TRAIN.VAL_ONLY:
            train_epoch(
                train_loader,
                model,
                optimizer,
                scaler,
                train_meter,
                cur_epoch,
                cfg,
                wandb_run
            )
        epoch_timer.epoch_toc()
        logger.info(
            "Epoch %s takes %.2fs. Epochs "
            "from %s to %s take "
            "%.2fs in average and "
            "%.2fs in median.",
            cur_epoch, epoch_timer.last_epoch_time(),
            start_epoch, cur_epoch,
            epoch_timer.avg_epoch_time(),
            epoch_timer.median_epoch_time()
        )
        logger.info(
            "For epoch %s, each iteraction takes "
            "%.2fs in average. "
            "From epoch %s to %s, each iteraction takes "
            "%.2fs in average.",
            cur_epoch, epoch_timer.last_epoch_time()/len(train_loader),
            start_epoch, cur_epoch,
            epoch_timer.avg_epoch_time()/len(train_loader)
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )


        # Save a checkpoint.
        cfg_to_save = cfg.clone()
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg_to_save,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.1
        if is_eval_epoch:
            val_acc = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_run)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    cur_epoch,
                    cfg_to_save,
                    scaler if cfg.TRAIN.MIXED_PRECISION else None,
                    best=True
                )
        if cfg.TRAIN.VAL_ONLY:
            break


    return wandb_run
