#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions that handle saving and loading of checkpoints."""

import copy
import os
import pickle
import shutil
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch

import trokens.utils.distributed as du
import trokens.utils.logging as logging
from trokens.utils.c2_model_loading import get_name_convert_func
from trokens.utils.env import checkpoint_pathmgr as pathmgr

logger = logging.get_logger(__name__)


_CHECKPOINT_EXCLUDE_PREFIXES = (
    "dinotxt_visual_model.",
    "dinotxt_text_model.",
)

_BEST_CHECKPOINT_ARCHIVE_PATHS = {}
_PROTECTED_BEST_CHECKPOINT_JOBS = set()


def _filter_checkpoint_state_dict(state_dict):
    """Drop frozen external backbones that are reloaded from their source weights."""
    return OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if not key.startswith(_CHECKPOINT_EXCLUDE_PREFIXES)
    )


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if du.is_master_proc() and not pathmgr.exists(checkpoint_dir):
        try:
            pathmgr.mkdirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def _train_cfg_value(cfg, key, default):
    train_cfg = getattr(cfg, "TRAIN", None)
    if train_cfg is None:
        return default
    try:
        return train_cfg.get(key, default)
    except AttributeError:
        return getattr(train_cfg, key, default)


def _safe_filename_token(value):
    value = str(value).strip()
    if not value:
        value = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    token = "".join(char if char in allowed else "_" for char in value)
    return token.strip("._-") or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _unique_checkpoint_path(path):
    if not pathmgr.exists(path):
        return path
    root, ext = os.path.splitext(path)
    version = 2
    while True:
        candidate = "{}_v{:02d}{}".format(root, version, ext)
        if not pathmgr.exists(candidate):
            return candidate
        version += 1


def _get_best_archive_dir(path_to_job, cfg):
    archive_dir = str(
        _train_cfg_value(cfg, "CHECKPOINT_ARCHIVE_DIR", "best_by_run")
    ).strip()
    if not archive_dir:
        archive_dir = "best_by_run"
    if os.path.isabs(archive_dir):
        return archive_dir
    return os.path.join(get_checkpoint_dir(path_to_job), archive_dir)


def _copy_checkpoint_file(src_path, dst_path):
    pathmgr.mkdirs(os.path.dirname(dst_path))
    with pathmgr.open(src_path, "rb") as src_file:
        with pathmgr.open(dst_path, "wb") as dst_file:
            shutil.copyfileobj(src_file, dst_file, length=16 * 1024 * 1024)


def _archive_existing_best_checkpoint(path_to_job, cfg, best_path):
    key = os.path.abspath(path_to_job)
    if key in _PROTECTED_BEST_CHECKPOINT_JOBS:
        return
    _PROTECTED_BEST_CHECKPOINT_JOBS.add(key)

    if not pathmgr.exists(best_path):
        return

    try:
        timestamp = datetime.fromtimestamp(os.path.getmtime(best_path)).strftime(
            "%Y%m%d_%H%M%S"
        )
    except OSError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(
        _get_best_archive_dir(path_to_job, cfg),
        "checkpoint_best_previous_{}.pyth".format(timestamp),
    )
    archive_path = _unique_checkpoint_path(archive_path)
    _copy_checkpoint_file(best_path, archive_path)
    logger.info(
        "Preserved existing best checkpoint at %s before overwrite.",
        archive_path,
    )


def _get_run_best_archive_path(path_to_job, cfg):
    key = os.path.abspath(path_to_job)
    if key in _BEST_CHECKPOINT_ARCHIVE_PATHS:
        return _BEST_CHECKPOINT_ARCHIVE_PATHS[key]

    run_id = _safe_filename_token(
        _train_cfg_value(cfg, "CHECKPOINT_RUN_ID", "")
    )
    archive_path = os.path.join(
        _get_best_archive_dir(path_to_job, cfg),
        "checkpoint_best_{}.pyth".format(run_id),
    )
    archive_path = _unique_checkpoint_path(archive_path)
    _BEST_CHECKPOINT_ARCHIVE_PATHS[key] = archive_path
    return archive_path


def _archive_run_best_checkpoint(path_to_job, cfg, best_path):
    if not _train_cfg_value(cfg, "CHECKPOINT_ARCHIVE_BEST", True):
        return
    archive_path = _get_run_best_archive_path(path_to_job, cfg)
    _copy_checkpoint_file(best_path, archive_path)
    logger.info("Archived run best checkpoint to %s.", archive_path)


def get_path_to_checkpoint(path_to_job, epoch, best=False):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    if best:
        name = "checkpoint_best.pyth"
    else:
        name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def get_best_checkpoint(path_to_job):
    """
    Get the checkpoint saved from the best validation metric.
    """
    return os.path.join(get_checkpoint_dir(path_to_job), "checkpoint_best.pyth")


def has_best_checkpoint(path_to_job):
    """
    Determine if the best validation checkpoint exists.
    """
    return pathmgr.exists(get_best_checkpoint(path_to_job))


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = pathmgr.ls(d) if pathmgr.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cfg, cur_epoch, multigrid_schedule=None):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cfg (CfgNode): configs to save.
        cur_epoch (int): current number of epoch of the model.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cfg.TRAIN.VAL_ONLY:
        return False
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(path_to_job, model, optimizer, epoch, cfg, scaler=None, best=False):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
        scaler (GradScaler): the mixed precision scale.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        return
    # Ensure that the checkpoint dir exists.
    pathmgr.mkdirs(get_checkpoint_dir(path_to_job))
    # Omit the DDP wrapper in the multi-gpu setting.

    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    normalized_sd = _filter_checkpoint_state_dict(sub_to_normal_bn(sd))

    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": normalized_sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    # Write the checkpoint.
    if not best:
        path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1)
        previous_checkpoint = get_path_to_checkpoint(path_to_job, epoch)
        if os.path.exists(previous_checkpoint):
            os.remove(previous_checkpoint)
    else:
        path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1, best=True)
        if _train_cfg_value(cfg, "CHECKPOINT_ARCHIVE_BEST", True):
            _archive_existing_best_checkpoint(path_to_job, cfg, path_to_checkpoint)

    with pathmgr.open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    if best:
        _archive_run_best_checkpoint(path_to_job, cfg, path_to_checkpoint)
    return path_to_checkpoint


def inflate_weight(state_dict_2d, state_dict_3d):
    """
    Inflate 2D model weights in state_dict_2d to the 3D model weights in
    state_dict_3d. The details can be found in:
    Joao Carreira, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    Args:
        state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
        state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
    Returns:
        state_dict_inflated (OrderedDict): a dict of inflated parameters.
    """
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info(
                "Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = (
                v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            )
        elif v2d.shape == v3d.shape:
            v3d = v2d
        else:
            logger.info(
                "Unexpected {}: {} -|> {}: {}".format(
                    k, v2d.shape, k, v3d.shape
                )
            )
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def validate_text_concat_checkpoint(model, state_dict, allow_init=False):
    """Guard concat semantics; v1 -> v2 is explicit encoder initialization only."""
    module = getattr(model, 'text_concat_st', None)
    prefix = 'text_concat_st.'
    present = any(name.startswith(prefix) for name in state_dict)
    if module is None:
        if present:
            raise ValueError('Text-concat checkpoint requires TEXT_CONCAT_ST.ENABLE.')
        return False
    signature = state_dict.get(prefix + 'architecture_signature')
    expected_signature = module.architecture_signature
    initializing = (allow_init and signature is not None and signature.shape == (4,)
                    and expected_signature.shape == (6,) and signature[0].item() == 1
                    and expected_signature[0].item() == 2
                    and torch.equal(signature[1:].cpu(), expected_signature[1:4].cpu())
                    and not any(name.startswith('query_evidence_verifier.') for name in state_dict))
    if not initializing and (signature is None or signature.shape != expected_signature.shape
            or not torch.equal(signature.cpu(), expected_signature.cpu())):
        raise ValueError('Text-concat checkpoint architecture/token-count mismatch; start a fresh run.')
    expected = {name: value for name, value in model.named_parameters() if value.requires_grad
                and not (initializing and name.startswith('query_evidence_verifier.'))}
    missing = [name for name, value in expected.items()
               if name not in state_dict or state_dict[name].shape != value.shape]
    if missing:
        raise ValueError('Incomplete Text-concat checkpoint: ' + ', '.join(missing[:6]))
    model._text_concat_st_checkpoint_loaded = not initializing
    if initializing:
        logger.warning('Initializing concat v2 encoder from v1; new evidence heads, epoch/optimizer reset required.')
        state_dict.pop(prefix + 'architecture_signature')
    return initializing


def validate_label_pre_temporal_checkpoint(model, state_dict, allow_init=False):
    """Guard routing/loss semantics; allow explicit compatible initialization."""
    concat_init = validate_text_concat_checkpoint(model, state_dict, allow_init)
    module = getattr(model, "label_pre_temporal", None)
    prefix = "label_pre_temporal."
    present = any(name.startswith(prefix) for name in state_dict)
    if module is None:
        if present:
            raise ValueError("Label-pre-temporal checkpoint requires LABEL_PRE_TEMPORAL.ENABLE.")
        return concat_init
    if not present:
        if allow_init:
            logger.warning("Initializing Label-pre-temporal routing from legacy weights; resetting epoch/optimizer.")
            model._label_pre_temporal_checkpoint_loaded = False
            return True
        raise ValueError("Label-pre-temporal weights missing: legacy initialization requires explicit CHECKPOINT_EPOCH_RESET=True.")
    version_key = prefix + "architecture_version"
    version = state_dict.get(version_key)
    if version is None or version.numel() != 1:
        raise ValueError("Label-pre-temporal checkpoint architecture mismatch.")
    source, destination = version.item(), module.architecture_version.item()
    reinitializing = source != destination
    if reinitializing:
        # v29-v32 reuse early weights for main pooling; v21-v28 used separate
        # Post scores. v13-v20 belonged to other, incompatible experiments.
        if (source not in (*range(1, 13), *range(21, 33))
                or destination not in range(29, 33)):
            raise ValueError("Label-pre-temporal checkpoint architecture mismatch.")
        source_cosine = source in (*range(9, 13), *range(25, 29), 31, 32)
        if source_cosine != (module.scoring == "cosine"):
            # Deliberately no implicit Q/K addition/removal or optimizer
            # migration. This experiment must be trained independently.
            raise ValueError(
                "Label-pre-temporal SCORING mismatch: projected_cosine and cosine "
                "checkpoints cannot resume/evaluate or initialize each other. "
                "Start a fresh run with the intended SCORING configuration.")
        if not allow_init:
            raise ValueError(
                "Label-pre-temporal attention/routing/readout/rho/global classification semantics changed "
                f"(v{source} -> v{destination}). Requires explicit "
                "CHECKPOINT_EPOCH_RESET=True initialization, not resume/evaluation.")
    expected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    expected.update({prefix + name: value for name, value in module.state_dict().items()})
    missing = [name for name, value in expected.items()
               if name not in state_dict or state_dict[name].shape != value.shape]
    if missing:
        raise ValueError("Incomplete Label-pre-temporal checkpoint: " + ", ".join(missing[:6]))
    if reinitializing:
        logger.warning("Initializing Label-pre-temporal v%s from v%s weights; routing, readout, rho or "
                       "classification semantics differ. Resetting epoch, optimizer and scaler.",
                       destination, source)
        # Do not overwrite the destination contract with the source marker.
        # load_checkpoint validates before building its matching-state copy.
        state_dict.pop(version_key)
        model._label_pre_temporal_checkpoint_loaded = False
        return True
    model._label_pre_temporal_checkpoint_loaded = True
    return False


def validate_verifier_checkpoint(model, state_dict, allow_init=False):
    """Reject random/partial learned heads at evaluation or resume.

    An explicit training initialization from a backbone checkpoint may omit the
    entire verifier. That requires resetting epoch and optimizer, not resuming.
    """
    verifier = getattr(model, "query_evidence_verifier", None)
    if verifier is None:
        return False
    prefix = "query_evidence_verifier."
    version_key = prefix + "architecture_version"
    if version_key in state_dict:
        version = state_dict[version_key]
        if version.numel() != 1 or version.item() != verifier.architecture_version.item():
            raise ValueError(
                "Checkpoint evidence architecture version does not match the configured quality/uniqueness heads. "
                "semantic_relation + metric is v3; relation_decision + metric is v5. "
                "Changing heads requires a new training run; incompatible weights cannot be resumed or evaluated."
            )
    expected = {prefix + name: value for name, value in verifier.state_dict().items()}
    missing = [name for name, value in expected.items()
               if name not in state_dict or state_dict[name].shape != value.shape]
    if missing:
        if allow_init and not any(name.startswith(prefix) for name in state_dict):
            logger.warning("Checkpoint has no Query evidence weights: initializing requested heads; resetting epoch/optimizer.")
            model._verifier_checkpoint_loaded = False
            return True
        raise ValueError(
            "Checkpoint lacks complete Query evidence weights. Train the heads first. Missing/mismatched: " + ", ".join(missing[:6])
        )
    model._verifier_checkpoint_loaded = True
    return False


def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    scaler=None,
    inflation=False,
    convert_from_caffe2=False,
    epoch_reset=False,
    clear_name_pattern=(),
    replace_name_pattern=(),
    should_split_qkv=False,
    allow_verifier_init=False,
    allow_label_pre_temporal_init=False,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        scaler (GradScaler): GradScaler to load the mixed precision scale.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
        clear_name_pattern (string): if given, this (sub)string will be cleared
            from a layer name if it can be matched.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert pathmgr.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    logger.info("Loading network weights from {}.".format(path_to_checkpoint))

    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model
    if convert_from_caffe2:
        with pathmgr.open(path_to_checkpoint, "rb") as f:
            caffe2_checkpoint = pickle.load(f, encoding="latin1")
        state_dict = OrderedDict()
        name_convert_func = get_name_convert_func()
        for key in caffe2_checkpoint["blobs"].keys():
            converted_key = name_convert_func(key)
            converted_key = c2_normal_to_sub_bn(converted_key, ms.state_dict())
            if converted_key in ms.state_dict():
                c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
                model_blob_shape = ms.state_dict()[converted_key].shape

                # expand shape dims if they differ (eg for converting linear to conv params)
                if len(c2_blob_shape) < len(model_blob_shape):
                    c2_blob_shape += (1,) * (
                        len(model_blob_shape) - len(c2_blob_shape)
                    )
                    caffe2_checkpoint["blobs"][key] = np.reshape(
                        caffe2_checkpoint["blobs"][key], c2_blob_shape
                    )
                # Load BN stats to Sub-BN.
                if (
                    len(model_blob_shape) == 1
                    and len(c2_blob_shape) == 1
                    and model_blob_shape[0] > c2_blob_shape[0]
                    and model_blob_shape[0] % c2_blob_shape[0] == 0
                ):
                    caffe2_checkpoint["blobs"][key] = np.concatenate(
                        [caffe2_checkpoint["blobs"][key]]
                        * (model_blob_shape[0] // c2_blob_shape[0])
                    )
                    c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

                if c2_blob_shape == tuple(model_blob_shape):
                    state_dict[converted_key] = torch.tensor(
                        caffe2_checkpoint["blobs"][key]
                    ).clone()
                    logger.info(
                        "{}: {} => {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
                else:
                    logger.warn(
                        "!! {}: {} does not match {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
            else:
                if not any(
                    prefix in key for prefix in ["momentum", "lr", "model_iter"]
                ):
                    logger.warn(
                        "!! {}: can not be converted, got {}".format(
                            key, converted_key
                        )
                    )
        diff = set(ms.state_dict()) - set(state_dict)
        diff = {d for d in diff if "num_batches_tracked" not in d}
        if len(diff) > 0:
            logger.warn("Not loaded {}".format(diff))
        if validate_label_pre_temporal_checkpoint(ms, state_dict, allow_label_pre_temporal_init):
            epoch_reset = True
        if validate_verifier_checkpoint(ms, state_dict, allow_verifier_init):
            epoch_reset = True
        ms.load_state_dict(state_dict, strict=False)
        epoch = -1
    else:
        # Load the checkpoint on CPU to avoid GPU mem spike.
        with pathmgr.open(path_to_checkpoint, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        model_state_dict_3d = (
            model.module.state_dict() if data_parallel else model.state_dict()
        )
        checkpoint["model_state"] = normal_to_sub_bn(
            checkpoint["model_state"], model_state_dict_3d
        )
        if inflation:
            # Validate the source: inflate_weight can substitute initialized
            # model tensors for incompatible shapes, hiding an incomplete head.
            if validate_label_pre_temporal_checkpoint(ms, checkpoint["model_state"], allow_label_pre_temporal_init):
                epoch_reset = True
            if validate_verifier_checkpoint(ms, checkpoint["model_state"], allow_verifier_init):
                epoch_reset = True
            # Try to inflate the model.
            inflated_model_dict = inflate_weight(
                checkpoint["model_state"], model_state_dict_3d
            )
            ms.load_state_dict(inflated_model_dict, strict=False)
        else:
            if clear_name_pattern:
                for item in clear_name_pattern:
                    model_state_dict_new = OrderedDict()
                    for k in checkpoint["model_state"]:
                        if item in k:
                            k_re = k.replace(item, "")
                            model_state_dict_new[k_re] = checkpoint[
                                "model_state"
                            ][k]
                            logger.info("renaming: {} -> {}".format(k, k_re))
                        else:
                            model_state_dict_new[k] = checkpoint["model_state"][
                                k
                            ]
                    checkpoint["model_state"] = model_state_dict_new

            pre_train_dict = checkpoint["model_state"]

            should_split_qkv = epoch_reset and should_split_qkv
            if should_split_qkv:
                pre_train_dict = split_qkv(pre_train_dict)
            if len(replace_name_pattern) > 0:
                new_pre_train_dict = {}
                for k,v in pre_train_dict.items():
                    for a,b in replace_name_pattern:
                        if a in k: k = k.replace(a, b)
                    new_pre_train_dict[k] = v
                pre_train_dict = new_pre_train_dict

            if validate_label_pre_temporal_checkpoint(ms, pre_train_dict, allow_label_pre_temporal_init):
                epoch_reset = True
            model_dict = ms.state_dict()
            # Match pre-trained weights that have same shape as current model.
            pre_train_dict_match = {
                k: v
                for k, v in pre_train_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
            # Weights that do not have match from the pre-trained model.
            not_load_layers = [
                k
                for k in model_dict.keys()
                if k not in pre_train_dict_match.keys()
            ]
            # Log weights that are not loaded with the pre-trained weights.
            if not_load_layers:
                preview = ", ".join(not_load_layers[:5])
                if len(not_load_layers) > 5:
                    preview = f"{preview}, ..."
                logger.debug(
                    "Skipped loading %d unmatched network weights. First few: %s",
                    len(not_load_layers),
                    preview,
                )
            # Load pre-trained weights.
            if validate_verifier_checkpoint(ms, pre_train_dict, allow_verifier_init):
                epoch_reset = True
            ms.load_state_dict(pre_train_dict_match, strict=False)
            epoch = -1

            # Load the optimizer state (commonly not done when fine-tuning)
        if "epoch" in checkpoint.keys() and not epoch_reset:
            epoch = checkpoint["epoch"]
            if optimizer:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                except ValueError as err:
                    logger.warning(
                        "Skip optimizer state from checkpoint due to parameter group mismatch: %s",
                        err,
                    )
            if scaler:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state"])
                except (ValueError, KeyError) as err:
                    logger.warning(
                        "Skip scaler state from checkpoint: %s",
                        err,
                    )
        else:
            epoch = -1

    return epoch


def sub_to_normal_bn(sd):
    """
    Convert the Sub-BN paprameters to normal BN parameters in a state dict.
    There are two copies of BN layers in a Sub-BN implementation: `bn.bn` and
    `bn.split_bn`. `bn.split_bn` is used during training and
    "compute_precise_bn". Before saving or evaluation, its stats are copied to
    `bn.bn`. We rename `bn.bn` to `bn` and store it to be consistent with normal
    BN layers.
    Args:
        sd (OrderedDict): a dict of parameters whitch might contain Sub-BN
        parameters.
    Returns:
        new_sd (OrderedDict): a dict with Sub-BN parameters reshaped to
        normal parameters.
    """
    new_sd = copy.deepcopy(sd)
    modifications = [
        ("bn.bn.running_mean", "bn.running_mean"),
        ("bn.bn.running_var", "bn.running_var"),
        ("bn.split_bn.num_batches_tracked", "bn.num_batches_tracked"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    for key in sd:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                new_sd[new_key] = new_sd.pop(key)

        for rm in to_remove:
            if rm in key and key in new_sd:
                del new_sd[key]

    for key in new_sd:
        if key.endswith("bn.weight") or key.endswith("bn.bias"):
            if len(new_sd[key].size()) == 4:
                assert all(d == 1 for d in new_sd[key].size()[1:])
                new_sd[key] = new_sd[key][:, 0, 0, 0]

    return new_sd


def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key


def normal_to_sub_bn(checkpoint_sd, model_sd):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        checkpoint_sd (OrderedDict): source dict of parameters.
        model_sd (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            if "bn.split_bn." in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape
            c2_blob_shape = checkpoint_sd[key].shape

            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = torch.cat(
                    [checkpoint_sd[key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
                logger.info(
                    "{} {} -> {}".format(
                        key, before_shape, checkpoint_sd[key].shape
                    )
                )
    return checkpoint_sd


def load_test_checkpoint(cfg, model):
    """
    Loading checkpoint logic for testing.
    """
    if cfg.TEST.TEST_EPOCH_NUM > 0:
        n = cfg.TEST.TEST_EPOCH_NUM
        name = "checkpoint_epoch_{:05d}.pyth".format(n)
        cfg.TEST.CHECKPOINT_FILE_PATH = os.path.join(cfg.OUTPUT_DIR, f'checkpoints', name)
    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in MODEL_VIS.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TEST.CHECKPOINT_FILE_PATH and test it.
        epoch = load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
            should_split_qkv = cfg.SPLIT_QKV_CHECKPOINT,
        )
    elif has_best_checkpoint(cfg.OUTPUT_DIR):
        best_checkpoint = get_best_checkpoint(cfg.OUTPUT_DIR)
        logger.info(
            "Loading best validation checkpoint for testing: %s",
            best_checkpoint,
        )
        epoch = load_checkpoint(best_checkpoint, model, cfg.NUM_GPUS > 1)
    elif has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        logger.info(
            "Best validation checkpoint not found. Loading last checkpoint for testing: %s",
            last_checkpoint,
        )
        epoch = load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        epoch = load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            should_split_qkv = cfg.SPLIT_QKV_CHECKPOINT,
        )
    else:
        logger.info(
            "Unknown way of loading checkpoint. Using with random initialization, only for debugging."
        )
        unwrapped = model.module if hasattr(model, "module") else model
        validate_label_pre_temporal_checkpoint(unwrapped, {})
        validate_verifier_checkpoint(unwrapped, {})
        epoch = -1
    return epoch


def load_train_checkpoint(cfg, model, optimizer, scaler=None):
    """
    Loading checkpoint logic for training.
    """
    if cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler=scaler,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            replace_name_pattern=cfg.TRAIN.CHECKPOINT_REPLACE_NAME_PATTERN,
            should_split_qkv = cfg.SPLIT_QKV_CHECKPOINT,
            allow_verifier_init=not cfg.TRAIN.VAL_ONLY,
            allow_label_pre_temporal_init=(cfg.TRAIN.CHECKPOINT_EPOCH_RESET and not cfg.TRAIN.VAL_ONLY),
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.AUTO_RESUME and has_checkpoint(cfg.OUTPUT_DIR):
        if cfg.TRAIN.VAL_ONLY and cfg.TEST.TEST_EPOCH_NUM > 0:
            n = cfg.TEST.TEST_EPOCH_NUM
            last_checkpoint = os.path.join(cfg.OUTPUT_DIR, 'checkpoints', f'checkpoint_epoch_{n:05}.pyth')
            logger.info("Load checkpoint epoch {}: {}.".format(n, last_checkpoint))
        else:
            last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
            logger.info("Load from last checkpoint, {}.".format(last_checkpoint))
        checkpoint_epoch = load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer, scaler=scaler
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.NEW_TRAIN:
        logger.info("Start from scratch.")
        start_epoch = 0
    else:
        start_epoch = 0

    if cfg.TRAIN.VAL_ONLY:
        unwrapped = model.module if hasattr(model, "module") else model
        if (getattr(unwrapped, 'text_concat_st', None) is not None
                and not getattr(unwrapped, '_text_concat_st_checkpoint_loaded', False)):
            raise ValueError('VAL_ONLY requires a complete Text-concat checkpoint.')
        if (getattr(unwrapped, "label_pre_temporal", None) is not None
                and not getattr(unwrapped, "_label_pre_temporal_checkpoint_loaded", False)):
            raise ValueError("VAL_ONLY requires a complete Label-pre-temporal checkpoint.")
        if (getattr(unwrapped, "query_evidence_verifier", None) is not None
                and not getattr(unwrapped, "_verifier_checkpoint_loaded", False)):
            raise ValueError("VAL_ONLY requires a trained learned-verifier checkpoint.")
    return start_epoch

def split_qkv(d):
    out = OrderedDict()
    names = ['q', 'k', 'v']
    for k, v in d.items():
        if 'qkv' in k:
            # split
            for a, new_v in zip(names, v.chunk(3,dim=0)):
                new_k = k.replace('qkv', a)
                out[new_k]=new_v
        else:
            out[k] = v
    return out
