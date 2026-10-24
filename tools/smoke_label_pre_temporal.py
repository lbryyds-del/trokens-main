"""Bounded single-/multi-GPU smoke; never save checkpoints or train an epoch.

Uses the actual Pointformer constructor/forward, motion modules, losses and
optimizer. By default, frozen towers/text use synthetic inputs. --real-data
uses locally cached towers and SAV episodes (no downloads). Run from the repo
root, selecting idle GPUs via CUDA_VISIBLE_DEVICES; torchrun enables DDP.
Also supports TEXT_CONCAT_ST v1 (judgments off) and v2 (Raw m / Post rho).
"""

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trokens.config.defaults import get_cfg
from trokens.models.pointformer import Pointformer
from trokens.models.optimizer import construct_optimizer, get_epoch_lr, set_lr
from tools.few_shot_multilabel import (
    compute_query_partial_q2s_loss, get_episode_class_ids, episode_labels_from_global,
    multilabel_classification_loss,
)
from trokens.models.losses import get_loss_func
from trokens.models.query_evidence_verifier import set_verifier_training_progress
from trokens.utils.amp import checked_amp_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="configs/trokens/sav_label_pre_temporal.yaml")
    parser.add_argument("--steps", type=int, default=2, help="Number of successful optimizer updates.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--video-chunk", type=int)
    parser.add_argument("--class-chunk", type=int)
    parser.add_argument("--batch-raw-quality", action="store_true")
    parser.add_argument("--mlp-checkpoint", action="store_true",
                        help="Enable MLP checkpoint, disabling full routing checkpoint.")
    parser.add_argument("--readout-checkpoint", action="store_true",
                        help="Checkpoint text readout, disabling full routing checkpoint.")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable all routing, MLP, readout and verifier activation checkpoints.")
    parser.add_argument("--no-encoder-checkpoint", action="store_true",
                        help="Disable full routing checkpoint; selective MLP/readout flags are independent.")
    parser.add_argument("--no-verifier-checkpoint", action="store_true")
    parser.add_argument("--real-data", action="store_true", help="Use cached frozen towers and real SAV episodes; never download or save checkpoints.")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--max-skipped-updates", type=int, default=4,
                        help="Bound AMP scale calibration; skipped updates never count as successful steps.")
    args = parser.parse_args()
    if args.steps < 1 or args.grad_accum < 1 or args.max_skipped_updates < 0 or not torch.cuda.is_available():
        parser.error("A visible idle CUDA GPU and positive --steps are required.")
    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.autograd.set_detect_anomaly(args.anomaly)
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.NUM_GPUS = world
    cfg.RNG_SEED = args.seed
    cfg.SOLVER.GRAD_ACCUM_STEPS = args.grad_accum
    concat = bool(cfg.FEW_SHOT.TEXT_CONCAT_ST.ENABLE)
    encoder_cfg = cfg.FEW_SHOT.TEXT_CONCAT_ST if concat else cfg.FEW_SHOT.LABEL_PRE_TEMPORAL
    if concat and args.batch_raw_quality:
        parser.error('Text concat has no early quality reader.')
    if args.video_chunk is not None:
        encoder_cfg.VIDEO_CHUNK_SIZE = args.video_chunk
    if args.class_chunk is not None:
        encoder_cfg.CLASS_CHUNK_SIZE = args.class_chunk
    if args.batch_raw_quality:
        cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.BATCH_RAW_QUALITY = True
    if args.no_checkpoint or args.no_encoder_checkpoint:
        encoder_cfg.ACTIVATION_CHECKPOINT = False
    if args.no_checkpoint or args.no_verifier_checkpoint:
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.ACTIVATION_CHECKPOINT = False
    if args.no_checkpoint:
        encoder_cfg.MLP_CHECKPOINT = False
        encoder_cfg.READOUT_CHECKPOINT = False
    if args.mlp_checkpoint:
        encoder_cfg.ACTIVATION_CHECKPOINT = False
        encoder_cfg.MLP_CHECKPOINT = True
    if args.readout_checkpoint:
        encoder_cfg.ACTIVATION_CHECKPOINT = False
        encoder_cfg.READOUT_CHECKPOINT = True
    if not (cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ENABLE or concat):
        parser.error("This smoke requires LABEL_PRE_TEMPORAL or TEXT_CONCAT_ST.")
    if args.real_data:
        probe = Pointformer.__new__(Pointformer)
        for filename in ("dinov2_vitl14_reg4_pretrain.pth",
                         "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
                         "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
                         "bpe_simple_vocab_16e6.txt.gz"):
            if probe._find_torchhub_checkpoint(filename) is None:
                raise FileNotFoundError(f"Required local asset missing (downloads disabled): {filename}")
        if not any((p / "facebookresearch_dinov2_main").is_dir() for p in probe._torchhub_dirs()):
            raise FileNotFoundError("Local DINOv2 hub source is required (downloads disabled).")
        model = Pointformer(cfg).cuda().train()
    else:
        with patch.object(Pointformer, "_load_dinotxt_visual_model", return_value=nn.Identity()), \
             patch.object(Pointformer, "_load_dinotxt_text_model", return_value=nn.Identity()), \
             patch.object(Pointformer, "_load_dinotxt_tokenizer", return_value=None), \
             patch.object(Pointformer, "_load_atomic_label_names", return_value=[]), \
             patch.object(Pointformer, "_load_label_prompt_groups", return_value=[]):
            model = Pointformer(cfg).cuda().train()
    k = cfg.FEW_SHOT.N_WAY
    s = k * cfg.FEW_SHOT.K_SHOT
    q = k * cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS
    b, t, n, d = s + q, cfg.DATA.NUM_FRAMES, cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE, model.embed_dim
    text = F.normalize(torch.randn(k, d, device="cuda"), dim=-1)
    if not args.real_data:
        model._get_pot_label_text_features = lambda ids, dtype: text[ids].to(dtype)
    # Shared model/text, but distinct per-rank inputs exercise gradient sync.
    torch.manual_seed(args.seed + rank)
    labels = F.one_hot(torch.arange(b, device="cuda") % k, k).float()
    global_labels = F.pad(labels, (0, cfg.MODEL.NUM_CLASSES - k))
    mask = torch.ones(b, t, n, device="cuda", dtype=torch.bool)
    visibility = mask.clone()
    visibility[s, 1] = False
    metadata = dict(
        support_mask=torch.arange(b, device="cuda") < s,
        episode_class_ids=torch.arange(k, device="cuda"),
        episode_positive_labels=labels,
        pred_query_mask=mask, pred_visibility=visibility,
        pred_tracks=torch.rand(b, t, n, 2, device="cuda") * 2 - 1,
        hod_feat=torch.randn(b, n, t, cfg.POINT_INFO.HOD.NUM_BINS, device="cuda"),
    )
    inputs = dict(video=torch.empty(b, t, 3, 1, 1, device="cuda"),
                  metadata=metadata, skip_feat_extractor=True)
    core = model
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    real_loader = None
    if args.real_data:
        from trokens.datasets.loader import construct_loader
        real_loader = iter(construct_loader(cfg, "train"))
    optimizer = construct_optimizer(model, cfg)
    # Match the actual epoch-0 learning-rate warmup as well as verifier warmup.
    set_lr(optimizer, get_epoch_lr(0, cfg))
    dtype = getattr(torch, args.dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    required = ("blocks.0.temporal_attn.qkv.weight", "blocks.0.attn.qkv.weight",
                "hod_motion_module.fc2.c_fc.weight", "cross_motion_module.fc2.c_fc.weight")
    learned_routing = not concat and core.label_pre_temporal.scoring == "projected_cosine"
    if concat:
        assert core.label_pre_temporal is None
        verifier = core.query_evidence_verifier
        if verifier is not None:
            required += tuple('query_evidence_verifier.' + name for name, _ in verifier.named_parameters())
        contract = dict(
            architecture='text_concat_st',
            architecture_signature=core.text_concat_st.architecture_signature.tolist(),
            main_readout='spatial_text_attention',
            quality_enabled=bool(verifier is not None and verifier.quality),
            uniqueness_enabled=bool(verifier is not None and verifier.uniqueness),
            video_chunk=encoder_cfg.VIDEO_CHUNK_SIZE, class_chunk=encoder_cfg.CLASS_CHUNK_SIZE,
            checkpoint=encoder_cfg.ACTIVATION_CHECKPOINT,
            mlp_checkpoint=encoder_cfg.MLP_CHECKPOINT,
            readout_checkpoint=encoder_cfg.READOUT_CHECKPOINT,
            base_only_loss=cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.BASE_ONLY_LOSS_ENABLE,
            loader_workers=cfg.DATA_LOADER.NUM_WORKERS if args.real_data else 0,
        )
    else:
        required += ("query_evidence_verifier.semantic_quality.projection.weight",
                     "query_evidence_verifier.metric_temperature_raw", "query_evidence_verifier.metric_scale_raw")
        required += tuple("query_evidence_verifier.semantic_quality." + name
                          for name, _ in core.query_evidence_verifier.semantic_quality.named_parameters())
        contract = dict(
            routing_version=int(core.label_pre_temporal.architecture_version.item()),
            routing_scoring=core.label_pre_temporal.scoring,
            routing_parameters=sum(p.numel() for p in core.label_pre_temporal.parameters()),
            main_readout="reuse_early_weights",
            temporal_bias_scale=core.label_pre_temporal.temporal_bias_scale,
            spatial_bias_scale=core.label_pre_temporal.spatial_bias_scale,
            video_chunk=core.label_pre_temporal.video_chunk_size,
            class_chunk=core.label_pre_temporal.class_chunk_size,
            batch_raw_quality=core.label_pre_temporal.batch_raw_quality,
            checkpoint=core.label_pre_temporal.activation_checkpoint,
            mlp_checkpoint=core.blocks[0].checkpoint_mlp,
            readout_checkpoint=core.label_pre_temporal.readout_checkpoint,
            verifier_checkpoint=core.query_evidence_verifier.activation_checkpoint,
            quality_mode=core.query_evidence_verifier.quality_mode,
            verifier_version=int(core.query_evidence_verifier.architecture_version.item()),
        )
    watched_parameter = "label_pre_temporal.query.weight" if learned_routing else required[0]
    if learned_routing:
        required += ("label_pre_temporal.query.weight", "label_pre_temporal.key.weight")
    if cfg.FEW_SHOT.CLASS_LOSS_LAMBDA > 0:
        required += ("head.weight", "head.bias")
    classification_loss_fn = get_loss_func(cfg)(reduction="mean")
    step, attempt, skipped = 0, 0, 0
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        forward_backward_events = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        updates_per_epoch = (cfg.FEW_SHOT.TRAIN_EPISODES + world * args.grad_accum - 1) // (world * args.grad_accum)
        set_verifier_training_progress(model, attempt, cfg.SOLVER.MAX_EPOCH * updates_per_epoch)
        attempt += 1
        for micro in range(args.grad_accum):
            targets = labels[s:]
            if real_loader is not None:
                from trokens.utils.misc import iter_to_cuda
                video, global_labels, _, meta = next(real_loader)
                video, global_labels, meta = iter_to_cuda([video, global_labels, meta])
                meta["support_mask"] = torch.as_tensor(np.array(meta["sample_type"]) == "support", device="cuda")
                class_ids = get_episode_class_ids(meta, global_labels.device)
                meta["episode_positive_labels"] = episode_labels_from_global(global_labels, class_ids)
                targets = meta["episode_positive_labels"][~meta["support_mask"]]
                inputs = dict(video=video, metadata=meta)
            sync = model.no_sync() if world > 1 and micro + 1 < args.grad_accum else nullcontext()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            with sync:
                with torch.autocast("cuda", dtype=dtype):
                    output = model(inputs)
                    if concat and core.query_evidence_verifier is None:
                        assert torch.equal(output['query_partial_q2s_logits'], output['query_partial_q2s_base_logits'])
                    q2s_loss, _, _ = compute_query_partial_q2s_loss(
                        output["query_partial_q2s_logits"], targets, output, cfg)
                    cls_loss = q2s_loss.new_zeros(())
                    if cfg.FEW_SHOT.CLASS_LOSS_LAMBDA > 0:
                        preds = torch.nan_to_num(
                            output["global_classification_logits"] / cfg.SOLVER.TEMPRATURE,
                            nan=0., posinf=30., neginf=-30.)
                        cls_loss = multilabel_classification_loss(
                            preds, global_labels, cfg, classification_loss_fn)
                    loss = (cfg.FEW_SHOT.CLASS_LOSS_LAMBDA * cls_loss
                            + cfg.FEW_SHOT.Q2S_LOSS_LAMBDA * q2s_loss)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss before backward.")
                scaler.scale(loss / args.grad_accum).backward()
            end.record()
            forward_backward_events.append((begin, end))
        scaler.unscale_(optimizer)
        parameters = dict(core.named_parameters())
        bad = [name for name, p in parameters.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
        missing = [name for name in required if parameters[name].grad is None]
        if missing or not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss/gradients={bad}, missing core gradients={missing}")
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL)
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM)
        before = parameters[watched_parameter].detach().clone()
        head_before = core.head.weight.detach().clone()
        scale_before = checked_amp_scale(scaler, cfg.SOLVER.AMP_MIN_SCALE)
        scaler.step(optimizer)
        scaler.update()
        scale_after = checked_amp_scale(scaler, cfg.SOLVER.AMP_MIN_SCALE)
        torch.cuda.synchronize()
        updated = not torch.equal(before, parameters[watched_parameter])
        head_updated = not torch.equal(head_before, core.head.weight)
        optimizer_skipped = scale_after < scale_before
        if optimizer_skipped:
            skipped += 1
            if updated or not bad or skipped > args.max_skipped_updates:
                raise RuntimeError(f"AMP calibration failed after {skipped} skips: {bad}")
        elif bad or not updated:
            raise RuntimeError(f"Optimizer update did not change {watched_parameter}.")
        elif cfg.FEW_SHOT.CLASS_LOSS_LAMBDA > 0 and not head_updated:
            raise RuntimeError("Global classifier optimizer update did not change its parameters.")
        else:
            step += 1
        if world > 1:
            for name, parameter in core.named_parameters():
                if parameter.requires_grad:
                    replica = parameter.detach().clone()
                    dist.broadcast(replica, src=0)
                    if not torch.equal(replica, parameter):
                        raise RuntimeError(f"DDP parameters diverged: {name}")
            del replica
        print(json.dumps(dict(
            **contract,
            synthetic=not args.real_data, excludes_frozen_backbone=not args.real_data, includes_motion=True,
            rank=rank, world_size=world, grad_accum=args.grad_accum,
            transport_progress=(core.query_evidence_verifier.transport_progress
                                if core.query_evidence_verifier is not None else None),
            lr=optimizer.param_groups[0]["lr"],
            shape=[b, k, t, n, d], step=step, attempt=attempt, dtype=args.dtype,
            loss=float(loss.detach()), cls_loss=float(cls_loss.detach()), q2s_loss=float(q2s_loss.detach()),
            classifier_updated=head_updated, seconds=time.perf_counter() - start,
            forward_backward_seconds=sum(a.elapsed_time(b) for a, b in forward_backward_events) / 1000,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
            scaler_scale=scale_after, routing_updated=updated if learned_routing else None,
            watched_parameter=watched_parameter, watched_parameter_updated=updated,
            optimizer_skipped=optimizer_skipped, nonfinite_gradients=bad,
            replicas_equal=True if world > 1 else None,
        )), flush=True)
        del output, loss, before, head_before
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
