"""Compare verifier implementations/architectures, without training.

Run from the repo root with CUDA_VISIBLE_DEVICES set to an idle GPU. Models do
forward/backward only: no optimizer updates, checkpoint writes or DDP workers.
The sequential reference is test-only, not the heuristic BACKEND=legacy.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from trokens.config.defaults import get_cfg
from trokens.models.pointformer import Pointformer
from tools.few_shot_multilabel import (
    compute_query_partial_q2s_loss, get_text_align_loss, multilabel_classification_loss,
)


def _reference_type():
    spec = importlib.util.spec_from_file_location(
        "verifier_benchmark_reference", REPO_ROOT / "tests/reference_query_verifier.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SequentialQueryEvidenceVerifier


def _synthetic_route(cfg):
    model = Pointformer.__new__(Pointformer)
    nn.Module.__init__(model)
    model.cfg, model.embed_dim = cfg, 1024
    model.pot_route_cfg = cfg.FEW_SHOT.POT_ROUTE
    model.use_frame_softmax_route = model.use_query_partial_q2s = True
    model.use_support_text_fusion = cfg.FEW_SHOT.SUPPORT_TEXT_FUSION.ENABLE
    model.support_text_fusion_cfg = cfg.FEW_SHOT.SUPPORT_TEXT_FUSION
    model._init_temporal_similarity_refinement()
    model._init_query_evidence_verifier()
    model.cuda().train()
    k, t, n, d = cfg.FEW_SHOT.N_WAY, cfg.DATA.NUM_FRAMES, cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE, 1024
    s, q = k * cfg.FEW_SHOT.K_SHOT, k * cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS
    text = torch.randn(k, d, device="cuda")
    model._get_pot_label_text_features = lambda ids, dtype: text.to(dtype)
    post = torch.randn(s + q, t, n, d, device="cuda", dtype=torch.float16, requires_grad=True)
    raw = torch.randn_like(post)
    mask = torch.ones(s + q, t, n, device="cuda", dtype=torch.bool)
    visibility = mask.clone()
    visibility[s, 1] = False
    labels = torch.eye(k, device="cuda").repeat(cfg.FEW_SHOT.K_SHOT + cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS, 1)
    meta = dict(support_mask=torch.arange(s + q, device="cuda") < s, pred_query_mask=mask,
                pred_visibility=visibility, episode_positive_labels=labels,
                episode_class_ids=torch.arange(k, device="cuda"))

    def forward():
        post.grad = None
        aux = model._build_frame_softmax_q2s_aux(post, meta, matchability_evidence_tokens=raw)
        loss, _, _ = compute_query_partial_q2s_loss(aux["query_partial_q2s_logits"], labels[s:], aux, cfg)
        return loss, aux

    return model, forward, dict(Q=q, S=s, K=k, T=t, N=n, D=d)


def _real_episode(cfg):
    from trokens.datasets.loader import construct_loader
    from trokens.utils.misc import iter_to_cuda
    # Cache exactly one real episode on GPU: decoding/H2D is outside timing.
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.FEW_SHOT.TRAIN_EPISODES = 1
    data = next(iter(construct_loader(cfg, "train")))
    videos, labels, meta = iter_to_cuda([data[0], data[1], data[3]])
    support = torch.as_tensor(np.array(meta["sample_type"]) == "support", device="cuda")
    classes = meta["episode_class_ids"]
    classes = classes[0] if classes.ndim == 2 else classes
    meta["support_mask"] = support
    meta["episode_positive_labels"] = labels.float().index_select(1, classes.long())
    meta["raw_positive_labels"] = labels.float()
    model = Pointformer(cfg).cuda().train()
    criterion = nn.BCEWithLogitsLoss()

    def forward():
        predictions, patch_tokens, aux = model(dict(video=videos, metadata=meta))
        predictions = torch.nan_to_num(predictions / cfg.SOLVER.TEMPRATURE,
                                       nan=0.0, posinf=30.0, neginf=-30.0)
        cls_loss = multilabel_classification_loss(predictions, labels, cfg, criterion)
        q2s_loss, _, _ = compute_query_partial_q2s_loss(
            aux["query_partial_q2s_logits"], meta["episode_positive_labels"][~support], aux, cfg,
        )
        loss = (cfg.FEW_SHOT.CLASS_LOSS_LAMBDA * cls_loss
                + cfg.FEW_SHOT.Q2S_LOSS_LAMBDA * q2s_loss
                + cfg.FEW_SHOT.TEXT_ALIGN.LOSS_WEIGHT * get_text_align_loss(aux, patch_tokens))
        return loss, aux

    return model, forward, dict(Q=int((~support).sum()), S=int(support.sum()), K=classes.numel(),
                                T=videos.shape[1], N=meta["pred_tracks"].shape[2], D=model.embed_dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(REPO_ROOT / "configs/trokens/sav.yaml"))
    parser.add_argument("--mode", choices=("route", "episode"), default="route")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=[
        "sequential", "batched_fp32", "batched_fp32_large", "batched_amp"],
        choices=("sequential", "batched_fp32", "batched_fp32_large", "batched_amp",
                 "metric_fp32", "metric_amp", "semantic_fp32", "semantic_amp"))
    parser.add_argument("--loss-scale", type=float, default=1024.0,
                        help="Scale backward then unscale gradients; no optimizer step is performed")
    args = parser.parse_args()
    if min(args.shot, args.steps, args.warmup) < 1:
        parser.error("shot, steps and warmup must be positive")
    if not torch.cuda.is_available():
        parser.error("an idle CUDA GPU is required")
    torch.set_num_threads(1)
    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)
    torch.backends.cudnn.deterministic = True
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.NUM_GPUS, cfg.FEW_SHOT.K_SHOT = 1, args.shot
    # Canonical v1 seed supplies weights for unchanged heads only. v3 quality
    # uses a separate fixed seed, shared by its FP32/AMP benchmark variants.
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.QUALITY_MODE = "semantic_attention"
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.UNIQUENESS_MODE = "relation_attention"
    model, forward, shapes = (_real_episode if args.mode == "episode" else _synthetic_route)(cfg)
    optimized = model.query_evidence_verifier
    if optimized is None:
        parser.error("the config must enable learned verification")
    if len(set(args.variants)) != len(args.variants) or len(args.variants) < 2:
        parser.error("choose at least two distinct variants")
    if not 0 < args.loss_scale < float("inf"):
        parser.error("loss-scale must be finite and positive")
    versions = {}
    for name in args.variants:
        vc = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.clone()
        vc.AMP_ENABLE = name in ("batched_amp", "metric_amp", "semantic_amp")
        vc.QUALITY_MODE = "semantic_relation" if name.startswith("semantic_") else "semantic_attention"
        vc.UNIQUENESS_MODE = "metric" if name.startswith(("metric_", "semantic_")) else "relation_attention"
        if name in ("sequential", "batched_fp32"):
            vc.QUERY_CHUNK_SIZE, vc.ACTIVATION_CHECKPOINT = 4, True
        cls = _reference_type() if name == "sequential" else type(optimized)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(779)
            verifier = cls(model.embed_dim, vc, optimized.quality, optimized.uniqueness).cuda().train()
        if name.startswith(("metric_", "semantic_")):
            state = verifier.state_dict()
            state.update({key: value for key, value in optimized.state_dict().items()
                          if key in state and key != "architecture_version"})
            verifier.load_state_dict(state, strict=True)
        else:
            verifier.load_state_dict(optimized.state_dict(), strict=True)
        versions[name] = verifier
    model.query_evidence_verifier = next(iter(versions.values()))
    del optimized
    print(json.dumps(dict(mode=args.mode, shape=shapes, torch=torch.__version__,
                          gpu=torch.cuda.get_device_name(), tf32=torch.backends.cuda.matmul.allow_tf32,
                          variants={name: dict(chunk=v.chunk_size, checkpoint=v.activation_checkpoint,
                                    amp=getattr(v, "amp_enabled", False),
                                    quality_mode=getattr(v, "quality_mode", "semantic_attention"),
                                    architecture_version=int(v.architecture_version),
                                    parameters=sum(p.numel() for p in v.parameters()))
                                    for name, v in versions.items()},
                          loss_scale=args.loss_scale, warmup=args.warmup, steps=args.steps)), flush=True)
    scaler = torch.cuda.amp.GradScaler(init_scale=args.loss_scale)
    # Unscale for the correctness checks only; never step or update optimizer.
    optimizers = {name: torch.optim.SGD(v.parameters(), lr=0) for name, v in versions.items()}

    def step(name, capture=False):
        model.query_evidence_verifier = versions[name]
        model.zero_grad(set_to_none=True)
        for verifier in versions.values():
            verifier.zero_grad(set_to_none=True)
        torch.manual_seed(812)  # identical dropout/main-model state in every comparison
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.TRAIN.MIXED_PRECISION):
            loss, aux = forward()
        torch.cuda.synchronize()
        middle = time.perf_counter()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        metrics = ((middle - start) * 1000, (end - middle) * 1000, (end - start) * 1000,
                   torch.cuda.max_memory_allocated() / 1024 ** 2)
        captured = None
        if capture:
            scaler.unscale_(optimizers[name])
            assert torch.isfinite(loss)
            keys = ("query_quality_logits", "query_relation_logits", "query_frame_target_mass",
                    "query_frame_confuser_mass", "query_frame_null_mass", "query_partial_q2s_logits",
                    "query_partial_q2s_base_logits", "query_partial_query_prototypes",
                    "query_partial_support_prototypes", "query_patch_conditional_weights")
            outputs = {key: aux[key].detach().cpu() for key in keys if key in aux}
            outputs["loss"] = loss.detach().cpu()
            grads = {}
            for key, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise AssertionError(f"Non-finite gradient: {key}")
            for key, parameter in versions[name].named_parameters():
                assert parameter.grad is not None, key
                grads[key] = parameter.grad.detach().cpu()
            captured = outputs, grads
        return metrics, captured

    checks = [step(name, capture=True)[1] for name in versions]
    errors = {}
    for name, actual in zip(list(versions)[1:], checks[1:]):
        mixed = any(getattr(versions[v], "amp_enabled", False) for v in (args.variants[0], name))
        cross_architecture = versions[name].architecture_version != versions[args.variants[0]].architecture_version
        errors[name] = {}
        for kind, left, right in zip(("outputs", "verifier_gradients"), checks[0], actual):
            if cross_architecture:
                # Changed heads/g/loss and all loss gradients intentionally
                # differ. Compare the base route and only unchanged heads.
                if kind == "verifier_gradients":
                    continue
                unchanged = ["query_partial_q2s_base_logits",
                             "query_partial_query_prototypes", "query_partial_support_prototypes",
                             "query_patch_conditional_weights"]
                before, after = versions[args.variants[0]], versions[name]
                if getattr(before, "quality_mode", "semantic_attention") == getattr(after, "quality_mode", "semantic_attention"):
                    unchanged.append("query_quality_logits")
                if getattr(before, "uniqueness_mode", "relation_attention") == getattr(after, "uniqueness_mode", "relation_attention"):
                    unchanged.append("query_relation_logits")
                left, right = ({k: data[k] for k in unchanged if k in data} for data in (left, right))
            atol, rtol = ((2e-3, 5e-3) if kind == "outputs" else (2e-4, 1e-2)) if mixed else (5e-6, 5e-5)
            for key in left:
                torch.testing.assert_close(left[key], right[key], atol=atol, rtol=rtol,
                                           msg=lambda detail, key=key: f"{name}: {key}\n{detail}")
            errors[name][kind + "_max_abs_error"] = max(float((left[key] - right[key]).abs().max()) for key in left)
        errors[name]["scope"] = "base_route_and_unchanged_heads_only" if cross_architecture else "full_equivalence"
    print(json.dumps(dict(equivalence=errors, finite_scaled_gradients=True)), flush=True)
    del checks
    for _ in range(args.warmup):
        for name in versions:
            step(name)
    measured = {name: [] for name in versions}
    # Alternate order to reduce clock/warmup bias; nothing is optimized/updated.
    for index in range(args.steps):
        for name in (list(versions) if index % 2 == 0 else list(reversed(versions))):
            measured[name].append(step(name)[0])
    result = {}
    for name, rows in measured.items():
        result[name] = {key: round(statistics.median(row[i] for row in rows), 3)
                        for i, key in enumerate(("forward_ms", "backward_ms", "total_ms", "peak_allocated_MiB"))}
    old = result[args.variants[0]]["total_ms"]
    for name in args.variants[1:]:
        new = result[name]["total_ms"]
        result[name].update(time_reduction_percent=round(100 * (1 - new / old), 2),
                            speedup=round(old / new, 3))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
