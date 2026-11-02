#!/usr/bin/env python3
"""Paired inference-only motion interventions using a frozen experiment checkout.

Appearance sampling and attention masks NEVER change across conditions. This is
not a retrained ablation and its metrics must not replace the formal run table.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch


CONDITIONS = (
    "real", "motion_off", "hod_only", "ctd_only", "static_geometry",
    "shuffle_motion_time", "null_inputs", "validity_masked", "hod_null",
    "ctd_static", "hod_validity_only", "ctd_validity_only",
)


def write_json(path, data):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def valid_displacement_mask(visibility):
    valid = torch.zeros_like(visibility, dtype=torch.bool)
    valid[:, 1:] = visibility[:, 1:].bool() & visibility[:, :-1].bool()
    return valid


def shuffle_pair(hod, ctd, seed):
    # Independent permutations per video; H/C use the SAME permutation. Keep
    # values unchanged, unlike re-differencing shuffled coordinates.
    rng = np.random.default_rng(seed)
    b, t = hod.shape[:2]
    order = np.stack([rng.permutation(t) for _ in range(b)])
    for row in order:
        if np.array_equal(row, np.arange(t)) and t > 1:
            row[:] = np.roll(row, 1)
    order = torch.as_tensor(order, device=hod.device)
    batch = torch.arange(b, device=hod.device)[:, None]
    return hod[batch, order], ctd[batch, order]


def condition_motion(model, metadata, hod, ctd, seed):
    """Return already-encoded branch pairs; gain is required to be one."""
    tracks = metadata["pred_tracks"]
    visibility = metadata["pred_visibility"]
    hist = metadata["hod_feat"].float()
    zero_hod = model.hod_motion_module(torch.zeros_like(hist))
    # All-zero pair descriptors, including removal of valid spatial geometry.
    zero_ctd = model.cross_motion_module(torch.zeros_like(tracks),
                                          torch.zeros_like(visibility))
    static_tracks = tracks[:, :1].expand_as(tracks).clone()
    static_ctd = model.cross_motion_module(static_tracks, visibility.clone())
    valid = valid_displacement_mask(visibility)
    hist_valid = hist * valid.permute(0, 2, 1).unsqueeze(-1)
    masked_hod = model.hod_motion_module(hist_valid) * valid.unsqueeze(-1)
    masked_ctd = ctd * visibility.bool().unsqueeze(-1)
    return {
        "real": (hod, ctd),
        "motion_off": (torch.zeros_like(hod), torch.zeros_like(ctd)),
        "hod_only": (hod, torch.zeros_like(ctd)),
        "ctd_only": (torch.zeros_like(hod), ctd),
        # Original visibility/attention masks retained; this isolates coordinate
        # dynamics, not every source of time variation in the CTD input.
        "static_geometry": (zero_hod, static_ctd),
        "shuffle_motion_time": shuffle_pair(hod, ctd, seed),
        "null_inputs": (zero_hod, zero_ctd),
        "validity_masked": (masked_hod, masked_ctd),
        "hod_null": (zero_hod, ctd),
        "ctd_static": (hod, static_ctd),
        "hod_validity_only": (masked_hod, ctd),
        "ctd_validity_only": (hod, masked_ctd),
    }


def metric_triple(aps):
    values = np.asarray(aps, dtype=np.float64)
    b, n = np.nanmean(values, axis=0) * 100
    return dict(base_map=float(b), novel_map=float(n), hm_map=float(2*b*n/(b+n)))


def bootstrap_deltas(ap_by_condition, repeats=2000):
    # Paired episodes, NOT independently resampled scores or pooled class AP.
    rng = np.random.default_rng(891)
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in ap_by_condition.items()}
    size = len(arrays["real"])
    deltas = {key: [] for key in arrays if key != "real"}
    for _ in range(repeats):
        ids = rng.integers(0, size, size=size)
        metrics = {key: metric_triple(value[ids])["hm_map"] for key, value in arrays.items()}
        for key in deltas:
            deltas[key].append(metrics[key] - metrics["real"])
    return {key: list(map(float, np.percentile(value, [2.5, 97.5])))
            for key, value in deltas.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    assert args.episodes > 0
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu
    # Refuse to admit on an already occupied physical GPU, prior to CUDA init.
    sys.path[:0] = [str(args.source / "tools"), str(args.source)]
    from queue_motion_ablation import nvidia_rows, verify_snapshot
    rows = nvidia_rows("--query-gpu=index,uuid,memory.used,memory.free")
    row = next(row for row in rows if row[0] == args.gpu)
    apps = nvidia_rows("--query-compute-apps=gpu_uuid,pid")
    assert not any(app[0] == row[1] for app in apps), "GPU busy; no probe launched"
    assert float(row[2]) < 512 and float(row[3]) > 20000, row
    manifest_path = args.source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_snapshot(args.source.parent, manifest)
    os.chdir(args.source)

    from trokens.config.defaults import get_cfg, assert_and_infer_cfg
    from trokens.config.ablation import resolve_ablation_cfg
    from trokens.config.feature_scale import validate_feature_scale_checkpoint
    from trokens.models import build_model
    from trokens.datasets.loader import construct_loader
    from trokens.utils.misc import iter_to_cuda
    from test_few_shot import process_patch_tokens, autocast_context
    from few_shot_multilabel import (
        support_query_split_multilabel, unpack_episode_output, compute_episode_ap,
        empty_ap_storage, update_ap_storage, compute_base_novel_hm,
    )

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = get_cfg()
    saved_cfg = type(cfg).load_cfg(ck["cfg"])
    # Runtime-only fields are added after model/loader construction in training.
    for field in ("num_patches", "wandb", "test_log_file", "csv_dump_name"):
        saved_cfg.pop(field, None)
    cfg.merge_from_other_cfg(saved_cfg)
    cfg.NUM_GPUS = 1
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.TRAIN.AUTO_RESUME = False
    cfg.TEST.CHECKPOINT_FILE_PATH = str(args.checkpoint)
    cfg.FEW_SHOT.TEST_EPISODES = args.episodes
    cfg.DATA_LOADER.NUM_WORKERS = 2
    cfg.RNG_SEED = args.seed
    cfg.OUTPUT_DIR = str(args.output)
    resolve_ablation_cfg(cfg)
    assert_and_infer_cfg(cfg)
    assert cfg.TEST.DATASET == "SAV" and cfg.FEW_SHOT.N_WAY == 5 and cfg.FEW_SHOT.K_SHOT == 1
    assert not any(cfg.FEW_SHOT.DECOUPLING.values())
    assert cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE
    assert cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE
    assert cfg.MODEL.FEATURE_SCALE.HOD_GAIN == cfg.MODEL.FEATURE_SCALE.CTD_GAIN == 1
    assert cfg.MODEL.FEATURE_SCALE.APPEARANCE_GAIN == 9.770436058356436
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.yaml").write_text(cfg.dump())
    started = time.monotonic()
    status = dict(state="INITIALIZING", pid=os.getpid(), gpu=args.gpu,
                  started_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                  checkpoint=str(args.checkpoint), checkpoint_sha256=checksum(args.checkpoint),
                  epoch=ck["epoch"] + 1, source=str(args.source),
                  script_sha256=checksum(Path(__file__).resolve()),
                  episodes=args.episodes, completed_episodes=0, conditions=list(CONDITIONS),
                  note="Paired fixed-checkpoint inference diagnostic; not retraining or formal 10000-episode test.")
    write_json(args.output / "status.json", status)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(2)
        model = build_model(cfg).eval()
        validate_feature_scale_checkpoint(model, ck["cfg"])
        saved_state = ck["model_state"]
        incompatible = model.load_state_dict(saved_state, strict=False)
        assert not incompatible.unexpected_keys, incompatible.unexpected_keys
        assert all(key.startswith("dinotxt_visual_model.") for key in incompatible.missing_keys), incompatible.missing_keys
        model.requires_grad_(False)
        del ck
        val_loader = construct_loader(cfg, "test")
        capture = {}
        sample_original = model._sample_point_features

        def sample(*a, **kw):
            result = sample_original(*a, **kw)
            capture["app"] = result
            return result

        model._sample_point_features = sample
        def save_branch(name):
            def hook(_module, _args, output):
                capture[name] = output
            return hook
        model.hod_motion_module.register_forward_hook(save_branch("hod"))
        model.cross_motion_module.register_forward_hook(save_branch("ctd"))
        stores = {name: empty_ap_storage(cfg.MODEL.NUM_CLASSES) for name in CONDITIONS}
        ap_by_condition = {name: [] for name in CONDITIONS}
        diagnostics = []
        max_replay_error = 0.0
        replay_checks = 0
        status["state"] = "RUNNING"
        write_json(args.output / "status.json", status)
        with torch.inference_mode(), (args.output / "episodes.jsonl").open("x") as episode_file:
            for index, (inputs, labels, _, meta) in enumerate(val_loader):
                inputs, labels, meta = iter_to_cuda([inputs, labels, meta])
                record = {"episode": index, "videos": list(meta["video_name"]), "conditions": {}}
                capture.clear()
                with autocast_context(cfg.TRAIN.MIXED_PRECISION):
                    actual = model({"video": inputs, "metadata": meta})
                    _, real_patches, _ = unpack_episode_output(actual)
                    app, hod, ctd = capture["app"], capture["hod"], capture["ctd"]
                    assert all(torch.isfinite(x).all() for x in (app, hod, ctd, real_patches))
                    branches = condition_motion(model, meta, hod, ctd, 15000 + index)
                    if index < 3:
                        replay = model.pt_forward(app + hod + ctd, meta)[1]
                        error = (real_patches - replay).float().abs().max().item()
                        max_replay_error = max(max_replay_error, error)
                        assert torch.allclose(real_patches, replay, atol=1e-5, rtol=1e-5), error
                        replay_checks += 1
                    def rms(x):
                        return x.float().square().mean().sqrt().item()
                    diagnostics.append(dict(app_rms=rms(app), hod_rms=rms(hod), ctd_rms=rms(ctd),
                        hod_null_rms=rms(branches["null_inputs"][0]),
                        ctd_null_rms=rms(branches["null_inputs"][1]),
                        hod_real_minus_null_rms=rms(hod-branches["null_inputs"][0]),
                        ctd_real_minus_null_rms=rms(ctd-branches["null_inputs"][1])))
                    reference_labels = reference_classes = None
                    for condition in CONDITIONS:
                        h, c = branches[condition]
                        patches = real_patches if condition == "real" else model.pt_forward(app + h + c, meta)[1]
                        split = support_query_split_multilabel(patches, labels, meta)
                        logits = process_patch_tokens(cfg, split["support_preds"], split["query_preds"])
                        assert torch.isfinite(logits).all(), (index, condition)
                        targets = split["query_batch_labels"]
                        classes = split["episode_class_ids"]
                        if reference_labels is None:
                            reference_labels, reference_classes = targets.clone(), classes.clone()
                        else:
                            assert torch.equal(targets, reference_labels) and torch.equal(classes, reference_classes)
                        update_ap_storage(stores[condition], logits, targets, classes)
                        episode = stores[condition]["episodes"][-1]
                        scores_np = np.asarray(episode["scores"], dtype=np.float32)
                        labels_np = np.asarray(episode["labels"], dtype=np.float32)
                        aps = [compute_episode_ap(labels_np, scores_np, episode["episode_class_ids"],
                               subset, cfg.TEST.DROP_EMPTY_QUERY_ROWS)
                               for subset in (cfg.TEST.SEEN_LABELS, cfg.TEST.NOVEL_LABELS)]
                        ap_by_condition[condition].append([v if v is not None else np.nan for v in aps])
                        record["conditions"][condition] = {**episode, "base_ap": aps[0], "novel_ap": aps[1]}
                episode_file.write(json.dumps(record) + "\n")
                if (index+1) % 25 == 0 or index+1 == args.episodes:
                    episode_file.flush()
                    elapsed = time.monotonic()-started
                    status.update(completed_episodes=index+1, elapsed_seconds=elapsed,
                                  estimated_remaining_seconds=elapsed/(index+1)*(args.episodes-index-1))
                    write_json(args.output / "status.json", status)
                    print(json.dumps({k:status[k] for k in ("completed_episodes", "elapsed_seconds", "estimated_remaining_seconds")}), flush=True)
        assert len(diagnostics) == args.episodes
        for key, value in saved_state.items():
            assert torch.equal(model.state_dict()[key].cpu(), value), f"Weight changed: {key}"
        ci = bootstrap_deltas(ap_by_condition)
        results = {}
        baseline = metric_triple(ap_by_condition["real"])["hm_map"]
        for condition in CONDITIONS:
            result = metric_triple(ap_by_condition[condition])
            official = compute_base_novel_hm(stores[condition], cfg)
            assert np.allclose(list(result.values()), official[:3])
            result.update(hm_delta_vs_real=result["hm_map"]-baseline,
                          paired_episode_bootstrap_95ci=ci.get(condition),
                          class_aps={str(k):v*100 for k,v in official[3].items()})
            results[condition] = result
        summary = dict(protocol=status["note"], episodes=args.episodes, checkpoint=status["checkpoint"],
                       checkpoint_epoch=status["epoch"], results=results,
                       mean_feature_statistics={key:float(np.mean([row[key] for row in diagnostics])) for key in diagnostics[0]},
                       cached_forward_max_error=max_replay_error, cached_forward_checks=replay_checks,
                       checkpoint_weights_unchanged=True,
                       limitations=["One checkpoint and seed; bootstrap only captures episode sampling uncertainty.",
                                    "Motion-only interventions; visual features, sampling and attention masks held fixed.",
                                    "Static condition retains original visibility; shuffled condition permutes encoded motion in time.",
                                    "Validity condition changes both HoD and CTD validity handling; cannot attribute them separately."])
        write_json(args.output / "results.json", summary)
        status.update(state="COMPLETE", elapsed_seconds=time.monotonic()-started)
        write_json(args.output / "status.json", status)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        status.update(state="FAILED", error=str(exc), elapsed_seconds=time.monotonic()-started)
        write_json(args.output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
