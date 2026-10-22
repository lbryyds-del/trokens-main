#!/usr/bin/env python3
"""Visualize measured post-decoupling patch scores from an isolated git revision.

No training, checkout, current config changes, or generated image content. The
historical dataset/model implementation is extracted to a temporary directory.
Figures use trajectory coordinates, not an invented 16x16 token arrangement.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


REVISION = "a5134b8a69e49a2c4bc3c5ace2bd58e40cc957c3"
CHECKPOINT = (
    "output/sav_semantic_relation_v3/checkpoints/best_by_run/"
    "checkpoint_best_run_20261011_133133.pyth"
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--checkpoint", type=Path, default=Path(CHECKPOINT))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="3", help="Physical CUDA GPU index")
    parser.add_argument("--episode", type=int, default=0, help="Zero-based test episode")
    parser.add_argument("--query", type=int, default=0, help="Query position in episode")
    parser.add_argument("--frame", type=int, default=3, help="Highlighted frame, zero-based")
    parser.add_argument("--video-labels", action="store_true",
                        help="Plot only this video's annotated classes, using pure-text Post scores")
    return parser.parse_args()


def export_revision(repo, revision, destination):
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"{revision}^{{commit}}"], text=True
    ).strip()
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", commit, "trokens"]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination)
    return commit


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_focus(output, arrays, info):
    """Compact single-frame view, with the same global linear weight scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    t = info["highlight_frame"]
    rgb = arrays["query_rgb"][t]
    xy = arrays["query_xy_pixels"][t]
    weights = arrays["actual_query_weights"][:, t]
    visible = arrays["query_visibility"][t]
    norm = Normalize(*info["plot_scales"]["weight_percent"])
    count = len(weights) + 1
    fig, axes = plt.subplots((count + 2) // 3, 3, figsize=(11, 8.5), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    axes.flat[0].imshow(rgb)
    axes.flat[0].set_title("Original | sampled frame " + str(t + 1) + "/8\nClip GT: "
                           + ", ".join(info["query_all_clip_labels"]), fontsize=11)
    for k, weight in enumerate(weights):
        ax = axes.flat[k + 1]
        ax.imshow(rgb)
        for valid, marker in ((visible, "o"), (~visible, "x")):
            ax.scatter(xy[valid, 0], xy[valid, 1], c=100 * weight[valid],
                       cmap="magma", norm=norm, marker=marker, s=23, alpha=0.85)
        top = np.argsort(weight, kind="stable")[-8:]
        ax.scatter(xy[top, 0], xy[top, 1], s=110, facecolors="none", edgecolors="cyan", linewidths=1.4)
        ax.set_xlim(-0.5, rgb.shape[1] - 0.5)
        ax.set_ylim(rgb.shape[0] - 0.5, -0.5)
        present = info["query_labels_in_episode"][k]
        ax.set_title(info["class_names"][k] + (" | GT present" if present else " | GT absent")
                     + f"\nTop-8 mass: {weight[top].sum():.1%}", fontsize=11,
                     color="#117744" if present else "#994444")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="magma"),
                 ax=list(axes.flat), shrink=0.7, label="Actual Query patch weight % (shared linear scale)")
    fig.suptitle("Post-decoupling: actual Text+Support Query routing\n"
                 "Cyan: top-8 for display only; x: tracker invisible; full 8-frame inference", fontsize=12)
    fig.savefig(output / "single_frame_focus.png", dpi=180)
    plt.close(fig)


def save_video_labels(output, arrays, info):
    """Full 8-frame diagnostics for the video's own labels, not episode classes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    rgb, xy = arrays["query_rgb"], arrays["query_xy_pixels"]
    visible = arrays["query_visibility"]
    scores, weights = arrays["post_text_cosine"], arrays["post_text_weights"]
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite data in true-label visualization")
    if not np.allclose(weights.sum(-1), 1, atol=1e-5):
        raise ValueError("Text-only weights do not sum to one")
    np.savez_compressed(output / "patch_scores.npz", **arrays)
    cnorm = Normalize(float(scores.min()), float(scores.max()))
    wnorm = Normalize(0, float(weights.max() * 100))
    info["plot_scales"] = {"cosine": [cnorm.vmin, cnorm.vmax],
                           "weight_percent": [wnorm.vmin, wnorm.vmax]}
    with (output / "patch_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_zero_based", "class_id_zero_based", "class_name", "token", "source_point",
                         "x_224", "y_224", "tracker_visible", "routing_valid", "text_post_cosine", "text_only_weight"])
        for k, cid in enumerate(info["class_ids_zero_based"]):
            for t in range(len(rgb)):
                for n in range(xy.shape[1]):
                    writer.writerow([t, cid, info["class_names"][k], n, int(arrays["point_indices"][n]),
                                     *xy[t, n], int(visible[t, n]), int(arrays["query_routing_mask"][t, n]),
                                     scores[k, t, n], weights[k, t, n]])

    def panel(ax, t, values=None, norm=None, cmap=None, top=None):
        ax.imshow(rgb[t])
        ax.set_xlim(-0.5, rgb.shape[2] - 0.5)
        ax.set_ylim(rgb.shape[1] - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if values is not None:
            for valid, marker in ((visible[t], "o"), (~visible[t], "x")):
                ax.scatter(xy[t, valid, 0], xy[t, valid, 1], c=values[valid],
                           cmap=cmap, norm=norm, marker=marker, s=19, linewidths=0.6, alpha=0.85)
            ax.scatter(xy[t, top, 0], xy[t, top, 1], s=68, facecolors="none",
                       edgecolors="cyan", linewidths=1.0)

    # Main overviews: original + each of the actual video labels, all frames.
    for kind, values, norm, cmap, title in (
        ("cosine", scores, cnorm, "inferno", "Pure text vs Post patch: cosine"),
        ("weights", weights * 100, wnorm, "magma", "Pure-text Post softmax weight (%)"),
    ):
        fig, axes = plt.subplots(len(scores) + 1, len(rgb), figsize=(22, 3 * (len(scores) + 1)),
                                 layout="constrained", squeeze=False)
        for t in range(len(rgb)):
            panel(axes[0, t], t)
            axes[0, t].set_title(f"Frame {t + 1}")
            for k, name in enumerate(info["class_names"]):
                top = np.argsort(weights[k, t], kind="stable")[-8:]
                panel(axes[k + 1, t], t, values[k, t], norm, cmap, top)
                if t == 0:
                    axes[k + 1, t].set_ylabel(name, fontsize=12)
        axes[0, 0].set_ylabel("Original", fontsize=12)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=axes[1:].ravel().tolist(), shrink=0.65, label=title + " | shared linear scale")
        fig.suptitle("Video's annotated classes: " + ", ".join(info["class_names"])
                     + "\n" + title + "; cyan: top-8 display only; x: tracker invisible; full 8-frame context", fontsize=13)
        fig.savefig(output / f"all_frames_{kind}.png", dpi=160)
        plt.close(fig)

    for k, name in enumerate(info["class_names"]):
        fig, axes = plt.subplots(3, len(rgb), figsize=(22, 9), layout="constrained")
        for t in range(len(rgb)):
            top = np.argsort(weights[k, t], kind="stable")[-8:]
            panel(axes[0, t], t)
            axes[0, t].set_title(f"Frame {t + 1}")
            panel(axes[1, t], t, scores[k, t], cnorm, "inferno", top)
            panel(axes[2, t], t, 100 * weights[k, t], wnorm, "magma", top)
            axes[2, t].set_xlabel(f"top-8 mass {weights[k, t, top].sum():.1%}", fontsize=10)
        for ax, label in zip(axes[:, 0], ["Original", "Text/Post cosine", "Text-only softmax"]):
            ax.set_ylabel(label, fontsize=11)
        for row, norm, cmap, title in [(1, cnorm, "inferno", "Cosine"), (2, wnorm, "magma", "Weight %")]:
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[row]),
                         shrink=0.8, label=title)
        fig.suptitle(name + " | annotated VIDEO label, not per-frame ground truth\n"
                     "All 8 frames; pure-text diagnostic (not Support-fused Query routing); shared scales across classes", fontsize=13)
        fig.savefig(output / f"class_{name}_8frames.png", dpi=160)
        plt.close(fig)
    (output / "provenance.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def render(output, arrays, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np
    from PIL import Image

    rgb = arrays["query_rgb"]
    xy = arrays["query_xy_pixels"]
    visible = arrays["query_visibility"]
    text = arrays["post_text_cosine"]
    fused = arrays["post_fused_cosine"]
    text_weights = arrays["post_text_weights"]
    weights = arrays["actual_query_weights"]
    classes = info["class_names"]
    labels = info["query_labels_in_episode"]
    # One linear scale across ALL classes and frames; no per-panel min/max.
    cosine_norm = Normalize(min(text.min(), fused.min()), max(text.max(), fused.max()))
    weight_norm = Normalize(0, 100 * max(text_weights.max(), weights.max()))
    info["plot_scales"] = {
        "cosine": [float(cosine_norm.vmin), float(cosine_norm.vmax)],
        "weight_percent": [0, float(weight_norm.vmax)],
    }

    def base(ax, t):
        ax.imshow(rgb[t])
        ax.set_xlim(-0.5, rgb.shape[2] - 0.5)
        ax.set_ylim(rgb.shape[1] - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    def scatter(ax, t, values, norm, cmap, top=None):
        base(ax, t)
        # X markers mean tracker-invisible, still used by QUERY_MASK_MODE=all.
        for valid, marker in ((visible[t], "o"), (~visible[t], "x")):
            ax.scatter(xy[t, valid, 0], xy[t, valid, 1], c=values[valid],
                       norm=norm, cmap=cmap, marker=marker, s=21, linewidths=0.6,
                       alpha=0.85)
        if top is not None:
            ax.scatter(xy[t, top, 0], xy[t, top, 1], s=90, facecolors="none",
                       edgecolors="cyan", linewidths=1.3)
        return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    columns = ["Original / clip label", "Text vs Post: cosine",
               "Text-only softmax (diagnostic)", "Text+Support vs Post: cosine",
               "Actual Query softmax / top-8"]
    for t in range(rgb.shape[0]):
        fig, axes = plt.subplots(len(classes), 5, figsize=(16.5, 3.0 * len(classes)),
                                 layout="constrained")
        for k, name in enumerate(classes):
            base(axes[k, 0], t)
            axes[k, 0].set_ylabel(
                f"{name}\nclip GT: {'present' if labels[k] else 'absent'}",
                fontsize=11, color="#117744" if labels[k] else "#994444")
            top = np.argsort(weights[k, t], kind="stable")[-8:]
            sc = scatter(axes[k, 1], t, text[k, t], cosine_norm, "inferno")
            sw = scatter(axes[k, 2], t, 100 * text_weights[k, t], weight_norm, "magma")
            scatter(axes[k, 3], t, fused[k, t], cosine_norm, "inferno")
            scatter(axes[k, 4], t, 100 * weights[k, t], weight_norm, "magma", top)
            axes[k, 4].set_xlabel(
                f"top-8 mass={weights[k, t, top].sum():.1%}; "
                f"top-1={weights[k, t].max():.1%}", fontsize=10)
        for ax, title in zip(axes[0], columns):
            ax.set_title(title, fontsize=11)
        fig.colorbar(sc, ax=axes[:, [1, 3]].ravel().tolist(), shrink=0.55,
                     label="Cosine (shared scale)")
        fig.colorbar(sw, ax=axes[:, [2, 4]].ravel().tolist(), shrink=0.55,
                     label="Patch weight % (shared scale)")
        fig.suptitle(
            f"Post-decoupling | {info['query_video_name']} | sampled frame {t + 1}/8\n"
            "Full 8-frame inference; 256 tracked tokens; cyan = top-8 diagnostic, "
            "x = tracker invisible; no spatial smoothing", fontsize=12)
        fig.savefig(output / f"frame_{t:02d}.png", dpi=150)
        if t == info["highlight_frame"]:
            fig.savefig(output / "single_frame.png", dpi=180)
        plt.close(fig)
        Image.fromarray((rgb[t] * 255).round().astype("uint8")).save(
            output / f"original_{t:02d}.png")

    # Compact view: one original row plus one actual-routing row per class.
    fig, axes = plt.subplots(len(classes) + 1, rgb.shape[0], figsize=(20, 2.6 * (len(classes) + 1)),
                             layout="constrained")
    for t in range(rgb.shape[0]):
        base(axes[0, t], t)
        axes[0, t].set_title(f"Frame {t + 1}")
        for k in range(len(classes)):
            top = np.argsort(weights[k, t], kind="stable")[-8:]
            sw = scatter(axes[k + 1, t], t, 100 * weights[k, t], weight_norm, "magma", top)
    axes[0, 0].set_ylabel("Original")
    for k, name in enumerate(classes):
        axes[k + 1, 0].set_ylabel(name + ("\nGT present" if labels[k] else "\nGT absent"))
    fig.colorbar(sw, ax=axes[1:].ravel().tolist(), shrink=0.6, label="Actual patch weight %")
    fig.suptitle("Actual Query routing across all 8 sampled frames (one shared color scale)")
    fig.savefig(output / "all_frames.png", dpi=150)
    plt.close(fig)

    # Show the actual episode supports so fusion is not a hidden condition.
    support_rgb = arrays["support_rgb"]
    fig, axes = plt.subplots(1, len(support_rgb), figsize=(3.4 * len(support_rgb), 4),
                             layout="constrained", squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.imshow(support_rgb[i, info["highlight_frame"]])
        ax.axis("off")
        names = [name for name, y in zip(classes, info["support_labels_in_episode"][i]) if y]
        ax.set_title("\n".join(names), fontsize=10)
        ax.set_xlabel(info["support_video_names"][i], fontsize=7)
    fig.suptitle("Episode Support clips: only their known labels enter prototype construction")
    fig.savefig(output / "supports.png", dpi=150)
    plt.close(fig)
    render_focus(output, arrays, info)


def run(args, commit):
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from fvcore.common.config import CfgNode
    from trokens.datasets.build import build_dataset
    from trokens.datasets.custom_sampler import FewShotEpisodeSampler
    from trokens.models.pointformer import Pointformer

    torch.set_num_threads(4)
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Checkpoints contain the complete merged configuration plus runtime keys
    # (e.g. num_patches), which are not present in the defaults schema.
    cfg = CfgNode.load_cfg(checkpoint["cfg"])
    saved_cfg = cfg.dump()
    cfg.NUM_GPUS = 1
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.DATA_LOADER.PIN_MEMORY = False
    cfg.num_patches = cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE
    if args.episode < 0 or args.episode >= cfg.FEW_SHOT.TEST_EPISODES:
        raise ValueError("Episode index outside test sampler range")
    if not 0 <= args.frame < cfg.DATA.NUM_FRAMES:
        raise ValueError("Frame index outside sampled clip")
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    print("Constructing historical test dataset", flush=True)
    dataset = build_dataset(cfg.TEST.DATASET, cfg, "test")
    sampler = FewShotEpisodeSampler(dataset, cfg, "test")
    sampler.local_episode_ids = [args.episode]
    episode_spec = next(iter(sampler))
    videos, labels, _, meta = default_collate([dataset[item] for item in episode_spec])
    support = torch.tensor([s == "support" for s in meta["sample_type"]])
    query_indices = (~support).nonzero().flatten()
    if not 0 <= args.query < len(query_indices):
        raise ValueError("Query index outside episode")
    query_index = int(query_indices[args.query])
    class_ids = meta["episode_class_ids"][0].long()
    episode_labels = labels[:, class_ids].bool()
    meta["support_mask"] = support
    meta["raw_positive_labels"] = labels.float()
    meta["episode_positive_labels"] = episode_labels
    print("Episode", args.episode, "classes", class_ids.tolist(), "query", meta["video_name"][query_index], flush=True)
    print("Loading historical model and frozen DinoTxt towers", flush=True)
    model = Pointformer(cfg).cuda().eval()
    state = checkpoint["model_state"]
    result = model.load_state_dict(state, strict=False)
    # Historical checkpoints intentionally exclude frozen pretrained towers.
    allowed = ("dinotxt_visual_model.", "dinotxt_text_model.")
    unexpected_missing = [k for k in result.missing_keys if not k.startswith(allowed)]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {unexpected_missing}; {result.unexpected_keys}")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name not in state:
            raise RuntimeError(f"Trainable parameter not loaded: {name}")
    print(f"Loaded all {len(state)} stored tensors; only frozen towers omitted", flush=True)
    meta_gpu = {k: v.cuda() if torch.is_tensor(v) else v for k, v in meta.items()}
    video_gpu = videos.cuda()
    with torch.inference_mode(), torch.autocast("cuda", enabled=cfg.TRAIN.MIXED_PRECISION):
        _, post, aux = model({"video": video_gpu, "metadata": meta_gpu})
        text = model._get_pot_label_text_features(class_ids.cuda(), post.dtype)
        # The learned verifier branch does not expose route_text in its aux;
        # reproduce it from the exact Support prototypes returned by inference.
        fused, _, _ = model._fuse_episode_text_with_support_visual(
            text, aux["query_partial_support_prototypes"])
        query_post = post[query_indices.cuda()]
        query_mask = meta_gpu["pred_query_mask"][query_indices.cuda()]
        cosine_text = model._compute_batched_point_text_similarity(query_post, text)
        cosine_fused = model._compute_batched_point_text_similarity(query_post, fused)
        _, weights_text = model._compute_batched_frame_softmax_text_prototypes(query_post, query_mask, text)
        rebuilt_prototypes, rebuilt_weights = model._compute_batched_frame_softmax_text_prototypes(
            query_post, query_mask, fused)
        actual_weights = aux["query_patch_conditional_weights"]
        weight_error = float((actual_weights.float() - rebuilt_weights.float()).abs().max())
        proto_error = float((aux["query_partial_query_prototypes"].float() - rebuilt_prototypes.float()).abs().max())
        if weight_error > 1e-3 or proto_error > 5e-3:
            raise RuntimeError(f"Reconstructed route differs from inference: {weight_error}, {proto_error}")
        if not torch.allclose(actual_weights.float().sum(-1), torch.ones_like(actual_weights[..., 0]).float(), atol=1e-3):
            raise RuntimeError("Patch weights do not sum to one")

    def numpy(value):
        return value.detach().float().cpu().numpy()

    mean = torch.tensor(dataset.data_mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(dataset.data_std).view(1, 1, 3, 1, 1)
    rgb = (videos * std + mean).clamp(0, 1).permute(0, 1, 3, 4, 2).numpy()
    # Dataset normalization divides coordinates by width/height, not width-1.
    xy = (meta["pred_tracks"][query_index].numpy() + 1) / 2
    xy *= np.array([rgb.shape[3], rgb.shape[2]])
    arrays = {
        "query_rgb": rgb[query_index], "support_rgb": rgb[support],
        "query_xy_pixels": xy,
        "query_visibility": meta["pred_visibility"][query_index].bool().numpy(),
        "query_routing_mask": meta["pred_query_mask"][query_index].bool().numpy(),
        "point_indices": meta["point_indices"][query_index].numpy(),
        "post_text_cosine": numpy(cosine_text[args.query]),
        "post_fused_cosine": numpy(cosine_fused[args.query]),
        "post_text_weights": numpy(weights_text[args.query]),
        "actual_query_weights": numpy(actual_weights[args.query]),
        "query_transport_weights": numpy(aux["query_patch_transport_weights"][args.query]),
    }
    info = {
        "revision": commit, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "checkpoint_run_id": cfg.TRAIN.CHECKPOINT_RUN_ID,
        "episode": args.episode, "seed": cfg.RNG_SEED, "query_position": args.query,
        "highlight_frame": args.frame, "class_ids_zero_based": class_ids.tolist(),
        "class_names": [model.atomic_label_names[c] for c in class_ids],
        "text_prompts": [model.atomic_label_prompt_groups[c] for c in class_ids],
        "query_video_name": meta["video_name"][query_index],
        "query_video_path": dataset._path_to_videos[int(episode_spec[query_index][0])],
        "query_feature_path": dataset._feat_paths[int(episode_spec[query_index][0])],
        "query_labels_in_episode": episode_labels[query_index].int().tolist(),
        "query_all_clip_labels": [model.atomic_label_names[c] for c in labels[query_index].nonzero().flatten()],
        "support_video_names": [v for v, s in zip(meta["video_name"], support) if s],
        "support_labels_in_episode": episode_labels[support].int().tolist(),
        "episode_dataset_indices": [int(item[0]) for item in episode_spec],
        "frame_softmax_tau": cfg.FEW_SHOT.POT_ROUTE.FRAME_SOFTMAX_TAU,
        "routing_mask_mode": cfg.POINT_INFO.QUERY_MASK_MODE,
        "loaded_checkpoint_tensors": len(state),
        "missing_frozen_tower_keys": len(result.missing_keys),
        "actual_vs_recomputed_weight_max_abs_error": weight_error,
        "actual_vs_recomputed_prototype_max_abs_error": proto_error,
        "base_logits": numpy(aux["query_partial_q2s_base_logits"][args.query]).tolist(),
        "verified_logits": numpy(aux["query_partial_q2s_logits"][args.query]).tolist(),
        "notes": [
            "One deterministic test episode, no selection by model score.",
            "Clip labels are not per-frame or spatial ground truth.",
            "Text-only Query weights are a diagnostic; actual Query uses fused Support+text.",
            "Cyan top-8 is a display aid, not a hard top-k operation in the routing model.",
            "Post tokens include 8-frame temporal and spatial context.",
            "Coordinates are tracked positions; duplicate/out-of-view/invisible tokens are retained.",
            "Conditional weights are before frame-level m*rho transport; transport weights are saved separately.",
        ],
    }
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite visualization data: {name}")
    if args.video_labels:
        # Labels select visualization rows only AFTER the normal forward pass.
        # No Query GT is injected into routing or used to invent Support data.
        video_classes = labels[query_index].nonzero().flatten().long()
        with torch.inference_mode(), torch.autocast("cuda", enabled=cfg.TRAIN.MIXED_PRECISION):
            true_text = model._get_pot_label_text_features(video_classes.cuda(), post.dtype)
            selected_post = post[query_index:query_index + 1]
            selected_mask = meta_gpu["pred_query_mask"][query_index:query_index + 1]
            true_cosine = model._compute_batched_point_text_similarity(selected_post, true_text)
            _, true_weights = model._compute_batched_frame_softmax_text_prototypes(
                selected_post, selected_mask, true_text)
        original_episode = {k: info.pop(k) for k in (
            "class_ids_zero_based", "class_names", "text_prompts", "query_labels_in_episode",
            "support_video_names", "support_labels_in_episode", "base_logits", "verified_logits")}
        info.update(original_episode=original_episode,
                    class_ids_zero_based=video_classes.tolist(),
                    class_names=[model.atomic_label_names[c] for c in video_classes],
                    text_prompts=[model.atomic_label_prompt_groups[c] for c in video_classes],
                    visualization="Pure-text vs Post patch for the video's annotated labels; no Support fusion")
        info["notes"] = [
            "Only the video's ground-truth clip labels select the visualization rows, after inference.",
            "Eight-frame Post features are unchanged; Query GT does not condition the forward pass.",
            "Pure-text softmax is a diagnostic, not the actual Support-fused Query route.",
            "Clip labels are not per-frame or spatial ground truth.",
            "Cyan top-8 is display only; tracker-invisible and duplicate tokens are retained.",
        ]
        arrays = {k: arrays[k] for k in ("query_rgb", "query_xy_pixels", "query_visibility",
                                       "query_routing_mask", "point_indices")}
        arrays.update(post_text_cosine=numpy(true_cosine[0]), post_text_weights=numpy(true_weights[0]))
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "checkpoint_config.yaml").write_text(saved_cfg)
        print("Rendering annotated video classes:", info["class_names"], flush=True)
        save_video_labels(args.output, arrays, info)
        print("Completed:", args.output, flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoint_config.yaml").write_text(saved_cfg)
    np.savez_compressed(args.output / "patch_scores.npz", **arrays)
    with (args.output / "patch_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_zero_based", "class_id", "class_name", "clip_label", "token", "source_point",
                         "x_224", "y_224", "tracker_visible", "routing_valid", "post_text_cosine", "post_fused_cosine",
                         "text_only_weight", "actual_query_weight", "transport_weight"])
        for t in range(cfg.DATA.NUM_FRAMES):
            for k, cid in enumerate(class_ids.tolist()):
                for n in range(xy.shape[1]):
                    writer.writerow([t, cid, info["class_names"][k], info["query_labels_in_episode"][k], n,
                                     int(arrays["point_indices"][n]), *xy[t, n],
                                     int(arrays["query_visibility"][t, n]), int(arrays["query_routing_mask"][t, n]),
                                     arrays["post_text_cosine"][k, t, n], arrays["post_fused_cosine"][k, t, n],
                                     arrays["post_text_weights"][k, t, n], arrays["actual_query_weights"][k, t, n],
                                     arrays["query_transport_weights"][k, t, n]])
    # Release the inference GPU before rendering the figures.
    del model, video_gpu, meta_gpu, post, aux, query_post
    print("Inference validated; rendering measured scores", flush=True)
    render(args.output, arrays, info)
    (args.output / "provenance.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps({"output": str(args.output), "weight_error": weight_error,
                      "prototype_error": proto_error, "query": info["query_video_name"]}), flush=True)


def main():
    args = arguments()
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    if not args.checkpoint.is_absolute():
        args.checkpoint = args.repo / args.checkpoint
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    os.chdir(args.repo)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    with tempfile.TemporaryDirectory(prefix="trokens_post_viz_") as scratch:
        commit = export_revision(args.repo, args.revision, Path(scratch))
        sys.path.insert(0, scratch)
        print("Isolated historical code:", commit, flush=True)
        run(args, commit)


if __name__ == "__main__":
    main()
