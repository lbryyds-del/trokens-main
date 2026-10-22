#!/usr/bin/env python3
"""Run specified true-label examples sequentially and build a local gallery."""

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True, help="episode:query pairs")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-note", default="Samples explicitly selected before inference.")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pairs = [tuple(int(x) for x in item.split(":")) for item in args.samples]
    if any(len(p) != 2 or min(p) < 0 for p in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("Expected unique nonnegative episode:query pairs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"selection_note": args.selection_note, "requested_samples": args.samples,
                "mode": "video_labels", "samples": []}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for number, (episode, query) in enumerate(pairs, 1):
        name = f"{number:02d}_ep{episode}_q{query}"
        folder = output / name
        command = [sys.executable, str(repo / "tools/visualize_post_patch.py"),
                   "--gpu", args.gpu, "--episode", str(episode), "--query", str(query),
                   "--video-labels", "--output", str(folder)]
        print(f"[{number}/{len(pairs)}] Starting {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        meta = json.loads((folder / "provenance.json").read_text())
        manifest["samples"].append({"folder": name, "video": meta["query_video_name"],
                                    "class_names": meta["class_names"],
                                    "class_ids_zero_based": meta["class_ids_zero_based"],
                                    "checkpoint_sha256": meta["checkpoint_sha256"]})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[{number}/{len(pairs)}] Completed {meta['query_video_name']}: {meta['class_names']}", flush=True)

    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Check saved artifacts and provide an original-frame contact sheet.
    fig, axes = plt.subplots(len(pairs), 8, figsize=(20, 2.8 * len(pairs)),
                             layout="constrained", squeeze=False)
    validated_pngs = 0
    validated_values = 0
    for i, sample in enumerate(manifest["samples"]):
        folder = output / sample["folder"]
        with np.load(folder / "patch_scores.npz") as data:
            assert data["post_text_cosine"].shape == (len(sample["class_names"]), 8, 256)
            assert np.isfinite(data["post_text_cosine"]).all()
            assert np.isfinite(data["post_text_weights"]).all()
            assert np.allclose(data["post_text_weights"].sum(-1), 1, atol=1e-5)
            validated_values += data["post_text_cosine"].size
            for t, ax in enumerate(axes[i]):
                ax.imshow(data["query_rgb"][t])
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"Frame {t + 1}")
            axes[i, 0].set_ylabel(f"Video {i + 1}\n" + "\n".join(sample["class_names"]), fontsize=10)
        for path in folder.glob("*.png"):
            with Image.open(path) as im:
                im.verify()
            validated_pngs += 1
    fig.suptitle("Additional videos: original 8 sampled frames and each video's annotated labels")
    fig.savefig(output / "video_contact_sheet.png", dpi=150)
    plt.close(fig)
    manifest.update(validated_pngs=validated_pngs, validated_patch_scores=validated_values)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8">',
             '<title>后解耦：更多视频的真实类别可视化</title>',
             '<style>body{font-family:sans-serif;margin:24px;line-height:1.7}img{width:100%;height:auto}'
             'section{border-top:1px solid #bbb;margin-top:30px}a{margin-right:14px}</style>',
             f'<h1>{len(pairs)} 个新增视频：各自真实标签 × 完整 8 帧</h1>',
             '<p>使用 a5134b8 历史代码与 20261011_133133 归档权重。纯文本/Post patch 诊断，'
             '不是 Support 融合后的 Query 权重。视频标签不是逐帧或区域真值。</p>',
             '<p>同一视频内，各类别与各帧共用色标；不同视频的色标范围可能不同。'
             '青圈是 top-8 展示标记，x 是跟踪不可见点。</p>',
             '<p>样本按标签覆盖选择，没有根据模型分数或热图效果筛选。</p>',
             '<a href="video_contact_sheet.png">全部视频原图预览</a>',
             '<a href="manifest.json">样本清单</a>']
    md = ["# 新增视频可视化", "", "每个视频只展示自身真实标签；每类完整 8 帧。", "",
          "纯文本/Post patch 诊断。视频内共用色标，跨视频的色标范围可能不同。", ""]
    for i, sample in enumerate(manifest["samples"], 1):
        folder = sample["folder"]
        labels = ", ".join(sample["class_names"])
        parts += [f'<section><h2>{i}. {html.escape(labels)}</h2>',
                  f'<p>{html.escape(sample["video"])}</p>',
                  f'<a href="{folder}/all_frames_cosine.png">8 帧相似度总览</a>',
                  f'<a href="{folder}/all_frames_weights.png">8 帧 softmax 总览</a>']
        md += [f"## {i}. {labels}", "", sample["video"], "",
               f"[8 帧相似度]({folder}/all_frames_cosine.png) · [softmax 权重]({folder}/all_frames_weights.png)", ""]
        for label in sample["class_names"]:
            parts.append(f'<a href="{folder}/class_{label}_8frames.png">{html.escape(label)} 单类大图</a>')
        parts += [f'<p><a href="{folder}/patch_scores.csv">原始 patch 数值</a></p>',
                  f'<img loading="lazy" src="{folder}/all_frames_cosine.png" alt="{html.escape(labels)}"></section>']
    parts.append('</html>')
    (output / "index.html").write_text("\n".join(parts))
    (output / "README.md").write_text("\n".join(md))
    print(f"Validated {validated_pngs} PNGs, {validated_values} patch scores; gallery: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
