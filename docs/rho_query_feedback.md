# Original rho with Query-only evidence feedback

> 历史记录，非当前运行说明。2026-11-06 已删除废弃实验代码和旧配置；当前入口及恢复方法见 [ρ 收敛记录](rho_cleanup_20261106.md)。本文原始实验叙述保留，不代表当前模型。

Config: `configs/trokens/sav_rho_grad_query_s1.yaml` (standalone, no YAML base).
This is a training-path experiment, not another competition scoring module.

## What changes

`FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.RHO_QUERY_FEEDBACK: True`
bypasses `DETACH_INPUTS` only for the original rho's Query Post patches and
Query spatial readout weights. Support evidence construction and ingestion
remain detached (`DETACH_INPUTS: True`). Both the existing relation MIL loss
and the final matching loss can now differentiate through rho into Query.

Support's normal matching and global-classification paths still train.
The encoder is shared: stopping the Support reference gradient does not freeze
the encoder or keep future Support references constant. Frozen DinoTxt stays
frozen; the feedback trains the existing downstream space-time encoder.

No extra parameters, new evidence formula, loss, difficulty weighting, or
gradient warmup are introduced. With identical weights/inputs, forward values
are unchanged. The existing objective remains:

`L_global + 0.6 L_base + 0.4 L_final + 0.05 L_rho`.

The existing 5% transport warmup still scales only the matching correction.
The rho MIL gradient is active from the first update, independent of that
transport warmup. The LR warmup remains 3 epochs.

## Fixed protocol

Matches the archived `rho_only.yaml` from
`output/sav/decoupling_ablation_lr3e5_e10_w3_20261103_031350/`:

- SAV, 5-way 1-shot, seed 1, frozen DinoTxt ViT-L/14 reg4, 8 frames.
- Query and Support decoupling on; original rho on; m and pair competition off.
- HoD/CTD off; no appearance amplification (gain 1).
- LR 3e-5; cosine schedule, 10 epochs, 3-epoch LR warmup, end LR 1e-6.
- 2,000 global episodes/epoch; 2 GPUs, accumulation 2/rank; 500 updates/epoch
  before any AMP skips. Train queries/class 6, test queries/class 1.
- 10,000 formal test episodes; original label splits/masks/loss weights.
- Fresh training, no automatic resume; output `output/sav/rho_grad_query_s1`.

Existing SAV/Tiny/competition configurations and launcher defaults are not
changed. The new flag defaults to false for all existing configurations.
Invalid combinations (m on, either role off, rho off, competition on, or
`DETACH_INPUTS: False`) fail at config validation.

A distinct checkpoint signature records the new training semantics. Use this
config when testing/resuming its own weights; normal loading rejects crossing
old-rho/new-feedback profiles, despite their identical learned tensor shapes.
This experiment starts fresh rather than resuming the old rho optimizer state.

## Run manually

From the repository root, when GPUs 0 and 1 are available:

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_rho_grad_query_s1.yaml \
  --gpus 0,1 --wandb-mode disabled
```

Append `--dry-run` to validate/print the command without launching training.
Use a new `--output-dir` if this output path already contains a previous run.
No real training or checkpoint evaluation is launched by this change.

## Verification and interpretation

`tests/test_rho_query_feedback.py` checks exact forward/initialization parity,
the Query/Support gradient boundaries (including early Support pooling),
unchanged loss algebra, empty/unknown-label cases, Query-truth invariance,
checkpoint guards, activation checkpointing, and two-rank CPU DDP accumulation.
These use synthetic tensors and do not establish accuracy or GPU memory needs.

After retraining, compare Base/Novel/HM on matched episodes and the same
checkpoint-selection rule. Also compare the same trained weights with normal
rho versus effective mass=1, and examine false positives and true co-occurring
actions. A higher final HM alone would not prove that localization improved or
that absent-class representations became more discriminative. The historical
pipeline selects checkpoints using test-split evaluation; this config preserves
it for comparability, not as a claim of independent held-out model selection.
