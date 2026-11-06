# SAV C: positive-match-conditioned counter-evidence

> 历史记录，非当前运行说明。2026-11-06 已删除废弃实验代码和旧配置；当前入口及恢复方法见 [ρ 收敛记录](rho_cleanup_20261106.md)。本文原始实验叙述保留，不代表当前模型。

## Status and scope

Implemented opt-in C (original rho + pair competition with feature gradients).
No experiment has been launched by this change. CPU tests do not establish GPU
memory/performance, successful full training, or a recognition gain.

Use `configs/trokens/sav_competition_c_s1.yaml`. It inherits `sav.yaml`; after
resolution the only differences are `PAIR_COMPETITION.ENABLE` and `OUTPUT_DIR`.
The current SAV protocol is 5-way 1-shot, 6 training Queries/class, LR 3e-5,
10 epochs, 3 warmup epochs, cosine ending at 1e-6, 2,000 episodes/epoch,
10,000 test episodes, AdamW, accumulation 2. Keep the same launch-time GPU
count, accumulation and other overrides as the baseline. YAML currently has
NUM_GPUS=1; a previous launcher may have overridden it to 2. Merely using the
same YAML LR does not make different effective episode/update budgets equal.

Both role decouplings and original rho remain ON; m/HoD/CTD remain OFF,
appearance gain 1.0. No changes were made to the user's `sav.yaml`.

## Computation

Use the main pooled Query `[Q,K,T,D]`, individual pooled Support `[S,K,U,D]`
and unchanged positive prototype `[K,U,D]`. Negative membership uses only
known episode Support labels. All references are under the same candidate k.

With unit vectors, for each positive prototype frame u:

```
a[k,u,r] = stopgrad(softmax_r(cos(p[k,u], n[k,r]) / tau))
C[q,k,t,u] = cos(q[q,k,t], p[k,u])
E[q,k,t,u] = sum_r a[k,u,r] * cos(q[q,k,t], n[k,r])
P = tau * softplus((E - C) / tau)
B = min(C * effective_original_rho_mass, C)
B_new = B - lambda_effective * P
z_new = z_original_rho + alpha * (BiMHM(B_new) - BiMHM(B))
```

`tau=0.1`, final `lambda=0.5`, alpha is the existing matching scale (10).
Lambda uses the same first-5%-of-updates warmup progress as original rho.
Evaluation uses full strength. LR warmup remains the original three epochs.

Main pooling/positive prototypes and the original rho's patch metric/gradient
boundaries are unchanged. New comparison gradients bypass rho's detached
reference tensors and flow through the MAIN pooled frames to Post encoders
and readout. DinoTxt remains frozen. Selection weights are recomputed each
forward but are not differentiated. No extra learned head/auxiliary loss is
introduced: original global/base/final/rho losses remain as before.

Unknown labels cannot be negatives. Invalid Query/reference frames, missing
positive references or no negative references contribute zero new penalty.
These evidence masks do NOT change the historical main BiMHM masks. Zero
strength returns original rho scores exactly. No Query truth enters the API.

## Controls (implemented, NOT queued)

* A: `PAIR_COMPETITION.ENABLE=False` (same original rho).
* B: `DETACH_INPUTS=True` (same forward, no new comparison feature gradient).
* C: `DETACH_INPUTS=False`, `REDUCTION=pair`, `SELECTION=conditioned`.
* D: `REDUCTION=mean` (same P computed, broadcast full-matrix mean including
  zero invalid entries; exactly a video/class score offset).
* `SELECTION=mean_weights`: share the mean conditioned negative weights over
  valid positive time positions. `uniform`: all valid negative frames equal.

The controls have distinct checkpoint signatures. Do not silently load old
rho-only checkpoints or use another arm's config. This experiment starts a new
task-network training run; use a matching saved config for resume/evaluation.

## Manual launch

From `/data3/duanwei/trokens-main`, keep your existing launch command and replace
only its `--cfg` argument with `configs/trokens/sav_competition_c_s1.yaml`.
Output is `output/sav/competition_c_s1`. Use a fresh output directory if already
used; do not overwrite or automatically resume mixed experiments.

Example matching the current **single-GPU YAML**, not an implicit two-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_HOME=/data3/duanwei/trokens-main/.torch-cache WANDB_MODE=disabled \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/run_net.py \
  --cfg configs/trokens/sav_competition_c_s1.yaml
```

If the baseline was launched on two GPUs, use the same two-GPU launcher and
NUM_GPUS override for both models. Learning rate/epoch equality alone does not
control differences in shot, Query count, accumulated episodes or data split.

Logs include positive/negative-group penalty means, Support-column variation,
logit deltas, eligible fractions, and both directional winner-change fractions
through the existing training/evaluation metric logger.

## Evaluation boundary

At the user's request, data split and checkpoint selection match current SAV:
training-time evaluation uses the existing shortened TEST loader every epoch,
and the final test evaluates the selected checkpoint on 10,000 episodes.
This preserves historical comparability but retains test-set-selection bias;
it is not an untouched independent validation/test protocol. The discarded
held-out-development draft is not used by this config.

One C run cannot establish that pairing/gradient feedback causes improvement.
Those claims still require matched B/D and reference-selection controls.

## Checks

`tests/test_pair_competition.py` checks scalar math, main-similarity parity,
unknown/empty masks, no-Query-truth leakage, positive/negative/Query gradient
paths, no selection gradient, B/C forward parity, zero-strength identity,
mean-offset identity, full/selective-checkpoint encoder backward, signature
guards, configuration equality, and CPU two-rank/two-update accumulation.

Six older tests requiring historical all-off/motion-on main configs already
fail on unmodified HEAD plus the user's current sav.yaml. They were reproduced
in an isolated `/tmp` checkout; this implementation does not change current
configs to satisfy those obsolete expectations.
