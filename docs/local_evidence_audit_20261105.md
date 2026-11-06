# SAV local competition evidence audit (2026-11-05)

> 历史记录，非当前运行说明。2026-11-06 已删除废弃实验代码和旧配置；当前入口及恢复方法见 [ρ 收敛记录](rho_cleanup_20261106.md)。本文原始实验叙述保留，不代表当前模型。

This is frozen-checkpoint inference and CPU analysis, not training. Existing
working-tree training changes and SAV/TinyVIRAT YAMLs are untouched.

## Remote deployment

### Current decision (supersedes the initial 14 plan below)

The user requested using **69** after reviewing the small logit differences.
The selected deployment is now:

- `liborui@10.16.50.69:22`, physical GPUs **0,1**, both RTX A6000.
- Runtime: `/data2/liborui/codes/trokens-main/output/sav/local_evidence_audit_20261105_69_v2/runtime`.
- Results: `/data2/liborui/codes/trokens-main/output/sav/local_evidence_audit_20261105_69_v2/run`.
- Python: `/home/liborui/miniconda3/envs/trokens/bin/python`.
- First run 16 untouched-forward episodes per rank; the instrumented replay
  must match those same-host outputs within `2e-5`.
- Preserve a separate, disclosed cross-machine audit against all 10,000 old
  predictions: maximum absolute logit change <0.05, mean <0.005, each Base/Novel/HM
  change <0.05 percentage points, and positive/negative pair-order change <0.5%.
  These thresholds were locked before full replay in the v2 protocol; they
  do not claim bitwise equality. Any failure stops before development export.
- Then export 10,000 development episodes and run CPU statistics automatically.
- No automatic run on 14. Its strict smoke also failed (episode 0: 0.00875354;
  episode 1: 0.00471854), so identical GPU model alone did not restore exact
  outputs. These remain logit discrepancies, not measured AP/HM losses.

Use the 69 `run/status.json`, not the prepared 14/v1 directories, for current
progress. The command example at the end documents the earlier 14 deployment
and is **not** the current launch command.

### Initial migration history

- Primary after migration checks: `xuxin@172.24.2.14:22`, physical GPUs **2,3**.
- Main checkout: `/data1/xuxin/trokens-main` (not modified).
- Isolated bundle: `/data1/xuxin/trokens-main/output/sav/local_evidence_audit_20261105_14_v1/runtime`.
- Results/logs: `/data1/xuxin/trokens-main/output/sav/local_evidence_audit_20261105_14_v1/run`.
- Python: `/data1/xuxin/envs/trokens/bin/python`.
- Fixed rho-only epoch 3 (checkpoint epoch field 2), SHA256
  `fbd5809b7df83aaa8debe5285e790bc7c7c00a405eb7be7fcc388db03b89a73a`.

No password is stored in these scripts or in the handoff.

Server 69 (`liborui@10.16.50.69`, checkout
`/data2/liborui/codes/trokens-main`) was checked first. Its A6000 strict smoke
failed historical score equality: maximum absolute differences were
0.0034633875 (episode 0) and 0.0065181255 (episode 1). This is **not** a measured
performance loss. The first 32 episodes' decoded inputs and trajectory metadata
hashes matched the original machine exactly, and frozen weights/source matched.
This supports, but does not by itself prove, cross-GPU numerical variation.
The full 69 diagnostic was not launched. Its v1/v2 preparation artifacts are
retained; do not mistake them for completed experiments. Server 14 uses A100,
matching the original machine. The 14 protocol keeps the strict `2e-5` score
check rather than the optional, explicitly separate cross-hardware audit mode.

## Automatic stages

1. Replay all 10,000 original episodes using two preserved even/odd streams,
   with seed 1 and four data workers per stream. Compare class IDs, labels and
   original scores against the saved historical cache by episode ID.
2. Only after replay passes, export 10,000 development episodes. This uses a
   fixed half-split of the original test pool, grouped by the first two filename
   components (seed 261105); train membership is unchanged. Development has
   379 groups/18,338 effective videos, reserved has 379 groups/18,971 videos.
   Both contain all 15 classes, and no retained full label combo has fewer
   than two distinct videos. This partition is **not historically untouched**:
   the original test pool already influenced checkpoint selection/diagnostics.
3. Run CPU analysis, write `result.json`, `REPORT.md`, `compact_statistics.npz`,
   `examples.json`, and progress records. The reserved set is never evaluated.

No optimizer, new training, score-formula tuning or automatic next experiment.
Failures stop this run and its own child processes only, recording `ERROR.txt`.

## Export correctness

The original frozen `_metric_relation_chunk` is called and its outputs returned
unchanged. Additional math, in the same AMP context, records scaled per-patch
positive-minus-negative evidence and patch weights. Explicit class indices and
query chunk offsets identify records. Identical weight rows are never used as
identifiers. Frozen one-outer-chunk layout and reconstruction are asserted.

Float32 `sum(pi * ell)` must match the original relation logits within `2e-5`.
Patch arrays are saved in float16; logits, rho and scores remain float32.
Every serialized batch is reloaded and its reconstruction error measured
separately (maximum allowed 0.02, actual maximum saved in worker status).

Only Support labels affect the verifier. Query truth is used for original
evaluation targets and subsequent CPU groups, not for competitor selection.

## Analysis and preregistered decision

- `c`: target attention-weighted strongest other-class evidence.
- `g`: target attention-weighted excess of strongest other-class evidence over
  own evidence; **primary** statistic, with c secondary.
- Missing reference classes are excluded; unavailable evidence is NaN, not zero.
- A/B/C/D groups use episode-candidate labels, not global label cardinality.
- Classwise A-vs-B/A-vs-C AUC, distributions, and original episode-mAP.
- Matched positive/negative Query pairs in the same episode/class use all-valid
  rho frames, mean absolute rho difference <=0.05, base/final logit differences
  <=0.25. Compare against the *same cohort's* rho mean and original final score,
  not a hard-coded historical 54.6% reference. Ties count as half.
- Five seeded valid-point shuffles and a per-frame valid-point mean-field
  control preserve own ell/pi and never consume the model RNG.
- Episode-paired bootstrap (400 draws) for primary ranking and class-macro AUC
  differences, plus A-vs-B attention overlap difference. These intervals are
  descriptive: clips recur across episodes and only one checkpoint is used.
- Overlap uses `sum(min(pi_k, pi_j))` with known-present other candidate labels;
  saves both per-target summaries and all pair identities.
- Calibration uses ten shared logit intervals across classes, with counts,
  positive rates and cross-class spread; no class-specific quantile boundaries.

The decision requires >=1,000 close pairs and enough A/B/C examples. Otherwise
it is INCONCLUSIVE. Conditional on coverage, primary A/B AUC >=0.60, matched
ranking >=0.58, B/C standardized mean gap <=0.30, and positive bootstrap lower
bounds versus the same-cohort rho, spatial controls and A/B overlap must all
pass to recommend *score testing only*. Failure stops this particular frozen
competition route; it does not prove all representation-learning approaches
impossible. Thresholds are saved before inference in `protocol.json`.

## Operations

Read the remote `run/status.json` and `run/{replay,development}/rank*/status.json`
for authoritative progress, and `run/runner.log` / per-GPU logs for errors.
Do not run the command twice. `runner.lock` protects against duplicate launch.
Create `run/STOP` to request a stop of this diagnostic only; a terminal failure
preserves partial shards and does not restart automatically.

If launching a **fresh, separately prepared** directory manually on 14:

```bash
cd /data1/xuxin/trokens-main/output/sav/local_evidence_audit_20261105_14_v1/runtime
TORCH_HOME="$PWD/.torch-cache" /data1/xuxin/envs/trokens/bin/python tools/run_local_evidence_audit.py --root ../run
```

Do not use this as a resume command over a partial run; the exporter refuses to
overwrite existing stage directories. New protocols should use new run paths.

Validation before deployment: 8 synthetic CPU tests passed, including masked
evidence, exclusion of self/missing references, RNG-isolated shuffles, ties and
weighted AUC, candidate grouping, matched-pair validity, and full analysis with
insufficient pairs. Historical reproduction and data-dependent evidence results
are separate from these unit tests.
