# SAV trajectory-cluster change pilot (2026-11-06)

One fresh seed-1 training run, not a multi-seed claim. The old `sav.yaml`,
TinyVIRAT config, sampler, original rho matching and backbone remain unchanged.

## Locked method

`final = original_rho_final + lambda_state * state + lambda_geo * geometry`.
There is **no Support reliability gate**. Both signed lambdas start at exactly
zero. Temperature/scale start at 0.1/10 through inverse-softplus initialization.
The six new scalars use 100x the original learning-rate schedule and zero decay:
peak 3e-3 versus 3e-5. Original parameter groups and their decay are unchanged.
`set_lr()` preserves the multiplier, including after optimizer-state resume.

Input Post features, attention and geometry are detached. Auxiliary losses can
update the new scalars, not the encoder directly. Fused final BCE nevertheless
changes gradients through the original main score. Lambda=0 guarantees forward
identity, **not the same total loss**, because channel auxiliary BCE is added.

### State evidence

`obj_ids` denotes visual feature clusters, not verified object instances.
Stable slots are sorted by number of distinct `point_indices` (ties: original
cluster ID), with >=4 distinct tracks and at most 16 slots. The main sampler is
unchanged; duplicate slots do not satisfy the minimum-count threshold.

For interval t, validity is `visible[t] & visible[t+1] & visible[0]`, where
visibility is intersected with the existing routing mask. Endpoint **raw**
attention is averaged then renormalized on this common mask. Within each cluster
we pool weighted Post differences, divide by cluster mass, and unit normalize.
No pointwise spatial centering. Each interval needs >=4 distinct visible tracks,
nonzero weight and norm >1e-6. Alpha sums visibility-normalized attention per
cluster, averaged over time, and is renormalized over usable query clusters.

For each Support/class, select up to four valid clusters by alpha (stable ties).
Compare Query interval vectors against all selected positive/negative Support
intervals by count-normalized log-mean-exp of cosine. Average valid intervals
within a Query cluster, then use alpha to combine valid clusters.

### Geometry evidence

Use the **same eight selected frames** and same sampled physical tracks. No
camera compensation. Velocity is normalized [-1,1] coordinate displacement
divided by the gap in original cached-frame indices; units are not seconds.
Each physical track contributes once. Intervals require >=4 distinct visible
tracks; a valid descriptor needs motion above 1e-6 on >=3 intervals.

Descriptor: 16-bin magnitude-weighted soft direction histograms for the whole,
first half and second half of the seven intervals (48 values), plus
`log1p(mean_speed/0.01)` and `log1p(std_speed/0.01)`. Histogram blocks are
normalized separately, then the direction block is unit normalized. The last
two values are **not** included in that normalization. Pair similarity is
`0.75*cos(direction) + 0.25*exp(-mean((amplitude_q-amplitude_s)^2))`.
These fixed choices are engineering specifications, not M0-validated optima.
Static/all-invisible clusters are invalid rather than zero-motion references.

Both channels exclude unknown Support labels. Missing positive/negative
references, absent valid Query clusters and unknown Query supervision contribute
neither a residual nor an auxiliary BCE. Query truth is used only by losses and
post-hoc metrics, never by reference selection or evidence computation.

## Training/evaluation contract

Config: `configs/trokens/sav_entity_change_s1.yaml`.

- Frozen DinoTxt; dual decoupling/rho ON, m/HoD/CTD/competition/Query feedback OFF.
- 5-way 1-shot, 6 train Queries/class, 1 test Query/class; eight frames/256 slots.
- 2000 global episodes/epoch, two GPUs, accumulation 2 = 500 updates/epoch.
- Ten epochs, three warmup, original cosine schedule and AdamW decay 0.5.
- Training pool unchanged. Training-time evaluation uses the existing
  development CSV, with the original `less_iters=True` episode count.
- Save **all ten epoch checkpoints**. Automatic final evaluation is locked to
  epoch 10 (`TEST.TEST_EPOCH_NUM: 10`), on the existing reserved CSV, 10000 episodes.
- Primary historical comparison: original rho epoch 10, the same reserved CSV,
  seed, two-rank episode partition and four DataLoader workers/rank. Development
  best is supplementary; do not compare it to historical test-selected best.
- These pools/checkpoints have historical exposure. Do not describe this as a
  pristine independent holdout or a multi-seed generalization result.

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_entity_change_s1.yaml --gpus 0,1 --wandb-mode disabled
```

Use a fresh output directory for a fresh run. To explicitly resume this exact
experiment, use its full checkpoint with `TRAIN.NEW_TRAIN False`,
`TRAIN.CHECKPOINT_FILE_PATH <path>` and `TRAIN.AUTO_RESUME False`; preserve its
config and optimizer. Do not initialize training from the old rho checkpoint.

Historical epoch-10 comparator (evaluation only, separate output):

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_entity_change_s1.yaml --gpus 0,1 --wandb-mode disabled \
  --train-enable False --output-dir output/sav/entity_change_baseline_e10 \
  -- FEW_SHOT.ENTITY_CHANGE.ENABLE False TEST.TEST_EPOCH_NUM -1 \
  TEST.CHECKPOINT_FILE_PATH /data3/duanwei/trokens-main/output/sav/decoupling_ablation_lr3e5_e10_w3_20261103_031350/rho_only/checkpoints/checkpoint_epoch_00010.pyth
```

## Readouts and checks

Original score: `query_partial_q2s_rho_logits`. Channel logits/valid masks:
`entity_{state,geo}_{logits,valid}`. Log lambda, temperature, scale, valid fraction,
auxiliary BCE and mean absolute logit contribution. Parameters being nonzero
does not establish useful information or improved generalization.

In eval mode, `STATE_TEST_MULT: 0` / `GEO_TEST_MULT: 0` disables each residual
without changing the architecture or checkpoint. These are interventions on a
jointly trained model, not separately retrained ablations.

`CORRESPONDENCE_SHUFFLE` is inference-only. Permutes distinct common-visible
tracks within the same cluster, propagating the mapping to duplicate slots,
using a local fixed RNG. It also disrupts attention-to-feature association;
declining accuracy alone does not isolate physical trajectory correspondence.

Strict common-only oracle: separate eval model, `GRANULARITY: frame`,
`STRICT_COMMON_REPLAY: True`, `GEO_ENABLE: False`, `TRAIN.ENABLE: False`.
Explicitly import rho-only weights with `initialize_from_rho_only()`; this
inference helper cannot restore optimizer/epoch and normal checkpoint loading
still rejects different training architectures. The oracle borrows the original
rho metric temperature/scale, keeps the first-frame intersection and averages
seven valid interval logits. Set lambda_state=2 for the locked stage-2 coefficient.
The exact historical A/B worker streams/text-cache initialization must also be
replayed before claiming reproduction of HM 85.1973; unit tests alone are not a
10000-episode reproduction.

Changing cluster granularity, frame order, or text injection for a trained model
requires a dedicated intervention harness that changes inputs/readout only and
keeps the comparison pool fixed. Do not bypass normal checkpoint guards by
silently loading incompatible configs. Text-free geometry descriptors do not
make their class-weighted scores text-independent. A time-constant additive
Post shift cancels algebraically; nonlinear text conditioning need not cancel.

Reliability fitting, text-strength/frame-order studies and bootstrap comparisons
are future diagnostics; this implementation does not automatically run them.
