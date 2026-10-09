# SAV / TinyVIRAT trajectory-mask and AMP fix

The shared implementation handles backward-tracked observations, empty frames,
and completely invisible trajectories without inventing valid points. It does
not change labels, class prompts, episode sampling, or Base/Novel AP filtering.

## Explicit mask policy

With `POINT_INFO.USE_PT_QUERY_MASK: True`, `QUERY_MASK_MODE` selects:

| Mode | Meaning |
| --- | --- |
| `visibility` | Use the stored tracker visibility, including visible points before their seed frame. Selected by the TinyVIRAT YAML. |
| `seed` | Admit a trajectory only at/after its seed frame. For forward-only tracking; default for other configs. |
| `all` | Explicit unmasked-routing control, reproducing historical SAV behavior for dumps containing `point_queries` but no `per_point_queries`. Selected by the SAV YAML for the historical-mask ablation. |

The loader recognizes both key spellings, but key spelling alone is NOT used
to infer tracking direction. The current nongrid CoTracker producer requests
backward tracking. TinyVIRAT retains the visibility policy. On 2026-10-08, SAV
was switched back to `all` for a historical-mask ablation, retaining the
numerical fixes below and leaving temporal refinement and both Query judgments
disabled. This is a mask-policy restoration, not a full historical-code rollback.

Historical SAV's loader read only `per_point_queries`; these dumps contain
`point_queries`, so it defaulted seed indices to zero and produced all-True
query masks. Thus the new visibility policy is **not numerically identical to
historical SAV** and should be reported as a mask-policy change in experiments.
For a historical-policy ablation set `POINT_INFO.QUERY_MASK_MODE all` explicitly
in the target dataset config, while retaining the numerical fixes. Visibility still has
its original separate consumers (motion/evidence); `all` is not global removal
of every visibility check.

## Numerical changes

- Both scalar and batched routed aggregation compute in FP32 outside autocast.
- Empty frames get zero prototypes and zero routed gradient. Nonempty weighted
  averages keep their existing denominator; fractional weights are not clamped
  to one. Fallback integer point counts use a safe denominator of at least one.
- Attention evaluates all-masked softmax rows on finite constants and returns
  zero attention mass/output for those rows, including after projection bias.
  Nonempty rows retain their existing attention equation.
- Training logs scaler drops/skipped optimizer updates, epoch update counts,
  and AMP scale. `SOLVER.AMP_MIN_SCALE: 0.0001` in both YAMLs stops pathological
  scale collapse and rejects the previous failed scaler state on resume. The
  global default is 0 (no floor). The guard does not reset checkpoints silently.

## Restart and verification

Both dataset YAMLs retain `TRAIN.AUTO_RESUME: False` and an empty
`TRAIN.CHECKPOINT_FILE_PATH`. Use a separate output directory, e.g.
`OUTPUT_DIR output/tinyvirat_maskfix`, for a fresh experiment. Do not resume the
stalled run's optimizer/scaler state. Existing checkpoints and logs are retained.

On 2026-09-22, 119 tests passed with the trokens interpreter and GPU 4 exposed,
including CUDA FP16/BF16 empty-prototype backward tests and masked-attention
forward/backward tests:

```bash
CUDA_VISIBLE_DEVICES=4 /home/duanwei/miniconda3/envs/trokens/bin/python -m pytest -q tests
```

Three real episodes from each dataset were checked after the fix, using saved
weights but no optimizer steps. All 37 q2s-connected parameter tensors had
finite, nonzero gradients in each episode. Both datasets were also checked at
effective loss scale 32768 on the production path (no in-memory replacements).
This verifies gradient recovery, **not** recovery of historical
mAP. The visibility-policy effect and final accuracy require new controlled runs.
