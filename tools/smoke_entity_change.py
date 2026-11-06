"""One real episode preflight; no checkpoint is saved or reused for training."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/trokens/sav.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import numpy as np
    import torch
    from trokens.config.defaults import get_cfg, assert_and_infer_cfg
    from trokens.datasets import loader
    from trokens.models import build_model
    from trokens.models.optimizer import construct_optimizer
    from trokens.utils.misc import iter_to_cuda
    from tools.few_shot_multilabel import (get_episode_class_ids, episode_labels_from_global,
        single_path_episode_targets, compute_query_partial_q2s_loss, multilabel_classification_loss)
    cfg = get_cfg(); cfg.merge_from_file(args.cfg); cfg = assert_and_infer_cfg(cfg)
    cfg.NUM_GPUS = 1; cfg.DATA_LOADER.NUM_WORKERS = 0
    torch.set_num_threads(1); torch.manual_seed(cfg.RNG_SEED); np.random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    start = time.monotonic()
    dl = loader.construct_loader(cfg, 'train')
    inputs, labels, _, meta = next(iter(dl))
    inputs, labels, meta = iter_to_cuda([inputs, labels, meta])
    meta['support_mask'] = torch.as_tensor(np.asarray(meta['sample_type']) == 'support', device='cuda')
    ids = get_episode_class_ids(meta, labels.device)
    meta['raw_positive_labels'] = labels.float()
    meta['episode_positive_labels'] = episode_labels_from_global(labels, ids)
    model = build_model(cfg).eval()
    print('Real model and 35-video train episode loaded.', flush=True)
    batch = dict(video=inputs, metadata=meta)
    amp = lambda: torch.autocast('cuda', dtype=torch.float16)
    # Warm text cache, then compare the identical encoder with/without new readout.
    with torch.no_grad(), amp():
        model._get_pot_label_text_features(ids, torch.float16)
        enabled = model(batch)['query_partial_q2s_logits'].clone()
        change = model.entity_change_verifier
        model.entity_change_verifier = None
        disabled = model(batch)['query_partial_q2s_logits']
        model.entity_change_verifier = change
    error = (enabled-disabled).abs().max().item()
    assert error == 0., ('lambda-zero identity failed', error)
    del enabled, disabled
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model.train(); optimizer = construct_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(init_scale=128.)
    tick = time.monotonic()
    with amp():
        out = model(batch)
        targets = single_path_episode_targets(out, labels, meta)['query_batch_labels']
        qloss, _, diagnostics = compute_query_partial_q2s_loss(out['query_partial_q2s_logits'], targets, out, cfg)
        global_loss = multilabel_classification_loss(out['global_classification_logits'], labels, cfg,
                                                     torch.nn.BCEWithLogitsLoss())
        loss = global_loss + qloss
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    gradients = {name: float(p.grad.detach().norm()) for name,p in change.named_parameters() if p.grad is not None}
    assert len(gradients) == 6 and all(np.isfinite(list(gradients.values())))
    unused = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(json.dumps(dict(unused_parameters=unused, nonfinite_gradients=nonfinite)), flush=True)
    assert not nonfinite, nonfinite
    scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
    result = dict(state='PASS', real_episode_videos=len(labels), physical_gpu_env=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        lambda_zero_max_error=error, finite_gradients=True, scalar_gradient_norms=gradients,
        lambda_after_one_smoke_step={n:float(p.detach()) for n,p in change.named_parameters() if n.startswith('lambda')},
        loss=float(loss.detach()), unused_parameters=unused,
        diagnostics={k:float(v) for k,v in diagnostics.items() if 'entity' in k},
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        train_step_seconds=time.monotonic()-tick, total_seconds=time.monotonic()-start,
        retained_checkpoint=False, note='Smoke weights discarded; main run starts fresh.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
