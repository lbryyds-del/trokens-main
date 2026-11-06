"""Compact geometry for existing sampled tracks. No labels, RNG or camera fit."""
import math

import torch
import torch.nn.functional as F


def distinct_slots(point_indices):
    """One representative per original track (sampling may duplicate slots)."""
    ids = torch.as_tensor(point_indices).long()
    if ids.ndim != 1:
        raise ValueError('point_indices must be [N].')
    return ~(ids[:, None].eq(ids[None, :]) & torch.ones(
        ids.numel(), ids.numel(), device=ids.device, dtype=torch.bool).tril(-1)).any(-1)


def cluster_slots(obj_ids, point_indices, max_entities, min_points):
    ids = torch.as_tensor(obj_ids).long()
    unique = distinct_slots(point_indices)
    if ids.shape != unique.shape:
        raise ValueError('obj_ids and point_indices must align.')
    eligible = [(int((ids[unique] == k).sum()), int(k))
                for k in ids.unique(sorted=True) if int(k) >= 0]
    eligible = sorted((x for x in eligible if x[0] >= min_points), key=lambda x: (-x[0], x[1]))
    out = torch.full_like(ids, -1)
    for slot, (_, k) in enumerate(eligible[:max_entities]):
        out[ids == k] = slot
    return out


def geometry_descriptor(tracks, valid, slots, point_indices, frame_indices, cfg, num_entities):
    """[3*B direction, log1p mean/std speed]; the two blocks are NOT jointly normalized.

    Velocities use normalized image coordinates per original cached-frame step.
    No global displacement subtraction. Each physical track contributes once.
    A temporal pyramid (whole/first half/second half) retains coarse order.
    """
    tracks, valid = tracks.float(), valid.bool()
    t, n, d = tracks.shape
    if d != 2 or valid.shape != (t, n) or slots.shape != (n,):
        raise ValueError('Geometry expects tracks[T,N,2], valid[T,N], slots[N].')
    dt = torch.as_tensor(frame_indices, device=tracks.device).float().diff()
    if dt.shape != (t-1,) or not (dt > 0).all():
        raise ValueError('Geometry frame indices must be strictly increasing.')
    distinct = distinct_slots(point_indices).to(tracks.device)
    velocity = tracks.diff(dim=0) / dt[:, None, None]
    speed = velocity.norm(dim=-1)
    common = valid[1:] & valid[:-1] & distinct[None] & torch.isfinite(velocity).all(-1)
    bins = cfg.GEO_NUM_BINS
    out = tracks.new_zeros(num_entities, bins * 3 + 2)
    usable = torch.zeros(num_entities, dtype=torch.bool, device=tracks.device)
    angle = torch.remainder(torch.atan2(velocity[..., 1], velocity[..., 0]), 2 * math.pi)
    position = angle * (bins / (2 * math.pi))
    lo = position.floor().long() % bins
    frac = position - position.floor()
    middle = (t-1+1)//2
    for g in range(num_entities):
        mask = common & (slots == g)[None]
        min_points = cfg.MIN_POINTS if cfg.GRANULARITY == 'cluster' else 1
        mask &= (mask.sum(-1) >= min_points)[:, None]
        moving = mask & (speed > cfg.NORM_EPS)
        if moving.any(-1).sum() < cfg.GEO_MIN_STEPS:
            continue
        histograms = []
        for start, end in ((0, t-1), (0, middle), (middle, t-1)):
            h = tracks.new_zeros(bins)
            m = moving[start:end]
            v = speed[start:end][m]
            idx, f = lo[start:end][m], frac[start:end][m]
            h.scatter_add_(0, idx, v * (1-f))
            h.scatter_add_(0, (idx+1) % bins, v * f)
            histograms.append(F.normalize(h, dim=0, eps=cfg.NORM_EPS))
        # Direction cosine and amplitude RBF will be evaluated separately.
        out[g, :3*bins] = F.normalize(torch.cat(histograms), dim=0, eps=cfg.NORM_EPS)
        values = speed[mask]  # retain static valid points in magnitude statistics
        out[g, 3*bins:] = torch.log1p(torch.stack((values.mean(), values.std(unbiased=False)))
                                     / cfg.GEO_SPEED_SCALE)
        usable[g] = True
    return out, usable


def build_entity_metadata(tracks, valid, obj_ids, point_indices, frame_indices, cfg):
    n = tracks.shape[1]
    if cfg.GRANULARITY == 'frame':
        slots, count = torch.zeros(n, dtype=torch.long), 1
    elif cfg.GRANULARITY == 'point':
        # Duplicated physical tracks share a slot rather than create references.
        _, slots = torch.unique(torch.as_tensor(point_indices), sorted=True, return_inverse=True)
        count = n  # fixed collatable size
    else:
        slots = cluster_slots(obj_ids, point_indices, cfg.MAX_ENTITIES, cfg.MIN_POINTS)
        count = cfg.MAX_ENTITIES
    result = {'entity_index': slots, 'entity_distinct': distinct_slots(point_indices)}
    if cfg.GEO_ENABLE:
        result['entity_geo'], result['entity_geo_valid'] = geometry_descriptor(
            tracks, valid, slots, point_indices, frame_indices, cfg, count)
    return result
