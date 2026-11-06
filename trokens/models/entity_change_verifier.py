"""Detached trajectory-cluster change evidence, additive AFTER original rho.

Six learned scalars only: two temperatures, two scales, two signed residual
coefficients. No reliability head, Query-label conditioning, or new encoder.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .query_matching import normalize_patch_region_weights


def _positive(raw):
    return F.softplus(raw.float()) + 1e-4


def _raw_positive(value):
    value = float(value) - 1e-4
    if value <= 0:
        raise ValueError('Temperature/scale must exceed 1e-4.')
    return torch.tensor(value + math.log(-math.expm1(-value)), dtype=torch.float32)


def _unit(x):
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _join(parts):
    return {key: torch.cat([p[key] for p in parts], 0) for key in parts[0]}


class EntityChangeVerifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        for channel in ('state', 'geo'):
            if not getattr(cfg, channel.upper() + '_ENABLE'):
                continue
            for kind in ('temperature', 'scale'):
                self.register_parameter(channel + '_' + kind + '_raw', nn.Parameter(
                    _raw_positive(getattr(cfg, channel.upper() + '_' + kind.upper() + '_INIT'))))
            self.register_parameter('lambda_' + channel, nn.Parameter(torch.tensor(
                float(getattr(cfg, 'LAMBDA_' + channel.upper() + '_INIT')), dtype=torch.float32)))

    def readout(self, patches, weights, valid, slots, distinct, point_indices=None):
        """Consume bounded [b,k,T,N,D] chunks, return compact [b,k,T-1,G,D].

        Exactly the common_only mask/weight order, restricted to each cluster.
        Distinct-slot masks affect minimum-count validity, not the original
        sampler or attention mass. Frame replay deliberately retains duplicates.
        """
        cfg = self.cfg
        patches, weights = patches.detach().float(), weights.detach().float()
        b, k, t, n, _ = patches.shape
        valid, slots, distinct = valid.bool(), slots.long(), distinct.bool()
        if slots.shape != (b, n) or distinct.shape != (b, n) or valid.shape != (b, t, n):
            raise ValueError('Misaligned entity/visibility metadata.')
        g = 1 if cfg.GRANULARITY == 'frame' else n if cfg.GRANULARITY == 'point' else cfg.MAX_ENTITIES
        membership = F.one_hot(slots.clamp_min(0), g).float() * (slots >= 0)[..., None]
        with torch.autocast(device_type=patches.device.type, enabled=False):
            pi = normalize_patch_region_weights(weights, valid)
            alpha = torch.einsum('bkn,bng->bkg', pi.mean(2), membership)
            if not cfg.STATE_ENABLE:
                return {'alpha': alpha}
            common = valid[:, 1:] & valid[:, :-1] & valid[:, :1]
            w = (weights[:, :, 1:] + weights[:, :, :-1]) * .5
            w = w.masked_fill(~common[:, None], 0.)
            # Normalize before group pooling, exactly as common_only does.
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            after = patches[:, :, 1:]
            if cfg.CORRESPONDENCE_SHUFFLE:
                if self.training:
                    raise ValueError('Correspondence shuffle is inference-only.')
                if point_indices is None:
                    raise ValueError('Correspondence shuffle needs original point_indices.')
                after = self._shuffle(after, common, slots, distinct, point_indices)
            delta = (after - patches[:, :, :-1]).masked_fill(~common[:, None, :, :, None], 0.)
            if cfg.GRANULARITY == 'frame':
                state = torch.einsum('bktn,bktnd->bktd', w, delta).unsqueeze(-2)
                mass = w.sum(-1, keepdim=True)
            else:
                mass = torch.einsum('bktn,bng->bktg', w, membership)
                state = torch.einsum('bktn,bktnd,bng->bktgd', w, delta, membership)
                state = state / mass.clamp_min(1e-12)[..., None]
            counts = torch.einsum('btn,bng->btg', (common & distinct[:, None]).float(), membership)
            min_points = cfg.MIN_POINTS if cfg.GRANULARITY == 'cluster' else 1
            norm = state.norm(dim=-1)
            usable = ((counts[:, None] >= min_points) & (mass > 1e-12)
                      & torch.isfinite(norm) & (norm > cfg.NORM_EPS))
            state = _unit(state.masked_fill(~usable[..., None], 0.))
        return dict(state=state, valid=usable, norm=norm.masked_fill(~usable, 0.), alpha=alpha)

    def _shuffle(self, after, valid, slots, distinct, point_indices):
        """Fixed local RNG; permute distinct valid tracks and their duplicate slots.

        This also perturbs attention-to-feature assignment: not by itself proof
        that physical correspondence, rather than pooling weights, is causal.
        """
        result = after.clone()
        for b in range(after.shape[0]):
            for t in range(after.shape[2]):
                for g in slots[b].unique().tolist():
                    if g < 0:
                        continue
                    ix = ((slots[b] == g) & valid[b, t] & distinct[b]).nonzero().flatten()
                    if ix.numel() < 2:
                        continue
                    generator = torch.Generator(device=after.device).manual_seed(
                        self.cfg.SHUFFLE_SEED + 97*t + int(g))
                    perm = ix[torch.randperm(ix.numel(), generator=generator, device=after.device)]
                    for source, destination in zip(ix.tolist(), perm.tolist()):
                        copies = point_indices[b].eq(point_indices[b, source]) & valid[b, t]
                        result[b, :, t, copies] = after[b, :, t, destination, None]
        return result

    def _reference_mask(self, valid, alpha):
        # valid [S,I,G] or [S,G]; alpha [S,G]. Select per Support, not globally.
        available = valid.any(1) if valid.ndim == 3 else valid
        order = alpha.masked_fill(~available, -1).argsort(dim=-1, descending=True, stable=True)
        selected = torch.zeros_like(available)
        selected.scatter_(1, order[:, :min(self.cfg.REF_TOPK, order.shape[1])], True)
        return valid & (selected[:, None] if valid.ndim == 3 else selected)

    def _metric(self, q, refs, valid, labels, known, channel):
        """Count-normalized positive-minus-negative LME. All-empty is finite."""
        temperature = _positive(getattr(self, channel + '_temperature_raw'))
        scale = _positive(getattr(self, channel + '_scale_raw'))
        flat = refs.reshape(-1, refs.shape[-1])
        if channel == 'state':
            similarity = torch.matmul(_unit(q), _unit(flat).T)
        else:
            split = 3 * self.cfg.GEO_NUM_BINS
            direction = torch.matmul(_unit(q[..., :split]), _unit(flat[..., :split]).T)
            amplitude = torch.exp(-((q[..., None, split:] - flat[..., split:])**2).mean(-1))
            mix = self.cfg.GEO_AMPLITUDE_WEIGHT
            similarity = (1-mix) * direction + mix * amplitude
        known = known & torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
        extra = (None,) * (valid.ndim-1)
        pos = (valid & ((labels > .5) & known)[(...,) + extra]).flatten()
        neg = (valid & ((labels <= .5) & known)[(...,) + extra]).flatten()

        def aggregate(mask):
            # Avoid logsumexp(all -inf), including its undefined backward.
            scores = (similarity / temperature).masked_fill(~mask, -1e4)
            value = temperature * (scores.logsumexp(-1) - mask.sum().clamp_min(1).float().log())
            return torch.where(mask.any(), value, torch.zeros_like(value))

        present = pos.any() & neg.any()
        return scale * (aggregate(pos) - aggregate(neg)) * present, present

    def evidence(self, query, support, qgeo, sgeo, qgv, sgv, labels, known, rho=None):
        """Support labels only. Query truth is never an input."""
        result = {}
        qa, sa = query['alpha'], support['alpha']
        if self.cfg.STRICT_COMMON_REPLAY:
            return self._replay(query, support, labels, known, rho)
        with torch.autocast(device_type=qa.device.type, enabled=False):
            for channel, enabled in (('state', self.cfg.STATE_ENABLE), ('geo', self.cfg.GEO_ENABLE)):
                if not enabled:
                    continue
                logits, validity = [], []
                for k in range(labels.shape[1]):
                    if channel == 'state':
                        q, s = query['state'][:, k], support['state'][:, k]
                        qv, sv = query['valid'][:, k], support['valid'][:, k]
                    else:
                        q, s, qv, sv = qgeo.detach().float(), sgeo.detach().float(), qgv.bool(), sgv.bool()
                    sv = self._reference_mask(sv, sa[:, k])
                    value, present = self._metric(q, s, sv, labels[:, k], known[:, k], channel)
                    if channel == 'state':
                        value = (value * qv).sum(1) / qv.sum(1).clamp_min(1)
                        qv = qv.any(1)
                    weights = qa[:, k].masked_fill(~qv, 0.)
                    good = present & (weights.sum(-1) > 1e-12)
                    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
                    logits.append((value * weights).sum(-1) * good)
                    validity.append(good)
                result['entity_' + channel + '_logits'] = torch.stack(logits, 1)
                result['entity_' + channel + '_valid'] = torch.stack(validity, 1)
        return result

    def _replay(self, query, support, labels, known, rho):
        """Inference oracle: exact original rho kernel/temperature/scale, no top-K.

        Load a baseline through initialize_from_rho_only(); not a train variant.
        """
        if self.training or rho is None:
            raise ValueError('STRICT_COMMON_REPLAY requires eval mode and original rho.')
        q, s = query['state'].squeeze(-2), support['state'].squeeze(-2)
        qv, sv = query['valid'].squeeze(-1), support['valid'].squeeze(-1)
        values, masks = [], []
        for k in range(labels.shape[1]):
            pos = (sv[:, k] & ((labels[:, k] > .5) & known[:, k])[:, None]).flatten()[None]
            neg = (sv[:, k] & ((labels[:, k] <= .5) & known[:, k])[:, None]).flatten()[None]
            z, _, _ = rho._metric_relation_chunk(q[:, k, :, None],
                q.new_ones(q.shape[0], 1, q.shape[2], 1), s[:, k].flatten(0, 1)[None], pos, neg)
            valid = qv[:, k] & pos.any() & neg.any()
            values.append((z[:, 0] * valid).sum(-1) / valid.sum(-1).clamp_min(1))
            masks.append(valid.any(-1))
        return dict(entity_state_logits=torch.stack(values, 1), entity_state_valid=torch.stack(masks, 1))

    def fuse(self, result, evidence):
        original = result['query_partial_q2s_logits']
        delta = torch.zeros_like(original, dtype=torch.float32)
        diagnostics = {}
        for channel in ('state', 'geo'):
            key = 'entity_' + channel + '_logits'
            if key not in evidence:
                continue
            coefficient = getattr(self, 'lambda_' + channel)
            multiplier = 1. if self.training else getattr(self.cfg, channel.upper() + '_TEST_MULT')
            contribution = multiplier * coefficient * evidence[key]
            contribution = contribution.masked_fill(~evidence['entity_' + channel + '_valid'], 0.)
            delta = delta + contribution
            diagnostics['entity_' + channel + '_contribution'] = contribution
            diagnostics['entity_lambda_' + channel] = coefficient
            diagnostics['entity_' + channel + '_temperature'] = _positive(getattr(self, channel + '_temperature_raw'))
            diagnostics['entity_' + channel + '_scale'] = _positive(getattr(self, channel + '_scale_raw'))
        result.update(evidence)
        result.update(diagnostics)
        result['query_partial_q2s_rho_logits'] = original
        result['entity_logit_delta'] = delta
        result['query_partial_q2s_logits'] = original + delta
        result['query_partial_q2s_temporal_logits'] = result['query_partial_q2s_logits']
        return result


def initialize_from_rho_only(model, state_dict):
    """Explicit inference-only baseline import; NEVER resume optimizer/epoch.

    Normal checkpoint loading still rejects architecture mismatches. New scalar
    defaults must be zero for the no-op check; replay may set lambda afterwards.
    """
    if model.training or getattr(model, 'entity_change_verifier', None) is None:
        raise ValueError('Baseline import requires eval-mode ENTITY_CHANGE model.')
    if any(key.startswith('entity_change_verifier.') for key in state_dict):
        raise ValueError('Expected a rho-only source checkpoint.')
    signature = state_dict.get('text_concat_st.architecture_signature')
    expected = model.text_concat_st.architecture_signature[:6]
    if signature is None or not torch.equal(signature.cpu(), expected.cpu()) or signature.tolist()[-2:] != [0, 1]:
        raise ValueError('Expected compatible dual-decoupled rho-only checkpoint.')
    for name, value in model.named_parameters():
        if name.startswith('entity_change_verifier.') or not value.requires_grad:
            continue
        if name not in state_dict or state_dict[name].shape != value.shape:
            raise ValueError('Incomplete rho source: ' + name)
    weights = {k: v for k, v in state_dict.items() if k != 'text_concat_st.architecture_signature'}
    return model.load_state_dict(weights, strict=False)
