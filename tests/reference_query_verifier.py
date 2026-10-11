# Frozen pre-vectorization learned verifier for equivalence tests/benchmarks only.
# This is NOT the heuristic BACKEND=legacy; never import it from training code.
"""Learned, class-shared Query evidence verification (no Query labels).

Raw semantic acceptance m and Post positive/confuser preference rho are learned
independently. Neither attention output is inserted into the visual prototypes.
The three-state masses are operational gates, not calibrated latent-state labels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint



def _unit(x):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=-1, eps=1e-6)


def _masked_softmax(logits, valid):
    """FP32 softmax with an explicitly finite path for all-invalid rows."""
    valid = valid.bool().expand_as(logits)
    has_any = valid.any(dim=-1, keepdim=True)
    scores = logits.float().masked_fill(~valid, -torch.inf)
    scores = torch.where(has_any, scores, torch.zeros_like(scores))
    return scores.softmax(dim=-1) * valid.float()


def _frame_mean(value, valid, fallback=0.0):
    valid = valid.expand_as(value)
    count = valid.sum(dim=-1)
    mean = (value.float() * valid.float()).sum(dim=-1) / count.clamp_min(1)
    return torch.where(count > 0, mean, torch.full_like(mean, fallback))


class ReferenceAttention(nn.Module):
    """One-way reading of a shared frame-reference set, with safe empty sets."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, query, memory, valid):
        shape = query.shape
        dim = shape[-1]
        # A dummy key prevents NaN softmax/gradients even if a rank has no
        # positive/negative reference. Its output is explicitly masked to zero.
        memory = torch.cat((memory, memory.new_zeros(1, dim)), dim=0)
        available = valid.any()
        mask = torch.cat((valid, ~available.reshape(1)))

        def split(x):
            return x.reshape(1, -1, self.heads, dim // self.heads).transpose(1, 2)

        q = split(self.q(query.reshape(-1, dim)))
        k = split(self.k(memory))
        v = split(self.v(memory))
        # All verifier arithmetic remains FP32, including under outer AMP.
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None, None, :])
        context = context.transpose(1, 2).reshape(*shape)
        return self.out(context) * available.to(context.dtype)


class SequentialQueryEvidenceVerifier(nn.Module):
    def __init__(self, input_dim, cfg, quality=True, uniqueness=True):
        super().__init__()
        d = int(getattr(cfg, "HIDDEN_DIM", 128))
        heads = int(getattr(cfg, "NUM_HEADS", 4))
        ff = int(getattr(cfg, "FFN_DIM", 256))
        if min(d, heads, ff) <= 0 or d % heads:
            raise ValueError("Verifier dimensions must be positive and HIDDEN_DIM divisible by NUM_HEADS.")
        if int(getattr(cfg, "NUM_LAYERS", 1)) != 1:
            raise ValueError("learned_dual_verifier v1 implements exactly one reference-attention layer.")
        if bool(getattr(cfg, "LEGACY_DIAGNOSTICS", False)):
            raise ValueError("Use BACKEND=legacy for old score diagnostics; learned verification never runs them.")
        self.quality = quality
        self.uniqueness = uniqueness
        self.detach_inputs = bool(getattr(cfg, "DETACH_INPUTS", True))
        self.chunk_size = int(getattr(cfg, "QUERY_CHUNK_SIZE", 4))
        if self.chunk_size <= 0:
            raise ValueError("QUERY_CHUNK_SIZE must be positive.")
        self.activation_checkpoint = bool(getattr(cfg, "ACTIVATION_CHECKPOINT", True))
        self.warmup_fraction = float(getattr(cfg, "TRANSPORT_WARMUP_FRACTION", 0.05))
        if not math.isfinite(self.warmup_fraction) or not 0 <= self.warmup_fraction <= 1:
            raise ValueError("TRANSPORT_WARMUP_FRACTION must be in [0,1].")
        self.transport_progress = 1.0
        self.register_buffer("architecture_version", torch.tensor(1, dtype=torch.int64))
        if quality:
            self.raw_projection = nn.Linear(input_dim, d, bias=False)
            self.semantic = nn.Sequential(
                nn.Linear(3 * d, ff), nn.GELU(), nn.Linear(ff, d), nn.LayerNorm(d),
            )
            # A common spatial-logit bias cancels under softmax.
            self.semantic_pool = nn.Linear(d, 1, bias=False)
            self.semantic_attention = ReferenceAttention(d, heads)
            self.semantic_fusion = nn.Sequential(
                nn.LayerNorm(3 * d), nn.Linear(3 * d, ff), nn.GELU(), nn.Linear(ff, d),
            )
            self.quality_head = nn.Linear(d, 1)
        if uniqueness:
            self.post_projection = nn.Linear(input_dim, d, bias=False)
            self.relation_attention = ReferenceAttention(d, heads)
            self.relation = nn.Sequential(
                nn.LayerNorm(4 * d), nn.Linear(4 * d, ff), nn.GELU(),
                # A shared final additive bias would cancel in positive-minus-negative.
                nn.Linear(ff, d, bias=False),
            )
            self.relation_head = nn.Linear(d, 1, bias=False)

    def _input(self, x):
        return x.detach() if self.detach_inputs else x

    def _call(self, function, *args):
        if self.training and self.activation_checkpoint and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _semantic_tokens(self, raw, text):
        text = text.expand_as(raw)
        return self.semantic(torch.cat((raw, text, raw * text), dim=-1))

    def _semantic_references(self, raw, text, valid):
        u = self._semantic_tokens(raw, text)
        weights = _masked_softmax(self.semantic_pool(u).squeeze(-1), valid)
        return (weights.unsqueeze(-1) * u).sum(dim=-2)

    def _quality_chunk(self, raw, text, valid, memory, positive, negative):
        u = self._semantic_tokens(raw, text)
        weights = _masked_softmax(self.semantic_pool(u).squeeze(-1), valid)
        pos = self.semantic_attention(u, memory, positive)
        neg = self.semantic_attention(u, memory, negative)
        h = u + self.semantic_fusion(torch.cat((u, pos, neg), dim=-1))
        logits = self.quality_head((weights.unsqueeze(-1) * h).sum(dim=-2)).squeeze(-1)
        return logits, weights

    def _relation_chunk(self, query, weights, memory, positive, negative):
        def read(valid):
            reference = self.relation_attention(query, memory, valid)
            h = self.relation(torch.cat(
                (query, reference, query - reference, query * reference), dim=-1,
            ))
            return self.relation_head((weights.unsqueeze(-1) * h).sum(dim=-2)).squeeze(-1)
        pos, neg = read(positive), read(negative)
        return pos - neg, pos, neg

    def forward(self, *, raw_query, raw_support, text, post_query, post_references,
                query_weights, query_valid, support_raw_valid, support_frame_valid,
                support_labels, support_known=None):
        """Only Support labels enter this API; outputs are [Q,K,T] (or weights).

        post_references is [S,K,T,D], never the class-averaged matcher prototype.
        Detach is applied BEFORE learned projections, not after learned memory.
        """
        with torch.autocast(device_type=post_query.device.type, enabled=False):
            return self._forward_fp32(
                raw_query, raw_support, text, post_query, post_references,
                query_weights, query_valid, support_raw_valid, support_frame_valid,
                support_labels, support_known,
            )

    def _forward_fp32(self, raw_query, raw_support, text, post_query, post_references,
                      query_weights, query_valid, support_raw_valid, support_frame_valid,
                      support_labels, support_known):
        q, t, n, _ = post_query.shape
        k = text.shape[0]
        if support_labels.ndim != 2 or support_labels.shape[1] != k:
            raise ValueError("Verifier Support labels must have shape [S,K].")
        known = torch.isfinite(support_labels) & (support_labels >= 0) & (support_labels <= 1)
        if support_known is not None:
            if support_known.shape != support_labels.shape:
                raise ValueError("Support known-label mask must have shape [S,K].")
            known = known & support_known.bool()
        positive_labels = (support_labels > 0.5) & known
        negative_labels = (support_labels <= 0.5) & known
        frame_valid = query_valid.any(dim=-1)
        if self.quality:
            if raw_query is None or raw_support is None:
                raise ValueError("Learned quality requires aligned pre-Pointformer Raw DinoTxt tokens.")
            if raw_query.shape[:3] != (q, t, n) or raw_support.shape[:3] != support_raw_valid.shape:
                raise ValueError("Raw verifier tokens and validity masks must align in B/T/N.")
            qr = self.raw_projection(_unit(self._input(raw_query)))
            sr = self.raw_projection(_unit(self._input(raw_support)))
            et = self.raw_projection(_unit(self._input(text)))
        if self.uniqueness:
            qp = self.post_projection(_unit(self._input(post_query)))
            sp = self.post_projection(_unit(self._input(post_references)))
            pi = self._input(query_weights).float()
        output = {}

        def append(name, value):
            output.setdefault(name, []).append(value)

        for c in range(k):
            if self.quality:
                valid_support_frames = support_raw_valid.any(dim=-1)
                pos = (valid_support_frames & positive_labels[:, c, None]).flatten()
                neg = (valid_support_frames & negative_labels[:, c, None]).flatten()
                available = pos.any() & neg.any()
                reference = self._call(self._semantic_references, sr, et[c], support_raw_valid)
                memory = reference.reshape(-1, reference.shape[-1])
                logits, weights = [], []
                for start in range(0, q, self.chunk_size):
                    chunk_logit, chunk_weight = self._call(
                        self._quality_chunk, qr[start:start + self.chunk_size], et[c],
                        query_valid[start:start + self.chunk_size], memory, pos, neg,
                    )
                    logits.append(chunk_logit)
                    weights.append(chunk_weight)
                logits, weights = torch.cat(logits), torch.cat(weights)
                valid = frame_valid & available
                mass = torch.where(available, logits.sigmoid(), torch.ones_like(logits))
                mass = torch.where(frame_valid, mass, torch.zeros_like(mass))
                append("quality", mass)
                append("quality_logits", logits)
                append("quality_valid", valid)
                append("quality_weights", weights)
                append("quality_reference_available", available)
            if self.uniqueness:
                valid_support_frames = support_frame_valid[:, c]
                pos = (valid_support_frames & positive_labels[:, c, None]).flatten()
                neg = (valid_support_frames & negative_labels[:, c, None]).flatten()
                available = pos.any() & neg.any()
                memory = sp[:, c].reshape(-1, sp.shape[-1])
                logits, positive_logits, negative_logits = [], [], []
                for start in range(0, q, self.chunk_size):
                    logit, pos_logit, neg_logit = self._call(
                        self._relation_chunk, qp[start:start + self.chunk_size],
                        pi[start:start + self.chunk_size, c], memory, pos, neg,
                    )
                    logits.append(logit)
                    positive_logits.append(pos_logit)
                    negative_logits.append(neg_logit)
                logits = torch.cat(logits)
                valid = frame_valid & available
                rho = torch.where(valid, logits.sigmoid(), torch.ones_like(logits))
                append("relation", rho)
                append("relation_logits", logits)
                append("relation_positive_logits", torch.cat(positive_logits))
                append("relation_negative_logits", torch.cat(negative_logits))
                append("relation_valid", valid)
                append("relation_reference_available", available)
        return {name: torch.stack(values, dim=0 if values[0].ndim == 0 else 1)
                for name, values in output.items()}
