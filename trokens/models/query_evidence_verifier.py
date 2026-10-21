"""Learned, class-shared Query evidence verification (no Query labels).

Raw semantic acceptance m and Post positive/confuser preference rho are learned
independently. Verifier features are never inserted into visual prototypes.
The three-state masses are operational gates, not calibrated latent-state labels.
"""

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from trokens.config.query_judgments import resolve_query_judgments


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


class FP32LayerNorm(nn.LayerNorm):
    """Keep normalization in FP32 without changing checkpoint parameter keys."""

    def forward(self, value):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return F.layer_norm(value.float(), self.normalized_shape, self.weight, self.bias, self.eps)


class SemanticRelationQuality(nn.Module):
    """Raw semantic evidence plus a bounded, Support-conditioned vector relation.

    No class-specific parameters, old Top-K scores, Support thresholds or Post
    features. Spatial pooling is private to this head: it never replaces pi.
    Reference attention keeps every Support frame until the Query reads it.
    """

    def __init__(self, input_dim, cfg):
        super().__init__()
        d = int(getattr(cfg, "QUALITY_HIDDEN_DIM", 64))
        if d <= 0:
            raise ValueError("QUALITY_HIDDEN_DIM must be positive.")
        self.dim = d
        for attr, field, default, positive in (
            ("patch_temperature", "QUALITY_PATCH_TEMPERATURE", 0.07, True),
            ("reference_temperature", "QUALITY_REFERENCE_TEMPERATURE", 0.10, True),
            ("max_pool_delta", "QUALITY_MAX_POOL_DELTA", 0.5, False),
            ("max_reference_delta", "QUALITY_MAX_REFERENCE_DELTA", 0.5, False),
            ("max_evidence_delta", "QUALITY_MAX_EVIDENCE_DELTA", 0.10, False),
        ):
            value = float(getattr(cfg, field, default))
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"{field} must be finite and {'positive' if positive else 'non-negative'}.")
            # Save the semantic bounds in the checkpoint, not as learned knobs.
            self.register_buffer(attr, torch.tensor(value, dtype=torch.float32))
        self.relation_strength = float(getattr(cfg, "QUALITY_RELATION_STRENGTH", 1.0))
        if not math.isfinite(self.relation_strength) or not 0 <= self.relation_strength <= 1:
            raise ValueError("QUALITY_RELATION_STRENGTH must be in [0,1].")
        scale = float(getattr(cfg, "QUALITY_SCALE_INIT", 10.0))
        bias = float(getattr(cfg, "QUALITY_BIAS_INIT", 0.0))
        if not math.isfinite(scale) or scale <= 1e-4:
            raise ValueError("QUALITY_SCALE_INIT must be finite and greater than 1e-4.")
        if not math.isfinite(bias):
            raise ValueError("QUALITY_BIAS_INIT must be finite.")
        value = scale - 1e-4
        self.scale_raw = nn.Parameter(torch.tensor(value + math.log(-math.expm1(-value))))
        self.bias = nn.Parameter(torch.tensor(bias))
        self.projection = nn.Linear(input_dim, d, bias=False)
        self.pool = nn.Sequential(
            FP32LayerNorm(2 * d + 1), nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1),
        )
        self.relation = nn.Sequential(
            FP32LayerNorm(5 * d + 5), nn.Linear(5 * d + 5, 2 * d), nn.GELU(), nn.Linear(2 * d, 1),
        )
        # Near the semantic anchor, but not exactly zero: all lower layers can
        # receive gradients on the first step (including checkpointed/DDP use).
        for head in (self.pool[-1], self.relation[-1]):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def _pool(self, raw, projected, text, projected_text, valid):
        with torch.autocast(device_type=raw.device.type, enabled=False):
            evidence = torch.matmul(raw.float(), text.float().T).permute(3, 0, 1, 2)
        x, e = projected[None], projected_text[:, None, None, None]
        feature = torch.cat((x - e, x * e, evidence.unsqueeze(-1)), dim=-1)
        delta = self.max_pool_delta * self.pool(feature).squeeze(-1).float().tanh()
        weights = _masked_softmax(evidence / self.patch_temperature + delta, valid[None])
        return weights, evidence, delta

    def _references(self, raw, projected, text, projected_text, valid):
        weights, _, _ = self._pool(raw, projected, text, projected_text, valid)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            # Contract K as a matrix row axis, not a broadcast batch axis:
            # otherwise matmul materializes K copies of all D=1024 tokens.
            references = _unit(torch.einsum("kstn,stnd->kstd", weights, raw.float()))
            reference_text = (references * text[:, None, None]).sum(-1).flatten(1)
        references = references.flatten(1, 2)
        return references, self.projection(references), reference_text

    def _relation_inputs(self, raw, projected, text, projected_text, valid,
                         references, projected_references, reference_text, reference_mask):
        """Shared v3 evidence reading, before either quality decision rule."""
        weights, evidence, pool_delta = self._pool(raw, projected, text, projected_text, valid)
        # Original Raw geometry determines both the absolute evidence and the
        # reference-read anchor. Only low-dimensional relation GEMMs use AMP.
        with torch.autocast(device_type=raw.device.type, enabled=False):
            # Query patches are shared by classes. One GEMM avoids expanding
            # [Q,T,N,D] to [K,Q,T,N,D] just to compare class reference sets.
            cosine = raw.float().reshape(-1, raw.shape[-1]) @ references.float().flatten(0, 1).T
            cosine = cosine.view(*raw.shape[:3], *references.shape[:2]).permute(3, 0, 1, 2, 4)
        x, e = projected[None], projected_text[:, None, None, None]
        ref_difference = projected_references - projected_text[:, None]
        delta = torch.matmul(x - e, ref_difference[:, None, None].transpose(-2, -1)).float()
        delta = self.max_reference_delta * (delta / math.sqrt(self.dim)).tanh()
        probability = _masked_softmax(
            (cosine / self.reference_temperature + delta)[:, None].expand(-1, 2, -1, -1, -1, -1),
            reference_mask[:, :, None, None, None, :],
        )
        reads = torch.matmul(probability.to(projected_references.dtype),
                             projected_references[:, None, None, None])
        pos, neg = reads.unbind(1)
        # Retain absolute evidence even if all references are weak: softmax
        # reference weights alone cannot represent that situation.
        read_cosine = (probability * cosine[:, None]).sum(-1)
        read_text = (probability * reference_text[:, None, None, None, None]).sum(-1)
        pc, nc = read_cosine.unbind(1)
        pt, nt = read_text.unbind(1)
        relation = torch.cat((
            (x - e).expand_as(pos), x - pos, x - neg, x * pos, x * neg,
            torch.stack((evidence, pc, nc, pt, nt), dim=-1),
        ), dim=-1)
        return weights, evidence, pool_delta, relation

    def _chunk(self, raw, projected, text, projected_text, valid,
               references, projected_references, reference_text, reference_mask):
        weights, evidence, pool_delta, relation = self._relation_inputs(
            raw, projected, text, projected_text, valid,
            references, projected_references, reference_text, reference_mask)
        residual = self.relation(relation).squeeze(-1).float().tanh()
        anchor = (weights * evidence).sum(-1)
        correction = self.relation_strength * self.max_evidence_delta * (weights * residual).sum(-1)
        scale = F.softplus(self.scale_raw.float()) + 1e-4
        logits = scale * (anchor + correction) + self.bias.float()
        saturation = (weights * (residual.abs() > 0.95).float()).sum(-1)
        pool_shift = (weights * pool_delta.abs()).sum(-1)
        return tuple(value.transpose(0, 1) for value in
                     (logits, weights, anchor, correction, saturation, pool_shift))

    def forward(self, raw_query, raw_support, text, query_valid, support_valid,
                positive, negative, call, chunk_size):
        qr, sr, et = (self.projection(x) for x in (raw_query, raw_support, text))
        references = call(self._references, raw_support, sr, text, et, support_valid)
        support_frames = support_valid.any(-1)[None]
        mask = torch.stack(((support_frames & positive.T[:, :, None]).flatten(1),
                            (support_frames & negative.T[:, :, None]).flatten(1)), dim=1)
        available = mask.any(-1).all(-1)
        parts = []
        for start in range(0, raw_query.shape[0], chunk_size):
            sl = slice(start, start + chunk_size)
            parts.append(call(self._chunk, raw_query[sl], qr[sl], text, et, query_valid[sl], *references, mask))
        logits, weights, anchor, correction, saturation, pool_shift = (
            torch.cat(values) for values in zip(*parts))
        frame_valid = query_valid.any(-1)[:, None]
        valid = frame_valid & available[None, :, None]
        mass = torch.where(available[None, :, None], logits.sigmoid(), torch.ones_like(logits))
        mass = torch.where(frame_valid, mass, torch.zeros_like(mass))
        return dict(quality=mass, quality_logits=logits, quality_valid=valid,
                    quality_weights=weights, quality_reference_available=available,
                    quality_anchor=anchor, quality_correction=correction,
                    quality_correction_saturation=saturation, quality_pool_shift=pool_shift)


class RelationDecisionQuality(SemanticRelationQuality):
    """V3 evidence reader with a directly learned visual-relation decision.

    Inherit the reader operations, NOT the v3 residual/scale parameters. Raw
    cosine remains a feature and retrieval prior, never a mandatory logit term.
    This is frame-level acceptance, not patch supervision/prototype rewriting.
    """

    def __init__(self, input_dim, cfg):
        nn.Module.__init__(self)
        d = int(getattr(cfg, "QUALITY_HIDDEN_DIM", 64))
        local = int(getattr(cfg, "QUALITY_DECISION_LOCAL_DIM", 32))
        readout = int(getattr(cfg, "QUALITY_DECISION_READOUT_DIM", 16))
        for name, value in (("QUALITY_HIDDEN_DIM", d),
                            ("QUALITY_DECISION_LOCAL_DIM", local),
                            ("QUALITY_DECISION_READOUT_DIM", readout)):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        self.dim = d
        for attr, field, default, positive in (
            ("patch_temperature", "QUALITY_PATCH_TEMPERATURE", .07, True),
            ("reference_temperature", "QUALITY_REFERENCE_TEMPERATURE", .10, True),
            ("max_pool_delta", "QUALITY_MAX_POOL_DELTA", .5, False),
            ("max_reference_delta", "QUALITY_MAX_REFERENCE_DELTA", .5, False),
        ):
            value = float(getattr(cfg, field, default))
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"{field} must be finite and {'positive' if positive else 'non-negative'}.")
            self.register_buffer(attr, torch.tensor(value, dtype=torch.float32))
        self.projection = nn.Linear(input_dim, d, bias=False)
        self.pool = nn.Sequential(
            FP32LayerNorm(2 * d + 1), nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1),
        )
        self.relation_encoder = nn.Sequential(
            FP32LayerNorm(5 * d + 5), nn.Linear(5 * d + 5, 2 * d), nn.GELU(),
            nn.Linear(2 * d, local), nn.GELU(),
        )
        self.decision = nn.Sequential(nn.Linear(local, readout), nn.GELU(), nn.Linear(readout, 1))
        for head in (self.pool[-1], self.decision[-1]):
            nn.init.normal_(head.weight, std=.01)
            nn.init.zeros_(head.bias)

    def _chunk(self, raw, projected, text, projected_text, valid,
               references, projected_references, reference_text, reference_mask):
        weights, evidence, pool_delta, relation = self._relation_inputs(
            raw, projected, text, projected_text, valid,
            references, projected_references, reference_text, reference_mask)
        local = self.relation_encoder(relation)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            pooled = (weights.unsqueeze(-1) * local.float()).sum(-2)
            semantic_evidence = (weights * evidence).sum(-1)
            pool_shift = (weights * pool_delta.abs()).sum(-1)
            feature_norm = pooled.norm(dim=-1)
        logits = self.decision(pooled).squeeze(-1).float()
        return tuple(value.transpose(0, 1) for value in
                     (logits, weights, semantic_evidence, pool_shift, feature_norm))

    def forward(self, raw_query, raw_support, text, query_valid, support_valid,
                positive, negative, call, chunk_size):
        qr, sr, et = (self.projection(x) for x in (raw_query, raw_support, text))
        references = call(self._references, raw_support, sr, text, et, support_valid)
        support_frames = support_valid.any(-1)[None]
        mask = torch.stack(((support_frames & positive.T[:, :, None]).flatten(1),
                            (support_frames & negative.T[:, :, None]).flatten(1)), dim=1)
        available = mask.any(-1).all(-1)
        parts = []
        for start in range(0, raw_query.shape[0], chunk_size):
            sl = slice(start, start + chunk_size)
            parts.append(call(self._chunk, raw_query[sl], qr[sl], text, et, query_valid[sl], *references, mask))
        logits, weights, evidence, pool_shift, feature_norm = (
            torch.cat(values) for values in zip(*parts))
        frame_valid = query_valid.any(-1)[:, None]
        valid = frame_valid & available[None, :, None]
        mass = torch.where(available[None, :, None], logits.sigmoid(), torch.ones_like(logits))
        mass = torch.where(frame_valid, mass, torch.zeros_like(mass))
        return dict(quality=mass, quality_logits=logits, quality_valid=valid,
                    quality_weights=weights, quality_reference_available=available,
                    quality_semantic_evidence=evidence, quality_pool_shift=pool_shift,
                    quality_relation_feature_norm=feature_norm)


class QueryEvidenceVerifier(nn.Module):
    def __init__(self, input_dim, cfg, quality=True, uniqueness=True):
        super().__init__()
        self.quality = quality
        self.uniqueness = uniqueness
        self.quality_mode = str(getattr(cfg, "QUALITY_MODE", "semantic_relation"))
        if self.quality_mode not in ("semantic_relation", "relation_decision"):
            raise ValueError("QUALITY_MODE must be semantic_relation or relation_decision.")
        self.detach_inputs = bool(getattr(cfg, "DETACH_INPUTS", True))
        self.chunk_size = int(getattr(cfg, "QUERY_CHUNK_SIZE", 4))
        if self.chunk_size <= 0:
            raise ValueError("QUERY_CHUNK_SIZE must be positive.")
        self.activation_checkpoint = bool(getattr(cfg, "ACTIVATION_CHECKPOINT", True))
        # Opt in per dataset. Inherit outer CUDA autocast; never force FP16 in
        # FP32 evaluation/training, and leave CPU verification in FP32.
        self.amp_enabled = bool(getattr(cfg, "AMP_ENABLE", False))
        self.warmup_fraction = float(getattr(cfg, "TRANSPORT_WARMUP_FRACTION", 0.05))
        if not math.isfinite(self.warmup_fraction) or not 0 <= self.warmup_fraction <= 1:
            raise ValueError("TRANSPORT_WARMUP_FRACTION must be in [0,1].")
        self.transport_progress = 1.0
        # Preserve trained checkpoint identifiers, including quality ablation.
        version = (5 if self.quality_mode == "relation_decision" else 3) if quality else 2
        self.register_buffer("architecture_version", torch.tensor(version, dtype=torch.int64))
        if quality:
            quality_head = RelationDecisionQuality if self.quality_mode == "relation_decision" else SemanticRelationQuality
            self.semantic_quality = quality_head(input_dim, cfg)
        if uniqueness:
            # Only aggregation/calibration are learned. Never rotate/compress
            # the Post metric space or add a feature-only classifier shortcut.
            for name, field, default in (
                ("metric_temperature_raw", "METRIC_TEMPERATURE_INIT", 0.1),
                ("metric_scale_raw", "METRIC_SCALE_INIT", 10.0),
            ):
                initial = float(getattr(cfg, field, default))
                if not math.isfinite(initial) or initial <= 1e-4:
                    raise ValueError(f"{field} must be finite and greater than 1e-4.")
                value = initial - 1e-4
                # Stable inverse softplus; both scalars stay positive in FP32.
                raw = value + math.log(-math.expm1(-value))
                self.register_parameter(name, nn.Parameter(torch.tensor(raw, dtype=torch.float32)))

    def _input(self, x):
        return x.detach() if self.detach_inputs else x

    def _call(self, function, *args):
        if self.training and self.activation_checkpoint and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    @staticmethod
    def _support_signs(support_labels, support_known, classes):
        if support_labels.ndim != 2 or support_labels.shape[1] != classes:
            raise ValueError("Verifier Support labels must have shape [S,K].")
        known = torch.isfinite(support_labels) & (support_labels >= 0) & (support_labels <= 1)
        if support_known is not None:
            if support_known.shape != support_labels.shape:
                raise ValueError("Support known-label mask must have shape [S,K].")
            known = known & support_known.bool()
        return (support_labels > 0.5) & known, (support_labels <= 0.5) & known

    def compute_raw_quality(self, raw_query, raw_support, text, query_valid,
                            support_raw_valid, support_labels, support_known=None):
        """Class-batched Raw-only m; learned references remain differentiable.

        Independent of Post/gates/rho, so the conditional encoder can hoist
        this once per class group instead of recomputing per video AND class.
        """
        if not self.quality:
            return {}
        if raw_query is None or raw_support is None:
            raise ValueError("Learned quality requires aligned pre-Pointformer Raw DinoTxt tokens.")
        precision = (nullcontext() if self.amp_enabled and raw_query.is_cuda
                     else torch.autocast(device_type=raw_query.device.type, enabled=False))
        with precision:
            positive, negative = self._support_signs(support_labels, support_known, text.shape[0])
            if raw_query.shape[:3] != query_valid.shape or raw_support.shape[:3] != support_raw_valid.shape:
                raise ValueError("Raw verifier tokens and validity masks must align in B/T/N.")
            qr, sr, et = (_unit(self._input(x)) for x in (raw_query, raw_support, text))
            return self.semantic_quality(
                qr, sr, et, query_valid, support_raw_valid, positive, negative,
                self._call, self.chunk_size)


    def _metric_aggregate(self, similarities, valid):
        """Count-normalized smooth maximum, separately for each reference set.

        tau * log(mean(exp(cosine / tau))) is monotone in every valid
        similarity. Its derivative w.r.t. evidence is softmax(cosine / tau).
        Subtracting log(count) avoids a positive/negative set-size logit prior.
        All-invalid sets return finite zeros with a connected zero gradient.
        """
        temperature = F.softplus(self.metric_temperature_raw.float()) + 1e-4
        valid = valid[:, None, None, None, :]
        count = valid.sum(-1)
        available = count > 0
        scores = (similarities.float() / temperature).masked_fill(~valid, -torch.inf)
        scores = torch.where(available.unsqueeze(-1), scores, torch.zeros_like(scores))
        aggregate = temperature * (scores.logsumexp(-1) - count.clamp_min(1).float().log())
        return torch.where(available, aggregate, torch.zeros_like(aggregate))

    def _metric_relation_chunk(self, query, weights, references, positive, negative):
        # [K,Q,T,N,R]; classes/Query frames never share a normalization axis.
        # Inputs already unit-normalized in FP32; only the GEMM follows AMP.
        scores = torch.matmul(query.unsqueeze(0), references[:, None, None].transpose(-2, -1)).float()
        pos = self._metric_aggregate(scores, positive)
        neg = self._metric_aggregate(scores, negative)
        scale = F.softplus(self.metric_scale_raw.float()) + 1e-4
        pi = weights.transpose(0, 1)
        pos = (scale * (pi * pos).sum(-1)).transpose(0, 1)
        neg = (scale * (pi * neg).sum(-1)).transpose(0, 1)
        return pos - neg, pos, neg

    def forward(self, *, raw_query, raw_support, text, post_query, post_references,
                query_weights, query_valid, support_raw_valid, support_frame_valid,
                support_labels, support_known=None, quality_output=None):
        """Only Support labels enter this API; outputs are [Q,K,T] (or weights).

        post_references is [S,K,T,D], never the class-averaged matcher prototype.
        Detach is applied BEFORE learned projections, not after learned memory.
        Optional quality_output reuses an externally computed Raw-only m;
        an empty dict requests rho only when the caller merges m afterward.
        """
        precision = (nullcontext() if self.amp_enabled and post_query.is_cuda
                     else torch.autocast(device_type=post_query.device.type, enabled=False))
        with precision:
            return self._forward_impl(
                raw_query, raw_support, text, post_query, post_references,
                query_weights, query_valid, support_raw_valid, support_frame_valid,
                support_labels, support_known, quality_output,
            )

    def _forward_impl(self, raw_query, raw_support, text, post_query, post_references,
                      query_weights, query_valid, support_raw_valid, support_frame_valid,
                      support_labels, support_known, quality_output=None):
        q, t, n, _ = post_query.shape
        k = text.shape[0]
        positive_labels, negative_labels = self._support_signs(support_labels, support_known, k)
        frame_valid = query_valid.any(dim=-1)
        if self.quality and quality_output is None:
            if raw_query is None or raw_support is None:
                raise ValueError("Learned quality requires aligned pre-Pointformer Raw DinoTxt tokens.")
            if raw_query.shape[:3] != (q, t, n) or raw_support.shape[:3] != support_raw_valid.shape:
                raise ValueError("Raw verifier tokens and validity masks must align in B/T/N.")
            qr, sr, et = (_unit(self._input(x)) for x in (raw_query, raw_support, text))
        if self.uniqueness:
            qp = _unit(self._input(post_query))
            sp = _unit(self._input(post_references))
            pi = self._input(query_weights).float()
        output = {} if quality_output is None else dict(quality_output)

        def masks(valid_frames):
            # [K,S,T] -> [K,R], preserving per-class Support identities.
            pos = (valid_frames & positive_labels.T[:, :, None]).flatten(1)
            neg = (valid_frames & negative_labels.T[:, :, None]).flatten(1)
            valid = torch.stack((pos, neg), dim=1)
            available = valid.any(dim=-1)
            # Retain the safe dummy element used for empty reference sets.
            return torch.cat((valid, ~available.unsqueeze(-1)), dim=-1), available

        if self.quality and quality_output is None:
            output.update(self.semantic_quality(
                qr, sr, et, query_valid, support_raw_valid, positive_labels, negative_labels,
                self._call, self.chunk_size,
            ))
        if self.uniqueness:
            mask, signs_available = masks(support_frame_valid.permute(1, 0, 2))
            available = signs_available.all(dim=1)
            references = sp.permute(1, 0, 2, 3).flatten(1, 2)
            logits, positive_logits, negative_logits = [], [], []
            for start in range(0, q, self.chunk_size):
                logit, pos_logit, neg_logit = self._call(
                    self._metric_relation_chunk, qp[start:start + self.chunk_size],
                    pi[start:start + self.chunk_size], references,
                    mask[:, 0, :-1], mask[:, 1, :-1],
                )
                logits.append(logit)
                positive_logits.append(pos_logit)
                negative_logits.append(neg_logit)
            logits = torch.cat(logits)
            valid = frame_valid[:, None] & available[None, :, None]
            rho = torch.where(valid, logits.sigmoid(), torch.ones_like(logits))
            output.update(relation=rho, relation_logits=logits, relation_valid=valid,
                          relation_positive_logits=torch.cat(positive_logits),
                          relation_negative_logits=torch.cat(negative_logits),
                          relation_reference_available=available)
        return output


def set_verifier_training_progress(model, update_index, total_updates):
    """Global optimizer-update based warmup; evaluation always uses full mass."""
    unwrapped = model.module if hasattr(model, "module") else model
    verifier = getattr(unwrapped, "query_evidence_verifier", None)
    if verifier is not None:
        warmup = math.ceil(max(1, total_updates) * verifier.warmup_fraction)
        verifier.transport_progress = min(1.0, max(0.0, update_index / warmup)) if warmup else 1.0


def build_query_q2s_aux(model, values, metadata, raw):
    """Preserve existing routing and matcher, replacing only evidence estimates."""
    from trokens.models.query_matching import (
        build_class_support_references, classwise_frame_similarity,
        factorize_frame_hypothesis_states, hypothesis_mass_bimhm_logits,
        normalize_patch_region_weights,
    )

    cfg = model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    judgments = resolve_query_judgments(cfg)
    support_mask = metadata["support_mask"].to(values.device).bool()
    query_indices = (~support_mask).nonzero(as_tuple=False).flatten()
    if not query_indices.numel():
        return None
    point_mask = metadata["pred_query_mask" if model.cfg.POINT_INFO.USE_PT_QUERY_MASK
                          else "pred_visibility"].to(values.device).bool()
    visibility = metadata.get("pred_visibility", point_mask).to(values.device).bool()
    evidence_mask = point_mask & visibility
    label_values = metadata["episode_positive_labels"].to(values.device)
    known = torch.isfinite(label_values) & (label_values >= 0) & (label_values <= 1)
    if "episode_label_known_mask" in metadata:
        known = known & metadata["episode_label_known_mask"].to(values.device).bool()
    labels = (label_values > 0.5) & known
    class_ids = metadata["episode_class_ids"].to(values.device).long()
    class_ids = (class_ids[0] if class_ids.ndim == 2 else class_ids).flatten()
    if not class_ids.numel():
        return None
    text = model._get_pot_label_text_features(class_ids, values.dtype)
    support_prototypes = model._build_frame_softmax_support_prototypes(
        values, point_mask, support_mask, labels, text,
    )
    route_text = text
    if bool(getattr(model, "use_support_text_fusion", False)):
        route_text, _, _ = model._fuse_episode_text_with_support_visual(text, support_prototypes)
    # Query and Support share the complete-episode text/visual route.
    routed, weights = model._compute_batched_frame_softmax_text_prototypes(
        values, point_mask, route_text,
    )
    query_prototypes = routed.index_select(0, query_indices)
    query_weights = weights.index_select(0, query_indices)
    diag = model._compute_bidirectional_frame_similarity(query_prototypes, support_prototypes)
    alpha = float(getattr(model.pot_route_cfg, "QUERY_PARTIAL_LOGIT_ALPHA", 10.0))
    bias = float(getattr(model.pot_route_cfg, "QUERY_PARTIAL_LOGIT_BIAS", -2.0))
    base_logits = alpha * diag + bias
    result = {
        "query_evidence_enabled": True,
        "query_partial_q2s_base_logits": base_logits,
        "query_partial_q2s_pre_refinement_logits": base_logits,
        "query_partial_query_prototypes": query_prototypes,
        "query_partial_query_prototypes_before_refinement": query_prototypes,
        "query_partial_support_prototypes": support_prototypes,
        "query_partial_diag_similarity": diag,
        "query_partial_diag_similarity_before_refinement": diag,
        "query_partial_alpha_sim_term": alpha * diag,
        "query_partial_bias_term": torch.full_like(base_logits, bias),
        "query_partial_query_sample_indices": query_indices,
        "query_partial_label_axis_global_labels": class_ids,
        "query_partial_target_label_indices": torch.arange(class_ids.numel(), device=values.device),
    }
    if not (judgments.quality or judgments.uniqueness):
        result.update(query_partial_q2s_logits=base_logits,
                      query_partial_q2s_temporal_logits=base_logits,
                      query_class_matchability=torch.ones_like(base_logits))
        return result
    verifier = getattr(model, "query_evidence_verifier", None)
    if verifier is None:
        raise RuntimeError("Learned verifier must be registered at model construction, before optimizer/DDP.")
    if (verifier.quality, verifier.uniqueness) != (judgments.quality, judgments.uniqueness):
        raise ValueError("Rebuild the model after changing learned judgment switches.")
    if judgments.quality and (raw is None or raw.shape[:3] != values.shape[:3]):
        raise ValueError("Learned quality requires aligned pre-Pointformer Raw DinoTxt tokens.")
    references = support_frame_valid = None
    if judgments.uniqueness:
        references, support_frame_valid, _, _, _ = build_class_support_references(
            model, values, point_mask, visibility, support_mask, labels, route_text,
            precomputed_prototypes=routed,
        )
    query_valid = evidence_mask.index_select(0, query_indices)
    hypothesis_weights = normalize_patch_region_weights(query_weights, query_valid)
    outputs = verifier(
        raw_query=raw.index_select(0, query_indices) if judgments.quality else None,
        raw_support=raw[support_mask] if judgments.quality else None,
        text=text, post_query=values.index_select(0, query_indices),
        post_references=references, query_weights=hypothesis_weights,
        query_valid=query_valid, support_raw_valid=evidence_mask[support_mask],
        support_frame_valid=support_frame_valid, support_labels=labels[support_mask],
        support_known=known[support_mask],
    )
    return finish_query_evidence_aux(
        model, result, query_prototypes, support_prototypes, query_weights,
        query_valid, hypothesis_weights, outputs)


def finish_query_evidence_aux(model, result, query_prototypes, support_prototypes,
                              query_weights, query_valid, hypothesis_weights, outputs):
    """Shared transport/diagnostics for legacy and single-path encoders."""
    cfg = model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    judgments = resolve_query_judgments(cfg)
    base_logits = result["query_partial_q2s_base_logits"]
    if not (judgments.quality or judgments.uniqueness):
        result.update(query_partial_q2s_logits=base_logits,
                      query_partial_q2s_temporal_logits=base_logits,
                      query_class_matchability=torch.ones_like(base_logits))
        return result
    from trokens.models.query_matching import (
        classwise_frame_similarity, factorize_frame_hypothesis_states,
        hypothesis_mass_bimhm_logits,
    )
    verifier = model.query_evidence_verifier
    alpha = float(getattr(model.pot_route_cfg, "QUERY_PARTIAL_LOGIT_ALPHA", 10.0))
    shape = (*base_logits.shape, query_prototypes.shape[2])
    neutral = base_logits.new_ones(shape, dtype=torch.float32)
    mass = outputs.get("quality", neutral)
    rho = outputs.get("relation", neutral)
    states = factorize_frame_hypothesis_states(mass, rho)
    strength = float(getattr(cfg, "ABSOLUTE_MASS_TRANSPORT_STRENGTH", 1.0))
    if model.training:
        strength *= verifier.transport_progress
    transport = hypothesis_mass_bimhm_logits(
        classwise_frame_similarity(query_prototypes, support_prototypes),
        states["target_mass"], base_logits, alpha=alpha, transport_strength=strength,
        unmatched_cost=float(getattr(cfg, "ABSOLUTE_MASS_UNMATCHED_COST", 0.0)),
        one_sided_transport=bool(getattr(cfg, "ABSOLUTE_MASS_ONE_SIDED", True)),
    )
    frame_valid = query_valid.any(dim=-1).unsqueeze(1).expand_as(mass)
    result.update({
        "query_partial_q2s_logits": transport["logits"],
        "query_partial_q2s_temporal_logits": transport["logits"],
        "query_class_matchability": _frame_mean(rho, frame_valid, 1.0),
        "query_class_transport_mass": _frame_mean(transport["effective_patch_mass"], frame_valid),
        "query_patch_conditional_weights": query_weights,
        "query_patch_transport_weights": query_weights.float() * transport["effective_patch_mass"].unsqueeze(-1),
        "query_frame_patch_mass": transport["effective_patch_mass"],
        "query_frame_unmatched_mass": transport["unmatched_mass"],
        "query_frame_transport_valid": frame_valid,
        "query_frame_patch_mass_mean": _frame_mean(transport["effective_patch_mass"], frame_valid),
        "query_frame_unmatched_mass_mean": _frame_mean(transport["unmatched_mass"], frame_valid),
        "query_frame_transport_logit_delta": transport["transport_delta"],
        "query_frame_temporal_logit_delta": transport["logits"] - base_logits.float(),
        "query_frame_max_t_switch_fraction": transport["winner_switch_fraction"],
        "query_verifier_transport_strength": base_logits.new_tensor(strength, dtype=torch.float32),
    })
    for state, value in states.items():
        result["query_frame_" + state] = value
        result["query_frame_" + state + "_mean"] = _frame_mean(value, frame_valid)
    if judgments.quality:
        quality_weights = outputs["quality_weights"]
        result.update({
            "query_frame_absolute_patch_mass": mass,
            "query_quality_mass_mean": _frame_mean(mass, frame_valid),
            "query_quality_logits": outputs["quality_logits"],
            "query_quality_valid": outputs["quality_valid"],
            "query_quality_logit_mean": _frame_mean(outputs["quality_logits"], outputs["quality_valid"]),
            "query_quality_reference_available": outputs["quality_reference_available"],
            "query_quality_patch_weights": quality_weights,
            "query_quality_effective_patches_mean": _frame_mean(
                1 / quality_weights.square().sum(-1).clamp_min(1e-6), frame_valid),
            "query_quality_top1_mass_mean": _frame_mean(quality_weights.max(-1).values, frame_valid),
        })
        names = (("anchor", "correction", "correction_saturation", "pool_shift")
                 if verifier.quality_mode == "semantic_relation"
                 else ("semantic_evidence", "pool_shift", "relation_feature_norm"))
        for name in names:
            value = outputs["quality_" + name]
            result["query_quality_" + name] = value
            result["query_quality_" + name + "_mean"] = _frame_mean(value, outputs["quality_valid"])
        if verifier.quality_mode == "semantic_relation":
            result["query_quality_correction_abs_mean"] = _frame_mean(
                outputs["quality_correction"].abs(), outputs["quality_valid"])
            head = verifier.semantic_quality
            result.update(query_quality_scale=F.softplus(head.scale_raw.detach().float()) + 1e-4,
                          query_quality_bias=head.bias.detach().float(),
                          query_quality_relation_strength=mass.new_tensor(head.relation_strength),
                          query_quality_max_evidence_delta=head.max_evidence_delta.detach().float())
    if judgments.uniqueness:
        result.update({
            "query_frame_matchability": rho,
            "query_frame_matchability_mean": _frame_mean(rho, frame_valid, 1.0),
            "query_relation_logits": outputs["relation_logits"],
            "query_relation_valid": outputs["relation_valid"],
            "query_relation_logit_mean": _frame_mean(outputs["relation_logits"], outputs["relation_valid"]),
            "query_relation_reference_available": outputs["relation_reference_available"],
            "query_relation_positive_logits": outputs["relation_positive_logits"],
            "query_relation_negative_logits": outputs["relation_negative_logits"],
            "query_evidence_patch_weights": hypothesis_weights,
            "query_evidence_effective_patches_mean": _frame_mean(
                1 / hypothesis_weights.square().sum(-1).clamp_min(1e-6), frame_valid),
            "query_evidence_top1_mass_mean": _frame_mean(hypothesis_weights.max(-1).values, frame_valid),
        })
        result.update({
            "query_metric_temperature": (F.softplus(verifier.metric_temperature_raw.detach().float()) + 1e-4),
            "query_metric_scale": (F.softplus(verifier.metric_scale_raw.detach().float()) + 1e-4),
        })
    return result
