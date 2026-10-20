"""Single-path, trajectory-preserving label priors for temporal/spatial attention.

Only the first temporal projections/PRE-softmax scores are shared. Spatial QKV
and MLP parameters are shared, but their inputs and computation are classwise.
No Query labels are read here. Only bounded class/video blocks are expanded.
"""

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from trokens.models.query_evidence_verifier import (
    _masked_softmax, finish_query_evidence_aux,
)
from trokens.models.query_matching import normalize_patch_region_weights


class LabelPreTemporalRouting(nn.Module):
    def __init__(self, dim, cfg, global_classifier=False):
        super().__init__()
        self.scoring = str(cfg.SCORING)
        if self.scoring not in ("projected_cosine", "cosine"):
            raise ValueError("LABEL_PRE_TEMPORAL.SCORING must be projected_cosine or cosine.")
        self.temperature = float(cfg.ROUTE_TEMPERATURE)
        self.strength = float(cfg.GATE_STRENGTH)
        self.temporal_bias_scale = float(cfg.TEMPORAL_BIAS_SCALE)
        self.spatial_bias_scale = float(cfg.SPATIAL_BIAS_SCALE)
        self.video_chunk_size = int(cfg.VIDEO_CHUNK_SIZE)
        self.class_chunk_size = int(cfg.CLASS_CHUNK_SIZE)
        self.activation_checkpoint = bool(cfg.ACTIVATION_CHECKPOINT)
        self.batch_raw_quality = bool(cfg.BATCH_RAW_QUALITY)
        self.readout_checkpoint = bool(cfg.READOUT_CHECKPOINT)
        self.post_support_readout = bool(cfg.POST_SUPPORT_READOUT)
        self.global_classifier = bool(global_classifier)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("ROUTE_TEMPERATURE must be finite and positive.")
        if not math.isfinite(self.strength) or not 0 <= self.strength < 1:
            raise ValueError("GATE_STRENGTH must be in [0,1).")
        for name, scale in (("TEMPORAL_BIAS_SCALE", self.temporal_bias_scale),
                            ("SPATIAL_BIAS_SCALE", self.spatial_bias_scale)):
            if not math.isfinite(scale) or scale < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.video_chunk_size <= 0 or self.class_chunk_size <= 0:
            raise ValueError("CLASS_CHUNK_SIZE and VIDEO_CHUNK_SIZE must be positive.")
        if self.scoring == "projected_cosine":
            self.query = nn.Linear(dim, dim, bias=False)
            self.key = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.query.weight)
            nn.init.eye_(self.key.weight)
        else:
            # No dormant/frozen matrices in the optimizer or checkpoint.
            # This does NOT remove the original Transformer temporal/spatial QKV.
            self.query = nn.Identity()
            self.key = nn.Identity()
        # v5/v6 retain asymmetric conditions and replace v3/v4 Value gating
        # with temporal AND spatial logit bias, without/with global loss.
        # v7/v8 change only final Query readout to Post Support + text,
        # without/with global loss. Encoding conditions remain Raw-based.
        # v9-v12 are the corresponding original-space cosine scorers.
        # Within a scorer family, changed semantics require explicit init;
        # cross-scorer loading is rejected, never silently resume.
        self.register_buffer("architecture_version", torch.tensor(
            (7 if self.post_support_readout else 5) + int(self.global_classifier)
            + (4 if self.scoring == "cosine" else 0),
            dtype=torch.int64))

    def routing_weights(self, raw, labels, mask):
        """Patch-axis softmax for shared [K,D] or per-video [B,K,D] conditions.

        Only the small condition tensor varies by video role. Raw Keys are
        projected (or normalized directly) once for the whole episode.
        The cosine variant uses the legacy safe-normalization/clamped-cosine
        formula, in FP32 even under AMP. Neither path aggregates trajectories.
        """
        if self.scoring == "projected_cosine":
            # Preserve the v5-v8 projection precision and eps exactly.
            query = F.normalize(self.query(labels).float(), dim=-1, eps=1e-6)
            keys = F.normalize(self.key(raw).float(), dim=-1, eps=1e-6)
        else:
            query = F.normalize(torch.nan_to_num(labels.float(), nan=0., posinf=0., neginf=0.),
                                dim=-1, eps=1e-12)
            keys = F.normalize(torch.nan_to_num(raw.float(), nan=0., posinf=0., neginf=0.),
                               dim=-1, eps=1e-12)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            equation = "kd,btnd->bktn" if query.ndim == 2 else "bkd,btnd->bktn"
            scores = torch.einsum(equation, query, keys)
            if self.scoring == "cosine":
                scores = scores.clamp(-1., 1.)
            return _masked_softmax(scores / self.temperature, mask[:, None])

    def forward(self, raw, labels, mask):
        weights = self.routing_weights(raw, labels, mask)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            count = mask.sum(-1, keepdim=True)[:, None]
            gate = 1 + self.strength * (count * weights - 1).tanh()
            # Invalid tokens are governed by the original attention mask, not
            # invented evidence. An empty routing row has a neutral gate.
            gate = torch.where(mask[:, None], gate, torch.ones_like(gate))
        return gate

    def attention_biases(self, gate):
        """Convert [B,T,N] positive priors to source-time/source-patch biases.

        Divide by 1+eps so g=1 maps EXACTLY to zero, including the neutral
        spatial CLS key. This subtracts a common log(1+eps) from each row and
        is softmax-equivalent to log(g+eps). No Value multiplication remains.
        """
        b, t, n = gate.shape
        log_prior = ((gate.float() + 1e-6) / (1.0 + 1e-6)).log()
        temporal = (self.temporal_bias_scale * log_prior.transpose(1, 2).reshape(b * n, t)
                    if self.temporal_bias_scale else None)
        spatial = (self.spatial_bias_scale * log_prior.reshape(b * t, n)
                   if self.spatial_bias_scale else None)
        return temporal, spatial


def _call(router, function, *args):
    if router.training and router.activation_checkpoint and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


def _initial_conditions(model, raw, evidence_mask, support_indices, positive, text):
    """Build QUERY conditions from text-weighted positive Raw Support only.

    This coarse summary is not the final matching prototype. Support encoding
    and its matching readout remain text-only; all main trajectories survive.
    """
    support_raw = raw.index_select(0, support_indices)
    support_valid = evidence_mask.index_select(0, support_indices)
    frames, _ = model._compute_batched_frame_softmax_text_prototypes(
        support_raw, support_valid, text)
    valid = support_valid.any(-1)[:, None] & positive[:, :, None]
    valid = valid & frames.float().norm(dim=-1).gt(1e-12)
    prototypes = (frames.float() * valid[..., None]).sum(0)
    prototypes = prototypes / valid.sum(0).clamp_min(1)[..., None]
    return model._fuse_episode_text_with_support_visual(text, prototypes)[0]


def _prepare(model, features, mask):
    """Preserve original CLS/pos-drop layout and project temporal QKV once."""
    b, t, n, d = features.shape
    tokens = features.permute(0, 2, 1, 3).reshape(b, n * t, d)
    token_mask = mask.transpose(1, 2).reshape(b, n * t)
    if model.cfg.MODEL.USE_CLS_TOKEN:
        tokens = torch.cat((model.cls_token.expand(b, -1, -1), tokens), 1)
        token_mask = torch.cat((mask.new_ones(b, 1), token_mask), 1)
    tokens = model.pos_drop(tokens)
    patches = tokens[:, 1:] if model.cfg.MODEL.USE_CLS_TOKEN else tokens
    block = model.blocks[0]
    temporal_mask = mask.transpose(1, 2).reshape(b * n, t)
    state = block.temporal_attn.prepare_logits(
        block.temporal_norm1(patches.reshape(b * n, t, d)), temporal_mask)
    # A leading B,N layout permits per-video slices without expanding K.
    state = tuple(x.reshape(b, n, *x.shape[1:]) if x is not None else None for x in state)
    return tokens, token_mask, state


def _encode(model, tokens, token_mask, state, gate, return_global=False):
    b, t, n = gate.shape
    state = tuple(x.flatten(0, 1) if x is not None else None for x in state)
    temporal_bias, spatial_bias = model.label_pre_temporal.attention_biases(gate)
    encoded, _ = model.blocks[0](
        tokens, (t, model.point_grid_size, n // model.point_grid_size), token_mask,
        temporal_state=state, temporal_bias=temporal_bias, spatial_bias=spatial_bias)
    encoded = model.norm(encoded)
    global_feature = None
    if return_global:
        # Match pt_forward's CLS / USE_PATCH_AS_CLS / no-CLS readout, before
        # pre_logits. Reuse this encoding; never run a second neutral branch.
        if model.cfg.MODEL.USE_CLS_TOKEN and not model.cfg.MODEL.USE_PATCH_AS_CLS:
            global_feature = encoded[:, 0]
        else:
            patches = encoded[:, 1:] if model.cfg.MODEL.USE_CLS_TOKEN else encoded
            global_feature = patches.mean(dim=1)
    if model.cfg.MODEL.USE_CLS_TOKEN:
        encoded = encoded[:, 1:]
    patches = encoded.reshape(b, n, t, -1).transpose(1, 2)
    return (patches, global_feature) if return_global else patches


def _support_chunk(model, tokens, token_mask, state, gate, route_mask, texts):
    encoded, global_sum = _encode_with_global(model, tokens, token_mask, state, gate)
    frames = torch.stack([
        _readout(model, value, route_mask, text)[0]
        for value, text in zip(encoded, texts)
    ], 1)
    # Text readout for matching; fused-condition readout for rho references.
    return frames[:, :, 0], frames[:, :, 1], global_sum


def _readout(model, value, route_mask, text):
    function = model._compute_batched_frame_softmax_text_prototypes
    if model.label_pre_temporal.readout_checkpoint and model.training and torch.is_grad_enabled():
        return checkpoint(function, value, route_mask, text, use_reentrant=False)
    return function(value, route_mask, text)


def _encode_classes(model, tokens, token_mask, state, gates, return_global=False):
    """Batch independent classes as videos; never attend across class axes.

    Input gates: [B,C,T,N], output: [C,B,T,N,D]. Expansion happens INSIDE
    the chunk checkpoint, so expanded tokens/attention states aren't retained
    across the entire episode. Temporal QKV is still projected only once.
    """
    b, c, t, n = gates.shape
    if c == 1:
        encoded = _encode(model, tokens, token_mask, state, gates[:, 0], return_global)
        return (tuple(x.unsqueeze(0) for x in encoded) if return_global
                else encoded.unsqueeze(0))

    def repeat(value):
        return (value.unsqueeze(0).expand(c, *value.shape).reshape(c * b, *value.shape[1:])
                if value is not None else None)

    encoded = _encode(model, repeat(tokens), repeat(token_mask),
                      tuple(repeat(value) for value in state),
                      gates.transpose(0, 1).reshape(c * b, t, n), return_global)
    if return_global:
        patches, global_features = encoded
        return patches.reshape(c, b, t, n, -1), global_features.reshape(c, b, -1)
    return encoded.reshape(c, b, t, n, -1)


def _encode_with_global(model, tokens, token_mask, state, gates):
    enabled = model.label_pre_temporal.global_classifier
    encoded = _encode_classes(model, tokens, token_mask, state, gates, enabled)
    if not enabled:
        return encoded, None
    patches, global_features = encoded
    # Sum, not mean: the last class chunk may contain fewer classes. Divide
    # once by the episode class count after assembling all video chunks.
    return patches, global_features.float().sum(dim=0)


def _query_chunk(model, tokens, token_mask, state, gate, route_mask, raw_query,
                 query_valid, raw_support, support_valid, text, readout_condition,
                 references, reference_valid, support_labels, support_known):
    encoded, global_sum = _encode_with_global(model, tokens, token_mask, state, gate)
    parts = []
    verifier = model.query_evidence_verifier
    for k, value in enumerate(encoded):
        frames, weights = _readout(model, value, route_mask, readout_condition[k:k+1])
        hypothesis_weights = normalize_patch_region_weights(weights, query_valid)
        outputs = {}
        # Singleton class calls preserve the existing metric/quality kernels.
        # Query and ALL positive/negative Supports refer to the same candidate
        # class, with shared encoder parameters but asymmetric gates (Support:
        # text; Query: Raw-fused condition). Rho keeps its Raw-fused Support
        # reference readout and consumes the final Query hypothesis weights,
        # which change when Post Support readout is enabled.
        # Never flatten class-specific tokens into shared references.
        if verifier is not None:
            outputs = verifier(
                raw_query=raw_query, raw_support=raw_support, text=text[k:k+1],
                post_query=value, post_references=references[:, k:k+1],
                query_weights=hypothesis_weights, query_valid=query_valid,
                support_raw_valid=support_valid, support_frame_valid=reference_valid[:, k:k+1],
                support_labels=support_labels[:, k:k+1], support_known=support_known[:, k:k+1],
                quality_output={} if model.label_pre_temporal.batch_raw_quality else None)
        parts.append((frames, weights, hypothesis_weights, outputs))
    frames, weights, hypotheses, outputs = zip(*parts)
    return (torch.cat(frames, 1), torch.cat(weights, 1), torch.cat(hypotheses, 1),
            _merge_outputs(outputs, class_axis=True), global_sum)


def _merge_outputs(parts, class_axis=False):
    if not parts or not parts[0]:
        return {}
    result = {}
    for key in parts[0]:
        values = [part[key] for part in parts]
        if key.endswith("reference_available"):
            result[key] = torch.cat(values, 0) if class_axis else values[0]
        else:
            result[key] = torch.cat(values, 1 if class_axis else 0)
    return result


def _detach_diagnostics(result):
    """Expose only loss inputs to DDP's autograd output traversal.

    In PyTorch 2.0, DDP(find_unused_parameters=True) sends undefined gradients
    through unused differentiable outputs. With CUDA checkpoints, these can
    unpack the same checkpoint concurrently on different autograd threads,
    before its saved tensors have finished recomputing. Diagnostics keep their
    values but must not be extra backward roots. Detach only at the final
    output boundary, AFTER the full differentiable logits have been computed.
    """
    # Keep in sync with compute_query_partial_q2s_loss (dual BCE + two MILs).
    loss_inputs = {
        "query_partial_q2s_logits", "query_partial_q2s_base_logits",
        "query_quality_logits", "query_relation_logits",
        "global_classification_logits",
    }
    return {key: value.detach() if torch.is_tensor(value) and key not in loss_inputs else value
            for key, value in result.items()}


def label_conditioned_episode(model, features, raw, metadata):
    """Return compact episode matching outputs; never run a neutral encoder."""
    router = model.label_pre_temporal
    if raw is None or raw.shape != features.shape:
        raise ValueError("LABEL_PRE_TEMPORAL requires aligned Raw [B,T,N,D] trajectories.")
    device = features.device
    support_mask = metadata["support_mask"].to(device).bool()
    si = support_mask.nonzero(as_tuple=False).flatten()
    qi = (~support_mask).nonzero(as_tuple=False).flatten()
    ids = metadata["episode_class_ids"].to(device).long()
    ids = (ids[0] if ids.ndim == 2 else ids).flatten()
    if not si.numel() or not qi.numel() or not ids.numel():
        raise ValueError("Label-conditioned episodes require Support, Query and candidate classes.")
    # Select Support BEFORE consulting labels/known masks. Query labels are
    # supervision-only, even though the dataset carries them in metadata.
    labels = metadata["episode_positive_labels"].to(device).index_select(0, si)
    if labels.shape != (si.numel(), ids.numel()):
        raise ValueError("Support labels must match episode [S,K].")
    known = torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
    if "episode_label_known_mask" in metadata:
        known = known & metadata["episode_label_known_mask"].to(device).bool().index_select(0, si)
    positive = (labels > .5) & known
    route_mask = metadata["pred_query_mask" if model.cfg.POINT_INFO.USE_PT_QUERY_MASK
                          else "pred_visibility"].to(device).bool()
    evidence_mask = route_mask & metadata.get("pred_visibility", route_mask).to(device).bool()
    text = model._get_pot_label_text_features(ids, raw.dtype)
    condition = _initial_conditions(model, raw, evidence_mask, si, positive, text)
    # Prepare BOTH roles' conditions before encoding, without using Support
    # Post features or Query truth. Preserve original (possibly interleaved)
    # video order and avoid expanding the large [B,K,T,N,D] Raw tensor.
    video_conditions = torch.where(support_mask[:, None, None], text[None], condition[None])
    gates = _call(router, router, raw, video_conditions, route_mask)
    tokens, token_mask, state = _prepare(model, features, route_mask)

    def chunks(indices):
        # Cache common inputs once per video chunk, never once per class.
        result = []
        for ix in indices.split(router.video_chunk_size):
            result.append((ix, tokens.index_select(0, ix), token_mask.index_select(0, ix),
                           tuple(x.index_select(0, ix) if x is not None else None for x in state),
                           route_mask.index_select(0, ix), raw.index_select(0, ix),
                           evidence_mask.index_select(0, ix)))
        return result

    support_chunks, query_chunks = chunks(si), chunks(qi)
    raw_support = raw.index_select(0, si)
    support_valid = evidence_mask.index_select(0, si)
    raw_query = raw.index_select(0, qi) if router.batch_raw_quality else None
    query_valid = evidence_mask.index_select(0, qi)
    all_query, all_support, all_weights, all_hypotheses, all_outputs = [], [], [], [], []
    global_sum = features.new_zeros(features.shape[0], features.shape[-1], dtype=torch.float32)
    for start in range(0, ids.numel(), router.class_chunk_size):
        classes = slice(start, start + router.class_chunk_size)
        quality = {}
        if router.batch_raw_quality and model.query_evidence_verifier is not None:
            quality = _call(
                router, model.query_evidence_verifier.compute_raw_quality,
                raw_query, raw_support, text[classes], query_valid, support_valid,
                labels[:, classes], known[:, classes])
        support_parts = []
        texts = torch.stack((text[classes], condition[classes]), 1)
        for ix, xt, xm, st, rm, _, _ in support_chunks:
            support_parts.append(_call(
                router, partial(_support_chunk, model), xt, xm, st,
                gates[:, classes].index_select(0, ix), rm, texts))
        match_parts, reference_parts, support_globals = zip(*support_parts)
        match_frames, references = torch.cat(match_parts, 0), torch.cat(reference_parts, 0)
        if router.global_classifier:
            global_sum = global_sum.index_add(0, si, torch.cat(support_globals, 0))
        # Retain original positive-Support mean for the matcher.
        pos = positive[:, classes, None, None]
        support_prototype = (match_frames * pos).sum(0) / pos.sum(0).clamp_min(1)
        all_support.append(support_prototype)
        # Reuse the matching Post prototype, not the Raw summary or rho
        # references. Fusion honors VISUAL_DETACH without detaching the
        # prototype used by the matcher. No extra encoder / Query truth.
        readout_condition = (
            model._fuse_episode_text_with_support_visual(text[classes], support_prototype)[0]
            if router.post_support_readout else condition[classes]
        )
        reference_valid = support_valid.any(-1)[:, None] & references.float().norm(dim=-1).gt(1e-12)
        query_parts = []
        for ix, xt, xm, st, rm, rq, ev in query_chunks:
            query_parts.append(_call(
                router, partial(_query_chunk, model), xt, xm, st,
                gates[:, classes].index_select(0, ix), rm, rq, ev,
                raw_support, support_valid, text[classes], readout_condition,
                references, reference_valid, labels[:, classes], known[:, classes]))
        frames, weights, hypotheses, outputs, query_globals = zip(*query_parts)
        if router.global_classifier:
            global_sum = global_sum.index_add(0, qi, torch.cat(query_globals, 0))
        all_query.append(torch.cat(frames, 0))
        all_weights.append(torch.cat(weights, 0))
        all_hypotheses.append(torch.cat(hypotheses, 0))
        group_outputs = _merge_outputs(outputs)
        group_outputs.update(quality)
        all_outputs.append(group_outputs)

    query_prototypes = torch.cat(all_query, 1)
    support_prototypes = torch.cat(all_support, 0)
    query_weights = torch.cat(all_weights, 1)
    hypothesis_weights = torch.cat(all_hypotheses, 1)
    outputs = _merge_outputs(all_outputs, class_axis=True)
    diag = model._compute_bidirectional_frame_similarity(query_prototypes, support_prototypes)
    alpha = float(model.pot_route_cfg.QUERY_PARTIAL_LOGIT_ALPHA)
    bias = float(model.pot_route_cfg.QUERY_PARTIAL_LOGIT_BIAS)
    base = alpha * diag + bias
    result = dict(
        single_path_episode=True,
        query_evidence_enabled=model.query_evidence_verifier is not None,
        query_partial_q2s_base_logits=base,
        query_partial_q2s_pre_refinement_logits=base,
        query_partial_query_prototypes=query_prototypes,
        query_partial_query_prototypes_before_refinement=query_prototypes,
        query_partial_support_prototypes=support_prototypes,
        query_partial_diag_similarity=diag,
        query_partial_diag_similarity_before_refinement=diag,
        query_partial_alpha_sim_term=alpha * diag,
        query_partial_bias_term=torch.full_like(base, bias),
        query_partial_query_sample_indices=qi,
        query_partial_label_axis_global_labels=ids,
        query_partial_target_label_indices=torch.arange(ids.numel(), device=device),
    )
    result = finish_query_evidence_aux(
        model, result, query_prototypes, support_prototypes, query_weights,
        evidence_mask.index_select(0, qi), hypothesis_weights, outputs)
    if router.global_classifier:
        # All videos (Support AND Query), in the original batch order. No
        # labels choose a class branch: every candidate has equal weight.
        global_features = global_sum / ids.numel()
        result["global_classification_logits"] = model.head(
            model.head_drop(model.pre_logits(global_features)))
    return _detach_diagnostics(result)
