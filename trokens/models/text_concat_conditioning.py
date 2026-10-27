"""Candidate-text concatenation and same-attention spatial readout.

Each role independently selects text concatenation or ordinary visual-only
space-time attention. Only Support labels form prototypes; Query truth never
conditions the encoder or spatial aggregation. Evidence heads require
class-conditioned Query. When both roles are shared no text is read.
"""
from functools import partial
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from trokens.models.episode_outputs import _detach_diagnostics, _merge_outputs
from trokens.models.query_evidence_verifier import finish_query_evidence_aux
from trokens.models.query_matching import normalize_patch_region_weights


class TextConcatST(nn.Module):
    def __init__(self, cfg, global_classifier=True, quality=False, uniqueness=False):
        super().__init__()
        self.temporal_text_tokens = int(cfg.TEMPORAL_TEXT_TOKENS)
        self.spatial_text_tokens = int(cfg.SPATIAL_TEXT_TOKENS)
        self.class_chunk_size = int(cfg.CLASS_CHUNK_SIZE)
        self.video_chunk_size = int(cfg.VIDEO_CHUNK_SIZE)
        self.activation_checkpoint = bool(cfg.ACTIVATION_CHECKPOINT)
        self.mlp_checkpoint = bool(getattr(cfg, 'MLP_CHECKPOINT', False))
        self.readout_checkpoint = bool(getattr(cfg, 'READOUT_CHECKPOINT', False))
        self.shared_query = bool(getattr(cfg, 'SHARED_QUERY_ENABLE', False))
        self.shared_support = bool(getattr(cfg, 'SHARED_SUPPORT_ENABLE', False))
        if self.shared_query and (quality or uniqueness):
            raise ValueError('Shared Query requires both quality and uniqueness OFF.')
        if self.activation_checkpoint and (self.mlp_checkpoint or self.readout_checkpoint):
            raise ValueError('Use either full TEXT_CONCAT_ST checkpoint or selective MLP/readout checkpoints.')
        self.global_classifier = bool(global_classifier)
        if min(self.temporal_text_tokens, self.spatial_text_tokens,
               self.class_chunk_size, self.video_chunk_size) < 1:
            raise ValueError('TEXT_CONCAT_ST token counts and chunk sizes must be positive.')
        if cfg.READOUT != 'spatial_text_attention':
            raise ValueError('TEXT_CONCAT_ST requires spatial_text_attention readout.')
        # No learned prompt, gate, extra QKV or readout projection.
        signature = [1, self.temporal_text_tokens, self.spatial_text_tokens, int(global_classifier)]
        if quality or uniqueness:
            # v2 verifies Raw m / same-spatial-readout Post rho. v1 remains intact.
            signature = [2, *signature[1:], int(quality), int(uniqueness)]
        if self.shared_query:
            # v3 has different Query semantics, even though parameter shapes match.
            signature = [3, *signature[1:]]
        if self.shared_support and self.shared_query:
            # v4 was the separate text-free Query CLS experiment. Reserve v5
            # for both roles visual-only; configured text counts are unused.
            signature = [5, 0, 0, int(global_classifier)]
        elif self.shared_support:
            # Query-only text conditioning: generic Support, class-wise Query.
            signature = [6, self.temporal_text_tokens, self.spatial_text_tokens,
                         int(global_classifier), int(quality), int(uniqueness)]
        self.register_buffer('architecture_signature', torch.tensor(signature, dtype=torch.int64))


def _encode_chunk(model, features, mask, text, evidence_readout=None):
    """Expand ONLY the bounded current chunk, then pool before returning."""
    b, t, n, d = features.shape
    c = text.shape[0]
    tokens = features.transpose(1, 2).reshape(b, n * t, d)
    token_mask = mask.transpose(1, 2).reshape(b, n * t)
    if model.cfg.MODEL.USE_CLS_TOKEN:
        tokens = torch.cat((model.cls_token.expand(b, -1, -1), tokens), 1)
        token_mask = torch.cat((mask.new_ones(b, 1), token_mask), 1)
    tokens = model.pos_drop(tokens)
    tokens = tokens[:, None].expand(b, c, *tokens.shape[1:]).reshape(b * c, -1, d)
    token_mask = token_mask[:, None].expand(b, c, -1).reshape(b * c, -1)
    class_text = text[None].expand(b, c, d).reshape(b * c, d)
    config = model.text_concat_st
    encoded, weights = model.blocks[0].forward_text_concat(
        tokens, (t, model.point_grid_size, n // model.point_grid_size), token_mask,
        class_text, config.temporal_text_tokens, config.spatial_text_tokens)
    readout = partial(_pool_chunk, model, b, c, t, n, d, evidence_readout=evidence_readout)
    return (checkpoint(readout, encoded, weights, use_reentrant=False)
            if config.readout_checkpoint and model.training and torch.is_grad_enabled()
            else readout(encoded, weights))


def _pool_chunk(model, b, c, t, n, d, encoded, weights, evidence_readout=None):
    """Original Post normalization/readout, optionally recomputed on backward."""
    encoded = model.norm(encoded)
    cls = int(model.cfg.MODEL.USE_CLS_TOKEN)
    patches = encoded[:, cls:]
    global_features = (encoded[:, 0] if cls and not model.cfg.MODEL.USE_PATCH_AS_CLS
                       else patches.mean(1))
    patches = patches.reshape(b * c, n, t, d).transpose(1, 2)
    # Final Post visual values, weighted by this same spatial attention's
    # text-query/visual-key probabilities. Keep gradients through both paths.
    with torch.autocast(device_type=encoded.device.type, enabled=False):
        frames = torch.einsum('btn,btnd->btd', weights.float(), patches.float())
    result = (frames.reshape(b, c, t, d), weights.reshape(b, c, t, n),
              global_features.float().reshape(b, c, d).sum(1))
    if evidence_readout is not None:
        # Consume Post patches inside the bounded readout/checkpoint, never
        # return/cache the full [B,K,T,N,D] tensor at the episode boundary.
        return (*result, evidence_readout(patches.reshape(b, c, t, n, d), result[1]))
    return result


def _pool_shared_query_chunk(model, b, t, n, d, mask, encoded, evidence_readout=None):
    """Visual-only mean per frame; no text attention or class axis."""
    encoded = model.norm(encoded)
    cls = int(model.cfg.MODEL.USE_CLS_TOKEN)
    patches = encoded[:, cls:]
    global_features = (encoded[:, 0] if cls and not model.cfg.MODEL.USE_PATCH_AS_CLS
                       else patches.mean(1))
    patches = patches.reshape(b, n, t, d).transpose(1, 2)
    with torch.autocast(device_type=encoded.device.type, enabled=False):
        values = patches.float().masked_fill(~mask[..., None], 0.)
        frames = values.sum(2) / mask.sum(2, keepdim=True).clamp_min(1)
    result = (frames, global_features.float())
    return (*result, evidence_readout(patches)) if evidence_readout is not None else result


def _encode_shared_query_chunk(model, features, mask, evidence_readout=None):
    """The ordinary pt_forward encoder, pooled before leaving this chunk."""
    b, t, n, d = features.shape
    tokens = features.transpose(1, 2).reshape(b, n * t, d)
    token_mask = mask.transpose(1, 2).reshape(b, n * t)
    if model.cfg.MODEL.USE_CLS_TOKEN:
        tokens = torch.cat((model.cls_token.expand(b, -1, -1), tokens), 1)
        token_mask = torch.cat((mask.new_ones(b, 1), token_mask), 1)
    encoded = model.pos_drop(tokens)
    thw = (t, model.point_grid_size, n // model.point_grid_size)
    for block in model.blocks:
        encoded, _ = block(encoded, thw, token_mask)
    readout = partial(_pool_shared_query_chunk, model, b, t, n, d, mask,
                      evidence_readout=evidence_readout)
    return (checkpoint(readout, encoded, use_reentrant=False)
            if model.text_concat_st.readout_checkpoint and model.training and torch.is_grad_enabled()
            else readout(encoded))


def _shared_query_encodings(model, features, mask, text, si, qi):
    """Encode Support per class, but Query only once, outside the class loop."""
    config = model.text_concat_st

    def run(function, *args):
        return (checkpoint(function, *args, use_reentrant=False)
                if config.activation_checkpoint and model.training and torch.is_grad_enabled()
                else function(*args))

    support_frames, support_weights = [], []
    support_global = features.new_zeros(si.numel(), features.shape[-1], dtype=torch.float32)
    for class_text in text.split(config.class_chunk_size):
        parts = [run(partial(_encode_chunk, model), features[ix], mask[ix], class_text)
                 for ix in si.split(config.video_chunk_size)]
        support_frames.append(torch.cat([p[0] for p in parts], 0))
        support_weights.append(torch.cat([p[1] for p in parts], 0))
        if config.global_classifier:
            support_global = support_global + torch.cat([p[2] for p in parts], 0)
    query_parts = [run(partial(_encode_shared_query_chunk, model), features[ix], mask[ix])
                   for ix in qi.split(config.video_chunk_size)]
    query_frames = torch.cat([p[0] for p in query_parts], 0)
    global_features = None
    if config.global_classifier:
        query_global = torch.cat([p[1] for p in query_parts], 0)
        order = torch.cat((si, qi)).argsort()
        global_features = torch.cat((support_global / text.shape[0], query_global), 0)[order]
    return (torch.cat(support_frames, 1), query_frames,
            torch.cat(support_weights, 1), global_features)


def _shared_visual_encodings(model, features, mask):
    """One ordinary encoding per video, without text or a candidate-class axis."""
    config = model.text_concat_st
    function = partial(_encode_shared_query_chunk, model)
    parts = []
    for start in range(0, features.shape[0], config.video_chunk_size):
        end = start + config.video_chunk_size
        args = (features[start:end], mask[start:end])
        parts.append(checkpoint(function, *args, use_reentrant=False)
                     if config.activation_checkpoint and model.training and torch.is_grad_enabled()
                     else function(*args))
    frames = torch.cat([p[0] for p in parts], 0)
    globals_ = torch.cat([p[1] for p in parts], 0) if config.global_classifier else None
    return frames, globals_


def _shared_support_evidence(model, valid, patches):
    """Uniform valid visual patches; no text attention on a non-decoupled Support."""
    verifier = model.query_evidence_verifier
    weights = normalize_patch_region_weights(valid[:, None].float(), valid)[:, 0]
    with torch.autocast(device_type=patches.device.type, enabled=False):
        return torch.einsum('stn,stnd->std', weights, verifier._input(patches).float())


def _shared_support_encodings(model, features, raw, mask, evidence_mask, text, si, qi, labels, known):
    """Encode Support once, then class-conditioned Query; optional independent m/rho.

    Only compact Support frame references acquire a class axis. Never copy or
    encode Support patch tensors per class, including when rho is enabled.
    """
    config, verifier = model.text_concat_st, model.query_evidence_verifier
    quality = verifier is not None and verifier.quality
    uniqueness = verifier is not None and verifier.uniqueness
    if quality and (raw is None or raw.shape != features.shape):
        raise ValueError('Text-concat quality requires aligned pre-encoder Raw DinoTxt tokens.')

    def run(function, *args):
        return (checkpoint(function, *args, use_reentrant=False)
                if config.activation_checkpoint and model.training and torch.is_grad_enabled()
                else function(*args))

    support_parts = [run(partial(_encode_shared_query_chunk, model,
                         evidence_readout=partial(_shared_support_evidence, model, evidence_mask[ix])
                         if uniqueness else None), features[ix], mask[ix])
                     for ix in si.split(config.video_chunk_size)]
    support = torch.cat([p[0] for p in support_parts], 0)
    support_global = torch.cat([p[1] for p in support_parts], 0)
    references = torch.cat([p[2] for p in support_parts], 0) if uniqueness else None
    support_valid = evidence_mask[si]
    valid_frames = (support_valid.any(-1) & references.float().norm(dim=-1).gt(1e-12)
                    if uniqueness else None)
    queries, weights, all_outputs = [], [], []
    query_global = features.new_zeros(qi.numel(), features.shape[-1], dtype=torch.float32)
    for start in range(0, text.shape[0], config.class_chunk_size):
        class_text = text[start:start + config.class_chunk_size]
        signs, available = labels[:, start:start + class_text.shape[0]], known[:, start:start + class_text.shape[0]]
        outputs = (verifier.compute_raw_quality(raw[qi], raw[si], class_text,
                   evidence_mask[qi], support_valid, signs, available) if quality else {})
        refs = references[:, None].expand(-1, class_text.shape[0], -1, -1) if uniqueness else None
        valid = valid_frames[:, None].expand(-1, class_text.shape[0], -1) if uniqueness else None
        parts = [run(partial(_encode_chunk, model,
                     evidence_readout=partial(_query_evidence, model, class_text, evidence_mask[ix],
                         support_valid, refs, valid, signs, available) if uniqueness else None),
                     features[ix], mask[ix], class_text)
                 for ix in qi.split(config.video_chunk_size)]
        queries.append(torch.cat([p[0] for p in parts], 0))
        weights.append(torch.cat([p[1] for p in parts], 0))
        if config.global_classifier:
            query_global = query_global + torch.cat([p[2] for p in parts], 0)
        if uniqueness:
            outputs.update(_merge_outputs([p[3] for p in parts]))
        all_outputs.append(outputs)
    globals_ = None
    if config.global_classifier:
        order = torch.cat((si, qi)).argsort()
        globals_ = torch.cat((support_global, query_global / text.shape[0]), 0)[order]
    return (support, torch.cat(queries, 1), torch.cat(weights, 1), globals_,
            _merge_outputs(all_outputs, class_axis=True))


def _support_evidence(model, valid, patches, weights):
    verifier = model.query_evidence_verifier
    pi = normalize_patch_region_weights(verifier._input(weights), valid)
    with torch.autocast(device_type=patches.device.type, enabled=False):
        return torch.einsum('bktn,bktnd->bktd', pi, verifier._input(patches).float())


def _query_evidence(model, text, valid, support_valid, references, reference_valid,
                    labels, known, patches, weights):
    verifier = model.query_evidence_verifier
    pi = normalize_patch_region_weights(weights, valid)
    # Each singleton call compares only the SAME candidate-class encodings.
    # Raw quality is computed once per class group, not per Query/video chunk.
    return _merge_outputs([
        verifier(raw_query=None, raw_support=None, text=text[k:k+1],
                 post_query=patches[:, k], post_references=references[:, k:k+1],
                 query_weights=pi[:, k:k+1], query_valid=valid,
                 support_raw_valid=support_valid, support_frame_valid=reference_valid[:, k:k+1],
                 support_labels=labels[:, k:k+1], support_known=known[:, k:k+1],
                 quality_output={})
        for k in range(text.shape[0])
    ], class_axis=True)


def _verified_encodings(model, features, raw, mask, evidence_mask, text, si, qi, labels, known):
    """Support-first streaming, one encoder pass per video/class (also negatives).

    Only per-Support frame references survive across Query chunks. Main visual
    pooling/masks are unchanged; visibility masks apply exclusively to evidence.
    """
    config, verifier = model.text_concat_st, model.query_evidence_verifier
    if verifier.quality and (raw is None or raw.shape != features.shape):
        raise ValueError('Text-concat quality requires aligned pre-encoder Raw DinoTxt tokens.')
    support_valid, query_valid = evidence_mask[si], evidence_mask[qi]
    order = torch.cat((si, qi)).argsort()
    all_frames, all_weights, all_outputs = [], [], []
    global_sum = features.new_zeros(features.shape[0], features.shape[-1], dtype=torch.float32)

    def encode(indices, class_text, reader=None):
        function = partial(_encode_chunk, model, evidence_readout=reader)
        args = (features[indices], mask[indices], class_text)
        return (checkpoint(function, *args, use_reentrant=False)
                if config.activation_checkpoint and model.training and torch.is_grad_enabled()
                else function(*args))

    for start in range(0, text.shape[0], config.class_chunk_size):
        classes = slice(start, start + config.class_chunk_size)
        class_text, signs, available = text[classes], labels[:, classes], known[:, classes]
        quality = (verifier.compute_raw_quality(raw[qi], raw[si], class_text,
                   query_valid, support_valid, signs, available) if verifier.quality else {})
        support_parts = [encode(ix, class_text, partial(_support_evidence, model, evidence_mask[ix])
                                if verifier.uniqueness else None)
                         for ix in si.split(config.video_chunk_size)]
        support_frames, support_weights, support_globals = (
            torch.cat([p[i] for p in support_parts], 0) for i in range(3))
        references = (torch.cat([p[3] for p in support_parts], 0) if verifier.uniqueness else None)
        reference_valid = (support_valid.any(-1)[:, None] & references.float().norm(dim=-1).gt(1e-12)
                           if references is not None else None)
        query_parts = []
        for ix in qi.split(config.video_chunk_size):
            reader = (partial(_query_evidence, model, class_text, evidence_mask[ix], support_valid,
                              references, reference_valid, signs, available)
                      if verifier.uniqueness else None)
            query_parts.append(encode(ix, class_text, reader))
        query_frames, query_weights, query_globals = (
            torch.cat([p[i] for p in query_parts], 0) for i in range(3))
        outputs = _merge_outputs([p[3] for p in query_parts]) if verifier.uniqueness else {}
        outputs.update(quality)
        all_outputs.append(outputs)
        all_frames.append(torch.cat((support_frames, query_frames), 0)[order])
        all_weights.append(torch.cat((support_weights, query_weights), 0)[order])
        if config.global_classifier:
            global_sum = global_sum + torch.cat((support_globals, query_globals), 0)[order]
    return (torch.cat(all_frames, 1), torch.cat(all_weights, 1), global_sum,
            _merge_outputs(all_outputs, class_axis=True))


def text_concat_episode(model, features, metadata, raw=None):
    config = model.text_concat_st
    device = features.device
    support_mask = metadata['support_mask'].to(device).bool()
    si = support_mask.nonzero(as_tuple=False).flatten()
    qi = (~support_mask).nonzero(as_tuple=False).flatten()
    ids = metadata['episode_class_ids'].to(device).long()
    ids = (ids[0] if ids.ndim == 2 else ids).flatten()
    if not si.numel() or not qi.numel() or not ids.numel():
        raise ValueError('Text-concat episodes require Support, Query and candidate classes.')
    labels = metadata['episode_positive_labels'].to(device).index_select(0, si)
    if labels.shape != (si.numel(), ids.numel()):
        raise ValueError('Support labels must match episode [S,K].')
    known = torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
    if 'episode_label_known_mask' in metadata:
        known = known & metadata['episode_label_known_mask'].to(device).bool().index_select(0, si)
    positive = (labels > .5) & known
    mask = metadata['pred_query_mask' if model.cfg.POINT_INFO.USE_PT_QUERY_MASK
                    else 'pred_visibility'].to(device).bool()
    if features.ndim != 4 or mask.shape != features.shape[:3]:
        raise ValueError('Text-concat requires aligned features [B,T,N,D] and mask [B,T,N].')
    # Fully shared encoding must not tokenize, cache or run candidate text.
    text = (None if config.shared_support and config.shared_query
            else model._get_pot_label_text_features(ids, features.dtype))
    verifier = model.query_evidence_verifier
    if config.shared_support and config.shared_query:
        frames, global_features = _shared_visual_encodings(model, features, mask)
        support = frames.index_select(0, si)
        query_prototypes = frames.index_select(0, qi)
    elif config.shared_support:
        evidence_mask = mask & metadata.get('pred_visibility', mask).to(device).bool()
        support, query_prototypes, query_weights, global_features, outputs = _shared_support_encodings(
            model, features, raw, mask, evidence_mask, text, si, qi, labels, known)
    elif config.shared_query:
        support, query_prototypes, support_weights, global_features = _shared_query_encodings(
            model, features, mask, text, si, qi)
    elif verifier is not None:
        evidence_mask = mask & metadata.get('pred_visibility', mask).to(device).bool()
        frames, weights, global_sum, outputs = _verified_encodings(
            model, features, raw, mask, evidence_mask, text, si, qi, labels, known)
    else:
        # Keep the v1 execution path intact, including its video/RNG ordering.
        all_frames, all_weights = [], []
        global_sum = features.new_zeros(features.shape[0], features.shape[-1], dtype=torch.float32)
        for class_text in text.split(config.class_chunk_size):
            parts = []
            for start in range(0, features.shape[0], config.video_chunk_size):
                end = start + config.video_chunk_size
                args = (features[start:end], mask[start:end], class_text)
                function = partial(_encode_chunk, model)
                parts.append(checkpoint(function, *args, use_reentrant=False)
                             if config.activation_checkpoint and model.training and torch.is_grad_enabled()
                             else function(*args))
            frames, weights, globals_ = zip(*parts)
            all_frames.append(torch.cat(frames, 0))
            all_weights.append(torch.cat(weights, 0))
            if config.global_classifier:
                global_sum = global_sum + torch.cat(globals_, 0)
        frames = torch.cat(all_frames, 1)
        weights = torch.cat(all_weights, 1)
    if not config.shared_query and not config.shared_support:
        support = frames.index_select(0, si)
        query_prototypes = frames.index_select(0, qi)
        global_features = global_sum / ids.numel()
    if config.shared_support:
        # Only labels form the class axis, after shared visual encoding.
        # No [S,K,T,N,D] expansion; multi-label positives may serve several classes.
        with torch.autocast(device_type=device.type, enabled=False):
            support_prototypes = torch.einsum('sk,std->ktd', positive.float(), support.float())
            support_prototypes = support_prototypes / positive.sum(0).clamp_min(1)[:, None, None]
    else:
        pos = positive[:, :, None, None]
        support_prototypes = (support * pos).sum(0) / pos.sum(0).clamp_min(1)
    diag = model._compute_bidirectional_frame_similarity(query_prototypes, support_prototypes)
    alpha = float(model.pot_route_cfg.QUERY_PARTIAL_LOGIT_ALPHA)
    bias = float(model.pot_route_cfg.QUERY_PARTIAL_LOGIT_BIAS)
    base = alpha * diag + bias
    result = dict(
        single_path_episode=True, query_evidence_enabled=verifier is not None,
        query_partial_q2s_logits=base, query_partial_q2s_base_logits=base,
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
        query_class_matchability=torch.ones_like(base),
    )
    if config.shared_support:
        result.update(shared_support=True, shared_query=config.shared_query)
        if not config.shared_query:
            result.update(text_concat_query_spatial_weights=query_weights,
                          text_concat_query_sample_indices=qi)
    elif config.shared_query:
        result.update(shared_query=True, text_concat_support_spatial_weights=support_weights,
                      text_concat_support_sample_indices=si)
    else:
        result['text_concat_spatial_weights'] = weights
    if verifier is not None:
        query_weights = query_weights if config.shared_support else weights[qi]
        query_valid = evidence_mask[qi]
        result = finish_query_evidence_aux(
            model, result, query_prototypes, support_prototypes, query_weights, query_valid,
            normalize_patch_region_weights(query_weights, query_valid), outputs)
    if config.global_classifier:
        result['global_classification_logits'] = model.head(
            model.head_drop(model.pre_logits(global_features)))
    return _detach_diagnostics(result)
