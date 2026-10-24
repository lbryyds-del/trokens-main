"""Symmetric candidate-text concatenation and same-attention spatial readout.

Support and Query share one encoder. Only Support labels form prototypes;
Query truth is never used for encoding, conditioning or spatial aggregation.
"""
from functools import partial
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from trokens.models.label_conditioned_routing import _detach_diagnostics, _merge_outputs
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
    text = model._get_pot_label_text_features(ids, features.dtype)
    verifier = model.query_evidence_verifier
    if verifier is not None:
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
    support = frames.index_select(0, si)
    pos = positive[:, :, None, None]
    support_prototypes = (support * pos).sum(0) / pos.sum(0).clamp_min(1)
    query_prototypes = frames.index_select(0, qi)
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
        text_concat_spatial_weights=weights,
    )
    if verifier is not None:
        query_weights, query_valid = weights[qi], evidence_mask[qi]
        result = finish_query_evidence_aux(
            model, result, query_prototypes, support_prototypes, query_weights, query_valid,
            normalize_patch_region_weights(query_weights, query_valid), outputs)
    if config.global_classifier:
        result['global_classification_logits'] = model.head(
            model.head_drop(model.pre_logits(global_sum / ids.numel())))
    return _detach_diagnostics(result)
