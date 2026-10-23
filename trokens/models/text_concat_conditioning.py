"""Symmetric candidate-text concatenation and same-attention spatial readout.

Support and Query share one encoder. Only Support labels form prototypes;
Query truth is never used for encoding, conditioning or spatial aggregation.
"""
from functools import partial
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from trokens.models.label_conditioned_routing import _detach_diagnostics


class TextConcatST(nn.Module):
    def __init__(self, cfg, global_classifier=True):
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
        self.register_buffer('architecture_signature', torch.tensor(
            [1, self.temporal_text_tokens, self.spatial_text_tokens, int(global_classifier)],
            dtype=torch.int64))


def _encode_chunk(model, features, mask, text):
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
    readout = partial(_pool_chunk, model, b, c, t, n, d)
    return (checkpoint(readout, encoded, weights, use_reentrant=False)
            if config.readout_checkpoint and model.training and torch.is_grad_enabled()
            else readout(encoded, weights))


def _pool_chunk(model, b, c, t, n, d, encoded, weights):
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
    return (frames.reshape(b, c, t, d), weights.reshape(b, c, t, n),
            global_features.float().reshape(b, c, d).sum(1))


def text_concat_episode(model, features, metadata):
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
        single_path_episode=True, query_evidence_enabled=False,
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
    if config.global_classifier:
        result['global_classification_logits'] = model.head(
            model.head_drop(model.pre_logits(global_sum / ids.numel())))
    return _detach_diagnostics(result)
