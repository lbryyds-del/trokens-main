"""Same-condition positive/negative Support verification of Query evidence.

Only Post relative evidence rho is learned here. Trajectory-change residuals
live in entity_change_verifier; neither branch modifies visual prototypes.
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


def _frame_mean(value, valid, fallback=0.0):
    valid = valid.expand_as(value)
    count = valid.sum(dim=-1)
    mean = (value.float() * valid.float()).sum(dim=-1) / count.clamp_min(1)
    return torch.where(count > 0, mean, torch.full_like(mean, fallback))


class QueryEvidenceVerifier(nn.Module):
    def __init__(self, input_dim, cfg, uniqueness=True):
        super().__init__()
        self.uniqueness = uniqueness
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
        # Keep the rho-only checkpoint version and scalar names unchanged.
        self.register_buffer("architecture_version", torch.tensor(2, dtype=torch.int64))
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

    def forward(self, *, text, post_query, post_references, query_weights,
                query_valid, support_frame_valid, support_labels, support_known=None):
        """Support labels alone define references; evidence outputs are [Q,K,T].

        post_references is [S,K,T,D], not the class-averaged matcher prototype.
        """
        precision = (nullcontext() if self.amp_enabled and post_query.is_cuda
                     else torch.autocast(device_type=post_query.device.type, enabled=False))
        with precision:
            return self._forward_impl(
                text, post_query, post_references, query_weights, query_valid,
                support_frame_valid, support_labels, support_known,
            )

    def _forward_impl(self, text, post_query, post_references, query_weights,
                      query_valid, support_frame_valid, support_labels, support_known):
        q, t, n, _ = post_query.shape
        k = text.shape[0]
        positive_labels, negative_labels = self._support_signs(support_labels, support_known, k)
        frame_valid = query_valid.any(dim=-1)
        if self.uniqueness:
            qp = _unit(self._input(post_query))
            sp = _unit(self._input(post_references))
            pi = self._input(query_weights).float()
        output = {}

        def masks(valid_frames):
            # [K,S,T] -> [K,R], preserving per-class Support identities.
            pos = (valid_frames & positive_labels.T[:, :, None]).flatten(1)
            neg = (valid_frames & negative_labels.T[:, :, None]).flatten(1)
            valid = torch.stack((pos, neg), dim=1)
            available = valid.any(dim=-1)
            # Retain the safe dummy element used for empty reference sets.
            return torch.cat((valid, ~available.unsqueeze(-1)), dim=-1), available

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
        build_class_support_references, normalize_patch_region_weights,
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
    if not judgments.uniqueness:
        result.update(query_partial_q2s_logits=base_logits,
                      query_partial_q2s_temporal_logits=base_logits,
                      query_class_matchability=torch.ones_like(base_logits))
        return result
    verifier = getattr(model, "query_evidence_verifier", None)
    if verifier is None:
        raise RuntimeError("Learned verifier must be registered at model construction, before optimizer/DDP.")
    if verifier.uniqueness != judgments.uniqueness:
        raise ValueError("Rebuild the model after changing learned judgment switches.")
    references = support_frame_valid = None
    if judgments.uniqueness:
        references, support_frame_valid, _, _, _ = build_class_support_references(
            model, values, point_mask, visibility, support_mask, labels, route_text,
            precomputed_prototypes=routed,
        )
    query_valid = evidence_mask.index_select(0, query_indices)
    hypothesis_weights = normalize_patch_region_weights(query_weights, query_valid)
    outputs = verifier(
        text=text, post_query=values.index_select(0, query_indices),
        post_references=references, query_weights=hypothesis_weights,
        query_valid=query_valid,
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
    if not judgments.uniqueness:
        result.update(query_partial_q2s_logits=base_logits,
                      query_partial_q2s_temporal_logits=base_logits,
                      query_class_matchability=torch.ones_like(base_logits))
        return result
    from trokens.models.query_matching import (
        classwise_frame_similarity, rho_frame_hypothesis_states,
        hypothesis_mass_bimhm_logits,
    )
    verifier = model.query_evidence_verifier
    alpha = float(getattr(model.pot_route_cfg, "QUERY_PARTIAL_LOGIT_ALPHA", 10.0))
    shape = (*base_logits.shape, query_prototypes.shape[2])
    neutral = base_logits.new_ones(shape, dtype=torch.float32)
    rho = outputs.get("relation", neutral)
    states = rho_frame_hypothesis_states(rho)
    strength = float(getattr(cfg, "ABSOLUTE_MASS_TRANSPORT_STRENGTH", 1.0))
    if model.training:
        strength *= verifier.transport_progress
    transport = hypothesis_mass_bimhm_logits(
        classwise_frame_similarity(query_prototypes, support_prototypes),
        states["target_mass"], base_logits, alpha=alpha, transport_strength=strength,
        unmatched_cost=float(getattr(cfg, "ABSOLUTE_MASS_UNMATCHED_COST", 0.0)),
        one_sided_transport=bool(getattr(cfg, "ABSOLUTE_MASS_ONE_SIDED", True)),
    )
    frame_valid = query_valid.any(dim=-1).unsqueeze(1).expand_as(rho)
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
