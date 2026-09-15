"""Visual-guided class-response attention along tracked trajectories."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


AuxOutput = Tuple[Tensor, Dict[str, Tensor]]


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Apply softmax while mapping an all-masked row to exact zeros."""
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if logits.shape[dim] == 0:
        return torch.zeros_like(logits)
    weights = torch.softmax(logits.masked_fill(~mask, -1e4), dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    normalizer = weights.sum(dim=dim, keepdim=True)
    return torch.where(
        normalizer > 0,
        weights / normalizer.clamp_min(1e-12),
        torch.zeros_like(weights),
    )


class TrajectoryCostAttentionRefiner(nn.Module):
    """Return bounded spatial-logit corrections for trajectory observations.

    ``similarity`` is ``[K,T,N]`` or ``[B,K,T,N]`` and contains cosine
    responses before the spatial temperature is applied. ``patch_tokens`` is
    the matching ``[T,N,D]`` or ``[B,T,N,D]`` visual tensor. Attention is
    independent across videos, classes and trajectories, and is local only in
    time.

    The additional branch deliberately detaches both inputs. The canonical
    similarity path and the final prototype aggregation remain attached to the
    visual backbone in :class:`Pointformer`, while this module learns only how
    to turn fixed per-step evidence into a routing-logit correction.
    """

    def __init__(
        self,
        feature_dim: int,
        cost_dim: int = 16,
        guidance_dim: int = 16,
        attention_dim: int = 16,
        radius: int = 2,
        max_logit_delta: float = 0.5,
    ) -> None:
        super().__init__()
        for name, value in {
            "feature_dim": feature_dim,
            "cost_dim": cost_dim,
            "guidance_dim": guidance_dim,
            "attention_dim": attention_dim,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            raise ValueError("radius must be a non-negative integer.")
        if not math.isfinite(max_logit_delta) or max_logit_delta < 0:
            raise ValueError("max_logit_delta must be finite and non-negative.")

        self.feature_dim = feature_dim
        self.cost_dim = cost_dim
        self.guidance_dim = guidance_dim
        self.attention_dim = attention_dim
        self.radius = radius
        self.max_logit_delta = float(max_logit_delta)

        self.cost_embed = nn.Sequential(
            nn.Linear(1, cost_dim),
            nn.GELU(),
            nn.Linear(cost_dim, cost_dim),
        )
        self.visual_proj = nn.Linear(feature_dim, guidance_dim, bias=False)
        self.visual_norm = nn.LayerNorm(guidance_dim)
        qk_dim = cost_dim + guidance_dim
        self.q_proj = nn.Linear(qk_dim, attention_dim, bias=False)
        self.k_proj = nn.Linear(qk_dim, attention_dim, bias=False)
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
        self.relative_bias = nn.Parameter(-offsets.abs() / 2.0)
        self.out_proj = nn.Linear(cost_dim, 1, bias=False)

        # Start exactly from the no-temporal route. Only this final projection
        # is zeroed so upstream attention layers receive gradients after its
        # first update without a second zero-valued gate.
        nn.init.zeros_(self.out_proj.weight)

    def _validate_parameter_dtype(self) -> None:
        non_fp32 = [
            name for name, parameter in self.named_parameters()
            if parameter.dtype != torch.float32
        ]
        if non_fp32:
            raise ValueError(
                "Keep temporal refiner parameters FP32 and use autocast; "
                f"non-FP32 parameters: {', '.join(non_fp32)}."
            )

    def forward(
        self,
        similarity: Tensor,
        patch_tokens: Tensor,
        temporal_mask: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, AuxOutput]:
        if similarity.ndim not in (3, 4):
            raise ValueError("similarity must have shape [K,T,N] or [B,K,T,N].")
        if not similarity.is_floating_point() or not patch_tokens.is_floating_point():
            raise ValueError("similarity and patch_tokens must be floating point.")

        single_video = similarity.ndim == 3
        if single_video:
            if patch_tokens.ndim != 3 or temporal_mask.ndim != 2:
                raise ValueError(
                    "Single-video input requires patch_tokens [T,N,D] and "
                    "temporal_mask [T,N]."
                )
            similarity = similarity.unsqueeze(0)
            patch_tokens = patch_tokens.unsqueeze(0)
            temporal_mask = temporal_mask.unsqueeze(0)

        batch, num_classes, temporal_dim, num_tracks = similarity.shape
        expected_tokens = (
            batch,
            temporal_dim,
            num_tracks,
            self.feature_dim,
        )
        if patch_tokens.ndim != 4 or tuple(patch_tokens.shape) != expected_tokens:
            raise ValueError(
                "patch_tokens must match similarity as [B,T,N,feature_dim]; "
                f"expected {expected_tokens}, got {tuple(patch_tokens.shape)}."
            )
        expected_mask = (batch, temporal_dim, num_tracks)
        if temporal_mask.ndim != 3 or tuple(temporal_mask.shape) != expected_mask:
            raise ValueError(
                "temporal_mask must match similarity as [B,T,N]; "
                f"expected {expected_mask}, got {tuple(temporal_mask.shape)}."
            )
        if patch_tokens.device != similarity.device:
            raise ValueError("patch_tokens and similarity must be on the same device.")
        if next(self.parameters()).device != similarity.device:
            raise ValueError("Move the temporal refiner to the input device first.")
        self._validate_parameter_dtype()

        output_shape = (batch, num_classes, temporal_dim, num_tracks)
        if similarity.numel() == 0:
            delta = torch.zeros(output_shape, device=similarity.device, dtype=torch.float32)
            attention = None
            if return_aux:
                attention = delta.new_zeros(
                    batch,
                    num_classes,
                    num_tracks,
                    temporal_dim,
                    temporal_dim,
                )
            return self._format_output(delta, attention, single_video, return_aux)

        valid = temporal_mask.to(device=similarity.device, dtype=torch.bool)
        valid_bnt = valid.permute(0, 2, 1)

        # Keep Linear projections under the caller's autocast context. Both
        # inputs are detached only for this extra temporal branch.
        scores = torch.nan_to_num(
            similarity.detach().float(),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        scores_bknt = scores.permute(0, 1, 3, 2)
        cost = self.cost_embed(scores_bknt.unsqueeze(-1))
        # Parameter-free normalization balances the learned response embedding
        # against the LayerNorm-normalized visual guidance without increasing
        # the documented 17,765-parameter design.
        cost = F.layer_norm(cost.float(), (self.cost_dim,))
        cost = cost * valid_bnt[:, None, :, :, None].to(cost.dtype)

        visual = torch.nan_to_num(
            patch_tokens.detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        autocast_enabled = torch.is_autocast_enabled()
        if similarity.device.type == "cpu":
            autocast_enabled = torch.is_autocast_cpu_enabled()
        if not autocast_enabled:
            visual = visual.float()
        visual_eps = 1e-6 if visual.dtype in (torch.float16, torch.bfloat16) else 1e-12
        visual = F.normalize(visual, dim=-1, eps=visual_eps)
        guidance = self.visual_norm(self.visual_proj(visual)).float()
        guidance = guidance.permute(0, 2, 1, 3)

        # ``Linear([E;G])`` is evaluated as two projections plus a broadcasted
        # sum. This is mathematically identical to concatenation but avoids
        # copying the class-shared visual guidance K times.
        query = F.linear(
            cost,
            self.q_proj.weight[:, :self.cost_dim],
        ) + F.linear(
            guidance,
            self.q_proj.weight[:, self.cost_dim:],
        )[:, None]
        key = F.linear(
            cost,
            self.k_proj.weight[:, :self.cost_dim],
        ) + F.linear(
            guidance,
            self.k_proj.weight[:, self.cost_dim:],
        )[:, None]

        # For short clips, offsets outside the clip have no valid source.
        # Exclude them instead of padding an empty slice to more than T frames.
        effective_radius = min(self.radius, temporal_dim - 1)
        offsets = tuple(range(-effective_radius, effective_radius + 1))

        # The short attention, mask normalization, and bounded residual use
        # FP32 for stable all-masked behavior under CUDA AMP.
        with torch.autocast(device_type=similarity.device.type, enabled=False):
            query = F.layer_norm(query.float(), (self.attention_dim,))
            key = F.layer_norm(key.float(), (self.attention_dim,))
            attention_logits_by_offset = []
            allowed_by_offset = []
            for offset in offsets:
                if offset < 0:
                    target_slice = slice(-offset, temporal_dim)
                    source_slice = slice(0, temporal_dim + offset)
                    time_padding = (-offset, 0)
                elif offset > 0:
                    target_slice = slice(0, temporal_dim - offset)
                    source_slice = slice(offset, temporal_dim)
                    time_padding = (0, offset)
                else:
                    target_slice = slice(0, temporal_dim)
                    source_slice = target_slice
                    time_padding = (0, 0)

                offset_logits = (
                    query[..., target_slice, :]
                    * key[..., source_slice, :]
                ).sum(dim=-1) / math.sqrt(self.attention_dim)
                offset_logits = F.pad(offset_logits, time_padding)
                attention_logits_by_offset.append(
                    offset_logits + self.relative_bias[offset + self.radius]
                )

                offset_allowed = (
                    valid_bnt[..., target_slice]
                    & valid_bnt[..., source_slice]
                )
                allowed_by_offset.append(F.pad(offset_allowed, time_padding))

            attention_logits = torch.stack(attention_logits_by_offset, dim=-1)
            allowed_bntw = torch.stack(allowed_by_offset, dim=-1)
            local_attention = _masked_softmax(
                attention_logits,
                allowed_bntw[:, None],
                dim=-1,
            )
            context = torch.zeros_like(cost, dtype=torch.float32)
            for offset_index, offset in enumerate(offsets):
                if offset < 0:
                    target_slice = slice(-offset, temporal_dim)
                    source_slice = slice(0, temporal_dim + offset)
                elif offset > 0:
                    target_slice = slice(0, temporal_dim - offset)
                    source_slice = slice(offset, temporal_dim)
                else:
                    target_slice = slice(0, temporal_dim)
                    source_slice = target_slice
                context[..., target_slice, :] = (
                    context[..., target_slice, :]
                    + local_attention[..., target_slice, offset_index].unsqueeze(-1)
                    * cost.float()[..., source_slice, :]
                )

            # An observation must have at least one valid *external* neighbor.
            # A self-only row is therefore an exact no-op by construction.
            external_offset = torch.tensor(
                [offset != 0 for offset in offsets],
                device=similarity.device,
                dtype=torch.bool,
            )
            has_external_neighbor = (allowed_bntw & external_offset).any(dim=-1)
            difference = context - cost.float()
            raw_delta = F.linear(difference, self.out_proj.weight).squeeze(-1)
            delta_bknt = self.max_logit_delta * torch.tanh(raw_delta)
            delta_bknt = delta_bknt * has_external_neighbor[:, None].to(
                delta_bknt.dtype
            )
            delta = delta_bknt.permute(0, 1, 3, 2).contiguous()
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            attention = None
            if return_aux:
                attention = local_attention.new_zeros(
                    batch,
                    num_classes,
                    num_tracks,
                    temporal_dim,
                    temporal_dim,
                )
                for offset_index, offset in enumerate(offsets):
                    if offset < 0:
                        target_slice = slice(-offset, temporal_dim)
                        source_slice = slice(0, temporal_dim + offset)
                    elif offset > 0:
                        target_slice = slice(0, temporal_dim - offset)
                        source_slice = slice(offset, temporal_dim)
                    else:
                        target_slice = slice(0, temporal_dim)
                        source_slice = target_slice
                    attention[
                        ...,
                        target_slice,
                        source_slice,
                    ] += torch.diag_embed(
                        local_attention[
                            ...,
                            target_slice,
                            offset_index,
                        ]
                    )

        return self._format_output(delta, attention, single_video, return_aux)

    @staticmethod
    def _format_output(
        delta: Tensor,
        attention: Optional[Tensor],
        single_video: bool,
        return_aux: bool,
    ) -> Union[Tensor, AuxOutput]:
        if single_video:
            delta = delta[0]
            if attention is not None:
                attention = attention[0]
        if not return_aux:
            return delta
        if attention is None:
            raise RuntimeError("Attention diagnostics were not constructed.")
        return delta, {
            "attention": attention,
            "delta_logits": delta,
        }
