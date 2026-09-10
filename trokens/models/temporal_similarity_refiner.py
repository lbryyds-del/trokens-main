"""Bounded temporal refinement of class responses along tracked points."""

import math

import torch
from torch import nn


class TrajectoryTemporalSimilarityRefiner(nn.Module):
    """Refine ``[K,T,N]`` similarities without mixing classes or trajectories.

    ``point_mask`` is the observation-validity mask ``[T,N]``, not merely a
    trajectory seed-frame mask. Invalid observations never enter the temporal
    convolution, including when the optional visibility *channel* is disabled.
    The spatial softmax and its existing routing mask remain the caller's job.
    """

    def __init__(
        self,
        hidden_dim=16,
        kernel_size=3,
        max_delta=0.05,
        gate_init=0.1,
        use_visibility=True,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0 or int(kernel_size) != kernel_size:
            raise ValueError("kernel_size must be a positive odd integer.")
        if hidden_dim <= 0 or int(hidden_dim) != hidden_dim:
            raise ValueError("hidden_dim must be a positive integer.")
        if not math.isfinite(max_delta) or max_delta < 0:
            raise ValueError("max_delta must be finite and non-negative.")
        if not math.isfinite(gate_init):
            raise ValueError("gate_init must be finite.")

        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.max_delta = float(max_delta)
        self.use_visibility = bool(use_visibility)
        self.temporal_conv = nn.Conv1d(
            2 if self.use_visibility else 1,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.act = nn.GELU()
        self.out_proj = nn.Conv1d(self.hidden_dim, 1, kernel_size=1)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, similarity, point_mask):
        if similarity.ndim != 3:
            raise ValueError("similarity must have shape [K,T,N].")
        if not similarity.is_floating_point():
            raise ValueError("similarity must be floating point.")
        k, t, n = similarity.shape
        if point_mask.ndim != 2 or tuple(point_mask.shape) != (t, n):
            raise ValueError("point_mask must match similarity [T,N].")

        # The route supplies finite cosines. Sanitization also makes standalone
        # calls safe; exact zero-gate identity applies to finite [-1,1] inputs.
        scores = torch.nan_to_num(
            similarity.float(), nan=0.0, posinf=1.0, neginf=-1.0,
        ).clamp(-1.0, 1.0)
        if similarity.numel() == 0:
            return scores.to(dtype=similarity.dtype)
        valid = point_mask.to(device=similarity.device).bool()
        visibility_knt = valid.T.unsqueeze(0).expand(k, -1, -1)
        masked_scores = scores.permute(0, 2, 1) * visibility_knt
        if self.use_visibility:
            temporal_input = torch.stack(
                [masked_scores, visibility_knt.to(scores.dtype)], dim=2,
            ).reshape(k * n, 2, t)
        else:
            temporal_input = masked_scores.reshape(k * n, 1, t)

        # Each Conv batch row is one (class, trajectory). Only T is convolved.
        # Keep the small residual in FP32 even under training autocast.
        with torch.autocast(device_type=similarity.device.type, enabled=False):
            delta = self.out_proj(self.act(self.temporal_conv(temporal_input)))
            delta = delta.reshape(k, n, t).permute(0, 2, 1)
            correction = self.max_delta * self.gate.tanh() * delta.tanh()
            refined = scores + correction * valid.unsqueeze(0)
        refined = torch.nan_to_num(
            refined, nan=0.0, posinf=1.0, neginf=-1.0,
        ).clamp(-1.0, 1.0)
        return refined.to(dtype=similarity.dtype)
