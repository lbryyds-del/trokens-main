"""Lightweight temporal residuals for class-conditioned trajectory routing."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TrajectoryTemporalSimilarityRefiner(nn.Module):
    """Return temporal routing-logit residuals for ``[K,T,N]`` scores.

    A leading batch dimension is also accepted.  Flattening ``B*K*N`` into
    one Conv1D batch keeps Support/Query samples independent while avoiding
    dozens of tiny CUDA launches per episode.
    """

    def __init__(
        self,
        hidden_dim=16,
        kernel_size=3,
        long_dilation=2,
        max_logit_delta=0.5,
        use_visibility=True,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0 or int(kernel_size) != kernel_size:
            raise ValueError("kernel_size must be a positive odd integer.")
        if hidden_dim <= 0 or int(hidden_dim) != hidden_dim:
            raise ValueError("hidden_dim must be a positive integer.")
        if long_dilation <= 0 or int(long_dilation) != long_dilation:
            raise ValueError("long_dilation must be a positive integer.")
        if not math.isfinite(max_logit_delta) or max_logit_delta < 0:
            raise ValueError("max_logit_delta must be finite and non-negative.")

        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.long_dilation = int(long_dilation)
        self.max_logit_delta = float(max_logit_delta)
        self.use_visibility = bool(use_visibility)
        self.temporal_conv = nn.Conv1d(
            3 if self.use_visibility else 2,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.long_temporal_conv = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size // 2) * self.long_dilation,
            dilation=self.long_dilation,
        )
        self.act = nn.GELU()
        self.out_proj = nn.Conv1d(self.hidden_dim, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    @staticmethod
    def _linear_conv1d(value, layer):
        """Apply a Conv1D exactly as temporal windows followed by a Linear.

        cuDNN's deterministic kernel is extremely slow for the large number
        of independent length-8 trajectories used here, especially for the
        dilated 16-to-16 layer.  This equivalent formulation stays on the
        deterministic GEMM path and retains the Conv1D parameters/state dict.
        """
        if layer.groups != 1 or layer.stride != (1,):
            raise ValueError("The lightweight temporal conv requires groups=1/stride=1.")
        kernel_size = layer.kernel_size[0]
        dilation = layer.dilation[0]
        padding = layer.padding[0]
        temporal_dim = value.shape[-1]
        padded = F.pad(value, (padding, padding))
        windows = torch.stack(
            [
                padded[..., offset:offset + temporal_dim]
                for offset in range(0, kernel_size * dilation, dilation)
            ],
            dim=-1,
        )
        rows = windows.permute(0, 2, 1, 3).reshape(
            -1,
            layer.in_channels * kernel_size,
        )
        output = F.linear(
            rows,
            layer.weight.reshape(layer.out_channels, -1),
            layer.bias,
        )
        return output.reshape(
            value.shape[0],
            temporal_dim,
            layer.out_channels,
        ).permute(0, 2, 1)

    def forward(self, similarity, point_mask):
        if similarity.ndim not in (3, 4):
            raise ValueError("similarity must have shape [K,T,N] or [B,K,T,N].")
        if not similarity.is_floating_point():
            raise ValueError("similarity must be floating point.")
        squeeze_batch = similarity.ndim == 3
        if squeeze_batch:
            similarity = similarity.unsqueeze(0)
            point_mask = point_mask.unsqueeze(0)
        b, k, t, n = similarity.shape
        if point_mask.ndim != 3 or tuple(point_mask.shape) != (b, t, n):
            raise ValueError("point_mask must match similarity [B,T,N].")

        scores = torch.nan_to_num(
            similarity.float(), nan=0.0, posinf=1.0, neginf=-1.0,
        ).clamp(-1.0, 1.0)
        if similarity.numel() == 0:
            empty = torch.zeros_like(scores)
            return empty.squeeze(0) if squeeze_batch else empty
        valid = point_mask.to(device=similarity.device).bool()
        visibility_bknt = (
            valid.permute(0, 2, 1).unsqueeze(1).expand(-1, k, -1, -1)
        )
        masked_scores = scores.permute(0, 1, 3, 2) * visibility_bknt
        previous_valid = torch.zeros_like(valid)
        if t > 1:
            previous_valid[:, 1:] = valid[:, :-1]
        pair_valid = valid & previous_valid
        diff = torch.zeros_like(scores)
        if t > 1:
            diff[:, :, 1:] = (
                (scores[:, :, 1:] - scores[:, :, :-1])
                * pair_valid[:, 1:].unsqueeze(1)
            )
        channels = [masked_scores, diff.permute(0, 1, 3, 2)]
        if self.use_visibility:
            channels.append(visibility_bknt.to(scores.dtype))
        # Keep the channel axis adjacent to each (batch, trajectory) row
        # before flattening; otherwise rows would mix neighboring trajectories.
        temporal_input = torch.stack(channels, dim=3).reshape(
            b * k * n, len(channels), t,
        )
        valid_rows = visibility_bknt.reshape(b * k * n, 1, t).to(scores.dtype)

        # Each Conv batch row is one (class, trajectory). Only T is convolved.
        # Keep the small residual in FP32 even under training autocast.
        with torch.autocast(device_type=similarity.device.type, enabled=False):
            hidden = self.act(
                self._linear_conv1d(temporal_input, self.temporal_conv)
            ) * valid_rows
            hidden = self.act(
                self._linear_conv1d(hidden, self.long_temporal_conv)
            ) * valid_rows
            output = self._linear_conv1d(hidden, self.out_proj).reshape(
                b, k, n, t,
            ).permute(0, 1, 3, 2)
            delta_logits = self.max_logit_delta * output.tanh()
            delta_logits = delta_logits * valid.unsqueeze(1).to(delta_logits.dtype)
        delta_logits = torch.nan_to_num(
            delta_logits, nan=0.0, posinf=0.0, neginf=0.0,
        )
        return delta_logits.squeeze(0) if squeeze_batch else delta_logits
