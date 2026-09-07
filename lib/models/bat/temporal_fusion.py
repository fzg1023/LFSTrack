"""
Temporal Convolutional Fusion (TCN-based) — CADTrack-style.

Matches CADTrack's design:
  - Conv1d per-token temporal modeling (O(T), causal)
  - Dilated convolutions for multi-scale temporal receptive field
  - Residual connection with LayerNorm
  - Per-frame output (not pooled), enabling per-frame loss

Unlike CADTrack's dual-stream (separate RGB/TIR TCNs), LFSTrack's
single-stream uses a single TCN on fused features.
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath


class TemporalConvBlock(nn.Module):
    """Single TCN layer: depthwise Conv1d + pointwise + residual.

    CADTrack-style: causal padding (left-only), dilated, depthwise-separable.
    """
    def __init__(self, channels, kernel_size=3, dilation=1, drop_path=0.):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: pad left only
        self.conv_dw = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation, groups=channels)
        self.conv_pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B*N, T, C]
        x_t = x.transpose(1, 2)                    # [B*N, C, T]
        out = self.conv_dw(x_t)
        out = out[..., :x.size(1)]                  # trim causal padding
        out = self.conv_pw(out)
        out = out.transpose(1, 2)                   # [B*N, T, C]
        return x + self.drop_path(self.act(self.norm(out)))


class LFSTrackTemporalFusion(nn.Module):
    """TCN-based temporal fusion for single-stream LFSTrack.

    Matches CADTrack's TCNTemporalFusion design:
      - Per-token Conv1d across time
      - Dilated multi-layer for enlarged receptive field
      - Causal: frame_t only sees frames [t-2, t-1, t]
      - Residual + LayerNorm at input and output
      - Zero-init: last conv_pw zeroed → gentle start

    Args:
        d_model:    feature dimension (768 for ViT-B)
        num_layers: number of TCN layers (default 2)
        kernel_size: temporal kernel size (default 3 → sees 3 frames)
    """
    def __init__(self, d_model=768, num_layers=2, kernel_size=3, drop_path=0.):
        super().__init__()
        self.d_model = d_model
        self.norm_in = nn.LayerNorm(d_model)
        self.tcn_layers = nn.ModuleList([
            TemporalConvBlock(d_model, kernel_size, dilation=2**i, drop_path=drop_path)
            for i in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # Zero-init: last layer outputs ≈0 → TCN starts as identity
        for layer in self.tcn_layers:
            nn.init.constant_(layer.conv_pw.weight, 0.0)
            nn.init.constant_(layer.conv_pw.bias, 0.0)
        nn.init.constant_(self.tcn_layers[-1].conv_dw.weight, 0.0)

        # Zero-init norm_in and norm_out so x + TCN(x) ≈ x at init
        # LayerNorm(w,b): output = w * (x-μ)/σ + b. With w=0,b=0: output=0 → residual = x
        nn.init.constant_(self.norm_in.weight, 0.0)
        nn.init.constant_(self.norm_in.bias, 0.0)
        nn.init.constant_(self.norm_out.weight, 0.0)
        nn.init.constant_(self.norm_out.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: [B, T, N, C] — T frames, N tokens per frame, C channels
        Returns:
            out: [B, T, N, C] — temporally enhanced features (same shape)
        """
        B, T, N, C = x.shape
        # Per-token TCN: [B, T, N, C] → [B*N, T, C]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, C).contiguous()
        x_flat = self.norm_in(x_flat)
        for layer in self.tcn_layers:
            x_flat = layer(x_flat)
        x_flat = self.norm_out(x_flat)
        # Back: [B*N, T, C] → [B, T, N, C]
        x_out = x_flat.view(B, N, T, C).permute(0, 2, 1, 3)
        return x + x_out  # residual connection (CADTrack-style)
