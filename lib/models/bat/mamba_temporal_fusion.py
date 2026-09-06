"""
Real Mamba2 Temporal Fusion (using mamba_ssm).

Drop-in replacement for LFSTrackTemporalFusion (TCN-based).
Uses mamba_ssm.Mamba2 for true selective state space modeling:
  - Hardware-optimized selective scan (via causal_conv1d)
  - Input-dependent gating + SSM dynamics
  - O(T) linear complexity per token
  - Zero-init -> identity at start, safe for pre-trained backbone

Requires: mamba_ssm >= 2.0, causal_conv1d

Interface: same as LFSTrackTemporalFusion
  - Input:  [B, T, N, C]
  - Output: [B, T, N, C]
"""
import torch
import torch.nn as nn


class Mamba2TemporalBlock(nn.Module):
    """Single Mamba2 temporal mixing block with zero-init.

    Wraps mamba_ssm.Mamba2 with:
      - Pre-norm (LayerNorm, zero-init)
      - Mamba2 core (selective SSM + conv1d + gating)
      - Output scale (zero-init -> residual identity at start)
      - Residual connection

    Args:
        d_model:    feature dimension
        d_state:    SSM state dimension (default 16)
        d_conv:     local conv kernel size (default 4)
        expand:     expansion factor (default 2)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = None  # lazy init for device/dtype placement
        self._d_model = d_model
        self._d_state = d_state
        self._d_conv = d_conv
        self._expand = expand

        # Output scale: zero-init -> mamba output is zeroed -> residual = x
        self.out_scale = nn.Parameter(torch.zeros(1, 1, d_model))

        # LayerNorm zero-init: w=0,b=0 -> norm_out = 0 -> residual = x
        nn.init.constant_(self.norm.weight, 0.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def _init_mamba(self, device, dtype):
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(
            d_model=self._d_model,
            d_state=self._d_state,
            d_conv=self._d_conv,
            expand=self._expand,
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        """
        Args:
            x: [B*N, T, C]
        Returns:
            out: [B*N, T, C]
        """
        if self.mamba is None:
            self._init_mamba(x.device, x.dtype)

        residual = x
        x_norm = self.norm(x)              # [BNT, T, C]
        out = self.mamba(x_norm)           # [BNT, T, C]
        out = out * self.out_scale         # zero-init gate
        return residual + out


class MambaTemporalFusion(nn.Module):
    """Real Mamba2 temporal fusion for LFSTrack 3-frame pipeline.

    Drop-in replacement for LFSTrackTemporalFusion.

    Args:
        d_model:    feature dimension (768 for ViT-B)
        num_layers: number of Mamba2 blocks (default 2)
        d_state:    SSM state size (default 16)
        d_conv:     conv kernel size (default 4)
        expand:     Mamba2 expansion factor (default 2)
        drop_path:  stochastic depth rate (reserved, not implemented)
    """

    def __init__(self, d_model=768, num_layers=2, d_state=16, d_conv=4,
                 expand=2, drop_path=0.0):
        super().__init__()
        self.d_model = d_model

        self.norm_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            Mamba2TemporalBlock(d_model, d_state=d_state, d_conv=d_conv,
                                expand=expand)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # Zero-init norms -> identity at start
        nn.init.constant_(self.norm_in.weight, 0.0)
        nn.init.constant_(self.norm_in.bias, 0.0)
        nn.init.constant_(self.norm_out.weight, 0.0)
        nn.init.constant_(self.norm_out.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: [B, T, N, C] - T frames, N tokens, C channels
        Returns:
            out: [B, T, N, C]
        """
        B, T, N, C = x.shape

        # Flatten: per-token temporal processing
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, C).contiguous()

        x_flat = self.norm_in(x_flat)
        for block in self.blocks:
            x_flat = block(x_flat)
        x_flat = self.norm_out(x_flat)

        # Reshape back
        x_out = x_flat.view(B, N, T, C).permute(0, 2, 1, 3)

        return x + x_out  # residual
