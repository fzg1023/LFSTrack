"""
Gated Single-Stream ViT Block with Uncertainty-aware Channel Gating.

Inspired by the dual-stream SelfAttention_For_Fusion_uncertainty,
but adapted for single-stream: instead of gating between RGB/TIR streams,
we compute per-channel confidence gates that suppress noisy features
and amplify reliable ones.

The gate is computed from pooled token features via a lightweight MLP,
making it modality-aware: channels dominated by RGB vs TIR respond
differently, and the gate learns to trust the reliable modality.
"""
import torch
import torch.nn as nn

from timm.models.layers import DropPath, Mlp

from lib.models.layers.attn import Attention


class ChannelGate(nn.Module):
    """Lightweight per-channel gating mechanism.

    Pool over tokens → small MLP → sigmoid → per-channel weights in [0,1].
    Similar to SE-Net but computed from token statistics rather than spatial.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over token dimension
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, N, C]
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, C]
        return self.mlp(pooled).unsqueeze(1)  # [B, 1, C]


class GatedSSBlock(nn.Module):
    """Single-Stream ViT Block with Uncertainty Channel Gating.

    Standard ViT block + channel gate applied to MLP output.
    The gate learns to suppress channels carrying unreliable modality features.

    Used in the last 3 layers (9-11) where modality fusion matters most.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 gate_reduction=4, adapter_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Channel gate: applied to MLP output
        self.gate = ChannelGate(dim, reduction=gate_reduction)

        # Optional parallel bottleneck Adapter (ViPT/AdaptFormer-style), see
        # lib.models.layers.adapter_blocks.Adapter. Off by default.
        if adapter_dim is not None:
            from lib.models.layers.adapter_blocks import Adapter
            self.adapter_attn = Adapter(dim, bottleneck_dim=adapter_dim)
            self.adapter_mlp = Adapter(dim, bottleneck_dim=adapter_dim)
        else:
            self.adapter_attn = None
            self.adapter_mlp = None

    def forward(self, x, mask=None, return_attention=False):
        # ── Self-Attention with residual ──
        x_norm1 = self.norm1(x)
        if return_attention:
            feat, attn = self.attn(x_norm1, mask=mask, return_attention=True)
        else:
            feat = self.attn(x_norm1, mask=mask)
            attn = None
        if self.adapter_attn is not None:
            feat = feat + self.adapter_attn(x_norm1)
        x = x + self.drop_path(feat)

        # ── Gated MLP ──
        residual = x
        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed)
        if self.adapter_mlp is not None:
            mlp_out = mlp_out + self.adapter_mlp(x_normed)

        # Compute channel gate from normalized features
        gate = self.gate(x_normed)  # [B, 1, C]

        x = residual + self.drop_path(gate * mlp_out)

        if return_attention:
            return x, attn
        return x
