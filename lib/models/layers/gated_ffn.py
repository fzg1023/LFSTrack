"""
Gated Feed-Forward Network (SwiGLU) for ViT blocks.

Replaces standard MLP (Linear→GELU→Linear) with gated SwiGLU
(Linear→SiLU ⊗ Linear→Linear), providing better gradient flow and
representation power.

Reference:
    PaLM: Scaling Language Modeling with Pathways (https://arxiv.org/abs/2204.02311)
    SwiGLU: GLU Variants Improve Transformer (https://arxiv.org/abs/2002.05202)

Integration: drop-in replacement for timm.models.layers.Mlp.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFFN(nn.Module):
    """SwiGLU-based Feed-Forward Network.

    Standard  FFN: x → fc1 → GELU → fc2
    GatedFFN:      x → w1 → SiLU ─┐
                       → w2 ─────→ × → w3

    The gate (w2 + SiLU) controls information flow per channel,
    giving better gradient propagation and feature selection.

    Args:
        in_features:  input dimension
        hidden_features: intermediate dimension (default 4× in_features)
        out_features: output dimension (default = in_features)
        drop: dropout rate
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)

        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        # SwiGLU: SiLU(x @ w1) ⊙ (x @ w2) @ w3
        # Only w1 is gated (SiLU); w2 is linear projection.
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))
