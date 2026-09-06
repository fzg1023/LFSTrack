"""
Lightweight Bottleneck Adapter for parameter-efficient fine-tuning.

Inspired by ViPT (CVPR 2023, Visual Prompt Multi-Modal Tracking) and
AdaptFormer (NeurIPS 2022): freeze a converged backbone and inject small
parallel bottleneck branches into each transformer block. Only the adapters
(+ optionally the head) are trained, at a much higher LR than would be safe
for the full backbone.

Design notes (see docs/LFSTrack_ARCHITECTURE.md for the overall philosophy):
  - Parallel branch (AdaptFormer-style): adapter reads the SAME normalized
    input as the attention/MLP sublayer and its output is ADDED to the
    sublayer output before the residual connection, rather than being
    inserted serially after it. This avoids compounding depth and lets the
    adapter be a pure "correction" signal on top of the frozen computation.
  - Zero-init up-projection: the adapter's second linear layer is
    zero-initialized, so at the start of training the adapter contributes
    exactly 0 and the model is IDENTICAL to the frozen checkpoint (same
    zero-init-branch pattern already used by Multi-Layer LFN's alpha gate
    in lib/models/bat/ostrack_adapter.py — keeps new experiments from ever
    regressing below the baseline at epoch 0).
  - Single-stream caveat: unlike dual-stream BAT (AAAI 2024) which inserts
    bi-directional adapters BETWEEN two modality-specific ViT streams, this
    codebase fuses RGB+TIR at the input (4-channel patch embed) into one
    stream. There is no second stream to bridge, so this module implements
    the ViPT/AdaptFormer style of "frozen-backbone + adapter" tuning rather
    than a literal cross-modal bridge.
"""
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Bottleneck adapter: down-proj -> act -> dropout -> up-proj -> scale.

    Zero-init on the up-projection ensures the adapter starts as a no-op
    (output == 0), so inserting it into a frozen, already-converged backbone
    does not perturb behavior until gradients start flowing.
    """

    def __init__(self, dim, bottleneck_dim=64, act_layer=nn.GELU, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim)

        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.dropout(self.act(self.down(x))))
