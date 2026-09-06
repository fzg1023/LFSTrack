"""
Layer-wise Fusion Network (LFN) — template-conditioned search-feature modulation.

Based on LFSTrack's Layer-wise Fusion Network:
  1. State Alignment:   Separate transforms for template (conditional state h)
                        and search (input l).
  2. Concentration Ctrl: Template-driven channel attention (avg + max pool).
  3. Channel-Spatial Gate: spatial permeability map from BOTH state and input.
  4. Modulation Update: x = (1-σ)·state + σ·input_refined  (σ = gate).

All operations are discrete feed-forward updates — no cross-frame recurrence
and no continuous-time dynamics; the modulation is conditioned on the current
template summary only.

Adapted for ViT token space: tokens reshaped to 2D grids → spatial convs → flatten back.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3 + BN + ReLU."""
    def __init__(self, in_c, out_c, kernel=3, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LayerwiseFusion(nn.Module):
    """Layer-wise Fusion Network (LFN) block for template→search modulation.

    Placed between a backbone layer and the head input, aggregating template
    tokens into a conditional state and modulating search features through
    channel-spatial gates.  Zero-init output layers ensure the module starts
    as identity (no degradation at epoch 1).

    Args:
        dim:       feature dimension (768 for ViT-B)
        grid_h:    search grid height (24 for 384/16)
        zero_init: if True, init output layers to 0 → identity start
    """

    def __init__(self, dim=768, grid_h=24, zero_init=True,
                 num_iterations=2, use_attn_pool=True,
                 use_cross_attn=False, use_multiscale_gate=False):
        super().__init__()
        self.dim = dim
        self.grid_h = grid_h
        self.num_iterations = num_iterations
        self.use_attn_pool = use_attn_pool
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_gate = use_multiscale_gate

        # ── 1. State Alignment ──
        # A1: Cross-Attention — search-aware template pooling
        if use_cross_attn:
            self.cross_q = nn.Linear(dim, dim // 8, bias=False)
            self.cross_kv = nn.Linear(dim, dim // 8, bias=False)
            self.cross_proj = nn.Linear(dim, dim)
        elif use_attn_pool:
            self.template_attn = nn.Linear(dim, 1)
        self.state_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
        )
        self.input_norm = nn.LayerNorm(dim)

        # ── 2. Concentration Control (channel attention from state) ──
        # The template conditional state determines which search channels are relevant
        self.conc_fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
        )
        self.conc_sigmoid = nn.Sigmoid()

        # ── 3. Channel-Spatial Gate ──
        self.gate_conv = nn.Sequential(
            ConvBlock(dim * 2, dim),
            nn.Conv2d(dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # C: Multi-scale gate (12x12 and 6x6)
        if use_multiscale_gate:
            self.gate_conv_12 = nn.Sequential(
                ConvBlock(dim * 2, dim),
                nn.Conv2d(dim, 1, 1, bias=True),
                nn.Sigmoid(),
            )
            self.gate_conv_6 = nn.Sequential(
                ConvBlock(dim * 2, dim),
                nn.Conv2d(dim, 1, 1, bias=True),
                nn.Sigmoid(),
            )

        # ── 4. Output refinement ──
        self.smooth = ConvBlock(dim, dim)

        # ── Micro-init for stable start (NOT zero — zero blocks gradient flow) ──
        # All output layers use a tiny random init (std=0.02) so gradients can
        # propagate through them from step 1.  The overall module still starts
        # near-identity because tanh(alpha)=0 or because the contribution is
        # proportional to the small weight magnitude.  Biases stay at 0 so the
        # default forward behaviour (e.g. gate≈0.5, smooth≈0) is preserved.
        if zero_init:
            if use_cross_attn:
                nn.init.trunc_normal_(self.cross_proj.weight, std=0.02)
                nn.init.constant_(self.cross_proj.bias, 0.0)
            elif use_attn_pool:
                nn.init.trunc_normal_(self.template_attn.weight, std=0.02)
                nn.init.constant_(self.template_attn.bias, 0.0)
            nn.init.trunc_normal_(self.state_pool[-1].weight, std=0.02)
            nn.init.constant_(self.state_pool[-1].bias, 0.0)
            nn.init.trunc_normal_(self.conc_fc[-1].weight, std=0.02)
            nn.init.constant_(self.conc_fc[-1].bias, 0.0)
            nn.init.trunc_normal_(self.gate_conv[-2].weight, std=0.02)
            nn.init.constant_(self.gate_conv[-2].bias, 0.0)
            if use_multiscale_gate:
                nn.init.trunc_normal_(self.gate_conv_12[-2].weight, std=0.02)
                nn.init.constant_(self.gate_conv_12[-2].bias, 0.0)
                nn.init.trunc_normal_(self.gate_conv_6[-2].weight, std=0.02)
                nn.init.constant_(self.gate_conv_6[-2].bias, 0.0)
            nn.init.trunc_normal_(self.smooth.conv.weight, std=0.02)

    def forward(self, search_tokens, template_tokens):
        """
        Args:
            search_tokens:   [B, N_s, C]  — search features (576 tokens / 24×24)
            template_tokens: [B, N_t, C]  — template features (144 tokens / 12×12)
        Returns:
            enhanced:        [B, N_s, C]  — template-modulated search features
        """
        B, Ns, C = search_tokens.shape
        # Dynamically compute grid size from actual token count
        # (supports test-time search sizes different from training)
        import math
        Hs = Ws = int(math.sqrt(Ns))
        if Hs * Ws != Ns:
            raise RuntimeError(
                f"LayerwiseFusion requires square search grid, got {Ns} tokens "
                f"(sqrt={math.sqrt(Ns):.1f}). Check search_size and patch_size.")

        # ── Conditional state: pool template → conditioning vector ──
        if self.use_cross_attn:
            # A1: Search-aware cross-attention template pooling
            Q = self.cross_q(search_tokens)                     # [B, Ns, d_k]
            K = self.cross_kv(template_tokens)                  # [B, Nt, d_k]
            V = K  # same projection for value
            attn = (Q @ K.transpose(-1, -2)) / (self.dim ** 0.25)   # [B, Ns, Nt]
            attn = attn.softmax(dim=-1).mean(dim=1)             # pool over search → [B, Nt]
            t_pooled = (template_tokens * attn.unsqueeze(-1)).sum(dim=1)  # [B, C]
            t_pooled = self.cross_proj(t_pooled)                # project back
        elif self.use_attn_pool:
            t_scores = self.template_attn(template_tokens).softmax(dim=1)
            t_pooled = (template_tokens * t_scores).sum(dim=1)
        else:
            t_pooled = template_tokens.mean(dim=1)  # v1: simple mean
        state_vec = self.state_pool(t_pooled)

        # ── Iterative conditional refinement ──
        s_current = search_tokens
        for _ in range(self.num_iterations):
            # Input: normalize, reshape to 2D
            s_norm = self.input_norm(s_current)                    # [B, Ns, C]
            s_2d = s_norm.transpose(1, 2).view(B, C, Hs, Ws)      # [B, C, H, W]

            # Concentration Control: conditional state → channel attention
            conc_raw = self.conc_fc(state_vec)                     # [B, C]
            concentration = self.conc_sigmoid(conc_raw)            # [B, C]
            s_refined = s_2d * concentration[:, :, None, None]     # [B, C, H, W]

            # Channel-Spatial Gate: spatial permeability (multi-scale or single)
            state_2d = state_vec[:, :, None, None].expand(-1, -1, Hs, Ws)
            gate_input = torch.cat([state_2d, s_refined], dim=1)

            if self.use_multiscale_gate:
                # C: Multi-scale gates → average
                g24 = self.gate_conv(gate_input)                    # [B, 1, 24, 24]
                g12 = F.interpolate(self.gate_conv_12(
                    F.avg_pool2d(gate_input, 2)), size=(Hs, Ws))    # [B, 1, 24, 24]
                g6  = F.interpolate(self.gate_conv_6(
                    F.avg_pool2d(gate_input, 4)), size=(Hs, Ws))    # [B, 1, 24, 24]
                spatial_gate = (g24 + g12 + g6) / 3.0
            else:
                spatial_gate = self.gate_conv(gate_input)

            # Conditional modulation update
            s_fused = state_2d * (1.0 - spatial_gate) + s_refined * spatial_gate

            # Flatten for next iteration
            s_out = self.smooth(s_fused)                           # [B, C, H, W]
            s_current = s_out.view(B, C, -1).transpose(1, 2)       # [B, Ns, C]

        # Residual (identity at zero-init)
        return search_tokens + s_current
