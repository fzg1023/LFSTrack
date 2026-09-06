"""
Single-Stream Vision Transformer for RGB-T Tracking.

Core idea: Extend RGB (3ch) to RGBT (6ch) at the input level.
All tokens go through a single unified ViT — self-attention naturally
learns cross-modal relationships without any explicit fusion module.

Token sequence: [z_1 ... z_64 | x_1 ... x_256]  (same 2-segment as DropTrack)
Each token internally encodes both modalities via 4-channel patch embedding.

Compatible with DropTrack pretrained weights (2-segment attention pattern).
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, Mlp, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.layers.attn import Attention
from lib.models.layers.attn_adapt_blocks import candidate_elimination
from lib.models.layers.gated_blocks import GatedSSBlock
from lib.models.bat.base_backbone import BaseBackbone
from lib.models.bat.utils import combine_tokens, recover_tokens

_logger = logging.getLogger(__name__)


class SSBlock(nn.Module):
    """Single-Stream ViT Block.

    Same structure as standard ViT Block but uses Attention from lib.models.layers.attn
    which supports attention mask and return_attention modes.

    Optionally carries a pair of parallel bottleneck Adapters (ViPT/AdaptFormer-style,
    see lib.models.layers.adapter_blocks.Adapter) for parameter-efficient fine-tuning
    of an otherwise-frozen block. Off by default (adapter_dim=None) — zero overhead,
    unchanged behavior for all existing experiments.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if adapter_dim is not None:
            from lib.models.layers.adapter_blocks import Adapter
            self.adapter_attn = Adapter(dim, bottleneck_dim=adapter_dim)
            self.adapter_mlp = Adapter(dim, bottleneck_dim=adapter_dim)
        else:
            self.adapter_attn = None
            self.adapter_mlp = None

    def forward(self, x, mask=None, return_attention=False):
        x_norm1 = self.norm1(x)
        if return_attention:
            feat, attn = self.attn(x_norm1, mask=mask, return_attention=True)
        else:
            feat = self.attn(x_norm1, mask=mask)
            attn = None
        if self.adapter_attn is not None:
            feat = feat + self.adapter_attn(x_norm1)
        x = x + self.drop_path(feat)

        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        if self.adapter_mlp is not None:
            mlp_out = mlp_out + self.adapter_mlp(x_norm2)
        x = x + self.drop_path(mlp_out)
        if return_attention:
            return x, attn
        return x


class VisionTransformerSingleStream(BaseBackbone):
    """Single-Stream Vision Transformer for RGB-T Tracking.

    Takes 4-channel RGBT input and processes as a single [template | search] token sequence.
    No explicit cross-modal fusion — self-attention handles it naturally.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=4, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None,
                 search_size=None, template_size=None, new_patch_size=None,
                 lfn_mid_layer=None, lfn_early_layer=None,
                 gate_layers=None, adapter_layers=None, adapter_dim=64,
                 use_modality_prompt=False, modality_prompt_num=2):
        """
        Args:
            search_size: [H, W] of search region
            template_size: [H, W] of template region
            new_patch_size: backbone stride (typically 16)
            ce_loc: list of layer indices where CE is applied, e.g. [3, 6, 9]
            ce_keep_ratio: keep ratios for CE layers, e.g. [1.0, 1.0, 1.0]
            lfn_mid_layer: if set, capture normalized [z|x] tokens right after this
                block index (0-indexed) into aux_dict['mid_tokens'], for Multi-Layer LFN.
                None (default) disables capture — zero overhead, unchanged behavior.
            lfn_early_layer: if set, capture normalized [z|x] tokens right after this
                block index (0-indexed) into aux_dict['early_tokens'], for 3-Layer LFN.
                Must be < lfn_mid_layer. None disables capture.
            gate_layers: list of layer indices using GatedSSBlock, e.g. [9, 10, 11].
                None or empty list → all layers use standard SSBlock (default).
            adapter_layers: list of layer indices that get a parallel bottleneck
                Adapter (ViPT/AdaptFormer-style), e.g. list(range(12)) for all layers.
                None or empty list → no adapters (default, zero overhead).
            adapter_dim: bottleneck dimension for the Adapter modules (default 64).
            use_modality_prompt: prepend learnable modality prompt tokens (ViPT-style)
                to the single-stream token sequence. Keeps the one-stream backbone
                while giving an explicit modality prior. Off by default.
            modality_prompt_num: number of prompt tokens when enabled.
        """
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # ── 4-channel Patch Embedding ──
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # Legacy pos_embed (not used for tracking, kept for weight loading compatibility)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ── Tracking-specific position embeddings ──
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W

        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        # ── Modality Prompt (ViPT-style) ──
        # Learnable prompt tokens prepended to the [z|x] token sequence. Small
        # random init → first forward ≈ original model (zero-risk start).
        self.use_modality_prompt = bool(use_modality_prompt)
        self.num_prompt = modality_prompt_num if use_modality_prompt else 0
        if self.use_modality_prompt:
            if self.num_prompt < 1:
                raise ValueError("modality_prompt_num must be >= 1 when USE_MODALITY_PROMPT is true")
            self.modality_prompt = nn.Parameter(torch.zeros(1, self.num_prompt, embed_dim))
            self.modality_prompt_pos = nn.Parameter(torch.zeros(1, self.num_prompt, embed_dim))
            nn.init.trunc_normal_(self.modality_prompt, std=0.02)
            nn.init.zeros_(self.modality_prompt_pos)
        else:
            self.modality_prompt = None
            self.modality_prompt_pos = None

        # ── CE configuration ──
        self.ce_loc = ce_loc if ce_loc is not None else []
        self.ce_keep_ratio = ce_keep_ratio if ce_keep_ratio is not None else []
        # Modality prompts prepend K tokens → candidate elimination token
        # bookkeeping (attention-based pruning) is not implemented for that
        # case. Fail fast instead of silently producing misaligned tokens.
        if self.use_modality_prompt and any(r < 1.0 for r in self.ce_keep_ratio):
            raise ValueError(
                "USE_MODALITY_PROMPT is incompatible with active CE pruning "
                f"(CE_KEEP_RATIO={self.ce_keep_ratio}). Use CE_KEEP_RATIO=[1,...] "
                "or disable the modality prompt.")

        # ── Multi-Layer LFN: optional intermediate-layer token capture ──
        self.lfn_mid_layer = lfn_mid_layer
        self.lfn_early_layer = lfn_early_layer
        if lfn_mid_layer is not None:
            if not (0 <= lfn_mid_layer < depth - 1):
                raise ValueError(
                    f"lfn_mid_layer={lfn_mid_layer} is out of range for depth={depth}; "
                    f"must satisfy 0 <= lfn_mid_layer < {depth - 1} (strictly before the "
                    f"final layer, otherwise the mid-layer branch is redundant with the "
                    f"final-layer LFN).")
            # Fail fast if any CE-pruning layer occurs before the capture point — token
            # order/count would no longer match [template | search] and the mid-layer
            # split in BATrack.forward would silently produce garbage features.
            pruning_before_capture = any(
                loc < lfn_mid_layer and ratio < 1.0
                for loc, ratio in zip(self.ce_loc, self.ce_keep_ratio))
            if pruning_before_capture:
                raise ValueError(
                    f"lfn_mid_layer={lfn_mid_layer} is set but CE_KEEP_RATIO has an "
                    f"active pruning layer (ratio < 1.0) at or before it (CE_LOC="
                    f"{self.ce_loc}, CE_KEEP_RATIO={self.ce_keep_ratio}). This would "
                    f"misalign the captured mid-layer tokens. Either move CE pruning "
                    f"after lfn_mid_layer, disable CE (keep_ratio=1.0), or choose a "
                    f"deeper lfn_mid_layer.")
        # ── 3-Layer LFN: early-layer token capture ──
        if lfn_early_layer is not None:
            if lfn_mid_layer is None:
                raise ValueError(
                    f"lfn_early_layer={lfn_early_layer} requires lfn_mid_layer to also be set; "
                    f"3-layer LFN needs both mid and early capture points.")
            if not (0 <= lfn_early_layer < lfn_mid_layer):
                raise ValueError(
                    f"lfn_early_layer={lfn_early_layer} must satisfy 0 <= early < "
                    f"mid_layer={lfn_mid_layer} (early must come before mid).")
            pruning_before_early = any(
                loc < lfn_early_layer and ratio < 1.0
                for loc, ratio in zip(self.ce_loc, self.ce_keep_ratio))
            if pruning_before_early:
                raise ValueError(
                    f"lfn_early_layer={lfn_early_layer} is set but CE pruning occurs before it.")

        # ── Transformer blocks (with optional gated layers + optional adapters) ──
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        gate_set = set(gate_layers) if gate_layers else set()
        adapter_set = set(adapter_layers) if adapter_layers else set()
        blocks = []
        for i in range(depth):
            block_adapter_dim = adapter_dim if i in adapter_set else None
            if i in gate_set:
                blocks.append(GatedSSBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer, act_layer=act_layer, adapter_dim=block_adapter_dim))
            else:
                blocks.append(SSBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer, act_layer=act_layer, adapter_dim=block_adapter_dim))
        self.blocks = nn.Sequential(*blocks)

        self.norm = norm_layer(embed_dim)

        # ── Tracking mode flags ──
        self.cat_mode = 'direct'
        self.add_cls_token = False
        self.add_sep_seg = False

        trunc_normal_(self.pos_embed_z, std=.02)
        trunc_normal_(self.pos_embed_x, std=.02)

    @staticmethod
    def _interpolate_pos_embed(pos_embed, target_num_tokens):
        """Interpolate a 2D position embedding to match a different token count.

        pos_embed: [1, N_src, C]  — stored position embedding (square grid)
        target_num_tokens: int     — desired number of tokens

        Returns: [1, target_num_tokens, C]
        """
        src_N = pos_embed.shape[1]
        if src_N == target_num_tokens:
            return pos_embed

        import math
        src_H = int(math.sqrt(src_N))
        if src_H * src_H != src_N:
            # Non-square, fall back to linear interpolation
            return F.interpolate(
                pos_embed.transpose(1, 2),  # [1, C, N_src]
                size=target_num_tokens,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)  # [1, N_tgt, C]

        tgt_H = int(math.sqrt(target_num_tokens))
        if tgt_H * tgt_H != target_num_tokens:
            return F.interpolate(
                pos_embed.transpose(1, 2),
                size=target_num_tokens,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        # 2D bicubic interpolation
        pos_2d = pos_embed.reshape(1, src_H, src_H, -1).permute(0, 3, 1, 2)  # [1, C, H, W]
        pos_2d = F.interpolate(pos_2d, size=(tgt_H, tgt_H), mode='bicubic', align_corners=False)
        return pos_2d.flatten(2).transpose(1, 2)  # [1, N_tgt, C]

    def _add_pos_embed(self, z_tokens, x_tokens):
        """Add position embeddings, interpolating if token counts don't match."""
        pos_z = self._interpolate_pos_embed(self.pos_embed_z, z_tokens.shape[1])
        pos_x = self._interpolate_pos_embed(self.pos_embed_x, x_tokens.shape[1])
        return z_tokens + pos_z, x_tokens + pos_x

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,
                         dynamic_template=None, Test=None, template_masks=None):
        """
        Args:
            z: template images — during training: list of 2 tensors [B,6,128,128] each
                                 during testing:  single tensor [B,6,128,128]
            x: search images [B, 6, 256, 256]
        Returns:
            x: output tokens [B_out, N_z+N_x, C]
            aux_dict: auxiliary outputs
        """
        # ── Handle template pairs (training mode) ──
        # During training, z is a list of 2 templates; we need to double the batch
        # so each batch item sees both templates (same pattern as VisionTransformerCE).
        # NOTE: Single-stream has NO TemplateRouter, so we skip the forward/backward
        # reordering that the dual-stream does at token level.
        if isinstance(z, (list, tuple)):
            z1, z2 = z[0], z[1]  # each [B, 6, 128, 128]
            # 6ch RGBT: RGB(0:3) + TIR(3:6)，全通道输入
            z1 = z1[:, :self.in_chans, :, :]
            z2 = z2[:, :self.in_chans, :, :]
            x_in = x[:, :self.in_chans, :, :]

            B_orig = x_in.shape[0]

            # Simple pair creation: interleave templates, duplicate search
            z_list, x_list = [], []
            for i in range(B_orig):
                z_list.append(torch.cat([z1[i].unsqueeze(0), z2[i].unsqueeze(0)], dim=0))
                x_list.append(torch.cat([x_in[i].unsqueeze(0), x_in[i].unsqueeze(0)], dim=0))
            z_in = torch.cat(z_list, dim=0)
            x_in = torch.cat(x_list, dim=0)
        else:
            # Test mode: single template
            z_in = z[:, :self.in_chans, :, :]
            x_in = x[:, :self.in_chans, :, :]
            B_orig = x_in.shape[0]

        B = x_in.shape[0]

        # ── Patch embedding ──
        # PatchEmbed returns tuple (x, x_center) when H==16 (search), single tensor otherwise
        z_pe = self.patch_embed(z_in)
        if isinstance(z_pe, tuple):
            z_tokens = z_pe[0]
        else:
            z_tokens = z_pe

        x_pe = self.patch_embed(x_in)
        if isinstance(x_pe, tuple):
            x_tokens = x_pe[0]        # [B, 256, C] — main search tokens
            # x_center = x_pe[1]     # [B, 64, C] — center tokens (unused in single-stream)
        else:
            x_tokens = x_pe

        # ── Add position embeddings (with interpolation for size mismatch) ──
        z_tokens, x_tokens = self._add_pos_embed(z_tokens, x_tokens)

        # Capture actual token counts (after possible interpolation) for downstream use
        actual_num_search = x_tokens.shape[1]
        actual_num_template = z_tokens.shape[1]

        # ── Build attention mask ──
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)
            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)
            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)
        else:
            mask_x = None

        # ── Concatenate: [z | x] ──
        x = combine_tokens(z_tokens, x_tokens, mode=self.cat_mode)
        x = self.pos_drop(x)

        # ── Modality Prompt: prepend learnable prompt tokens (ViPT-style) ──
        if self.num_prompt:
            prompt = self.modality_prompt + self.modality_prompt_pos
            x = torch.cat([prompt.expand(x.shape[0], -1, -1), x], dim=1)
            if mask_x is not None:
                # prompt tokens are never masked
                mask_x = torch.cat([
                    torch.zeros(mask_x.shape[0], self.num_prompt, dtype=torch.bool, device=mask_x.device),
                    mask_x], dim=1)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device).repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device).repeat(B, 1)

        # ── Forward through transformer blocks ──
        removed_indexes_s = []
        ce_index = 0
        last_attn = None
        mid_tokens = None
        early_tokens = None

        for i, blk in enumerate(self.blocks):
            # Determine CE keep ratio for this layer
            keep_ratio_search = None
            if i in self.ce_loc and ce_index < len(self.ce_keep_ratio):
                keep_ratio_search = self.ce_keep_ratio[ce_index]
                ce_index += 1

            need_attn = (keep_ratio_search is not None and keep_ratio_search < 1.0)
            # Always capture attention for the last layer when requested
            is_last = (i == len(self.blocks) - 1)

            if need_attn or (return_last_attn and is_last):
                x, attn = blk(x, mask=mask_x, return_attention=True)
            else:
                x = blk(x, mask=mask_x)
                attn = None

            if need_attn and attn is not None:
                x, global_index_s, removed_index_s = candidate_elimination(
                    attn, x, lens_z, keep_ratio_search, global_index_s, ce_template_mask)
                if i in self.ce_loc:
                    removed_indexes_s.append(removed_index_s)

            if return_last_attn and is_last:
                last_attn = attn

            # ── Multi-Layer LFN: capture intermediate [z|x] tokens (normed) ──
            if self.lfn_early_layer is not None and i == self.lfn_early_layer:
                early_tokens = self.norm(x)
                if self.num_prompt:
                    early_tokens = early_tokens[:, self.num_prompt:]
            if self.lfn_mid_layer is not None and i == self.lfn_mid_layer:
                mid_tokens = self.norm(x)
                if self.num_prompt:
                    mid_tokens = mid_tokens[:, self.num_prompt:]

        # ── Final norm ──
        x = self.norm(x)
        if self.num_prompt:
            # drop prompt tokens — downstream expects [z|x] only
            x = x[:, self.num_prompt:]

        # ── Split back: [z | x] ──
        lens_z_new = global_index_t.shape[1]
        z_out = x[:, :lens_z_new]
        x_out = x[:, lens_z_new:]

        # ── Recover pruned tokens (if CE was applied) ──
        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
            pruned_lens_x = lens_x - global_index_s.shape[1]
            pad_x = torch.zeros([B, pruned_lens_x, x_out.shape[2]], device=x_out.device)
            x_out = torch.cat([x_out, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            C = x_out.shape[-1]
            x_out = torch.zeros_like(x_out).scatter_(
                dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_out)

        x_out = recover_tokens(x_out, lens_z_new, lens_x, mode=self.cat_mode)
        x_out = torch.cat([z_out, x_out], dim=1)

        aux_dict = {
            "attn": last_attn,
            "removed_indexes_s": removed_indexes_s,
            # Multi-Layer LFN: intermediate-layer [z|x] tokens, or None if unset
            "mid_tokens": mid_tokens,
            # 3-Layer LFN: early-layer [z|x] tokens, or None if lfn_early_layer unset
            "early_tokens": early_tokens,
            # Dynamic search token count (supports test-time search sizes != training)
            "num_search": actual_num_search,
            "num_template": actual_num_template,
            # Dummy keys for BATActor compatibility (single-stream has no uncertainty/temporal modules)
            "future_loss": torch.tensor(0.0, device=x_out.device),
            "weights": [],
            # B_orig tracks the logical batch size before template-pair doubling;
            # the actor expects relative_score/score with shape [B_orig, 2]
            "relative_score": torch.zeros(B_orig, 2, device=x_out.device),
            "score": torch.zeros(B_orig, 2, device=x_out.device),
            # u_m / u / ui: uncertainty tensors [B_orig, num_search_tokens, 1]
            # tracker.track() does u_m[0] so each must be a subscriptable tensor
            "u_m": [
                torch.zeros(B_orig, self.num_patches_search, 1, device=x_out.device),
                torch.zeros(B_orig, self.num_patches_search, 1, device=x_out.device),
                torch.zeros(B_orig, self.num_patches_search, 1, device=x_out.device),
            ],
        }

        return x_out, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False,
                dynamic_template=None, Test=None, template_masks=None):
        x, aux_dict = self.forward_features(
            z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            dynamic_template=dynamic_template, Test=Test, template_masks=template_masks)
        # Return 3 values for BATrack.forward compatibility; track_token is unused in training
        return x, aux_dict, dynamic_template


def _create_vision_transformer_single_stream(pretrained=None, **kwargs):
    model = VisionTransformerSingleStream(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_single_stream(pretrained=None, **kwargs):
    """ViT-Base Single-Stream for RGBT tracking (6-channel input)."""
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer_single_stream(pretrained=pretrained, **model_kwargs)
    return model
