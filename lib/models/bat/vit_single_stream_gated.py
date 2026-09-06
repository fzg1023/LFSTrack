"""
Single-Stream ViT + Uncertainty-Gated Blocks for RGB-T Tracking.

Same 4ch input as vit_single_stream.py, but layers 9-11 use GatedSSBlock
with per-channel uncertainty gating. This lets the model learn which feature
channels to trust based on modality reliability — without needing dual streams.
"""
import math
import logging
from functools import partial

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
    """Standard ViT Block (layers 0-8)."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), mask=mask, return_attention=True)
        else:
            feat = self.attn(self.norm1(x), mask=mask)
            attn = None
        x = x + self.drop_path(feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class VisionTransformerSingleStreamGated(BaseBackbone):
    """Single-Stream ViT with Uncertainty-Gated blocks (layers 9-11)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=4, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None,
                 search_size=None, template_size=None, new_patch_size=None,
                 gate_layers=None):
        """
        Args:
            gate_layers: list of layer indices using GatedSSBlock, e.g. [9, 10, 11]
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

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W

        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.ce_loc = ce_loc if ce_loc is not None else []
        self.ce_keep_ratio = ce_keep_ratio if ce_keep_ratio is not None else []

        # Which layers use uncertainty gating
        self.gate_layers = gate_layers if gate_layers is not None else [9, 10, 11]

        # ── Build blocks: layers 0-8 standard, 9-11 gated ──
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for i in range(depth):
            if i in self.gate_layers:
                blocks.append(GatedSSBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer, act_layer=act_layer))
            else:
                blocks.append(SSBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer, act_layer=act_layer))
        self.blocks = nn.Sequential(*blocks)

        self.norm = norm_layer(embed_dim)

        self.cat_mode = 'direct'
        self.add_cls_token = False
        self.add_sep_seg = False

        trunc_normal_(self.pos_embed_z, std=.02)
        trunc_normal_(self.pos_embed_x, std=.02)

    @staticmethod
    def _interpolate_pos_embed(pos_embed, target_num_tokens):
        """Interpolate a 2D position embedding to match a different token count."""
        import math
        src_N = pos_embed.shape[1]
        if src_N == target_num_tokens:
            return pos_embed
        src_H = int(math.sqrt(src_N))
        if src_H * src_H != src_N:
            return F.interpolate(
                pos_embed.transpose(1, 2), size=target_num_tokens,
                mode='linear', align_corners=False).transpose(1, 2)
        tgt_H = int(math.sqrt(target_num_tokens))
        if tgt_H * tgt_H != target_num_tokens:
            return F.interpolate(
                pos_embed.transpose(1, 2), size=target_num_tokens,
                mode='linear', align_corners=False).transpose(1, 2)
        pos_2d = pos_embed.reshape(1, src_H, src_H, -1).permute(0, 3, 1, 2)
        pos_2d = F.interpolate(pos_2d, size=(tgt_H, tgt_H), mode='bicubic', align_corners=False)
        return pos_2d.flatten(2).transpose(1, 2)

    def _add_pos_embed(self, z_tokens, x_tokens):
        """Add position embeddings, interpolating if token counts don't match."""
        pos_z = self._interpolate_pos_embed(self.pos_embed_z, z_tokens.shape[1])
        pos_x = self._interpolate_pos_embed(self.pos_embed_x, x_tokens.shape[1])
        return z_tokens + pos_z, x_tokens + pos_x

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,
                         dynamic_template=None, Test=None, template_masks=None):
        # ── Template pair handling ──
        if isinstance(z, (list, tuple)):
            z1, z2 = z[0], z[1]
            z1 = z1[:, :4, :, :]
            z2 = z2[:, :4, :, :]
            x_in = x[:, :4, :, :]
            B_orig = x_in.shape[0]
            z_list, x_list = [], []
            for i in range(B_orig):
                z_list.append(torch.cat([z1[i].unsqueeze(0), z2[i].unsqueeze(0)], dim=0))
                x_list.append(torch.cat([x_in[i].unsqueeze(0), x_in[i].unsqueeze(0)], dim=0))
            z_in = torch.cat(z_list, dim=0)
            x_in = torch.cat(x_list, dim=0)
        else:
            z_in = z[:, :4, :, :]
            x_in = x[:, :4, :, :]
            B_orig = x_in.shape[0]

        B = x_in.shape[0]

        z_pe = self.patch_embed(z_in)
        z_tokens = z_pe[0] if isinstance(z_pe, tuple) else z_pe
        x_pe = self.patch_embed(x_in)
        x_tokens = x_pe[0] if isinstance(x_pe, tuple) else x_pe

        z_tokens, x_tokens = self._add_pos_embed(z_tokens, x_tokens)

        # Capture actual token counts (after possible interpolation) for downstream use
        actual_num_search = x_tokens.shape[1]
        actual_num_template = z_tokens.shape[1]

        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)
            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)
            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)
        else:
            mask_x = None

        x = combine_tokens(z_tokens, x_tokens, mode=self.cat_mode)
        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device).repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device).repeat(B, 1)

        removed_indexes_s = []
        ce_index = 0
        last_attn = None

        for i, blk in enumerate(self.blocks):
            keep_ratio_search = None
            if i in self.ce_loc and ce_index < len(self.ce_keep_ratio):
                keep_ratio_search = self.ce_keep_ratio[ce_index]
                ce_index += 1

            need_attn = (keep_ratio_search is not None and keep_ratio_search < 1.0)
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

        x = self.norm(x)

        lens_z_new = global_index_t.shape[1]
        z_out = x[:, :lens_z_new]
        x_out = x[:, lens_z_new:]

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
            # Dynamic search token count (supports test-time search sizes != training)
            "num_search": actual_num_search,
            "num_template": actual_num_template,
            # Multi-Layer LFN not supported in gated variant
            "mid_tokens": None,
            "future_loss": torch.tensor(0.0, device=x_out.device),
            "weights": [],
            "relative_score": torch.zeros(B_orig, 2, device=x_out.device),
            "score": torch.zeros(B_orig, 2, device=x_out.device),
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
        return x, aux_dict, dynamic_template


def vit_base_patch16_224_single_stream_gated(pretrained=None, **kwargs):
    """ViT-Base Single-Stream + Uncertainty Gating for RGBT tracking."""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformerSingleStreamGated(**model_kwargs)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print(f'[Gated] Load pretrained from: {pretrained}')
        print(f'[Gated] missing_keys: {missing_keys}')
        print(f'[Gated] unexpected_keys: {unexpected_keys}')

    return model
