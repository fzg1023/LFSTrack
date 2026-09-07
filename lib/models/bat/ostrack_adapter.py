"""
LFSTrack model adapter — single-stream variants only.
"""
import math
import os
from typing import List
from timm.models.layers import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.bat.vit_single_stream import vit_base_patch16_224_single_stream
from lib.models.bat.vit_single_stream_gated import vit_base_patch16_224_single_stream_gated
from lib.utils.box_ops import box_xyxy_to_cxcywh
import torch.nn.functional as F


class BATrack(nn.Module):
    """ This is the base class for BATrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 use_lfn=True, lfn_num_iterations=2, lfn_use_attn_pool=True,
                 lfn_use_cross_attn=False, lfn_use_multiscale_gate=False,
                 use_multilayer_lfn=False, lfn_mid_layer=7,
                 use_3layer_lfn=False, lfn_early_layer=3):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_lfn: enable the Layer-wise Fusion Network (LFN) — asymmetric
                template→search modulation.
            use_multilayer_lfn: enable a second LFN branch reading backbone layer
                `lfn_mid_layer`, fused into the final LFN output via a zero-init gate.
                Requires use_lfn=True and the backbone to expose
                aux_dict['mid_tokens'] (see vit_single_stream.py, lfn_mid_layer arg).
            use_3layer_lfn: enable a third LFN branch reading backbone layer
                `lfn_early_layer`, fused into the final LFN output via a zero-init gate.
                Requires use_multilayer_lfn=True.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type in ("CORNER", "CENTER", "CONDITIONAL_CENTER"):
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        # Layer-wise Fusion Network (LFN): asymmetric template→search modulation (zero-init)
        if use_lfn:
            from lib.models.bat.layerwise_fusion import LayerwiseFusion
            grid_h = int(box_head.feat_sz)  # 24 for 384/16
            self.lfn = LayerwiseFusion(dim=transformer.embed_dim, grid_h=grid_h,
                                       num_iterations=lfn_num_iterations,
                                       use_attn_pool=lfn_use_attn_pool,
                                       use_cross_attn=lfn_use_cross_attn,
                                       use_multiscale_gate=lfn_use_multiscale_gate)
        else:
            self.lfn = None

        # Multi-Layer LFN: second LFN branch on an intermediate layer, combined into
        # the final-layer LFN output through a zero-init projection + zero-init gate
        # (tanh(alpha), alpha=0 → contributes exactly 0 at init). Off by default —
        # does not affect any existing single_stream / single_stream_gated experiment.
        if use_multilayer_lfn and use_lfn:
            from lib.models.bat.layerwise_fusion import LayerwiseFusion
            grid_h = int(box_head.feat_sz)
            self.lfn_mid_layer = lfn_mid_layer
            self.lfn_mid = LayerwiseFusion(dim=transformer.embed_dim, grid_h=grid_h,
                                           num_iterations=1, use_attn_pool=False,
                                           use_cross_attn=False, use_multiscale_gate=False)
            self.lfn_mid_proj = nn.Linear(transformer.embed_dim, transformer.embed_dim)
            # NOTE: only `alpha` is zero-init'd here. If both alpha AND proj were
            # zero-init'd simultaneously, the branch becomes permanently dead:
            # d(loss)/d(alpha) ∝ proj(mid_enhanced) == 0 (since proj is all-zero), and
            # d(loss)/d(proj.weight) ∝ tanh(alpha) == 0 (since alpha is zero) — a
            # bilinear deadlock where neither side ever receives a gradient, so the
            # module (and everything upstream in lfn_mid) never trains.
            # Keeping proj at its default (small, non-zero) init lets alpha receive a
            # real gradient from step 1, while the overall contribution still starts
            # at exactly 0 because tanh(alpha)=0 with alpha initialized to 0.
            nn.init.trunc_normal_(self.lfn_mid_proj.weight, std=0.02)
            nn.init.zeros_(self.lfn_mid_proj.bias)
            self.lfn_mid_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.lfn_mid_layer = None
            self.lfn_mid = None
            self.lfn_mid_proj = None
            self.lfn_mid_alpha = None

        # ── 3-Layer LFN: third LFN branch on an early layer ──
        if use_3layer_lfn and use_multilayer_lfn and use_lfn:
            from lib.models.bat.layerwise_fusion import LayerwiseFusion
            grid_h = int(box_head.feat_sz)
            self.lfn_early_layer = lfn_early_layer
            self.lfn_early = LayerwiseFusion(dim=transformer.embed_dim, grid_h=grid_h,
                                             num_iterations=1, use_attn_pool=False,
                                             use_cross_attn=False, use_multiscale_gate=False)
            self.lfn_early_proj = nn.Linear(transformer.embed_dim, transformer.embed_dim)
            nn.init.trunc_normal_(self.lfn_early_proj.weight, std=0.02)
            nn.init.zeros_(self.lfn_early_proj.bias)
            self.lfn_early_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.lfn_early_layer = None
            self.lfn_early = None
            self.lfn_early_proj = None
            self.lfn_early_alpha = None

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.last_template = None
        self.dynamic_template = None
        # self.dynamic_template_list = []
        self.seq_name = None

    def reset_dynamic_template(self):
        """P4: clear per-sequence template state.

        Called at tracker.initialize() so that no template memory leaks
        between consecutive test sequences when a tracker instance is reused.
        """
        self.dynamic_template = None
        self.last_template = None


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                seq_name=None,
                Test=None,
                frame_id=None,
                template_masks = None
                ):
        ###---for template ---###
        # if self.last_template == None:
        #     self.last_template=template
        #     template = torch.cat([template,self.last_template],dim=1)
        # else:
        #     template = torch.cat([template,self.last_template],dim=1)
        # print(frame_id)
        #start from 1

        # ── P4 fix: dynamic template state management ──
        # The dynamic-template memory may ONLY persist during online tracking
        # (Test is not None). During training it must be stateless: carrying
        # the previous sample's track_token into the next sample pollutes the
        # batch and keeps stale GPU tensors referenced across iterations.
        # Note: the current single-stream backbone treats `dynamic_template`
        # as a pass-through (it is echoed back but not consumed inside
        # forward_features), so the state has no numerical effect today —
        # this fix keeps it correct/hygienic if a dynamic-template-consuming
        # backbone is ever used, and fixes the dropped `template_masks` kwarg.
        if Test is not None:
            if self.dynamic_template is None or frame_id == 1:
                x, aux_dict, track_token = self.backbone(z=template, x=search,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn,
                                            dynamic_template=None, Test=Test,
                                            template_masks=template_masks)
            else:
                x, aux_dict, track_token = self.backbone(z=template, x=search,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn,
                                            dynamic_template=self.dynamic_template,
                                            Test=Test, template_masks=template_masks)
            self.dynamic_template = track_token
        else:
            # Training: stateless forward, never reuse another sample's template
            x, aux_dict, track_token = self.backbone(z=template, x=search,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn,
                                        dynamic_template=None, Test=Test,
                                        template_masks=template_masks)
        # if Test is None:
        #     x_f_list = aux_dict["x_f"]
        #     out1 = self.forward_head(x_f_list[0],None)
        #     out2 = self.forward_head(x_f_list[1],None)
        #     out3 = self.forward_head(x_f_list[2],None)
        #     out_list = [out1,out2,out3]
        # Forward head with optional Layer-wise Fusion Network (LFN)
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        # ── Layer-wise Fusion Network: template modulates search (asymmetric) ──
        if self.lfn is not None:
            # Use dynamic search token count from backbone (supports test-time size changes)
            num_search = aux_dict.get('num_search', self.feat_len_s)
            template_tokens = feat_last[:, :-num_search, :]  # [B, Nt, C]
            search_tokens = feat_last[:, -num_search:, :]    # [B, Ns, C]
            search_enhanced = self.lfn(search_tokens, template_tokens)

            # ── Multi-Layer LFN: fuse an intermediate-layer LFN branch in (zero-init) ──
            if self.lfn_mid is not None and aux_dict.get("mid_tokens") is not None:
                mid_tokens = aux_dict["mid_tokens"]
                mid_template = mid_tokens[:, :-num_search, :]
                mid_search = mid_tokens[:, -num_search:, :]
                mid_enhanced = self.lfn_mid(mid_search, mid_template)
                search_enhanced = search_enhanced + torch.tanh(self.lfn_mid_alpha) * \
                    self.lfn_mid_proj(mid_enhanced)

            # ── 3-Layer LFN: fuse an early-layer LFN branch in (zero-init) ──
            if self.lfn_early is not None and aux_dict.get("early_tokens") is not None:
                early_tokens = aux_dict["early_tokens"]
                early_template = early_tokens[:, :-num_search, :]
                early_search = early_tokens[:, -num_search:, :]
                early_enhanced = self.lfn_early(early_search, early_template)
                search_enhanced = search_enhanced + torch.tanh(self.lfn_early_alpha) * \
                    self.lfn_early_proj(early_enhanced)

            feat_last = torch.cat([template_tokens, search_enhanced], dim=1)
            # Store dynamic search token count so forward_head can use correct split
            aux_dict['num_search'] = search_enhanced.shape[1]

        out = self.forward_head(feat_last, None, feat_len_s=aux_dict.get('num_search', self.feat_len_s))
        out.update(aux_dict)
        out['backbone_feat'] = x
        # if Test is None:
        #     return out,out_list
        return out

    def forward_head(self, cat_feature, gt_score_map=None, feat_len_s=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if feat_len_s is None:
            feat_len_s = self.feat_len_s
        feat_sz = int(math.sqrt(feat_len_s))
        #print("cat_feature",cat_feature.shape)
        enc_opt = cat_feature[:, -feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, feat_sz, feat_sz)
        #print("opt_feat", opt_feat.shape)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type in ("CENTER", "CONDITIONAL_CENTER"):
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            # print("outputs_coord", outputs_coord.shape)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,}
            return out
        else:
            raise NotImplementedError

    def forward_heads(self, cat_feature, gt_score_map=None, feat_len_s=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if feat_len_s is None:
            feat_len_s = self.feat_len_s
        feat_sz = int(math.sqrt(feat_len_s))
        #print("cat_feature",cat_feature.shape)
        enc_opt = cat_feature[:, -feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, feat_sz, feat_sz)
        #print("opt_feat", opt_feat.shape)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type in ("CENTER", "CONDITIONAL_CENTER"):
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            # print("outputs_coord", outputs_coord.shape)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,}
            return out
        else:
            raise NotImplementedError


def resize_pos_embed(posemb, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_grid = posemb[0, :]
    
    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to new token with height:{} width: {}'.format(posemb_grid.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    # posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb_grid


def remap_legacy_keys(state_dict):
    """Map legacy parameter names to the renamed Layer-wise Fusion Network (LFN)
    and Conditional Center Head conventions, so that checkpoints saved before
    the rename remain loadable.

    Legacy → Current:
        liquid_fusion*            → lfn*   (e.g. liquid_fusion_mid_proj.weight → lfn_mid_proj.weight)
        box_head.gate*_ctr.log_tau → box_head.gate*_ctr.log_decay
    """
    remapped = {}
    for k, v in state_dict.items():
        if 'liquid_fusion' in k:
            k = k.replace('liquid_fusion', 'lfn')
        if 'log_tau' in k:
            k = k.replace('log_tau', 'log_decay')
        remapped[k] = v
    return remapped


def build_single_stream_track(cfg, training=True):
    """Build single-stream RGBT tracker with 4-channel input.

    Uses a unified ViT where RGB+T are fused at the input channel level.
    Self-attention naturally learns cross-modal relationships.
    DropTrack pretrained weights are loaded with 4ch patch embedding expansion.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_single_stream':
        lfn_multilayer = getattr(cfg.MODEL, 'LFN_MULTILAYER', False)
        lfn_3layer = getattr(cfg.MODEL, 'LFN_3LAYER', False)
        lfn_early_layer = getattr(cfg.MODEL, 'LFN_EARLY_LAYER', 3) if lfn_3layer else None
        gate_layers = getattr(cfg.MODEL.BACKBONE, 'GATE_LAYERS', [])
        if not gate_layers:
            gate_layers = None  # None = all standard blocks
        adapter_layers = getattr(cfg.MODEL.BACKBONE, 'ADAPTER_LAYERS', [])
        if not adapter_layers:
            adapter_layers = None  # None = no adapters
        adapter_dim = getattr(cfg.MODEL.BACKBONE, 'ADAPTER_DIM', 64)
        use_modality_prompt = getattr(cfg.MODEL, 'USE_MODALITY_PROMPT', False)
        modality_prompt_num = getattr(cfg.MODEL, 'MODALITY_PROMPT_NUM', 2)
        in_chans = int(getattr(cfg.MODEL, 'IN_CHANS', 4))
        backbone = vit_base_patch16_224_single_stream(
            pretrained='',  # We load weights manually below
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
            template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
            new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
            in_chans=in_chans,
            lfn_mid_layer=getattr(cfg.MODEL, 'LFN_MID_LAYER', 7) if lfn_multilayer else None,
            lfn_early_layer=lfn_early_layer,
            gate_layers=gate_layers,
            adapter_layers=adapter_layers,
            adapter_dim=adapter_dim,
            use_modality_prompt=use_modality_prompt,
            modality_prompt_num=modality_prompt_num,
        )
        hidden_dim = backbone.embed_dim
    else:
        raise NotImplementedError(f"Unknown backbone type: {cfg.MODEL.BACKBONE.TYPE}")

    box_head = build_box_head(cfg, hidden_dim)

    model = BATrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        use_lfn=getattr(cfg.MODEL, 'USE_LFN', True),
        lfn_num_iterations=getattr(cfg.MODEL, 'LFN_NUM_ITERATIONS', 2),
        lfn_use_attn_pool=getattr(cfg.MODEL, 'LFN_USE_ATTN_POOL', True),
        lfn_use_cross_attn=getattr(cfg.MODEL, 'LFN_USE_CROSS_ATTN', False),
        lfn_use_multiscale_gate=getattr(cfg.MODEL, 'LFN_USE_MULTISCALE_GATE', False),
        use_multilayer_lfn=lfn_multilayer,
        lfn_mid_layer=getattr(cfg.MODEL, 'LFN_MID_LAYER', 7),
        use_3layer_lfn=lfn_3layer,
        lfn_early_layer=getattr(cfg.MODEL, 'LFN_EARLY_LAYER', 3),
    )

    # ── Load DropTrack pretrained weights (with 3ch → in_chans expansion) ──
    if training and ('OSTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE):
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        param_dict = dict()
        in_chans = int(getattr(cfg.MODEL, 'IN_CHANS', 4))

        for k, v in checkpoint["net"].items():
            # Handle patch embedding: expand from 3ch → in_chans (TIR channels copy RGB)
            if 'patch_embed.proj.weight' in k:
                v_new = torch.zeros(v.shape[0], in_chans, v.shape[2], v.shape[3])
                v_new[:, :3] = v
                for c in range(3, in_chans):
                    v_new[:, c] = v[:, (c - 3) % 3]
                v = v_new
                print(f'[Single-Stream] Expanded patch_embed from 3ch → {in_chans}ch (TIR channels copied from RGB)')

            # Resize pos_embed to match configured template/search sizes
            tgt_z = cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.STRIDE
            tgt_x = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
            if 'pos_embed_x' in k:
                v = resize_pos_embed(v, tgt_x, tgt_x)
                temporal_key = 'backbone.temporal_pos_embed_x'
                if temporal_key in checkpoint["net"]:
                    v = v + checkpoint["net"][temporal_key]
            elif 'pos_embed_z' in k:
                v = resize_pos_embed(v, tgt_z, tgt_z)
                temporal_key = 'backbone.temporal_pos_embed_z'
                if temporal_key in checkpoint["net"]:
                    v = v + checkpoint["net"][temporal_key]

            param_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(param_dict), strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print(f"[Single-Stream] missing_keys: {[k for k in missing_keys if 'pos_embed' not in k and 'cls_token' not in k and 'dist_token' not in k]}")
        print(f"[Single-Stream] unexpected_keys: {unexpected_keys}")

    # ── Load single_stream checkpoint as pretrained weights ──
    elif training and pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        # ep13 checkpoint was saved from BATrack, keys already have 'backbone.' prefix — direct match
        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(checkpoint["net"]), strict=False)
        print(f'[Single-Stream] Load pretrained from: {pretrained}')
        print(f"[Single-Stream] missing_keys: {missing_keys}")
        print(f"[Single-Stream] unexpected_keys: {unexpected_keys}")

    # ── Adapter tuning: freeze everything except Adapters (+ box_head) ──
    # ViPT/AdaptFormer-style parameter-efficient fine-tuning. Explicit requires_grad=False
    # (rather than just BACKBONE_MULTIPLIER=0.0) so frozen params are excluded from the
    # optimizer entirely — no wasted AdamW state / gradient memory for the frozen backbone.
    if training and adapter_layers is not None and getattr(cfg.TRAIN, 'ADAPTER_TUNING', False):
        num_frozen, num_trainable = 0, 0
        for name, param in model.named_parameters():
            is_adapter_param = 'adapter_attn' in name or 'adapter_mlp' in name
            is_head_param = 'box_head' in name
            if is_adapter_param or is_head_param:
                param.requires_grad = True
                num_trainable += param.numel()
            else:
                param.requires_grad = False
                num_frozen += param.numel()
        print(f"[Adapter-Tuning] Frozen: {num_frozen:,}  Trainable (adapters+head): {num_trainable:,} "
              f"({100 * num_trainable / (num_frozen + num_trainable):.2f}%)")

    # ── Modality-prompt tuning: freeze everything except box_head + prompts ──
    elif training and getattr(cfg.TRAIN, 'PROMPT_TUNING', False):
        num_frozen, num_trainable = 0, 0
        for name, param in model.named_parameters():
            if 'box_head' in name or 'modality_prompt' in name:
                param.requires_grad = True
                num_trainable += param.numel()
            else:
                param.requires_grad = False
                num_frozen += param.numel()
        print(f"[Prompt-Tuning] Frozen: {num_frozen:,}  Trainable (head+prompts): {num_trainable:,} "
              f"({100 * num_trainable / (num_frozen + num_trainable):.2f}%)")

    elif training and getattr(cfg.TRAIN, 'HEAD_ONLY_TUNING', False):
        num_frozen, num_trainable = 0, 0
        for name, param in model.named_parameters():
            if 'box_head' in name:
                param.requires_grad = True
                num_trainable += param.numel()
            else:
                param.requires_grad = False
                num_frozen += param.numel()
        print(f"[Head-Only] Frozen: {num_frozen:,}  Trainable (head only): {num_trainable:,} "
              f"({100 * num_trainable / (num_frozen + num_trainable):.2f}%)")

    return model


def build_single_stream_mamba_track(cfg, training=True):
    """Build single-stream + Mamba FPN tracker.

    Two-stage training:
      Stage1 (mamba_stage1): freeze ViT backbone, train MambaFPN neck + head
      Stage2 (mamba_stage2): unfreeze all, fine-tune at lower LR
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # ── Build backbone with Mamba FPN ──
    fp_layers = getattr(cfg.MODEL.BACKBONE, 'FP_LAYERS', [3, 6, 9])

    backbone = vit_base_patch16_224_single_stream_mamba(
        pretrained='',  # loaded below
        drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
        template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
        new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
        fp_layers=fp_layers,
    )
    hidden_dim = backbone.embed_dim

    box_head = build_box_head(cfg, hidden_dim)

    model = BATrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    # ── Load pretrained weights (single_stream checkpoint) ──
    if training and pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        # The pretrained checkpoint (single_stream_rgbt_reg_ep20) has 'backbone.' prefix
        # but does NOT have mamba_neck keys. Load with strict=False — mamba_neck stays random init.
        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(checkpoint["net"]), strict=False)
        print(f'[Mamba] Load pretrained backbone+head from: {pretrained}')
        mamba_keys = [k for k in missing_keys if 'mamba_neck' in k]
        other_keys = [k for k in missing_keys if 'mamba_neck' not in k]
        if mamba_keys:
            print(f'[Mamba] MambaFPN keys (random init, expected): {len(mamba_keys)} params')
        if other_keys:
            print(f'[Mamba] Other missing keys: {other_keys}')
        print(f'[Mamba] Unexpected keys: {unexpected_keys}')

    return model


def build_single_stream_gated_track(cfg, training=True):
    """Build single-stream + Uncertainty Gating tracker.

    Layers 9-11 use GatedSSBlock with per-channel uncertainty gating.
    DropTrack pretrained weights loaded with 4ch expansion; gate weights init randomly.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    gate_layers = getattr(cfg.MODEL.BACKBONE, 'GATE_LAYERS', [9, 10, 11])

    backbone = vit_base_patch16_224_single_stream_gated(
        pretrained='',
        drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
        template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
        new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
        gate_layers=gate_layers,
    )
    hidden_dim = backbone.embed_dim

    box_head = build_box_head(cfg, hidden_dim)

    model = BATrack(backbone, box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE,
                    use_lfn=getattr(cfg.MODEL, 'USE_LFN', True),
                    lfn_num_iterations=getattr(cfg.MODEL, 'LFN_NUM_ITERATIONS', 2),
                    lfn_use_attn_pool=getattr(cfg.MODEL, 'LFN_USE_ATTN_POOL', True),
                    lfn_use_cross_attn=getattr(cfg.MODEL, 'LFN_USE_CROSS_ATTN', False),
                    lfn_use_multiscale_gate=getattr(cfg.MODEL, 'LFN_USE_MULTISCALE_GATE', False))

    # ── Load DropTrack/OSTrack pretrained weights (with 4ch expansion) ──
    if training and ('OSTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE):
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        pd = {}
        tz = cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.STRIDE
        tx = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
        for k, v in checkpoint['net'].items():
            if k == 'backbone.patch_embed.proj.weight' and v.shape[1] == 3:
                nw = torch.zeros(v.shape[0], 4, v.shape[2], v.shape[3])
                nw[:, :3] = v; nw[:, 3] = v[:, 1]; pd[k] = nw; continue
            if 'pos_embed_x' in k: v = resize_pos_embed(v, tx, tx)
            elif 'pos_embed_z' in k: v = resize_pos_embed(v, tz, tz)
            pd[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(pd), strict=False)
        gate_keys = [k for k in missing_keys if 'gate' in k]
        lfn_keys = [k for k in missing_keys if 'lfn' in k]
        other_keys = [k for k in missing_keys if 'gate' not in k and 'lfn' not in k]
        print(f'[Gate+LFN] Loaded pretrained: {cfg.MODEL.PRETRAIN_FILE}')
        if gate_keys: print(f'[Gate+LFN] Gate params (random init): {len(gate_keys)} keys')
        if lfn_keys: print(f'[Gate+LFN] LFN params (random init): {len(lfn_keys)} keys')
        if other_keys: print(f'[Gate+LFN] Other missing: {other_keys}')
    elif training and pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(checkpoint["net"]), strict=False)
        print(f'[Gated] Load pretrained from: {pretrained}')

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 3-Frame Temporal Model (TCAM — cross-frame attention)
# ═══════════════════════════════════════════════════════════════════════════

class BATrack3Frame(BATrack):
    """BATrack with CADTrack-style TCN temporal fusion.

    Processes frames sequentially through shared backbone,
    applies TCN across search tokens from all frames,
    then passes each frame's enhanced features to the head.
    Outputs a list of per-frame prediction dicts.
    """

    def __init__(self, backbone, box_head, temporal_fusion, head_type="CENTER"):
        super().__init__(backbone, box_head, head_type=head_type)
        self.temporal_fusion = temporal_fusion
        self._temporal_state = None  # for future streaming support
        # Learnable gate for fusing previous frame's features (zero-init → no impact initially)
        self.temporal_gate = nn.Parameter(torch.zeros(1, 1, backbone.embed_dim))

    def forward(self, template, search, num_frames=3, **kwargs):
        """
        Args:
            template:    [B*N, 4, Ht, Wt] or [B, 4, Ht, Wt]
            search:      [B*N, 4, Hs, Ws] — N frames concatenated in batch dim
            num_frames:  number of temporal frames
        Returns:
            list of per-frame dicts (matching CADTrack's output format),
            or single dict when num_frames==1 (matching BATrack for tracking)
        """
        # ── Single-frame shortcut: temporal-enhanced tracking ──
        if num_frames == 1:
            track_query_before = kwargs.get('track_query_before', None)
            out = super().forward(template=template, search=search,
                                  ce_template_mask=kwargs.get('ce_template_mask'),
                                  ce_keep_rate=kwargs.get('ce_keep_rate'),
                                  return_last_attn=kwargs.get('return_last_attn', False),
                                  Test=kwargs.get('Test'),
                                  frame_id=kwargs.get('frame_id'),
                                  template_masks=kwargs.get('template_masks'))
            feat = out.get('backbone_feat', None)
            if feat is not None:
                if isinstance(feat, list):
                    feat = feat[-1]
                search_tokens = feat[:, -self.feat_len_s:, :]  # [B, HW, C]

                # ── Temporal enhancement: fuse previous frame's features ──
                if track_query_before is not None:
                    # Pool previous frame's search tokens → per-channel bias
                    prev_pooled = track_query_before.mean(dim=1, keepdim=True)  # [B, 1, C]
                    # Learned gate (zero-init → identity at start)
                    enhanced = search_tokens + self.temporal_gate * prev_pooled
                    # Re-run head with temporally-enhanced search tokens
                    template_len = feat.shape[1] - self.feat_len_s
                    template_tokens = feat[:, :template_len, :]
                    cat_feat = torch.cat([template_tokens, enhanced], dim=1)
                    enhanced_out = self.forward_head(cat_feat)
                    out.update({k: enhanced_out[k] for k in
                                ['pred_boxes', 'score_map', 'size_map', 'offset_map']})

                # Store current search tokens as temporal state for next frame
                out['track_query_before'] = search_tokens.detach()
            return [out]  # wrap in list for consistent return type

        B_total = search.shape[0]
        B = B_total // num_frames

        # ── Step 1: Backbone forward (shared weights, all frames in batch) ──
        x, aux_dict, track_token = self.backbone(
            z=template, x=search,
            ce_template_mask=kwargs.get('ce_template_mask'),
            ce_keep_rate=kwargs.get('ce_keep_rate'),
            return_last_attn=kwargs.get('return_last_attn', False),
            dynamic_template=None,
            Test=kwargs.get('Test'),
            template_masks=kwargs.get('template_masks'))
        if isinstance(x, list):
            x = x[-1]
        # x: [B*N, template_tokens + search_tokens, C]

        # ── Step 2: Extract search tokens, reshape for TCN ──
        # x layout: [s0_f0, s1_f0, ..., sB-1_f0, s0_f1, ..., sB-1_fT-1]
        # i.e., frame-major: x[t*B + b] = sample b, frame t
        # Reshape to [T, B, HW, C] then permute to [B, T, HW, C]
        search_tokens = x[:, -self.feat_len_s:, :]          # [B*T, HW, C]
        search_tokens = search_tokens.reshape(num_frames, B, self.feat_len_s, -1)  # [T, B, HW, C]
        search_tokens = search_tokens.permute(1, 0, 2, 3)   # [B, T, HW, C]

        # ── Step 3: TCN temporal fusion ──
        search_enhanced = self.temporal_fusion(search_tokens)  # [B, T, HW, C]

        # ── Step 4: Layer-wise Fusion Network — template modulates search ──
        template_len = x.shape[1] - self.feat_len_s
        search_enhanced_lf = search_enhanced.clone()
        for t in range(num_frames):
            frame_template = x[t * B:(t + 1) * B, :template_len, :]       # [B, Nt, C]
            frame_search = search_enhanced[:, t, :, :]                      # [B, HW, C]
            search_enhanced_lf[:, t, :, :] = self.lfn(frame_search, frame_template)

        # ── Step 5: Per-frame head forward ──
        out_list = []
        for t in range(num_frames):
            frame_x = x[t * B:(t + 1) * B]                                # [B, L, C]
            frame_template = frame_x[:, :template_len]                     # [B, Nt, C]
            frame_search = search_enhanced_lf[:, t, :, :]                  # [B, HW, C]
            cat_feat = torch.cat([frame_template, frame_search], dim=1)
            frame_out = self.forward_head(cat_feat)
            frame_out.update(aux_dict)
            frame_out['backbone_feat'] = x
            out_list.append(frame_out)

        # Last frame: store temporal state for tracking
        out_list[-1]['track_query_before'] = search_enhanced[:, -1, :, :].detach()
        return out_list

    def reset_temporal_state(self):
        """Reset temporal state between sequences (CADTrack-compatible)."""
        self._temporal_state = None


def build_single_stream_3frame_track(cfg, training=True):
    """Build single-stream + CADTrack-style TCN temporal fusion."""
    from lib.models.bat.temporal_fusion import LFSTrackTemporalFusion

    backbone = vit_base_patch16_224_single_stream(
        pretrained='', drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
        template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
        new_patch_size=cfg.MODEL.BACKBONE.STRIDE)
    hidden_dim = backbone.embed_dim
    box_head = build_box_head(cfg, hidden_dim)
    tcn = LFSTrackTemporalFusion(d_model=hidden_dim, num_layers=2, kernel_size=3)
    model = BATrack3Frame(backbone, box_head, tcn, head_type=cfg.MODEL.HEAD.TYPE)

    if training and ('DropTrack' in cfg.MODEL.PRETRAIN_FILE or 'OSTrack' in cfg.MODEL.PRETRAIN_FILE):
        import pickle as _pk
        from types import SimpleNamespace
        _O = getattr(_pk, 'Unpickler', _pk._Unpickler)
        class _S(_O):
            def find_class(s, m, n):
                try: return super().find_class(m, n)
                except: return SimpleNamespace
        _pk.Unpickler = _S
        try: ck = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        finally: _pk.Unpickler = _O
        pd = {}
        tz = cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.STRIDE
        tx = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
        for k, v in ck['net'].items():
            # 4ch patch embed: use green channel copy (same as 1-frame, NOT mean!)
            if k == 'backbone.patch_embed.proj.weight' and v.shape[1] == 3:
                nw = torch.zeros(v.shape[0], 4, v.shape[2], v.shape[3])
                nw[:, :3] = v
                nw[:, 3] = v[:, 1]  # copy green channel
                pd[k] = nw
                continue
            # pos_embed: resize + add temporal_pos_embed (matching 1-frame behavior)
            if 'pos_embed_x' in k:
                v = resize_pos_embed(v, tx, tx)
                temporal_key = 'backbone.temporal_pos_embed_x'
                if temporal_key in ck['net']:
                    v = v + ck['net'][temporal_key]
            elif 'pos_embed_z' in k:
                v = resize_pos_embed(v, tz, tz)
                temporal_key = 'backbone.temporal_pos_embed_z'
                if temporal_key in ck['net']:
                    v = v + ck['net'][temporal_key]
            pd[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(pd), strict=False)
        print(f'[3Frame-TCN] Loaded pretrained: {cfg.MODEL.PRETRAIN_FILE}')
        tc = [k for k in missing_keys if 'temporal_fusion' in k]
        if tc: print(f'[3Frame-TCN] TCN random init: {len(tc)} keys')

    return model


def build_single_stream_mamba_hybrid_track(cfg, training=True):
    """Build single-stream + Mamba-Enhanced tracker.

    Layers 6-8:  MambaEnhancedBlock (ViT + parallel Mamba branch, zero-init gate)
    Layers 9-11: GatedSSBlock (channel gate)
    Pretrained from single_stream_gated best checkpoint.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    mamba_layers = getattr(cfg.MODEL.BACKBONE, 'MAMBA_LAYERS', [6, 7, 8])
    gate_layers = getattr(cfg.MODEL.BACKBONE, 'GATE_LAYERS', [9, 10, 11])

    backbone = VisionTransformerSingleStreamMambaHybrid(
        weight_init='skip',  # weights loaded from pretrained below
        drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
        template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
        new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
        mamba_layers=mamba_layers,
        gate_layers=gate_layers,
    )
    hidden_dim = backbone.embed_dim

    box_head = build_box_head(cfg, hidden_dim)

    model = BATrack(backbone, box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)

    if training and pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(checkpoint["net"]), strict=False)
        mamba_keys = [k for k in missing_keys if 'mamba' in k]
        gate_keys = [k for k in missing_keys if 'gate' in k]
        other_keys = [k for k in missing_keys if 'mamba' not in k and 'gate' not in k]
        print(f'[MambaEnhanced] Load pretrained backbone+head from: {pretrained}')
        if mamba_keys:
            print(f'[MambaEnhanced] Mamba params (random init, expected): {len(mamba_keys)} keys')
        if gate_keys:
            print(f'[MambaEnhanced] Gate params (random init, expected): {len(gate_keys)} keys')
        if other_keys:
            print(f'[MambaEnhanced] Other missing keys: {other_keys}')
        print(f'[MambaEnhanced] Unexpected keys: {unexpected_keys}')

    return model


def build_single_stream_gatedffn_track(cfg, training=True):
    """Build single-stream + ADDITIVE GatedFFN (SwiGLU) tracker.

    ADDITIVE design (same proven pattern as MambaEnhancedBlock):
      - All layers KEEP original Mlp (fc1/fc2) — pretrained weights load directly
      - All layers ADD parallel GatedFFN (SwiGLU)
      - Gate initialized to 1.0 (fully open), w3 initialized to N(0,1e-4)
      - At init: GatedFFN output ≈ 0 → exact pretrained behavior
      - w3 gets full gradients from epoch 1 → learns quickly
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    gate_layers = getattr(cfg.MODEL.BACKBONE, 'GATE_LAYERS', [9, 10, 11])

    backbone = vit_base_patch16_224_single_stream_gatedffn(
        pretrained='',  # loaded externally below
        drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
        template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
        new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
        gate_layers=gate_layers,
    )
    hidden_dim = backbone.embed_dim

    box_head = build_box_head(cfg, hidden_dim)

    model = BATrack(backbone, box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)

    # ── Re-init GatedFFN w3 to near-zero ──
    # With gate=1.0 and w3≈0: GatedFFN output ≈ 0 → preserves pretrained behavior.
    # But w3 gets FULL gradients (gate=1.0) → learns quickly from epoch 1.
    # w1, w2 gradients scale with w3 → grow as w3 grows.
    with torch.no_grad():
        w3_count = 0
        for n, p in model.named_parameters():
            if 'gated_ffn.w3.weight' in n:
                nn.init.normal_(p, std=1e-4)
                w3_count += 1
        if w3_count > 0:
            print(f'[GatedFFN] Re-initialized {w3_count} w3 weights to N(0, 1e-4)')

    if training and pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        # ── Direct load (no conversion needed!) ──
        # The ADDITIVE design keeps original Mlp (fc1/fc2) in all blocks,
        # so pretrained weights match directly. Only new GatedFFN params
        # (w1/w2/w3) and gate params are missing — expected.
        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(checkpoint["net"]), strict=False)
        gated_ffn_keys = [k for k in missing_keys if 'gated_ffn' in k]
        gated_ffn_gate_keys = [k for k in missing_keys if 'gated_ffn_gate' in k]
        norm_g_keys = [k for k in missing_keys if 'norm_g' in k]
        other_keys = [k for k in missing_keys if 'gated_ffn' not in k and 'gated_ffn_gate' not in k and 'norm_g' not in k]
        print(f'[GatedFFN] Load pretrained backbone+head from: {pretrained}')
        print(f'[GatedFFN] Original MLP (fc1/fc2) loaded directly — exact pretrained match')
        if gated_ffn_keys:
            print(f'[GatedFFN] GatedFFN params (random init, expected): {len(gated_ffn_keys)} keys')
        if gated_ffn_gate_keys:
            print(f'[GatedFFN] GatedFFN gates (zero init, expected): {len(gated_ffn_gate_keys)} keys')
        if norm_g_keys:
            print(f'[GatedFFN] GatedFFN norms (random init, expected): {len(norm_g_keys)} keys')
        if other_keys:
            print(f'[GatedFFN] Other missing keys: {other_keys}')
        if unexpected_keys:
            print(f'[GatedFFN] Unexpected keys: {unexpected_keys}')

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Mamba Temporal Fusion Model
# ═══════════════════════════════════════════════════════════════════════════

def build_single_stream_mamba_temporal_track(cfg, training=True, pretrained_ckpt=None):
    """Build single-stream + Mamba temporal fusion.

    Uses MambaTemporalFusion (selective SSM) instead of TCN.
    Designed to be loaded on top of a pre-trained single_stream checkpoint
    (e.g., V2 ep50) — backbone weights are inherited, Mamba module is zero-init.

    Args:
        cfg:            config object
        training:       whether in training mode
        pretrained_ckpt: path to pre-trained checkpoint (backbone + head weights)
    """
    from lib.models.bat.mamba_temporal_fusion import MambaTemporalFusion

    backbone = vit_base_patch16_224_single_stream(
        pretrained='', drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
        template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
        new_patch_size=cfg.MODEL.BACKBONE.STRIDE)
    hidden_dim = backbone.embed_dim
    box_head = build_box_head(cfg, hidden_dim)

    # Mamba temporal fusion (zero-init → identity at start)
    mamba_layers = getattr(cfg.TRAIN, 'MAMBA_TEMPORAL_LAYERS', 2)
    mamba_conv = getattr(cfg.TRAIN, 'MAMBA_TEMPORAL_KERNEL', 4)
    mamba_state = getattr(cfg.TRAIN, 'MAMBA_TEMPORAL_STATE', 16)
    mamba_expand = getattr(cfg.TRAIN, 'MAMBA_TEMPORAL_EXPAND', 2)
    mamba_fusion = MambaTemporalFusion(
        d_model=hidden_dim, num_layers=mamba_layers,
        d_state=mamba_state, d_conv=mamba_conv,
        expand=mamba_expand, drop_path=cfg.TRAIN.DROP_PATH_RATE)

    model = BATrack3Frame(backbone, box_head, mamba_fusion, head_type=cfg.MODEL.HEAD.TYPE)

    # ── Load pre-trained checkpoint (V2 ep50) ──
    if pretrained_ckpt:
        ck = torch.load(pretrained_ckpt, map_location='cpu')
        pd = {}
        tz = cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.STRIDE
        tx = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
        for k, v in ck['net'].items():
            # pos_embed resize for different search/template sizes
            if 'pos_embed_x' in k:
                v = resize_pos_embed(v, tx, tx)
            elif 'pos_embed_z' in k:
                v = resize_pos_embed(v, tz, tz)
            pd[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(remap_legacy_keys(pd), strict=False)
        print(f'[Mamba-3Frame] Loaded pretrained: {pretrained_ckpt}')
        mk = [k for k in missing_keys if 'temporal_fusion' in k or 'temporal_gate' in k]
        if mk:
            print(f'[Mamba-3Frame] Mamba temporal fusion (zero-init): {len(mk)} keys')
        other = [k for k in missing_keys if 'temporal_fusion' not in k and 'temporal_gate' not in k]
        if other:
            print(f'[Mamba-3Frame] Other missing: {other}')
        if unexpected_keys:
            print(f'[Mamba-3Frame] Unexpected: {unexpected_keys}')

    return model
