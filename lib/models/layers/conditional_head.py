"""
Conditional Center Head for LFSTrack.

A center head whose prediction-branch updates are constrained by learnable
positive decay parameters: after each convolutional stage, a ConditionalGate
derived from the stage features gates the response of that stage, which is
then shared with the size/offset branches for branch consistency.

Keeps the same I/O contract as CenterPredictor:

    forward(x) -> score_map_ctr, bbox, size_map, offset_map
    cal_bbox(score_map_ctr, size_map, offset_map, return_score=False)

So the existing training actor and test tracker can reuse it without changes.
"""

import torch
import torch.nn as nn

from lib.models.layers.frozen_bn import FrozenBatchNorm2d


def _conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                dilation=1, freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class ConditionalGate(nn.Module):
    """State-dependent gate with a learnable positive decay parameter.

    gate(x) = sigmoid(g(x))
    cond_gate = gate + (1 - gate) * exp(-decay),  decay > 0

    The decay term keeps every branch at least partially open, avoiding
    over-aggressive shutdown and keeping gradients stable early in training.
    """

    def __init__(self, in_channels: int, init_decay: float = 2.0, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.log_decay = nn.Parameter(torch.tensor(float(init_decay)).log())
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_gate = self.gate_net(x).sigmoid()      # [B, 1]
        decay = self.log_decay.exp()               # scalar > 0
        cond_gate = raw_gate + (1.0 - raw_gate) * (-decay).exp()
        return cond_gate.view(-1, 1, 1, 1)


class ConditionalCenterHead(nn.Module):
    """Conditionally-gated center head.

    Differences vs CenterPredictor:
    - Insert a ConditionalGate after each conv stage of the center branch.
    - Share the center-branch gates with size/offset branches to keep branch
      consistency at each depth.
    """

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16,
                 freeze_bn=False, init_decay=2.0):
        super().__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # center branch
        self.conv1_ctr = _conv_block(inplanes, channel, freeze_bn=freeze_bn)
        self.gate1_ctr = ConditionalGate(channel, init_decay)
        self.conv2_ctr = _conv_block(channel, channel // 2, freeze_bn=freeze_bn)
        self.gate2_ctr = ConditionalGate(channel // 2, init_decay)
        self.conv3_ctr = _conv_block(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.gate3_ctr = ConditionalGate(channel // 4, init_decay)
        self.conv4_ctr = _conv_block(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.gate4_ctr = ConditionalGate(channel // 8, init_decay)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # offset branch
        self.conv1_offset = _conv_block(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = _conv_block(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = _conv_block(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = _conv_block(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size branch
        self.conv1_size = _conv_block(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = _conv_block(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = _conv_block(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = _conv_block(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        score_map_ctr, size_map, offset_map = self.get_score_map(x)
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)
        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        bbox = torch.cat([
            (idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
            (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
            size.squeeze(-1)
        ], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        return size * self.feat_sz, offset

    def get_score_map(self, x):
        def _sigmoid(v):
            return torch.clamp(v.sigmoid_(), min=1e-4, max=1 - 1e-4)

        h_ctr = self.conv1_ctr(x)
        g1 = self.gate1_ctr(h_ctr)
        h_ctr = h_ctr * g1

        h_ctr = self.conv2_ctr(h_ctr)
        g2 = self.gate2_ctr(h_ctr)
        h_ctr = h_ctr * g2

        h_ctr = self.conv3_ctr(h_ctr)
        g3 = self.gate3_ctr(h_ctr)
        h_ctr = h_ctr * g3

        h_ctr = self.conv4_ctr(h_ctr)
        g4 = self.gate4_ctr(h_ctr)
        h_ctr = h_ctr * g4
        score_map_ctr = self.conv5_ctr(h_ctr)

        h_offset = self.conv1_offset(x) * g1
        h_offset = self.conv2_offset(h_offset) * g2
        h_offset = self.conv3_offset(h_offset) * g3
        h_offset = self.conv4_offset(h_offset) * g4
        score_map_offset = self.conv5_offset(h_offset)

        h_size = self.conv1_size(x) * g1
        h_size = self.conv2_size(h_size) * g2
        h_size = self.conv3_size(h_size) * g3
        h_size = self.conv4_size(h_size) * g4
        score_map_size = self.conv5_size(h_size)

        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset


def build_conditional_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = getattr(cfg.MODEL.HEAD, "NUM_CHANNELS", 256)
    init_decay = getattr(cfg.MODEL.HEAD, "COND_INIT_DECAY", 2.0)
    # FIX_BN controls whether the trainer freezes head BN layers to eval mode.
    # When true (the standard setting for this codebase), BN acts as a stable,
    # deterministic per-channel affine transform — no train/val mismatch.
    # The freeze_bn parameter here chooses the BN implementation accordingly.
    freeze_bn = getattr(cfg.TRAIN, "FIX_BN", False)
    print(f"[ConditionalHead] building CONDITIONAL_CENTER: feat_sz={feat_sz}, channel={channel}, init_decay={init_decay}, freeze_bn={freeze_bn}")
    return ConditionalCenterHead(
        inplanes=hidden_dim,
        channel=channel,
        feat_sz=feat_sz,
        stride=stride,
        freeze_bn=freeze_bn,
        init_decay=init_decay,
    )
