"""
Offline Knowledge Distillation Actor: Dual-Stream Teacher → Single-Stream Student.

Teacher (BATrack, dual-stream, frozen) provides soft supervision at feature
and score-map levels. Student learns to mimic the teacher's internal
representations, indirectly acquiring modality-fusion knowledge.
"""
import torch
import torch.nn.functional as F

from lib.train.actors.bat import BATActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class DistillActor(BATActor):
    """Actor for offline knowledge distillation."""

    def __init__(self, net, objective, loss_weight, settings, cfg,
                 teacher_net=None, distill_weight=None):
        """
        Args:
            net: student network (single-stream)
            teacher_net: teacher network (dual-stream, frozen)
            distill_weight: dict with keys 'feature', 'score_map'
        """
        super().__init__(net, objective, loss_weight, settings, cfg)
        self.teacher = teacher_net
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.distill_weight = distill_weight or {'feature': 0.1, 'score_map': 0.5, 'attention': 0.05}
        self.criterion = torch.nn.BCEWithLogitsLoss()  # for template router (inherited)

    def __call__(self, data):
        # ── Teacher forward (no grad) ──
        with torch.no_grad():
            t_out = self._teacher_forward(data)

        # ── Student forward (override to request attention) ──
        s_out = self.forward_pass(data, return_last_attn=True)

        # ── Compute losses ──
        loss, status = self.compute_losses(s_out, data, t_out)
        return loss, status

    def forward_pass(self, data, return_last_attn=False):
        """Override to support attention distillation."""
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])

        box_mask_z = None
        ce_keep_rate = None
        self.epoch = data['epoch']
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        B, C, H_S, W_S = search_img.size()
        search_bboxes = torch.tensor(data['search_anno'][-1], dtype=torch.float32, device='cuda')
        self.search_bboxes_masks = self.get_bboxes_masks(search_bboxes, B, H_S, W_S)
        self.search_bboxes_masks = self.search_bboxes_masks.view(B, -1)
        self.search_masks = torch.where(self.search_bboxes_masks, torch.tensor(1).to('cuda'), torch.tensor(0).to('cuda')).to('cuda')

        B, C, H_T, W_T = template_img_i.size()
        template_bboxes = torch.tensor(data['template_anno'][-1], dtype=torch.float32, device='cuda')
        template_bboxes_masks = self.get_bboxes_masks(template_bboxes, B, H_T, W_T)
        self.template_bboxes_masks = template_bboxes_masks.view(B, -1)
        self.template_masks = torch.where(self.template_bboxes_masks, torch.tensor(1).to('cuda'), torch.tensor(0).to('cuda')).to('cuda')

        if len(data['template_images']) == 2:
            template_bboxes = torch.tensor(data['template_anno'][0], dtype=torch.float32, device='cuda')
            template_bboxes_masks = self.get_bboxes_masks(template_bboxes, B, H_T, W_T)
            self.template_bboxes_masks = template_bboxes_masks.view(B, -1)
            self.template_masks2 = torch.where(self.template_bboxes_masks, torch.tensor(1).to('cuda'), torch.tensor(0).to('cuda')).to('cuda')
            search_mask_list = []
            for i in range(B):
                search_mask_list.append(torch.cat([self.search_masks[i].unsqueeze(0), self.search_masks[i].unsqueeze(0)], dim=0))
            self.search_masks = torch.cat(search_mask_list, dim=0)

        out_dict = self.net(
            template=template_list,
            search=search_img,
            ce_template_mask=box_mask_z,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            template_masks=self.template_masks,
        )
        return out_dict

    def _teacher_forward(self, data):
        """Forward teacher with same data format as student."""
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])

        if len(template_list) == 1:
            template_list = template_list[0]

        return self.teacher(
            template=template_list,
            search=search_img,
            ce_template_mask=None,
            ce_keep_rate=None,
            return_last_attn=True,  # request attention for distill
        )

    def compute_losses(self, pred_dict, gt_dict, t_dict=None, return_status=True):
        """Compute detection + distillation losses."""
        # ── Standard detection loss (same as BATActor) ──
        gt_bbox = gt_dict['search_anno'][-1]
        B_orig, _ = gt_bbox.shape
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Duplicate GT for template pairs
        gt_bbox_list, gt_gaussian_maps_list = [], []
        for i in range(B_orig):
            gt_bbox_list.append(torch.cat([gt_bbox[i].unsqueeze(0), gt_bbox[i].unsqueeze(0)], dim=0))
            gt_gaussian_maps_list.append(torch.cat([gt_gaussian_maps[i].unsqueeze(0), gt_gaussian_maps[i].unsqueeze(0)], dim=0))
        gt_bbox = torch.cat(gt_bbox_list, dim=0)
        gt_gaussian_maps = torch.cat(gt_gaussian_maps_list, dim=0)

        # Dummy losses (single-stream has no uncertainty/temporal)
        futrue_loss = pred_dict.get('future_loss', torch.tensor(0.0, device=pred_dict['pred_boxes'].device))
        if isinstance(futrue_loss, torch.Tensor) and futrue_loss.numel() > 0:
            futrue_loss = futrue_loss.mean() * 0.1
        else:
            futrue_loss = torch.tensor(0.0)

        parameters = pred_dict.get('weights', [])
        sigma = 0.1 * (self.epoch / 30)
        y = self.search_masks
        y_n = 1 - y
        num = len(parameters)
        if num > 0:
            total_unc = 0
            for i in range(num):
                weight = (parameters[i].float().squeeze(1) - 1) * y + 1
                weight2 = (parameters[i].float().squeeze(1) - 1) * y_n + 1
                loss1 = torch.sum(torch.exp(-(weight - 1)), dim=1, keepdim=True) / 256
                loss2 = torch.sum(torch.exp(-(weight2 - 1)), dim=1, keepdim=True) / 256
                mseloss = -torch.log(loss2 / (loss1 + loss2))
                total_unc = total_unc + mseloss
            uncertainty_loss = sigma * (total_unc / num).mean()
        else:
            uncertainty_loss = torch.tensor(0.0, device=pred_dict['pred_boxes'].device)

        # Template router dummy
        scores = pred_dict.get("relative_score", torch.zeros(B_orig, 2))
        label_list = []
        pred_boxes = pred_dict['pred_boxes']
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat(1, pred_boxes.size(1), 1).view(-1, 4).clamp(min=0.0, max=1.0)
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        for i in range(0, B_orig * 2, 2):
            t1, t2 = iou[i], iou[i + 1]
            if t1 > t2:
                label = torch.tensor([1, 0]).unsqueeze(0)
            elif t1 == t2:
                label = torch.tensor([1, 1]).unsqueeze(0)
            else:
                label = torch.tensor([0, 1]).unsqueeze(0)
            label_list.append(label)
        labels = torch.cat(label_list, dim=0)
        templaterouter = 0.01 * self.criterion(scores.to('cuda'), labels.float().to('cuda'))

        # L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        # Location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # ── Detection loss ──
        det_loss = (
            self.loss_weight['giou'] * giou_loss +
            self.loss_weight['l1'] * l1_loss +
            self.loss_weight['focal'] * location_loss +
            uncertainty_loss + templaterouter + futrue_loss
        )

        # ── Distillation losses ──
        distill_loss = torch.tensor(0.0, device=det_loss.device)
        distill_feat_loss = torch.tensor(0.0)
        distill_score_loss = torch.tensor(0.0)

        if t_dict is not None:
            # Feature distillation: MSE on backbone features (search region)
            if 'backbone_feat' in t_dict and 'backbone_feat' in pred_dict:
                t_feat = t_dict['backbone_feat']
                s_feat = pred_dict['backbone_feat']
                if isinstance(t_feat, list):
                    t_feat = t_feat[-1]
                if isinstance(s_feat, list):
                    s_feat = s_feat[-1]
                # Only distill search-region tokens (last 256)
                t_search = t_feat[:, -256:, :]
                s_search = s_feat[:, -256:, :]
                distill_feat_loss = F.mse_loss(s_search, t_search)
                distill_loss = distill_loss + self.distill_weight['feature'] * distill_feat_loss

            # Score map distillation: MSE on score maps
            if 'score_map' in t_dict and 'score_map' in pred_dict:
                t_score = t_dict['score_map']
                s_score = pred_dict['score_map']
                if t_score is not None and s_score is not None:
                    distill_score_loss = F.mse_loss(s_score, t_score)
                    distill_loss = distill_loss + self.distill_weight['score_map'] * distill_score_loss

            # Attention distillation: KL on search-region self-attention
            # Teacher: dual-stream CEABlock_Enhancement produces different-size attn maps
            #   (RGB ~640 tokens, TIR ~448 tokens due to z_f prepend in RGB only).
            # Student: single-stream with 384 tokens (128 template + 256 search).
            # Strategy: extract last 256×256 submatrix (search self-attention) from all,
            #   average teacher's two streams, compare via KL with student.
            if 'attn' in t_dict and 'attn' in pred_dict:
                t_attn = t_dict['attn']
                s_attn = pred_dict['attn']
                if t_attn is not None and s_attn is not None and t_attn.dim() == 4 and s_attn.dim() == 4:
                    # Extract search-search attention: last 256 tokens
                    def extract_search_attn(a):
                        # a: [B, H, N, N] → extract last 256 tokens (search region)
                        return a[:, :, -256:, -256:]

                    t_search_attn = extract_search_attn(t_attn)
                    if 'i_attn' in t_dict and t_dict['i_attn'] is not None:
                        t_i_search_attn = extract_search_attn(t_dict['i_attn'])
                        t_search_attn = (t_search_attn + t_i_search_attn) / 2.0

                    s_search_attn = extract_search_attn(s_attn)

                    # KL divergence on attention distributions
                    t_prob = F.softmax(t_search_attn, dim=-1)
                    s_log = F.log_softmax(s_search_attn, dim=-1)
                    distill_attn_loss = F.kl_div(s_log, t_prob, reduction='batchmean')
                    distill_loss = distill_loss + self.distill_weight['attention'] * distill_attn_loss
                else:
                    distill_attn_loss = torch.tensor(0.0, device=det_loss.device)
            else:
                distill_attn_loss = torch.tensor(0.0, device=det_loss.device)

        loss = det_loss + distill_loss

        if return_status:
            mean_iou = iou.detach().mean()
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "Loss/distill_feat": distill_feat_loss.item() if isinstance(distill_feat_loss, torch.Tensor) else 0.0,
                "Loss/distill_score": distill_score_loss.item() if isinstance(distill_score_loss, torch.Tensor) else 0.0,
                "Loss/distill_attn": distill_attn_loss.item() if isinstance(distill_attn_loss, torch.Tensor) else 0.0,
                "IoU": mean_iou.item(),
                "u_m": uncertainty_loss.item() if isinstance(uncertainty_loss, torch.Tensor) else 0.0,
                "templaterouter": templaterouter.item(),
                "furuteLoss": futrue_loss.item() if isinstance(futrue_loss, torch.Tensor) else 0.0,
            }
            return loss, status
        else:
            return loss
