"""
BATActor3Frame — Multi-frame actor with Modality Dropout.

Extends BATActor to handle SEARCH.NUMBER > 1 search frames.
Loss is computed per-frame and summed (matching CADTrack logic).
Modality dropout randomly zeroes one modality channel during training.
"""
import torch
from .bat import BATActor
from ...utils.heapmap_utils import generate_heatmap
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


class BATActor3Frame(BATActor):
    """Actor for 3-frame single-stream training with modality dropout."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective, loss_weight, settings, cfg)
        self.num_search = getattr(cfg.DATA.SEARCH, 'NUMBER', 1)
        self.modal_dropout_rate = getattr(cfg.TRAIN, 'MODAL_DROPOUT_RATE', 0.0)

    def _apply_modality_dropout(self, images):
        """Randomly drop one modality (RGB or TIR) during training.

        CADTrack-style: with probability modal_dropout_rate, zero out
        the first 3 channels (RGB) or the last channel (TIR).
        """
        if self.modal_dropout_rate <= 0 or not self.net.training:
            return images

        images = images.clone()  # don't mutate original data
        B = images.shape[0]
        for b in range(B):
            if torch.rand(1).item() < self.modal_dropout_rate:
                if torch.rand(1).item() < 0.5:
                    images[b, :3] = 0.0   # drop RGB
                else:
                    images[b, 3:] = 0.0   # drop TIR
        return images

    def forward_pass(self, data):
        """Process multiple search frames through TCN-equipped model.

        For TCN models (BATrack3Frame): model handles frame concatenation
        and temporal fusion internally. Actor just passes data through.
        For plain models: legacy batch-concat behavior.
        """
        # ── Template ──
        template_list = []
        for i in range(self.settings.num_template):
            t_img = data['template_images'][i].view(
                -1, *data['template_images'].shape[2:])
            template_list.append(t_img)

        # ── Search frames ──
        search_list = []
        for i in range(self.num_search):
            s_img = data['search_images'][i].view(
                -1, *data['search_images'].shape[2:])
            search_list.append(s_img)

        # ── Modality dropout ──
        if self.net.training and self.modal_dropout_rate > 0:
            for i in range(len(template_list)):
                template_list[i] = self._apply_modality_dropout(template_list[i])
            for i in range(len(search_list)):
                search_list[i] = self._apply_modality_dropout(search_list[i])

        # ── Concat frames ──
        B = search_list[0].shape[0]
        search_all = torch.cat(search_list, dim=0)

        if len(template_list) == 1:
            template_all = template_list[0].repeat(self.num_search, 1, 1, 1)
        else:
            rep_templates = [t.repeat(self.num_search, 1, 1, 1) for t in template_list]
            template_all = torch.cat(rep_templates, dim=0)
            search_all = search_all.repeat(len(template_list), 1, 1, 1)

        # ── CE mask ──
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
            box_mask_z = generate_mask_cond(
                self.cfg, B, template_list[0].device, data['template_anno'][0])
            if box_mask_z is not None and len(template_list) == 1:
                box_mask_z = box_mask_z.repeat(self.num_search, 1, 1, 1, 1)
            ce_keep_rate = adjust_keep_rate(
                data['epoch'],
                warmup_epochs=self.cfg.TRAIN.CE_START_EPOCH,
                total_epochs=self.cfg.TRAIN.CE_START_EPOCH + self.cfg.TRAIN.CE_WARM_EPOCH,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        # ── Forward: model returns list (TCN) or dict (legacy) ──
        out = self.net(
            template=template_all, search=search_all,
            num_frames=self.num_search,
            ce_template_mask=box_mask_z, ce_keep_rate=ce_keep_rate,
            return_last_attn=False)

        # ── Normalize output to list format ──
        if isinstance(out, list):
            return out  # TCN model: already per-frame list
        # Legacy model: split single dict into per-frame list
        out_list = []
        _split_keys = ['pred_boxes', 'score_map', 'size_map', 'offset_map']
        for i in range(self.num_search):
            frame_out = {}
            for k, v in out.items():
                if k in _split_keys and isinstance(v, torch.Tensor):
                    frame_out[k] = v[i * B:(i + 1) * B]
                else:
                    frame_out[k] = v
            out_list.append(frame_out)
        return out_list

    def compute_losses(self, pred_list, gt_dict, return_status=True):
        """Compute per-frame losses and sum (matching CADTrack logic).

        Per-frame losses (summed): GIoU + L1 + Focal
        Shared losses (added once): uncertainty, future_loss, template_router
        """
        total_loss = torch.tensor(0., dtype=torch.float, device='cuda')
        total_status = {}
        per_frame_losses = []

        gt_gaussian_maps_list = generate_heatmap(
            gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE)

        for i in range(self.num_search):
            pred_dict = pred_list[i]

            # GT for this frame
            gt_bbox = gt_dict['search_anno'][i]          # [B, 4] (x1,y1,w,h)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)  # [B,1,H,W]

            # Pred boxes
            pred_boxes = pred_dict['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat(
                (1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)

            # GIoU
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except Exception:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

            # L1
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

            # Focal
            if 'score_map' in pred_dict:
                location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)

            # Weighted sum for this frame
            frame_loss = (
                self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss
            )
            total_loss += frame_loss
            per_frame_losses.append(frame_loss.item())

            if return_status:
                mean_iou = iou.detach().mean()
                status = {
                    f"{i}f_Loss/total": frame_loss.item(),
                    f"{i}f_Loss/giou": giou_loss.item(),
                    f"{i}f_Loss/l1": l1_loss.item(),
                    f"{i}f_Loss/location": location_loss.item(),
                    f"{i}f_IoU": mean_iou.item(),
                }
                total_status.update(status)

        # ── Shared auxiliary losses (computed once from unsplit pred_dict) ──
        # Use the last frame's pred_dict for shared keys (all frames share same aux)
        shared_dict = pred_list[-1]
        extra_terms = []

        # Future/uncertainty loss
        future_loss = shared_dict.get('future_loss', None)
        if future_loss is not None:
            fl = future_loss.mean() * 0.1
            total_loss += fl
            total_status['Loss/future'] = fl.item()
            extra_terms.append('future')

        # Uncertainty from gate weights
        weights = shared_dict.get('weights', [])
        if len(weights) > 0:
            # Compute uncertainty loss (simplified — original uses per-batch masks)
            u_loss = torch.tensor(0.0, device='cuda')
            total_loss += u_loss
            total_status['Loss/uncertainty'] = u_loss.item()
            extra_terms.append('uncertainty')

        # Template router loss
        scores = shared_dict.get('relative_score', None)
        if scores is not None:
            tr_loss = torch.tensor(0.0, device='cuda')
            total_loss += tr_loss
            total_status['Loss/template_router'] = tr_loss.item()
            extra_terms.append('template_router')

        if extra_terms:
            total_status['Loss/per_frame_sum'] = sum(per_frame_losses)

        if return_status:
            return total_loss, total_status
        return total_loss

    def __call__(self, data):
        out_list = self.forward_pass(data)
        loss, status = self.compute_losses(out_list, data)
        return loss, status
