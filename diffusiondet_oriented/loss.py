# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet model and criterion classes.
"""
import torch
import torch.nn.functional as F
import math
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .util import box_ops
from .util.misc import get_world_size, is_dist_avail_and_initialized
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from iou_op import generalized_box_iou, generalized_polygon_iou, polygon_iou
from structures_rotated_boxes import pairwise_giou_rotated, pairwise_iou, pairwise_diou_rotated, diou_loss_rotated



def corners_to_center_height_width_angle(corners):
    """
    Convert corners representation to center, height, width, and angle format.

    Arguments:
        corners (Tensor[N, 8] or List[N, 8]): Tensor or list containing the corner coordinates
            in the order (x0, y0, x1, y1, x2, y2, x3, y3) for each box.

    Returns:
        boxes (Tensor[N, 5]): Tensor containing the boxes in the format
            (x_center, y_center, width, height, angle) for each box.
    """
    # Convert corners to tensor if it's not already a tensor
    if not isinstance(corners, torch.Tensor):
        corners = torch.tensor(corners)

    # Calculate the center coordinates
    x_center = (corners[:, 0] + corners[:, 2] + corners[:, 4] + corners[:, 6]) / 4
    y_center = (corners[:, 1] + corners[:, 3] + corners[:, 5] + corners[:, 7]) / 4

    # Calculate the width and height
    width = torch.sqrt((corners[:, 0] - corners[:, 2]) ** 2 + (corners[:, 1] - corners[:, 3]) ** 2)
    height = torch.sqrt((corners[:, 2] - corners[:, 4]) ** 2 + (corners[:, 3] - corners[:, 5]) ** 2)

    # Calculate the angle (in radians)
    angle_rad = torch.atan2(corners[:, 3] - corners[:, 1], corners[:, 2] - corners[:, 0])

    # Convert the angle from radians to degrees
    angle_deg = angle_rad * 180 / math.pi

    # Stack the center coordinates, width, height, and angle into a tensor
    boxes = torch.stack([x_center, y_center, width, height, angle_deg], dim=1)

    return boxes

class SetCriterionDynamicK(nn.Module):
    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_fed_loss:
            self.fed_loss_num_classes = 50
            from detectron2.data.detection_utils import get_fed_loss_cls_weights
            cls_weight_fun = lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER)  # noqa
            fed_loss_cls_weights = cls_weight_fun()
            assert (
                    len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        batch_size = len(targets)

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        # target_classes[idx] = target_classes_o
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        if self.use_focal or self.use_fed_loss:
            num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            gt_classes = torch.argmax(target_classes_onehot, dim=-1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            src_logits = src_logits.flatten(0, 1)
            target_classes_onehot = target_classes_onehot.flatten(0, 1)
            if self.use_focal:
                cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
            else:
                cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
            if self.use_fed_loss:
                K = self.num_classes
                N = src_logits.shape[0]
                fed_loss_classes = self.get_fed_loss_classes(
                    gt_classes,
                    num_fed_loss_classes=self.fed_loss_num_classes,
                    num_classes=K,
                    weight=self.fed_loss_cls_weights,
                )
                fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
                fed_loss_classes_mask[fed_loss_classes] = 1
                fed_loss_classes_mask = fed_loss_classes_mask[:K]
                weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()

                loss_ce = torch.sum(cls_loss * weight) / num_boxes
            else:
                loss_ce = torch.sum(cls_loss) / num_boxes

            losses = {'loss_ce': loss_ce}
        else:
            raise NotImplementedError

        return losses

    def oriented_box_loss(src_boxes, tgt_boxes):
        """
        Computes a corner-based loss between predicted and ground truth oriented bounding boxes.

        Args:
            src_boxes (torch.Tensor): Tensor of shape (num_boxes, 8) representing predicted oriented
                bounding boxes with coordinates (x0, y0, x1, y1, x2, y2, x3, y3).
            tgt_boxes (torch.Tensor): Tensor of shape (num_boxes, 8) representing ground truth oriented
                bounding boxes with coordinates (x0, y0, x1, y1, x2, y2, x3, y3).

        Returns:
            torch.Tensor: Tensor of shape (num_boxes,) containing the corner-based loss for each box.
        """
        num_boxes = src_boxes.size(0)
        losses = torch.zeros(num_boxes, device=src_boxes.device)

        for i in range(num_boxes):
            src_corners = src_boxes[i].view(-1, 2)  # (4, 2)
            tgt_corners = tgt_boxes[i].view(-1, 2)  # (4, 2)

            # Compute the distance between corresponding corners
            corner_distances = torch.sqrt(((src_corners - tgt_corners) ** 2).sum(dim=1))

            # Compute the mean distance across all corners
            losses[i] = corner_distances.mean()

        return losses
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 8]
        The target boxes are expected in format (x0, y0, x1, y1, x2, y2, x3, y3), absolute coordinates.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']  # Assuming src_boxes are in (x0, y0, x1, y1, x2, y2, x3, y3) format

        batch_size = len(targets)
        pred_box_list = []
        tgt_box_list = []

        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue

            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]["boxes"]  # Assuming (x0, y0, x1, y1, x2, y2, x3, y3) format

            pred_box_list.append(bz_src_boxes[valid_query])
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            target_boxes = torch.cat(tgt_box_list)
            num_boxes = src_boxes.shape[0]

            losses = {}

            # Compute oriented box loss between oriented bounding boxes
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # Compute GIoU loss between oriented bounding boxes
            loss_giou = diou_loss_rotated(corners_to_center_height_width_angle(src_boxes), corners_to_center_height_width_angle(target_boxes))
            losses['loss_giou'] = loss_giou.sum() / num_boxes

        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, _ = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1, use_focal: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.ota_k = cfg.MODEL.DiffusionDet.OTA_K
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.use_focal or self.use_fed_loss:
                out_prob = outputs["pred_logits"].sigmoid()  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 8]
            else:
                out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 8]

            indices = []
            matched_ids = []
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]  # [num_proposals, 8]
                bz_out_prob = out_prob[batch_idx]
                bz_tgt_ids = targets[batch_idx]["labels"]
                num_insts = len(bz_tgt_ids)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    continue
                bz_gtboxs = targets[batch_idx]['boxes']  # [num_gt, 8] (x0, y0, x1, y1, x2, y2, x3, y3)
               
                fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                    bz_boxes,  # (x0, y0, x1, y1, x2, y2, x3, y3)
                    bz_gtboxs,  # (x0, y0, x1, y1, x2, y2, x3, y3)
                    expanded_strides=32
                )
                
                
                

               

                pair_wise_ious = pairwise_iou(corners_to_center_height_width_angle(bz_boxes), corners_to_center_height_width_angle(bz_gtboxs)) 

                # Compute the classification cost.
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                else:
                    cost_class = -bz_out_prob[:, bz_tgt_ids]

                # Compute the corner distance cost between boxes
                cost_bbox = torch.cdist(bz_boxes, bz_gtboxs, p=1)
                cost_giou = -pairwise_diou_rotated(corners_to_center_height_width_angle(bz_boxes), corners_to_center_height_width_angle(bz_gtboxs))
                # Final cost matrix
                # Check dimensions and raise ValueError if there is a mismatch
                #if cost_bbox.shape != cost_class.shape or cost_bbox.shape != cost_giou.shape or cost_class.shape != cost_giou.shape:
              #  raise ValueError(f"Dimension mismatch: cost_bbox.shape={cost_bbox.shape}, "
               #                  f"cost_class.shape={cost_class.shape}, cost_giou.shape={cost_giou.shape}, is_in_boxes_and_center={is_in_boxes_and_center.shape}")
                cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * (-cost_giou) 
                #+ 100.0 * (~is_in_boxes_and_center)
                cost[~fg_mask] = cost[~fg_mask] + 10000.0

                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        """
        Determines if points (represented by the centers of the boxes) are inside or outside the ground truth oriented bounding boxes.
        
        Args:
            boxes (torch.Tensor): Tensor of shape (num_boxes, 8) representing predicted oriented bounding boxes with coordinates (x0, y0, x1, y1, x2, y2, x3, y3).
            target_gts (torch.Tensor): Tensor of shape (num_gts, 8) representing ground truth oriented bounding boxes with coordinates (x0, y0, x1, y1, x2, y2, x3, y3)
            expanded_strides (int): Stride value used for the center radius calculation.
            
        Returns:
            Tuple containing:
            is_in_boxes_anchor (torch.Tensor): Tensor of shape (num_boxes,) indicating whether the center of each box is inside any ground truth box.
            is_in_boxes_and_center (torch.Tensor): Tensor of shape (num_boxes, num_gts) indicating whether the center of each box is inside each ground truth box and within a certain radius from the ground truth box center.
        """
        
        num_boxes = boxes.size(0)
        num_gts = target_gts.size(0)
        
        # Calculate the centers of the boxes and target ground truth boxes
        box_centers = boxes.view(-1, 4, 2).mean(dim=1)  # (num_boxes, 2)
        gt_centers = target_gts.view(-1, 4, 2).mean(dim=1)  # (num_gts, 2)
        
        # Repeat the box centers and ground truth centers for efficient computation
        expanded_box_centers = box_centers.unsqueeze(1).repeat(1, num_gts, 1)  # (num_boxes, num_gts, 2)
        expanded_gt_centers = gt_centers.unsqueeze(0).repeat(num_boxes, 1, 1)  # (num_boxes, num_gts, 2)
        
        # Compute the distances between box centers and ground truth centers
        center_distances = torch.sqrt(((expanded_box_centers - expanded_gt_centers) ** 2).sum(-1))  # (num_boxes, num_gts)
        
        # Compute the polygon areas of the ground truth boxes
        target_gts_poly = target_gts.view(-1, 4, 2)  # (num_gts, 4, 2)
        v1 = torch.cat([target_gts_poly[:, 1, :] - target_gts_poly[:, 0, :], torch.zeros_like(target_gts_poly[:, 0, :1])], dim=-1)  # (num_gts, 3)
        v2 = torch.cat([target_gts_poly[:, 2, :] - target_gts_poly[:, 0, :], torch.zeros_like(target_gts_poly[:, 0, :1])], dim=-1)  # (num_gts, 3)
        gt_areas = torch.abs(torch.cross(v1, v2, dim=-1)).sum(-1) / 2  # (num_gts,)
        
        # Compute the maximum diagonal length of the ground truth boxes
        max_diag_lengths = torch.sqrt(((target_gts_poly.max(dim=1)[0] - target_gts_poly.min(dim=1)[0]) ** 2).sum(-1))  # (num_gts,)
        
        # Compute the center radius for each ground truth box
        center_radii = max_diag_lengths / expanded_strides  # (num_gts,)
        
        # Determine if the box centers are inside the ground truth boxes
        is_in_boxes = (center_distances <= gt_areas.view(1, -1) / center_radii.view(1, -1))  # (num_boxes, num_gts)
        
        # Determine if the box centers are inside the ground truth boxes and within the center radius
        is_in_centers = (center_distances <= center_radii.view(1, -1))  # (num_boxes, num_gts)
        is_in_boxes_and_center = is_in_boxes & is_in_centers  # (num_boxes, num_gts)
        is_in_boxes_anchor = is_in_boxes.any(dim=1)  # (num_boxes,)
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return (selected_query, gt_indices), matched_query_id
