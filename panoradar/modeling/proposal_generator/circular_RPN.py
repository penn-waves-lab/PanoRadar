import torch
import torch.nn.functional as F
from typing import List

from panoradar.structures.circular_boxes import circular_pairwise_iou, CircularBoxes
from panoradar.modeling.circular_box_regression import CircularBox2BoxTransform

from detectron2.modeling.proposal_generator import PROPOSAL_GENERATOR_REGISTRY, RPN_HEAD_REGISTRY, RPN, StandardRPNHead
from detectron2.modeling.proposal_generator.proposal_utils import _is_tracing
from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import batched_nms, cat, Conv2d

def find_top_circ_rpn_proposals(proposals, pred_objectness_logits: List[torch.Tensor], image_sizes, nms_thresh, pre_nms_topk, post_nms_topk, min_box_size, training):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RRPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RRPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size(float): minimum proposal box side length in pixels (absolute units wrt
            input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """

    num_images = len(image_sizes)
    device = proposals[0].device if torch.jit.is_scripting() else ("cpu" if torch.jit.is_tracing() else proposals[0].device)

    # 1. Select top-k anchor for every level and every image
    topk_scores = [] #lvl tensor, each of shape N x topk
    topk_proposals = []
    level_ids = [] #lvl Tensor, each of shape (topk, )
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor): # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)
        
        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx] # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i, ), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = CircularBoxes(topk_proposals[n]) # CHANGED HERE
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/Nan. Training has diverged"
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold = min_box_size)
        lvl = lvl.to(keep.device)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        # Double up boxes that do not go across so that IOU calculations are correct
        boxes_tensor = boxes.tensor

        no_circular_indices = torch.where(boxes_tensor[:, 2] > boxes_tensor[:, 0])[0]

        # Keep a set of the indices of boxes that do not wrap around
        no_circular_indices_set = set(no_circular_indices.tolist())
        
        # unwrawp boxes to pass through batched nms

        boxes_tensor = torch.t(torch.stack((
            boxes_tensor[:, 0],
            boxes_tensor[:, 1],
            torch.where(boxes_tensor[:, 2] < boxes_tensor[:, 0], boxes_tensor[:, 2] + image_size[1], boxes_tensor[:, 2]),
            boxes_tensor[:, 3]
        )))

        n = len(boxes_tensor)

        # Double everything
        boxes_with_double = torch.concat((boxes_tensor, boxes_tensor))
        boxes_with_double[n:, [0,2]] = boxes_with_double[n:, [0,2]] + 512
        scores_double = torch.concat((scores_per_img, scores_per_img))
        lvl_double = torch.concat((lvl, lvl))
        # Pass through nms
        keep = batched_nms(boxes_with_double, scores_double, lvl_double, nms_thresh)

        # Set of indices we decided to keep after nms
        keep_set = set(keep.tolist())

        new_keep = keep[torch.where(keep < n)[0]] # Only keep the indices less than n, so original boxes
        # Only keep indices that wrap around, or where both copies are kept
        keep_from_keep = [ind for ind, keep_ind in enumerate(new_keep.tolist())
                          if keep_ind not in no_circular_indices_set or keep_ind + n in keep_set]


        new_keep = new_keep[torch.tensor(keep_from_keep)] if len(keep_from_keep) > 0 else torch.tensor([], dtype=torch.int)

        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.

        new_keep = new_keep[:post_nms_topk].to(device=boxes.tensor.device) # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[new_keep]
        res.proposal_boxes.clip(image_size)
        res.objectness_logits = scores_per_img[new_keep]
        results.append(res)

    return results

@RPN_HEAD_REGISTRY.register()
class CircularRPNHead(StandardRPNHead):
    """
    RPN classification and regression head but with circular padding to allow for circular object detection
    """

    # Remove padding in conv to add circular padding in forward function
    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride=1,
            padding=0, # Set padding 0, use own pad function later
            activation=torch.nn.ReLU(),
        )

    def pad(self, x, padding):
        """
        Apply circular padding to input along the azimuth axis, constant padding along elevation
        
        Args:
            x (Tensor): input that is to be padded.
            padding (int): size of padding
        """
        x = F.pad(x, (padding, padding, 0, 0), 'circular') # circular padding along azimuth
        x = F.pad(x, (0, 0, padding, padding), 'constant')
        return x


    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: list of feature maps
        
        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            # Pad input first with circular padding, then pass through convolution
            x = self.pad(x, padding=1)
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas

@PROPOSAL_GENERATOR_REGISTRY.register()
class CircularRPN(RPN):
    """
    Regional Proposal Network that supports circular proposals and bounding boxes
    """

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["box2box_transform"] = CircularBox2BoxTransform(
            weights = cfg.MODEL.RPN.BBOX_REG_WEIGHTS
        )
        # Use newly defined box2box transform here
        return ret

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[CircularBoxes], gt_instances: List[Instances]):
        """
        Args:
            anchors: list of anchor boxes
            gt_instances: the ground truth instances for each image
        
        Returns:
            list[Tensor]:
                List of #img tensors, i-th element is a vector of labels whose length is the total number of anchors across feature maps. Label values are in [-1, 0, 1]
                -1 = ignore, 0 = negative class, 1 = positive class
            list[Tensor]:
                i-th element is a NX5 tensor, where N is the total number of anchors across feature maps. The values are the matched gt boxes for each anchor, values undefined
                for those anchors not labeled as 1.
        """
        anchors = CircularBoxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        if not isinstance(gt_boxes[0], CircularBoxes):
            gt_boxes = [CircularBoxes(x.tensor, width=anchors.width) for x in gt_boxes]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for gt_boxes_i in gt_boxes:
            # gt_boxes_i: ground-truth boxes for the i-th image

            # Calculate IOU matrix
            match_quality_matrix = retry_if_cuda_oom(circular_pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)

            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)

            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
            
            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
        
        return gt_labels, matched_gt_boxes

    @torch.no_grad()
    def predict_proposals(self, anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Args:
            anchors (List[CircularBoxes]): List of anchor boxes
            pred_objectness_logits (List[Tensor]): List of predicted scores
            pred_anchor_deltas (List[Tensor]): List of predicted deltas that will be applied to the anchor boxes
            image_sizes (List[Tuple[int, int]]): List of image sizes for each anchor box
        
        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_circ_rpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_size,
            self.training
        )