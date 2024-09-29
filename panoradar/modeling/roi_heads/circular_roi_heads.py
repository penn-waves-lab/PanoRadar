import torch
import numpy as np
from typing import List, Dict

from panoradar.structures.circular_boxes import CircularBoxes, circular_pairwise_iou
from panoradar.modeling.roi_heads.circular_box_head import CircularFastRCNNOutputLayers

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads, build_box_head
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals

from detectron2.utils.events import get_event_storage
from detectron2.layers import ShapeSpec
from detectron2.structures import Instances

@ROI_HEADS_REGISTRY.register()
class CircularROIHeads(StandardROIHeads):


    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        
        # Assume all channel counts are equal
        in_channels = [input_shape[f].channels for f in in_features]
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # RCNN head is split into box head and box_predictor
        # box head contains all the conv layers and fc layers and 
        # box_predictor contains the two output layers for classification and bbox regression
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = CircularFastRCNNOutputLayers(cfg, box_head.output_shape) # This is a change to include the custom head
        
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor
        }
    
    def update_to_circ_box_gt(self, instance):
        instance.gt_boxes = CircularBoxes(instance.gt_boxes.tensor)
        return instance

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        if self.proposal_append_gt:
            targets = [self.update_to_circ_box_gt(x) for x in targets]
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = circular_pairwise_iou(
                CircularBoxes(targets_per_image.gt_boxes.tensor), CircularBoxes(proposals_per_image.proposal_boxes.tensor)
            ) # CHANGE HERE
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[sampled_targets]

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    
    def update_proposal_box(self, instances: Instances):
        """
        Helper method to unwrap the wrapped boxes in order to pass into pooling
        
        Args:
            instances: Instances object that contains proposals
        
        Returns:
            instances: Instances object with the unwrapped proposal boxes
        """
        instances.proposal_boxes.tensor = torch.t(
            torch.stack((
                instances.proposal_boxes.tensor[:, 0],
                instances.proposal_boxes.tensor[:, 1],
                torch.where(instances.proposal_boxes.tensor[:, 2] < instances.proposal_boxes.tensor[:, 0], instances.proposal_boxes.tensor[:, 2] + 512, instances.proposal_boxes.tensor[:, 2]),
                instances.proposal_boxes.tensor[:, 3]
            )))
        return instances


    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        
        # Double the image in the horizontal dimension by concat
        features = [torch.cat((x, x), dim=3) for x in features]

        # Modify boxes so that the wrapped boxes go over instead of wrapping
        proposal_boxes = [self.update_proposal_box(x).proposal_boxes for x  in proposals] # CHANGE HERE so that the boxes can pass through ROI pooling

        # Pass through ROI pooler to get box features
        box_features = self.box_pooler(features, proposal_boxes)

        # Pass through box head for box features to go through convolutions
        box_features = self.box_head(box_features)

        # Pass through last layer
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = CircularBoxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances