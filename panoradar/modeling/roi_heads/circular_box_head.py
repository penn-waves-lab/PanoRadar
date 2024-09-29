import torch

from panoradar.modeling.circular_box_regression import CircularBox2BoxTransform
from panoradar.structures.circular_boxes import CircularBoxes

from detectron2.layers import batched_nms
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

def circular_fast_rcnn_inference(
        boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    """
    Call `circular_fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """

    result_per_image = [
        circular_fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

@torch.no_grad()
def circular_fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying rotated non-maximum suppression (Rotated NMS).

    Args:
        Same as `circular_fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `circular_fast_rcnn_inference`, but for only one image.
    """

    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to CircularBoxes to use the `clip` function ...
    boxes = CircularBoxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    
    # Unwrap boxes so that can be passed through nms
    boxes.tensor = torch.t(torch.stack((
        boxes.tensor[:, 0],
        boxes.tensor[:, 1],
        torch.where(boxes.tensor[:, 2] < boxes.tensor[:, 0], boxes.tensor[:, 2] + image_shape[1], boxes.tensor[:, 2]),
        boxes.tensor[:, 3]
    )))

    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    boxes_tensor = boxes.clone()

    no_circular_indices = torch.where(boxes_tensor[:, 2] > boxes_tensor[:, 0])[0]

    # Keep a set of the indices of boxes that do not wrap around
    no_circular_indices_set = set(no_circular_indices.tolist())
    
    # unwrawp boxes to pass through batched nms
    boxes_tensor = torch.t(torch.stack((
        boxes_tensor[:, 0],
        boxes_tensor[:, 1],
        torch.where(boxes_tensor[:, 2] < boxes_tensor[:, 0], boxes_tensor[:, 2] + 512, boxes_tensor[:, 2]),
        boxes_tensor[:, 3]
    )))

    n = len(boxes_tensor)

    # Double everything
    boxes_with_double = torch.concat((boxes_tensor, boxes_tensor))
    boxes_with_double[n:, [0,2]] = boxes_with_double[n:, [0,2]] + 512
    scores_double = torch.concat((scores, scores))
    lvl_double = torch.concat((filter_inds[:,1], filter_inds[:, 1]))
    # Pass through nms
    keep = batched_nms(boxes_with_double, scores_double, lvl_double, nms_thresh)
    # Set of indices we decided to keep after nms
    keep_set = set(keep.tolist())

    new_keep = keep[torch.where(keep < n)[0]] # Only keep the indices less than n, so original boxes

    # Only keep indices that wrap around, or where both copies are kept
    keep_from_keep = [ind for ind, keep_ind in enumerate(new_keep.tolist()) 
                      if keep_ind not in no_circular_indices_set or keep_ind + n in keep_set]

    new_keep = new_keep[torch.tensor(keep_from_keep, dtype=torch.long)] if len(keep_from_keep) > 0 else torch.tensor([], dtype=torch.long)
    
    if topk_per_image >= 0:
        new_keep = new_keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[new_keep], scores[new_keep], filter_inds[new_keep]

    result = Instances(image_shape)
    result.pred_boxes = CircularBoxes(boxes)
    result.pred_boxes.clip(image_shape)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

class CircularFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting FastRCNN outputs with circular boxes
    """

    @classmethod
    def from_config(cls, cfg, input_shape):
        args = super().from_config(cfg, input_shape)
        args["box2box_transform"] = CircularBox2BoxTransform(
            weights = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )
        return args
    
    def inference(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `circ_fast_rcnn_inference`.
            list[Tensor]: same as `circ_fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return circular_fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )