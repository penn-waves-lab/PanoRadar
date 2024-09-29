import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess, detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, build_sem_seg_head

__all__ = ["GeneralizedRCNNSemanticSegmentor"]

@META_ARCH_REGISTRY.register()
class GeneralizedRCNNSemanticSegmentor(nn.Module):
    """
    Generalized R-CNN with an additional sementic segmentation head,
    which contains the following four components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    4. Sementic segmentation head
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        two_stage: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            two_stage: whether the model is part of panoradar two-stage model.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.sem_seg_head = sem_seg_head
        self.two_stage = two_stage

        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        if cfg.MODEL.CIRCULAR_SEG_OBJ:
            assert cfg.MODEL.ROI_HEADS.NAME == "CircularROIHeads", "ROI_HEADS.NAME must be CircularROIHeads"
            assert cfg.MODEL.RPN.HEAD_NAME == "CircularRPNHead", "RPN.HEAD_NAME must be CircularRPNHead"
            assert cfg.MODEL.ANCHOR_GENERATOR.NAME == "CircularAnchorGenerator", "ANCHOR_GENERATOR.NAME must be CircularAnchorGenerator"
            assert cfg.MODEL.PROPOSAL_GENERATOR.NAME == "CircularRPN", "PROPOSAL_GENERATOR.NAME must be CircularRPN"
        backbone = build_backbone(cfg)
        proposal_generator = build_proposal_generator(cfg, backbone.output_shape())
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        # fmt: on

        return {
            "backbone": backbone,
            "proposal_generator": proposal_generator,
            "roi_heads": roi_heads,
            "sem_seg_head": sem_seg_head,
            "input_format": cfg.INPUT.FORMAT,
            "two_stage": cfg.MODEL.TWO_STAGE,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                * sem_seg: semantic segmentation ground truth

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(
        seg_results,
        obj_instances,
        batched_inputs: List[Dict[str, torch.Tensor]],
        image_sizes,
    ):
        """
        Args:
            seg_results: list of semantic segmentation results
            obj_instances: list of object detection and segmentation results
            batched_inputs: a list of dicts
            image_sizes: list of (h, w) tuples
        Returns:
            results: list of dict
        """

        processed_results = []
        for seg_result, obj_result, input_per_image, image_size in zip(
            seg_results, obj_instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            seg_r = sem_seg_postprocess(seg_result, image_size, height, width)
            obj_r = detector_postprocess(obj_result, height, width)
            processed_results.append(
                {"sem_seg": seg_r, "instances": obj_r}
            )

        return processed_results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                * sem_seg: semantic segmentation ground truth

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            loss dictionary
        """
        if not self.training:
            if self.two_stage:
                return self.inference(batched_inputs, do_postprocess=False)
            else:
                return self.inference(batched_inputs, do_postprocess=True)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if "sem_seg" in batched_inputs[0]:
            gt_seg_targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_seg_targets = ImageList.from_tensors(
                gt_seg_targets,
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value,
                self.backbone.padding_constraints,
            ).tensor
        else:
            gt_seg_targets = None

        # 1. backbone forward
        features = self.backbone(images.tensor)

        # 2. proposal generator forward
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # 3. roi head forward
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # 4. semantic segmentor head forward
        _, sem_seg_losses = self.sem_seg_head(features, gt_seg_targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(sem_seg_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True
            list[dict]:
                Each dict is the output for one input image.
                    1. The dict contains a key "instances" whose value is a :class:`Instances`.
                    The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
                    2. The dict contains a key "sem_seg" whose value is a
                    Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.

            Otherwise, a tuple (list[ImageList], list[Instances]), 0 index for seg, 1 index for obj
        """

        assert not self.training

        # Pass through backbone
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # Obj
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            obj_results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            obj_results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # Seg
        seg_results, _ = self.sem_seg_head(features, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNNSemanticSegmentor._postprocess(
                seg_results, obj_results, batched_inputs, images.image_sizes
            )
        return seg_results, obj_results
