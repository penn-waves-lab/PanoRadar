import torch
import torch.nn as nn

from typing import Dict, List

from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.modeling import build_model
from detectron2.modeling.postprocessing import sem_seg_postprocess, detector_postprocess
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY

__all__ = ["TwoStageModel"]

@META_ARCH_REGISTRY.register()
class TwoStageModel(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        cfg,
        depth_sn_model: nn.Module,
        seg_obj_model: nn.Module,
    ):
        """
        Args:
            depth_sn_model: a depth and surface normal estimation model
            seg_obj_model: a object detection and segmentation model
        """
        super().__init__()
        self.cfg = cfg
        self.depth_sn_model = depth_sn_model
        self.seg_obj_model = seg_obj_model

        self.dummy_param = nn.Parameter(torch.empty(0)) # TODO:verify if we need this

    @classmethod
    def from_config(cls, cfg):
        cfg.MODEL.META_ARCHITECTURE = "DepthSnModel"
        depth_sn_model = build_model(cfg)

        cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNNSemanticSegmentor"
        cfg.MODEL.BACKBONE.NAME = "build_resnet_deeplab_fpn_backbone"
        cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
        cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
        seg_obj_model = build_model(cfg)
        cfg.MODEL.META_ARCHITECTURE = "TwoStageModel"
        return {
            "cfg": cfg,
            "depth_sn_model": depth_sn_model,
            "seg_obj_model": seg_obj_model,
        }

    @property
    def device(self):
        return self.dummy_param.device

    @staticmethod
    def _postprocess(
        depth_results,
        sn_results,
        seg_results,
        obj_instances,
        batched_inputs: List[Dict[str, torch.Tensor]],
        image_sizes,
    ):
        """
        Args:
            depth_results: list of depth results
            sn_results: list of surface normal results
            seg_results: list of semantic segmentation results
            obj_instances: list of object detection and segmentation results
            batched_inputs: a list of dicts
            image_sizes: list of (h, w) tuples
        Returns:
            results: list of dict
        """

        processed_results = []      
        for depth_result, sn_result, seg_result, obj_result, input_per_image, image_size in zip(
            depth_results, sn_results, seg_results, obj_instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            seg_r = sem_seg_postprocess(seg_result, image_size, height, width)
            obj_r = detector_postprocess(obj_result, height, width)
            processed_results.append(
                {"depth": depth_result, "sn": sn_result, "sem_seg": seg_r, "instances": obj_r}
            )
        return processed_results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        losses = {}
        # 1. depth and sn stage
        (depth_results, sn_results), depth_sn_losses = self.depth_sn_model(batched_inputs, return_outputs=True)
        losses.update(depth_sn_losses)

        # 2. transform
        depth = depth_results.repeat(1, 3, 1, 1).clamp(min=0, max=None) * 255
        depth = [depth[i] for i in range(depth.shape[0])]
        for i in range(len(batched_inputs)):
            batched_inputs[i]["image"] = depth[i]

        # 3. obj seg stage
        obj_seg_losses = self.seg_obj_model(batched_inputs)
        losses.update(obj_seg_losses)

        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        assert not self.training

        # images
        rf = [x["image"].to(self.device) for x in batched_inputs]
        rf = ImageList.from_tensors(rf)

        # 1. depth and sn stage
        (depth_results, sn_results) = self.depth_sn_model(batched_inputs, return_outputs=True)

         # 2. transform
        depth = depth_results.repeat(1, 3, 1, 1).clamp(min=0, max=None) * 255 
        depth = [depth[i] for i in range(depth.shape[0])]
        for i in range(len(batched_inputs)):
            batched_inputs[i]["image"] = depth[i]

        # 3. obj seg stage
        seg_results, obj_results = self.seg_obj_model(batched_inputs)

        for i in range(len(batched_inputs)):
            batched_inputs[i]["image"] = rf.tensor[i]

        return self._postprocess(
            depth_results,
            sn_results,
            seg_results,
            obj_results,
            batched_inputs,
            rf.image_sizes,
        )
