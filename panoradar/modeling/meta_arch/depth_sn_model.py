import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY

from typing import Dict, List

from ..decoder.depth_sn_head import build_depth_sn_head

@META_ARCH_REGISTRY.register()
class DepthSnModel(nn.Module):
    """This is the model for depth and surface normal."""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        depth_sn_head: nn.Module,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface.
            depth_sn_head: a depth and surface normal decoder module.
        """
        super().__init__()
        self.backbone = backbone
        self.depth_sn_head = depth_sn_head

        # TODO: verify if we really need this
        self.dummy_param = nn.Parameter(torch.empty(0))

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        depth_sn_head = build_depth_sn_head(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "depth_sn_head": depth_sn_head,
        }

    @property
    def device(self):
        return self.dummy_param.device

    @staticmethod
    def _postprocess(depth_results, sn_results):
        """
        Post processing of the inference results.
        """

        processed_results = []
        for depth_result, sn_result in zip(depth_results, sn_results):
            processed_results.append({"depth": depth_result, "sn": sn_result})

        return processed_results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], return_outputs=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * depth: Tensor, groundtruth depth in (1, H, W) format
                * sn: Tensor, groundtruth normal in (3, H, W) format

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            loss dictionary and optionally the output predictions.
        """
        if not self.training:
            return self.inference(batched_inputs, return_outputs)

        # images
        rf = [x["image"].to(self.device) for x in batched_inputs]
        rf = ImageList.from_tensors(rf).tensor

        # depth
        gt_depth = [x["depth"].to(self.device) for x in batched_inputs]
        gt_depth = ImageList.from_tensors(gt_depth).tensor

        # surface normal
        gt_sn = [x["sn"].to(self.device) for x in batched_inputs]
        gt_sn = ImageList.from_tensors(gt_sn).tensor

        # 1. backbone forward
        features = self.backbone(rf)

        # 2. depth and sn forward
        (depth_results, sn_results), depth_sn_losses = self.depth_sn_head(
            features=features,
            targets=(gt_depth, gt_sn),
            return_features=False
        )

        losses = {}
        losses.update(depth_sn_losses)
        if return_outputs:
            return (depth_results, sn_results), losses
        else:
            return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], return_outputs=False):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`

        Returns:
            list[dict]: a list of dicts containing the inference results.
                Each dict contains the following keys:

                * depth: Tensor, the predicted depth in (1, H, W) format.
                * sn: Tensor, the predicted surface normal in (3, H, W) format.
        """

        assert not self.training

        # images
        rf = [x["image"].to(self.device) for x in batched_inputs]
        rf = ImageList.from_tensors(rf).tensor

        # Pass through backbone
        features = self.backbone(rf)

        # Depth and sn
        (depth_results, sn_results), _ = self.depth_sn_head(
            features=features,
            targets=None,
            return_features=False
        )

        if return_outputs:
            return (depth_results, sn_results)
        else:
            return self._postprocess(depth_results, sn_results)
