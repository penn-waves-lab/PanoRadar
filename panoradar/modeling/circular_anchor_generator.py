import torch
from typing import List

from panoradar.structures.circular_boxes import CircularBoxes

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.anchor_generator import ANCHOR_GENERATOR_REGISTRY

@ANCHOR_GENERATOR_REGISTRY.register()
class CircularAnchorGenerator(DefaultAnchorGenerator):
    """
    Compute anchor boxes but return them in the CircularBoxes class, with each anchor box coordinate modified to be between 0 and 512 (img_width)
    """

    def forward(self, features: List[torch.Tensor], image_size = (64, 512)):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)

        # Mod all widths by 512 so that the x values fall between 0 and 512
        anchors_modded_widths = [torch.cat((x[:, [0, 2]] % 512, x), dim=1) for x in anchors_over_all_feature_maps]
        anchors_over_all_feature_maps_new = [torch.stack((x[:, 0], x[:, 3], x[:, 1], x[:, 5]), dim = 1) for x in anchors_modded_widths]

        # Return CircularBoxes instead of Boxes
        return [CircularBoxes(x, width=image_size[1]) for x in anchors_over_all_feature_maps_new]