import math
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, Backbone
from detectron2.layers import Conv2d, get_norm

def pad(x, padding, circular=True):
    """
    Pad the input tensor.
    Args:
        x (torch.Tensor): input tensor.
        padding (int): padding size.
        circular (bool): whether to use circular padding.
    Returns:
        torch.Tensor: padded tensor.
    """
    if circular:
        x = F.pad(x, (padding, padding, 0, 0), 'circular')
    else:
        x = F.pad(x, (padding, padding, 0, 0), 'constant')
    x = F.pad(x, (0, 0, padding, padding), 'constant')
    return x

class Last2LevelMaxPool(LastLevelMaxPool):
    """
    Generate two more downsampled features maps from the last feature map.
    """
    def __init__(self):
        super().__init__()
        self.num_levels = 2

    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=1, stride=2, padding=0)
        return [x, F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class CustomFPN(FPN):
    """
    Custom FPN that also returns the ResNet outputs besides the FPN output.
    Args:
        bottom_up (Backbone): a backbone module.
        in_features (list[str]): names of the input feature maps coming from the backbone to which FPN is attached.
        out_channels (int): number of output channels (used at each FPN level).
        norm (str or callable): normalization for the convolution layers.
        top_block (nn.Module, optional): an extra block after FPN.
        fuse_type (str): how to fuse the two feature maps in the top-down pathway. It can be "sum" or "avg".
        square_pad (int): pad the feature map with 0-s to a square shape (used for RetinaNet).
        circular (bool): whether to use circular padding.
    Returns:
        Backbone: a backbone with a custom FPN.
    """

    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        square_pad=0,
        circular=True,
    ):

        super().__init__(
            bottom_up, in_features, out_channels, norm, top_block, fuse_type, square_pad
        )

        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type
        self.circular = circular

        # Get the res out_features from backbone
        self.res_out_features = self.bottom_up._out_features
        res_out_features_strides = self.bottom_up._out_feature_strides
        res_out_feature_channels = self.bottom_up._out_feature_channels

        # change the in_feature adaptively accroding to the current strides in the backbone
        bottom_fpn_key_out = max(self._out_feature_strides, key=lambda k: self._out_feature_strides[k])
        bottom_fpn_key_in = f'p{int(bottom_fpn_key_out[-1]) - self.top_block.num_levels}'
        self.top_block.in_feature = bottom_fpn_key_in
        
        # Add to FPN out_features
        self._out_features.extend(self.res_out_features)
        self._out_feature_channels.update(res_out_feature_channels)
        self._out_feature_strides.update(res_out_features_strides)

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        if isinstance(x, dict): # if we run the backbone already
            bottom_up_features = x
        else:
            bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        tmp = pad(prev_features, padding=1, circular=self.circular)
        results.append(self.output_convs[0](tmp))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                tmp = pad(prev_features, padding=1, circular=self.circular)
                results.insert(0, output_conv(tmp))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) - len(self.res_out_features) == len(results)

        # Update custom output
        out = {f: res for f, res in zip(self._out_features, results)}
        out.update(bottom_up_features)
        return out
