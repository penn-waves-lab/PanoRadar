"""
This resnet is the same as https://github.com/facebookresearch/detectron2/blob/main/projects/DeepLab/deeplab/semantic_seg.py
except:
1. All convolutions use circular padding.
"""

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.layers import ASPP, Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from .loss import DeepLabCE

from ..backbone.custom_unet_fpn_backbone import pad

__all__ = ["CustomDeepLabV3PlusHead"]

class CircularASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        *,
        norm,
        activation,
        pool_kernel_size=None,
        dropout: float = 0.0,
        use_depthwise_separable_conv=False,
        circular: bool = True,
    ):
        """
        Args:
            in_channels (int): number of input channels for ASPP.
            out_channels (int): number of output channels.
            dilations (list): a list of 3 dilations in ASPP.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format. norm is
                applied to all conv layers except the conv following
                global average pooling.
            activation (callable): activation function.
            pool_kernel_size (tuple, list): the average pooling size (kh, kw)
                for image pooling layer in ASPP. If set to None, it always
                performs global average pooling. If not None, it must be
                divisible by the shape of inputs in forward(). It is recommended
                to use a fixed input feature size in training, and set this
                option to match this size, so that it performs global average
                pooling in training, and the size of the pooling window stays
                consistent in inference.
            dropout (float): apply dropout on the output of ASPP. It is used in
                the official DeepLab implementation with a rate of 0.1:
                https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/model.py#L532  # noqa
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                for 3x3 convs in ASPP, proposed in :paper:`DeepLabV3+`.
        """
        super(CircularASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        self.circular = circular
        self.dilations = []
        use_bias = norm == ""
        self.convs = nn.ModuleList()
        # conv 1x1
        self.convs.append(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, out_channels),
                activation=deepcopy(activation),
            )
        )
        weight_init.c2_xavier_fill(self.convs[-1])
        # atrous convs
        for dilation in dilations:
            if use_depthwise_separable_conv:
                self.convs.append(
                    DepthwiseSeparableConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        norm1=norm,
                        activation1=deepcopy(activation),
                        norm2=norm,
                        activation2=deepcopy(activation),
                    )
                )
            else:
                self.convs.append(
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=0,
                        dilation=dilation,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                        activation=deepcopy(activation),
                    )
                )
                self.dilations.append(dilation)
                weight_init.c2_xavier_fill(self.convs[-1])
        # image pooling
        # We do not add BatchNorm because the spatial resolution is 1x1,
        # the original TF implementation has BatchNorm.
        if pool_kernel_size is None:
            image_pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
            )
        else:
            image_pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
            )
        weight_init.c2_xavier_fill(image_pooling[1])
        self.convs.append(image_pooling)

        self.project = Conv2d(
            5 * out_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels),
            activation=deepcopy(activation),
        )
        weight_init.c2_xavier_fill(self.project)

    def forward(self, x):
        size = x.shape[-2:]
        if self.pool_kernel_size is not None:
            if size[0] % self.pool_kernel_size[0] or size[1] % self.pool_kernel_size[1]:
                raise ValueError(
                    "`pool_kernel_size` must be divisible by the shape of inputs. "
                    "Input size: {} `pool_kernel_size`: {}".format(size, self.pool_kernel_size)
                )
        res = []
        idx = 0
        for conv in self.convs:
            if idx == 0 or idx == len(self.convs) - 1:
                res.append(conv(x))
            else:
                out = pad(x, padding=self.dilations[idx - 1], circular=self.circular)
                res.append(conv(out))
            idx += 1
        res[-1] = F.interpolate(res[-1], size=size, mode="bilinear", align_corners=False)
        res = torch.cat(res, dim=1)
        res = self.project(res)
        res = F.dropout(res, self.dropout, training=self.training) if self.dropout > 0 else res
        return res

@SEM_SEG_HEADS_REGISTRY.register()
class CustomDeepLabV3PlusHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3+`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        norm: Union[str, Callable],
        train_size: Optional[Tuple],
        loss_weight: float = 1.0,
        loss_type: str = "cross_entropy",
        ignore_value: int = -1,
        num_classes: Optional[int] = None,
        use_depthwise_separable_conv: bool = False,
        top_k_percent_pixels: float = 0.2,
        label_smoothing: float = 0.0,
        circular: bool = True,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shape of the input features. They will be ordered by stride
                and the last one (with largest stride) is used as the input to the
                decoder (i.e.  the ASPP module); the rest are low-level feature for
                the intermediate levels of decoder.
            project_channels (list[int]): a list of low-level feature channels.
                The length should be len(in_features) - 1.
            aspp_dilations (list(int)): a list of 3 dilations in ASPP.
            aspp_dropout (float): apply dropout on the output of ASPP.
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            common_stride (int): output stride of decoder.
            norm (str or callable): normalization for all conv layers.
            train_size (tuple): (height, width) of training images.
            loss_weight (float): loss weight.
            loss_type (str): type of loss function, 2 opptions:
                (1) "cross_entropy" is the standard cross entropy loss.
                (2) "hard_pixel_mining" is the loss in DeepLab that samples
                    top k% hardest pixels.
            ignore_value (int): category to be ignored during training.
            num_classes (int): number of classes, if set to None, the decoder
                will not construct a predictor.
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                in ASPP and decoder.
            top_k_percent_pixels (float): percentage of top pixels to keep.
            label_smoothing (float): label smoothing factor in cross entropy.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        # fmt: off
        self.in_features      = [k for k, v in input_shape]  # starting from "res2" to "res5"
        in_channels           = [x[1].channels for x in input_shape]
        in_strides            = [x[1].stride for x in input_shape]
        aspp_channels         = decoder_channels[-1]
        self.ignore_value     = ignore_value
        self.common_stride    = common_stride  # output stride
        self.loss_weight      = loss_weight
        self.loss_type        = loss_type
        self.decoder_only     = num_classes is None
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        self.top_k_percent_pixels = top_k_percent_pixels
        self.label_smoothing  = label_smoothing
        self.circular         = circular
        # fmt: on

        assert (
            len(project_channels) == len(self.in_features) - 1
        ), "Expected {} project_channels, got {}".format(
            len(self.in_features) - 1, len(project_channels)
        )
        assert len(decoder_channels) == len(
            self.in_features
        ), "Expected {} decoder_channels, got {}".format(
            len(self.in_features), len(decoder_channels)
        )
        self.decoder = nn.ModuleDict()

        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            decoder_stage = nn.ModuleDict()

            if idx == len(self.in_features) - 1:
                # ASPP module
                if train_size is not None:
                    train_h, train_w = train_size
                    encoder_stride = in_strides[-1]
                    if train_h % encoder_stride or train_w % encoder_stride:
                        raise ValueError("Crop size need to be divisible by encoder stride.")
                    pool_h = train_h // encoder_stride
                    pool_w = train_w // encoder_stride
                    pool_kernel_size = (pool_h, pool_w)
                else:
                    pool_kernel_size = None
                project_conv = CircularASPP(
                    in_channel,
                    aspp_channels,
                    aspp_dilations,
                    norm=norm,
                    activation=F.relu,
                    pool_kernel_size=pool_kernel_size,
                    dropout=aspp_dropout,
                    use_depthwise_separable_conv=use_depthwise_separable_conv,
                    circular=self.circular,
                )
                fuse_conv = None
            else:
                project_conv = Conv2d(
                    in_channel,
                    project_channels[idx],
                    kernel_size=1,
                    bias=use_bias,
                    norm=get_norm(norm, project_channels[idx]),
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(project_conv)
                if use_depthwise_separable_conv:
                    # We use a single 5x5 DepthwiseSeparableConv2d to replace
                    # 2 3x3 Conv2d since they have the same receptive field,
                    # proposed in :paper:`Panoptic-DeepLab`.
                    fuse_conv = DepthwiseSeparableConv2d(
                        project_channels[idx] + decoder_channels[idx + 1],
                        decoder_channels[idx],
                        kernel_size=5,
                        padding=2,
                        norm1=norm,
                        activation1=F.relu,
                        norm2=norm,
                        activation2=F.relu,
                    )
                else:
                    fuse_conv = nn.Sequential(
                        Conv2d(
                            project_channels[idx] + decoder_channels[idx + 1],
                            decoder_channels[idx],
                            kernel_size=3,
                            padding=0,
                            bias=use_bias,
                            norm=get_norm(norm, decoder_channels[idx]),
                            activation=F.relu,
                        ),
                        Conv2d(
                            decoder_channels[idx],
                            decoder_channels[idx],
                            kernel_size=3,
                            padding=0,
                            bias=use_bias,
                            norm=get_norm(norm, decoder_channels[idx]),
                            activation=F.relu,
                        ),
                    )
                    weight_init.c2_xavier_fill(fuse_conv[0])
                    weight_init.c2_xavier_fill(fuse_conv[1])

            decoder_stage["project_conv"] = project_conv
            decoder_stage["fuse_conv"] = fuse_conv

            self.decoder[self.in_features[idx]] = decoder_stage

        if not self.decoder_only:
            self.predictor = Conv2d(
                decoder_channels[0], num_classes, kernel_size=1, stride=1, padding=0
            )
            nn.init.normal_(self.predictor.weight, 0, 0.001)
            nn.init.constant_(self.predictor.bias, 0)

            if self.loss_type == "cross_entropy":
                self.loss = nn.CrossEntropyLoss(
                    reduction="mean",
                    ignore_index=self.ignore_value,
                    label_smoothing=label_smoothing,
                )
            elif self.loss_type == "hard_pixel_mining":
                self.loss = DeepLabCE(
                    ignore_label=self.ignore_value,
                    top_k_percent_pixels=top_k_percent_pixels,
                    label_smoothing=label_smoothing,
                )
            else:
                raise ValueError("Unexpected loss type: %s" % self.loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            # common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            common_stride = 2 * cfg.MODEL.RESNETS.STEM_STRIDE if cfg.MODEL.RESNETS.STEM_MAXPOOL else cfg.MODEL.RESNETS.STEM_STRIDE,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            train_size=train_size,
            loss_weight=cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            loss_type=cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
            top_k_percent_pixels=cfg.MODEL.SEM_SEG_HEAD.TOP_K_PERCENT_PIXELS,
            label_smoothing=cfg.MODEL.SEM_SEG_HEAD.LABEL_SMOOTHING,
            circular=cfg.MODEL.CIRCULAR_SEG_OBJ,
        )
        return ret

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.decoder_only:
            # Output from self.layers() only contains decoder feature.
            return y
        if self.training:
            return None, self.losses(y, targets)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for f in self.in_features[::-1]:
            x = features[f]
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = pad(y, padding=1, circular=self.circular)
                y = self.decoder[f]["fuse_conv"][0](y)
                y = pad(y, padding=1, circular=self.circular)
                y = self.decoder[f]["fuse_conv"][1](y)
        if not self.decoder_only:
            y = self.predictor(y)
        return y

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
