import pyiqa
import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec

from functools import partial

from ..backbone.custom_unet_fpn_backbone import BasicBlock, pad_azimuth, pad

__all__ = ["DepthSnHead"]

def Upsample(dim):
    """
    Upsample the spatial dimensions by a factor of 2.
    Args:
        dim (int): number of input and output channels.
    Returns:
        nn.Module: the upsampling layer.
    """
    return nn.ConvTranspose2d(dim, dim, 4, 2, padding=3)

def UpsampleAzimuth(dim):
    """
    Upsample the spatial dimensions by a factor of 2 along azimuth (width).
    Args:
        dim (int): number of input and output channels.
    Returns:
        nn.Module: the upsampling layer.
    """
    return nn.ConvTranspose2d(dim, dim, (1, 4), (1, 2), padding=(0, 3))

class DepthSnHead(nn.Module):
    """
    A depth and surface normal head with a unet style architecture for CustomUnetBackbone.
    """
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.num_down_elev = 3  # after this layer only down azimuth
        dim = cfg.MODEL.BACKBONE.STEM_OUT_CHANNELS  # stem output channel
        dim_copy = 64
        dim_mults = cfg.MODEL.BACKBONE.DIM_MULTS
        resnet_groups = 8

        self.final_res = len(dim_mults) + 1
        self.loss_weight_depth = cfg.MODEL.DPETH_SN_HEAD.LOSS_WEIGHT_DEPTH
        self.loss_weight_sn = cfg.MODEL.DPETH_SN_HEAD.LOSS_WEIGHT_SN
        self.loss_weight_percep = cfg.MODEL.DPETH_SN_HEAD.LOSS_WEIGHT_PERCEP
        self.circular = cfg.MODEL.CIRCULAR_DEPTH

        # dimensions
        dims = [dim, *map(lambda m: dim_copy * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        block_klass = partial(BasicBlock, groups=resnet_groups, circular=self.circular)
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            azimuth_only = ind < (num_resolutions - self.num_down_elev - 1) and not is_last
            if is_last:
                upsample = nn.Identity()
            elif azimuth_only:
                upsample = UpsampleAzimuth(dim_in)
            else:
                upsample = Upsample(dim_in)
            self.ups.append(
                nn.ModuleList(
                    [block_klass(dim_out * 2, dim_in), block_klass(dim_in, dim_in), upsample]
                )
            )

        # final conv
        self.final_conv_depth = nn.Sequential(block_klass(dim * 2, dim), nn.Conv2d(dim, 1, 1))
        self.final_conv_sn = nn.Sequential(block_klass(dim * 2, dim), nn.Conv2d(dim, 3, 1))

        # loss
        self.l1_loss_func = nn.SmoothL1Loss(reduction="none", beta=0.005)
        self.percep_loss_func = [
            pyiqa.create_metric(
                'lpips',
                device=torch.device('cuda'), # TODO: make this adaptive
                as_loss=True,
                net='vgg',
                eval_mode=True,
                pnet_tune=False,
            )
        ]  # do not register into model

        self._out_features = [
            'pred_depth',
            'pred_sn',
            'depth_stem',
            'depth_res2',
            'depth_res3',
            'depth_res4',
            'depth_res5',
        ]
        self._out_feature_strides = {
            'pred_depth': 1,
            'pred_sn': 1,
            'depth_stem': 1,
            'depth_res2': 1,
            'depth_res3': 2,
            'depth_res4': 4,
            'depth_res5': 8,
        }
        self._out_feature_channels = {
            'pred_depth': 1,
            'pred_sn': 3,
            'depth_stem': dim_copy * 4,
            'depth_res2': dim_copy * 4,
            'depth_res3': dim_copy * 8,
            'depth_res4': dim_copy * 16,
            'depth_res5': dim_copy * 32,
        }

    def forward(self, features, targets=None, return_features=False):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (1xHxW depth predictions, {})
        """
        y, features = self.layers(features)

        if self.training:
            if return_features:
                return features, self.losses(y, targets)
            else:
                return y, self.losses(y, targets)
        else:
            return y, features

    def layers(self, features):
        """
        Args:
            features (dict[str->Tensor]): input features
        Returns:
            tuple[tuple[Tensor, Tensor], dict[str->Tensor]]: ((depth, sn), features)
        """
        x = features[f'res{self.final_res}']
        h = [features['stem']]
        h.extend([features[f'res{i}'] for i in range(2, self.final_res + 1)])

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for i, (block1, block2, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            features[f"depth_res{self.final_res-i}"] = x
            x = block1(x)
            x = block2(x)
            is_last = (i + 1) >= len(self.ups)
            azimuth_only = i < (len(self.ups) - self.num_down_elev - 1) and not is_last
            if azimuth_only:
                x = pad_azimuth(x, padding=1, circular=self.circular)
            elif not is_last:
                x = pad(x, padding=1, circular=self.circular)
            x = upsample(x)

        x = torch.cat((x, h.pop()), dim=1)
        features["depth_stem"] = x
        ret = (self.final_conv_depth(x), self.final_conv_sn(x))
        features["pred_depth"] = ret[0]
        features["pred_sn"] = ret[1]

        return ret, features

    def losses(self, predictions, targets):
        """
        Args:
            predictions (tuple): (pred_depth, pred_sn)
            targets (tuple): (target_depth, target_sn)
        Returns:
            dict[str: Tensor]: dict of losses
        """
        target_depth, target_sn = targets
        pred_depth, pred_sn = predictions

        mask_depth = target_depth > 0
        mask_sn = target_sn > -10

        loss_depth = (
            self.l1_loss_func(pred_depth, target_depth) * mask_depth
        ).sum() / mask_depth.sum()
        loss_sn = (self.l1_loss_func(pred_sn, target_sn) * mask_sn).sum() / mask_sn.sum()
        loss_percep = self.percep_loss_func[0](pred_depth * mask_depth, target_depth * mask_depth)

        losses = {
            "loss_depth": loss_depth * self.loss_weight_depth,
            "loss_sn": loss_sn * self.loss_weight_sn,
            "loss_percep": loss_percep * self.loss_weight_percep,
        }
        return losses

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

def build_depth_sn_head(cfg, input_shape):
    """
    Build a depth head from cfg.MODEL.DEPTH_HEAD.NAME
    """
    return DepthSnHead(cfg, input_shape)
