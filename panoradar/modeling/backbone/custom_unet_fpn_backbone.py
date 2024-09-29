import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone

from functools import partial

from .custom_fpn import CustomFPN, Last2LevelMaxPool

def Downsample(dim):
    """
    Downsample the feature map by 2x.
    Args:
        dim (int): number of channels in the input and output feature map.
    Returns:
        nn.Module: the downsample layer.
    """
    return nn.Conv2d(dim, dim, 4, 2, padding=0)

def DownsampleAzimuth(dim):
    """
    Downsample the feature map by 2x along azimuth (width).
    Args:
        dim (int): number of channels in the input and output feature map.
    Returns:
        nn.Module: the downsample layer.
    """
    return nn.Conv2d(dim, dim, (1, 4), (1, 2), padding=0)

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

def pad_azimuth(x, padding, circular=True):
    """
    Pad the input tensor along azimuth (width).
    Args:
        x (torch.Tensor): input tensor.
        padding (int): padding size.
        circular (bool): whether to use circular padding.
    Returns:
        torch.Tensor: padded tensor.
    """
    if circular:
        return F.pad(x, (padding, padding, 0, 0), 'circular')
    else:
        return F.pad(x, (padding, padding, 0, 0), 'constant')

class ConvBlock(nn.Module):
    """
    A basic convolutional block with GroupNorm and SiLU activation.
    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel (int): kernel size.
        padding (int): padding size.
        groups (int): number of groups for GroupNorm.
        circular (bool): whether to use circular padding.
    Returns:
        nn.Module: the convolutional block.
    """
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, groups=8, circular=True):
        super().__init__()
        self.padding = padding
        self.circular = circular
        self.proj = nn.Conv2d(in_channels, out_channels, kernel, padding=0)
        self.norm = nn.GroupNorm(groups, out_channels, eps=1e-4)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = pad(x, padding=self.padding, circular=self.circular)
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)

class BasicBlock(nn.Module):
    """
    A basic residual block with basic convolutional blocks.
    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        groups (int): number of groups for GroupNorm.
        circular (bool): whether to use circular padding.
    Returns:
        nn.Module: the residual block.
    """
    def __init__(self, in_channels, out_channels, groups=8, circular=True):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels, groups=groups, circular=circular)
        self.block2 = ConvBlock(out_channels, out_channels, groups=groups, circular=circular)
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class CustomUnetBackbone(Backbone):
    """
    This is the backbone for our unet model.

    1. It uses basic blocks instead of bottleneck block (but ativation first then addition).
    2. Conv2d (k=4x4, stride=2, padding=0) for downsample.
    3. Downsampling happens at the end of each layer.
    4. It has stem layer with large kernel size (7x7).
    5. It only downsample along with azimuth (width) after the 3rd layer.
    6. Group Norm, SiLU, learnable bias.
    Args:
        cfg (CfgNode): configs.
        input_shape (ShapeSpec): input shape.
    Returns:
        Backbone: the backbone module.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        self.num_down_elev = 3  # after this layer only down azimuth
        dim = cfg.MODEL.BACKBONE.STEM_OUT_CHANNELS  # stem output channel
        dim_copy = 64
        dim_mults = cfg.MODEL.BACKBONE.DIM_MULTS
        num_blocks_per_down = cfg.MODEL.BACKBONE.NUM_BLOCKS_PER_DOWN
        resnet_groups = 8
        init_kernel_size = 7
        self.circular = cfg.MODEL.CIRCULAR_DEPTH

        assert (init_kernel_size % 2) == 1
        self.init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv2d(input_shape.channels, dim, init_kernel_size, padding=0)

        # dimensions
        dims = [dim, *map(lambda m: dim_copy * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(BasicBlock, groups=resnet_groups, circular=self.circular)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            azimuth_only = ind >= self.num_down_elev and not is_last
            if is_last:
                downsample = nn.Identity()
            elif azimuth_only:
                downsample = DownsampleAzimuth(dim_out)
            else:
                downsample = Downsample(dim_out)

            num_blocks = nn.ModuleList([])
            num_blocks.append(block_klass(dim_in, dim_out))
            for _ in range(num_blocks_per_down[ind] - 1):
                num_blocks.append(block_klass(dim_out, dim_out))
            num_blocks.append(downsample)
            self.downs.append(num_blocks)

        # some variables that FPN might need
        self._out_features = ['stem', 'res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {'stem': 1, 'res2': 1, 'res3': 2, 'res4': 4, 'res5': 8}
        self._out_feature_channels = {'stem': dim, 'res2': dim_copy, 'res3': dim_copy * 2, 'res4': dim_copy * 4, 'res5': dim_copy * 8}

    def forward(self, x):
        x = pad(x, padding=self.init_padding, circular=self.circular)
        x = self.init_conv(x)
        outputs = {'stem': x}

        for i, down_blocks in enumerate(self.downs):
            for block in down_blocks[:-1]:
                x = block(x)
            outputs[f'res{i+2}'] = x
            is_last = (i + 1) >= len(self.downs)
            azimuth_only = i >= self.num_down_elev and not is_last
            if azimuth_only:
                x = pad_azimuth(x, padding=1, circular=self.circular)
            elif not is_last:
                x = pad(x, padding=1, circular=self.circular)
            x = down_blocks[-1](x)

        return outputs

@BACKBONE_REGISTRY.register()
def custom_unet_fpn_backbone(cfg, input_shape):
    """
    Create a resnet backbone for unet with additional FPN.
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module
    """
    bottom_up = CustomUnetBackbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = CustomFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=Last2LevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def custom_unet_backbone(cfg, input_shape):
    """
    Create a resnet backbone from config with deeplab STEM and additional FPN
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module
    """
    backbone = CustomUnetBackbone(cfg, input_shape)
    return backbone
