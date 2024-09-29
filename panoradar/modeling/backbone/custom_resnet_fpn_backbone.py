from detectron2.modeling import BACKBONE_REGISTRY
from .custom_resnet import build_custom_resnet_deeplab_backbone
from .custom_fpn import CustomFPN, Last2LevelMaxPool

@BACKBONE_REGISTRY.register()
def build_resnet_deeplab_fpn_backbone(cfg, input_shape):
    """
    Create a resnet backbone from config with deeplab STEM and additional FPN
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module
    """
    bottom_up = build_custom_resnet_deeplab_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = CustomFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=Last2LevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        circular=cfg.MODEL.CIRCULAR_SEG_OBJ,
    )
    return backbone
