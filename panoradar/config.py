from detectron2.config import CfgNode
from detectron2.projects.deeplab.config import add_deeplab_config

def add_depth_sn_config(cfg):
    """
    Add config for depth and surface normal estimation model.
    Args:
        cfg: a detectron2 CfgNode instance.
    """
    cfg.MODEL.DPETH_SN_HEAD = CfgNode()
    cfg.MODEL.DPETH_SN_HEAD.LOSS_WEIGHT_DEPTH = 1.0
    cfg.MODEL.DPETH_SN_HEAD.LOSS_WEIGHT_SN = 0.1
    cfg.MODEL.DPETH_SN_HEAD.LOSS_WEIGHT_PERCEP = 0.1
    cfg.MODEL.CIRCULAR_DEPTH = True
    
def add_seg_obj_config(cfg):
    """
    Add config for segmentation and object detection model.
    Args:
        cfg: a detectron2 CfgNode instance.
    """
    cfg.MODEL.CIRCULAR_SEG_OBJ = False
    cfg.MODEL.TWO_STAGE = True

def add_custom_deeplab_config(cfg):
    add_deeplab_config(cfg)
    cfg.MODEL.RESNETS.STEM_STRIDE = 2
    cfg.MODEL.RESNETS.STEM_MAXPOOL = True
    cfg.MODEL.SEM_SEG_HEAD.TOP_K_PERCENT_PIXELS = 0.2
    cfg.MODEL.SEM_SEG_HEAD.LABEL_SMOOTHING = 0.1

def get_panoradar_cfg() -> CfgNode:
    """
    Get a copy of the default config, and additional config for PanoRadar.
    Returns:
        a detectron2 CfgNode instance.
    """
    from detectron2.config.defaults import _C
    
    # ================ Data ======================
    _C.DATASETS.BASE_PATH = "./data"
    
    # ================ Solver ====================
    _C.SOLVER.NAME = "SGD"
    _C.SOLVER.VALUES = (0.1, 0.01)

    # =============== Augmentation ===============
    _C.INPUT.ROTATE = CfgNode()
    _C.INPUT.ROTATE.ENABLED = True
    _C.INPUT.ROTATE.ROTATE_P = 1.0
    _C.INPUT.ROTATE.HFLIP_P = 0.5
    _C.INPUT.CROP_AND_RESIZE = CfgNode()
    _C.INPUT.CROP_AND_RESIZE.ENABLED = True
    _C.INPUT.CROP_AND_RESIZE.CROP_LENGTH = (8, 16)  # (half height, half width)
    _C.INPUT.CROP_AND_RESIZE.DROP_BOX_THRES = (5, 8)  # (height, width)
    _C.INPUT.CROP_AND_RESIZE.CROP_AND_RESIZE_P = 0.5
    _C.INPUT.SCALE_TRANSFORM = CfgNode()
    _C.INPUT.SCALE_TRANSFORM.ENABLED = True
    _C.INPUT.SCALE_TRANSFORM.SCALE_RANGE = (0.8, 1.2)
    _C.INPUT.SCALE_TRANSFORM.SCALE_P = 0.5
    _C.INPUT.JITTER = CfgNode()
    _C.INPUT.JITTER.ENABLED = True
    _C.INPUT.JITTER.MEAN = 0.0
    _C.INPUT.JITTER.STD = 0.003
    _C.INPUT.JITTER.JITTER_P = 0.5
    _C.INPUT.FIRST_REFL = CfgNode()
    _C.INPUT.FIRST_REFL.ENABLED = True
    _C.INPUT.FIRST_REFL.JITTER_P = 0.5

    # ===== Depth/Surface Normal Estimation =======
    _C.MODEL.BACKBONE.NUM_BLOCKS_PER_DOWN = (2, 2, 2, 2)
    _C.MODEL.BACKBONE.DIM_MULTS = (1, 2, 4, 8)
    _C.MODEL.BACKBONE.STEM_OUT_CHANNELS = 64
    add_depth_sn_config(_C)
    add_seg_obj_config(_C)
    add_custom_deeplab_config(_C)

    return _C.clone()
