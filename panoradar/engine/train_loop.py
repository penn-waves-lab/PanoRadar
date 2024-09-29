import os
import torch
import logging
import numpy as np

from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import SemSegEvaluator
from detectron2.solver.lr_scheduler import LRMultiplier, WarmupParamScheduler

from fvcore.common.param_scheduler import MultiStepParamScheduler

from ..data import LidarTwoTasksMapper, RfFourTasksMapper
from .hooks import ImageVisualizationHook
from ..evaluation import (
    DepthEvaluator,
    SnEvaluator,
    SemSegEvaluator,
    CircularObjEvaluator,
)

__all__ = ["LidarTwoTasksTrainer", "RfFourTasksTrainer", "RfDepthSnTrainer"]

class LidarTwoTasksTrainer(DefaultTrainer):
    """The customized trainer for our training task."""

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        """Take in the config file and return a torch Dataloader for training"""
        mapper = LidarTwoTasksMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Take in the config file and return a torch Dataloader for testing"""
        mapper = LidarTwoTasksMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build evaluator(s) for metrics"""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)

        ret = [
            SemSegEvaluator(
                dataset_name,
                output_dir=output_folder,
                sem_seg_loading_fn=lambda name, dtype: np.load(name)[0].astype(dtype),
            ),
            CircularObjEvaluator(dataset_name, output_dir=output_folder)
        ]
        return ret

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Returns a torch.optim.Optimizer"""
        if cfg.SOLVER.NAME == 'SGD':
            args = {
                "params": model.parameters(),
                "lr": cfg.SOLVER.BASE_LR,
                "momentum": cfg.SOLVER.MOMENTUM,
                "nesterov": cfg.SOLVER.NESTEROV,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            }
            optim_cls = torch.optim.SGD
        elif cfg.SOLVER.NAME == 'AdamW':
            args = {"params": model.parameters(), "lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.00001}
            optim_cls = torch.optim.AdamW
        else:
            raise NameError(f'Unrecognize solver name: {cfg.SOLVER.NAME}')
        return optim_cls(**args)

class RfFourTasksTrainer(DefaultTrainer):
    """The customized trainer for our training task."""

    def __init__(self, cfg):
        cfg.MODEL.PIXEL_MEAN = [0 for _ in range(256)]
        cfg.MODEL.PIXEL_STD = [1 for _ in range(256)]
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        """Take in the config file and return a torch Dataloader for training"""
        mapper = RfFourTasksMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Take in the config file and return a torch Dataloader for testing"""
        mapper = RfFourTasksMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build evaluator(s) for metrics"""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)

        ret = [
            DepthEvaluator(output_folder),
            SnEvaluator(output_folder),
            CircularObjEvaluator(dataset_name, output_dir=output_folder),
            SemSegEvaluator(
                dataset_name,
                output_dir=output_folder,
                sem_seg_loading_fn=lambda name, dtype: np.load(name)[0].astype(dtype),
            ),
        ]
        return ret

    def build_hooks(self):
        """Overwrite this function so that new hooks can be added.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        ret = super().build_hooks()
        if self.cfg.VIS_PERIOD > 0:
            ret.append(ImageVisualizationHook(self.cfg))
        return ret

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Returns a torch.optim.Optimizer"""
        if cfg.SOLVER.NAME == 'SGD':
            args = {
                "params": model.parameters(),
                "lr": cfg.SOLVER.BASE_LR,
                "momentum": cfg.SOLVER.MOMENTUM,
                "nesterov": cfg.SOLVER.NESTEROV,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            }
            optim_cls = torch.optim.SGD
        elif cfg.SOLVER.NAME == 'AdamW':
            args = {"params": model.parameters(), "lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.00001}
            optim_cls = torch.optim.AdamW
        else:
            raise NameError(f'Unrecognize solver name: {cfg.SOLVER.NAME}')
        return optim_cls(**args)

class RfDepthSnTrainer(DefaultTrainer):
    """The customized trainer for training depth and surface normal."""

    def __init__(self, cfg):
        cfg.MODEL.PIXEL_MEAN = [0 for _ in range(256)]
        cfg.MODEL.PIXEL_STD = [1 for _ in range(256)]
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        """Take in the config file and return a torch Dataloader for training"""
        mapper = RfFourTasksMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Take in the config file and return a torch Dataloader for testing"""
        mapper = RfFourTasksMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build evaluator(s) for metrics"""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)

        ret = [
            DepthEvaluator(output_folder),
            SnEvaluator(output_folder),
        ]
        return ret

    def build_hooks(self):
        """Overwrite this function so that new hooks can be added.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        ret = super().build_hooks()
        if self.cfg.VIS_PERIOD > 0:
            ret.append(ImageVisualizationHook(self.cfg))
        return ret

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Returns a torch.optim.Optimizer"""
        if cfg.SOLVER.NAME == 'SGD':
            args = {
                "params": model.parameters(),
                "lr": cfg.SOLVER.BASE_LR,
                "momentum": cfg.SOLVER.MOMENTUM,
                "nesterov": cfg.SOLVER.NESTEROV,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            }
            optim_cls = torch.optim.SGD
        elif cfg.SOLVER.NAME == 'AdamW':
            args = {"params": model.parameters(), "lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.00001}
            optim_cls = torch.optim.AdamW
        else:
            raise NameError(f'Unrecognize solver name: {cfg.SOLVER.NAME}')
        return optim_cls(**args)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """Returns a torch.optim.lr_scheduler.LambdaLR"""
        name = cfg.SOLVER.LR_SCHEDULER_NAME

        if name == "WarmupMultiStepLR":
            steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
            values = [x for x in cfg.SOLVER.VALUES]
            if len(steps) != len(cfg.SOLVER.STEPS):
                logger = logging.getLogger(__name__)
                logger.warning(
                    "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                    "These values will be ignored."
                )
            sched = MultiStepParamScheduler(
                values=[1] + values,
                milestones=steps,
                num_updates=cfg.SOLVER.MAX_ITER,
            )
        else:
            raise ValueError("Unknown LR scheduler: {}".format(name))

        sched = WarmupParamScheduler(
            sched,
            cfg.SOLVER.WARMUP_FACTOR,
            min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
            cfg.SOLVER.WARMUP_METHOD,
            cfg.SOLVER.RESCALE_INTERVAL,
        )
        return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)

def get_trainer_class(cfg):
    """
    Get the trainer class according to the dataset.
    Args:
        cfg (CfgNode): the config node.
    Returns:
       Trainer: a trainer instance of DefaultTrainer class. 
    """
    
    arch = cfg.MODEL.META_ARCHITECTURE
    if cfg.MODEL.CIRCULAR_SEG_OBJ:
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CircularRPN"
        cfg.MODEL.RPN.HEAD_NAME = "CircularRPNHead"
        cfg.MODEL.ANCHOR_GENERATOR.NAME = "CircularAnchorGenerator"
        cfg.MODEL.ROI_HEADS.NAME = "CircularROIHeads"

    if 'GeneralizedRCNNSemanticSegmentor' == arch:
        return LidarTwoTasksTrainer
    elif 'TwoStageModel' == arch:
        return RfFourTasksTrainer
    elif 'DepthSnModel' == arch:
        return RfDepthSnTrainer
    else:
        raise NameError(f'Unrecognized META_ARCHITECTURE: {arch}')
