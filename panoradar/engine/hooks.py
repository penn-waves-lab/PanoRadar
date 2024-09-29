import torch

from torch.utils.tensorboard import SummaryWriter

from detectron2.data import MetadataCatalog
from detectron2.engine import HookBase

from ..utils import draw_vis_image

class ImageVisualizationHook(HookBase):
    def __init__(self, cfg):
        """
        The hook for visualizing validation set images and log them to tensorboard.
        Args:
            cfg: the config object
        """
        super().__init__()
        self.cfg = cfg
        self.period = cfg.VIS_PERIOD
        self.writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)  # create another tensorboard file for images logging
        self.dataset_name = cfg.DATASETS.TEST[0]
        self.dataset_metadata = MetadataCatalog.get(self.dataset_name)
        self.vis_indices = self.dataset_metadata.vis_ind
        self.depth_only = True if cfg.MODEL.META_ARCHITECTURE == 'DepthSnModel' else False

    def after_step(self):
        if (self.trainer.iter + 1) % self.period != 0: # only visualize every period
            return

        dataloader = self.trainer.build_test_loader(self.cfg, self.dataset_name)
        self.trainer.model.eval()

        with torch.no_grad():
            vis_title = 0  # image title in the tensorboard

            for idx, input_dict in enumerate(dataloader):
                if idx not in self.vis_indices: # only visualize the selected images
                    continue

                # get model prediction, draw image, and log it to the tensorboard
                preds = self.trainer.model(input_dict)
                whole_img = draw_vis_image(
                    input_dict[0],
                    preds[0],
                    dataset_metadata=self.dataset_metadata,
                    depth_only=self.depth_only,
                    return_rgb=True,
                )
                self.writer.add_image(
                    f"val image {vis_title}", whole_img, global_step=self.trainer.iter
                )
                vis_title += 1

        self.trainer.model.train()
