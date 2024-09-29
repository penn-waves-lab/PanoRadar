"""
Custom Evaluators
1. DepthEvaluator: evaluate depth l1 metrics
2. SnEvaluator: evaluate surface normal l1 and angle metrics
3. AdvDepthEvaluator: evaluate sub-distance depth metrics.
4. AdvSnEvaluator: evaluate sub-distance surface normal metrics.
"""

import os
import logging
import numpy as np

from collections import OrderedDict

import torch
import torch.nn.functional as F

from detectron2.evaluation import DatasetEvaluator

from .iqa_evaluation import masked_psnr, masked_ssim

__all__ = ["DepthEvaluator", "SnEvaluator", "AdvDepthEvaluator", "AdvSnEvaluator"]


class DepthEvaluator(DatasetEvaluator):
    """Evaluate depth l1 metrics."""

    def __init__(self, output_dir=None):
        """
        Args:
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir

        self._l1_mean = []
        self._l1_median = []
        self._l1_80 = []
        self._l1_90 = []
        self._psnr = []
        self._ssim = []

    def reset(self):
        self._l1_mean.clear()
        self._l1_median.clear()
        self._l1_80.clear()
        self._l1_90.clear()
        self._psnr.clear()
        self._ssim.clear()

    def process(self, inputs, outputs):
        """Process the pair of inputs and outputs.
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input, output in zip(inputs, outputs):
            gt_depth = input['depth'].clone().unsqueeze(0)  # (1, 1, 64, 512)
            pred_depth = output['depth'].detach().cpu().clone().unsqueeze(0)  # (1, 1, 64, 512)
            mask_valid = gt_depth > 0
            mask_inr = mask_valid & (gt_depth < 0.96)

            l1_loss = F.l1_loss(pred_depth, gt_depth, reduction='none')[mask_inr]
            self._l1_mean.append(l1_loss.sum() / mask_inr.sum())
            self._l1_median.append(l1_loss.quantile(0.5))
            self._l1_80.append(l1_loss.quantile(0.8))
            self._l1_90.append(l1_loss.quantile(0.9))
            self._psnr.append(masked_psnr(pred_depth, gt_depth, mask_inr, data_range=1.0))
            self._ssim.append(masked_ssim(pred_depth, gt_depth, mask_inr, data_range=1.0))

    def evaluate(self):
        """
        Returns:
            dict: A dict of {metric name: score} pairs.
        """
        res = {
            'depth_l1_mean': np.mean(self._l1_mean),
            'depth_l1_median': np.mean(self._l1_median),
            'depth_l1_80': np.mean(self._l1_80),
            'depth_l1_90': np.mean(self._l1_90),
            'depth_psnr': np.mean(self._psnr),
            'depth_ssim': np.mean(self._ssim),
        }

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "depth_evaluation.pth")
            with open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"depth": res})
        self._logger.info(results)
        return results


class SnEvaluator(DatasetEvaluator):
    """Evaluate surface normal l1 and angle metrics."""

    def __init__(self, output_dir=None):
        """
        Args:
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir

        self._angle_mean = []
        self._angle_median = []
        self._angle_80 = []
        self._angle_90 = []

    def reset(self):
        """
        Reset the internal state to prepare for a new round of evaluation.
        """
        self._angle_mean.clear()
        self._angle_median.clear()
        self._angle_80.clear()
        self._angle_90.clear()

    def process(self, inputs, outputs):
        """Process the pair of inputs and outputs.
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input, output in zip(inputs, outputs):
            gt_depth = input['depth'].clone()[0]  # (64, 512)
            gt_sn = input['sn'].clone()  # (3, 64, 512)
            pred_sn = output['sn'].detach().cpu().clone()  # (3, 64, 512)
            mask = torch.logical_and(gt_sn[0] > -10, gt_depth < 0.96)  # (64, 512)

            cos_angle = F.cosine_similarity(pred_sn, gt_sn, dim=0, eps=1e-4).clamp(-1.0, 1.0)
            angle = torch.rad2deg(torch.acos(cos_angle[mask]))

            self._angle_mean.append(angle.mean())
            self._angle_median.append(angle.quantile(0.5))
            self._angle_80.append(angle.quantile(0.8))
            self._angle_90.append(angle.quantile(0.9))

    def evaluate(self):
        """
        Returns:
            dict: A dict of {metric name: score} pairs.
        """
        res = {
            'sn_angle_mean': np.mean(self._angle_mean),
            'sn_angle_median': np.mean(self._angle_median),
            'sn_angle_80': np.mean(self._angle_80),
            'sn_angle_90': np.mean(self._angle_90),
        }

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sn_evaluation.pth")
            with open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"sn": res})
        self._logger.info(results)
        return results


class AdvDepthEvaluator(DatasetEvaluator):
    """Evaluate depth metrics for the final model in the inference notebook.
    This is an advance evaluator. It has more metrics and thus time-comsuming.

    Metrics include:
        1. mean, median, 90%, 95% of all the L1 depth error
        2. CDF plots (error-percentile) of all the L1 depth error
        3. mean, median, 90%, 95% L1 depth error grouped by distance (0-3m, 3-6m, 6-10m)
        ----
        4. the above 1,2,3 separated by foreground and background semantics
    """

    def __init__(self, output_dir=None):
        """
        Args:
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir

        self.fg_class = torch.tensor([0, 1, 2, 4, 6])  # foreground class indices
        self.l1_errors = []  # 1D float32
        self.fg_positions = []  # 1D bool
        self.gt_depths = []  # 1D float32

    def reset(self):
        """
        Reset the internal state to prepare for a new round of evaluation.
        """
        self.l1_errors = []
        self.fg_positions = []
        self.gt_depths = []

    def process(self, inputs, outputs):
        """Process the pair of inputs and outputs.
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input, output in zip(inputs, outputs):
            gt_depth = input['depth'].clone().unsqueeze(0)  # (1, 1, 64, 512)
            pred_depth = output['depth'].detach().cpu().clone().unsqueeze(0)  # (1, 1, 64, 512)
            mask_valid = gt_depth > 0
            mask_inr = mask_valid & (gt_depth < 0.96)

            # get foreground and background mask
            gt_sem_seg = input["sem_seg"].unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 512)
            mask_fg = torch.isin(gt_sem_seg, self.fg_class)

            # store the L1 errors per pixel
            l1_loss = F.l1_loss(pred_depth, gt_depth, reduction='none')
            self.l1_errors.append(l1_loss[mask_inr].numpy())
            self.fg_positions.append(mask_fg[mask_inr].numpy())
            self.gt_depths.append(gt_depth[mask_inr].numpy())

    def evaluate(self):
        """
        Evaluate the statistics to get the metrics.
        """
        self.l1_errors = np.hstack(self.l1_errors)
        self.fg_positions = np.hstack(self.fg_positions)
        self.gt_depths = np.hstack(self.gt_depths)

        # group by distance and foreground/background
        self.inds_0to3m_all = self.gt_depths <= 0.3
        self.inds_3to6m_all = np.logical_and(self.gt_depths > 0.3, self.gt_depths <= 0.6)
        self.inds_6to10m_all = np.logical_and(self.gt_depths > 0.6, self.gt_depths < 0.96)
        inds_0to3m_fg = np.logical_and(self.inds_0to3m_all, self.fg_positions)
        inds_3to6m_fg = np.logical_and(self.inds_3to6m_all, self.fg_positions)
        inds_6to10m_fg = np.logical_and(self.inds_6to10m_all, self.fg_positions)
        inds_0to3m_bg = np.logical_and(self.inds_0to3m_all, ~self.fg_positions)
        inds_3to6m_bg = np.logical_and(self.inds_3to6m_all, ~self.fg_positions)
        inds_6to10m_bg = np.logical_and(self.inds_6to10m_all, ~self.fg_positions)

        res = {
            'l1_mean_0to10m_all': np.mean(self.l1_errors),
            'l1_mean_0to3m_all': np.mean(self.l1_errors[self.inds_0to3m_all]),
            'l1_mean_3to6m_all': np.mean(self.l1_errors[self.inds_3to6m_all]),
            'l1_mean_6to10m_all': np.mean(self.l1_errors[self.inds_6to10m_all]),
            'l1_median_0to10m_all': np.median(self.l1_errors),
            'l1_median_0to3m_all': np.median(self.l1_errors[self.inds_0to3m_all]),
            'l1_median_3to6m_all': np.median(self.l1_errors[self.inds_3to6m_all]),
            'l1_median_6to10m_all': np.median(self.l1_errors[self.inds_6to10m_all]),
            'l1_90_0to10m_all': np.quantile(self.l1_errors, 0.9),
            'l1_90_0to3m_all': np.quantile(self.l1_errors[self.inds_0to3m_all], 0.9),
            'l1_90_3to6m_all': np.quantile(self.l1_errors[self.inds_3to6m_all], 0.9),
            'l1_90_6to10m_all': np.quantile(self.l1_errors[self.inds_6to10m_all], 0.9),
            'l1_95_0to10m_all': np.quantile(self.l1_errors, 0.95),
            'l1_95_0to3m_all': np.quantile(self.l1_errors[self.inds_0to3m_all], 0.95),
            'l1_95_3to6m_all': np.quantile(self.l1_errors[self.inds_3to6m_all], 0.95),
            'l1_95_6to10m_all': np.quantile(self.l1_errors[self.inds_6to10m_all], 0.95),
            # ----
            'l1_mean_0to10m_fg': np.mean(self.l1_errors[self.fg_positions]),
            'l1_mean_0to3m_fg': np.mean(self.l1_errors[inds_0to3m_fg]),
            'l1_mean_3to6m_fg': np.mean(self.l1_errors[inds_3to6m_fg]),
            'l1_mean_6to10m_fg': np.mean(self.l1_errors[inds_6to10m_fg]),
            'l1_median_0to10m_fg': np.median(self.l1_errors[self.fg_positions]),
            'l1_median_0to3m_fg': np.median(self.l1_errors[inds_0to3m_fg]),
            'l1_median_3to6m_fg': np.median(self.l1_errors[inds_3to6m_fg]),
            'l1_median_6to10m_fg': np.median(self.l1_errors[inds_6to10m_fg]),
            'l1_90_0to10m_fg': np.quantile(self.l1_errors[self.fg_positions], 0.9),
            'l1_90_0to3m_fg': np.quantile(self.l1_errors[inds_0to3m_fg], 0.9),
            'l1_90_3to6m_fg': np.quantile(self.l1_errors[inds_3to6m_fg], 0.9),
            'l1_90_6to10m_fg': np.quantile(self.l1_errors[inds_6to10m_fg], 0.9),
            'l1_95_0to10m_fg': np.quantile(self.l1_errors[self.fg_positions], 0.95),
            'l1_95_0to3m_fg': np.quantile(self.l1_errors[inds_0to3m_fg], 0.95),
            'l1_95_3to6m_fg': np.quantile(self.l1_errors[inds_3to6m_fg], 0.95),
            'l1_95_6to10m_fg': np.quantile(self.l1_errors[inds_6to10m_fg], 0.95),
            # ----
            'l1_mean_0to10m_bg': np.mean(self.l1_errors[~self.fg_positions]),
            'l1_mean_0to3m_bg': np.mean(self.l1_errors[inds_0to3m_bg]),
            'l1_mean_3to6m_bg': np.mean(self.l1_errors[inds_3to6m_bg]),
            'l1_mean_6to10m_bg': np.mean(self.l1_errors[inds_6to10m_bg]),
            'l1_median_0to10m_bg': np.median(self.l1_errors[~self.fg_positions]),
            'l1_median_0to3m_bg': np.median(self.l1_errors[inds_0to3m_bg]),
            'l1_median_3to6m_bg': np.median(self.l1_errors[inds_3to6m_bg]),
            'l1_median_6to10m_bg': np.median(self.l1_errors[inds_6to10m_bg]),
            'l1_90_0to10m_bg': np.quantile(self.l1_errors[~self.fg_positions], 0.9),
            'l1_90_0to3m_bg': np.quantile(self.l1_errors[inds_0to3m_bg], 0.9),
            'l1_90_3to6m_bg': np.quantile(self.l1_errors[inds_3to6m_bg], 0.9),
            'l1_90_6to10m_bg': np.quantile(self.l1_errors[inds_6to10m_bg], 0.9),
            'l1_95_0to10m_bg': np.quantile(self.l1_errors[~self.fg_positions], 0.95),
            'l1_95_0to3m_bg': np.quantile(self.l1_errors[inds_0to3m_bg], 0.95),
            'l1_95_3to6m_bg': np.quantile(self.l1_errors[inds_3to6m_bg], 0.95),
            'l1_95_6to10m_bg': np.quantile(self.l1_errors[inds_6to10m_bg], 0.95),
        }

        if self.output_dir:
            file_path = os.path.join(self.output_dir, "depth_adv_evaluation.pth")
            states = {
                'l1_errors': self.l1_errors,
                'fg_positions': self.fg_positions,
                'gt_depths': self.gt_depths,
            }
            with open(file_path, "wb") as f:
                torch.save({'res': res, 'states': states}, f)

        results = OrderedDict({"depth_advance": res})
        self.logger.info(results)
        return results


class AdvSnEvaluator(DatasetEvaluator):
    """Evaluate surface normal metrics for the final model in the inference notebook.
    This is an advance evaluator. It has more metrics and thus time-comsuming.

    Metrics include:
        1. mean, median, 90%, 95% of all the L1 angle error
        2. CDF plots (error-percentile) of all the L1 angle error
        3. mean, median, 90%, 95% L1 angle error grouped by distance (0-3m, 3-6m, 6-10m)
        ----
        4. the above 1,2,3 separated by foreground and background semantics
    """

    def __init__(self, output_dir=None):
        """
        Args:
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir

        self.fg_class = torch.tensor([0, 1, 2, 4, 6])  # foreground class indices
        self.angle_errors = []  # 1D float32, in deg
        self.fg_positions = []  # 1D bool
        self.gt_depths = []  # 1D float32

    def reset(self):
        """
        Reset the internal state to prepare for a new round of evaluation.
        """
        self.angle_errors = []
        self.fg_positions = []
        self.gt_depths = []

    def process(self, inputs, outputs):
        """Process the pair of inputs and outputs.
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input, output in zip(inputs, outputs):
            gt_depth = input['depth'].clone()[0]  # (64, 512)
            gt_sn = input['sn'].clone()  # (3, 64, 512)
            pred_sn = output['sn'].detach().cpu().clone()  # (3, 64, 512)
            mask_sn = torch.logical_and(gt_sn[0] > -10, gt_depth < 0.96)  # (64, 512)

            # compute angle error
            cos_angle = F.cosine_similarity(pred_sn, gt_sn, dim=0, eps=1e-4).clamp(-1.0, 1.0)
            angle_err = torch.rad2deg(torch.acos(cos_angle))

            # get foreground and background mask
            gt_sem_seg = input["sem_seg"]  # (64, 512)
            mask_fg = torch.isin(gt_sem_seg, self.fg_class)

            # store the angle errors per pixel
            self.angle_errors.append(angle_err[mask_sn].numpy())
            self.fg_positions.append(mask_fg[mask_sn].numpy())
            self.gt_depths.append(gt_depth[mask_sn].numpy())

    def evaluate(self):
        """
        Evaluate the statistics to get the metrics.
        """
        self.angle_errors = np.hstack(self.angle_errors)
        self.fg_positions = np.hstack(self.fg_positions)
        self.gt_depths = np.hstack(self.gt_depths)

        # group by distance and foreground/background
        self.inds_0to3m_all = self.gt_depths <= 0.3
        self.inds_3to6m_all = np.logical_and(self.gt_depths > 0.3, self.gt_depths <= 0.6)
        self.inds_6to10m_all = np.logical_and(self.gt_depths > 0.6, self.gt_depths < 0.96)
        inds_0to3m_fg = np.logical_and(self.inds_0to3m_all, self.fg_positions)
        inds_3to6m_fg = np.logical_and(self.inds_3to6m_all, self.fg_positions)
        inds_6to10m_fg = np.logical_and(self.inds_6to10m_all, self.fg_positions)
        inds_0to3m_bg = np.logical_and(self.inds_0to3m_all, ~self.fg_positions)
        inds_3to6m_bg = np.logical_and(self.inds_3to6m_all, ~self.fg_positions)
        inds_6to10m_bg = np.logical_and(self.inds_6to10m_all, ~self.fg_positions)

        res = {
            'angle_mean_0to10m_all': np.mean(self.angle_errors),
            'angle_mean_0to3m_all': np.mean(self.angle_errors[self.inds_0to3m_all]),
            'angle_mean_3to6m_all': np.mean(self.angle_errors[self.inds_3to6m_all]),
            'angle_mean_6to10m_all': np.mean(self.angle_errors[self.inds_6to10m_all]),
            'angle_median_0to10m_all': np.median(self.angle_errors),
            'angle_median_0to3m_all': np.median(self.angle_errors[self.inds_0to3m_all]),
            'angle_median_3to6m_all': np.median(self.angle_errors[self.inds_3to6m_all]),
            'angle_median_6to10m_all': np.median(self.angle_errors[self.inds_6to10m_all]),
            'angle_90_0to10m_all': np.quantile(self.angle_errors, 0.9),
            'angle_90_0to3m_all': np.quantile(self.angle_errors[self.inds_0to3m_all], 0.9),
            'angle_90_3to6m_all': np.quantile(self.angle_errors[self.inds_3to6m_all], 0.9),
            'angle_90_6to10m_all': np.quantile(self.angle_errors[self.inds_6to10m_all], 0.9),
            'angle_95_0to10m_all': np.quantile(self.angle_errors, 0.95),
            'angle_95_0to3m_all': np.quantile(self.angle_errors[self.inds_0to3m_all], 0.95),
            'angle_95_3to6m_all': np.quantile(self.angle_errors[self.inds_3to6m_all], 0.95),
            'angle_95_6to10m_all': np.quantile(self.angle_errors[self.inds_6to10m_all], 0.95),
            # ----
            'angle_mean_0to10m_fg': np.mean(self.angle_errors[self.fg_positions]),
            'angle_mean_0to3m_fg': np.mean(self.angle_errors[inds_0to3m_fg]),
            'angle_mean_3to6m_fg': np.mean(self.angle_errors[inds_3to6m_fg]),
            'angle_mean_6to10m_fg': np.mean(self.angle_errors[inds_6to10m_fg]),
            'angle_median_0to10m_fg': np.median(self.angle_errors[self.fg_positions]),
            'angle_median_0to3m_fg': np.median(self.angle_errors[inds_0to3m_fg]),
            'angle_median_3to6m_fg': np.median(self.angle_errors[inds_3to6m_fg]),
            'angle_median_6to10m_fg': np.median(self.angle_errors[inds_6to10m_fg]),
            'angle_90_0to10m_fg': np.quantile(self.angle_errors[self.fg_positions], 0.9),
            'angle_90_0to3m_fg': np.quantile(self.angle_errors[inds_0to3m_fg], 0.9),
            'angle_90_3to6m_fg': np.quantile(self.angle_errors[inds_3to6m_fg], 0.9),
            'angle_90_6to10m_fg': np.quantile(self.angle_errors[inds_6to10m_fg], 0.9),
            'angle_95_0to10m_fg': np.quantile(self.angle_errors[self.fg_positions], 0.95),
            'angle_95_0to3m_fg': np.quantile(self.angle_errors[inds_0to3m_fg], 0.95),
            'angle_95_3to6m_fg': np.quantile(self.angle_errors[inds_3to6m_fg], 0.95),
            'angle_95_6to10m_fg': np.quantile(self.angle_errors[inds_6to10m_fg], 0.95),
            # ----
            'angle_mean_0to10m_bg': np.mean(self.angle_errors[~self.fg_positions]),
            'angle_mean_0to3m_bg': np.mean(self.angle_errors[inds_0to3m_bg]),
            'angle_mean_3to6m_bg': np.mean(self.angle_errors[inds_3to6m_bg]),
            'angle_mean_6to10m_bg': np.mean(self.angle_errors[inds_6to10m_bg]),
            'angle_median_0to10m_bg': np.median(self.angle_errors[~self.fg_positions]),
            'angle_median_0to3m_bg': np.median(self.angle_errors[inds_0to3m_bg]),
            'angle_median_3to6m_bg': np.median(self.angle_errors[inds_3to6m_bg]),
            'angle_median_6to10m_bg': np.median(self.angle_errors[inds_6to10m_bg]),
            'angle_90_0to10m_bg': np.quantile(self.angle_errors[~self.fg_positions], 0.9),
            'angle_90_0to3m_bg': np.quantile(self.angle_errors[inds_0to3m_bg], 0.9),
            'angle_90_3to6m_bg': np.quantile(self.angle_errors[inds_3to6m_bg], 0.9),
            'angle_90_6to10m_bg': np.quantile(self.angle_errors[inds_6to10m_bg], 0.9),
            'angle_95_0to10m_bg': np.quantile(self.angle_errors[~self.fg_positions], 0.95),
            'angle_95_0to3m_bg': np.quantile(self.angle_errors[inds_0to3m_bg], 0.95),
            'angle_95_3to6m_bg': np.quantile(self.angle_errors[inds_3to6m_bg], 0.95),
            'angle_95_6to10m_bg': np.quantile(self.angle_errors[inds_6to10m_bg], 0.95),
        }

        if self.output_dir:
            file_path = os.path.join(self.output_dir, "sn_adv_evaluation.pth")
            states = {
                'angle_errors': self.angle_errors,
                'fg_positions': self.fg_positions,
                'gt_depths': self.gt_depths,
                'inds_0to3m_all': self.inds_0to3m_all,
                'inds_3to6m_all': self.inds_3to6m_all,
                'inds_6to10m_all': self.inds_6to10m_all,
            }
            with open(file_path, "wb") as f:
                torch.save({'res': res, 'states': states}, f)

        results = OrderedDict({"sn_advance": res})
        self.logger.info(results)
        return results
