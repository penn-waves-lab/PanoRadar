import os
import math
import torch
import logging
import numpy as np

from copy import deepcopy
from collections import OrderedDict

from detectron2.evaluation import DatasetEvaluator

from .circular_det_evaluation import CircularBoxes, circular_pairwise_iou

__all__ = ["HumanLocalizationEvaluator", "AdvHumanLocalizationEvaluator"]

def calculate_localization_error(gt_depth, gt_boxes, gt_classes, pred_depth, pred_boxes, pred_classes, iou_thresh=0.5):
    """
    Function to calculate human localization error between ground truth and prediction
    Args:
        gt_depth: Ground truth tensor of depth map (1, 64, 512)
        gt_boxes: CircularBoxes, ground truth boxes (N,4)
        gt_classes: Ground truth classes, ith index represents class of the ith box (N)
        pred_depth: Predicted depth map (1, 64, 512)
        pred_boxes: CircularBoxes, predicted boxes for the image (N2, 4)
        pred_classes: Predicted classes, ith index represents class of the ith predicted box (N2)
        iou_thresh: float, iou threshold for considering two boxes to be aligned
    Output:
        list of (depth_err, angle_err, l2_dist_err)
        depth_err: Depth error
        angle_err: Angle error
        l2_dist_err: L2 distance between predicted point and actual point in cartesian plane/coordinates
    """

    # Only want boxes that have a human, and round to integer
    gt_human_indices = [i for i, v in enumerate(gt_classes) if v == 0]
    gt_boxes.tensor = torch.round(gt_boxes.tensor[gt_human_indices]).detach().cpu().clone()
    gt_boxes = CircularBoxes(gt_boxes.tensor)
    gt_boxes.clip((64, 512))

    pred_human_indices = [i for i, v in enumerate(pred_classes) if v == 0]
    pred_boxes.tensor = torch.round(pred_boxes.tensor[pred_human_indices]).detach().cpu().clone()
    pred_boxes = CircularBoxes(pred_boxes.tensor)
    pred_boxes.clip((64, 512))
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return None

    # Get pairwise IOU for the GT and pred human boxes
    iou_mat = circular_pairwise_iou(gt_boxes, pred_boxes)
    _, gt_max_indices = iou_mat.max(dim=1) # Get the predicted box with maxIOU for each gt box
    
    # For each gt box match with a pred box by checking iou thresh, if no match then -1
    match_indices = [gt_max_indices[i].detach().item() if iou_mat[i][gt_max_indices[i]] > iou_thresh else -1 for i in range(len(gt_boxes))] # [N], where N is num gt boxes

    if len(match_indices) == 0:
        return None

    # Get the gt boxes and pred boxes that are matched
    gt_matched_indices = [i for i, v in enumerate(match_indices) if v != -1]
    pred_matched_indices = [v for v in match_indices if v != -1]

    gt_matched_boxes = gt_boxes.tensor[gt_matched_indices]
    pred_matched_boxes = pred_boxes.tensor[pred_matched_indices]

    gt_localization = get_human_localization(gt_depth[0], gt_matched_boxes)
    pred_localization = get_human_localization(pred_depth[0], pred_matched_boxes)

    results = []

    # Iterate through gt and pred localizations to calculate errors
    for gt_localize, pred_localize in zip(gt_localization, pred_localization):

        range_diff = np.abs(gt_localize[0] - pred_localize[0])

        angle_diff = np.rad2deg(np.abs(gt_localize[1] - pred_localize[1]))

        euclid_diff = np.linalg.norm(np.array(gt_localize[2]) - np.array(pred_localize[2]))

        results.append([range_diff, angle_diff, euclid_diff])
    
    return results

def calculate_localization_error_adv(gt_depth, gt_boxes, gt_classes, pred_depth, pred_boxes, pred_classes, iou_thresh=0.5):
    """
    Function to calculate human localization error between ground truth and prediction, separating into different distances
    Args:
        gt_depth: Ground truth tensor of depth map (1, 64, 512)
        gt_boxes: CircularBoxes, ground truth boxes (N,4)
        gt_classes: Ground truth classes, ith index represents class of the ith box (N)
        pred_depth: Predicted depth map (1, 64, 512)
        pred_boxes: CircularBoxes, predicted boxes for the image (N2, 4)
        pred_classes: Predicted classes, ith index represents class of the ith predicted box (N2)
        iou_thresh: float, iou threshold for considering two boxes to be aligned
    Output:
        list of (depth_err, angle_err, l2_dist_err, dist_class)
        depth_err: Depth error
        angle_err: Angle error
        l2_dist_err: L2 distance between predicted point and actual point in cartesian plane/coordinates
        dist_class: The distance class for which the human box belongs to, 0 is 0-3m, 1 is 3-6m, and 2 is 6-10m
    """

    # Only want boxes that have a human, and round to integer
    gt_human_indices = [i for i, v in enumerate(gt_classes) if v == 0]
    gt_boxes.tensor = torch.round(gt_boxes.tensor[gt_human_indices]).detach().cpu().clone()
    gt_boxes = CircularBoxes(gt_boxes.tensor)
    gt_boxes.clip((64, 512))

    pred_human_indices = [i for i, v in enumerate(pred_classes) if v == 0]
    pred_boxes.tensor = torch.round(pred_boxes.tensor[pred_human_indices]).detach().cpu().clone()
    pred_boxes = CircularBoxes(pred_boxes.tensor)
    pred_boxes.clip((64, 512))
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return None

    # Get pairwise IOU for the GT and pred human boxes
    iou_mat = circular_pairwise_iou(gt_boxes, pred_boxes)
    _, gt_max_indices = iou_mat.max(dim=1) # Get the predicted box with maxIOU for each gt box
    
    # For each gt box match with a pred box by checking iou thresh, if no match then -1
    match_indices = [gt_max_indices[i].detach().item() if iou_mat[i][gt_max_indices[i]] > 0.5 else -1 for i in range(len(gt_boxes))] # [N], where N is num gt boxes

    if len(match_indices) == 0:
        return None

    # Get the gt boxes and pred boxes that are matched
    gt_matched_indices = [i for i, v in enumerate(match_indices) if v != -1]
    pred_matched_indices = [v for v in match_indices if v != -1]

    gt_matched_boxes = gt_boxes.tensor[gt_matched_indices]
    pred_matched_boxes = pred_boxes.tensor[pred_matched_indices]

    gt_localization = get_human_localization(gt_depth[0], gt_matched_boxes)
    pred_localization = get_human_localization(pred_depth[0], pred_matched_boxes)

    results = []

    # Iterate through gt and pred localizations to calculate errors
    for gt_localize, pred_localize in zip(gt_localization, pred_localization):

        range_diff = np.abs(gt_localize[0] - pred_localize[0])

        angle_diff = np.rad2deg(np.abs(gt_localize[1] - pred_localize[1]))

        euclid_diff = np.linalg.norm(np.array(gt_localize[2]) - np.array(pred_localize[2]))

        if gt_localize[0] >= 0 and gt_localize[0] < 3:
            dist_class = 0
        elif gt_localize[0] >= 3 and gt_localize[0] < 6:
            dist_class = 1
        else:
            dist_class = 2

        results.append([range_diff, angle_diff, euclid_diff, dist_class])
    
    return results

def nan_if(arr, value):
    """
    Function to convert array elements that have a value into nan. This used for the -1000 spots in predicted depth
    """
    return np.where(arr == value, np.nan, arr)

def polar_to_cartesian(range, theta):
    """
    Converts polar coordinates (r, theta) to cartesian coordinates (x,y)
    Args:
        r: range value in m
        theta: angle, in radians
    Output:
        (x,y): cartesian coordinates in m
    """
    x = range * math.cos(theta)
    y = range * math.sin(theta)

    return [x, y]

def get_human_localization(depth, boxes):
    """
    Function to get the human localization range, angle and coordinate for each human in the image
    Args: 
        depth: torch.Tensor, Depth map (1, 64, 512)
        boxes: torch.Tensor, HUMAN boxes for the image (N, 4)
    Returns:
        List of (range, angle, coordinate) of length N, each index corresponds to each box
    """
    result = []
    for box in boxes:
        box = box.tolist()
        x1 = int(box[0])
        x2 = int(box[2]) if box[0] < box[2] else int(box[2]) + 512
        y1, y2 = int(box[1]), int(box[3])

        # Azimuth coordinate
        midazi = ((x2 + x1) / 2) % 512
        theta_angle = - ((2 * math.pi / 512) * midazi) # Convert to radian angle

        # Get the depth as the median of all depths in the bounding box
        newx2 = int(box[2]) % 512
        if box[2] < box[0] or box[2] >= 512:
            boxDepth = torch.cat((depth[y1: y2 + 1, x1: 512], depth[y1: y2 + 1, 0: newx2 + 1]), dim=1)
        else:
            boxDepth = depth[y1:y2 + 1, x1:x2 + 1]
        
        # Ignore NAN values/ -1000 values
        range = np.nanquantile(nan_if(boxDepth, -1000), 0.25) * 10

        coordinate = polar_to_cartesian(range, theta_angle)
        result.append([range, theta_angle, coordinate])

    return result

class HumanLocalizationEvaluator(DatasetEvaluator):
    """
    Class to evaluate human localization errors
    """


    def __init__(self, output_dir=None):
        """
        Args:
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir

        self._depth_errors = []
        self._angle_errors = []
        self._l2_dist_errors = []

    def reset(self):
        self._depth_errors.clear()
        self._angle_errors.clear()
        self._l2_dist_errors.clear()

    def process(self, inputs, outputs):
        
        for input, output in zip(inputs, outputs):
            # Iterate through all images

            gt_depth = input['depth'].detach().cpu().clone()
            gt_boxes = deepcopy(input['instances'].gt_boxes)
            gt_classes = input['instances'].gt_classes.detach().cpu().clone()
            pred_depth = output['depth'].detach().cpu().clone()
            pred_boxes = deepcopy(output['instances'].pred_boxes)
            pred_classes = output['instances'].pred_classes.detach().cpu().clone()

            # errors for all boxes for 1 image
            localization_errors = calculate_localization_error(gt_depth, gt_boxes, gt_classes, pred_depth, pred_boxes, pred_classes)

            if localization_errors: 
                
                localization_errors = np.array(localization_errors)

                depth_errs = localization_errors[:, 0]
                angle_errs = localization_errors[:, 1]
                l2_errs = localization_errors[:, 2]

                # Ignore nan values
                depth_errs = depth_errs[~np.isnan(depth_errs)].tolist()
                angle_errs = angle_errs[~np.isnan(depth_errs)].tolist()
                l2_errs = l2_errs[~np.isnan(depth_errs)].tolist()

                self._depth_errors.extend(depth_errs)
                self._angle_errors.extend(angle_errs)
                self._l2_dist_errors.extend(l2_errs)
    
    def evaluate(self):
        res = {
            "human_loc_depth_mean": np.mean(self._depth_errors),
            "human_loc_depth_median": np.median(self._depth_errors),
            "human_loc_angle_mean": np.mean(self._angle_errors),
            "human_loc_angle_median": np.median(self._angle_errors),
            "human_loc_l2_dist_mean": np.mean(self._l2_dist_errors),
            "human_loc_l2_dist_median": np.median(self._l2_dist_errors),
        }

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "human_loc_evaluation.pth")
            with open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"human_loc": res})
        self._logger.info(results)
        return results



class AdvHumanLocalizationEvaluator(DatasetEvaluator):
    """
    Class to evaluate human localization errors, with separate data on how far the human is
    Ranges from 0-3m, 3-6m, and 6-10m
    """

    def __init__(self, output_dir=None):
        """
        Args:
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir

        self._depth_errors_all = []
        self._depth_errors_0_3 = []
        self._depth_errors_3_6 = []
        self._depth_errors_6_10 = []
        self._angle_errors_all = []
        self._angle_errors_0_3 = []
        self._angle_errors_3_6 = []
        self._angle_errors_6_10 = []
        self._l2_dist_errors_all = []
        self._l2_dist_errors_0_3 = []
        self._l2_dist_errors_3_6 = []
        self._l2_dist_errors_6_10 = []

    def reset(self):
        self._depth_errors_all.clear()
        self._depth_errors_0_3.clear()
        self._depth_errors_3_6.clear()
        self._depth_errors_6_10.clear()
        self._angle_errors_all.clear()
        self._angle_errors_0_3.clear()
        self._angle_errors_3_6.clear()
        self._angle_errors_6_10.clear()
        self._l2_dist_errors_all.clear()
        self._l2_dist_errors_0_3.clear()
        self._l2_dist_errors_3_6.clear()
        self._l2_dist_errors_6_10.clear()

    def process(self, inputs, outputs):
        
        for input, output in zip(inputs, outputs):
            # Iterate through all images

            gt_depth = input['depth'].detach().cpu().clone()
            gt_boxes = deepcopy(input['instances'].gt_boxes)
            gt_classes = input['instances'].gt_classes.detach().cpu().clone()
            pred_depth = output['depth'].detach().cpu().clone()
            pred_boxes = deepcopy(output['instances'].pred_boxes)
            pred_classes = output['instances'].pred_classes.detach().cpu().clone()

            # errors for all boxes for 1 image
            localization_errors = calculate_localization_error_adv(gt_depth, gt_boxes, gt_classes, pred_depth, pred_boxes, pred_classes)

            if localization_errors: 
                
                localization_errors = np.array(localization_errors)

                depth_errs = localization_errors[:, 0]
                angle_errs = localization_errors[:, 1]
                l2_errs = localization_errors[:, 2]

                # Separate the different classes
                depth_errs_0_3 = depth_errs[np.where(localization_errors[:, 3] == 0)]
                depth_errs_3_6 = depth_errs[np.where(localization_errors[:, 3] == 1)]
                depth_errs_6_10 = depth_errs[np.where(localization_errors[:, 3] == 2)]

                angle_errs_0_3 = angle_errs[np.where(localization_errors[:, 3] == 0)]
                angle_errs_3_6 = angle_errs[np.where(localization_errors[:, 3] == 1)]
                angle_errs_6_10 = angle_errs[np.where(localization_errors[:, 3] == 2)]

                l2_errs_0_3 = l2_errs[np.where(localization_errors[:, 3] == 0)]
                l2_errs_3_6 = l2_errs[np.where(localization_errors[:, 3] == 1)]
                l2_errs_6_10 = l2_errs[np.where(localization_errors[:, 3] == 2)]

                # Ignore nan values
                depth_errs = depth_errs[~np.isnan(depth_errs)].tolist()
                depth_errs_0_3 = depth_errs_0_3[~np.isnan(depth_errs_0_3)].tolist()
                depth_errs_3_6 = depth_errs_3_6[~np.isnan(depth_errs_3_6)].tolist()
                depth_errs_6_10 = depth_errs_6_10[~np.isnan(depth_errs_6_10)].tolist()

                angle_errs = angle_errs[~np.isnan(depth_errs)].tolist()
                angle_errs_0_3 = angle_errs_0_3[~np.isnan(angle_errs_0_3)].tolist()
                angle_errs_3_6 = angle_errs_3_6[~np.isnan(angle_errs_3_6)].tolist()
                angle_errs_6_10 = angle_errs_6_10[~np.isnan(angle_errs_6_10)].tolist()

                l2_errs = l2_errs[~np.isnan(depth_errs)].tolist()
                l2_errs_0_3 = l2_errs_0_3[~np.isnan(l2_errs_0_3)].tolist()
                l2_errs_3_6 = l2_errs_3_6[~np.isnan(l2_errs_3_6)].tolist()
                l2_errs_6_10 = l2_errs_6_10[~np.isnan(l2_errs_6_10)].tolist()

                self._depth_errors_all.extend(depth_errs)
                self._depth_errors_0_3.extend(depth_errs_0_3)
                self._depth_errors_3_6.extend(depth_errs_3_6)
                self._depth_errors_6_10.extend(depth_errs_6_10)

                self._angle_errors_all.extend(angle_errs)
                self._angle_errors_0_3.extend(angle_errs_0_3)
                self._angle_errors_3_6.extend(angle_errs_3_6)
                self._angle_errors_6_10.extend(angle_errs_6_10)
                
                self._l2_dist_errors_all.extend(l2_errs)
                self._l2_dist_errors_0_3.extend(l2_errs_0_3)
                self._l2_dist_errors_3_6.extend(l2_errs_3_6)
                self._l2_dist_errors_6_10.extend(l2_errs_6_10)

    
    def evaluate(self):
        res = {
            "human_loc_depth_mean_all": np.mean(self._depth_errors_all),
            "human_loc_depth_mean_0to3m": np.mean(self._depth_errors_0_3),
            "human_loc_depth_mean_3to6m": np.mean(self._depth_errors_3_6),
            "human_loc_depth_mean_6to10m": np.mean(self._depth_errors_6_10),
            "human_loc_depth_median_all": np.median(self._depth_errors_all),
            "human_loc_depth_median_0to3m": np.median(self._depth_errors_0_3),
            "human_loc_depth_median_3to6m": np.median(self._depth_errors_3_6),
            "human_loc_depth_median_6to10m": np.median(self._depth_errors_6_10),
            
            "human_loc_angle_mean_all": np.mean(self._angle_errors_all),
            "human_loc_angle_mean_0to3m": np.mean(self._angle_errors_0_3),
            "human_loc_angle_mean_3to6m": np.mean(self._angle_errors_3_6),
            "human_loc_angle_mean_6to10m": np.mean(self._angle_errors_6_10),
            "human_loc_angle_median_all": np.median(self._angle_errors_all),
            "human_loc_angle_median_0to3m": np.median(self._angle_errors_0_3),
            "human_loc_angle_median_3to6m": np.median(self._angle_errors_3_6),
            "human_loc_angle_median_6to10m": np.median(self._angle_errors_6_10),

            "human_loc_l2_dist_mean_all": np.mean(self._l2_dist_errors_all),
            "human_loc_l2_dist_mean_0to3m": np.mean(self._l2_dist_errors_0_3),
            "human_loc_l2_dist_mean_3to6m": np.mean(self._l2_dist_errors_3_6),
            "human_loc_l2_dist_mean_6to10m": np.mean(self._l2_dist_errors_6_10),
            "human_loc_l2_dist_median_all": np.median(self._l2_dist_errors_all),
            "human_loc_l2_dist_median_0to3m": np.median(self._l2_dist_errors_0_3),
            "human_loc_l2_dist_median_3to6m": np.median(self._l2_dist_errors_3_6),
            "human_loc_l2_dist_median_6to10m": np.median(self._l2_dist_errors_6_10),
        }

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "human_loc_advanced_evaluation.pth")
            with open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"human_loc_advanced": res})
        self._logger.info(results)
        return results