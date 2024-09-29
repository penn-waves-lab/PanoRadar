"""This is to modified the original detectron2 `COCOEvaluator` to support AP30."""

import os
import json
import logging
import itertools
import numpy as np
import torch

from collections import OrderedDict
from typing import Dict, List

from detectron2.evaluation import DatasetEvaluator, SemSegEvaluator
from detectron2.evaluation.sem_seg_evaluation import load_image_into_numpy_array
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False

__all__ = ["ObjEvaluator", "AdvObjEvaluator", "SemSegEvaluator", "AdvSemSegEvaluator"]


class ObjEvaluator(DatasetEvaluator):
    """Evaluate object detection metrics for model.

    The original detectron2 `COCOEvaluator` reads inputs from a cached file in the
    `inference` folder, which is generated directly from the dataset annotation before
    the dataset mapper.

    In contrast, this evaluator ensures that the metrics are computed from the given
    inputs and outputs, both of which should have an Instances object.

    Metrics include:
        1. mAP[0.5:0.05:0.95], mAP30, mAP50, mAP75
        2. Per-class AP[0.5:0.05:0.95], AP30, AP50, AP75
    """

    def __init__(self, dataset_name, output_dir=None):
        """
        Args:
            dataset_name: the name of the dataset
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        class_labels = MetadataCatalog.get(dataset_name).thing_classes  # list
        self.category_keys = [{'id': idx, 'name': label} for idx, label in enumerate(class_labels)]

        self.input_instances = []  # to store input Instances()
        self.output_instances = []  # to store output Instances()

    def reset(self):
        """
        Reset the internal state to prepare for a new round of evaluation.
        """
        self.input_instances.clear()
        self.output_instances.clear()

    def process(self, inputs, outputs):
        """Process the pair of inputs and outputs.
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input, output in zip(inputs, outputs):
            self.input_instances.append(input['instances'])
            self.output_instances.append(output['instances'].to('cpu'))

    def evaluate(self):
        """
        Evaluate the statistics to get the metrics.
        Returns:
            results (OrderedDict): the metrics
        """
        coco_eval_5095 = self._compute_COCO_AP(
            self.input_instances,
            self.output_instances,
            np.linspace(0.5, 0.95, 10),
            self.category_keys,
        )
        coco_eval_30 = self._compute_COCO_AP(
            self.input_instances,
            self.output_instances,
            [0.3],
            self.category_keys,
        )

        # whether prediction is empty (no box detected)
        # set them to zeros to avoid errors
        if not np.any(coco_eval_5095.stats):
            coco_eval_5095.stats = [0.0, 0.0, 0.0]
            coco_eval_30.stats = [0.0, 0.0, 0.0]
            coco_eval_5095.eval = {'precision': np.zeros((6, 1, len(self.category_keys), 1, 1))}
            coco_eval_30.eval = {'precision': np.zeros((6, 1, len(self.category_keys), 1, 1))}

        precision5095 = coco_eval_5095.eval['precision']
        precision30 = coco_eval_30.eval['precision']

        res = {
            'AP': coco_eval_5095.stats[0],
            'AP30': coco_eval_30.stats[0],
            'AP50': coco_eval_5095.stats[1],
            'AP75': coco_eval_5095.stats[2],
        }

        # per-class AP
        for category_key in self.category_keys:
            class_id = category_key['id']
            class_name = category_key['name']

            res[f'AP-{class_name}'] = np.mean(precision5095[:, :, class_id, 0, -1])
            res[f'AP30-{class_name}'] = np.mean(precision30[:, :, class_id, 0, -1])
            res[f'AP50-{class_name}'] = np.mean(precision5095[0, :, class_id, 0, -1])
            res[f'AP75-{class_name}'] = np.mean(precision5095[5, :, class_id, 0, -1])

        if self.output_dir:
            file_path = os.path.join(self.output_dir, "obj_evaluation.pth")
            with open(file_path, "wb") as f:
                torch.save({'res': res}, f)

        results = OrderedDict({"bbox": res})
        self.logger.info(results)
        return results

    @staticmethod
    def _compute_COCO_AP(inputs: List, outputs: List, iouThrs: List[float], category_keys: List[Dict]) -> COCOeval:
        """Convert your inputs Instances() (ground truths) and outputs Instances()
        (predictions) to COCO format.
        Args:
            inputs, outputs: list of gt and prediction Instances()
            iouThrs: the IoU threshold that wants to be evaluated on
            category_keys: labels, e.g. [{'id': 0, 'name': 'person'}, {'id': 1, 'name': 'non-person'}]
        Returns:
            coco_eval: The COCOeval object. See https://cocodataset.org/#detection-eval
        """
        gt_annotations = []
        pred_annotations = []
        image_entries = []

        gt_id_cnt = 0
        pred_id_cnt = 0

        for input, output, image_id in zip(inputs, outputs, range(len(inputs))):
            image_entries.append({'id': image_id})

            # For ground truths
            for box, category_id in zip(input.gt_boxes.tensor.tolist(), input.gt_classes.tolist()):
                x1, y1, x2, y2 = box
                gt_annotations.append(
                    {
                        'image_id': image_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'category_id': category_id,
                        'id': gt_id_cnt,
                        'iscrowd': 0,
                        'area': (x2 - x1) * (y2 - y1),
                    }
                )
                gt_id_cnt += 1

            # For predictions
            for box, category_id, score in zip(
                output.pred_boxes.tensor.tolist(),
                output.pred_classes.tolist(),
                output.scores.tolist(),
            ):
                x1, y1, x2, y2 = box
                pred_annotations.append(
                    {
                        'image_id': image_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'category_id': category_id,
                        'score': score,
                        'id': pred_id_cnt,
                    }
                )
                pred_id_cnt += 1

        # if no prediction box, need to return earlier to avoid error (func loadRes)
        if not pred_annotations:
            return COCOeval(iouType='bbox')

        # construct COCO format gt and pred; start eval
        gt_coco = COCO()
        gt_coco.dataset = {
            'annotations': gt_annotations,
            'images': image_entries,
            'categories': category_keys,
        }
        gt_coco.createIndex()
        pred_coco = gt_coco.loadRes(pred_annotations)

        coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
        coco_eval.params.iouThrs = np.asarray(iouThrs)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval


class AdvObjEvaluator(DatasetEvaluator):
    """Advance object detection metrics for model.

    The original detectron2 `COCOEvaluator` reads inputs from a cached file in the
    `inference` folder, which is generated directly from the dataset annotation before
    the dataset mapper.

    In contrast, this evaluator ensures that the metrics are computed from the given
    inputs and outputs, both of which should have an Instances object.

    This evaluator also does advance evaluation, including distance-error analysis
    and precision & recall.

    Metrics include:
        1. mAP[0.5:0.05:0.95], mAP30, mAP50, mAP75
        2. Per-class AP[0.5:0.05:0.95], AP30, AP50, AP75
        3. precision, recall
        4. the above metrics group by distance
    """

    def __init__(self, dataset_name, output_dir=None):
        """
        Args:
            dataset_name: the name of the dataset
            output_dir (str): an output directory to dump results.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        class_labels = MetadataCatalog.get(dataset_name).thing_classes  # list
        self.category_keys = [{'id': idx, 'name': label} for idx, label in enumerate(class_labels)]

        # To store Instances(). They have additional field bbox_dist
        self.input_instances = []
        self.output_instances = []

    def reset(self):
        """
        Reset the internal state to prepare for a new round of evaluation.
        """
        self.input_instances.clear()
        self.output_instances.clear()

    def process(self, inputs, outputs):
        """Process the pair of inputs and outputs.
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input, output in zip(inputs, outputs):
            input_inst = input['instances']
            output_inst = output['instances'].to('cpu')
            gt_depth = torch.clamp(input['depth'].squeeze(), 0, 0.96)  # (64, 512)

            # compute and record the range distances for bbox
            input_dist = [self._box_dist(gt_depth, bbox) for bbox in input_inst.gt_boxes.tensor]
            output_dist = [self._box_dist(gt_depth, bbox) for bbox in output_inst.pred_boxes.tensor]
            input_inst.bbox_dist = torch.tensor(input_dist)  # add a new field
            output_inst.bbox_dist = torch.tensor(output_dist)

            self.input_instances.append(input_inst)
            self.output_instances.append(output_inst)

    def evaluate(self):
        """
        Evaluate the statistics to get the metrics.
        Returns:
            results (OrderedDict): the metrics
        """
        res = {}
        res.update(self.evaluate_at_distance(min_dist=0, max_dist=0.3))
        res.update(self.evaluate_at_distance(min_dist=0.3, max_dist=0.6))
        res.update(self.evaluate_at_distance(min_dist=0.6, max_dist=1.0))
        res.update(self.evaluate_at_distance(min_dist=0, max_dist=1.0))

        if self.output_dir:
            file_path = os.path.join(self.output_dir, "obj_adv_evaluation.pth")
            with open(file_path, "wb") as f:
                torch.save({'res': res}, f)

        results = OrderedDict({"bbox_advance": res})
        self.logger.info(results)
        return results

    def evaluate_at_distance(self, min_dist: float, max_dist: float) -> Dict:
        """
        Evaluate the statistics to get the metrics at a distance range.
        Args:
            min_dist, max_dist: the distance range to evaluate
        Returns:
            res: the metrics
        """
        coco_eval_5095 = self._compute_COCO_AP(
            self.input_instances,
            self.output_instances,
            np.linspace(0.5, 0.95, 10),
            self.category_keys,
            min_dist,
            max_dist,
        )
        coco_eval_30 = self._compute_COCO_AP(
            self.input_instances,
            self.output_instances,
            [0.3],
            self.category_keys,
            min_dist,
            max_dist,
        )

        # whether prediction is empty (no box detected)
        # set them to zeros to avoid errors
        if not np.any(coco_eval_5095.stats):
            coco_eval_5095.stats = [0.0, 0.0, 0.0]
            coco_eval_30.stats = [0.0, 0.0, 0.0]
            coco_eval_5095.eval = {'precision': np.zeros((6, 1, len(self.category_keys), 1, 1))}
            coco_eval_30.eval = {'precision': np.zeros((6, 1, len(self.category_keys), 1, 1))}

        precision5095 = coco_eval_5095.eval['precision']
        precision30 = coco_eval_30.eval['precision']

        suffix = f'{min_dist*10:.0f}to{max_dist*10:.0f}m'
        res = {
            f'AP_{suffix}': coco_eval_5095.stats[0],
            f'AP30_{suffix}': coco_eval_30.stats[0],
            f'AP50_{suffix}': coco_eval_5095.stats[1],
            f'AP75_{suffix}': coco_eval_5095.stats[2],
        }

        # per-class AP
        for category_key in self.category_keys:
            class_id = category_key['id']
            class_name = category_key['name']

            res[f'AP-{class_name}_{suffix}'] = np.mean(precision5095[:, :, class_id, 0, -1])
            res[f'AP30-{class_name}_{suffix}'] = np.mean(precision30[:, :, class_id, 0, -1])
            res[f'AP50-{class_name}_{suffix}'] = np.mean(precision5095[0, :, class_id, 0, -1])
            res[f'AP75-{class_name}_{suffix}'] = np.mean(precision5095[5, :, class_id, 0, -1])

        return res

    @staticmethod
    def _compute_COCO_AP(
        inputs: List,
        outputs: List,
        iouThrs: List[float],
        category_keys: List[Dict],
        min_dist: float,
        max_dist: float,
    ) -> COCOeval:
        """Convert your inputs Instances() (ground truths) and outputs Instances()
        (predictions) to COCO format.
        Args:
            inputs, outputs: list of gt and prediction Instances()
            iouThrs: the IoU threshold that wants to be evaluated on
            category_keys: labels, e.g. [{'id': 0, 'name': 'person'}, {'id': 1, 'name': 'non-person'}]
            min_dist, max_dist: the distance range to evaluate
        Returns:
            coco_eval: The COCOeval object. See https://cocodataset.org/#detection-eval
        """
        gt_annotations = []
        pred_annotations = []
        image_entries = []

        gt_id_cnt = 0
        pred_id_cnt = 0

        for input, output, image_id in zip(inputs, outputs, range(len(inputs))):
            image_entries.append({'id': image_id})

            # For ground truths
            for box, category_id, dist in zip(
                input.gt_boxes.tensor.tolist(), input.gt_classes.tolist(), input.bbox_dist.tolist()
            ):
                if dist < min_dist or dist >= max_dist:
                    continue

                x1, y1, x2, y2 = box
                gt_annotations.append(
                    {
                        'image_id': image_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'category_id': category_id,
                        'id': gt_id_cnt,
                        'iscrowd': 0,
                        'area': (x2 - x1) * (y2 - y1),
                    }
                )
                gt_id_cnt += 1

            # For predictions
            for box, category_id, score, dist in zip(
                output.pred_boxes.tensor.tolist(),
                output.pred_classes.tolist(),
                output.scores.tolist(),
                output.bbox_dist.tolist(),
            ):
                if dist < min_dist or dist >= max_dist:
                    continue

                x1, y1, x2, y2 = box
                pred_annotations.append(
                    {
                        'image_id': image_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'category_id': category_id,
                        'score': score,
                        'id': pred_id_cnt,
                    }
                )
                pred_id_cnt += 1

        # if no prediction box, need to return earlier to avoid error (func loadRes)
        if not pred_annotations:
            return COCOeval(iouType='bbox')

        # construct COCO format gt and pred; start eval
        gt_coco = COCO()
        gt_coco.dataset = {
            'annotations': gt_annotations,
            'images': image_entries,
            'categories': category_keys,
        }
        gt_coco.createIndex()
        pred_coco = gt_coco.loadRes(pred_annotations)

        coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
        coco_eval.params.iouThrs = np.asarray(iouThrs)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval

    @staticmethod
    def _box_dist(gt_depth: torch.Tensor, bbox: torch.Tensor) -> float:
        """Get the range distance of the bbox.
        Args:
            gt_depth: the depth ground truth, shape (H, W)
            bbox: bounding box [x1, y1, x2, y2], shape (4, )
        Returns:
            dist: the range distance of the bbox.
        """
        bbox = torch.round(bbox).to(torch.int32)
        dist = torch.quantile(gt_depth[bbox[1] : bbox[3], bbox[0] : bbox[2]], 0.5)
        return dist


class AdvSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate sub-distance semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn("SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata.")
        if ignore_label is not None:
            self._logger.warn("SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata.")
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )

    def reset(self):
        """
        Preparation for a new round of evaluation.
        """
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)

        # confidence matrices for advanced eval
        self._conf_matrix_0_3 = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._conf_matrix_3_6 = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._conf_matrix_6_10 = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)

        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            # Advanced metrics
            gt_depth = input['depth'].clone()[0].numpy()  # get ground truth depth

            gt_0_3 = gt.copy()
            gt_3_6 = gt.copy()
            gt_6_10 = gt.copy()

            # Ignore all values that are not within the range
            gt_0_3[gt_depth >= 0.3] = self._num_classes
            gt_3_6[np.logical_or(gt_depth < 0.3, gt_depth >= 0.6)] = self._num_classes
            gt_6_10[np.logical_or(gt_depth < 0.6, gt_depth >= 1)] = self._num_classes

            # Update confidence matrices
            self._conf_matrix_0_3 += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt_0_3.reshape(-1),
                minlength=self._conf_matrix_0_3.size,
            ).reshape(self._conf_matrix_0_3.shape)

            self._conf_matrix_3_6 += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt_3_6.reshape(-1),
                minlength=self._conf_matrix_3_6.size,
            ).reshape(self._conf_matrix_3_6.shape)

            self._conf_matrix_6_10 += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt_6_10.reshape(-1),
                minlength=self._conf_matrix_6_10.size,
            ).reshape(self._conf_matrix_6_10.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        Returns:
            dict: standard semantic segmentation metrics
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            conf_matrix_0_3_list = all_gather(self._conf_matrix_0_3)
            conf_matrix_3_6_list = all_gather(self._conf_matrix_3_6)
            conf_matrix_6_10_list = all_gather(self._conf_matrix_6_10)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._conf_matrix_0_3 = np.zeros_like(self._conf_matrix_0_3)
            for conf_matrix in conf_matrix_0_3_list:
                self._conf_matrix_0_3 += conf_matrix

            self._conf_matrix_3_6 = np.zeros_like(self._conf_matrix_3_6)
            for conf_matrix in conf_matrix_3_6_list:
                self._conf_matrix_3_6 += conf_matrix

            self._conf_matrix_6_10 = np.zeros_like(self._conf_matrix_6_10)
            for conf_matrix in conf_matrix_6_10_list:
                self._conf_matrix_6_10 += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "advanced_sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU_all"] = 100 * miou
        res["fwIoU_all"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}_all"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC_all"] = 100 * macc
        res["pACC_all"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}_all"] = 100 * acc[i]

        # Evaluate advanced metrics
        # 0-3m
        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix_0_3.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix_0_3[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix_0_3[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res["mIoU_0to3m"] = 100 * miou
        res["fwIoU_0to3m"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}_0to3m"] = 100 * iou[i]

        res["mACC_0to3m"] = 100 * macc
        res["pACC_0to3m"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}_0to3m"] = 100 * acc[i]

        # 3-6m
        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix_3_6.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix_3_6[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix_3_6[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res["mIoU_3to6m"] = 100 * miou
        res["fwIoU_3to6m"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}_3to6m"] = 100 * iou[i]

        res["mACC_3to6m"] = 100 * macc
        res["pACC_3to6m"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}_3to6m"] = 100 * acc[i]

        # 6-10m
        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix_6_10.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix_6_10[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix_6_10[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res["mIoU_6to10m"] = 100 * miou
        res["fwIoU_6to10m"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}_6to10m"] = 100 * iou[i]

        res["mACC_6to10m"] = 100 * macc
        res["pACC_6to10m"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}_6to10m"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "advanced_sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"advanced_sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert label in self._contiguous_id_to_dataset_id, "Label {} is not in the metadata info for {}".format(
                    label, self._dataset_name
                )
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append({"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle})
        return json_list

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        """
        See https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/sem_seg_evaluation.html
        """
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
