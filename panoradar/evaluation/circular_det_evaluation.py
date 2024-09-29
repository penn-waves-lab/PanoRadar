import os
import logging
import torch
import numpy as np

from typing import OrderedDict, List, Dict

from panoradar.structures.circular_boxes import CircularBoxes, circular_pairwise_iou

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures.boxes import BoxMode

__all__ = ["CircularObjEvaluator", "AdvCircularObjEvaluator"]

class CircularBoxMode:
    @staticmethod
    def convert(box, from_mode: "BoxMode", to_mode: "BoxMode"):
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
            arr[:, 2] = (arr[:, 2] + arr[:, 0]) % 512
            arr[:, 3] += arr[:, 1]
        elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
            arr[:, 2] = torch.where(
                arr[:, 2] < arr[:, 0], arr[:, 2] + 512 - arr[:, 0], arr[:, 2] - arr[:, 0]
            )
            arr[:, 3] -= arr[:, 1]
        else:
            raise NotImplementedError(
                "Conversion from BoxMode {} to {} is not supported yet".format(from_mode, to_mode)
            )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class CircularCOCOeval(COCOeval):
    def boxlist_to_tensor(self, boxlist, output_box_dim):
        if type(boxlist) == np.ndarray:
            box_tensor = torch.from_numpy(boxlist)
        elif type(boxlist) == list:
            if boxlist == []:
                return torch.zeros((0, output_box_dim), dtype=torch.float32)
            else:
                box_tensor = torch.FloatTensor(boxlist)
        else:
            raise Exception()

        return box_tensor

    def compute_iou_dt_gt(self, dt, gt, is_crowd):
        dt = CircularBoxes(self.boxlist_to_tensor(dt, output_box_dim=4))
        gt = CircularBoxes(self.boxlist_to_tensor(gt, output_box_dim=4))
        return circular_pairwise_iou(dt, gt)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        assert p.iouType == "bbox", "unsupported iouType for iou computation"

        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]

        # Note: this function is copied from cocoeval.py in cocoapi
        # and the major difference is here.
        ious = self.compute_iou_dt_gt(d, g, iscrowd)
        return ious


class CircularObjEvaluator(DatasetEvaluator):
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
        """Evaluate the statistics to get the metrics."""
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
            'mAP': coco_eval_5095.stats[0],
            'mAP30': coco_eval_30.stats[0],
            'mAP50': coco_eval_5095.stats[1],
            'mAP75': coco_eval_5095.stats[2],
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
    def _compute_COCO_AP(
        inputs: List, outputs: List, iouThrs: List[float], category_keys: List[Dict]
    ) -> COCOeval:
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
                        'bbox': [x1, y1, x2, y2],
                        'category_id': category_id,
                        'id': gt_id_cnt,
                        'iscrowd': 0,
                        'area': (x2 - x1) * (y2 - y1) if x2 > x1 else (x2 + 512 - x1) * (y2 - y1),
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
                        'bbox': [x1, y1, x2, y2],
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

        coco_eval = CircularCOCOeval(gt_coco, pred_coco, 'bbox')
        coco_eval.params.iouThrs = np.asarray(iouThrs)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval


class AdvCircularObjEvaluator(DatasetEvaluator):
    """Advance circular object detection metrics for model.

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
        """Evaluate the statistics to get the metrics."""
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
        """Evaluate the statistics to get the metrics at a distance range."""
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
                        'bbox': [x1, y1, x2, y2],
                        'category_id': category_id,
                        'id': gt_id_cnt,
                        'iscrowd': 0,
                        'area': (x2 - x1) * (y2 - y1) if x2 > x1 else (x2 + 512 - x1) * (y2 - y1),
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
                        'bbox': [x1, y1, x2, y2],
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

        coco_eval = CircularCOCOeval(gt_coco, pred_coco, 'bbox')
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
        if bbox[0] < bbox[2]:
            dist = torch.quantile(gt_depth[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1], 0.5)
        else:
            dist = torch.quantile(
                torch.cat([gt_depth[bbox[1] : bbox[3] + 1, bbox[0] : 512], gt_depth[bbox[1] : bbox[3] + 1, 0 : bbox[2] + 1]], dim=1), 0.5
            ) 
        return dist
