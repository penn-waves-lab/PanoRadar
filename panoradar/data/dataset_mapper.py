import cv2
import copy
import torch
import numpy as np

from typing import List, Dict, Tuple

from detectron2.data.detection_utils import annotations_to_instances

from .utils import sn, get_smooth_surface_normal

__all__ = ["LidarTwoTasksMapper", "RfFourTasksMapper"]

# ==========================================================
# =====================  Augmentation  =====================
# ==========================================================
def crop_and_resize(
    image: np.ndarray,
    annos: List[Dict],
    sem_seg: np.ndarray,
    glass_seg: np.ndarray,
    max_crop_length=(16, 16),
    drop_box_thres=(5, 8),
    crop_and_resize_p=0.5,
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """Crop and resize augmentation.
    NOTE: this function should be called after the wrap_bbox() function.
    Args:
        image: RGB or Lidar npy image, shape (C, H, W)
        annos: the annotation dict in detectron2 format
        sem_seg: semantic segmentation npy image, shape (H, W)
        glass_seg: glass segmentation npy image, shape (H, W)
        max_crop_length: the length of the crop, (half height, half width)
        drop_box_thres: The threshold (height, width) for dropping a box after cropping
        crop_and_resize_p: the probability for crop and resize
    Returns:
        image_aug, annos_aug, sem_seg_aug: the augmented image, annotation and segmentation
    """
    if np.random.rand() < crop_and_resize_p:
        H, W = image.shape[1:]
        crop_offset_h0 = np.random.randint(0, max_crop_length[0] + 1)
        crop_offset_w0 = np.random.randint(0, max_crop_length[1] + 1)
        crop_offset_h1 = H - crop_offset_h0
        crop_offset_w1 = W - crop_offset_w0

        # image cropping and resize
        image = image[:, crop_offset_h0:crop_offset_h1, crop_offset_w0:crop_offset_w1]
        image = cv2.resize(image.transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)  # back to (C, H, W)

        # segmentation cropping and resize
        sem_seg = sem_seg[crop_offset_h0:crop_offset_h1, crop_offset_w0:crop_offset_w1]
        sem_seg = cv2.resize(sem_seg, (W, H), interpolation=cv2.INTER_NEAREST)
        glass_seg = glass_seg[crop_offset_h0:crop_offset_h1, crop_offset_w0:crop_offset_w1]
        glass_seg = cv2.resize(glass_seg, (W, H), interpolation=cv2.INTER_NEAREST)

        # deal with boxes cropping and resize
        annos_aug = []
        h_ratio = H / (crop_offset_h1 - crop_offset_h0)
        w_ratio = W / (crop_offset_w1 - crop_offset_w0)
        for anno in annos:
            x0, y0, x1, y1 = anno['bbox']
            if (
                y1 < crop_offset_h0 + drop_box_thres[0]
                or y0 >= crop_offset_h1 - drop_box_thres[0]
                or x1 < crop_offset_w0 + drop_box_thres[1]
                or x0 >= crop_offset_w1 - drop_box_thres[1]
            ):
                continue

            anno['bbox'] = [
                (x0 - crop_offset_w0) * w_ratio,
                (y0 - crop_offset_h0) * h_ratio,
                (x1 - crop_offset_w0) * w_ratio,
                (y1 - crop_offset_h0) * h_ratio,
            ]
            annos_aug.append(anno)
        annos = annos_aug

    return image, annos, sem_seg, glass_seg


def jitter_image(image: np.ndarray, mean=0.0, std=0.003, jitter_p=0.5) -> np.ndarray:
    """Jitter the image (or depth lidar npy) with Gaussian noise.
    Args:
        image: RGB or Lidar npy image, shape (C, H, W)
        mean, std: the mean and standard deviation for the Guassian distribution
        jitter_p: the probability for jittering
    Returns:
        image_aug: the augmented image
    """
    if np.random.rand() < jitter_p:
        jitter = np.random.randn(*image.shape) * std + mean
        image = image + jitter
    return image


def scaling_transform(image: np.ndarray, scale_range=(0.8, 1.2), scaling_p=0.5) -> np.ndarray:
    """Scale transform for the image (or depth lidar npy).
    Args:
        image: RGB or Lidar npy image, shape (C, H, W)
        scale_range: the (min, max) ratio for scaling
        scaling_p: the probability for the scaling
    Returns:
        image_aug: the augmented image
    """
    if np.random.rand() < scaling_p:
        scale = np.random.uniform(*scale_range)
        image = image * scale
    return image


def wrap_bbox(annos: List[Dict], width: int, circular: bool = False) -> List[Dict]:
    """Wrap the bounding box coordinate to be within the azimuth size.
    Args:
        annos: the annotation dict in detectron2 format
    Returns:
        annos_wrap: the annotation dict after wrapping the bbox
    """
    annos_wrap = []
    for anno in annos:
        x0_, y0_, x1_, y1_ = anno['bbox']
        x0 = int(min(x0_, x1_)) % width
        x1 = int(max(x0_, x1_)) % width
        y0 = int(min(y0_, y1_))
        y1 = int(max(y0_, y1_))

        if circular:
            # If x1 is over 512 then keep it that way for now, wrapping taken care of in model
            # COMMENT OUT HERE IF SWITCH OFF
            if x1 < x0:
                x1 = x1 + width
            anno_warp = copy.deepcopy(anno)
            anno_warp['bbox'] = [x0, y0, x1, y1]
            annos_wrap.append(anno_warp)
            # END COMMENT OUT
        else:
            if x0 > x1:
                if x0 < width - 1:
                    anno_warp = copy.deepcopy(anno)
                    anno_warp['bbox'] = [x0, y0, width - 1, y1]
                    annos_wrap.append(anno_warp)
                if x1 > 0:
                    anno_warp = copy.deepcopy(anno)
                    anno_warp['bbox'] = [0, y0, x1, y1]
                    annos_wrap.append(anno_warp)
            else:
                anno_warp = copy.deepcopy(anno)
                anno_warp['bbox'] = [x0, y0, x1, y1]
                annos_wrap.append(anno_warp)

    return annos_wrap


def mix_after_first_reflection(
    image0: np.ndarray, depth0: np.ndarray, image1: np.ndarray, jitter_p: float = 0.5
) -> np.ndarray:
    """Find the range bin of the first reflection from lidar.
    Then jitter the values in range bins after it.
    Args:
        image0: rf data to be augmented, (256, 64, 512)
        depth0: the lidar depth data, (1, 64, 512)
        image1: rf data used to augment image0, (256, 64, 512)
        jitter_p: the probability to do the jitter augmentation
    Return:
        refl_data: The data for mixing after the first reflection
    """
    y_per_bin = 0.003747
    guard_bin = 3

    if np.random.rand() < jitter_p:
        C, H, W = image0.shape
        start_bin = depth0 / y_per_bin + guard_bin + np.random.randint(-2, 3)  # (1,H,W)
        start_bin[start_bin < 0] = C  # don't jitter failure and glass region
        mask = np.arange(0, C, dtype=np.float32).reshape(-1, 1, 1)
        mask = np.tile(mask, (1, H, W)) > start_bin
        image0 = image0 + (image1 - image0) * mask * 0.5

    return image0


# ==========================================================
# ========================  Mapper  ========================
# ==========================================================
class LidarTwoTasksMapper:
    def __init__(self, cfg, is_train: bool):
        """Map the lidar npy files to model compatible format (ddd, 0~255).
        The two tasks are: semantic seg and obj detection.

        The callable currently does the following:
        1. Read the image from "file_name"
        2. Applies augmentation to the image and annotations
        3. Prepare data and annotations to Tensor and :class:`Instances`

        Args:
            cfg: the config object, CfgNode
            is_train: whether it's for training, control augmentation
        """
        self.cfg = cfg
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT

        self.is_train = is_train
        self.need_rotate_aug = cfg.INPUT.ROTATE.ENABLED
        self.rotate_p = cfg.INPUT.ROTATE.ROTATE_P
        self.hflip_p = cfg.INPUT.ROTATE.HFLIP_P

        self.need_crop_resize_aug = cfg.INPUT.CROP_AND_RESIZE.ENABLED
        self.crop_length = cfg.INPUT.CROP_AND_RESIZE.CROP_LENGTH
        self.drop_box_thres = cfg.INPUT.CROP_AND_RESIZE.DROP_BOX_THRES
        self.crop_and_resize_p = cfg.INPUT.CROP_AND_RESIZE.CROP_AND_RESIZE_P

        self.need_scaling = cfg.INPUT.SCALE_TRANSFORM.ENABLED
        self.scale_range = cfg.INPUT.SCALE_TRANSFORM.SCALE_RANGE
        self.scale_p = cfg.INPUT.SCALE_TRANSFORM.SCALE_P

        self.need_jitter_aug = cfg.INPUT.JITTER.ENABLED
        self.jitter_mean = cfg.INPUT.JITTER.MEAN
        self.jitter_std = cfg.INPUT.JITTER.STD
        self.jitter_p = cfg.INPUT.JITTER.JITTER_P

    def __call__(self, dataset_dict):
        """Do the mapping for the dataset_dict
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # load data, image of float32, sem_seg of uint8
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = np.tile(np.load(dataset_dict['depth_file_name']), (3, 1, 1))  # (3,H=64,W=512)
        sem_seg = np.squeeze(np.load(dataset_dict["sem_seg_file_name"]))  # shape (H=64,W=512)
        glass_seg = np.squeeze(np.load(dataset_dict["glass_file_name"])).astype(np.uint8)  # 64,512
        annos = dataset_dict['annotations']  # List of Dict
        dataset_dict['fail_region'] = image[0] < 0  # for vis

        # deal with augmentation
        if self.need_rotate_aug and self.is_train:
            image, annos, sem_seg, glass_seg = self.rotate_and_hflip(
                image, annos, sem_seg, glass_seg, self.rotate_p, self.hflip_p
            )
        annos = wrap_bbox(annos, dataset_dict['width'], self.cfg.MODEL.CIRCULAR_SEG_OBJ)

        if self.need_crop_resize_aug and self.is_train:
            image, annos, sem_seg, glass_seg = crop_and_resize(
                image,
                annos,
                sem_seg,
                glass_seg,
                self.crop_length,
                self.drop_box_thres,
                self.crop_and_resize_p,
            )

        if self.need_scaling and self.is_train:
            image = scaling_transform(image, self.scale_range, self.scale_p)

        if self.need_jitter_aug and self.is_train:
            image = jitter_image(image, self.jitter_mean, self.jitter_std, self.jitter_p)

        # save to the dataset dict
        image = np.clip(image, 0, a_max=None) * 255
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image, np.float32))
        dataset_dict["sem_seg"] = torch.as_tensor(np.ascontiguousarray(sem_seg, np.int64))
        dataset_dict["glass_seg"] = torch.as_tensor(np.ascontiguousarray(glass_seg, np.int64))
        dataset_dict['instances'] = annotations_to_instances(
            annos,
            (dataset_dict['height'], dataset_dict['width']),
            mask_format=self.instance_mask_format,
        )

        return dataset_dict

    @staticmethod
    def rotate_and_hflip(
        image: np.ndarray,
        annos: List[Dict],
        sem_seg: np.ndarray,
        glass_seg: np.ndarray,
        rotate_p=1.0,
        hflip_p=0.5,
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray, np.ndarray]:
        """Rotation and horizontal flip augmentation.
        Args:
            image: RGB or Lidar npy image, shape (C, H, W)
            annos: the annotation dict in detectron2 format
            sem_seg: semantic segmentation npy image, shape (H, W)
            rotate_p, flip_p: the probabilities for rotation and hflip
        Returns:
            image_aug, annos_aug: the augmented image and annotation
        """
        WIDTH = image.shape[-1]

        if np.random.rand() < rotate_p:
            rot_ind = int(WIDTH * np.random.rand())
            image = np.concatenate((image[:, :, rot_ind:], image[:, :, :rot_ind]), axis=-1)
            sem_seg = np.concatenate((sem_seg[:, rot_ind:], sem_seg[:, :rot_ind]), axis=-1)
            glass_seg = np.concatenate((glass_seg[:, rot_ind:], glass_seg[:, :rot_ind]), axis=-1)
            for anno in annos:
                x0, y0, x1, y1 = anno['bbox']
                anno['bbox'] = [x0 - rot_ind, y0, x1 - rot_ind, y1]

        if np.random.rand() < hflip_p:
            image = np.flip(image, axis=-1)
            sem_seg = np.flip(sem_seg, axis=-1)
            glass_seg = np.flip(glass_seg, axis=-1)
            for anno in annos:
                x0, y0, x1, y1 = anno['bbox']
                anno['bbox'] = [WIDTH - 1 - x0, y0, WIDTH - 1 - x1, y1]

        return image, annos, sem_seg, glass_seg


class RfFourTasksMapper:
    def __init__(self, cfg, is_train: bool):
        """Load and map the rf npy files.
        The four tasks are: depth, surface normal, semantic seg, obj detection

        The callable currently does the following:
        1. Read the image from "file_name"
        2. Applies augmentation to the image and annotations
        3. Prepare data and annotations to Tensor and :class:`Instances`

        Args:
            cfg: the config object, CfgNode
            is_train: whether it's for training, control augmentation
        """
        self.cfg = cfg
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT

        self.is_train = is_train
        self.need_rotate_aug = cfg.INPUT.ROTATE.ENABLED
        self.rotate_p = cfg.INPUT.ROTATE.ROTATE_P
        self.hflip_p = cfg.INPUT.ROTATE.HFLIP_P

        self.need_jitter_aug = cfg.INPUT.JITTER.ENABLED
        self.jitter_mean = cfg.INPUT.JITTER.MEAN
        self.jitter_std = cfg.INPUT.JITTER.STD
        self.jitter_p = cfg.INPUT.JITTER.JITTER_P

        self.need_first_refl_aug = cfg.INPUT.FIRST_REFL.ENABLED
        self.first_refl_p = cfg.INPUT.FIRST_REFL.JITTER_P

        self.prev_data = None  # for mix_after_first_reflection

    def __call__(self, dataset_dict):
        """Do the mapping for the dataset_dict
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # load data, image of float32, sem_seg of uint8
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = np.load(dataset_dict['file_name'])  # rf (256,H=64,W=512)
        sem_seg = np.squeeze(np.load(dataset_dict["sem_seg_file_name"]))  # shape (H=64,W=512)
        depth = np.load(dataset_dict['depth_file_name'])  # shape (1,H=64,W=512)
        glass = np.load(dataset_dict['glass_file_name'])  # shape (1,H=64,W=512)
        annos = dataset_dict['annotations']  # List of Dict

        depth[glass] = -1e3
        normal, _ = get_smooth_surface_normal(sn, depth.squeeze())
        normal = sn.get_rhs_from_normal(normal, True).astype(np.float32)
        normal = normal.transpose(2, 0, 1)  # (3, 64, 512)

        # deal with augmentation
        if self.need_rotate_aug and self.is_train:
            image, annos, sem_seg, depth, normal = self.rotate_and_hflip(
                image, annos, sem_seg, depth, normal, self.rotate_p, self.hflip_p
            )
        annos = wrap_bbox(annos, dataset_dict['width'], self.cfg.MODEL.CIRCULAR_SEG_OBJ)

        if self.need_first_refl_aug and self.is_train:
            if self.prev_data is None:
                self.prev_data = image  # store it initially
            else:
                image_ = mix_after_first_reflection(image, depth, self.prev_data, self.first_refl_p)
                self.prev_data = image
                image = image_

        if self.need_jitter_aug and self.is_train:
            image = jitter_image(image, self.jitter_mean, self.jitter_std, self.jitter_p)

        # save to the dataset dict
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image, np.float32))
        dataset_dict["sem_seg"] = torch.as_tensor(np.ascontiguousarray(sem_seg, np.int64))
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth, np.float32))
        dataset_dict["sn"] = torch.as_tensor(np.ascontiguousarray(normal, np.float32))
        dataset_dict['annotations'] = annos
        dataset_dict['instances'] = annotations_to_instances(
            annos,
            (dataset_dict['height'], dataset_dict['width']),
            mask_format=self.instance_mask_format,
        )

        return dataset_dict

    @staticmethod
    def rotate_and_hflip(
        image: np.ndarray,
        annos: List[Dict],
        sem_seg: np.ndarray,
        depth: np.ndarray,
        normal: np.ndarray,
        rotate_p=1.0,
        hflip_p=0.5,
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray, np.ndarray, np.ndarray]:
        """Rotation and horizontal flip augmentation.
        Args:
            image: RGB or Lidar npy image, shape (C, H, W)
            annos: the annotation dict in detectron2 format
            sem_seg: semantic segmentation npy image, shape (H, W)
            depth: depth map, shape (1, H, W)
            normal: surface normal, shape (3, H, W)
            rotate_p, flip_p: the probabilities for rotation and hflip
        Returns:
            image_aug, annos_aug: the augmented image and annotation
        """
        WIDTH = image.shape[-1]

        if np.random.rand() < rotate_p:
            rot_ind = int(WIDTH * np.random.rand())
            image = np.concatenate((image[:, :, rot_ind:], image[:, :, :rot_ind]), axis=-1)
            sem_seg = np.concatenate((sem_seg[:, rot_ind:], sem_seg[:, :rot_ind]), axis=-1)
            depth = np.concatenate((depth[:, :, rot_ind:], depth[:, :, :rot_ind]), axis=-1)
            normal = np.concatenate((normal[:, :, rot_ind:], normal[:, :, :rot_ind]), axis=-1)
            for anno in annos:
                x0, y0, x1, y1 = anno['bbox']
                anno['bbox'] = [x0 - rot_ind, y0, x1 - rot_ind, y1]

        if np.random.rand() < hflip_p:
            image = np.flip(image, axis=-1)
            sem_seg = np.flip(sem_seg, axis=-1)
            depth = np.flip(depth, axis=-1)
            normal = np.flip(normal, axis=-1)
            for anno in annos:
                x0, y0, x1, y1 = anno['bbox']
                anno['bbox'] = [WIDTH - 1 - x0, y0, WIDTH - 1 - x1, y1]

        return image, annos, sem_seg, depth, normal
