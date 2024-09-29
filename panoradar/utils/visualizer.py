import cv2
import torch
import open3d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image

from detectron2.utils.visualizer import (
    Visualizer,
    _create_text_labels,
    ColorMode,
    GenericMask,
)
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode

from panoradar_SP.visualization import project_polar_to_cartesian

from ..data import sn, normal_color_mapping
from .visualization_utils import (
    visualize_depth_pred_with_seg,
    visualize_lidar_gt_with_seg,
    visualize_rf_heatmap_with_cfar,
)

CAM_PARAMS = open3d.io.read_pinhole_camera_parameters("panoradar/utils/open3d_camera_params.json")


class OBJVisualizer(Visualizer):
    """
    Slightly modified version of detectron2.utils.Visualizer class,
    changed so that boxes of each class will have the same colour,
    and wrapped boxes will be drawn as one box
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define image width
        self.img_width = self.img.shape[1]  # img is (H, W, C)
        self.color_palette = [[0, 106, 216], [0, 113, 31], [255, 255, 255]]

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        # This is the only change, remove jitter from the colors and select from the palette if no thing_colors
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [[x / 255 for x in self.metadata.thing_colors[c]] for c in classes]
            alpha = 0.8
        else:
            colors = [[x / 255 for x in self.color_palette[clss]] for clss in classes]  # Get colors from color palette
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy() if predictions.has("pred_masks") else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_dataset_dict(self, dic):
        """
        Draw ground truth annotations/segmentations in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) if len(x["bbox"]) == 4 else x["bbox"]
                for x in annos
            ]

            colors = None
            category_ids = [x["category_id"] for x in annos]
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                # Change here remove jitter
                colors = [[x / 255 for x in self.metadata.thing_colors[c]] for c in category_ids]
            else:
                colors = [[x / 255 for x in self.color_palette[clss]] for clss in category_ids]
            names = self.metadata.get("thing_classes", None)
            labels = _create_text_labels(
                category_ids,
                scores=None,
                class_names=names,
                is_crowd=[x.get("iscrowd", 0) for x in annos],
            )
            self.overlay_instances(labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors)

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype="uint8")
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)

        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """

        # Change draw box function so that boxes that wrap around are drawn properly and as one
        x0, y0, x1, y1 = box_coord
        height = y1 - y0
        x1 = x1 % self.img_width
        linewidth = max(self._default_font_size / 4, 1)

        if x1 < x0:
            # Box wraps around, so draw two rectangles in matplotlib
            width = self.img_width - x0 + 10  # Draw first rectangle to img_width+10 so the right edge does not show

            # First draw the box on the right
            self.output.ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0),
                    width,
                    height,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=linewidth * self.output.scale,
                    alpha=alpha,
                    linestyle=line_style,
                )
            )

            # Draw second box
            secondwidth = x1 + 10  # Add 10 since drawing from -10
            self.output.ax.add_patch(
                mpl.patches.Rectangle(
                    (-10, y0),
                    secondwidth,
                    height,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=linewidth * self.output.scale,
                    alpha=alpha,
                    linestyle=line_style,
                )
            )
        else:
            # Otherwise just draw one box
            width = x1 - x0

            self.output.ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0),
                    width,
                    height,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=linewidth * self.output.scale,
                    alpha=alpha,
                    linestyle=line_style,
                )
            )
        return self.output


def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
    """
    Modified version of detectron2.utils.Visualizer.draw_sem_seg function,
    changed so that removed the thick OFF_WHITE edge color.

    Args:
        sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
            Each value is the integer label of the pixel.
        area_threshold (int): segments with less than `area_threshold` are not drawn.
        alpha (float): the larger it is, the more opaque the segmentations are.

    Returns:
        output (VisImage): image object with visualizations.
    """
    if isinstance(sem_seg, torch.Tensor):
        sem_seg = sem_seg.numpy()
    labels, areas = np.unique(sem_seg, return_counts=True)
    sorted_idxs = np.argsort(-areas).tolist()
    labels = labels[sorted_idxs]
    for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
        try:
            mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
        except (AttributeError, IndexError):
            mask_color = None

        binary_mask = (sem_seg == label).astype(np.uint8)
        text = self.metadata.stuff_classes[label]
        self.draw_binary_mask(
            binary_mask,
            color=mask_color,
            edge_color=None,
            text=text,
            alpha=alpha,
            area_threshold=area_threshold,
        )
    return self.output


Visualizer.draw_sem_seg = draw_sem_seg


def draw_vis_image(
    input_dict,
    preds,
    dataset_metadata,
    depth_only: bool = False,
    return_rgb: bool = True,
):
    """draw the image for visualization.
    Args:
        input_dict: the input dictionary, not a list
        preds: the output predictions dictionary from the model, not a list
        dataset_metadata: the metadata from `MetadataCatalog.get`
        depth_only: If true, only draw depth estimation.
        return_rgb: If true, return a rgb image for tensorboard
    Returns:
        numpy.ndarray (3, H, W): the image for visualization if return_rgb is True
        (matplotlib.pyplot.Figure, numpy.ndarray): the figure and axes if return_rgb is False
    """

    # prepare the lidar Magma colormap
    gt_depth = input_dict['depth'].cpu().squeeze().numpy()  # (64, 512)
    bg_img = (np.clip(gt_depth, 0, 0.96) * 255).astype(np.uint8)
    bg_img = cv2.applyColorMap(-bg_img, cv2.COLORMAP_MAGMA)[:, :, ::-1]
    bg_img[gt_depth <= 0] = [255, 255, 255]

    # prepare pred depth Magma colormap
    pred_depth = preds['depth'].cpu().squeeze().numpy()  # (64, 512)
    pred_bg_img = (np.clip(pred_depth, 0, 0.96) * 255).astype(np.uint8)
    pred_bg_img = cv2.applyColorMap(-pred_bg_img, cv2.COLORMAP_MAGMA)[:, :, ::-1]
    pred_bg_img[pred_depth <= 0] = [255, 255, 255]

    # draw depth estimation
    st_id = 0
    rows = 3 if depth_only else 10
    fig, ax = plt.subplots(
        rows,
        2,
        gridspec_kw={'width_ratios': [4, 1]},
        figsize=(8, 2 * rows),
    )

    _plot_depth(
        input_dict['image'].cpu().permute(1, 0, 2).numpy(),
        preds['depth'].detach().cpu().squeeze().numpy(),
        gt_depth,
        ax[:3],
    )
    st_id += 3

    if not depth_only:
        _plot_surface_normal(
            preds['sn'].detach().cpu().permute(1, 2, 0).numpy(),
            input_dict['sn'].cpu().permute(1, 2, 0).numpy(),
            ax[3:5],
        )
        st_id += 2

        _plot_semantic_seg(
            bg_img,
            pred_bg_img,
            preds['sem_seg'].to("cpu").numpy(),
            input_dict['sem_seg'],
            dataset_metadata,
            ax[st_id : st_id + 3],
            confidence_threshold=0.5,
        )
        st_id += 3

        _plot_obj_detection(
            bg_img,
            pred_bg_img,
            preds['instances'].to('cpu'),
            input_dict,
            dataset_metadata,
            ax[st_id : st_id + 2],
        )

    fig.tight_layout()

    if return_rgb:
        fig.canvas.draw()
        whole_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        whole_img = whole_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        whole_img = whole_img.transpose(2, 0, 1)  # (3, H, W)
        plt.close()
        return whole_img
    else:
        return fig, ax


def draw_range_image(
    input_dict,
    preds,
    return_rgb: bool = True,
):
    """draw the range image frontal and pointcloud view for visualization.
    Args:
        input_dict: the input dictionary, not a list
        preds: the output predictions dictionary from the model, not a list
        return_rgb: If true, return a rgb image for tensorboard
    Returns:
        numpy.ndarray (3, H, W): the image for visualization if return_rgb is True
        (matplotlib.pyplot.Figure, numpy.ndarray): the figure and axes if return_rgb is False
    """

    # prepare the lidar Magma colormap
    gt_depth = input_dict['depth'].cpu().numpy()  # (64, 512)

    # prepare pred depth Magma colormap
    pred_depth = preds['depth'].cpu().numpy()  # (64, 512)

    # draw depth estimation
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(7, 12))

    # Visualize the 3D RF Heatmaps peak finding
    rf = np.load(input_dict['file_name'])

    gt_depth_copy = gt_depth.copy().squeeze()
    mask = gt_depth_copy > 0
    gt_depth_copy[mask == 0] = np.nan

    ax[0].imshow(-pred_depth.squeeze(), aspect='auto', cmap='magma', vmin=-0.8, vmax=-0.05)
    ax[0].axis('off')
    ax[0].text(
        0.99,
        0.95,
        'After ML',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[0].transAxes,
        color='black',
        fontsize=20,
    )
    ax[0].set_title('Range Images', fontsize=25, weight='bold')
    ax[1].imshow(-gt_depth_copy, aspect='auto', cmap='magma', vmin=-0.8, vmax=-0.05)
    ax[1].axis('off')
    ax[1].text(
        0.99,
        0.95,
        'LiDAR',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[1].transAxes,
        color='black',
        fontsize=20,
    )

    # prepare the lidar Magma colormap
    gt_seg = input_dict['sem_seg'].cpu().unsqueeze(0).numpy()  # (1, 64, 512)

    # prepare pred depth Magma colormap
    pred_seg = preds['sem_seg'].cpu().numpy()  # (11, 64, 512)
    pred_seg = np.argmax(pred_seg, axis=0, keepdims=True)  # (1, 64, 512)

    rf_vis = visualize_rf_heatmap_with_cfar(rf, CAM_PARAMS)
    gt_vis = visualize_lidar_gt_with_seg(gt_depth, gt_seg, CAM_PARAMS)
    pred_vis = visualize_depth_pred_with_seg(pred_depth, pred_seg, CAM_PARAMS)

    ax[2].imshow(rf_vis, aspect='auto')
    ax[2].axis('off')
    ax[2].text(
        0.99,
        0.95,
        'After SP',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[2].transAxes,
        color='black',
        fontsize=20,
    )
    ax[2].set_title('Point Clouds', fontsize=25, weight='bold')
    ax[3].imshow(pred_vis, aspect='auto')
    ax[3].axis('off')
    ax[3].text(
        0.99,
        0.95,
        'After ML',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[3].transAxes,
        color='black',
        fontsize=20,
    )
    ax[4].imshow(gt_vis, aspect='auto')
    ax[4].axis('off')
    ax[4].text(
        0.99,
        0.95,
        'LiDAR',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[4].transAxes,
        color='black',
        fontsize=20,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)

    if return_rgb:
        fig.canvas.draw()
        whole_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        whole_img = whole_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        whole_img = whole_img.transpose(2, 0, 1)  # (3, H, W)
        plt.close()
        return whole_img
    else:
        return fig, ax


def _plot_depth(
    rf_raw: np.ndarray,
    pred_depth: np.ndarray,
    y_depth: np.ndarray,
    ax: np.ndarray,
):
    """
    Private function for plotting depth result, including
    1 (left) beamforming result at z=0 plane, polar coordinate (range-azimuth)
    1 (right) beamforming result at z=0 plane, Cartesian coordinate (x-y)
    2 (left) LiDAR range image, polar coordinate (elevation-azimuth), range is the pixel value
    2 (right) LiDAR result at z=0 plane, Cartesian coordinate (x-y)
    3 (left) ML predicted range image, polar coordinate (elevation-azimuth), range is the pixel value
    3 (right) ML predicted result at z=0 plane, Cartesian coordinate (x-y)
    Args:
        rf_raw: the raw radar data, shape (256, 64, 512)
        pred_depth: the predicted depth, shape (64, 512)
        y_depth: the ground truth depth, shape (64, 512)
        ax: the axes for plotting
    """
    N_rings = rf_raw.shape[1]
    N_beams = rf_raw.shape[2]
    N_upsample_beams = N_beams * 6
    r_rings = 0.037474 * np.arange(N_rings)
    mask = y_depth > 0

    # transform to 2D polar coordinate floor plan
    thetas = np.linspace(0, -2 * np.pi, N_beams)
    valid_inds = (mask == 1)[31]
    lidar_slice = y_depth[31].copy()
    pred_slice = pred_depth[31].copy()
    lidar_slice[lidar_slice > 0.96] = 2.0  # out of range
    pred_slice[pred_slice > 0.96] = 2.0  # out of range
    # transform 2D polar to Cartesian
    pc_x = lidar_slice * np.cos(thetas) * 10
    pc_y = lidar_slice * np.sin(thetas) * 10
    pc_x, pc_y = pc_y[valid_inds], -pc_x[valid_inds]
    out_x = pred_slice * np.cos(thetas) * 10
    out_y = pred_slice * np.sin(thetas) * 10
    out_x, out_y = out_y[valid_inds], -out_x[valid_inds]

    # transform 2D polar to Cartesian
    img_x = cv2.resize(rf_raw[31], (N_upsample_beams, N_rings))  # width, height
    img_x = project_polar_to_cartesian(
        img_x,
        r_rings,
        0.96 * 10,
        0.04,
        np.linspace(0, -2 * np.pi, N_upsample_beams),
        default_value=np.nan,
        rotation_offset=-np.pi / 2,
    )

    y_depth = y_depth.copy()
    y_depth[mask == 0] = np.nan

    ax[0, 0].imshow(rf_raw[31], cmap='jet', origin='lower', vmin=0.1, vmax=0.5, aspect='auto')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(img_x, origin='lower', vmin=-0.1, vmax=0.5, cmap='jet')
    ax[0, 1].axis('off')
    ax[1, 0].imshow(-y_depth, aspect='auto', cmap='magma', vmin=-0.8, vmax=-0.05)
    ax[1, 0].axis('off')
    ax[1, 1].scatter(pc_x, pc_y, s=1)
    ax[1, 1].scatter(0, 0, c='red')
    ax[1, 1].set_xlim([-10, 10])
    ax[1, 1].set_ylim([-10, 10])
    ax[1, 1].set_aspect('equal')
    ax[1, 1].set_xticklabels([])
    ax[1, 1].set_yticklabels([])
    ax[1, 1].grid(True, which='both')
    ax[2, 0].imshow(-pred_depth, aspect='auto', cmap='magma', vmin=-0.8, vmax=-0.05)
    ax[2, 0].axis('off')
    ax[2, 1].scatter(out_x, out_y, s=1)
    ax[2, 1].scatter(0, 0, c='red')
    ax[2, 1].set_xlim([-10, 10])
    ax[2, 1].set_ylim([-10, 10])
    ax[2, 1].set_aspect('equal')
    ax[2, 1].set_xticklabels([])
    ax[2, 1].set_yticklabels([])
    ax[2, 1].grid(True, which='both')


def _plot_surface_normal(
    pred_sn: np.ndarray,
    y_sn: np.ndarray,
    ax: np.ndarray,
):
    """
    Private function for plotting the surface normal result.
    Args:
        pred_sn: the predicted surface normal, shape (H, W, 3)
        y_sn: the ground truth surface normal, shape (H, W, 3)
        ax: the axes for plotting
    """
    # transform surface normal to color version
    n_color_pred = normal_color_mapping(sn.get_normal_from_rhs(pred_sn))
    n_color_y = normal_color_mapping(sn.get_normal_from_rhs(y_sn))
    n_color_pred[:, :, 0] = 255 - n_color_pred[:, :, 0]  # flip x color
    n_color_y[:, :, 0] = 255 - n_color_y[:, :, 0]  # flip x color
    n_color_y[y_sn <= -10] = 255  # set invalid region to white

    ax[0, 0].imshow(n_color_y, aspect='auto')
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].imshow(n_color_pred, aspect='auto')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')


def _plot_semantic_seg(
    bg_img: np.ndarray,
    pred_bg_img: np.ndarray,
    pred_logits: np.ndarray,
    y_seg: np.ndarray,
    dataset_metadata,
    ax: np.ndarray,
    confidence_threshold: float = 0.7,
):
    """
    Private function for plotting the semantic segmentation result.
    Args:
        bg_img: the background image, shape (H, W, 3)
        pred_bg_img: the background image for prediction, shape (H, W, 3)
        pred_logits: the predicted logits, shape (C, H, W)
        y_seg: the ground truth segmentation, shape (H, W)
        dataset_metadata: the metadata from `MetadataCatalog.get`
        ax: the axes for plotting
        confidence_threshold: the confidence threshold for uncertain region
    """
    pred_bg_img = np.array([[(158, 154, 200)]])
    pred_bg_img = np.tile(pred_bg_img, (64, 512, 1))
    bg_img = pred_bg_img
    pred_seg = np.argmax(pred_logits, axis=0)
    pred_logits = np.exp(pred_logits) / np.sum(np.exp(pred_logits), axis=0)  # softmax
    max_probs = np.max(pred_logits, axis=0)
    uncertain_mask = max_probs < confidence_threshold

    v = Visualizer(pred_bg_img, metadata=dataset_metadata, scale=2.0)
    seg_pred_vis = v.draw_sem_seg(pred_seg, area_threshold=50, alpha=1.0).get_image()
    v = Visualizer(bg_img, metadata=dataset_metadata, scale=2.0)
    seg_gt_vis = v.draw_sem_seg(y_seg, area_threshold=50, alpha=1.0).get_image()

    # interpolate the uncertain region
    uncertain_mask = cv2.resize(
        uncertain_mask.astype(np.uint8),
        (seg_gt_vis.shape[1], seg_gt_vis.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    ax[0, 0].imshow(seg_gt_vis, aspect="auto")
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ax[1, 0].imshow(seg_pred_vis, aspect="auto")
    ax[1, 0].axis("off")
    ax[1, 1].axis("off")
    seg_pred_vis[uncertain_mask == 1] = [0, 0, 0]
    ax[2, 0].imshow(seg_pred_vis, aspect="auto")
    ax[2, 0].axis("off")
    ax[2, 1].axis("off")


def _plot_obj_detection(
    bg_img: np.ndarray,
    pred_bg_img: np.ndarray,
    pred_obj_instance,
    input_dict,
    dataset_metadata,
    ax: np.ndarray,
):
    """
    Private function for plotting the object detection result.
    Args:
        bg_img: the background image, shape (H, W, 3)
        pred_bg_img: the background image for prediction, shape (H, W, 3)
        pred_obj_instance: the predicted instance, shape (Instances)
        input_dict: the input dictionary, not a list
        dataset_metadata: the metadata from `MetadataCatalog.get`
        ax: the axes for plotting
    """
    input_dict.pop('sem_seg', None)
    input_dict.pop('sem_seg_file_name', None)

    v = OBJVisualizer(pred_bg_img, metadata=dataset_metadata, scale=2.0)
    obj_pred_vis = v.draw_instance_predictions(pred_obj_instance).get_image()
    v = OBJVisualizer(bg_img, metadata=dataset_metadata, scale=2.0)
    obj_gt_vis = v.draw_dataset_dict(input_dict).get_image()

    ax[0, 0].imshow(obj_gt_vis, aspect="auto")
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ax[1, 0].imshow(obj_pred_vis, aspect="auto")
    ax[1, 0].axis("off")
    ax[1, 1].axis("off")
