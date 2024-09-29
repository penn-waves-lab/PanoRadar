import cv2
import math
import open3d
import numpy as np

from ..data import seg_colors

PALETTE = np.array(seg_colors)
PALETTE = np.concatenate([PALETTE, np.array([[255, 255, 255]])], axis=0)


def to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to cartesian coordinates
    Args:
        r: radius
        theta: polar angle
        phi: azimuthal angle
    Returns:
        [x, y, z]: cartesian coordinates
    """
    return [
        r * math.sin(theta) * math.cos(phi),
        r * math.sin(theta) * math.sin(phi),
        r * math.cos(theta),
    ]


def get_all_cartesian_1channel(array):
    """
    Convert 1 channel spherical depth array to cartesian coordinates
    Args:
        array: (1, 64, 512) depth
    Returns:
        points: (N, 3) cartesian coordinates
    """
    points = []
    for theta in range(16, 48):
        for phi in range(512):
            theta_angle = ((math.pi) / 4) + (theta * (math.pi) / 128)
            phi_angle = -phi * (math.pi) / 256
            r = array[0, theta, phi]
            r = r * 10
            if r >= 0 and r < 6:
                point_coord = to_cartesian(r, theta_angle, phi_angle)
                points.append(point_coord)
    return np.array(points)


def get_semantic_colors(depth, seg):
    """
    Get the colors for the semantic segmentation
    Args:
        depth: (1, 64, 512) depth
        seg: (1, 64, 512) semantic segmentation
    Returns:
        colors: (N, 3) color array
    """
    colors = []
    for theta in range(16, 48):
        for phi in range(512):
            seg_class = min(seg[0, theta, phi], 11)
            r = depth[0, theta, phi]
            r = r * 10
            if r >= 0 and r < 6:
                point_color = PALETTE[seg_class] / 255
                colors.append(point_color)

    return np.array(colors)


def visualize_lidar_gt_with_seg(lidar_data, gt_seg, parameters):
    """
    Visualize the lidar data with semantic segmentation colors.
    Args:
        lidar_data: the lidar data, (1, 64, 512)
        gt_seg: the ground truth semantic segmentation, (1, 64, 512)
        parameters: the camera parameters
    Returns:
        ret: the visualization image
    """
    # Get points
    lidar_points = get_all_cartesian_1channel(lidar_data)

    # Get colors
    lidar_colors = get_semantic_colors(lidar_data, gt_seg)

    # Get pointcloud
    lidar_pointcloud = open3d.geometry.PointCloud()
    lidar_pointcloud.points = open3d.utility.Vector3dVector(lidar_points)
    lidar_pointcloud.colors = open3d.utility.Vector3dVector(lidar_colors)

    # Visualize
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='Lidar with semantic', width=1920, height=569, left=0, top=0, visible=True)
    vis.add_geometry(lidar_pointcloud)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters, True)

    # Save image as png
    vis.poll_events()
    vis.update_renderer()
    ret = vis.capture_screen_float_buffer()
    ret = np.asarray(ret)
    vis.destroy_window()

    return ret


def visualize_depth_pred_with_seg(depth_pred, seg_pred, parameters):
    """
    Visualize the predicted depth with semantic segmentation colors.
    Args:
        depth_pred: the predicted depth, (1, 64, 512)
        seg_pred: the predicted semantic segmentation, (1, 64, 512)
        parameters: the camera parameters
    Returns:
        ret: the visualization image
    """
    # Get points
    pred_points = get_all_cartesian_1channel(depth_pred)

    # Get colors
    pred_colors = get_semantic_colors(depth_pred, seg_pred)

    pred_pointcloud = open3d.geometry.PointCloud()
    pred_pointcloud.points = open3d.utility.Vector3dVector(pred_points)
    pred_pointcloud.colors = open3d.utility.Vector3dVector(pred_colors)

    # Visualize
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='Pred Depth with Semantic', width=1920, height=569, left=0, top=0, visible=True)
    vis.add_geometry(pred_pointcloud)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters, True)

    # Save image as png
    vis.poll_events()
    vis.update_renderer()
    ret = vis.capture_screen_float_buffer()
    ret = np.asarray(ret)
    vis.destroy_window()

    return ret


def cfar3d(
    data,
    r_train_num,
    r_guard_num,
    ele_train_num,
    ele_guard_num,
    azi_train_num,
    azi_guard_num,
    fp_rate,
):
    mx = np.max(data)
    mn = np.min(data)

    data = (data - mn) / (mx - mn)

    num_r, num_elevation, num_azimuth = data.shape
    peaks = np.zeros_like(data)

    r_guard_half = r_guard_num // 2
    r_window_half = r_train_num // 2 + r_guard_half
    ele_guard_half = ele_guard_num // 2
    ele_window_half = ele_train_num // 2 + ele_guard_half
    azi_guard_half = azi_guard_num // 2
    azi_window_half = azi_train_num // 2 + azi_guard_half

    second_thresh = np.mean(data) + 2 * np.std(data)

    for r_ind in range(r_window_half, num_r - r_window_half):
        for ele_ind in range(ele_window_half, num_elevation - ele_window_half):
            for azi_ind in range(azi_window_half, num_azimuth - azi_window_half):
                guard_sum = np.sum(
                    data[
                        r_ind - r_guard_half : r_ind,
                        ele_ind - ele_guard_half : ele_ind,
                        azi_ind - ele_guard_half : azi_ind,
                    ]
                ) + np.sum(
                    data[
                        r_ind + 1 : r_ind + ele_guard_half + 1,
                        ele_ind + 1 : ele_ind + ele_guard_half + 1,
                        azi_ind + 1 : azi_ind - ele_guard_half + 1,
                    ]
                )

                window_sum = np.sum(
                    data[
                        r_ind - r_window_half : r_ind,
                        ele_ind - ele_window_half : ele_ind,
                        azi_ind - azi_window_half : azi_ind,
                    ]
                ) + np.sum(
                    data[
                        r_ind + 1 : r_ind + r_window_half + 1,
                        ele_ind + 1 : ele_ind + ele_window_half + 1,
                        azi_ind + 1 : azi_ind - azi_window_half + 1,
                    ]
                )

                train_sum = window_sum - guard_sum

                num_train_cells = (
                    (r_window_half * ele_window_half * azi_window_half * 8)
                    - (r_guard_half * ele_guard_half * azi_guard_half * 8)
                ) - 1
                avg_noise = train_sum / num_train_cells
                threshold = avg_noise * num_train_cells * ((fp_rate ** (-1 / num_train_cells)) - 1)
                if data[r_ind, ele_ind, azi_ind] >= threshold and data[r_ind, ele_ind, azi_ind] > second_thresh:
                    peaks[r_ind, ele_ind, azi_ind] = 1

    return peaks


def get_points_3d_cfar(peak_indices, data):
    points = []
    colors = []
    for peak_index in peak_indices:
        r = peak_index[0] * 0.0375
        ele_angle = ((math.pi) / 4) + (peak_index[1] * (math.pi) / 128)
        azi_angle = -peak_index[2] * (math.pi) / 256
        color = [data[peak_index[0], peak_index[1], peak_index[2]], 0, 0]

        if r > 0 and r < 6 and peak_index[1] > 16 and peak_index[1] < 48:
            cartesian = to_cartesian(r, ele_angle, azi_angle)

            points.append(cartesian)
            colors.append(color)

    return np.array(points), np.array(colors)


def visualize_rf_heatmap_with_cfar(rf, parameters):
    peaks = cfar3d(rf, 40, 8, 32, 4, 8, 2, 3e-6)

    peak_indices = np.transpose(np.nonzero(peaks))

    cfar_3d_points, _ = get_points_3d_cfar(peak_indices, rf)

    # Get pointcloud
    cfar_3d_pointcloud = open3d.geometry.PointCloud()
    cfar_3d_pointcloud.points = open3d.utility.Vector3dVector(cfar_3d_points)

    # Visualize
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='cfar3d', width=1920, height=569, left=0, top=0, visible=True)
    vis.add_geometry(cfar_3d_pointcloud)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters, True)

    # Save image as png
    vis.poll_events()
    vis.update_renderer()
    ret = vis.capture_screen_float_buffer()
    ret = np.asarray(ret)
    vis.destroy_window()

    return ret
