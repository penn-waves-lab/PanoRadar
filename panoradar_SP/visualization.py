"""Visualization and interpretation of the imaging result."""

import numpy as np
import matplotlib.pyplot as plt


def project_polar_to_cartesian(
    heatmap: np.ndarray,
    r_rings: np.ndarray,
    max_range: float,
    grid_size: float,
    beam_angles: np.ndarray,
    default_value: float = 0.0,
    rotation_offset: float = 0,
) -> np.ndarray:
    """Project the polar system imaging result to Cartesian system.
    Args:
        heatmap: the polar system imaging result, (N_rings, N_beams)
        r_rings: radius of each ring, in meter, shape (N_rings, )
        max_range: maximum projection range, m x m image
        grid_size: the actual size of each grid/pixel
        beam_angles: the facing angle of each beam
    Return:
        proj_heatmap: the Cartesian system imaging result
    """
    PROJ_MAP_SZ = int(2 * max_range / grid_size)  # size of the projected heatmap
    proj_heatmap = np.full((PROJ_MAP_SZ, PROJ_MAP_SZ), default_value, dtype=np.float32)
    N_rings, N_beams = heatmap.shape

    cos_phi = np.cos(beam_angles + rotation_offset)
    sin_phi = np.sin(beam_angles + rotation_offset)

    # project polar to Cartesian
    for ring_id in range(0, N_rings):
        x_grid_id = (r_rings[ring_id] * cos_phi + max_range) / grid_size
        y_grid_id = (r_rings[ring_id] * sin_phi + max_range) / grid_size
        x_grid_id = np.round(x_grid_id).astype(np.int32)
        y_grid_id = np.round(y_grid_id).astype(np.int32)

        # bound to PROJ_MAP_SZ
        valid = np.logical_and(
            np.logical_and(x_grid_id >= 0, x_grid_id < PROJ_MAP_SZ),
            np.logical_and(y_grid_id >= 0, y_grid_id < PROJ_MAP_SZ),
        )

        proj_heatmap[y_grid_id[valid], x_grid_id[valid]] = heatmap[ring_id][valid]

    return proj_heatmap


def show_2d_imaging_plane(heatmap_abs, lidar_frame, r_rings, beam_angles: np.ndarray, save_path: str):
    """This is the function showing the large result figure"""
    MAX_RANGE = 10.0  # max imaging 10m*10m
    GRID_SIZE = 0.04  # grid size 4cm

    # ################# 1. Log  #################
    heatmap_n = np.log10(heatmap_abs + 0.001) / 3
    proj_heatmap_l = project_polar_to_cartesian(heatmap_n, r_rings, MAX_RANGE, GRID_SIZE, beam_angles, -0.2)

    # #################  visualize result  #################
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(lidar_frame[:, 0], lidar_frame[:, 1], s=1, c="C1")
    plt.imshow(proj_heatmap_l, origin="lower", extent=(-10, 10, -10, 10), cmap="jet", vmin=-0.2, vmax=0.5)
    plt.subplot(1, 2, 2)
    plt.imshow(proj_heatmap_l, origin="lower", extent=(-10, 10, -10, 10), cmap="jet", vmin=-0.2, vmax=0.5)

    if save_path:
        plt.savefig(f'{save_path}/imaging_result.png')
    else:
        plt.show()


def get_range_image_from_lidar(
    lidar_frame: np.ndarray, lidar_transform: np.ndarray, out_azi_size: int = None
) -> np.ndarray:
    """Get the range image from lidar point cloud.
    Args:
        lidar_frame: a single frame of point cloud, shape (channel, horizontal, xyz)
        out_azi_size: the output azimuth size
    Returns:
        polar: each pixel is the range for the point. shape (elev, azimuth)
    """
    transform_3d = np.eye(3)
    transform_3d[:2, :2] = lidar_transform
    transform_3d = transform_3d[None]
    lidar_frame = lidar_frame @ transform_3d

    point_ranges = np.linalg.norm(lidar_frame, ord=2, axis=-1)
    point_azimuth = np.arctan2(lidar_frame[:, :, 1], lidar_frame[:, :, 0])
    point_azimuth[point_azimuth < 0] += 2 * np.pi  # shift -pi~pi to 0~2pi

    if out_azi_size is None:
        out_azi_size = lidar_frame.shape[1]
    out_elev_size = lidar_frame.shape[0]

    azimuth_min = 0
    azimuth_num = out_azi_size
    azimuth_bin = 2 * np.pi / azimuth_num
    azimuth_inds = np.round((point_azimuth - azimuth_min) / azimuth_bin).astype(np.int32)
    valid = np.logical_and(azimuth_inds >= 0, azimuth_inds < azimuth_num)

    polar = np.zeros((out_elev_size, out_azi_size), dtype=np.float32)
    for i in range(64):
        valid_i = valid[i]
        polar[i, azimuth_inds[i][valid_i]] = point_ranges[i][valid_i]

    polar = np.flip(polar, axis=1)  # from counter-clockwise to clockwise, align with radar
    return polar
