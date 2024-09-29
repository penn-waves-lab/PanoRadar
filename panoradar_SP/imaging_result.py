import numpy as np
import cupy as cp
from pathlib import Path

from panoradar_SP.dataset import get_lidar_2d_projection, get_motion_params, get_all_from_dataset
import panoradar_SP.radar_imaging_3d as imaging_3d
from panoradar_SP.visualization import show_2d_imaging_plane


def imaging_one_frame(
    frame_id: int,
    radar_frames,
    lidar_frames,
    v_esti: np.ndarray,
    delta_s0_esti: np.ndarray,
    theta_v_esti: np.ndarray,
    static_refl: np.ndarray,
    vbf_compen: cp.ndarray,
    window: cp.ndarray,
    params,
    save_path: str,
):
    """
    Processes the raw signals and visualizes the result

    Args:
        frame_id: the frame number we want to visualize
        radar_frames: RadarFrames object containing all radar frames for the trajectory
        lidar-frames: LidarFrames object containing all lidar frames for the trajectory
        v_esti: velocity estimation
        delta_s0_esti, theta_v_esti: motion estimation parameters
        static_refl:
        vbf_compen:
        window:
        params:
    """
    r_radar = params["r_radar"]
    r_rings = params["r_rings"]
    N_syn_ante = params["N_syn_ante"]
    lambda_i = params["lambda_i"]
    lidar_transform = lidar_frames.lidar_transform
    elev = 0  # elevation angle

    # prepare the signal
    radar_frame, beam_angles = radar_frames.get_a_frame_data(frame_id, pad=True)
    radar_frame = cp.array(radar_frame - static_refl, dtype=cp.complex64)
    lidar_frame = lidar_frames.get_a_frame_data(frame_id)
    lidar_frame_2d = get_lidar_2d_projection(lidar_frame, lidar_transform, elevation=0)
    #
    azimuth_compen = imaging_3d.azimuth_compensation(r_radar, -beam_angles[1], elev, N_syn_ante, lambda_i, window)
    signal_bf = imaging_3d.vertical_beam_forming(radar_frame, vbf_compen)

    # start imaging
    heatmap = imaging_3d.motion_imaging(
        signal_bf,
        azimuth_compen,
        0,
        delta_s0_esti[frame_id],
        theta_v_esti[frame_id],
        beam_angles,
        lambda_i,
    )
    heatmap = cp.asnumpy(heatmap)

    # check last frame
    if heatmap.shape[1] != len(beam_angles):
        return

    # Visualize the result
    show_2d_imaging_plane(heatmap, lidar_frame_2d, r_rings, beam_angles.get(), save_path)


def imaging_frame_from_traj(traj_folder: str, frame_ind: int, out_path: str):
    """
    Processes raw signal data for one frame in a trajectory and plots the resulting 2D floormap

    Args:
        traj_folder: name of the trajectory that we are imaging
        frame_ind: the frame number that we want to visualize
        out_path: the path to save the image. None for directly showing it.
    """
    # Check data type
    traj_folder: Path = Path(traj_folder)
    if 'static' in traj_folder.parent.name:
        radar_frames, lidar_frames, _, params = get_all_from_dataset(traj_folder, static=True)
        motion = {
            'v_esti': np.zeros(len(radar_frames)),
            'delta_s0_esti': np.zeros(len(radar_frames)),
            'theta_v_esti': np.zeros(len(radar_frames)),
        }
    else:
        radar_frames, lidar_frames, _, params = get_all_from_dataset(traj_folder, static=False)
        motion = get_motion_params(traj_folder, need_smooth=True)

    static_refl = np.mean(radar_frames.radar_frames, axis=0, keepdims=True)
    vbf_compen = imaging_3d.vertical_beam_forming_compen(elevation=0, lambda_i=params['lambda_i'])
    window = imaging_3d.get_window(params["N_syn_ante"], params["N_rings"])

    # Plot the imaging result for the specified frame
    imaging_one_frame(
        frame_ind,
        radar_frames,
        lidar_frames,
        motion['v_esti'],
        motion['delta_s0_esti'],
        motion['theta_v_esti'],
        static_refl,
        vbf_compen,
        window,
        params,
        out_path,
    )
