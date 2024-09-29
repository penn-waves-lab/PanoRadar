import numpy as np
import cupy as cp
from tqdm import tqdm
from pathlib import Path

from panoradar_SP.dataset import get_all_from_dataset
import panoradar_SP.radar_imaging_3d as imaging_3d
from panoradar_SP.velocity_estimation import estimate_v_line_detect


def estimate_params_for_trajectory(traj_name: str):
    """Estimate the motion parameters for one whole trajectory.
       Also save the lidar ground truth for evaluation.
    Args:
        traj_name: name of the trajectory to be estimated

    Save the following parameters:
        v_esti, delta_s0_esti, theta_v_esti, omega_imu
        v_gt, delta_s0_gt, theta_v_gt, omega_gt
    """

    # read the trajectory
    print('Loading data')
    radar_frames, lidar_frames, imu_frames, params = get_all_from_dataset(traj_name, False)

    r_radar = params["r_radar"]
    N_syn_ante = params["N_syn_ante"]
    lambda_i = params["lambda_i"]
    lambda_0 = params["lambda_0"]
    N_AoA_bins = params["N_AoA_bins"]
    N_rings = params["N_rings"]
    delta_t = radar_frames.delta_t
    elev = 0  # elevation angle

    esti_all = []  # [(A, theta_v), ...]
    gt_all = []  # [(v, theta_v, omega), ...]

    static_refl = np.mean(radar_frames.radar_frames, axis=0, keepdims=True)
    vbf_compen = imaging_3d.vertical_beam_forming_compen(elev, lambda_i)
    window = imaging_3d.get_window(N_syn_ante, N_rings)

    print('Start estimation')
    for pick_frame in tqdm(range(len(radar_frames))):
        # prepare signals
        radar_frame, beam_angles = radar_frames.get_a_frame_data(pick_frame, pad=True)
        radar_frame = cp.array(radar_frame - static_refl, dtype=cp.complex64)
        signal_bf = imaging_3d.vertical_beam_forming(radar_frame, vbf_compen)
        azimuth_compen = imaging_3d.azimuth_compensation(r_radar, -beam_angles[1], elev, N_syn_ante, lambda_i, window)

        # estimate motion
        params_esti, *_ = estimate_v_line_detect(signal_bf, azimuth_compen, len(beam_angles), params)
        esti_all.append(params_esti)

        # get ground truth motion
        gt_all.append(lidar_frames.get_velocity_gt(pick_frame))

    # convert to numpy array
    esti_all = np.asarray(esti_all)
    gt_all = np.asarray(gt_all)

    # compute motion parameter (esti and gt), and save them
    np.savez(
        f'{traj_name}/motion_output.npz',
        v_esti=-esti_all[:, 0] * lambda_0 / (delta_t * N_AoA_bins * 2),
        theta_v_esti=esti_all[:, 1],
        delta_s0_esti=-esti_all[:, 0] * lambda_0 / (N_AoA_bins * 2),
        omega_imu=imu_frames.get_all_omega(),
        v_gt=gt_all[:, 0],
        theta_v_gt=gt_all[:, 1],
        delta_s0_gt=gt_all[:, 0] * delta_t,
        omega_gt=gt_all[:, 2],
    )


def estimate_whole_building(in_building_folder: str):
    """Estimates the motion parameters for all trajectories in this building.
    Calls 'estimates_params_for_trajectory' for each trajectory.

    Args:
        in_building_folder: the name of the building folder

    Saves the following paramters for each trajectory into motion_output.npz file
    in the trajectory folder:
        v_esti, delta_s0_esti, theta_v_esti, omega_imu
        v_gt, delta_s0_gt, theta_v_gt, omega_gt
    """
    # Check data type
    in_building_folder = Path(in_building_folder)
    if 'static' in in_building_folder.name:
        print("You don't need to do motion estimation for static data. Exit.")
        exit(0)

    all_traj_names = sorted(in_building_folder.iterdir())
    N_traj = len(all_traj_names)  # Total number of trajectories

    for i, traj_name in enumerate(all_traj_names):
        print(f'({i+1} / {N_traj}) {traj_name}:')
        estimate_params_for_trajectory(traj_name)
