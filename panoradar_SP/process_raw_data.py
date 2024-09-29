import numpy as np
import cupy as cp
import torch
from torchvision.transforms.functional import resize
import cv2
from tqdm import tqdm
from pathlib import Path

from panoradar_SP.dataset import get_all_from_dataset, get_motion_params
import panoradar_SP.radar_imaging_3d as imaging_3d
from panoradar_SP.visualization import get_range_image_from_lidar


def process_data(rf: np.ndarray, lidar: np.ndarray):
    """
    Process rf and lidar data to ready-to-use format by reshaping and resizing.

    Args:
        rf: rf data for 1 frame
        lidar: lidar data for 1 frame

    Returns:
        rf, lidar as np arrays after processing
    """
    azimuth_size = 512

    rf = np.log10(rf + 0.001) / 3
    rf = torch.from_numpy(rf.transpose((1, 0, 2)))  # (#range_bin, #elev, #azimuth)
    rf = resize(rf, (rf.shape[1], azimuth_size))
    rf = rf.numpy().astype(np.float32)

    lidar = torch.from_numpy(lidar.copy()) / 10
    lidar[lidar == 0] = -1e3  # avoid failure points to affect the resizing
    lidar = resize(lidar.unsqueeze(0), (lidar.shape[0], azimuth_size))
    lidar = lidar.numpy().astype(np.float32)

    # use median filter to fix lidar failure regions
    lidar_mf = cv2.medianBlur(lidar[0], ksize=3)[np.newaxis]  # (1, #elev, #azimuth)
    fail_region = lidar < 0
    lidar[fail_region] = lidar_mf[fail_region]

    return rf, lidar


def save_static_data(traj_name: Path, out_folder: Path):
    """
    Process the raw data of one specific *static* trajectory

    Args:
        traj_name: the name of the trajectory to process
        out_folder: the output folder for the processed data

    Saves for one trajectory:
        - rf: (#elev, #range_bin, #azimuth), float32
        - lidar: (#elev, #azimuth), float32
    """
    # make dirs for output folders
    target_parent_folder = out_folder / traj_name.relative_to(traj_name.parent.parent)
    lidar_npy_folder = target_parent_folder / Path('lidar_npy')
    rf_npy_folder = target_parent_folder / Path('rf_npy')
    lidar_npy_folder.mkdir(parents=True, exist_ok=True)
    rf_npy_folder.mkdir(parents=True, exist_ok=True)

    print('loading data...')
    radar_frames, lidar_frames, *_, params = get_all_from_dataset(traj_name, static=True)
    r_radar = params["r_radar"]
    N_syn_ante = params["N_syn_ante"]
    N_rings = params["N_rings"]
    lambda_i = params["lambda_i"]
    lidar_transform = lidar_frames.lidar_transform

    # prepare for 3D imaging
    elevs = np.linspace(np.deg2rad(45), np.deg2rad(-45), 64)
    vbf_compens = [imaging_3d.vertical_beam_forming_compen(elev, lambda_i) for elev in elevs]
    static_refl = np.mean(radar_frames.radar_frames, axis=0, keepdims=True)
    window = imaging_3d.get_window(N_syn_ante, N_rings)

    print('imaging...')
    for f in tqdm(range(len(radar_frames))):
        # prepare the signal
        radar_frame, beam_angles = radar_frames.get_a_frame_data(f, pad=True)
        radar_frame = cp.array(radar_frame - static_refl, dtype=cp.complex64)
        N_beams = len(beam_angles)

        # rf img
        heatmaps = []
        for elev, vbf_compen in zip(elevs, vbf_compens):
            signal_bf = imaging_3d.vertical_beam_forming(radar_frame, vbf_compen)
            azimuth_compen = imaging_3d.azimuth_compensation(
                r_radar, -beam_angles[1], elev, N_syn_ante, lambda_i, window
            )
            heatmap = imaging_3d.static_imaging(signal_bf, azimuth_compen, N_beams)
            heatmaps.append(cp.asnumpy(heatmap))
        heatmaps = np.stack(heatmaps)

        # lidar
        lidar_frame = lidar_frames.get_a_frame_data(f)
        lidar_range_img = get_range_image_from_lidar(lidar_frame, lidar_transform)

        # further process
        heatmaps, lidar_range_img = process_data(heatmaps, lidar_range_img)

        # save
        np.save(lidar_npy_folder / f'{f:05d}.npy', lidar_range_img)
        np.save(rf_npy_folder / f'{f:05d}.npy', heatmaps)

    print()


def save_moving_data(traj_name: Path, out_folder: Path):
    """
    Process the raw data of one specific *moving* trajectory
    Args:
        traj_name: the name of the trajectory to process
        out_folder: the output folder for the processed data

    Saves for one trajectory:
        - rf: (#elev, #range_bin, #azimuth), float32
        - lidar: (#elev, #azimuth), float32
    """
    # make dirs for output folders
    target_parent_folder = out_folder / traj_name.relative_to(traj_name.parent.parent)
    lidar_npy_folder = target_parent_folder / Path('lidar_npy')
    rf_npy_folder = target_parent_folder / Path('rf_npy')
    lidar_npy_folder.mkdir(parents=True, exist_ok=True)
    rf_npy_folder.mkdir(parents=True, exist_ok=True)

    print('loading data...')
    radar_frames, lidar_frames, _, params = get_all_from_dataset(traj_name, static=False)
    r_radar = params["r_radar"]
    N_syn_ante = params["N_syn_ante"]
    N_rings = params["N_rings"]
    lambda_i = params["lambda_i"]
    lidar_transform = lidar_frames.lidar_transform

    # read motion estimation parameters
    motion = get_motion_params(traj_name)
    delta_s0_esti = motion['delta_s0_esti']
    theta_v_esti = motion['theta_v_esti']

    # prepare for 3D imaging
    elevs = np.linspace(np.deg2rad(45), np.deg2rad(-45), 64)
    vbf_compens = [imaging_3d.vertical_beam_forming_compen(elev, lambda_i) for elev in elevs]
    static_refl = np.mean(radar_frames.radar_frames, axis=0, keepdims=True)
    window = imaging_3d.get_window(N_syn_ante, N_rings)

    print('imaging...')
    for f in tqdm(range(len(radar_frames))):
        # prepare the signal
        radar_frame, beam_angles = radar_frames.get_a_frame_data(f, pad=True)
        radar_frame = cp.array(radar_frame - static_refl, dtype=cp.complex64)

        # rf img
        heatmaps = []
        for elev, vbf_compen in zip(elevs, vbf_compens):
            signal_bf = imaging_3d.vertical_beam_forming(radar_frame, vbf_compen)
            azimuth_compen = imaging_3d.azimuth_compensation(
                r_radar, -beam_angles[1], elev, N_syn_ante, lambda_i, window
            )
            heatmap = imaging_3d.motion_imaging(
                signal_bf,
                azimuth_compen,
                elev,
                delta_s0_esti[f],
                theta_v_esti[f],
                beam_angles,
                lambda_i,
            )
            heatmaps.append(cp.asnumpy(heatmap))
        heatmaps = np.stack(heatmaps)

        # lidar
        lidar_frame = lidar_frames.get_a_frame_data(f)
        lidar_range_img = get_range_image_from_lidar(lidar_frame, lidar_transform)

        # further process
        heatmaps, lidar_range_img = process_data(heatmaps, lidar_range_img)

        # save
        np.save(lidar_npy_folder / f'{f:05d}.npy', lidar_range_img)
        np.save(rf_npy_folder / f'{f:05d}.npy', heatmaps)

    print()


def process_dataset(in_building_folder: str, out_folder: str):
    """
    Processes the complete signal dataset into format for neural network input

    Args:
        in_building_folder: the name of the trajectory folder
        out_folder: the output folder for the processed data
    Saves:
        - rf: (#elev, #range_bin, #azimuth), float32
        - lidar: (#elev, #azimuth), float32
    """
    # Check data type
    in_building_folder: Path = Path(in_building_folder)
    out_folder: Path = Path(out_folder)
    func = save_static_data if 'static' in in_building_folder.name else save_moving_data

    all_traj_names = sorted(in_building_folder.iterdir())
    N_traj = len(all_traj_names)  # Total number of trajectories

    for i, traj_name in enumerate(all_traj_names):
        print(f'({i+1} / {N_traj}) {traj_name}:')
        func(traj_name, out_folder)

    print('done')
