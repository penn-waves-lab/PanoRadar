"""This file is about the dataset operation."""
import numpy as np
import cupy as cp
from scipy.ndimage import gaussian_filter
import open3d as o3d
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List

# =========================  General Interfaces  =========================
class LidarFrames:
    """Representation of the raw LiDAR frames data."""

    def __init__(
        self,
        lidar_frames: Union[np.ndarray, List],
        lidar_ts: np.ndarray,
        frame_sep_ids: Tuple[int],
        lidar_transform: np.ndarray,
    ) -> None:
        """Init the lidar frames object
        Args:
            lidar_frames: raw .npy data or raw Tuple[frame]
            lidar_ts: timestamps, shape (#frame, )
            frame_sep_ids: Index for separating each frame
            lidar_transform: the transformation aligning ladar to radar
        """
        self.lidar_frames = lidar_frames
        self.lidar_ts = lidar_ts
        self.frame_sep_ids = frame_sep_ids
        self.lidar_transform = lidar_transform

        self.delta_t = np.median(np.diff(lidar_ts))  # time between two frame

    def __len__(self) -> int:
        return len(self.frame_sep_ids) - 1

    def get_a_frame_data(self, frame_id: int) -> np.ndarray:
        """Get the data of the lidar frame for a cycle.
        Args:
            frame_id: frame index, which frame do you want, start from 0
        Returns:
            lidar_frame: (min, max, data) or (64, 2048, 3)
        """
        st_id = self.frame_sep_ids[frame_id]
        frame = self.lidar_frames[st_id]

        return frame

    def get_a_frame_ts(self, frame_id: int) -> np.ndarray:
        """Get the timestamps of the lidar frame for a cycle.
        Args:
            frame_id: frame index, which frame do you want, start from 0
        Returns:
            lidar_ts: timestamps in seconds
        """
        st_id = self.frame_sep_ids[frame_id]
        return self.lidar_ts[st_id]

    def get_velocity_gt(self, frame_id: int) -> Tuple[float, float, float]:
        """Get the ground truth velocity by registering two lidar frames.
        Args:
            frame_id: frame index, which frame do you want, start from 0
        Returns:
            v: the magnitude of the velocity, m/s
            phi_v: the angle between the velocity and the 0-beam (forward direction). rad
            omega_v: external rotation (angular velocity), rad/s
        """
        frame0 = self.get_a_frame_data(frame_id)
        frame1 = self.get_a_frame_data(frame_id + 1)
        frame0 = get_lidar_2d_projection(frame0, self.lidar_transform, 0)  # convert to 2D
        frame1 = get_lidar_2d_projection(frame1, self.lidar_transform, 0)
        frame0 = np.hstack((frame0, np.zeros((frame0.shape[0], 1))))
        frame1 = np.hstack((frame1, np.zeros((frame1.shape[0], 1))))

        frame0_pcd = o3d.geometry.PointCloud()
        frame0_pcd.points = o3d.utility.Vector3dVector(frame0)
        frame0_pcd, _ = frame0_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
        frame1_pcd = o3d.geometry.PointCloud()
        frame1_pcd.points = o3d.utility.Vector3dVector(frame1)
        frame1_pcd, _ = frame1_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)

        # regi first pass
        threshold = 0.5
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            frame1_pcd,
            frame0_pcd,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-21, relative_rmse=1e-21, max_iteration=300
            ),
        )

        # regi second pass
        threshold = 0.06
        trans_init = reg_p2p.transformation
        reg_p2p = o3d.pipelines.registration.registration_icp(
            frame1_pcd,
            frame0_pcd,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-21, relative_rmse=1e-21, max_iteration=300
            ),
        )

        # get the velocity
        t = (self.frame_sep_ids[frame_id + 1] - self.frame_sep_ids[frame_id]) * self.delta_t
        v = np.sqrt(reg_p2p.transformation[0, 3] ** 2 + reg_p2p.transformation[1, 3] ** 2)
        phi_v = np.arctan2(reg_p2p.transformation[1, 3], reg_p2p.transformation[0, 3])
        omega_v = np.arcsin(reg_p2p.transformation[1, 0])

        return v / t, phi_v, omega_v / t


class ImuFrames:
    """Representation of the IMU data. Current it only angular velocity for z axis."""

    def __init__(
        self,
        imu_frames: np.ndarray,
        imu_ts: np.ndarray,
        frame_sep_ids: np.ndarray,
    ) -> None:
        """Init the imu frames object
        Args:
            imu_frames: raw .npy data
            imu_ts: timestamps, shape (#frame, )
            frame_sep_ids: Index for separating each frame
        """
        self.imu_ts = imu_ts
        self.frame_sep_ids = frame_sep_ids

        imu_frames = _ourlier_filtering(imu_frames, 20, 2)  # outlier removal
        self.imu_frames = gaussian_filter(imu_frames, sigma=2.0, truncate=6)
        self.imu_frames = np.deg2rad(self.imu_frames)

    def __len__(self) -> int:
        return len(self.frame_sep_ids) - 1

    def get_all_omega(self) -> np.ndarray:
        """Get all the external rotation for the whole trajectory
        Only give the readings at the beginning of each cycle. Mainly for gt eval.
        Returns:
            external_omegas: delta omega in radian, counter-clockwise positive
        """
        omegas = []
        for i in range(len(self)):
            st_frame_id = self.frame_sep_ids[i]
            end_frame_id = self.frame_sep_ids[i + 1]
            omegas.append(np.mean(self.imu_frames[st_frame_id:end_frame_id]))
        return np.array(omegas)

    def get_a_frame_omegas(self, frame_id: int) -> np.ndarray:
        """Get the external rotation readings during a cycle.
        Returns:
            external_omegas: delta omega in radian, counter-clockwise positive
        """
        st_id = self.frame_sep_ids[frame_id]
        end_id = self.frame_sep_ids[frame_id + 1]
        return self.imu_frames[st_id:end_id]

    def get_a_frame_ts(self, frame_id: int) -> np.ndarray:
        """Get the timestamps of the imu frame for a cycle (the beginning).
        Args:
            frame_id: frame index, which frame do you want, start from 0
        Returns:
            imu_ts: timestamps in seconds
        """
        st_id = self.frame_sep_ids[frame_id]
        return self.imu_ts[st_id]


class RadarFrames:
    """Representation of the raw radar frames data."""

    def __init__(
        self,
        radar_frames: np.ndarray,
        radar_ts: np.ndarray,
        radar_frame_sep: np.ndarray,
        imu_obj: ImuFrames,
        N_syn_ante: int,
        static: bool,
    ) -> None:
        """Init the radar frames object
        Args:
            radar_frames: raw .npy data, shape (#total_chirps, #N_v_ante, #adc_samples)
            radar_ts: timestamps, shape (#total_chirps, )
            radar_frame_sep: Index for separating chirps for each frame, as starting beam
            imu_obj: the imu data class object
            N_syn_ante: the number of antennas used in the horizontal arc
            static: whether this is static data (the robot doesn't move)
        """
        INT16_MAX = 32768
        self.radar_frames = radar_frames / INT16_MAX
        self.radar_ts = radar_ts
        self.radar_st_frame_sep = radar_frame_sep
        self.half_syn_ante = (N_syn_ante - 1) // 2
        self.static = static

        # compute the time between two beams
        self.delta_t = np.mean(np.diff(radar_ts[radar_frame_sep]) / np.diff(radar_frame_sep))

        # determine the radar end frame separation
        self.radar_end_frame_sep = (
            np.append(radar_frame_sep[1:], 0)
            if static
            else self.__find_radar_end_frame_sep(imu_obj.imu_frames, imu_obj.frame_sep_ids)
        )

    def __find_radar_end_frame_sep(self, imu_frames: np.ndarray, imu_frame_sep: np.ndarray) -> np.ndarray:
        """find the end frame separation using the imu omega measurement.
        Args:
            imu_frames: imu omega measurement data, shape (#measurement, ) in rad/s
            imu_frame_sep: the separation indeices for each rotation cycle
        Return:
            radar_end_frame_sep: the indices indicating the end of each frame
        """
        delta_theta0 = np.mean(2 * np.pi / np.diff(self.radar_st_frame_sep))
        radar_end_frame_sep = []

        for frame_id in range(len(self)):
            imu_st_ind = imu_frame_sep[frame_id]
            imu_end_ind = imu_frame_sep[frame_id + 1]
            omega = np.mean(imu_frames[imu_st_ind:imu_end_ind])  # ccw +

            N_beams = round(2 * np.pi / (delta_theta0 - self.delta_t * omega))
            radar_end_frame_sep.append(self.radar_st_frame_sep[frame_id] + N_beams)

        # last one is not used. For the same length as radar_st_frame_sep
        radar_end_frame_sep.append(0)
        return np.array(radar_end_frame_sep)

    def __len__(self) -> int:
        return len(self.radar_st_frame_sep) - 1

    def get_a_frame_data(self, frame_id: int, pad=True) -> Tuple[np.ndarray, cp.ndarray]:
        """Get the data of the radar frame for a cycle.
        Args:
            frame_id: frame index, which frame do you want, start from 0
            pad: whether we pad some chirps at the beginning and the end.
                Note that real time series signal is used. If self.static is true,
                circular padding method is used.
        Returns:
            radar_frame: shape (#chirps, #N_v_ante, #adc_samples)
            beam_angles: the facing angle of each beams
        """
        st_ind = self.radar_st_frame_sep[frame_id]
        end_ind = self.radar_end_frame_sep[frame_id]

        if pad and self.static:
            frame = np.concatenate(
                (
                    self.radar_frames[end_ind - self.half_syn_ante : end_ind],
                    self.radar_frames[st_ind:end_ind],
                    self.radar_frames[st_ind : st_ind + self.half_syn_ante],
                ),
                axis=0,
            )
        elif pad and not self.static:
            frame = self.radar_frames[st_ind - self.half_syn_ante : end_ind + self.half_syn_ante]
        else:
            frame = self.radar_frames[st_ind:end_ind]

        beam_angles = cp.linspace(0, -2 * np.pi, end_ind - st_ind + 1)[:-1]
        return frame, beam_angles

    def get_a_frame_ts(self, frame_id: int) -> np.ndarray:
        """Get the timestamps of the radar frame for a cycle.
        Args:
            frame_id: frame index, which frame do you want, start from 0
        Returns:
            radar_ts: timestamps in seconds with shape (N_beams, )
        """
        st_ind = self.radar_st_frame_sep[frame_id]
        end_ind = self.radar_end_frame_sep[frame_id]
        return self.radar_ts[st_ind:end_ind]


def _ourlier_filtering(in_array: np.ndarray, jitter_thres: float, assign_neighbor: int) -> np.ndarray:
    """filter outliers using thresholds and assign them with their neighbors.
    Args:
        in_array: the input array, should be 1D array
        jitter_thres: if abs(a_n-a_{n-1}) > jitter_thres, then a_n is an outlier
        assign_neighbor: then a_n will be assigned as a_{n-assign_neighbor}
    """
    in_array = np.copy(in_array)
    diff = np.abs(np.hstack((np.diff(in_array), 0)))
    (jitter_ids,) = np.where(diff > jitter_thres)
    jitter_ids[jitter_ids < assign_neighbor] = assign_neighbor
    in_array[jitter_ids] = in_array[jitter_ids - assign_neighbor]
    return in_array


def get_all_from_dataset(dataset_name: str, static: bool) -> Tuple[RadarFrames, LidarFrames, ImuFrames, Dict[str, Any]]:
    """Parse the new high rotary speed dataset.
    Args:
        dataset_name: the name of the dataset.
        root_folder_name: the name of the root folder that holds the SP data
        static: whether this is static data (the robot doesn't move)
        st_beam_offset: the offset to the starting beam, percentage 0~100
    Returns:
        radar_frames, lidar_frames, video_frames: the data
        param: A dictionary containing the parameters {'name':value}
    """
    # result given by the 'find_loop.ipynb' notebook
    try:
        data_params = np.load(f'{dataset_name}/data_params.npz', allow_pickle=True)
        radar_sep = data_params['radar_sep']
        lidar_sep = data_params['lidar_sep']
        imu_sep = data_params['imu_sep']
        r_radar = data_params['r_radar'].item()
        transform = data_params['transform']
    except FileNotFoundError:
        raise ValueError("The dataset_name is incorrect!")

    # --- Read Raw Data and Timestamp ---
    # read radar data, #frame, #chirps * #tx, #rx, #adc_samples
    # reshape to (total_chirps, N_v_ante, N_samples)
    radar_frames = np.load(f"{dataset_name}/radar_data.npy", mmap_mode='r')
    radar_frames = radar_frames.reshape(radar_frames.shape[0], -1, radar_frames.shape[3])
    radar_ts = np.loadtxt(f"{dataset_name}/radar_ts.txt")

    # read lidar data  # (frames, 64, 2048, 3)
    lidar_frames = np.load(f"{dataset_name}/lidar_data.npy", mmap_mode='r')
    lidar_ts = np.loadtxt(f"{dataset_name}/lidar_ts.txt")

    # read imu data (external rotation, i.e. delta omega in deg/s) (N,)
    imu_frames = np.load(f"{dataset_name}/imu_data.npy", mmap_mode='r')
    imu_ts = np.loadtxt(f"{dataset_name}/imu_ts.txt")

    # --- Useful Parameters ---
    c = 2.99792e8  # m/s
    fs = 5120e3  # sampling rate Hz/s
    freq_slope = 79.951e12  # Hz/s
    freq_st = 77e9  # start frequency, Hz
    freq_end = 81e9  # end frequency, Hz
    FOV = np.deg2rad(90)  # degree

    # N_syn_ante: how many antennas for the fov
    # N_AoA_bins: the number of bins in Angle fft (zero padding)
    N_syn_ante = int(FOV / (2 * np.pi) * (radar_sep[1] - radar_sep[0]))
    N_syn_ante = N_syn_ante if N_syn_ante % 2 == 1 else N_syn_ante + 1
    N_AoA_bins = N_syn_ante if N_syn_ante > 300 else 300
    N_rings = radar_frames.shape[2]  # also = N_samples
    N_samples = N_rings

    r_rings = c / 2 / (freq_end - freq_st) * np.arange(N_rings)  # in meter
    lambda_i = c / cp.linspace(freq_st, freq_end, N_rings)

    # --- Data and Param ---
    imu_frames = ImuFrames(imu_frames, imu_ts, imu_sep)
    radar_frames = RadarFrames(radar_frames, radar_ts, radar_sep, imu_frames, N_syn_ante, static)
    lidar_frames = LidarFrames(lidar_frames, lidar_ts, lidar_sep, transform)

    params = {
        "r_radar": r_radar,
        "c": c,
        "fs": fs,
        "freq_slope": freq_slope,
        "freq_st": freq_st,
        "freq_end": freq_end,
        "FOV": FOV,
        "N_rings": N_rings,
        "N_samples": N_samples,
        "r_rings": r_rings,
        "N_syn_ante": N_syn_ante,
        "lambda_i": lambda_i,
        "lambda_0": 3.8e-3,
        "N_AoA_bins": N_AoA_bins,
        # "st_beam_offset": st_beam_offset,
    }

    return radar_frames, lidar_frames, imu_frames, params


def get_lidar_2d_projection(lidar_frame: np.ndarray, transform: np.ndarray, elevation: float) -> np.ndarray:
    """Parse Ouster Lidar frames and return ndarray(x,y) n*2 for a certain elevation.
    Args:
        lidar_frame: point clouds with shape (64, #h_points, xyz(3) )
        transform: shape (2,2), rotation matrix from lidar to radar
        elevation: elevation angle, horizontal 0 and up positive, in rad.
    Returns:
        xy: N*2 xy points
    """
    elevations = np.linspace(np.pi / 4, -np.pi / 4, 64)
    channel = np.argmin(np.abs(elevations - elevation))

    return lidar_frame[channel, :, :2] @ transform


def get_motion_params(dataset_name: str, need_smooth: bool = True) -> Dict[str, np.ndarray]:
    """Load motion parmaeters (v, theta_v, etc...). Also smooth it if needed.
    Args:
        dataset_name: the name of the trajectory folder.
        root_folder_name: the name of the root folder that holds the SP data
        need_smooth: whether smoothing is needed
    Returns:
        motion: a dictionary having keys {
            v_esti, delta_s0_esti, theta_v_esti, omega_imu
            v_gt, delta_s0_gt, theta_v_gt, omega_gt }
    """
    motion = np.load(f'{dataset_name}/motion_output.npz')
    motion = dict(motion)  # convert to real dict

    # unwrap and correct direction system error
    motion['theta_v_esti'][motion['v_esti'] < 0] += np.pi  # -v = +180 deg
    motion['theta_v_esti'] %= 2 * np.pi
    motion['theta_v_esti'][motion['theta_v_esti'] > 5] -= 2 * np.pi
    motion['v_esti'] = np.abs(motion['v_esti'])
    motion['delta_s0_esti'] = np.abs(motion['delta_s0_esti'])
    motion['theta_v_esti'] -= np.deg2rad(5.5)  # sys error

    if not need_smooth:
        return motion

    # do some smoothing (Gaussian filtering)
    motion['v_esti'] = gaussian_filter(motion['v_esti'], sigma=1.0, truncate=4)
    motion['delta_s0_esti'] = gaussian_filter(motion['delta_s0_esti'], sigma=1.0, truncate=4)
    motion['theta_v_esti'] = gaussian_filter(motion['theta_v_esti'], sigma=1.0, truncate=4)
    motion['v_gt'] = gaussian_filter(motion['v_gt'], sigma=1.0, truncate=4)
    motion['delta_s0_gt'] = gaussian_filter(motion['delta_s0_gt'], sigma=1.0, truncate=4)
    motion['theta_v_gt'] = gaussian_filter(motion['theta_v_gt'], sigma=1.0, truncate=4)
    motion['omega_imu'] = gaussian_filter(motion['omega_imu'], sigma=1.0, truncate=4)
    motion['omega_gt'] = gaussian_filter(motion['omega_gt'], sigma=1.0, truncate=4)

    return motion
