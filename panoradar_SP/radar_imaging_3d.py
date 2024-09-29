import numpy as np
import cupy as cp


def vertical_beam_forming_compen(
    elevation: float,
    lambda_i: cp.ndarray,
    ante_spacing: float = 1.9e-3,
) -> cp.ndarray:
    """Perform vertical beam forming to focus on a certain elevation angle.
    Args:
        elevation: the elevation angle, in rad. The horizontal plane is zero,
            upper part is positive and lower part is negative.
        lambda_i: wavelength of each sample points, (N_samples,)
        ante_spacing: the spacing between two Rx antennas, in m
    Returns:
        vbf_compen: the compensation for vertical beamforming. Shape (1, N_ante, N_samples)
    """
    dist = ante_spacing * np.sin(elevation) * cp.array([7, 6, 5, 4, 3, 2, 1, 0])
    dist = dist.reshape(1, -1, 1)
    lambda_i = lambda_i.reshape(1, 1, -1)
    #
    compensation = cp.exp(-2j * np.pi * dist / lambda_i)
    return compensation.astype(cp.complex64)


def vertical_beam_forming(radar_frame: cp.ndarray, vbf_compen: cp.ndarray) -> cp.ndarray:
    """Do vertical beamforming
    Args:
        signal: the circular signal. Shape (N_frames + N_syn_ante, N_ante, N_samples)
        vbf_compen: Shape (1, N_ante, N_samples)
    Returns:
        signal_bf: shape(N_frames + N_syn_ante, N_samples)
    """
    return cp.mean(radar_frame * vbf_compen, axis=1)


def get_window(N_syn_ante: int, N_samples: int) -> cp.ndarray:
    """Get hanning window for combining different chirps and FFT."""
    return cp.hanning(N_syn_ante)[:, None] * cp.hanning(N_samples)[None]


def azimuth_compensation(
    r_radar: float,
    delta_theta: float,
    elev: float,
    N_syn_ante: int,
    lambda_i: cp.ndarray,
    window: cp.ndarray,
) -> cp.ndarray:
    """Get the azimuth compensation term to make all chirps aligned with the same direction.
    This compensation makes them aligned at the rotation center
    Args:
        r_radar: radius of the radar, m
        delta_theta: azimuth angle between two virtual antennas, rad
        elev: the elevation angle, in rad
        N_syn_ante: the number of synthetic antenna
        lambda_i: wavelength of each sample points, (N_samples,)
        window: the window function, multiply to the azimuth_compen
    Returns:
        azimuth_compen: the compensation term for azimuth. Shape (N_syn_ante, N_samples)
    """
    assert N_syn_ante % 2 == 1, "N_syn_ante should be odd!"
    half_N_syn_ante = (N_syn_ante - 1) // 2

    h_theta = delta_theta * cp.arange(half_N_syn_ante, -half_N_syn_ante - 1, -1)
    h_theta = h_theta.reshape(-1, 1)
    ante_x = r_radar * cp.cos(h_theta)
    lambda_i = lambda_i.reshape(1, -1)  # (1, N_samples)
    azimuth_compen = cp.exp(4j * np.pi * ante_x * cp.cos(elev) / lambda_i) * window

    return azimuth_compen.astype(cp.complex64)


def motion_imaging(
    signal_bf: cp.ndarray,
    azimuth_compen: cp.ndarray,
    elevation: float,
    delta_s0: float,
    theta_v: float,
    beam_angles: cp.ndarray,
    lambda_i: cp.ndarray,
) -> cp.ndarray:
    """The imaging algorithm for the moving plation. The moving velocity is given.
    Args:
        elevation: the elevation angle, in rad
        delta_s0: The maximum movement for delta_t.
                   delta_s0 = -A * lambda_0 / (2 * N_AoA_bins)
        theta_v: velocity direction, angle between v and beam 0, left+ right-
        beam_angles: the facing angle of each beam, 0 ~ -2pi
        lambda_0: the wavelength
    Returns:
        heatmap: the imaging result of shape (N_rings, N_beams)
    """
    N_syn_ante = azimuth_compen.shape[0]
    N_half = (N_syn_ante - 1) // 2
    N_beams = signal_bf.shape[0] - N_syn_ante + 1
    lambda_i = lambda_i.reshape(1, 1, -1)

    delta_s = (delta_s0 * cp.cos(beam_angles - theta_v) * cp.cos(elevation)).reshape(-1, 1, 1)
    move_d = cp.arange(N_beams).reshape(-1, 1, 1) + cp.arange(-N_half, N_half + 1).reshape(1, -1, 1)
    move_d = 2 * move_d * delta_s  # shape (N_beams, N_syn_ante, 1)
    move_compen = cp.exp(2j * np.pi * move_d / lambda_i).astype(cp.complex64)

    indices = np.arange(N_beams, step=1)[:, None] + np.arange(N_syn_ante)[None]
    heatmap = cp.sum(signal_bf[indices] * azimuth_compen * move_compen, axis=1)
    heatmap = cp.abs(cp.fft.fft(heatmap, axis=1)).T

    return heatmap


def static_imaging(signal_bf: cp.ndarray, azimuth_compen: cp.ndarray, N_beams: int) -> cp.ndarray:
    """The imaging algorithm for the moving plation. The moving velocity is given.
    Args:
        signal_bf: the signal after vertical beamforming, shape (N_beams+N_syn_ante-1, N_samples)
        azimuth_compen: azimuth compensation, shape (N_syn_ante, N_samples)
        N_beams: the number of beams
    Returns:
        heatmap: the imaging result of shape (N_rings, N_beams)
    """
    N_syn_ante = azimuth_compen.shape[0]

    azimuth_compen = azimuth_compen[None]  # (1, N_syn_ante, N_samples)
    indices = np.arange(N_beams, step=1)[:, None] + np.arange(N_syn_ante)[None]

    heatmap = cp.sum(signal_bf[indices] * azimuth_compen, axis=1)  # (N_beams, N_samples)
    heatmap = cp.abs(cp.fft.fft(heatmap, axis=1)).T  # (N_rings, N_beams)

    return heatmap


def beam_origin_compen(delta_s0, theta_v, N_beams) -> np.ndarray:
    """Get the beam origin compensation when projecting to Catesian
    Args:
        delta_s0: the maximum movement in a delta_t interval
        theta_v: The moving direction
    Returns:
        origin_compen: The beam origin compensation, (2, N_beams)
    """
    origin_compen_x = np.arange(N_beams) * delta_s0 * np.cos(theta_v)
    origin_compen_y = np.arange(N_beams) * delta_s0 * np.sin(theta_v)
    origin_compen = np.vstack((origin_compen_x, origin_compen_y))
    return origin_compen
