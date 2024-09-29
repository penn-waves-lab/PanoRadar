"""This file do velocity estimation for the spinning radar and do motion imaging."""

import numpy as np
import cupy as cp
import scipy.optimize
import cupyx.scipy.ndimage
import skimage
import cv2


def detect_lines(binary_img, rho=1, theta=np.pi / 36, threshold=30, min_line_length=30, max_line_gap=5):
    lines = cv2.HoughLinesP(binary_img.T, rho, theta, threshold, None, min_line_length, max_line_gap)
    if lines is not None:
        lines = lines[:, :, [1, 0, 3, 2]]
    return lines


def binarize_and_thining(range_aoa_s: cp.ndarray, vertical_winsize: int = 3, alpha: float = 8.0) -> np.ndarray:
    """Binarize the image to make lines thin.
    Args:
        range_aoa_s: shape (N_beams, N_AoA_bins, N_samples)
        vertical_winsize: window size for the maximum filter along AoA axis
        alpha: coefficient for global noise suppression
    Return:
        binary_imgs: ready for cv2 line_detection, shape (N_beams, N_AoA_bins, N_samples)
    """
    vertical_max = cupyx.scipy.ndimage.maximum_filter(range_aoa_s, size=(1, vertical_winsize, 1))
    global_noise = cp.std(range_aoa_s, axis=(0, 1), keepdims=True)
    out = cp.logical_and(range_aoa_s == vertical_max, range_aoa_s > alpha * global_noise)

    return cp.asnumpy(out).astype(np.uint8)


def peaks_on_lines(merged_lines, img):
    peak_rs = []
    peak_cs = []
    for l in merged_lines:
        (c0, r0, c1, r1) = l
        c0 = np.clip(c0, 0, img.shape[1] - 1)
        c1 = np.clip(c1, 0, img.shape[1] - 1)
        r0 = np.clip(r0, 0, img.shape[0] - 1)
        r1 = np.clip(r1, 0, img.shape[0] - 1)
        rr, cc = skimage.draw.line(r0, c0, r1, c1)
        p_on_l = img[rr, cc]
        r = (rr * p_on_l).sum() / p_on_l.sum()
        c = (cc * p_on_l).sum() / p_on_l.sum()
        peak_rs.append(np.round(r).astype(int))
        peak_cs.append(np.round(c).astype(int))
    return np.array(peak_cs), np.array(peak_rs)


def faster_range_AoA_images(signal_bf: cp.ndarray, azimuth_compen: cp.ndarray, params) -> cp.ndarray:
    """Get range-AoA image in a faster way."""
    N_syn_ante = params['N_syn_ante']
    N_AoA_bins = params['N_AoA_bins']
    N_beams = signal_bf.shape[0] - azimuth_compen.shape[0] + 1

    signal_stacked = cp.stack([signal_bf[beam_id : beam_id + N_syn_ante] for beam_id in range(N_beams)])
    signal_compen_stacked = signal_stacked * azimuth_compen
    range_fft_stac = cp.fft.fft(signal_compen_stacked, axis=2)
    range_aoa_stac = cp.abs(cp.fft.fftshift(cp.fft.fft(range_fft_stac, axis=1, n=N_AoA_bins), axes=1))

    return range_aoa_stac


def fit_cos(angles, dopplers, valid_gap: float = 20):
    valid_inds = np.full_like(angles, True, dtype=bool)
    prev_valid_inds = np.copy(valid_inds)
    cos_fitting = lambda phi, A, phi_v, offset: A * np.cos(phi + phi_v) + offset

    for _ in range(20):
        params_hat, _ = scipy.optimize.curve_fit(
            cos_fitting,
            angles[valid_inds],
            dopplers[valid_inds],
            p0=[35, 0, 0],
        )
        # A_hat, phi_v_hat, offset_hat = params_hat
        pred = cos_fitting(angles, *params_hat)
        valid_inds = np.abs(dopplers - pred) < valid_gap
        if np.array_equal(valid_inds, prev_valid_inds):
            break
        prev_valid_inds = np.copy(valid_inds)

    return params_hat, valid_inds


def estimate_v_line_detect(signal_bf, azimuth_compen, N_beams, params):
    N_rings = params["N_rings"]
    N_AoA_bins = params["N_AoA_bins"]

    range_aoa_s = faster_range_AoA_images(signal_bf, azimuth_compen, params)  # cp
    binary_imgs = binarize_and_thining(range_aoa_s, vertical_winsize=20, alpha=12)
    range_aoa_s = cp.asnumpy(range_aoa_s)

    # free GPU memory
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    all_peak_cs = []
    all_peak_rs = []
    for range_bin in range(15, N_rings - 15):
        micro_AoA = range_aoa_s[:, :, range_bin].T  # (N_AoA, N_beams)
        binary_img = binary_imgs[:, :, range_bin].T  # (N_AoA, N_beams)

        lines = detect_lines(binary_img, rho=0.5, theta=np.pi / 360, threshold=40, min_line_length=50, max_line_gap=5)

        if lines is not None:
            lines = lines[:, 0]  # shape (N, 4)
            peak_cs, peak_rs = peaks_on_lines(lines, micro_AoA)

            all_peak_cs.append(peak_cs)
            all_peak_rs.append(peak_rs)

    all_peak_cs = np.hstack(all_peak_cs)
    all_peak_rs = np.hstack(all_peak_rs)
    all_peak_cs = all_peak_cs / N_beams * 2 * np.pi
    all_peak_rs = all_peak_rs - N_AoA_bins / 2

    # only use the points in the middle. Avoid edge cases.
    valid_inds = (all_peak_cs > 0.5) & (all_peak_cs < 5.98)
    all_peak_cs = all_peak_cs[valid_inds]
    all_peak_rs = all_peak_rs[valid_inds]

    (A_hat, phi_v_hat, offset_hat), valid_inds = fit_cos(all_peak_cs, all_peak_rs, valid_gap=15)

    return (A_hat, phi_v_hat), offset_hat, valid_inds, all_peak_cs, all_peak_rs
