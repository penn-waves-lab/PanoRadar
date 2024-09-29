import cv2
import numpy as np
from scipy.ndimage import correlate1d

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class SurfaceNormal:
    """Surface Normal from range images."""

    def __init__(
        self,
        N_azimuth: int,
        azi_range: Tuple[float, float],
        N_elev: int,
        elev_range: Tuple[float, float],
    ):
        """Init.
        Args:
            N_azimuth: the number of azimuth channels
            azi_range: azimuth range [min, max], in rad
            N_elev: the number of elevation channels
            elev_range: elevation angle range [min, max] in rad
        """
        theta = np.linspace(*azi_range, N_azimuth).reshape(1, -1)
        phi = np.linspace(*elev_range, N_elev).reshape(-1, 1)

        self.delta_theta = theta[0, 0] - theta[0, 1]
        self.delta_phi = phi[0, 0] - phi[1, 0]
        self.cos_phi = np.cos(phi)
        self.R_theta_phi = self.get_sf_rot_matrix(theta, phi)
        self.R_theta_phi_inv = np.linalg.inv(self.R_theta_phi)

    def get_sf_rot_matrix(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Get the rotation matrix R_{theta, phi} for the surface normal.
        Returns:
            R_theta_phi: the rotation matrix, shape (N_elev, N_azimuth, 3, 3)
        """
        r0 = np.stack((np.cos(theta), -np.sin(theta), np.zeros_like(theta)), axis=2)
        r1 = np.stack((np.sin(theta), np.cos(theta), np.zeros_like(theta)), axis=2)
        r2 = np.stack((np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)), axis=2)
        R_theta = np.stack((r0, r1, r2), axis=3)

        r0 = np.stack((np.cos(phi), np.zeros_like(phi), -np.sin(phi)), axis=2)
        r1 = np.stack((np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)), axis=2)
        r2 = np.stack((np.sin(phi), np.zeros_like(phi), np.cos(phi)), axis=2)
        R_phi = np.stack((r0, r1, r2), axis=3)

        R_theta_phi = R_theta @ R_phi  # shape (N_elev, N_azimuth, 3, 3)
        return R_theta_phi

    def compute_surface_normal(self, range_img: np.ndarray) -> np.ndarray:
        """Compute the surface normal for the range image
        Args:
            range_img: the input range image, shape (N_elev, N_azimuth)
        Returns:
            normal: the normal (xyz) at each pixel location, shape (N_elev, N_azimuth, 3)
        """
        range_img = np.copy(range_img)
        range_img[range_img <= 0] = np.nan

        kernel_th = np.array([1.0, 1.0, 1.0, 0, -1.0, -1.0, -1.0])
        kernel_ph = np.array([1.0, 0, -1.0])
        dRdth = correlate1d(range_img, kernel_th, axis=1, mode='wrap') / (12 * self.delta_theta)
        dRdph = correlate1d(range_img, kernel_ph, axis=0, mode='constant', cval=np.nan) / (2 * self.delta_phi)

        rhs = (
            np.ones_like(dRdth),
            1 / (range_img * self.cos_phi) * dRdth,
            1 / range_img * dRdph,
        )
        rhs = np.stack(rhs, axis=2)[:, :, :, np.newaxis]  # shape (N_elev, N_azimuth, 3, 1)
        normal = np.squeeze(self.R_theta_phi @ rhs)

        return normal

    def compute_surface_normal_min(self, range_img: np.ndarray) -> np.ndarray:
        """Compute the surface normal for the range image
        Args:
            range_img: the input range image, shape (N_elev, N_azimuth)
        Returns:
            normal: the normal (xyz) at each pixel location, shape (N_elev, N_azimuth, 3)
        """
        range_img = np.copy(range_img)
        range_img[range_img <= 0] = np.nan

        kernel_left = np.array([1.0, -1.0, 0])
        kernel_right = np.array([0, 1.0, -1.0])

        dRdth_left = correlate1d(range_img, kernel_left, axis=1, mode='wrap') / (1 * self.delta_theta)
        dRdth_right = correlate1d(range_img, kernel_right, axis=1, mode='wrap') / (1 * self.delta_theta)
        dRdth = np.where(np.abs(dRdth_left) < np.abs(dRdth_right), dRdth_left, dRdth_right)

        dRdph_left = correlate1d(range_img, kernel_left, axis=0, mode='constant', cval=np.nan) / (1 * self.delta_phi)
        dRdph_right = correlate1d(range_img, kernel_right, axis=0, mode='constant', cval=np.nan) / (1 * self.delta_phi)
        dRdph = np.where(np.abs(dRdph_left) < np.abs(dRdph_right), dRdph_left, dRdph_right)

        rhs = (
            np.ones_like(dRdth),
            1 / (range_img * self.cos_phi) * dRdth,
            1 / range_img * dRdph,
        )
        rhs = np.stack(rhs, axis=2)[:, :, :, np.newaxis]  # shape (N_elev, N_azimuth, 3, 1)
        normal = np.squeeze(self.R_theta_phi @ rhs)

        return normal

    def get_rhs_from_normal(self, normal: np.ndarray, mask_cable_area: bool = False) -> np.ndarray:
        """Get the right-hand-side [1/(r cos phi) dr/d theta, 1/r dr/d phi] from
        the surface normal map.
        Args:
            normal: surface normal xyz, shape (N_elev, N_azimuth, 3)
            mask_cable_area: make the mask for the cable area larger
        Returns:
            rhs: the right hand side which is position invariant, shape (N_elev, N_azimuth, 3)
        """
        invalid_region = np.all(normal < -10, axis=2)

        if mask_cable_area:
            lower_region = invalid_region[44:63, :].astype(np.uint8) * 255
            kernel1 = np.ones((3, 7), np.uint8)
            kernel2 = np.ones((1, 7), np.uint8)
            lower_region = cv2.dilate(lower_region, kernel1)
            lower_region = cv2.dilate(lower_region, kernel2).astype(bool)
            invalid_region[44:63] = lower_region

        rhs = (self.R_theta_phi_inv @ normal[:, :, :, None]).squeeze()
        rhs[invalid_region] = -1e3
        return rhs

    def get_normal_from_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Get the surface normal from the right hand side.
        Args:
            rhs: [1/(r cos phi) dr/d theta, 1/r dr/d phi], shape (N_elev, N_azimuth, 3)
        Returns:
            normal: surface normal xyz, shape (N_elev, N_azimuth, 3)
        """
        invalid_region = np.all(rhs < -10, axis=2)
        normal = (self.R_theta_phi @ rhs[:, :, :, None]).squeeze()
        normal[invalid_region] = -1e3
        return normal


def normal_color_mapping(normal: np.ndarray) -> np.ndarray:
    """Map surface normal (x,y,z) to RGB image for visualization.
       x (-1,1)  -> R (0, 255)
       y (-1,1)  -> R (0, 255)
       z (-1,1)  -> R (127, 255)
    Args:
        normal: the normal (xyz) at each pixel location, shape (N_elev, N_azimuth, 3)
    Returns:
        n_color: normal with color. shape (N_elev, N_azimuth, 3)
    """
    n_color = normal / np.linalg.norm(normal, ord=2, axis=2, keepdims=True)  # (-1, 1)
    n_color[:, :, :2] = np.round((n_color[:, :, :2] + 1) / 2 * 255)
    n_color[:, :, 2] = np.round((n_color[:, :, 2] + 1) / 2 * 127) + 127
    n_color[np.isnan(n_color)] = 0  # deal with NaN, set color to black (0,0,0)

    return n_color.astype(np.uint8)


def color_back_to_normal(n_color: np.ndarray) -> np.ndarray:
    """Convert the RGB color back to normal xyz
        R (0, 255)   ->  x (-1,1)
        R (0, 255)   ->  y (-1,1)
        R (127, 255) ->  z (-1,1)
    Args:
        n_color: normal with color. shape (N_elev, N_azimuth, 3)
    Returns:
        normal: the normal (xyz) at each pixel location, shape (N_elev, N_azimuth, 3)
            Set invalid region to -1e3
    """
    invalid_region = np.all(n_color == 0, axis=2)

    normal = np.copy(n_color).astype(np.float32)
    normal[:, :, :2] /= 255 / 2
    normal[:, :, 2] -= 127
    normal[:, :, 2] /= 127 / 2
    normal -= 1

    normal[invalid_region] = -1e3
    return normal


def get_smooth_surface_normal(sn: SurfaceNormal, range_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Utility. Get the smooth surface normal from range image after filteration.
    Args:
        sn: The SurfaceNormal class object
        range_img: the input range image, shape (N_elev, N_azimuth)
    Returns:
        n_xyz: surface normal in xyz, shape (N_elev, N_azimuth, 3), in range (-1,1)
            invalid region is set to -1e3
        n_color: surface normal in RGB, for vis purpose, shape (N_elev, N_azimuth, 3)
    """
    range_img = cv2.medianBlur(range_img, 3)
    # normal = sn.compute_surface_normal(range_img)
    normal = sn.compute_surface_normal_min(range_img)
    n_color = normal_color_mapping(normal)
    n_color = cv2.bilateralFilter(n_color, 15, 50, 50)
    normal = color_back_to_normal(n_color)

    return normal, n_color


sn = SurfaceNormal(512, (0, -2 * np.pi), 64, (np.pi / 4, -np.pi / 4))
