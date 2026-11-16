"""Display a saved array spectrum on its physical azimuth–zenith coordinates."""

import numpy as np

from shared.sensing.arrays import forward_hemisphere_angles


def angular_spectrum_mesh(angular):
    """Return warped cell corners and peak power over range/Doppler.

    Spatial FFT coordinates are direction cosines, not a rectangular grid of
    spherical angles. Clip cell edges to the forward hemisphere and mask bins
    whose centers are outside it. No interpolation or new beamforming is used.
    """
    elevation = angular.elevation_direction_cosine
    azimuth = angular.azimuth_direction_cosine
    if min(len(elevation), len(azimuth)) < 2:
        raise ValueError("a two-dimensional angular product is required")

    def edges(axis):
        return np.r_[
            axis[0] - (axis[1] - axis[0]) / 2,
            (axis[:-1] + axis[1:]) / 2,
            axis[-1] + (axis[-1] - axis[-2]) / 2,
        ]

    z, y = np.meshgrid(np.clip(edges(elevation), -1, 1), edges(azimuth), indexing="ij")
    radius = np.sqrt(np.maximum(1 - z * z, 0))
    y = np.clip(y, -radius, radius)
    x = np.sqrt(np.maximum(1 - y * y - z * z, 0))
    azimuth_deg = np.degrees(np.arctan2(y, x))
    zenith_deg = np.degrees(np.arccos(z))
    physical, _, _ = forward_hemisphere_angles(elevation, azimuth)
    power = np.max(angular.power_linear, axis=(2, 3))
    return azimuth_deg, zenith_deg, np.where(physical, power, np.nan)
