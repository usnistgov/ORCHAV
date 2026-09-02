"""Helpers for generating beamforming visualizations in Open3D."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

_C = 299_792_458.0  # Speed of light (m/s)

# Bound the element-by-direction workspace used by the array-factor routines.
# This applies to both GUI-created arrays and arbitrary arrays supplied by
# frame metadata.
MAX_BEAM_PATTERN_WORK_ITEMS = 8_000_000

# Lightweight colormaps avoid importing matplotlib at module load.

_COLORMAP_CACHE: dict[str, np.ndarray] = {}


def _get_colormap_lut(name: str, n: int = 256) -> np.ndarray:
    """Return an (n, 3) float64 LUT for the given colormap name."""
    key = (name, n)
    if key not in _COLORMAP_CACHE:
        try:
            from matplotlib import colormaps

            cmap = colormaps[name]
            lut = cmap(np.linspace(0.0, 1.0, n))[:, :3].astype(np.float64)
        except (ImportError, KeyError):
            # Fallback: red → blue linear ramp
            t = np.linspace(0.0, 1.0, n, dtype=np.float64)
            lut = np.column_stack([t, 0.3 + 0.7 * (1.0 - t), 1.0 - t])
        _COLORMAP_CACHE[key] = lut
    return _COLORMAP_CACHE[key]


# Element pattern functions


def _dipole_pattern(theta: np.ndarray) -> np.ndarray:
    """Half-wave dipole element pattern (voltage, not power)."""
    return np.abs(np.sin(theta))


def _tr38901_pattern(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """3GPP TR 38.901 single-element pattern (voltage, not power).

    Peak at (θ=90°, φ=0°), i.e. boresight along +X.
    Vertical HPBW ≈ 65°, horizontal HPBW ≈ 65°.
    SLA_v = 30 dB, A_m = 30 dB.
    """
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    # Vertical: A_V(θ) = -min(12*((θ-90)/65)^2, 30)
    a_v = -np.minimum(12.0 * ((theta_deg - 90.0) / 65.0) ** 2, 30.0)
    # Horizontal: A_H(φ) = -min(12*(φ/65)^2, 30)
    a_h = -np.minimum(12.0 * (phi_deg / 65.0) ** 2, 30.0)
    # Combined: A(θ,φ) = -min(-(A_V + A_H), 30) + 8 dBi
    a_db = -np.minimum(-(a_v + a_h), 30.0) + 8.0
    # Field magnitude is the square root of linear power, normalized to peak.
    amplitude = 10.0 ** (a_db / 20.0)
    peak = amplitude.max()
    if peak > 0:
        amplitude /= peak
    return amplitude


# Internal helpers


def _ensure_complex_weights(weights: np.ndarray) -> np.ndarray:
    """Return complex-valued weight vector from raw storage."""
    arr = np.asarray(weights)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)
    return arr.astype(np.complex128, copy=False).reshape(-1)


def bounded_pattern_sample_counts(
    weights: np.ndarray,
    azimuth_samples: int,
    elevation_samples: int,
    *,
    minimum_azimuth_samples: int = 12,
    minimum_elevation_samples: int = 9,
) -> tuple[int, int]:
    """Return sample counts that fit the array-factor memory work budget.

    Elevation is reduced first because it usually has fewer samples and this
    preserves azimuthal shape detail. A ``ValueError`` is raised only when the
    element count cannot fit even the requested minimum grid.
    """
    num_elements = int(_ensure_complex_weights(weights).size)
    if num_elements <= 0:
        raise ValueError("weights array is empty")

    azimuth = max(1, int(azimuth_samples))
    elevation = max(1, int(elevation_samples))
    # Never increase a caller's requested resolution merely to establish the
    # lower bound used when a large array has to be reduced.
    min_azimuth = min(azimuth, max(1, int(minimum_azimuth_samples)))
    min_elevation = min(elevation, max(1, int(minimum_elevation_samples)))

    if num_elements * azimuth * elevation <= MAX_BEAM_PATTERN_WORK_ITEMS:
        return azimuth, elevation

    allowed_elevation = MAX_BEAM_PATTERN_WORK_ITEMS // max(1, num_elements * azimuth)
    elevation = min(elevation, max(min_elevation, int(allowed_elevation)))

    if num_elements * azimuth * elevation > MAX_BEAM_PATTERN_WORK_ITEMS:
        allowed_azimuth = MAX_BEAM_PATTERN_WORK_ITEMS // max(1, num_elements * elevation)
        azimuth = min(azimuth, max(min_azimuth, int(allowed_azimuth)))

    if num_elements * azimuth * elevation > MAX_BEAM_PATTERN_WORK_ITEMS:
        raise ValueError("antenna array is too large for the minimum beam-pattern sampling grid")
    return azimuth, elevation


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return rotation matrix for yaw(Z), pitch(Y), roll(X) convention."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


# Main mesh generation


def generate_beamforming_mesh(
    weights: np.ndarray,
    element_positions: np.ndarray,
    carrier_frequency_hz: float,
    *,
    orientation_radians: Optional[tuple[float, float, float]] = None,
    origin: Optional[np.ndarray] = None,
    scale: float = 1.0,
    azimuth_samples: int = 72,
    elevation_samples: int = 37,
    db_scale: bool = False,
    dynamic_range_db: float = 40.0,
    colormap: str = "jet",
    element_pattern: str = "isotropic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices, triangles, colors) describing an antenna pattern surface.

    Args:
        weights: Complex array of shape ``(N,)`` or ``(N, 2)`` (real/imag).
        element_positions: ``(N, 3)`` array of element positions in metres.
        carrier_frequency_hz: Carrier frequency in Hz.
        orientation_radians: ``(yaw, pitch, roll)`` to rotate the pattern.
        origin: ``(3,)`` translation applied after rotation.
        scale: Radial scale factor (1.0 = unit sphere at peak).
        azimuth_samples: Number of azimuth grid points.
        elevation_samples: Number of elevation grid points.
        db_scale: If *True*, map amplitude to dB before colour/radius mapping.
        dynamic_range_db: Floor (dB below peak) when *db_scale* is True.
        colormap: Matplotlib colormap name (e.g. ``"jet"``, ``"viridis"``).
        element_pattern: Per-element radiation pattern to multiply with the
            array factor.  One of ``"isotropic"``, ``"dipole"``, ``"tr38901"``.
    """
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive")

    w = _ensure_complex_weights(weights)
    if w.size == 0:
        raise ValueError("weights array is empty")

    azimuth_samples, elevation_samples = bounded_pattern_sample_counts(
        w,
        azimuth_samples,
        elevation_samples,
    )

    positions = np.asarray(element_positions, dtype=np.float64)
    if positions.shape[0] != w.size:
        raise ValueError("weights and element_positions must have matching length")

    wavelength = _C / float(carrier_frequency_hz)
    k = 2.0 * math.pi / wavelength

    phi = np.linspace(-math.pi, math.pi, azimuth_samples, endpoint=False, dtype=np.float64)
    theta = np.linspace(0.0, math.pi, elevation_samples, dtype=np.float64)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    sin_theta = np.sin(theta_grid)
    cos_theta = np.cos(theta_grid)
    cos_phi = np.cos(phi_grid)
    sin_phi = np.sin(phi_grid)

    # Array factor
    x_term = positions[:, 0][:, np.newaxis, np.newaxis] * sin_theta * cos_phi
    y_term = positions[:, 1][:, np.newaxis, np.newaxis] * sin_theta * sin_phi
    z_term = positions[:, 2][:, np.newaxis, np.newaxis] * cos_theta

    phase = np.exp(1j * k * (x_term + y_term + z_term))
    field = np.sum(w[:, np.newaxis, np.newaxis] * phase, axis=0)
    amplitude = np.abs(field)

    if element_pattern == "dipole":
        amplitude *= _dipole_pattern(theta_grid)
    elif element_pattern == "tr38901":
        amplitude *= _tr38901_pattern(theta_grid, phi_grid)
    # "isotropic" → no modification

    # Normalise to [0, 1]
    max_amp = float(np.max(amplitude)) if amplitude.size else 1.0
    if max_amp <= 0.0:
        max_amp = 1.0
    amplitude /= max_amp

    # dB-scale mapping
    if db_scale:
        amp_db = 20.0 * np.log10(np.maximum(amplitude, 1e-10))
        floor = -abs(dynamic_range_db)
        amplitude = np.clip((amp_db - floor) / (-floor), 0.0, 1.0)

    # Radius and vertex positions
    radius = amplitude * float(scale)
    xs = radius * sin_theta * cos_phi
    ys = radius * sin_theta * sin_phi
    zs = radius * cos_theta

    vertices = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)

    # Vectorised triangulation
    n_az = azimuth_samples
    n_el = elevation_samples
    ii, jj = np.meshgrid(np.arange(n_el - 1), np.arange(n_az), indexing="ij")
    nj = (jj + 1) % n_az
    idx0 = (ii * n_az + jj).ravel()
    idx1 = (ii * n_az + nj).ravel()
    idx2 = ((ii + 1) * n_az + jj).ravel()
    idx3 = ((ii + 1) * n_az + nj).ravel()
    triangles = np.empty((len(idx0) * 2, 3), dtype=np.int32)
    triangles[0::2, 0] = idx0
    triangles[0::2, 1] = idx2
    triangles[0::2, 2] = idx1
    triangles[1::2, 0] = idx1
    triangles[1::2, 1] = idx2
    triangles[1::2, 2] = idx3

    # Color mapping via LUT
    norm_amp = amplitude.reshape(-1)
    lut = _get_colormap_lut(colormap)
    indices = np.clip((norm_amp * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
    colors = lut[indices]

    if orientation_radians is not None:
        rot = _rotation_matrix(*orientation_radians)
        vertices = vertices @ rot.T

    if origin is not None:
        origin_arr = np.asarray(origin, dtype=np.float64)
        vertices += origin_arr

    return vertices.astype(np.float64), triangles, colors


# Pattern metrics


def compute_pattern_metrics(
    weights: np.ndarray,
    element_positions: np.ndarray,
    carrier_frequency_hz: float,
    *,
    element_pattern: str = "isotropic",
    azimuth_samples: int = 360,
    elevation_samples: int = 181,
) -> dict[str, float]:
    """Compute antenna pattern metrics (peak gain, HPBW, SLL).

    Returns a dict with keys:
    - ``peak_gain_dbi``: Peak directivity in dBi.
    - ``hpbw_az_deg``: Half-power beamwidth in the azimuth plane.
    - ``hpbw_el_deg``: Half-power beamwidth in the elevation plane.
    - ``sll_db``: First sidelobe level relative to peak (negative dB).
    """
    w = _ensure_complex_weights(weights)
    if w.size == 0:
        return {"peak_gain_dbi": 0.0, "hpbw_az_deg": 0.0, "hpbw_el_deg": 0.0, "sll_db": 0.0}

    try:
        azimuth_samples, elevation_samples = bounded_pattern_sample_counts(
            w,
            azimuth_samples,
            elevation_samples,
        )
    except ValueError:
        return {"peak_gain_dbi": 0.0, "hpbw_az_deg": 0.0, "hpbw_el_deg": 0.0, "sll_db": 0.0}

    positions = np.asarray(element_positions, dtype=np.float64)
    wavelength = _C / float(carrier_frequency_hz)
    k = 2.0 * math.pi / wavelength

    phi = np.linspace(-math.pi, math.pi, azimuth_samples, endpoint=False)
    theta = np.linspace(0.0, math.pi, elevation_samples)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    sin_theta = np.sin(theta_grid)
    cos_theta = np.cos(theta_grid)

    x_term = positions[:, 0][:, np.newaxis, np.newaxis] * sin_theta * np.cos(phi_grid)
    y_term = positions[:, 1][:, np.newaxis, np.newaxis] * sin_theta * np.sin(phi_grid)
    z_term = positions[:, 2][:, np.newaxis, np.newaxis] * cos_theta

    phase = np.exp(1j * k * (x_term + y_term + z_term))
    field = np.sum(w[:, np.newaxis, np.newaxis] * phase, axis=0)
    amplitude = np.abs(field)

    if element_pattern == "dipole":
        amplitude *= _dipole_pattern(theta_grid)
    elif element_pattern == "tr38901":
        amplitude *= _tr38901_pattern(theta_grid, phi_grid)

    peak = float(amplitude.max())
    if peak <= 0:
        return {"peak_gain_dbi": 0.0, "hpbw_az_deg": 0.0, "hpbw_el_deg": 0.0, "sll_db": 0.0}

    # Peak gain (dBi): directivity relative to isotropic
    # D = 4π * |F_peak|² / ∫∫|F|² sin(θ) dθ dφ
    power = amplitude**2
    d_theta = theta[1] - theta[0] if len(theta) > 1 else math.pi
    d_phi = phi[1] - phi[0] if len(phi) > 1 else 2 * math.pi
    total_power = float(np.sum(power * sin_theta) * d_theta * d_phi)
    if total_power > 0:
        directivity = 4.0 * math.pi * peak**2 / total_power
        peak_gain_dbi = 10.0 * math.log10(max(directivity, 1e-10))
    else:
        peak_gain_dbi = 0.0

    # Find peak location
    peak_idx = np.unravel_index(np.argmax(amplitude), amplitude.shape)
    peak_el_idx, peak_az_idx = peak_idx

    # HPBW in azimuth (cut at peak elevation)
    az_cut = amplitude[peak_el_idx, :]
    half_power = peak / math.sqrt(2.0)
    above_half = az_cut >= half_power
    hpbw_az_deg = float(np.count_nonzero(above_half)) * 360.0 / azimuth_samples

    # HPBW in elevation (cut at peak azimuth)
    el_cut = amplitude[:, peak_az_idx]
    above_half_el = el_cut >= half_power
    hpbw_el_deg = float(np.count_nonzero(above_half_el)) * 180.0 / elevation_samples

    # Sidelobe level: find the highest peak that isn't the main lobe
    main_lobe_mask = amplitude >= half_power
    sidelobe_amp = amplitude.copy()
    sidelobe_amp[main_lobe_mask] = 0.0
    sll_linear = float(sidelobe_amp.max()) / peak if peak > 0 else 0.0
    sll_db = 20.0 * math.log10(max(sll_linear, 1e-10))

    return {
        "peak_gain_dbi": round(peak_gain_dbi, 1),
        "hpbw_az_deg": round(hpbw_az_deg, 1),
        "hpbw_el_deg": round(hpbw_el_deg, 1),
        "sll_db": round(sll_db, 1),
    }
