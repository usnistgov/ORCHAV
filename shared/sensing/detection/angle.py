"""Angle-cube helpers for optional sensing frame extensions.

The functions here start from range-Doppler detections and a stored azimuth
power cube, then produce sparse angle candidates and display-friendly detection
coordinates. They operate on NumPy arrays from ``StandardMPCFrame["sensing"]``
payloads and do not depend on generator runtime objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnglePeakDetectionConfig:
    """Thresholds and gates for extracting azimuth peaks at RD detections."""

    max_peaks_per_rd_cell: int = 2
    peak_relative_floor_db: float = 10.0
    peak_nms_radius_bins: int = 1
    edge_exclusion_bins: int = 0
    secondary_peak_max_gap_db: float = 3.0
    secondary_peak_min_separation_bins: int = 2
    secondary_peak_min_prominence_db: float = 3.0
    # Profile SNR: secondary peak must be this many dB above the median
    # of the azimuth profile (the profile's noise floor).  Rejects sidelobe
    # maxima that are local peaks but not truly above the background.
    secondary_peak_min_profile_snr_db: float = 6.0
    # Valley depth: the valley between strongest and secondary must drop
    # by at least this many dB below the weaker peak.  Ensures two peaks
    # are truly resolved, not part of the same broad lobe.
    secondary_peak_min_valley_depth_db: float = 3.0
    cluster_azimuth_gate_deg: float | None = None
    sidelobe_suppression_enabled: bool = False
    sidelobe_range_tolerance_bins: int = 2
    sidelobe_doppler_tolerance_bins: int = 2
    sidelobe_azimuth_tolerance_bins: int = 3
    sidelobe_min_power_ratio_db: float = 4.0


@dataclass
class AnglePeakDetectionResult:
    """Sparse azimuth-candidate products aligned with a range-Doppler grid."""

    power_cube: np.ndarray
    candidate_mask: np.ndarray
    aoa_az_map: np.ndarray
    peak_count_map: np.ndarray


def _local_maxima_1d(profile: np.ndarray) -> np.ndarray:
    arr = np.asarray(profile, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"profile must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        return np.empty(0, dtype=np.int32)

    maxima: list[int] = []
    for idx in range(arr.size):
        value = float(arr[idx])
        if value <= 0.0 or not np.isfinite(value):
            continue
        left = float(arr[idx - 1]) if idx > 0 else -np.inf
        right = float(arr[idx + 1]) if idx + 1 < arr.size else -np.inf
        if value >= left and value >= right:
            maxima.append(idx)

    if maxima:
        return np.asarray(maxima, dtype=np.int32)
    if np.any(arr > 0.0):
        return np.asarray([int(np.argmax(arr))], dtype=np.int32)
    return np.empty(0, dtype=np.int32)


def extract_rd_anchored_angle_candidates(
    rd_peak_mask: np.ndarray,
    azimuth_cube: np.ndarray,
    azimuth_axis_deg: np.ndarray | list[float],
    *,
    config: AnglePeakDetectionConfig,
) -> AnglePeakDetectionResult:
    """Lift RD peaks into sparse azimuth candidates from a stored angle cube.

    ``rd_peak_mask`` selects the range-Doppler cells worth inspecting. For each
    selected cell, the azimuth profile is filtered for one or more resolved
    local peaks, with secondary peaks kept only when they are sufficiently
    separated and prominent relative to the profile floor.
    """
    rd_mask = np.asarray(rd_peak_mask, dtype=bool)
    cube = np.maximum(np.asarray(azimuth_cube, dtype=np.float64), 0.0)
    axis = np.asarray(azimuth_axis_deg, dtype=np.float64)

    if cube.ndim != 3:
        raise ValueError(f"azimuth_cube must be 3-D, got shape {cube.shape}")
    if rd_mask.shape != cube.shape[1:]:
        raise ValueError(
            f"rd_peak_mask must match azimuth_cube RD shape {cube.shape[1:]}, got {rd_mask.shape}"
        )
    if axis.shape != (cube.shape[0],):
        raise ValueError(
            "azimuth_axis_deg must match azimuth_cube azimuth bins: "
            f"expected {(cube.shape[0],)}, got {axis.shape}"
        )

    max_peaks = max(int(config.max_peaks_per_rd_cell), 1)
    peak_radius = max(int(config.peak_nms_radius_bins), 0)
    edge_exclusion_bins = max(int(config.edge_exclusion_bins), 0)
    secondary_gap_db = max(float(config.secondary_peak_max_gap_db), 0.0)
    secondary_min_sep = max(int(config.secondary_peak_min_separation_bins), 0)
    secondary_min_prominence_db = max(float(config.secondary_peak_min_prominence_db), 0.0)
    secondary_min_profile_snr_db = max(float(config.secondary_peak_min_profile_snr_db), 0.0)
    secondary_min_valley_depth_db = max(float(config.secondary_peak_min_valley_depth_db), 0.0)

    candidate_mask = np.zeros_like(cube, dtype=bool)
    peak_count_map = np.zeros(rd_mask.shape, dtype=np.int16)
    aoa_az_map = np.full(rd_mask.shape, np.nan, dtype=np.float32)

    for doppler_idx, range_idx in np.argwhere(rd_mask):
        profile = cube[:, doppler_idx, range_idx]
        local_maxima = _local_maxima_1d(profile)
        if edge_exclusion_bins > 0 and local_maxima.size > 0:
            local_maxima = local_maxima[
                (local_maxima >= edge_exclusion_bins)
                & (local_maxima < cube.shape[0] - edge_exclusion_bins)
            ]
        if local_maxima.size == 0:
            continue

        max_power = float(np.max(profile[local_maxima]))
        if max_power <= 0.0:
            continue
        min_power = 0.0
        if config.peak_relative_floor_db > 0.0:
            min_power = max_power / (10.0 ** (float(config.peak_relative_floor_db) / 10.0))

        ordered = local_maxima[np.argsort(-profile[local_maxima])]
        selected: list[int] = []
        strongest_idx = int(ordered[0])
        strongest_power = float(profile[strongest_idx])

        # Profile noise floor: median of positive profile values
        positive_vals = profile[profile > 0.0]
        profile_median = float(np.median(positive_vals)) if len(positive_vals) > 0 else 0.0

        for azimuth_idx in ordered:
            power = float(profile[azimuth_idx])
            if power < min_power:
                continue
            if any(abs(int(azimuth_idx) - kept_idx) <= peak_radius for kept_idx in selected):
                continue
            if selected:
                if strongest_power <= 0.0:
                    continue
                gap_db = 10.0 * np.log10(strongest_power / max(power, 1e-30))
                if gap_db > secondary_gap_db:
                    continue
                if abs(int(azimuth_idx) - strongest_idx) < secondary_min_sep:
                    continue
                # Prominence: valley between this peak and strongest must be deep
                lo = min(int(azimuth_idx), strongest_idx)
                hi = max(int(azimuth_idx), strongest_idx)
                valley_power = float(np.min(profile[lo : hi + 1]))
                prominence_db = 10.0 * np.log10(max(power, 1e-30) / max(valley_power, 1e-30))
                if prominence_db < secondary_min_prominence_db:
                    continue
                # Profile SNR: peak must be above the profile's noise floor
                if secondary_min_profile_snr_db > 0.0 and profile_median > 0.0:
                    profile_snr_db = 10.0 * np.log10(max(power, 1e-30) / profile_median)
                    if profile_snr_db < secondary_min_profile_snr_db:
                        continue
                # Valley depth: the valley must drop sufficiently below the weaker peak
                if secondary_min_valley_depth_db > 0.0 and valley_power > 0.0:
                    valley_depth_db = 10.0 * np.log10(max(power, 1e-30) / valley_power)
                    if valley_depth_db < secondary_min_valley_depth_db:
                        continue
            selected.append(int(azimuth_idx))
            if len(selected) >= max_peaks:
                break

        if not selected:
            selected = [int(ordered[0])]

        candidate_mask[selected, doppler_idx, range_idx] = True
        peak_count_map[doppler_idx, range_idx] = len(selected)
        aoa_az_map[doppler_idx, range_idx] = np.float32(axis[selected[0]])

    return AnglePeakDetectionResult(
        power_cube=cube,
        candidate_mask=candidate_mask,
        aoa_az_map=aoa_az_map,
        peak_count_map=peak_count_map,
    )


def augment_detections_with_sensor_frame_xy(
    detections: list[dict[str, Any]],
    *,
    range_mode: str = "monostatic",
) -> list[dict[str, Any]]:
    """Project monostatic radius and receiver-local bearing into RX coordinates.

    Bistatic range is the total TX-target-RX path length, not the RX radius.
    This boundary has no calibrated RX pose for ellipse localization, so it
    omits Cartesian coordinates in bistatic mode instead of inventing them.
    """
    if range_mode not in {"monostatic", "bistatic"}:
        raise ValueError("range_mode must be 'monostatic' or 'bistatic'")
    if range_mode == "bistatic":
        return [
            {k: v for k, v in det.items() if k not in {"x_m", "y_m", "z_m"}} for det in detections
        ]
    augmented: list[dict[str, Any]] = []
    for det in detections:
        out = dict(det)
        try:
            range_m = float(out.get("range_m", np.nan))
            azimuth_deg = float(out.get("aoa_az_deg", np.nan))
        except (TypeError, ValueError):
            range_m = float("nan")
            azimuth_deg = float("nan")

        if np.isfinite(range_m) and np.isfinite(azimuth_deg):
            azimuth_rad = np.radians(azimuth_deg)
            try:
                elevation_deg = float(out.get("aoa_el_deg", np.nan))
            except (TypeError, ValueError):
                elevation_deg = float("nan")
            if np.isfinite(elevation_deg):
                elevation_rad = np.radians(elevation_deg)
                horiz_range = range_m * np.cos(elevation_rad)
                out["x_m"] = float(horiz_range * np.cos(azimuth_rad))
                out["y_m"] = float(horiz_range * np.sin(azimuth_rad))
                out["z_m"] = float(range_m * np.sin(elevation_rad))
            else:
                out["x_m"] = float(range_m * np.cos(azimuth_rad))
                out["y_m"] = float(range_m * np.sin(azimuth_rad))
                out.setdefault("z_m", float("nan"))
        else:
            out["x_m"] = float("nan")
            out["y_m"] = float("nan")
            out.setdefault("z_m", float("nan"))
        augmented.append(out)
    return augmented


def suppress_angle_sidelobes(
    detections: list[dict[str, Any]],
    *,
    config: AnglePeakDetectionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Suppress weaker detections that closely match a stronger angle-bearing peak."""
    if not config.sidelobe_suppression_enabled or len(detections) < 2:
        return detections, {"enabled": False, "kept": len(detections), "rejected": 0}

    sorted_dets = sorted(
        detections,
        key=lambda det: float(det.get("power_linear", 0.0)),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    rejected = 0

    for det in sorted_dets:
        det_power = max(float(det.get("power_linear", 0.0)), 1e-30)
        det_range_bin = int(det.get("range_bin", -10_000))
        det_doppler_bin = int(det.get("doppler_bin", -10_000))
        det_azimuth_bin = int(det.get("azimuth_bin", -10_000))
        suppress = False

        for stronger in kept:
            stronger_power = max(float(stronger.get("power_linear", 0.0)), 1e-30)
            power_ratio_db = 10.0 * np.log10(stronger_power / det_power)
            if power_ratio_db < float(config.sidelobe_min_power_ratio_db):
                continue

            if (
                abs(int(stronger.get("range_bin", -10_000)) - det_range_bin)
                <= int(config.sidelobe_range_tolerance_bins)
                and abs(int(stronger.get("doppler_bin", -10_000)) - det_doppler_bin)
                <= int(config.sidelobe_doppler_tolerance_bins)
                and abs(int(stronger.get("azimuth_bin", -10_000)) - det_azimuth_bin)
                <= int(config.sidelobe_azimuth_tolerance_bins)
            ):
                suppress = True
                break

        if suppress:
            rejected += 1
            continue
        kept.append(det)

    return kept, {
        "enabled": True,
        "kept": len(kept),
        "rejected": rejected,
        "min_power_ratio_db": float(config.sidelobe_min_power_ratio_db),
    }
