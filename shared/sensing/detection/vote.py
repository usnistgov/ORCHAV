"""3-D vote-fusion helpers for azimuth-range-Doppler sensing cubes.

Vote fusion detects sparse targets by projecting a 3-D azimuth cube onto three
2-D views, running CFAR/NMS in each view, and voting candidate peaks back into
the original cube. The outputs remain NumPy contracts suitable for optional
``StandardMPCFrame["sensing"]`` payloads and post-processing diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.sensing.detection.cfar import (
    CFAR2DResult,
    CFAREdgePolicy,
    _cluster_unit_radius_components,
    _validated_axis,
    _validated_boolean_mask,
    _validated_cell_count,
    _validated_cell_pair,
    _validated_cfar_parameters,
    _validated_optional_map,
    _validated_power_array,
    _validated_real_scalar,
    cfar_2d,
    nms_peaks_2d,
)
from shared.sensing.signal_processing import (
    doppler_to_monostatic_equivalent_velocity_m_s,
    doppler_to_path_closing_rate_m_s,
)


@dataclass(frozen=True)
class ProjectionCfarConfig:
    """CFAR settings for one 2-D projection of a 3-D power cube."""

    guard_cells: tuple[int, int]
    reference_cells: tuple[int, int]
    threshold_scale: float
    method: str = "ca"
    os_rank: float = 0.75
    edge_policy: CFAREdgePolicy = "full_window"


@dataclass(frozen=True)
class VoteFusionConfig:
    """Configuration for three-projection vote fusion and clustering."""

    rd: ProjectionCfarConfig
    raz: ProjectionCfarConfig
    daz: ProjectionCfarConfig
    min_votes: int = 2
    pre_threshold_db: float = 20.0
    nms_suppression_radius: tuple[int, int, int] = (1, 2, 4)
    nms_max_peaks: int = 200
    post_nms_min_peak_divisor: float = 55.0
    cluster_range_gate_m: float = 1.0
    cluster_velocity_gate_m_s: float = 5.0
    cluster_azimuth_gate_deg: float | None = None


@dataclass
class VoteFusionResult:
    """Intermediate projection products and fused 3-D detection masks."""

    power_cube: np.ndarray
    rd_power: np.ndarray
    raz_power: np.ndarray
    daz_power: np.ndarray
    rd_cfar: CFAR2DResult
    raz_cfar: CFAR2DResult
    daz_cfar: CFAR2DResult
    projection_edge_policies: dict[str, CFAREdgePolicy]
    rd_peak_mask: np.ndarray
    raz_peak_mask: np.ndarray
    daz_peak_mask: np.ndarray
    votes: np.ndarray
    fused_mask: np.ndarray
    rd_fused_mask: np.ndarray
    aoa_az_map: np.ndarray


@dataclass
class ClusteredDetection3D:
    """One 3-D clustered detection in range/velocity/azimuth space."""

    range_m: float
    velocity_m_s: float
    doppler_hz: float
    path_closing_rate_m_s: float
    monostatic_equivalent_velocity_m_s: float
    power_linear: float
    snr_db: float
    range_bin: int
    doppler_bin: int
    azimuth_bin: int
    num_detections: int
    cluster_id: int
    vote_count: int
    centroid_range_m: float = 0.0
    centroid_velocity_m_s: float = 0.0
    centroid_doppler_hz: float = 0.0
    centroid_path_closing_rate_m_s: float = 0.0
    centroid_monostatic_equivalent_velocity_m_s: float = 0.0
    aoa_az_deg: float = float("nan")
    centroid_aoa_az_deg: float = float("nan")


def _normalize_projection_config(config: ProjectionCfarConfig) -> ProjectionCfarConfig:
    """Validate projection settings without truncation or method substitution."""
    guards = _validated_cell_pair(config.guard_cells, name="guard_cells", positive=False)
    refs = _validated_cell_pair(config.reference_cells, name="reference_cells", positive=True)
    scale, method, rank, policy = _validated_cfar_parameters(
        threshold_scale=config.threshold_scale,
        method=config.method,
        os_rank=config.os_rank,
        edge_policy=config.edge_policy,
    )
    return ProjectionCfarConfig(guards, refs, scale, method, rank, policy)


def _validated_radius(radius: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(radius, tuple) or len(radius) != 3:
        raise ValueError("suppression_radius must be a three-item tuple")
    values = tuple(
        _validated_cell_count(v, name="suppression_radius", positive=False) for v in radius
    )
    return values[0], values[1], values[2]


def _validate_fusion_config(config: VoteFusionConfig) -> None:
    votes = _validated_cell_count(config.min_votes, name="min_votes", positive=True)
    if votes > 3:
        raise ValueError("min_votes must be in [1, 3]")
    _validated_real_scalar(config.pre_threshold_db, name="pre_threshold_db", positive=False)
    _validated_radius(config.nms_suppression_radius)
    _validated_cell_count(config.nms_max_peaks, name="nms_max_peaks", positive=True)
    divisor = _validated_real_scalar(
        config.post_nms_min_peak_divisor, name="post_nms_min_peak_divisor", positive=True
    )
    if divisor < 1.0:
        raise ValueError("post_nms_min_peak_divisor must be at least one")
    _validated_real_scalar(config.cluster_range_gate_m, name="cluster_range_gate_m", positive=True)
    _validated_real_scalar(
        config.cluster_velocity_gate_m_s, name="cluster_velocity_gate_m_s", positive=True
    )
    if config.cluster_azimuth_gate_deg is not None:
        _validated_real_scalar(
            config.cluster_azimuth_gate_deg, name="cluster_azimuth_gate_deg", positive=True
        )


def derive_aoa_az_map_from_cube(
    azimuth_cube: np.ndarray,
    azimuth_axis_deg: np.ndarray | list[float],
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Derive a per-RD azimuth map from the strongest azimuth bin.

    Args:
        azimuth_cube: Power cube with shape ``(azimuth, doppler, range)``.
        azimuth_axis_deg: Azimuth axis in degrees with shape ``(azimuth,)``.
        mask: Optional boolean mask with the same shape as ``azimuth_cube``.
            When provided, only masked cells contribute.
    """
    cube = _validated_power_array(azimuth_cube, dimensions=3, name="azimuth_cube")
    axis = _validated_axis(
        azimuth_axis_deg, expected_length=cube.shape[0], name="azimuth_axis_deg", nonnegative=False
    )

    if mask is not None:
        valid_mask = _validated_boolean_mask(mask, dimensions=3, name="mask")
        if valid_mask.shape != cube.shape:
            raise ValueError("mask must match azimuth_cube shape")
        masked = np.where(valid_mask, cube, 0.0)
    else:
        masked = cube

    aoa_az_map = np.full(cube.shape[1:], np.nan, dtype=np.float32)
    if masked.size == 0:
        return aoa_az_map

    valid = np.any(masked > 0.0, axis=0)
    if not np.any(valid):
        return aoa_az_map

    best_bins = np.argmax(masked, axis=0)
    aoa_az_map[valid] = axis[best_bins[valid]].astype(np.float32, copy=False)
    return aoa_az_map


def vote_fusion_3d(
    azimuth_cube: np.ndarray,
    *,
    azimuth_axis_deg: np.ndarray | list[float],
    config: VoteFusionConfig,
) -> VoteFusionResult:
    """Run sparse 3-projection vote fusion on an azimuth cube.

    CFAR runs on uncensored range-Doppler / range-azimuth / Doppler-azimuth
    power projections, preserving their measured training noise. NMS peaks
    cast votes back into the original 3-D grid. The relative pre-threshold
    gates final candidates only; it never changes the noise/SNR denominator.
    """
    cube = _validated_power_array(azimuth_cube, dimensions=3, name="azimuth_cube")
    axis = _validated_axis(
        azimuth_axis_deg, expected_length=cube.shape[0], name="azimuth_axis_deg", nonnegative=False
    )
    _validate_fusion_config(config)

    rd_cfg = _normalize_projection_config(config.rd)
    raz_cfg = _normalize_projection_config(config.raz)
    daz_cfg = _normalize_projection_config(config.daz)

    power_cube = cube.copy()
    peak_value = float(np.max(power_cube)) if power_cube.size else 0.0
    candidate_mask = power_cube > 0.0
    if peak_value > 0.0 and config.pre_threshold_db > 0.0:
        pre_threshold_linear = peak_value * (10.0 ** (-float(config.pre_threshold_db) / 10.0))
        candidate_mask &= power_cube >= pre_threshold_linear

    rd_power = power_cube.sum(axis=0)
    raz_power = power_cube.sum(axis=1)
    daz_power = power_cube.sum(axis=2)

    rd_cfar = cfar_2d(
        rd_power,
        guard_cells=rd_cfg.guard_cells,
        reference_cells=rd_cfg.reference_cells,
        threshold_scale=rd_cfg.threshold_scale,
        method=rd_cfg.method,
        os_rank=rd_cfg.os_rank,
        edge_policy=rd_cfg.edge_policy,
    )
    raz_cfar = cfar_2d(
        raz_power,
        guard_cells=raz_cfg.guard_cells,
        reference_cells=raz_cfg.reference_cells,
        threshold_scale=raz_cfg.threshold_scale,
        method=raz_cfg.method,
        os_rank=raz_cfg.os_rank,
        edge_policy=raz_cfg.edge_policy,
    )
    daz_cfar = cfar_2d(
        daz_power,
        guard_cells=daz_cfg.guard_cells,
        reference_cells=daz_cfg.reference_cells,
        threshold_scale=daz_cfg.threshold_scale,
        method=daz_cfg.method,
        os_rank=daz_cfg.os_rank,
        edge_policy=daz_cfg.edge_policy,
    )

    rd_peak_mask = nms_peaks_2d(
        rd_cfar.detections,
        rd_power,
        suppression_radius=rd_cfg.guard_cells,
    )
    raz_peak_mask = nms_peaks_2d(
        raz_cfar.detections,
        raz_power,
        suppression_radius=raz_cfg.guard_cells,
    )
    daz_peak_mask = nms_peaks_2d(
        daz_cfar.detections,
        daz_power,
        suppression_radius=daz_cfg.guard_cells,
    )

    votes = np.zeros_like(power_cube, dtype=np.int8)
    for doppler_idx, range_idx in np.argwhere(rd_peak_mask):
        votes[:, doppler_idx, range_idx] += 1
    for azimuth_idx, range_idx in np.argwhere(raz_peak_mask):
        votes[azimuth_idx, :, range_idx] += 1
    for azimuth_idx, doppler_idx in np.argwhere(daz_peak_mask):
        votes[azimuth_idx, doppler_idx, :] += 1

    fused_mask = (votes >= int(config.min_votes)) & candidate_mask
    rd_fused_mask = np.any(fused_mask, axis=0)
    aoa_az_map = derive_aoa_az_map_from_cube(power_cube, axis, mask=fused_mask)

    return VoteFusionResult(
        power_cube=power_cube,
        rd_power=rd_power,
        raz_power=raz_power,
        daz_power=daz_power,
        rd_cfar=rd_cfar,
        raz_cfar=raz_cfar,
        daz_cfar=daz_cfar,
        projection_edge_policies={
            "rd": rd_cfg.edge_policy,
            "raz": raz_cfg.edge_policy,
            "daz": daz_cfg.edge_policy,
        },
        rd_peak_mask=rd_peak_mask,
        raz_peak_mask=raz_peak_mask,
        daz_peak_mask=daz_peak_mask,
        votes=votes,
        fused_mask=fused_mask,
        rd_fused_mask=rd_fused_mask,
        aoa_az_map=aoa_az_map,
    )


def nms_peaks_3d(
    detections_mask: np.ndarray,
    power_cube: np.ndarray,
    *,
    suppression_radius: tuple[int, int, int] = (1, 2, 4),
    max_peaks: int | None = None,
) -> np.ndarray:
    """Greedy non-wrapping 3D NMS with row-major ties and a positive quota.

    Radius zero leaves candidates unsuppressed along that dimension. Only
    strictly positive power cells are candidates; empty masks still validate.
    """
    mask = _validated_boolean_mask(detections_mask, dimensions=3, name="detections_mask")
    cube = _validated_power_array(power_cube, dimensions=3, name="power_cube")
    ra, rd, rr = _validated_radius(suppression_radius)
    if max_peaks is not None:
        max_peaks = _validated_cell_count(max_peaks, name="max_peaks", positive=True)
    if mask.shape != cube.shape:
        raise ValueError(
            f"detections_mask and power_cube must match, got {mask.shape} vs {cube.shape}"
        )
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    result = np.zeros_like(mask, dtype=bool)
    suppressed = np.zeros_like(mask, dtype=bool)

    candidate_coords = np.argwhere(mask & (cube > 0.0))
    if candidate_coords.size == 0:
        return result

    candidate_powers = cube[
        candidate_coords[:, 0],
        candidate_coords[:, 1],
        candidate_coords[:, 2],
    ]
    order = np.argsort(-candidate_powers, kind="stable")
    peaks_kept = 0
    for az_idx, doppler_idx, range_idx in candidate_coords[order]:
        if suppressed[az_idx, doppler_idx, range_idx]:
            continue
        result[az_idx, doppler_idx, range_idx] = True
        peaks_kept += 1
        if max_peaks is not None and peaks_kept >= int(max_peaks):
            break
        az_lo = max(0, az_idx - ra)
        az_hi = min(mask.shape[0], az_idx + ra + 1)
        dop_lo = max(0, doppler_idx - rd)
        dop_hi = min(mask.shape[1], doppler_idx + rd + 1)
        rng_lo = max(0, range_idx - rr)
        rng_hi = min(mask.shape[2], range_idx + rr + 1)
        suppressed[az_lo:az_hi, dop_lo:dop_hi, rng_lo:rng_hi] = True

    return result


def _resolve_cluster_azimuth_gate_deg(
    azimuth_axis_deg: np.ndarray,
    configured_gate_deg: float | None,
) -> float:
    if configured_gate_deg is not None:
        return _validated_real_scalar(
            configured_gate_deg, name="cluster_azimuth_gate_deg", positive=True
        )
    if azimuth_axis_deg.size <= 1:
        return 1.0
    spacing = np.abs((np.diff(azimuth_axis_deg) + 180.0) % 360.0 - 180.0)
    spacing = spacing[spacing > 0.0]
    if spacing.size == 0:
        return 1.0
    return max(float(np.median(spacing)) * 2.0, 1.0)


def collapse_cluster_labels_to_rd(
    cluster_labels_3d: np.ndarray,
    power_cube: np.ndarray,
) -> np.ndarray:
    """Collapse 3D cluster labels to a 2D RD preview for compatibility."""
    labels = np.asarray(cluster_labels_3d, dtype=np.int32)
    cube = np.asarray(power_cube, dtype=np.float64)
    if labels.shape != cube.shape:
        raise ValueError(
            f"cluster_labels_3d and power_cube must match, got {labels.shape} vs {cube.shape}"
        )

    rd_labels = np.full(labels.shape[1:], -1, dtype=np.int32)
    for doppler_idx in range(labels.shape[1]):
        for range_idx in range(labels.shape[2]):
            label_column = labels[:, doppler_idx, range_idx]
            valid = label_column >= 0
            if not np.any(valid):
                continue
            power_column = np.where(valid, cube[:, doppler_idx, range_idx], -np.inf)
            best_az = int(np.argmax(power_column))
            rd_labels[doppler_idx, range_idx] = int(label_column[best_az])
    return rd_labels


def cluster_detections_3d(
    detections_mask: np.ndarray,
    power_cube: np.ndarray,
    range_axis_m: np.ndarray,
    doppler_axis_hz: np.ndarray,
    azimuth_axis_deg: np.ndarray | list[float],
    *,
    wavelength: float,
    snr_map: np.ndarray | None = None,
    snr_valid_mask: np.ndarray | None = None,
    cluster_range_gate_m: float = 1.0,
    cluster_velocity_gate_m_s: float = 5.0,
    cluster_azimuth_gate_deg: float | None = None,
    return_cluster_labels: bool = False,
    range_mode: str = "monostatic",
    tx_pos: np.ndarray | None = None,
    rx_pos: np.ndarray | None = None,
    votes: np.ndarray | None = None,
) -> list[ClusteredDetection3D] | tuple[list[ClusteredDetection3D], np.ndarray]:
    """Cluster positive detections using physical, circular-azimuth distance.

    Gates are ellipsoidal neighbour distances, not maximum cluster diameters;
    connected chains may span more than a gate. Doppler does not wrap.
    SNR values are RD-projection SNR, not per-angle voxel SNR.
    """
    mask = _validated_boolean_mask(detections_mask, dimensions=3, name="detections_mask")
    cube = _validated_power_array(power_cube, dimensions=3, name="power_cube")
    if mask.shape != cube.shape:
        raise ValueError("detections_mask and power_cube must match")
    range_axis = _validated_axis(
        range_axis_m, expected_length=cube.shape[2], name="range_axis_m", nonnegative=True
    )
    doppler_axis = _validated_axis(
        doppler_axis_hz, expected_length=cube.shape[1], name="doppler_axis_hz", nonnegative=False
    )
    azimuth_axis = _validated_axis(
        azimuth_axis_deg, expected_length=cube.shape[0], name="azimuth_axis_deg", nonnegative=False
    )
    wavelength = _validated_real_scalar(wavelength, name="wavelength", positive=True)
    range_gate = _validated_real_scalar(
        cluster_range_gate_m, name="cluster_range_gate_m", positive=True
    )
    velocity_gate = _validated_real_scalar(
        cluster_velocity_gate_m_s, name="cluster_velocity_gate_m_s", positive=True
    )
    azimuth_gate = _resolve_cluster_azimuth_gate_deg(azimuth_axis, cluster_azimuth_gate_deg)
    if not isinstance(range_mode, str) or range_mode not in ("monostatic", "bistatic"):
        raise ValueError("range_mode must be exactly 'monostatic' or 'bistatic'")
    snr_map = _validated_optional_map(
        snr_map, expected_shape=cube.shape[1:], name="snr_map", allow_nan=False
    )
    if snr_valid_mask is not None:
        valid_snr = _validated_boolean_mask(snr_valid_mask, dimensions=2, name="snr_valid_mask")
        if valid_snr.shape != cube.shape[1:]:
            raise ValueError("snr_valid_mask must match the RD shape")
        if snr_map is None:
            raise ValueError("snr_valid_mask requires snr_map")
        mask = mask & valid_snr[np.newaxis, :, :]
    if np.any(mask & (cube <= 0.0)):
        raise ValueError("detections_mask may select only strictly positive power cells")
    if votes is not None:
        votes = np.asarray(votes)
        if votes.shape != cube.shape or votes.dtype.kind not in "iu":
            raise ValueError("votes must be an integer array matching power_cube")
        if np.any((votes < 0) | (votes > 3)):
            raise ValueError("votes must lie in [0, 3]")

    if not np.any(mask):
        empty_labels = np.full(mask.shape, -1, dtype=np.int32)
        return ([], empty_labels) if return_cluster_labels else []

    det_coords = np.argwhere(mask)
    if det_coords.size == 0:
        empty_labels = np.full(mask.shape, -1, dtype=np.int32)
        return ([], empty_labels) if return_cluster_labels else []

    velocity_axis = np.asarray(doppler_axis_hz, dtype=np.float64) * float(wavelength) / 2.0
    det_azimuth_deg = azimuth_axis[det_coords[:, 0]]
    det_velocity_m_s = velocity_axis[det_coords[:, 1]]
    det_range_m = range_axis[det_coords[:, 2]]

    phys_coords = np.column_stack(
        [
            det_range_m / range_gate,
            det_velocity_m_s / velocity_gate,
            det_azimuth_deg / azimuth_gate,
        ]
    )
    # On a scan spanning at most 180 degrees the usual Euclidean graph is
    # already exact. Otherwise add seam neighbours using shortest arc length.
    if np.ptp(det_azimuth_deg) <= 180.0:
        labels = _cluster_unit_radius_components(phys_coords)
    else:
        labels = np.full(len(det_coords), -1, dtype=np.int32)
        next_label = 0
        for seed in range(len(det_coords)):
            if labels[seed] >= 0:
                continue
            labels[seed] = next_label
            pending = [seed]
            while pending:
                current = pending.pop()
                delta = phys_coords[:, :2] - phys_coords[current, :2]
                angular = (
                    (det_azimuth_deg - det_azimuth_deg[current] + 180.0) % 360.0 - 180.0
                ) / azimuth_gate
                neighbours = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) + angular**2 <= 1.0)
                for neighbour in neighbours:
                    if labels[neighbour] < 0:
                        labels[neighbour] = next_label
                        pending.append(int(neighbour))
            next_label += 1

    cluster_labels = np.full(mask.shape, -1, dtype=np.int32)
    cluster_labels[
        det_coords[:, 0],
        det_coords[:, 1],
        det_coords[:, 2],
    ] = labels.astype(np.int32, copy=False)

    unique_labels = np.unique(labels[labels >= 0])
    clustered: list[ClusteredDetection3D] = []
    votes_arr = np.asarray(votes, dtype=np.int16) if votes is not None else None
    snr_arr = np.asarray(snr_map, dtype=np.float64) if snr_map is not None else None

    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_indices = det_coords[cluster_mask]

        powers = cube[tuple(cluster_indices.T)]
        peak_index = cluster_indices[int(np.argmax(powers))]
        peak_az_idx, peak_dopp_idx, peak_range_idx = peak_index
        max_power = float(np.max(powers))
        # Normalize first so finite powers cannot overflow the centroid sums.
        weights = powers / max_power if max_power > 0.0 else np.ones(len(powers))
        total_weight = float(np.sum(weights))
        weights = weights / total_weight
        azimuths = np.radians(azimuth_axis[cluster_indices[:, 0]])
        centroid_az_deg = float(
            np.degrees(
                np.arctan2(np.dot(weights, np.sin(azimuths)), np.dot(weights, np.cos(azimuths)))
            )
        )
        # Integer-bin and measurement fields refer to an actual detected peak;
        # the continuous cluster centroid is published separately.
        range_m = float(range_axis[peak_range_idx])
        doppler_hz = float(doppler_axis[peak_dopp_idx])
        centroid_range_m = float(np.dot(weights, range_axis[cluster_indices[:, 2]]))
        centroid_doppler_hz = float(np.dot(weights, doppler_axis[cluster_indices[:, 1]]))

        # A detection owns an observed Doppler, not a target identity. Keep
        # the legacy velocity projection target-independent; geometry-aware
        # bistatic speed belongs exclusively to truth evaluation.
        path_closing_rate_m_s = doppler_to_path_closing_rate_m_s(doppler_hz, wavelength)
        velocity_m_s = doppler_to_monostatic_equivalent_velocity_m_s(doppler_hz, wavelength)
        centroid_path_closing_rate_m_s = doppler_to_path_closing_rate_m_s(
            centroid_doppler_hz, wavelength
        )
        centroid_velocity_m_s = doppler_to_monostatic_equivalent_velocity_m_s(
            centroid_doppler_hz, wavelength
        )
        snr_db = float(snr_arr[peak_dopp_idx, peak_range_idx]) if snr_arr is not None else 0.0
        vote_count = (
            int(
                np.max(
                    votes_arr[cluster_indices[:, 0], cluster_indices[:, 1], cluster_indices[:, 2]]
                )
            )
            if votes_arr is not None
            else 0
        )

        clustered.append(
            ClusteredDetection3D(
                range_m=range_m,
                velocity_m_s=velocity_m_s,
                doppler_hz=doppler_hz,
                path_closing_rate_m_s=path_closing_rate_m_s,
                monostatic_equivalent_velocity_m_s=velocity_m_s,
                power_linear=float(max_power),
                snr_db=snr_db,
                range_bin=int(peak_range_idx),
                doppler_bin=int(peak_dopp_idx),
                azimuth_bin=int(peak_az_idx),
                num_detections=int(np.sum(cluster_mask)),
                cluster_id=int(cluster_id),
                vote_count=vote_count,
                centroid_range_m=centroid_range_m,
                centroid_velocity_m_s=centroid_velocity_m_s,
                centroid_doppler_hz=centroid_doppler_hz,
                centroid_path_closing_rate_m_s=centroid_path_closing_rate_m_s,
                centroid_monostatic_equivalent_velocity_m_s=centroid_velocity_m_s,
                aoa_az_deg=float(azimuth_axis[peak_az_idx]),
                centroid_aoa_az_deg=centroid_az_deg,
            )
        )

    clustered.sort(key=lambda det: det.power_linear, reverse=True)

    if return_cluster_labels:
        return clustered, cluster_labels
    return clustered
