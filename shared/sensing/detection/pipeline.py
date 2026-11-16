"""Shared detection pipeline for range-Doppler style frame extensions.

The functions here operate on already-computed sensing payloads, not on raw
Sionna RT paths. They provide the reusable CFAR/NMS/clustering/filtering stages
used by post-processing and diagnostics before results are written back as
optional frame-extension data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from shared.sensing.detection.angle import (
    AnglePeakDetectionConfig,
    augment_detections_with_sensor_frame_xy,
    extract_rd_anchored_angle_candidates,
    suppress_angle_sidelobes,
)
from shared.sensing.detection.cfar import (
    CFAR2DResult,
    CFAREdgePolicy,
    cfar_2d,
    cluster_detections_2d,
    nms_peaks_2d,
)
from shared.sensing.detection.vote import (
    VoteFusionConfig,
    cluster_detections_3d,
    collapse_cluster_labels_to_rd,
    nms_peaks_3d,
    vote_fusion_3d,
)
from shared.sensing.signal_processing import RangeDopplerResult

DetectionFilter = Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, Any]]]

DEFAULT_CLUSTER_RANGE_GATE_M = 1.0
DEFAULT_CLUSTER_VELOCITY_GATE_M_S = 5.0
DEFAULT_MIN_SNR_DB = 6.0
DB_POWER_SCALE = 10.0
POWER_FLOOR_LINEAR = 1e-30
SNR_CENSOR_FLOOR_DB = -300.0
SNR_CENSOR_CEILING_DB = 300.0


def _validated_real_scalar(
    value: float,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Return one exact finite real scalar satisfying the requested bound."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real value")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real value")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _validated_count(value: int, *, name: str, positive: bool) -> int:
    """Return one exact integral bin count."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be an exact {qualifier} integer")
    count = int(value)
    if (positive and count <= 0) or (not positive and count < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return count


def _validated_count_tuple(
    value: tuple[int, ...],
    *,
    name: str,
    length: int,
    positive: bool,
) -> tuple[int, ...]:
    """Validate one fixed-length tuple of exact integral bin counts."""

    if not isinstance(value, tuple) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-item tuple")
    return tuple(
        _validated_count(item, name=f"{name}[{index}]", positive=positive)
        for index, item in enumerate(value)
    )


def _validated_power_array(values: np.ndarray, *, dimensions: int, name: str) -> np.ndarray:
    """Return finite, nonnegative real linear power without complex coercion."""

    raw = np.asarray(values)
    if raw.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-D, got shape {raw.shape}")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a real numeric linear-power array")
    power = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(power)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(power < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return power


def _validated_boolean_array(
    values: np.ndarray, *, shape: tuple[int, ...], name: str
) -> np.ndarray:
    """Return an exact boolean array with no numeric truth-value coercion."""

    mask = np.asarray(values)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError(f"{name} must be a boolean array")
    return mask


def _validated_axis(
    values: np.ndarray | list[float],
    *,
    length: int,
    name: str,
    nonnegative: bool,
) -> np.ndarray:
    """Return one finite, strictly increasing physical coordinate axis."""

    raw = np.asarray(values)
    if raw.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {raw.shape}")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a real numeric array")
    axis = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(axis < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _validated_rd_inputs(
    rd_result: RangeDopplerResult,
    detection_power_map: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate one aligned range-Doppler power map and its physical axes."""

    source = rd_result.power_linear if detection_power_map is None else detection_power_map
    power = _validated_power_array(source, dimensions=2, name="detection power map")
    range_axis = _validated_axis(
        rd_result.range_axis_m,
        length=power.shape[1],
        name="range_axis_m",
        nonnegative=True,
    )
    doppler_axis = _validated_axis(
        rd_result.doppler_axis_hz,
        length=power.shape[0],
        name="doppler_axis_hz",
        nonnegative=False,
    )
    return power, range_axis, doppler_axis


def _validated_azimuth_inputs(
    azimuth_cube: np.ndarray,
    azimuth_axis_deg: np.ndarray | list[float],
    *,
    rd_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one finite nonnegative azimuth-Doppler-range power cube."""

    cube = _validated_power_array(azimuth_cube, dimensions=3, name="azimuth_cube")
    if cube.shape[1:] != rd_shape:
        raise ValueError(
            "azimuth_cube RD shape must match the detection power map: "
            f"expected {rd_shape}, got {cube.shape[1:]}"
        )
    axis = _validated_axis(
        azimuth_axis_deg,
        length=cube.shape[0],
        name="azimuth_axis_deg",
        nonnegative=False,
    )
    return cube, axis


@dataclass(frozen=True)
class DetectionPipelineConfig:
    """Tunable gates for CFAR, NMS, clustering, and post-detection filtering."""

    cfar_guard_cells: tuple[int, int]
    cfar_reference_cells: tuple[int, int]
    cfar_threshold_scale: float
    cfar_method: str = "ca"
    cfar_os_rank: float = 0.75
    cfar_edge_policy: CFAREdgePolicy = "full_window"
    nms_suppression_radius: tuple[int, int] = (2, 4)
    cluster_range_gate_m: float = DEFAULT_CLUSTER_RANGE_GATE_M
    cluster_velocity_gate_m_s: float = DEFAULT_CLUSTER_VELOCITY_GATE_M_S
    min_snr_db: float = DEFAULT_MIN_SNR_DB
    # Optional power span below the strongest cell in this detector input.
    # This is a sensitivity bound, not calibrated receiver noise or an SNR.
    cfar_max_dynamic_range_db: float | None = None
    min_detection_range_m: float | None = None
    max_detection_range_m: float | None = None
    ghost_filter_velocity_gate_m_s: float | None = None
    ghost_filter_azimuth_gate_deg: float | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or nonphysical detector configuration values."""

        _validated_count_tuple(
            self.cfar_guard_cells,
            name="cfar_guard_cells",
            length=2,
            positive=False,
        )
        _validated_count_tuple(
            self.cfar_reference_cells,
            name="cfar_reference_cells",
            length=2,
            positive=True,
        )
        _validated_real_scalar(
            self.cfar_threshold_scale,
            name="cfar_threshold_scale",
            nonnegative=True,
        )
        if not isinstance(self.cfar_method, str) or self.cfar_method not in ("ca", "os"):
            raise ValueError("cfar_method must be exactly 'ca' or 'os'")
        rank = _validated_real_scalar(self.cfar_os_rank, name="cfar_os_rank")
        if not 0.0 <= rank <= 1.0:
            raise ValueError("cfar_os_rank must lie between zero and one")
        if not isinstance(self.cfar_edge_policy, str) or self.cfar_edge_policy not in (
            "full_window",
            "truncate",
        ):
            raise ValueError("cfar_edge_policy must be exactly 'full_window' or 'truncate'")
        _validated_count_tuple(
            self.nms_suppression_radius,
            name="nms_suppression_radius",
            length=2,
            positive=False,
        )
        _validated_real_scalar(
            self.cluster_range_gate_m,
            name="cluster_range_gate_m",
            positive=True,
        )
        _validated_real_scalar(
            self.cluster_velocity_gate_m_s,
            name="cluster_velocity_gate_m_s",
            positive=True,
        )
        _validated_real_scalar(self.min_snr_db, name="min_snr_db")
        if self.cfar_max_dynamic_range_db is not None:
            _validated_real_scalar(
                self.cfar_max_dynamic_range_db, name="cfar_max_dynamic_range_db", nonnegative=True
            )

        min_range = (
            None
            if self.min_detection_range_m is None
            else _validated_real_scalar(
                self.min_detection_range_m,
                name="min_detection_range_m",
                nonnegative=True,
            )
        )
        max_range = (
            None
            if self.max_detection_range_m is None
            else _validated_real_scalar(
                self.max_detection_range_m,
                name="max_detection_range_m",
                nonnegative=True,
            )
        )
        if min_range is not None and max_range is not None and min_range > max_range:
            raise ValueError("min_detection_range_m must not exceed max_detection_range_m")

        if self.ghost_filter_velocity_gate_m_s is not None:
            _validated_real_scalar(
                self.ghost_filter_velocity_gate_m_s,
                name="ghost_filter_velocity_gate_m_s",
                positive=True,
            )
        if self.ghost_filter_azimuth_gate_deg is not None:
            _validated_real_scalar(
                self.ghost_filter_azimuth_gate_deg,
                name="ghost_filter_azimuth_gate_deg",
                positive=True,
            )
            if self.ghost_filter_velocity_gate_m_s is None:
                raise ValueError(
                    "ghost_filter_azimuth_gate_deg requires ghost_filter_velocity_gate_m_s"
                )


def _validate_wavelength_and_range_mode(wavelength_m: float, range_mode: str) -> float:
    """Validate the target-independent Doppler projection contract."""

    wavelength = _validated_real_scalar(
        wavelength_m,
        name="wavelength_m",
        positive=True,
    )
    if not isinstance(range_mode, str) or range_mode not in ("monostatic", "bistatic"):
        raise ValueError("range_mode must be exactly 'monostatic' or 'bistatic'")
    return wavelength


def _validate_projection_cfar_config(config: Any, *, name: str) -> None:
    """Validate one vote-fusion projection without lossy normalization."""

    _validated_count_tuple(
        config.guard_cells,
        name=f"{name}.guard_cells",
        length=2,
        positive=False,
    )
    _validated_count_tuple(
        config.reference_cells,
        name=f"{name}.reference_cells",
        length=2,
        positive=True,
    )
    _validated_real_scalar(
        config.threshold_scale,
        name=f"{name}.threshold_scale",
        nonnegative=True,
    )
    if not isinstance(config.method, str) or config.method not in ("ca", "os"):
        raise ValueError(f"{name}.method must be exactly 'ca' or 'os'")
    rank = _validated_real_scalar(config.os_rank, name=f"{name}.os_rank")
    if not 0.0 <= rank <= 1.0:
        raise ValueError(f"{name}.os_rank must lie between zero and one")
    if not isinstance(config.edge_policy, str) or config.edge_policy not in (
        "full_window",
        "truncate",
    ):
        raise ValueError(f"{name}.edge_policy must be exactly 'full_window' or 'truncate'")


def _validate_vote_config(config: VoteFusionConfig) -> None:
    """Validate the public vote-fusion configuration without lossy coercion."""

    _validate_projection_cfar_config(config.rd, name="vote_config.rd")
    _validate_projection_cfar_config(config.raz, name="vote_config.raz")
    _validate_projection_cfar_config(config.daz, name="vote_config.daz")
    min_votes = _validated_count(config.min_votes, name="vote_config.min_votes", positive=True)
    if min_votes > 3:
        raise ValueError("vote_config.min_votes must lie in [1, 3]")
    _validated_real_scalar(
        config.pre_threshold_db,
        name="vote_config.pre_threshold_db",
        nonnegative=True,
    )
    _validated_count_tuple(
        config.nms_suppression_radius,
        name="vote_config.nms_suppression_radius",
        length=3,
        positive=False,
    )
    _validated_count(config.nms_max_peaks, name="vote_config.nms_max_peaks", positive=True)
    divisor = _validated_real_scalar(
        config.post_nms_min_peak_divisor,
        name="vote_config.post_nms_min_peak_divisor",
        positive=True,
    )
    if divisor < 1.0:
        raise ValueError("vote_config.post_nms_min_peak_divisor must be at least one")
    _validated_real_scalar(
        config.cluster_range_gate_m,
        name="vote_config.cluster_range_gate_m",
        positive=True,
    )
    _validated_real_scalar(
        config.cluster_velocity_gate_m_s,
        name="vote_config.cluster_velocity_gate_m_s",
        positive=True,
    )
    if config.cluster_azimuth_gate_deg is not None:
        _validated_real_scalar(
            config.cluster_azimuth_gate_deg,
            name="vote_config.cluster_azimuth_gate_deg",
            positive=True,
        )


def _validate_angle_config(config: AnglePeakDetectionConfig) -> None:
    """Reject angle-detector parameters that otherwise get silently clamped."""

    _validated_count(
        config.max_peaks_per_rd_cell,
        name="angle_config.max_peaks_per_rd_cell",
        positive=True,
    )
    for name in (
        "peak_nms_radius_bins",
        "edge_exclusion_bins",
        "secondary_peak_min_separation_bins",
        "sidelobe_range_tolerance_bins",
        "sidelobe_doppler_tolerance_bins",
        "sidelobe_azimuth_tolerance_bins",
    ):
        _validated_count(
            getattr(config, name),
            name=f"angle_config.{name}",
            positive=False,
        )
    for name in (
        "peak_relative_floor_db",
        "secondary_peak_max_gap_db",
        "secondary_peak_min_prominence_db",
        "secondary_peak_min_profile_snr_db",
        "secondary_peak_min_valley_depth_db",
        "sidelobe_min_power_ratio_db",
    ):
        _validated_real_scalar(
            getattr(config, name),
            name=f"angle_config.{name}",
            nonnegative=True,
        )
    if config.cluster_azimuth_gate_deg is not None:
        _validated_real_scalar(
            config.cluster_azimuth_gate_deg,
            name="angle_config.cluster_azimuth_gate_deg",
            positive=True,
        )
    if not isinstance(config.sidelobe_suppression_enabled, (bool, np.bool_)):
        raise ValueError("angle_config.sidelobe_suppression_enabled must be boolean")


@dataclass
class DetectionStageArtifacts:
    """Intermediate masks and maps produced while deriving detections."""

    cfar_input_power: np.ndarray
    cfar_input_power_db: np.ndarray
    cfar_mask: np.ndarray
    cfar_noise_floor: np.ndarray
    cfar_threshold: np.ndarray
    cfar_training_cell_count: np.ndarray
    cfar_eligible_mask: np.ndarray
    nms_mask: np.ndarray
    snr_map: np.ndarray
    snr_valid_mask: np.ndarray
    cluster_labels: np.ndarray
    extra_artifacts: dict[str, np.ndarray] | None = None


@dataclass
class DetectionPipelineResult:
    """Detections, metadata, and optional stage artifacts from one pipeline run."""

    detections: list[dict[str, Any]]
    metadata: dict[str, Any]
    stages: DetectionStageArtifacts | None = None


def _compute_snr_map(
    power_map: np.ndarray,
    noise_map: np.ndarray,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite SNR values and an explicit denominator-validity mask.

    SNR is defined only where CFAR has an eligible training window and its
    estimated noise power is strictly positive.  Zero-noise cells are not
    assigned an arbitrary large dB value.  Valid ratios are clipped to a
    declared finite storage interval; invalid cells carry the neutral numeric
    sentinel ``0`` and must be interpreted through ``snr_valid_mask``.
    """

    power = _validated_power_array(power_map, dimensions=2, name="SNR input power")
    noise = _validated_power_array(noise_map, dimensions=2, name="SNR noise estimate")
    if power.shape != noise.shape:
        raise ValueError("SNR power and noise maps must have identical shapes")
    if eligible_mask is None:
        eligible = np.ones(power.shape, dtype=bool)
    else:
        eligible = _validated_boolean_array(
            eligible_mask,
            shape=power.shape,
            name="SNR eligibility mask",
        )

    valid = eligible & (noise > 0.0)
    snr_map = np.zeros(power.shape, dtype=np.float64)
    positive_signal = valid & (power > 0.0)
    snr_map[valid & ~positive_signal] = SNR_CENSOR_FLOOR_DB
    # Subtract logarithms rather than forming the ratio first, which avoids an
    # intermediate overflow for valid extreme dynamic ranges.
    snr_map[positive_signal] = np.clip(
        DB_POWER_SCALE * (np.log10(power[positive_signal]) - np.log10(noise[positive_signal])),
        SNR_CENSOR_FLOOR_DB,
        SNR_CENSOR_CEILING_DB,
    )
    return snr_map, valid


def _cfar_stage_metadata(
    result: CFAR2DResult,
    *,
    method: str,
    threshold_scale: float,
    os_rank: float,
    edge_policy: CFAREdgePolicy,
    snr_valid_mask: np.ndarray,
) -> dict[str, Any]:
    """Describe exact CFAR/SNR semantics and observed eligibility counts."""

    counts = result.training_cell_count[result.eligible]
    return {
        "cfar": {
            "input_domain": "power_linear",
            "input_unit": "1",
            "method": method,
            "threshold_semantics": "raw_linear_power_multiplier",
            "threshold_scale": float(threshold_scale),
            "threshold_unit": "same_as_input_power",
            "os_rank_fraction": float(os_rank),
            "os_rank_rounding": "nearest_integer_ties_to_even_of_fraction_times_k_minus_one",
            "edge_policy": edge_policy,
            "cut_comparison": "strict_greater_than_threshold",
            "ineligible_numeric_sentinel": 0.0,
            "eligible_cell_count": int(np.count_nonzero(result.eligible)),
            "training_cell_count_min": int(np.min(counts)) if counts.size else None,
            "training_cell_count_max": int(np.max(counts)) if counts.size else None,
            "power_db_artifact": {
                "semantics": "10_log10_max_power_over_unit_reference",
                "linear_floor": POWER_FLOOR_LINEAR,
                "unit_reference": 1.0,
            },
        },
        "snr": {
            "semantics": "cell_power_over_cfar_noise_estimate",
            "validity": "eligible_training_window_and_strictly_positive_noise_estimate",
            "invalid_cell_policy": "reject_from_detection_and_store_zero_with_validity_mask",
            "finite_censor_floor_db": SNR_CENSOR_FLOOR_DB,
            "finite_censor_ceiling_db": SNR_CENSOR_CEILING_DB,
            "valid_cell_count": int(np.count_nonzero(snr_valid_mask)),
        },
    }


def _detection_stage_artifacts(
    power_map: np.ndarray,
    cfar_result: CFAR2DResult,
    nms_mask: np.ndarray,
    snr_map: np.ndarray,
    snr_valid_mask: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    extra_artifacts: dict[str, np.ndarray] | None = None,
) -> DetectionStageArtifacts:
    """Build one complete, copy-safe set of detector-stage artifacts."""

    power = np.asarray(power_map, dtype=np.float64)
    return DetectionStageArtifacts(
        cfar_input_power=np.array(power, copy=True),
        cfar_input_power_db=DB_POWER_SCALE * np.log10(np.maximum(power, POWER_FLOOR_LINEAR)),
        cfar_mask=np.array(cfar_result.detections, copy=True),
        cfar_noise_floor=np.array(cfar_result.noise, copy=True),
        cfar_threshold=np.array(cfar_result.threshold, copy=True),
        cfar_training_cell_count=np.array(cfar_result.training_cell_count, copy=True),
        cfar_eligible_mask=np.array(cfar_result.eligible, copy=True),
        nms_mask=np.array(nms_mask, copy=True),
        snr_map=np.array(snr_map, copy=True),
        snr_valid_mask=np.array(snr_valid_mask, copy=True),
        cluster_labels=np.array(cluster_labels, copy=True),
        extra_artifacts=extra_artifacts,
    )


def _apply_post_detection_filters(
    detections: list[dict[str, Any]],
    *,
    config: DetectionPipelineConfig,
    post_detection_filter: DetectionFilter | None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    detections_after_snr = len(detections)
    detections, range_gate_meta = apply_detection_range_gate(
        detections,
        min_detection_range_m=config.min_detection_range_m,
        max_detection_range_m=config.max_detection_range_m,
    )

    detections, ghost_meta = apply_ghost_filter(
        detections,
        velocity_gate_m_s=config.ghost_filter_velocity_gate_m_s,
        azimuth_gate_deg=config.ghost_filter_azimuth_gate_deg,
    )

    if post_detection_filter is not None:
        detections, tracker_meta = post_detection_filter(detections)
    else:
        tracker_meta = {
            "enabled": False,
            "raw": len(detections),
            "kept": len(detections),
            "active_tracks": 0,
            "confirmed_tracks": 0,
        }

    return detections, detections_after_snr, range_gate_meta, ghost_meta, tracker_meta


def apply_detection_range_gate(
    detections: list[dict[str, Any]],
    *,
    min_detection_range_m: float | None = None,
    max_detection_range_m: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply optional physical min/max range gating on detections."""
    min_range = (
        None
        if min_detection_range_m is None
        else _validated_real_scalar(
            min_detection_range_m,
            name="min_detection_range_m",
            nonnegative=True,
        )
    )
    max_range = (
        None
        if max_detection_range_m is None
        else _validated_real_scalar(
            max_detection_range_m,
            name="max_detection_range_m",
            nonnegative=True,
        )
    )
    if min_range is not None and max_range is not None and min_range > max_range:
        raise ValueError("min_detection_range_m must not exceed max_detection_range_m")
    if min_detection_range_m is None and max_detection_range_m is None:
        return detections, {"enabled": False, "kept": len(detections), "rejected": 0}

    kept: list[dict[str, Any]] = []
    rejected = 0
    for det in detections:
        if "range_m" not in det:
            raise ValueError("range-gated detection is missing range_m")
        rng = _validated_real_scalar(det["range_m"], name="detection range_m", nonnegative=True)
        if min_range is not None and rng < min_range:
            rejected += 1
            continue
        if max_range is not None and rng > max_range:
            rejected += 1
            continue
        kept.append(det)

    return kept, {
        "enabled": True,
        "min_range_m": min_range,
        "max_range_m": max_range,
        "kept": len(kept),
        "rejected": rejected,
    }


def apply_ghost_filter(
    detections: list[dict[str, Any]],
    *,
    velocity_gate_m_s: float | None = None,
    azimuth_gate_deg: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Heuristically keep the shortest-range detection per velocity group.

    Detections are grouped by velocity proximity. If a shorter-range detection
    with similar velocity already exists, the longer-range detection is
    suppressed as a possible multipath duplicate. This rule does not establish
    path provenance: the direct return may be absent, reflected-path Doppler
    can differ from direct-path Doppler, and a farther physical target can be
    suppressed.

    Args:
        detections: Input detections with ``range_m`` and ``velocity_m_s``.
        velocity_gate_m_s: Velocity grouping threshold in m/s. ``None``
            disables the filter.
        azimuth_gate_deg: Optional azimuth grouping threshold in degrees.
            When provided, a farther detection is only suppressed if it is
            close in both velocity and azimuth. If either detection lacks a
            finite azimuth estimate, the function falls back to velocity-only
            grouping for that pair.

    Returns:
        Filtered detections and metadata dict.
    """
    if velocity_gate_m_s is None:
        if azimuth_gate_deg is not None:
            raise ValueError("azimuth_gate_deg requires velocity_gate_m_s")
        return detections, {"enabled": False, "kept": len(detections), "rejected": 0}
    velocity_gate = _validated_real_scalar(
        velocity_gate_m_s,
        name="velocity_gate_m_s",
        positive=True,
    )
    azimuth_gate = (
        None
        if azimuth_gate_deg is None
        else _validated_real_scalar(
            azimuth_gate_deg,
            name="azimuth_gate_deg",
            positive=True,
        )
    )
    if not detections:
        return detections, {
            "enabled": True,
            "velocity_gate_m_s": velocity_gate,
            "azimuth_gate_deg": azimuth_gate,
            "kept": 0,
            "rejected": 0,
        }

    def _azimuth_difference_deg(lhs: Any, rhs: Any) -> float | None:
        try:
            lhs_value = float(lhs)
            rhs_value = float(rhs)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(lhs_value) or not np.isfinite(rhs_value):
            return None
        return abs(((lhs_value - rhs_value + 180.0) % 360.0) - 180.0)

    validated: list[tuple[dict[str, Any], float, float]] = []
    for detection in detections:
        if "range_m" not in detection or "velocity_m_s" not in detection:
            raise ValueError("ghost-filtered detections require range_m and velocity_m_s")
        detection_range = _validated_real_scalar(
            detection["range_m"],
            name="detection range_m",
            nonnegative=True,
        )
        detection_velocity = _validated_real_scalar(
            detection["velocity_m_s"],
            name="detection velocity_m_s",
        )
        validated.append((detection, detection_range, detection_velocity))

    sorted_dets = sorted(validated, key=lambda item: item[1])
    kept: list[dict[str, Any]] = []
    kept_velocities: list[float] = []
    for det, _, det_v in sorted_dets:
        is_ghost = False
        for kept_det, kept_v in zip(kept, kept_velocities, strict=True):
            if abs(det_v - kept_v) >= velocity_gate:
                continue
            if azimuth_gate is None:
                is_ghost = True
                break
            azimuth_diff = _azimuth_difference_deg(
                det.get("aoa_az_deg"),
                kept_det.get("aoa_az_deg"),
            )
            if azimuth_diff is None or azimuth_diff <= azimuth_gate:
                is_ghost = True
                break
        if not is_ghost:
            kept.append(det)
            kept_velocities.append(det_v)

    return kept, {
        "enabled": True,
        "velocity_gate_m_s": velocity_gate,
        "azimuth_gate_deg": azimuth_gate,
        "kept": len(kept),
        "rejected": len(detections) - len(kept),
    }


def run_detection_pipeline(
    rd_result: RangeDopplerResult,
    *,
    config: DetectionPipelineConfig,
    wavelength_m: float,
    detection_power_map: np.ndarray | None = None,
    post_detection_filter: DetectionFilter | None = None,
    return_stage_artifacts: bool = False,
    range_mode: str = "monostatic",
    tx_pos: np.ndarray | None = None,
    rx_pos: np.ndarray | None = None,
    aoa_az_map: np.ndarray | None = None,
    aoa_el_map: np.ndarray | None = None,
    reference_peak_power: float | None = None,
) -> DetectionPipelineResult:
    """Run CFAR -> NMS -> DBSCAN -> SNR filter -> optional gating/filtering."""
    power_map, range_axis_m, doppler_axis_hz = _validated_rd_inputs(
        rd_result,
        detection_power_map,
    )
    wavelength = _validate_wavelength_and_range_mode(wavelength_m, range_mode)
    reference_peak = float(np.max(power_map))
    if reference_peak_power is not None:
        reference_peak = max(
            reference_peak,
            _validated_real_scalar(
                reference_peak_power, name="reference_peak_power", nonnegative=True
            ),
        )

    cfar_result = cfar_2d(
        power_map,
        guard_cells=config.cfar_guard_cells,
        reference_cells=config.cfar_reference_cells,
        threshold_scale=config.cfar_threshold_scale,
        method=config.cfar_method,  # type: ignore[arg-type]
        os_rank=config.cfar_os_rank,
        edge_policy=config.cfar_edge_policy,
    )

    power_floor = None
    if config.cfar_max_dynamic_range_db is not None:
        power_floor = reference_peak * 10 ** (-config.cfar_max_dynamic_range_db / 10)
        threshold = np.maximum(cfar_result.threshold, power_floor)
        cfar_result = CFAR2DResult(
            detections=(power_map > threshold) & cfar_result.eligible,
            threshold=threshold,
            noise=cfar_result.noise,
            eligible=cfar_result.eligible,
            training_cell_count=cfar_result.training_cell_count,
        )

    snr_map, snr_valid_mask = _compute_snr_map(
        power_map,
        cfar_result.noise,
        cfar_result.eligible,
    )

    nms_mask = nms_peaks_2d(
        cfar_result.detections & snr_valid_mask,
        power_map,
        suppression_radius=config.nms_suppression_radius,
    )

    clustered_result = cluster_detections_2d(
        detections_mask=nms_mask,
        power_map=power_map,
        range_axis_m=range_axis_m,
        doppler_axis_hz=doppler_axis_hz,
        wavelength=wavelength,
        snr_map=snr_map,
        snr_valid_mask=snr_valid_mask,
        aoa_az_map=aoa_az_map,
        aoa_el_map=aoa_el_map,
        cluster_range_gate_m=config.cluster_range_gate_m,
        cluster_velocity_gate_m_s=config.cluster_velocity_gate_m_s,
        return_cluster_labels=return_stage_artifacts,
        range_mode=range_mode,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
    )
    if return_stage_artifacts:
        clustered, cluster_labels = clustered_result
    else:
        clustered = clustered_result
        cluster_labels = None

    detections: list[dict[str, Any]] = []
    for cluster in clustered:
        if cluster.snr_db < config.min_snr_db:
            continue
        detections.append(
            {
                "range_m": cluster.range_m,
                "velocity_m_s": cluster.velocity_m_s,
                "power_linear": cluster.power_linear,
                "snr_db": cluster.snr_db,
                "doppler_hz": cluster.doppler_hz,
                "path_closing_rate_m_s": cluster.path_closing_rate_m_s,
                "monostatic_equivalent_velocity_m_s": (cluster.monostatic_equivalent_velocity_m_s),
                "cluster_id": cluster.cluster_id,
                "num_detections": cluster.num_detections,
                "range_bin": cluster.range_bin,
                "doppler_bin": cluster.doppler_bin,
                "centroid_range_m": cluster.centroid_range_m,
                "centroid_velocity_m_s": cluster.centroid_velocity_m_s,
                "centroid_doppler_hz": cluster.centroid_doppler_hz,
                "centroid_path_closing_rate_m_s": cluster.centroid_path_closing_rate_m_s,
                "centroid_monostatic_equivalent_velocity_m_s": (
                    cluster.centroid_monostatic_equivalent_velocity_m_s
                ),
                # Cluster internals use NaN for an unavailable angle. Omit
                # unmeasured coordinates at the public detection boundary;
                # radial trackers need no fabricated angular measurement.
                **({"aoa_az_deg": cluster.aoa_az_deg} if aoa_az_map is not None else {}),
                **({"aoa_el_deg": cluster.aoa_el_deg} if aoa_el_map is not None else {}),
            }
        )

    detections = augment_detections_with_sensor_frame_xy(detections, range_mode=range_mode)

    (
        detections,
        detections_after_snr,
        range_gate_meta,
        ghost_meta,
        tracker_meta,
    ) = _apply_post_detection_filters(
        detections,
        config=config,
        post_detection_filter=post_detection_filter,
    )

    metadata = {
        **_cfar_stage_metadata(
            cfar_result,
            method=config.cfar_method,
            threshold_scale=config.cfar_threshold_scale,
            os_rank=config.cfar_os_rank,
            edge_policy=config.cfar_edge_policy,
            snr_valid_mask=snr_valid_mask,
        ),
        "raw_cfar_cells": int(np.sum(cfar_result.detections)),
        "snr_valid_cfar_cells": int(np.sum(cfar_result.detections & snr_valid_mask)),
        "nms_peaks": int(np.sum(nms_mask)),
        "clusters": len(clustered),
        "snr_kept": detections_after_snr,
        "range_gate": range_gate_meta,
        "ghost_filter": ghost_meta,
        "tracker": tracker_meta,
        "final_kept": len(detections),
        "cartesian_position_policy": (
            "receiver_local_monostatic_radius"
            if range_mode == "monostatic"
            else "unavailable_bistatic_localization_requires_calibrated_rx_pose"
        ),
        "cfar_max_dynamic_range_db": config.cfar_max_dynamic_range_db,
        "cfar_power_floor_linear": power_floor,
        "cfar_reference_peak_power_linear": reference_peak,
    }

    stages = None
    if return_stage_artifacts:
        stages = _detection_stage_artifacts(
            power_map,
            cfar_result,
            nms_mask=nms_mask,
            snr_map=snr_map,
            snr_valid_mask=snr_valid_mask,
            cluster_labels=cluster_labels,
        )

    return DetectionPipelineResult(detections=detections, metadata=metadata, stages=stages)


def run_vote_fusion_pipeline(
    rd_result: RangeDopplerResult,
    *,
    azimuth_cube: np.ndarray,
    azimuth_axis_deg: np.ndarray | list[float],
    config: DetectionPipelineConfig,
    vote_config: VoteFusionConfig,
    wavelength_m: float,
    post_detection_filter: DetectionFilter | None = None,
    return_stage_artifacts: bool = False,
    range_mode: str = "monostatic",
    tx_pos: np.ndarray | None = None,
    rx_pos: np.ndarray | None = None,
) -> DetectionPipelineResult:
    """Run 3-projection vote fusion on an azimuth cube and keep detections in 3D."""
    if config.cfar_max_dynamic_range_db is not None:
        raise ValueError("cfar_max_dynamic_range_db supports only rd and beam_scan detection")
    rd_power, range_axis_m, doppler_axis_hz = _validated_rd_inputs(rd_result, None)
    power_cube, azimuth_axis = _validated_azimuth_inputs(
        azimuth_cube,
        azimuth_axis_deg,
        rd_shape=rd_power.shape,
    )
    wavelength = _validate_wavelength_and_range_mode(wavelength_m, range_mode)
    _validate_vote_config(vote_config)

    fusion_result = vote_fusion_3d(
        power_cube,
        azimuth_axis_deg=azimuth_axis,
        config=vote_config,
    )
    snr_map, snr_valid_mask = _compute_snr_map(
        fusion_result.rd_power,
        fusion_result.rd_cfar.noise,
        fusion_result.rd_cfar.eligible,
    )
    # Ineligible candidates must neither suppress usable cells nor consume
    # the peak quota. SNR validity is a candidate contract, not a post-filter.
    nms_mask_3d = nms_peaks_3d(
        fusion_result.fused_mask & snr_valid_mask[np.newaxis, :, :],
        fusion_result.power_cube,
        suppression_radius=vote_config.nms_suppression_radius,
        max_peaks=vote_config.nms_max_peaks,
    )
    if np.any(nms_mask_3d) and vote_config.post_nms_min_peak_divisor > 1.0:
        nms_values = fusion_result.power_cube[nms_mask_3d]
        if nms_values.size > 0:
            min_peak_value = float(np.max(nms_values)) / float(
                vote_config.post_nms_min_peak_divisor
            )
            nms_mask_3d &= fusion_result.power_cube >= min_peak_value

    clustered_result = cluster_detections_3d(
        nms_mask_3d,
        fusion_result.power_cube,
        range_axis_m,
        doppler_axis_hz,
        azimuth_axis,
        wavelength=wavelength,
        snr_map=snr_map,
        snr_valid_mask=snr_valid_mask,
        cluster_range_gate_m=vote_config.cluster_range_gate_m,
        cluster_velocity_gate_m_s=vote_config.cluster_velocity_gate_m_s,
        cluster_azimuth_gate_deg=vote_config.cluster_azimuth_gate_deg,
        return_cluster_labels=return_stage_artifacts,
        range_mode=range_mode,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        votes=fusion_result.votes,
    )
    if return_stage_artifacts:
        clustered, cluster_labels_3d = clustered_result
    else:
        clustered = clustered_result
        cluster_labels_3d = None

    detections: list[dict[str, Any]] = []
    snr_invalid_clusters_rejected = 0
    for cluster in clustered:
        if not snr_valid_mask[cluster.doppler_bin, cluster.range_bin]:
            snr_invalid_clusters_rejected += 1
            continue
        if not np.isfinite(cluster.snr_db):
            raise ValueError("vote-fusion cluster produced a nonfinite SNR")
        if cluster.snr_db < config.min_snr_db:
            continue
        detections.append(
            {
                "range_m": cluster.range_m,
                "velocity_m_s": cluster.velocity_m_s,
                "power_linear": cluster.power_linear,
                "snr_db": cluster.snr_db,
                "doppler_hz": cluster.doppler_hz,
                "path_closing_rate_m_s": cluster.path_closing_rate_m_s,
                "monostatic_equivalent_velocity_m_s": (cluster.monostatic_equivalent_velocity_m_s),
                "cluster_id": cluster.cluster_id,
                "num_detections": cluster.num_detections,
                "range_bin": cluster.range_bin,
                "doppler_bin": cluster.doppler_bin,
                "azimuth_bin": cluster.azimuth_bin,
                "centroid_aoa_az_deg": cluster.centroid_aoa_az_deg,
                "vote_count": cluster.vote_count,
                "centroid_range_m": cluster.centroid_range_m,
                "centroid_velocity_m_s": cluster.centroid_velocity_m_s,
                "centroid_doppler_hz": cluster.centroid_doppler_hz,
                "centroid_path_closing_rate_m_s": cluster.centroid_path_closing_rate_m_s,
                "centroid_monostatic_equivalent_velocity_m_s": (
                    cluster.centroid_monostatic_equivalent_velocity_m_s
                ),
                "aoa_az_deg": cluster.aoa_az_deg,
                "aoa_el_deg": float("nan"),
            }
        )

    (
        detections,
        detections_after_snr,
        range_gate_meta,
        ghost_meta,
        tracker_meta,
    ) = _apply_post_detection_filters(
        detections,
        config=config,
        post_detection_filter=post_detection_filter,
    )

    metadata = {
        **_cfar_stage_metadata(
            fusion_result.rd_cfar,
            method=vote_config.rd.method,
            threshold_scale=vote_config.rd.threshold_scale,
            os_rank=vote_config.rd.os_rank,
            edge_policy=vote_config.rd.edge_policy,
            snr_valid_mask=snr_valid_mask,
        ),
        "detection_mode": "vote_fusion_3d",
        "projection_edge_policies": dict(fusion_result.projection_edge_policies),
        "vote_min_votes": int(vote_config.min_votes),
        "vote_pre_threshold_db": float(vote_config.pre_threshold_db),
        "raw_cfar_cells_rd": int(np.sum(fusion_result.rd_cfar.detections)),
        "raw_cfar_cells_raz": int(np.sum(fusion_result.raz_cfar.detections)),
        "raw_cfar_cells_daz": int(np.sum(fusion_result.daz_cfar.detections)),
        "projection_peaks_rd": int(np.sum(fusion_result.rd_peak_mask)),
        "projection_peaks_raz": int(np.sum(fusion_result.raz_peak_mask)),
        "projection_peaks_daz": int(np.sum(fusion_result.daz_peak_mask)),
        "raw_fused_cells": int(np.sum(fusion_result.fused_mask)),
        "nms_peaks": int(np.sum(nms_mask_3d)),
        "clusters": len(clustered),
        "snr_invalid_clusters_rejected": snr_invalid_clusters_rejected,
        "snr_kept": detections_after_snr,
        "range_gate": range_gate_meta,
        "ghost_filter": ghost_meta,
        "tracker": tracker_meta,
        "final_kept": len(detections),
    }

    stages = None
    if return_stage_artifacts:
        rd_cluster_labels = (
            collapse_cluster_labels_to_rd(cluster_labels_3d, fusion_result.power_cube)
            if cluster_labels_3d is not None
            else np.full(rd_power.shape, -1, dtype=np.int32)
        )
        stages = _detection_stage_artifacts(
            fusion_result.rd_power,
            fusion_result.rd_cfar,
            nms_mask=np.any(nms_mask_3d, axis=0),
            snr_map=snr_map,
            snr_valid_mask=snr_valid_mask,
            cluster_labels=rd_cluster_labels,
            extra_artifacts={
                "vote_raz_projection_power_db": DB_POWER_SCALE
                * np.log10(np.maximum(fusion_result.raz_power, POWER_FLOOR_LINEAR)),
                "vote_daz_projection_power_db": DB_POWER_SCALE
                * np.log10(np.maximum(fusion_result.daz_power, POWER_FLOOR_LINEAR)),
                "vote_rd_peak_mask": np.array(fusion_result.rd_peak_mask, copy=True),
                "vote_raz_peak_mask": np.array(fusion_result.raz_peak_mask, copy=True),
                "vote_daz_peak_mask": np.array(fusion_result.daz_peak_mask, copy=True),
                "vote_raz_cfar_mask": np.array(fusion_result.raz_cfar.detections, copy=True),
                "vote_daz_cfar_mask": np.array(fusion_result.daz_cfar.detections, copy=True),
                "vote_rd_fused_mask": np.array(fusion_result.rd_fused_mask, copy=True),
                "vote_aoa_az_map": np.array(fusion_result.aoa_az_map, copy=True),
                "vote_counts": np.array(fusion_result.votes, copy=True),
                "vote_fused_mask": np.array(fusion_result.fused_mask, copy=True),
                "vote_nms_mask": np.array(nms_mask_3d, copy=True),
                "vote_cluster_labels": (
                    np.array(cluster_labels_3d, copy=True)
                    if cluster_labels_3d is not None
                    else np.full(fusion_result.power_cube.shape, -1, dtype=np.int32)
                ),
            },
        )

    return DetectionPipelineResult(detections=detections, metadata=metadata, stages=stages)


def run_angle_enhanced_detection_pipeline(
    rd_result: RangeDopplerResult,
    *,
    azimuth_cube: np.ndarray,
    azimuth_axis_deg: np.ndarray | list[float],
    config: DetectionPipelineConfig,
    angle_config: AnglePeakDetectionConfig,
    wavelength_m: float,
    detection_power_map: np.ndarray | None = None,
    post_detection_filter: DetectionFilter | None = None,
    return_stage_artifacts: bool = False,
    range_mode: str = "monostatic",
    tx_pos: np.ndarray | None = None,
    rx_pos: np.ndarray | None = None,
) -> DetectionPipelineResult:
    """Run RD CFAR, then lift RD peaks into a sparse azimuth-aware 3D detector."""
    if config.cfar_max_dynamic_range_db is not None:
        raise ValueError("cfar_max_dynamic_range_db supports only rd and beam_scan detection")
    power_map, range_axis_m, doppler_axis_hz = _validated_rd_inputs(
        rd_result,
        detection_power_map,
    )
    azimuth_power_cube, azimuth_axis = _validated_azimuth_inputs(
        azimuth_cube,
        azimuth_axis_deg,
        rd_shape=power_map.shape,
    )
    wavelength = _validate_wavelength_and_range_mode(wavelength_m, range_mode)
    _validate_angle_config(angle_config)

    cfar_result = cfar_2d(
        power_map,
        guard_cells=config.cfar_guard_cells,
        reference_cells=config.cfar_reference_cells,
        threshold_scale=config.cfar_threshold_scale,
        method=config.cfar_method,  # type: ignore[arg-type]
        os_rank=config.cfar_os_rank,
        edge_policy=config.cfar_edge_policy,
    )
    snr_map, snr_valid_mask = _compute_snr_map(
        power_map,
        cfar_result.noise,
        cfar_result.eligible,
    )
    rd_nms_mask = nms_peaks_2d(
        cfar_result.detections & snr_valid_mask,
        power_map,
        suppression_radius=config.nms_suppression_radius,
    )

    angle_result = extract_rd_anchored_angle_candidates(
        rd_nms_mask,
        azimuth_power_cube,
        azimuth_axis,
        config=angle_config,
    )
    clustered_result = cluster_detections_3d(
        angle_result.candidate_mask,
        angle_result.power_cube,
        range_axis_m,
        doppler_axis_hz,
        azimuth_axis,
        wavelength=wavelength,
        snr_map=snr_map,
        snr_valid_mask=snr_valid_mask,
        cluster_range_gate_m=config.cluster_range_gate_m,
        cluster_velocity_gate_m_s=config.cluster_velocity_gate_m_s,
        cluster_azimuth_gate_deg=angle_config.cluster_azimuth_gate_deg,
        return_cluster_labels=return_stage_artifacts,
        range_mode=range_mode,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
    )
    if return_stage_artifacts:
        clustered, cluster_labels_3d = clustered_result
    else:
        clustered = clustered_result
        cluster_labels_3d = None

    detections: list[dict[str, Any]] = []
    snr_invalid_clusters_rejected = 0
    for cluster in clustered:
        if not snr_valid_mask[cluster.doppler_bin, cluster.range_bin]:
            snr_invalid_clusters_rejected += 1
            continue
        if not np.isfinite(cluster.snr_db):
            raise ValueError("angle-enhanced cluster produced a nonfinite SNR")
        if cluster.snr_db < config.min_snr_db:
            continue
        detections.append(
            {
                "range_m": cluster.range_m,
                "velocity_m_s": cluster.velocity_m_s,
                "power_linear": cluster.power_linear,
                "snr_db": cluster.snr_db,
                "doppler_hz": cluster.doppler_hz,
                "path_closing_rate_m_s": cluster.path_closing_rate_m_s,
                "monostatic_equivalent_velocity_m_s": (cluster.monostatic_equivalent_velocity_m_s),
                "cluster_id": cluster.cluster_id,
                "num_detections": cluster.num_detections,
                "range_bin": cluster.range_bin,
                "doppler_bin": cluster.doppler_bin,
                "azimuth_bin": cluster.azimuth_bin,
                "centroid_aoa_az_deg": cluster.centroid_aoa_az_deg,
                "centroid_range_m": cluster.centroid_range_m,
                "centroid_velocity_m_s": cluster.centroid_velocity_m_s,
                "centroid_doppler_hz": cluster.centroid_doppler_hz,
                "centroid_path_closing_rate_m_s": cluster.centroid_path_closing_rate_m_s,
                "centroid_monostatic_equivalent_velocity_m_s": (
                    cluster.centroid_monostatic_equivalent_velocity_m_s
                ),
                "aoa_az_deg": cluster.aoa_az_deg,
                "aoa_el_deg": float("nan"),
            }
        )

    detections, sidelobe_meta = suppress_angle_sidelobes(detections, config=angle_config)
    detections = augment_detections_with_sensor_frame_xy(detections, range_mode=range_mode)
    (
        detections,
        detections_after_snr,
        range_gate_meta,
        ghost_meta,
        tracker_meta,
    ) = _apply_post_detection_filters(
        detections,
        config=config,
        post_detection_filter=post_detection_filter,
    )

    metadata = {
        **_cfar_stage_metadata(
            cfar_result,
            method=config.cfar_method,
            threshold_scale=config.cfar_threshold_scale,
            os_rank=config.cfar_os_rank,
            edge_policy=config.cfar_edge_policy,
            snr_valid_mask=snr_valid_mask,
        ),
        "detection_mode": "rd_angle_peaks_3d",
        "raw_cfar_cells": int(np.sum(cfar_result.detections)),
        "snr_valid_cfar_cells": int(np.sum(cfar_result.detections & snr_valid_mask)),
        "rd_nms_peaks": int(np.sum(rd_nms_mask)),
        "angle_candidate_cells": int(np.sum(angle_result.candidate_mask)),
        "rd_cells_with_multiple_angle_peaks": int(np.sum(angle_result.peak_count_map > 1)),
        "clusters": len(clustered),
        "snr_invalid_clusters_rejected": snr_invalid_clusters_rejected,
        "sidelobe_suppression": sidelobe_meta,
        "snr_kept": detections_after_snr,
        "range_gate": range_gate_meta,
        "ghost_filter": ghost_meta,
        "tracker": tracker_meta,
        "final_kept": len(detections),
    }

    stages = None
    if return_stage_artifacts:
        rd_cluster_labels = (
            collapse_cluster_labels_to_rd(cluster_labels_3d, angle_result.power_cube)
            if cluster_labels_3d is not None
            else np.full(power_map.shape, -1, dtype=np.int32)
        )
        stages = _detection_stage_artifacts(
            power_map,
            cfar_result,
            nms_mask=rd_nms_mask,
            snr_map=snr_map,
            snr_valid_mask=snr_valid_mask,
            cluster_labels=rd_cluster_labels,
            extra_artifacts={
                "angle_candidate_mask": np.array(angle_result.candidate_mask, copy=True),
                "angle_peak_count_map": np.array(angle_result.peak_count_map, copy=True),
                "angle_aoa_az_map": np.array(angle_result.aoa_az_map, copy=True),
            },
        )

    return DetectionPipelineResult(detections=detections, metadata=metadata, stages=stages)
