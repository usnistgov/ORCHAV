"""Beam-wise detection on retained spatial-FFT power.

Each beam gets its own RD background estimate and NMS. At each RD cell,
only the strongest visible beam supplies a candidate. This avoids fixed
broadside voltage cancellation and duplicate sidelobe detections; it does
not resolve two targets occupying exactly the same RD cell. Directions use
the forward hemisphere of the receiving planar array, without target truth.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from shared.sensing.arrays import forward_hemisphere_angles
from shared.sensing.detection.angle import augment_detections_with_sensor_frame_xy
from shared.sensing.detection.cfar import CFAR2DResult
from shared.sensing.detection.pipeline import (
    DetectionFilter,
    DetectionPipelineConfig,
    DetectionPipelineResult,
    _apply_post_detection_filters,
    _detection_stage_artifacts,
    _validated_power_array,
    run_detection_pipeline,
)
from shared.sensing.signal_processing import RangeDopplerResult, apply_pre_cfar_doppler_guards


def _array_observability(num_rows: int, num_cols: int) -> dict[str, Any]:
    """Describe measured spatial coordinates separately from plot assumptions."""
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1
        for v in (num_rows, num_cols)
    ):
        raise ValueError("Array dimensions must be positive integers")
    num_rows, num_cols = int(num_rows), int(num_cols)
    return {
        "aoa_az_observable": num_cols > 1 and num_rows > 1,
        "aoa_el_observable": num_rows > 1,
        "azimuth_direction_cosine_observable": num_cols > 1,
        "elevation_direction_cosine_observable": num_rows > 1,
        "angle_assumptions": (["zero_horizontal_direction_cosine"] if num_cols == 1 else [])
        + (["zero_vertical_direction_cosine"] if num_rows == 1 else []),
    }


def refine_array_directions(
    detections: list[dict[str, Any]],
    element_rd: np.ndarray,
    *,
    num_rows: int,
    num_cols: int,
    horizontal_spacing_wavelengths: float,
    vertical_spacing_wavelengths: float,
    oversampling: int,
) -> list[dict[str, Any]]:
    """Estimate directions on a finer grid around each already detected beam.

    Zero-padding interpolates the array response; it does not increase aperture
    or resolve additional targets in a shared RD cell. The search stays within
    one original bin spacing of the selected beam, in the positive-x hemisphere.
    Singleton axes retain a zero direction cosine as an explicit assumption,
    never an estimated bearing. Observability flags distinguish these assumed
    coordinates from measurements. A horizontal ULA measures uy, not spherical
    azimuth without an elevation assumption. Detection support and power
    remain those of the original beam; coarse angles are retained explicitly.
    """
    values = np.asarray(element_rd)
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in (num_rows, num_cols)
    ):
        raise ValueError("Array dimensions must be integers")
    rows, cols = int(num_rows), int(num_cols)
    if rows < 1 or cols < 1 or values.ndim != 3 or values.shape[0] != rows * cols:
        raise ValueError("Angle refinement requires an element-by-Doppler-by-range tensor")
    if not np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("Angle refinement requires finite complex element samples")
    if (
        isinstance(oversampling, (bool, np.bool_))
        or not isinstance(oversampling, (int, np.integer))
        or not 1 <= oversampling <= 16
    ):
        raise ValueError("Angle oversampling must be an integer from 1 to 16")
    dx, dz = float(horizontal_spacing_wavelengths), float(vertical_spacing_wavelengths)
    if not all(np.isfinite(v) and 0 < v <= 0.5 for v in (dx, dz)):
        raise ValueError("Angle refinement requires spacings within (0, 0.5] wavelengths")
    ne = rows * oversampling if rows > 1 else 1
    na = cols * oversampling if cols > 1 else 1
    uy = np.fft.fftshift(np.fft.fftfreq(na, d=dx))
    uz = -np.fft.fftshift(np.fft.fftfreq(ne, d=dz))
    z, y = np.meshgrid(uz, uy, indexing="ij")
    x_squared = 1.0 - y**2 - z**2
    output = deepcopy(detections)
    for detection in output:
        d, r = detection["doppler_bin"], detection["range_bin"]
        if not (
            not isinstance(d, (bool, np.bool_))
            and not isinstance(r, (bool, np.bool_))
            and isinstance(d, (int, np.integer))
            and isinstance(r, (int, np.integer))
            and 0 <= d < values.shape[1]
            and 0 <= r < values.shape[2]
        ):
            raise ValueError("Detection RD indices must lie within the complex element tensor")
        az, el = float(detection["aoa_az_deg"]), float(detection["aoa_el_deg"])
        if not np.isfinite(az) or not np.isfinite(el):
            raise ValueError("Coarse beam angles must be finite")
        center_y = np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el))
        center_z = np.sin(np.deg2rad(el))
        visible = (
            (x_squared > 0)
            & (abs(y - center_y) <= 1 / (cols * dx))
            & (abs(z - center_z) <= 1 / (rows * dz))
        )
        if not np.any(visible):
            raise ValueError("The coarse beam has no physical local refinement neighborhood")
        grid = values[:, d, r].reshape(cols, rows).T
        power = abs(np.fft.fftshift(np.fft.fft2(grid, s=(ne, na)))) ** 2
        if not np.any(power[visible] > 0):
            raise ValueError("Cannot refine a direction without element signal energy")
        peak = np.unravel_index(np.argmax(np.where(visible, power, -np.inf)), power.shape)
        detection.update(
            coarse_aoa_az_deg=az,
            coarse_aoa_el_deg=el,
            aoa_az_deg=float(np.rad2deg(np.arctan2(y[peak], np.sqrt(x_squared[peak])))),
            aoa_el_deg=float(np.rad2deg(np.arcsin(z[peak]))),
            beam_angle_oversampling=int(oversampling),
            **_array_observability(rows, cols),
        )
        for key in ("x_m", "y_m", "z_m"):
            detection.pop(key, None)
    return augment_detections_with_sensor_frame_xy(output)


def visible_array_beams(
    power: np.ndarray,
    elevation_direction_cosine: np.ndarray,
    azimuth_direction_cosine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select physical forward beams from an ``[E,A,D,R]`` power tensor.

    Planar spatial FFT axes are direction cosines, not independent spherical
    angles. Joint conversion uses x = sqrt(1 - y**2 - z**2). Exact endfire
    bins are excluded because their forward/backward position is singular.
    """
    cube = _validated_power_array(power, dimensions=4, name="angular power")
    visible, azimuth, elevation = forward_hemisphere_angles(
        elevation_direction_cosine, azimuth_direction_cosine
    )
    if cube.shape[:2] != visible.shape:
        raise ValueError("Beam direction-cosine coordinates must match the angular tensor")
    return cube[visible], azimuth[visible], elevation[visible]


def run_beam_scan_detection_pipeline(
    rd_result: RangeDopplerResult,
    *,
    beam_power: np.ndarray,
    beam_azimuth_deg: np.ndarray,
    beam_elevation_deg: np.ndarray,
    config: DetectionPipelineConfig,
    wavelength_m: float,
    edge_guard_bins: int = 0,
    dc_guard_bins: int = 0,
    post_detection_filter: DetectionFilter | None = None,
    return_stage_artifacts: bool = False,
    range_mode: str = "monostatic",
    tx_pos: np.ndarray | None = None,
    rx_pos: np.ndarray | None = None,
    reference: dict[str, Any] | None = None,
    num_rows: int | None = None,
    num_cols: int | None = None,
) -> DetectionPipelineResult:
    """Detect independently in beams and track their combined candidates once.

    Optional 2-D stage maps select each cell's strongest beam throughout:
    power, noise, eligibility, NMS and labels refer to that same beam. The
    metadata identifies this projection; it is not CFAR on a max-pooled RD
    background and does not inherit a single-beam false-alarm calibration.
    Producers supply physical row/column counts so coarse outputs carry the same
    observability contract as refined outputs, before tracking or fusion. Omitting
    both counts retains compatibility with older geometry-free direct callers.
    """
    if (num_rows is None) != (num_cols is None):
        raise ValueError("Array observability requires both row and column counts")
    observability = _array_observability(num_rows, num_cols) if num_rows is not None else {}
    if range_mode != "monostatic":
        raise ValueError("beam_scan currently requires monostatic range for Cartesian projection")
    beams = _validated_power_array(beam_power, dimensions=3, name="beam_power")
    if beams.shape[0] == 0 or beams.shape[1:] != rd_result.power_linear.shape:
        raise ValueError("Beam power must share the supplied range-Doppler grid")
    az = np.asarray(beam_azimuth_deg, dtype=np.float64)
    el = np.asarray(beam_elevation_deg, dtype=np.float64)
    if any(v.shape != (len(beams),) or not np.all(np.isfinite(v)) for v in (az, el)):
        raise ValueError("Each beam requires finite azimuth and elevation coordinates")
    reference_peaks = None
    if reference is not None:
        reference_peaks = np.asarray(reference.get("power_linear"), dtype=float)
        if (
            reference.get("basis") != "pre_clutter_spatial_fft_peak"
            or reference_peaks.shape != (len(beams),)
            or not np.all(np.isfinite(reference_peaks))
            or np.any(reference_peaks < 0)
            or not np.array_equal(reference.get("azimuth_deg"), az)
            or not np.array_equal(reference.get("elevation_deg"), el)
        ):
            raise ValueError(
                "Detector power reference must match the exact visible beam coordinates"
            )
    strongest = np.argmax(beams, axis=0)
    # Physical and ghost gates run after beam selection; tracker time advances
    # exactly once per CPI, including a CPI with no surviving detections.
    beam_config = replace(
        config,
        min_detection_range_m=None,
        max_detection_range_m=None,
        ghost_filter_velocity_gate_m_s=None,
        ghost_filter_azimuth_gate_deg=None,
    )
    detections: list[dict[str, Any]] = []
    projected: dict[str, np.ndarray] = {}
    cluster_offset = 0
    for index, beam in enumerate(beams):
        guarded = apply_pre_cfar_doppler_guards(
            beam,
            edge_guard_bins=edge_guard_bins,
            dc_guard_bins=dc_guard_bins,
        )
        result = run_detection_pipeline(
            RangeDopplerResult(
                power_linear=guarded,
                range_axis_m=rd_result.range_axis_m,
                doppler_axis_hz=rd_result.doppler_axis_hz,
                cube=None,
            ),
            config=beam_config,
            wavelength_m=wavelength_m,
            return_stage_artifacts=return_stage_artifacts,
            range_mode=range_mode,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            reference_peak_power=(
                None if reference_peaks is None else float(reference_peaks[index])
            ),
        )
        selected = strongest == index
        for detection in result.detections:
            cell = (detection["doppler_bin"], detection["range_bin"])
            if not selected[cell]:
                continue
            detection.update(
                aoa_az_deg=float(az[index]),
                aoa_el_deg=float(el[index]),
                beam_index=index,
                **deepcopy(observability),
                cluster_id=int(detection["cluster_id"]) + cluster_offset,
            )
            detections.append(detection)
        if result.stages is not None:
            for key in (
                "cfar_input_power",
                "cfar_mask",
                "cfar_noise_floor",
                "cfar_threshold",
                "cfar_training_cell_count",
                "cfar_eligible_mask",
                "nms_mask",
                "snr_map",
                "snr_valid_mask",
                "cluster_labels",
            ):
                values = getattr(result.stages, key)
                if key == "cluster_labels":
                    values = np.where(values >= 0, values + cluster_offset, -1)
                if key not in projected:
                    projected[key] = np.empty_like(values)
                projected[key][selected] = values[selected]
        cluster_offset += int(result.metadata["clusters"])
    detections = augment_detections_with_sensor_frame_xy(detections)
    detections, snr_kept, range_meta, ghost_meta, tracker_meta = _apply_post_detection_filters(
        detections,
        config=config,
        post_detection_filter=post_detection_filter,
    )
    metadata = {
        "array_detection_mode": "beam_scan",
        "beam_count": len(beams),
        "beam_selection": "strongest_visible_beam_per_rd_cell_first_index_on_ties",
        "cfar_background": "within_each_beam",
        "cfar_max_dynamic_range_db": config.cfar_max_dynamic_range_db,
        "cfar_power_floor_reference": (
            "max_pre_clutter_and_guarded_peak_within_each_beam"
            if reference is not None
            else "strongest_guarded_cell_within_each_beam"
        ),
        "detector_power_reference": deepcopy(reference),
        "angular_hemisphere": "rx_local_positive_x",
        "stage_projection": "strongest_beam_per_rd_cell",
        "snr_kept": snr_kept,
        "range_gate": range_meta,
        "ghost_filter": ghost_meta,
        "tracker": tracker_meta,
        "final_kept": len(detections),
    }
    stages = None
    if return_stage_artifacts:
        stages = _detection_stage_artifacts(
            projected["cfar_input_power"],
            CFAR2DResult(
                detections=projected["cfar_mask"],
                threshold=projected["cfar_threshold"],
                noise=projected["cfar_noise_floor"],
                eligible=projected["cfar_eligible_mask"],
                training_cell_count=projected["cfar_training_cell_count"],
            ),
            nms_mask=projected["nms_mask"],
            snr_map=projected["snr_map"],
            snr_valid_mask=projected["snr_valid_mask"],
            cluster_labels=projected["cluster_labels"],
            extra_artifacts={"selected_beam_index": strongest},
        )
    return DetectionPipelineResult(detections=detections, metadata=metadata, stages=stages)
