"""Range-Doppler and array products from one coherent processing interval.

The processor supplies a Sionna CIR stack from one solve. Range projection,
slow-time FFT, optional array processing, and detection populate a sensing
payload with explicit axes and processing metadata.
"""

from __future__ import annotations

import os
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from generator.core.sensing.profiling import SensingStageProfiler
from shared.frames.sionna_metadata import SIONNA_INTERACTION_LABELS_LOWER, SIONNA_INVALID_OBJECT_ID
from shared.logging import get_logger
from shared.sensing.arrays import (
    PLANAR_ELEMENT_ORDER,
    SPATIAL_FFT_NORMALIZATION,
    SPATIAL_PHASE_CONVENTION,
    planar_element_coordinates,
    planar_spatial_fft,
    strongest_array_directions,
)
from shared.sensing.detection.beam_scan import refine_array_directions, visible_array_beams
from shared.sensing.products import (
    TEMPORAL_SUPPORT_FULL_ACQUISITION,
    TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
    add_product_spec,
    catalog_axis,
    dense_axis,
)
from shared.sensing.range_response import frequency_to_delay, paths_to_frequency
from shared.sensing.signal_processing import (
    C0,
    DELAY_PROJECTION_POINT_BIN_FLOOR,
    DOPPLER_FFT_NORMALIZATION_UNITARY,
    RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
    WINDOW_NORMALIZATION_ENERGY,
    RangeDopplerResult,
    apply_aoa_filter,
    apply_pre_cfar_doppler_guards,
    cir_to_samples_batch,
    cir_to_samples_per_antenna,
    coherent_clutter_metadata,
    inject_target_relative_complex_awgn,
    remove_window_weighted_mean,
    sensing_window,
    window_energy_correction,
)

if TYPE_CHECKING:
    from ..processor import CoherentCpiFftProcessor

logger = get_logger(__name__)


def _interaction_label(value: Any) -> str:
    try:
        return SIONNA_INTERACTION_LABELS_LOWER.get(int(value), f"type_{int(value)}")
    except (TypeError, ValueError):
        return "unknown"


def _target_mesh_snapshot(simulation_objects: Any) -> list[str]:
    target_managers = getattr(simulation_objects, "target_managers", None) or []
    snapshot: list[str] = []
    for manager in target_managers:
        name = getattr(getattr(manager, "config", None), "name", "target")
        mesh_idx = getattr(manager, "current_mesh_idx", None)
        mesh_name = None
        meshes = getattr(manager, "meshes", None)
        if isinstance(mesh_idx, int) and meshes and 0 <= mesh_idx < len(meshes):
            mesh_name = os.path.basename(meshes[mesh_idx])
        if mesh_name:
            snapshot.append(f"{name}:{mesh_name}")
        elif mesh_idx is not None:
            snapshot.append(f"{name}:mesh_idx={mesh_idx}")
        else:
            snapshot.append(str(name))
    return snapshot


def _scene_object_name_map(scene: Any) -> dict[int, str]:
    objects = getattr(scene, "objects", None)
    if not objects:
        return {}

    mapping: dict[int, str] = {}
    for obj_name, obj in objects.items():
        obj_id = getattr(obj, "object_id", None)
        if obj_id is None:
            continue
        try:
            mapping[int(obj_id)] = str(obj_name)
        except (TypeError, ValueError):
            continue
    return mapping


def _log_nonfinite_cir_path_details(
    *,
    a_raw: np.ndarray,
    tau_raw: np.ndarray,
    keep_mask: np.ndarray,
    frame_idx: int,
    paths: Any,
    simulation_objects: Any,
) -> None:
    if paths is None:
        return

    try:
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        interactions = np.asarray(paths.interactions.numpy())
        objects = np.asarray(paths.objects.numpy())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return

    if valid.ndim != 3 or interactions.ndim != 4 or objects.ndim != 4:
        return

    a_base_shape = a_raw.shape[:-1]
    if not a_base_shape:
        return

    tau_shape = tau_raw.shape
    tau_size = int(np.prod(tau_shape)) if tau_shape else 0
    object_name_map = _scene_object_name_map(getattr(simulation_objects, "scene", None))
    mesh_snapshot = _target_mesh_snapshot(simulation_objects)

    bad_flat_indices = np.flatnonzero(~keep_mask)
    max_examples = min(4, bad_flat_indices.size)
    for flat_idx in bad_flat_indices[:max_examples]:
        try:
            a_index = tuple(int(v) for v in np.unravel_index(int(flat_idx), a_base_shape))
        except ValueError:
            a_index = (int(flat_idx),)

        rx_idx = tx_idx = path_idx = None
        tau_index: tuple[int, ...] | None = None
        if tau_size > 0:
            tau_linear = int(flat_idx) % tau_size
            try:
                tau_index = tuple(int(v) for v in np.unravel_index(tau_linear, tau_shape))
            except ValueError:
                tau_index = None
            if tau_index is not None:
                if len(tau_index) >= 3:
                    rx_idx, tx_idx, path_idx = tau_index[-3], tau_index[-2], tau_index[-1]
                elif len(tau_index) == 1:
                    path_idx = tau_index[0]

        object_chain: list[str] = []
        interaction_chain: list[str] = []
        if (
            rx_idx is not None
            and tx_idx is not None
            and path_idx is not None
            and 0 <= rx_idx < valid.shape[0]
            and 0 <= tx_idx < valid.shape[1]
            and 0 <= path_idx < valid.shape[2]
        ):
            if bool(valid[rx_idx, tx_idx, path_idx]):
                depth_count = interactions.shape[0]
                for depth_idx in range(depth_count):
                    interaction_value = interactions[depth_idx, rx_idx, tx_idx, path_idx]
                    object_value = objects[depth_idx, rx_idx, tx_idx, path_idx]
                    try:
                        interaction_int = int(interaction_value)
                    except (TypeError, ValueError):
                        continue
                    if interaction_int <= 0:
                        continue
                    interaction_chain.append(_interaction_label(interaction_int))
                    try:
                        object_int = int(object_value)
                    except (TypeError, ValueError):
                        object_chain.append(str(object_value))
                        continue
                    if object_int == int(SIONNA_INVALID_OBJECT_ID):
                        object_chain.append("INVALID")
                    else:
                        object_chain.append(object_name_map.get(object_int, f"object_{object_int}"))

        logger.warning(
            "Non-finite coherent CIR path at frame %d: flat_row=%d a_index=%s tau_index=%s "
            "rx=%s tx=%s path=%s objects=%s interactions=%s target_meshes=%s",
            int(frame_idx),
            int(flat_idx),
            a_index,
            tau_index,
            rx_idx,
            tx_idx,
            path_idx,
            object_chain or ["unresolved"],
            interaction_chain or ["unresolved"],
            mesh_snapshot or ["none"],
        )


def _drop_nonfinite_cir_paths(
    a_2d: np.ndarray,
    tau_1d: np.ndarray,
    *,
    frame_idx: int,
    a_raw: Optional[np.ndarray] = None,
    tau_raw: Optional[np.ndarray] = None,
    paths: Any = None,
    simulation_objects: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove nonfinite CIR rows and return their mask on the original path axis.

    Sionna occasionally returns a path with finite delay but NaN complex gains
    for all coherent samples. One such row contaminates the whole CPI because
    it maps into a fast-time bin before the Doppler FFT. Drop those rows here
    so a single bad path does not poison the RD products.
    """
    if a_2d.ndim != 2 or tau_1d.ndim != 1 or a_2d.shape[0] != tau_1d.size:
        raise ValueError("Coherent CIR gains and delays must share an explicit path axis")

    finite_tau = np.isfinite(tau_1d)
    finite_a = np.all(np.isfinite(a_2d), axis=1)
    keep_mask = finite_tau & finite_a

    if np.all(keep_mask):
        return a_2d, tau_1d, keep_mask

    dropped = int((~keep_mask).sum())
    logger.warning(
        "Dropping %d/%d non-finite coherent CIR paths at frame %d",
        dropped,
        int(keep_mask.size),
        int(frame_idx),
    )
    if a_raw is not None and tau_raw is not None:
        _log_nonfinite_cir_path_details(
            a_raw=a_raw,
            tau_raw=tau_raw,
            keep_mask=keep_mask,
            frame_idx=frame_idx,
            paths=paths,
            simulation_objects=simulation_objects,
        )
    return a_2d[keep_mask], tau_1d[keep_mask], keep_mask


def _extract_link_tau(tau_raw: np.ndarray, num_paths: int) -> np.ndarray:
    """Return the first link's path delays as a 1-D array."""
    tau_arr = np.asarray(tau_raw, dtype=np.float64)
    if tau_arr.ndim == 0:
        return tau_arr.reshape(1)
    if tau_arr.shape[-1] == num_paths:
        return tau_arr.reshape(-1, num_paths)[0].copy()
    flat = tau_arr.reshape(-1)
    return flat[:num_paths].copy()


def _cfr_shared_delays(tau_raw: np.ndarray, num_paths: int, elements: int) -> np.ndarray:
    """Admit declared shared-delay synthetic arrays, never select element zero."""
    delays = np.asarray(tau_raw, dtype=np.float64)
    if delays.shape == (num_paths,):
        return delays
    if delays.shape == (1, 1, num_paths):
        return delays[0, 0]
    if elements == 1 and delays.shape == (1, 1, 1, 1, num_paths):
        return delays[0, 0, 0, 0]
    raise ValueError(
        "cfr_ifft requires shared absolute path delays [path] or [rx=1,tx=1,path]; "
        f"element-dependent native array delays are unsupported (received {delays.shape})"
    )


def _prepare_rx_beamforming(
    simulation_objects: Any,
    a_raw: np.ndarray,
    tau_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Extract per-RX-antenna CIR for spatial beamforming.

    Supports both ULA (1 row, N cols) and UPA (M rows, N cols).

    Scope:
    - coherent CPI tensors shaped ``[rx, rx_ant, tx, tx_ant, paths, cpi]``
    - single RX / single TX / single TX antenna
    - multi-element RX array (``num_rx_ant > 1``)
    """
    a_arr = np.asarray(a_raw)
    if a_arr.ndim != 6:
        return None

    num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_cpi = a_arr.shape
    if num_rx != 1 or num_tx != 1 or num_tx_ant != 1 or num_rx_ant <= 1:
        return None

    # Get array dimensions from our AntennaConfig (Sionna PlanarArray
    # does not expose num_rows/num_cols as attributes after construction).
    sim_config = getattr(simulation_objects, "simulation_config", None)
    rx_antenna_cfg = getattr(sim_config, "rx_antenna", None) if sim_config else None
    if rx_antenna_cfg is None:
        # Multi-element interpretation requires the authored planar-array
        # topology.  Inferring a half-wavelength ULA from tensor shape would
        # assign unpersisted spatial coordinates and combining semantics.
        return None

    try:
        num_rows = int(rx_antenna_cfg.num_rows)
        num_cols = int(rx_antenna_cfg.num_cols)
        horizontal_spacing = float(rx_antenna_cfg.horizontal_spacing)
        vertical_spacing = float(rx_antenna_cfg.vertical_spacing)
        element_pattern = str(rx_antenna_cfg.pattern)
        polarization = str(rx_antenna_cfg.polarization)
    except (TypeError, ValueError):
        return None

    if num_rows * num_cols != num_rx_ant:
        return None

    tau_link = _extract_link_tau(tau_raw, num_paths)
    if tau_link.size != num_paths:
        return None

    a_link = np.asarray(a_arr[0, :, 0, 0, :, :], dtype=np.complex128)
    return (
        a_link,
        tau_link,
        {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "horizontal_spacing_lambda": horizontal_spacing,
            "vertical_spacing_lambda": vertical_spacing,
            "element_pattern": element_pattern,
            "polarization": polarization,
            "num_paths": num_paths,
            "num_cpi": num_cpi,
        },
    )


def process_coherent_cpi_fft(
    processor: "CoherentCpiFftProcessor",
    *,
    simulation_objects: Any,
    paths: Any,
    cir_stack: Tuple[np.ndarray, np.ndarray],
    frame_idx: int,
    reference_time_s: float,
) -> Optional[Dict[str, Any]]:
    """Process a complete coherent CIR stack, returning its sensing payload.

    ``cir_stack`` contains complex path gains and delays in seconds. Gains
    use ``[path, time]`` for SISO or Sionna's
    ``[rx, rx_ant, tx, tx_ant, path, time]`` order. ``reference_time_s`` is
    the first sample's scenario time. Returns ``None`` when sensing is disabled.
    """
    if not processor.config.enabled:
        return None

    start_time = time.time()
    profiler = SensingStageProfiler()

    with profiler.stage("cir_extract"):
        a_raw, tau_raw = cir_stack

    expected_cpi = processor.execution_config.acquisition.cpi_length_samples
    expected_rx_elements = processor.execution_config.topology.rx_elements
    cfr_mode = processor.config.range_response.model == "cfr_ifft"
    if a_raw.ndim == 6:
        expected_prefix = (1, expected_rx_elements, 1, 1)
        if tuple(a_raw.shape[:4]) != expected_prefix:
            raise ValueError(
                "coherent_cpi CIR topology disagrees with the resolved contract: "
                f"expected prefix {expected_prefix}, got {tuple(a_raw.shape[:4])}"
            )
    elif a_raw.ndim != 2 or expected_rx_elements != 1:
        raise ValueError(
            "coherent_cpi requires an explicit SISO [path,time] or Sionna "
            "[rx,rx_ant,tx,tx_ant,path,time] CIR tensor"
        )
    if a_raw.shape[-1] != expected_cpi:
        raise ValueError(
            "Coherent CIR sample count disagrees with the resolved CPI length: "
            f"expected {expected_cpi}, got {a_raw.shape[-1]}"
        )

    if cfr_mode:
        tau_raw = _cfr_shared_delays(tau_raw, a_raw.shape[-2], expected_rx_elements)
        finite_coefficients = np.all(
            np.isfinite(a_raw), axis=(0, 1, 2, 3, 5) if a_raw.ndim == 6 else 1
        )
        nonfinite_before_selection = int(
            np.count_nonzero(~np.isfinite(tau_raw) | ~finite_coefficients)
        )

    beamforming_ctx: dict[str, Any] | None = None

    # Flatten to (N_paths, N_cpi) and (N_paths,)
    with profiler.stage("flatten_cir"):
        prepared_beamforming = _prepare_rx_beamforming(simulation_objects, a_raw, tau_raw)
        if expected_rx_elements > 1 and prepared_beamforming is None:
            raise ValueError(
                "coherent_cpi could not map the declared RX array elements to an "
                "explicit beamforming tensor; implicit flattening is forbidden"
            )
        if prepared_beamforming is not None:
            a_ant, tau_1d, beamforming_ctx = prepared_beamforming
            n_cpi = a_ant.shape[-1]
            num_rx_ant = a_ant.shape[0]
            a_path_major = np.transpose(a_ant, (1, 0, 2)).reshape(a_ant.shape[1], -1)
            a_path_major, tau_1d, path_mask = _drop_nonfinite_cir_paths(
                a_path_major,
                tau_1d,
                frame_idx=frame_idx,
                a_raw=a_raw,
                tau_raw=tau_raw,
                paths=paths,
                simulation_objects=simulation_objects,
            )
            a_path_major, tau_1d = apply_aoa_filter(
                a_path_major, tau_1d, paths, processor.config, frame_idx, path_mask=path_mask
            )
            a_ant = np.transpose(
                a_path_major.reshape(-1, num_rx_ant, n_cpi),
                (1, 0, 2),
            )
            # The primary broadside view is an explicitly unitary coherent
            # voltage sum.  Equal-phase array gain therefore scales by sqrt(E)
            # in voltage rather than by the accidental unnormalized E factor.
            a_2d = a_ant.sum(axis=0) / np.sqrt(num_rx_ant)
            beamforming_ctx["num_rx_ant"] = int(num_rx_ant)
        elif a_raw.ndim > 2:
            n_cpi = a_raw.shape[-1]
            a_2d = a_raw.reshape(-1, n_cpi)
            tau_1d = tau_raw.flatten()
            if tau_1d.size != a_2d.shape[0]:
                raise ValueError(
                    "coherent_cpi CIR delay/path axes disagree with the explicit "
                    f"SISO tensor: {tau_1d.size} delays for {a_2d.shape[0]} path rows"
                )
        else:
            a_2d = a_raw
            tau_1d = tau_raw.flatten()
            n_cpi = a_2d.shape[-1] if a_2d.ndim == 2 else 1

        if not np.iscomplexobj(a_2d):
            a_2d = a_2d.astype(np.complex128)

        if beamforming_ctx is None:
            a_2d, tau_1d, path_mask = _drop_nonfinite_cir_paths(
                a_2d,
                tau_1d,
                frame_idx=frame_idx,
                a_raw=a_raw,
                tau_raw=tau_raw,
                paths=paths,
                simulation_objects=simulation_objects,
            )

            a_2d, tau_1d = apply_aoa_filter(
                a_2d, tau_1d, paths, processor.config, frame_idx, path_mask=path_mask
            )

    c0 = C0
    range_factor = 2.0 if processor.range_mode == "monostatic" else 1.0
    fft_r = processor.config.fft_size_range
    fft_d = processor.execution_config.processing.fft_size_doppler
    fs = processor.config.bandwidth
    prf = processor.config.prf_hz

    # Build slow-time CIR matrix: (N_cpi, fft_r)
    with profiler.stage("cir_projection"):
        if cfr_mode:
            element_coefficients = a_ant if beamforming_ctx is not None else a_2d[np.newaxis, ...]
            sample_times = reference_time_s + np.arange(n_cpi) / prf
            with profiler.stage("frequency_projection"):
                cfr_observation, cfr_diagnostics = paths_to_frequency(
                    element_coefficients,
                    tau_1d,
                    tau_1d >= 0,
                    sample_times,
                    bandwidth=fs,
                    config=processor.config.range_response,
                    realization_index=frame_idx,
                )
            with profiler.stage("range_ifft"):
                slow_time_matrix_ant, cfr_axes, cfr_metadata = frequency_to_delay(
                    cfr_observation,
                    bandwidth=fs,
                    config=processor.config.range_response,
                    fft_size=fft_r,
                    range_mode=processor.range_mode,
                )
            cfr_metadata["carrier_frequency_hz"] = processor.carrier_frequency_hz
            cfr_diagnostics["discarded_nonfinite_paths_before_aoa_selection"] = (
                nonfinite_before_selection
            )
            slow_time_matrix = slow_time_matrix_ant.sum(axis=0) / np.sqrt(expected_rx_elements)
        elif beamforming_ctx is not None:
            slow_time_matrix_ant = cir_to_samples_per_antenna(
                a_ant,
                tau_1d,
                sampling_frequency=fs,
                fft_size=fft_r,
            )
            slow_time_matrix = np.sum(slow_time_matrix_ant, axis=0) / np.sqrt(
                slow_time_matrix_ant.shape[0]
            )
        else:
            slow_time_matrix = cir_to_samples_batch(
                a_2d,
                tau_1d,
                sampling_frequency=fs,
                fft_size=fft_r,
            )

    noise_injection_applications: list[dict[str, Any]] = []
    if processor.noise_enabled and processor.noise_snr_db is not None:
        if beamforming_ctx is not None:
            noise_result = inject_target_relative_complex_awgn(
                slow_time_matrix_ant,
                processor.noise_snr_db,
                rng=processor._noise_rng,
            )
            slow_time_matrix_ant = noise_result.samples
            noise_injection_applications.append(
                {**noise_result.metadata(), "stage": "per_rx_element_slow_time"}
            )
            slow_time_matrix = np.sum(slow_time_matrix_ant, axis=0) / np.sqrt(
                slow_time_matrix_ant.shape[0]
            )
        else:
            noise_result = inject_target_relative_complex_awgn(
                slow_time_matrix, processor.noise_snr_db, rng=processor._noise_rng
            )
            slow_time_matrix = noise_result.samples
            noise_injection_applications.append(
                {**noise_result.metadata(), "stage": "combined_slow_time"}
            )

    aoa_az_map: np.ndarray | None = None
    aoa_el_map: np.ndarray | None = None
    azimuth_cube: np.ndarray | None = None
    azimuth_axis_deg: np.ndarray | None = None
    elevation_axis_deg: np.ndarray | None = None
    angle_cube: np.ndarray | None = None
    azimuth_direction_cosine: np.ndarray | None = None
    elevation_direction_cosine: np.ndarray | None = None
    with profiler.stage("rd_fft"):
        delay_data = slow_time_matrix[-fft_d:]
        delay_taper = sensing_window(processor.config.window_range, fft_r)
        delay_taper_correction = window_energy_correction(delay_taper)
        delay_data = delay_data * delay_taper[np.newaxis, :] * delay_taper_correction

        range_axis = (
            cfr_axes["range_axis_m"] if cfr_mode else (np.arange(fft_r) / fs) * c0 / range_factor
        )
        doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_d, d=1.0 / prf))
        range_axis_label = (
            "Range (m)" if processor.range_mode == "monostatic" else "Path Length (m)"
        )

        doppler_taper = sensing_window(processor.config.window_doppler, fft_d)
        if cfr_mode and not np.any(doppler_taper):
            raise ValueError(
                "Doppler window has zero energy; use boxcar or at least three Hann samples"
            )
        doppler_taper_correction = window_energy_correction(doppler_taper)
        rd_raw_input = delay_data * doppler_taper[:, np.newaxis] * doppler_taper_correction
        rd_complex_raw = np.fft.fftshift(
            np.fft.fft(rd_raw_input, n=fft_d, axis=0), axes=0
        ) / np.sqrt(fft_d)
        windowed_mean = processor.config.coherent_clutter_mode == "windowed_mean"
        mean_disabled = processor.config.coherent_clutter_mode == "none"
        rd_metadata = {
            "delay_projection": DELAY_PROJECTION_POINT_BIN_FLOOR,
            "range_processing": RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
            "range_fft_applied": False,
            "delay_taper": str(processor.config.window_range),
            "doppler_taper": str(processor.config.window_doppler),
            "fft_axis": "slow_time",
            "fft_normalization": DOPPLER_FFT_NORMALIZATION_UNITARY,
            "window_normalization": WINDOW_NORMALIZATION_ENERGY,
            **coherent_clutter_metadata(processor.config.coherent_clutter_mode),
        }
        if cfr_mode:
            rd_metadata.update(cfr_metadata)
        rd_raw_metadata = {
            **rd_metadata,
            "slow_time_mean_removal_applied": False,
        }
        rd_result_raw = RangeDopplerResult(
            power_linear=np.abs(rd_complex_raw) ** 2,
            range_axis_m=range_axis,
            doppler_axis_hz=doppler_axis,
            cube=rd_complex_raw,
            range_axis_label=range_axis_label,
            doppler_axis_label="Doppler (Hz)",
            metadata=rd_raw_metadata,
        )

        # The legacy policy preserves identical rows for compatibility.
        # They can represent static clutter; windowed_mean removes them.
        row_variation = np.max(np.abs(np.diff(delay_data, axis=0)))
        mean_removal_applied = not mean_disabled and (windowed_mean or bool(row_variation > 0))
        if mean_removal_applied:
            rd_clean_input = (
                remove_window_weighted_mean(delay_data, doppler_taper, axis=0)
                if windowed_mean
                else delay_data - delay_data.mean(axis=0, keepdims=True)
            )
            rd_clean_input = (
                rd_clean_input * doppler_taper[:, np.newaxis] * doppler_taper_correction
            )
            rd_complex = np.fft.fftshift(
                np.fft.fft(rd_clean_input, n=fft_d, axis=0), axes=0
            ) / np.sqrt(fft_d)
        else:
            logger.debug("Coherent mean removal disabled or legacy identical-PRI bypass")
            rd_complex = rd_complex_raw
        rd_primary_metadata = {
            **rd_metadata,
            "slow_time_mean_removal_applied": mean_removal_applied,
        }
        rd_power = np.abs(rd_complex) ** 2
        rd_result = RangeDopplerResult(
            power_linear=rd_power,
            range_axis_m=range_axis,
            doppler_axis_hz=doppler_axis,
            cube=rd_complex,
            range_axis_label=range_axis_label,
            doppler_axis_label="Doppler (Hz)",
            metadata=rd_primary_metadata,
        )

        if beamforming_ctx is not None:
            ant_delay_data = (
                slow_time_matrix_ant[:, -fft_d:, :]
                * delay_taper[np.newaxis, np.newaxis, :]
                * delay_taper_correction
            )
            rd_ant_raw_input = (
                ant_delay_data * doppler_taper[np.newaxis, :, np.newaxis] * doppler_taper_correction
            )
            rd_ant_complex_raw = np.fft.fftshift(
                np.fft.fft(rd_ant_raw_input, n=fft_d, axis=1),
                axes=1,
            ) / np.sqrt(fft_d)
            if mean_removal_applied:
                rd_ant_clean_input = (
                    remove_window_weighted_mean(ant_delay_data, doppler_taper, axis=1)
                    if windowed_mean
                    else ant_delay_data - ant_delay_data.mean(axis=1, keepdims=True)
                )
                rd_ant_clean_input = (
                    rd_ant_clean_input
                    * doppler_taper[np.newaxis, :, np.newaxis]
                    * doppler_taper_correction
                )
                rd_ant_complex = np.fft.fftshift(
                    np.fft.fft(rd_ant_clean_input, n=fft_d, axis=1),
                    axes=1,
                ) / np.sqrt(fft_d)
            else:
                rd_ant_complex = rd_ant_complex_raw

            bf_rows = beamforming_ctx["num_rows"]
            bf_cols = beamforming_ctx["num_cols"]

            spatial = planar_spatial_fft(
                rd_ant_complex,
                num_rows=bf_rows,
                num_cols=bf_cols,
                vertical_spacing_wavelengths=beamforming_ctx["vertical_spacing_lambda"],
                horizontal_spacing_wavelengths=beamforming_ctx["horizontal_spacing_lambda"],
            )
            angle_cube = np.abs(spatial.spectrum) ** 2
            azimuth_direction_cosine = spatial.azimuth_direction_cosine
            elevation_direction_cosine = spatial.elevation_direction_cosine
            azimuth_axis_deg = spatial.broadside_azimuth_deg
            elevation_axis_deg = spatial.broadside_elevation_deg
            beamforming_ctx["grating_lobes_possible"] = spatial.grating_lobes_possible

            # Measurement angles use one joint physical peak. The independent
            # broadside projection axes remain available for display/storage.
            aoa_az_map, joint_el_map = strongest_array_directions(
                angle_cube, elevation_direction_cosine, azimuth_direction_cosine
            )
            aoa_az_map[rd_power <= 0.0] = np.nan
            aoa_el_map: np.ndarray | None = None
            if bf_rows > 1:
                aoa_el_map = joint_el_map
                aoa_el_map[rd_power <= 0.0] = np.nan

            # For storage: collapse to 3D azimuth cube (backward compat)
            azimuth_cube = angle_cube.max(axis=0)

    # Build sensing_data
    beamforming_extra: dict[str, Any] = {
        "cir_time_steps": n_cpi,
        "rd_ready": True,
        **rd_primary_metadata,
    }
    if beamforming_ctx is not None and azimuth_axis_deg is not None:
        bf_mode = "rx_az_el_fft" if beamforming_ctx["num_rows"] > 1 else "rx_azimuth_fft"
        beamforming_extra.update(
            {
                "beamforming_mode": bf_mode,
                "antenna_rows": int(beamforming_ctx["num_rows"]),
                "antenna_cols": int(beamforming_ctx["num_cols"]),
                "azimuth_axis_deg": azimuth_axis_deg.astype(np.float32).tolist(),
                "rx_element_order": PLANAR_ELEMENT_ORDER,
                "spatial_fft_normalization": SPATIAL_FFT_NORMALIZATION,
                "spatial_phase_convention": SPATIAL_PHASE_CONVENTION,
                "rx_combine_policy": "rx_broadside_unitary_voltage_sum",
                "grating_lobes_possible": bool(
                    beamforming_ctx.get("grating_lobes_possible", False)
                ),
            }
        )
        if elevation_axis_deg is not None and beamforming_ctx["num_rows"] > 1:
            beamforming_extra["elevation_axis_deg"] = elevation_axis_deg.astype(np.float32).tolist()
    sensing_data = processor._build_sensing_metadata(
        frame_idx=frame_idx,
        processing_strategy="coherent_cpi_fft",
        rd_source="fft_cpi",
        rd_model="cfr_ifft_doppler_fft" if cfr_mode else "point_binned_cir_doppler_fft",
        frame_role="cpi_end",
        fft_d=fft_d,
        n_history=n_cpi,
        raw_payload_policy=processor.raw_payload_policy,
        extra=beamforming_extra,
    )
    if cfr_mode:
        sensing_data.update(cfr_axes)
        sensing_data["range_response"] = {**cfr_metadata, **cfr_diagnostics}
        for key in ("config", "config_resolved"):
            sensing_data[key]["range_sample_spacing_m"] = cfr_metadata["range_sample_spacing_m"]
            sensing_data[key]["range_resolution_scale_m"] = cfr_metadata["range_resolution_scale_m"]
            sensing_data[key]["range_response_processing"] = sensing_data["range_response"]
        if processor._should_emit_raw_payload("cfr_by_pair_rx_element"):
            sensing_data["cfr_by_pair_rx_element"] = cfr_observation[np.newaxis, ...]
    if processor.config.cfar_max_dynamic_range_db is not None and beamforming_ctx is not None:
        raw_spatial = planar_spatial_fft(
            rd_ant_complex_raw,
            num_rows=beamforming_ctx["num_rows"],
            num_cols=beamforming_ctx["num_cols"],
            vertical_spacing_wavelengths=beamforming_ctx["vertical_spacing_lambda"],
            horizontal_spacing_wavelengths=beamforming_ctx["horizontal_spacing_lambda"],
        )
        raw_beams, raw_azimuth, raw_elevation = visible_array_beams(
            abs(raw_spatial.spectrum) ** 2,
            raw_spatial.elevation_direction_cosine,
            raw_spatial.azimuth_direction_cosine,
        )
        # Reference the received signal before clutter suppression: a nearly
        # empty residual must not normalize its own rounding errors into hits.
        sensing_data["detector_power_reference"] = {
            "basis": "pre_clutter_spatial_fft_peak",
            "azimuth_deg": raw_azimuth.tolist(),
            "elevation_deg": raw_elevation.tolist(),
            "power_linear": raw_beams.max(axis=(1, 2)).tolist(),
        }
    if noise_injection_applications:
        sensing_data["noise_injection"] = {
            "contract_version": 1,
            "applications": noise_injection_applications,
        }

    # RD maps
    persist_detection_stages = bool(
        processor.persist_detection_stage_artifacts and processor.config.output_detections
    )
    if processor.config.output_range_doppler:
        sensing_data["range_doppler_map"] = rd_result.cube
    if processor.config.output_range_doppler or persist_detection_stages:
        sensing_data["range_doppler_by_pair"] = rd_result.cube[np.newaxis, ...]
        sensing_data["range_axis_m"] = np.asarray(range_axis, dtype=np.float64)
        sensing_data["doppler_axis_hz"] = np.asarray(doppler_axis, dtype=np.float64)
    if processor.config.output_range_doppler or persist_detection_stages:
        if processor._should_emit_raw_payload("range_doppler_map_raw"):
            sensing_data["range_doppler_map_raw"] = rd_result_raw.cube
            sensing_data["range_doppler_raw_by_pair"] = rd_result_raw.cube[np.newaxis, ...]
        if aoa_az_map is not None:
            sensing_data["range_doppler_aoa_az_map"] = np.asarray(aoa_az_map, dtype=np.float32)
        if aoa_el_map is not None:
            sensing_data["range_doppler_aoa_el_map"] = np.asarray(aoa_el_map, dtype=np.float32)
        if azimuth_cube is not None and processor._should_emit_raw_payload(
            "range_doppler_azimuth_cube"
        ):
            sensing_data["range_doppler_azimuth_cube"] = np.asarray(azimuth_cube, dtype=np.float32)
        if angle_cube is not None and (
            processor.config.array_detection_mode == "beam_scan"
            or processor.recording_profile in {"maps", "reprocessing", "full"}
            or processor._should_emit_raw_payload("range_doppler_azimuth_cube")
        ):
            sensing_data["range_doppler_angle_by_pair"] = np.asarray(
                angle_cube[np.newaxis, ...], dtype=np.float32
            )
            sensing_data["azimuth_direction_cosine"] = np.asarray(
                azimuth_direction_cosine, dtype=np.float64
            )
            sensing_data["elevation_direction_cosine"] = np.asarray(
                elevation_direction_cosine, dtype=np.float64
            )
            sensing_data["azimuth_broadside_angle_deg"] = np.asarray(
                azimuth_axis_deg, dtype=np.float64
            )
            sensing_data["elevation_broadside_angle_deg"] = np.asarray(
                elevation_axis_deg, dtype=np.float64
            )
    sensing_data["rd_fresh"] = True

    # Range profile from last PRI
    if processor.config.output_range_profile:
        last_rp = np.abs(slow_time_matrix[-1] * delay_taper * delay_taper_correction) ** 2
        sensing_data["range_profile"] = last_rp
        sensing_data["range_profile_by_pair"] = last_rp[np.newaxis, ...]
        sensing_data.setdefault("range_axis_m", np.asarray(range_axis, dtype=np.float64))
        if processor._should_emit_raw_payload("range_profile_raw"):
            sensing_data["range_profile_raw"] = last_rp.copy()
            sensing_data["range_profile_raw_by_pair"] = last_rp[np.newaxis, ...].copy()
        sensing_data["range_profile_kind"] = "power_linear"
        sensing_data["rp_fresh"] = True

    # Store CIR and slow_time_matrix for micro-Doppler in visualizer/diagnostic
    if processor._should_emit_raw_payload("cir"):
        sensing_data["cir"] = slow_time_matrix[-1].copy()
        element_slow_time = (
            slow_time_matrix_ant
            if beamforming_ctx is not None
            else slow_time_matrix[np.newaxis, ...]
        )
        sensing_data["cir_by_pair_rx_element"] = element_slow_time[np.newaxis, :, -1, :].copy()
    if processor._should_emit_raw_payload("slow_time_matrix"):
        sensing_data["slow_time_matrix"] = slow_time_matrix.copy()
        element_slow_time = (
            slow_time_matrix_ant
            if beamforming_ctx is not None
            else slow_time_matrix[np.newaxis, ...]
        )
        sensing_data["slow_time_by_pair_rx_element"] = element_slow_time[np.newaxis, ...].copy()

    rows = int(beamforming_ctx["num_rows"]) if beamforming_ctx is not None else 1
    cols = int(beamforming_ctx["num_cols"]) if beamforming_ctx is not None else 1
    horizontal_spacing = (
        float(beamforming_ctx["horizontal_spacing_lambda"]) if beamforming_ctx is not None else 0.5
    )
    vertical_spacing = (
        float(beamforming_ctx["vertical_spacing_lambda"]) if beamforming_ctx is not None else 0.5
    )
    simulation_config = getattr(simulation_objects, "simulation_config", None)
    rx_antenna_config = (
        getattr(simulation_config, "rx_antenna", None) if simulation_config is not None else None
    )
    element_pattern = (
        str(beamforming_ctx["element_pattern"])
        if beamforming_ctx is not None
        else (str(rx_antenna_config.pattern) if rx_antenna_config is not None else "not_available")
    )
    polarization = (
        str(beamforming_ctx["polarization"])
        if beamforming_ctx is not None
        else (
            str(rx_antenna_config.polarization)
            if rx_antenna_config is not None
            else "not_available"
        )
    )
    sensing_data["rx_element_axis"] = [
        {"element_index": index, "row": row, "column": column}
        for index, (row, column) in enumerate(planar_element_coordinates(rows, cols))
    ]
    sensing_data["rx_array"] = {
        "rows": rows,
        "columns": cols,
        "element_count": rows * cols,
        "element_order": PLANAR_ELEMENT_ORDER,
        "horizontal_spacing_wavelengths": horizontal_spacing,
        "vertical_spacing_wavelengths": vertical_spacing,
        "local_horizontal_direction": "+y",
        "local_vertical_direction": "+z",
        "spatial_phase_convention": SPATIAL_PHASE_CONVENTION,
        "spatial_fft_normalization": SPATIAL_FFT_NORMALIZATION,
        "element_pattern": element_pattern,
        "polarization": polarization,
    }

    # Pre-CFAR Doppler guards operate on a copy so the stored RD map remains
    # unchanged. Edge guards stay on by default; the DC guard is explicit and
    # defaults off to avoid suppressing legitimate stationary targets.
    doppler_edge_guard_bins = int(getattr(processor.config, "pre_cfar_doppler_edge_guard_bins", 2))
    doppler_dc_guard_bins = int(getattr(processor.config, "pre_cfar_doppler_dc_guard_bins", 0))
    if doppler_edge_guard_bins > 0 or doppler_dc_guard_bins > 0:
        rd_power_guarded = apply_pre_cfar_doppler_guards(
            rd_power,
            edge_guard_bins=doppler_edge_guard_bins,
            dc_guard_bins=doppler_dc_guard_bins,
        )
        rd_result_guarded = RangeDopplerResult(
            power_linear=rd_power_guarded,
            range_axis_m=range_axis,
            doppler_axis_hz=doppler_axis,
            cube=rd_complex,
            range_axis_label=range_axis_label,
            doppler_axis_label="Doppler (Hz)",
            metadata=dict(rd_primary_metadata),
        )
    else:
        rd_result_guarded = rd_result

    # Use the same complex element RD samples that supplied the coarse beams.
    # Refinement is optional and runs after coarse pruning, before tracking.
    beam_angle_refiner = None
    if processor.config.beam_angle_oversampling > 1 and beamforming_ctx is not None:
        beam_angle_refiner = partial(
            refine_array_directions,
            element_rd=rd_ant_complex,
            num_rows=beamforming_ctx["num_rows"],
            num_cols=beamforming_ctx["num_cols"],
            horizontal_spacing_wavelengths=beamforming_ctx["horizontal_spacing_lambda"],
            vertical_spacing_wavelengths=beamforming_ctx["vertical_spacing_lambda"],
            oversampling=processor.config.beam_angle_oversampling,
        )

    # CFAR
    with profiler.stage("detection"):
        cfar_metadata = processor._run_2d_cfar(
            rd_result_guarded,
            sensing_data,
            update_time_s=reference_time_s,
            aoa_az_map=aoa_az_map,
            aoa_el_map=aoa_el_map,
            angular_power=angle_cube,
            azimuth_direction_cosine=azimuth_direction_cosine,
            elevation_direction_cosine=elevation_direction_cosine,
            beam_angle_refiner=beam_angle_refiner,
        )
    if cfar_metadata:
        cfar_metadata.setdefault("range_doppler", {})["pre_cfar_doppler_guards"] = {
            "edge_bins": doppler_edge_guard_bins,
            "dc_bins": doppler_dc_guard_bins,
        }
        sensing_data["cfar_metadata"] = cfar_metadata

    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    range_product_axis = dense_axis(
        "range",
        size=fft_r,
        unit="m",
        coordinate_key="range_axis_m",
        coordinate_semantics=(
            "monostatic_range" if processor.range_mode == "monostatic" else "bistatic_path_length"
        ),
        reference_frame="tx_target_rx_path",
    )
    range_doppler_axes = (
        pair_axis,
        dense_axis(
            "doppler_frequency",
            size=fft_d,
            unit="Hz",
            coordinate_key="doppler_axis_hz",
            coordinate_semantics="fftshifted_doppler_frequency",
        ),
        range_product_axis,
    )
    element_axis = catalog_axis(
        "rx_element",
        size=rows * cols,
        catalog_key="rx_element_axis",
        coordinate_semantics="sionna_rx_element_index",
        reference_frame="rx_array_local",
    )
    slow_time_axis = dense_axis(
        "slow_time",
        size=n_cpi,
        unit="s",
        coordinate_key="slow_time_axis_s",
        coordinate_semantics="scenario_clock_sample_time",
        reference_frame="scenario_clock",
    )
    combine = {
        "input_axis": "rx_element",
        "policy": "rx_broadside_unitary_voltage_sum",
        "normalization": "one_over_sqrt_element_count",
    }
    delay_processing = (
        cfr_metadata if cfr_mode else {"delay_projection": DELAY_PROJECTION_POINT_BIN_FLOOR}
    )
    if "cfr_by_pair_rx_element" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="cfr_by_pair_rx_element",
            kind="baseband_frequency_observation",
            axes=(
                pair_axis,
                element_axis,
                slow_time_axis,
                dense_axis(
                    "frequency",
                    size=cfr_observation.shape[-1],
                    unit="Hz",
                    coordinate_key="frequency_axis_hz",
                    coordinate_semantics="offset_from_carrier",
                    reference_frame="carrier_frequency_hz",
                ),
            ),
            representation="complex_voltage",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            combine=combine,
            processing={
                **cfr_metadata,
                **cfr_diagnostics,
                "window_applied": False,
                "noise_included": processor.config.range_response.frequency_noise.variance > 0,
            },
        )
    if "cir_by_pair_rx_element" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="cir_by_pair_rx_element",
            kind="delay_domain_cir",
            axes=(pair_axis, element_axis, range_product_axis),
            representation="complex_voltage",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
            combine=combine,
            processing=delay_processing,
        )
    if "slow_time_by_pair_rx_element" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="slow_time_by_pair_rx_element",
            kind="delay_domain_slow_time",
            axes=(pair_axis, element_axis, slow_time_axis, range_product_axis),
            representation="complex_voltage",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            source_data_keys=(
                ("cfr_by_pair_rx_element",)
                if "cfr_by_pair_rx_element" in sensing_data
                else ("cir_by_pair_rx_element",) if "cir_by_pair_rx_element" in sensing_data else ()
            ),
            combine=combine,
            processing=delay_processing,
        )
    if "range_profile_by_pair" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="range_profile_by_pair",
            kind="range_profile",
            axes=(pair_axis, range_product_axis),
            representation="power_linear",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
            source_data_keys=(
                ("cir_by_pair_rx_element",) if "cir_by_pair_rx_element" in sensing_data else ()
            ),
            combine=combine,
        )
    if "range_profile_raw_by_pair" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="range_profile_raw_by_pair",
            kind="range_profile",
            variant="raw",
            axes=(pair_axis, range_product_axis),
            representation="power_linear",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
            source_data_keys=(
                ("cir_by_pair_rx_element",) if "cir_by_pair_rx_element" in sensing_data else ()
            ),
            combine=combine,
        )
    if "range_doppler_by_pair" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="range_doppler_by_pair",
            kind="range_doppler",
            axes=range_doppler_axes,
            representation="complex_voltage",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            source_data_keys=(
                ("slow_time_by_pair_rx_element",)
                if "slow_time_by_pair_rx_element" in sensing_data
                else ()
            ),
            combine=combine,
            processing=rd_primary_metadata,
        )
    if "range_doppler_raw_by_pair" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="range_doppler_raw_by_pair",
            kind="range_doppler",
            variant="raw",
            axes=range_doppler_axes,
            representation="complex_voltage",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            source_data_keys=(
                ("slow_time_by_pair_rx_element",)
                if "slow_time_by_pair_rx_element" in sensing_data
                else ()
            ),
            combine=combine,
            processing=rd_raw_metadata,
        )
    processor._queue_detection_stage_product_specs(
        sensing_data,
        primary_data_key=(
            "range_doppler_angle_by_pair"
            if processor.config.array_detection_mode == "beam_scan"
            else "range_doppler_by_pair"
        ),
        axes=range_doppler_axes,
        input_processing={
            "source_projection": (
                "strongest_visible_beam_per_rd_cell"
                if processor.config.array_detection_mode == "beam_scan"
                else "squared_magnitude_of_complex_voltage"
            ),
            "pre_cfar_doppler_guard_policy": "zero_selected_doppler_rows",
            "pre_cfar_doppler_edge_guard_bins": doppler_edge_guard_bins,
            "pre_cfar_doppler_dc_guard_bins": doppler_dc_guard_bins,
            "pre_cfar_doppler_guard_fill_power": 0.0,
            "pre_cfar_doppler_dc_reference": "fftshifted_zero_doppler_bin",
        },
    )
    if "range_doppler_angle_by_pair" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="range_doppler_angle_by_pair",
            kind="range_doppler_angle",
            axes=(
                pair_axis,
                dense_axis(
                    "elevation_direction_cosine",
                    size=rows,
                    unit="1",
                    coordinate_key="elevation_direction_cosine",
                    alternate_coordinate_keys=("elevation_broadside_angle_deg",),
                    coordinate_semantics="rx_array_local_z_direction_cosine",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "azimuth_direction_cosine",
                    size=cols,
                    unit="1",
                    coordinate_key="azimuth_direction_cosine",
                    alternate_coordinate_keys=("azimuth_broadside_angle_deg",),
                    coordinate_semantics="rx_array_local_y_direction_cosine",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "doppler_frequency",
                    size=fft_d,
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                    coordinate_semantics="fftshifted_doppler_frequency",
                ),
                range_product_axis,
            ),
            representation="power_linear",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            source_data_keys=(
                ("slow_time_by_pair_rx_element",)
                if "slow_time_by_pair_rx_element" in sensing_data
                else ()
            ),
            processing={
                "spatial_fft_normalization": SPATIAL_FFT_NORMALIZATION,
                "spatial_phase_convention": SPATIAL_PHASE_CONVENTION,
                "grating_lobes_possible": bool(
                    beamforming_ctx.get("grating_lobes_possible", False)
                    if beamforming_ctx is not None
                    else False
                ),
            },
        )
    sensing_data["compatibility_views"] = {
        "range_doppler_map": "range_doppler_by_pair[pair=0]",
        "range_profile": "range_profile_by_pair[pair=0]",
        "cir": "unitary_combination_of_cir_by_pair_rx_element[pair=0]",
        "slow_time_matrix": ("unitary_combination_of_slow_time_by_pair_rx_element[pair=0]"),
    }

    if processor.recording_profile is not None:
        with profiler.stage("display_products"):
            processor._record_display_products(
                sensing_data,
                samples=slow_time_matrix,
                times_s=reference_time_s + np.arange(n_cpi) / prf,
                range_axis_m=range_axis,
                frequency_samples=(
                    cfr_observation.sum(axis=0) / np.sqrt(expected_rx_elements)
                    if cfr_mode and processor.recording_profile != "results"
                    else None
                ),
            )
        with profiler.stage("retention"):
            sensing_data = processor._finalize_recording(sensing_data)
    elapsed = (time.time() - start_time) * 1000.0
    profiler.attach(sensing_data, total_ms=elapsed)
    logger.debug("Coherent sensing processing took %.2f ms (%d CPI samples)", elapsed, n_cpi)
    return sensing_data
