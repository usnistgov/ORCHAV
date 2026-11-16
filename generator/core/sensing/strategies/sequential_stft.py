"""Accumulate independently solved pulses and apply a sliding Doppler FFT.

The range response is either point bins or finite-bandwidth CFR/IFFT. Both
preserve the solver's complex phase; neither synthesizes extra carrier/Doppler
phase. Cross-solve continuity needs qualification for the scattering model.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from generator.core.sensing.profiling import SensingStageProfiler
from generator.core.sensing.strategies._shared_helpers import compute_range_profile_block
from shared.logging import get_logger
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
    SLOW_TIME_MEAN_REMOVAL_POLICY_NONZERO_VARIATION,
    WINDOW_NORMALIZATION_ENERGY,
    RangeDopplerResult,
    apply_aoa_filter,
    cir_to_samples,
    coherent_clutter_metadata,
    extract_cir_link,
    inject_target_relative_complex_awgn,
    remove_window_weighted_mean,
    sensing_window,
    window_energy_correction,
)

if TYPE_CHECKING:
    from ..processor import SequentialStftProcessor

logger = get_logger(__name__)


def process_sequential_stft(
    processor: "SequentialStftProcessor",
    *,
    frame_idx: int,
    paths: Any,
    reference_time_s: float,
    source_time_s: float,
) -> Optional[Dict[str, Any]]:
    """Append one independently solved pulse and return available sensing products.

    A range profile can be emitted during warm-up; range-Doppler products need
    ``fft_size_doppler`` pulses and then use a sliding window. ``source_time_s``
    timestamps this pulse, while ``reference_time_s`` is the window midpoint,
    both on the scenario clock. Returns ``None`` when disabled or CIR
    extraction fails.
    """
    if not processor.config.enabled:
        return None

    start_time = time.time()
    profiler = SensingStageProfiler()
    cfr_mode = processor.config.range_response.model == "cfr_ifft"
    acquisition = processor.execution_config.acquisition
    result_due = acquisition.window_due(processor._source_sample_count + 1)

    # 1. Compute CIR from Paths
    try:
        with profiler.stage("cir_extract"):
            a_link, tau_link = extract_cir_link(paths)

            # AoA filtering (before CIR summation)
            if a_link.ndim == 1:
                a_link_2d = a_link.reshape(-1, 1)
                a_link_2d, tau_link = apply_aoa_filter(
                    a_link_2d, tau_link, paths, processor.config, frame_idx
                )
                a_link = a_link_2d.ravel()
            else:
                a_link, tau_link = apply_aoa_filter(
                    a_link, tau_link, paths, processor.config, frame_idx
                )

    except (ValueError, TypeError, IndexError, RuntimeError) as e:
        logger.error("Failed to compute CIR from paths: %s", e)
        return None

    # Build metadata
    fft_d = processor.config.fft_size_doppler
    sensing_data = processor._build_sensing_metadata(
        frame_idx=frame_idx,
        processing_strategy="sequential_stft",
        rd_source="fft_cpi",
        rd_model="cfr_ifft_doppler_fft" if cfr_mode else "point_binned_cir_doppler_fft",
        frame_role="accumulating",
        fft_d=fft_d,
        n_history=(
            min(fft_d, len(processor._fast_history_raw) + 1)
            if cfr_mode
            else len(processor._fast_history_raw) + 1
        ),
        raw_payload_policy=processor.raw_payload_policy,
    )
    if processor.clutter_enabled:
        for config_key in ("config", "config_resolved"):
            sensing_data[config_key]["clutter_removal_enabled"] = True
            sensing_data[config_key]["clutter_removal_window"] = processor.clutter_window

    clutter_metadata: Dict[str, Any] = {}
    cfar_metadata: Dict[str, Any] = {}

    # Point-bin profiles are built after sample-domain noise injection so the
    # profile, its CFAR mask, and retained CIR describe the same acquisition.
    rp_available = False

    # 3. CIR samples → slow-time accumulation
    with profiler.stage("cir_projection"):
        if cfr_mode:
            with profiler.stage("frequency_projection"):
                observation, diagnostics = paths_to_frequency(
                    np.asarray(a_link, dtype=np.complex128).reshape(1, -1, 1),
                    tau_link,
                    tau_link >= 0,
                    np.array([source_time_s]),
                    bandwidth=processor.config.bandwidth,
                    config=processor.config.range_response,
                    realization_index=frame_idx,
                )
            with profiler.stage("range_ifft"):
                delay, cfr_axes, cfr_metadata = frequency_to_delay(
                    observation,
                    bandwidth=processor.config.bandwidth,
                    config=processor.config.range_response,
                    fft_size=processor.config.fft_size_range,
                    range_mode=processor.range_mode,
                )
            cir_samples = delay[0, 0]
            processor._frequency_history.append(observation[0, 0])
            processor._frequency_noise_history.append(diagnostics["noise"])
            cfr_metadata["carrier_frequency_hz"] = processor.carrier_frequency_hz
            sensing_data.update(cfr_axes)
            sensing_data["range_response"] = {**cfr_metadata, **diagnostics}
            for key in ("config", "config_resolved"):
                # The execution recipe is stable across pulses. Per-pulse
                # path/noise diagnostics belong to the observation descriptor.
                sensing_data[key]["range_response_processing"] = dict(cfr_metadata)
                sensing_data[key]["range_sample_spacing_m"] = cfr_metadata["range_sample_spacing_m"]
                sensing_data[key]["range_resolution_scale_m"] = cfr_metadata[
                    "range_resolution_scale_m"
                ]
            if processor.config.output_range_profile:
                sensing_data["range_profile"] = abs(cir_samples) ** 2
                sensing_data["range_profile_kind"] = "power_linear"
                if processor._should_emit_raw_payload("range_profile_raw"):
                    sensing_data["range_profile_raw"] = sensing_data["range_profile"].copy()
                rp_available = True
        else:
            cir_samples = cir_to_samples(
                a_link,
                tau_link,
                sampling_frequency=processor.config.bandwidth,
                fft_size=processor.config.fft_size_range,
            )
    if processor.noise_enabled and processor.noise_snr_db is not None:
        noise_result = inject_target_relative_complex_awgn(
            cir_samples, processor.noise_snr_db, rng=processor._noise_rng
        )
        cir_samples = noise_result.samples
        sensing_data["noise_injection"] = {
            "contract_version": 1,
            "applications": [{**noise_result.metadata(), "stage": "per_frame_delay_domain_cir"}],
        }
    if not cfr_mode:
        rp_available = compute_range_profile_block(
            processor,
            a_link,
            tau_link,
            sensing_data,
            clutter_metadata,
            cfar_metadata,
            profiler,
            acquired_samples=cir_samples if processor.noise_enabled else None,
            compute_detections=acquisition.cpi_hop_samples == 1 or result_due,
        )
    if processor._should_emit_raw_payload("cir"):
        sensing_data["cir"] = cir_samples
        sensing_data["cir_by_pair_rx_element"] = cir_samples[np.newaxis, np.newaxis, :]
    processor._fast_history_raw.append(cir_samples)

    if processor._slow_time_clutter is not None:
        slow_time_result = processor._slow_time_clutter.process(cir_samples)
        processor._fast_history_clean.append(slow_time_result.cleaned)
        clutter_metadata["range_doppler"] = {
            "enabled": True,
            "window_frames": processor.clutter_window,
            "baseline_frames": slow_time_result.frames_used,
            "history_frames": len(processor._fast_history_clean),
            "method": "causal_previous_sample_mean_subtraction",
            **slow_time_result.processing,
        }
    else:
        processor._fast_history_clean.append(cir_samples.copy())

    # 4. Range-Doppler Map (when enough history accumulated)
    c0 = C0
    range_factor = 2.0 if processor.range_mode == "monostatic" else 1.0
    fft_r = processor.config.fft_size_range
    fs = processor.config.bandwidth
    prf = processor.config.prf_hz
    want_rd = processor.config.output_range_doppler or processor.config.output_detections
    history_ready = len(processor._fast_history_raw) >= fft_d
    range_axis = (
        cfr_axes["range_axis_m"] if cfr_mode else (np.arange(fft_r) / fs) * c0 / range_factor
    )
    sensing_data["range_axis_m"] = np.asarray(range_axis, dtype=np.float64)
    sensing_data["rx_element_axis"] = [{"element_index": 0, "row": 0, "column": 0}]
    if cfr_mode and processor._should_emit_raw_payload("cfr_by_pair_rx_element"):
        # A warm-up frame retains only its acquired pulse. A complete frame
        # retains the exact overlapping window, including its original noise.
        retained = np.asarray(processor._frequency_history) if history_ready else observation[0]
        sensing_data["cfr_by_pair_rx_element"] = retained[np.newaxis, np.newaxis, ...]
        sensing_data["observation_time_axis_s"] = np.array([source_time_s])

    rd_primary: Optional[RangeDopplerResult] = None
    rd_result_raw: Optional[RangeDopplerResult] = None

    # Declare the requested processing recipe before the first complete window.
    # Persisted accumulating CIR and completed STFT frames must retain exactly
    # the same resolved configuration for acquisition replay.
    rd_metadata = {
        "delay_projection": DELAY_PROJECTION_POINT_BIN_FLOOR,
        "range_processing": RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
        "range_fft_applied": False,
        "delay_taper": str(processor.config.window_range),
        "doppler_taper": str(processor.config.window_doppler),
        "fft_axis": "slow_time",
        "fft_normalization": DOPPLER_FFT_NORMALIZATION_UNITARY,
        "window_normalization": WINDOW_NORMALIZATION_ENERGY,
        "slow_time_mean_removal_policy": SLOW_TIME_MEAN_REMOVAL_POLICY_NONZERO_VARIATION,
        "slow_time_mean_removal_operator": "finite_cpi_mean_projection",
        "slow_time_mean_removal_transfer_model": "finite_record_projection_not_lti",
        "slow_time_mean_removal_tone_power_gain": "1-abs(mean(exp(j*2*pi*f*n)))**2",
        "slow_time_mean_removal_group_delay": "undefined_for_non_lti_projection",
        "slow_time_mean_removal_result_availability": "after_complete_cpi",
        "slow_time_mean_removal_decision_statistic": "max_abs_first_difference",
        "slow_time_mean_removal_decision_threshold": 0.0,
    }
    if cfr_mode:
        rd_metadata.update(cfr_metadata)
        rd_metadata.update(coherent_clutter_metadata(processor.config.coherent_clutter_mode))
        rd_metadata["coherent_clutter_mode"] = processor.config.coherent_clutter_mode
    if want_rd:
        sensing_data.update(rd_metadata)
        for config_key in ("config", "config_resolved"):
            config = sensing_data.get(config_key)
            if isinstance(config, dict):
                config.update(rd_metadata)

    if history_ready and not result_due:
        sensing_data["frame_role"] = "acquired_unprocessed"
        sensing_data["rd_ready"] = False
        sensing_data["rd_map_frames_used"] = 0

    if want_rd and result_due:
        sensing_data["frame_role"] = "stft_ready"

        with profiler.stage("rd_fft"):
            doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_d, d=1.0 / prf))
            range_axis_label = (
                "Range (m)" if processor.range_mode == "monostatic" else "Path Length (m)"
            )

            delay_taper = sensing_window(processor.config.window_range, fft_r)
            delay_taper_correction = window_energy_correction(delay_taper)
            doppler_taper = sensing_window(processor.config.window_doppler, fft_d)
            if cfr_mode and not np.any(doppler_taper):
                raise ValueError(
                    "Doppler window has zero energy; use boxcar or at least three Hann samples"
                )
            doppler_taper_correction = window_energy_correction(doppler_taper)
            fast_time_raw = np.array(list(processor._fast_history_raw)[-fft_d:])
            delay_data_raw = fast_time_raw * delay_taper[np.newaxis, :] * delay_taper_correction
            rd_raw_input = delay_data_raw * doppler_taper[:, np.newaxis] * doppler_taper_correction
            rd_complex_raw = np.fft.fftshift(
                np.fft.fft(rd_raw_input, n=fft_d, axis=0), axes=0
            ) / np.sqrt(fft_d)
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

            if processor.clutter_enabled:
                fast_time_clean = np.array(list(processor._fast_history_clean)[-fft_d:])
                delay_data_clean = (
                    fast_time_clean * delay_taper[np.newaxis, :] * delay_taper_correction
                )
            else:
                delay_data_clean = delay_data_raw

            row_variation = np.max(np.abs(np.diff(delay_data_clean, axis=0)))
            mean_removal_applied = bool(row_variation > 0)
            windowed_mean = cfr_mode and processor.config.coherent_clutter_mode == "windowed_mean"
            if cfr_mode:
                mean_removal_applied = processor.config.coherent_clutter_mode != "none" and (
                    windowed_mean or mean_removal_applied
                )
            if mean_removal_applied:
                rd_clean_input = (
                    remove_window_weighted_mean(delay_data_clean, doppler_taper, axis=0)
                    if windowed_mean
                    else delay_data_clean - delay_data_clean.mean(axis=0, keepdims=True)
                )
            else:
                rd_clean_input = delay_data_clean
            if mean_removal_applied or processor.clutter_enabled:
                # History subtraction remains part of the primary input even
                # when it leaves a constant (including all-zero) sequence.
                rd_clean_input = (
                    rd_clean_input * doppler_taper[:, np.newaxis] * doppler_taper_correction
                )
                rd_complex = np.fft.fftshift(
                    np.fft.fft(rd_clean_input, n=fft_d, axis=0), axes=0
                ) / np.sqrt(fft_d)
            else:
                rd_complex = rd_complex_raw

            rd_primary_metadata = {
                **rd_metadata,
                "slow_time_mean_removal_applied": mean_removal_applied,
            }
            rd_power = np.abs(rd_complex) ** 2
            rd_primary = RangeDopplerResult(
                power_linear=rd_power,
                range_axis_m=range_axis,
                doppler_axis_hz=doppler_axis,
                cube=rd_complex,
                range_axis_label=range_axis_label,
                doppler_axis_label="Doppler (Hz)",
                metadata=rd_primary_metadata,
            )

        logger.debug(
            "Sequential STFT RD map from %d accumulated pulses (frame %d)",
            fft_d,
            frame_idx,
        )
    elif want_rd:
        logger.debug(
            "Sequential STFT: accumulating history %d/%d",
            len(processor._fast_history_raw),
            fft_d,
        )

    # Emit RD maps
    if (
        rd_result_raw is not None
        and processor.config.output_range_doppler
        and processor._should_emit_raw_payload("range_doppler_map_raw")
    ):
        sensing_data["range_doppler_map_raw"] = rd_result_raw.cube
        sensing_data["range_doppler_raw_by_pair"] = rd_result_raw.cube[np.newaxis, ...]
    persist_detection_stages = bool(
        processor.persist_detection_stage_artifacts and processor.config.output_detections
    )
    if rd_primary is not None and processor.config.output_range_doppler:
        sensing_data["range_doppler_map"] = rd_primary.cube
    if rd_primary is not None and (
        processor.config.output_range_doppler or persist_detection_stages
    ):
        sensing_data["range_doppler_by_pair"] = rd_primary.cube[np.newaxis, ...]
        sensing_data["doppler_axis_hz"] = np.asarray(rd_primary.doppler_axis_hz, dtype=np.float64)
    if rd_primary is not None and processor.config.output_detections:
        with profiler.stage("detection"):
            cfar_metadata.update(
                processor._run_2d_cfar(
                    rd_primary,
                    sensing_data,
                    update_time_s=reference_time_s,
                )
            )

    # Store slow_time_matrix for micro-Doppler
    if processor._should_emit_raw_payload("slow_time_matrix") and (
        history_ready or processor.recording_profile is not None
    ):
        # Keep each actually acquired warm-up pulse once. Complete windows use
        # the same canonical key and existing overlapping acquisition support.
        sensing_data["slow_time_matrix"] = (
            np.array(list(processor._fast_history_raw)[-fft_d:]).copy()
            if history_ready
            else cir_samples[np.newaxis, :].copy()
        )
        if not history_ready:
            sensing_data["observation_time_axis_s"] = np.array([source_time_s])
        sensing_data["slow_time_by_pair_rx_element"] = sensing_data["slow_time_matrix"][
            np.newaxis, np.newaxis, ...
        ]

    if "range_profile" in sensing_data:
        sensing_data["range_profile_by_pair"] = np.asarray(sensing_data["range_profile"])[
            np.newaxis, ...
        ]
    if "range_profile_raw" in sensing_data:
        sensing_data["range_profile_raw_by_pair"] = np.asarray(sensing_data["range_profile_raw"])[
            np.newaxis, ...
        ]

    sensing_data["rd_fresh"] = rd_primary is not None
    sensing_data["rp_fresh"] = rp_available

    if clutter_metadata:
        sensing_data["clutter_metadata"] = clutter_metadata
    if cfar_metadata:
        sensing_data["cfar_metadata"] = cfar_metadata

    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    element_axis = catalog_axis(
        "rx_element",
        size=1,
        catalog_key="rx_element_axis",
        coordinate_semantics="sionna_rx_element_index",
        reference_frame="rx_array_local",
    )
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
    if "cfr_by_pair_rx_element" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="cfr_by_pair_rx_element",
            kind="baseband_frequency_observation",
            axes=(
                pair_axis,
                element_axis,
                dense_axis(
                    "slow_time",
                    size=fft_d if history_ready else 1,
                    unit="s",
                    coordinate_key=(
                        "slow_time_axis_s" if history_ready else "observation_time_axis_s"
                    ),
                    coordinate_semantics="scenario_clock_sample_time",
                    reference_frame="scenario_clock",
                ),
                dense_axis(
                    "frequency",
                    size=processor.config.range_response.num_frequency_bins,
                    unit="Hz",
                    coordinate_key="frequency_axis_hz",
                    coordinate_semantics="offset_from_carrier",
                    reference_frame="carrier_frequency_hz",
                ),
            ),
            representation="complex_voltage",
            unit="1",
            temporal_support=(
                TEMPORAL_SUPPORT_FULL_ACQUISITION
                if history_ready
                else TEMPORAL_SUPPORT_LAST_VALID_SAMPLE
            ),
            combine={"policy": "siso_identity", "normalization": "identity"},
            processing={
                **cfr_metadata,
                "window_applied": False,
                "noise_included": processor.config.range_response.frequency_noise.variance > 0,
                "noise": {
                    "domain": "unwindowed_frequency_observation",
                    "realization_policy": "once_per_independent_pulse",
                    "pulses": (
                        list(processor._frequency_noise_history)
                        if history_ready
                        else [diagnostics["noise"]]
                    ),
                },
                "phase_contract": processor.execution_config.acquisition.phase_contract,
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
            combine={"policy": "siso_identity", "normalization": "identity"},
            processing=(
                cfr_metadata if cfr_mode else {"delay_projection": DELAY_PROJECTION_POINT_BIN_FLOOR}
            ),
        )
    if "slow_time_by_pair_rx_element" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="slow_time_by_pair_rx_element",
            kind="delay_domain_slow_time",
            axes=(
                pair_axis,
                element_axis,
                dense_axis(
                    "slow_time",
                    size=fft_d if history_ready else 1,
                    unit="s",
                    coordinate_key=(
                        "slow_time_axis_s" if history_ready else "observation_time_axis_s"
                    ),
                    coordinate_semantics="scenario_clock_sample_time",
                    reference_frame="scenario_clock",
                ),
                range_product_axis,
            ),
            representation="complex_voltage",
            unit="1",
            temporal_support=(
                TEMPORAL_SUPPORT_FULL_ACQUISITION
                if history_ready
                else TEMPORAL_SUPPORT_LAST_VALID_SAMPLE
            ),
            source_data_keys=(
                ("cfr_by_pair_rx_element",)
                if cfr_mode and "cfr_by_pair_rx_element" in sensing_data
                else ()
            ),
            combine={"policy": "siso_identity", "normalization": "identity"},
            processing={
                **(cfr_metadata if cfr_mode else {}),
                "phase_contract": "independent_solve_sequence",
            },
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
            combine={"policy": "siso_identity", "normalization": "identity"},
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
            combine={"policy": "siso_identity", "normalization": "identity"},
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
            combine={"policy": "siso_identity", "normalization": "identity"},
            processing=rd_primary.metadata if rd_primary.metadata is not None else {},
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
            combine={"policy": "siso_identity", "normalization": "identity"},
            processing=rd_result_raw.metadata if rd_result_raw.metadata is not None else {},
        )
    processor._queue_detection_stage_product_specs(
        sensing_data,
        primary_data_key="range_doppler_by_pair",
        axes=range_doppler_axes,
        input_processing={
            "source_projection": "squared_magnitude_of_complex_voltage",
            "pre_cfar_doppler_guard_policy": "none",
        },
    )
    sensing_data["compatibility_views"] = {
        "range_doppler_map": "range_doppler_by_pair[pair=0]",
        "range_profile": "range_profile_by_pair[pair=0]",
        "cir": "cir_by_pair_rx_element[pair=0,rx_element=0]",
        "slow_time_matrix": "slow_time_by_pair_rx_element[pair=0,rx_element=0]",
    }

    if processor.recording_profile is not None:
        if not history_ready or result_due:
            with profiler.stage("display_products"):
                history = np.asarray(processor._fast_history_raw)
                times = np.asarray(
                    [*processor._source_time_history, source_time_s][-len(history) :]
                )
                processor._record_display_products(
                    sensing_data,
                    samples=history,
                    times_s=times,
                    range_axis_m=range_axis,
                    frequency_samples=(
                        np.asarray(processor._frequency_history) if cfr_mode else None
                    ),
                    sequential=True,
                )
        with profiler.stage("retention"):
            sensing_data = processor._finalize_recording(sensing_data)
    elapsed = (time.time() - start_time) * 1000.0
    profiler.attach(sensing_data, total_ms=elapsed)
    logger.debug("Sequential STFT sensing processing took %.2f ms", elapsed)
    return sensing_data
