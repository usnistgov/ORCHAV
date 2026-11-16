"""Diagnostic range-Doppler products from Sionna's per-path Doppler values.

Path powers are accumulated directly into range/Doppler bins. A zero-phase
complex magnitude surrogate supports storage and optional noise injection;
the analytic map does not represent a sampled waveform.
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
from shared.sensing.signal_processing import (
    C0,
    DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED,
    DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF,
    RangeDopplerResult,
    cir_to_samples,
    extract_cir_link,
    inject_target_relative_complex_awgn,
)

if TYPE_CHECKING:
    from ..processor import AnalyticPathDopplerOracleProcessor

logger = get_logger(__name__)


def _direct_doppler_clutter_cutoff_hz(
    processor: "AnalyticPathDopplerOracleProcessor",
    *,
    apply_clutter_filter: bool,
) -> float:
    """Return the exact direct path-Doppler rejection cutoff."""

    if not (processor.clutter_enabled and apply_clutter_filter):
        return 0.0
    return float(processor.config.prf_hz) / (2.0 * max(processor.clutter_window, 1))


def _direct_doppler_clutter_processing(
    processor: "AnalyticPathDopplerOracleProcessor",
    *,
    apply_clutter_filter: bool,
) -> dict[str, str | float]:
    """Describe the exact direct Doppler filter executed for one RD product."""

    cutoff_hz = _direct_doppler_clutter_cutoff_hz(
        processor,
        apply_clutter_filter=apply_clutter_filter,
    )
    policy = (
        DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF
        if cutoff_hz > 0.0
        else DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED
    )
    return {
        "direct_doppler_clutter_filter_policy": policy,
        "direct_doppler_clutter_cutoff_hz": cutoff_hz,
    }


def _power_map_to_cube(power_map: np.ndarray) -> np.ndarray:
    """Convert an analytic power map into a storage/noise surrogate RD cube."""
    magnitude = np.sqrt(np.maximum(np.asarray(power_map, dtype=np.float64), 0.0))
    return magnitude.astype(np.complex128)


def _build_direct_rd_map(
    processor: "AnalyticPathDopplerOracleProcessor",
    a_link: np.ndarray,
    tau_link: np.ndarray,
    doppler_link: np.ndarray,
    *,
    apply_clutter_filter: bool = True,
    aoa_az_deg: np.ndarray | None = None,
    aoa_el_deg: np.ndarray | None = None,
) -> tuple[RangeDopplerResult, np.ndarray | None, np.ndarray | None]:
    """Build range-Doppler map from per-path Doppler values.

    Returns:
        Tuple of (RangeDopplerResult, aoa_az_map, aoa_el_map).
        AoA maps are [doppler, range] power-weighted averages, or None.
    """
    c0 = C0
    fft_r = processor.config.fft_size_range
    fft_d = processor.config.fft_size_doppler
    fs = processor.config.bandwidth
    prf = processor.config.prf_hz
    range_factor = 2.0 if processor.range_mode == "monostatic" else 1.0

    range_axis = (np.arange(fft_r) / fs) * c0 / range_factor
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_d, d=1.0 / prf))

    rd_power = np.zeros((fft_d, fft_r), dtype=np.float64)
    path_power = np.abs(np.asarray(a_link, dtype=np.complex128)) ** 2
    delays = np.asarray(tau_link, dtype=np.float64)
    dopplers = np.asarray(doppler_link, dtype=np.float64)
    range_bin_spacing = c0 / (range_factor * fs)

    # AoA accumulation maps (power-weighted sum, divided later)
    has_aoa = aoa_az_deg is not None and aoa_el_deg is not None
    az_weighted_sum = np.zeros((fft_d, fft_r), dtype=np.float64) if has_aoa else None
    el_weighted_sum = np.zeros((fft_d, fft_r), dtype=np.float64) if has_aoa else None

    doppler_clutter_hz = _direct_doppler_clutter_cutoff_hz(
        processor,
        apply_clutter_filter=apply_clutter_filter,
    )

    valid = (delays > 0.0) & (path_power > 0.0)
    if doppler_clutter_hz > 0.0:
        valid &= np.abs(dopplers) >= doppler_clutter_hz

    if np.any(valid):
        path_ranges = delays[valid] * c0 / range_factor
        range_bins = np.rint(path_ranges / range_bin_spacing).astype(np.int64)
        in_range = (range_bins >= 0) & (range_bins < fft_r)

        if np.any(in_range):
            range_bins = range_bins[in_range]
            dopplers_valid = dopplers[valid][in_range]
            path_power_valid = path_power[valid][in_range]

            if fft_d > 1:
                doppler_step = float(doppler_axis[1] - doppler_axis[0])
                doppler_bins = np.rint((dopplers_valid - doppler_axis[0]) / doppler_step).astype(
                    np.int64
                )
                doppler_bins = np.clip(doppler_bins, 0, fft_d - 1)
            else:
                doppler_bins = np.zeros_like(range_bins)

            np.add.at(rd_power, (doppler_bins, range_bins), path_power_valid)

            if has_aoa:
                az_valid = np.asarray(aoa_az_deg)[valid][in_range]
                el_valid = np.asarray(aoa_el_deg)[valid][in_range]
                np.add.at(az_weighted_sum, (doppler_bins, range_bins), path_power_valid * az_valid)
                np.add.at(el_weighted_sum, (doppler_bins, range_bins), path_power_valid * el_valid)

    # Convert weighted sums to weighted averages
    aoa_az_map: np.ndarray | None = None
    aoa_el_map: np.ndarray | None = None
    if has_aoa:
        nonzero = rd_power > 0
        aoa_az_map = np.zeros_like(rd_power)
        aoa_el_map = np.zeros_like(rd_power)
        aoa_az_map[nonzero] = az_weighted_sum[nonzero] / rd_power[nonzero]
        aoa_el_map[nonzero] = el_weighted_sum[nonzero] / rd_power[nonzero]

    range_label = "Range (m)" if processor.range_mode == "monostatic" else "Path Length (m)"
    rd_result = RangeDopplerResult(
        power_linear=rd_power,
        range_axis_m=range_axis,
        doppler_axis_hz=doppler_axis,
        cube=_power_map_to_cube(rd_power),
        range_axis_label=range_label,
        doppler_axis_label="Doppler (Hz)",
    )
    return rd_result, aoa_az_map, aoa_el_map


def process_analytic_path_doppler_oracle(
    processor: "AnalyticPathDopplerOracleProcessor",
    *,
    frame_idx: int,
    paths: Any,
    reference_time_s: float,
) -> Optional[Dict[str, Any]]:
    """Build oracle products for one path snapshot at ``reference_time_s``.

    Updates the processor's diagnostic history and optional detection state.
    Returns ``None`` when disabled or CIR extraction yields no usable paths;
    valid per-path Doppler is required once CIR extraction succeeds.
    """
    if not processor.config.enabled:
        return None

    start_time = time.time()
    profiler = SensingStageProfiler()

    # 1. Compute CIR from Paths
    try:
        with profiler.stage("cir_extract"):
            a_link, tau_link = extract_cir_link(paths)
            if a_link.size == 0 or tau_link.size == 0:
                logger.debug("Analytic path-Doppler oracle skipped a no-path scene")
                return None
            if a_link.size != tau_link.size:
                raise ValueError(
                    "analytic_path_doppler_oracle requires one complex gain per path delay"
                )
            processor._cir_history.append((a_link, tau_link))

            doppler_link: Optional[np.ndarray] = None
            raw_doppler = getattr(paths, "doppler", None)
            if raw_doppler is not None:
                if hasattr(raw_doppler, "numpy"):
                    raw_doppler = raw_doppler.numpy()
                raw_doppler = np.asarray(raw_doppler).flatten()
                if raw_doppler.size == tau_link.size and np.all(np.isfinite(raw_doppler)):
                    doppler_link = raw_doppler

    except (ValueError, TypeError, IndexError, RuntimeError) as e:
        logger.error("Failed to compute CIR from paths: %s", e)
        return None

    # Extract per-path AoA if available (for per-cluster AoA estimation)
    path_aoa_az: Optional[np.ndarray] = None
    path_aoa_el: Optional[np.ndarray] = None
    phi_r = getattr(paths, "phi_r", None)
    theta_r = getattr(paths, "theta_r", None)
    if phi_r is not None and theta_r is not None:
        phi_np = np.asarray(phi_r.numpy() if hasattr(phi_r, "numpy") else phi_r).flatten()
        theta_np = np.asarray(theta_r.numpy() if hasattr(theta_r, "numpy") else theta_r).flatten()
        if phi_np.size == tau_link.size:
            path_aoa_az = (np.degrees(phi_np) + 180.0) % 360.0 - 180.0
            path_aoa_el = 90.0 - np.degrees(theta_np)

    # Build metadata
    want_rd = processor.config.output_range_doppler or processor.config.output_detections
    use_direct_rd = doppler_link is not None
    if not use_direct_rd:
        raise RuntimeError(
            "analytic_path_doppler_oracle requires valid Sionna paths.doppler values; "
            "a waveform-derived fallback is intentionally unavailable"
        )
    processing_strategy = "analytic_path_doppler_oracle"
    rd_source = "path_doppler"
    rd_model = "analytic_path_power_oracle_nonwaveform"
    sensing_data = processor._build_sensing_metadata(
        frame_idx=frame_idx,
        processing_strategy=processing_strategy,
        rd_source=rd_source,
        rd_model=rd_model,
        frame_role="analytic_oracle_ready",
        fft_d=processor.config.fft_size_doppler,
        n_history=len(processor._cir_history),
        raw_payload_policy=processor.raw_payload_policy,
    )
    if processor.clutter_enabled:
        for config_key in ("config", "config_resolved"):
            sensing_data[config_key]["clutter_removal_enabled"] = True
            sensing_data[config_key]["clutter_removal_window"] = processor.clutter_window

    clutter_metadata: Dict[str, Any] = {}
    cfar_metadata: Dict[str, Any] = {}
    noise_injection_applications: list[dict[str, Any]] = []

    # 2. Range Profile
    rp_available = compute_range_profile_block(
        processor, a_link, tau_link, sensing_data, clutter_metadata, cfar_metadata, profiler
    )

    # CIR samples for micro-Doppler
    with profiler.stage("cir_projection"):
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
        noise_injection_applications.append(
            {**noise_result.metadata(), "stage": "per_frame_delay_domain_cir"}
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

    # 3. Range-Doppler Map
    rd_primary: Optional[RangeDopplerResult] = None
    rd_result_raw: Optional[RangeDopplerResult] = None

    aoa_az_map: Optional[np.ndarray] = None
    aoa_el_map: Optional[np.ndarray] = None

    if use_direct_rd and want_rd:
        with profiler.stage("direct_rd"):
            rd_result_raw, _, _ = _build_direct_rd_map(
                processor, a_link, tau_link, doppler_link, apply_clutter_filter=False
            )
            if processor.clutter_enabled:
                rd_primary, aoa_az_map, aoa_el_map = _build_direct_rd_map(
                    processor,
                    a_link,
                    tau_link,
                    doppler_link,
                    apply_clutter_filter=True,
                    aoa_az_deg=path_aoa_az,
                    aoa_el_deg=path_aoa_el,
                )
            else:
                rd_primary = rd_result_raw
                if path_aoa_az is not None:
                    _, aoa_az_map, aoa_el_map = _build_direct_rd_map(
                        processor,
                        a_link,
                        tau_link,
                        doppler_link,
                        apply_clutter_filter=False,
                        aoa_az_deg=path_aoa_az,
                        aoa_el_deg=path_aoa_el,
                    )
        logger.debug(
            "Direct RD map from %d paths (Doppler range %.1f–%.1f Hz, clutter=%s)",
            len(doppler_link),
            float(np.min(doppler_link)),
            float(np.max(doppler_link)),
            processor.clutter_enabled,
        )
        # Inject noise into the direct RD map power
        if (
            processor.noise_enabled
            and processor.noise_snr_db is not None
            and rd_primary is not None
        ):
            # Snapshot RD is analytic power, so noise is applied to a zero-phase
            # magnitude surrogate cube and then converted back to power.
            noise_result = inject_target_relative_complex_awgn(
                (
                    rd_primary.cube
                    if rd_primary.cube is not None
                    else _power_map_to_cube(rd_primary.power_linear)
                ),
                processor.noise_snr_db,
                rng=processor._noise_rng,
            )
            noisy_cube = noise_result.samples
            noise_injection_applications.append(
                {**noise_result.metadata(), "stage": "analytic_oracle_rd_surrogate"}
            )
            rd_primary = RangeDopplerResult(
                power_linear=np.abs(noisy_cube) ** 2,
                range_axis_m=rd_primary.range_axis_m,
                doppler_axis_hz=rd_primary.doppler_axis_hz,
                cube=noisy_cube,
                range_axis_label=rd_primary.range_axis_label,
                doppler_axis_label=rd_primary.doppler_axis_label,
            )
    if noise_injection_applications:
        sensing_data["noise_injection"] = {
            "contract_version": 1,
            "applications": noise_injection_applications,
        }
    elif processor.config.output_range_doppler:
        logger.debug(
            "Range–Doppler not computed (history %d/%d)",
            len(processor._fast_history_raw),
            processor.config.fft_size_doppler,
        )

    if (
        rd_result_raw is not None
        and processor.config.output_range_doppler
        and processor._should_emit_raw_payload("range_doppler_map_raw")
    ):
        sensing_data["range_doppler_map_raw"] = rd_result_raw.cube
        sensing_data["range_doppler_raw_power_by_pair"] = rd_result_raw.power_linear[
            np.newaxis, ...
        ]
    persist_detection_stages = bool(
        processor.persist_detection_stage_artifacts and processor.config.output_detections
    )
    if rd_primary is not None and processor.config.output_range_doppler:
        sensing_data["range_doppler_map"] = rd_primary.cube
    if rd_primary is not None and (
        processor.config.output_range_doppler or persist_detection_stages
    ):
        # The analytic oracle computes power directly.  Its canonical typed
        # product remains power; the historical complex magnitude surrogate is
        # retained only as an explicitly named compatibility view.
        sensing_data["range_doppler_power_by_pair"] = rd_primary.power_linear[np.newaxis, ...]
        sensing_data["range_axis_m"] = np.asarray(rd_primary.range_axis_m, dtype=np.float64)
        sensing_data["doppler_axis_hz"] = np.asarray(rd_primary.doppler_axis_hz, dtype=np.float64)
    if rd_primary is not None and processor.config.output_detections:
        with profiler.stage("detection"):
            cfar_metadata.update(
                processor._run_2d_cfar(
                    rd_primary,
                    sensing_data,
                    update_time_s=reference_time_s,
                    aoa_az_map=aoa_az_map,
                    aoa_el_map=aoa_el_map,
                )
            )
    sensing_data["rd_fresh"] = rd_primary is not None
    sensing_data["rp_fresh"] = rp_available

    c0 = C0
    range_factor = 2.0 if processor.range_mode == "monostatic" else 1.0
    fft_r = processor.config.fft_size_range
    range_axis = (np.arange(fft_r) / processor.config.bandwidth) * c0 / range_factor
    sensing_data.setdefault("range_axis_m", np.asarray(range_axis, dtype=np.float64))
    sensing_data["rx_element_axis"] = [{"element_index": 0, "row": 0, "column": 0}]
    if "range_profile" in sensing_data:
        sensing_data["range_profile_by_pair"] = np.asarray(sensing_data["range_profile"])[
            np.newaxis, ...
        ]
    if "range_profile_raw" in sensing_data:
        sensing_data["range_profile_raw_by_pair"] = np.asarray(sensing_data["range_profile_raw"])[
            np.newaxis, ...
        ]

    # Store per-path analytical data for oracle micro-Doppler
    if doppler_link is not None:
        sensing_data["analytical_paths"] = {
            "amplitudes": a_link,
            "delays_s": tau_link,
            "doppler_hz": doppler_link,
        }

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
            size=processor.config.fft_size_doppler,
            unit="Hz",
            coordinate_key="doppler_axis_hz",
            coordinate_semantics="fftshifted_doppler_frequency",
        ),
        range_product_axis,
    )
    if "cir_by_pair_rx_element" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="cir_by_pair_rx_element",
            kind="delay_domain_cir",
            axes=(pair_axis, element_axis, range_product_axis),
            representation="complex_voltage",
            unit="1",
            evidence_role="oracle_assisted",
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
    if "range_profile_by_pair" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="range_profile_by_pair",
            kind="range_profile",
            axes=(pair_axis, range_product_axis),
            representation="power_linear",
            unit="1",
            evidence_role="oracle_assisted",
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
            evidence_role="oracle_assisted",
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
            source_data_keys=(
                ("cir_by_pair_rx_element",) if "cir_by_pair_rx_element" in sensing_data else ()
            ),
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
    if "range_doppler_power_by_pair" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="range_doppler_power_by_pair",
            kind="range_doppler",
            axes=range_doppler_axes,
            representation="power_linear",
            unit="1",
            evidence_role="oracle_assisted",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            combine={"policy": "siso_identity", "normalization": "identity"},
            processing={
                "model": "analytic_path_power_oracle_nonwaveform",
                "doppler_assignment": "nearest_bin",
                **_direct_doppler_clutter_processing(
                    processor,
                    apply_clutter_filter=True,
                ),
            },
        )
    if "range_doppler_raw_power_by_pair" in sensing_data:
        add_product_spec(
            sensing_data,
            data_key="range_doppler_raw_power_by_pair",
            kind="range_doppler",
            variant="raw",
            axes=range_doppler_axes,
            representation="power_linear",
            unit="1",
            evidence_role="oracle_assisted",
            temporal_support=TEMPORAL_SUPPORT_FULL_ACQUISITION,
            combine={"policy": "siso_identity", "normalization": "identity"},
            processing={
                "model": "analytic_path_power_oracle_nonwaveform",
                "doppler_assignment": "nearest_bin",
                **_direct_doppler_clutter_processing(
                    processor,
                    apply_clutter_filter=False,
                ),
            },
        )
    processor._queue_detection_stage_product_specs(
        sensing_data,
        primary_data_key="range_doppler_power_by_pair",
        axes=range_doppler_axes,
        evidence_role="oracle_assisted",
        input_processing={
            "source_projection": "identity_power_linear",
            "pre_cfar_doppler_guard_policy": "none",
        },
    )
    sensing_data["compatibility_views"] = {
        "range_doppler_map": (
            "sqrt(range_doppler_power_by_pair[pair=0]) with zero phase; " "not complex voltage"
        ),
        "range_profile": "range_profile_by_pair[pair=0]",
        "cir": "cir_by_pair_rx_element[pair=0,rx_element=0]",
    }

    elapsed = (time.time() - start_time) * 1000.0
    profiler.attach(sensing_data, total_ms=elapsed)
    logger.debug("Sensing processing took %.2f ms", elapsed)
    return processor._finalize_recording(sensing_data)
