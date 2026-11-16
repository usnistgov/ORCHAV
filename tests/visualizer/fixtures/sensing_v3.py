"""Small schema-v4 sensing payloads for visualizer unit tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
from uuid import UUID, uuid5

import numpy as np

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
)
from shared.sensing.products import (
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)

_GENERATION_ID = "432c0a09-35db-4941-bc8b-de76440cd0ef"
_ACQUISITION_NAMESPACE = UUID("3d8e2346-b56c-4774-a04b-e6368281f112")
_PROCESSING_RUN_ID = "dd0f865c-8441-4b03-8694-b361bd493208"


def build_visualizer_sensing_payload(
    products: Mapping[str, Any] | None = None,
    *,
    scene_frame_id: int = 0,
    sample_time_s: float = 0.0,
) -> dict[str, Any]:
    """Return a valid one-sample v4 payload extended with visual products."""
    temporal = build_sensing_temporal_contract(
        clock_id=_GENERATION_ID,
        acquisition_id=str(uuid5(_ACQUISITION_NAMESPACE, f"{scene_frame_id!r}:{sample_time_s!r}")),
        acquisition_config_id=canonical_sensing_hash({"samples": 1}),
        pair_id=canonical_sensing_hash(
            {"tx_sensor_id": "visualizer-tx", "rx_sensor_id": "visualizer-rx"}
        ),
        tx_sensor_id="visualizer-tx",
        rx_sensor_id="visualizer-rx",
        samples=[
            {
                "sample_index": 0,
                "sample_time_s": sample_time_s,
                "valid": True,
                "phase_coherent": True,
                "source": {
                    "source_generation_id": _GENERATION_ID,
                    "scene_frame_id": scene_frame_id,
                    "sample_offset": 0,
                },
            }
        ],
        processing_run_id=_PROCESSING_RUN_ID,
        processing_recipe_id=canonical_sensing_hash({"algorithm": "visualizer-test-fixture"}),
        reference_time_s=sample_time_s,
        reference_time_policy="sample_time",
        display_scene_frame_id=scene_frame_id,
        display_policy="exact",
        display_reason="Single-frame visualizer test fixture.",
    )
    payload: dict[str, Any] = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "processing_strategy": "analytic_path_doppler_oracle",
    }
    if products is not None:
        payload.update(products)
    _ensure_visualizer_resolved_config(payload)
    payload["temporal"]["result"]["processing_recipe_id"] = canonical_sensing_hash(
        payload["config_resolved"]["processing_recipe"]
    )
    materialize_visualizer_product_specs(payload)
    return payload


def _ensure_visualizer_resolved_config(payload: dict[str, Any]) -> None:
    """Give native-v4 visualizer fixtures a complete exact runtime envelope."""

    range_size = 1
    doppler_size = 1
    if payload.get("range_axis_m") is not None:
        range_size = int(np.asarray(payload["range_axis_m"]).size)
    for key in (
        "range_profile_by_pair",
        "cir_by_pair_rx_element",
        "slow_time_by_pair_rx_element",
        "range_doppler_by_pair",
        "range_doppler_power_by_pair",
    ):
        if payload.get(key) is not None:
            range_size = int(np.asarray(payload[key]).shape[-1])
            break
    if payload.get("doppler_axis_hz") is not None:
        doppler_size = int(np.asarray(payload["doppler_axis_hz"]).size)
    else:
        for key in ("range_doppler_by_pair", "range_doppler_power_by_pair"):
            if payload.get(key) is not None:
                doppler_size = int(np.asarray(payload[key]).shape[-2])
                break

    default_resolved: dict[str, Any] = {
        "contract_version": 1,
        "sampling_frequency": 1.0e9,
        "bandwidth": 1.0e9,
        "bandwidth_hz": 1.0e9,
        "prf_hz": 1000.0,
        "fft_size_range": range_size,
        "fft_size_doppler": doppler_size,
        "carrier_freq_hz": 28.0e9,
        "range_mode": "monostatic",
        "window_range": "hann",
        "window_doppler": "hann",
        "processing_mode": "analytic_path_doppler_oracle",
        "acquisition": {
            "sample_source": "analytic_path_snapshot",
            "rt_solve_stride_scene_steps": 1,
            "sample_period_s": None,
            "sample_rate_hz": None,
            "cpi_length_samples": 1,
            "cpi_hop_samples": 1,
            "overlap_samples": 0,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": "analytic_nonwaveform_path_doppler",
        },
        "processing": {
            "mode": "analytic_path_doppler_oracle",
            "strategy": "analytic_path_doppler_oracle",
            "fft_size_doppler": doppler_size,
            "result_stride_samples": 1,
            "product_semantics": "analytic_oracle_nonwaveform",
            "evidence_status": "diagnostic_quarantined",
        },
        "topology": {
            "pair_count": 1,
            "pair_index": 0,
            "tx_device_count": 1,
            "rx_device_count": 1,
            "tx_elements": 1,
            "rx_elements": 1,
            "rx_array_processing": "siso",
        },
        "processing_recipe": {
            "window_range": "hann",
            "window_doppler": "hann",
        },
    }
    supplied = payload.get("config_resolved")
    if not isinstance(supplied, Mapping):
        supplied = payload.get("config")
    if isinstance(supplied, Mapping):
        for key, value in supplied.items():
            if isinstance(value, Mapping) and isinstance(default_resolved.get(key), Mapping):
                default_resolved[key] = {
                    **dict(default_resolved[key]),
                    **deepcopy(dict(value)),
                }
            else:
                default_resolved[key] = deepcopy(value)
    payload["config_resolved"] = default_resolved
    payload["config"] = deepcopy(default_resolved)


def materialize_visualizer_product_specs(payload: dict[str, Any]) -> None:
    """Describe canonical visualizer fixture arrays against their current temporal IDs."""
    payload.pop("products", None)
    payload.pop("product_contract_version", None)
    payload.pop("_product_specs", None)

    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    range_coordinates = payload.get("range_axis_m")
    config = payload.get("config_resolved")
    if not isinstance(config, Mapping):
        config = payload.get("config")
    range_mode = config.get("range_mode") if isinstance(config, Mapping) else None
    range_semantics = (
        "bistatic_path_length"
        if isinstance(range_mode, str) and range_mode.lower().startswith("bista")
        else "monostatic_range"
    )

    element_count: int | None = None
    range_size: int | None = None
    cir_values = payload.get("cir_by_pair_rx_element")
    if cir_values is not None:
        cir = np.asarray(cir_values)
        if cir.ndim != 3 or cir.shape[0] != 1:
            raise ValueError("cir_by_pair_rx_element fixture must use [1,E,R]")
        element_count = int(cir.shape[1])
        range_size = int(cir.shape[2])
    slow_values = payload.get("slow_time_by_pair_rx_element")
    if slow_values is not None:
        slow_time = np.asarray(slow_values)
        if slow_time.ndim != 4 or slow_time.shape[0] != 1:
            raise ValueError("slow_time_by_pair_rx_element fixture must use [1,E,N,R]")
        if element_count is not None and element_count != int(slow_time.shape[1]):
            raise ValueError("Typed CIR and slow-time fixture RX-element sizes disagree")
        if range_size is not None and range_size != int(slow_time.shape[3]):
            raise ValueError("Typed CIR and slow-time fixture range sizes disagree")
        element_count = int(slow_time.shape[1])
        range_size = int(slow_time.shape[3])

    element_axis = None
    if element_count is not None:
        payload.setdefault(
            "rx_element_axis",
            [{"element_index": index, "row": 0, "column": index} for index in range(element_count)],
        )
        element_axis = catalog_axis(
            "rx_element",
            size=element_count,
            catalog_key="rx_element_axis",
            coordinate_semantics="sionna_rx_element_index",
            reference_frame="rx_array_local",
        )
        if range_coordinates is None:
            raise ValueError("Typed CIR/slow-time fixture requires range_axis_m")
        if np.asarray(range_coordinates).shape != (range_size,):
            raise ValueError("Typed CIR/slow-time fixture range_axis_m size is inconsistent")
        fast_time_range_axis = dense_axis(
            "range",
            size=int(range_size),
            unit="m",
            coordinate_key="range_axis_m",
            coordinate_semantics=range_semantics,
            reference_frame="tx_target_rx_path",
        )
        fast_time_combine = (
            {"policy": "siso_identity", "normalization": "identity"}
            if element_count == 1
            else {
                "input_axis": "rx_element",
                "policy": "rx_broadside_unitary_voltage_sum",
                "normalization": "one_over_sqrt_element_count",
            }
        )
        if cir_values is not None:
            add_product_spec(
                payload,
                data_key="cir_by_pair_rx_element",
                kind="delay_domain_cir",
                axes=(pair_axis, element_axis, fast_time_range_axis),
                representation="complex_voltage",
                unit="1",
                combine=fast_time_combine,
            )
        if slow_values is not None:
            slow_time = np.asarray(slow_values)
            add_product_spec(
                payload,
                data_key="slow_time_by_pair_rx_element",
                kind="delay_domain_slow_time",
                axes=(
                    pair_axis,
                    element_axis,
                    dense_axis(
                        "slow_time",
                        size=int(slow_time.shape[2]),
                        unit="s",
                        coordinate_key="slow_time_axis_s",
                        coordinate_semantics="scenario_clock_sample_time",
                        reference_frame="scenario_clock",
                    ),
                    fast_time_range_axis,
                ),
                representation="complex_voltage",
                unit="1",
                combine=fast_time_combine,
            )

    if "range_profile_by_pair" in payload:
        profile = np.asarray(payload["range_profile_by_pair"])
        if range_coordinates is None:
            raise ValueError("range_profile_by_pair fixture requires range_axis_m")
        add_product_spec(
            payload,
            data_key="range_profile_by_pair",
            kind="range_profile",
            axes=(
                pair_axis,
                dense_axis(
                    "range",
                    size=profile.shape[-1],
                    unit="m",
                    coordinate_key="range_axis_m",
                    coordinate_semantics=range_semantics,
                    reference_frame="tx_target_rx_path",
                ),
            ),
            representation="power_linear",
            unit="1",
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
    for data_key, representation in (
        ("range_doppler_by_pair", "complex_voltage"),
        ("range_doppler_power_by_pair", "power_linear"),
    ):
        if data_key not in payload:
            continue
        rd_values = np.asarray(payload[data_key])
        if range_coordinates is None or payload.get("doppler_axis_hz") is None:
            raise ValueError(f"{data_key} fixture requires exact range and Doppler axes")
        add_product_spec(
            payload,
            data_key=data_key,
            kind="range_doppler",
            axes=(
                pair_axis,
                dense_axis(
                    "doppler_frequency",
                    size=rd_values.shape[-2],
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                ),
                dense_axis(
                    "range",
                    size=rd_values.shape[-1],
                    unit="m",
                    coordinate_key="range_axis_m",
                    coordinate_semantics=range_semantics,
                    reference_frame="tx_target_rx_path",
                ),
            ),
            representation=representation,
            unit="1",
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
    angular_values = payload.get("range_doppler_angle_by_pair")
    if angular_values is not None:
        angular = np.asarray(angular_values)
        if angular.ndim != 5 or angular.shape[0] != 1:
            raise ValueError("range_doppler_angle_by_pair fixture must use [1,E,A,D,R]")
        required_coordinates = (
            "elevation_direction_cosine",
            "azimuth_direction_cosine",
            "elevation_broadside_angle_deg",
            "azimuth_broadside_angle_deg",
            "doppler_axis_hz",
            "range_axis_m",
        )
        missing_coordinates = [key for key in required_coordinates if key not in payload]
        if missing_coordinates:
            raise ValueError(
                "range_doppler_angle_by_pair fixture lacks exact coordinates: "
                + ", ".join(missing_coordinates)
            )
        add_product_spec(
            payload,
            data_key="range_doppler_angle_by_pair",
            kind="range_doppler_angle",
            axes=(
                pair_axis,
                dense_axis(
                    "elevation_direction_cosine",
                    size=angular.shape[1],
                    unit="1",
                    coordinate_key="elevation_direction_cosine",
                    alternate_coordinate_keys=("elevation_broadside_angle_deg",),
                    coordinate_semantics="rx_array_local_z_direction_cosine",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "azimuth_direction_cosine",
                    size=angular.shape[2],
                    unit="1",
                    coordinate_key="azimuth_direction_cosine",
                    alternate_coordinate_keys=("azimuth_broadside_angle_deg",),
                    coordinate_semantics="rx_array_local_y_direction_cosine",
                    reference_frame="rx_array_local",
                ),
                dense_axis(
                    "doppler_frequency",
                    size=angular.shape[3],
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                    coordinate_semantics="fftshifted_doppler_frequency",
                ),
                dense_axis(
                    "range",
                    size=angular.shape[4],
                    unit="m",
                    coordinate_key="range_axis_m",
                    coordinate_semantics=range_semantics,
                    reference_frame="tx_target_rx_path",
                ),
            ),
            representation="power_linear",
            unit="1",
        )
    materialize_product_specs(payload)


__all__ = [
    "build_visualizer_sensing_payload",
    "materialize_visualizer_product_specs",
]
