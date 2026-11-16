"""CPU-only reconstruction from a retained unwindowed frequency observation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np

from shared.sensing.products import (
    TEMPORAL_SUPPORT_FULL_ACQUISITION,
    TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
    add_product_spec,
    build_product_descriptor,
    dense_axis,
    select_product_by_contract,
)
from shared.sensing.range_response import (
    RangeResponseConfig,
    frequency_grid,
    frequency_to_delay,
    validate_sample_times,
)
from shared.sensing.signal_processing import (
    WINDOW_NORMALIZATION_ENERGY,
    _window,
    _window_energy_correction,
    coherent_clutter_metadata,
    remove_window_weighted_mean,
)


def processing_config_dict(config: Any) -> dict[str, Any]:
    """Keep historical recipe identities when new optional overrides are absent."""
    result = asdict(config)
    for key in ("range_response", "range_ifft_size", "coherent_clutter_mode"):
        if result.get(key) is None:
            result.pop(key, None)
    return result


def select_frequency_observation(sensing: Mapping[str, Any]) -> tuple:
    try:
        selected = select_product_by_contract(
            sensing,
            kind="baseband_frequency_observation",
            variant="primary",
            axis_names=("pair", "rx_element", "slow_time", "frequency"),
            representations=("complex_voltage",),
        )
    except KeyError as exc:
        raise ValueError(
            "CFR replay requires retained unwindowed baseband_frequency_observation; "
            "point-bin CIR, RD power and MPC summaries cannot recover fractional delays. "
            "Generate cfr_ifft with recording.profile: reprocessing or full."
        ) from exc
    key, values, axes, descriptor = selected
    if values.ndim != 4 or values.shape[0] != 1 or not np.iscomplexobj(values):
        raise ValueError(
            "CFR observation requires explicit complex [pair=1,element,time,frequency] axes"
        )
    if tuple(axis.get("unit") for axis in descriptor["axes"]) != ("1", "1", "s", "Hz"):
        raise ValueError("CFR axes must have units [1,1,s,Hz]")
    if descriptor.get("processing", {}).get("window_applied") is not False:
        raise ValueError("CFR replay requires an explicitly unwindowed observation")
    samples = sensing.get("temporal", {}).get("acquisition", {}).get("samples", [])
    if not samples or any(sample.get("valid") is not True for sample in samples):
        raise ValueError("CFR replay requires a complete valid acquisition")
    required = sensing["config_resolved"]["fft_size_doppler"]
    if values.shape[2] != required or len(samples) != required:
        raise ValueError("CFR Doppler replay requires a complete analysis window")
    times = validate_sample_times(axes[2], values.shape[2])
    if not np.array_equal(times, [sample["sample_time_s"] for sample in samples]):
        raise ValueError("CFR sample times disagree with acquisition membership")
    return selected


def reconstruct_delay_observation(sensing: Mapping[str, Any]) -> dict[str, Any]:
    """Return an owned typed delay view for explicitly requested signal analysis.

    Unlike Doppler replay this also accepts an authored single-pulse frequency
    observation during sequential warm-up. Existing arrays and metadata remain
    unchanged; the new product names its retained frequency parent.
    """
    _, values, axes, source = select_product_by_contract(
        sensing,
        kind="baseband_frequency_observation",
        variant="primary",
        axis_names=("pair", "rx_element", "slow_time", "frequency"),
        representations=("complex_voltage",),
    )
    if values.ndim != 4 or values.shape[0] != 1 or not np.isfinite(values).all():
        raise ValueError("Delay reconstruction requires one pair of finite frequency observations")
    if tuple(axis.get("unit") for axis in source["axes"]) != ("1", "1", "s", "Hz"):
        raise ValueError("Frequency observation axes must use [1,1,s,Hz]")
    cfg = sensing["config_resolved"]
    response = RangeResponseConfig.parse(cfg.get("range_response"))
    if (
        response.model != "cfr_ifft"
        or source.get("processing", {}).get("window_applied") is not False
    ):
        raise ValueError("Delay reconstruction requires an unwindowed cfr_ifft observation")
    if sensing.get("noise_injection", {}).get("applications"):
        raise ValueError("Retained CFR does not include independently injected delay-domain noise")
    if not np.array_equal(axes[3], frequency_grid(cfg["bandwidth"], response.num_frequency_bins)):
        raise ValueError("Retained frequency coordinates disagree with the recorded bandwidth")
    if source.get("processing", {}).get("carrier_frequency_hz") != cfg["carrier_freq_hz"]:
        raise ValueError("Retained frequency carrier disagrees with the recorded recipe")
    times = validate_sample_times(axes[2], values.shape[2], minimum_samples=1)
    samples = sensing["temporal"]["acquisition"]["samples"]
    valid_times = [row["sample_time_s"] for row in samples if row.get("valid") is True]
    if np.array_equal(times, valid_times) and len(valid_times) == len(samples):
        support = TEMPORAL_SUPPORT_FULL_ACQUISITION
    elif len(times) == 1 and valid_times and times[0] == valid_times[-1]:
        support = TEMPORAL_SUPPORT_LAST_VALID_SAMPLE
    else:
        raise ValueError("Retained frequency epochs do not match their acquisition support")
    delay, coordinates, metadata = frequency_to_delay(
        values[0],
        bandwidth=cfg["bandwidth"],
        config=response,
        fft_size=cfg["fft_size_range"],
        range_mode=cfg["range_mode"],
    )
    result = dict(sensing)
    key, range_key = "slow_time_by_pair_rx_element", "reconstructed_range_axis_m"
    result[key] = delay[None]
    result[range_key] = coordinates["range_axis_m"]
    pair, element, slow_time, _ = source["axes"]
    descriptor = build_product_descriptor(
        result,
        data_key=key,
        kind="delay_domain_slow_time",
        axes=(
            pair,
            element,
            slow_time,
            dense_axis(
                "range",
                size=delay.shape[-1],
                unit="m",
                coordinate_key=range_key,
                coordinate_semantics=(
                    "monostatic_range"
                    if cfg["range_mode"] == "monostatic"
                    else "bistatic_path_length"
                ),
                reference_frame="tx_target_rx_path",
            ),
        ),
        representation="complex_voltage",
        unit="1",
        scale_semantics=source["scale_semantics"],
        pair_ids=source["pair_ids"],
        temporal_support=support,
        combine=source.get("combine"),
        source_product_ids=(source["product_id"],),
        processing={**metadata, "reconstructed_from_frequency": True},
    )
    result["products"] = {**sensing["products"], key: descriptor}
    return result


def rebuild_frequency_observation(
    sensing: Mapping[str, Any],
    *,
    response: RangeResponseConfig | None = None,
    fft_size: int | None = None,
    window_doppler: str | None = None,
    mti_enabled: bool = True,
    coherent_clutter_mode: str | None = None,
) -> dict[str, Any]:
    """Rewindow/resize on the exact retained grid without resampling noise."""
    key, values, coordinates, descriptor = select_frequency_observation(sensing)
    cfg = sensing["config_resolved"]
    stored = RangeResponseConfig.parse(cfg.get("range_response"))
    requested = response or stored
    if stored.model != "cfr_ifft" or requested.model != "cfr_ifft":
        raise ValueError("CFR replay cannot convert a point-bin acquisition")
    if (
        requested.num_frequency_bins != stored.num_frequency_bins
        or requested.frequency_noise != stored.frequency_noise
    ):
        raise ValueError(
            "CFR replay cannot change the retained frequency grid or noise realization; acquire new paths"
        )
    band = float(cfg["bandwidth"])
    frequencies = np.asarray(coordinates[3])
    if not np.array_equal(frequencies, frequency_grid(band, requested.num_frequency_bins)):
        raise ValueError("Changed bandwidth/frequency grid requires a new acquisition")
    processing = descriptor["processing"]
    if processing.get("carrier_frequency_hz") != cfg["carrier_freq_hz"]:
        raise ValueError("Changed carrier requires a new acquisition")
    times = np.asarray(coordinates[2])
    if not np.isclose(1 / np.diff(times)[0], cfg["prf_hz"], rtol=1e-9):
        raise ValueError("Changed slow-time sampling requires a new acquisition")
    size = cfg["fft_size_range"] if fft_size is None else fft_size
    delay, axes, metadata = frequency_to_delay(
        values[0], bandwidth=band, config=requested, fft_size=size, range_mode=cfg["range_mode"]
    )
    combined = delay.sum(axis=0) / np.sqrt(delay.shape[0])
    recipe = cfg["processing_recipe"]
    # Derived recordings persist effective windows at the top level.
    taper_name = window_doppler or cfg.get("window_doppler", recipe.get("window_doppler"))
    taper = _window(taper_name, delay.shape[1])
    if not np.any(taper):
        raise ValueError(
            "Doppler window has zero energy; use boxcar or at least three Hann samples"
        )
    correction = _window_energy_correction(taper)
    clutter = coherent_clutter_mode
    if clutter is None:
        clutter = cfg.get("coherent_clutter_mode", recipe.get("coherent_clutter_mode", "legacy"))
    if clutter not in {"none", "legacy", "windowed_mean"}:
        raise ValueError("coherent_clutter_mode must be none, legacy or windowed_mean")
    if not mti_enabled:
        clutter = "none"
    mean_applied = clutter != "none" and (
        clutter == "windowed_mean" or bool(np.max(abs(np.diff(combined, axis=0))) > 0)
    )

    def doppler(array: np.ndarray, *, clean: bool, axis: int) -> np.ndarray:
        if clean and mean_applied:
            array = (
                remove_window_weighted_mean(array, taper, axis=axis)
                if clutter == "windowed_mean"
                else array - array.mean(axis=axis, keepdims=True)
            )
        shape = [1] * array.ndim
        shape[axis] = taper.size
        return np.fft.fftshift(
            np.fft.fft(array * taper.reshape(shape) * correction, axis=axis), axes=axis
        ) / np.sqrt(taper.size)

    range_metadata = {
        **metadata,
        "carrier_frequency_hz": cfg["carrier_freq_hz"],
        "noise": processing["noise"],
        "source_product_id": descriptor["product_id"],
        "slow_time_mean_removal_applied": False,
        "doppler_window_applied": False,
        "doppler_fft_applied": False,
    }
    metadata.update(
        {
            **range_metadata,
            "slow_time_mean_removal_applied": mean_applied,
            "coherent_clutter_mode": clutter,
            **coherent_clutter_metadata(clutter),
            "doppler_taper": taper_name,
            "doppler_window_applied": True,
            "doppler_fft_applied": True,
            "window_normalization": WINDOW_NORMALIZATION_ENERGY,
            "fft_axis": "slow_time",
            "fft_normalization": "unitary_1_over_sqrt_n",
        }
    )
    return {
        "source_key": key,
        "source_descriptor": descriptor,
        "observation": values,
        "response_config": requested.to_dict(),
        "delay": delay,
        "range_profile": abs(combined[-1]) ** 2,
        "rd_cube": doppler(combined, clean=True, axis=0),
        "rd_raw": doppler(combined, clean=False, axis=0),
        "element_rd": doppler(delay, clean=True, axis=1),
        "element_rd_raw": doppler(delay, clean=False, axis=1),
        "slow_time_axis_s": times,
        "doppler_axis_hz": np.fft.fftshift(np.fft.fftfreq(taper.size, d=1 / cfg["prf_hz"])),
        **axes,
        "metadata": metadata,
        "range_metadata": range_metadata,
    }


def publish_rebuilt_frequency(sensing: dict[str, Any], rebuilt: dict[str, Any]) -> None:
    """Describe replay-owned arrays with immutable upstream observation lineage."""
    source = rebuilt["source_descriptor"]
    source_id = source["product_id"]
    metadata = rebuilt["metadata"]
    # Native products also expose response annotations at the block's top
    # level. Rewindowing/resizing must refresh those mirrors, while the exact
    # upstream observation and its descriptor retain their acquisition identity.
    for key in metadata.keys() & sensing.keys():
        sensing[key] = deepcopy(metadata[key])
    for key in ("frequency_axis_hz", "delay_axis_s", "range_axis_m", "slow_time_axis_s"):
        sensing[key] = rebuilt[key]
    sensing["range_response"] = metadata
    sensing["fft_size_range"] = metadata["fft_size_range"]
    sensing["num_samples"] = metadata["fft_size_range"]
    pair, element, slow_time, frequency = source["axes"]
    range_axis = dense_axis(
        "range",
        size=metadata["fft_size_range"],
        unit="m",
        coordinate_key="range_axis_m",
        coordinate_semantics=(
            "monostatic_range"
            if sensing["config_resolved"]["range_mode"] == "monostatic"
            else "bistatic_path_length"
        ),
        reference_frame="tx_target_rx_path",
    )
    doppler = dense_axis(
        "doppler_frequency",
        size=rebuilt["rd_cube"].shape[0],
        unit="Hz",
        coordinate_key="doppler_axis_hz",
        coordinate_semantics="fftshifted_doppler_frequency",
    )
    specs = [
        (
            "cfr_by_pair_rx_element",
            "baseband_frequency_observation",
            rebuilt["observation"],
            (pair, element, slow_time, frequency),
            "primary",
            source["processing"],
        ),
        (
            "slow_time_by_pair_rx_element",
            "delay_domain_slow_time",
            rebuilt["delay"][None],
            (pair, element, slow_time, range_axis),
            "primary",
            rebuilt["range_metadata"],
        ),
        (
            "range_doppler_complex_by_pair",
            "range_doppler_complex_evidence",
            rebuilt["rd_cube"][None],
            (pair, doppler, range_axis),
            "primary",
            metadata,
        ),
        (
            "range_doppler_raw_by_pair",
            "range_doppler",
            rebuilt["rd_raw"][None],
            (pair, doppler, range_axis),
            "raw",
            {**metadata, "slow_time_mean_removal_applied": False},
        ),
        (
            "range_profile_by_pair",
            "range_profile",
            rebuilt["range_profile"][None],
            (pair, range_axis),
            "primary",
            rebuilt["range_metadata"],
        ),
    ]
    for key, kind, values, axes, variant, processing in specs:
        sensing[key] = values
        add_product_spec(
            sensing,
            data_key=key,
            kind=kind,
            axes=axes,
            variant=variant,
            representation="complex_voltage" if np.iscomplexobj(values) else "power_linear",
            unit="1",
            source_product_ids=(source_id,),
            combine=source.get("combine"),
            processing=processing,
            temporal_support=(
                TEMPORAL_SUPPORT_LAST_VALID_SAMPLE
                if kind == "range_profile"
                else TEMPORAL_SUPPORT_FULL_ACQUISITION
            ),
        )
    spatial = rebuilt.get("spatial")
    if spatial is not None:
        from shared.sensing.arrays import SPATIAL_FFT_NORMALIZATION, SPATIAL_PHASE_CONVENTION

        sensing["azimuth_direction_cosine"] = spatial.azimuth_direction_cosine
        sensing["elevation_direction_cosine"] = spatial.elevation_direction_cosine
        sensing["azimuth_broadside_angle_deg"] = spatial.broadside_azimuth_deg
        sensing["elevation_broadside_angle_deg"] = spatial.broadside_elevation_deg
        sensing["range_doppler_angle_by_pair"] = (abs(spatial.spectrum) ** 2)[None].astype(
            np.float32
        )
        angular = tuple(
            dense_axis(
                name,
                size=size,
                unit="1",
                coordinate_key=name,
                coordinate_semantics=semantics,
                alternate_coordinate_keys=(alternate,),
                reference_frame="rx_array_local",
            )
            for name, size, semantics, alternate in (
                (
                    "elevation_direction_cosine",
                    spatial.spectrum.shape[0],
                    "rx_array_local_z_direction_cosine",
                    "elevation_broadside_angle_deg",
                ),
                (
                    "azimuth_direction_cosine",
                    spatial.spectrum.shape[1],
                    "rx_array_local_y_direction_cosine",
                    "azimuth_broadside_angle_deg",
                ),
            )
        )
        add_product_spec(
            sensing,
            data_key="range_doppler_angle_by_pair",
            kind="range_doppler_angle",
            axes=(pair, *angular, doppler, range_axis),
            representation="power_linear",
            unit="1",
            source_product_ids=(source_id,),
            processing={
                **metadata,
                "spatial_fft_normalization": SPATIAL_FFT_NORMALIZATION,
                "spatial_phase_convention": SPATIAL_PHASE_CONVENTION,
            },
        )
