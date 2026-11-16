"""Conservative, observation-only reliability decisions for native fusion reports."""

from dataclasses import replace
from numbers import Integral

from shared.sensing.fusion.core import COMPONENTS, FusionConfig, Observation


def native_observation(sensor_id, frame_id, index, time_s, detection, sensing, source):
    """Preserve producer evidence without using truth or local SNR as uncertainty."""
    import numpy as np

    guards = (
        sensing.get("cfar_metadata", {}).get("range_doppler", {}).get("pre_cfar_doppler_guards", {})
    )
    resolved = sensing.get("config_resolved", {})
    fft_size = resolved.get("fft_size_doppler")
    mirror = sensing.get("fft_size_doppler")
    for name, value in (
        ("config_resolved.fft_size_doppler", fft_size),
        ("fft_size_doppler", mirror),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive exact integer")
    fft_size = int(fft_size) if fft_size is not None else None
    if fft_size is not None and mirror is not None and mirror != fft_size:
        raise ValueError("fft_size_doppler mirror contradicts authoritative config_resolved")
    native = {
        **source,
        "storage_frame_id": frame_id,
        "detection_index": index,
        "range_bin": detection.get("range_bin"),
        "doppler_bin": detection.get("doppler_bin"),
        "fft_size_doppler": fft_size,
        "dc_exclusion_bin_count": guards.get("dc_bins"),
        "dc_exclusion_convention": (
            "fftshift_centered_total_width" if "dc_bins" in guards else "not_recorded"
        ),
        "local_snr_db": detection.get("snr_db"),
    }
    values = [
        detection.get("range_m"),
        detection.get("aoa_az_deg"),
        detection.get("aoa_el_deg"),
        detection.get("velocity_m_s"),
    ]
    # Refined singleton-axis coordinates are plotting assumptions, not
    # spatial observations. Older producers without flags keep their contract.
    for i, flag in ((1, "aoa_az_observable"), (2, "aoa_el_observable")):
        if detection.get(flag) is False:
            values[i] = None
    native["angle_assumptions"] = list(detection.get("angle_assumptions", []))
    for i in (1, 2):
        if values[i] is not None:
            values[i] = float(np.radians(values[i]))
    components = tuple(c for c, value in zip(COMPONENTS, values) if value is not None)
    return Observation(
        sensor_id,
        f"{frame_id}:{index}",
        time_s,
        *values,
        components=components,
        native_source=native,
    )


def apply_quality(observation: Observation, config: FusionConfig) -> Observation:
    """Omit radial velocity beside an authored DC exclusion; preserve spatial data."""
    reasons = ["quality_unknown_declared_fallback"]
    components = observation.components
    if config.quality_policy == "legacy":
        return replace(observation, quality_reasons=("legacy_quality_policy",))
    evidence = observation.native_source
    count, size, index = (
        evidence.get(k) for k in ("dc_exclusion_bin_count", "fft_size_doppler", "doppler_bin")
    )
    if evidence.get("dc_exclusion_convention") != "fftshift_centered_total_width" or any(
        type(v) is not int for v in (count, size, index)
    ):
        reasons.append("dc_boundary_quality_unknown")
    else:
        if size <= 0 or count < 0 or not 0 <= index < size:
            raise ValueError("Malformed producer Doppler-bin evidence")
        count = min(count, size)
        lo = max(0, size // 2 - count // 2)
        hi = min(size, lo + count)
        lo = max(0, hi - count)
        width = config.dc_boundary_width_bins
        if count and lo <= index < hi:
            reasons.append("radial_velocity_inside_documented_dc_exclusion")
            components = tuple(c for c in components if c != COMPONENTS[3])
        elif count and (lo - width <= index < lo or hi <= index < hi + width):
            reasons.append("radial_velocity_adjacent_to_dc_exclusion")
            components = tuple(c for c in components if c != COMPONENTS[3])
    covariance = observation.covariance
    if covariance is not None and components != observation.components:
        indices = [observation.components.index(c) for c in components]
        covariance = (
            tuple(tuple(covariance[i][j] for j in indices) for i in indices) if indices else None
        )
    return replace(
        observation, components=components, covariance=covariance, quality_reasons=tuple(reasons)
    )
