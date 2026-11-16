"""Scenario-backed sensing acquisition and processing diagrams.

The renderer in this module is intentionally independent of generator runtime
objects.  It consumes the same persisted schema-v4 sensing payload that an
HDF5 reader or postprocessor sees, builds a small immutable explanatory model,
and renders that model with Matplotlib.  The generator summary layer is only a
thin adapter.

The figure is a timing and processing explainer, not a waveform plot.  ORCHAV
supports point-delay projection or a retained finite-bandwidth CFR-to-delay
IFFT before the slow-time Doppler FFT. Neither route synthesizes an RF chirp,
matched filter, or pulse-compression waveform.
"""

from __future__ import annotations

import json
import math
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from shared.extensions.sensing import (
    SENSING_DEFAULT_REFERENCE_POLICY,
    SENSING_SCHEMA_VERSION,
    assert_supported_sensing_schema,
)
from shared.sensing.signal_processing import (
    DELAY_PROJECTION_POINT_BIN_FLOOR,
    DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED,
    DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF,
    DOPPLER_FFT_NORMALIZATION_UNITARY,
    RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
    WINDOW_NORMALIZATION_ENERGY,
)

COHERENT_MODE = "coherent_cpi"
SEQUENTIAL_MODE = "sequential_stft"
ANALYTIC_ORACLE_MODE = "analytic_path_doppler_oracle"

_MODE_TITLES = {
    COHERENT_MODE: "Coherent CPI",
    SEQUENTIAL_MODE: "Sequential STFT",
    ANALYTIC_ORACLE_MODE: "Analytic path-Doppler oracle",
}

_MODE_STATUS = {
    COHERENT_MODE: ("PRIMARY", "controlled specular validation", "#147D64"),
    SEQUENTIAL_MODE: ("EXPERIMENTAL", "controlled specular validation only", "#C77800"),
    ANALYTIC_ORACLE_MODE: ("DIAGNOSTIC", "non-waveform oracle", "#A13A56"),
}

_MODE_EVIDENCE_STATUS = {
    COHERENT_MODE: "phase2_controlled_specular_validated",
    SEQUENTIAL_MODE: "phase2_controlled_specular_validated_experimental",
    ANALYTIC_ORACLE_MODE: "diagnostic_quarantined",
}

_SLOW_TIME_MEAN_REMOVAL_POLICY = "when_row_variation_nonzero"

_VALID_WINDOW_NAMES = frozenset(
    {
        "hann",
        "hanning",
        "hamming",
        "blackman",
        "blackman-harris",
        "blackmanharris",
        "kaiser",
        "rect",
        "rectangular",
        "boxcar",
        "none",
    }
)

_INK = "#17233C"
_MUTED = "#5D687B"
_PANEL = "#F6F8FC"
_GRID = "#D9E0EC"
_BLUE = "#316BD6"
_CYAN = "#2F9BB5"
_PURPLE = "#7758C7"
_GOLD = "#D99019"
_GREEN = "#168268"
_RED = "#B64C65"


@dataclass(frozen=True, slots=True)
class SensingDiagramWindow:
    """One CPI or sliding-STFT window expressed in exact sample indices."""

    label: str
    start_sample: int
    length_samples: int
    result_emitted_with_sample: int

    @property
    def stop_sample(self) -> int:
        """Return the exclusive sample-index boundary."""

        return self.start_sample + self.length_samples


@dataclass(frozen=True, slots=True)
class SensingDiagramStage:
    """One explanatory processing stage."""

    title: str
    detail: str
    color: str


@dataclass(frozen=True, slots=True)
class SensingDiagramSlot:
    """One exact persisted acquisition slot."""

    index: int
    time_s: float
    valid: bool
    phase_coherent: bool
    source_generation_id: str | None
    scene_frame_id: int | str | None
    sample_offset: int | None
    solve_id: str | None
    invalid_reason: str | None

    @property
    def composite_source_identity(self) -> tuple[str, int | str, int] | None:
        """Return the source identity used to verify overlap membership."""

        if (
            self.source_generation_id is None
            or self.scene_frame_id is None
            or self.sample_offset is None
        ):
            return None
        return (self.source_generation_id, self.scene_frame_id, self.sample_offset)


@dataclass(frozen=True, slots=True)
class SensingDiagramRtEpoch:
    """One RT source epoch and the acquisition members it produced."""

    solve_id: str
    scene_frame_id: int | str
    source_epoch_s: float
    member_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SensingDiagramAcquisition:
    """One exact persisted acquisition envelope."""

    acquisition_id: str
    completeness: str
    slots: tuple[SensingDiagramSlot, ...]
    rt_epochs: tuple[SensingDiagramRtEpoch, ...]
    start_time_s: float
    end_time_s: float
    expected_sample_count: int
    default_reference_policy: str
    default_reference_time_s: float
    result_reference_time_s: float | None
    result_reference_policy: str | None
    processing_run_id: str | None
    processing_recipe_id: str | None
    slow_time_mean_removal_applied: bool | None

    @property
    def valid_slots(self) -> tuple[SensingDiagramSlot, ...]:
        """Return valid observed slots in persisted order."""

        return tuple(slot for slot in self.slots if slot.valid)


@dataclass(frozen=True, slots=True)
class SensingDiagramModel:
    """Immutable, JSON-serializable model behind one sensing diagram."""

    schema_version: int
    contract_version: int
    mode: str
    mode_title: str
    strategy: str
    evidence_status: str
    evidence_detail: str
    evidence_color: str
    sample_source: str
    phase_contract: str
    sample_period_s: float | None
    sample_rate_hz: float | None
    cpi_length_samples: int
    cpi_hop_samples: int
    overlap_samples: int
    overlap_fraction: float
    rt_solve_stride_scene_steps: int
    result_stride_samples: int
    fft_size_range: int
    fft_size_doppler: int
    bandwidth_hz: float
    delay_sampling_frequency_hz: float
    range_mode: str
    delay_projection: str | None
    range_processing: str | None
    range_fft_applied: bool | None
    delay_taper: str | None
    doppler_taper: str | None
    fft_axis: str | None
    fft_normalization: str | None
    window_normalization: str | None
    slow_time_mean_removal_policy: str | None
    slow_time_mean_removal_applied: bool | None
    direct_doppler_clutter_filter_policy: str | None
    direct_doppler_clutter_cutoff_hz: float | None
    clutter_removal_enabled: bool
    output_detections: bool
    cfar_type: str | None
    pair_id: str
    acquisition_config_id: str
    processing_run_id: str | None
    processing_recipe_id: str | None
    acquisition_completeness: str
    observed_sample_count: int
    expected_sample_count: int
    result_reference_policy: str | None
    acquisitions: tuple[SensingDiagramAcquisition, ...]
    observed_revisit_interval_s: float | None
    windows: tuple[SensingDiagramWindow, ...]
    rt_solve_samples: tuple[int, ...]
    processing_stages: tuple[SensingDiagramStage, ...]

    @property
    def cpi_duration_s(self) -> float | None:
        """Return the exact symbolic slot duration ``N * PRI``."""

        if self.sample_period_s is None:
            return None
        return self.cpi_length_samples * self.sample_period_s

    @property
    def hop_duration_s(self) -> float | None:
        """Return the exact result/window hop duration."""

        if self.sample_period_s is None:
            return None
        return self.cpi_hop_samples * self.sample_period_s

    @property
    def sample_center_span_s(self) -> float | None:
        """Return first-to-last sample-center separation within a window."""

        if self.sample_period_s is None:
            return None
        return max(self.cpi_length_samples - 1, 0) * self.sample_period_s


@dataclass(frozen=True, slots=True)
class _ProcessingDescriptorSemantics:
    """Result-invariant processing claims carried by the primary RD product."""

    delay_projection: str | None
    range_processing: str | None
    range_fft_applied: bool | None
    fft_axis: str | None
    fft_normalization: str | None
    window_normalization: str | None
    slow_time_mean_removal_policy: str | None
    direct_doppler_clutter_filter_policy: str | None
    direct_doppler_clutter_cutoff_hz: float | None


def _required_mapping(parent: Mapping[str, Any], key: str, *, path: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be a mapping")
    return value


def _required_string(parent: Mapping[str, Any], key: str, *, path: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _required_int(parent: Mapping[str, Any], key: str, *, path: str, minimum: int = 1) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if minimum == 0 else "positive"
        raise ValueError(f"{path}.{key} must be a {qualifier} integer")
    return int(value)


def _required_positive_float(parent: Mapping[str, Any], key: str, *, path: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{path}.{key} must be a finite positive number")
    return normalized


def _optional_positive_float(parent: Mapping[str, Any], key: str, *, path: str) -> float | None:
    value = parent.get(key)
    if value is None:
        return None
    return _required_positive_float(parent, key, path=path)


def _recipe_value(
    config: Mapping[str, Any],
    recipe: Mapping[str, Any],
    key: str,
    *,
    required: bool = True,
) -> Any:
    """Return one exact recipe value and reject contradictory duplicates."""

    top_present = key in config and config.get(key) is not None
    recipe_present = key in recipe and recipe.get(key) is not None
    if top_present and recipe_present and config[key] != recipe[key]:
        raise ValueError(
            f"sensing.config_resolved.{key} disagrees with "
            f"sensing.config_resolved.processing_recipe.{key}"
        )
    if top_present:
        return config[key]
    if recipe_present:
        return recipe[key]
    if required:
        raise ValueError(f"Sensing diagram requires exact config_resolved/processing_recipe.{key}")
    return None


def _required_bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a Boolean")
    return value


def _finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{path} must be a finite number")
    return normalized


def _require_matching_time(actual: float, expected: float, *, path: str, meaning: str) -> None:
    """Reject a persisted temporal summary that contradicts its exact slots."""

    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{path} must equal {meaning}")


def _extract_diagram_acquisition(
    sensing_data: Mapping[str, Any],
    *,
    mode: str,
    expected_count: int,
    sample_period_s: float | None,
) -> SensingDiagramAcquisition:
    """Extract exact sample membership and RT epochs from one sensing event."""

    temporal = _required_mapping(sensing_data, "temporal", path="sensing")
    acquisition = _required_mapping(temporal, "acquisition", path="sensing.temporal")
    samples = acquisition.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("sensing.temporal.acquisition.samples must be an ordered sequence")
    if len(samples) != expected_count:
        raise ValueError(
            "sensing.temporal.acquisition.samples must match the resolved CPI/window length"
        )
    persisted_expected_count = _required_int(
        acquisition,
        "expected_sample_count",
        path="sensing.temporal.acquisition",
        minimum=0,
    )
    if persisted_expected_count != expected_count:
        raise ValueError(
            "sensing.temporal.acquisition.expected_sample_count must match the resolved "
            "CPI/window length"
        )

    slots: list[SensingDiagramSlot] = []
    for position, sample in enumerate(samples):
        sample_path = f"sensing.temporal.acquisition.samples[{position}]"
        if not isinstance(sample, Mapping):
            raise ValueError(f"{sample_path} must be a mapping")
        if sample.get("sample_index") != position:
            raise ValueError(
                "sensing.temporal.acquisition.samples must use contiguous ordered sample_index"
            )
        timestamp = _finite_float(sample.get("sample_time_s"), path=f"{sample_path}.sample_time_s")
        valid = _required_bool(sample.get("valid"), path=f"{sample_path}.valid")
        phase_coherent = _required_bool(
            sample.get("phase_coherent"),
            path=f"{sample_path}.phase_coherent",
        )
        source = sample.get("source")
        if valid:
            if not isinstance(source, Mapping):
                raise ValueError(f"{sample_path}.source is required for a valid sample")
            generation_id = _required_string(
                source,
                "source_generation_id",
                path=f"{sample_path}.source",
            )
            scene_frame_id = source.get("scene_frame_id")
            if isinstance(scene_frame_id, bool) or not isinstance(scene_frame_id, (int, str)):
                raise ValueError(f"{sample_path}.source.scene_frame_id must be int or string")
            sample_offset = _required_int(
                source,
                "sample_offset",
                path=f"{sample_path}.source",
                minimum=0,
            )
            solve_id = _required_string(source, "solve_id", path=f"{sample_path}.source")
            invalid_reason = None
        else:
            if source is not None:
                raise ValueError(f"{sample_path}.source must be absent for an invalid sample")
            generation_id = None
            scene_frame_id = None
            sample_offset = None
            solve_id = None
            invalid_reason = _required_string(sample, "invalid_reason", path=sample_path)
        slots.append(
            SensingDiagramSlot(
                index=position,
                time_s=timestamp,
                valid=valid,
                phase_coherent=phase_coherent,
                source_generation_id=generation_id,
                scene_frame_id=scene_frame_id,
                sample_offset=sample_offset,
                solve_id=solve_id,
                invalid_reason=invalid_reason,
            )
        )

    observed_count = _required_int(
        acquisition,
        "observed_sample_count",
        path="sensing.temporal.acquisition",
        minimum=0,
    )
    valid_count = _required_int(
        acquisition,
        "valid_sample_count",
        path="sensing.temporal.acquisition",
        minimum=0,
    )
    valid_slots = tuple(slot for slot in slots if slot.valid)
    if observed_count != valid_count or valid_count != len(valid_slots):
        raise ValueError("Sensing temporal observed/valid sample counts are inconsistent")
    if sample_period_s is not None and len(valid_slots) > 1:
        for left, right in zip(valid_slots, valid_slots[1:]):
            if not math.isclose(
                right.time_s - left.time_s,
                sample_period_s,
                rel_tol=1e-7,
                abs_tol=max(1e-12, sample_period_s * 1e-9),
            ):
                raise ValueError(
                    "Sensing temporal sample cadence disagrees with the resolved sample period"
                )

    start_time_s = _finite_float(
        acquisition.get("start_time_s"),
        path="sensing.temporal.acquisition.start_time_s",
    )
    end_time_s = _finite_float(
        acquisition.get("end_time_s"),
        path="sensing.temporal.acquisition.end_time_s",
    )
    _require_matching_time(
        start_time_s,
        slots[0].time_s,
        path="sensing.temporal.acquisition.start_time_s",
        meaning="the first sample time",
    )
    _require_matching_time(
        end_time_s,
        slots[-1].time_s,
        path="sensing.temporal.acquisition.end_time_s",
        meaning="the last sample time",
    )
    default_reference_policy = _required_string(
        acquisition,
        "default_reference_policy",
        path="sensing.temporal.acquisition",
    )
    if default_reference_policy != SENSING_DEFAULT_REFERENCE_POLICY:
        raise ValueError(
            "sensing.temporal.acquisition.default_reference_policy must equal "
            f"{SENSING_DEFAULT_REFERENCE_POLICY!r}"
        )
    default_reference_time_s = _finite_float(
        acquisition.get("default_reference_time_s"),
        path="sensing.temporal.acquisition.default_reference_time_s",
    )
    _require_matching_time(
        default_reference_time_s,
        0.5 * (start_time_s + end_time_s),
        path="sensing.temporal.acquisition.default_reference_time_s",
        meaning="the first/last sample-center midpoint",
    )

    epochs_by_solve: dict[str, list[SensingDiagramSlot]] = {}
    for slot in valid_slots:
        if slot.solve_id is None:
            raise ValueError("Valid sensing slots require an RT solve identity")
        epochs_by_solve.setdefault(slot.solve_id, []).append(slot)
    epochs: list[SensingDiagramRtEpoch] = []
    for solve_id, members in epochs_by_solve.items():
        scene_ids = {member.scene_frame_id for member in members}
        if len(scene_ids) != 1 or None in scene_ids:
            raise ValueError("One RT solve identity cannot span multiple scene-frame identities")
        source_epochs = []
        for member in members:
            if member.sample_offset is None:
                raise ValueError("Valid sensing slots require sample_offset")
            source_epochs.append(
                member.time_s
                if sample_period_s is None
                else member.time_s - member.sample_offset * sample_period_s
            )
        source_epoch_s = source_epochs[0]
        if any(
            not math.isclose(
                value,
                source_epoch_s,
                rel_tol=0.0,
                abs_tol=max(1e-12, abs(source_epoch_s) * 1e-12),
            )
            for value in source_epochs[1:]
        ):
            raise ValueError("Sensing members disagree on their RT solve/source epoch")
        epochs.append(
            SensingDiagramRtEpoch(
                solve_id=solve_id,
                scene_frame_id=next(iter(scene_ids)),  # type: ignore[arg-type]
                source_epoch_s=source_epoch_s,
                member_indices=tuple(member.index for member in members),
            )
        )
    epochs.sort(key=lambda epoch: (epoch.source_epoch_s, epoch.solve_id))

    if valid_slots:
        if mode == COHERENT_MODE:
            if len(epochs) != 1:
                raise ValueError("A coherent CPI must derive from exactly one RT solve")
            if any(not slot.phase_coherent for slot in valid_slots):
                raise ValueError("Every observed coherent-CPI slot must be phase coherent")
            offsets = tuple(slot.sample_offset for slot in valid_slots)
            if offsets != tuple(range(len(valid_slots))):
                raise ValueError("Coherent-CPI sample offsets must be contiguous from zero")
        elif mode == SEQUENTIAL_MODE:
            if len(epochs) != len(valid_slots):
                raise ValueError("Sequential STFT requires one independent RT solve per slot")
            if any(slot.phase_coherent for slot in valid_slots):
                raise ValueError("Sequential STFT slots must not claim intra-solve coherence")
            if any(slot.sample_offset != 0 for slot in valid_slots):
                raise ValueError("Sequential STFT source samples require sample_offset=0")
        else:
            if len(valid_slots) != 1 or len(epochs) != 1:
                raise ValueError("Analytic oracle acquisition requires one snapshot/RT solve")
            if valid_slots[0].phase_coherent:
                raise ValueError("Analytic oracle snapshot must not claim phase coherence")

    completeness = _required_string(
        acquisition,
        "completeness",
        path="sensing.temporal.acquisition",
    )
    temporal_result = temporal.get("result")
    if temporal_result is not None and not isinstance(temporal_result, Mapping):
        raise ValueError("sensing.temporal.result must be a mapping when present")
    result_reference_time_s = (
        _finite_float(
            temporal_result.get("reference_time_s"),
            path="sensing.temporal.result.reference_time_s",
        )
        if isinstance(temporal_result, Mapping)
        else None
    )
    result_reference_policy = (
        _required_string(
            temporal_result,
            "reference_time_policy",
            path="sensing.temporal.result",
        )
        if isinstance(temporal_result, Mapping)
        else None
    )
    processing_recipe_id = (
        _required_string(
            temporal_result,
            "processing_recipe_id",
            path="sensing.temporal.result",
        )
        if isinstance(temporal_result, Mapping)
        else None
    )
    processing_run_id = (
        _required_string(
            temporal_result,
            "processing_run_id",
            path="sensing.temporal.result",
        )
        if isinstance(temporal_result, Mapping)
        else None
    )
    if result_reference_time_s is not None:
        if not valid_slots or not epochs:
            raise ValueError("A result-bearing sensing acquisition requires valid RT source slots")
        if mode == SEQUENTIAL_MODE:
            expected_result_policy = SENSING_DEFAULT_REFERENCE_POLICY
            expected_result_time_s = default_reference_time_s
        else:
            expected_result_policy = "solve_start"
            expected_result_time_s = epochs[0].source_epoch_s
        if result_reference_policy != expected_result_policy:
            raise ValueError(
                f"{mode} result reference policy must equal {expected_result_policy!r}"
            )
        _require_matching_time(
            result_reference_time_s,
            expected_result_time_s,
            path="sensing.temporal.result.reference_time_s",
            meaning=f"the {expected_result_policy} epoch",
        )
    return SensingDiagramAcquisition(
        acquisition_id=_required_string(
            acquisition,
            "acquisition_id",
            path="sensing.temporal.acquisition",
        ),
        completeness=completeness,
        slots=tuple(slots),
        rt_epochs=tuple(epochs),
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        expected_sample_count=persisted_expected_count,
        default_reference_policy=default_reference_policy,
        default_reference_time_s=default_reference_time_s,
        result_reference_time_s=result_reference_time_s,
        result_reference_policy=result_reference_policy,
        processing_run_id=processing_run_id,
        processing_recipe_id=processing_recipe_id,
        slow_time_mean_removal_applied=None,
    )


def _primary_range_doppler_descriptor(
    sensing_data: Mapping[str, Any],
    *,
    mode: str,
    result_expected: bool,
) -> Mapping[str, Any] | None:
    """Return the exact primary RD descriptor without loading numeric arrays."""

    products = sensing_data.get("products")
    if not isinstance(products, Mapping):
        if result_expected:
            raise ValueError("A result-bearing sensing event requires typed products")
        return None
    expected_representation = "power_linear" if mode == ANALYTIC_ORACLE_MODE else "complex_voltage"
    matches: list[Mapping[str, Any]] = []
    for descriptor in products.values():
        if not isinstance(descriptor, Mapping):
            continue
        axes = descriptor.get("axes")
        axis_names = (
            tuple(axis.get("name") if isinstance(axis, Mapping) else None for axis in axes)
            if isinstance(axes, list)
            else ()
        )
        if (
            descriptor.get("kind") == "range_doppler"
            and descriptor.get("variant") == "primary"
            and descriptor.get("representation") == expected_representation
            and axis_names == ("pair", "doppler_frequency", "range")
        ):
            matches.append(descriptor)
    if not matches:
        if result_expected:
            raise ValueError("A result-bearing sensing event has no primary typed RD product")
        return None
    if len(matches) != 1:
        raise ValueError("A sensing event has ambiguous primary typed RD products")
    return matches[0]


def _validate_processing_descriptor(
    descriptor: Mapping[str, Any] | None,
    *,
    mode: str,
    delay_taper: str,
    doppler_taper: str,
    clutter_enabled: bool,
    clutter_window: int,
    prf_hz: float,
    cfr_response: Mapping[str, Any] | None = None,
    fft_size_range: int | None = None,
) -> tuple[_ProcessingDescriptorSemantics | None, bool | None]:
    if descriptor is None:
        return (None, None)
    processing = _required_mapping(descriptor, "processing", path="sensing.products.primary_rd")
    if mode == ANALYTIC_ORACLE_MODE:
        if descriptor.get("evidence_role") != "oracle_assisted":
            raise ValueError("Analytic oracle RD products must be labeled oracle_assisted")
        if processing.get("model") != "analytic_path_power_oracle_nonwaveform":
            raise ValueError("Analytic oracle RD product has an unexpected processing model")
        if processing.get("doppler_assignment") != "nearest_bin":
            raise ValueError(
                "Analytic oracle RD product must declare nearest-bin Doppler assignment"
            )
        if (
            "slow_time_mean_removal_policy" in processing
            or "slow_time_mean_removal_applied" in processing
        ):
            raise ValueError("Analytic oracle RD products must not claim slow-time mean removal")
        expected_policy = (
            DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF
            if clutter_enabled
            else DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED
        )
        if processing.get("direct_doppler_clutter_filter_policy") != expected_policy:
            raise ValueError(
                "Analytic oracle RD processing.direct_doppler_clutter_filter_policy "
                f"must equal the executed value {expected_policy!r}"
            )
        cutoff_hz = _finite_float(
            processing.get("direct_doppler_clutter_cutoff_hz"),
            path=("sensing.products.primary_rd.processing." "direct_doppler_clutter_cutoff_hz"),
        )
        expected_cutoff_hz = prf_hz / (2.0 * clutter_window) if clutter_enabled else 0.0
        if cutoff_hz < 0.0 or not math.isclose(
            cutoff_hz,
            expected_cutoff_hz,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Analytic oracle RD processing.direct_doppler_clutter_cutoff_hz "
                f"must equal the executed value {expected_cutoff_hz!r}"
            )
        return (
            _ProcessingDescriptorSemantics(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                expected_policy,
                expected_cutoff_hz,
            ),
            None,
        )

    if (
        "direct_doppler_clutter_filter_policy" in processing
        or "direct_doppler_clutter_cutoff_hz" in processing
    ):
        raise ValueError("Sampled RD products must not claim direct path-Doppler filtering")

    expected: dict[str, Any] = {
        "delay_projection": DELAY_PROJECTION_POINT_BIN_FLOOR,
        "range_processing": RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
        "range_fft_applied": False,
        "delay_taper": delay_taper,
        "doppler_taper": doppler_taper,
        "fft_axis": "slow_time",
        "fft_normalization": DOPPLER_FFT_NORMALIZATION_UNITARY,
        "window_normalization": WINDOW_NORMALIZATION_ENERGY,
    }
    if cfr_response is not None:
        expected.update(
            model="cfr_ifft",
            model_version=1,
            delay_projection="cfr_ifft",
            range_processing="finite_bandwidth_delay_response",
            range_fft_applied=True,
            range_transform="ifft",
            range_transform_axis="frequency",
            range_transform_normalization="unit_coherent_peak",
            frequency_window=cfr_response["frequency_window"],
            num_frequency_bins=cfr_response["num_frequency_bins"],
            fft_size_range=fft_size_range,
        )
    for key, value in expected.items():
        if processing.get(key) != value:
            raise ValueError(
                f"Primary typed RD processing.{key} must equal the executed value {value!r}"
            )
    mean_removal_policy = processing.get("slow_time_mean_removal_policy")
    if mean_removal_policy not in {
        _SLOW_TIME_MEAN_REMOVAL_POLICY,
        "window_weighted_mean",
        "disabled",
    }:
        raise ValueError(
            "Primary typed RD processing.slow_time_mean_removal_policy must equal "
            f"{_SLOW_TIME_MEAN_REMOVAL_POLICY!r}"
        )
    mean_removal_applied = _required_bool(
        processing.get("slow_time_mean_removal_applied"),
        path="sensing.products.primary_rd.processing.slow_time_mean_removal_applied",
    )
    if mean_removal_policy == "disabled" and mean_removal_applied:
        raise ValueError("Disabled mean-removal policy cannot be applied")
    return (
        _ProcessingDescriptorSemantics(
            expected["delay_projection"],
            expected["range_processing"],
            expected["range_fft_applied"],
            "slow_time",
            DOPPLER_FFT_NORMALIZATION_UNITARY,
            WINDOW_NORMALIZATION_ENERGY,
            mean_removal_policy,
            None,
            None,
        ),
        mean_removal_applied,
    )


def _validate_acquisition_overlap(
    acquisitions: Sequence[SensingDiagramAcquisition],
    *,
    mode: str,
    hop: int,
    overlap: int,
) -> float | None:
    complete = [
        acquisition
        for acquisition in acquisitions
        if acquisition.result_reference_time_s is not None
        and len(acquisition.valid_slots) == len(acquisition.slots)
    ]
    if len(complete) < 2:
        return None
    first, second = complete[:2]
    first_sources = tuple(slot.composite_source_identity for slot in first.valid_slots)
    second_sources = tuple(slot.composite_source_identity for slot in second.valid_slots)
    if any(identity is None for identity in (*first_sources, *second_sources)):
        raise ValueError("Complete sensing acquisitions require composite source identity")
    if mode == SEQUENTIAL_MODE:
        shared = set(first_sources) & set(second_sources)
        if len(shared) != overlap or first_sources[hop:] != second_sources[:overlap]:
            raise ValueError("Sequential STFT acquisitions disagree with the resolved overlap/hop")
    elif mode == COHERENT_MODE and set(first_sources) & set(second_sources):
        raise ValueError("Non-overlapping coherent CPIs must not reuse source samples")
    first_start = first.valid_slots[0].time_s
    second_start = second.valid_slots[0].time_s
    if second_start <= first_start:
        raise ValueError("Sensing acquisition start times must be strictly increasing")
    return second_start - first_start


def _windows_for_mode(mode: str, *, length: int, hop: int) -> tuple[SensingDiagramWindow, ...]:
    if mode == COHERENT_MODE:
        return (
            SensingDiagramWindow("CPI 1", 0, length, length - 1),
            SensingDiagramWindow("CPI 2", length, length, 2 * length - 1),
        )
    if mode == SEQUENTIAL_MODE:
        return (
            SensingDiagramWindow("STFT W1", 0, length, length - 1),
            SensingDiagramWindow("STFT W2", hop, length, length + hop - 1),
        )
    return (SensingDiagramWindow("oracle snapshot", 0, 1, 0),)


def _rt_solve_samples(mode: str, *, length: int, hop: int) -> tuple[int, ...]:
    if mode == COHERENT_MODE:
        return (0, length)
    if mode == SEQUENTIAL_MODE:
        return tuple(range(length + hop))
    return (0,)


def _processing_stages(
    *,
    mode: str,
    sample_source: str,
    phase_contract: str,
    fft_size_range: int,
    fft_size_doppler: int,
    bandwidth_hz: float,
    delay_sampling_frequency_hz: float,
    delay_taper: str | None,
    doppler_taper: str | None,
    slow_time_mean_removal_policy: str | None,
    slow_time_mean_removal_applied: bool | None,
    direct_doppler_clutter_filter_policy: str | None,
    direct_doppler_clutter_cutoff_hz: float | None,
    clutter_enabled: bool,
    output_detections: bool,
    cfar_type: str | None,
    cfr_response: Mapping[str, Any] | None = None,
) -> tuple[SensingDiagramStage, ...]:
    if mode == ANALYTIC_ORACLE_MODE:
        if direct_doppler_clutter_filter_policy == (
            DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF
        ):
            if direct_doppler_clutter_cutoff_hz is None or direct_doppler_clutter_cutoff_hz <= 0.0:
                raise ValueError("Enabled oracle direct Doppler filtering requires a cutoff")
            direct_filter_detail = (
                "exclude otherwise-valid paths with "
                f"|f_D| < {_format_rate(direct_doppler_clutter_cutoff_hz)} "
                "(paths exactly at the cutoff are retained)"
            )
        elif (
            direct_doppler_clutter_filter_policy == DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED
            and direct_doppler_clutter_cutoff_hz == 0.0
        ):
            direct_filter_detail = "direct path-Doppler clutter filter disabled"
        else:
            raise ValueError("Analytic oracle direct Doppler filter semantics are invalid")
        stages = [
            SensingDiagramStage(
                "RT paths",
                "one geometry solve; read the complete per-path complex-gain, delay, "
                "and paths.doppler set",
                _BLUE,
            ),
            SensingDiagramStage(
                "Direct path binning",
                f"{direct_filter_detail}; deposit retained path power into range/Doppler "
                "cells with nearest-bin Doppler assignment; no CPI or STFT",
                _PURPLE,
            ),
            SensingDiagramStage(
                "Oracle RD power",
                "diagnostic non-waveform map; no Doppler FFT and no window overlap",
                _RED,
            ),
        ]
    else:
        acquisition_detail = (
            f"one RT solve → N={fft_size_doppler} complex CIR offsets; "
            "phase coherent within that solve"
            if mode == COHERENT_MODE
            else f"N={fft_size_doppler} independent solves → one complex CIR row "
            "per PRI; cross-solve phase is experimental"
        )
        if slow_time_mean_removal_policy not in {
            _SLOW_TIME_MEAN_REMOVAL_POLICY,
            "window_weighted_mean",
            "disabled",
        }:
            raise ValueError(
                "Sampled sensing processing requires the persisted mean-removal policy"
            )
        if slow_time_mean_removal_applied is None:
            raise ValueError("Sampled sensing processing requires an applied mean-removal state")
        if mode == COHERENT_MODE:
            taper_name = "CPI tapers"
            taper_detail = (
                f"delay-bin weighting={delay_taper}; CPI Doppler window={doppler_taper}; "
                "energy normalized; coherent strategy does not consume the configured "
                "sliding-baseline switch"
            )
        else:
            taper_name = "STFT tapers / clutter"
            taper_detail = (
                f"delay-bin weighting={delay_taper}; rolling STFT window={doppler_taper}; "
                "energy normalized; configured sliding-baseline removal "
                f"{'enabled' if clutter_enabled else 'disabled'}"
            )
        stages = [
            SensingDiagramStage(
                "RT → CIR paths",
                acquisition_detail,
                _BLUE,
            ),
            SensingDiagramStage(
                "Delay projection",
                f"floor(τ·F_delay), F_delay={_format_rate(delay_sampling_frequency_hz)} "
                f"(configured B={_format_rate(bandwidth_hz)}), point-delay sampling → "
                f"[N,{fft_size_range}] complex delay-bin grid; no range FFT",
                _CYAN,
            ),
            SensingDiagramStage(
                taper_name,
                taper_detail,
                _GOLD,
            ),
            SensingDiagramStage(
                "Slow-time DC removal",
                (
                    "persisted policy: no mean removal; displayed result: "
                    if slow_time_mean_removal_policy == "disabled"
                    else (
                        "persisted policy: subtract the Doppler-taper-weighted mean in every complete CPI, including constant rows; displayed result: "
                        if slow_time_mean_removal_policy == "window_weighted_mean"
                        else "persisted policy: subtract every delay bin's slow-time mean only when "
                        "the sampled matrix has any row-to-row variation; displayed result: "
                    )
                )
                + f"{'applied' if slow_time_mean_removal_applied else 'not applied'}",
                _RED,
            ),
            SensingDiagramStage(
                "CPI Doppler DFT" if mode == COHERENT_MODE else "Rolling STFT DFT",
                (
                    f"unitary {fft_size_doppler}-point CPI Doppler DFT + fftshift "
                    "along slow time only"
                    if mode == COHERENT_MODE
                    else f"unitary {fft_size_doppler}-point rolling STFT DFT + "
                    "fftshift along slow time only"
                ),
                _PURPLE,
            ),
            SensingDiagramStage(
                "Range–Doppler power",
                "|complex RD|² on persisted range and Doppler coordinates",
                _GREEN,
            ),
        ]
    if cfr_response is not None:
        stages[0] = SensingDiagramStage(
            "RT → coherent paths",
            f"one frozen-geometry solve → N={fft_size_doppler} coherent coefficient/delay offsets; rigid path Doppler",
            _BLUE,
        )
        stages[1] = SensingDiagramStage(
            "CFR → delay IFFT",
            f"{cfr_response['num_frequency_bins']} frequency samples over B={_format_rate(bandwidth_hz)}; "
            f"frequency taper={cfr_response['frequency_window']}; {fft_size_range}-point IFFT; "
            "unit coherent peak normalization; no point-bin projection",
            _CYAN,
        )
        stages[2] = SensingDiagramStage(
            "CPI Doppler taper",
            f"Doppler window={doppler_taper}; energy normalized; delay taper={delay_taper} (identity); "
            "frequency taper applied before IFFT",
            _GOLD,
        )
        # The coherent processor removes the mean before Doppler tapering.
        stages[2], stages[3] = stages[3], stages[2]
    if output_detections:
        stages.append(
            SensingDiagramStage(
                "Detection output",
                f"{(cfar_type or 'configured').upper()}-CFAR → NMS → DBSCAN → SNR filter",
                _RED,
            )
        )
    return tuple(stages)


def build_sensing_diagram_model(
    sensing_data: Mapping[str, Any],
    *,
    sensing_event_history: Sequence[Mapping[str, Any]] | None = None,
) -> SensingDiagramModel:
    """Build a strict diagram model from persisted schema-v4 sensing events.

    ``sensing_event_history`` may contain bounded metadata-only snapshots. The
    latest full payload remains the configuration authority; history provides
    observed first/next acquisition membership without retaining numeric arrays.
    """

    assert_supported_sensing_schema(sensing_data)
    config = _required_mapping(sensing_data, "config_resolved", path="sensing")
    contract_version = _required_int(
        config,
        "contract_version",
        path="sensing.config_resolved",
    )
    acquisition_plan = _required_mapping(
        config,
        "acquisition",
        path="sensing.config_resolved",
    )
    processing_plan = _required_mapping(
        config,
        "processing",
        path="sensing.config_resolved",
    )
    recipe = _required_mapping(
        config,
        "processing_recipe",
        path="sensing.config_resolved",
    )

    mode = _required_string(config, "processing_mode", path="sensing.config_resolved")
    if mode not in _MODE_TITLES:
        raise ValueError(f"Unsupported sensing diagram mode {mode!r}")
    if processing_plan.get("mode") != mode:
        raise ValueError("Resolved sensing processing mode fields disagree")
    strategy = _required_string(
        processing_plan,
        "strategy",
        path="sensing.config_resolved.processing",
    )
    if sensing_data.get("processing_strategy") != strategy:
        raise ValueError("Persisted sensing strategy disagrees with config_resolved.processing")

    sample_source = _required_string(
        acquisition_plan,
        "sample_source",
        path="sensing.config_resolved.acquisition",
    )
    phase_contract = _required_string(
        acquisition_plan,
        "phase_contract",
        path="sensing.config_resolved.acquisition",
    )
    sample_period_s = _optional_positive_float(
        acquisition_plan,
        "sample_period_s",
        path="sensing.config_resolved.acquisition",
    )
    sample_rate_hz = _optional_positive_float(
        acquisition_plan,
        "sample_rate_hz",
        path="sensing.config_resolved.acquisition",
    )
    if (sample_period_s is None) != (sample_rate_hz is None):
        raise ValueError("Resolved sensing sample period and rate must be present together")
    if (
        sample_period_s is not None
        and sample_rate_hz is not None
        and not math.isclose(
            sample_period_s * sample_rate_hz,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Resolved sensing sample period and rate are not reciprocal")
    if mode == ANALYTIC_ORACLE_MODE and sample_period_s is not None:
        raise ValueError("Analytic oracle diagrams must not claim a sampled PRI")
    if mode != ANALYTIC_ORACLE_MODE and sample_period_s is None:
        raise ValueError("Sampled sensing modes require an exact physical PRI")

    length = _required_int(
        acquisition_plan,
        "cpi_length_samples",
        path="sensing.config_resolved.acquisition",
    )
    hop = _required_int(
        acquisition_plan,
        "cpi_hop_samples",
        path="sensing.config_resolved.acquisition",
    )
    overlap = _required_int(
        acquisition_plan,
        "overlap_samples",
        path="sensing.config_resolved.acquisition",
        minimum=0,
    )
    if hop > length or overlap != length - hop:
        raise ValueError("Resolved sensing overlap must equal CPI length minus hop")
    if mode == COHERENT_MODE and (hop != length or overlap != 0):
        raise ValueError("Coherent CPI requires H=N and O=0")
    rt_stride = _required_int(
        acquisition_plan,
        "rt_solve_stride_scene_steps",
        path="sensing.config_resolved.acquisition",
    )
    result_stride = _required_int(
        processing_plan,
        "result_stride_samples",
        path="sensing.config_resolved.processing",
    )
    if result_stride != hop:
        raise ValueError("Resolved sensing result stride must equal CPI/window hop")
    fft_size_doppler = _required_int(
        processing_plan,
        "fft_size_doppler",
        path="sensing.config_resolved.processing",
    )
    if fft_size_doppler != _required_int(
        config,
        "fft_size_doppler",
        path="sensing.config_resolved",
    ):
        raise ValueError("Resolved sensing Doppler FFT sizes disagree")
    if mode != ANALYTIC_ORACLE_MODE and length != fft_size_doppler:
        raise ValueError("Sampled sensing CPI/window length must equal Doppler FFT length")

    fft_size_range = _required_int(
        config,
        "fft_size_range",
        path="sensing.config_resolved",
    )
    bandwidth_hz = _required_positive_float(
        config,
        "bandwidth_hz",
        path="sensing.config_resolved",
    )
    prf_hz = _required_positive_float(
        config,
        "prf_hz",
        path="sensing.config_resolved",
    )
    delay_sampling_frequency_hz = _required_positive_float(
        config,
        "sampling_frequency",
        path="sensing.config_resolved",
    )
    range_mode = _required_string(config, "range_mode", path="sensing.config_resolved")
    if range_mode not in {"monostatic", "bistatic"}:
        raise ValueError("Sensing diagram range_mode must be monostatic or bistatic")

    evidence_status = _required_string(
        processing_plan,
        "evidence_status",
        path="sensing.config_resolved.processing",
    )
    evidence_label, evidence_detail, evidence_color = _MODE_STATUS[mode]
    cfr_response = None
    range_response = config.get("range_response", {"model": "point_bin"})
    if not isinstance(range_response, Mapping):
        raise ValueError("Sensing diagram range_response must be a mapping")
    if range_response.get("model", "point_bin") == "cfr_ifft":
        if (
            mode != COHERENT_MODE
            or processing_plan.get("product_semantics") != "cfr_ifft_slow_time_fft"
        ):
            raise ValueError("CFR sensing diagrams require the bounded coherent CPI contract")
        if recipe.get("range_response") != range_response:
            raise ValueError("CFR sensing diagram range_response disagrees with processing recipe")
        frequency_window = _required_string(
            range_response, "frequency_window", path="range_response"
        )
        if frequency_window not in {"hann", "boxcar"}:
            raise ValueError("CFR sensing diagram requires a supported frequency window")
        _required_int(range_response, "num_frequency_bins", path="range_response")
        cfr_response = range_response
        evidence_label, evidence_detail, evidence_color = (
            "BOUNDED",
            "finite-bandwidth coherent processing",
            "#147D64",
        )
        expected_evidence = "finite_bandwidth_bounded_coherent"
    elif range_response.get("model", "point_bin") == "point_bin":
        expected_evidence = _MODE_EVIDENCE_STATUS[mode]
    else:
        raise ValueError("Unsupported sensing diagram range response")
    if evidence_status != expected_evidence:
        raise ValueError(f"{mode} evidence_status must equal {expected_evidence!r}")

    delay_taper_value = _recipe_value(config, recipe, "window_range")
    doppler_taper_value = _recipe_value(config, recipe, "window_doppler")
    if cfr_response is not None and delay_taper_value != "boxcar":
        raise ValueError("CFR sensing diagrams require identity boxcar delay taper")
    if not isinstance(delay_taper_value, str) or not delay_taper_value:
        raise ValueError("Sensing diagram delay taper must be a non-empty string")
    if not isinstance(doppler_taper_value, str) or not doppler_taper_value:
        raise ValueError("Sensing diagram Doppler taper must be a non-empty string")
    if delay_taper_value.lower() not in _VALID_WINDOW_NAMES:
        raise ValueError(f"Unsupported executed delay taper {delay_taper_value!r}")
    if doppler_taper_value.lower() not in _VALID_WINDOW_NAMES:
        raise ValueError(f"Unsupported executed Doppler taper {doppler_taper_value!r}")
    output_detections = _required_bool(
        _recipe_value(config, recipe, "output_detections"),
        path="sensing.config_resolved.processing_recipe.output_detections",
    )
    clutter_enabled = _required_bool(
        _recipe_value(config, recipe, "clutter_removal_enabled"),
        path="sensing.config_resolved.processing_recipe.clutter_removal_enabled",
    )
    clutter_window_value = _recipe_value(config, recipe, "clutter_removal_window")
    if (
        isinstance(clutter_window_value, bool)
        or not isinstance(clutter_window_value, int)
        or clutter_window_value < 1
    ):
        raise ValueError(
            "sensing.config_resolved.processing_recipe.clutter_removal_window "
            "must be a positive integer"
        )
    clutter_window = int(clutter_window_value)
    cfar_type_value = _recipe_value(config, recipe, "cfar_type", required=False)
    if cfar_type_value is not None and not isinstance(cfar_type_value, str):
        raise ValueError("Sensing diagram CFAR type must be a string when present")

    history_events = list(sensing_event_history or ())
    history_has_result = any(
        isinstance(event.get("temporal"), Mapping)
        and isinstance(event["temporal"].get("result"), Mapping)
        for event in history_events
    )
    display_latest = not history_events or not history_has_result
    candidate_events = [(event, True) for event in history_events]
    candidate_events.append((sensing_data, display_latest))
    acquisitions_by_id: dict[str, SensingDiagramAcquisition] = {}
    validated_acquisitions_by_id: dict[str, SensingDiagramAcquisition] = {}
    pair_id: str | None = None
    acquisition_config_id: str | None = None
    processing_recipe_id: str | None = None
    descriptor_semantics: _ProcessingDescriptorSemantics | None = None
    for event, display_event in candidate_events:
        assert_supported_sensing_schema(event)
        event_config = _required_mapping(event, "config_resolved", path="sensing")
        if (
            event_config.get("processing_mode") != mode
            or event_config.get("acquisition") != acquisition_plan
            or event_config.get("processing") != processing_plan
            or event.get("processing_strategy") != strategy
            or event_config.get("range_response") != config.get("range_response")
        ):
            raise ValueError("Sensing diagram history mixes incompatible execution contracts")
        event_acquisition = _extract_diagram_acquisition(
            event,
            mode=mode,
            expected_count=length,
            sample_period_s=sample_period_s,
        )
        temporal = _required_mapping(event, "temporal", path="sensing")
        temporal_acquisition = _required_mapping(
            temporal,
            "acquisition",
            path="sensing.temporal",
        )
        event_pair_id = _required_string(
            temporal_acquisition,
            "pair_id",
            path="sensing.temporal.acquisition",
        )
        event_config_id = _required_string(
            temporal_acquisition,
            "acquisition_config_id",
            path="sensing.temporal.acquisition",
        )
        if pair_id is None:
            pair_id = event_pair_id
            acquisition_config_id = event_config_id
        elif pair_id != event_pair_id or acquisition_config_id != event_config_id:
            raise ValueError("Sensing diagram history changes pair or acquisition configuration")
        descriptor = _primary_range_doppler_descriptor(
            event,
            mode=mode,
            result_expected=event_acquisition.result_reference_time_s is not None,
        )
        event_semantics, mean_removal_applied = _validate_processing_descriptor(
            descriptor,
            mode=mode,
            delay_taper=delay_taper_value,
            doppler_taper=doppler_taper_value,
            clutter_enabled=clutter_enabled,
            clutter_window=clutter_window,
            prf_hz=prf_hz,
            cfr_response=cfr_response,
            fft_size_range=fft_size_range,
        )
        event_acquisition = replace(
            event_acquisition,
            slow_time_mean_removal_applied=mean_removal_applied,
        )
        if descriptor is not None:
            if event_acquisition.processing_run_id is None:
                raise ValueError("A result-bearing acquisition requires processing_run_id")
            if event_acquisition.processing_recipe_id is None:
                raise ValueError("A result-bearing acquisition requires processing_recipe_id")
            lineage = _required_mapping(
                descriptor,
                "lineage",
                path="sensing.products.primary_rd",
            )
            descriptor_processing_run_id = _required_string(
                lineage,
                "processing_run_id",
                path="sensing.products.primary_rd.lineage",
            )
            descriptor_processing_recipe_id = _required_string(
                lineage,
                "processing_recipe_id",
                path="sensing.products.primary_rd.lineage",
            )
            if (
                descriptor_processing_run_id != event_acquisition.processing_run_id
                or descriptor_processing_recipe_id != event_acquisition.processing_recipe_id
            ):
                raise ValueError(
                    "Primary RD descriptor processing identity disagrees with temporal.result"
                )
            if processing_recipe_id is None:
                processing_recipe_id = event_acquisition.processing_recipe_id
            elif processing_recipe_id != event_acquisition.processing_recipe_id:
                raise ValueError("Sensing diagram history changes processing recipe identity")
            if descriptor_semantics is None:
                descriptor_semantics = event_semantics
            elif descriptor_semantics != event_semantics:
                raise ValueError("Sensing diagram history changes primary RD processing semantics")
        existing = validated_acquisitions_by_id.get(event_acquisition.acquisition_id)
        if existing is not None and existing != event_acquisition:
            raise ValueError("Duplicate sensing acquisition identity has conflicting metadata")
        if existing is None:
            validated_acquisitions_by_id[event_acquisition.acquisition_id] = event_acquisition
        if display_event and event_acquisition.acquisition_id not in acquisitions_by_id:
            acquisitions_by_id[event_acquisition.acquisition_id] = event_acquisition

    acquisitions = tuple(acquisitions_by_id.values())
    if not acquisitions:
        raise ValueError("Sensing diagram requires at least one exact acquisition")
    if descriptor_semantics is None:
        raise ValueError("Sensing diagram requires at least one result-bearing acquisition")
    observed_revisit_interval_s = _validate_acquisition_overlap(
        acquisitions,
        mode=mode,
        hop=hop,
        overlap=overlap,
    )
    delay_projection = descriptor_semantics.delay_projection
    range_processing = descriptor_semantics.range_processing
    range_fft_applied = descriptor_semantics.range_fft_applied
    fft_axis = descriptor_semantics.fft_axis
    fft_normalization = descriptor_semantics.fft_normalization
    window_normalization = descriptor_semantics.window_normalization
    slow_time_mean_removal_policy = descriptor_semantics.slow_time_mean_removal_policy
    direct_doppler_clutter_filter_policy = descriptor_semantics.direct_doppler_clutter_filter_policy
    direct_doppler_clutter_cutoff_hz = descriptor_semantics.direct_doppler_clutter_cutoff_hz
    if mode == ANALYTIC_ORACLE_MODE:
        displayed_delay_taper = None
        displayed_doppler_taper = None
    else:
        displayed_delay_taper = delay_taper_value
        displayed_doppler_taper = doppler_taper_value

    primary_acquisition = next(
        (
            acquisition
            for acquisition in reversed(acquisitions)
            if acquisition.result_reference_time_s is not None
        ),
        acquisitions[-1],
    )
    result_reference_policy = primary_acquisition.result_reference_policy
    completeness = primary_acquisition.completeness
    observed_count = len(primary_acquisition.valid_slots)
    expected_count = primary_acquisition.expected_sample_count
    if pair_id is None or acquisition_config_id is None:
        raise ValueError("Sensing diagram could not resolve acquisition identity")

    stages = _processing_stages(
        mode=mode,
        sample_source=sample_source,
        phase_contract=phase_contract,
        fft_size_range=fft_size_range,
        fft_size_doppler=fft_size_doppler,
        bandwidth_hz=bandwidth_hz,
        delay_sampling_frequency_hz=delay_sampling_frequency_hz,
        delay_taper=displayed_delay_taper,
        doppler_taper=displayed_doppler_taper,
        slow_time_mean_removal_policy=slow_time_mean_removal_policy,
        slow_time_mean_removal_applied=primary_acquisition.slow_time_mean_removal_applied,
        direct_doppler_clutter_filter_policy=direct_doppler_clutter_filter_policy,
        direct_doppler_clutter_cutoff_hz=direct_doppler_clutter_cutoff_hz,
        clutter_enabled=clutter_enabled,
        output_detections=output_detections,
        cfar_type=cfar_type_value,
        cfr_response=cfr_response,
    )
    return SensingDiagramModel(
        schema_version=SENSING_SCHEMA_VERSION,
        contract_version=contract_version,
        mode=mode,
        mode_title=_MODE_TITLES[mode],
        strategy=strategy,
        evidence_status=evidence_label,
        evidence_detail=f"{evidence_detail} · {evidence_status}",
        evidence_color=evidence_color,
        sample_source=sample_source,
        phase_contract=phase_contract,
        sample_period_s=sample_period_s,
        sample_rate_hz=sample_rate_hz,
        cpi_length_samples=length,
        cpi_hop_samples=hop,
        overlap_samples=overlap,
        overlap_fraction=overlap / length,
        rt_solve_stride_scene_steps=rt_stride,
        result_stride_samples=result_stride,
        fft_size_range=fft_size_range,
        fft_size_doppler=fft_size_doppler,
        bandwidth_hz=bandwidth_hz,
        delay_sampling_frequency_hz=delay_sampling_frequency_hz,
        range_mode=range_mode,
        delay_projection=delay_projection,
        range_processing=range_processing,
        range_fft_applied=range_fft_applied,
        delay_taper=displayed_delay_taper,
        doppler_taper=displayed_doppler_taper,
        fft_axis=fft_axis,
        fft_normalization=fft_normalization,
        window_normalization=window_normalization,
        slow_time_mean_removal_policy=slow_time_mean_removal_policy,
        slow_time_mean_removal_applied=primary_acquisition.slow_time_mean_removal_applied,
        direct_doppler_clutter_filter_policy=direct_doppler_clutter_filter_policy,
        direct_doppler_clutter_cutoff_hz=direct_doppler_clutter_cutoff_hz,
        clutter_removal_enabled=clutter_enabled,
        output_detections=output_detections,
        cfar_type=cfar_type_value,
        pair_id=pair_id,
        acquisition_config_id=acquisition_config_id,
        processing_run_id=primary_acquisition.processing_run_id,
        processing_recipe_id=processing_recipe_id,
        acquisition_completeness=completeness,
        observed_sample_count=observed_count,
        expected_sample_count=expected_count,
        result_reference_policy=result_reference_policy,
        acquisitions=acquisitions,
        observed_revisit_interval_s=observed_revisit_interval_s,
        windows=_windows_for_mode(mode, length=length, hop=hop),
        rt_solve_samples=_rt_solve_samples(mode, length=length, hop=hop),
        processing_stages=stages,
    )


def sensing_diagram_manifest(model: SensingDiagramModel) -> dict[str, Any]:
    """Return the JSON manifest stored beside a rendered diagram."""

    manifest = asdict(model)
    manifest["cpi_duration_s"] = model.cpi_duration_s
    manifest["sample_center_span_s"] = model.sample_center_span_s
    manifest["hop_duration_s"] = model.hop_duration_s
    manifest["note"] = (
        "Chirp/PRI slots are symbolic timing slots. ORCHAV persists CIR/path samples; "
        "it does not synthesize an RF waveform, matched filter, pulse-compress, or "
        "apply a range FFT. For sampled modes, N·PRI is the symbolic window extent, "
        "while (N−1)·PRI is the first-to-last sample-center span. The result-bearing "
        "output is attached to its final observed sample; no processing-completion or "
        "result-availability timestamp is persisted."
    )
    if model.range_processing == "finite_bandwidth_delay_response":
        manifest["note"] = (
            "Chirp/PRI slots are symbolic timing slots. Retained frequency observations undergo "
            "a frequency-windowed IFFT to delay followed by the slow-time Doppler FFT. "
            "No RF waveform, matched filter or pulse compression is synthesized. "
            "Geometry is frozen within each CPI; rigid path Doppler does not model intra-CPI range migration. "
            "N·PRI is the symbolic extent; (N−1)·PRI is the sample-center span."
        )
    return manifest


def write_sensing_diagram_manifest(model: SensingDiagramModel, output_path: str | Path) -> Path:
    """Write the exact diagram model as deterministic JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sensing_diagram_manifest(model), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    magnitude = abs(seconds)
    if magnitude >= 1.0:
        return f"{seconds:.3g} s"
    if magnitude >= 1e-3:
        return f"{seconds * 1e3:.3g} ms"
    if magnitude >= 1e-6:
        return f"{seconds * 1e6:.3g} µs"
    return f"{seconds * 1e9:.3g} ns"


def _format_rate(rate_hz: float | None) -> str:
    if rate_hz is None:
        return "n/a"
    if rate_hz >= 1e9:
        return f"{rate_hz / 1e9:.3g} GHz"
    if rate_hz >= 1e6:
        return f"{rate_hz / 1e6:.3g} MHz"
    if rate_hz >= 1e3:
        return f"{rate_hz / 1e3:.3g} kHz"
    return f"{rate_hz:.3g} Hz"


def _panel(ax: Axes, title: str) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.01),
            0.99,
            0.97,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            facecolor=_PANEL,
            edgecolor=_GRID,
            linewidth=1.0,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.025,
        0.93,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=_INK,
    )


def _metric_box(ax: Axes, x: float, width: float, label: str, value: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.05),
            width,
            0.43,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="white",
            edgecolor=_GRID,
            linewidth=0.9,
            transform=ax.transAxes,
        )
    )
    ax.text(x + 0.015, 0.36, label.upper(), fontsize=7.5, color=_MUTED, transform=ax.transAxes)
    ax.text(
        x + 0.015,
        0.16,
        value,
        fontsize=10.5,
        fontweight="bold",
        color=_INK,
        transform=ax.transAxes,
    )


def _draw_slot(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    color: str = _BLUE,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            transform=ax.transAxes,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=_INK,
        transform=ax.transAxes,
    )


def _draw_diamond(ax: Axes, x: float, y: float, *, color: str = _PURPLE) -> None:
    size_x = 0.009
    size_y = 0.025
    ax.add_patch(
        Polygon(
            ((x, y + size_y), (x + size_x, y), (x, y - size_y), (x - size_x, y)),
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            transform=ax.transAxes,
        )
    )


def _draw_result(ax: Axes, x: float, y: float) -> None:
    ax.add_patch(
        Polygon(
            ((x, y + 0.026), (x + 0.012, y - 0.024), (x - 0.012, y - 0.024)),
            closed=True,
            facecolor=_GREEN,
            edgecolor="white",
            linewidth=0.8,
            transform=ax.transAxes,
        )
    )


def _draw_coherent_timeline(ax: Axes, model: SensingDiagramModel) -> None:
    group_width = 0.37
    starts = (0.07, 0.56)
    slot_labels = ("1", "2", "3", "⋯", f"N={model.cpi_length_samples}")
    result_xs: list[float] = []
    for group_index, start in enumerate(starts, start=1):
        ax.add_patch(
            FancyBboxPatch(
                (start - 0.012, 0.42),
                group_width + 0.024,
                0.34,
                boxstyle="round,pad=0.008,rounding_size=0.016",
                facecolor="#E9F0FF",
                edgecolor=_BLUE,
                linewidth=1.0,
                alpha=0.75,
                transform=ax.transAxes,
            )
        )
        ax.text(
            start,
            0.79,
            f"CPI {group_index} · {model.cpi_length_samples} slots",
            fontsize=9.5,
            fontweight="bold",
            color=_BLUE,
            transform=ax.transAxes,
        )
        gap = 0.009
        slot_width = (group_width - gap * 4) / 5
        for index, label in enumerate(slot_labels):
            _draw_slot(
                ax,
                start + index * (slot_width + gap),
                0.51,
                slot_width,
                0.16,
                label,
            )
        _draw_diamond(ax, start, 0.34)
        result_x = start + 4 * (slot_width + gap) + slot_width / 2
        result_xs.append(result_x)
        _draw_result(ax, result_x, 0.34)

    ax.text(0.07, 0.23, "RT solve", fontsize=8, color=_PURPLE, transform=ax.transAxes)
    ax.text(0.56, 0.23, "RT solve", fontsize=8, color=_PURPLE, transform=ax.transAxes)
    for result_x in result_xs:
        ax.text(
            result_x,
            0.23,
            "RD output",
            fontsize=8,
            color=_GREEN,
            ha="center",
            transform=ax.transAxes,
        )
    ax.text(
        0.5,
        0.28,
        "result attached to final observed sample output; reference: "
        f"{model.result_reference_policy or 'unavailable'}",
        fontsize=7.8,
        color=_MUTED,
        ha="center",
        transform=ax.transAxes,
    )
    ax.annotate(
        "",
        xy=(0.145, 0.46),
        xytext=(0.075, 0.46),
        arrowprops={"arrowstyle": "<->", "color": _MUTED, "linewidth": 1.0},
        xycoords=ax.transAxes,
    )
    ax.text(
        0.11,
        0.40,
        f"PRI = {_format_duration(model.sample_period_s)}",
        fontsize=8,
        color=_MUTED,
        ha="center",
        transform=ax.transAxes,
    )
    ax.annotate(
        "",
        xy=(0.56, 0.14),
        xytext=(0.07, 0.14),
        arrowprops={"arrowstyle": "<->", "color": _BLUE, "linewidth": 1.2},
        xycoords=ax.transAxes,
    )
    ax.text(
        0.315,
        0.07,
        (
            f"inter-CPI revisit: observed {_format_duration(model.observed_revisit_interval_s)}"
            if model.observed_revisit_interval_s is not None
            else f"configured hop H=N={model.cpi_hop_samples} · {_format_duration(model.hop_duration_s)}"
        ),
        fontsize=8.5,
        color=_BLUE,
        ha="center",
        transform=ax.transAxes,
    )


def _draw_sequential_timeline(ax: Axes, model: SensingDiagramModel) -> None:
    labels = (
        "1",
        f"H+1={model.cpi_hop_samples + 1}",
        "⋯",
        f"N={model.cpi_length_samples}",
        "⋯",
        f"N+H={model.cpi_length_samples + model.cpi_hop_samples}",
    )
    left = 0.08
    right = 0.92
    width = 0.09
    centers = [left + index * (right - left) / (len(labels) - 1) for index in range(len(labels))]
    for center, label in zip(centers, labels, strict=True):
        _draw_slot(ax, center - width / 2, 0.46, width, 0.15, label, color=_CYAN)
        _draw_diamond(ax, center, 0.35, color=_PURPLE)

    ax.add_patch(
        FancyBboxPatch(
            (centers[0] - 0.055, 0.71),
            centers[3] - centers[0] + 0.11,
            0.095,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="#DDEAFF",
            edgecolor=_BLUE,
            linewidth=1.2,
            transform=ax.transAxes,
        )
    )
    ax.text(
        centers[0],
        0.825,
        "rolling STFT W1: samples 1…N",
        fontsize=9,
        color=_BLUE,
        transform=ax.transAxes,
    )
    ax.add_patch(
        FancyBboxPatch(
            (centers[1] - 0.055, 0.62),
            centers[-1] - centers[1] + 0.11,
            0.075,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="#EAE2FB",
            edgecolor=_PURPLE,
            linewidth=1.2,
            transform=ax.transAxes,
        )
    )
    ax.text(
        (centers[1] + centers[-1]) / 2,
        0.657,
        f"rolling STFT W2: samples H+1…H+N "
        f"({model.cpi_hop_samples + 1}…{model.cpi_length_samples + model.cpi_hop_samples})",
        fontsize=8.5,
        color=_PURPLE,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    _draw_result(ax, centers[3], 0.87)
    _draw_result(ax, centers[-1], 0.87)
    ax.text(
        (centers[3] + centers[-1]) / 2,
        0.90,
        "RD outputs",
        fontsize=8,
        color=_GREEN,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.24,
        (
            "independent RT solve/source epoch every PRI · "
            f"result reference: {model.result_reference_policy or 'unavailable'}"
        ),
        fontsize=8.5,
        color=_PURPLE,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.10,
        (
            f"window N={model.cpi_length_samples} · hop H={model.cpi_hop_samples} "
            f"({_format_duration(model.hop_duration_s)}) · overlap O={model.overlap_samples} "
            f"({100.0 * model.overlap_fraction:.1f}%)"
        ),
        fontsize=9,
        color=_INK,
        fontweight="bold",
        ha="center",
        transform=ax.transAxes,
    )


def _draw_oracle_timeline(ax: Axes, model: SensingDiagramModel) -> None:
    _draw_slot(ax, 0.15, 0.43, 0.26, 0.22, "one scene-epoch path set", color=_BLUE)
    _draw_diamond(ax, 0.43, 0.54, color=_PURPLE)
    ax.annotate(
        "",
        xy=(0.70, 0.54),
        xytext=(0.45, 0.54),
        arrowprops={"arrowstyle": "->", "color": _PURPLE, "linewidth": 1.6},
        xycoords=ax.transAxes,
    )
    _draw_result(ax, 0.75, 0.54)
    ax.text(0.40, 0.68, "one RT solve", fontsize=9, color=_PURPLE, transform=ax.transAxes)
    ax.text(
        0.55,
        0.37,
        "read the complete gain/delay/paths.doppler set",
        fontsize=9,
        color=_MUTED,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.75,
        0.68,
        "oracle RD output",
        fontsize=9,
        color=_GREEN,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.17,
        "No PRI sequence · no coherent CPI · no rolling STFT window · no Doppler FFT"
        f" · observed scene-snapshot revisit {_format_duration(model.observed_revisit_interval_s)}",
        fontsize=10,
        color=_RED,
        fontweight="bold",
        ha="center",
        transform=ax.transAxes,
    )


def _draw_processing(ax: Axes, model: SensingDiagramModel) -> None:
    stages = model.processing_stages
    left = 0.025
    right = 0.975
    gap = 0.014
    width = (right - left - gap * (len(stages) - 1)) / len(stages)
    y = 0.30
    height = 0.54
    for index, stage in enumerate(stages):
        x = left + index * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                facecolor="white",
                edgecolor=stage.color,
                linewidth=1.4,
                transform=ax.transAxes,
            )
        )
        ax.add_patch(
            Rectangle(
                (x, y + height - 0.055),
                width,
                0.055,
                facecolor=stage.color,
                edgecolor=stage.color,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x + width / 2,
            y + height - 0.082,
            stage.title,
            fontsize=9,
            fontweight="bold",
            color=_INK,
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
        detail = textwrap.fill(stage.detail, width=max(18, int(width * 130)))
        ax.text(
            x + width / 2,
            y + height - 0.16,
            detail,
            fontsize=7.0,
            color=_MUTED,
            ha="center",
            va="top",
            linespacing=1.2,
            transform=ax.transAxes,
        )
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.88, y + height / 2),
                xytext=(x + width + gap * 0.12, y + height / 2),
                arrowprops={"arrowstyle": "->", "color": _MUTED, "linewidth": 1.0},
                xycoords=ax.transAxes,
            )

    if model.mode == ANALYTIC_ORACLE_MODE:
        formula = (
            "Diagnostic path-power binning uses persisted delay/Doppler coordinates. "
            "Configured delay-bin/Doppler window names are not applied to this direct "
            "oracle map."
        )
    elif model.mode == SEQUENTIAL_MODE:
        formula = (
            f"STFT membership: Wₘ = x[mH : mH+N], N={model.cpi_length_samples}, "
            f"H={model.cpi_hop_samples}, O=N−H={model.overlap_samples}. "
            "The optional summary micro-Doppler STFT is a separate analysis layer."
        )
    else:
        formula = (
            f"One RT solve supplies all N={model.cpi_length_samples} phase-coherent offsets; "
            f"the symbolic CPI extent is N·PRI={_format_duration(model.cpi_duration_s)}, "
            "while first-to-last sample centers span "
            f"(N−1)·PRI={_format_duration(model.sample_center_span_s)}. The result-bearing "
            "output is attached to sample N; no completion timestamp is persisted. The "
            "next non-overlapping CPI starts after H=N slots."
        )
    ax.add_patch(
        FancyBboxPatch(
            (0.025, 0.045),
            0.95,
            0.16,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor="#EDF1F7",
            edgecolor="none",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.125,
        textwrap.fill(formula, width=145),
        fontsize=8.5,
        color=_INK,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


def _comparison_manifests(
    manifests: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate and order one manifest for each supported sensing mode."""

    by_mode: dict[str, Mapping[str, Any]] = {}
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            raise ValueError("Sensing mode comparison manifests must be mappings")
        if manifest.get("schema_version") != SENSING_SCHEMA_VERSION:
            raise ValueError("Sensing mode comparison requires schema-v4 manifests")
        mode = manifest.get("mode")
        if mode not in _MODE_TITLES:
            raise ValueError(f"Unsupported sensing comparison mode {mode!r}")
        if mode in by_mode:
            raise ValueError(f"Duplicate sensing comparison mode {mode!r}")
        if manifest.get("mode_title") != _MODE_TITLES[mode]:
            raise ValueError(f"Sensing comparison {mode} has an unexpected mode_title")
        expected_label, expected_detail, expected_color = _MODE_STATUS[mode]
        if manifest.get("evidence_status") != expected_label:
            raise ValueError(f"Sensing comparison {mode} has an unexpected evidence label")
        if manifest.get("evidence_color") != expected_color:
            raise ValueError(f"Sensing comparison {mode} has an unexpected evidence color")
        if manifest.get("evidence_detail") != (
            f"{expected_detail} · {_MODE_EVIDENCE_STATUS[mode]}"
        ):
            raise ValueError(f"Sensing comparison {mode} has an unexpected evidence detail")
        for key in (
            "strategy",
            "sample_source",
            "phase_contract",
            "pair_id",
            "acquisition_config_id",
            "note",
        ):
            if not isinstance(manifest.get(key), str) or not manifest[key]:
                raise ValueError(f"Sensing comparison manifest requires non-empty {key}")
        for key in ("cpi_length_samples", "cpi_hop_samples", "rt_solve_stride_scene_steps"):
            value = manifest.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Sensing comparison manifest requires positive {key}")
        for key in ("overlap_samples", "result_stride_samples"):
            value = manifest.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Sensing comparison manifest requires non-negative {key}")
        length = int(manifest["cpi_length_samples"])
        hop = int(manifest["cpi_hop_samples"])
        overlap = int(manifest["overlap_samples"])
        if hop > length or overlap != length - hop:
            raise ValueError("Sensing comparison overlap must equal length minus hop")
        if manifest.get("result_stride_samples") != hop:
            raise ValueError("Sensing comparison result stride must equal hop")
        clutter_enabled = manifest.get("clutter_removal_enabled")
        if not isinstance(clutter_enabled, bool):
            raise ValueError("Sensing comparison manifest requires Boolean clutter state")
        sample_period_s = manifest.get("sample_period_s")
        if mode == ANALYTIC_ORACLE_MODE:
            if (
                length != 1
                or hop != 1
                or overlap != 0
                or sample_period_s is not None
                or manifest.get("sample_rate_hz") is not None
                or manifest.get("slow_time_mean_removal_policy") is not None
                or manifest.get("slow_time_mean_removal_applied") is not None
            ):
                raise ValueError("Analytic oracle comparison timing/processing claims are invalid")
            direct_filter_policy = manifest.get("direct_doppler_clutter_filter_policy")
            direct_filter_cutoff_hz = _finite_float(
                manifest.get("direct_doppler_clutter_cutoff_hz"),
                path="comparison.direct_doppler_clutter_cutoff_hz",
            )
            expected_direct_filter_policy = (
                DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF
                if clutter_enabled
                else DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED
            )
            if direct_filter_policy != expected_direct_filter_policy:
                raise ValueError(
                    "Analytic oracle comparison has an invalid direct Doppler filter policy"
                )
            if (
                direct_filter_cutoff_hz <= 0.0
                if clutter_enabled
                else not math.isclose(
                    direct_filter_cutoff_hz,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(
                    "Analytic oracle comparison has an invalid direct Doppler filter cutoff"
                )
            expected_result_policy = "solve_start"
        else:
            if (
                isinstance(sample_period_s, bool)
                or not isinstance(sample_period_s, (int, float))
                or not math.isfinite(float(sample_period_s))
                or float(sample_period_s) <= 0.0
            ):
                raise ValueError("Sampled comparison modes require a finite positive PRI")
            if manifest.get("slow_time_mean_removal_policy") not in {
                _SLOW_TIME_MEAN_REMOVAL_POLICY,
                "window_weighted_mean",
                "disabled",
            } or not isinstance(manifest.get("slow_time_mean_removal_applied"), bool):
                raise ValueError("Sampled comparison modes require exact mean-removal semantics")
            if (
                manifest.get("direct_doppler_clutter_filter_policy") is not None
                or manifest.get("direct_doppler_clutter_cutoff_hz") is not None
            ):
                raise ValueError("Sampled comparison modes must not claim direct Doppler filtering")
            expected_result_policy = (
                "solve_start" if mode == COHERENT_MODE else SENSING_DEFAULT_REFERENCE_POLICY
            )
        if mode == COHERENT_MODE and (hop != length or overlap != 0):
            raise ValueError("Coherent comparison requires H=N and O=0")
        if manifest.get("result_reference_policy") != expected_result_policy:
            raise ValueError(f"Sensing comparison {mode} has an invalid result reference policy")

        acquisitions = manifest.get("acquisitions")
        if (
            not isinstance(acquisitions, Sequence)
            or isinstance(acquisitions, (str, bytes))
            or not acquisitions
        ):
            raise ValueError("Sensing comparison manifest requires exact acquisitions")
        result_acquisitions: list[Mapping[str, Any]] = []
        for acquisition in acquisitions:
            if not isinstance(acquisition, Mapping):
                raise ValueError("Sensing comparison acquisition entries must be mappings")
            slots = acquisition.get("slots")
            if (
                not isinstance(slots, Sequence)
                or isinstance(slots, (str, bytes))
                or len(slots) != acquisition.get("expected_sample_count")
                or len(slots) != length
            ):
                raise ValueError("Sensing comparison acquisition sample count is malformed")
            if any(not isinstance(slot, Mapping) for slot in slots):
                raise ValueError("Sensing comparison acquisition slots must be mappings")
            slot_times = [
                _finite_float(slot.get("time_s"), path="comparison.acquisition.slot.time_s")
                for slot in slots
            ]
            start_time_s = _finite_float(
                acquisition.get("start_time_s"), path="comparison.acquisition.start_time_s"
            )
            end_time_s = _finite_float(
                acquisition.get("end_time_s"), path="comparison.acquisition.end_time_s"
            )
            default_reference_time_s = _finite_float(
                acquisition.get("default_reference_time_s"),
                path="comparison.acquisition.default_reference_time_s",
            )
            if (
                not math.isclose(start_time_s, slot_times[0], rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(end_time_s, slot_times[-1], rel_tol=0.0, abs_tol=1e-12)
                or acquisition.get("default_reference_policy") != SENSING_DEFAULT_REFERENCE_POLICY
                or not math.isclose(
                    default_reference_time_s,
                    0.5 * (slot_times[0] + slot_times[-1]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("Sensing comparison acquisition temporal summary is inconsistent")
            valid_count = sum(slot.get("valid") is True for slot in slots)
            if valid_count != len(slots):
                result_fields = (
                    "result_reference_time_s",
                    "result_reference_policy",
                    "processing_run_id",
                    "processing_recipe_id",
                    "slow_time_mean_removal_applied",
                )
                if any(acquisition.get(key) is not None for key in result_fields):
                    raise ValueError(
                        "Incomplete sensing comparison acquisition must not carry a result"
                    )
                continue
            if acquisition.get("result_reference_time_s") is not None:
                if (
                    acquisition.get("result_reference_policy") != expected_result_policy
                    or not isinstance(acquisition.get("processing_run_id"), str)
                    or not acquisition.get("processing_run_id")
                    or not isinstance(acquisition.get("processing_recipe_id"), str)
                    or not acquisition.get("processing_recipe_id")
                ):
                    raise ValueError("Sensing comparison result acquisition metadata is malformed")
                if mode == SEQUENTIAL_MODE:
                    expected_reference_time_s = default_reference_time_s
                else:
                    epochs = acquisition.get("rt_epochs")
                    if (
                        not isinstance(epochs, Sequence)
                        or isinstance(epochs, (str, bytes))
                        or not epochs
                        or not isinstance(epochs[0], Mapping)
                    ):
                        raise ValueError("Sensing comparison result requires an RT source epoch")
                    expected_reference_time_s = _finite_float(
                        epochs[0].get("source_epoch_s"),
                        path="comparison.acquisition.rt_epochs[0].source_epoch_s",
                    )
                if not math.isclose(
                    float(acquisition["result_reference_time_s"]),
                    expected_reference_time_s,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("Sensing comparison result reference time is inconsistent")
                if mode == ANALYTIC_ORACLE_MODE:
                    if acquisition.get("slow_time_mean_removal_applied") is not None:
                        raise ValueError("Oracle comparison must not claim mean removal")
                elif not isinstance(acquisition.get("slow_time_mean_removal_applied"), bool):
                    raise ValueError("Sampled comparison acquisition requires mean-removal state")
                result_acquisitions.append(acquisition)
        if not result_acquisitions:
            raise ValueError("Sensing comparison manifest requires a result acquisition")
        if any(
            acquisition.get("processing_recipe_id") != manifest.get("processing_recipe_id")
            for acquisition in result_acquisitions
        ) or result_acquisitions[-1].get("processing_run_id") != manifest.get("processing_run_id"):
            raise ValueError("Sensing comparison displayed processing identity is inconsistent")
        if manifest.get("slow_time_mean_removal_applied") != result_acquisitions[-1].get(
            "slow_time_mean_removal_applied"
        ):
            raise ValueError("Sensing comparison displayed mean-removal state is inconsistent")

        revisit = manifest.get("observed_revisit_interval_s")
        if len(result_acquisitions) >= 2:
            first_slots = result_acquisitions[0]["slots"]
            second_slots = result_acquisitions[1]["slots"]
            first_sources = tuple(
                (
                    slot.get("source_generation_id"),
                    slot.get("scene_frame_id"),
                    slot.get("sample_offset"),
                )
                for slot in first_slots
            )
            second_sources = tuple(
                (
                    slot.get("source_generation_id"),
                    slot.get("scene_frame_id"),
                    slot.get("sample_offset"),
                )
                for slot in second_slots
            )
            if mode == SEQUENTIAL_MODE:
                if (
                    first_sources[hop:] != second_sources[:overlap]
                    or len(set(first_sources) & set(second_sources)) != overlap
                ):
                    raise ValueError("Sequential comparison acquisition overlap is inconsistent")
            elif mode == COHERENT_MODE and set(first_sources) & set(second_sources):
                raise ValueError("Coherent comparison acquisitions must not reuse source samples")
            observed_revisit = float(second_slots[0]["time_s"]) - float(first_slots[0]["time_s"])
            if (
                isinstance(revisit, bool)
                or not isinstance(revisit, (int, float))
                or not math.isclose(float(revisit), observed_revisit, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError("Sensing comparison observed revisit interval is inconsistent")
        elif revisit is not None:
            raise ValueError("One acquisition cannot establish an observed revisit interval")
        stages = manifest.get("processing_stages")
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or not stages:
            raise ValueError("Sensing comparison manifest requires processing_stages")
        if any(
            not isinstance(stage, Mapping)
            or not isinstance(stage.get("title"), str)
            or not stage.get("title")
            or not isinstance(stage.get("detail"), str)
            or not stage.get("detail")
            or not isinstance(stage.get("color"), str)
            or not stage.get("color")
            for stage in stages
        ):
            raise ValueError("Sensing comparison processing stages are malformed")
        by_mode[mode] = manifest

    missing = [mode for mode in _MODE_TITLES if mode not in by_mode]
    if missing:
        raise ValueError(f"Sensing mode comparison is missing modes: {', '.join(missing)}")
    ordered = (
        by_mode[COHERENT_MODE],
        by_mode[SEQUENTIAL_MODE],
        by_mode[ANALYTIC_ORACLE_MODE],
    )
    for shared_key in (
        "pair_id",
        "bandwidth_hz",
        "delay_sampling_frequency_hz",
        "range_mode",
        "fft_size_range",
    ):
        if len({manifest.get(shared_key) for manifest in ordered}) != 1:
            raise ValueError(f"Sensing comparison modes disagree on {shared_key}")
    return ordered


def sensing_mode_comparison_manifest(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the validated, exact manifests behind a three-mode comparison."""

    ordered = _comparison_manifests(manifests)
    return {
        "schema_version": SENSING_SCHEMA_VERSION,
        "kind": "sensing_mode_comparison",
        "modes": [dict(manifest) for manifest in ordered],
        "note": (
            "The mode cards are rendered from these exact schema-v4 sensing-diagram "
            "manifests in coherent, sequential, then analytic-oracle order."
        ),
    }


def write_sensing_mode_comparison_manifest(
    manifests: Sequence[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Write the exact validated three-mode comparison contract as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sensing_mode_comparison_manifest(manifests), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _draw_comparison_slot(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    label: str,
    *,
    color: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.065,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor="white",
            edgecolor=color,
            linewidth=1.1,
            transform=ax.transAxes,
        )
    )
    ax.text(
        x + width / 2,
        y + 0.0325,
        label,
        fontsize=7.5,
        fontweight="bold",
        color=_INK,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


def _draw_comparison_card(ax: Axes, manifest: Mapping[str, Any]) -> None:
    """Render one compact mode card from its exact JSON-compatible manifest."""

    mode = str(manifest["mode"])
    length = int(manifest["cpi_length_samples"])
    hop = int(manifest["cpi_hop_samples"])
    overlap = int(manifest["overlap_samples"])
    evidence_color = str(manifest["evidence_color"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.005),
            0.99,
            0.99,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            facecolor=_PANEL,
            edgecolor=_GRID,
            linewidth=1.2,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.05,
        0.95,
        str(manifest["mode_title"]),
        fontsize=13,
        fontweight="bold",
        color=_INK,
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.05,
        0.902,
        str(manifest["strategy"]),
        fontsize=8.2,
        color=_MUTED,
        va="top",
        transform=ax.transAxes,
    )
    evidence_detail = str(manifest["evidence_detail"]).split(" · ", maxsplit=1)[0]
    ax.text(
        0.95,
        0.945,
        str(manifest["evidence_status"]),
        fontsize=7.5,
        fontweight="bold",
        color="white",
        ha="right",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.4,rounding_size=0.12",
            "facecolor": evidence_color,
            "edgecolor": "none",
        },
        transform=ax.transAxes,
    )
    ax.text(
        0.05,
        0.855,
        evidence_detail,
        fontsize=7.5,
        color=evidence_color,
        va="top",
        transform=ax.transAxes,
    )

    if mode == ANALYTIC_ORACLE_MODE:
        timing = "PRI: not acquired"
        window = (
            "one scene-epoch path set · observed revisit "
            f"{_format_duration(manifest.get('observed_revisit_interval_s'))}"
        )
    else:
        timing = (
            f"PRI {_format_duration(manifest.get('sample_period_s'))} · "
            f"N·PRI {_format_duration(manifest.get('cpi_duration_s'))} · "
            f"centers {_format_duration(manifest.get('sample_center_span_s'))}"
        )
        window_kind = "CPI Doppler window" if mode == COHERENT_MODE else "rolling STFT window"
        window = f"{window_kind}: H={hop} · O={overlap} " f"({100.0 * overlap / length:.1f}%)"
    ax.text(0.05, 0.805, timing, fontsize=8.2, color=_INK, transform=ax.transAxes)
    ax.text(0.05, 0.77, window, fontsize=8.2, color=_INK, transform=ax.transAxes)

    lane_y = 0.645
    if mode == COHERENT_MODE:
        _draw_diamond(ax, 0.075, lane_y + 0.032, color=_PURPLE)
        labels: tuple[str, ...] = ("1", "2", "3", "…", f"N={length}")
        for index, label in enumerate(labels):
            _draw_comparison_slot(ax, 0.14 + 0.115 * index, lane_y, 0.095, label, color=_BLUE)
        _draw_result(ax, 0.14 + 0.115 * 4 + 0.0475, lane_y + 0.145)
        ax.text(
            0.50,
            0.59,
            "one RT solve owns every phase-coherent CPI slot · result attached to sample N",
            fontsize=7.5,
            color=_PURPLE,
            ha="center",
            transform=ax.transAxes,
        )
    elif mode == SEQUENTIAL_MODE:
        labels = ("1", f"H+1={hop + 1}", "…", f"N={length}", "…", f"N+H={length + hop}")
        xs = [0.055 + 0.145 * index for index in range(len(labels))]
        ax.add_patch(
            Rectangle(
                (xs[0] - 0.01, lane_y + 0.092),
                xs[3] - xs[0] + 0.105,
                0.025,
                facecolor="#DDEAFF",
                edgecolor=_BLUE,
                transform=ax.transAxes,
            )
        )
        ax.add_patch(
            Rectangle(
                (xs[1] - 0.01, lane_y + 0.057),
                xs[-1] - xs[1] + 0.105,
                0.025,
                facecolor="#EAE2FB",
                edgecolor=_PURPLE,
                transform=ax.transAxes,
            )
        )
        for x, label in zip(xs, labels, strict=True):
            _draw_comparison_slot(ax, x, lane_y, 0.085, label, color=_CYAN)
            _draw_diamond(ax, x + 0.0425, lane_y - 0.035, color=_PURPLE)
        _draw_result(ax, xs[3] + 0.0425, lane_y + 0.145)
        _draw_result(ax, xs[-1] + 0.0425, lane_y + 0.145)
        ax.text(
            0.50,
            0.565,
            f"independent solve each PRI · rolling W2 is samples H+1…H+N (H={hop})",
            fontsize=7.5,
            color=_PURPLE,
            ha="center",
            transform=ax.transAxes,
        )
    else:
        _draw_comparison_slot(ax, 0.08, lane_y, 0.31, "one scene-epoch path set", color=_BLUE)
        _draw_diamond(ax, 0.45, lane_y + 0.032, color=_PURPLE)
        ax.annotate(
            "",
            xy=(0.72, lane_y + 0.032),
            xytext=(0.50, lane_y + 0.032),
            arrowprops={"arrowstyle": "->", "color": _PURPLE, "linewidth": 1.3},
            xycoords=ax.transAxes,
        )
        _draw_result(ax, 0.80, lane_y + 0.032)
        ax.text(
            0.50,
            0.59,
            "path delay + paths.doppler → direct oracle bins",
            fontsize=7.5,
            color=_PURPLE,
            ha="center",
            transform=ax.transAxes,
        )

    ax.text(
        0.05,
        0.505,
        "EXECUTED PROCESSING",
        fontsize=7.4,
        fontweight="bold",
        color=_MUTED,
        transform=ax.transAxes,
    )
    stages = manifest["processing_stages"]
    stage_height = min(0.052, 0.39 / len(stages))
    stage_gap = 0.008
    y = 0.445
    for stage in stages:
        stage_color = str(stage["color"])
        ax.add_patch(
            FancyBboxPatch(
                (0.05, y),
                0.90,
                stage_height,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor="white",
                edgecolor=_GRID,
                linewidth=0.8,
                transform=ax.transAxes,
            )
        )
        ax.add_patch(
            Rectangle(
                (0.05, y),
                0.018,
                stage_height,
                facecolor=stage_color,
                edgecolor=stage_color,
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.09,
            y + stage_height / 2,
            str(stage["title"]),
            fontsize=7.5,
            color=_INK,
            va="center",
            transform=ax.transAxes,
        )
        y -= stage_height + stage_gap

    result_policy = manifest.get("result_reference_policy") or "snapshot epoch"
    ax.text(
        0.05,
        0.035,
        f"result ref: {result_policy} · source: {manifest['sample_source']}",
        fontsize=6.9,
        color=_MUTED,
        transform=ax.transAxes,
    )


def create_sensing_mode_comparison_figure(
    manifests: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path,
) -> Path:
    """Render a compact, exact three-mode comparison from diagram manifests."""

    ordered = _comparison_manifests(manifests)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 9.2), facecolor="white")
    fig.text(
        0.035,
        0.965,
        "ORCHAV sensing modes · acquisition and processing comparison",
        fontsize=20,
        fontweight="bold",
        color=_INK,
        va="top",
    )
    fig.text(
        0.035,
        0.925,
        "Exact schema-v4 execution contracts and observed first/next acquisition semantics",
        fontsize=10.5,
        color=_MUTED,
        va="top",
    )
    grid = fig.add_gridspec(
        1,
        3,
        left=0.03,
        right=0.97,
        top=0.885,
        bottom=0.095,
        wspace=0.035,
    )
    for index, manifest in enumerate(ordered):
        _draw_comparison_card(fig.add_subplot(grid[index]), manifest)
    fig.text(
        0.035,
        0.052,
        (
            "Symbolic chirp/PRI slots explain timing only. ORCHAV persists CIR/path "
            "samples; it does not synthesize an RF waveform, matched filter, "
            "pulse-compress, or apply a range FFT."
        ),
        fontsize=8.5,
        fontweight="bold",
        color=_RED,
        va="bottom",
    )
    fig.savefig(output, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    write_sensing_mode_comparison_manifest(ordered, output.with_suffix(".json"))
    return output


def create_sensing_acquisition_processing_figure(
    sensing_data: Mapping[str, Any],
    *,
    output_path: str | Path,
    sensing_event_history: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Render one scenario-backed acquisition/processing diagram."""

    model = build_sensing_diagram_model(
        sensing_data,
        sensing_event_history=sensing_event_history,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9.7), facecolor="white")
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=(1.15, 2.55, 2.85, 0.55),
        hspace=0.18,
        left=0.035,
        right=0.965,
        top=0.97,
        bottom=0.035,
    )
    header = fig.add_subplot(grid[0])
    header.set_xlim(0.0, 1.0)
    header.set_ylim(0.0, 1.0)
    header.axis("off")
    header.text(
        0.0,
        0.93,
        "ORCHAV sensing acquisition & processing",
        fontsize=20,
        fontweight="bold",
        color=_INK,
        va="top",
    )
    header.text(
        0.0,
        0.67,
        f"{model.mode_title} · {model.strategy}",
        fontsize=12.5,
        color=_MUTED,
        va="top",
    )
    displayed_evidence_detail = model.evidence_detail.split(" · ", maxsplit=1)[0]
    header.text(
        0.985,
        0.84,
        f"{model.evidence_status}  ·  {displayed_evidence_detail}",
        fontsize=9,
        fontweight="bold",
        color="white",
        ha="right",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.5,rounding_size=0.15",
            "facecolor": model.evidence_color,
            "edgecolor": "none",
        },
    )

    if model.mode == ANALYTIC_ORACLE_MODE:
        metrics = (
            ("ACQUISITION", "single scene-epoch path set"),
            ("PRI / SAMPLE RATE", "not defined"),
            (
                "SCENE SNAPSHOT REVISIT",
                _format_duration(model.observed_revisit_interval_s),
            ),
            ("RESULT", model.result_reference_policy or "snapshot epoch"),
        )
    else:
        window_label = (
            "CPI DOPPLER WINDOW" if model.mode == COHERENT_MODE else "ROLLING STFT WINDOW"
        )
        metrics = (
            (
                "PRI / SAMPLE RATE",
                f"{_format_duration(model.sample_period_s)} · {_format_rate(model.sample_rate_hz)}",
            ),
            (
                "N·PRI / CENTER SPAN",
                f"{_format_duration(model.cpi_duration_s)} / "
                f"{_format_duration(model.sample_center_span_s)}",
            ),
            (
                window_label,
                f"N={model.cpi_length_samples} · H={model.cpi_hop_samples} · "
                f"O={model.overlap_samples} ({model.overlap_fraction * 100:.1f}%)",
            ),
            (
                "RESULT CADENCE / REF",
                f"every {model.result_stride_samples} sample(s) · {model.result_reference_policy or 'n/a'}",
            ),
        )
    metric_gap = 0.012
    metric_width = (1.0 - metric_gap * 3) / 4
    for index, (label, value) in enumerate(metrics):
        _metric_box(header, index * (metric_width + metric_gap), metric_width, label, value)

    timeline = fig.add_subplot(grid[1])
    timeline_title = (
        "1 · RT snapshot and direct oracle result"
        if model.mode == ANALYTIC_ORACLE_MODE
        else "1 · Acquisition timeline (symbolic chirp/PRI/CIR sample slots*)"
    )
    _panel(timeline, timeline_title)
    if model.mode == COHERENT_MODE:
        _draw_coherent_timeline(timeline, model)
    elif model.mode == SEQUENTIAL_MODE:
        _draw_sequential_timeline(timeline, model)
    else:
        _draw_oracle_timeline(timeline, model)

    processing = fig.add_subplot(grid[2])
    _panel(processing, "2 · Processing path")
    _draw_processing(processing, model)

    footer = fig.add_subplot(grid[3])
    footer.axis("off")
    footer.text(
        0.0,
        0.72,
        (
            "Derived from persisted sensing.config_resolved contract v"
            f"{model.contract_version} + sensing.temporal "
            f"({model.acquisition_completeness}, {model.observed_sample_count}/"
            f"{model.expected_sample_count} samples). Schematic spacing is not proportional."
        ),
        fontsize=8.2,
        color=_MUTED,
        va="top",
    )
    footer.text(
        0.0,
        0.28,
        (
            "* Symbolic PRI slots: retained CFR → frequency-windowed delay IFFT → slow-time DFT; "
            "no synthesized RF waveform, matched filter or pulse compression."
            if model.range_processing == "finite_bandwidth_delay_response"
            else "* “Chirp/PRI slot” is a timing analogy: ORCHAV stores CIR/path samples; "
            "it does not synthesize an RF waveform, matched filter, pulse-compress, "
            "or apply a range FFT."
        ),
        fontsize=8.2,
        color=_RED,
        va="top",
        fontweight="bold",
    )

    fig.savefig(output, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


__all__ = [
    "ANALYTIC_ORACLE_MODE",
    "COHERENT_MODE",
    "SEQUENTIAL_MODE",
    "SensingDiagramAcquisition",
    "SensingDiagramModel",
    "SensingDiagramRtEpoch",
    "SensingDiagramSlot",
    "SensingDiagramStage",
    "SensingDiagramWindow",
    "build_sensing_diagram_model",
    "create_sensing_acquisition_processing_figure",
    "create_sensing_mode_comparison_figure",
    "sensing_diagram_manifest",
    "sensing_mode_comparison_manifest",
    "write_sensing_diagram_manifest",
    "write_sensing_mode_comparison_manifest",
]
