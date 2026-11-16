"""Resolve sensing acquisition and processing semantics before ray tracing.

The YAML model intentionally permits ``processing_mode: auto`` as an input
convenience.  Runtime services must not interpret that value independently.
This module converts it into one immutable execution contract shared by the
scheduler, ray-tracing service, processor, temporal binder, and persistence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigurationError

if TYPE_CHECKING:
    from ..configuration import SensingConfig, SimulationConfig


AUTO_MODE = "auto"
COHERENT_MODE = "coherent_cpi"
SEQUENTIAL_MODE = "sequential_stft"
ANALYTIC_ORACLE_MODE = "analytic_path_doppler_oracle"

VALID_INPUT_MODES = frozenset({AUTO_MODE, COHERENT_MODE, SEQUENTIAL_MODE, ANALYTIC_ORACLE_MODE})
PARTIAL_WINDOW_POLICY = "retain_unprocessed"


@dataclass(frozen=True)
class SensingAcquisitionPlan:
    """Immutable description of how physical sensing samples are acquired."""

    sample_source: str
    rt_solve_stride_scene_steps: int
    sample_period_s: float | None
    sample_rate_hz: float | None
    cpi_length_samples: int
    cpi_hop_samples: int
    overlap_samples: int
    partial_window_policy: str
    phase_contract: str

    def window_due(self, observed_samples: int) -> bool:
        """Schedule complete windows from acquired sample count, never scene IDs."""
        return (
            observed_samples >= self.cpi_length_samples
            and (observed_samples - self.cpi_length_samples) % self.cpi_hop_samples == 0
        )


@dataclass(frozen=True)
class SensingProcessingPlan:
    """Immutable description of the processing strategy applied to samples."""

    mode: str
    strategy: str
    fft_size_doppler: int
    result_stride_samples: int
    product_semantics: str
    evidence_status: str


@dataclass(frozen=True)
class SensingTopologyRequirement:
    """Sensor-pair and antenna-element topology admitted by the selected strategy."""

    pair_count: int
    pair_index: int
    tx_device_count: int
    rx_device_count: int
    tx_elements: int
    rx_elements: int
    rx_array_processing: str


@dataclass(frozen=True)
class ResolvedSensingExecutionConfig:
    """Single runtime authority for sensing acquisition and mode semantics."""

    requested_mode: str
    acquisition: SensingAcquisitionPlan
    processing: SensingProcessingPlan
    topology: SensingTopologyRequirement
    contract_version: int = 1

    @property
    def is_coherent_one_solve(self) -> bool:
        return self.acquisition.sample_source == "coherent_one_solve"

    @property
    def is_sequential(self) -> bool:
        return self.processing.mode == SEQUENTIAL_MODE

    @property
    def is_analytic_oracle(self) -> bool:
        return self.processing.mode == ANALYTIC_ORACLE_MODE

    def persisted(self) -> dict[str, Any]:
        """Return canonical persisted semantics without the unresolved input.

        ``requested_mode`` is deliberately omitted: persisted identity records
        the explicit resolved result, never ``auto``.
        """

        return {
            "contract_version": self.contract_version,
            "acquisition": asdict(self.acquisition),
            "processing": asdict(self.processing),
            "topology": asdict(self.topology),
        }


def _positive_finite(value: Any, *, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ConfigurationError(f"{name} must be a finite positive number")
    return normalized


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _antenna_elements(antenna: Any) -> int:
    if antenna is None:
        return 1
    rows = _positive_int(getattr(antenna, "num_rows", 1), name="antenna.num_rows")
    cols = _positive_int(getattr(antenna, "num_cols", 1), name="antenna.num_cols")
    return rows * cols


def _require_single_pair(
    *,
    tx_device_count: int,
    rx_device_count: int,
) -> None:
    if tx_device_count != 1 or rx_device_count != 1:
        raise ConfigurationError(
            "Phase-2 sensing strategies require exactly one transmitter device and "
            "one receiver device; multi-pair acquisition is deferred to the typed "
            "pair/product contract"
        )


def resolve_sensing_execution_config(
    simulation_config: "SimulationConfig",
    *,
    frame_dt_s: float,
    tx_device_count: int,
    rx_device_count: int,
) -> ResolvedSensingExecutionConfig | None:
    """Resolve and validate sensing execution before any ray-tracing work.

    Returns ``None`` for a disabled sensing extension. Every enabled row either
    returns a fully explicit frozen contract or raises :class:`ConfigurationError`.
    """

    sensing = simulation_config.sensing
    if sensing is None or not sensing.enabled:
        return None

    if (
        sensing.track_confirmation_enabled
        and sensing.tracker_type == "cartesian3d"
        and min(simulation_config.rx_antenna.num_rows, simulation_config.rx_antenna.num_cols) <= 1
    ):
        raise ConfigurationError("Cartesian 3D tracking requires a planar RX array")

    resolved = resolve_sensing_execution_values(
        sensing,
        cir_time_steps=simulation_config.cir_time_steps,
        cir_sampling_frequency_hz=simulation_config.cir_sampling_frequency_hz,
        frame_dt_s=frame_dt_s,
        tx_device_count=tx_device_count,
        rx_device_count=rx_device_count,
        tx_elements=_antenna_elements(simulation_config.tx_antenna),
        rx_elements=_antenna_elements(simulation_config.rx_antenna),
    )
    if resolved.is_coherent_one_solve and simulation_config.mesh_update_interval_s is not None:
        from ..target.mesh import mesh_update_step_interval

        mesh_stride = mesh_update_step_interval(
            duration=simulation_config.duration,
            num_steps=simulation_config.num_steps,
            mesh_update_interval_s=simulation_config.mesh_update_interval_s,
        )
        cpi_stride = resolved.acquisition.rt_solve_stride_scene_steps
        if mesh_stride is not None and mesh_stride % cpi_stride != 0:
            raise ConfigurationError(
                "coherent_cpi requires the mesh update step interval to be an exact "
                f"multiple of the {cpi_stride}-step RT/CPI cadence; got {mesh_stride}"
            )
    return resolved


def resolve_sensing_execution_values(
    sensing: "SensingConfig",
    *,
    cir_time_steps: int,
    cir_sampling_frequency_hz: float | None,
    frame_dt_s: float,
    tx_device_count: int = 1,
    rx_device_count: int = 1,
    tx_elements: int = 1,
    rx_elements: int = 1,
) -> ResolvedSensingExecutionConfig:
    """Value-level resolver used by runtime orchestration and focused tests."""

    requested_raw = getattr(sensing, "processing_mode", None)
    if not isinstance(requested_raw, str) or not requested_raw.strip():
        raise ConfigurationError(
            "sensing.processing_mode must be one of " f"{sorted(VALID_INPUT_MODES)}"
        )
    requested_mode = requested_raw.strip().lower()
    if requested_mode == "snapshot_direct_rd":
        raise ConfigurationError(
            "sensing.processing_mode='snapshot_direct_rd' was removed because it is "
            "an analytic path-Doppler oracle, not waveform/coherent sensing; use "
            "'analytic_path_doppler_oracle' for an explicit diagnostic oracle"
        )
    if requested_mode not in VALID_INPUT_MODES:
        raise ConfigurationError(
            f"Invalid sensing.processing_mode {requested_raw!r}; expected one of "
            f"{sorted(VALID_INPUT_MODES)}"
        )

    cir_steps = _positive_int(cir_time_steps, name="raytracing.cir_time_steps")
    fft_d = _positive_int(
        getattr(sensing, "fft_size_doppler", None),
        name="sensing.fft_size_doppler",
    )
    prf_hz = _positive_finite(getattr(sensing, "prf_hz", None), name="sensing.prf_hz")
    scene_period_s = _positive_finite(frame_dt_s, name="scenario frame interval")
    tx_count = _positive_int(tx_device_count, name="transmitter device count")
    rx_count = _positive_int(rx_device_count, name="receiver device count")
    tx_element_count = _positive_int(tx_elements, name="TX element count")
    rx_element_count = _positive_int(rx_elements, name="RX element count")
    _require_single_pair(tx_device_count=tx_count, rx_device_count=rx_count)

    if requested_mode == AUTO_MODE:
        if cir_steps <= 1:
            raise ConfigurationError(
                "sensing.processing_mode='auto' is ambiguous with cir_time_steps=1; "
                "select 'sequential_stft' for independent solves or "
                "'analytic_path_doppler_oracle' for the diagnostic oracle"
            )
        resolved_mode = COHERENT_MODE
    else:
        resolved_mode = requested_mode

    requested_hop = getattr(sensing, "cpi_hop_samples", None)
    if requested_hop is not None:
        requested_hop = _positive_int(requested_hop, name="sensing.cpi_hop_samples")
        if resolved_mode != SEQUENTIAL_MODE:
            raise ConfigurationError("sensing.cpi_hop_samples requires sequential_stft")
        if requested_hop > fft_d:
            raise ConfigurationError("sensing.cpi_hop_samples must not exceed fft_size_doppler")

    sensing.range_response.validate_processing(
        bandwidth=sensing.bandwidth,
        fft_size=sensing.fft_size_range,
        window_range=sensing.window_range,
        noise_enabled=sensing.noise_enabled,
        noise_snr_db=sensing.noise_snr_db,
        processing_mode=resolved_mode,
    )
    cfr_mode = sensing.range_response.model == "cfr_ifft"

    if sensing.coherent_clutter_mode != "legacy" and not (
        resolved_mode == COHERENT_MODE or (resolved_mode == SEQUENTIAL_MODE and cfr_mode)
    ):
        raise ConfigurationError("coherent_clutter_mode requires coherent_cpi processing")
    if cfr_mode and resolved_mode == SEQUENTIAL_MODE and sensing.clutter_removal_enabled:
        raise ConfigurationError(
            "sequential cfr_ifft does not support causal history clutter; "
            "use coherent_mode none, windowed_mean or unweighted_mean"
        )

    if sensing.array_detection_mode == "beam_scan" and (
        resolved_mode != COHERENT_MODE or rx_element_count <= 1
    ):
        raise ConfigurationError("beam_scan requires coherent_cpi with a multi-element RX array")
    if sensing.track_confirmation_enabled and sensing.tracker_type in {"cartesian", "cartesian3d"}:
        if (
            sensing.range_mode != "monostatic"
            or resolved_mode != COHERENT_MODE
            or rx_element_count <= 1
        ):
            raise ConfigurationError(
                "Cartesian tracking requires monostatic coherent_cpi with a multi-element RX array"
            )
    if sensing.track_association_azimuth_gate_deg is not None and (
        resolved_mode != COHERENT_MODE or rx_element_count <= 1
    ):
        raise ConfigurationError("Azimuth tracking gates require coherent_cpi with an RX array")

    if resolved_mode == COHERENT_MODE:
        if cir_steps < 2:
            raise ConfigurationError("coherent_cpi requires raytracing.cir_time_steps >= 2")
        cir_rate_hz = _positive_finite(
            cir_sampling_frequency_hz,
            name="raytracing.cir_sampling_frequency_hz",
        )
        if not math.isclose(cir_rate_hz, prf_hz, rel_tol=1e-9, abs_tol=1e-9):
            raise ConfigurationError(
                "coherent_cpi requires sensing.prf_hz to equal "
                "raytracing.cir_sampling_frequency_hz"
            )
        if fft_d != cir_steps:
            raise ConfigurationError(
                "coherent_cpi requires sensing.fft_size_doppler to equal "
                "raytracing.cir_time_steps; zero-padding/truncation is not part of "
                "the Phase-2 acquisition contract"
            )
        if tx_element_count != 1:
            raise ConfigurationError(
                "coherent_cpi currently requires one TX antenna element; explicit "
                "TX element combining is deferred to the typed-axis contract"
            )
        acquisition = SensingAcquisitionPlan(
            sample_source="coherent_one_solve",
            rt_solve_stride_scene_steps=cir_steps,
            sample_period_s=1.0 / cir_rate_hz,
            sample_rate_hz=cir_rate_hz,
            cpi_length_samples=cir_steps,
            cpi_hop_samples=cir_steps,
            overlap_samples=0,
            partial_window_policy=PARTIAL_WINDOW_POLICY,
            phase_contract="sionna_intra_solve_phase_coherent",
        )
        processing = SensingProcessingPlan(
            mode=COHERENT_MODE,
            strategy="coherent_cpi_fft",
            fft_size_doppler=fft_d,
            result_stride_samples=cir_steps,
            product_semantics=(
                "cfr_ifft_slow_time_fft" if cfr_mode else "cir_point_bin_slow_time_fft"
            ),
            evidence_status=(
                "finite_bandwidth_bounded_coherent"
                if cfr_mode
                else "phase2_controlled_specular_validated"
            ),
        )
        rx_array_processing = "explicit_planar_rx_fft" if rx_element_count > 1 else "siso"
    elif resolved_mode == SEQUENTIAL_MODE:
        if cir_steps != 1:
            raise ConfigurationError("sequential_stft requires raytracing.cir_time_steps == 1")
        if cir_sampling_frequency_hz is not None:
            raise ConfigurationError(
                "sequential_stft samples the scenario clock and requires "
                "raytracing.cir_sampling_frequency_hz to be omitted"
            )
        if fft_d < 2:
            raise ConfigurationError("sequential_stft requires fft_size_doppler >= 2")
        expected_period_s = 1.0 / prf_hz
        if not math.isclose(scene_period_s, expected_period_s, rel_tol=1e-9, abs_tol=1e-12):
            raise ConfigurationError(
                "sequential_stft requires the scenario frame interval to equal "
                "1 / sensing.prf_hz; adjust timeline.sample_rate_hz, "
                "timeline.duration_s/steps, or the PRF"
            )
        if tx_element_count != 1 or rx_element_count != 1:
            raise ConfigurationError(
                "sequential_stft is Phase-2 SISO only; array-element phase across "
                "independent solves is not yet validated"
            )
        acquisition = SensingAcquisitionPlan(
            sample_source="independent_solves",
            rt_solve_stride_scene_steps=1,
            sample_period_s=scene_period_s,
            sample_rate_hz=prf_hz,
            cpi_length_samples=fft_d,
            cpi_hop_samples=requested_hop or 1,
            overlap_samples=fft_d - (requested_hop or 1),
            partial_window_policy=PARTIAL_WINDOW_POLICY,
            phase_contract="independent_solve_phase_continuity_controlled_specular_only",
        )
        processing = SensingProcessingPlan(
            mode=SEQUENTIAL_MODE,
            strategy="sequential_stft",
            fft_size_doppler=fft_d,
            result_stride_samples=requested_hop or 1,
            product_semantics=(
                "cfr_ifft_slow_time_fft" if cfr_mode else "cross_solve_cir_doppler_fft"
            ),
            evidence_status="phase2_controlled_specular_validated_experimental",
        )
        rx_array_processing = "siso"
    else:
        if cir_steps != 1:
            raise ConfigurationError(
                "analytic_path_doppler_oracle requires raytracing.cir_time_steps == 1"
            )
        if cir_sampling_frequency_hz is not None:
            raise ConfigurationError(
                "analytic_path_doppler_oracle does not acquire a CIR time series; "
                "raytracing.cir_sampling_frequency_hz must be omitted"
            )
        if tx_element_count != 1 or rx_element_count != 1:
            raise ConfigurationError(
                "analytic_path_doppler_oracle is a single-pair SISO diagnostic"
            )
        acquisition = SensingAcquisitionPlan(
            sample_source="analytic_path_snapshot",
            rt_solve_stride_scene_steps=1,
            sample_period_s=None,
            sample_rate_hz=None,
            cpi_length_samples=1,
            cpi_hop_samples=1,
            overlap_samples=0,
            partial_window_policy=PARTIAL_WINDOW_POLICY,
            phase_contract="analytic_nonwaveform_path_doppler",
        )
        processing = SensingProcessingPlan(
            mode=ANALYTIC_ORACLE_MODE,
            strategy="analytic_path_doppler_oracle",
            fft_size_doppler=fft_d,
            result_stride_samples=1,
            product_semantics="analytic_oracle_nonwaveform",
            evidence_status="diagnostic_quarantined",
        )
        rx_array_processing = "siso"

    topology = SensingTopologyRequirement(
        pair_count=1,
        pair_index=0,
        tx_device_count=tx_count,
        rx_device_count=rx_count,
        tx_elements=tx_element_count,
        rx_elements=rx_element_count,
        rx_array_processing=rx_array_processing,
    )
    return ResolvedSensingExecutionConfig(
        requested_mode=requested_mode,
        acquisition=acquisition,
        processing=processing,
        topology=topology,
        contract_version=2 if cfr_mode else 1,
    )
