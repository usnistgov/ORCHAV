#!/usr/bin/env python3
"""Normalized configuration dataclasses for generator-core execution.

Scenario YAML is parsed elsewhere into scenario-model objects.  The dataclasses
in this module are the lower-level representation used by services and
propagation after names, defaults, and units have been normalized.  They should
stay free of Sionna scene objects and other runtime allocations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from shared.logging import get_logger
from shared.scenarios.actors import FixedOrientationSpec, OrientationSpec

from ..exceptions import ConfigurationError
from ..mobility import MobilityPattern
from ..orientation.base import PreparedOrientationSource
from ..utils import point_to_tuple
from .defaults import (
    CUSTOM_QUALITY_BASE_PRESET,
    DEFAULT_ANTENNA_HORIZONTAL_SPACING,
    DEFAULT_ANTENNA_NUM_COLS,
    DEFAULT_ANTENNA_NUM_ROWS,
    DEFAULT_ANTENNA_PATTERN,
    DEFAULT_ANTENNA_POLARIZATION,
    DEFAULT_ANTENNA_VERTICAL_SPACING,
    DEFAULT_BANDWIDTH_HZ,
    DEFAULT_CARRIER_FREQUENCY_HZ,
    DEFAULT_CIR_SAMPLING_FREQUENCY_HZ,
    DEFAULT_CIR_TIME_STEPS,
    DEFAULT_COVERAGE_ENABLED,
    DEFAULT_COVERAGE_MAX_PATHS,
    DEFAULT_COVERAGE_METRIC,
    DEFAULT_COVERAGE_QUALITY_PRESET,
    DEFAULT_COVERAGE_RESOLUTION_M,
    DEFAULT_COVERAGE_SAVE_COMPRESSION,
    DEFAULT_COVERAGE_STRIDE,
    DEFAULT_COVERAGE_TX_COMBINATION,
    DEFAULT_COVERAGE_TX_MODE,
    DEFAULT_DEBUG_LEVEL,
    DEFAULT_DURATION_S,
    DEFAULT_EXPORT_PATH_METRICS,
    DEFAULT_FILE_FORMAT,
    DEFAULT_GENERATE_VIDEO,
    DEFAULT_MESH_UPDATE_INTERVAL_S,
    DEFAULT_NUM_STEPS,
    DEFAULT_OUTPUT_MODE,
    DEFAULT_QUALITY_PRESET,
    DEFAULT_SCENE_MATERIAL_SCATTERING_COEFFICIENT_PRESET,
    DEFAULT_SCENE_NAME,
    DEFAULT_SENSING_AOA_FILTER_ENABLED,
    DEFAULT_SENSING_BANDWIDTH_HZ,
    DEFAULT_SENSING_CFAR_GUARD_CELLS,
    DEFAULT_SENSING_CFAR_MIN_SNR_DB,
    DEFAULT_SENSING_CFAR_OS_RANK,
    DEFAULT_SENSING_CFAR_RD_GUARD_CELLS_DOPPLER,
    DEFAULT_SENSING_CFAR_RD_GUARD_CELLS_RANGE,
    DEFAULT_SENSING_CFAR_RD_REF_CELLS_DOPPLER,
    DEFAULT_SENSING_CFAR_RD_REF_CELLS_RANGE,
    DEFAULT_SENSING_CFAR_REF_CELLS,
    DEFAULT_SENSING_CFAR_THRESHOLD_SCALE,
    DEFAULT_SENSING_CFAR_TYPE,
    DEFAULT_SENSING_CHIRPS_PER_FRAME,
    DEFAULT_SENSING_CLUTTER_REMOVAL_ENABLED,
    DEFAULT_SENSING_CLUTTER_REMOVAL_WINDOW,
    DEFAULT_SENSING_ENABLED,
    DEFAULT_SENSING_EVAL_ASSOCIATION_RANGE_GATE_M,
    DEFAULT_SENSING_EVAL_ASSOCIATION_VELOCITY_GATE_M_S,
    DEFAULT_SENSING_FFT_SIZE_DOPPLER,
    DEFAULT_SENSING_FFT_SIZE_RANGE,
    DEFAULT_SENSING_NOISE_ENABLED,
    DEFAULT_SENSING_OUTPUT_DETECTIONS,
    DEFAULT_SENSING_OUTPUT_RANGE_DOPPLER,
    DEFAULT_SENSING_OUTPUT_RANGE_PROFILE,
    DEFAULT_SENSING_PERSIST_DETECTION_STAGE_ARTIFACTS,
    DEFAULT_SENSING_PRE_CFAR_DOPPLER_DC_GUARD_BINS,
    DEFAULT_SENSING_PRE_CFAR_DOPPLER_EDGE_GUARD_BINS,
    DEFAULT_SENSING_PRF_HZ,
    DEFAULT_SENSING_PROCESSING_MODE,
    DEFAULT_SENSING_RANGE_MODE,
    DEFAULT_SENSING_RAW_PAYLOAD_POLICY,
    DEFAULT_SENSING_TRACK_ASSOCIATION_RANGE_GATE_M,
    DEFAULT_SENSING_TRACK_ASSOCIATION_VELOCITY_GATE_M_S,
    DEFAULT_SENSING_TRACK_CONFIRMATION_ENABLED,
    DEFAULT_SENSING_TRACK_CONFIRMATION_M,
    DEFAULT_SENSING_TRACK_CONFIRMATION_N,
    DEFAULT_SENSING_TRACK_MAX_MISSED_FRAMES,
    DEFAULT_SENSING_TRACK_OUTPUT_TENTATIVE,
    DEFAULT_SENSING_TRACKER_TYPE,
    DEFAULT_SENSING_WINDOW,
    DEFAULT_START_STEP,
    DEFAULT_TEMPERATURE_K,
    DEFAULT_VIEW,
    VALID_ANTENNA_PATTERNS,
    VALID_FILE_FORMATS,
    VALID_OUTPUT_MODES,
    VALID_POLARIZATIONS,
    VALID_SCENE_MATERIAL_SCATTERING_COEFFICIENT_PRESETS,
    VALID_SENSING_PROCESSING_MODES,
    VALID_SENSING_RANGE_MODES,
    VALID_SENSING_RAW_PAYLOAD_POLICIES,
    VALID_SENSING_TRACKER_TYPES,
    VALID_VIEWS,
)
from .presets import AVAILABLE_SCENES, QUALITY_PRESETS

logger = get_logger(__name__)


@dataclass
class TransmitterConfig:
    """Parsed transmitter definition used before live Sionna objects exist.

    Mobility and orientation stay as generator-side policies here.  The initial
    position property resolves the starting point that ``SceneService`` needs
    when it creates the live Sionna transmitter.
    """

    name: str
    mobility: MobilityPattern
    orientation: Optional[OrientationSpec | PreparedOrientationSource] = None
    _initial_position: Optional[Tuple] = None
    power_dbm: Optional[float] = None

    def __post_init__(self):
        if self.orientation is None:
            self.orientation = FixedOrientationSpec()
        mobility = cast(Any, self.mobility)
        if self._initial_position is not None and hasattr(mobility, "start_pos"):
            # mobility.start_pos and initial_position may both name the start,
            # but they must resolve to one physical position.
            mobility_start = point_to_tuple(mobility.start_pos)
            config_start = point_to_tuple(self._initial_position)
            if mobility_start != config_start:
                raise ValueError(
                    f"TransmitterConfig {self.name}: mobility.start_pos {mobility_start} doesn't match initial_position {config_start}"
                )

    @property
    def initial_position(self):
        """Return the transmitter start position expected by scene creation."""
        mobility = cast(Any, self.mobility)
        if hasattr(mobility, "start_pos"):
            return mobility.start_pos
        elif hasattr(mobility, "transmitter_pos") and mobility.transmitter_pos:
            tx_pos = mobility.transmitter_pos
            return point_to_tuple(tx_pos)
        elif hasattr(mobility, "receiver_pos") and mobility.receiver_pos:
            rx_pos = mobility.receiver_pos
            return point_to_tuple(rx_pos)
        if self._initial_position is None:
            raise ValueError(
                f"TransmitterConfig {self.name}: {self.mobility.__class__.__name__} requires explicit initial_position"
            )
        return self._initial_position


@dataclass
class ReceiverConfig:
    """Parsed receiver definition used before live Sionna objects exist."""

    name: str
    mobility: MobilityPattern
    orientation: Optional[OrientationSpec | PreparedOrientationSource] = None
    _initial_position: Optional[Tuple] = None

    def __post_init__(self):
        if self.orientation is None:
            self.orientation = FixedOrientationSpec()
        mobility = cast(Any, self.mobility)
        if self._initial_position is not None and hasattr(mobility, "start_pos"):
            # mobility.start_pos and initial_position may both name the start,
            # but they must resolve to one physical position.
            mobility_start = point_to_tuple(mobility.start_pos)
            config_start = point_to_tuple(self._initial_position)
            if mobility_start != config_start:
                raise ValueError(
                    f"ReceiverConfig {self.name}: mobility.start_pos {mobility_start} doesn't match initial_position {config_start}"
                )

    @property
    def initial_position(self):
        """Return the receiver start position expected by scene creation."""
        mobility = cast(Any, self.mobility)
        if hasattr(mobility, "start_pos"):
            return mobility.start_pos
        elif hasattr(mobility, "receiver_pos") and mobility.receiver_pos:
            rx_pos = mobility.receiver_pos
            return point_to_tuple(rx_pos)
        elif hasattr(mobility, "transmitter_pos") and mobility.transmitter_pos:
            tx_pos = mobility.transmitter_pos
            return point_to_tuple(tx_pos)
        elif self._initial_position is not None:
            return self._initial_position
        else:
            raise ValueError(
                f"ReceiverConfig {self.name}: {self.mobility.__class__.__name__} requires explicit initial_position"
            )


@dataclass
class CoverageConfig:
    """Normalized coverage-map settings consumed by the coverage package.

    ``CoverageService`` fills this from YAML-shaped coverage blocks.  The solver
    and writer then consume this object rather than reaching back into the
    scenario model.
    """

    enabled: bool = DEFAULT_COVERAGE_ENABLED
    bbox: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    bbox_xy: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    resolution: Tuple[float, float] = DEFAULT_COVERAGE_RESOLUTION_M
    heights: Optional[List[float]] = None
    metric: str = DEFAULT_COVERAGE_METRIC
    stride: int = DEFAULT_COVERAGE_STRIDE
    quality: str = DEFAULT_COVERAGE_QUALITY_PRESET
    max_paths: int = DEFAULT_COVERAGE_MAX_PATHS
    tx_combination: str = DEFAULT_COVERAGE_TX_COMBINATION
    tx_index: Optional[int] = None
    tx_mode: str = DEFAULT_COVERAGE_TX_MODE
    tx_selected: Optional[Union[int, str]] = None
    metrics_store: Optional[List[str]] = None
    metrics_derived: Optional[List[str]] = None
    # RadioMapSolver-specific path termination controls.
    rr_depth: Optional[int] = None
    rr_prob: Optional[float] = None
    stop_threshold: Optional[float] = None
    seed: Optional[int] = None
    solver_settings: Optional[Dict[str, Any]] = None
    # Optional transmit precoding vector passed through to coverage solving.
    precoding_vec: Optional[List[float]] = None
    save_path: Optional[str] = None
    save_compression: str = DEFAULT_COVERAGE_SAVE_COMPRESSION


@dataclass
class SensingConfig:
    """Settings carried for an optional sensing-capable pipeline extension.

    Runs without that extension leave sensing disabled. Enabling these settings
    requires the active installation to provide the processor.

    Carrier frequency comes from ``SimulationConfig.carrier_frequency_hz``.
    ``bandwidth`` is the radar chirp bandwidth and remains distinct from
    ``SimulationConfig.bandwidth_hz`` for the channel model.
    """

    enabled: bool = DEFAULT_SENSING_ENABLED

    # Waveform parameters
    bandwidth: float = DEFAULT_SENSING_BANDWIDTH_HZ
    chirps_per_frame: int = DEFAULT_SENSING_CHIRPS_PER_FRAME
    prf_hz: float = DEFAULT_SENSING_PRF_HZ  # Pulse Repetition Frequency
    # Range interpretation: "bistatic" (path length) or "monostatic" (round-trip/2)
    range_mode: str = DEFAULT_SENSING_RANGE_MODE

    # Processing parameters
    fft_size_range: int = DEFAULT_SENSING_FFT_SIZE_RANGE
    fft_size_doppler: int = DEFAULT_SENSING_FFT_SIZE_DOPPLER
    window_range: str = DEFAULT_SENSING_WINDOW
    window_doppler: str = DEFAULT_SENSING_WINDOW

    # Detection parameters
    cfar_type: str = DEFAULT_SENSING_CFAR_TYPE  # cell-averaging
    cfar_guard_cells: int = DEFAULT_SENSING_CFAR_GUARD_CELLS
    cfar_ref_cells: int = DEFAULT_SENSING_CFAR_REF_CELLS
    cfar_threshold_scale: float = DEFAULT_SENSING_CFAR_THRESHOLD_SCALE
    cfar_os_rank: float = DEFAULT_SENSING_CFAR_OS_RANK
    cfar_rd_guard_cells_range: int = DEFAULT_SENSING_CFAR_RD_GUARD_CELLS_RANGE
    cfar_rd_guard_cells_doppler: int = DEFAULT_SENSING_CFAR_RD_GUARD_CELLS_DOPPLER
    cfar_rd_ref_cells_range: int = DEFAULT_SENSING_CFAR_RD_REF_CELLS_RANGE
    cfar_rd_ref_cells_doppler: int = DEFAULT_SENSING_CFAR_RD_REF_CELLS_DOPPLER
    cfar_min_snr_db: float = DEFAULT_SENSING_CFAR_MIN_SNR_DB  # Minimum SNR, dB.
    pre_cfar_doppler_edge_guard_bins: int = DEFAULT_SENSING_PRE_CFAR_DOPPLER_EDGE_GUARD_BINS
    pre_cfar_doppler_dc_guard_bins: int = DEFAULT_SENSING_PRE_CFAR_DOPPLER_DC_GUARD_BINS

    # Physical plausibility gate on detections (optional)
    min_detection_range_m: Optional[float] = None
    max_detection_range_m: Optional[float] = None

    # Multi-frame track confirmation (optional)
    track_confirmation_enabled: bool = DEFAULT_SENSING_TRACK_CONFIRMATION_ENABLED
    track_confirmation_m: int = DEFAULT_SENSING_TRACK_CONFIRMATION_M
    track_confirmation_n: int = DEFAULT_SENSING_TRACK_CONFIRMATION_N
    track_max_missed_frames: int = DEFAULT_SENSING_TRACK_MAX_MISSED_FRAMES
    track_association_range_gate_m: float = DEFAULT_SENSING_TRACK_ASSOCIATION_RANGE_GATE_M
    track_association_velocity_gate_m_s: float = DEFAULT_SENSING_TRACK_ASSOCIATION_VELOCITY_GATE_M_S
    track_output_tentative: bool = DEFAULT_SENSING_TRACK_OUTPUT_TENTATIVE
    tracker_type: str = DEFAULT_SENSING_TRACKER_TYPE

    # Evaluation/association gates for TP/FP/FN metrics
    eval_association_range_gate_m: float = DEFAULT_SENSING_EVAL_ASSOCIATION_RANGE_GATE_M
    eval_association_velocity_gate_m_s: float = DEFAULT_SENSING_EVAL_ASSOCIATION_VELOCITY_GATE_M_S

    # Clutter removal
    clutter_removal_enabled: bool = DEFAULT_SENSING_CLUTTER_REMOVAL_ENABLED
    clutter_removal_window: int = DEFAULT_SENSING_CLUTTER_REMOVAL_WINDOW

    # Display options (summary figures only, not processing)
    display_range_xlim: Optional[list] = None  # [min_m, max_m] to zoom RD figure
    display_velocity_ylim: Optional[list] = None  # [min_m_s, max_m_s] to zoom RD figure

    # Output control
    output_range_doppler: bool = DEFAULT_SENSING_OUTPUT_RANGE_DOPPLER
    output_range_profile: bool = DEFAULT_SENSING_OUTPUT_RANGE_PROFILE
    output_detections: bool = DEFAULT_SENSING_OUTPUT_DETECTIONS
    persist_detection_stage_artifacts: bool = DEFAULT_SENSING_PERSIST_DETECTION_STAGE_ARTIFACTS
    raw_payload_policy: str = DEFAULT_SENSING_RAW_PAYLOAD_POLICY
    rng_seed: Optional[int] = None

    # Noise injection (postprocessor can also inject noise independently)
    noise_enabled: bool = DEFAULT_SENSING_NOISE_ENABLED
    noise_snr_db: Optional[float] = None

    # AoA filtering: discard paths outside angular bounds before CIR computation.
    # Angles in degrees, using Sionna convention (az: 0-360, el: elevation).
    # Bounds are applied after wrapping azimuth to [-180, 180).
    aoa_filter_enabled: bool = DEFAULT_SENSING_AOA_FILTER_ENABLED
    aoa_az_min_deg: Optional[float] = None
    aoa_az_max_deg: Optional[float] = None
    aoa_el_min_deg: Optional[float] = None
    aoa_el_max_deg: Optional[float] = None

    # Processing mode selection: "auto" (default), "coherent_cpi",
    # "snapshot_direct_rd", or "sequential_stft".
    processing_mode: str = DEFAULT_SENSING_PROCESSING_MODE

    def __post_init__(self):
        if self.enabled:
            if self.bandwidth <= 0:
                raise ConfigurationError("Sensing bandwidth must be positive")
            if self.prf_hz <= 0:
                raise ConfigurationError("Sensing PRF must be positive")
        # Normalize policy-like strings here so downstream processors only need
        # to handle canonical values.
        policy = self.raw_payload_policy
        if not isinstance(policy, str):
            logger.warning(
                "Invalid raw_payload_policy %r, defaulting to %s",
                self.raw_payload_policy,
                DEFAULT_SENSING_RAW_PAYLOAD_POLICY,
            )
            policy = DEFAULT_SENSING_RAW_PAYLOAD_POLICY
        policy = policy.lower()
        if policy not in VALID_SENSING_RAW_PAYLOAD_POLICIES:
            logger.warning(
                "Invalid raw_payload_policy '%s', defaulting to %s",
                policy,
                DEFAULT_SENSING_RAW_PAYLOAD_POLICY,
            )
            policy = DEFAULT_SENSING_RAW_PAYLOAD_POLICY
        self.raw_payload_policy = policy
        if self.rng_seed is not None:
            try:
                self.rng_seed = int(self.rng_seed)
            except (TypeError, ValueError):
                logger.warning("Invalid sensing rng_seed %r; clearing", self.rng_seed)
                self.rng_seed = None
        if self.noise_enabled and self.noise_snr_db is None:
            logger.warning("noise_enabled=True but noise_snr_db is None; disabling noise")
            self.noise_enabled = False

        mode = (self.range_mode or DEFAULT_SENSING_RANGE_MODE).lower()
        if mode not in VALID_SENSING_RANGE_MODES:
            logger.warning(
                "Invalid range_mode '%s', defaulting to %s",
                self.range_mode,
                DEFAULT_SENSING_RANGE_MODE,
            )
            mode = DEFAULT_SENSING_RANGE_MODE
        self.range_mode = mode

        pm = (self.processing_mode or DEFAULT_SENSING_PROCESSING_MODE).lower()
        if pm not in VALID_SENSING_PROCESSING_MODES:
            logger.warning(
                "Invalid processing_mode '%s', defaulting to %s",
                self.processing_mode,
                DEFAULT_SENSING_PROCESSING_MODE,
            )
            pm = DEFAULT_SENSING_PROCESSING_MODE
        self.processing_mode = pm

        tracker_type = str(self.tracker_type or DEFAULT_SENSING_TRACKER_TYPE).lower()
        if tracker_type not in VALID_SENSING_TRACKER_TYPES:
            logger.warning(
                "Invalid tracker_type '%s', defaulting to %s",
                self.tracker_type,
                DEFAULT_SENSING_TRACKER_TYPE,
            )
            tracker_type = DEFAULT_SENSING_TRACKER_TYPE
        self.tracker_type = tracker_type

        if self.min_detection_range_m is not None:
            self.min_detection_range_m = float(self.min_detection_range_m)
            if self.min_detection_range_m < 0.0:
                raise ConfigurationError("min_detection_range_m must be >= 0")
        if self.max_detection_range_m is not None:
            self.max_detection_range_m = float(self.max_detection_range_m)
            if self.max_detection_range_m <= 0.0:
                raise ConfigurationError("max_detection_range_m must be > 0")
        if (
            self.min_detection_range_m is not None
            and self.max_detection_range_m is not None
            and self.min_detection_range_m >= self.max_detection_range_m
        ):
            raise ConfigurationError(
                "min_detection_range_m must be smaller than max_detection_range_m"
            )

        self.track_confirmation_m = max(1, int(self.track_confirmation_m))
        self.track_confirmation_n = max(self.track_confirmation_m, int(self.track_confirmation_n))
        self.track_max_missed_frames = max(0, int(self.track_max_missed_frames))
        self.track_association_range_gate_m = max(0.0, float(self.track_association_range_gate_m))
        self.track_association_velocity_gate_m_s = max(
            0.0, float(self.track_association_velocity_gate_m_s)
        )
        self.pre_cfar_doppler_edge_guard_bins = int(self.pre_cfar_doppler_edge_guard_bins)
        self.pre_cfar_doppler_dc_guard_bins = int(self.pre_cfar_doppler_dc_guard_bins)
        if self.pre_cfar_doppler_edge_guard_bins < 0:
            raise ConfigurationError("pre_cfar_doppler_edge_guard_bins must be >= 0")
        if self.pre_cfar_doppler_dc_guard_bins < 0:
            raise ConfigurationError("pre_cfar_doppler_dc_guard_bins must be >= 0")

        self.eval_association_range_gate_m = max(0.0, float(self.eval_association_range_gate_m))
        self.eval_association_velocity_gate_m_s = max(
            0.0, float(self.eval_association_velocity_gate_m_s)
        )


@dataclass
class AntennaConfig:
    """Generator-facing description of a Sionna planar antenna array."""

    pattern: str = DEFAULT_ANTENNA_PATTERN
    polarization: str = DEFAULT_ANTENNA_POLARIZATION
    num_rows: int = DEFAULT_ANTENNA_NUM_ROWS
    num_cols: int = DEFAULT_ANTENNA_NUM_COLS
    vertical_spacing: float = DEFAULT_ANTENNA_VERTICAL_SPACING
    horizontal_spacing: float = DEFAULT_ANTENNA_HORIZONTAL_SPACING

    def __post_init__(self) -> None:
        if self.pattern not in VALID_ANTENNA_PATTERNS:
            raise ConfigurationError(
                f"Invalid antenna pattern: {self.pattern!r}. "
                f"Available: {sorted(VALID_ANTENNA_PATTERNS)}"
            )
        if self.polarization not in VALID_POLARIZATIONS:
            raise ConfigurationError(
                f"Invalid polarization: {self.polarization!r}. "
                f"Available: {sorted(VALID_POLARIZATIONS)}"
            )
        if self.num_rows < 1:
            raise ConfigurationError(f"num_rows must be >= 1, got {self.num_rows}")
        if self.num_cols < 1:
            raise ConfigurationError(f"num_cols must be >= 1, got {self.num_cols}")
        if self.vertical_spacing <= 0:
            raise ConfigurationError(f"vertical_spacing must be > 0, got {self.vertical_spacing}")
        if self.horizontal_spacing <= 0:
            raise ConfigurationError(
                f"horizontal_spacing must be > 0, got {self.horizontal_spacing}"
            )


@dataclass
class SimulationConfig:
    """Top-level normalized config shared by one generator run.

    Services intentionally share this object by reference.  Some setup steps
    normalize fields such as coverage options or ``num_steps``, and later
    services should observe those run-local updates.
    """

    scene_name: str = DEFAULT_SCENE_NAME
    duration: float = DEFAULT_DURATION_S
    num_steps: int = DEFAULT_NUM_STEPS
    start_step: int = DEFAULT_START_STEP
    quality: str = DEFAULT_QUALITY_PRESET
    view: str = DEFAULT_VIEW
    generate_video: bool = DEFAULT_GENERATE_VIDEO
    debug_level: str = DEFAULT_DEBUG_LEVEL
    file_format: str = DEFAULT_FILE_FORMAT  # File-output route currently writes HDF5 frames.
    output_mode: str = DEFAULT_OUTPUT_MODE
    grpc_config: Optional[Dict[str, Any]] = None
    scene_material_scattering_coefficient_preset: str = (
        DEFAULT_SCENE_MATERIAL_SCATTERING_COEFFICIENT_PRESET
    )
    coverage: Optional[CoverageConfig] = None
    export_path_metrics: bool = DEFAULT_EXPORT_PATH_METRICS

    # Radio-frequency parameters assigned to the Sionna scene by SceneService.
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY_HZ
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ
    temperature_k: Optional[float] = DEFAULT_TEMPERATURE_K

    # Mesh animation cadence is separate from the ray-tracing cadence.  None
    # means targets may update their mesh on every solved RT step.
    mesh_update_interval_s: Optional[float] = DEFAULT_MESH_UPDATE_INTERVAL_S

    # Sionna-native CIR evolution: when > 1, paths.cir() generates multiple
    # phase-coherent time samples from one ray-tracing solve.
    cir_time_steps: int = DEFAULT_CIR_TIME_STEPS
    cir_sampling_frequency_hz: Optional[float] = DEFAULT_CIR_SAMPLING_FREQUENCY_HZ

    # Per-material scattering/thickness overrides from YAML
    material_overrides: Optional[Dict[str, Dict[str, Any]]] = None

    # Dictionaries mapping names to config objects
    transmitters: Optional[Dict[str, TransmitterConfig]] = None
    receivers: Optional[Dict[str, ReceiverConfig]] = None

    # Optional sub-module configs
    sensing: Optional[SensingConfig] = None
    tx_antenna: Optional[AntennaConfig] = None
    rx_antenna: Optional[AntennaConfig] = None

    def __post_init__(self):
        if self.coverage is None:
            self.coverage = CoverageConfig()
        if self.sensing is None:
            self.sensing = SensingConfig()
        if self.transmitters is None:
            self.transmitters = {}
        if self.receivers is None:
            self.receivers = {}
        if self.cir_time_steps > 1 and self.cir_sampling_frequency_hz is None:
            raise ConfigurationError(
                "cir_sampling_frequency_hz is required when cir_time_steps > 1"
            )
        if (
            self.sensing.enabled
            and self.cir_time_steps > 1
            and self.sensing.prf_hz
            and self.cir_sampling_frequency_hz
            and abs(self.sensing.prf_hz - self.cir_sampling_frequency_hz) > 0.01
        ):
            logger.warning(
                "sensing.prf_hz (%.1f) != cir_sampling_frequency_hz (%.1f) — "
                "these should typically match for coherent processing",
                self.sensing.prf_hz,
                self.cir_sampling_frequency_hz,
            )
        # Coherent CIR mode solves only at CIR boundaries, so target mesh
        # cadence is quantized to those boundaries when both features are used.
        if self.cir_time_steps > 1 and self.mesh_update_interval_s is not None:
            from ..target.mesh import mesh_update_step_interval

            mesh_step_interval = mesh_update_step_interval(
                duration=self.duration,
                num_steps=self.num_steps,
                mesh_update_interval_s=self.mesh_update_interval_s,
            )
            if mesh_step_interval is not None:
                effective_step_interval = (
                    max(1, math.ceil(mesh_step_interval / self.cir_time_steps))
                    * self.cir_time_steps
                )
                if mesh_step_interval != self.cir_time_steps:
                    logger.warning(
                        "mesh_update_interval_s gives nominal step_interval=%d but "
                        "coherent mode only updates meshes on cir_time_steps=%d "
                        "boundaries; effective mesh cadence will be quantized to %d steps",
                        mesh_step_interval,
                        self.cir_time_steps,
                        effective_step_interval,
                    )

    QUALITY_PRESETS = QUALITY_PRESETS
    AVAILABLE_SCENES = AVAILABLE_SCENES

    def is_custom_xml(self) -> bool:
        """Return whether ``scene_name`` points to a scenario XML file."""
        return self.scene_name.endswith(".xml")

    def get_scene_display_name(self) -> str:
        """Return a compact scene label for logs and output metadata."""
        if self.is_custom_xml():
            return self.scene_name.split("/")[-1].replace(".xml", "")
        return self.scene_name

    def get_quality_profile(self) -> Dict[str, Any]:
        """Get quality settings for the current quality preset.

        When quality is ``"custom"``, the base profile is ``"medium"``.
        Custom overrides are merged on top by ``raytracing_service``.
        """
        if self.quality == "custom":
            return dict(self.QUALITY_PRESETS[CUSTOM_QUALITY_BASE_PRESET])
        return dict(
            self.QUALITY_PRESETS.get(self.quality, self.QUALITY_PRESETS[DEFAULT_QUALITY_PRESET])
        )

    def get_coverage_quality_settings(self) -> Dict[str, Any]:
        """Map the coverage quality preset to RadioMapSolver parameters.

        Coverage uses a grid solver with different keyword names and cost
        tradeoffs. Its default budget is independent from ordinary MPC solving,
        so this method reads ``coverage.quality`` rather than ``quality``.

        Returns:
            Dict with keys: max_depth, samples_per_tx, los, specular_reflection,
            diffuse_reflection, refraction, diffraction
        """
        coverage_quality = (
            self.coverage.quality if self.coverage is not None else DEFAULT_COVERAGE_QUALITY_PRESET
        )
        profile = self.QUALITY_PRESETS.get(
            coverage_quality,
            self.QUALITY_PRESETS[DEFAULT_COVERAGE_QUALITY_PRESET],
        )
        settings = {
            "max_depth": int(profile.get("max_depth", 3)),
            # Map ray tracing samples_per_src to coverage samples_per_tx
            "samples_per_tx": int(profile.get("samples_per_src", 100000)),
            "los": bool(profile.get("los", True)),
            "specular_reflection": bool(profile.get("specular_reflection", True)),
            "diffuse_reflection": bool(profile.get("diffuse_reflection", True)),
            "refraction": bool(profile.get("refraction", True)),
            "diffraction": bool(profile.get("diffraction", False)),
        }
        return settings

    def validate(self):
        """Validate cross-field constraints before services allocate runtime state."""
        valid_qualities = set(self.QUALITY_PRESETS) | {"custom"}
        if self.quality not in valid_qualities:
            raise ConfigurationError(
                f"Invalid quality: {self.quality}. Available: {sorted(valid_qualities)}"
            )
        if self.view not in VALID_VIEWS:
            raise ConfigurationError(
                f"Invalid view: {self.view}. Available: {', '.join(VALID_VIEWS)}"
            )
        if self.num_steps < 1:
            raise ConfigurationError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.start_step < 0:
            raise ConfigurationError(f"start_step must be >= 0, got {self.start_step}")
        if self.start_step >= self.num_steps:
            raise ConfigurationError(
                f"start_step ({self.start_step}) must be < num_steps ({self.num_steps})"
            )
        if self.is_custom_xml():
            import os

            if not os.path.exists(self.scene_name):
                raise ConfigurationError(f"Custom XML file not found: {self.scene_name}")
        if not self.is_custom_xml() and self.scene_name not in self.AVAILABLE_SCENES:
            logger.warning(f"'{self.scene_name}' is not in the predefined scenes list.")
            logger.warning(
                "This might be a custom scene or a scene from a different Sionna RT version."
            )
            logger.warning(f"Available predefined scenes: {list(self.AVAILABLE_SCENES.keys())}")
            logger.warning("Use a path ending in .xml for custom XML files.")
            logger.warning(f"Attempting to load '{self.scene_name}' anyway.")
        if self.file_format not in VALID_FILE_FORMATS:
            raise ConfigurationError(
                f"Invalid file_format: {self.file_format}. Only HDF5 format is supported."
            )
        if self.output_mode not in VALID_OUTPUT_MODES:
            raise ConfigurationError(
                f"Invalid output_mode: {self.output_mode}. Available: local, grpc"
            )
        if self.output_mode == "grpc" and not self.grpc_config:
            logger.warning(
                "gRPC mode selected without grpc_config; using the default live endpoint"
            )
        if (
            self.scene_material_scattering_coefficient_preset
            not in VALID_SCENE_MATERIAL_SCATTERING_COEFFICIENT_PRESETS
        ):
            raise ConfigurationError(
                "scene_material_scattering_coefficient_preset must be 'none' or 'itu', "
                f"got {self.scene_material_scattering_coefficient_preset!r}"
            )


def build_simulation_config(*args, **kwargs):
    """Build a SimulationConfig from a ScenarioConfiguration.

    Delegates to configuration.loader.load_simulation_config.
    """
    from .loader import load_simulation_config

    return load_simulation_config(*args, **kwargs)
