"""Offline post-processor for sensing data.

Re-runs detection pipeline (RD map reconstruction, MTI, CFAR, NMS, DBSCAN,
range gating) from saved HDF5 frames without requiring GPU or regeneration.

Supports noise injection, GT-based evaluation, and CFAR threshold sweeps.

Usage as library::

    from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor

    pp = SensingPostProcessor(PostProcessingConfig(cfar_threshold_scale=5.0))
    result = pp.process_from_slow_time_matrix(stm, meta)

Usage as CLI::

    python -m shared.sensing.postprocessor frames_coherent/ --cfar-threshold 5.0 -o analysis/cfar-5/

    # With noise and evaluation:
    python -m shared.sensing.postprocessor frames_coherent/ --noise-snr 20 --evaluate --figures -o eval/

    # With a threshold sweep:
    python -m shared.sensing.postprocessor frames_coherent/ --threshold-sweep \
        --threshold-range 1.0 15.0 -o threshold-sweep/
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid5

import numpy as np

from shared.extensions.sensing import (
    SENSING_DENSE_DATASET_KEYS,
    SENSING_SCHEMA_VERSION,
    assert_sensing_contract,
    build_sensing_temporal_contract,
    canonical_processing_recipe_id,
    canonical_sensing_hash,
    resolve_sensing_config,
)
from shared.frames.manifest import FRAMES_MANIFEST_FILENAME
from shared.frames.providers import Hdf5Provider
from shared.frames.types import StandardMPCFrame
from shared.scenarios.frame_paths import resolve_scenario_frames_dir
from shared.sensing.angle import select_range_doppler_angle_product
from shared.sensing.cfr_replay import processing_config_dict, select_frequency_observation
from shared.sensing.detection.angle import AnglePeakDetectionConfig
from shared.sensing.detection.beam_scan import (
    refine_array_directions,
    run_beam_scan_detection_pipeline,
    visible_array_beams,
)
from shared.sensing.detection.cfar import CFAREdgePolicy
from shared.sensing.detection.pipeline import (
    DetectionFilter,
    DetectionPipelineConfig,
    DetectionPipelineResult,
    run_angle_enhanced_detection_pipeline,
    run_detection_pipeline,
    run_vote_fusion_pipeline,
)
from shared.sensing.detection.vote import (
    ProjectionCfarConfig,
    VoteFusionConfig,
    derive_aoa_az_map_from_cube,
)
from shared.sensing.evaluation import (
    DEFAULT_RANGE_GATE_M,
    DEFAULT_VELOCITY_GATE_M_S,
    MOTMetrics,
    ScenarioEvaluationResult,
    ThresholdSweepPoint,
    associate_detections_to_gt,
    compute_mot_metrics,
    evaluate_scenario,
)
from shared.sensing.evaluation_truth import evaluation_velocity_mask
from shared.sensing.gt import GT_CHANNEL_PRECEDENCE
from shared.sensing.loader import iter_frames_sensing, load_frames_sensing, load_saved_results
from shared.sensing.plotting.figures import (
    compute_vertex_velocity_overlay,
    generate_figures,
    generate_micro_doppler_figures,
    generate_rd_gif,
    generate_rd_gif_3d,
    generate_threshold_sweep_figures,
    load_cir_sequence,
    load_gt_entries_for_tracking,
)
from shared.sensing.products import (
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
    select_product_by_contract,
)
from shared.sensing.range_response import RangeResponseConfig
from shared.sensing.signal_processing import (
    C0,
    RangeDopplerResult,
    _range_axis,
    _window,
    _window_energy_correction,
    apply_detection_velocity_metadata,
    apply_pre_cfar_doppler_guards,
    cir_to_samples,
    inject_target_relative_complex_awgn,
    remove_window_weighted_mean,
)
from shared.sensing.tracker import (
    CartesianKalmanTracker,
    JPDAFTracker,
    KalmanTracker,
    MofNTracker,
    TrackerConfig,
    TrackerInterface,
    prune_correlated_ghost_tracks,
)
from shared.sensing.writer import (
    ColumnarResultWriter,
    _json_default,
    artifacts_to_dict,
    build_reprocessed_sensing_block,
    save_stage_artifacts,
    validate_postprocessed_result,
)

logger = logging.getLogger(__name__)

_SENSING_INSTALL_GUIDANCE = "Install ORCHAV with: python -m pip install orchav"
_VERTEX_OVERLAY_INSTALL_GUIDANCE = (
    "Vertex overlay uses trimesh, which is not included in the standard ORCHAV installation. "
    'Install it separately with: python -m pip install "trimesh>=3.20"'
)
_SENSING_DEPENDENCY_NAMES = {
    "scipy": "SciPy",
    "sklearn": "scikit-learn",
    "matplotlib": "Matplotlib",
    "PIL": "Pillow",
}
_SENSING_DEPENDENCY_PROBES = {
    "scipy": ("scipy.optimize", "scipy.ndimage"),
    "sklearn": ("sklearn.cluster",),
    "matplotlib": ("matplotlib.pyplot",),
    "PIL": ("PIL.Image",),
    "trimesh": ("trimesh",),
}
_HUNGARIAN_TRACKERS = frozenset({"kalman", "jpdaf", "cartesian", "cartesian3d"})

FIGURE_CHOICES = (
    "detection_stages",
    "tracking_decisions",
    "range_doppler",
    "range_profile",
    "range_azimuth",
    "doppler_azimuth",
    "angular_overview",
    "azimuth_elevation",
    "range_elevation",
    "detections_3d",
    "detection_timeline",
    "tracking",
    "tracking_xy",
    "tracking_xyz",
    "angular_tracking",
    "angular_errors",
    "detection_summary",
    "detection_events",
    "cumulative_detection_performance",
    "error_distribution",
    "detections_per_frame",
)


@dataclass(frozen=True)
class SensingEvaluationEventKey:
    """Exact temporal identity used to join sensing truth and results."""

    clock_id: str
    acquisition_id: str
    pair_id: str
    reference_time_s: float


@dataclass(frozen=True)
class _JoinedEvaluationEvent:
    """One identity-aligned truth/result event, including absent counterparts."""

    key: SensingEvaluationEventKey
    target_ids: tuple[str, ...]
    gt_ranges: np.ndarray
    gt_velocities: np.ndarray
    detections: tuple[dict[str, Any], ...]
    frame_present: bool
    result_present: bool
    gt_velocity_valid: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class _EvaluationEventJoin:
    """Identity join plus explicit missing/extra result diagnostics."""

    events: tuple[_JoinedEvaluationEvent, ...]
    missing_result_keys: tuple[SensingEvaluationEventKey, ...]
    extra_result_keys: tuple[SensingEvaluationEventKey, ...]
    skipped_invalid_truth_keys: tuple[SensingEvaluationEventKey, ...]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PostProcessingConfig:
    """Tunable parameters for offline re-processing."""

    # Windowing
    # ``None`` inherits exact persisted v4 recipe fields. Schema-less direct
    # array calls retain their historical Hann behavior in
    # ``_effective_processing_windows``.
    window_range: str | None = None
    window_doppler: str | None = None
    range_response: RangeResponseConfig | None = None
    range_ifft_size: int | None = None

    # MTI clutter removal
    mti_enabled: bool = True
    # None inherits the coherent source recipe; explicit mean suppression should
    # use windowed_mean. Schema-less historical calls retain their old operator.
    coherent_clutter_mode: str | None = None

    # 2-D CFAR
    array_detection_mode: str = "rd"
    # Offline angle estimation from retained per-element complex samples.
    # One preserves the original detected beam coordinates exactly.
    beam_angle_oversampling: int = 1
    cfar_type: str = "ca"
    cfar_threshold_scale: float = 3.0
    cfar_min_snr_db: float = 6.0
    cfar_max_dynamic_range_db: float | None = None
    cfar_guard_cells_range: int = 2
    cfar_guard_cells_doppler: int = 1
    cfar_ref_cells_range: int = 8
    cfar_ref_cells_doppler: int = 4
    cfar_os_rank: float = 0.75
    cfar_edge_policy: CFAREdgePolicy = "full_window"
    pre_cfar_doppler_edge_guard_bins: int | None = None
    pre_cfar_doppler_dc_guard_bins: int | None = None

    # NMS
    nms_radius_doppler: int = 2
    nms_radius_range: int = 4

    # DBSCAN clustering (physical-space gates)
    cluster_range_gate_m: float = 1.0
    cluster_velocity_gate_m_s: float = 5.0

    # Range gating
    min_detection_range_m: float | None = None
    max_detection_range_m: float | None = None

    # Ghost filter (shortest-range per velocity group)
    ghost_filter_velocity_gate_m_s: float | None = None  # None = disabled
    ghost_filter_azimuth_gate_deg: float | None = None  # None = disabled

    # Vote fusion (postprocessor-only, requires stored azimuth cube)
    vote_fusion_enabled: bool = False
    vote_min_votes: int = 2
    vote_raz_threshold_scale: float | None = None
    vote_daz_threshold_scale: float | None = None
    vote_azimuth_guard_cells: int = 4
    vote_azimuth_ref_cells: int = 8
    vote_pre_threshold_db: float = 20.0
    vote_nms_max_peaks: int = 200
    vote_post_nms_min_peak_divisor: float = 55.0

    # Angle-enhanced RD detection (postprocessor-only, requires stored azimuth cube)
    angle_peak_detection_enabled: bool = False
    angle_max_peaks_per_rd_cell: int = 2
    angle_peak_relative_floor_db: float = 10.0
    angle_peak_nms_radius_bins: int = 1
    angle_edge_exclusion_bins: int = 0
    angle_secondary_peak_max_gap_db: float = 3.0
    angle_secondary_peak_min_separation_bins: int = 2
    angle_secondary_peak_min_prominence_db: float = 3.0
    angle_secondary_peak_min_profile_snr_db: float = 6.0
    angle_secondary_peak_min_valley_depth_db: float = 3.0
    angle_cluster_azimuth_gate_deg: float | None = None
    angle_sidelobe_suppression_enabled: bool = False
    angle_sidelobe_range_tolerance_bins: int = 2
    angle_sidelobe_doppler_tolerance_bins: int = 2
    angle_sidelobe_azimuth_tolerance_bins: int = 3
    angle_sidelobe_min_power_ratio_db: float = 4.0

    # Per-frame tracker
    tracker_type: str = "none"  # "none", "m_of_n", "kalman", "jpdaf", "cartesian", "cartesian3d"
    tracker_m: int = 2
    tracker_n: int = 3
    tracker_confirmation_mode: str = dataclass_field(default="latched", kw_only=True)
    tracker_max_missed_frames: int = 1
    tracker_range_gate_m: float = 1.0
    tracker_velocity_gate_m_s: float = 1.0
    tracker_azimuth_gate_deg: float | None = None
    tracker_azimuth_cost_scale_deg: float | None = None
    tracker_min_lifetime: int = 0  # Minimum hits before output (0 = use tracker_m)
    tracker_xy_gate_m: float = 2.0  # Euclidean distance gate for Cartesian tracker (metres)
    tracker_acceleration_psd_m2_s3: float | None = None
    ghost_track_correlation_threshold: float | None = None  # None = disabled
    ghost_track_correlation_min_overlap: int = 5  # Minimum overlap frames for correlation

    # JPDAF-specific
    jpda_pd: float = 0.9  # Detection probability
    jpda_beta_fa: float = 0.01  # False alarm spatial density
    jpda_gate_threshold: float = 9.21  # Chi-squared gate (2 DOF, 99%)
    jpda_max_joint_nodes: int = dataclass_field(default=100_000, kw_only=True)
    jpda_birth_confirmation_frames: int = dataclass_field(default=3, kw_only=True)
    jpda_birth_separation_chi2: float = dataclass_field(default=1.0, kw_only=True)

    # M-of-N tracker parameters
    m_of_n_m: int | None = None
    m_of_n_n: int | None = None
    m_of_n_range_tol_m: float = 0.5
    m_of_n_velocity_tol_m_s: float = 0.3

    # Dimensionless target-relative complex AWGN.  These historical option
    # names do not imply thermal-noise physics or an absolute link-budget scale.
    noise_enabled: bool = False
    noise_snr_db: float | None = None
    noise_rng_seed: int | None = None

    # Skip the detection pipeline entirely (produce RD map / range profile only)
    skip_detection: bool = False

    # Optional debug observability
    include_tracking_diagnostics: bool = dataclass_field(default=False, kw_only=True)
    include_stage_artifacts: bool = False

    def __post_init__(self) -> None:
        if self.range_response is not None:
            self.range_response = RangeResponseConfig.parse(self.range_response)
        if self.range_ifft_size is not None and (
            type(self.range_ifft_size) is not int or self.range_ifft_size < 2
        ):
            raise ValueError("range_ifft_size must be an integer >= 2")
        if self.cfar_max_dynamic_range_db is not None:
            value = self.cfar_max_dynamic_range_db
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError("cfar_max_dynamic_range_db must be finite and nonnegative")
            if self.vote_fusion_enabled or self.angle_peak_detection_enabled:
                raise ValueError("The dynamic-range threshold supports rd and beam_scan detection")
        factor = self.beam_angle_oversampling
        if isinstance(factor, bool) or not isinstance(factor, int) or not 1 <= factor <= 16:
            raise ValueError("beam_angle_oversampling must be an integer from 1 to 16")
        if factor > 1 and self.array_detection_mode != "beam_scan":
            raise ValueError("beam_angle_oversampling requires array_detection_mode='beam_scan'")
        if self.array_detection_mode not in {"rd", "beam_scan"}:
            raise ValueError("array_detection_mode must be 'rd' or 'beam_scan'")
        if self.array_detection_mode == "beam_scan":
            if self.vote_fusion_enabled or self.angle_peak_detection_enabled:
                raise ValueError(
                    "beam_scan cannot be combined with angle-peak or vote-fusion detection"
                )
            if self.noise_enabled:
                raise ValueError(
                    "beam_scan uses retained angular products; noise changes require rebuilding per-element products"
                )


_TYPED_ANGULAR_POSTPROCESSING_DEFERRED = (
    "typed P,E,A,D,R postprocessing not implemented; use rd or beam_scan detection"
)


def _angular_postprocessing_requested(config: PostProcessingConfig) -> bool:
    """Return whether a request needs the deferred typed angular consumer."""

    return bool(
        (
            config.ghost_filter_azimuth_gate_deg is not None
            and config.ghost_filter_azimuth_gate_deg > 0.0
            and config.array_detection_mode != "beam_scan"
        )
        or config.vote_fusion_enabled
        or config.angle_peak_detection_enabled
    )


def validate_typed_processing_config(config: PostProcessingConfig) -> None:
    """Reject deferred algorithms consistently at typed API and workflow boundaries."""

    if config.coherent_clutter_mode not in {None, "none", "legacy", "windowed_mean"}:
        raise ValueError("coherent_clutter_mode must be none, legacy or windowed_mean")
    if not config.mti_enabled and config.coherent_clutter_mode not in {None, "none"}:
        raise ValueError("Disabled MTI conflicts with an explicit mean-removal mode")
    if _angular_postprocessing_requested(config):
        raise ValueError(_TYPED_ANGULAR_POSTPROCESSING_DEFERRED)


@dataclass(frozen=True)
class _TypedDetectionInput:
    """One schema-v4 product selected without storage-key or dtype inference."""

    data_key: str
    product_id: str
    kind: str
    representation: str
    values: np.ndarray
    range_axis_m: np.ndarray
    descriptor: Mapping[str, Any]
    rx_element_axis: tuple[dict[str, Any], ...] | None = None
    slow_time_axis_s: np.ndarray | None = None
    doppler_axis_hz: np.ndarray | None = None
    combine_policy: str | None = None


# ---------------------------------------------------------------------------
# Post-processor
# ---------------------------------------------------------------------------


class SensingPostProcessor:
    """Recompute RD maps and/or detections from saved sensing data."""

    def __init__(
        self,
        config: PostProcessingConfig | None = None,
        *,
        tracker_factory: Callable[[PostProcessingConfig], TrackerInterface | None] | None = None,
    ) -> None:
        """Use an optional research tracker implementing the timestamped interface.

        Record custom algorithm settings in a PostProcessingConfig subclass so
        saved recipe identities describe the executed processor. The factory is
        called lazily once for each processor instance; no truth is passed to it.
        """
        self.config = config or PostProcessingConfig()
        if tracker_factory is None:
            self._validate_builtin_tracker_type(self.config.tracker_type)
        self._uses_builtin_tracker = tracker_factory is None
        self._tracker_factory = tracker_factory or self._create_tracker
        self._tracker_range_mode: str | None = None
        self._noise_rng = np.random.default_rng(
            int(self.config.noise_rng_seed) if self.config.noise_rng_seed is not None else None
        )
        self._tracker: TrackerInterface | None = None
        self._tracker_initialized = False

    def _ensure_tracker(self, range_mode: str = "monostatic") -> None:
        """Bind built-in state dynamics to the resolved observation convention."""
        if self._tracker_initialized:
            if (
                self._uses_builtin_tracker
                and self._tracker is not None
                and self._tracker_range_mode != range_mode
            ):
                raise ValueError(
                    "A stateful tracker cannot mix monostatic range and bistatic path length"
                )
            return
        if self._uses_builtin_tracker:
            self._tracker = self._create_tracker(self.config, range_mode=range_mode)
        else:
            self._tracker = self._tracker_factory(self.config)
        self._tracker_range_mode = range_mode
        self._tracker_initialized = True

    @staticmethod
    def _validate_builtin_tracker_type(tracker_type: str) -> None:
        supported = {"none", "m_of_n", "kalman", "cartesian", "cartesian3d", "jpdaf"}
        if not isinstance(tracker_type, str) or tracker_type.lower() not in supported:
            raise ValueError(
                f"Unsupported tracker_type {tracker_type!r}; choose one of {sorted(supported)} "
                "or provide a custom tracker_factory"
            )

    @staticmethod
    def _create_tracker(
        config: PostProcessingConfig,
        *,
        range_mode: str = "monostatic",
    ) -> CartesianKalmanTracker | JPDAFTracker | KalmanTracker | MofNTracker | None:
        SensingPostProcessor._validate_builtin_tracker_type(config.tracker_type)
        tracker_type = config.tracker_type.lower()
        # Legacy --m-of-n flag overrides tracker_type
        if tracker_type == "none" and config.m_of_n_m is not None and config.m_of_n_n is not None:
            tracker_type = "m_of_n"
        if tracker_type == "none":
            return None
        if range_mode not in {"monostatic", "bistatic"}:
            raise ValueError("Tracker range_mode must be monostatic or bistatic")
        if range_mode == "bistatic" and tracker_type in {"cartesian", "cartesian3d"}:
            raise ValueError(
                "Native bistatic reports require explicit localization before Cartesian tracking"
            )
        m = config.m_of_n_m if config.m_of_n_m is not None else config.tracker_m
        n = config.m_of_n_n if config.m_of_n_n is not None else config.tracker_n
        rg = (
            config.m_of_n_range_tol_m
            if config.m_of_n_m is not None
            else config.tracker_range_gate_m
        )
        vg = (
            config.m_of_n_velocity_tol_m_s
            if config.m_of_n_m is not None
            else config.tracker_velocity_gate_m_s
        )
        tracker_cfg = TrackerConfig(
            confirmation_mode=config.tracker_confirmation_mode,
            m=m,
            n=n,
            max_missed_frames=config.tracker_max_missed_frames,
            range_gate_m=rg,
            velocity_gate_m_s=vg,
            azimuth_gate_deg=config.tracker_azimuth_gate_deg,
            azimuth_cost_scale_deg=config.tracker_azimuth_cost_scale_deg,
            min_track_lifetime=config.tracker_min_lifetime,
            xy_gate_m=config.tracker_xy_gate_m,
            record_diagnostics=config.include_tracking_diagnostics,
            range_rate_scale=2.0 if range_mode == "bistatic" else 1.0,
        )
        acceleration_kwargs = (
            {}
            if config.tracker_acceleration_psd_m2_s3 is None
            else {"acceleration_psd_m2_s3": config.tracker_acceleration_psd_m2_s3}
        )
        if tracker_type == "cartesian3d":
            return CartesianKalmanTracker(
                tracker_cfg,
                include_z=True,
                **acceleration_kwargs,
            )
        if tracker_type == "cartesian":
            return CartesianKalmanTracker(tracker_cfg, **acceleration_kwargs)
        if tracker_type == "kalman":
            return KalmanTracker(tracker_cfg, **acceleration_kwargs)
        if tracker_type == "jpdaf":
            return JPDAFTracker(
                tracker_cfg,
                pd=config.jpda_pd,
                beta_fa=config.jpda_beta_fa,
                gate_threshold=config.jpda_gate_threshold,
                max_joint_nodes=config.jpda_max_joint_nodes,
                birth_confirmation_frames=config.jpda_birth_confirmation_frames,
                birth_separation_chi2=config.jpda_birth_separation_chi2,
                **acceleration_kwargs,
            )
        if tracker_type == "m_of_n":
            return MofNTracker(tracker_cfg)
        raise ValueError(f"Unsupported tracker_type {tracker_type!r}")

    # ---- entry points by data availability ----

    def process_from_frequency_observation(
        self,
        sensing: dict[str, Any],
        *,
        update_time_s: float | None = None,
    ) -> dict[str, Any]:
        """Rebuild CFR, Doppler and compatible spatial/detector products on CPU."""
        from functools import partial

        validate_typed_processing_config(self.config)
        from shared.sensing.arrays import planar_spatial_fft, strongest_array_directions
        from shared.sensing.cfr_replay import rebuild_frequency_observation
        from shared.sensing.detection.beam_scan import refine_array_directions, visible_array_beams

        cfg = self._resolve_meta(sensing)
        if self.config.noise_enabled or self.config.noise_snr_db is not None:
            raise ValueError(
                "CFR replay reuses retained frequency noise; legacy noise settings conflict"
            )
        if self.config.window_range not in {None, "boxcar"}:
            raise ValueError("CFR replay requires window_range: boxcar; use frequency_window")
        rebuilt = rebuild_frequency_observation(
            sensing,
            response=self.config.range_response,
            fft_size=self.config.range_ifft_size,
            window_doppler=self.config.window_doppler,
            mti_enabled=self.config.mti_enabled,
            coherent_clutter_mode=self.config.coherent_clutter_mode,
        )
        rd = rebuilt["rd_cube"]
        rd_result = RangeDopplerResult(
            power_linear=abs(rd) ** 2,
            cube=rd,
            range_axis_m=rebuilt["range_axis_m"],
            doppler_axis_hz=rebuilt["doppler_axis_hz"],
            range_axis_label=(
                "Range (m)" if cfg["range_mode"] == "monostatic" else "Path Length (m)"
            ),
            doppler_axis_label="Doppler (Hz)",
            metadata=rebuilt["metadata"],
        )
        detection_kwargs = {}
        if rebuilt["delay"].shape[0] > 1:
            array = sensing["rx_array"]
            spatial_args = dict(
                num_rows=array["rows"],
                num_cols=array["columns"],
                horizontal_spacing_wavelengths=array["horizontal_spacing_wavelengths"],
                vertical_spacing_wavelengths=array["vertical_spacing_wavelengths"],
            )
            spatial = planar_spatial_fft(rebuilt["element_rd"], **spatial_args)
            power = abs(spatial.spectrum) ** 2
            rebuilt["spatial"] = spatial
            azimuth, elevation = strongest_array_directions(
                power, spatial.elevation_direction_cosine, spatial.azimuth_direction_cosine
            )
            azimuth[rd_result.power_linear <= 0] = np.nan
            detection_kwargs["aoa_az_map"] = azimuth
            if array["rows"] > 1:
                elevation[rd_result.power_linear <= 0] = np.nan
                detection_kwargs["aoa_el_map"] = elevation
            if self.config.array_detection_mode == "beam_scan" and not self.config.skip_detection:
                detection_kwargs["beam_array_shape"] = (array["rows"], array["columns"])
                detection_kwargs["beam_inputs"] = visible_array_beams(
                    power, spatial.elevation_direction_cosine, spatial.azimuth_direction_cosine
                )
                if self.config.cfar_max_dynamic_range_db is not None:
                    raw = planar_spatial_fft(rebuilt["element_rd_raw"], **spatial_args)
                    beams, az, el = visible_array_beams(
                        abs(raw.spectrum) ** 2,
                        raw.elevation_direction_cosine,
                        raw.azimuth_direction_cosine,
                    )
                    detection_kwargs["beam_reference"] = {
                        "basis": "pre_clutter_spatial_fft_peak",
                        "azimuth_deg": az.tolist(),
                        "elevation_deg": el.tolist(),
                        "power_linear": beams.max(axis=(1, 2)).tolist(),
                    }
                if self.config.beam_angle_oversampling > 1:
                    detection_kwargs["beam_angle_refiner"] = partial(
                        refine_array_directions,
                        element_rd=rebuilt["element_rd"],
                        oversampling=self.config.beam_angle_oversampling,
                        **spatial_args,
                    )
        detection = (
            DetectionPipelineResult(detections=[], metadata={}, stages=None)
            if self.config.skip_detection
            else self._detect(rd_result, cfg, update_time_s=update_time_s, **detection_kwargs)
        )
        return {
            "rd_map": rd_result.power_linear,
            "rd_cube": rd,
            "range_axis": rebuilt["range_axis_m"],
            "doppler_axis": rebuilt["doppler_axis_hz"],
            "range_mode": cfg["range_mode"],
            "detections": detection.detections,
            "detection_filter_metadata": detection.metadata,
            "stage_artifacts": _artifacts_to_dict(detection.stages),
            "config": processing_config_dict(self.config),
            "cfr_rebuilt": rebuilt,
        }

    def process_from_slow_time_matrix(
        self,
        slow_time_matrix: np.ndarray,
        sensing_meta: dict[str, Any],
        *,
        range_axis_m: np.ndarray | None = None,
        slow_time_axis_s: np.ndarray | None = None,
        update_time_s: float | None = None,
    ) -> dict[str, Any]:
        """Recompute RD map + detections from a saved slow_time_matrix.

        The slow_time_matrix has shape ``(fft_size_doppler, fft_size_range)``
        (slow-time rows x fast-time columns).

        Args:
            slow_time_matrix: Complex slow-time matrix from HDF5.
            sensing_meta: Sensing config dict (from frame ``sensing.config``
                or ``sensing.config_resolved``).
            range_axis_m: Exact persisted range coordinates. Direct array
                callers may omit them to retain the low-level config-derived
                API; persisted-frame callers must provide them.
            slow_time_axis_s: Exact persisted sample times. When supplied,
                Doppler coordinates are derived only from their declared,
                validated uniform spacing.

        Returns:
            Dict with ``rd_map``, ``range_axis``, ``doppler_axis``,
            ``detections``, and ``config``.
        """
        if self.config.array_detection_mode == "beam_scan":
            raise ValueError(
                "beam_scan requires retained angular products, not combined slow-time samples"
            )
        cfg = self._resolve_meta(sensing_meta)
        if cfg.get("_native_v4_config") is True:
            validate_typed_processing_config(self.config)
        is_cfr = cfg.get("range_response", {}).get("model") == "cfr_ifft"
        if is_cfr and range_axis_m is None:
            raise ValueError(
                "CFR delay processing requires exact retained range_axis_m coordinates"
            )
        if self.config.range_response is not None or self.config.range_ifft_size is not None:
            raise ValueError(
                "Frequency window/IFFT overrides require process_from_frequency_observation and retained CFR"
            )
        if is_cfr and (self.config.noise_enabled or self.config.noise_snr_db is not None):
            raise ValueError(
                "CFR delay samples already include observation noise; legacy noise settings conflict"
            )

        fft_r = int(cfg["fft_size_range"])
        fft_d = int(cfg["fft_size_doppler"])
        fs = float(cfg["sampling_frequency"])
        prf = float(cfg["prf_hz"])
        range_mode = cfg["range_mode"]

        stm = np.asarray(slow_time_matrix)
        if stm.ndim != 2:
            raise ValueError(f"slow_time_matrix must be 2-D, got shape {stm.shape}")
        if stm.shape != (fft_d, fft_r):
            raise ValueError(
                "slow_time_matrix shape must match resolved "
                f"[fft_size_doppler,fft_size_range]=[{fft_d},{fft_r}]; got {stm.shape}"
            )

        # Target-relative complex AWGN (before windowing, on raw samples).
        noise_injection: dict[str, Any] | None = None
        if self.config.noise_enabled and self.config.noise_snr_db is not None:
            noise_result = inject_target_relative_complex_awgn(
                stm,
                self.config.noise_snr_db,
                rng=self._noise_rng,
            )
            stm = noise_result.samples
            noise_injection = {
                "contract_version": 1,
                "applications": [
                    {**noise_result.metadata(), "stage": "postprocessor_slow_time_input"}
                ],
            }

        # Windowing. Persisted v4 data inherits its exact stored recipe unless
        # the caller supplied an explicit postprocessing override.
        window_range, window_doppler = _effective_processing_windows(cfg, self.config)
        if is_cfr and window_range != "boxcar":
            raise ValueError(
                "CFR requires window_range: boxcar; rewindow retained frequency observations instead"
            )
        wr = _window(window_range, fft_r)
        wd = _window(window_doppler, stm.shape[0])
        wr_c = _window_energy_correction(wr)
        wd_c = _window_energy_correction(wd)
        windowed = stm * wr[np.newaxis, :] * wr_c

        validate_typed_processing_config(self.config)
        clutter_mode = self.config.coherent_clutter_mode
        if clutter_mode is None:
            clutter_mode = cfg.get("coherent_clutter_mode") or cfg.get("processing_recipe", {}).get(
                "coherent_clutter_mode", "legacy" if is_cfr else "unweighted_mean"
            )
        if clutter_mode not in {"none", "legacy", "windowed_mean", "unweighted_mean"}:
            raise ValueError(f"Unsupported saved coherent clutter mode: {clutter_mode!r}")
        if not self.config.mti_enabled:
            clutter_mode = "none"
        mean_applied = clutter_mode != "none" and (
            clutter_mode != "legacy" or bool(np.max(np.abs(np.diff(windowed, axis=0))) > 0)
        )
        if mean_applied:
            windowed = (
                remove_window_weighted_mean(windowed, wd, axis=0)
                if clutter_mode == "windowed_mean"
                else windowed - windowed.mean(axis=0, keepdims=True)
            )
        rd_processing = {
            "coherent_clutter_mode": clutter_mode,
            "slow_time_mean_removal_applied": mean_applied,
            "slow_time_mean_removal_policy": {
                "none": "disabled",
                "legacy": "when_row_variation_nonzero",
                "windowed_mean": "window_weighted_mean",
                "unweighted_mean": "unconditional_unweighted_mean",
            }[clutter_mode],
        }

        windowed = windowed * wd[:, np.newaxis] * wd_c

        # Doppler FFT
        rd_cube = np.fft.fftshift(np.fft.fft(windowed, n=fft_d, axis=0), axes=0) / np.sqrt(fft_d)
        rd_power = np.abs(rd_cube) ** 2

        if range_axis_m is None:
            range_axis, range_label = _range_axis(fft_r, fs, range_mode)
        else:
            range_axis = _dense_coordinate(
                range_axis_m,
                data_key="slow_time_matrix",
                axis_name="range",
                size=fft_r,
            )
            range_label = (
                "Monostatic range (m)"
                if str(range_mode).lower().startswith("mono")
                else "Bistatic path length (m)"
            )
        persisted_slow_time_axis: np.ndarray | None = None
        if slow_time_axis_s is None:
            doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_d, d=1.0 / prf))
        else:
            persisted_slow_time_axis = _dense_coordinate(
                slow_time_axis_s,
                data_key="slow_time_matrix",
                axis_name="slow_time",
                size=stm.shape[0],
            )
            if persisted_slow_time_axis.size == 1:
                doppler_axis = np.zeros((fft_d,), dtype=np.float64)
            else:
                sample_deltas = np.diff(persisted_slow_time_axis)
                sample_period_s = float(sample_deltas[0])
                tolerance = max(abs(sample_period_s) * 1e-9, 1e-12)
                if sample_period_s <= 0.0 or not np.allclose(
                    sample_deltas,
                    sample_period_s,
                    rtol=1e-9,
                    atol=tolerance,
                ):
                    raise ValueError(
                        "Typed slow_time coordinates must be strictly increasing and "
                        "uniform for Doppler FFT reprocessing"
                    )
                exact_prf_hz = 1.0 / sample_period_s
                if not np.isclose(
                    exact_prf_hz,
                    prf,
                    rtol=1e-9,
                    atol=max(abs(prf) * 1e-9, 1e-9),
                ):
                    raise ValueError(
                        "Typed slow_time spacing disagrees with config_resolved.prf_hz: "
                        f"coordinates imply {exact_prf_hz}, config declares {prf}"
                    )
                doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_d, d=sample_period_s))

        rd_result = RangeDopplerResult(
            power_linear=rd_power,
            range_axis_m=range_axis,
            doppler_axis_hz=doppler_axis,
            cube=rd_cube,
            range_axis_label=range_label,
            doppler_axis_label="Doppler (Hz)",
        )

        if self.config.skip_detection:
            detection_result = DetectionPipelineResult(detections=[], metadata={}, stages=None)
        else:
            detection_result = self._detect(rd_result, cfg, update_time_s=update_time_s)
        velocity_meta: dict[str, object] = {}
        apply_detection_velocity_metadata(velocity_meta, range_mode=range_mode)

        return {
            "rd_map": rd_power,
            "rd_cube": rd_cube,
            "rd_processing": rd_processing,
            "range_axis": range_axis,
            "doppler_axis": doppler_axis,
            **(
                {"slow_time_axis": persisted_slow_time_axis}
                if persisted_slow_time_axis is not None
                else {}
            ),
            "range_mode": range_mode,
            "detections": detection_result.detections,
            "detection_filter_metadata": detection_result.metadata,
            **({"noise_injection": noise_injection} if noise_injection is not None else {}),
            **(
                {"stage_artifacts": _artifacts_to_dict(detection_result.stages)}
                if detection_result.stages is not None
                else {}
            ),
            **velocity_meta,
            "config": processing_config_dict(self.config),
        }

    def process_from_cir_frames(
        self,
        cir_list: list[tuple[np.ndarray, np.ndarray]],
        sensing_meta: dict[str, Any],
        *,
        update_time_s: float | None = None,
    ) -> dict[str, Any]:
        """Recompute RD map + detections from per-frame CIR taps.

        Args:
            cir_list: List of ``(amplitudes, delays_s)`` tuples, one per pulse.
            sensing_meta: Sensing config dict.

        Returns:
            Dict with ``rd_map``, ``range_axis``, ``doppler_axis``,
            ``detections``, and ``config``.
        """
        cfg = self._resolve_meta(sensing_meta)

        fft_r = int(cfg["fft_size_range"])
        fs = float(cfg["sampling_frequency"])

        # Build slow-time matrix from CIR taps
        fast_rows = []
        for cir, delays in cir_list:
            samples = cir_to_samples(cir, delays, sampling_frequency=fs, fft_size=fft_r)
            fast_rows.append(samples)

        stm = np.stack(fast_rows, axis=0)
        return self.process_from_slow_time_matrix(
            stm,
            sensing_meta,
            update_time_s=update_time_s,
        )

    def process_from_rd_map(
        self,
        rd_power: np.ndarray,
        sensing_meta: dict[str, Any],
        *,
        range_axis_m: np.ndarray | None = None,
        doppler_axis_hz: np.ndarray | None = None,
        rd_cube: np.ndarray | None = None,
        aoa_az_map: np.ndarray | None = None,
        aoa_el_map: np.ndarray | None = None,
        azimuth_cube: np.ndarray | None = None,
        azimuth_axis_deg: np.ndarray | list[float] | None = None,
        update_time_s: float | None = None,
    ) -> dict[str, Any]:
        """Re-run detection pipeline on an existing RD map.

        Args:
            rd_power: Power RD map, shape ``(doppler_bins, range_bins)``.
            sensing_meta: Sensing config dict.
            range_axis_m: Exact persisted range coordinates. Direct array
                callers may omit this low-level override.
            doppler_axis_hz: Exact persisted Doppler-frequency coordinates.
            rd_cube: Optional complex RD cube for SNR computation.
            aoa_az_map: Optional azimuth estimate per RD cell in degrees.
            aoa_el_map: Optional elevation estimate per RD cell in degrees.
            azimuth_cube: Optional RD-azimuth power cube.
            azimuth_axis_deg: Optional azimuth axis corresponding to
                ``azimuth_cube``.

        Returns:
            Dict with ``rd_map``, ``range_axis``, ``doppler_axis``,
            ``detections``, and ``config``.
        """
        cfg = self._resolve_meta(sensing_meta)
        if self.config.coherent_clutter_mode is not None:
            raise ValueError(
                "Clutter overrides require retained complex slow-time or frequency samples; "
                "they cannot be applied to an already processed RD map"
            )
        if self.config.range_response is not None or self.config.range_ifft_size is not None:
            raise ValueError(
                "Frequency window/IFFT overrides require process_from_frequency_observation and retained CFR"
            )
        if cfg.get("range_response", {}).get("model") == "cfr_ifft":
            if self.config.noise_enabled or self.config.noise_snr_db is not None:
                raise ValueError(
                    "CFR RD products already include observation noise; legacy noise settings conflict"
                )
            if self.config.window_range not in {None, "boxcar"}:
                raise ValueError(
                    "CFR requires window_range: boxcar; rebuild retained frequency observations instead"
                )
            if range_axis_m is None:
                raise ValueError(
                    "CFR RD processing requires exact retained range_axis_m coordinates"
                )
        native_v4 = cfg.get("_native_v4_config") is True
        if native_v4:
            if self.config.array_detection_mode == "beam_scan":
                stored_windows = _native_v4_processing_windows(cfg)
                for window_name, stored in zip(("window_range", "window_doppler"), stored_windows):
                    requested = getattr(self.config, window_name)
                    if requested is not None and requested != stored:
                        raise ValueError(
                            "beam_scan window changes require rebuilding per-element products"
                        )
            validate_typed_processing_config(self.config)
            # A normal 2-D reprocessing request may coexist with an optional
            # typed angular product, but it must not reinterpret legacy maps.
            aoa_az_map = None
            aoa_el_map = None
            azimuth_cube = None
            azimuth_axis_deg = None

        fft_r = int(cfg["fft_size_range"])
        fs = float(cfg["sampling_frequency"])
        prf = float(cfg["prf_hz"])
        range_mode = cfg["range_mode"]

        rd_power = np.asarray(rd_power)
        if rd_power.ndim != 2:
            raise ValueError(f"rd_power must be 2-D, got shape {rd_power.shape}")
        if rd_power.shape[1] != fft_r:
            raise ValueError(
                f"rd_power range dimension {rd_power.shape[1]} does not match "
                f"fft_size_range={fft_r}"
            )
        fft_d = rd_power.shape[0]
        if range_axis_m is None:
            range_axis, range_label = _range_axis(fft_r, fs, range_mode)
        else:
            range_axis = _dense_coordinate(
                range_axis_m,
                data_key="range_doppler",
                axis_name="range",
                size=rd_power.shape[1],
            )
            range_label = (
                "Monostatic range (m)"
                if str(range_mode).lower().startswith("mono")
                else "Bistatic path length (m)"
            )
        if doppler_axis_hz is None:
            doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_d, d=1.0 / prf))
        else:
            doppler_axis = _dense_coordinate(
                doppler_axis_hz,
                data_key="range_doppler",
                axis_name="doppler_frequency",
                size=fft_d,
            )
        resolved_azimuth_axis = azimuth_axis_deg
        if resolved_azimuth_axis is None and not native_v4 and isinstance(sensing_meta, dict):
            resolved_azimuth_axis = sensing_meta.get("azimuth_axis_deg")
        if aoa_az_map is None and azimuth_cube is not None and resolved_azimuth_axis is not None:
            aoa_az_map = derive_aoa_az_map_from_cube(
                azimuth_cube,
                resolved_azimuth_axis,
            )

        rd_result = RangeDopplerResult(
            power_linear=rd_power,
            range_axis_m=range_axis,
            doppler_axis_hz=doppler_axis,
            cube=rd_cube,
            range_axis_label=range_label,
            doppler_axis_label="Doppler (Hz)",
        )

        beam_inputs = None
        beam_array_shape = None
        angular_product_id = None
        if native_v4 and self.config.array_detection_mode == "rd":
            angular = select_range_doppler_angle_product(sensing_meta)
            if angular is not None:
                if not np.array_equal(angular.range_axis_m, range_axis) or not np.array_equal(
                    angular.doppler_axis_hz, doppler_axis
                ):
                    raise ValueError("RD angular and signal coordinate grids must agree exactly")
                from shared.sensing.arrays import strongest_array_directions

                # Recover the native joint-peak direction from the declared
                # spatial product. Untyped compatibility arguments above stay
                # ignored; an RD-only replay must not silently lose saved AoA.
                aoa_az_map, joint_el_map = strongest_array_directions(
                    angular.power_linear,
                    angular.elevation_direction_cosine,
                    angular.azimuth_direction_cosine,
                )
                aoa_az_map[rd_power <= 0] = np.nan
                if len(angular.elevation_direction_cosine) > 1:
                    aoa_el_map = joint_el_map
                    aoa_el_map[rd_power <= 0] = np.nan
                angular_product_id = angular.product_id
        if self.config.array_detection_mode == "beam_scan":
            if native_v4:
                angular = select_range_doppler_angle_product(sensing_meta)
                if angular is None:
                    raise ValueError(
                        "beam_scan requires a retained typed range-Doppler-angle product"
                    )
                if not np.array_equal(angular.range_axis_m, range_axis) or not np.array_equal(
                    angular.doppler_axis_hz, doppler_axis
                ):
                    raise ValueError("beam_scan angular and RD coordinate grids must agree exactly")
                beam_array_shape = tuple(angular.power_linear.shape[:2])
                beam_inputs = visible_array_beams(
                    angular.power_linear,
                    angular.elevation_direction_cosine,
                    angular.azimuth_direction_cosine,
                )
                angular_product_id = angular.product_id
            elif azimuth_cube is not None and resolved_azimuth_axis is not None:
                # Explicit low-level ULA array API: these are broadside angles,
                # not untyped aliases read from a persisted schema-v4 frame.
                beam_array_shape = (1, int(np.asarray(azimuth_cube).shape[0]))
                beam_inputs = visible_array_beams(
                    np.asarray(azimuth_cube)[np.newaxis, ...],
                    np.array([0.0]),
                    np.sin(np.deg2rad(resolved_azimuth_axis)),
                )
            else:
                raise ValueError("beam_scan requires angular power and its physical coordinates")

        beam_angle_refiner = None
        angle_estimation = None
        if self.config.beam_angle_oversampling > 1 and not self.config.skip_detection:
            beam_angle_refiner, angle_estimation = self._beam_angle_refiner(sensing_meta)
        if self.config.skip_detection:
            detection_result = DetectionPipelineResult(detections=[], metadata={}, stages=None)
        else:
            detection_result = self._detect(
                rd_result,
                cfg,
                aoa_az_map=aoa_az_map,
                aoa_el_map=aoa_el_map,
                azimuth_cube=azimuth_cube,
                azimuth_axis_deg=resolved_azimuth_axis,
                update_time_s=update_time_s,
                beam_inputs=beam_inputs,
                beam_array_shape=beam_array_shape,
                beam_reference=sensing_meta.get("detector_power_reference") if native_v4 else None,
                beam_angle_refiner=beam_angle_refiner,
            )
        velocity_meta: dict[str, object] = {}
        apply_detection_velocity_metadata(velocity_meta, range_mode=range_mode)

        result = {
            "rd_map": rd_power,
            "rd_cube": rd_cube,
            "range_axis": range_axis,
            "doppler_axis": doppler_axis,
            "range_mode": range_mode,
            "detections": detection_result.detections,
            "detection_filter_metadata": detection_result.metadata,
            **(
                {"stage_artifacts": _artifacts_to_dict(detection_result.stages)}
                if detection_result.stages is not None
                else {}
            ),
            **velocity_meta,
            "config": processing_config_dict(self.config),
        }
        if angular_product_id is not None:
            result["angular_input_product_id"] = angular_product_id
            result["detection_filter_metadata"]["angular_input_product_id"] = angular_product_id
        if angle_estimation is not None:
            result["detection_filter_metadata"]["angle_estimation"] = angle_estimation
        if aoa_az_map is not None:
            result["range_doppler_aoa_az_map"] = np.array(aoa_az_map, copy=True)
        if aoa_el_map is not None:
            result["range_doppler_aoa_el_map"] = np.array(aoa_el_map, copy=True)
        if azimuth_cube is not None:
            result["range_doppler_azimuth_cube"] = np.array(azimuth_cube, copy=True)
        if resolved_azimuth_axis is not None:
            result["azimuth_axis_deg"] = np.array(resolved_azimuth_axis, copy=True)
        resolved_el_axis = (
            sensing_meta.get("elevation_axis_deg") if isinstance(sensing_meta, dict) else None
        )
        if resolved_el_axis is not None:
            result["elevation_axis_deg"] = np.array(resolved_el_axis, copy=True)
        for key in ("beamforming_mode", "antenna_rows", "antenna_cols"):
            value = sensing_meta.get(key) if isinstance(sensing_meta, dict) else None
            if value is not None:
                result[key] = value
        return result

    # ---- detection pipeline ----

    def _beam_angle_refiner(self, sensing: dict[str, Any]) -> tuple[Callable, dict[str, Any]]:
        """Return the angle refiner and its exact retained observation provenance.

        Power-only angular products cannot supply interpolated complex array
        phases. Require the retained native analytical-Doppler element samples
        and inherit their declared window/mean-removal recipe.
        """
        if (
            sensing.get("schema_version") != SENSING_SCHEMA_VERSION
            or sensing.get("processing_strategy") != "coherent_cpi_fft"
        ):
            raise ValueError(
                "Beam angle refinement requires native analytical-Doppler element samples"
            )
        source_key = "slow_time_by_pair_rx_element"
        delay_reconstructed = source_key not in sensing
        if delay_reconstructed:
            if any(
                d.get("kind") == "baseband_frequency_observation"
                for d in sensing.get("products", {}).values()
            ):
                from shared.sensing.cfr_replay import reconstruct_delay_observation

                source_key, _, _, source_descriptor = select_frequency_observation(sensing)
                sensing = reconstruct_delay_observation(sensing)
            else:
                raise ValueError(
                    "Beam angle refinement requires retained slow_time_by_pair_rx_element or canonical frequency observations; use recording.profile: reprocessing or full"
                )
        assert_sensing_contract(
            {"sensing": sensing}, require_schema_version=True, require_described_products=True
        )
        if not delay_reconstructed:
            source_descriptor = sensing["products"][source_key]
        array = sensing["rx_array"]
        rows, cols = int(array["rows"]), int(array["columns"])
        raw = np.asarray(sensing["slow_time_by_pair_rx_element"])
        if raw.ndim != 4 or raw.shape[1] != rows * cols:
            raise ValueError("Retained slow-time element dimensions do not match the array")
        pair = int(sensing.get("pair_index", 0))
        h = raw[pair]
        cfg = _resolve_native_v4_config(sensing)
        if h.shape[1:] != (int(cfg["fft_size_doppler"]), int(cfg["fft_size_range"])):
            raise ValueError(
                "Angle refinement requires one complete native CPI on the retained RD grid"
            )
        range_window, doppler_window = _native_v4_processing_windows(cfg)
        wr, wd = _window(range_window, h.shape[2]), _window(doppler_window, h.shape[1])
        values = h * wr[None, None, :] * _window_energy_correction(wr)
        if sensing["slow_time_mean_removal_applied"]:
            policy = sensing.get("slow_time_mean_removal_policy")
            if policy == "window_weighted_mean":
                values = remove_window_weighted_mean(values, wd, axis=1)
            elif policy == "when_row_variation_nonzero":
                values = values - values.mean(axis=1, keepdims=True)
            else:
                raise ValueError("Unsupported recorded mean-removal policy")
        values = values * wd[None, :, None] * _window_energy_correction(wd)
        element_rd = np.fft.fftshift(np.fft.fft(values, axis=1), axes=1) / np.sqrt(h.shape[1])
        refiner = partial(
            refine_array_directions,
            element_rd=element_rd,
            num_rows=rows,
            num_cols=cols,
            horizontal_spacing_wavelengths=array["horizontal_spacing_wavelengths"],
            vertical_spacing_wavelengths=array["vertical_spacing_wavelengths"],
            oversampling=self.config.beam_angle_oversampling,
        )
        return refiner, {
            "method": "local_zero_padded_spatial_fft",
            "oversampling": self.config.beam_angle_oversampling,
            "input": source_key,
            "stage": "after_coarse_beam_gates_before_tracking",
            "increases_aperture_resolution": False,
            "source_product_id": source_descriptor["product_id"],
            "delay_reconstructed": delay_reconstructed,
        }

    def _detect(
        self,
        rd_result: RangeDopplerResult,
        cfg: dict[str, Any],
        *,
        aoa_az_map: np.ndarray | None = None,
        aoa_el_map: np.ndarray | None = None,
        azimuth_cube: np.ndarray | None = None,
        azimuth_axis_deg: np.ndarray | list[float] | None = None,
        update_time_s: float | None = None,
        beam_inputs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        beam_array_shape: tuple[int, int] | None = None,
        beam_angle_refiner: Callable | None = None,
        beam_reference: dict[str, Any] | None = None,
    ) -> DetectionPipelineResult:
        """Run CFAR -> NMS -> DBSCAN -> range gate on an RD map."""
        self._ensure_tracker(cfg["range_mode"])
        post_detection_filter: DetectionFilter | None = None
        if self._tracker is not None:
            if update_time_s is None:
                raise ValueError("update_time_s is required when a sensing tracker is enabled")
            tracker = self._tracker

            def track_detections(
                detections: list[dict[str, Any]],
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                return tracker.filter_detections(
                    detections,
                    update_time_s=float(update_time_s),
                )

            post_detection_filter = track_detections
        if beam_angle_refiner is not None:
            tracker_filter = post_detection_filter

            def refine_then_track(detections):
                refined = beam_angle_refiner(detections)
                if tracker_filter is not None:
                    return tracker_filter(refined)
                return refined, {"enabled": False, "raw": len(refined), "kept": len(refined)}

            post_detection_filter = refine_then_track
        carrier_hz = float(cfg["carrier_freq_hz"])
        wavelength = C0 / carrier_hz if carrier_hz > 0 else 1.0
        edge_guard_bins, dc_guard_bins = self._resolve_pre_cfar_guard_bins(cfg)
        rd_power_for_detection = (
            apply_pre_cfar_doppler_guards(
                rd_result.power_linear,
                edge_guard_bins=edge_guard_bins,
                dc_guard_bins=dc_guard_bins,
            )
            if edge_guard_bins > 0 or dc_guard_bins > 0
            else rd_result.power_linear
        )
        pipeline_cfg = DetectionPipelineConfig(
            cfar_guard_cells=(
                self.config.cfar_guard_cells_doppler,
                self.config.cfar_guard_cells_range,
            ),
            cfar_reference_cells=(
                self.config.cfar_ref_cells_doppler,
                self.config.cfar_ref_cells_range,
            ),
            cfar_threshold_scale=self.config.cfar_threshold_scale,
            cfar_method=self.config.cfar_type.lower(),
            cfar_os_rank=self.config.cfar_os_rank,
            cfar_edge_policy=self.config.cfar_edge_policy,
            nms_suppression_radius=(
                self.config.nms_radius_doppler,
                self.config.nms_radius_range,
            ),
            cluster_range_gate_m=self.config.cluster_range_gate_m,
            cluster_velocity_gate_m_s=self.config.cluster_velocity_gate_m_s,
            min_snr_db=self.config.cfar_min_snr_db,
            cfar_max_dynamic_range_db=self.config.cfar_max_dynamic_range_db,
            min_detection_range_m=self.config.min_detection_range_m,
            max_detection_range_m=self.config.max_detection_range_m,
            ghost_filter_velocity_gate_m_s=self.config.ghost_filter_velocity_gate_m_s,
            ghost_filter_azimuth_gate_deg=self.config.ghost_filter_azimuth_gate_deg,
        )
        if self.config.array_detection_mode == "beam_scan":
            if beam_inputs is None:
                raise ValueError("beam_scan requires resolved angular beam inputs")
            beams, azimuths, elevations = beam_inputs
            return run_beam_scan_detection_pipeline(
                rd_result,
                beam_power=beams,
                beam_azimuth_deg=azimuths,
                beam_elevation_deg=elevations,
                num_rows=beam_array_shape[0] if beam_array_shape is not None else None,
                num_cols=beam_array_shape[1] if beam_array_shape is not None else None,
                reference=beam_reference,
                config=pipeline_cfg,
                wavelength_m=wavelength,
                edge_guard_bins=edge_guard_bins,
                dc_guard_bins=dc_guard_bins,
                post_detection_filter=post_detection_filter,
                return_stage_artifacts=self.config.include_stage_artifacts,
                range_mode=cfg["range_mode"],
            )
        if (
            self.config.angle_peak_detection_enabled
            and azimuth_cube is not None
            and azimuth_axis_deg is not None
        ):
            detection_azimuth_cube = np.asarray(azimuth_cube, dtype=np.float64)
            if edge_guard_bins > 0 or dc_guard_bins > 0:
                guarded_cube = np.zeros_like(detection_azimuth_cube, dtype=np.float64)
                for azimuth_idx in range(detection_azimuth_cube.shape[0]):
                    guarded_cube[azimuth_idx] = apply_pre_cfar_doppler_guards(
                        detection_azimuth_cube[azimuth_idx],
                        edge_guard_bins=edge_guard_bins,
                        dc_guard_bins=dc_guard_bins,
                    )
                detection_azimuth_cube = guarded_cube

            angle_cfg = AnglePeakDetectionConfig(
                max_peaks_per_rd_cell=self.config.angle_max_peaks_per_rd_cell,
                peak_relative_floor_db=self.config.angle_peak_relative_floor_db,
                peak_nms_radius_bins=self.config.angle_peak_nms_radius_bins,
                edge_exclusion_bins=self.config.angle_edge_exclusion_bins,
                secondary_peak_max_gap_db=self.config.angle_secondary_peak_max_gap_db,
                secondary_peak_min_separation_bins=self.config.angle_secondary_peak_min_separation_bins,
                secondary_peak_min_prominence_db=self.config.angle_secondary_peak_min_prominence_db,
                secondary_peak_min_profile_snr_db=self.config.angle_secondary_peak_min_profile_snr_db,
                secondary_peak_min_valley_depth_db=self.config.angle_secondary_peak_min_valley_depth_db,
                cluster_azimuth_gate_deg=self.config.angle_cluster_azimuth_gate_deg,
                sidelobe_suppression_enabled=self.config.angle_sidelobe_suppression_enabled,
                sidelobe_range_tolerance_bins=self.config.angle_sidelobe_range_tolerance_bins,
                sidelobe_doppler_tolerance_bins=self.config.angle_sidelobe_doppler_tolerance_bins,
                sidelobe_azimuth_tolerance_bins=self.config.angle_sidelobe_azimuth_tolerance_bins,
                sidelobe_min_power_ratio_db=self.config.angle_sidelobe_min_power_ratio_db,
            )
            return run_angle_enhanced_detection_pipeline(
                rd_result,
                azimuth_cube=detection_azimuth_cube,
                azimuth_axis_deg=azimuth_axis_deg,
                config=pipeline_cfg,
                angle_config=angle_cfg,
                wavelength_m=wavelength,
                detection_power_map=rd_power_for_detection,
                post_detection_filter=post_detection_filter,
                return_stage_artifacts=self.config.include_stage_artifacts,
                range_mode=cfg["range_mode"],
            )
        if (
            self.config.vote_fusion_enabled
            and azimuth_cube is not None
            and azimuth_axis_deg is not None
        ):
            detection_azimuth_cube = np.asarray(azimuth_cube, dtype=np.float64)
            if edge_guard_bins > 0 or dc_guard_bins > 0:
                guarded_cube = np.zeros_like(detection_azimuth_cube, dtype=np.float64)
                for azimuth_idx in range(detection_azimuth_cube.shape[0]):
                    guarded_cube[azimuth_idx] = apply_pre_cfar_doppler_guards(
                        detection_azimuth_cube[azimuth_idx],
                        edge_guard_bins=edge_guard_bins,
                        dc_guard_bins=dc_guard_bins,
                    )
                detection_azimuth_cube = guarded_cube

            base_projection_cfg = ProjectionCfarConfig(
                guard_cells=(
                    self.config.cfar_guard_cells_doppler,
                    self.config.cfar_guard_cells_range,
                ),
                reference_cells=(
                    self.config.cfar_ref_cells_doppler,
                    self.config.cfar_ref_cells_range,
                ),
                threshold_scale=self.config.cfar_threshold_scale,
                method=self.config.cfar_type.lower(),
                os_rank=self.config.cfar_os_rank,
                edge_policy=self.config.cfar_edge_policy,
            )
            raz_threshold_scale = (
                self.config.vote_raz_threshold_scale
                if self.config.vote_raz_threshold_scale is not None
                else base_projection_cfg.threshold_scale
            )
            daz_threshold_scale = (
                self.config.vote_daz_threshold_scale
                if self.config.vote_daz_threshold_scale is not None
                else base_projection_cfg.threshold_scale
            )
            vote_cfg = VoteFusionConfig(
                rd=base_projection_cfg,
                raz=ProjectionCfarConfig(
                    guard_cells=(
                        self.config.vote_azimuth_guard_cells,
                        self.config.cfar_guard_cells_range,
                    ),
                    reference_cells=(
                        self.config.vote_azimuth_ref_cells,
                        self.config.cfar_ref_cells_range,
                    ),
                    threshold_scale=raz_threshold_scale,
                    method=base_projection_cfg.method,
                    os_rank=base_projection_cfg.os_rank,
                    edge_policy=base_projection_cfg.edge_policy,
                ),
                daz=ProjectionCfarConfig(
                    guard_cells=(
                        self.config.vote_azimuth_guard_cells,
                        self.config.cfar_guard_cells_doppler,
                    ),
                    reference_cells=(
                        self.config.vote_azimuth_ref_cells,
                        self.config.cfar_ref_cells_doppler,
                    ),
                    threshold_scale=daz_threshold_scale,
                    method=base_projection_cfg.method,
                    os_rank=base_projection_cfg.os_rank,
                    edge_policy=base_projection_cfg.edge_policy,
                ),
                min_votes=self.config.vote_min_votes,
                pre_threshold_db=self.config.vote_pre_threshold_db,
                nms_suppression_radius=(
                    max(self.config.vote_azimuth_guard_cells // 4, 1),
                    self.config.nms_radius_doppler,
                    self.config.nms_radius_range,
                ),
                nms_max_peaks=self.config.vote_nms_max_peaks,
                post_nms_min_peak_divisor=self.config.vote_post_nms_min_peak_divisor,
                cluster_range_gate_m=self.config.cluster_range_gate_m,
                cluster_velocity_gate_m_s=self.config.cluster_velocity_gate_m_s,
            )
            return run_vote_fusion_pipeline(
                rd_result,
                azimuth_cube=detection_azimuth_cube,
                azimuth_axis_deg=azimuth_axis_deg,
                config=pipeline_cfg,
                vote_config=vote_cfg,
                wavelength_m=wavelength,
                post_detection_filter=post_detection_filter,
                return_stage_artifacts=self.config.include_stage_artifacts,
                range_mode=cfg["range_mode"],
            )
        pipeline_result = run_detection_pipeline(
            rd_result,
            config=pipeline_cfg,
            wavelength_m=wavelength,
            detection_power_map=rd_power_for_detection,
            post_detection_filter=post_detection_filter,
            return_stage_artifacts=self.config.include_stage_artifacts,
            aoa_az_map=aoa_az_map,
            aoa_el_map=aoa_el_map,
            range_mode=cfg["range_mode"],
        )
        return pipeline_result

    def _resolve_pre_cfar_guard_bins(self, cfg: dict[str, Any]) -> tuple[int, int]:
        processing_mode = str(cfg.get("processing_mode", "auto")).lower()
        source_processing = cfg.get("source_processing")
        if processing_mode == "offline_reprocess" and isinstance(source_processing, Mapping):
            processing_mode = str(source_processing.get("mode", processing_mode)).lower()
        explicit_override = (
            self.config.pre_cfar_doppler_edge_guard_bins is not None
            or self.config.pre_cfar_doppler_dc_guard_bins is not None
        )
        if processing_mode != "coherent_cpi" and not explicit_override:
            return 0, 0

        edge_guard_bins = self.config.pre_cfar_doppler_edge_guard_bins
        dc_guard_bins = self.config.pre_cfar_doppler_dc_guard_bins
        if edge_guard_bins is None:
            if cfg.get("_native_v4_config") is True and (
                "pre_cfar_doppler_edge_guard_bins" not in cfg
            ):
                raise ValueError(
                    "Native schema-v4 coherent reprocessing requires "
                    "config_resolved.pre_cfar_doppler_edge_guard_bins or an explicit "
                    "postprocessing override"
                )
            edge_guard_bins = int(cfg.get("pre_cfar_doppler_edge_guard_bins", 2))
        if dc_guard_bins is None:
            if cfg.get("_native_v4_config") is True and (
                "pre_cfar_doppler_dc_guard_bins" not in cfg
            ):
                raise ValueError(
                    "Native schema-v4 coherent reprocessing requires "
                    "config_resolved.pre_cfar_doppler_dc_guard_bins or an explicit "
                    "postprocessing override"
                )
            dc_guard_bins = int(cfg.get("pre_cfar_doppler_dc_guard_bins", 0))
        return max(0, int(edge_guard_bins)), max(0, int(dc_guard_bins))

    # ---- helpers ----

    @staticmethod
    def _resolve_meta(meta: dict[str, Any]) -> dict[str, Any]:
        """Extract numeric semantics, strictly for native persisted v4 payloads.

        Schema-less callers retain the direct-array convenience API and its
        historical defaults. A persisted schema-v4 payload instead has one
        authoritative ``config_resolved`` execution envelope; using outer
        aliases, ``config``, or library defaults there would fabricate source
        semantics.
        """

        if meta.get("schema_version") == SENSING_SCHEMA_VERSION:
            return _resolve_native_v4_config(meta)
        preferred_block = meta.get("config_resolved") or meta.get("config") or meta
        resolved = resolve_sensing_config(
            sensing_block=preferred_block,
            defaults={
                "fft_size_range": 256,
                "fft_size_doppler": 128,
                "sampling_frequency": 100e6,
                "prf_hz": 2000.0,
                "range_mode": "bistatic",
                "carrier_freq_hz": 28e9,
                "processing_mode": "auto",
            },
        )
        cfg = preferred_block
        source_processing = cfg.get("source_processing")
        return {
            "fft_size_range": int(resolved["fft_size_range"]),
            "fft_size_doppler": int(resolved["fft_size_doppler"]),
            "sampling_frequency": float(resolved["sampling_frequency"]),
            "prf_hz": float(resolved["prf_hz"]),
            "range_mode": str(resolved["range_mode"]),
            "carrier_freq_hz": float(resolved["carrier_freq_hz"]),
            "processing_mode": str(resolved["processing_mode"]),
            "source_processing": (
                deepcopy(dict(source_processing))
                if isinstance(source_processing, Mapping)
                else None
            ),
            "pre_cfar_doppler_edge_guard_bins": int(cfg.get("pre_cfar_doppler_edge_guard_bins", 2)),
            "pre_cfar_doppler_dc_guard_bins": int(cfg.get("pre_cfar_doppler_dc_guard_bins", 0)),
            "_native_v4_config": False,
        }


# ---------------------------------------------------------------------------
# M-of-N cross-frame confirmation filter
# ---------------------------------------------------------------------------


def apply_m_of_n_filter(
    per_frame_detections: list[list[dict[str, Any]]],
    m: int,
    n: int,
    range_tol_m: float = 0.5,
    velocity_tol_m_s: float = 0.3,
) -> list[list[dict[str, Any]]]:
    """Filter detections: keep only those confirmed in M of the last N frames.

    A detection in frame *i* is confirmed if at least *M* − 1 of the
    preceding *N* − 1 frames contain a detection within *range_tol_m* and
    *velocity_tol_m_s* of the current detection (the current frame counts
    as one of the M).

    Args:
        per_frame_detections: List of per-frame detection lists.
        m: Minimum number of frames with a matching detection (including current).
        n: Window size (number of consecutive frames to look back, including current).
        range_tol_m: Range matching tolerance in metres.
        velocity_tol_m_s: Velocity matching tolerance in m/s.

    Returns:
        Filtered per-frame detection lists (same length as input).
    """
    if m < 1 or n < 1 or m > n:
        raise ValueError(f"Invalid M-of-N: M={m}, N={n} (need 1 <= M <= N)")

    n_frames = len(per_frame_detections)
    filtered: list[list[dict[str, Any]]] = []

    for i in range(n_frames):
        if not per_frame_detections[i]:
            filtered.append([])
            continue

        # Window of preceding frames (up to N-1 back, excluding current)
        window_start = max(0, i - n + 1)
        history = per_frame_detections[window_start:i]  # excludes current frame

        confirmed: list[dict[str, Any]] = []
        for det in per_frame_detections[i]:
            r = det["range_m"]
            v = det["velocity_m_s"]
            # Count matches in history (current frame already counts as 1)
            hits = 1
            for prev_dets in history:
                for pd in prev_dets:
                    if (
                        abs(pd["range_m"] - r) <= range_tol_m
                        and abs(pd["velocity_m_s"] - v) <= velocity_tol_m_s
                    ):
                        hits += 1
                        break  # one match per frame is enough
                if hits >= m:
                    break  # early exit
            if hits >= m:
                confirmed.append(det)
        filtered.append(confirmed)

    return filtered


# ---------------------------------------------------------------------------
# Schema-v4 persisted-product selection
# ---------------------------------------------------------------------------


_NATIVE_V4_MODE_STRATEGIES = {
    "coherent_cpi": "coherent_cpi_fft",
    "sequential_stft": "sequential_stft",
    "analytic_path_doppler_oracle": "analytic_path_doppler_oracle",
    "measurement_import": "nist_mpc_csv_import",
    "offline_reprocess": "shared_sensing_postprocessor",
}

_NATIVE_V4_ACQUISITION_FIELDS = (
    "sample_source",
    "rt_solve_stride_scene_steps",
    "sample_period_s",
    "sample_rate_hz",
    "cpi_length_samples",
    "cpi_hop_samples",
    "overlap_samples",
    "partial_window_policy",
    "phase_contract",
)

_NATIVE_V4_PROCESSING_FIELDS = (
    "mode",
    "strategy",
    "fft_size_doppler",
    "result_stride_samples",
    "product_semantics",
    "evidence_status",
)

_NATIVE_V4_TOPOLOGY_FIELDS = (
    "pair_count",
    "pair_index",
    "tx_device_count",
    "rx_device_count",
    "tx_elements",
    "rx_elements",
    "rx_array_processing",
)


def _native_v4_exact_int(value: Any, *, path: str, minimum: int = 0) -> int:
    """Return an exact integer from a persisted contract field."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Native schema-v4 {path} must be an exact integer")
    normalized = int(value)
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"Native schema-v4 {path} must be {qualifier}")
    return normalized


def _native_v4_positive_float(value: Any, *, path: str) -> float:
    """Return one finite positive persisted scalar without alias fallback."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Native schema-v4 {path} must be a finite positive number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Native schema-v4 {path} must be a finite positive number") from exc
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"Native schema-v4 {path} must be a finite positive number")
    return normalized


def _native_v4_nonempty_string(value: Any, *, path: str) -> str:
    """Return one non-empty persisted string field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Native schema-v4 {path} must be a non-empty string")
    return value.strip()


def _native_v4_required_mapping(
    value: Any,
    *,
    path: str,
    fields: Sequence[str],
) -> Mapping[str, Any]:
    """Require a mapping with every named execution-contract field."""

    if not isinstance(value, Mapping):
        raise ValueError(f"Native schema-v4 {path} must be a mapping")
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(
            f"Native schema-v4 {path} is incomplete; missing fields: " + ", ".join(missing)
        )
    return value


def _validate_native_v4_acquisition_plan(
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach the complete ADR-SEN-004 acquisition plan."""

    acquisition = _native_v4_required_mapping(
        acquisition,
        path="config_resolved.acquisition",
        fields=_NATIVE_V4_ACQUISITION_FIELDS,
    )
    sample_source = _native_v4_nonempty_string(
        acquisition["sample_source"],
        path="config_resolved.acquisition.sample_source",
    )
    if sample_source not in {
        "coherent_one_solve",
        "independent_solves",
        "analytic_path_snapshot",
        "measured_pulse_stream",
    }:
        raise ValueError(
            "Native schema-v4 config_resolved.acquisition.sample_source is unsupported: "
            f"{sample_source!r}"
        )

    cpi_length = _native_v4_exact_int(
        acquisition["cpi_length_samples"],
        path="config_resolved.acquisition.cpi_length_samples",
        minimum=1,
    )
    cpi_hop = _native_v4_exact_int(
        acquisition["cpi_hop_samples"],
        path="config_resolved.acquisition.cpi_hop_samples",
        minimum=1,
    )
    if cpi_hop > cpi_length:
        raise ValueError(
            "Native schema-v4 config_resolved.acquisition.cpi_hop_samples must not "
            "exceed cpi_length_samples"
        )
    overlap = _native_v4_exact_int(
        acquisition["overlap_samples"],
        path="config_resolved.acquisition.overlap_samples",
    )
    if overlap != cpi_length - cpi_hop:
        raise ValueError(
            "Native schema-v4 config_resolved.acquisition.overlap_samples must equal "
            "cpi_length_samples - cpi_hop_samples"
        )
    if acquisition["partial_window_policy"] != "retain_unprocessed":
        raise ValueError(
            "Native schema-v4 config_resolved.acquisition.partial_window_policy must "
            "equal 'retain_unprocessed'"
        )
    _native_v4_nonempty_string(
        acquisition["phase_contract"],
        path="config_resolved.acquisition.phase_contract",
    )

    rt_stride = acquisition["rt_solve_stride_scene_steps"]
    if sample_source == "measured_pulse_stream":
        if rt_stride is not None:
            raise ValueError(
                "Native schema-v4 measured_pulse_stream acquisition must not claim an "
                "RT solve stride"
            )
    else:
        normalized_stride = _native_v4_exact_int(
            rt_stride,
            path="config_resolved.acquisition.rt_solve_stride_scene_steps",
            minimum=1,
        )
        if sample_source == "coherent_one_solve" and normalized_stride != cpi_length:
            raise ValueError(
                "Native schema-v4 coherent_one_solve RT stride must equal " "cpi_length_samples"
            )
        if sample_source != "coherent_one_solve" and normalized_stride != 1:
            raise ValueError("Native schema-v4 non-coherent RT stride must equal one scene step")

    sample_period = acquisition["sample_period_s"]
    sample_rate = acquisition["sample_rate_hz"]
    if sample_source == "analytic_path_snapshot":
        if sample_period is not None or sample_rate is not None:
            raise ValueError(
                "Native schema-v4 analytic_path_snapshot acquisition must not claim a "
                "sample period or sample rate"
            )
    else:
        normalized_period = _native_v4_positive_float(
            sample_period,
            path="config_resolved.acquisition.sample_period_s",
        )
        normalized_rate = _native_v4_positive_float(
            sample_rate,
            path="config_resolved.acquisition.sample_rate_hz",
        )
        if not np.isclose(
            normalized_period * normalized_rate,
            1.0,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                "Native schema-v4 acquisition sample_period_s and sample_rate_hz must "
                "be reciprocal"
            )
    return deepcopy(dict(acquisition))


def _resolve_native_v4_config(sensing: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one persisted v4 execution envelope without any fallback.

    This is intentionally separate from :func:`resolve_sensing_config`, whose
    precedence/default behavior remains useful to schema-less direct-array
    APIs. Persisted v4 reprocessing is an evidence boundary: missing values
    mean the source must be regenerated or explicitly converted.
    """

    if sensing.get("schema_version") != SENSING_SCHEMA_VERSION:
        raise ValueError(
            f"Persisted sensing reprocessing requires schema version {SENSING_SCHEMA_VERSION}"
        )
    config = sensing.get("config_resolved")
    if not isinstance(config, Mapping):
        raise ValueError(
            "Native schema-v4 persisted sensing requires a complete config_resolved; "
            "config and outer sensing aliases are not fallback sources"
        )
    contract_version = _native_v4_exact_int(
        config.get("contract_version"),
        path="config_resolved.contract_version",
        minimum=1,
    )
    if contract_version not in {1, 2}:
        raise ValueError("Native schema-v4 config_resolved.contract_version must equal 1 or 2")
    from shared.sensing.range_response import validate_execution_range_response

    validate_execution_range_response(config)

    fft_r = _native_v4_exact_int(
        config.get("fft_size_range"),
        path="config_resolved.fft_size_range",
        minimum=1,
    )
    fft_d = _native_v4_exact_int(
        config.get("fft_size_doppler"),
        path="config_resolved.fft_size_doppler",
        minimum=1,
    )
    sampling_frequency = _native_v4_positive_float(
        config.get("sampling_frequency"),
        path="config_resolved.sampling_frequency",
    )
    prf_hz = _native_v4_positive_float(
        config.get("prf_hz"),
        path="config_resolved.prf_hz",
    )
    carrier_freq_hz = _native_v4_positive_float(
        config.get("carrier_freq_hz"),
        path="config_resolved.carrier_freq_hz",
    )
    range_mode = _native_v4_nonempty_string(
        config.get("range_mode"),
        path="config_resolved.range_mode",
    ).lower()
    if range_mode not in {"monostatic", "bistatic"}:
        raise ValueError(
            "Native schema-v4 config_resolved.range_mode must be exactly "
            "'monostatic' or 'bistatic'"
        )

    acquisition = _validate_native_v4_acquisition_plan(config.get("acquisition"))
    processing = _native_v4_required_mapping(
        config.get("processing"),
        path="config_resolved.processing",
        fields=_NATIVE_V4_PROCESSING_FIELDS,
    )
    mode = _native_v4_nonempty_string(
        processing["mode"],
        path="config_resolved.processing.mode",
    )
    strategy = _native_v4_nonempty_string(
        processing["strategy"],
        path="config_resolved.processing.strategy",
    )
    expected_strategy = _NATIVE_V4_MODE_STRATEGIES.get(mode)
    if expected_strategy is None or strategy != expected_strategy:
        raise ValueError(
            "Native schema-v4 config_resolved.processing mode/strategy pair is unsupported: "
            f"{mode!r}/{strategy!r}"
        )
    if config.get("processing_mode") != mode:
        raise ValueError(
            "Native schema-v4 config_resolved.processing_mode must exactly mirror "
            "config_resolved.processing.mode"
        )
    processing_fft_d = _native_v4_exact_int(
        processing["fft_size_doppler"],
        path="config_resolved.processing.fft_size_doppler",
        minimum=1,
    )
    if processing_fft_d != fft_d:
        raise ValueError(
            "Native schema-v4 config_resolved.processing.fft_size_doppler must match "
            "config_resolved.fft_size_doppler"
        )
    result_stride = _native_v4_exact_int(
        processing["result_stride_samples"],
        path="config_resolved.processing.result_stride_samples",
        minimum=1,
    )
    if result_stride != int(acquisition["cpi_hop_samples"]):
        raise ValueError(
            "Native schema-v4 processing.result_stride_samples must equal "
            "acquisition.cpi_hop_samples"
        )
    if mode != "analytic_path_doppler_oracle" and processing_fft_d != int(
        acquisition["cpi_length_samples"]
    ):
        raise ValueError(
            "Native schema-v4 processing.fft_size_doppler must equal "
            "acquisition.cpi_length_samples"
        )
    _native_v4_nonempty_string(
        processing["product_semantics"],
        path="config_resolved.processing.product_semantics",
    )
    _native_v4_nonempty_string(
        processing["evidence_status"],
        path="config_resolved.processing.evidence_status",
    )
    expected_source = {
        "coherent_cpi": "coherent_one_solve",
        "sequential_stft": "independent_solves",
        "analytic_path_doppler_oracle": "analytic_path_snapshot",
        "measurement_import": "measured_pulse_stream",
    }.get(mode)
    if expected_source is not None and acquisition["sample_source"] != expected_source:
        raise ValueError(
            "Native schema-v4 acquisition.sample_source does not match the resolved "
            f"processing mode {mode!r}"
        )
    if mode == "coherent_cpi" and int(acquisition["cpi_hop_samples"]) != int(
        acquisition["cpi_length_samples"]
    ):
        raise ValueError(
            "Native schema-v4 coherent_cpi acquisition must use a non-overlapping "
            "cpi_hop_samples equal to cpi_length_samples"
        )
    if mode == "analytic_path_doppler_oracle" and int(acquisition["cpi_hop_samples"]) != 1:
        raise ValueError(f"Native schema-v4 {mode} acquisition must use cpi_hop_samples=1")

    topology = _native_v4_required_mapping(
        config.get("topology"),
        path="config_resolved.topology",
        fields=_NATIVE_V4_TOPOLOGY_FIELDS,
    )
    pair_count = _native_v4_exact_int(
        topology["pair_count"],
        path="config_resolved.topology.pair_count",
        minimum=1,
    )
    pair_index = _native_v4_exact_int(
        topology["pair_index"],
        path="config_resolved.topology.pair_index",
    )
    if pair_count != 1 or pair_index != 0:
        raise ValueError("Native schema-v4 reprocessing currently requires topology P=1")
    for field in ("tx_device_count", "rx_device_count", "tx_elements", "rx_elements"):
        _native_v4_exact_int(
            topology[field],
            path=f"config_resolved.topology.{field}",
            minimum=1,
        )
    if topology["tx_device_count"] != 1 or topology["rx_device_count"] != 1:
        raise ValueError(
            "Native schema-v4 reprocessing currently requires one TX and one RX device"
        )
    _native_v4_nonempty_string(
        topology["rx_array_processing"],
        path="config_resolved.topology.rx_array_processing",
    )
    processing_recipe = config.get("processing_recipe")
    if not isinstance(processing_recipe, Mapping):
        raise ValueError("Native schema-v4 config_resolved.processing_recipe must be a mapping")

    resolved: dict[str, Any] = {
        "fft_size_range": fft_r,
        "fft_size_doppler": fft_d,
        "sampling_frequency": sampling_frequency,
        "prf_hz": prf_hz,
        "range_mode": range_mode,
        "carrier_freq_hz": carrier_freq_hz,
        "processing_mode": mode,
        "source_processing": (
            deepcopy(dict(config["source_processing"]))
            if isinstance(config.get("source_processing"), Mapping)
            else None
        ),
        "acquisition": acquisition,
        "processing": deepcopy(dict(processing)),
        "topology": deepcopy(dict(topology)),
        "processing_recipe": deepcopy(dict(processing_recipe)),
        "_native_v4_config": True,
    }
    # Derived recipes keep effective windows and clutter policy at the top
    # level. Preserve them when projecting the persisted execution envelope.
    for key in ("window_range", "window_doppler", "coherent_clutter_mode"):
        if key in config:
            resolved[key] = deepcopy(config[key])
    if contract_version == 2:
        for key in (
            "range_response",
            "range_response_processing",
            "bandwidth",
            "range_resolution_scale_m",
            "range_sample_spacing_m",
        ):
            resolved[key] = deepcopy(config[key])
    for guard_key in (
        "pre_cfar_doppler_edge_guard_bins",
        "pre_cfar_doppler_dc_guard_bins",
    ):
        if guard_key in config:
            resolved[guard_key] = _native_v4_exact_int(
                config[guard_key],
                path=f"config_resolved.{guard_key}",
            )
    return resolved


def _native_v4_processing_windows(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return exact stored window functions without assuming Hann defaults."""

    recipe = config.get("processing_recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("Native schema-v4 config_resolved.processing_recipe must be a mapping")
    windows: list[str] = []
    for field in ("window_range", "window_doppler"):
        top_value = config.get(field)
        recipe_value = recipe.get(field)
        if top_value is not None and recipe_value is not None and top_value != recipe_value:
            raise ValueError(
                f"Native schema-v4 config_resolved.{field} disagrees with "
                f"config_resolved.processing_recipe.{field}"
            )
        value = top_value if top_value is not None else recipe_value
        windows.append(
            _native_v4_nonempty_string(
                value,
                path=f"config_resolved.{field} or processing_recipe.{field}",
            )
        )
    return windows[0], windows[1]


def _effective_processing_windows(
    config: Mapping[str, Any],
    postprocessing_config: PostProcessingConfig,
) -> tuple[str, str]:
    """Resolve stored v4 windows or schema-less compatibility defaults."""

    if config.get("_native_v4_config") is True:
        inherited_range, inherited_doppler = _native_v4_processing_windows(config)
    else:
        inherited_range, inherited_doppler = "hann", "hann"
    return (
        (
            postprocessing_config.window_range
            if postprocessing_config.window_range is not None
            else inherited_range
        ),
        (
            postprocessing_config.window_doppler
            if postprocessing_config.window_doppler is not None
            else inherited_doppler
        ),
    )


def _native_v4_frame_windows(frame: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve the exact source windows for one loaded persisted frame."""

    sensing = _typed_sensing_payload(frame)
    if sensing is None:
        raise ValueError("Persisted sensing frame has no schema-v4 payload")
    _resolve_native_v4_config(sensing)
    config = sensing["config_resolved"]
    assert isinstance(config, Mapping)
    return _native_v4_processing_windows(config)


def _typed_sensing_payload(frame: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the complete loaded schema-v4 sensing payload, when present."""

    typed = frame.get("_typed_sensing")
    if isinstance(typed, Mapping):
        if typed.get("schema_version") == SENSING_SCHEMA_VERSION:
            _resolve_native_v4_config(typed)
        return typed
    if isinstance(frame.get("products"), Mapping):
        if frame.get("schema_version") == SENSING_SCHEMA_VERSION:
            _resolve_native_v4_config(frame)
        return frame
    return None


def _optional_product_by_contract(
    sensing: Mapping[str, Any],
    *,
    kind: str,
    axis_names: Sequence[str],
    representations: Sequence[str],
) -> (
    tuple[
        str,
        np.ndarray,
        tuple[np.ndarray | list[Any], ...],
        Mapping[str, Any],
    ]
    | None
):
    """Select one contract product while distinguishing absence from corruption."""

    try:
        return select_product_by_contract(
            sensing,
            kind=kind,
            variant="primary",
            axis_names=axis_names,
            representations=representations,
        )
    except KeyError as exc:
        message = str(exc)
        if (
            "No loaded sensing product matches" in message
            or "has no typed product descriptor mapping" in message
        ):
            return None
        raise ValueError(f"Invalid typed sensing product: {exc}") from exc


def _descriptor_axis_units(
    descriptor: Mapping[str, Any],
    *,
    data_key: str,
) -> tuple[str | None, ...]:
    axes = descriptor.get("axes")
    if not isinstance(axes, list):
        raise ValueError(f"Typed sensing product {data_key!r} has no ordered axes")
    return tuple(axis.get("unit") if isinstance(axis, Mapping) else None for axis in axes)


def _dense_coordinate(
    value: np.ndarray | list[Any],
    *,
    data_key: str,
    axis_name: str,
    size: int,
) -> np.ndarray:
    """Validate one persisted dense coordinate without reconstructing it."""

    coordinates = np.asarray(value, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.shape != (size,):
        raise ValueError(
            f"Typed sensing product {data_key!r} axis {axis_name!r} requires "
            f"one-dimensional coordinates of length {size}; got {coordinates.shape}"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"Typed sensing product {data_key!r} axis {axis_name!r} must be finite")
    return coordinates


def _select_typed_slow_time_input(
    frame: Mapping[str, Any],
    *,
    reconstruct_frequency: bool = False,
) -> _TypedDetectionInput | None:
    """Select the element-resolved slow-time product and its named combiner."""

    sensing = _typed_sensing_payload(frame)
    if sensing is None:
        return None
    if (
        reconstruct_frequency
        and not any(
            d.get("kind") == "delay_domain_slow_time" for d in sensing.get("products", {}).values()
        )
        and any(
            d.get("kind") == "baseband_frequency_observation"
            for d in sensing.get("products", {}).values()
        )
    ):
        from shared.sensing.cfr_replay import reconstruct_delay_observation

        sensing = reconstruct_delay_observation(sensing)
    selected = _optional_product_by_contract(
        sensing,
        kind="delay_domain_slow_time",
        axis_names=("pair", "rx_element", "slow_time", "range"),
        representations=("complex_voltage",),
    )
    if selected is None:
        return None
    data_key, values, coordinates, descriptor = selected
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError(
            f"Typed slow-time product {data_key!r} requires shape [P=1,E,N,R]; "
            f"received {values.shape}"
        )
    if not np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Typed slow-time product {data_key!r} declares complex_voltage and "
            "must contain finite complex samples"
        )
    if _descriptor_axis_units(descriptor, data_key=data_key) != ("1", "1", "s", "m"):
        raise ValueError(f"Typed slow-time product {data_key!r} must use axis units [1,1,s,m]")
    element_catalog = coordinates[1]
    if (
        len(coordinates[0]) != 1
        or not isinstance(element_catalog, list)
        or len(element_catalog) != values.shape[1]
        or any(not isinstance(entry, Mapping) for entry in element_catalog)
    ):
        raise ValueError(
            f"Typed slow-time product {data_key!r} has inconsistent pair/element catalogs"
        )

    combine = descriptor.get("combine")
    if not isinstance(combine, Mapping):
        raise ValueError(
            f"Typed slow-time product {data_key!r} requires an explicit combine policy"
        )
    policy = combine.get("policy")
    normalization = combine.get("normalization")
    element_count = int(values.shape[1])
    if policy == "siso_identity":
        if element_count != 1 or normalization != "identity":
            raise ValueError(
                "siso_identity requires exactly one RX element and identity normalization"
            )
    elif policy == "rx_broadside_unitary_voltage_sum":
        if (
            combine.get("input_axis") != "rx_element"
            or normalization != "one_over_sqrt_element_count"
        ):
            raise ValueError(
                "rx_broadside_unitary_voltage_sum requires input_axis='rx_element' "
                "and one_over_sqrt_element_count normalization"
            )
    else:
        raise ValueError(
            f"Typed slow-time product {data_key!r} uses unsupported combine policy {policy!r}"
        )

    product_id = descriptor.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError(f"Typed sensing product {data_key!r} has no product_id")
    return _TypedDetectionInput(
        data_key=data_key,
        product_id=product_id,
        kind="delay_domain_slow_time",
        representation="complex_voltage",
        values=values,
        range_axis_m=_dense_coordinate(
            coordinates[3],
            data_key=data_key,
            axis_name="range",
            size=values.shape[3],
        ),
        descriptor=descriptor,
        rx_element_axis=tuple(deepcopy(dict(entry)) for entry in element_catalog),
        slow_time_axis_s=_dense_coordinate(
            coordinates[2],
            data_key=data_key,
            axis_name="slow_time",
            size=values.shape[2],
        ),
        combine_policy=str(policy),
    )


def _select_typed_range_doppler_input(
    frame: Mapping[str, Any],
) -> _TypedDetectionInput | None:
    """Select one exact range-Doppler product without dtype interpretation."""

    sensing = _typed_sensing_payload(frame)
    if sensing is None:
        return None
    selected = _optional_product_by_contract(
        sensing,
        kind="range_doppler",
        axis_names=("pair", "doppler_frequency", "range"),
        representations=("complex_voltage", "power_linear"),
    )
    if selected is None:
        return None
    data_key, values, coordinates, descriptor = selected
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError(
            f"Typed range-Doppler product {data_key!r} requires shape [P=1,D,R]; "
            f"received {values.shape}"
        )
    if _descriptor_axis_units(descriptor, data_key=data_key) != ("1", "Hz", "m"):
        raise ValueError(f"Typed range-Doppler product {data_key!r} must use axis units [1,Hz,m]")
    if len(coordinates[0]) != 1:
        raise ValueError(f"Typed range-Doppler product {data_key!r} has no one-pair catalog")

    representation = descriptor.get("representation")
    if representation == "complex_voltage":
        if not np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Typed range-Doppler product {data_key!r} declares complex_voltage "
                "and must contain finite complex samples"
            )
    elif representation == "power_linear":
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(
                f"Typed range-Doppler product {data_key!r} declares power_linear "
                "and must contain finite nonnegative real values"
            )
    else:  # The contract selector above makes this defensive branch unreachable.
        raise ValueError(
            f"Typed range-Doppler product {data_key!r} has unsupported representation "
            f"{representation!r}"
        )

    product_id = descriptor.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError(f"Typed sensing product {data_key!r} has no product_id")
    return _TypedDetectionInput(
        data_key=data_key,
        product_id=product_id,
        kind="range_doppler",
        representation=str(representation),
        values=values,
        range_axis_m=_dense_coordinate(
            coordinates[2],
            data_key=data_key,
            axis_name="range",
            size=values.shape[2],
        ),
        descriptor=descriptor,
        doppler_axis_hz=_dense_coordinate(
            coordinates[1],
            data_key=data_key,
            axis_name="doppler_frequency",
            size=values.shape[1],
        ),
    )


def _select_typed_detection_input(
    frame: Mapping[str, Any],
    *,
    rebuild_rd: bool,
    rebuild_cfr: bool = False,
) -> _TypedDetectionInput | None:
    """Choose the declared primary product for one persisted processing route."""

    sensing = _typed_sensing_payload(frame)
    kinds = {d.get("kind") for d in sensing.get("products", {}).values()} if sensing else set()
    frequency_retained = (
        "baseband_frequency_observation" in kinds and "delay_domain_slow_time" not in kinds
    )
    if rebuild_cfr or (rebuild_rd and frequency_retained):
        sensing = _typed_sensing_payload(frame)
        if sensing is None:
            raise ValueError("CFR replay requires a typed retained frequency observation")
        key, values, axes, descriptor = select_frequency_observation(sensing)
        return _TypedDetectionInput(
            data_key=key,
            product_id=descriptor["product_id"],
            kind="baseband_frequency_observation",
            representation="complex_voltage",
            values=values,
            range_axis_m=np.asarray(sensing["range_axis_m"]),
            descriptor=descriptor,
            slow_time_axis_s=np.asarray(axes[2]),
        )
    if rebuild_rd:
        slow_time = _select_typed_slow_time_input(frame)
        if slow_time is None:
            range_doppler = _select_typed_range_doppler_input(frame)
            if range_doppler is not None:
                raise ValueError(
                    "RD rebuilding requires a typed delay_domain_slow_time product; "
                    "a stored range-Doppler product cannot satisfy --rebuild-rd"
                )
            return None
        return slow_time
    range_doppler = _select_typed_range_doppler_input(frame)
    if range_doppler is not None:
        return range_doppler
    slow_time = _select_typed_slow_time_input(frame)
    if slow_time is not None:
        return slow_time

    sensing = _typed_sensing_payload(frame)
    legacy_keys = tuple(
        key
        for key in ("slow_time_matrix", "range_doppler_map")
        if key in frame or (isinstance(sensing, Mapping) and key in sensing)
    )
    if legacy_keys:
        raise ValueError(
            "Persisted sensing detection inputs require schema-v4 typed products; "
            f"found only compatibility arrays {legacy_keys!r}"
        )
    return None


def _combine_typed_slow_time(product: _TypedDetectionInput) -> np.ndarray:
    """Apply the producer-declared RX-element voltage combiner."""

    values = product.values[0]
    if product.combine_policy == "siso_identity":
        return np.asarray(values[0])
    if product.combine_policy == "rx_broadside_unitary_voltage_sum":
        return np.sum(values, axis=0) / np.sqrt(values.shape[0])
    raise ValueError(f"Unsupported typed slow-time combine policy {product.combine_policy!r}")


def _typed_range_doppler_arrays(
    product: _TypedDetectionInput,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return power and optional voltage from an explicit representation."""

    values = np.asarray(product.values[0])
    if product.representation == "power_linear":
        return values, None
    if product.representation == "complex_voltage":
        return np.abs(values) ** 2, values
    raise ValueError(f"Unsupported typed range-Doppler representation {product.representation!r}")


# ---------------------------------------------------------------------------
# Detection threshold sweep
# ---------------------------------------------------------------------------


def sweep_thresholds(
    frames: list[dict[str, Any]],
    *,
    threshold_range: tuple[float, float] = (1.0, 10.0),
    threshold_steps: int = 20,
    base_config: PostProcessingConfig | None = None,
    range_gate: float = DEFAULT_RANGE_GATE_M,
    velocity_gate: float = DEFAULT_VELOCITY_GATE_M_S,
    rebuild_rd: bool = False,
    rebuild_cfr: bool = False,
) -> list[ThresholdSweepPoint]:
    """Sweep CFAR threshold and compute detection metrics at each point.

    Pre-computes RD maps once, then re-runs only CFAR -> NMS -> DBSCAN ->
    evaluate for each threshold value.

    Args:
        frames: Loaded sensing frame dicts (from ``_load_frames_sensing``).
        threshold_range: ``(min, max)`` threshold scale values.
        threshold_steps: Number of threshold values to test.
        base_config: Base postprocessor config (threshold will be overridden).
        range_gate: Association range gate in metres.
        velocity_gate: Association velocity gate in m/s.
        rebuild_rd: Require the typed slow-time product instead of preferring
            a stored typed range-Doppler product.
        rebuild_cfr: Rebuild from the typed frequency observation before the
            threshold sweep, preserving the requested frequency/IFFT recipe.

    Returns:
        List of :class:`ThresholdSweepPoint` sorted by ascending threshold.
    """
    base_config = base_config or PostProcessingConfig()
    if base_config.array_detection_mode == "beam_scan":
        raise ValueError(
            "For beam_scan sweeps, run each detector recipe on retained angular products"
        )
    pp = SensingPostProcessor(base_config)
    frames_by_key = _index_evaluation_frames(frames)
    if not frames_by_key:
        return []

    # Pre-compute RD maps and build RangeDopplerResult objects once.
    # The sweep loop then only varies the CFAR threshold, skipping
    # redundant config resolution, axis computation, and dict construction.
    precomputed: list[tuple[SensingEvaluationEventKey, RangeDopplerResult, dict[str, Any]]] = []
    for frame_index, frame in enumerate(frames):
        event_key = _frame_evaluation_event_key(frame, frame_index=frame_index)
        if event_key not in frames_by_key:
            continue
        typed_sensing = _typed_sensing_payload(frame)
        meta = dict(typed_sensing) if isinstance(typed_sensing, Mapping) else dict(frame)
        update_time_s = None
        if base_config.tracker_type.lower() != "none" or (
            base_config.m_of_n_m is not None and base_config.m_of_n_n is not None
        ):
            update_time_s = event_key.reference_time_s
        typed_input = _select_typed_detection_input(
            frame, rebuild_rd=rebuild_rd, rebuild_cfr=rebuild_cfr
        )
        if typed_input is None:
            continue
        if typed_input.kind == "baseband_frequency_observation":
            result = pp.process_from_frequency_observation(meta, update_time_s=update_time_s)
        elif typed_input.kind == "delay_domain_slow_time":
            assert typed_input.slow_time_axis_s is not None
            result = pp.process_from_slow_time_matrix(
                _combine_typed_slow_time(typed_input),
                meta,
                range_axis_m=typed_input.range_axis_m,
                slow_time_axis_s=typed_input.slow_time_axis_s,
                update_time_s=update_time_s,
            )
        else:
            assert typed_input.doppler_axis_hz is not None
            rd_power, rd_cube = _typed_range_doppler_arrays(typed_input)
            result = pp.process_from_rd_map(
                rd_power,
                meta,
                range_axis_m=typed_input.range_axis_m,
                doppler_axis_hz=typed_input.doppler_axis_hz,
                rd_cube=rd_cube,
                update_time_s=update_time_s,
            )
        cfg = pp._resolve_meta(meta)
        range_mode = cfg["range_mode"]
        rd_map = result["rd_map"]
        range_axis = np.asarray(result["range_axis"], dtype=np.float64)
        doppler_axis = np.asarray(result["doppler_axis"], dtype=np.float64)
        range_label = (
            "Monostatic range (m)"
            if str(range_mode).lower().startswith("mono")
            else "Bistatic path length (m)"
        )
        rd_result_obj = RangeDopplerResult(
            power_linear=rd_map,
            range_axis_m=range_axis,
            doppler_axis_hz=doppler_axis,
            cube=result.get("rd_cube"),
            range_axis_label=range_label,
            doppler_axis_label="Doppler (Hz)",
        )
        precomputed.append((event_key, rd_result_obj, cfg))

    precomputed.sort(key=lambda item: _evaluation_event_sort_key(item[0]))
    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_steps)
    sweep_results: list[ThresholdSweepPoint] = []

    for threshold in thresholds:
        sweep_cfg = copy(base_config)
        sweep_cfg.cfar_threshold_scale = float(threshold)
        sweep_pp = SensingPostProcessor(sweep_cfg)

        detections_by_key: dict[SensingEvaluationEventKey, tuple[np.ndarray, np.ndarray]] = {}
        total_dets = 0
        for event_key, rd_result_obj, cfg in precomputed:
            detection_result = sweep_pp._detect(
                rd_result_obj, cfg, update_time_s=event_key.reference_time_s
            )
            dets = detection_result.detections
            total_dets += len(dets)
            det_r = np.array([d["range_m"] for d in dets]) if dets else np.array([])
            det_v = np.array([d["velocity_m_s"] for d in dets]) if dets else np.array([])
            detections_by_key[event_key] = (det_r, det_v)

        per_event_gt: list[tuple[np.ndarray, np.ndarray]] = []
        per_event_dets: list[tuple[np.ndarray, np.ndarray]] = []
        velocity_masks = []
        for event_key in sorted(frames_by_key, key=_evaluation_event_sort_key):
            _, gt_ranges, gt_velocities = _ground_truth_for_evaluation(frames_by_key[event_key])
            per_event_gt.append((gt_ranges, gt_velocities))
            velocity_masks.append(
                evaluation_velocity_mask(frames_by_key[event_key].get("_gt") or {})
            )
            per_event_dets.append(
                detections_by_key.get(
                    event_key,
                    (np.array([], dtype=np.float64), np.array([], dtype=np.float64)),
                )
            )

        eval_result = evaluate_scenario(
            per_event_gt,
            per_event_dets,
            range_gate=range_gate,
            velocity_gate=velocity_gate,
            per_event_velocity_valid=(
                [
                    mask if mask is not None else [True] * len(gt[0])
                    for mask, gt in zip(velocity_masks, per_event_gt, strict=True)
                ]
                if any(mask is not None for mask in velocity_masks)
                else None
            ),
        )

        sweep_results.append(
            ThresholdSweepPoint(
                threshold_scale=float(threshold),
                pd=eval_result.cumulative_pd,
                mean_false_alarms_per_evaluated_event=(
                    eval_result.mean_false_alarms_per_evaluated_event
                ),
                miss_rate=(
                    None if eval_result.cumulative_pd is None else 1.0 - eval_result.cumulative_pd
                ),
                evaluated_event_count=eval_result.evaluated_event_count,
                target_opportunity_count=eval_result.target_opportunity_count,
                matched_pair_count=eval_result.matched_pair_count,
                total_detections=total_dets,
                total_tp=eval_result.total_tp,
                total_fp=eval_result.total_fp,
                total_fn=eval_result.total_fn,
                range_rmse=eval_result.cumulative_range_rmse,
                velocity_rmse=eval_result.cumulative_velocity_rmse,
                velocity_matched_pair_count=eval_result.velocity_matched_pair_count,
                velocity_excluded_matched_pair_count=eval_result.velocity_excluded_matched_pair_count,
                velocity_unavailable_opportunity_count=eval_result.velocity_unavailable_opportunity_count,
                evaluation_reference_protocol=eval_result.evaluation_reference_protocol,
            )
        )

    return sweep_results


# ---------------------------------------------------------------------------
# HDF5 frame loading
# ---------------------------------------------------------------------------


_load_frames_sensing = load_frames_sensing
_iter_frames_sensing = iter_frames_sensing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_load_saved_results = load_saved_results
_validate_postprocessed_result = validate_postprocessed_result
_artifacts_to_dict = artifacts_to_dict
_build_reprocessed_sensing_block = build_reprocessed_sensing_block
_ColumnarResultWriter = ColumnarResultWriter
_save_stage_artifacts = save_stage_artifacts


def _resolve_source_frames_dir(source: Path) -> Path:
    """Resolve a scenario/profile or direct packed-v2 frame-set path once."""

    requested = Path(source).expanduser()
    if requested.suffix.lower() in {".yaml", ".yml"} or (
        requested.is_dir() and (requested / "scenario.yaml").is_file()
    ):
        candidate = resolve_scenario_frames_dir(requested)
    elif (requested / FRAMES_MANIFEST_FILENAME).is_file():
        candidate = requested
    else:
        candidate = requested / "frames"

    resolved = candidate.resolve()
    if not (resolved / FRAMES_MANIFEST_FILENAME).is_file():
        raise FileNotFoundError(f"No packed HDF5 v2 {FRAMES_MANIFEST_FILENAME} found at {resolved}")
    return resolved


def _analysis_output_root(
    source_frames: Path,
    requested_output: Path | None,
    *,
    existing: bool,
) -> Path:
    """Validate the explicit analysis root without creating or changing it."""

    if requested_output is None:
        mode = "--gif-only" if existing else "frame-producing postprocessing"
        raise ValueError(f"{mode} requires an explicit --output-dir")

    source = Path(source_frames).resolve()
    output_entry = Path(requested_output).expanduser().absolute()
    try:
        output_entry.lstat()
    except FileNotFoundError:
        output_entry_exists = False
    except OSError as exc:
        raise ValueError(f"could not inspect --output-dir {output_entry}: {exc}") from exc
    else:
        output_entry_exists = True

    if existing and output_entry_exists:
        is_junction = getattr(output_entry, "is_junction", None)
        if output_entry.is_symlink() or (callable(is_junction) and is_junction()):
            raise ValueError("--gif-only analysis root must be a real directory")

    output = output_entry.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError(
            "--output-dir must not equal, contain, or be inside the source frame directory"
        )

    if existing:
        if not output.is_dir():
            raise FileNotFoundError(
                f"--gif-only requires an existing completed analysis root: {output}"
            )
        if not (output / "frames" / FRAMES_MANIFEST_FILENAME).is_file():
            raise FileNotFoundError(
                f"--gif-only requires completed derived frames under {output / 'frames'}"
            )
    elif output_entry_exists:
        raise FileExistsError(f"--output-dir must name a new absent analysis root: {output_entry}")
    return output


def _claim_analysis_root(output_root: Path) -> None:
    """Create the validated analysis root with exclusive filesystem semantics."""

    Path(output_root).mkdir(parents=True, exist_ok=False)


def _has_detection_input(frame: dict[str, Any]) -> bool:
    """Return whether a frame has typed input or retained incomplete lineage."""

    if _unprocessed_acquisition_temporal(frame) is not None:
        return True
    return _select_typed_detection_input(frame, rebuild_rd=False) is not None


def _frame_temporal(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached schema-v4 temporal record from one loaded frame."""

    sensing_json = frame.get("_sensing_json")
    if not isinstance(sensing_json, Mapping):
        sensing_json = frame.get("_typed_sensing")
    temporal = sensing_json.get("temporal") if isinstance(sensing_json, Mapping) else None
    if not isinstance(temporal, Mapping):
        raise ValueError(
            f"Sensing frame {frame.get('_frame_index')!r} has no schema-v4 temporal record"
        )
    return deepcopy(dict(temporal))


def _evaluation_event_key_from_temporal(
    temporal: Mapping[str, Any],
    *,
    context: str,
    grouped_reference_time_s: Any = None,
) -> SensingEvaluationEventKey:
    """Extract and validate one exact truth/result join key."""

    clock = temporal.get("clock")
    acquisition = temporal.get("acquisition")
    if not isinstance(clock, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError(f"{context} lacks temporal clock/acquisition identity")
    clock_id = clock.get("clock_id")
    acquisition_id = acquisition.get("acquisition_id")
    pair_id = acquisition.get("pair_id")
    for field_name, value in (
        ("clock_id", clock_id),
        ("acquisition_id", acquisition_id),
        ("pair_id", pair_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} has no non-empty {field_name}")

    result = temporal.get("result")
    if grouped_reference_time_s is not None:
        reference_time_raw = grouped_reference_time_s
    elif isinstance(result, Mapping):
        reference_time_raw = result.get("reference_time_s")
    else:
        reference_time_raw = acquisition.get("default_reference_time_s")
    if isinstance(reference_time_raw, (bool, np.bool_)):
        raise ValueError(f"{context} reference_time_s must be finite")
    try:
        reference_time_s = float(reference_time_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} reference_time_s must be finite") from exc
    if not np.isfinite(reference_time_s):
        raise ValueError(f"{context} reference_time_s must be finite")
    return SensingEvaluationEventKey(
        clock_id=clock_id,
        acquisition_id=acquisition_id,
        pair_id=pair_id,
        reference_time_s=reference_time_s,
    )


def _frame_evaluation_event_key(
    frame: Mapping[str, Any],
    *,
    frame_index: int,
) -> SensingEvaluationEventKey:
    """Return the exact event key represented by one source frame."""

    grouped = frame.get("_grouped_temporal")
    if isinstance(grouped, Mapping):
        return _evaluation_event_key_from_temporal(
            grouped,
            context=f"Evaluation frame {frame_index}",
            grouped_reference_time_s=grouped.get("reference_time_s"),
        )
    return _evaluation_event_key_from_temporal(
        _frame_temporal(frame),
        context=f"Evaluation frame {frame_index}",
    )


def _result_evaluation_identity(
    result: Mapping[str, Any],
    *,
    result_index: int,
) -> tuple[SensingEvaluationEventKey, str]:
    """Return the temporal event and processing occurrence for one result."""

    temporal = result.get("temporal")
    if not isinstance(temporal, Mapping):
        raise ValueError(f"Evaluation result {result_index} has no temporal identity")
    event_key = _evaluation_event_key_from_temporal(
        temporal,
        context=f"Evaluation result {result_index}",
    )
    temporal_result = temporal.get("result")
    processing_run_id = (
        temporal_result.get("processing_run_id") if isinstance(temporal_result, Mapping) else None
    )
    if not isinstance(processing_run_id, str) or not processing_run_id:
        raise ValueError(f"Evaluation result {result_index} has no processing_run_id")
    return event_key, processing_run_id


def _evaluation_event_sort_key(
    key: SensingEvaluationEventKey,
) -> tuple[str, str, float, str]:
    """Sort each clock/pair stream by physical result epoch."""

    return key.clock_id, key.pair_id, key.reference_time_s, key.acquisition_id


def _truth_velocity_is_evaluable(
    frame: Mapping[str, Any],
    *,
    context: str,
) -> bool:
    """Return whether every active target has a valid velocity observable."""

    gt = frame.get("_gt")
    if gt is None:
        return True
    if not isinstance(gt, Mapping):
        raise ValueError(f"{context} ground truth must be a mapping")
    if gt.get("_evaluation_unavailable_reason"):
        return False
    target_ranges = gt.get("target_ranges", ())
    if not isinstance(target_ranges, Sequence) or isinstance(target_ranges, (str, bytes)):
        raise ValueError(f"{context} target_ranges must be an ordered sequence")
    if len(target_ranges) == 0:
        return True
    velocity_valid = gt.get("target_velocity_valid")
    if not isinstance(velocity_valid, Sequence) or isinstance(velocity_valid, (str, bytes)):
        raise ValueError(f"{context} requires target_velocity_valid for every active target")
    if len(velocity_valid) != len(target_ranges) or any(
        not isinstance(valid, (bool, np.bool_)) for valid in velocity_valid
    ):
        raise ValueError(f"{context} target_velocity_valid must align as booleans")
    # Explicit new-policy support retains range-only opportunities. Legacy
    # unavailable truth keeps its established whole-event exclusion behavior.
    return evaluation_velocity_mask(gt) is not None or all(bool(valid) for valid in velocity_valid)


def _index_evaluation_frames(
    frames: Sequence[Mapping[str, Any]],
) -> dict[SensingEvaluationEventKey, Mapping[str, Any]]:
    """Index complete source events by exact identity and reject duplicates."""

    indexed: dict[SensingEvaluationEventKey, Mapping[str, Any]] = {}
    seen_keys: set[SensingEvaluationEventKey] = set()
    for frame_index, frame in enumerate(frames):
        grouped = frame.get("_grouped_temporal")
        temporal = grouped if isinstance(grouped, Mapping) else _frame_temporal(frame)
        acquisition = temporal.get("acquisition")
        if not isinstance(acquisition, Mapping):
            raise ValueError(f"Evaluation frame {frame_index} lacks acquisition metadata")
        if acquisition.get("completeness") != "complete":
            continue
        key = _frame_evaluation_event_key(frame, frame_index=frame_index)
        if key in seen_keys:
            raise ValueError(f"Duplicate sensing evaluation frame key: {key}")
        seen_keys.add(key)
        if not _truth_velocity_is_evaluable(
            frame,
            context=f"Evaluation frame {frame_index}",
        ):
            continue
        indexed[key] = frame
    return indexed


def _ground_truth_for_evaluation(
    frame: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Read canonical event-bound truth, or an explicit empty truth event."""

    if frame is None or frame.get("_gt") is None:
        empty = np.array([], dtype=np.float64)
        return (), empty, empty.copy()
    gt = frame["_gt"]
    if not isinstance(gt, Mapping):
        raise ValueError("Sensing evaluation ground truth must be a mapping")
    gt_ranges = np.asarray(gt.get("target_ranges", ()), dtype=np.float64).ravel()
    gt_velocities = np.asarray(gt.get("target_speeds", ()), dtype=np.float64).ravel()
    target_ids_raw = gt.get("target_ids", ())
    if not isinstance(target_ids_raw, Sequence) or isinstance(target_ids_raw, (str, bytes)):
        raise ValueError("Sensing evaluation target_ids must be an ordered sequence")
    target_ids = tuple(target_ids_raw)
    if any(not isinstance(target_id, str) or not target_id for target_id in target_ids):
        raise ValueError("Sensing evaluation target_ids must be non-empty semantic strings")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Sensing evaluation target_ids must be unique within an event")
    if len(target_ids) != len(gt_ranges) or len(gt_velocities) != len(gt_ranges):
        raise ValueError("Sensing evaluation target_ids/ranges/velocities must align exactly")
    return target_ids, gt_ranges, gt_velocities


def _detections_for_evaluation(
    result: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Read one result's detections, or explicit emptiness for a missing result."""

    if result is None:
        return ()
    detections = result.get("detections", ())
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        raise ValueError("Sensing evaluation detections must be an ordered sequence")
    if any(not isinstance(detection, Mapping) for detection in detections):
        raise ValueError("Every sensing evaluation detection must be a mapping")
    return tuple(dict(detection) for detection in detections)


def _join_evaluation_events(
    frames: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> _EvaluationEventJoin:
    """Full outer-join truth and results by immutable temporal identity.

    A missing result becomes an empty detection set (and therefore contributes
    misses when truth exists). An extra result becomes an empty-truth event (and
    therefore contributes false alarms). Empty placeholder dictionaries are
    treated as absent results; non-empty results must carry exact lineage.
    """

    skipped_invalid_truth: list[SensingEvaluationEventKey] = []
    for frame_index, frame in enumerate(frames):
        grouped = frame.get("_grouped_temporal")
        temporal = grouped if isinstance(grouped, Mapping) else _frame_temporal(frame)
        acquisition = temporal.get("acquisition")
        if (
            isinstance(acquisition, Mapping)
            and acquisition.get("completeness") == "complete"
            and not _truth_velocity_is_evaluable(
                frame,
                context=f"Evaluation frame {frame_index}",
            )
        ):
            skipped_invalid_truth.append(
                _frame_evaluation_event_key(frame, frame_index=frame_index)
            )
    if len(set(skipped_invalid_truth)) != len(skipped_invalid_truth):
        raise ValueError("Duplicate sensing evaluation frame key in invalid-truth events")

    frames_by_key = _index_evaluation_frames(frames)
    results_by_key: dict[SensingEvaluationEventKey, Mapping[str, Any]] = {}
    run_by_key: dict[SensingEvaluationEventKey, str] = {}
    seen_occurrences: set[tuple[SensingEvaluationEventKey, str]] = set()
    for result_index, result in enumerate(results):
        if not result:
            continue
        key, processing_run_id = _result_evaluation_identity(
            result,
            result_index=result_index,
        )
        occurrence = (key, processing_run_id)
        if occurrence in seen_occurrences:
            raise ValueError(
                "Duplicate sensing evaluation result occurrence for "
                f"event {key} and processing_run_id {processing_run_id!r}"
            )
        seen_occurrences.add(occurrence)
        if key in results_by_key:
            raise ValueError(
                "Multiple sensing evaluation results exist for one temporal event; "
                "select one processing_run_id before evaluation: "
                f"{run_by_key[key]!r}, {processing_run_id!r} for {key}"
            )
        results_by_key[key] = result
        run_by_key[key] = processing_run_id

    for key in skipped_invalid_truth:
        results_by_key.pop(key, None)
    all_keys = sorted(set(frames_by_key) | set(results_by_key), key=_evaluation_event_sort_key)
    missing_result_keys = tuple(key for key in all_keys if key not in results_by_key)
    extra_result_keys = tuple(key for key in all_keys if key not in frames_by_key)
    events: list[_JoinedEvaluationEvent] = []
    for key in all_keys:
        frame = frames_by_key.get(key)
        result = results_by_key.get(key)
        target_ids, gt_ranges, gt_velocities = _ground_truth_for_evaluation(frame)
        mask = evaluation_velocity_mask((frame or {}).get("_gt") or {})
        events.append(
            _JoinedEvaluationEvent(
                key=key,
                target_ids=target_ids,
                gt_ranges=gt_ranges,
                gt_velocities=gt_velocities,
                detections=_detections_for_evaluation(result),
                frame_present=frame is not None,
                result_present=result is not None,
                gt_velocity_valid=None if mask is None else tuple(mask),
            )
        )
    return _EvaluationEventJoin(
        events=tuple(events),
        missing_result_keys=missing_result_keys,
        extra_result_keys=extra_result_keys,
        skipped_invalid_truth_keys=tuple(
            sorted(skipped_invalid_truth, key=_evaluation_event_sort_key)
        ),
    )


def _evaluation_join_metadata(join: _EvaluationEventJoin) -> dict[str, Any]:
    """Return JSON-safe evidence for missing and extra temporal events."""

    return {
        "evaluated_event_count": len(join.events),
        "missing_result_count": len(join.missing_result_keys),
        "extra_result_count": len(join.extra_result_keys),
        "skipped_invalid_truth_event_count": len(join.skipped_invalid_truth_keys),
        "missing_result_keys": [asdict(key) for key in join.missing_result_keys],
        "extra_result_keys": [asdict(key) for key in join.extra_result_keys],
        "skipped_invalid_truth_event_keys": [
            asdict(key) for key in join.skipped_invalid_truth_keys
        ],
    }


def _single_pulse_sample(frame: Mapping[str, Any], *, sample_index: int) -> dict[str, Any]:
    """Resolve the physical sample represented by one per-pulse storage frame."""

    temporal = _frame_temporal(frame)
    acquisition = temporal.get("acquisition")
    samples = acquisition.get("samples") if isinstance(acquisition, Mapping) else None
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("sensing.temporal.acquisition.samples must be an ordered sequence")

    outer_id = frame.get("_frame_index")
    matching: list[Mapping[str, Any]] = []
    valid: list[Mapping[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("valid") is not True:
            continue
        valid.append(sample)
        source = sample.get("source")
        if isinstance(source, Mapping) and source.get("scene_frame_id") == outer_id:
            matching.append(sample)
    candidates = matching or valid
    if len(candidates) != 1:
        raise ValueError(
            f"Per-pulse frame {outer_id!r} must identify exactly one physical source sample; "
            f"found {len(candidates)}"
        )

    resolved = deepcopy(dict(candidates[0]))
    resolved["sample_index"] = int(sample_index)
    source = resolved.get("source")
    if isinstance(source, dict):
        frame_set_id = frame.get("_source_frame_set_id")
        if frame_set_id is not None:
            source.setdefault("source_frame_set_id", str(frame_set_id))
    return resolved


def _acquisition_from_samples(
    *,
    acquisition_id: str,
    acquisition_config_id: str,
    pair_id: str,
    tx_sensor_id: str,
    rx_sensor_id: str,
    samples: Sequence[Mapping[str, Any]],
    completeness: str,
) -> dict[str, Any]:
    """Build the derived acquisition summary from authoritative ordered slots."""

    copied_samples = [deepcopy(dict(sample)) for sample in samples]
    if not copied_samples:
        raise ValueError("A sensing acquisition must contain at least one expected sample")
    times = [float(sample["sample_time_s"]) for sample in copied_samples]
    if not np.all(np.isfinite(times)) or any(
        current <= previous for previous, current in zip(times, times[1:])
    ):
        raise ValueError("Sensing sample timestamps must be finite and strictly increasing")
    observed = sum(isinstance(sample.get("source"), Mapping) for sample in copied_samples)
    valid = sum(sample.get("valid") is True for sample in copied_samples)
    return {
        "acquisition_id": acquisition_id,
        "acquisition_config_id": acquisition_config_id,
        "pair_id": pair_id,
        "tx_sensor_id": tx_sensor_id,
        "rx_sensor_id": rx_sensor_id,
        "start_time_s": times[0],
        "end_time_s": times[-1],
        "default_reference_time_s": (times[0] + times[-1]) / 2.0,
        "default_reference_policy": "cpi_center",
        "expected_sample_count": len(copied_samples),
        "observed_sample_count": observed,
        "valid_sample_count": valid,
        "completeness": completeness,
        "samples": copied_samples,
    }


def _grouped_slow_time_sensing(
    source_frames: Sequence[Mapping[str, Any]],
    products: Sequence[_TypedDetectionInput],
    *,
    clock: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one typed CPI product from exact one-pulse typed products."""

    if not source_frames or len(source_frames) != len(products):
        raise ValueError("Grouped slow-time construction requires one product per source frame")
    first = products[0]
    if first.rx_element_axis is None:
        raise ValueError("Typed slow-time grouping requires the RX-element catalog")
    expected_shape = (1, first.values.shape[1], 1, first.values.shape[3])
    expected_combine = first.descriptor.get("combine")
    for product in products:
        if product.values.shape != expected_shape:
            raise ValueError(
                "Every per-pulse typed slow-time product must use the same [P=1,E,N=1,R] "
                f"shape; expected {expected_shape}, got {product.values.shape}"
            )
        if product.rx_element_axis != first.rx_element_axis:
            raise ValueError("Per-pulse typed slow-time products changed RX-element identity")
        if not np.array_equal(product.range_axis_m, first.range_axis_m):
            raise ValueError("Per-pulse typed slow-time products changed exact range coordinates")
        if product.descriptor.get("combine") != expected_combine:
            raise ValueError("Per-pulse typed slow-time products changed combine policy")

    source_sensing = _typed_sensing_payload(source_frames[-1])
    if source_sensing is None:
        raise ValueError("Grouped slow-time construction requires schema-v4 source metadata")
    grouped: dict[str, Any] = {
        "schema_version": source_sensing.get("schema_version"),
        "temporal": {
            "clock": deepcopy(dict(clock)),
            "acquisition": deepcopy(dict(acquisition)),
        },
        "slow_time_by_pair_rx_element": np.concatenate(
            [product.values for product in products],
            axis=2,
        ),
        "range_axis_m": np.array(first.range_axis_m, copy=True),
        "rx_element_axis": [deepcopy(entry) for entry in first.rx_element_axis],
    }
    for key in ("config", "config_resolved", "processing_strategy", "rx_array"):
        if key in source_sensing:
            grouped[key] = deepcopy(source_sensing[key])

    source_axes = first.descriptor.get("axes")
    if not isinstance(source_axes, list) or len(source_axes) != 4:
        raise ValueError("Typed slow-time product has no complete axis metadata")
    element_axis = source_axes[1]
    slow_axis = source_axes[2]
    range_axis = source_axes[3]
    if not all(isinstance(axis, Mapping) for axis in (element_axis, slow_axis, range_axis)):
        raise ValueError("Typed slow-time product axis metadata is malformed")
    grouped_values = grouped["slow_time_by_pair_rx_element"]
    add_product_spec(
        grouped,
        data_key="slow_time_by_pair_rx_element",
        kind="delay_domain_slow_time",
        variant=str(first.descriptor.get("variant")),
        axes=(
            catalog_axis("pair", size=1, catalog_key="pair_axis"),
            catalog_axis(
                "rx_element",
                size=grouped_values.shape[1],
                catalog_key="rx_element_axis",
                coordinate_semantics=str(element_axis.get("coordinate_semantics")),
                reference_frame=str(element_axis.get("reference_frame")),
            ),
            dense_axis(
                "slow_time",
                size=grouped_values.shape[2],
                unit="s",
                coordinate_key="slow_time_axis_s",
                coordinate_semantics=str(slow_axis.get("coordinate_semantics")),
                reference_frame=str(slow_axis.get("reference_frame")),
            ),
            dense_axis(
                "range",
                size=grouped_values.shape[3],
                unit="m",
                coordinate_key="range_axis_m",
                coordinate_semantics=str(range_axis.get("coordinate_semantics")),
                reference_frame=str(range_axis.get("reference_frame")),
            ),
        ),
        representation="complex_voltage",
        unit=str(first.descriptor.get("unit")),
        scale_semantics=str(first.descriptor.get("scale_semantics")),
        evidence_role=str(first.descriptor.get("evidence_role")),
        source_product_ids=tuple(product.product_id for product in products),
        combine=(dict(expected_combine) if isinstance(expected_combine, Mapping) else None),
        processing=(
            dict(first.descriptor["processing"])
            if isinstance(first.descriptor.get("processing"), Mapping)
            else None
        ),
    )
    materialize_product_specs(grouped)
    return grouped


def _group_per_pulse_frames(
    frames: Sequence[dict[str, Any]],
    *,
    acquisition_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Group per-pulse v4 samples from their exact acquisition plan.

    The grouped frame index belongs to the derived frame set. The selected
    source storage row is retained separately so the writer can copy its
    physical scene into that local namespace. CPI length, hop, and partial
    policy have no compatibility defaults at this persisted-v4 boundary.
    """

    plan = _validate_native_v4_acquisition_plan(acquisition_plan)
    cpi_size = int(plan["cpi_length_samples"])
    cpi_hop_samples = int(plan["cpi_hop_samples"])
    partial_window_policy = str(plan["partial_window_policy"])
    if partial_window_policy != "retain_unprocessed":
        raise ValueError("Only partial_window_policy='retain_unprocessed' is supported")
    if plan["sample_source"] not in {"independent_solves", "measured_pulse_stream"}:
        raise ValueError(
            "Single-pulse CPI grouping requires an independent-solve or "
            "measured-pulse acquisition plan"
        )
    grouped_frames: list[dict[str, Any]] = []
    if cpi_hop_samples == 1:
        # Match online sequential processing: emit one right-aligned window per
        # observed pulse, retaining warm-up windows as incomplete/unprocessed.
        windows = [(max(0, end - cpi_size + 1), end + 1) for end in range(len(frames))]
    else:
        # Measured streams may explicitly declare a larger hop. It remains a
        # source acquisition fact, never a fallback inferred from CPI length.
        hop = cpi_hop_samples
        windows = [
            (start, min(start + cpi_size, len(frames))) for start in range(0, len(frames), hop)
        ]

    for start, stop in windows:
        source_frames = list(frames[start:stop])
        slow_time_products: list[_TypedDetectionInput] = []
        for source_frame in source_frames:
            source_sensing = _typed_sensing_payload(source_frame)
            if source_sensing is None:
                raise ValueError("Per-pulse CPI grouping requires schema-v4 source metadata")
            source_config = _resolve_native_v4_config(source_sensing)
            if source_config["acquisition"] != plan:
                raise ValueError(
                    "Every per-pulse source frame must carry the same exact resolved "
                    "acquisition plan used for grouping"
                )
            product = _select_typed_slow_time_input(source_frame)
            if product is None:
                raise ValueError(
                    "Per-pulse CPI grouping requires a typed "
                    "delay_domain_slow_time product in every source frame"
                )
            if product.values.shape[2] != 1:
                raise ValueError(
                    "Per-pulse CPI grouping requires N=1 typed slow-time products; "
                    f"received {product.values.shape}"
                )
            slow_time_products.append(product)
        samples = [
            _single_pulse_sample(frame, sample_index=index)
            for index, frame in enumerate(source_frames)
        ]
        for product, sample in zip(slow_time_products, samples, strict=True):
            assert product.slow_time_axis_s is not None
            if not np.isclose(
                float(product.slow_time_axis_s[0]),
                float(sample["sample_time_s"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Per-pulse typed slow-time coordinates must match temporal sample_time_s"
                )
        for sample in samples:
            sample["phase_coherent"] = False
        clocks = [_frame_temporal(frame).get("clock") for frame in source_frames]
        if not clocks or any(clock != clocks[0] for clock in clocks[1:]):
            raise ValueError("All samples in a CPI must use the same scenario clock")
        clock = clocks[0]
        if not isinstance(clock, Mapping):
            raise ValueError("A grouped CPI requires sensing.temporal.clock")

        # End-of-run missing slots are scheduled from the declared PRF, not inferred
        # from source identifier gaps.
        if len(samples) < cpi_size:
            sample_period_s = _native_v4_positive_float(
                plan["sample_period_s"],
                path="config_resolved.acquisition.sample_period_s",
            )
            next_time_s = float(samples[-1]["sample_time_s"]) + sample_period_s
            while len(samples) < cpi_size:
                samples.append(
                    {
                        "sample_index": len(samples),
                        "sample_time_s": next_time_s,
                        "valid": False,
                        "phase_coherent": False,
                        "invalid_reason": "end_of_run_missing_sample",
                    }
                )
                next_time_s += sample_period_s

        completeness = (
            "complete"
            if all(sample.get("valid") is True for sample in samples)
            else "incomplete_unprocessed"
        )
        source_acquisitions = [
            _frame_temporal(frame).get("acquisition", {}) for frame in source_frames
        ]
        pair_identities = [
            (
                (
                    acquisition.get("pair_id"),
                    acquisition.get("tx_sensor_id"),
                    acquisition.get("rx_sensor_id"),
                )
                if isinstance(acquisition, Mapping)
                else (None, None, None)
            )
            for acquisition in source_acquisitions
        ]
        if not pair_identities or any(
            not all(isinstance(value, str) and value.strip() for value in identity)
            for identity in pair_identities
        ):
            raise ValueError("Every per-pulse acquisition requires stable sensor pair identity")
        pair_identity = pair_identities[0]
        if any(identity != pair_identity for identity in pair_identities[1:]):
            raise ValueError("All samples in a CPI must use the same TX/RX pair identity")
        pair_id, tx_sensor_id, rx_sensor_id = pair_identity
        acquisition_identity: dict[str, Any] = {
            "kind": "per_pulse_cpi",
            "cpi_size": cpi_size,
            "cpi_hop_samples": cpi_hop_samples,
            "partial_window_policy": partial_window_policy,
            "pair": {
                "pair_id": pair_id,
                "tx_sensor_id": tx_sensor_id,
                "rx_sensor_id": rx_sensor_id,
            },
        }
        acquisition_identity["resolved_acquisition"] = dict(plan)
        acquisition_config_id = canonical_sensing_hash(acquisition_identity)
        clock_id = str(clock.get("clock_id"))
        membership_hash = canonical_sensing_hash(
            {
                "clock_id": clock_id,
                "acquisition_config_id": acquisition_config_id,
                "pair_id": pair_id,
                "samples": samples,
            }
        )
        acquisition_id = str(uuid5(UUID(clock_id), membership_hash))
        acquisition = _acquisition_from_samples(
            acquisition_id=acquisition_id,
            acquisition_config_id=acquisition_config_id,
            pair_id=pair_id,
            tx_sensor_id=tx_sensor_id,
            rx_sensor_id=rx_sensor_id,
            samples=samples,
            completeness=completeness,
        )
        reference_time_s = float(acquisition["default_reference_time_s"])
        eligible = [
            sample
            for sample in samples
            if sample.get("valid") is True
            and float(sample["sample_time_s"]) <= reference_time_s
            and isinstance(sample.get("source"), Mapping)
        ]
        display_scene_id = eligible[-1]["source"]["scene_frame_id"] if eligible else None
        display_sample_position = (
            int(eligible[-1]["sample_index"]) if eligible else len(samples) - 1
        )
        source_frame_step = int(
            source_frames[display_sample_position].get(
                "_frame_index",
                start + display_sample_position,
            )
        )
        derived_frame_id = len(grouped_frames)

        representative = deepcopy(source_frames[-1])
        # A CPI-center result has no honest last-pulse truth binding. Omit the
        # convenience projection; _source_sensing_payload also removes the
        # persisted source GT channels for this grouped result.
        representative.pop("_gt", None)
        representative.pop("slow_time_matrix", None)
        representative.pop("range_doppler_map", None)
        if completeness == "complete":
            grouped_sensing = _grouped_slow_time_sensing(
                source_frames,
                slow_time_products,
                clock=clock,
                acquisition=acquisition,
            )
            representative["_typed_sensing"] = grouped_sensing
            representative["_grouped_typed_sensing"] = grouped_sensing
        # Packed frame IDs are local, numeric, and strictly increasing. The
        # writer copies the selected source scene under this derived identity.
        representative["_frame_index"] = derived_frame_id
        representative["_pulse_range"] = (start, stop)
        representative["_source_frame_step"] = source_frame_step
        representative["_grouped_temporal"] = {
            "clock": deepcopy(dict(clock)),
            "acquisition": acquisition,
            "reference_time_s": reference_time_s,
            "reference_time_policy": "cpi_center",
            "display_scene_frame_id": display_scene_id,
            "display_policy": "nearest_at_or_before_reference",
            "display_reason": "evolving per-pulse CPI binds to the latest source scene at or before center",
        }
        grouped_frames.append(representative)
    return grouped_frames


def _source_display_binding(frame: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the source-scene binding that a derived result supersedes."""

    grouped = frame.get("_grouped_temporal")
    if isinstance(grouped, Mapping):
        return {
            "scene_frame_id": grouped.get("display_scene_frame_id"),
            "policy": grouped.get("display_policy"),
            "reason": grouped.get("display_reason"),
        }

    source_temporal = _frame_temporal(frame)
    source_result = source_temporal.get("result")
    display = source_temporal.get("display_binding")
    acquisition = source_temporal.get("acquisition")
    if isinstance(source_result, Mapping) and isinstance(display, Mapping):
        return deepcopy(dict(display))
    if source_result is not None or display is not None:
        raise ValueError("Source temporal result and display binding must appear together")
    if not isinstance(acquisition, Mapping) or acquisition.get("completeness") != "complete":
        return None

    reference_time_s = float(acquisition["default_reference_time_s"])
    samples = acquisition.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("A metadata-only acquisition requires ordered samples")
    eligible = [
        sample
        for sample in samples
        if isinstance(sample, Mapping)
        and sample.get("valid") is True
        and isinstance(sample.get("source"), Mapping)
        and float(sample["sample_time_s"]) <= reference_time_s
    ]
    if not eligible:
        raise ValueError("A metadata-only acquisition has no source scene at its reference")
    return {
        "scene_frame_id": eligible[-1]["source"]["scene_frame_id"],
        "policy": "nearest_at_or_before_reference",
        "reason": (
            "metadata-only acquisition selects the latest source scene at or before "
            "its default reference"
        ),
    }


def _source_scene_frame_step(frame: Mapping[str, Any]) -> int:
    """Resolve the packed source row whose physical scene must be copied."""

    explicit_step = frame.get("_source_frame_step")
    if explicit_step is not None:
        return int(explicit_step)
    source_binding = _source_display_binding(frame)
    if source_binding is None:
        return int(frame.get("_frame_index", 0))
    scene_frame_id = source_binding.get("scene_frame_id")
    if not isinstance(scene_frame_id, (int, np.integer)) or isinstance(
        scene_frame_id, (bool, np.bool_)
    ):
        raise ValueError(
            "Self-contained derived HDF5 requires a numeric source display scene frame ID"
        )
    return int(scene_frame_id)


def _source_sensing_payload(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the source product payload independently of copied scene pose."""

    grouped = frame.get("_grouped_typed_sensing")
    if isinstance(grouped, Mapping):
        return deepcopy(dict(grouped))
    typed = frame.get("_typed_sensing")
    if isinstance(typed, Mapping):
        # Descriptors can introduce new products and coordinates without
        # extending the legacy compatibility-array whitelist.
        return deepcopy(dict(typed))
    metadata = frame.get("_sensing_json")
    sensing = deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else {}
    for key in SENSING_DENSE_DATASET_KEYS:
        if key in frame:
            sensing[key] = np.array(frame[key], copy=True)
    if not _has_result_bound_truth(frame):
        for key in GT_CHANNEL_PRECEDENCE:
            sensing.pop(key, None)
    return sensing


def _has_result_bound_truth(frame: Mapping[str, Any]) -> bool:
    """Return whether frame GT is explicitly bound to its result epoch."""

    if isinstance(frame.get("_grouped_temporal"), Mapping):
        return False
    temporal = _frame_temporal(frame)
    result = temporal.get("result")
    truth = temporal.get("truth")
    if not isinstance(result, Mapping) or not isinstance(truth, Mapping):
        return False
    try:
        result_time_s = float(result["reference_time_s"])
        truth_time_s = float(truth["reference_time_s"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(result_time_s)
        and np.isfinite(truth_time_s)
        and np.isclose(result_time_s, truth_time_s, rtol=0.0, atol=1e-12)
    )


def _tracking_enabled(config: PostProcessingConfig) -> bool:
    """Mirror tracker activation, including the legacy M-of-N shortcut."""

    return config.tracker_type.lower() != "none" or (
        config.m_of_n_m is not None and config.m_of_n_n is not None
    )


def _temporal_stream_identity(frame: Mapping[str, Any]) -> tuple[str, str]:
    """Return the clock/pair identity that scopes stateful tracking."""

    grouped = frame.get("_grouped_temporal")
    temporal = grouped if isinstance(grouped, Mapping) else _frame_temporal(frame)
    clock = temporal.get("clock")
    acquisition = temporal.get("acquisition")
    if not isinstance(clock, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError("Tracking requires temporal clock and acquisition metadata")
    clock_id = clock.get("clock_id")
    pair_id = acquisition.get("pair_id")
    if not isinstance(clock_id, str) or not isinstance(pair_id, str):
        raise ValueError("Tracking requires stable clock_id and pair_id values")
    return clock_id, pair_id


def _source_sensing_metadata(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the source sensing metadata mapping used for recipe resolution."""

    raw = frame.get("_sensing_json")
    if isinstance(raw, Mapping):
        return raw
    return frame


def _build_effective_processing_recipe(
    frame: Mapping[str, Any],
    *,
    postprocessing_config: PostProcessingConfig,
    input_kind: str,
) -> dict[str, Any]:
    """Describe every inherited and overridden value used by reprocessing."""

    source = _source_sensing_metadata(frame)
    inherited = SensingPostProcessor._resolve_meta(dict(source))
    resolved_block = source.get("config_resolved")
    if not isinstance(resolved_block, Mapping):
        raise ValueError("Native schema-v4 reprocessing recipe requires source config_resolved")
    source_recipe = resolved_block.get("processing_recipe")
    source_execution = resolved_block.get("processing")
    recipe = {
        "tool": "shared.sensing_postprocessor",
        "input_kind": input_kind,
        "inherited_numeric_model": inherited,
        "source_processing": (
            deepcopy(dict(source_execution)) if isinstance(source_execution, Mapping) else None
        ),
        "source_recipe": (
            deepcopy(dict(source_recipe)) if isinstance(source_recipe, Mapping) else None
        ),
        "effective_overrides": processing_config_dict(postprocessing_config),
    }
    if frame.get("_output_selection") is not None:
        recipe["output_selection"] = deepcopy(frame["_output_selection"])
    if frame.get("_output_recording_profile") is not None:
        recipe["output_recording_profile"] = frame["_output_recording_profile"]
    return recipe


def _build_reprocessed_temporal(
    frame: Mapping[str, Any],
    *,
    processing_run_id: str,
    postprocessing_config: PostProcessingConfig,
    input_kind: str,
) -> dict[str, Any]:
    """Create immutable result lineage for one reprocessed acquisition."""

    source_display = _source_display_binding(frame)
    grouped = frame.get("_grouped_temporal")
    if isinstance(grouped, Mapping):
        clock = grouped.get("clock")
        acquisition = grouped.get("acquisition")
        reference_time_s = grouped.get("reference_time_s")
        reference_policy = grouped.get("reference_time_policy")
        truth = None
    else:
        source_temporal = _frame_temporal(frame)
        clock = source_temporal.get("clock")
        acquisition = source_temporal.get("acquisition")
        source_result = source_temporal.get("result")
        display = source_temporal.get("display_binding")
        if isinstance(source_result, Mapping) and isinstance(display, Mapping):
            reference_time_s = source_result.get("reference_time_s")
            reference_policy = source_result.get("reference_time_policy")
            truth = source_temporal.get("truth")
        elif source_result is None and display is None and isinstance(acquisition, Mapping):
            # A complete acquisition may intentionally be stored without a
            # result. Reprocessing creates the first result at the acquisition's
            # declared default epoch and binds it to an actual contributing
            # source scene without interpreting opaque frame IDs as time.
            if acquisition.get("completeness") != "complete":
                raise ValueError(
                    "An incomplete_unprocessed acquisition requires an explicit partial-processing policy"
                )
            reference_time_s = acquisition.get("default_reference_time_s")
            reference_policy = acquisition.get("default_reference_policy")
            truth = None
        else:
            raise ValueError("Source temporal result and display binding must appear together")

    if not isinstance(clock, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError("Reprocessing requires temporal clock and acquisition metadata")
    if source_display is None:
        raise ValueError("Reprocessing a complete acquisition requires a source display binding")
    derived_scene_frame_id = frame.get("_frame_index")
    if not isinstance(derived_scene_frame_id, (int, np.integer)) or isinstance(
        derived_scene_frame_id, (bool, np.bool_)
    ):
        raise ValueError("Derived sensing output requires a numeric local frame ID")
    effective_recipe = _build_effective_processing_recipe(
        frame,
        postprocessing_config=postprocessing_config,
        input_kind=input_kind,
    )
    recipe_id = canonical_processing_recipe_id(effective_recipe)
    return build_sensing_temporal_contract(
        clock_id=str(clock["clock_id"]),
        acquisition_id=str(acquisition["acquisition_id"]),
        acquisition_config_id=str(acquisition["acquisition_config_id"]),
        pair_id=str(acquisition["pair_id"]),
        tx_sensor_id=str(acquisition["tx_sensor_id"]),
        rx_sensor_id=str(acquisition["rx_sensor_id"]),
        samples=acquisition["samples"],
        processing_run_id=processing_run_id,
        processing_recipe_id=recipe_id,
        reference_time_s=float(reference_time_s),
        reference_time_policy=str(reference_policy),
        display_scene_frame_id=int(derived_scene_frame_id),
        display_policy="exact_derived_scene",
        display_reason=(
            "derived frame carries the source-bound physical scene in its local frame namespace"
        ),
        completeness=str(acquisition["completeness"]),
        truth=truth if isinstance(truth, Mapping) else None,
    )


def _unprocessed_acquisition_temporal(frame: Mapping[str, Any]) -> dict[str, Any] | None:
    """Preserve incomplete or explicitly off-cadence native observations."""

    grouped = frame.get("_grouped_temporal")
    temporal = grouped if isinstance(grouped, Mapping) else _frame_temporal(frame)
    clock = temporal.get("clock")
    acquisition = temporal.get("acquisition")
    if not isinstance(clock, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError("Sensing metadata is missing temporal clock/acquisition")
    if acquisition.get("completeness") != "incomplete_unprocessed":
        source = _source_sensing_metadata(frame)
        config = source.get("config_resolved", {})
        preserve_native_cadence = (
            frame.get("_output_selection") is None
            and config.get("processing", {}).get("mode") == "sequential_stft"
            and config.get("acquisition", {}).get("cpi_hop_samples", 1) > 1
            and temporal.get("result") is None
            and temporal.get("display_binding") is None
        )
        if not preserve_native_cadence:
            return None
    return {
        "clock": deepcopy(dict(clock)),
        "acquisition": deepcopy(dict(acquisition)),
    }


def _remove_empty_claimed_analysis_root(output_root: Path) -> None:
    """Release this run's analysis root when no output survived a failure."""

    try:
        Path(output_root).rmdir()
    except FileNotFoundError:
        return
    except OSError:
        # A nonempty run contains useful diagnostics or partial optional
        # products and requires an explicit user cleanup decision.
        return


@dataclass(frozen=True)
class _StreamingSourceScan:
    """Bounded preflight facts needed before a streaming output is claimed."""

    frame_count: int
    range_windows: frozenset[str]
    doppler_windows: frozenset[str]
    has_slow_time: bool
    has_per_pulse_slow_time: bool
    first_default_source_step: int | None
    first_rebuild_source_step: int | None


def _has_complex_rebuild_input(frame: Mapping[str, Any]) -> bool:
    """Inspect retained kinds without materializing an inverse transform."""
    sensing = _typed_sensing_payload(frame)
    return bool(
        sensing
        and any(
            descriptor.get("kind") in {"delay_domain_slow_time", "baseband_frequency_observation"}
            for descriptor in sensing.get("products", {}).values()
        )
    )


def _scan_streaming_source(
    frames_dir: Path, *, frame_ids: tuple[int, ...] | None = None
) -> _StreamingSourceScan:
    """Validate source recipes without retaining the frame sequence."""

    frame_count = 0
    range_windows: set[str] = set()
    doppler_windows: set[str] = set()
    has_slow_time = False
    has_per_pulse_slow_time = False
    first_default_source_step: int | None = None
    first_rebuild_source_step: int | None = None
    source_iterator = (
        _iter_frames_sensing(frames_dir)
        if frame_ids is None
        else _iter_frames_sensing(frames_dir, frame_ids=frame_ids)
    )
    with closing(source_iterator) as iterator:
        for frame in iterator:
            frame_count += 1
            range_window, doppler_window = _native_v4_frame_windows(frame)
            range_windows.add(range_window)
            doppler_windows.add(doppler_window)
            slow_time = _select_typed_slow_time_input(frame)
            has_slow_time = has_slow_time or _has_complex_rebuild_input(frame)
            if slow_time is not None:
                has_slow_time = True
                has_per_pulse_slow_time = has_per_pulse_slow_time or slow_time.values.shape[2] == 1
            incomplete_temporal = (
                _unprocessed_acquisition_temporal(frame) if frame_ids is None else None
            )
            if incomplete_temporal is not None:
                default_writable = True
                rebuild_writable = True
            else:
                default_writable = (
                    _select_typed_detection_input(frame, rebuild_rd=False) is not None
                )
                # Rebuild eligibility is exactly the presence of the typed
                # slow-time product selected above.  Do not probe the strict
                # rebuild route here: it intentionally rejects a valid stored
                # range-Doppler-only frame, which remains writable in the
                # default streaming lane.
                rebuild_writable = _has_complex_rebuild_input(frame)
            if first_default_source_step is None and default_writable:
                first_default_source_step = _source_scene_frame_step(frame)
            if first_rebuild_source_step is None and rebuild_writable:
                first_rebuild_source_step = _source_scene_frame_step(frame)
    return _StreamingSourceScan(
        frame_count=frame_count,
        range_windows=frozenset(range_windows),
        doppler_windows=frozenset(doppler_windows),
        has_slow_time=has_slow_time,
        has_per_pulse_slow_time=has_per_pulse_slow_time,
        first_default_source_step=first_default_source_step,
        first_rebuild_source_step=first_rebuild_source_step,
    )


def _streaming_incompatible_options(args: Any) -> tuple[str, ...]:
    """Return CLI features that require retained cross-frame state."""

    incompatible: list[str] = []
    for enabled, label in (
        (args.evaluate, "--evaluate"),
        (args.report, "--report"),
        (args.threshold_sweep, "--threshold-sweep"),
        (args.figures or bool(args.figure), "--figures/--figure"),
        (
            args.gif
            or args.gif_only
            or args.gif_3d
            or args.gif_per_frame_vmax
            or args.gif_no_detections,
            "GIF generation",
        ),
        (args.micro_doppler or args.vertex_overlay is not None, "micro-Doppler/vertex overlay"),
        (args.ghost_track_correlation is not None, "batch track-correlation ghost pruning"),
        (args.dry_run, "--dry-run"),
    ):
        if enabled:
            incompatible.append(label)
    return tuple(incompatible)


def _requested_cli_dependencies(args: Any) -> dict[str, set[str]]:
    """Map optional import modules to the requested operations that need them."""

    requirements: dict[str, set[str]] = {}

    def require(module: str, reason: str) -> None:
        requirements.setdefault(module, set()).add(reason)

    # A normal dry run stops before processing or plotting. Vertex overlay is
    # currently an explicit pre-summary computation and remains an exception.
    if args.dry_run and args.vertex_overlay is None:
        return requirements

    evaluation_requested = bool(
        args.evaluate or args.report or args.threshold_sweep or args.figures or args.figure
    )
    if evaluation_requested:
        require("scipy", "evaluation/report/threshold metrics")

    tracker_type = str(args.tracker).lower()
    if tracker_type in _HUNGARIAN_TRACKERS:
        require("scipy", f"{tracker_type} Hungarian tracker association")

    if args.vote_fusion:
        require("sklearn", "vote-fusion 3D clustering")
    if args.angle_peak_detection:
        require("sklearn", "angle-enhanced 3D clustering")

    plotting_requested = bool(
        args.figures
        or args.figure
        or args.gif
        or args.gif_only
        or args.gif_3d
        or args.micro_doppler
        or args.vertex_overlay is not None
    )
    if plotting_requested:
        require("matplotlib", "requested sensing plots")

    if args.gif or args.gif_only or args.gif_3d:
        require("PIL", "requested sensing GIF")

    if args.vertex_overlay is not None:
        require("scipy", "vertex-velocity smoothing")
        require("trimesh", "vertex-overlay PLY loading")

    return requirements


def _preflight_cli_dependencies(args: Any) -> None:
    """Fail before output publication when dependencies for the requested work are absent."""

    requirements = _requested_cli_dependencies(args)
    missing: list[str] = []
    for module in requirements:
        try:
            loaded_module = importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != module:
                raise
            missing.append(module)
            continue
        if module == "matplotlib":
            # Match every sensing figure path: probe pyplot only after forcing
            # the renderer-free backend, never whichever interactive backend
            # happens to be configured on the host.
            loaded_module.use("Agg")
        for probe in _SENSING_DEPENDENCY_PROBES[module]:
            if probe != module:
                # Once the distribution root imports, failures in an expected
                # runtime submodule indicate a broken installation and must
                # retain their original traceback rather than masquerading as
                # a missing dependency.
                importlib.import_module(probe)

    if not missing:
        return

    messages: list[str] = []
    sensing_missing = [module for module in missing if module in _SENSING_DEPENDENCY_NAMES]
    if sensing_missing:
        details = "; ".join(
            f"{_SENSING_DEPENDENCY_NAMES[module]} ({', '.join(sorted(requirements[module]))})"
            for module in sensing_missing
        )
        messages.append(f"Missing sensing dependencies: {details}.")
        messages.append(_SENSING_INSTALL_GUIDANCE)
    if "trimesh" in missing:
        messages.append(_VERTEX_OVERLAY_INSTALL_GUIDANCE)
    raise RuntimeError(" ".join(messages))


def _run_streaming_result_generation(
    *,
    args: Any,
    output_dir: Path,
    config: PostProcessingConfig,
    postprocessor: SensingPostProcessor,
    total_sensing_frames: int,
    first_source_step: int,
) -> None:
    """Write results with one input/result live plus one fixed-size output chunk."""

    source_provider = Hdf5Provider(
        args.frames_dir.parent,
        frames_subdir=args.frames_dir.name,
    )
    selection = getattr(args, "_window_selection", None)

    def source_scene(step):
        if selection is None:
            return source_provider.load_frame(step)
        # Its sensing block is replaced by the selected acquisition's output.
        # Copy the full physical scene without reading unrelated CFR windows.
        from dataclasses import fields

        from shared.frames.contracts import FrameComponent, FrameReadRequest

        projected = source_provider.load_frame_projection(
            step,
            FrameReadRequest(components=frozenset(FrameComponent) - {FrameComponent.SENSING}),
        ).frame
        return StandardMPCFrame(
            **{field.name: getattr(projected, field.name) for field in fields(StandardMPCFrame)}
        )

    try:
        first_source_frame = source_scene(first_source_step)
    except BaseException:
        source_provider.close()
        raise

    try:
        _claim_analysis_root(output_dir)
    except BaseException:
        source_provider.close()
        raise
    try:
        columnar_writer = _ColumnarResultWriter(output_dir, chunk_size=100)
    except BaseException:
        source_provider.close()
        _remove_empty_claimed_analysis_root(output_dir)
        raise

    processed = 0
    processing_run_id = str(uuid4())
    tracker_stream_identity: tuple[str, str] | None = None
    try:
        source_iterator = (
            _iter_frames_sensing(args.frames_dir)
            if selection is None
            else _iter_frames_sensing(args.frames_dir, frame_ids=selection.frame_ids)
        )
        with closing(source_iterator) as iterator:
            for idx, frame in enumerate(iterator):
                if selection is not None:
                    selection.bind(frame, idx)
                if getattr(args, "recording_profile", None) is not None:
                    frame["_output_recording_profile"] = args.recording_profile
                if not _has_result_bound_truth(frame):
                    frame.pop("_gt", None)
                frame_step = int(frame.get("_frame_index", idx))
                if not _has_detection_input(frame):
                    logger.debug("Frame %d: no suitable data found, skipping", idx)
                    continue

                src_step = _source_scene_frame_step(frame)
                if src_step == first_source_step and first_source_frame is not None:
                    source_frame = first_source_frame
                    first_source_frame = None
                else:
                    source_frame = source_scene(src_step)
                source_sensing = _source_sensing_payload(frame)
                if not source_sensing:
                    source_sensing = deepcopy(dict(source_frame.sensing or {}))
                meta = source_sensing or frame.get("_sensing_json") or frame.get("config", frame)
                incomplete_temporal = _unprocessed_acquisition_temporal(frame)
                if incomplete_temporal is not None:
                    columnar_writer.add_incomplete_acquisition(
                        temporal=incomplete_temporal,
                        frame_index=frame_step,
                        source_frame=source_frame,
                        source_sensing=source_sensing,
                        postprocessing_config=config,
                    )
                    logger.info(
                        "Frame %d: retained unprocessed acquisition without coherent result",
                        frame_step,
                    )
                    continue

                typed_input = _select_typed_detection_input(
                    frame,
                    rebuild_rd=bool(args.rebuild_rd),
                    rebuild_cfr=config.range_response is not None
                    or config.range_ifft_size is not None,
                )
                if typed_input is None:
                    logger.debug("Frame %d: no typed detection product found, skipping", idx)
                    continue
                input_kind = typed_input.data_key
                source_display_binding = _source_display_binding(frame)
                if _tracking_enabled(config):
                    current_stream_identity = _temporal_stream_identity(frame)
                    if tracker_stream_identity is None:
                        tracker_stream_identity = current_stream_identity
                    elif current_stream_identity != tracker_stream_identity:
                        raise ValueError(
                            "Tracking cannot share state across sensing clock or sensor-pair "
                            f"changes: expected {tracker_stream_identity}, got "
                            f"{current_stream_identity}"
                        )

                temporal = _build_reprocessed_temporal(
                    frame,
                    processing_run_id=processing_run_id,
                    postprocessing_config=config,
                    input_kind=input_kind,
                )
                update_time_s = float(temporal["result"]["reference_time_s"])
                if typed_input.kind == "baseband_frequency_observation":
                    result = postprocessor.process_from_frequency_observation(
                        meta, update_time_s=update_time_s
                    )
                elif typed_input.kind == "delay_domain_slow_time":
                    assert typed_input.slow_time_axis_s is not None
                    result = postprocessor.process_from_slow_time_matrix(
                        _combine_typed_slow_time(typed_input),
                        meta,
                        range_axis_m=typed_input.range_axis_m,
                        slow_time_axis_s=typed_input.slow_time_axis_s,
                        update_time_s=update_time_s,
                    )
                else:
                    assert typed_input.doppler_axis_hz is not None
                    rd_power, rd_cube = _typed_range_doppler_arrays(typed_input)
                    result = postprocessor.process_from_rd_map(
                        rd_power,
                        meta,
                        range_axis_m=typed_input.range_axis_m,
                        doppler_axis_hz=typed_input.doppler_axis_hz,
                        rd_cube=rd_cube,
                        update_time_s=update_time_s,
                    )
                result["processing_recipe_resolved"] = _build_effective_processing_recipe(
                    frame,
                    postprocessing_config=config,
                    input_kind=input_kind,
                )
                result["temporal"] = temporal
                result["_frame_index"] = frame_step
                result["_input_kind"] = input_kind
                columnar_writer.add_frame(
                    result,
                    frame_index=frame_step,
                    source_frame=source_frame,
                    source_sensing=source_sensing,
                    postprocessing_config=config,
                    input_kind=input_kind,
                    source_display_binding=source_display_binding,
                )
                if args.save_stage_artifacts:
                    _save_stage_artifacts(output_dir, result, frame_index=frame_step)
                logger.info(
                    "Frame %d: %d detections",
                    frame_step,
                    len(result.get("detections", [])),
                )
                processed += 1
        if selection is not None and processed != len(selection.frame_ids):
            raise ValueError("Derived output count does not match its selected window schedule")
    except BaseException:
        columnar_writer.abort()
        _remove_empty_claimed_analysis_root(output_dir)
        raise
    finally:
        source_provider.close()

    try:
        columnar_writer.finalize()
        from shared.sensing.result_scenario import write_result_scenario

        write_result_scenario(args.frames_dir, output_dir)
    except BaseException:
        columnar_writer.abort()
        _remove_empty_claimed_analysis_root(output_dir)
        raise

    logger.info(
        "Done (streaming). Processed %d/%d frames. Results in %s",
        processed,
        total_sensing_frames,
        output_dir,
    )
    summary_path = output_dir / "postprocessing_config.json"
    with open(summary_path, "w") as handle:
        json.dump(processing_config_dict(config), handle, indent=2)
    logger.info("Config saved to %s", summary_path)


def main() -> None:
    """CLI entry point for reprocessing sensing extensions from saved frames."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Re-run sensing detection pipeline from saved HDF5 frames.",
    )
    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Directory containing HDF5 frames with sensing data.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "New analysis-run root for derived frames, figures, and reports. "
            "Required except for --dry-run; --gif-only requires an existing completed root."
        ),
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        help=(
            "Compare named processing recipes from a JSON file on complete saved RD results. "
            "Use with the source and --output-dir; put processing/evaluation options in the file."
        ),
    )
    parser.add_argument(
        "--frequency-window",
        choices=("boxcar", "hann"),
        help="Rewindow the same retained CFR observation in a new processing run.",
    )
    parser.add_argument(
        "--range-ifft-size",
        type=int,
        help="CFR IFFT length (>= retained frequency count); changes sampling only.",
    )
    parser.add_argument(
        "--rebuild-rd",
        action="store_true",
        help="Rebuild RD map from slow_time_matrix (re-apply windowing + FFT).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Bound sensing-frame/result retention while generating derived HDF5 results. "
            "Cross-frame evaluation, reports, sweeps, figures/GIFs, micro-Doppler, "
            "batch ghost pruning, dry-run, and per-pulse CPI grouping are unsupported."
        ),
    )
    parser.add_argument(
        "--result-hop-samples",
        type=int,
        default=None,
        help="Select existing complete windows at this sample spacing for a new streaming result run.",
    )
    parser.add_argument(
        "--recording-profile",
        choices=("maps",),
        default=None,
        help=(
            "Retain power maps and saved display products from selected complete-window CFR replay, "
            "omitting complex observations; requires --streaming --result-hop-samples --rebuild-rd."
        ),
    )
    parser.add_argument(
        "--no-mti",
        action="store_true",
        help="Disable MTI clutter removal.",
    )
    parser.add_argument(
        "--coherent-clutter-mode",
        choices=("none", "windowed_mean", "legacy"),
        default=None,
        help="Rebuild complex samples with no mean removal, recommended window-weighted "
        "mean removal, or the historical conditional unweighted mean. Omission inherits "
        "the stored coherent recipe. Requires retained complex samples.",
    )
    parser.add_argument(
        "--cfar-threshold",
        type=float,
        default=3.0,
        help="CFAR threshold scale (default: 3.0).",
    )
    parser.add_argument(
        "--cfar-min-snr",
        type=float,
        default=6.0,
        help="Minimum SNR in dB for valid detection (default: 6.0).",
    )
    parser.add_argument(
        "--cfar-max-dynamic-range",
        type=float,
        default=None,
        help="Optional detector power span in dB below the strongest input cell (per beam for beam_scan); a sensitivity bound, not receiver noise.",
    )
    parser.add_argument("--cfar-guard-range", type=int, default=2, help="Range CFAR guard cells.")
    parser.add_argument(
        "--cfar-guard-doppler", type=int, default=1, help="Doppler CFAR guard cells."
    )
    parser.add_argument(
        "--cfar-ref-range", type=int, default=8, help="Range CFAR reference extent."
    )
    parser.add_argument(
        "--cfar-ref-doppler", type=int, default=4, help="Doppler CFAR reference extent."
    )
    parser.add_argument(
        "--array-detection-mode",
        choices=["rd", "beam_scan"],
        default="rd",
        help="Detect on the fixed broadside RD map or within retained spatial-FFT beams.",
    )
    parser.add_argument(
        "--beam-angle-oversampling",
        type=int,
        default=1,
        help="Offline beam-angle interpolation factor (1–16, default 1/off); requires beam_scan and retained complex per-element slow time.",
    )
    parser.add_argument(
        "--cfar-type",
        choices=["ca", "os"],
        default="ca",
        help="CFAR method (default: ca).",
    )
    parser.add_argument(
        "--cfar-edge-policy",
        choices=["full_window", "truncate"],
        default="full_window",
        help=(
            "CFAR edge-cell policy: require the complete configured training window "
            "or use the exact available cells (default: full_window)."
        ),
    )
    parser.add_argument(
        "--pre-cfar-doppler-edge-guard-bins",
        type=int,
        default=None,
        help="Override coherent pre-CFAR Doppler edge guard bins (default: use saved config, else 2).",
    )
    parser.add_argument(
        "--pre-cfar-doppler-dc-guard-bins",
        type=int,
        default=None,
        help="Override coherent pre-CFAR Doppler DC guard bins (default: use saved config, else 0).",
    )
    parser.add_argument(
        "--window-range",
        default=None,
        help=(
            "Override the stored range window function. Without this option, native "
            "schema-v4 reprocessing inherits the exact persisted processing recipe."
        ),
    )
    parser.add_argument(
        "--window-doppler",
        default=None,
        help=(
            "Override the stored Doppler window function. Without this option, native "
            "schema-v4 reprocessing inherits the exact persisted processing recipe."
        ),
    )
    parser.add_argument(
        "--min-range",
        type=float,
        default=None,
        help="Minimum detection range in meters.",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Maximum detection range in meters.",
    )
    # Clustering gates (physical-space, anisotropic)
    parser.add_argument(
        "--cluster-range-gate",
        type=float,
        default=1.0,
        help="Clustering range gate in metres (default: 1.0). "
        "Detections within this range difference are grouped as one target.",
    )
    parser.add_argument(
        "--cluster-velocity-gate",
        type=float,
        default=5.0,
        help="Clustering velocity gate in m/s (default: 5.0). "
        "Set wide for extended targets with micro-Doppler (pedestrians).",
    )
    # Ghost filter (shortest-range per velocity group)
    parser.add_argument(
        "--ghost-filter-velocity-gate",
        type=float,
        default=None,
        help="Enable a heuristic that keeps only the shortest-range detection per "
        "velocity group. Value is the velocity gate in m/s for grouping (e.g., 3.0). "
        "This can suppress a farther physical target and does not establish path provenance.",
    )
    parser.add_argument(
        "--ghost-filter-azimuth-gate",
        type=float,
        default=None,
        help="Optional azimuth condition in degrees for the ghost-mitigation heuristic. "
        "When provided, "
        "a farther detection is only suppressed if it is close in both velocity "
        "and azimuth. Falls back to velocity-only filtering when azimuth data "
        "is unavailable.",
    )
    parser.add_argument(
        "--angle-peak-detection",
        action="store_true",
        help="Enable RD-anchored azimuth peak extraction and 3D angular clustering "
        "on stored range-doppler-azimuth cubes.",
    )
    parser.add_argument(
        "--angle-max-peaks-per-rd",
        type=int,
        default=2,
        help="Maximum azimuth peaks to keep per RD detection cell when "
        "--angle-peak-detection is enabled (default: 2).",
    )
    parser.add_argument(
        "--angle-peak-relative-floor-db",
        type=float,
        default=10.0,
        help="Only keep azimuth peaks within this many dB of the strongest "
        "peak in each RD cell (default: 10.0 dB).",
    )
    parser.add_argument(
        "--angle-edge-exclusion-bins",
        type=int,
        default=0,
        help="Ignore azimuth peaks that fall within this many bins of either "
        "edge of the azimuth axis (default: 0). Useful to suppress beamspace "
        "edge artifacts near +/-90 degrees.",
    )
    parser.add_argument(
        "--angle-secondary-peak-max-gap-db",
        type=float,
        default=3.0,
        help="Maximum allowed dB gap between the strongest azimuth peak and a "
        "secondary azimuth peak kept from the same RD cell (default: 3.0 dB).",
    )
    parser.add_argument(
        "--angle-secondary-peak-min-separation-bins",
        type=int,
        default=2,
        help="Minimum azimuth-bin separation required between the strongest peak "
        "and a secondary peak in the same RD cell (default: 2).",
    )
    parser.add_argument(
        "--angle-secondary-peak-min-prominence-db",
        type=float,
        default=3.0,
        help="Minimum valley-to-peak prominence in dB required for a secondary "
        "azimuth peak within the same RD cell (default: 3.0 dB).",
    )
    parser.add_argument(
        "--angle-secondary-peak-min-profile-snr-db",
        type=float,
        default=6.0,
        help="Secondary peak must be this many dB above the azimuth profile "
        "median (noise floor). Rejects sidelobe maxima that are just "
        "local bumps above the background (default: 6.0 dB).",
    )
    parser.add_argument(
        "--angle-secondary-peak-min-valley-depth-db",
        type=float,
        default=3.0,
        help="Valley between strongest and secondary peak must drop by this "
        "many dB below the weaker peak. Ensures peaks are truly "
        "resolved, not part of one broad lobe (default: 3.0 dB).",
    )
    parser.add_argument(
        "--angle-sidelobe-suppression",
        action="store_true",
        help="Enable post-cluster sidelobe suppression (softer rules).",
    )
    parser.add_argument(
        "--vote-fusion",
        action="store_true",
        help="Enable 3-projection vote fusion on stored range-doppler-azimuth cubes.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Minimum votes required out of the R-D, R-Az, and D-Az projections " "(default: 2).",
    )
    parser.add_argument(
        "--vote-raz-threshold-scale",
        type=float,
        default=None,
        help="Optional CFAR threshold scale override for the range-azimuth projection.",
    )
    parser.add_argument(
        "--vote-daz-threshold-scale",
        type=float,
        default=None,
        help="Optional CFAR threshold scale override for the doppler-azimuth projection.",
    )
    # Tracker (per-frame, replaces batch-style M-of-N)
    parser.add_argument(
        "--save-tracking-diagnostics",
        action="store_true",
        help="Record optional candidate assignments and track state/covariance for inspection.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["none", "m_of_n", "kalman", "jpdaf", "cartesian", "cartesian3d"],
        default="none",
        help="Tracker type for per-frame detection filtering (default: none).",
    )
    parser.add_argument(
        "--tracker-m",
        type=int,
        default=2,
        help="Tracker M parameter: required hits to confirm a track (default: 2).",
    )
    parser.add_argument(
        "--tracker-n",
        type=int,
        default=3,
        help="Tracker N parameter: confirmation window in frames (default: 3).",
    )
    parser.add_argument(
        "--tracker-confirmation-mode",
        choices=["latched", "continuous"],
        default="latched",
        help="Keep confirmation until retirement, or recheck M-of-N each update (default: latched).",
    )
    parser.add_argument(
        "--tracker-max-missed-frames",
        type=int,
        default=1,
        help=(
            "Maximum consecutive missed sensing frames before a track is deleted "
            "(default: 1). Increase this for intermittent extended-target detections."
        ),
    )
    parser.add_argument(
        "--tracker-min-lifetime",
        type=int,
        default=0,
        help="Minimum track lifetime (hits) before output. This rejects every shorter "
        "track regardless of physical cause. 0 = use tracker-m (default: 0).",
    )
    parser.add_argument(
        "--tracker-azimuth-gate",
        type=float,
        default=None,
        help="Optional azimuth gate in degrees for tracker association. "
        "When provided, detections farther than this angle from a track's "
        "last azimuth are not associated.",
    )
    parser.add_argument(
        "--tracker-azimuth-cost-scale",
        type=float,
        default=None,
        help="Optional soft azimuth cost scale in degrees for tracker association. "
        "Smaller values make azimuth disagreement more expensive without hard rejection.",
    )
    parser.add_argument(
        "--tracker-range-gate",
        type=float,
        default=1.0,
        help="Range gate in metres for range/velocity trackers (default: 1.0).",
    )
    parser.add_argument(
        "--tracker-velocity-gate",
        type=float,
        default=1.0,
        help="Velocity gate in m/s for tracker association (default: 1.0).",
    )
    parser.add_argument(
        "--tracker-xy-gate",
        type=float,
        default=2.0,
        help="Euclidean distance gate in x,y plane for Cartesian tracker (metres, default: 2.0). "
        "Only used with --tracker cartesian.",
    )
    parser.add_argument(
        "--tracker-acceleration-psd-m2-s3",
        type=float,
        default=None,
        help=(
            "Continuous white-acceleration spectral density used by timestamp-driven "
            "Kalman trackers. Defaults to 0.5 for radial/JPDA and 1.0 for Cartesian."
        ),
    )
    parser.add_argument(
        "--ghost-track-correlation",
        type=float,
        default=None,
        help="Enable heuristic track-correlation candidate pruning. Value is the Pearson "
        "rho threshold (default: None = disabled; example: 0.85). Requires --tracker. "
        "The farther track in a correlated candidate pair is pruned; this does not prove "
        "multipath provenance.",
    )
    parser.add_argument(
        "--ghost-track-correlation-min-overlap",
        type=int,
        default=5,
        help="Minimum overlapping frames required for track correlation computation "
        "(default: 5). Lower values admit shorter-lived candidate pairs but increase "
        "the risk of pruning distinct tracks.",
    )
    # JPDAF-specific parameters
    parser.add_argument(
        "--jpda-pd",
        type=float,
        default=0.9,
        help="JPDAF detection probability Pd (default: 0.9).",
    )
    parser.add_argument(
        "--jpda-beta-fa",
        type=float,
        default=0.01,
        help="JPDAF false alarm spatial density (default: 0.01).",
    )
    parser.add_argument(
        "--jpda-gate-threshold",
        type=float,
        default=9.21,
        help="JPDAF Mahalanobis gate threshold, chi-squared 2 DOF (default: 9.21 = 99%%).",
    )
    parser.add_argument(
        "--jpda-max-joint-nodes",
        type=int,
        default=100_000,
        help="Maximum exact joint-association search nodes per update; exceeding it rejects the update (default: 100000).",
    )
    parser.add_argument(
        "--jpda-birth-confirmation-frames",
        type=int,
        default=3,
        help="Consecutive unclaimed reports before a nearby tentative JPDA birth (at least 2; default: 3).",
    )
    parser.add_argument(
        "--jpda-birth-separation-chi2",
        type=float,
        default=1.0,
        help="Nonnegative R-normalized squared separation for duplicate-birth suppression; zero blocks exact duplicates only (default: 1).",
    )
    # Legacy M-of-N (convenience alias — sets tracker_type=m_of_n)
    parser.add_argument(
        "--m-of-n",
        type=int,
        nargs=2,
        default=None,
        metavar=("M", "N"),
        help="Shorthand for --tracker m_of_n --tracker-m M --tracker-n N.",
    )
    parser.add_argument(
        "--m-of-n-range-tol",
        type=float,
        default=0.5,
        help="Range tolerance for M-of-N matching in metres (default: 0.5).",
    )
    parser.add_argument(
        "--m-of-n-velocity-tol",
        type=float,
        default=0.3,
        help="Velocity tolerance for M-of-N matching in m/s (default: 0.3).",
    )
    # Noise injection
    parser.add_argument(
        "--noise-snr",
        type=float,
        default=None,
        help=(
            "Inject dimensionless target-relative complex AWGN at the given relative "
            "SNR (dB); this is not physical thermal or link-budget noise. "
            "Implies --rebuild-rd."
        ),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="RNG seed for noise injection (default: random).",
    )
    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Evaluate detections against ground truth (TP/FP/FN, Pd, mean false "
            "alarms per evaluated event, RMSE)."
        ),
    )
    parser.add_argument(
        "--range-gate",
        type=float,
        default=DEFAULT_RANGE_GATE_M,
        help="Association range tolerance in metres (default: ±2.0).",
    )
    parser.add_argument(
        "--velocity-gate",
        type=float,
        default=DEFAULT_VELOCITY_GATE_M_S,
        help="Association velocity tolerance in m/s (default: ±2.0).",
    )
    # Detection report
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a JSON detection report with per-frame stats, GT association, and metrics.",
    )
    # Detection threshold sweep
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Run a CFAR threshold sweep and plot metrics for the selected input and truth set.",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=[1.0, 10.0],
        metavar=("MIN", "MAX"),
        help="Threshold sweep range (default: 1.0 10.0).",
    )
    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=20,
        help="Number of threshold sweep steps (default: 20).",
    )
    # Figures
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate summary figures (RD map, detection timeline, evaluation).",
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=FIGURE_CHOICES,
        default=[],
        metavar="NAME",
        help=(
            "Generate only a selected figure. Repeat for multiple figures. "
            "Implies --figures. Choices: " + ", ".join(FIGURE_CHOICES)
        ),
    )
    parser.add_argument(
        "--scene-overlay",
        action="store_true",
        help="Overlay scene walls/floor outline on x/y and 3D tracking figures.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Generate animated GIF of RD map evolution across all frames.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="Frames per second for the GIF animation (default: 10).",
    )
    parser.add_argument(
        "--gif-interp",
        default="nearest",
        help="Interpolation for GIF RD maps: nearest, bilinear, etc. (default: nearest).",
    )
    parser.add_argument(
        "--gif-stride",
        type=int,
        default=1,
        help="Subsample every N-th frame for GIF (default: 1 = all frames). "
        "Use e.g. 64 for 8000-frame scenarios to keep GIF size manageable.",
    )
    parser.add_argument(
        "--gif-per-frame-vmax",
        action="store_true",
        help="Also generate a GIF with per-frame color normalization "
        "(each frame uses its own max power instead of the global max). "
        "Saved alongside the standard GIF with '_perframe' suffix.",
    )
    parser.add_argument(
        "--gif-no-detections",
        action="store_true",
        help="Also generate a GIF without detection markers overlaid. "
        "Saved alongside the standard GIF with '_nodet' suffix.",
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Skip frame processing; generate GIFs from existing results in the output "
        "directory. Requires --gif.",
    )
    parser.add_argument(
        "--gif-3d",
        action="store_true",
        help="Generate 3D surface plot GIF of RD map evolution.",
    )
    parser.add_argument(
        "--micro-doppler",
        action="store_true",
        help="Generate micro-Doppler spectrograms from stored CIR data.",
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip the detection pipeline entirely. Useful with --micro-doppler "
        "to generate spectrograms without running CFAR/NMS/DBSCAN.",
    )
    parser.add_argument(
        "--micro-doppler-range-m",
        type=float,
        default=None,
        help="Target range in meters for single-bin micro-Doppler "
        "(default: auto-select strongest bin).",
    )
    parser.add_argument(
        "--stft-window",
        type=int,
        default=None,
        help="Override STFT window size (CPI) for micro-Doppler spectrograms "
        "(default: fft_size_doppler from stored config). Useful for CPI sweep analysis.",
    )
    parser.add_argument(
        "--vertex-overlay",
        type=str,
        default=None,
        metavar="MESH_DIR",
        help="Overlay ground-truth vertex radial velocities on micro-Doppler "
        "figures. Provide the path to the PLY mesh library directory. "
        "Requires --micro-doppler. The mesh FPS is inferred from the scenario "
        "mesh_update_interval_s or can be set via --vertex-overlay-fps.",
    )
    parser.add_argument(
        "--vertex-overlay-fps",
        type=float,
        default=None,
        help="Mesh frame rate for vertex velocity computation "
        "(default: inferred from 1/mesh_update_interval_s).",
    )
    parser.add_argument(
        "--vertex-overlay-bulk-velocity",
        type=float,
        default=0.0,
        help="Bulk radial velocity (m/s) to add to vertex velocities "
        "(default: 0.0 for stationary targets).",
    )
    parser.add_argument(
        "--save-stage-artifacts",
        action="store_true",
        help=(
            "Save optional CFAR/NMS/SNR/cluster stage artifacts as compressed NPZ sidecars "
            "and embed them in HDF5 outputs when enabled."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load frames and print summary without processing. Exit with code 0.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    if args.recipes is not None:
        if args.output_dir is None:
            parser.error("--recipes requires --output-dir")
        conflicts = [
            name
            for name, value in vars(args).items()
            if name not in {"frames_dir", "output_dir", "recipes", "verbose"}
            and value != parser.get_default(name)
        ]
        if conflicts:
            parser.error(
                "With --recipes, put processing/evaluation options in the recipe file: "
                + ", ".join(conflicts)
            )
        from shared.sensing.experiment import run_recipe_file

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(message)s",
        )
        try:
            run_recipe_file(args.frames_dir, args.recipes, args.output_dir)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
        return
    if args.result_hop_samples is not None and (not args.streaming or args.result_hop_samples < 1):
        parser.error("--result-hop-samples requires --streaming and a positive integer")
    if args.coherent_clutter_mode is not None:
        if args.no_mti and args.coherent_clutter_mode != "none":
            parser.error("--no-mti conflicts with the requested coherent clutter mode")
        args.rebuild_rd = True
    if args.recording_profile is not None and (
        not args.streaming or args.result_hop_samples is None or not args.rebuild_rd
    ):
        parser.error(
            "--recording-profile requires --streaming --result-hop-samples --rebuild-rd "
            "with complete CFR observations"
        )
    if args.figure:
        args.figures = True
    if args.streaming:
        incompatible = _streaming_incompatible_options(args)
        if incompatible:
            parser.error(
                "--streaming is a result-generation-only lane and cannot be combined "
                "with: " + ", ".join(incompatible)
            )

    if args.array_detection_mode == "beam_scan" and (args.rebuild_rd or args.threshold_sweep):
        parser.error(
            "beam_scan uses retained angular products; omit --rebuild-rd and run sweep recipes separately"
        )
    if args.gif_only and args.dry_run:
        parser.error("--gif-only and --dry-run cannot be combined")
    if args.vertex_overlay is not None and not args.micro_doppler:
        parser.error("--vertex-overlay requires --micro-doppler")
    if not args.dry_run and not args.gif_only:
        # Reject unavailable typed processing before suggesting optional
        # packages or opening input/output paths. Installing a clustering
        # dependency cannot make an unimplemented consumer available.
        try:
            validate_typed_processing_config(
                PostProcessingConfig(
                    array_detection_mode=args.array_detection_mode,
                    beam_angle_oversampling=args.beam_angle_oversampling,
                    ghost_filter_azimuth_gate_deg=args.ghost_filter_azimuth_gate,
                    vote_fusion_enabled=args.vote_fusion,
                    angle_peak_detection_enabled=args.angle_peak_detection,
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
    try:
        _preflight_cli_dependencies(args)
    except RuntimeError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        args.frames_dir = _resolve_source_frames_dir(args.frames_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    output_dir: Path | None = None
    if args.gif_only:
        try:
            output_dir = _analysis_output_root(
                args.frames_dir,
                args.output_dir,
                existing=True,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            parser.error(str(exc))
    elif not args.dry_run:
        try:
            output_dir = _analysis_output_root(
                args.frames_dir,
                args.output_dir,
                existing=False,
            )
        except (FileExistsError, OSError, ValueError) as exc:
            parser.error(str(exc))

    # --gif-only: skip frame processing, generate GIFs from existing NPY files
    if args.gif_only:
        assert output_dir is not None
        saved_results = _load_saved_results(output_dir)
        if not saved_results:
            logger.error(
                "No reprocessed results found in %s. "
                "Run the postprocessor first (without --gif-only) to generate results, "
                "then use --gif-only to create GIFs from them.",
                output_dir,
            )
            return
        n_frames = (
            max(int(result.get("_frame_index", idx)) for idx, result in enumerate(saved_results))
            + 1
        )
        logger.info(
            "GIF-only mode: found %d saved RD maps, max index %d",
            len(saved_results),
            n_frames - 1,
        )
        gif_kwargs = dict(
            fps=args.gif_fps,
            interpolation=args.gif_interp,
            stride=args.gif_stride,
        )
        _generate_rd_gif(output_dir, n_frames, all_results=saved_results, **gif_kwargs)
        if args.gif_per_frame_vmax:
            _generate_rd_gif(
                output_dir,
                n_frames,
                all_results=saved_results,
                **gif_kwargs,
                per_frame_vmax=True,
            )
        if args.gif_no_detections:
            _generate_rd_gif(
                output_dir,
                n_frames,
                all_results=saved_results,
                **gif_kwargs,
                show_detections=False,
            )
            if args.gif_per_frame_vmax:
                _generate_rd_gif(
                    output_dir,
                    n_frames,
                    all_results=saved_results,
                    **gif_kwargs,
                    per_frame_vmax=True,
                    show_detections=False,
                )
        if args.gif_3d:
            _generate_rd_gif_3d(output_dir, n_frames, fps=args.gif_fps, all_results=saved_results)
        logger.info("GIF-only mode complete.")
        return

    # --- Vertex velocity overlay (shared across micro-Doppler call sites) ---
    vel_overlay: dict[str, Any] | None = None
    if args.vertex_overlay is not None:
        mesh_dir = Path(args.vertex_overlay)
        if not mesh_dir.is_dir():
            logger.error("--vertex-overlay directory does not exist: %s", mesh_dir)
            sys.exit(1)
        # Infer mesh FPS from CLI or sensing config
        mesh_fps = args.vertex_overlay_fps
        if mesh_fps is None:
            # Try to read mesh_update_interval_s from first frame's config
            try:
                probe_frames = _load_frames_sensing(args.frames_dir)
                if probe_frames:
                    probe_cfg = probe_frames[0].get("_sensing_json") or probe_frames[0].get(
                        "config", {}
                    )
                    mui = probe_cfg.get("mesh_update_interval_s")
                    if mui and float(mui) > 0:
                        mesh_fps = 1.0 / float(mui)
            except (FileNotFoundError, KeyError, ValueError):
                pass
        if mesh_fps is None:
            mesh_fps = 120.0
            logger.warning(
                "Could not infer mesh FPS; using default %.0f. "
                "Set --vertex-overlay-fps explicitly.",
                mesh_fps,
            )
        # Infer radar/target positions from scenario config
        radar_pos = np.array([0.0, 0.0, 1.5])
        target_pos = np.array([0.0, 5.0, 0.9])
        try:
            probe_frames = _load_frames_sensing(args.frames_dir)
            if not probe_frames:
                raise ValueError("Vertex overlay requires a native schema-v4 sensing frame")
            probe_sensing = _typed_sensing_payload(probe_frames[0])
            if probe_sensing is None:
                raise ValueError("Vertex overlay requires native schema-v4 sensing metadata")
            probe_cfg = _resolve_native_v4_config(probe_sensing)
            prf_hz_overlay = float(probe_cfg["prf_hz"])
            carrier_hz_overlay = float(probe_cfg["carrier_freq_hz"])
        except (FileNotFoundError, KeyError, ValueError) as exc:
            parser.error(str(exc))
        logger.info(
            "Computing vertex velocity overlay from %s (%.0f fps, bulk=%.1f m/s)",
            mesh_dir,
            mesh_fps,
            args.vertex_overlay_bulk_velocity,
        )
        vel_overlay = compute_vertex_velocity_overlay(
            mesh_dir,
            mesh_fps=mesh_fps,
            radar_pos=radar_pos,
            target_pos=target_pos,
            bulk_velocity=args.vertex_overlay_bulk_velocity,
            carrier_hz=carrier_hz_overlay,
            prf_hz=prf_hz_overlay,
        )
        logger.info(
            "Vertex overlay: %d frames, v_min=[%.1f, %.1f], v_max=[%.1f, %.1f] m/s",
            len(vel_overlay["time_s"]),
            vel_overlay["v_min"].min(),
            vel_overlay["v_min"].max(),
            vel_overlay["v_max"].min(),
            vel_overlay["v_max"].max(),
        )

    # --noise-snr implies --rebuild-rd (noise is injected on the STM)
    noise_enabled = args.noise_snr is not None
    if noise_enabled and not args.rebuild_rd:
        logger.info("--noise-snr implies --rebuild-rd; enabling RD map rebuild")
        args.rebuild_rd = True

    # Warn about --no-mti without --rebuild-rd or --micro-doppler
    if args.no_mti and not args.rebuild_rd and not args.micro_doppler:
        logger.warning(
            "--no-mti has no effect without --rebuild-rd or --micro-doppler: "
            "MTI changes require rebuilding the RD map or micro-Doppler spectrogram"
        )

    # --figures and --report imply --evaluate when GT is available
    if (args.figures or args.report) and not args.evaluate:
        args.evaluate = True

    m_of_n_m = args.m_of_n[0] if args.m_of_n else None
    m_of_n_n = args.m_of_n[1] if args.m_of_n else None

    logger.info("Loading frames from %s", args.frames_dir)
    frames: list[dict[str, Any]] = []
    streaming_scan: _StreamingSourceScan | None = None
    try:
        if args.streaming:
            args._window_selection = None
            if args.result_hop_samples is not None:
                from shared.sensing.window_selection import select_saved_windows

                args._window_selection = select_saved_windows(
                    args.frames_dir, args.result_hop_samples
                )
            if args._window_selection is None:
                streaming_scan = _scan_streaming_source(args.frames_dir)
            else:
                streaming_scan = _scan_streaming_source(
                    args.frames_dir, frame_ids=args._window_selection.frame_ids
                )
            stored_range_windows = set(streaming_scan.range_windows)
            stored_doppler_windows = set(streaming_scan.doppler_windows)
            source_frame_count = streaming_scan.frame_count
            if args.recording_profile is not None or args._window_selection is not None:
                selected_ids = (
                    None if args._window_selection is None else args._window_selection.frame_ids
                )
                with closing(
                    _iter_frames_sensing(args.frames_dir, frame_ids=selected_ids)
                ) as iterator:
                    for index, frame in enumerate(iterator):
                        if args._window_selection is not None:
                            args._window_selection.bind(frame, index)
                            if (
                                _select_typed_detection_input(
                                    frame, rebuild_rd=bool(args.rebuild_rd)
                                )
                                is None
                            ):
                                raise ValueError(
                                    "A selected complete window has no requested processing input"
                                )
                        if args.recording_profile is not None:
                            select_frequency_observation(_typed_sensing_payload(frame))
        else:
            frames = _load_frames_sensing(args.frames_dir)
            stored_windows = [_native_v4_frame_windows(frame) for frame in frames]
            stored_range_windows = {windows[0] for windows in stored_windows}
            stored_doppler_windows = {windows[1] for windows in stored_windows}
            source_frame_count = len(frames)
    except ValueError as exc:
        parser.error(str(exc))
    if source_frame_count == 0:
        parser.error("No native schema-v4 sensing frames were found")

    # Validate the complete persisted execution envelope before choosing any
    # processing behavior or reserving output. Windows are inherited from the
    # exact source recipe unless the user supplied an explicit override.
    if args.window_range is None and len(stored_range_windows) != 1:
        parser.error(
            "Source frames use multiple stored range windows; pass --window-range "
            "to choose an explicit reprocessing override"
        )
    if args.window_doppler is None and len(stored_doppler_windows) != 1:
        parser.error(
            "Source frames use multiple stored Doppler windows; pass --window-doppler "
            "to choose an explicit reprocessing override"
        )
    effective_window_range = (
        str(args.window_range)
        if args.window_range is not None
        else next(iter(stored_range_windows))
    )
    effective_window_doppler = (
        str(args.window_doppler)
        if args.window_doppler is not None
        else next(iter(stored_doppler_windows))
    )

    response_override = None
    if args.frequency_window is not None or args.range_ifft_size is not None:
        if args.streaming or args.threshold_sweep:
            parser.error(
                "CFR rewindowing requires a single retained processing run; replay first, then stream/sweep its products"
            )
        if noise_enabled:
            parser.error("CFR replay reuses frequency noise; --noise-snr conflicts")
        try:
            first_sensing = _typed_sensing_payload(frames[0])
            stored_response = RangeResponseConfig.parse(
                first_sensing["config_resolved"].get("range_response")
            )
            response_fields = stored_response.to_dict()
            if args.frequency_window is not None:
                response_fields["frequency_window"] = args.frequency_window
            response_override = RangeResponseConfig.parse(response_fields)
            if response_override.model != "cfr_ifft":
                raise ValueError(
                    "Cannot reconstruct fractional-delay CFR from point-bin samples; acquire cfr_ifft with upstream retention"
                )
            for frame in frames:
                source = _typed_sensing_payload(frame)
                if _unprocessed_acquisition_temporal(frame) is not None:
                    continue
                select_frequency_observation(source)
                response_override.validate_processing(
                    bandwidth=source["config_resolved"]["bandwidth"],
                    fft_size=args.range_ifft_size or source["config_resolved"]["fft_size_range"],
                    window_range=effective_window_range,
                )
        except (ValueError, TypeError, KeyError) as exc:
            parser.error(str(exc))

    config = PostProcessingConfig(
        range_response=response_override,
        range_ifft_size=args.range_ifft_size,
        window_range=effective_window_range,
        window_doppler=effective_window_doppler,
        mti_enabled=not args.no_mti,
        coherent_clutter_mode=args.coherent_clutter_mode,
        cfar_type=args.cfar_type,
        cfar_threshold_scale=args.cfar_threshold,
        array_detection_mode=args.array_detection_mode,
        beam_angle_oversampling=args.beam_angle_oversampling,
        cfar_min_snr_db=args.cfar_min_snr,
        cfar_max_dynamic_range_db=args.cfar_max_dynamic_range,
        cfar_guard_cells_range=args.cfar_guard_range,
        cfar_guard_cells_doppler=args.cfar_guard_doppler,
        cfar_ref_cells_range=args.cfar_ref_range,
        cfar_ref_cells_doppler=args.cfar_ref_doppler,
        cfar_edge_policy=args.cfar_edge_policy,
        pre_cfar_doppler_edge_guard_bins=args.pre_cfar_doppler_edge_guard_bins,
        pre_cfar_doppler_dc_guard_bins=args.pre_cfar_doppler_dc_guard_bins,
        cluster_range_gate_m=args.cluster_range_gate,
        cluster_velocity_gate_m_s=args.cluster_velocity_gate,
        min_detection_range_m=args.min_range,
        max_detection_range_m=args.max_range,
        ghost_filter_velocity_gate_m_s=args.ghost_filter_velocity_gate,
        ghost_filter_azimuth_gate_deg=args.ghost_filter_azimuth_gate,
        angle_peak_detection_enabled=args.angle_peak_detection,
        angle_max_peaks_per_rd_cell=args.angle_max_peaks_per_rd,
        angle_peak_relative_floor_db=args.angle_peak_relative_floor_db,
        angle_edge_exclusion_bins=args.angle_edge_exclusion_bins,
        angle_secondary_peak_max_gap_db=args.angle_secondary_peak_max_gap_db,
        angle_secondary_peak_min_separation_bins=args.angle_secondary_peak_min_separation_bins,
        angle_secondary_peak_min_prominence_db=args.angle_secondary_peak_min_prominence_db,
        angle_secondary_peak_min_profile_snr_db=args.angle_secondary_peak_min_profile_snr_db,
        angle_secondary_peak_min_valley_depth_db=args.angle_secondary_peak_min_valley_depth_db,
        angle_sidelobe_suppression_enabled=args.angle_sidelobe_suppression,
        vote_fusion_enabled=args.vote_fusion,
        vote_min_votes=args.min_votes,
        vote_raz_threshold_scale=args.vote_raz_threshold_scale,
        vote_daz_threshold_scale=args.vote_daz_threshold_scale,
        tracker_type=args.tracker,
        tracker_m=args.tracker_m,
        tracker_n=args.tracker_n,
        tracker_confirmation_mode=args.tracker_confirmation_mode,
        tracker_max_missed_frames=args.tracker_max_missed_frames,
        tracker_range_gate_m=args.tracker_range_gate,
        tracker_velocity_gate_m_s=args.tracker_velocity_gate,
        tracker_azimuth_gate_deg=args.tracker_azimuth_gate,
        tracker_azimuth_cost_scale_deg=args.tracker_azimuth_cost_scale,
        tracker_min_lifetime=args.tracker_min_lifetime,
        tracker_xy_gate_m=args.tracker_xy_gate,
        tracker_acceleration_psd_m2_s3=args.tracker_acceleration_psd_m2_s3,
        ghost_track_correlation_threshold=args.ghost_track_correlation,
        ghost_track_correlation_min_overlap=args.ghost_track_correlation_min_overlap,
        jpda_pd=args.jpda_pd,
        jpda_beta_fa=args.jpda_beta_fa,
        jpda_gate_threshold=args.jpda_gate_threshold,
        jpda_max_joint_nodes=args.jpda_max_joint_nodes,
        jpda_birth_confirmation_frames=args.jpda_birth_confirmation_frames,
        jpda_birth_separation_chi2=args.jpda_birth_separation_chi2,
        m_of_n_m=m_of_n_m,
        m_of_n_n=m_of_n_n,
        m_of_n_range_tol_m=args.m_of_n_range_tol,
        m_of_n_velocity_tol_m_s=args.m_of_n_velocity_tol,
        noise_enabled=noise_enabled,
        noise_snr_db=args.noise_snr,
        noise_rng_seed=args.noise_seed,
        skip_detection=args.skip_detection,
        include_stage_artifacts=args.save_stage_artifacts,
        include_tracking_diagnostics=args.save_tracking_diagnostics,
    )
    try:
        validate_typed_processing_config(config)
    except ValueError as exc:
        parser.error(str(exc))

    pp = SensingPostProcessor(config)

    if not args.streaming:
        for frame in frames:
            if not _has_result_bound_truth(frame):
                frame.pop("_gt", None)
    logger.info("Found %d frames with sensing data", source_frame_count)

    # --skip-detection with --micro-doppler: generate spectrograms only
    if args.skip_detection and args.micro_doppler and not args.dry_run:
        assert output_dir is not None
        source_sequence = _load_cir_sequence(args.frames_dir)
        if not source_sequence[0]:
            parser.error(
                "No usable CIR or slow-time samples were found for micro-Doppler; "
                "--output-dir was not created"
            )
        try:
            _claim_analysis_root(output_dir)
        except FileExistsError as exc:
            parser.error(f"Could not claim new --output-dir: {exc}")
        logger.info("Skip-detection mode: generating micro-Doppler only")
        _generate_micro_doppler_figures(
            args.frames_dir,
            output_dir,
            config=config,
            target_range_m=args.micro_doppler_range_m,
            velocity_overlay=vel_overlay,
            stft_window_override=args.stft_window,
            source_sequence=source_sequence,
        )
        logger.info("Skip-detection mode complete. Output in %s", output_dir)
        return

    # --dry-run: print summary and exit
    if args.dry_run:
        has_stm = sum(1 for frame in frames if _select_typed_slow_time_input(frame) is not None)
        has_rd = sum(1 for frame in frames if _select_typed_range_doppler_input(frame) is not None)
        has_cir = sum(1 for f in frames if "cir" in f)
        strategy = "unknown"
        if frames:
            sj = frames[0].get("_sensing_json", {})
            strategy = sj.get("processing_strategy", strategy)
        logger.info(
            "Dry run: %d frames, %d with STM, %d with RD, %d with CIR, strategy=%s",
            len(frames),
            has_stm,
            has_rd,
            has_cir,
            strategy,
        )
        return

    # H5: A requested rebuild is a strict typed-product route, not a fallback hint.
    rebuild_has_stm = (
        bool(streaming_scan and streaming_scan.has_slow_time)
        if args.streaming
        else any(_has_complex_rebuild_input(frame) for frame in frames)
    )
    if args.rebuild_rd and not rebuild_has_stm:
        parser.error(
            "--rebuild-rd requires retained typed delay-domain or frequency complex observations"
        )

    # H1: Auto-rebuild RD when window function differs from stored config
    if not args.rebuild_rd:
        has_stm = (
            bool(streaming_scan and streaming_scan.has_slow_time)
            if args.streaming
            else any(_has_complex_rebuild_input(frame) for frame in frames)
        )
        windows_changed = any(
            stored_range != config.window_range for stored_range in stored_range_windows
        ) or any(
            stored_doppler != config.window_doppler for stored_doppler in stored_doppler_windows
        )
        if has_stm and windows_changed:
            args.rebuild_rd = True
            logger.info(
                "Window function changed (stored range=%s, effective range=%s; "
                "stored Doppler=%s, effective Doppler=%s); "
                "auto-enabling --rebuild-rd",
                sorted(stored_range_windows),
                config.window_range,
                sorted(stored_doppler_windows),
                config.window_doppler,
            )

    # --- Detect per-pulse data and group into CPIs if needed ---
    # If the first frame has a single-pulse STM (1, N), group consecutive
    # frames using the exact persisted acquisition plan.
    per_pulse_mode = False
    acquisition_plan: Mapping[str, Any] | None = None
    if args.streaming and streaming_scan is not None and streaming_scan.has_per_pulse_slow_time:
        parser.error(
            "--streaming does not support per-pulse slow-time CPI grouping; "
            "run the default retained mode so grouping preserves exact overlap semantics"
        )
    if frames:
        first_slow_time = _select_typed_slow_time_input(frames[0])
        if first_slow_time is not None and first_slow_time.values.shape[2] == 1:
            per_pulse_mode = True
            first_sensing = _typed_sensing_payload(frames[0])
            if first_sensing is None:
                raise ValueError("Per-pulse input has no schema-v4 sensing payload")
            first_config = _resolve_native_v4_config(first_sensing)
            acquisition_plan = first_config["acquisition"]
            if acquisition_plan.get("sample_source") not in {
                "independent_solves",
                "measured_pulse_stream",
            }:
                raise ValueError(
                    "Single-pulse slow-time rows require a resolved independent-solve "
                    "or measured-pulse acquisition"
                )
            logger.info(
                "Per-pulse mode detected (STM shape %s). CPI=%d, hop=%d, partial=%s.",
                first_slow_time.values.shape,
                int(acquisition_plan["cpi_length_samples"]),
                int(acquisition_plan["cpi_hop_samples"]),
                acquisition_plan["partial_window_policy"],
            )

    if per_pulse_mode:
        assert acquisition_plan is not None
        grouped_frames = _group_per_pulse_frames(
            frames,
            acquisition_plan=acquisition_plan,
        )
        n_complete = sum(
            _unprocessed_acquisition_temporal(frame) is None for frame in grouped_frames
        )
        logger.info(
            "Grouped %d pulses into %d CPIs (%d complete, cpi_length=%d)",
            len(frames),
            len(grouped_frames),
            n_complete,
            int(acquisition_plan["cpi_length_samples"]),
        )
        frames = grouped_frames

    if args.streaming:
        assert streaming_scan is not None
        first_streaming_source_step = (
            streaming_scan.first_rebuild_source_step
            if args.rebuild_rd
            else streaming_scan.first_default_source_step
        )
        if first_streaming_source_step is None:
            parser.error(
                "No usable sensing frames contain the requested schema-v4 typed input; "
                "--output-dir was not created"
            )
        assert output_dir is not None
        try:
            _run_streaming_result_generation(
                args=args,
                output_dir=output_dir,
                config=config,
                postprocessor=pp,
                total_sensing_frames=streaming_scan.frame_count,
                first_source_step=first_streaming_source_step,
            )
        except FileExistsError as exc:
            parser.error(f"Could not claim new --output-dir: {exc}")
        return

    usable_frames = [frame for frame in frames if _has_detection_input(frame)]
    if not usable_frames:
        parser.error(
            "No usable sensing frames contain a schema-v4 typed slow-time or "
            "range-Doppler product; --output-dir was not created"
        )

    # Derived output copies the full source frame around each replacement
    # sensing block. Validate that read path before reserving the output name.
    source_provider = Hdf5Provider(
        args.frames_dir.parent,
        frames_subdir=args.frames_dir.name,
    )
    first_usable = usable_frames[0]
    first_source_step = _source_scene_frame_step(first_usable)
    try:
        first_source_frame = source_provider.load_frame(first_source_step)
    except BaseException:
        source_provider.close()
        raise

    assert output_dir is not None
    try:
        _claim_analysis_root(output_dir)
    except FileExistsError as exc:
        source_provider.close()
        parser.error(f"Could not claim new --output-dir: {exc}")
    except BaseException:
        source_provider.close()
        raise

    # --- Create columnar writer for HDF5 output ---
    try:
        columnar_writer = _ColumnarResultWriter(output_dir, chunk_size=100)
    except BaseException:
        source_provider.close()
        _remove_empty_claimed_analysis_root(output_dir)
        raise

    # Process all frames
    all_results: list[dict[str, Any]] = []
    processed = 0
    processing_run_id = str(uuid4())
    tracker_stream_identity: tuple[str, str] | None = None
    try:
        for idx, frame in enumerate(frames):
            frame_step = int(frame.get("_frame_index", idx))
            if not _has_detection_input(frame):
                logger.debug("Frame %d: no suitable data found, skipping", idx)
                all_results.append({})
                continue

            source_frame = None
            if source_provider is not None:
                # Copy the explicitly bound source scene under the derived
                # output ID; product metadata remains sourced from ``frame``.
                src_step = _source_scene_frame_step(frame)
                if src_step == first_source_step and first_source_frame is not None:
                    source_frame = first_source_frame
                    first_source_frame = None
                else:
                    source_frame = source_provider.load_frame(src_step)
            source_sensing = _source_sensing_payload(frame)
            if not source_sensing and source_frame is not None:
                source_sensing = deepcopy(dict(source_frame.sensing or {}))
            meta = source_sensing or frame.get("_sensing_json") or frame.get("config", frame)
            incomplete_temporal = _unprocessed_acquisition_temporal(frame)
            if incomplete_temporal is not None:
                if source_frame is not None:
                    columnar_writer.add_incomplete_acquisition(
                        temporal=incomplete_temporal,
                        frame_index=frame_step,
                        source_frame=source_frame,
                        source_sensing=source_sensing,
                        postprocessing_config=config,
                    )
                logger.info(
                    "Frame %d: retained unprocessed acquisition without coherent result",
                    frame_step,
                )
                all_results.append({})
                continue

            typed_input = _select_typed_detection_input(
                frame,
                rebuild_rd=bool(args.rebuild_rd),
                rebuild_cfr=config.range_response is not None or config.range_ifft_size is not None,
            )
            if typed_input is None:
                logger.debug("Frame %d: no typed detection product found, skipping", idx)
                all_results.append({})
                continue
            input_kind = typed_input.data_key

            source_display_binding = _source_display_binding(frame)
            if _tracking_enabled(config):
                current_stream_identity = _temporal_stream_identity(frame)
                if tracker_stream_identity is None:
                    tracker_stream_identity = current_stream_identity
                elif current_stream_identity != tracker_stream_identity:
                    raise ValueError(
                        "Tracking cannot share state across sensing clock or sensor-pair changes: "
                        f"expected {tracker_stream_identity}, got {current_stream_identity}"
                    )

            temporal = _build_reprocessed_temporal(
                frame,
                processing_run_id=processing_run_id,
                postprocessing_config=config,
                input_kind=input_kind,
            )
            update_time_s = float(temporal["result"]["reference_time_s"])
            if typed_input.kind == "baseband_frequency_observation":
                result = pp.process_from_frequency_observation(meta, update_time_s=update_time_s)
            elif typed_input.kind == "delay_domain_slow_time":
                assert typed_input.slow_time_axis_s is not None
                result = pp.process_from_slow_time_matrix(
                    _combine_typed_slow_time(typed_input),
                    meta,
                    range_axis_m=typed_input.range_axis_m,
                    slow_time_axis_s=typed_input.slow_time_axis_s,
                    update_time_s=update_time_s,
                )
            else:
                assert typed_input.doppler_axis_hz is not None
                rd_power, rd_cube = _typed_range_doppler_arrays(typed_input)
                result = pp.process_from_rd_map(
                    rd_power,
                    meta,
                    range_axis_m=typed_input.range_axis_m,
                    doppler_axis_hz=typed_input.doppler_axis_hz,
                    rd_cube=rd_cube,
                    update_time_s=update_time_s,
                )
            result["processing_recipe_resolved"] = _build_effective_processing_recipe(
                frame,
                postprocessing_config=config,
                input_kind=input_kind,
            )
            result["temporal"] = temporal
            result["_frame_index"] = frame_step
            result["_input_kind"] = input_kind

            # Columnar HDF5 output
            if source_frame is not None:
                columnar_writer.add_frame(
                    result,
                    frame_index=frame_step,
                    source_frame=source_frame,
                    source_sensing=source_sensing,
                    postprocessing_config=config,
                    input_kind=input_kind,
                    source_display_binding=source_display_binding,
                )

            if args.save_stage_artifacts:
                _save_stage_artifacts(output_dir, result, frame_index=frame_step)
            n_det = len(result.get("detections", []))
            logger.info("Frame %d: %d detections", frame_step, n_det)
            all_results.append(result)
            processed += 1
    except BaseException:
        columnar_writer.abort()
        _remove_empty_claimed_analysis_root(output_dir)
        raise
    finally:
        if source_provider is not None:
            source_provider.close()

    # Finalize columnar output.  Tracking (M-of-N or Kalman) is now applied
    # per-frame inside _detect() via the post_detection_filter callback,
    # so results already contain tracker-filtered detections.
    try:
        columnar_writer.finalize()
        from shared.sensing.result_scenario import write_result_scenario

        write_result_scenario(args.frames_dir, output_dir)
    except BaseException:
        columnar_writer.abort()
        _remove_empty_claimed_analysis_root(output_dir)
        raise

    # --- Track-correlation ghost pruning (batch, requires full track history) ---
    pruned_ghost_track_ids: set[int] = set()
    if config.ghost_track_correlation_threshold is not None and processed > 0:
        logger.info(
            "Running track-correlation ghost pruning (rho > %.2f)",
            config.ghost_track_correlation_threshold,
        )
        all_results, corr_meta = prune_correlated_ghost_tracks(
            all_results,
            correlation_threshold=config.ghost_track_correlation_threshold,
            min_overlap_frames=config.ghost_track_correlation_min_overlap,
        )
        pruned_ghost_track_ids = set(corr_meta.get("ghost_tracks", []))
        if pruned_ghost_track_ids:
            logger.info(
                "Ghost tracks pruned: %s (%d detections removed)",
                sorted(pruned_ghost_track_ids),
                corr_meta["pruned"],
            )
        else:
            logger.info("No correlated ghost tracks found")

    logger.info(
        "Done. Processed %d/%d frames. Results in %s",
        processed,
        len(frames),
        output_dir,
    )

    # Write summary config
    summary_path = output_dir / "postprocessing_config.json"
    with open(summary_path, "w") as f:
        json.dump(processing_config_dict(config), f, indent=2)
    logger.info("Config saved to %s", summary_path)

    # --- Evaluation ---
    eval_result: ScenarioEvaluationResult | None = None
    mot_result: MOTMetrics | None = None
    evaluation_join: _EvaluationEventJoin | None = None
    per_event_gt: list[tuple[np.ndarray, np.ndarray]] = []

    if args.evaluate and processed > 0:
        evaluation_join = _join_evaluation_events(frames, all_results)
        if evaluation_join.missing_result_keys:
            logger.warning(
                "Evaluation has %d truth/source events without a result; "
                "they are evaluated as empty detections",
                len(evaluation_join.missing_result_keys),
            )
        if evaluation_join.extra_result_keys:
            logger.warning(
                "Evaluation has %d result events without a source/truth event; "
                "they are evaluated as empty truth",
                len(evaluation_join.extra_result_keys),
            )
        if evaluation_join.skipped_invalid_truth_keys:
            logger.warning(
                "Evaluation skipped %d events whose active-target velocity truth is unavailable; "
                "their detections are not counted as false alarms",
                len(evaluation_join.skipped_invalid_truth_keys),
            )

        per_event_gt = [(event.gt_ranges, event.gt_velocities) for event in evaluation_join.events]
        per_event_mot_gt = [
            (event.target_ids, event.gt_ranges, event.gt_velocities)
            for event in evaluation_join.events
        ]
        per_event_dets = []
        per_event_mot_dets: list[list[dict[str, Any]]] = []
        for event in evaluation_join.events:
            detections = list(event.detections)
            det_ranges = np.asarray(
                [detection["range_m"] for detection in detections],
                dtype=np.float64,
            )
            det_velocities = np.asarray(
                [detection["velocity_m_s"] for detection in detections],
                dtype=np.float64,
            )
            per_event_dets.append((det_ranges, det_velocities))
            per_event_mot_dets.append(detections)

        has_gt = any(gt_r.size > 0 for gt_r, _ in per_event_gt)
        if has_gt:
            eval_result = evaluate_scenario(
                per_event_gt,
                per_event_dets,
                range_gate=args.range_gate,
                velocity_gate=args.velocity_gate,
                per_event_velocity_valid=(
                    [
                        (
                            event.gt_velocity_valid
                            if event.gt_velocity_valid is not None
                            else [True] * len(event.gt_ranges)
                        )
                        for event in evaluation_join.events
                    ]
                    if any(event.gt_velocity_valid is not None for event in evaluation_join.events)
                    else None
                ),
            )
            pd_text = (
                "undefined"
                if eval_result.cumulative_pd is None
                else f"{eval_result.cumulative_pd:.1%}"
            )
            range_rmse_text = (
                "undefined"
                if eval_result.cumulative_range_rmse is None
                else f"{eval_result.cumulative_range_rmse:.2f} m"
            )
            velocity_rmse_text = (
                "undefined"
                if eval_result.cumulative_velocity_rmse is None
                else f"{eval_result.cumulative_velocity_rmse:.2f} m/s"
            )
            logger.info(
                "Evaluation: Pd=%s, mean false alarms/evaluated event=%s, "
                "Range RMSE=%s, Vel RMSE=%s "
                "(TP=%d, FP=%d, FN=%d, %d evaluated events)",
                pd_text,
                eval_result.mean_false_alarms_per_evaluated_event,
                range_rmse_text,
                velocity_rmse_text,
                eval_result.total_tp,
                eval_result.total_fp,
                eval_result.total_fn,
                eval_result.evaluated_event_count,
            )

            # Save evaluation results
            eval_path = output_dir / "evaluation_results.json"
            eval_dict = asdict(eval_result)
            eval_dict["event_join"] = _evaluation_join_metadata(evaluation_join)

            # Compute MOT metrics when tracker is active
            velocity_complete = all(
                event.gt_velocity_valid is None or all(event.gt_velocity_valid)
                for event in evaluation_join.events
            )
            if config.tracker_type != "none" and velocity_complete:
                mot_result = compute_mot_metrics(
                    per_event_mot_gt,
                    per_event_mot_dets,
                    range_gate=args.range_gate,
                    velocity_gate=args.velocity_gate,
                )
                mota_text = "undefined" if mot_result.mota is None else f"{mot_result.mota:.1%}"
                motp_text = "undefined" if mot_result.motp is None else f"{mot_result.motp:.3f} m"
                logger.info(
                    "MOT: MOTA=%s, MOTP=%s, IDSW=%d, Frag=%s",
                    mota_text,
                    motp_text,
                    mot_result.total_id_switches,
                    {k: v for k, v in mot_result.track_fragmentation.items()},
                )
                eval_dict["mot_metrics"] = asdict(mot_result)
            else:
                mot_result = None
                if config.tracker_type != "none":
                    eval_dict["mot_metrics_unavailable_reason"] = (
                        "Instantaneous velocity unavailable"
                    )

            with open(eval_path, "w") as f:
                json.dump(eval_dict, f, indent=2, default=_json_default)
            logger.info("Evaluation results saved to %s", eval_path)
        else:
            logger.warning("No ground truth found in frames; skipping evaluation")
            mot_result = None

    # --- Detection report ---
    if args.report and processed > 0:
        det_counts = []
        for result in all_results:
            dets = result.get("detections", [])
            det_counts.append(len(dets))
        det_counts_arr = np.array(det_counts, dtype=int)

        # Count GT targets (max across frames)
        n_gt_targets = 0
        frames_with_gt = 0
        for frame in frames:
            gt = frame.get("_gt")
            if gt is not None:
                n_targets = len(gt.get("target_ranges", []))
                n_gt_targets = max(n_gt_targets, n_targets)
                if n_targets > 0:
                    frames_with_gt += 1

        report: dict[str, Any] = {
            "scenario": str(args.frames_dir),
            "total_frames": len(frames),
            "frames_with_sensing": processed,
            "frames_with_detections": int(np.sum(det_counts_arr > 0)),
            "detection_statistics": {
                "total_detections": int(det_counts_arr.sum()),
                "mean_per_frame": (
                    round(float(det_counts_arr.mean()), 2) if len(det_counts_arr) else 0
                ),
                "median_per_frame": int(np.median(det_counts_arr)) if len(det_counts_arr) else 0,
                "max_per_frame": int(det_counts_arr.max()) if len(det_counts_arr) else 0,
                "min_per_frame": int(det_counts_arr.min()) if len(det_counts_arr) else 0,
                "distribution": {
                    str(c): int(np.sum(det_counts_arr == c))
                    for c in sorted(set(det_counts_arr.tolist()))
                },
            },
            "ground_truth": {
                "num_targets": n_gt_targets,
                "frames_with_gt": frames_with_gt,
            },
            "config_used": {
                "cfar_type": config.cfar_type,
                "cfar_threshold_scale": config.cfar_threshold_scale,
                "cfar_min_snr_db": config.cfar_min_snr_db,
                "cfar_edge_policy": config.cfar_edge_policy,
                "cluster_range_gate_m": config.cluster_range_gate_m,
                "cluster_velocity_gate_m_s": config.cluster_velocity_gate_m_s,
                "ghost_filter_velocity_gate_m_s": config.ghost_filter_velocity_gate_m_s,
                "ghost_filter_azimuth_gate_deg": config.ghost_filter_azimuth_gate_deg,
                "angle_peak_detection_enabled": config.angle_peak_detection_enabled,
                "angle_max_peaks_per_rd_cell": config.angle_max_peaks_per_rd_cell,
                "angle_peak_relative_floor_db": config.angle_peak_relative_floor_db,
                "angle_edge_exclusion_bins": config.angle_edge_exclusion_bins,
                "angle_secondary_peak_max_gap_db": config.angle_secondary_peak_max_gap_db,
                "angle_secondary_peak_min_separation_bins": config.angle_secondary_peak_min_separation_bins,
                "angle_secondary_peak_min_prominence_db": config.angle_secondary_peak_min_prominence_db,
                "angle_sidelobe_suppression_enabled": config.angle_sidelobe_suppression_enabled,
                "vote_fusion_enabled": config.vote_fusion_enabled,
                "vote_min_votes": config.vote_min_votes,
                "vote_raz_threshold_scale": config.vote_raz_threshold_scale,
                "vote_daz_threshold_scale": config.vote_daz_threshold_scale,
                "mti_enabled": config.mti_enabled,
                "tracker_type": config.tracker_type,
                "tracker_m": config.tracker_m,
                "tracker_n": config.tracker_n,
                "tracker_confirmation_mode": config.tracker_confirmation_mode,
                "tracker_max_missed_frames": config.tracker_max_missed_frames,
                "tracker_azimuth_gate_deg": config.tracker_azimuth_gate_deg,
                "tracker_azimuth_cost_scale_deg": config.tracker_azimuth_cost_scale_deg,
                "tracker_min_lifetime": config.tracker_min_lifetime,
                "tracker_xy_gate_m": config.tracker_xy_gate_m,
                "jpda_max_joint_nodes": config.jpda_max_joint_nodes,
                "jpda_birth_confirmation_frames": config.jpda_birth_confirmation_frames,
                "jpda_birth_separation_chi2": config.jpda_birth_separation_chi2,
            },
        }

        if eval_result is not None:
            miss_breakdown = {
                "missing_result": 0,
                "no_detection_present": 0,
                "detection_present_but_unmatched": 0,
            }
            assert evaluation_join is not None
            for event in evaluation_join.events:
                if event.gt_ranges.size == 0:
                    continue
                det_ranges = np.asarray(
                    [detection["range_m"] for detection in event.detections],
                    dtype=np.float64,
                )
                det_velocities = np.asarray(
                    [detection["velocity_m_s"] for detection in event.detections],
                    dtype=np.float64,
                )
                association = associate_detections_to_gt(
                    event.gt_ranges,
                    event.gt_velocities,
                    det_ranges,
                    det_velocities,
                    range_gate=args.range_gate,
                    velocity_gate=args.velocity_gate,
                )
                unmatched_gt = association.false_negatives
                if unmatched_gt <= 0:
                    continue
                if not event.result_present:
                    miss_breakdown["missing_result"] += int(unmatched_gt)
                elif det_ranges.size == 0:
                    miss_breakdown["no_detection_present"] += int(unmatched_gt)
                else:
                    miss_breakdown["detection_present_but_unmatched"] += int(unmatched_gt)

            precision_denominator = eval_result.total_tp + eval_result.total_fp
            report["association"] = {
                "association_protocol": eval_result.association_protocol,
                "range_gate_m": args.range_gate,
                "velocity_gate_m_s": args.velocity_gate,
                "evaluated_event_count": eval_result.evaluated_event_count,
                "target_opportunity_count": eval_result.target_opportunity_count,
                "matched_pair_count": eval_result.matched_pair_count,
                "true_positives": eval_result.total_tp,
                "false_positives": eval_result.total_fp,
                "false_negatives": eval_result.total_fn,
                "precision": (
                    round(eval_result.total_tp / precision_denominator, 4)
                    if precision_denominator > 0
                    else None
                ),
                "detection_probability": (
                    round(eval_result.cumulative_pd, 4)
                    if eval_result.cumulative_pd is not None
                    else None
                ),
                "mean_false_alarms_per_evaluated_event": (
                    round(eval_result.mean_false_alarms_per_evaluated_event, 4)
                    if eval_result.mean_false_alarms_per_evaluated_event is not None
                    else None
                ),
                "range_rmse_m": (
                    round(eval_result.cumulative_range_rmse, 4)
                    if eval_result.cumulative_range_rmse is not None
                    else None
                ),
                "velocity_rmse_m_s": (
                    round(eval_result.cumulative_velocity_rmse, 4)
                    if eval_result.cumulative_velocity_rmse is not None
                    else None
                ),
                "miss_breakdown": miss_breakdown,
                "event_join": _evaluation_join_metadata(evaluation_join),
            }

        if mot_result is not None:
            report["mot_metrics"] = {
                "mota": round(mot_result.mota, 4) if mot_result.mota is not None else None,
                "motp_m": round(mot_result.motp, 4) if mot_result.motp is not None else None,
                "id_switches": mot_result.total_id_switches,
                "total_gt": mot_result.total_gt,
                "total_matches": mot_result.total_matches,
                "total_misses": mot_result.total_misses,
                "total_fp": mot_result.total_fp,
                "num_tracks": mot_result.num_tracks,
                "num_gt_targets": mot_result.num_gt_targets,
                "track_fragmentation": {
                    str(k): v for k, v in mot_result.track_fragmentation.items()
                },
                "distinct_track_id_count": {
                    str(k): v for k, v in mot_result.distinct_track_id_count.items()
                },
                "metric_protocol": mot_result.metric_protocol,
                "mostly_tracked": mot_result.mostly_tracked,
                "mostly_lost": mot_result.mostly_lost,
            }

        report_path = output_dir / "detection_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)

        # Human-readable summary.
        ds = report["detection_statistics"]
        report_lines = [
            "",
            "=" * 60,
            "DETECTION REPORT",
            "=" * 60,
            f"Scenario:    {report['scenario']}",
            f"Frames:      {report['total_frames']} total, {report['frames_with_sensing']} with sensing",
            f"Detections:  {ds['total_detections']} total, {ds['mean_per_frame']:.1f} avg/frame, "
            f"max={ds['max_per_frame']}",
            f"GT targets:  {report['ground_truth']['num_targets']}",
        ]
        if "association" in report:
            a = report["association"]
            precision_text = "undefined" if a["precision"] is None else f"{a['precision']:.1%}"
            detection_probability_text = (
                "undefined"
                if a["detection_probability"] is None
                else f"{a['detection_probability']:.1%}"
            )
            mean_false_alarms_text = (
                "undefined"
                if a["mean_false_alarms_per_evaluated_event"] is None
                else f"{a['mean_false_alarms_per_evaluated_event']:.2f}"
            )
            range_rmse_text = (
                "undefined" if a["range_rmse_m"] is None else f"{a['range_rmse_m']:.3f} m"
            )
            velocity_rmse_text = (
                "undefined"
                if a["velocity_rmse_m_s"] is None
                else f"{a['velocity_rmse_m_s']:.3f} m/s"
            )
            report_lines.extend(
                [
                    "",
                    f"Association (gate: {a['range_gate_m']}m / {a['velocity_gate_m_s']}m/s):",
                    f"  TP={a['true_positives']}  FP={a['false_positives']}  FN={a['false_negatives']}",
                    f"  Precision={precision_text}  Pd={detection_probability_text}  "
                    f"Mean false alarms/evaluated event={mean_false_alarms_text}",
                    f"  Range RMSE={range_rmse_text}  Velocity RMSE={velocity_rmse_text}",
                ]
            )
        if "mot_metrics" in report:
            m = report["mot_metrics"]
            mota_text = "undefined" if m["mota"] is None else f"{m['mota']:.1%}"
            motp_text = "undefined" if m["motp_m"] is None else f"{m['motp_m']:.3f} m"
            report_lines.extend(
                [
                    "",
                    "MOT Metrics:",
                    f"  MOTA={mota_text}  MOTP={motp_text}  " f"ID Switches={m['id_switches']}",
                    f"  Mostly Tracked: {m['mostly_tracked']}/{m['num_gt_targets']}  "
                    f"Mostly Lost: {m['mostly_lost']}/{m['num_gt_targets']}",
                    f"  Track interruptions: {m['track_fragmentation']}",
                    f"  Distinct assigned IDs: {m['distinct_track_id_count']}",
                    f"  Metric protocol: {m['metric_protocol']}",
                ]
            )
        report_lines.extend(["", f"Report saved to: {report_path}", "=" * 60])
        logger.info("\n".join(report_lines))

    # --- Detection threshold sweep ---
    sweep_results: list[ThresholdSweepPoint] | None = None
    if args.threshold_sweep and processed > 0:
        sweep_frames_by_key = _index_evaluation_frames(frames)
        has_gt = any(
            _ground_truth_for_evaluation(frame)[1].size > 0
            for frame in sweep_frames_by_key.values()
        )
        if has_gt:
            logger.info(
                "Running threshold sweep: %.1f to %.1f in %d steps",
                args.threshold_range[0],
                args.threshold_range[1],
                args.threshold_steps,
            )
            sweep_results = sweep_thresholds(
                frames,
                threshold_range=tuple(args.threshold_range),
                threshold_steps=args.threshold_steps,
                base_config=config,
                range_gate=args.range_gate,
                velocity_gate=args.velocity_gate,
                rebuild_rd=bool(args.rebuild_rd),
                rebuild_cfr=config.range_response is not None or config.range_ifft_size is not None,
            )
            # Save sweep results
            sweep_path = output_dir / "threshold_sweep_results.json"
            with open(sweep_path, "w") as f:
                json.dump([asdict(pt) for pt in sweep_results], f, indent=2, default=_json_default)
            logger.info(
                "Threshold sweep results saved to %s (%d points)",
                sweep_path,
                len(sweep_results),
            )
        else:
            logger.warning("No ground truth; skipping threshold sweep")

    # --- Generate figures ---
    if args.figures and processed > 0:
        _generate_figures(
            output_dir,
            processed,
            frames=frames,
            all_results=all_results,
            eval_result=eval_result,
            pruned_ghost_track_ids=pruned_ghost_track_ids,
            scene_overlay=args.scene_overlay,
            figure_names=set(args.figure) if args.figure else None,
        )

    if args.figures and sweep_results:
        _generate_threshold_sweep_figures(output_dir, sweep_results)

    # Generate animated GIF
    if args.gif and processed > 0:
        gif_kwargs = dict(
            fps=args.gif_fps,
            interpolation=args.gif_interp,
            stride=args.gif_stride,
        )
        _generate_rd_gif(output_dir, processed, all_results=all_results, **gif_kwargs)
        if args.gif_per_frame_vmax:
            _generate_rd_gif(
                output_dir,
                processed,
                all_results=all_results,
                **gif_kwargs,
                per_frame_vmax=True,
            )
        if args.gif_no_detections:
            _generate_rd_gif(
                output_dir,
                processed,
                all_results=all_results,
                **gif_kwargs,
                show_detections=False,
            )
            if args.gif_per_frame_vmax:
                _generate_rd_gif(
                    output_dir,
                    processed,
                    all_results=all_results,
                    **gif_kwargs,
                    per_frame_vmax=True,
                    show_detections=False,
                )

    if args.gif_3d and processed > 0:
        _generate_rd_gif_3d(output_dir, processed, fps=args.gif_fps, all_results=all_results)

    # Generate micro-Doppler spectrograms from stored CIR
    if args.micro_doppler and processed > 0:
        _generate_micro_doppler_figures(
            args.frames_dir,
            output_dir,
            config=config,
            target_range_m=args.micro_doppler_range_m,
            velocity_overlay=vel_overlay,
            stft_window_override=args.stft_window,
        )


# Aliases for internal call sites (implementations in shared.sensing.plotting.figures)
_generate_figures = generate_figures
_generate_threshold_sweep_figures = generate_threshold_sweep_figures
_generate_rd_gif = generate_rd_gif
_generate_rd_gif_3d = generate_rd_gif_3d
_load_cir_sequence = load_cir_sequence
_load_gt_entries_for_tracking = load_gt_entries_for_tracking
_generate_micro_doppler_figures = generate_micro_doppler_figures


if __name__ == "__main__":
    main()
    frames: list[dict[str, Any]] | None = (None,)
