"""Sensing processors selected by a resolved acquisition contract.

``create_sensing_processor`` consumes the immutable execution contract from
``acquisition``. Each processor owns per-run history and detection/tracking
state, while ``strategies`` implements its numerical path. The factory does
not re-interpret YAML mode or CIR cadence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np

from generator.core.configuration import SensingConfig
from generator.core.configuration.defaults import (
    DEFAULT_SENSING_RAW_PAYLOAD_POLICY,
    VALID_SENSING_RAW_PAYLOAD_POLICIES,
)
from generator.core.sensing.acquisition import ResolvedSensingExecutionConfig
from generator.core.sensing.strategies import (
    process_analytic_path_doppler_oracle,
    process_coherent_cpi_fft,
    process_sequential_stft,
)
from shared.logging import get_logger
from shared.sensing.detection.beam_scan import run_beam_scan_detection_pipeline, visible_array_beams
from shared.sensing.detection.clutter import SlidingBaselineRemover
from shared.sensing.detection.pipeline import (
    SNR_CENSOR_CEILING_DB,
    SNR_CENSOR_FLOOR_DB,
    DetectionPipelineConfig,
    DetectionStageArtifacts,
    apply_detection_range_gate,
    run_detection_pipeline,
)
from shared.sensing.products import add_product_spec
from shared.sensing.signal_processing import (
    C0,
    RangeDopplerResult,
    apply_detection_velocity_metadata,
)
from shared.sensing.tracker import (
    CartesianKalmanTracker,
    JPDAFTracker,
    KalmanTracker,
    MofNTracker,
    TrackerConfig,
)

logger = get_logger(__name__)


DETECTION_STAGE_PRODUCT_KEYS = (
    "range_doppler_detection_input_power_by_pair",
    "range_doppler_cfar_noise_power_by_pair",
    "range_doppler_cfar_threshold_power_by_pair",
    "range_doppler_cfar_training_cell_count_by_pair",
    "range_doppler_cfar_eligible_mask_by_pair",
    "range_doppler_cfar_mask_by_pair",
    "range_doppler_nms_mask_by_pair",
    "range_doppler_snr_db_by_pair",
    "range_doppler_snr_valid_mask_by_pair",
    "range_doppler_cluster_labels_by_pair",
)
"""Complete canonical schema-v4 detector-stage numeric-product inventory."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class SensingProcessorBase(ABC):
    """Abstract base for sensing processors.

    Shared functionality: config access, window generation, CFAR detection
    building, and sensing metadata assembly.  Subclasses implement
    :meth:`process` which is the per-frame entry point.
    """

    def __init__(
        self,
        config: SensingConfig,
        carrier_frequency_hz: float,
        execution_config: ResolvedSensingExecutionConfig,
        *,
        uses_slow_time_history: bool = False,
    ):
        self.config = config
        self.carrier_frequency_hz = carrier_frequency_hz
        self.execution_config = execution_config
        self.recording_profile = getattr(config, "recording_profile", None)
        self._display_last_spectrum_time_s: float | None = None
        mode = getattr(self.config, "range_mode", "bistatic")
        mode = mode.lower() if isinstance(mode, str) else "bistatic"
        self.range_mode = "monostatic" if mode.startswith("mono") else "bistatic"
        self.raw_payload_policy = self._normalize_raw_payload_policy(
            getattr(
                self.config,
                "raw_payload_policy",
                DEFAULT_SENSING_RAW_PAYLOAD_POLICY,
            )
        )
        self.min_detection_range_m = getattr(self.config, "min_detection_range_m", None)
        self.max_detection_range_m = getattr(self.config, "max_detection_range_m", None)
        self.persist_detection_stage_artifacts = bool(
            getattr(self.config, "persist_detection_stage_artifacts", False)
        )

        self.track_confirmation_enabled = bool(
            getattr(self.config, "track_confirmation_enabled", False)
        )
        self.track_confirmation_m = max(1, int(getattr(self.config, "track_confirmation_m", 2)))
        self.track_confirmation_n = max(
            self.track_confirmation_m, int(getattr(self.config, "track_confirmation_n", 3))
        )
        self.track_max_missed_frames = max(
            0, int(getattr(self.config, "track_max_missed_frames", 1))
        )
        self.track_association_range_gate_m = max(
            0.0, float(getattr(self.config, "track_association_range_gate_m", 0.6))
        )
        self.track_association_velocity_gate_m_s = max(
            0.0, float(getattr(self.config, "track_association_velocity_gate_m_s", 0.3))
        )
        self.track_output_tentative = bool(getattr(self.config, "track_output_tentative", False))
        self.track_confirmation_mode = getattr(self.config, "track_confirmation_mode", "latched")
        tracker_cfg = TrackerConfig(
            confirmation_mode=self.track_confirmation_mode,
            record_diagnostics=bool(getattr(self.config, "persist_tracking_diagnostics", False)),
            range_rate_scale=2.0 if self.range_mode == "bistatic" else 1.0,
            m=self.track_confirmation_m,
            n=self.track_confirmation_n,
            max_missed_frames=self.track_max_missed_frames,
            range_gate_m=self.track_association_range_gate_m,
            velocity_gate_m_s=self.track_association_velocity_gate_m_s,
            azimuth_gate_deg=self.config.track_association_azimuth_gate_deg,
            output_tentative=self.track_output_tentative,
            **(
                {"xy_gate_m": self.config.track_association_position_gate_m}
                if self.config.track_association_position_gate_m is not None
                else {}
            ),
        )
        tracker_type = str(getattr(self.config, "tracker_type", "m_of_n")).lower()
        self._cartesian_tracking = tracker_type in {"cartesian", "cartesian3d"}
        if (
            self._cartesian_tracking
            and self.track_confirmation_enabled
            and self.range_mode == "bistatic"
        ):
            raise ValueError(
                "Native bistatic reports require explicit localization before Cartesian tracking"
            )
        if self._cartesian_tracking:
            acceleration = (
                {}
                if self.config.track_acceleration_psd_m2_s3 is None
                else {"acceleration_psd_m2_s3": self.config.track_acceleration_psd_m2_s3}
            )
            self._tracker = CartesianKalmanTracker(
                tracker_cfg, include_z=tracker_type == "cartesian3d", **acceleration
            )
        elif tracker_type == "kalman":
            self._tracker = KalmanTracker(tracker_cfg)
        elif tracker_type == "jpdaf":
            self._tracker = JPDAFTracker(
                tracker_cfg,
                max_joint_nodes=self.config.jpda_max_joint_nodes,
                birth_confirmation_frames=self.config.jpda_birth_confirmation_frames,
                birth_separation_chi2=self.config.jpda_birth_separation_chi2,
            )
        else:
            self._tracker = MofNTracker(tracker_cfg)

        # Sensor geometry cache. Target state is deliberately excluded: target
        # truth must not influence detection, clustering, or tracking outputs.
        self._cached_tx_pos: np.ndarray | None = None
        self._cached_rx_pos: np.ndarray | None = None

        # Dimensionless target-relative complex AWGN.  This is deliberately
        # not a physical receiver thermal-noise or link-budget model.
        self.noise_enabled = bool(getattr(self.config, "noise_enabled", False))
        self.noise_snr_db: float | None = getattr(self.config, "noise_snr_db", None)
        rng_seed = getattr(self.config, "rng_seed", None)
        self._noise_rng = np.random.default_rng(int(rng_seed) if rng_seed is not None else None)

        # Slow-time history and clutter removal (snapshot and sequential only)
        if uses_slow_time_history:
            history_len = max(1, int(self.config.fft_size_doppler))
            self._fast_history_raw: deque[np.ndarray] = deque(maxlen=history_len)
            self._fast_history_clean: deque[np.ndarray] = deque(maxlen=history_len)
            self.clutter_enabled = bool(getattr(self.config, "clutter_removal_enabled", False))
            self.clutter_window = max(1, int(getattr(self.config, "clutter_removal_window", 1)))
            self._range_clutter = (
                SlidingBaselineRemover(self.clutter_window, clip_zero=True)
                if self.clutter_enabled
                else None
            )
            self._slow_time_clutter = (
                SlidingBaselineRemover(self.clutter_window, clip_zero=False)
                if self.clutter_enabled
                else None
            )

    def _update_geometry_cache(self, simulation_objects: Any, frame_idx: int = 0) -> None:
        """Extract observation-system TX/RX positions from simulation objects.

        Args:
            simulation_objects: Contains ``*_positions_cache`` lists.
                Each cache is ``List[List[mi.Point3f]]`` — one trajectory
                per device, each trajectory a list of Point3f per step.
            frame_idx: Current simulation step index, used to pick the
                correct position along each trajectory.
        """
        if simulation_objects is None:
            return

        tx_cache = getattr(simulation_objects, "tx_positions_cache", None)
        if tx_cache and len(tx_cache) > 0:
            trajectory = tx_cache[0]  # first TX's trajectory
            idx = min(frame_idx, len(trajectory) - 1)
            self._cached_tx_pos = self._point3f_to_array(trajectory[idx])

        rx_cache = getattr(simulation_objects, "rx_positions_cache", None)
        if rx_cache and len(rx_cache) > 0:
            trajectory = rx_cache[0]  # first RX's trajectory
            idx = min(frame_idx, len(trajectory) - 1)
            self._cached_rx_pos = self._point3f_to_array(trajectory[idx])

    @staticmethod
    def _point3f_to_array(pt: Any) -> np.ndarray:
        """Convert a Mitsuba Point3f (or array-like) to a float64 numpy array."""
        if hasattr(pt, "numpy"):
            return np.asarray(pt.numpy(), dtype=np.float64).ravel()[:3]
        if hasattr(pt, "x") and hasattr(pt, "y") and hasattr(pt, "z"):
            # DrJit Float components may need .numpy() conversion
            def _scalar(v: Any) -> float:
                return float(v.numpy()) if hasattr(v, "numpy") else float(v)

            return np.array([_scalar(pt.x), _scalar(pt.y), _scalar(pt.z)], dtype=np.float64)
        return np.asarray(pt, dtype=np.float64).ravel()[:3]

    # ---- public entry point ----

    @abstractmethod
    def process(
        self,
        simulation_objects: Any,
        frame_idx: int,
        paths: Any,
        cir_stack: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        source_time_s: float,
    ) -> dict[str, Any] | None:
        """Compute sensing products for one frame / CPI."""

    # ---- shared helpers ----

    @staticmethod
    def _normalize_raw_payload_policy(policy: Any) -> str:
        if isinstance(policy, str):
            normalized = policy.lower()
            if normalized in VALID_SENSING_RAW_PAYLOAD_POLICIES:
                return normalized
        return DEFAULT_SENSING_RAW_PAYLOAD_POLICY

    def _should_emit_raw_payload(self, key: str) -> bool:
        if self.recording_profile is not None:
            from shared.sensing.recording import retain_raw_product

            return retain_raw_product(self.config, key)
        if self.raw_payload_policy == "always":
            return True
        if self.raw_payload_policy == "none":
            return False
        # diagnostic policy keeps only the minimum payloads needed for
        # downstream diagnostics (micro-Doppler and CIR inspection).
        return key in {"cir", "slow_time_matrix", "cfr_by_pair_rx_element"}

    def _record_display_products(self, sensing: dict[str, Any], **inputs: Any) -> None:
        if self.recording_profile in {None, "results"}:
            return
        from shared.sensing.recording import add_display_products

        self._display_last_spectrum_time_s = add_display_products(
            sensing,
            self.config,
            last_spectrum_time_s=self._display_last_spectrum_time_s,
            **inputs,
        )

    def _finalize_recording(self, sensing: dict[str, Any]) -> dict[str, Any]:
        if self.recording_profile is None:
            return sensing
        from shared.sensing.recording import apply_recording_profile

        return apply_recording_profile(sensing, self.config)

    @staticmethod
    def _validated_source_time_s(source_time_s: float) -> float:
        """Return a finite source timestamp for result/tracker identity."""
        timestamp_s = float(source_time_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("source_time_s must be finite")
        return timestamp_s

    def _build_sensing_metadata(
        self,
        *,
        frame_idx: int,
        processing_strategy: str,
        rd_source: str,
        rd_model: str,
        frame_role: str,
        fft_d: int,
        n_history: int,
        source_frame_step: int | None = None,
        pair_index: int = 0,
        raw_payload_policy: str = DEFAULT_SENSING_RAW_PAYLOAD_POLICY,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build shared axis, recipe, and acquisition metadata for all strategies."""
        c0 = C0
        range_factor = 2.0 if self.range_mode == "monostatic" else 1.0
        lambda_c = c0 / self.carrier_frequency_hz
        fs = self.config.bandwidth
        prf = self.config.prf_hz
        required_history = self.execution_config.acquisition.cpi_length_samples
        range_axis_label = "Range (m)" if self.range_mode == "monostatic" else "Path Length (m)"
        rng_seed_value = getattr(self.config, "rng_seed", None)
        try:
            rng_seed = int(rng_seed_value) if rng_seed_value is not None else None
        except (TypeError, ValueError):
            rng_seed = None
        if processing_strategy != self.execution_config.processing.strategy:
            raise RuntimeError(
                "Sensing strategy metadata disagrees with the resolved execution contract"
            )
        if fft_d != self.execution_config.processing.fft_size_doppler:
            raise RuntimeError(
                "Sensing Doppler FFT length disagrees with the resolved execution contract"
            )
        processing_recipe = asdict(self.config)
        # The pre-existing sequential cadence is one pulse. Keep its recipe
        # identity stable whether that default is omitted or stated explicitly.
        if processing_recipe["cpi_hop_samples"] in (None, 1):
            processing_recipe.pop("cpi_hop_samples")
        # New Cartesian options must not change identities of existing recipes.
        for key in ("track_association_position_gate_m", "track_acceleration_psd_m2_s3"):
            if processing_recipe[key] is None:
                processing_recipe.pop(key)
        # Preserve the exact historical identity when the new option is omitted
        # or explicitly selects the reference model.
        if self.config.range_response.model == "point_bin_floor":
            processing_recipe.pop("range_response", None)
        for display_only_key in ("enabled", "display_range_xlim", "display_velocity_ylim"):
            processing_recipe.pop(display_only_key, None)
        processing_recipe.update(
            {
                "processing_mode": self.execution_config.processing.mode,
                "processing_strategy": processing_strategy,
                "rd_source": rd_source,
                "rd_model": rd_model,
                "carrier_frequency_hz": self.carrier_frequency_hz,
                "noise_model": (
                    "target_relative_complex_awgn" if self.noise_enabled else "disabled"
                ),
            }
        )
        resolved_config: dict[str, Any] = {
            "carrier_freq_hz": self.carrier_frequency_hz,
            "carrier_frequency": self.carrier_frequency_hz,
            "bandwidth": fs,
            "bandwidth_hz": fs,
            "sampling_frequency": fs,
            "prf_hz": prf,
            "fft_size_range": self.config.fft_size_range,
            "fft_size_doppler": fft_d,
            "range_resolution_m": c0 / (range_factor * fs),
            "velocity_resolution_m_s": (lambda_c * prf) / (2 * fft_d),
            "range_mode": self.range_mode,
            "processing_mode": self.execution_config.processing.mode,
            "rd_required_history": required_history,
            "pre_cfar_doppler_edge_guard_bins": int(
                getattr(self.config, "pre_cfar_doppler_edge_guard_bins", 2)
            ),
            "pre_cfar_doppler_dc_guard_bins": int(
                getattr(self.config, "pre_cfar_doppler_dc_guard_bins", 0)
            ),
            "min_detection_range_m": self.min_detection_range_m,
            "max_detection_range_m": self.max_detection_range_m,
            "track_confirmation_enabled": self.track_confirmation_enabled,
            "track_confirmation_m": self.track_confirmation_m,
            "track_confirmation_n": self.track_confirmation_n,
            "track_confirmation_mode": self.track_confirmation_mode,
            "track_max_missed_frames": self.track_max_missed_frames,
            "track_association_range_gate_m": self.track_association_range_gate_m,
            "track_association_velocity_gate_m_s": self.track_association_velocity_gate_m_s,
            "track_output_tentative": self.track_output_tentative,
            "eval_association_range_gate_m": float(
                getattr(self.config, "eval_association_range_gate_m", 10.0)
            ),
            "eval_association_velocity_gate_m_s": float(
                getattr(self.config, "eval_association_velocity_gate_m_s", 5.0)
            ),
            "persist_detection_stage_artifacts": self.persist_detection_stage_artifacts,
            "persist_tracking_diagnostics": bool(
                getattr(self.config, "persist_tracking_diagnostics", False)
            ),
            "raw_payload_policy": self.raw_payload_policy,
            "rng_seed": rng_seed,
            "noise_model": ("target_relative_complex_awgn" if self.noise_enabled else "disabled"),
            "noise_sample_domain": "relative_complex_voltage",
            "noise_power_unit": "1",
            "noise_reference_policy": "mean_power_over_nonzero_input_samples",
            "physical_thermal_noise_model": False,
            "display_range_xlim": getattr(self.config, "display_range_xlim", None),
            "display_velocity_ylim": getattr(self.config, "display_velocity_ylim", None),
            "processing_recipe": processing_recipe,
            **self.execution_config.persisted(),
        }
        apply_detection_velocity_metadata(resolved_config, range_mode=self.range_mode)
        if self._cartesian_tracking:
            resolved_config.update(
                tracker_type=self.config.tracker_type,
                tracking_coordinate_frame="receiver_local",
            )
            for key in ("track_association_position_gate_m", "track_acceleration_psd_m2_s3"):
                value = getattr(self.config, key)
                if value is not None:
                    resolved_config[key] = value
        if self.config.range_response.model == "cfr_ifft":
            resolved_config["range_response"] = self.config.range_response.to_dict()
            resolved_config["noise_model"] = "frequency_observation_complex_awgn"
            resolved_config["noise_sample_domain"] = "unwindowed_frequency_observation"
            resolved_config["noise_reference_policy"] = "fixed_variance"
            processing_recipe["noise_model"] = resolved_config["noise_model"]

        data: dict[str, Any] = {
            "carrier_freq_hz": self.carrier_frequency_hz,
            "bandwidth_hz": fs,
            "prf_hz": prf,
            "sampling_frequency": fs,
            "fft_size_range": self.config.fft_size_range,
            "fft_size_doppler": fft_d,
            "num_samples": self.config.fft_size_range,
            "range_mode": self.range_mode,
            "range_axis_label": range_axis_label,
            "range_resolution_m": c0 / (range_factor * fs),
            "velocity_resolution_m_s": (lambda_c * prf) / (2 * fft_d),
            "rd_history_size": n_history,
            "rd_required_history": required_history,
            "rd_ready": n_history >= required_history,
            "rd_map_frames_used": min(n_history, required_history),
            "processing_strategy": processing_strategy,
            "rd_source": rd_source,
            "rd_model": rd_model,
            "frame_role": frame_role,
            "pair_index": int(pair_index),
            "source_frame_step": frame_idx if source_frame_step is None else int(source_frame_step),
            "raw_payload_policy": self._normalize_raw_payload_policy(raw_payload_policy),
            "rd_fresh": False,
            "rp_fresh": False,
            "config": dict(resolved_config),
            "config_resolved": dict(resolved_config),
        }
        apply_detection_velocity_metadata(data, range_mode=self.range_mode)
        if extra:
            data.update(extra)
            data["config"].update(extra)
            data["config_resolved"].update(extra)
        return data

    def _apply_range_gate(
        self, detections: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply optional physical min/max range gating on detections."""
        return apply_detection_range_gate(
            detections,
            min_detection_range_m=self.min_detection_range_m,
            max_detection_range_m=self.max_detection_range_m,
        )

    def _apply_track_confirmation(
        self,
        detections: list[dict[str, Any]],
        *,
        update_time_s: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply the selected tracker at the physical sensing result epoch."""
        if not self.track_confirmation_enabled:
            return detections, {
                "enabled": False,
                "raw": len(detections),
                "kept": len(detections),
                "active_tracks": 0,
                "confirmed_tracks": 0,
            }
        return self._tracker.filter_detections(
            detections,
            update_time_s=update_time_s,
        )

    def _publish_detection_stage_artifacts(
        self,
        sensing_data: dict[str, Any],
        stages: DetectionStageArtifacts,
    ) -> None:
        """Publish one complete detector-stage set with an explicit pair axis."""

        if not self.persist_detection_stage_artifacts:
            return
        artifacts = {
            "range_doppler_detection_input_power_by_pair": np.asarray(
                stages.cfar_input_power, dtype=np.float64
            ),
            "range_doppler_cfar_noise_power_by_pair": np.asarray(
                stages.cfar_noise_floor, dtype=np.float64
            ),
            "range_doppler_cfar_threshold_power_by_pair": np.asarray(
                stages.cfar_threshold, dtype=np.float64
            ),
            "range_doppler_cfar_training_cell_count_by_pair": np.asarray(
                stages.cfar_training_cell_count, dtype=np.int32
            ),
            "range_doppler_cfar_eligible_mask_by_pair": np.asarray(
                stages.cfar_eligible_mask, dtype=np.bool_
            ),
            "range_doppler_cfar_mask_by_pair": np.asarray(stages.cfar_mask, dtype=np.bool_),
            "range_doppler_nms_mask_by_pair": np.asarray(stages.nms_mask, dtype=np.bool_),
            "range_doppler_snr_db_by_pair": np.asarray(stages.snr_map, dtype=np.float64),
            "range_doppler_snr_valid_mask_by_pair": np.asarray(
                stages.snr_valid_mask, dtype=np.bool_
            ),
            "range_doppler_cluster_labels_by_pair": np.asarray(
                stages.cluster_labels, dtype=np.int32
            ),
        }
        shapes = {array.shape for array in artifacts.values()}
        if len(shapes) != 1 or any(array.ndim != 2 for array in artifacts.values()):
            raise RuntimeError("Detection-stage artifacts must share one [doppler,range] shape")
        for data_key, array in artifacts.items():
            sensing_data[data_key] = np.array(array[np.newaxis, ...], copy=True)

    def _queue_detection_stage_product_specs(
        self,
        sensing_data: dict[str, Any],
        *,
        primary_data_key: str,
        axes: Sequence[Mapping[str, Any]],
        evidence_role: str = "observation_derived",
        input_processing: Mapping[str, Any],
    ) -> None:
        """Queue the complete typed detector DAG rooted at one primary RD product."""

        if not self.persist_detection_stage_artifacts:
            return
        present = [key for key in DETECTION_STAGE_PRODUCT_KEYS if key in sensing_data]
        if not present:
            return
        missing = [key for key in DETECTION_STAGE_PRODUCT_KEYS if key not in sensing_data]
        if missing:
            raise RuntimeError(
                "Detection-stage publication must be complete; missing " + ", ".join(missing)
            )
        if primary_data_key not in sensing_data:
            raise RuntimeError(
                "Detection-stage publication requires its primary typed RD source "
                f"{primary_data_key!r}"
            )

        input_key = "range_doppler_detection_input_power_by_pair"
        noise_key = "range_doppler_cfar_noise_power_by_pair"
        threshold_key = "range_doppler_cfar_threshold_power_by_pair"
        count_key = "range_doppler_cfar_training_cell_count_by_pair"
        eligible_key = "range_doppler_cfar_eligible_mask_by_pair"
        cfar_mask_key = "range_doppler_cfar_mask_by_pair"
        nms_key = "range_doppler_nms_mask_by_pair"
        snr_key = "range_doppler_snr_db_by_pair"
        snr_valid_key = "range_doppler_snr_valid_mask_by_pair"
        labels_key = "range_doppler_cluster_labels_by_pair"

        beam_scan = self.config.array_detection_mode == "beam_scan"
        cfar_processing = {
            "algorithm": "two_dimensional_constant_false_alarm_rate",
            "input_domain": "power_linear",
            "method": self.config.cfar_type.lower(),
            "guard_cells": {
                "doppler": int(self.config.cfar_rd_guard_cells_doppler),
                "range": int(self.config.cfar_rd_guard_cells_range),
            },
            "reference_cells": {
                "doppler": int(self.config.cfar_rd_ref_cells_doppler),
                "range": int(self.config.cfar_rd_ref_cells_range),
            },
            "threshold_scale": float(self.config.cfar_threshold_scale),
            "threshold_semantics": "raw_linear_power_multiplier",
            "max_dynamic_range_db": self.config.cfar_max_dynamic_range_db,
            "power_floor_reference": (
                "max_pre_clutter_and_guarded_peak_within_each_beam"
                if sensing_data.get("detector_power_reference") is not None
                else "strongest_guarded_cell_per_detector_input_or_beam"
            ),
            "os_rank_fraction": float(self.config.cfar_os_rank),
            "os_rank_rounding": "round_fraction_times_training_count_minus_one",
            "edge_policy": self.config.cfar_edge_policy,
            "cut_comparison": "strict_greater_than_threshold",
        }
        if beam_scan:
            cfar_processing.update(
                angular_input_data_key=primary_data_key,
                execution="independently_per_visible_beam_before_projection",
                stage_projection="strongest_visible_beam_per_rd_cell",
            )
        nms_radius = {
            "doppler": max(int(self.config.cfar_rd_guard_cells_doppler), 2),
            "range": max(int(self.config.cfar_rd_guard_cells_range), 4),
        }

        add_product_spec(
            sensing_data,
            data_key=input_key,
            kind="range_doppler_detection_input_power",
            axes=axes,
            representation="power_linear",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(primary_data_key,),
            processing={
                "operation": "detector_input_power_projection",
                "output_domain": "power_linear",
                **dict(input_processing),
            },
        )
        add_product_spec(
            sensing_data,
            data_key=noise_key,
            kind="range_doppler_cfar_noise_power",
            axes=axes,
            representation="power_linear",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(input_key, primary_data_key) if beam_scan else (input_key,),
            processing={
                **cfar_processing,
                "operation": "training_cell_noise_power_estimate",
                "ineligible_fill_value": 0.0,
            },
        )
        add_product_spec(
            sensing_data,
            data_key=count_key,
            kind="range_doppler_cfar_training_cell_count",
            axes=axes,
            representation="integer_count",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(input_key,),
            processing={
                **cfar_processing,
                "operation": "actual_training_cell_count",
                "count_semantics": "training_cells_used_for_each_cell_under_test",
                "ineligible_fill_value": 0,
            },
        )
        add_product_spec(
            sensing_data,
            data_key=eligible_key,
            kind="range_doppler_cfar_eligible_mask",
            axes=axes,
            representation="boolean_mask",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(input_key, count_key),
            processing={
                **cfar_processing,
                "operation": "training_window_eligibility",
                "true_semantics": "cell_has_training_cells_required_by_edge_policy",
            },
        )
        add_product_spec(
            sensing_data,
            data_key=threshold_key,
            kind="range_doppler_cfar_threshold_power",
            axes=axes,
            representation="power_linear",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(noise_key, eligible_key),
            processing={
                **cfar_processing,
                "operation": "cfar_threshold_power",
                "formula": (
                    "max(noise_power * threshold_scale, input_peak_power * 10**(-max_dynamic_range_db/10))"
                    if self.config.cfar_max_dynamic_range_db is not None
                    else "noise_power * threshold_scale"
                ),
                "ineligible_fill_value": 0.0,
            },
        )
        add_product_spec(
            sensing_data,
            data_key=cfar_mask_key,
            kind="range_doppler_cfar_mask",
            axes=axes,
            representation="boolean_mask",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(input_key, threshold_key, eligible_key),
            processing={
                **cfar_processing,
                "operation": "cfar_cell_under_test_decision",
                "formula": "eligible and input_power > threshold_power and input_power > 0",
            },
        )
        add_product_spec(
            sensing_data,
            data_key=snr_key,
            kind="range_doppler_snr_db",
            axes=axes,
            representation="log_power_ratio_db",
            unit="dB",
            evidence_role=evidence_role,
            source_data_keys=(input_key, noise_key, eligible_key),
            processing={
                "operation": "cell_power_over_cfar_noise_estimate",
                "formula": "10 * log10(input_power / noise_power)",
                "validity": "eligible_training_window_and_strictly_positive_noise_power",
                "zero_input_valid_value_db": SNR_CENSOR_FLOOR_DB,
                "finite_censor_floor_db": SNR_CENSOR_FLOOR_DB,
                "finite_censor_ceiling_db": SNR_CENSOR_CEILING_DB,
                "invalid_fill_value_db": 0.0,
            },
        )
        add_product_spec(
            sensing_data,
            data_key=snr_valid_key,
            kind="range_doppler_snr_valid_mask",
            axes=axes,
            representation="boolean_mask",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(noise_key, eligible_key),
            processing={
                "operation": "snr_denominator_validity",
                "formula": "cfar_eligible and noise_power > 0",
                "invalid_snr_fill_value_db": 0.0,
            },
        )
        add_product_spec(
            sensing_data,
            data_key=nms_key,
            kind="range_doppler_nms_mask",
            axes=axes,
            representation="boolean_mask",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(
                (input_key, cfar_mask_key, snr_valid_key, primary_data_key)
                if beam_scan
                else (input_key, cfar_mask_key, snr_valid_key)
            ),
            processing={
                "algorithm": "greedy_local_maximum_nonmaximum_suppression_2d",
                **({"execution": "per_beam_before_strongest_beam_selection"} if beam_scan else {}),
                "candidate_mask": "cfar_mask and snr_valid_mask",
                "local_maximum_neighborhood_bins": {"doppler": 3, "range": 3},
                "candidate_order": "descending_input_power_via_numpy_argsort",
                "suppression_radius_bins": nms_radius,
                "boundary_policy": "clip_without_wrap",
            },
        )
        add_product_spec(
            sensing_data,
            data_key=labels_key,
            kind="range_doppler_cluster_labels",
            axes=axes,
            representation="integer_label",
            unit="1",
            evidence_role=evidence_role,
            source_data_keys=(
                (nms_key, snr_key, snr_valid_key, primary_data_key)
                if beam_scan
                else (nms_key, snr_key, snr_valid_key)
            ),
            processing={
                "algorithm": "dbscan_on_gate_normalized_range_velocity_coordinates",
                "input_mask": "nms_mask",
                "range_coordinate": "range_axis_m / cluster_range_gate_m",
                "velocity_coordinate": (
                    "(doppler_axis_hz * wavelength_m / 2) / cluster_velocity_gate_m_s"
                ),
                "cluster_range_gate_m": self.config.cluster_range_gate_m,
                "cluster_velocity_gate_m_s": self.config.cluster_velocity_gate_m_s,
                **({"execution": "per_beam_before_strongest_beam_selection"} if beam_scan else {}),
                "wavelength_m": (
                    C0 / self.carrier_frequency_hz if self.carrier_frequency_hz else 1.0
                ),
                "velocity_semantics": "monostatic_equivalent_velocity",
                "dbscan_eps": 1.0,
                "dbscan_min_samples": 1,
                "background_label": -1,
                "missing_sklearn_fallback": "connected_components_of_unit_radius_graph",
            },
        )

    def _run_2d_cfar(
        self,
        rd_result: RangeDopplerResult,
        sensing_data: dict[str, Any],
        *,
        update_time_s: float,
        aoa_az_map: np.ndarray | None = None,
        aoa_el_map: np.ndarray | None = None,
        angular_power: np.ndarray | None = None,
        azimuth_direction_cosine: np.ndarray | None = None,
        elevation_direction_cosine: np.ndarray | None = None,
        beam_angle_refiner: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Run 2-D CFAR on an RD map and populate sensing_data in-place.

        Returns CFAR metadata dict (empty if no CFAR configured).
        """
        cfar_metadata: dict[str, Any] = {}
        if not self.config.output_detections:
            return cfar_metadata

        pipeline_cfg = DetectionPipelineConfig(
            cfar_guard_cells=(
                self.config.cfar_rd_guard_cells_doppler,
                self.config.cfar_rd_guard_cells_range,
            ),
            cfar_reference_cells=(
                self.config.cfar_rd_ref_cells_doppler,
                self.config.cfar_rd_ref_cells_range,
            ),
            cfar_threshold_scale=self.config.cfar_threshold_scale,
            cfar_method=self.config.cfar_type.lower(),
            cfar_os_rank=self.config.cfar_os_rank,
            cfar_edge_policy=self.config.cfar_edge_policy,
            nms_suppression_radius=(
                max(self.config.cfar_rd_guard_cells_doppler, 2),
                max(self.config.cfar_rd_guard_cells_range, 4),
            ),
            cluster_range_gate_m=self.config.cluster_range_gate_m,
            cluster_velocity_gate_m_s=self.config.cluster_velocity_gate_m_s,
            min_snr_db=self.config.cfar_min_snr_db,
            cfar_max_dynamic_range_db=self.config.cfar_max_dynamic_range_db,
            min_detection_range_m=self.min_detection_range_m,
            max_detection_range_m=self.max_detection_range_m,
            ghost_filter_velocity_gate_m_s=self.config.ghost_filter_velocity_gate_m_s,
            ghost_filter_azimuth_gate_deg=self.config.ghost_filter_azimuth_gate_deg,
        )
        wavelength = C0 / self.carrier_frequency_hz if self.carrier_frequency_hz else 1.0
        if self.config.beam_angle_oversampling > 1 and beam_angle_refiner is None:
            raise ValueError("Beam angle refinement requires complex per-element RD samples")

        def track_detections(detections):
            # Preserve coarse-beam CFAR and pruning, then refine the positions
            # passed to the tracker using the same CPI's complex element data.
            if self.config.beam_angle_oversampling > 1:
                assert beam_angle_refiner is not None
                detections = beam_angle_refiner(detections)
            if self._cartesian_tracking and self.track_confirmation_enabled:
                array = sensing_data.get("rx_array", {})
                if (
                    self.config.tracker_type == "cartesian3d"
                    and min(array.get("rows", 0), array.get("columns", 0)) <= 1
                ):
                    raise ValueError("Cartesian 3D tracking requires a planar RX array")
            cartesian_reports = self._cartesian_tracking and self.track_confirmation_enabled
            if self.recording_profile is not None or cartesian_reports:
                sensing_data["detections_before_tracking"] = deepcopy(detections)
                if not cartesian_reports:
                    # Range/Doppler reports can have no angular localization.
                    # Omit unavailable optional coordinates from this new
                    # measurement contract instead of publishing NaN positions.
                    for report in sensing_data["detections_before_tracking"]:
                        for key in ("x_m", "y_m", "z_m", "aoa_az_deg", "aoa_el_deg"):
                            value = report.get(key)
                            if value is None or not np.isfinite(value):
                                report.pop(key, None)
                sensing_data["detections_before_tracking_metadata"] = {
                    "schema_version": 1 if cartesian_reports else 2,
                    "stage": "after_angle_refinement_before_tracking",
                    "coordinate_frame": (
                        "receiver_local" if cartesian_reports else "receiver_measurement"
                    ),
                    "tracking_applied": False,
                    "reference_time_s": update_time_s,
                }
            return self._apply_track_confirmation(detections, update_time_s=update_time_s)

        if self.config.array_detection_mode == "beam_scan":
            if (
                angular_power is None
                or azimuth_direction_cosine is None
                or elevation_direction_cosine is None
            ):
                raise ValueError(
                    "beam_scan requires the complete angular power tensor and coordinates"
                )
            beams, azimuths, elevations = visible_array_beams(
                angular_power,
                elevation_direction_cosine,
                azimuth_direction_cosine,
            )
            pipeline_result = run_beam_scan_detection_pipeline(
                rd_result,
                beam_power=beams,
                beam_azimuth_deg=azimuths,
                beam_elevation_deg=elevations,
                num_rows=int(angular_power.shape[0]),
                num_cols=int(angular_power.shape[1]),
                reference=sensing_data.get("detector_power_reference"),
                config=pipeline_cfg,
                wavelength_m=wavelength,
                edge_guard_bins=self.config.pre_cfar_doppler_edge_guard_bins,
                dc_guard_bins=self.config.pre_cfar_doppler_dc_guard_bins,
                post_detection_filter=track_detections,
                return_stage_artifacts=self.persist_detection_stage_artifacts,
                range_mode=self.range_mode,
                tx_pos=self._cached_tx_pos,
                rx_pos=self._cached_rx_pos,
            )
        else:
            pipeline_result = run_detection_pipeline(
                rd_result,
                config=pipeline_cfg,
                wavelength_m=wavelength,
                post_detection_filter=track_detections,
                return_stage_artifacts=self.persist_detection_stage_artifacts,
                range_mode=self.range_mode,
                tx_pos=self._cached_tx_pos,
                rx_pos=self._cached_rx_pos,
                aoa_az_map=aoa_az_map,
                aoa_el_map=aoa_el_map,
            )
        if pipeline_result.stages is not None:
            self._publish_detection_stage_artifacts(sensing_data, pipeline_result.stages)
        cfar_metadata["range_doppler"] = {
            "method": self.config.cfar_type.lower(),
            "guard_cells": {
                "doppler": self.config.cfar_rd_guard_cells_doppler,
                "range": self.config.cfar_rd_guard_cells_range,
            },
            "reference_cells": {
                "doppler": self.config.cfar_rd_ref_cells_doppler,
                "range": self.config.cfar_rd_ref_cells_range,
            },
            "threshold_scale": self.config.cfar_threshold_scale,
            "threshold_semantics": "raw_linear_power_multiplier",
            "max_dynamic_range_db": self.config.cfar_max_dynamic_range_db,
            "power_floor_reference": pipeline_result.metadata.get(
                "cfar_power_floor_reference", "strongest_guarded_cell_per_detector_input_or_beam"
            ),
            "os_rank": self.config.cfar_os_rank,
            "os_rank_rounding": "round_fraction_times_k_minus_one",
            "edge_policy": self.config.cfar_edge_policy,
            "cut_comparison": "strict_greater_than_threshold",
        }
        sensing_data["detections"] = pipeline_result.detections
        sensing_data["detection_filter_metadata"] = pipeline_result.metadata
        if self.config.beam_angle_oversampling > 1:
            sensing_data["detection_filter_metadata"]["angle_estimation"] = {
                "method": "local_zero_padded_spatial_fft",
                "oversampling": self.config.beam_angle_oversampling,
                "input": "complex_per_element_range_doppler",
                "stage": "after_coarse_beam_gates_before_tracking",
                "increases_aperture_resolution": False,
            }
        return cfar_metadata


# ---------------------------------------------------------------------------
# Analytic path-Doppler oracle processor
# ---------------------------------------------------------------------------


class AnalyticPathDopplerOracleProcessor(SensingProcessorBase):
    """Quarantined non-waveform oracle based on Sionna path Doppler."""

    def __init__(
        self,
        config: SensingConfig,
        carrier_frequency_hz: float,
        execution_config: ResolvedSensingExecutionConfig,
    ):
        super().__init__(
            config,
            carrier_frequency_hz,
            execution_config,
            uses_slow_time_history=True,
        )
        history_len = max(1, int(self.config.fft_size_doppler))
        self._cir_history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=history_len)

    def process(
        self,
        simulation_objects: Any,
        frame_idx: int,
        paths: Any,
        cir_stack: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        source_time_s: float,
    ) -> dict[str, Any] | None:
        self._update_geometry_cache(simulation_objects, frame_idx)
        reference_time_s = self._validated_source_time_s(source_time_s)
        return process_analytic_path_doppler_oracle(
            self,
            frame_idx=frame_idx,
            paths=paths,
            reference_time_s=reference_time_s,
        )


# ---------------------------------------------------------------------------
# Coherent CPI FFT processor
# ---------------------------------------------------------------------------


class CoherentCpiFftProcessor(SensingProcessorBase):
    """CPI sensing: process Sionna-native CIR stack with N coherent samples."""

    def process(
        self,
        simulation_objects: Any,
        frame_idx: int,
        paths: Any,
        cir_stack: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        source_time_s: float,
    ) -> dict[str, Any] | None:
        self._update_geometry_cache(simulation_objects, frame_idx)
        reference_time_s = self._validated_source_time_s(source_time_s)
        if cir_stack is None:
            raise RuntimeError(
                "coherent_cpi requires a Sionna coherent CIR stack; fallback is forbidden"
            )
        return process_coherent_cpi_fft(
            self,
            simulation_objects=simulation_objects,
            paths=paths,
            cir_stack=cir_stack,
            frame_idx=frame_idx,
            reference_time_s=reference_time_s,
        )


# ---------------------------------------------------------------------------
# Sequential STFT processor
# ---------------------------------------------------------------------------


class SequentialStftProcessor(SensingProcessorBase):
    """Sequential STFT: accumulate independent per-frame CIR and FFT for Doppler."""

    def __init__(
        self,
        config: SensingConfig,
        carrier_frequency_hz: float,
        execution_config: ResolvedSensingExecutionConfig,
    ):
        super().__init__(
            config,
            carrier_frequency_hz,
            execution_config,
            uses_slow_time_history=True,
        )
        history_len = max(1, int(self.config.fft_size_doppler))
        self._source_time_history: deque[float] = deque(maxlen=history_len)
        self._source_sample_count = 0
        self._frequency_history: deque[np.ndarray] = deque(maxlen=history_len)
        self._frequency_noise_history: deque[dict[str, Any]] = deque(maxlen=history_len)

    def process(
        self,
        simulation_objects: Any,
        frame_idx: int,
        paths: Any,
        cir_stack: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        source_time_s: float,
    ) -> dict[str, Any] | None:
        self._update_geometry_cache(simulation_objects, frame_idx)
        timestamp_s = self._validated_source_time_s(source_time_s)
        if self._source_time_history and timestamp_s <= self._source_time_history[-1]:
            raise ValueError("source_time_s must be strictly increasing in sequential mode")
        if self._source_time_history:
            if not np.isclose(
                timestamp_s - self._source_time_history[-1],
                1.0 / self.config.prf_hz,
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError(
                    "sequential sensing requires uniform source timestamps at the PRF; "
                    "a missing pulse cannot be compressed into a complete Doppler window"
                )
        candidate_times = [*self._source_time_history, timestamp_s]
        candidate_times = candidate_times[-self._source_time_history.maxlen :]
        reference_time_s = (candidate_times[0] + candidate_times[-1]) / 2.0
        result = process_sequential_stft(
            self,
            frame_idx=frame_idx,
            paths=paths,
            reference_time_s=reference_time_s,
            source_time_s=timestamp_s,
        )
        if result is not None:
            self._source_time_history.append(timestamp_s)
            self._source_sample_count += 1
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_sensing_processor(
    config: SensingConfig,
    carrier_frequency_hz: float,
    execution_config: ResolvedSensingExecutionConfig,
) -> SensingProcessorBase:
    """Create the processor declared by one resolved execution contract.

    Args:
        config: Sensing configuration.
        carrier_frequency_hz: Carrier frequency in Hz.
        execution_config: Frozen contract resolved before ray tracing.

    Returns:
        A sensing processor instance.
    """
    strategy = execution_config.processing.strategy
    if strategy == "sequential_stft":
        logger.info("Creating SequentialStftProcessor from resolved execution contract")
        return SequentialStftProcessor(config, carrier_frequency_hz, execution_config)
    if strategy == "coherent_cpi_fft":
        logger.info("Creating CoherentCpiFftProcessor from resolved execution contract")
        return CoherentCpiFftProcessor(config, carrier_frequency_hz, execution_config)
    if strategy == "analytic_path_doppler_oracle":
        logger.info("Creating quarantined AnalyticPathDopplerOracleProcessor")
        return AnalyticPathDopplerOracleProcessor(
            config,
            carrier_frequency_hz,
            execution_config,
        )
    raise RuntimeError(f"Unsupported resolved sensing strategy: {strategy!r}")
