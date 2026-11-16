"""Optional per-frame sensing metadata accumulation for file-output runs.

The pipeline writes output frames on the file schedule, while the ray-tracing
service may only compute fresh path data at acquisition boundaries. This module
keeps the state needed to attach sensing-derived metadata to the correct output
frame without making ``offline_pipeline`` import sensing internals unless the scenario
enables them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from shared.logging import get_logger

from ..configuration.defaults import (
    DEFAULT_SENSING_EVAL_ASSOCIATION_RANGE_GATE_M,
    DEFAULT_SENSING_EVAL_ASSOCIATION_VELOCITY_GATE_M_S,
)
from ..sensing.acquisition import ResolvedSensingExecutionConfig
from .schedule import StepScheduleEvent

logger = get_logger(__name__)

_VELOCITY_REFERENCE_POLICY = "instantaneous_relative_kinematics_v1"


@dataclass(frozen=True)
class _SensingPairIdentity:
    """Stable ordered device identity for one sensing TX/RX pair."""

    pair_id: str
    tx_sensor_id: str
    rx_sensor_id: str


@dataclass(frozen=True)
class _SensingPipelineHelpers:
    """Late-bound helper functions from ``generator.core.sensing``."""

    accumulate_gt: Callable[..., dict[str, Any]]
    build_gt_channel: Callable[..., dict[str, Any]]
    collect_fast_time_samples: Callable[..., None]
    compute_frame_association: Callable[..., dict[str, Any]]
    build_timing_metadata: Callable[..., dict[str, Any]]
    merge_timing_metadata: Callable[..., dict[str, Any]]
    build_temporal_contract: Callable[..., dict[str, Any]]
    canonical_processing_recipe_id: Callable[[Mapping[str, Any]], str]
    canonical_sensing_hash: Callable[[Any], str]
    materialize_product_specs: Callable[[dict[str, Any]], None]
    build_target_catalog: Callable[..., dict[str, Any]]
    build_target_state: Callable[..., dict[str, Any]]
    build_target_truth: Callable[..., dict[str, Any]]
    assert_sensing_contract: Callable[..., None]
    finalize_sensing_v4_publication: Callable[[dict[str, Any]], None]
    sensing_schema_version: int


def _load_sensing_pipeline_helpers() -> _SensingPipelineHelpers:
    """Load sensing helpers only for scenarios that explicitly enable them."""
    try:
        from shared.extensions.sensing import (
            SENSING_SCHEMA_VERSION,
            assert_sensing_contract,
            build_sensing_temporal_contract,
            canonical_processing_recipe_id,
            canonical_sensing_hash,
            finalize_sensing_v4_publication,
        )
        from shared.sensing.products import materialize_product_specs
        from shared.sensing.targets import (
            build_target_catalog,
            build_target_state,
            build_target_truth,
        )

        from ..sensing.hooks import (
            accumulate_sensing_gt,
            build_sensing_gt_channel,
            collect_fast_time_samples,
            compute_frame_association,
        )
        from ..sensing.timing import (
            build_sensing_timing_metadata,
            merge_sensing_timing_metadata,
        )
    except ImportError as exc:
        raise RuntimeError("Sensing output is not available in this ORCHAV installation.") from exc

    return _SensingPipelineHelpers(
        accumulate_gt=accumulate_sensing_gt,
        build_gt_channel=build_sensing_gt_channel,
        collect_fast_time_samples=collect_fast_time_samples,
        compute_frame_association=compute_frame_association,
        build_timing_metadata=build_sensing_timing_metadata,
        merge_timing_metadata=merge_sensing_timing_metadata,
        build_temporal_contract=build_sensing_temporal_contract,
        canonical_processing_recipe_id=canonical_processing_recipe_id,
        canonical_sensing_hash=canonical_sensing_hash,
        materialize_product_specs=materialize_product_specs,
        build_target_catalog=build_target_catalog,
        build_target_state=build_target_state,
        build_target_truth=build_target_truth,
        assert_sensing_contract=assert_sensing_contract,
        finalize_sensing_v4_publication=finalize_sensing_v4_publication,
        sensing_schema_version=SENSING_SCHEMA_VERSION,
    )


@dataclass
class SensingPipelineState:
    """Hold optional sensing state that spans file-output frames.

    For single-step RT, sensing data can be attached immediately to each solved
    frame. For multi-step coherent runs, the path solve happens at the interval
    start while the complete metadata belongs on the interval end; this state
    holds the pending block until that output frame is written.
    """

    enabled: bool
    frame_dt_s: float
    gt_dt_s: float
    first_step: int
    execution_config: ResolvedSensingExecutionConfig | None = None
    helpers: _SensingPipelineHelpers | None = None
    source_generation_id: str | None = None
    cir_sampling_frequency_hz: float | None = None
    range_mode: str = "bistatic"
    eval_range_gate_m: float = DEFAULT_SENSING_EVAL_ASSOCIATION_RANGE_GATE_M
    eval_velocity_gate_m_s: float = DEFAULT_SENSING_EVAL_ASSOCIATION_VELOCITY_GATE_M_S
    gt_history: list[dict[str, Any]] = field(default_factory=list)
    last_sensing_data: dict[str, Any] | None = None
    summary_sensing_events: list[dict[str, Any]] = field(default_factory=list)
    collect_summary_spectra: bool = False
    collect_summary_cartesian: bool = False
    collect_summary_tracking: bool = False
    summary_doppler_history: list[dict[str, Any]] = field(default_factory=list)
    summary_cartesian_history: list[dict[str, Any]] = field(default_factory=list)
    summary_tracking_detections: list[dict[str, Any]] = field(default_factory=list)
    summary_tracking_display_steps: dict[float, int] = field(default_factory=dict)
    summary_cartesian_scene: dict[str, Any] | None = field(default=None, repr=False)
    prev_target_ranges: dict[str, float] | None = None
    prev_gt_reference_time_s: float | None = None
    fast_time_samples: list[Any] = field(default_factory=list)
    fast_time_sample_times_s: list[float] = field(default_factory=list)
    analytical_paths: list[Any] = field(default_factory=list)
    analytical_path_sample_times_s: list[float] = field(default_factory=list)
    association_history: list[dict[str, Any]] = field(default_factory=list)
    pending_sensing: dict[str, Any] | None = None
    pending_gt_frame: dict[str, Any] | None = None
    pending_source_step: int | None = None
    pending_source_time_s: float | None = None
    pending_pair_identity: _SensingPairIdentity | None = None
    source_history: list[dict[str, Any]] = field(default_factory=list)
    source_sample_count: int = 0
    target_catalog: dict[str, Any] | None = None
    target_runtime_order: tuple[str, ...] = ()

    @classmethod
    def from_config(
        cls,
        scenario_configuration: Any,
        *,
        frame_dt_s: float,
        gt_dt_s: float,
        first_step: int,
        source_generation_id: str,
        execution_config: ResolvedSensingExecutionConfig | None,
    ) -> SensingPipelineState:
        """Create sensing state without importing sensing internals unless enabled."""
        sensing_cfg = getattr(scenario_configuration, "sensing", None) or {}
        enabled = bool(sensing_cfg.get("enabled", False))
        if enabled != (execution_config is not None):
            raise RuntimeError(
                "Scenario sensing enablement and resolved execution contract disagree"
            )
        helpers = _load_sensing_pipeline_helpers() if enabled else None
        target_entries, target_runtime_order = cls._scenario_target_catalog_entries(
            scenario_configuration
        )
        target_catalog = (
            helpers.build_target_catalog(target_entries) if helpers is not None else None
        )
        summary = getattr(scenario_configuration, "generator_summary", None) or {}
        return cls(
            enabled=enabled,
            collect_summary_spectra=bool(summary.get("enabled"))
            and "sensing" in (summary.get("create") or []),
            collect_summary_cartesian=bool(summary.get("enabled"))
            and "sensing" in (summary.get("create") or [])
            and bool(
                set((summary.get("sensing") or {}).get("figures") or [])
                & {"tracking_xy", "tracking_xyz", "tracking_cartesian"}
            ),
            collect_summary_tracking=bool(summary.get("enabled"))
            and "sensing" in (summary.get("create") or [])
            and "tracking" in ((summary.get("sensing") or {}).get("figures") or []),
            helpers=helpers,
            range_mode=str(sensing_cfg.get("range_mode", "bistatic")).lower(),
            eval_range_gate_m=float(
                sensing_cfg.get(
                    "eval_association_range_gate_m",
                    DEFAULT_SENSING_EVAL_ASSOCIATION_RANGE_GATE_M,
                )
                or DEFAULT_SENSING_EVAL_ASSOCIATION_RANGE_GATE_M
            ),
            eval_velocity_gate_m_s=float(
                sensing_cfg.get(
                    "eval_association_velocity_gate_m_s",
                    DEFAULT_SENSING_EVAL_ASSOCIATION_VELOCITY_GATE_M_S,
                )
                or DEFAULT_SENSING_EVAL_ASSOCIATION_VELOCITY_GATE_M_S
            ),
            frame_dt_s=frame_dt_s,
            gt_dt_s=gt_dt_s,
            first_step=first_step,
            execution_config=execution_config,
            source_generation_id=source_generation_id,
            cir_sampling_frequency_hz=(
                execution_config.acquisition.sample_rate_hz
                if execution_config is not None and execution_config.is_coherent_one_solve
                else None
            ),
            target_catalog=target_catalog,
            target_runtime_order=target_runtime_order,
        )

    @staticmethod
    def _scenario_target_catalog_entries(
        scenario_configuration: Any,
    ) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        """Extract authored target identity/static asset fields from a scenario."""

        actors = getattr(scenario_configuration, "actors", None)
        targets = getattr(actors, "targets", ()) if actors is not None else ()
        entries: list[dict[str, Any]] = []
        runtime_order: list[str] = []
        for target in targets or ():
            name = getattr(target, "name", None)
            if not isinstance(name, str) or not name:
                raise ValueError("Sensing target actors require exact non-empty authored names")
            asset = getattr(target, "asset", None)
            if hasattr(asset, "model_dump"):
                asset_value = asset.model_dump(mode="json")
            elif isinstance(asset, dict):
                asset_value = deepcopy(asset)
            else:
                asset_value = None
            entries.append(
                {
                    "target_id": name,
                    "display_name": name,
                    "identity_source": "scenario_actor_name",
                    "asset": asset_value,
                }
            )
            runtime_order.append(name)
        return entries, tuple(runtime_order)

    @property
    def cir_steps(self) -> int:
        """Return one-solve coherent sample count, otherwise one scene step."""

        execution = self.execution_config
        if execution is not None and execution.is_coherent_one_solve:
            return execution.acquisition.cpi_length_samples
        return 1

    def process_frame(
        self,
        frame_data: dict[str, Any],
        *,
        step_idx: int,
        is_rt_step: bool,
        schedule_event: StepScheduleEvent,
    ) -> None:
        """Attach optional sensing metadata for one scheduled output frame."""
        if not self.enabled:
            return

        sensing_block = frame_data.get("sensing")
        if is_rt_step and isinstance(sensing_block, dict):
            self._ensure_execution_metadata(sensing_block)

        if self.execution_config is not None and self.execution_config.is_coherent_one_solve:
            self._process_coherent_cpi(frame_data, step_idx, is_rt_step, schedule_event)
        elif is_rt_step and frame_data.get("sensing"):
            self._process_single_rt_frame(frame_data, step_idx)

        self._attach_timing_metadata(frame_data, step_idx, schedule_event)
        self._finalize_published_sensing(frame_data)

    def _finalize_published_sensing(self, frame_data: dict[str, Any]) -> None:
        """Strip internal aliases and validate the native v4 publication payload."""

        sensing = frame_data.get("sensing")
        if sensing is None:
            return
        helpers = self._helpers()
        if not isinstance(sensing, dict):
            helpers.assert_sensing_contract(
                {"sensing": sensing},
                require_schema_version=True,
            )
            return
        helpers.finalize_sensing_v4_publication(sensing)
        self._record_summary_sensing_event(sensing)
        if self.collect_summary_tracking and sensing["temporal"].get("result") is not None:
            # Match the source-scene steps used by native GT history. Retain
            # actual published states, including tentative reports, without
            # rerunning association or using truth to construct estimates.
            step = sensing["temporal"]["display_binding"]["scene_frame_id"]
            reference_time_s = float(sensing["temporal"]["result"]["reference_time_s"])
            self.summary_tracking_display_steps[reference_time_s] = int(step)
            self.summary_tracking_detections.extend(
                {**deepcopy(detection), "frame": step}
                for detection in sensing.get("detections") or []
                if detection.get("track_id") is not None
            )
        if self.collect_summary_cartesian:
            from shared.sensing.plotting.cartesian import cartesian_tracking_event

            event = cartesian_tracking_event(
                sensing, scene_context=self.summary_cartesian_scene or frame_data
            )
            self.summary_cartesian_scene = None
            if event is not None:
                self.summary_cartesian_history.append(event)
        if self.collect_summary_spectra:
            from shared.sensing.plotting.doppler_summary import doppler_summary_event

            event = doppler_summary_event(sensing)
            if event is not None:
                self.summary_doppler_history.append(event)

    def _record_summary_sensing_event(self, sensing: dict[str, Any]) -> None:
        """Retain bounded metadata for the first acquisition transitions.

        Numeric products remain in the authoritative frame payload.  Summary
        diagrams need only the immutable execution contract, exact acquisition
        membership, and typed product descriptors for one warm-up plus the
        first two result-bearing acquisitions.
        """

        temporal = sensing.get("temporal")
        acquisition = temporal.get("acquisition") if isinstance(temporal, dict) else None
        if not isinstance(acquisition, dict):
            return
        acquisition_id = acquisition.get("acquisition_id")
        if not isinstance(acquisition_id, str) or not acquisition_id:
            return
        if any(
            event.get("temporal", {}).get("acquisition", {}).get("acquisition_id") == acquisition_id
            for event in self.summary_sensing_events
        ):
            return

        has_result = isinstance(temporal.get("result"), dict)
        result_count = sum(
            isinstance(event.get("temporal", {}).get("result"), dict)
            for event in self.summary_sensing_events
        )
        has_warm_up = any(
            not isinstance(event.get("temporal", {}).get("result"), dict)
            for event in self.summary_sensing_events
        )
        if (has_result and result_count >= 2) or (not has_result and has_warm_up):
            return

        snapshot = {
            key: deepcopy(sensing[key])
            for key in (
                "schema_version",
                "processing_strategy",
                "config_resolved",
                "temporal",
                "products",
            )
            if key in sensing
        }
        self.summary_sensing_events.append(snapshot)

    def _ensure_execution_metadata(self, sensing: dict[str, Any]) -> None:
        """Install or verify the immutable resolved execution metadata."""

        execution = self.execution_config
        if execution is None:
            raise RuntimeError("Enabled sensing has no resolved execution contract")
        expected = execution.persisted()
        resolved = sensing.setdefault("config_resolved", {})
        if not isinstance(resolved, dict):
            raise ValueError("sensing.config_resolved must be a mapping")
        for key, expected_value in expected.items():
            existing = resolved.get(key)
            if existing is not None and existing != expected_value:
                raise ValueError(
                    f"sensing.config_resolved.{key} disagrees with the execution contract"
                )
            resolved[key] = deepcopy(expected_value)
        existing_mode = resolved.get("processing_mode")
        if existing_mode is not None and existing_mode != execution.processing.mode:
            raise ValueError(
                "sensing.config_resolved.processing_mode disagrees with the execution contract"
            )
        resolved["processing_mode"] = execution.processing.mode
        recipe = resolved.get("processing_recipe")
        if recipe is None:
            resolved["processing_recipe"] = {
                "processing": deepcopy(expected["processing"]),
                "processing_strategy": sensing.get("processing_strategy"),
                "rd_source": sensing.get("rd_source"),
                "rd_model": sensing.get("rd_model"),
                "raw_payload_policy": sensing.get("raw_payload_policy"),
            }
        elif not isinstance(recipe, dict):
            raise ValueError("sensing.config_resolved.processing_recipe must be a mapping")
        sensing["config"] = deepcopy(resolved)

    def generate_summary(
        self,
        scenario_configuration: Any,
        *,
        output_root: Path | None = None,
        strict: bool = False,
    ) -> None:
        """Generate summary figures when enough enabled-run state was collected."""
        summary = getattr(scenario_configuration, "generator_summary", None)
        summary = summary if isinstance(summary, dict) else {}
        selected = bool(summary.get("enabled", False)) and "sensing" in (
            summary.get("create", []) or []
        )
        if not self.enabled or not selected:
            return
        if not self.last_sensing_data:
            if strict:
                raise RuntimeError(
                    "Sensing summary was requested, but the run produced no usable sensing state"
                )
            return
        try:
            from shared.sensing.plotting.figures import build_tracking_history

            from ...figures.summary import maybe_generate_sensing_summary

            # Interpolated sequential truth has no exact scene step. Use the
            # result's declared display frame only as the plot's join key;
            # retain its physical midpoint and the original truth metadata.
            summary_gt = [
                (
                    {
                        **entry,
                        "step": self.summary_tracking_display_steps[entry["reference_time_s"]],
                    }
                    if "step" not in entry
                    and entry.get("reference_time_s") in self.summary_tracking_display_steps
                    else entry
                )
                for entry in self.gt_history
            ]
            last_gt = self.gt_history[-1] if self.gt_history else None
            sensing_figs = maybe_generate_sensing_summary(
                self.last_sensing_data,
                summary_gt,
                gt_state=last_gt,
                scenario_context=scenario_configuration,
                fast_time_samples=self.fast_time_samples,
                analytical_paths=self.analytical_paths or None,
                association_history=self.association_history,
                frame_dt_s=self.frame_dt_s,
                fast_time_sample_times_s=self.fast_time_sample_times_s,
                analytical_path_sample_times_s=self.analytical_path_sample_times_s,
                sensing_event_history=self.summary_sensing_events,
                doppler_history=self.summary_doppler_history,
                cartesian_history=self.summary_cartesian_history,
                tracking_history=build_tracking_history(self.summary_tracking_detections),
                output_root=output_root,
                strict=strict,
            )
            if sensing_figs:
                for path in sensing_figs:
                    logger.info("Sensing summary figure: %s", path)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            if strict:
                raise RuntimeError("Sensing summary generation failed") from exc
            logger.debug("Sensing summary generation skipped: %s", exc)

    def _helpers(self) -> _SensingPipelineHelpers:
        if self.helpers is None:
            raise RuntimeError("Sensing helpers are not loaded for this pipeline run")
        return self.helpers

    def _process_coherent_cpi(
        self,
        frame_data: dict[str, Any],
        step_idx: int,
        is_rt_step: bool,
        schedule_event: StepScheduleEvent,
    ) -> None:
        """Attach pending coherent-interval metadata at the interval boundary."""
        helpers = self._helpers()
        if is_rt_step and frame_data.get("sensing"):
            # The RT frame starts the interval. Hold its sensing block until the
            # interval end, where the full fast-time window is represented.
            source_time_s = self._required_scene_time_s(frame_data)
            pair_identity = self._resolve_pair_identity(frame_data, frame_data["sensing"])
            self.pending_sensing = frame_data.pop("sensing")
            self.pending_gt_frame = frame_data
            self.pending_source_step = step_idx
            self.pending_source_time_s = source_time_s
            self.pending_pair_identity = pair_identity

        if (
            schedule_event.is_interval_end
            and self.pending_sensing is not None
            and schedule_event.interval_size < self.cir_steps
        ):
            # Persist the observed acquisition prefix, but do not expose the
            # full-window result computed by the one-solve processor. The
            # remaining expected slots did not occur before the run ended.
            source_step = self.pending_source_step
            source_time_s = self.pending_source_time_s
            pair_identity = self.pending_pair_identity
            if source_step is None or source_time_s is None or pair_identity is None:
                raise RuntimeError(
                    "Coherent sensing acquisition lost its source scene, time, or sensor pair"
                )
            self._retain_incomplete_coherent_diagnostics(
                self.pending_sensing,
                observed_count=schedule_event.interval_size,
            )
            self._bind_incomplete_coherent_temporal(
                self.pending_sensing,
                source_step=source_step,
                source_time_s=source_time_s,
                pair_identity=pair_identity,
                observed_count=schedule_event.interval_size,
            )
            self._attach_target_contract(
                self.pending_sensing,
                self.pending_gt_frame or frame_data,
                truth_targets=None,
            )
            frame_data["sensing"] = self.pending_sensing
            logger.info(
                "Retaining incomplete last CPI acquisition (%d/%d samples)",
                schedule_event.interval_size,
                self.cir_steps,
            )
            self.pending_sensing = None
            self.pending_gt_frame = None
            self.pending_source_step = None
            self.pending_source_time_s = None
            self.pending_pair_identity = None
            return

        if not schedule_event.is_interval_end or self.pending_sensing is None:
            return

        frame_data["sensing"] = self.pending_sensing
        self.last_sensing_data = self.pending_sensing
        gt_frame = self.pending_gt_frame or frame_data
        if self.collect_summary_cartesian:
            # The complete CPI is stored at its end, but its physical result
            # and truth refer to the actual solve-start scene retained here.
            self.summary_cartesian_scene = gt_frame
        source_step = self.pending_source_step
        source_time_s = self.pending_source_time_s
        pair_identity = self.pending_pair_identity
        if source_step is None or source_time_s is None or pair_identity is None:
            raise RuntimeError(
                "Coherent sensing result lost its source scene, time, or sensor pair"
            )
        self._bind_coherent_temporal(
            self.pending_sensing,
            source_step=source_step,
            source_time_s=source_time_s,
            pair_identity=pair_identity,
        )
        self._prepare_frame_target_ids(gt_frame)
        gt_entry = helpers.accumulate_gt(
            gt_frame,
            self.range_mode,
            self._gt_elapsed_time_s(source_time_s),
            self.prev_target_ranges,
            sensing_data=self.pending_sensing,
            step=source_step,
            reference_time_s=source_time_s,
            velocity_reference_policy=_VELOCITY_REFERENCE_POLICY,
        )
        self._record_gt(gt_entry)
        truth_channel = helpers.build_gt_channel(gt_frame, gt_entry)
        self._attach_target_contract(
            self.pending_sensing,
            gt_frame,
            truth_targets=truth_channel["targets"],
            velocity_reference_policy=truth_channel["velocity_reference_policy"],
        )
        self._select_evaluation_truth(gt_entry, self.pending_sensing, gt_frame)
        from shared.sensing.angular import angular_event

        gt_entry["angular_summary"] = angular_event(self.pending_sensing)
        helpers.collect_fast_time_samples(
            self.pending_sensing,
            self.fast_time_samples,
            self.fast_time_sample_times_s,
        )
        self._record_association(gt_entry)
        self.pending_sensing = None
        self.pending_gt_frame = None
        self.pending_source_step = None
        self.pending_source_time_s = None
        self.pending_pair_identity = None

    def _process_single_rt_frame(self, frame_data: dict[str, Any], step_idx: int) -> None:
        """Attach metadata immediately for a single-step RT frame."""
        helpers = self._helpers()
        sensing = frame_data["sensing"]
        temporal_truth_frame, temporal_truth_step = self._bind_single_temporal(
            sensing,
            frame_data=frame_data,
            step_idx=step_idx,
        )
        if self.collect_summary_cartesian:
            self.summary_cartesian_scene = temporal_truth_frame
        self.last_sensing_data = sensing
        helpers.collect_fast_time_samples(
            sensing,
            self.fast_time_samples,
            self.fast_time_sample_times_s,
        )
        self._collect_analytical_path(sensing)
        if temporal_truth_frame is None:
            self._attach_target_contract(sensing, frame_data, truth_targets=None)
            return
        self._prepare_frame_target_ids(temporal_truth_frame)
        gt_entry = helpers.accumulate_gt(
            temporal_truth_frame,
            self.range_mode,
            self._gt_elapsed_time_s(float(sensing["temporal"]["truth"]["reference_time_s"])),
            self.prev_target_ranges,
            sensing_data=sensing,
            step=step_idx,
            reference_time_s=float(sensing["temporal"]["truth"]["reference_time_s"]),
            velocity_reference_policy=_VELOCITY_REFERENCE_POLICY,
        )
        if temporal_truth_step is None:
            gt_entry.pop("step", None)
        else:
            gt_entry["step"] = int(temporal_truth_step)
        self._record_gt(gt_entry)
        truth_channel = helpers.build_gt_channel(temporal_truth_frame, gt_entry)
        self._attach_target_contract(
            sensing,
            temporal_truth_frame,
            truth_targets=truth_channel["targets"],
            velocity_reference_policy=truth_channel["velocity_reference_policy"],
        )
        self._select_evaluation_truth(gt_entry, sensing, temporal_truth_frame)
        from shared.sensing.angular import angular_event

        gt_entry["angular_summary"] = angular_event(sensing)
        self._record_association(gt_entry)

    def _collect_analytical_path(self, sensing: dict[str, Any]) -> None:
        """Retain analytical path data only with its exact sample epoch."""

        analytical_paths = sensing.get("analytical_paths")
        if analytical_paths is None:
            return
        temporal = sensing.get("temporal")
        acquisition = temporal.get("acquisition") if isinstance(temporal, dict) else None
        samples = acquisition.get("samples") if isinstance(acquisition, dict) else None
        if not isinstance(samples, list):
            raise ValueError("Analytical sensing paths have no temporal acquisition membership")
        valid_times_s = [
            float(sample["sample_time_s"])
            for sample in samples
            if isinstance(sample, dict) and sample.get("valid") is True
        ]
        if not valid_times_s:
            raise ValueError("Analytical sensing paths have no valid physical sample epoch")
        sample_time_s = valid_times_s[-1]
        if (
            self.analytical_path_sample_times_s
            and sample_time_s <= self.analytical_path_sample_times_s[-1]
        ):
            raise ValueError("Analytical sensing path timestamps must be strictly increasing")
        self.analytical_paths.append(analytical_paths)
        self.analytical_path_sample_times_s.append(sample_time_s)

    def _bind_coherent_temporal(
        self,
        sensing: dict[str, Any],
        *,
        source_step: int,
        source_time_s: float,
        pair_identity: _SensingPairIdentity,
    ) -> None:
        """Bind one solved geometry epoch to its coherent sample offsets."""
        sample_rate_hz = self._required_cir_sample_rate_hz()
        solve_id = self._solve_id(source_step)
        samples = [
            self._observed_sample(
                sample_index=offset,
                sample_time_s=source_time_s + offset / sample_rate_hz,
                scene_frame_id=source_step,
                sample_offset=offset,
                solve_id=solve_id,
                phase_coherent=True,
            )
            for offset in range(self.cir_steps)
        ]
        temporal = self._build_complete_temporal(
            sensing,
            samples=samples,
            reference_time_s=source_time_s,
            reference_time_policy="solve_start",
            display_scene_frame_id=source_step,
            display_policy="source_scene_exact",
            display_reason="one_solve_coherent_geometry_epoch",
            truth={
                "reference_time_s": source_time_s,
                "scene_frame_id": source_step,
                "policy": "solve_start_state",
                "source": "generator_actor_state",
            },
            acquisition_extra={
                "sample_rate_hz": sample_rate_hz,
                "sample_count": self.cir_steps,
                "source_model": "one_solve_ordered_offsets",
            },
            pair_identity=pair_identity,
        )
        self._attach_temporal(sensing, temporal)

    def _bind_incomplete_coherent_temporal(
        self,
        sensing: dict[str, Any],
        *,
        source_step: int,
        source_time_s: float,
        pair_identity: _SensingPairIdentity,
        observed_count: int,
    ) -> None:
        """Bind a truncated final CPI without fabricating a processed result."""
        if observed_count < 1 or observed_count >= self.cir_steps:
            raise ValueError(
                "An incomplete coherent CPI must contain at least one and fewer "
                "than cir_steps observed samples"
            )
        sample_rate_hz = self._required_cir_sample_rate_hz()
        solve_id = self._solve_id(source_step)
        samples = [
            self._observed_sample(
                sample_index=offset,
                sample_time_s=source_time_s + offset / sample_rate_hz,
                scene_frame_id=source_step,
                sample_offset=offset,
                solve_id=solve_id,
                phase_coherent=True,
            )
            for offset in range(observed_count)
        ]
        samples.extend(
            {
                "sample_index": offset,
                "sample_time_s": source_time_s + offset / sample_rate_hz,
                "valid": False,
                "phase_coherent": False,
                "invalid_reason": "not_observed_before_end_of_run",
            }
            for offset in range(observed_count, self.cir_steps)
        )
        temporal = self._build_unprocessed_temporal(
            sensing,
            samples=samples,
            acquisition_extra={
                "sample_rate_hz": sample_rate_hz,
                "sample_count": self.cir_steps,
                "source_model": "one_solve_ordered_offsets",
            },
            pair_identity=pair_identity,
        )
        self._attach_temporal(sensing, temporal)

    @staticmethod
    def _retain_incomplete_coherent_diagnostics(
        sensing: dict[str, Any],
        *,
        observed_count: int,
    ) -> None:
        """Drop full-CPI results and retain only observed-prefix diagnostics."""
        import numpy as np

        slow_time = sensing.get("slow_time_matrix")
        if slow_time is not None:
            slow_time_array = np.asarray(slow_time)
            if slow_time_array.ndim < 1 or slow_time_array.shape[0] < observed_count:
                raise ValueError("slow_time_matrix must contain every observed coherent sample")
            observed_slow_time = np.array(
                slow_time_array[:observed_count],
                copy=True,
            )
            sensing["slow_time_matrix"] = observed_slow_time
            last_observed_cir = np.array(observed_slow_time[-1], copy=True)
            if "cir" in sensing:
                sensing["cir"] = last_observed_cir
            if "range_profile" in sensing:
                sensing["range_profile"] = np.abs(last_observed_cir) ** 2
            if "range_profile_raw" in sensing:
                sensing["range_profile_raw"] = np.abs(last_observed_cir) ** 2
            if "range_profile_by_pair" in sensing:
                sensing["range_profile_by_pair"] = (np.abs(last_observed_cir) ** 2)[np.newaxis, :]
            if "range_profile_raw_by_pair" in sensing:
                sensing["range_profile_raw_by_pair"] = (np.abs(last_observed_cir) ** 2)[
                    np.newaxis, :
                ]

            element_slow_time = sensing.pop("slow_time_by_pair_rx_element", None)
            if element_slow_time is not None:
                element_array = np.asarray(element_slow_time)
                if element_array.ndim != 4 or element_array.shape[2] < observed_count:
                    raise ValueError(
                        "slow_time_by_pair_rx_element must use [pair,element,time,range]"
                    )
                sensing["cir_by_pair_rx_element"] = np.array(
                    element_array[:, :, observed_count - 1, :],
                    copy=True,
                )
        else:
            # The processor's one-sample products refer to the final row of the
            # computed full window. Without the matrix, that row cannot be
            # replaced by the last schedule-observed sample.
            for key in (
                "cir",
                "range_profile",
                "range_profile_raw",
                "range_profile_kind",
                "cir_by_pair_rx_element",
                "range_profile_by_pair",
                "range_profile_raw_by_pair",
                "slow_time_by_pair_rx_element",
            ):
                sensing.pop(key, None)

        for key in tuple(sensing):
            if key.startswith("range_doppler_") or key == "range_profile_cfar_mask":
                sensing.pop(key, None)
        for key in (
            "cfar_metadata",
            "detection_filter_metadata",
            "detections",
            "detections_before_tracking",
            "detections_before_tracking_metadata",
            "gt",
            "gt_sensing",
            "gt_state",
            "range_profile_baseline",
            "temporal",
        ):
            sensing.pop(key, None)

        sensing["frame_role"] = "cpi_incomplete"
        # A precomputed display window can include schedule-unobserved samples.
        # Retain no such view on an incomplete CPI.
        if sensing.get("recording"):
            from shared.sensing.recording import DISPLAY_PRODUCT_KEYS

            for key in DISPLAY_PRODUCT_KEYS:
                sensing.pop(key, None)
            sensing["recording"]["canonical_observation"] = None
        # A projected full-CPI frequency observation cannot be advertised as
        # acquired evidence when the schedule ended before that CPI completed.
        sensing.pop("cfr_by_pair_rx_element", None)
        sensing["observed_sample_count"] = int(observed_count)
        sensing["rd_history_size"] = int(observed_count)
        sensing["rd_map_frames_used"] = int(observed_count)
        sensing["rd_ready"] = False
        sensing["rd_fresh"] = False
        sensing["rp_fresh"] = "range_profile" in sensing

    def _bind_single_temporal(
        self,
        sensing: dict[str, Any],
        *,
        frame_data: dict[str, Any],
        step_idx: int,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """Bind snapshot or sequential membership and return its truth state."""
        if self.execution_config is None:
            raise RuntimeError("Sensing temporal binding requires an execution contract")
        strategy = str(sensing.get("processing_strategy", ""))
        if strategy != self.execution_config.processing.strategy:
            raise ValueError("Persisted sensing strategy disagrees with the execution contract")
        source_time_s = self._required_scene_time_s(frame_data)
        pair_identity = self._resolve_pair_identity(frame_data, sensing)
        self._prepare_frame_target_ids(frame_data)
        source_record = {
            "scene_frame_id": int(step_idx),
            "sample_time_s": source_time_s,
            "solve_id": self._solve_id(step_idx),
            "pair_identity": pair_identity,
            "frame_data": frame_data,
        }

        if strategy != "sequential_stft":
            samples = [
                self._observed_sample(
                    sample_index=0,
                    sample_time_s=source_time_s,
                    scene_frame_id=step_idx,
                    sample_offset=0,
                    solve_id=source_record["solve_id"],
                    phase_coherent=False,
                )
            ]
            temporal = self._build_complete_temporal(
                sensing,
                samples=samples,
                reference_time_s=source_time_s,
                reference_time_policy="solve_start",
                display_scene_frame_id=step_idx,
                display_policy="source_scene_exact",
                display_reason="snapshot_result_uses_current_scene",
                truth={
                    "reference_time_s": source_time_s,
                    "scene_frame_id": step_idx,
                    "policy": "source_scene_exact",
                    "source": "generator_actor_state",
                },
                acquisition_extra={
                    "sample_count": 1,
                    "source_model": "analytic_path_snapshot",
                },
                pair_identity=pair_identity,
            )
            self._attach_temporal(sensing, temporal)
            return frame_data, step_idx

        expected_count = self.execution_config.acquisition.cpi_length_samples
        if self.source_history and any(
            record.get("pair_identity") != pair_identity for record in self.source_history
        ):
            raise ValueError(
                "Sequential sensing history cannot combine different TX/RX pair identities"
            )
        self.source_history.append(source_record)
        self.source_sample_count += 1
        if len(self.source_history) > expected_count:
            del self.source_history[: len(self.source_history) - expected_count]

        observed = list(self.source_history)
        samples = [
            self._observed_sample(
                sample_index=index,
                sample_time_s=float(record["sample_time_s"]),
                scene_frame_id=int(record["scene_frame_id"]),
                sample_offset=0,
                solve_id=str(record["solve_id"]),
                phase_coherent=False,
            )
            for index, record in enumerate(observed)
        ]
        for index in range(len(samples), expected_count):
            samples.append(
                {
                    "sample_index": index,
                    "sample_time_s": source_time_s + (index - len(observed) + 1) * self.frame_dt_s,
                    "valid": False,
                    "phase_coherent": False,
                    "invalid_reason": "not_yet_observed",
                }
            )

        if not self.execution_config.acquisition.window_due(self.source_sample_count):
            self._strip_unprocessed_result_claims(sensing)
            temporal = self._build_unprocessed_temporal(
                sensing,
                samples=samples,
                acquisition_extra={
                    "sample_interval_s": self.frame_dt_s,
                    "sample_count": expected_count,
                    "source_model": "ordered_independent_solves",
                },
                pair_identity=pair_identity,
            )
            self._attach_temporal(sensing, temporal)
            return None, None

        reference_time_s = (
            float(observed[0]["sample_time_s"]) + float(observed[-1]["sample_time_s"])
        ) / 2.0
        (
            truth_frame,
            truth_step,
            display_step,
            truth_policy,
            truth_reference_time_s,
        ) = self._sequential_truth_state(observed, reference_time_s=reference_time_s)
        temporal = self._build_complete_temporal(
            sensing,
            samples=samples,
            reference_time_s=reference_time_s,
            reference_time_policy="cpi_center",
            display_scene_frame_id=display_step,
            display_policy="nearest_at_or_before_reference",
            display_reason="bind_sliding_result_to_available_source_scene",
            truth={
                "reference_time_s": truth_reference_time_s,
                "scene_frame_id": truth_step,
                "policy": truth_policy,
                "source": "generator_actor_state",
            },
            acquisition_extra={
                "sample_interval_s": self.frame_dt_s,
                "sample_count": expected_count,
                "source_model": "ordered_independent_solves",
            },
            pair_identity=pair_identity,
        )
        self._attach_temporal(sensing, temporal)
        return truth_frame, truth_step

    def _observed_sample(
        self,
        *,
        sample_index: int,
        sample_time_s: float,
        scene_frame_id: int,
        sample_offset: int,
        solve_id: str,
        phase_coherent: bool,
    ) -> dict[str, Any]:
        generation_id = self._required_generation_id()
        return {
            "sample_index": int(sample_index),
            "sample_time_s": float(sample_time_s),
            "valid": True,
            "phase_coherent": bool(phase_coherent),
            "source": {
                "source_generation_id": generation_id,
                "scene_frame_id": int(scene_frame_id),
                "sample_offset": int(sample_offset),
                "solve_id": solve_id,
            },
        }

    @staticmethod
    def _strip_unprocessed_result_claims(sensing: dict[str, Any]) -> None:
        """Remove products and metadata that would claim a processed result."""

        for key in tuple(sensing):
            if key.startswith("range_doppler_") or key == "range_profile_cfar_mask":
                sensing.pop(key, None)
        for key in (
            "cfar_metadata",
            "detection_filter_metadata",
            "detections",
            "detections_before_tracking",
            "detections_before_tracking_metadata",
        ):
            sensing.pop(key, None)

    def _build_complete_temporal(
        self,
        sensing: dict[str, Any],
        *,
        samples: list[dict[str, Any]],
        reference_time_s: float,
        reference_time_policy: str,
        display_scene_frame_id: int | None,
        display_policy: str,
        display_reason: str,
        truth: dict[str, Any],
        acquisition_extra: dict[str, Any],
        pair_identity: _SensingPairIdentity,
    ) -> dict[str, Any]:
        helpers = self._helpers()
        acquisition_config_id = self._acquisition_config_id(
            sensing,
            acquisition_extra=acquisition_extra,
            pair_identity=pair_identity,
        )
        return helpers.build_temporal_contract(
            clock_id=self._required_generation_id(),
            acquisition_id=self._acquisition_id(
                acquisition_config_id=acquisition_config_id,
                samples=samples,
                pair_identity=pair_identity,
            ),
            acquisition_config_id=acquisition_config_id,
            pair_id=pair_identity.pair_id,
            tx_sensor_id=pair_identity.tx_sensor_id,
            rx_sensor_id=pair_identity.rx_sensor_id,
            samples=samples,
            processing_run_id=self._required_generation_id(),
            processing_recipe_id=self._processing_recipe_id(sensing),
            reference_time_s=reference_time_s,
            reference_time_policy=reference_time_policy,
            display_scene_frame_id=display_scene_frame_id,
            display_policy=display_policy,
            display_reason=display_reason,
            completeness="complete",
            truth=truth,
        )

    def _build_unprocessed_temporal(
        self,
        sensing: dict[str, Any],
        *,
        samples: list[dict[str, Any]],
        acquisition_extra: dict[str, Any],
        pair_identity: _SensingPairIdentity,
    ) -> dict[str, Any]:
        helpers = self._helpers()
        acquisition_config_id = self._acquisition_config_id(
            sensing,
            acquisition_extra=acquisition_extra,
            pair_identity=pair_identity,
        )
        return helpers.build_temporal_contract(
            clock_id=self._required_generation_id(),
            acquisition_id=self._acquisition_id(
                acquisition_config_id=acquisition_config_id,
                samples=samples,
                pair_identity=pair_identity,
            ),
            acquisition_config_id=acquisition_config_id,
            pair_id=pair_identity.pair_id,
            tx_sensor_id=pair_identity.tx_sensor_id,
            rx_sensor_id=pair_identity.rx_sensor_id,
            samples=samples,
        )

    def _attach_temporal(
        self,
        sensing: dict[str, Any],
        temporal: dict[str, Any],
    ) -> None:
        helpers = self._helpers()
        sensing["schema_version"] = helpers.sensing_schema_version
        sensing["temporal"] = temporal
        helpers.materialize_product_specs(sensing)

    def _acquisition_config_id(
        self,
        sensing: dict[str, Any],
        *,
        acquisition_extra: dict[str, Any],
        pair_identity: _SensingPairIdentity,
    ) -> str:
        helpers = self._helpers()
        resolved = sensing.get("config_resolved")
        acquisition = resolved.get("acquisition") if isinstance(resolved, dict) else None
        topology = resolved.get("topology") if isinstance(resolved, dict) else None
        if not isinstance(acquisition, dict) or not isinstance(topology, dict):
            raise ValueError("Sensing config_resolved lacks acquisition/topology semantics")
        return helpers.canonical_sensing_hash(
            {
                "acquisition": acquisition,
                "topology": topology,
                "pair": {
                    "pair_id": pair_identity.pair_id,
                    "tx_sensor_id": pair_identity.tx_sensor_id,
                    "rx_sensor_id": pair_identity.rx_sensor_id,
                },
            }
        )

    def _processing_recipe_id(self, sensing: dict[str, Any]) -> str:
        """Return the identity of the immutable processing recipe itself.

        Result identity and acquisition/pair identity already carry occurrence
        and source context.  Wrapping the recipe in mutable projection fields
        made equivalent recipes hash differently across generator and offline
        reprocessing paths, which prevented deterministic A/B replay.
        """
        helpers = self._helpers()
        resolved = sensing.get("config_resolved")
        recipe = resolved.get("processing_recipe") if isinstance(resolved, dict) else None
        if not isinstance(recipe, dict):
            raise ValueError("Sensing config_resolved lacks a processing_recipe")
        return helpers.canonical_processing_recipe_id(recipe)

    def _acquisition_id(
        self,
        *,
        acquisition_config_id: str,
        samples: list[dict[str, Any]],
        pair_identity: _SensingPairIdentity,
    ) -> str:
        helpers = self._helpers()
        digest = helpers.canonical_sensing_hash(
            {
                "acquisition_config_id": acquisition_config_id,
                "pair_id": pair_identity.pair_id,
                "samples": samples,
            }
        )
        return str(uuid5(UUID(self._required_generation_id()), digest))

    def _required_generation_id(self) -> str:
        generation_id = self.source_generation_id
        if generation_id is None:
            raise RuntimeError("Sensing temporal binding requires a source generation UUID")
        try:
            canonical = str(UUID(generation_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("source_generation_id must be a canonical UUID string") from exc
        if canonical != generation_id:
            raise ValueError("source_generation_id must be a canonical UUID string")
        return generation_id

    def _required_cir_sample_rate_hz(self) -> float:
        sample_rate_hz = self.cir_sampling_frequency_hz
        if sample_rate_hz is None or not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError(
                "cir_sampling_frequency_hz must be finite and positive for coherent sensing"
            )
        return sample_rate_hz

    def _solve_id(self, scene_frame_id: int) -> str:
        return f"{self._required_generation_id()}:{int(scene_frame_id)}"

    @staticmethod
    def _required_scene_time_s(frame_data: dict[str, Any]) -> float:
        """Return the explicit physical scene time attached by ray tracing."""

        value = frame_data.get("scene_time_s")
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(
                "Sensing source frames require an explicit finite scene_time_s timestamp"
            )
        timestamp_s = float(value)
        if not math.isfinite(timestamp_s):
            raise ValueError(
                "Sensing source frames require an explicit finite scene_time_s timestamp"
            )
        return timestamp_s

    def _resolve_pair_identity(
        self,
        frame_data: dict[str, Any],
        sensing: dict[str, Any],
    ) -> _SensingPairIdentity:
        """Resolve RX-major/TX-minor ``pair_index`` to stable device names."""

        raw_pair_index = sensing.get("pair_index")
        if not isinstance(raw_pair_index, Integral) or isinstance(raw_pair_index, bool):
            raise ValueError("Sensing pair_index must be an explicit integer")
        pair_index = int(raw_pair_index)
        if pair_index < 0:
            raise ValueError("Sensing pair_index must be nonnegative")

        tx_sensor_ids = self._stable_sensor_ids(frame_data.get("tx_list"), role="TX")
        rx_sensor_ids = self._stable_sensor_ids(frame_data.get("rx_list"), role="RX")
        execution = self.execution_config
        if execution is None:
            raise RuntimeError("Sensing pair binding requires an execution contract")
        topology = execution.topology
        if (
            len(tx_sensor_ids) != topology.tx_device_count
            or len(rx_sensor_ids) != topology.rx_device_count
        ):
            raise ValueError(
                "Sensing frame topology disagrees with the resolved execution contract"
            )
        if pair_index != topology.pair_index:
            raise ValueError("Sensing pair_index disagrees with the resolved execution contract")
        pair_count = len(tx_sensor_ids) * len(rx_sensor_ids)
        if pair_index >= pair_count:
            raise ValueError(
                f"Sensing pair_index {pair_index} is out of range for {len(tx_sensor_ids)} "
                f"TX and {len(rx_sensor_ids)} RX sensors"
            )

        tx_sensor_id = tx_sensor_ids[pair_index % len(tx_sensor_ids)]
        rx_sensor_id = rx_sensor_ids[pair_index // len(tx_sensor_ids)]
        pair_id = self._helpers().canonical_sensing_hash(
            {
                "tx_sensor_id": tx_sensor_id,
                "rx_sensor_id": rx_sensor_id,
            }
        )
        return _SensingPairIdentity(
            pair_id=pair_id,
            tx_sensor_id=tx_sensor_id,
            rx_sensor_id=rx_sensor_id,
        )

    @staticmethod
    def _stable_sensor_ids(devices: Any, *, role: str) -> tuple[str, ...]:
        """Return unique non-empty configured names without indexed fallbacks."""

        if not isinstance(devices, (list, tuple)) or not devices:
            raise ValueError(f"Sensing source frames require a non-empty {role.lower()}_list")
        sensor_ids: list[str] = []
        for index, device in enumerate(devices):
            raw_name = getattr(device, "name", None)
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(
                    f"Sensing {role} sensor at index {index} requires a stable non-empty name"
                )
            sensor_ids.append(raw_name.strip())
        if len(set(sensor_ids)) != len(sensor_ids):
            raise ValueError(f"Sensing {role} sensor names must be unique")
        return tuple(sensor_ids)

    @staticmethod
    def _sequential_truth_state(
        observed: list[dict[str, Any]],
        *,
        reference_time_s: float,
    ) -> tuple[dict[str, Any], int | None, int, str, float]:
        """Return exact or linearly interpolated actor state at CPI center."""
        reference_time_s = float(reference_time_s)
        if not math.isfinite(reference_time_s):
            raise ValueError("Sequential truth reference_time_s must be finite")
        lower = max(
            (record for record in observed if record["sample_time_s"] <= reference_time_s),
            key=lambda record: record["sample_time_s"],
        )
        upper = min(
            (record for record in observed if record["sample_time_s"] >= reference_time_s),
            key=lambda record: record["sample_time_s"],
        )
        display_step = int(lower["scene_frame_id"])
        lower_time = float(lower["sample_time_s"])
        upper_time = float(upper["sample_time_s"])
        if not math.isfinite(lower_time) or not math.isfinite(upper_time):
            raise ValueError("Sequential truth source epochs must be finite")
        if lower is upper or math.isclose(
            lower_time,
            upper_time,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return (
                lower["frame_data"],
                display_step,
                display_step,
                "source_scene_exact",
                lower_time,
            )

        import numpy as np

        if upper_time <= lower_time:
            raise ValueError("Sequential truth source epochs must increase strictly")
        alpha = (reference_time_s - lower_time) / (upper_time - lower_time)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Sequential truth reference epoch lies outside source bracket")
        interpolated = dict(lower["frame_data"])
        for key in ("tx_positions_snapshot", "rx_positions_snapshot"):
            lower_value = lower["frame_data"].get(key)
            upper_value = upper["frame_data"].get(key)
            if lower_value is None or upper_value is None:
                raise ValueError(f"Sequential truth interpolation requires exact {key}")
            lower_array = np.asarray(lower_value, dtype=np.float64)
            upper_array = np.asarray(upper_value, dtype=np.float64)
            if (
                lower_array.shape != upper_array.shape
                or not np.all(np.isfinite(lower_array))
                or not np.all(np.isfinite(upper_array))
            ):
                raise ValueError(f"Sequential truth interpolation requires aligned finite {key}")
            interpolated[key] = lower_array + alpha * (upper_array - lower_array)

        for role in ("tx", "rx"):
            key = f"{role}_velocities_snapshot"
            lower_value = lower["frame_data"].get(key)
            upper_value = upper["frame_data"].get(key)
            if lower_value is None or upper_value is None:
                interpolated.pop(key, None)
                continue
            lower_array = np.asarray(lower_value, dtype=np.float64)
            upper_array = np.asarray(upper_value, dtype=np.float64)
            expected_shape = interpolated[f"{role}_positions_snapshot"].shape
            if (
                lower_array.shape != expected_shape
                or upper_array.shape != expected_shape
                or not np.all(np.isfinite(lower_array))
                or not np.all(np.isfinite(upper_array))
            ):
                raise ValueError(f"Sequential truth interpolation requires aligned finite {key}")
            interpolated[key] = lower_array + alpha * (upper_array - lower_array)

        def target_ids(frame: dict[str, Any], count: int) -> tuple[str, ...] | None:
            internal = frame.get("_sensing_target_ids")
            if isinstance(internal, (list, tuple)) and len(internal) == count:
                if any(not isinstance(value, str) or not value for value in internal):
                    return None
                return tuple(internal)
            metadata = frame.get("targets_metadata")
            if not isinstance(metadata, list) or len(metadata) != count:
                return None
            names = tuple(row.get("name") if isinstance(row, dict) else None for row in metadata)
            if any(not isinstance(name, str) or not name for name in names):
                return None
            return names  # type: ignore[return-value]

        lower_positions = lower["frame_data"].get("target_positions_snapshot")
        upper_positions = upper["frame_data"].get("target_positions_snapshot")
        if lower_positions is None or upper_positions is None:
            # A lifecycle boundary cannot be interpolated. Retain the exact
            # latest state at or before the requested reference epoch and bind
            # truth to that source epoch rather than relabeling it as CPI-center truth.
            return (
                lower["frame_data"],
                display_step,
                display_step,
                "source_scene_at_or_before_lifecycle_discontinuity",
                lower_time,
            )
        lower_position_array = np.asarray(lower_positions, dtype=np.float64)
        upper_position_array = np.asarray(upper_positions, dtype=np.float64)
        lower_position_ids = target_ids(lower["frame_data"], len(lower_position_array))
        upper_position_ids = target_ids(upper["frame_data"], len(upper_position_array))
        if lower_position_ids is None or upper_position_ids is None:
            raise ValueError("Sequential target interpolation requires exact target IDs")
        if set(lower_position_ids) != set(upper_position_ids):
            return (
                lower["frame_data"],
                display_step,
                display_step,
                "source_scene_at_or_before_lifecycle_discontinuity",
                lower_time,
            )

        for key in ("target_positions_snapshot", "target_velocities_snapshot"):
            lower_value = lower["frame_data"].get(key)
            upper_value = upper["frame_data"].get(key)
            if lower_value is None or upper_value is None:
                if key == "target_velocities_snapshot":
                    interpolated.pop(key, None)
                    continue
                raise ValueError(f"Sequential target interpolation requires exact {key}")
            lower_array = np.asarray(lower_value, dtype=np.float64)
            upper_array = np.asarray(upper_value, dtype=np.float64)
            if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
                raise ValueError(f"Sequential target interpolation requires finite {key}")
            lower_ids = target_ids(lower["frame_data"], len(lower_array))
            upper_ids = target_ids(upper["frame_data"], len(upper_array))
            if lower_ids is None or upper_ids is None:
                raise ValueError(
                    "Sequential target interpolation requires exact target IDs on every row"
                )
            if set(lower_ids) != set(upper_ids):
                if key == "target_velocities_snapshot":
                    interpolated.pop(key, None)
                    continue
                raise ValueError("Sequential target position membership changed unexpectedly")
            upper_by_id = {
                target_id: upper_array[index] for index, target_id in enumerate(upper_ids)
            }
            upper_aligned = np.asarray([upper_by_id[target_id] for target_id in lower_ids])
            if lower_array.shape != upper_aligned.shape:
                raise ValueError(f"Sequential target interpolation requires aligned {key}")
            interpolated[key] = lower_array + alpha * (upper_aligned - lower_array)
        lower_scene_id = int(lower["scene_frame_id"])
        upper_scene_id = int(upper["scene_frame_id"])
        interpolated["frame_idx"] = (
            f"interpolated:{lower_scene_id}:{upper_scene_id}:{reference_time_s.hex()}"
        )
        return (
            interpolated,
            None,
            display_step,
            "linear_interpolation_between_source_scenes",
            reference_time_s,
        )

    def _prepare_frame_target_ids(self, frame_data: dict[str, Any]) -> tuple[str, ...]:
        """Bind source-array rows to authored IDs at the producer boundary."""

        import numpy as np

        positions = frame_data.get("target_positions_snapshot")
        target_count = 0 if positions is None else int(np.asarray(positions).shape[0])
        metadata = frame_data.get("targets_metadata")
        metadata_ids: list[str] = []
        if isinstance(metadata, list) and len(metadata) == target_count:
            for row in metadata:
                name = row.get("name") if isinstance(row, dict) else None
                if not isinstance(name, str) or not name:
                    metadata_ids = []
                    break
                metadata_ids.append(name)
        if len(metadata_ids) == target_count:
            target_ids = tuple(metadata_ids)
        elif len(self.target_runtime_order) == target_count:
            target_ids = self.target_runtime_order
        elif target_count == 0:
            target_ids = ()
        else:
            raise ValueError(
                "Sensing target rows cannot be bound to exact authored target identities"
            )
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("Sensing target row identities must be unique")

        helpers = self._helpers()
        catalog = self.target_catalog
        catalog_targets = catalog.get("targets") if isinstance(catalog, dict) else None
        if isinstance(catalog_targets, list) and not catalog_targets and target_ids:
            # Focused programmatic callers may not carry a typed scenario
            # object.  Exact frame metadata names are still authored IDs, so
            # establish the immutable catalog on the first sensing event.
            self.target_catalog = helpers.build_target_catalog(
                [
                    {
                        "target_id": target_id,
                        "display_name": target_id,
                        "identity_source": "frame_target_metadata_name",
                    }
                    for target_id in target_ids
                ]
            )
            self.target_runtime_order = target_ids
            catalog_targets = self.target_catalog["targets"]
        known_ids = {
            target.get("target_id") for target in catalog_targets or [] if isinstance(target, dict)
        }
        unknown = set(target_ids) - known_ids
        if unknown:
            raise ValueError(f"Sensing target rows contain IDs absent from catalog: {unknown}")
        frame_data["_sensing_target_ids"] = list(target_ids)
        return target_ids

    def _attach_target_contract(
        self,
        sensing: dict[str, Any],
        frame_data: dict[str, Any],
        *,
        truth_targets: list[dict[str, Any]] | None,
        velocity_reference_policy: str | None = None,
    ) -> None:
        """Attach canonical catalog/state/truth without positional aliases."""

        import numpy as np

        target_ids = self._prepare_frame_target_ids(frame_data)
        catalog = self.target_catalog
        if not isinstance(catalog, dict):
            raise RuntimeError("Enabled sensing has no target catalog")
        animation_by_id = {
            row.get("name"): row["animated_asset"]
            for row in frame_data.get("targets_metadata", [])
            if isinstance(row, dict) and isinstance(row.get("animated_asset"), dict)
        }
        if animation_by_id:
            catalog_rows = deepcopy(catalog["targets"])
            for catalog_row in catalog_rows:
                animation = animation_by_id.get(catalog_row["target_id"])
                if animation is not None:
                    metadata = animation["catalog"]
                    previous = catalog_row.get("animated_asset")
                    if previous is not None and previous != metadata:
                        raise ValueError(
                            "Animated asset identity changed within the target catalog"
                        )
                    catalog_row["animated_asset"] = deepcopy(metadata)
            catalog = self._helpers().build_target_catalog(catalog_rows)
            self.target_catalog = catalog
        temporal = sensing.get("temporal")
        acquisition = temporal.get("acquisition") if isinstance(temporal, dict) else None
        temporal_truth = temporal.get("truth") if isinstance(temporal, dict) else None
        if isinstance(temporal_truth, dict):
            reference_time_s = float(temporal_truth["reference_time_s"])
        elif isinstance(acquisition, dict):
            reference_time_s = float(acquisition["default_reference_time_s"])
        else:
            raise ValueError("Target state requires temporal acquisition identity")

        positions = np.asarray(
            frame_data.get("target_positions_snapshot", np.empty((0, 3))),
            dtype=np.float64,
        )
        velocities_raw = frame_data.get("target_velocities_snapshot")
        orientations_raw = frame_data.get("target_orientations_snapshot")
        velocities = (
            None if velocities_raw is None else np.asarray(velocities_raw, dtype=np.float64)
        )
        orientations = (
            None if orientations_raw is None else np.asarray(orientations_raw, dtype=np.float64)
        )
        row_by_id = {target_id: index for index, target_id in enumerate(target_ids)}
        state_targets: list[dict[str, Any]] = []
        for catalog_target in catalog.get("targets", []):
            target_id = catalog_target["target_id"]
            row = row_by_id.get(target_id)
            if row is None:
                state_targets.append(
                    {
                        "target_id": target_id,
                        "lifecycle": "inactive",
                        "valid": False,
                        "position_m": None,
                        "orientation_rad": None,
                        "velocity_m_s": None,
                        "invalid_reason": None,
                    }
                )
                continue
            position_valid = row < len(positions) and bool(np.all(np.isfinite(positions[row])))
            state_targets.append(
                {
                    "target_id": target_id,
                    "lifecycle": "active",
                    "valid": position_valid,
                    "position_m": positions[row].tolist() if position_valid else None,
                    "orientation_rad": (
                        orientations[row].tolist()
                        if position_valid
                        and orientations is not None
                        and row < len(orientations)
                        and np.all(np.isfinite(orientations[row]))
                        else None
                    ),
                    "velocity_m_s": (
                        velocities[row].tolist()
                        if position_valid
                        and velocities is not None
                        and row < len(velocities)
                        and np.all(np.isfinite(velocities[row]))
                        else None
                    ),
                    "invalid_reason": (None if position_valid else "target_position_not_finite"),
                }
            )

        for state_target in state_targets:
            animation = animation_by_id.get(state_target["target_id"])
            if animation is not None and state_target["lifecycle"] == "active":
                state_target["animated_asset"] = deepcopy(animation["state"])

        helpers = self._helpers()
        sensing["target_catalog"] = deepcopy(catalog)
        sensing["target_state"] = helpers.build_target_state(
            catalog,
            scene_frame_id=frame_data.get("frame_idx", 0),
            reference_time_s=reference_time_s,
            targets=state_targets,
        )
        if truth_targets is not None:
            truth_by_id = {
                target.get("target_id"): target
                for target in truth_targets
                if isinstance(target, dict) and isinstance(target.get("target_id"), str)
            }
            if len(truth_by_id) != len(truth_targets):
                raise ValueError("Canonical sensing truth contains duplicate or missing target IDs")
            pair_id = acquisition.get("pair_id") if isinstance(acquisition, dict) else None
            if not isinstance(pair_id, str) or not pair_id:
                raise ValueError("Canonical sensing truth requires exact acquisition pair identity")
            canonical_truth_targets: list[dict[str, Any]] = []
            for state_target in state_targets:
                target_id = state_target["target_id"]
                truth_target = deepcopy(state_target)
                observable = truth_by_id.get(target_id)
                pair_observables: list[dict[str, Any]] = []
                if truth_target["lifecycle"] == "active" and truth_target["valid"]:
                    if observable is None:
                        if velocity_reference_policy is not None:
                            raise ValueError(
                                "Versioned truth requires source evidence for every active target"
                            )
                        pair_observables.append(
                            {
                                "pair_id": pair_id,
                                "bistatic_path_length_m": None,
                                "monostatic_equivalent_range_m": None,
                                "path_closing_rate_m_s": None,
                                "monostatic_equivalent_velocity_m_s": None,
                                "bistatic_bisector_velocity_m_s": None,
                                "velocity_valid": False,
                                "velocity_derivation_source": "unavailable",
                                "velocity_reference_dt_s": None,
                                "undefined_reason": "truth_observable_unavailable",
                            }
                        )
                    else:
                        stored_range_m = float(observable["range_m"])
                        bistatic_path_length_m = (
                            2.0 * stored_range_m
                            if self.range_mode == "monostatic"
                            else stored_range_m
                        )
                        bisector_velocity = observable.get("bistatic_bisector_velocity_m_s")
                        bisector_velocity = (
                            None if bisector_velocity is None else float(bisector_velocity)
                        )
                        velocity_valid = observable.get("velocity_valid") is True
                        velocity_source = observable.get("velocity_derivation_source")
                        velocity_reference_dt_s = observable.get("velocity_reference_dt_s")
                        if velocity_valid:
                            path_closing_rate = float(observable["path_closing_rate_m_s"])
                            monostatic_velocity = float(
                                observable["monostatic_equivalent_velocity_m_s"]
                            )
                            undefined_reason = (
                                None
                                if bisector_velocity is not None
                                else "bistatic_bisector_undefined_geometry"
                            )
                        else:
                            path_closing_rate = None
                            monostatic_velocity = None
                            bisector_velocity = None
                            undefined_reason = observable.get("velocity_unavailable_reason")
                            if not isinstance(undefined_reason, str) or not undefined_reason:
                                raise ValueError(
                                    "Unavailable canonical truth velocity requires a reason"
                                )
                        pair_observables.append(
                            {
                                "pair_id": pair_id,
                                "bistatic_path_length_m": bistatic_path_length_m,
                                "monostatic_equivalent_range_m": (0.5 * bistatic_path_length_m),
                                "path_closing_rate_m_s": path_closing_rate,
                                "monostatic_equivalent_velocity_m_s": monostatic_velocity,
                                "bistatic_bisector_velocity_m_s": bisector_velocity,
                                "velocity_valid": velocity_valid,
                                "velocity_derivation_source": velocity_source,
                                "velocity_reference_dt_s": velocity_reference_dt_s,
                                "undefined_reason": undefined_reason,
                            }
                        )
                        if velocity_reference_policy is not None:
                            pair_observables[-1]["velocity_reference_evidence"] = deepcopy(
                                observable["velocity_reference_evidence"]
                            )
                if pair_observables and truth_target.get("position_m") is not None:
                    from shared.sensing.angular import angular_truth_fields

                    pair_observables[0].update(
                        angular_truth_fields(frame_data, truth_target["position_m"])
                    )
                truth_target["pair_observables"] = pair_observables
                canonical_truth_targets.append(truth_target)
            sensing["truth"] = helpers.build_target_truth(
                catalog,
                targets=canonical_truth_targets,
                velocity_reference_policy=velocity_reference_policy,
            )

    @staticmethod
    def _select_evaluation_truth(
        gt_entry: dict[str, Any], sensing: dict[str, Any], frame_data: dict[str, Any]
    ) -> None:
        """Use the same declared reference for native summaries and replay.

        Canonical truth remains the source observation. Evaluation history also
        retains its recorded velocities when a signal-model reference is selected.
        References join by target ID while preserving the native array order.
        """
        if not gt_entry.get("evaluation_reference_valid"):
            return
        from shared.sensing.evaluation_truth import extract_evaluation_truth

        selected = extract_evaluation_truth(sensing, frame_data)
        if selected is None:
            raise ValueError("Evaluation truth must preserve the exact native target identities")
        selected_ids = selected["target_ids"]
        native_ids = gt_entry["target_ids"]
        if (
            len(set(selected_ids)) != len(selected_ids)
            or len(set(native_ids)) != len(native_ids)
            or set(selected_ids) != set(native_ids)
        ):
            raise ValueError("Evaluation truth must preserve the exact native target identities")
        selected_row_by_id = {target_id: row for row, target_id in enumerate(selected_ids)}
        selected_rows = [selected_row_by_id[target_id] for target_id in native_ids]
        for key in (
            "target_speeds",
            "target_velocity_valid",
            "target_velocity_sources",
            "target_velocity_reference_dt_s",
            "target_velocity_unavailable_reasons",
            "target_velocity_references",
            "recorded_target_speeds",
            "recorded_target_velocity_sources",
            "recorded_target_velocity_valid",
            "recorded_target_velocity_reference_dt_s",
        ):
            gt_entry[key] = [deepcopy(selected[key][row]) for row in selected_rows]
        gt_entry["evaluation_reference_protocol"] = selected["evaluation_reference_protocol"]
        for key in (
            "target_path_closing_rates_m_s",
            "target_monostatic_equivalent_velocities_m_s",
            "target_bistatic_bisector_velocities_m_s",
        ):
            gt_entry[f"recorded_{key}"] = gt_entry.pop(key)

    def _record_gt(self, gt_entry: dict[str, Any]) -> None:
        """Store ground-truth history needed by summaries and next-frame ranges."""
        reference_time_s = gt_entry.get("reference_time_s")
        if not isinstance(reference_time_s, Real) or isinstance(reference_time_s, bool):
            raise ValueError("Sensing GT reference_time_s must be finite")
        timestamp_s = float(reference_time_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("Sensing GT reference_time_s must be finite")
        target_ids = gt_entry.get("target_ids")
        target_ranges = gt_entry.get("target_ranges")
        if not isinstance(target_ids, list) or not isinstance(target_ranges, list):
            raise ValueError("Sensing GT target IDs and ranges must be aligned lists")
        if len(target_ids) != len(target_ranges):
            raise ValueError("Sensing GT target IDs and ranges must be aligned lists")
        if any(not isinstance(target_id, str) or not target_id for target_id in target_ids):
            raise ValueError("Sensing GT requires exact non-empty target IDs")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("Sensing GT target IDs must be unique")
        prior_ranges: dict[str, float] = {}
        for target_id, target_range in zip(target_ids, target_ranges, strict=True):
            range_m = float(target_range)
            if not math.isfinite(range_m) or range_m < 0.0:
                raise ValueError("Sensing GT target ranges must be finite and non-negative")
            prior_ranges[target_id] = range_m

        self.gt_history.append(gt_entry)
        self.prev_gt_reference_time_s = timestamp_s
        self.prev_target_ranges = prior_ranges

    def _gt_elapsed_time_s(self, reference_time_s: float) -> float | None:
        """Return the exact elapsed time from the preceding retained GT epoch.

        No elapsed time exists at the first retained epoch, so ``None`` is
        returned and finite-difference velocity is unavailable. Every later
        estimate uses the temporal contract's physical reference epochs.
        """

        timestamp_s = float(reference_time_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("Sensing GT reference_time_s must be finite")
        if self.prev_gt_reference_time_s is None:
            return None
        elapsed_s = timestamp_s - self.prev_gt_reference_time_s
        if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
            raise ValueError(
                "Sensing GT reference_time_s must increase strictly "
                f"(previous={self.prev_gt_reference_time_s}, current={timestamp_s})"
            )
        return elapsed_s

    def _record_association(self, gt_entry: dict[str, Any]) -> None:
        """Store one frame of detection/ground-truth association metadata."""
        helpers = self._helpers()
        self.association_history.append(
            helpers.compute_frame_association(
                gt_entry,
                range_gate_m=self.eval_range_gate_m,
                velocity_gate_m_s=self.eval_velocity_gate_m_s,
            )
        )

    def _attach_timing_metadata(
        self,
        frame_data: dict[str, Any],
        step_idx: int,
        schedule_event: StepScheduleEvent,
    ) -> None:
        """Merge schedule-derived timing metadata into an existing sensing block."""
        sensing_block = frame_data.get("sensing")
        if not isinstance(sensing_block, dict):
            return
        helpers = self._helpers()
        config_resolved = sensing_block.get("config_resolved")
        if not isinstance(config_resolved, dict):
            raise ValueError(
                "Native sensing timing requires config_resolved; config and outer "
                "prf_hz are not fallback sources"
            )
        prf_raw = config_resolved.get("prf_hz")
        if isinstance(prf_raw, bool) or not isinstance(prf_raw, Real):
            raise ValueError(
                "Native sensing timing requires finite positive config_resolved.prf_hz"
            )
        prf_hz = float(prf_raw)
        if not math.isfinite(prf_hz) or prf_hz <= 0.0:
            raise ValueError(
                "Native sensing timing requires finite positive config_resolved.prf_hz"
            )
        timing_meta = helpers.build_timing_metadata(
            step_idx=step_idx,
            first_step=self.first_step,
            cir_steps=self.cir_steps,
            frame_dt_s=self.frame_dt_s,
            prf_hz=prf_hz,
        )
        timing_meta["cpi_index"] = int(schedule_event.interval_index)
        timing_meta["pri_index_in_cpi"] = int(schedule_event.sample_index_in_interval)
        timing_meta["timing_role"] = (
            "cpi_end" if self.cir_steps > 1 and schedule_event.is_interval_end else "pri_sample"
        )
        sensing_block["timing"] = helpers.merge_timing_metadata(
            timing_meta,
            sensing_block.get("timing"),
        )
