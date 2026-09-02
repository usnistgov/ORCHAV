#!/usr/bin/env python3
"""Live generator gRPC server.

This service is the live counterpart to ``file_server``. It computes or receives
raw generator frame dictionaries, keeps those raw dictionaries in
``GeneratorFrameCache`` when useful, normalizes them through
``standard_mpc_frame_from_raw``, and sends only ``StandardMPCFrame`` protobufs
to clients.

One controlling stream owns parameter/object updates and frame requests for a
server generation epoch. A second controller is rejected rather than sharing
mutable solver state. The service sends bounded protobuf messages through
insecure gRPC; loopback is the default and non-loopback binding is for
explicitly trusted networks only.
"""

import logging
import math
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast, overload

import grpc

from shared.frames.protobuf import standard_mpc_frame_to_proto
from shared.grpc_transport import (
    DEFAULT_GRPC_BIND_HOST,
    DEFAULT_GRPC_SHUTDOWN_GRACE_S,
    GRPC_MAX_MESSAGE_BYTES,
    GRPC_MESSAGE_OPTIONS,
    bind_grpc_server,
    format_grpc_endpoint,
    is_loopback_grpc_host,
)
from shared.logging import get_logger

from ...core.propagation import (
    LiveActorCategory,
    LiveActorOverride,
    LiveOverrideMap,
    apply_target_scale_overrides,
    category_from_value,
    normalize_live_overrides,
)
from ...core.runtime import SimulationObjects
from ..frames.conversion import standard_mpc_frame_from_raw
from .cache import GeneratorFrameCache
from .live_scene_update import LiveSceneXmlStager

# Import the gRPC generated code
try:
    from shared.protos import (
        visualizer_pb2 as _visualizer_pb2,
    )
    from shared.protos import (
        visualizer_pb2_grpc as _visualizer_pb2_grpc,
    )

    visualizer_pb2: Any = _visualizer_pb2
    visualizer_pb2_grpc: Any = _visualizer_pb2_grpc
except ImportError as e:
    logging.error(f"Could not import gRPC generated code: {e}")
    logging.error("Make sure the protobuf files are generated.")
    raise

from shared.logging import configure_logging

logger = get_logger(__name__)
# Logger level will be set by configure_logging() based on scenario.yaml debug_level
# Don't hardcode level here - let it inherit from parent 'generator' logger
logger.propagate = True  # Ensure messages propagate to root logger

LIVE_RAYTRACING_INTEGER_BOUNDS = {
    "max_depth": (1, 10),
    "samples_per_src": (1, 100_000_000),
    "max_num_paths_per_src": (1, 100_000_000),
    "seed": (0, 999_999),
}


def _finite_triple(values: Any, *, label: str) -> tuple[float, float, float]:
    """Return an exact finite three-value tuple for a live update."""
    try:
        if len(values) != 3:
            raise ValueError(f"{label} must contain exactly 3 values")
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain exactly 3 numeric values") from exc
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{label} values must all be finite")
    return normalized[0], normalized[1], normalized[2]


class GeneratorService(visualizer_pb2_grpc.GeneratorServiceServicer):
    """Live gRPC service that computes, caches, normalizes, and streams frames."""

    # Default retry configuration
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY_SECONDS = 0.5

    def __init__(
        self,
        frame_cache: GeneratorFrameCache,
        generator_config: Dict[str, Any],
        frame_provider: Any = None,
        provider_timeout_s: float = 5.0,
        max_retries: int | None = None,
        retry_delay_seconds: float | None = None,
    ):
        self.frame_cache = frame_cache
        self.generator_config = generator_config
        self.is_ready = True
        self.is_streaming = False
        self.start_time = time.time()
        self.frames_generated = 0
        self.frames_failed = 0
        self.external_frame_provider = frame_provider
        self._external_frames_compatible = True
        self.provider_timeout_s = provider_timeout_s
        self.generation_epoch = 0

        # Retry configuration
        self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self.retry_delay_seconds = (
            retry_delay_seconds
            if retry_delay_seconds is not None
            else self.DEFAULT_RETRY_DELAY_SECONDS
        )

        # Services
        services = generator_config.get("services", {})
        self.scene_service = services.get("scene_service")
        self.actor_state_service = services.get("actor_state_service")
        self.raytracing_service = services.get("raytracing_service")

        # Configs (needed for rebuilding)
        self.configs = generator_config.get("configs", {})

        if not self.raytracing_service:
            logger.warning("RayTracingService not found in generator_config! Streaming may fail.")

        # Serialize live frame computation against concurrent stream requests.
        self._frame_compute_lock = threading.RLock()

        # Scene state, solver settings, and cached frames form one mutable
        # session owned by a single controlling stream.
        self._stream_state_lock = threading.Lock()
        self._controller_active = False
        self._controller_idle = threading.Event()
        self._controller_idle.set()
        self._scene_xml_stager = LiveSceneXmlStager()

    def _bump_generation_epoch(self) -> int:
        """Advance the frame-generation epoch after cache-invalidating updates."""
        self.generation_epoch += 1
        return self.generation_epoch

    def _invalidate_generated_frames(self, *, reason: str) -> int:
        """Clear generated frames and advance the public state revision."""
        with self._frame_compute_lock:
            removed = self.frame_cache.clear()
            epoch = self._bump_generation_epoch()
        logger.info(
            "Cleared %d cached frame(s) after %s (generation_epoch=%d)",
            removed,
            reason,
            epoch,
        )
        return removed

    def _retire_external_frames(self, *, reason: str) -> None:
        """Stop serving dispatcher frames after persistent simulation changes."""
        if self.external_frame_provider is None or not self._external_frames_compatible:
            return
        self._external_frames_compatible = False
        logger.info("Ignoring pre-change dispatcher frames after %s", reason)

    def _resolve_live_actor_name(self, category: str, requested_name: str) -> str | None:
        """Resolve a live-edit name or positional alias to a configured actor."""
        config_key = {
            "tx": "tx_configs",
            "rx": "rx_configs",
            "target": "target_configs",
        }.get(category)
        if config_key is None:
            return None

        normalized_request = str(requested_name).strip().casefold()
        for index, config in enumerate(self.configs.get(config_key, ()), start=1):
            configured_name = self._read_config_value(config, "name") or f"{category}_{index}"
            aliases = {
                str(configured_name).casefold(),
                f"{category}_{index}",
                f"{category}{index}",
            }
            if normalized_request in aliases:
                return str(configured_name)
        return None

    def _claim_controller(self) -> bool:
        """Claim the single mutable live session for one streaming RPC."""
        with self._stream_state_lock:
            if self._controller_active:
                return False
            self._controller_active = True
            self._controller_idle.clear()
            self.is_streaming = True
            return True

    def _release_controller(self) -> None:
        """Release live-session ownership after its streaming RPC ends."""
        with self._stream_state_lock:
            self._controller_active = False
            self.is_streaming = False
            self._controller_idle.set()

    def _simulation_config(self) -> Any:
        """Return the active SimulationConfig when it is available."""
        return (
            self.configs.get("simulation_config")
            or self.configs.get("simulation")
            or self.generator_config.get("simulation_config")
        )

    @staticmethod
    def _read_config_value(config: Any, name: str) -> Any:
        """Read a value from a mapping or configuration object."""
        if isinstance(config, dict):
            return config.get(name)
        return getattr(config, name, None)

    @staticmethod
    def _write_config_value(config: Any, name: str, value: Any) -> None:
        """Write a value to a mapping or configuration object."""
        if isinstance(config, dict):
            config[name] = value
            return
        setattr(config, name, value)

    def _live_scene_source_path(self) -> Path:
        """Return the active XML file that anchors relative scene assets."""
        scenario_config = self.generator_config.get("scenario_configuration")
        scene_xml = self._read_config_value(scenario_config, "scene_xml")
        if scene_xml:
            return Path(scene_xml).resolve(strict=True)

        simulation_config = self._simulation_config()
        scene_name = self._read_config_value(simulation_config, "scene_name")
        if scene_name:
            return Path(scene_name).resolve(strict=True)

        raise ValueError("Live XML updates require an active scenario XML file")

    def _simulation_config_references(self) -> list[Any]:
        """Collect the distinct simulation-config objects used by live services."""
        candidates = [
            self._simulation_config(),
            self.generator_config.get("simulation_config"),
            self.configs.get("simulation_config"),
            getattr(self.scene_service, "simulation_config", None),
            getattr(self.actor_state_service, "simulation_config", None),
            getattr(self.raytracing_service, "simulation_config", None),
        ]
        unique: list[Any] = []
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            unique.append(candidate)
        return unique

    def _rebuild_live_services(self) -> SimulationObjects:
        """Rebuild the prepared live scene from the current configuration."""
        if self.scene_service is None:
            raise RuntimeError("SceneService is not available")
        if self.actor_state_service is None:
            raise RuntimeError("ActorStateService is not available")
        if self.raytracing_service is None:
            raise RuntimeError("RayTracingService is not available")

        tx_configs = self.configs["tx_configs"]
        rx_configs = self.configs["rx_configs"]
        target_configs = self.configs["target_configs"]
        self.scene_service.build_scene(tx_configs, rx_configs, target_configs)
        _manager, actor_state_cache = self.actor_state_service.prepare_actor_state(
            tx_configs,
            rx_configs,
            self.scene_service.target_managers,
            motion_mode="cached",
        )
        self.raytracing_service.prepare_simulation(
            self.scene_service,
            self.actor_state_service,
            tx_configs,
            rx_configs,
            target_configs,
            scenario_configuration=self.generator_config.get("scenario_configuration"),
            motion_mode="cached",
            actor_state_cache=actor_state_cache,
        )
        simulation = self._ensure_simulation_context()
        if simulation is None:
            raise RuntimeError("Live services did not produce a simulation context")
        return simulation

    def _activate_live_scene_xml(self, xml_payload: str) -> SimulationObjects:
        """Validate, stage, and activate one full-scene XML replacement.

        The source XML is never overwritten. A failed candidate is discarded,
        and the previous configuration is rebuilt before the error is returned.
        """
        source = self._live_scene_source_path()
        candidate = self._scene_xml_stager.stage(
            xml_payload,
            source_scene_path=source,
        )
        scenario_config = self.generator_config.get("scenario_configuration")
        simulation_configs = self._simulation_config_references()
        previous_scene_xml = self._read_config_value(scenario_config, "scene_xml")
        previous_scene_names = [
            self._read_config_value(config, "scene_name") for config in simulation_configs
        ]

        try:
            with self._frame_compute_lock:
                if scenario_config is not None:
                    self._write_config_value(scenario_config, "scene_xml", candidate)
                for config in simulation_configs:
                    self._write_config_value(config, "scene_name", str(candidate))
                simulation = self._rebuild_live_services()
        except Exception as activation_error:
            if scenario_config is not None:
                self._write_config_value(
                    scenario_config,
                    "scene_xml",
                    previous_scene_xml,
                )
            for config, previous_name in zip(
                simulation_configs,
                previous_scene_names,
                strict=True,
            ):
                self._write_config_value(config, "scene_name", previous_name)
            try:
                with self._frame_compute_lock:
                    self._rebuild_live_services()
            except Exception as rollback_error:
                self.is_ready = False
                self._invalidate_generated_frames(reason="failed live scene rollback")
                raise RuntimeError(
                    "Live scene activation failed and the previous scene could not be "
                    f"restored: activation={activation_error}; rollback={rollback_error}"
                ) from activation_error
            finally:
                self._scene_xml_stager.discard(candidate)
            raise RuntimeError(
                f"Live scene activation failed; the previous scene was restored: {activation_error}"
            ) from activation_error

        self._scene_xml_stager.accept(candidate)
        return simulation

    def close(self) -> None:
        """Wait for the controlling stream, then release owned scene files."""
        self.is_ready = False
        self._controller_idle.wait()
        with self._frame_compute_lock:
            self._scene_xml_stager.close()

    def _frame_info_metadata(self) -> Tuple[int, float, float]:
        """Resolve frame-count metadata even before any frame has been generated."""
        sim_config = self._simulation_config()

        total_frames = int(getattr(self.frame_cache, "total_frames", 0) or 0)
        for candidate in (
            getattr(sim_config, "num_steps", None),
            self.generator_config.get("num_steps"),
        ):
            try:
                total_frames = max(total_frames, int(candidate or 0))
            except (TypeError, ValueError):
                continue

        duration = float(getattr(self.frame_cache, "duration", 0.0) or 0.0)
        for candidate in (
            getattr(sim_config, "duration", None),
            getattr(sim_config, "duration_s", None),
            self.generator_config.get("duration"),
        ):
            try:
                duration = duration or float(candidate or 0.0)
            except (TypeError, ValueError):
                continue

        frame_rate = float(getattr(self.frame_cache, "frame_rate", 0.0) or 0.0)
        if frame_rate <= 0.0 and duration > 0.0 and total_frames > 0:
            frame_rate = float(total_frames) / duration

        return total_frames, duration, frame_rate

    def _compute_frame_with_retry(
        self, frame_idx: int, live_overrides: Optional[List] = None
    ) -> Optional[Dict[str, Any]]:
        """Compute a single frame with retry logic for transient failures.

        Uses exponential backoff between retries. Tracks failures for health monitoring.

        Args:
            frame_idx: Frame index to compute.
            live_overrides: Optional list of position/orientation overrides.

        Returns:
            Frame data dict on success, None on failure after all retries.
        """
        if not self.raytracing_service:
            logger.error("Cannot compute frame %s: RayTracingService not available", frame_idx)
            self.frames_failed += 1
            return None

        for attempt in range(self.max_retries):
            try:
                with self._frame_compute_lock:
                    frame_data = self.raytracing_service.compute_step(
                        frame_idx, live_overrides=live_overrides or None
                    )
                if frame_data is not None:
                    self.frames_generated += 1
                    return frame_data
                else:
                    # None return is not an exception, but indicates no data
                    logger.warning(
                        "Frame %s computation returned None (attempt %d/%d)",
                        frame_idx,
                        attempt + 1,
                        self.max_retries,
                    )
            except Exception as e:  # broad catch: gRPC handler retry loop
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay_seconds * (2**attempt)
                    logger.warning(
                        "Frame %s computation failed (attempt %d/%d), retrying in %.2fs: %s",
                        frame_idx,
                        attempt + 1,
                        self.max_retries,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Frame %s computation failed after %d attempts: %s",
                        frame_idx,
                        self.max_retries,
                        e,
                    )

        self.frames_failed += 1
        return None

    def _ensure_simulation_context(self) -> Optional[SimulationObjects]:
        """Ensure simulation context is available via RayTracingService."""
        if self.raytracing_service and self.raytracing_service.simulation_objects:
            return self.raytracing_service.simulation_objects
        return None

    def _apply_parameter_update(
        self,
        session_simulation: SimulationObjects,
        update_msg: Any,
    ) -> tuple[Dict[str, Any], bool]:
        """Apply a parameter update and report whether effective state changed."""
        applied_settings: Dict[str, Any] = {}

        if update_msg is None:
            return applied_settings, False

        if not update_msg.HasField("raytracing_config"):
            raise ValueError("Parameter update requires raytracing_config")

        cfg = update_msg.raytracing_config
        applied_settings = {
            "max_depth": int(cfg.max_depth),
            "samples_per_src": int(cfg.samples_per_src),
            "max_num_paths_per_src": int(cfg.max_num_paths_per_src),
            "los": bool(cfg.los),
            "specular_reflection": bool(cfg.specular_reflection),
            "diffuse_reflection": bool(cfg.diffuse_reflection),
            "refraction": bool(cfg.refraction),
            "diffraction": bool(cfg.diffraction),
            "synthetic_array": bool(cfg.synthetic_array),
            "seed": int(cfg.seed),
        }

        for name, (minimum, maximum) in LIVE_RAYTRACING_INTEGER_BOUNDS.items():
            value = applied_settings[name]
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")

        current_settings = getattr(session_simulation, "settings", None)
        if isinstance(current_settings, dict):
            state_changed = any(
                current_settings.get(key, object()) != value
                for key, value in applied_settings.items()
            )
        else:
            sim_config = getattr(session_simulation, "simulation_config", None)
            state_changed = any(
                not hasattr(sim_config, key) or getattr(sim_config, key) != value
                for key, value in applied_settings.items()
            )

        # Update simulation settings in-place so future computations pick up the change.
        if hasattr(session_simulation, "settings") and isinstance(
            session_simulation.settings, dict
        ):
            session_simulation.settings.update(applied_settings)

        # Keep generator configuration metadata in sync (useful for status RPCs).
        ray_settings = self.generator_config.setdefault("raytracing_settings", {})
        ray_settings.update(applied_settings)

        # Update the SimulationConfig if present so future rebuilds inherit values.
        sim_config = getattr(session_simulation, "simulation_config", None)
        if sim_config is not None:
            for key, value in applied_settings.items():
                if hasattr(sim_config, key):
                    setattr(sim_config, key, value)

        return applied_settings, state_changed

    def GetFrameInfo(self, request, context):
        """Get available frame information"""
        try:
            available_frames = self.frame_cache.get_available_frames()
            total_frames, duration, frame_rate = self._frame_info_metadata()

            return visualizer_pb2.GetFrameInfoResponse(
                success=True,
                message="Frame info retrieved successfully",
                total_frames=total_frames,
                duration=duration,
                frame_rate=frame_rate,
                available_frames=available_frames,
                generation_epoch=self.generation_epoch,
            )

        except Exception as e:  # broad catch: gRPC handler
            logger.error(f"Error getting frame info: {e}")
            return visualizer_pb2.GetFrameInfoResponse(success=False, message=f"Error: {str(e)}")

    def HealthCheck(self, request, context):
        """Return server health status for monitoring.

        Provides uptime, frame generation stats, and readiness status.
        """
        try:
            uptime = time.time() - self.start_time
            return visualizer_pb2.HealthCheckResponse(
                healthy=self.is_ready,
                uptime_seconds=uptime,
                frames_generated=self.frames_generated,
                frames_failed=self.frames_failed,
                is_streaming=self.is_streaming,
                is_ready=self.is_ready,
            )
        except Exception as e:  # broad catch: gRPC handler
            logger.error(f"Error in HealthCheck: {e}")
            return visualizer_pb2.HealthCheckResponse(
                healthy=False,
                uptime_seconds=0.0,
                frames_generated=0,
                frames_failed=0,
                is_streaming=False,
                is_ready=False,
            )

    def GetCacheStatus(self, request, context):
        """Return frame cache statistics for monitoring.

        Provides cache size, eviction stats, and configuration details.
        """
        try:
            stats = self.frame_cache.get_cache_stats()
            return visualizer_pb2.CacheStatusResponse(
                cached_frames=stats["cached_frames"],
                total_size_bytes=stats["total_size_bytes"],
                oldest_frame_age_seconds=stats["oldest_frame_age_seconds"],
                mode=stats["mode"],
                max_frames=stats["max_frames"],
                max_size_bytes=stats["max_size_bytes"],
                ttl_seconds=stats["ttl_seconds"],
                evictions_count=stats["evictions_count"],
                evictions_ttl=stats["evictions_ttl"],
                evictions_size=stats["evictions_size"],
                peak_size_bytes=stats["peak_size_bytes"],
                cache_hits=stats["cache_hits"],
                cache_misses=stats["cache_misses"],
                oversized_bypasses=stats["oversized_bypasses"],
            )
        except Exception as e:  # broad catch: gRPC handler
            logger.error(f"Error in GetCacheStatus: {e}")
            return visualizer_pb2.CacheStatusResponse(
                cached_frames=0,
                total_size_bytes=0,
                oldest_frame_age_seconds=0.0,
                mode="unknown",
                max_frames=0,
                max_size_bytes=0,
                ttl_seconds=0.0,
                evictions_count=0,
                evictions_ttl=0,
                evictions_size=0,
                peak_size_bytes=0,
                cache_hits=0,
                cache_misses=0,
                oversized_bypasses=0,
            )

    def StreamFrames(self, request_iterator, context):
        """Handle live frame requests, flow control, and scene/object updates.

        Request handling works in two phases: obtain a raw frame dictionary
        from the dispatcher, cache, or ray-tracing service; then convert that
        raw frame to ``StandardMPCFrame`` protobuf just before yielding the
        response. Keeping that boundary explicit prevents transport code from
        depending on Sionna objects directly.
        """

        session_simulation = self._ensure_simulation_context()
        if session_simulation is None:
            yield visualizer_pb2.FrameResponse(
                error=visualizer_pb2.ErrorDetails(
                    code="CONTEXT_UNAVAILABLE",
                    message="Generator context is not ready for streaming",
                ),
                generation_epoch=self.generation_epoch,
            )
            return

        live_overrides: List[Any] = []
        override_state: LiveOverrideMap = {"tx": {}, "rx": {}, "target": {}}
        session_overrides_changed = False
        buffer_capacity = 0  # Maximum frames to send to client (transmission limit)
        buffer_fill = 0  # Frames sent to client
        server_prefetch_limit = 0  # Maximum frames to compute ahead (computation limit, cache only)
        frames_computed = 0  # Frames computed and cached (may exceed buffer_capacity)
        next_frame_idx = 0
        sent_eof = False
        uncached_prefetch: tuple[int, Dict[str, Any]] | None = None

        total_steps = 0
        simulation_config = None

        def _serialize_override_state(state: LiveOverrideMap) -> List[Dict[str, Any]]:
            serialized: List[Dict[str, Any]] = []
            for category, entries in state.items():
                for key, entry in entries.items():
                    if not entry:
                        continue
                    payload: Dict[str, Any] = {
                        "name": entry.name or key,
                        "type": category,
                    }
                    if entry.position is not None:
                        payload["position"] = entry.position
                    if entry.orientation is not None:
                        payload["orientation"] = entry.orientation
                    if entry.scale is not None:
                        payload["scale"] = entry.scale
                    serialized.append(payload)
            return serialized

        def _replace_override_state(new_state: LiveOverrideMap) -> bool:
            nonlocal override_state, live_overrides, session_overrides_changed
            replacement: LiveOverrideMap = {
                "tx": dict(new_state.get("tx", {})),
                "rx": dict(new_state.get("rx", {})),
                "target": dict(new_state.get("target", {})),
            }
            if replacement == override_state:
                return False
            override_state = replacement
            live_overrides = _serialize_override_state(override_state)
            session_overrides_changed = True
            return True

        def _resolve_override_actor_names(new_state: LiveOverrideMap) -> LiveOverrideMap:
            """Return override state keyed by configured actors or reject it."""
            resolved: LiveOverrideMap = {"tx": {}, "rx": {}, "target": {}}
            for category, entries in new_state.items():
                category_key = cast(LiveActorCategory, category)
                for key, entry in entries.items():
                    configured_name = self._resolve_live_actor_name(
                        category,
                        entry.name or key,
                    )
                    if configured_name is None:
                        raise ValueError(f"Unknown {category.upper()} actor {entry.name or key!r}")
                    resolved[category][configured_name.casefold()] = LiveActorOverride(
                        name=configured_name,
                        category=category_key,
                        position=entry.position,
                        orientation=entry.orientation,
                        scale=entry.scale,
                    )
            return resolved

        def _update_node_override(
            category: str,
            name: str,
            *,
            position: Optional[Tuple[float, float, float]] = None,
            orientation: Optional[Tuple[float, float, float]] = None,
            scale: Optional[float] = None,
        ) -> bool:
            nonlocal override_state, live_overrides, session_overrides_changed
            key = category.lower()
            if key not in override_state:
                return False
            configured_name = self._resolve_live_actor_name(key, name)
            if configured_name is None:
                raise ValueError(f"Unknown {key.upper()} actor {name!r}")
            name = configured_name
            category_key = cast(LiveActorCategory, key)
            existing = override_state[key].get(name.lower())
            replacement = LiveActorOverride(
                name=existing.name if existing is not None else name,
                category=category_key,
                position=position if position is not None else getattr(existing, "position", None),
                orientation=(
                    orientation
                    if orientation is not None
                    else getattr(existing, "orientation", None)
                ),
                scale=scale if scale is not None else getattr(existing, "scale", None),
            )
            if replacement == existing:
                return False
            override_state[key][name.lower()] = replacement
            live_overrides = _serialize_override_state(override_state)
            session_overrides_changed = True
            return True

        def _reset_stream_after_invalidation() -> None:
            nonlocal buffer_fill, frames_computed, next_frame_idx, sent_eof, uncached_prefetch
            buffer_fill = 0
            frames_computed = 0
            next_frame_idx = 0
            sent_eof = False
            uncached_prefetch = None

        def _refresh_simulation_metadata():
            nonlocal simulation_config, total_steps
            simulation_config = getattr(session_simulation, "simulation_config", None)
            if simulation_config is not None:
                total_steps = getattr(simulation_config, "num_steps", 0) or 0
                duration = getattr(simulation_config, "duration", 0.0) or 0.0
                frame_rate = (float(total_steps) / float(duration)) if duration else 0.0
                self.frame_cache.update_stats(duration, frame_rate)
            else:
                total_steps = int(self.generator_config.get("num_steps") or 0)

        _refresh_simulation_metadata()

        def _produce_frame(frame_idx: int) -> Optional[Any]:
            nonlocal uncached_prefetch
            logger.debug(
                "[StreamFrames] preparing frame %s (next_frame_idx=%s, "
                "buffer_fill=%s/%s, total_steps=%s)",
                frame_idx,
                next_frame_idx,
                buffer_fill,
                buffer_capacity,
                total_steps,
            )

            # Check if frame is invalid (negative or beyond total_steps)
            if frame_idx < 0:
                logger.warning(
                    "[StreamFrames] Cannot produce frame %s: negative frame index", frame_idx
                )
                return visualizer_pb2.FrameResponse(
                    error=visualizer_pb2.ErrorDetails(
                        code="INVALID_FRAME",
                        message=f"Frame {frame_idx} is negative (invalid frame index)",
                    ),
                    frame_idx=frame_idx,
                    generation_epoch=self.generation_epoch,
                )
            if total_steps and frame_idx >= total_steps:
                logger.warning(
                    "[StreamFrames] Cannot produce frame %s: beyond total_steps (%s)",
                    frame_idx,
                    total_steps,
                )
                return visualizer_pb2.FrameResponse(
                    error=visualizer_pb2.ErrorDetails(
                        code="INVALID_FRAME",
                        message=f"Frame {frame_idx} is beyond total_steps ({total_steps})",
                    ),
                    frame_idx=frame_idx,
                    generation_epoch=self.generation_epoch,
                )

            # Get current override state (if any)
            external_overrides = _serialize_override_state(override_state)

            # Prefer the dispatcher when available because it has the current external positions.
            # Only check gRPC cache if dispatcher is unavailable (standalone mode)
            frame_data = None
            cached_frame = None
            one_shot_prefetch_used = False

            if uncached_prefetch is not None:
                prefetched_idx, prefetched_frame = uncached_prefetch
                uncached_prefetch = None
                if prefetched_idx == frame_idx:
                    frame_data = prefetched_frame
                    one_shot_prefetch_used = True
                    logger.debug(
                        "[StreamFrames] Frame %s served from one-shot oversized prefetch",
                        frame_idx,
                    )

            if (
                self.external_frame_provider is not None
                and self._external_frames_compatible
                and not external_overrides
            ):
                # Dispatcher available: use it as primary source (has correct streaming positions)
                frame_data = self.external_frame_provider.wait_for_frame(
                    frame_idx, timeout=self.provider_timeout_s
                )
                if frame_data is not None:
                    logger.debug(
                        "[StreamFrames] frame %s served from dispatcher (external provider)",
                        frame_idx,
                    )
                    # If frame came from dispatcher, remove any stale entry from gRPC cache
                    # This ensures dispatcher frames (with correct streaming positions) take precedence
                    if self.frame_cache.remove_frame(frame_idx):
                        logger.debug(
                            "[StreamFrames] Removed stale frame %s from gRPC cache (dispatcher has newer version)",
                            frame_idx,
                        )

            # Only check gRPC cache if dispatcher didn't provide the frame (standalone mode or dispatcher miss)
            if frame_data is None:
                cached_frame = self.frame_cache.get_frame(frame_idx)
                if cached_frame is not None:
                    logger.debug(
                        "[StreamFrames] Frame %s found in gRPC cache (standalone mode)", frame_idx
                    )
                    frame_data = cached_frame

            # Check for stored positions if frame needs to be recomputed
            # This allows recomputation with correct streaming positions even after frame eviction
            stored_positions_used = False
            if (
                frame_data is None
                and self.external_frame_provider is not None
                and not external_overrides
            ):
                stored_positions = self.external_frame_provider.get_stored_positions(frame_idx)
                if stored_positions:
                    logger.info(
                        "[StreamFrames] Frame %s not in dispatcher, but found stored streaming positions. "
                        "Recomputing with stored positions (ray tracing has stochastic components)",
                        frame_idx,
                    )
                    # Use stored positions as external overrides for recomputation.
                    external_overrides = stored_positions
                    stored_positions_used = True

            if frame_data is None:
                try:
                    with self._frame_compute_lock:
                        if stored_positions_used:
                            logger.debug(
                                "[StreamFrames] Computing frame %s with stored streaming positions (frame was evicted)",
                                frame_idx,
                            )
                        else:
                            logger.debug(
                                "[StreamFrames] Computing frame %s (cache miss, generating new)",
                                frame_idx,
                            )
                        if self.raytracing_service:
                            frame_data = self.raytracing_service.compute_step(
                                frame_idx, live_overrides=external_overrides or None
                            )
                        else:
                            # Should not happen if check in __init__ passes, but defensive
                            logger.error("RayTracingService missing, cannot compute frame")
                            frame_data = None
                        # Mark frame as recomputed from stored positions
                        if stored_positions_used and frame_data is not None:
                            frame_data["_recomputed_from_stored_positions"] = True
                except Exception as exc:  # broad catch: gRPC handler
                    logger.exception(
                        "Frame computation raised an exception for frame %s", frame_idx
                    )
                    self.frames_failed += 1
                    return visualizer_pb2.FrameResponse(
                        error=visualizer_pb2.ErrorDetails(
                            code="COMPUTATION_FAILED",
                            message=f"Exception while computing frame {frame_idx}: {exc}",
                        ),
                        frame_idx=frame_idx,
                        generation_epoch=self.generation_epoch,
                    )

                if frame_data is None:
                    logger.error("Frame computation returned no data for frame %s", frame_idx)
                    self.frames_failed += 1
                    return visualizer_pb2.FrameResponse(
                        error=visualizer_pb2.ErrorDetails(
                            code="COMPUTATION_FAILED",
                            message=f"Failed to compute frame {frame_idx}",
                        ),
                        frame_idx=frame_idx,
                        generation_epoch=self.generation_epoch,
                    )

            frame_data_dict = cast(Dict[str, Any], frame_data)
            self.frames_generated += 1
            # Only cache frames that were computed by gRPC server (not retrieved from dispatcher)
            # This avoids redundant caching: dispatcher already caches frames from bridge
            # gRPC cache is only for standalone mode (when dispatcher is unavailable)
            frame_source = (
                "dispatcher"
                if (self.external_frame_provider is not None and cached_frame is None)
                else ("cache" if cached_frame else "computed")
            )

            # Only add to gRPC cache if:
            # 1. Frame was computed by gRPC server (not from dispatcher)
            # 2. Not already in cache
            # This keeps gRPC cache for standalone mode only, avoiding redundancy with dispatcher
            if (
                not one_shot_prefetch_used
                and cached_frame is None
                and (
                    self.external_frame_provider is None
                    or not self._external_frames_compatible
                    or self.external_frame_provider.get_frame(frame_idx) is None
                )
            ):
                self.frame_cache.add_frame(frame_idx, frame_data_dict)

            logger.debug(
                "[StreamFrames] frame %s ready; sending to client (source: %s)",
                frame_idx,
                frame_source,
            )

            try:
                frame_pb = self._convert_to_protobuf_frame(frame_data_dict, frame_idx=frame_idx)
            except Exception as exc:  # noqa: BLE001 - return conversion failure on the wire.
                logger.exception("Frame %s does not satisfy the transport contract", frame_idx)
                self.frames_failed += 1
                return visualizer_pb2.FrameResponse(
                    error=visualizer_pb2.ErrorDetails(
                        code="ENCODING_FAILED",
                        message=f"Cannot encode frame {frame_idx}: {exc}",
                    ),
                    frame_idx=frame_idx,
                    generation_epoch=self.generation_epoch,
                )
            response = visualizer_pb2.FrameResponse(
                frame_data=frame_pb,
                frame_idx=frame_idx,
                generation_epoch=self.generation_epoch,
            )
            response_size = response.ByteSize()
            if response_size > GRPC_MAX_MESSAGE_BYTES:
                logger.error(
                    "Encoded frame %s is %d bytes, exceeding the live transport limit of %d bytes",
                    frame_idx,
                    response_size,
                    GRPC_MAX_MESSAGE_BYTES,
                )
                self.frames_failed += 1
                return visualizer_pb2.FrameResponse(
                    error=visualizer_pb2.ErrorDetails(
                        code="FRAME_TOO_LARGE",
                        message=(
                            f"Encoded frame {frame_idx} is {response_size} bytes; "
                            f"the live transport limit is {GRPC_MAX_MESSAGE_BYTES} bytes"
                        ),
                    ),
                    frame_idx=frame_idx,
                    generation_epoch=self.generation_epoch,
                )
            return response

        def _emit_eof(last_idx: int) -> Any:
            return visualizer_pb2.FrameResponse(
                eof=visualizer_pb2.EndOfStream(last_frame_idx=last_idx),
                frame_idx=last_idx,
                generation_epoch=self.generation_epoch,
            )

        def _compute_ahead():
            """Compute frames ahead and cache them (without sending to client).

            Yields None after each frame to allow other operations (like sending frames) to proceed.
            This prevents blocking when computing many frames ahead.

            Computes frames starting from the first frame that hasn't been computed yet,
            up to server_prefetch_limit frames ahead of the current position.
            """
            nonlocal next_frame_idx, frames_computed, sent_eof, uncached_prefetch
            if uncached_prefetch is not None:
                return
            # Compute frames starting from where we left off
            # At initialization: next_frame_idx=0, frames_computed=0, so compute from frame 0
            # After sending: next_frame_idx advances, but frames_computed tracks how many ahead we've computed
            # So compute frames starting from: next_frame_idx + frames_computed
            # Up to: next_frame_idx + server_prefetch_limit

            frame_to_compute = next_frame_idx + frames_computed
            compute_end = next_frame_idx + server_prefetch_limit

            while (
                server_prefetch_limit > 0
                and frames_computed < server_prefetch_limit
                and frame_to_compute < compute_end
            ):
                # Skip negative frames and frames beyond total_steps
                if frame_to_compute < 0:
                    frames_computed += 1
                    frame_to_compute += 1
                    continue
                if total_steps and frame_to_compute >= total_steps:
                    break

                # Check dispatcher first (if available) - don't prefetch frames that bridge will provide
                frame_already_available = False
                if self.external_frame_provider is not None:
                    if self.external_frame_provider.get_frame(frame_to_compute) is not None:
                        # Frame already in dispatcher (from bridge), skip prefetch
                        frame_already_available = True

                # Check gRPC cache only if dispatcher doesn't have it
                if not frame_already_available:
                    if self.frame_cache.get_frame(frame_to_compute) is not None:
                        frame_already_available = True

                if frame_already_available:
                    frames_computed += 1
                    frame_to_compute += 1
                    continue

                # Don't prefetch if dispatcher is available - bridge will provide frames on demand
                # Prefetching is only useful in standalone mode
                if self.external_frame_provider is not None:
                    logger.debug(
                        "[StreamFrames] Skipping prefetch for frame %s (dispatcher available, bridge will provide)",
                        frame_to_compute,
                    )
                    frames_computed += 1
                    frame_to_compute += 1
                    continue

                logger.debug(
                    "[StreamFrames] Computing ahead frame %s (frames_computed=%s/%s, next_frame=%s)",
                    frame_to_compute,
                    frames_computed,
                    server_prefetch_limit,
                    next_frame_idx,
                )

                # Compute frame and cache it (don't send) - only in standalone mode
                # Use retry logic for transient failures
                frame_data = self._compute_frame_with_retry(
                    frame_to_compute,
                    live_overrides,
                )

                if frame_data is not None:
                    admitted = self.frame_cache.add_frame(frame_to_compute, frame_data)
                    frames_computed += 1
                    if admitted:
                        logger.debug(
                            "[StreamFrames] Cached frame %s (computed ahead, not sent)",
                            frame_to_compute,
                        )
                    else:
                        uncached_prefetch = (frame_to_compute, frame_data)
                        logger.debug(
                            "[StreamFrames] Holding oversized frame %s for one immediate send",
                            frame_to_compute,
                        )
                else:
                    # After retries exhausted, skip this frame and continue with next
                    # Don't break entirely - allow streaming to continue with remaining frames
                    logger.warning(
                        "[StreamFrames] Skipping frame %s after failed computation",
                        frame_to_compute,
                    )
                    frames_computed += 1  # Count as "processed" to avoid infinite loop

                frame_to_compute += 1

                # Yield after each frame to allow sending frames and handling requests immediately
                # This prevents blocking when computing many frames ahead
                yield None
                if uncached_prefetch is not None:
                    break

        def _fill_buffer():
            """Send frames from cache to client (up to buffer_capacity)."""
            nonlocal next_frame_idx, buffer_fill, sent_eof
            # Start sending from the first unsent frame
            # next_frame_idx represents the next frame to compute/send
            # buffer_fill is how many frames have been sent
            # So first unsent frame is: next_frame_idx - buffer_fill
            # Ensure next_frame_idx is non-negative to prevent negative frame_to_send
            next_frame_idx = max(0, next_frame_idx)
            frame_to_send = max(0, next_frame_idx - buffer_fill)

            while buffer_capacity > 0 and buffer_fill < buffer_capacity:
                # Skip negative frames (shouldn't happen now, but defensive check)
                if frame_to_send < 0:
                    logger.warning(
                        "[StreamFrames] Negative frame_to_send detected: %s (next_frame_idx=%s, buffer_fill=%s), skipping",
                        frame_to_send,
                        next_frame_idx,
                        buffer_fill,
                    )
                    frame_to_send = 0
                    next_frame_idx = max(0, next_frame_idx)
                    buffer_fill = 0
                    continue
                if total_steps and frame_to_send >= total_steps:
                    if not sent_eof:
                        sent_eof = True
                        yield _emit_eof(total_steps - 1 if total_steps > 0 else frame_to_send - 1)
                    break

                # Check if frame is in cache (computed ahead)
                cached_frame = self.frame_cache.get_frame(frame_to_send)
                if cached_frame is None:
                    # Frame not computed yet - compute it now
                    logger.debug(
                        "[StreamFrames] Frame %s not in cache, computing now (buffer_fill=%s/%s)",
                        frame_to_send,
                        buffer_fill,
                        buffer_capacity,
                    )
                    response = _produce_frame(frame_to_send)
                    if response is None:
                        break
                    yield response
                    if response.WhichOneof("response_type") == "frame_data":
                        buffer_fill += 1
                        next_frame_idx = frame_to_send + 1
                        frame_to_send += 1
                    else:
                        break
                else:
                    logger.debug(
                        "[StreamFrames] Sending cached frame %s (buffer_fill=%s/%s)",
                        frame_to_send,
                        buffer_fill,
                        buffer_capacity,
                    )
                    response = _produce_frame(frame_to_send)
                    if response is None:
                        break
                    yield response
                    if response.WhichOneof("response_type") != "frame_data":
                        break
                    buffer_fill += 1
                    next_frame_idx = frame_to_send + 1
                    frame_to_send += 1

        controller_acquired = self._claim_controller()
        if not controller_acquired:
            details = "The live generator already has an active controlling client"
            abort = getattr(context, "abort", None)
            if callable(abort):
                abort(grpc.StatusCode.RESOURCE_EXHAUSTED, details)
            else:  # pragma: no cover - production gRPC contexts provide abort()
                set_code = getattr(context, "set_code", None)
                set_details = getattr(context, "set_details", None)
                if callable(set_code):
                    set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                if callable(set_details):
                    set_details(details)
            return

        try:
            for request in request_iterator:
                if not context.is_active():
                    logger.debug("[StreamFrames] gRPC context inactive; stopping stream")
                    break

                req_type = request.WhichOneof("request_type")

                if req_type == "override_cmd":
                    raw_overrides = list(request.override_cmd.overrides)
                    try:
                        for override in raw_overrides:
                            if category_from_value(override.type) is None:
                                raise ValueError(
                                    f"Override for {override.name or 'unnamed actor'} "
                                    "has an invalid actor type"
                                )
                            if not str(override.name).strip():
                                raise ValueError("Override actor name must not be empty")
                            _finite_triple(
                                (override.x, override.y, override.z),
                                label=f"Override position for {override.name or 'unnamed actor'}",
                            )
                            if override.orientation:
                                _finite_triple(
                                    override.orientation,
                                    label=(
                                        "Override orientation for "
                                        f"{override.name or 'unnamed actor'}"
                                    ),
                                )
                        normalized_overrides = _resolve_override_actor_names(
                            normalize_live_overrides(raw_overrides)
                        )
                    except ValueError as exc:
                        yield visualizer_pb2.FrameResponse(
                            error=visualizer_pb2.ErrorDetails(
                                code="INVALID_OVERRIDE",
                                message=str(exc),
                            ),
                            generation_epoch=self.generation_epoch,
                        )
                        continue
                    state_changed = _replace_override_state(normalized_overrides)
                    if state_changed:
                        self._invalidate_generated_frames(reason="live override change")
                        _reset_stream_after_invalidation()
                    logger.debug(
                        "[StreamFrames] received %s override(s); state_changed=%s",
                        len(live_overrides),
                        state_changed,
                    )

                elif req_type == "parameter_update":
                    update = request.parameter_update
                    logger.info(
                        "[StreamFrames] received parameter update (preset=%s, flush_cache=%s)",
                        getattr(update, "preset", ""),
                        getattr(update, "flush_cache", False),
                    )

                    success = True
                    message = "Parameter update applied"
                    cache_flushed = False

                    try:
                        with self._frame_compute_lock:
                            active_simulation = session_simulation
                            if active_simulation is None:
                                raise RuntimeError("Simulation context is not ready")
                            applied_settings, state_changed = self._apply_parameter_update(
                                active_simulation, update
                            )
                            if state_changed or update.flush_cache:
                                if state_changed:
                                    self._retire_external_frames(
                                        reason="ray-tracing parameter change"
                                    )
                                reason = (
                                    "ray-tracing parameter change"
                                    if state_changed
                                    else "requested parameter cache flush"
                                )
                                self._invalidate_generated_frames(reason=reason)
                                cache_flushed = True
                                _reset_stream_after_invalidation()
                                self.frames_generated = 0
                                self.frames_failed = 0
                            if applied_settings:
                                logger.info(
                                    "[StreamFrames] Updated raytracing settings: %s",
                                    applied_settings,
                                )
                    except Exception as exc:  # broad catch: gRPC handler
                        success = False
                        message = f"Parameter update failed: {exc}"
                        logger.exception("Parameter update failed")

                    response_pb = visualizer_pb2.ParameterUpdateResponse(
                        success=success,
                        message=message,
                        cache_flushed=cache_flushed,
                        generation_epoch=self.generation_epoch,
                    )
                    if success and update.HasField("raytracing_config"):
                        response_pb.applied_config.CopyFrom(update.raytracing_config)

                    logger.info(
                        "[StreamFrames] sending parameter update response (success=%s, cache_flushed=%s)",
                        success,
                        cache_flushed,
                    )

                    yield visualizer_pb2.FrameResponse(
                        param_update_response=response_pb,
                        generation_epoch=self.generation_epoch,
                    )

                    # After cache flush, DON'T immediately prefetch frames.
                    # Wait for client to send flow control or explicit frame request.
                    # This prevents excessive frame generation immediately after parameter updates.
                    if cache_flushed:
                        logger.info(
                            "[StreamFrames] Cache flushed - waiting for client request before prefetching"
                        )

                elif req_type == "object_update":
                    update = request.object_update
                    msg = (
                        "[StreamFrames] received object update "
                        f"(object_name={update.object_name or 'N/A'}, "
                        f"flush_cache={getattr(update, 'flush_cache', True)}, "
                        f"xml_payload_size={len(update.xml_payload) if update.xml_payload else 0})"
                    )
                    logger.info(msg)

                    success = True
                    message = "Object update applied"
                    cache_flushed = False
                    recomputation_triggered = False

                    try:
                        if update.xml_payload:
                            logger.debug(
                                "[StreamFrames] Processing ObjectUpdate with XML payload (%d bytes)",
                                len(update.xml_payload),
                            )
                            session_simulation = self._activate_live_scene_xml(update.xml_payload)
                            self._retire_external_frames(reason="XML scene update")
                            self._invalidate_generated_frames(reason="XML scene update")
                            _reset_stream_after_invalidation()
                            self.frames_generated = 0
                            self.frames_failed = 0
                            cache_flushed = True
                            recomputation_triggered = True
                            _refresh_simulation_metadata()
                            logger.info(
                                "[StreamFrames] Activated staged XML scene (total_steps=%d)",
                                total_steps,
                            )
                        else:
                            node_type = update.object_type
                            category = category_from_value(node_type)
                            if category is None:
                                success = False
                                try:
                                    node_type_int = int(node_type)
                                except (TypeError, ValueError):
                                    node_type_int = -1
                                if node_type_int == 0:
                                    message = "No xml_payload supplied in ObjectUpdate"
                                    logger.warning(
                                        "[StreamFrames] ObjectUpdate received without xml_payload field"
                                    )
                                else:
                                    message = f"Unsupported node_type '{node_type}' in ObjectUpdate"
                                    logger.warning(
                                        "[StreamFrames] Unsupported node_type in ObjectUpdate: %s",
                                        node_type,
                                    )
                            else:
                                node_name = update.object_name or f"{category}_1"
                                position_tuple = None
                                orientation_tuple = None
                                scale_value = None
                                if update.has_position:
                                    position_tuple = _finite_triple(
                                        (update.x, update.y, update.z),
                                        label=f"Position for {node_name}",
                                    )
                                if update.has_orientation:
                                    orientation_tuple = _finite_triple(
                                        update.orientation,
                                        label=f"Orientation for {node_name}",
                                    )
                                if update.has_scale:
                                    if category != "target":
                                        raise ValueError(
                                            "Scale updates are supported only for targets"
                                        )
                                    scale_value = float(update.scale)
                                    if not math.isfinite(scale_value) or scale_value <= 0.0:
                                        raise ValueError(
                                            f"Scale for {node_name} must be finite and positive"
                                        )
                                if (
                                    position_tuple is None
                                    and orientation_tuple is None
                                    and scale_value is None
                                ):
                                    success = False
                                    message = "Node update missing position/orientation/scale data"
                                    logger.warning(
                                        "[StreamFrames] Node update for %s has no editable fields",
                                        node_name,
                                    )
                                else:
                                    state_changed = _update_node_override(
                                        category,
                                        node_name,
                                        position=position_tuple,
                                        orientation=orientation_tuple,
                                        scale=scale_value,
                                    )
                                    logger.info(
                                        "[StreamFrames] Applied %s override for node '%s' "
                                        "(position=%s, orientation=%s, scale=%s)",
                                        category.upper(),
                                        node_name,
                                        position_tuple,
                                        orientation_tuple,
                                        scale_value,
                                    )
                                    if state_changed or getattr(update, "flush_cache", True):
                                        reason = (
                                            "live node change"
                                            if state_changed
                                            else "requested node cache flush"
                                        )
                                        self._invalidate_generated_frames(reason=reason)
                                        cache_flushed = True
                                        _reset_stream_after_invalidation()
                                        recomputation_triggered = True
                                        self.frames_generated = 0
                                        self.frames_failed = 0
                                    message = f"Updated {category.upper()} node '{node_name}'"
                    except Exception as exc:  # broad catch: gRPC handler
                        success = False
                        message = f"Object update failed: {exc}"
                        logger.exception(
                            "[StreamFrames] ObjectUpdate handling failed with exception"
                        )

                    response_pb = visualizer_pb2.ObjectUpdateResponse(
                        success=success,
                        message=message,
                        object_name=update.object_name,
                        cache_flushed=cache_flushed,
                        recomputation_triggered=recomputation_triggered,
                        generation_epoch=self.generation_epoch,
                    )

                    logger.info(
                        "[StreamFrames] sending object update response (success=%s, cache_flushed=%s)",
                        success,
                        cache_flushed,
                    )

                    yield visualizer_pb2.FrameResponse(
                        object_update_response=response_pb,
                        generation_epoch=self.generation_epoch,
                    )

                    if cache_flushed:
                        logger.info(
                            "[StreamFrames] Object update flushed caches - waiting for client request before prefetching"
                        )

                elif req_type == "cache_flush":
                    flush_req = request.cache_flush
                    reason = flush_req.reason or "Client-initiated cache flush"
                    logger.info(
                        "[StreamFrames] received cache flush request (flush_client=%s, flush_server=%s, reason=%s)",
                        flush_req.flush_client,
                        flush_req.flush_server,
                        reason,
                    )

                    success = True
                    message = "Cache flush completed"
                    frames_cleared = 0
                    server_cache_flushed = False

                    try:
                        if flush_req.flush_server:
                            frames_cleared = self._invalidate_generated_frames(reason=reason)
                            _reset_stream_after_invalidation()
                            self.frames_generated = 0
                            self.frames_failed = 0
                            server_cache_flushed = True
                    except Exception as exc:  # broad catch: gRPC handler
                        success = False
                        message = f"Cache flush failed: {exc}"
                        logger.exception("Cache flush request failed")

                    response_pb = visualizer_pb2.CacheFlushResponse(
                        success=success,
                        message=message,
                        client_cache_flushed=bool(flush_req.flush_client),
                        server_cache_flushed=server_cache_flushed,
                        frames_cleared=frames_cleared,
                        generation_epoch=self.generation_epoch,
                    )

                    yield visualizer_pb2.FrameResponse(
                        cache_flush_response=response_pb,
                        generation_epoch=self.generation_epoch,
                    )

                    if not success:
                        break

                    # If we flushed the server cache, wait for explicit client flow control/get_frame
                    if server_cache_flushed:
                        logger.info(
                            "[StreamFrames] Server cache flushed - waiting for client request before prefetching"
                        )

                elif req_type == "get_frame":
                    frame_command = request.get_frame
                    requested_idx = (
                        frame_command.frame_idx if frame_command.frame_idx >= 0 else next_frame_idx
                    )

                    # Check if requested frame is beyond total_steps
                    if total_steps and requested_idx >= total_steps:
                        logger.warning(
                            "[StreamFrames] Client requested frame %s beyond total_steps (%s), sending EOF",
                            requested_idx,
                            total_steps,
                        )
                        if not sent_eof:
                            sent_eof = True
                            yield _emit_eof(
                                total_steps - 1 if total_steps > 0 else requested_idx - 1
                            )
                        break

                    logger.debug("[StreamFrames] explicit frame request: idx=%s", requested_idx)
                    response = _produce_frame(requested_idx)
                    if response is not None:
                        yield response

                    if response is None or response.WhichOneof("response_type") != "frame_data":
                        logger.warning(
                            "[StreamFrames] Stopping stream after get_frame error or missing response"
                        )
                        break

                    buffer_fill = max(0, buffer_fill - 1)
                    # Ensure next_frame_idx is non-negative
                    next_frame_idx = max(0, requested_idx + 1)

                    # After explicit frame request, compute ahead and fill buffer incrementally
                    # This ensures frames are sent as soon as they're computed
                    for _ in _compute_ahead():
                        # After each frame computation, send any frames that are ready
                        for resp in _fill_buffer():
                            yield resp
                            if resp.WhichOneof("response_type") != "frame_data":
                                break

                    # Final pass: send any remaining frames
                    for resp in _fill_buffer():
                        yield resp
                        if resp.WhichOneof("response_type") != "frame_data":
                            break

                elif req_type == "flow_control":
                    signal = request.flow_control
                    max_flow_window = self.frame_cache.max_frames
                    requested_capacity = int(signal.buffer_capacity)
                    requested_prefetch = int(signal.server_prefetch_limit)
                    oversized_fields = [
                        ("buffer_capacity", requested_capacity),
                        ("server_prefetch_limit", requested_prefetch),
                    ]
                    oversized = next(
                        (
                            (name, value)
                            for name, value in oversized_fields
                            if value > max_flow_window
                        ),
                        None,
                    )
                    if oversized is not None:
                        field_name, field_value = oversized
                        yield visualizer_pb2.FrameResponse(
                            error=visualizer_pb2.ErrorDetails(
                                code="INVALID_FLOW_CONTROL",
                                message=(
                                    f"{field_name}={field_value} exceeds the live "
                                    f"frame-cache capacity of {max_flow_window}"
                                ),
                            ),
                            generation_epoch=self.generation_epoch,
                        )
                        continue

                    old_next_frame = next_frame_idx
                    old_buffer_fill = buffer_fill
                    old_frames_computed = frames_computed

                    # Zero pauses transmission; negative values leave the current limit unchanged.
                    if requested_capacity >= 0:
                        buffer_capacity = requested_capacity

                    if requested_prefetch >= 0:
                        server_prefetch_limit = requested_prefetch
                    elif server_prefetch_limit == 0 and buffer_capacity > 0:
                        server_prefetch_limit = buffer_capacity

                    consumed = max(0, int(signal.frames_consumed))
                    if consumed:
                        buffer_fill = max(0, buffer_fill - consumed)

                    # Handle initial connection: if current_display is -1, explicitly start from frame 0
                    if signal.current_display_frame_idx < 0:
                        # Initial connection: client hasn't displayed any frame yet, start from beginning
                        if next_frame_idx != 0:
                            logger.debug(
                                "[StreamFrames] INITIAL CONNECTION: resetting next_frame from %s to 0",
                                next_frame_idx,
                            )
                        next_frame_idx = 0
                        buffer_fill = 0
                        frames_computed = 0
                        logger.info(
                            "[StreamFrames] INITIAL CONNECTION: starting from frame 0 "
                            "(buffer_capacity=%s=transmission limit, server_prefetch_limit=%s=computation limit)",
                            buffer_capacity,
                            server_prefetch_limit,
                        )
                    elif signal.current_display_frame_idx >= 0:
                        # If client reports a position, use it to set next_frame
                        # This allows client to "reset" by sending a lower frame number
                        client_current_frame = int(signal.current_display_frame_idx)
                        client_next_frame = client_current_frame + 1

                        # Only update if client is ahead OR if frames_consumed=0 (which indicates a reset/buffer clear)
                        if consumed == 0:
                            # frames_consumed=0 with a position means client cleared buffer or reset
                            # Respect the client's position - reset next_frame to client's position + 1
                            logger.debug(
                                "[StreamFrames] BUFFER CLEAR detected: client at frame %s, resetting next_frame from %s to %s",
                                client_current_frame,
                                next_frame_idx,
                                client_next_frame,
                            )
                            # Ensure next_frame_idx is non-negative
                            next_frame_idx = max(0, client_next_frame)
                            # Also reset buffer_fill since client cleared their buffer
                            buffer_fill = 0
                        else:
                            # Normal flow control - only move forward
                            if client_next_frame > next_frame_idx:
                                logger.debug(
                                    "[StreamFrames] Client ahead: moving next_frame from %s to %s",
                                    next_frame_idx,
                                    client_next_frame,
                                )
                                # Ensure next_frame_idx is non-negative
                                next_frame_idx = max(0, client_next_frame)
                            else:
                                logger.debug(
                                    "[StreamFrames] Client at %s, keeping next_frame at %s",
                                    client_current_frame,
                                    next_frame_idx,
                                )
                                # Ensure next_frame_idx is non-negative
                                next_frame_idx = max(0, next_frame_idx)

                    logger.debug(
                        "[StreamFrames] flow control update: buffer_capacity=%s, server_prefetch_limit=%s, frames_consumed=%s, "
                        "current_display=%s, buffer_fill: %s -> %s, frames_computed: %s -> %s, next_frame: %s -> %s",
                        buffer_capacity,
                        server_prefetch_limit,
                        consumed,
                        signal.current_display_frame_idx,
                        old_buffer_fill,
                        buffer_fill,
                        old_frames_computed,
                        frames_computed,
                        old_next_frame,
                        next_frame_idx,
                    )

                    # Interleave computation and sending to avoid blocking:
                    # 1. Send any frames that are ready immediately
                    # 2. Compute a few frames ahead
                    # 3. Send newly computed frames
                    # Repeat until buffer is full or prefetch limit reached

                    # First, send any frames that are already in cache
                    for resp in _fill_buffer():
                        yield resp
                        if resp.WhichOneof("response_type") != "frame_data":
                            break

                    # Then compute ahead incrementally, sending frames as they become available
                    # This prevents blocking when computing many frames
                    for _ in _compute_ahead():
                        # After each frame computation, send any frames that are ready
                        for resp in _fill_buffer():
                            yield resp
                            if resp.WhichOneof("response_type") != "frame_data":
                                break

                    # Final pass: send any remaining frames that fit in buffer_capacity
                    for resp in _fill_buffer():
                        yield resp
                        if resp.WhichOneof("response_type") != "frame_data":
                            break

                else:
                    logger.debug("Received FrameRequest without a request_type; ignoring")

            if not sent_eof and total_steps and next_frame_idx >= total_steps:
                yield _emit_eof(total_steps - 1 if total_steps > 0 else next_frame_idx - 1)

        except grpc.RpcError as exc:
            status = None
            try:  # some environments raise base RpcError without code()
                status = exc.code()
            except AttributeError:
                status = None

            if status == grpc.StatusCode.CANCELLED or not context.is_active():
                logger.debug("StreamFrames ended after client disconnect")
            else:
                logger.exception("StreamFrames terminated with RPC error")
                yield visualizer_pb2.FrameResponse(
                    error=visualizer_pb2.ErrorDetails(
                        code="STREAM_RPC_ERROR",
                        message=str(exc),
                    ),
                    frame_idx=next_frame_idx,
                    generation_epoch=self.generation_epoch,
                )
        except Exception:  # broad catch: gRPC handler top-level
            if context is not None and not context.is_active():
                logger.debug("StreamFrames ended after inactive client context")
                return
            logger.exception("StreamFrames encountered an unexpected error")
            yield visualizer_pb2.FrameResponse(
                error=visualizer_pb2.ErrorDetails(
                    code="STREAM_ERROR",
                    message="Server-side error during streaming",
                ),
                frame_idx=next_frame_idx,
                generation_epoch=self.generation_epoch,
            )

        finally:
            try:
                if session_overrides_changed:
                    with self._frame_compute_lock:
                        active_simulation = self._ensure_simulation_context()
                        if active_simulation is not None:
                            apply_target_scale_overrides(
                                list(getattr(active_simulation, "target_managers", ()) or ()),
                                None,
                            )
                        self._invalidate_generated_frames(reason="live controller release")
            finally:
                self._release_controller()

    def GetGeneratorStatus(self, request, context):
        """Get generator status"""
        try:
            uptime = time.time() - self.start_time

            config = visualizer_pb2.GeneratorConfig(
                data_mode=self.generator_config.get("data_mode", "unknown"),
                motion_mode=self.generator_config.get("motion_mode", "unknown"),
                num_steps=self.generator_config.get("num_steps", 0),
                duration=self.generator_config.get("duration", 0.0),
                output_mode=self.generator_config.get("output_mode", "unknown"),
                enabled_patterns=self.generator_config.get("enabled_patterns", []),
            )

            return visualizer_pb2.GetGeneratorStatusResponse(
                success=True,
                message="Generator status retrieved successfully",
                is_ready=self.is_ready,
                is_streaming=self.is_streaming,
                frames_generated=self.frames_generated,
                uptime=uptime,
                config=config,
                generation_epoch=self.generation_epoch,
            )

        except Exception as e:  # broad catch: gRPC handler
            logger.error(f"Error getting generator status: {e}")
            return visualizer_pb2.GetGeneratorStatusResponse(
                success=False, message=f"Error: {str(e)}"
            )

    def _convert_to_protobuf_frame(
        self,
        frame_data: Dict[str, Any],
        frame_idx: int = -1,
    ) -> Any:
        """Normalize raw frame data and wrap it in the transport ``FrameData``."""
        standard_mpc_frame_pb = self._convert_to_standard_mpc_frame(frame_data, frame_idx)

        return visualizer_pb2.FrameData(standard_mpc_frame=standard_mpc_frame_pb)

    def _convert_to_standard_mpc_frame(self, frame_data: Dict[str, Any], frame_idx: int) -> Any:
        """Normalize and encode one frame, propagating contract failures."""
        standard_frame = standard_mpc_frame_from_raw(
            frame_data,
            frame_idx,
            source_provider="generator_grpc",
            simulation_config=frame_data.get("simulation_config") or self._simulation_config(),
        )
        if standard_frame.frame_index != frame_idx:
            raise ValueError(
                f"Normalized frame index {standard_frame.frame_index} does not match "
                f"requested frame {frame_idx}"
            )
        return standard_mpc_frame_to_proto(standard_frame)


def add_test_frame_data(frame_cache: GeneratorFrameCache, num_frames: int = 10):
    """Add test frame data to the cache for demonstration purposes"""

    logger.info("[STATS] Adding %d test frames to cache...", num_frames)

    for frame_idx in range(num_frames):
        # Create test frame data in the format expected by _convert_to_protobuf_frame
        frame_data = {
            "tx_positions": [
                {
                    "name": f"tx_{i}",
                    "x": float(i * 5.0),
                    "y": 0.0,
                    "z": 2.0,
                    "orientation": [0.0, 0.0, 0.0],
                }
                for i in range(2)
            ],
            "rx_positions": [
                {
                    "name": f"rx_{i}",
                    "x": float(10.0 + i * 5.0),
                    "y": float(i * 5.0),
                    "z": 1.5,
                    "orientation": [0.0, 0.0, 0.0],
                }
                for i in range(2)
            ],
            "targets": [
                {
                    "name": "test_target",
                    "x": 5.0 + frame_idx * 0.1,  # Move target slightly each frame
                    "y": 5.0 + frame_idx * 0.1,
                    "z": 1.0,
                    "orientation": [0.0, 0.0, float(frame_idx * 10.0)],  # Rotate target
                    "mesh_name": "test_mesh",
                    "mesh_index": 0,
                    "scale": 1.0,
                    "material_type": "default",
                }
            ],
            "paths": {
                "total_paths": 4,
                "valid_paths": 4,
                "path_types": [
                    {"path_type": "LOS", "count": 2, "percentage": 50.0},
                    {"path_type": "Reflection", "count": 2, "percentage": 50.0},
                ],
                "depth_info": [
                    {
                        "depth": 0,
                        "total_interactions": 2,
                        "interactions": [{"path_type": "LOS", "count": 2, "percentage": 100.0}],
                    },
                    {
                        "depth": 1,
                        "total_interactions": 2,
                        "interactions": [
                            {"path_type": "Reflection", "count": 2, "percentage": 100.0}
                        ],
                    },
                ],
            },
            "scene_info": {
                "scene_name": "test_scene",
                "frequency": 2.4e9,
                "quality": "high",
                "material_type": "default",
                "bounds": [-10.0, -10.0, -10.0, 20.0, 20.0, 10.0],
            },
        }

        # Add frame to cache
        frame_cache.add_frame(frame_idx, frame_data)

    logger.info("Added %d test frames to cache", num_frames)


@overload
def run_generator_server(
    port: int = 50051,
    generator_config: Dict[str, Any] | None = None,
    frame_cache: GeneratorFrameCache | None = None,
    *,
    bind_host: str = DEFAULT_GRPC_BIND_HOST,
    frame_dispatcher: Any = None,
    start_in_background: Literal[False] = False,
    provider_timeout_s: float = 5.0,
) -> GeneratorFrameCache: ...


@overload
def run_generator_server(
    port: int = 50051,
    generator_config: Dict[str, Any] | None = None,
    frame_cache: GeneratorFrameCache | None = None,
    *,
    bind_host: str = DEFAULT_GRPC_BIND_HOST,
    frame_dispatcher: Any = None,
    start_in_background: Literal[True] = True,
    provider_timeout_s: float = 5.0,
) -> tuple[grpc.Server, GeneratorService, GeneratorFrameCache]: ...


@overload
def run_generator_server(
    port: int = 50051,
    generator_config: Dict[str, Any] | None = None,
    frame_cache: GeneratorFrameCache | None = None,
    *,
    bind_host: str = DEFAULT_GRPC_BIND_HOST,
    frame_dispatcher: Any = None,
    start_in_background: bool = False,
    provider_timeout_s: float = 5.0,
) -> GeneratorFrameCache | tuple[grpc.Server, GeneratorService, GeneratorFrameCache]: ...


def run_generator_server(
    port: int = 50051,
    generator_config: Dict[str, Any] | None = None,
    frame_cache: GeneratorFrameCache | None = None,
    *,
    bind_host: str = DEFAULT_GRPC_BIND_HOST,
    frame_dispatcher: Any = None,
    start_in_background: bool = False,
    provider_timeout_s: float = 5.0,
) -> GeneratorFrameCache | tuple[grpc.Server, GeneratorService, GeneratorFrameCache]:
    """Run the live generator gRPC server and return its frame cache/handles."""
    # Configure logging FIRST - respect scenario.yaml debug_level setting
    sim_cfg = generator_config.get("simulation_config") if generator_config else None
    level = getattr(sim_cfg, "debug_level", "WARNING")
    try:
        configure_logging(level=level)
    except (ValueError, OSError) as exc:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.WARNING,
                format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
            )
        get_logger(__name__).warning("Logging configuration failed: %s", exc)

    # Ensure module logger propagates to root (but don't override the level set by configure_logging)
    module_logger = get_logger(__name__)
    module_logger.propagate = True

    # Log that we're starting the server (this will verify logging works)
    logger.info("=" * 60)
    logger.info("Generator gRPC Server starting - logging system initialized")
    logger.info("=" * 60)

    if generator_config is None:
        generator_config = {
            "data_mode": "live_grpc",
            "motion_mode": "step",
            "num_steps": 100,
            "duration": 100.0,
            "output_mode": "grpc",
            "enabled_patterns": ["mobility", "orientation"],
        }

    logger.info("Generator gRPC Server endpoint: %s", format_grpc_endpoint(bind_host, port))
    if not is_loopback_grpc_host(bind_host):
        logger.warning(
            "Live generator gRPC is binding to non-loopback host %s. "
            "Use this only on a trusted network; this service has no authentication or TLS.",
            bind_host,
        )
    logger.info("Generator will serve frame data to visualizers on demand")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Share the provided cache or allocate one for this server.
    standalone_mode = frame_dispatcher is None and not start_in_background
    if frame_cache is None:
        frame_cache = GeneratorFrameCache()
        if standalone_mode:
            add_test_frame_data(frame_cache, num_frames=20)

    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=GRPC_MESSAGE_OPTIONS,
    )
    generator_service = GeneratorService(
        frame_cache,
        generator_config,
        frame_provider=frame_dispatcher,
        provider_timeout_s=provider_timeout_s,
    )
    visualizer_pb2_grpc.add_GeneratorServiceServicer_to_server(generator_service, server)

    # Start server
    listen_addr, bound_port = bind_grpc_server(server, bind_host, port)
    server.start()

    if bound_port != port:
        listen_addr = format_grpc_endpoint(bind_host, bound_port)
    logger.info("Generator gRPC server started on %s", listen_addr)
    if not start_in_background:
        logger.info("Waiting for visualizer connections...")

    if not start_in_background:
        stats_stop = threading.Event()

        def display_stats():
            while not stats_stop.wait(5.0):
                available_frames = frame_cache.get_available_frames()
                if available_frames:
                    logger.info(
                        "Stats: %d frames available, range: %d-%d",
                        len(available_frames),
                        min(available_frames),
                        max(available_frames),
                    )

        stats_thread = threading.Thread(target=display_stats, daemon=True)
        stats_thread.start()

        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Stopping Generator gRPC server...")
        finally:
            stats_stop.set()
            server.stop(DEFAULT_GRPC_SHUTDOWN_GRACE_S).wait()
            generator_service.close()
            stats_thread.join(timeout=1.0)
            logger.info("Generator gRPC server stopped")
        return frame_cache

    return server, generator_service, frame_cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generator gRPC Server")
    parser.add_argument(
        "--port", type=int, default=50051, help="Port to listen on (default: 50051)"
    )
    parser.add_argument(
        "--bind-host",
        default=DEFAULT_GRPC_BIND_HOST,
        help="Host/interface to bind (default: 127.0.0.1; use non-loopback only on trusted LANs)",
    )
    args = parser.parse_args()

    run_generator_server(args.port, bind_host=args.bind_host)
