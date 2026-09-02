"""Scenario selection, loading, and live-source setup service."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import yaml

from shared.geometry.transforms import UnsupportedLightweightTransformError
from shared.logging import (
    get_current_log_level_name,
    get_logger,
    resolve_log_level,
    set_log_level,
)
from shared.scenarios.paths import PathPolicy, create_path_policy

from ..app.dialog_manager import DialogManager
from ..io.frame_sources import LiveGrpcSource, make_frame_source
from ..io.scenario_config import find_project_root, load_app_config, load_scenario
from ..scene.defaults import DEFAULT_NODE_MARKER_SIZE_M, NODE_MARKER_SIZE_BOUNDS_M
from ..scene.io import XMLSceneHandler, build_scene_from_root
from ..services.base import BaseService
from ..services.cache_service import CacheInvalidationScope, invalidate_visualizer_cache

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.scenario_loader")


@dataclass(frozen=True, slots=True)
class ScenarioLoadResult:
    """Resources prepared by one successful scenario-load attempt."""

    scenario: Any
    frame_source: Any
    app_config: Any
    frame_source_ready: bool


@dataclass(frozen=True, slots=True)
class ScenarioPreflightResult:
    """Read-only resources validated before replacing the active scenario."""

    requested_path: str
    scenario: Any
    app_config: Any
    path_policy: PathPolicy
    scene_xml_path: Path
    scene_xml_root: Any
    scene_mesh_entries: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ScenarioFrameSourcePreparation:
    """Opened candidate source validated before the active scenario is retired."""

    frame_source: Any
    frame_source_ready: bool
    available_frames: tuple[int, ...]


class ScenarioLoaderService(BaseService):
    """Encapsulates scenario selection, saving, and dialog flows."""

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        dialog_manager: DialogManager,
        project_root_resolver: Optional[Callable[[], Path]] = None,
    ) -> None:
        """Bind scenario loading to dialog and project-root helpers."""
        super().__init__()
        self.visualizer = visualizer
        self.dialog_manager = dialog_manager
        self._project_root_resolver = project_root_resolver
        self._online_source_cls = None
        self._last_frame_source_preparation_error: str | None = None

    @property
    def last_frame_source_preparation_error(self) -> str | None:
        """Return the user-facing error from the latest candidate-source probe."""
        return self._last_frame_source_preparation_error

    # Scenario selection helpers
    def _report_open_error(self, message: str) -> None:
        """Report expected load errors without blocking non-modal launch modes."""

        non_modal = (
            bool(getattr(self.visualizer, "_cli_driven_frame_run", False))
            or bool(getattr(self.visualizer, "_explicit_cli_scenario_startup", False))
            or getattr(self.visualizer, "_viewport_mode", "detached") == "embedded"
        )
        if non_modal:
            set_status = getattr(self.visualizer, "_set_status_message", None)
            if callable(set_status):
                set_status(message, 5000)
            return
        self.dialog_manager.show_error("Scenario Error", message)

    def open_scenario_via_dialog(self) -> bool:
        """Use DialogManager to pick a scenario and delegate loading."""
        if bool(getattr(self.visualizer, "_scenario_load_in_progress", False)):
            return False
        scenario_path = self.dialog_manager.select_scenario_file(self._default_directory())
        if not scenario_path:
            return False

        scenario_dir = scenario_path.parent if scenario_path.is_file() else scenario_path
        scenario_path_str = self._relative_to_project_root(scenario_dir)

        logger.info("Opening scenario via dialog: %s", scenario_path_str)
        outcome = self.visualizer.open_scenario(scenario_path_str)
        return bool(getattr(outcome, "succeeded", False))

    def save_scene_xml(self, xml_root: Any) -> bool:
        """Persist the current XML scene through the dialog flow."""
        if xml_root is None:
            logger.warning("No XML root available; skipping save")
            return False

        out = self.dialog_manager.select_save_file(
            title="Save XML As",
            default_name="scene_modified.xml",
            filter_str="XML Files (*.xml)",
        )
        if not out:
            return False

        try:
            XMLSceneHandler.debug_xml_structure(xml_root, "Before Saving")
            XMLSceneHandler.save_xml_scene(xml_root, out)
            self.dialog_manager.show_info("Saved", f"XML saved to {out}")
            return True
        except (OSError, IOError, ValueError, TypeError) as exc:  # pragma: no cover
            logger.error("Failed to save XML: %s", exc)
            self.dialog_manager.show_error("Save Error", f"Failed to save XML: {exc}")
            return False

    # Node editing feedback helpers
    def validate_node_editing_context(self, frame_source: Any) -> bool:
        """Ensure node editing is available and show the right dialogs otherwise."""
        source_cls = self._resolve_online_source_cls()
        if source_cls is None:
            self.dialog_manager.show_warning("Unavailable", "Node editing requires live gRPC mode.")
            return False

        if not isinstance(frame_source, source_cls):
            self.dialog_manager.show_info(
                "Offline Mode",
                "Connect to a generator in live gRPC mode to edit node properties.",
            )
            return False

        return True

    def warn_unsupported_node(self) -> None:
        """Warn that the selected object cannot be edited through live node RPC."""
        self.dialog_manager.show_warning(
            "Unsupported", "Only target, TX, or RX nodes can be edited."
        )

    def warn_missing_identifier(self) -> None:
        """Warn that a node edit lacks a stable target/TX/RX identifier."""
        self.dialog_manager.show_warning(
            "Missing Identifier", "Cannot determine the node name to update."
        )

    def inform_no_changes(self) -> None:
        """Inform the user that the edit dialog contains no changes to send."""
        self.dialog_manager.show_info(
            "No Changes", "Update at least one property before applying changes."
        )

    def show_node_update_error(self, title: str, message: str) -> None:
        """Show a node-update failure through the dialog manager."""
        self.dialog_manager.show_error(title, message)

    # Internal helpers
    def _default_directory(self) -> Optional[str]:
        """Return the default scenario picker directory if the project root resolves."""
        try:
            return str(self._resolve_project_root() / "scenarios")
        except (OSError, ValueError):
            return None

    def _relative_to_project_root(self, scenario_dir: Path) -> str:
        """Return a scenario path relative to the project root when possible."""
        try:
            root = self._resolve_project_root()
            return str(scenario_dir.relative_to(root))
        except (OSError, ValueError):
            return str(scenario_dir)

    def _resolve_project_root(self) -> Path:
        """Resolve the active project root through injected or default policy."""
        if self._project_root_resolver is not None:
            return self._project_root_resolver()
        return find_project_root()

    def _resolve_online_source_cls(self):
        """Resolve and cache the live-gRPC frame source class if available."""
        if self._online_source_cls is not None:
            return self._online_source_cls
        try:
            self._online_source_cls = LiveGrpcSource
        except ImportError:
            self._online_source_cls = None
        return self._online_source_cls

    def load_scenario(
        self,
        scenario_path: str,
        *,
        cleanup_scene_first: bool = True,
        preflight: ScenarioPreflightResult | None = None,
        prepared_frame_source: ScenarioFrameSourcePreparation | None = None,
    ) -> ScenarioLoadResult | None:
        """
        Load a scenario and return the loaded scenario object and frame source.

        Args:
            scenario_path: Path to scenario folder or YAML file (relative to project root)
            cleanup_scene_first: Whether XML loading must retire the current scene. The
                app scenario workflow passes ``False`` because it already performs the
                broader scenario cleanup before invoking this service.
            preflight: Validated, read-only scenario resources. Supplying
                this prevents reparsing the configuration after the app has retired
                the active scenario.
            prepared_frame_source: Candidate source opened and probed before the
                destructive scenario transition. Ownership transfers to this method.

        Returns:
            Named scenario resources, or ``None`` when expected input/configuration
            errors prevent the scenario from loading.
        """
        viz = self.visualizer
        if preflight is None:
            preflight = self.preflight_scenario(scenario_path)
        if preflight is None:
            return None

        load_completed = False
        try:

            def _record_stage(name: str, started_at: float) -> None:
                """Record scenario-load stage timing when startup telemetry is active."""
                recorder = getattr(viz, "record_startup_stage_timing", None)
                if callable(recorder):
                    recorder(name, (time.perf_counter() - started_at) * 1000.0)

            logger.info("Opening scenario: %s", scenario_path)
            viz.progress.note(f"Opening scenario {scenario_path}")
            viz._scene_boot_start = time.perf_counter()
            viz._scene_boot_logged = False
            viz.scene_boot_duration_ms = None
            viz.last_frame_duration_ms = None
            playback_cadence = getattr(viz, "playback_cadence", None)
            if playback_cadence is not None:
                playback_cadence.reset()
                viz.ui_controller.refresh_status_telemetry()

            policy = preflight.path_policy
            PROJECT_ROOT = policy.project_root
            BASE_DIR = policy.config_dir

            logger.debug("Project root: %s", PROJECT_ROOT)
            logger.debug("Base directory: %s", BASE_DIR)

            app_config = preflight.app_config
            scenario = preflight.scenario
            logger.info("Scenario loaded: %s", scenario.root)

            viz.scenario = scenario
            self._sync_node_actor_names_from_scenario(scenario)
            self._sync_node_marker_config_from_scenario(scenario)

            effective_log_level = self._apply_scenario_log_level(scenario.debug_level)
            logger.debug("Effective log level set to: %s", effective_log_level)

            viz.current_scenario_policy = policy
            viz.current_project_root = PROJECT_ROOT
            viz.current_base_dir = BASE_DIR

            # 3) Load scene from XML
            stage_start = time.perf_counter()
            with viz.progress.task("Loading scene geometry"):
                self._load_scene_xml(
                    preflight.scene_xml_path,
                    xml_root=preflight.scene_xml_root,
                    mesh_entries=preflight.scene_mesh_entries,
                    cleanup_first=cleanup_scene_first,
                )
            _record_stage("scenario_loader.load_scene_geometry_ms", stage_start)

            # Install a supplied source or create and initialize one for direct calls.
            frame_source = (
                prepared_frame_source.frame_source if prepared_frame_source is not None else None
            )
            frame_source_ready = (
                prepared_frame_source.frame_source_ready
                if prepared_frame_source is not None
                else False
            )
            try:
                stage_start = time.perf_counter()
                with viz.progress.task("Creating frame source"):
                    if frame_source is None:
                        frame_source = make_frame_source(scenario)
                        self._setup_frame_source(frame_source)
                    viz.frame_source = frame_source
                    logger.debug("Frame source set as main frame source")
                    viz.ui_controller.update_file_source_summary()

                    # Apply the authored timeline to playback controls.
                    self._update_animation_steps_from_scenario(scenario, viz)

                    if hasattr(viz.mpc_core, "set_frame_source"):
                        viz.mpc_core.set_frame_source(frame_source)
                        logger.debug("Frame source set on MPCCore")
                _record_stage("scenario_loader.create_frame_source_ms", stage_start)

                if prepared_frame_source is None:
                    frame_source_ready = True
            except (ValueError, AttributeError, RuntimeError, TypeError, OSError) as e:
                close_source = getattr(frame_source, "close", None)
                if callable(close_source):
                    try:
                        close_source()
                    except (OSError, RuntimeError):
                        logger.debug("Could not close failed frame source", exc_info=True)
                viz.frame_source = None
                raise RuntimeError(f"Could not set up frame source: {e}") from e

            # Probe for actual frames — even if source creation succeeded,
            # there may be no HDF5 files yet (scene-only mode).
            if prepared_frame_source is None and frame_source_ready and frame_source is not None:
                try:
                    stage_start = time.perf_counter()
                    if not frame_source.list_frames():
                        logger.info("No frames found; scene-only mode")
                        frame_source_ready = False
                    _record_stage("scenario_loader.frame_probe_ms", stage_start)
                except (OSError, RuntimeError) as exc:
                    close_source = getattr(frame_source, "close", None)
                    if callable(close_source):
                        try:
                            close_source()
                        except (OSError, RuntimeError):
                            logger.debug("Could not close failed frame source", exc_info=True)
                    viz.frame_source = None
                    raise RuntimeError(f"Could not inspect frame source: {exc}") from exc

            result = ScenarioLoadResult(
                scenario=scenario,
                frame_source=frame_source,
                app_config=app_config,
                frame_source_ready=frame_source_ready,
            )
            load_completed = True
            return result

        except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError) as e:
            logger.error(f"Failed to load scenario {scenario_path}: {e}")
            viz._scene_boot_start = None
            viz._scene_boot_logged = False
            self._report_open_error(f"Failed to open scenario:\n{str(e)}")
            return None
        finally:
            if prepared_frame_source is not None and not load_completed:
                close_source = getattr(prepared_frame_source.frame_source, "close", None)
                if callable(close_source):
                    try:
                        close_source()
                    except (OSError, RuntimeError):
                        logger.debug(
                            "Could not close rejected prepared frame source", exc_info=True
                        )
                if getattr(viz, "frame_source", None) is prepared_frame_source.frame_source:
                    viz.frame_source = None

    def prepare_frame_source(
        self,
        preflight: ScenarioPreflightResult,
    ) -> ScenarioFrameSourcePreparation | None:
        """Open and probe a candidate source without mutating visualizer state.

        This is deliberately separate from configuration/asset preflight because
        it may contact a remote endpoint. The app workflow calls it before tearing
        down the active scenario, so connection, index, and first-frame failures
        leave the current workspace usable.
        """
        frame_source = None
        self._last_frame_source_preparation_error = None
        try:
            scenario = preflight.scenario
            frame_source = make_frame_source(scenario)
            self._setup_frame_source(frame_source)
            available_frames = tuple(sorted(int(frame) for frame in frame_source.list_frames()))
            frame_source_ready = bool(available_frames)

            data_mode = str(getattr(scenario, "data_mode", "files"))
            if frame_source_ready and data_mode in {"files", "remote_hdf5"}:
                first_frame = available_frames[0]
                if frame_source.load_frame(first_frame) is None:
                    raise RuntimeError(f"Frame {first_frame} returned no data")

            if not frame_source_ready:
                logger.info("No frames found; candidate will open in scene-only mode")
            return ScenarioFrameSourcePreparation(
                frame_source=frame_source,
                frame_source_ready=frame_source_ready,
                available_frames=available_frames,
            )
        except Exception as exc:  # source/extension boundary: preserve active workspace
            close_source = getattr(frame_source, "close", None)
            if callable(close_source):
                try:
                    close_source()
                except Exception:
                    logger.debug("Could not close failed candidate frame source", exc_info=True)
            logger.error("Scenario frame-source preparation failed: %s", exc)
            self._last_frame_source_preparation_error = (
                f"Failed to prepare scenario data source: {exc}"
            )
            self._report_open_error(f"Failed to prepare scenario data source:\n{exc}")
            return None

    def preflight_scenario(
        self,
        scenario_path: str,
        *,
        data_mode_override: str | None = None,
        grpc_port_override: int | None = None,
    ) -> ScenarioPreflightResult | None:
        """Validate local scenario inputs without mutating visualizer state.

        Parsing, path-policy construction, scene XML resolution, and frame-source
        configuration all happen before the app tears down its current scenario.
        Remote endpoints are deliberately not contacted during preflight.
        Optional data-source overrides are applied to this candidate only, so
        probing and the later committed load use the same effective provider.
        """
        try:
            requested = Path(scenario_path).expanduser()
            if not requested.is_absolute():
                requested = self._resolve_project_root() / requested
            requested = requested.resolve(strict=False)
            yaml_path = requested / "scenario.yaml" if requested.is_dir() else requested

            policy = create_path_policy(yaml_path)
            app_config = load_app_config()
            scenario = load_scenario(
                requested,
                app_config,
                data_mode_override=data_mode_override,
                grpc_port_override=grpc_port_override,
            )
            scene_xml_path = self._resolve_scene_xml(scenario, app_config)
            scene_xml_root = XMLSceneHandler.load_xml_scene(str(scene_xml_path))
            scene_mesh_entries = build_scene_from_root(
                scene_xml_root,
                str(scene_xml_path),
            )
            self._validate_frame_source_configuration(scenario)
            return ScenarioPreflightResult(
                requested_path=str(scenario_path),
                scenario=scenario,
                app_config=app_config,
                path_policy=policy,
                scene_xml_path=scene_xml_path,
                scene_xml_root=scene_xml_root,
                scene_mesh_entries=scene_mesh_entries,
            )
        except (
            UnsupportedLightweightTransformError,
            ImportError,
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            SyntaxError,
            yaml.YAMLError,
        ) as exc:
            logger.error("Scenario preflight failed for %s: %s", scenario_path, exc)
            self._report_open_error(f"Failed to open scenario:\n{str(exc)}")
            return None

    @staticmethod
    def _validate_frame_source_configuration(scenario: Any) -> None:
        """Validate locally checkable frame-source settings without connecting."""
        data_spec = getattr(scenario, "data_spec", {})
        if data_spec is None:
            data_spec = {}
        if not isinstance(data_spec, Mapping):
            raise TypeError("Scenario data configuration must be a mapping")

        data_mode = str(getattr(scenario, "data_mode", "files"))
        if data_mode == "files":
            files_spec = data_spec.get("files", {})
            if files_spec is None:
                files_spec = {}
            if not isinstance(files_spec, Mapping):
                raise TypeError("data.files must be a mapping")
            frame_format = str(files_spec.get("format", "h5")).lower()
            if frame_format not in {"h5", "hdf5"}:
                raise ValueError(
                    f"Unsupported frame format: {frame_format}. Only HDF5 (hdf5/h5) is supported."
                )
            return

        if data_mode == "live_grpc":
            live_spec = data_spec.get("live_grpc", {})
            if live_spec is None:
                live_spec = {}
            if not isinstance(live_spec, Mapping):
                raise TypeError("data.live_grpc must be a mapping")
            buffer_size = int(live_spec.get("buffer_size", 50))
            if buffer_size < 1:
                raise ValueError("live_grpc.buffer_size must be >= 1")
            return

        if data_mode == "remote_hdf5":
            remote_spec = data_spec.get("remote_hdf5", {})
            if remote_spec is None:
                remote_spec = {}
            if not isinstance(remote_spec, Mapping):
                raise TypeError("data.remote_hdf5 must be a mapping")
            cache_size = int(remote_spec.get("cache_size", 50))
            connect_timeout = float(remote_spec.get("connect_timeout", 10.0))
            frame_index_ttl_s = float(remote_spec.get("frame_index_ttl_s", 0.0))
            if cache_size < 1:
                raise ValueError("remote_hdf5.cache_size must be >= 1")
            if connect_timeout <= 0.0:
                raise ValueError("remote_hdf5.connect_timeout must be > 0")
            if frame_index_ttl_s < 0.0:
                raise ValueError("remote_hdf5.frame_index_ttl_s must be >= 0")
            return

        from ..io.frame_source_extensions import registered_frame_source_modes

        if data_mode not in registered_frame_source_modes():
            raise ValueError(f"Unknown data mode: {data_mode}")

    def _sync_node_actor_names_from_scenario(self, scenario: Any) -> None:
        """Copy TX/RX actor names into visualizer state for name-mode labels."""
        viz = self.visualizer
        actors = getattr(scenario, "actors", None)

        def _names(kind: str) -> tuple[str, ...]:
            """Return configured actor names for one radio role."""

            entries = getattr(actors, kind, ()) if actors is not None else ()
            return tuple(str(getattr(entry, "name", "") or "") for entry in entries)

        tx_names = _names("tx")
        rx_names = _names("rx")
        if hasattr(viz, "set_state"):
            viz.set_state(tx_device_names=tx_names, rx_device_names=rx_names)
        else:
            state = getattr(viz, "app_state", None)
            if state is not None:
                setattr(state, "tx_device_names", tx_names)
                setattr(state, "rx_device_names", rx_names)
        invalidate_visualizer_cache(
            viz,
            CacheInvalidationScope.LABELS,
            reason="scenario_node_names",
        )

    def _apply_scenario_log_level(self, configured_level: str | int | None) -> str:
        """Apply scenario logging without overriding ``ORCHAV_LOG_LEVEL``."""
        set_log_level(resolve_log_level(configured_level))
        effective_level = get_current_log_level_name()

        combo = getattr(self.visualizer, "log_level_combo", None)
        if combo is None or not hasattr(combo, "setCurrentText"):
            return effective_level

        previous = combo.blockSignals(True) if hasattr(combo, "blockSignals") else None
        try:
            combo.setCurrentText(effective_level)
        finally:
            if previous is not None:
                combo.blockSignals(previous)
        return effective_level

    def _sync_node_marker_config_from_scenario(self, scenario: Any) -> None:
        """Copy optional YAML TX/RX marker settings into visualizer state."""
        viz = self.visualizer
        visualizer_cfg = getattr(scenario, "visualizer_cfg", None) or {}
        if not isinstance(visualizer_cfg, Mapping):
            visualizer_cfg = {}

        # Marker settings are scenario-owned, so every load starts from the
        # canonical shell defaults instead of inheriting the previous scene.
        viz.node_marker_config = {
            "default": {
                "shape": "sphere",
                "center": True,
            }
        }
        viz.tx_marker_size = DEFAULT_NODE_MARKER_SIZE_M
        viz.rx_marker_size = DEFAULT_NODE_MARKER_SIZE_M
        node_service = getattr(viz, "node_service", None)
        cache = getattr(node_service, "_node_marker_payload_cache", None)
        if hasattr(cache, "clear"):
            cache.clear()

        raw_markers = visualizer_cfg.get("node_markers")
        if raw_markers is None:
            raw_markers = visualizer_cfg.get("node_marker_config")
        if raw_markers is not None and not isinstance(raw_markers, Mapping):
            logger.warning("Ignoring visualizer node marker config; expected a mapping")
            raw_markers = None

        marker_config: dict[str, Any] = {}
        if isinstance(raw_markers, Mapping):
            flat_default = {
                key: value
                for key, value in raw_markers.items()
                if key
                not in {
                    "default",
                    "tx",
                    "rx",
                    "tx_size",
                    "rx_size",
                    "tx_marker_size",
                    "rx_marker_size",
                }
            }
            if flat_default:
                marker_config["default"] = dict(flat_default)

            for key in ("default", "tx", "rx"):
                section = raw_markers.get(key)
                if isinstance(section, Mapping):
                    marker_config.setdefault(key, {}).update(dict(section))

        def _bounded_marker_size(value: Any) -> Optional[float]:
            """Return a finite marker size inside the shared safety bounds."""
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                return None
            minimum, maximum = NODE_MARKER_SIZE_BOUNDS_M
            if math.isfinite(candidate) and minimum <= candidate <= maximum:
                return candidate
            return None

        def _configured_size(kind: str) -> Optional[float]:
            """Resolve TX/RX marker size from accepted nested and flat YAML keys."""
            section = marker_config.get(kind)
            if isinstance(section, Mapping):
                for key in ("size", "marker_size"):
                    size = _bounded_marker_size(section.get(key))
                    if size is not None:
                        return size
            if isinstance(raw_markers, Mapping):
                for key in (f"{kind}_marker_size", f"{kind}_size"):
                    size = _bounded_marker_size(raw_markers.get(key))
                    if size is not None:
                        return size
            for key in (f"{kind}_marker_size", f"{kind}_size"):
                size = _bounded_marker_size(visualizer_cfg.get(key))
                if size is not None:
                    return size
            return None

        scenario_root = Path(getattr(scenario, "root", Path(".")))
        for key in ("default", "tx", "rx"):
            section = marker_config.get(key)
            if not isinstance(section, dict):
                continue
            for path_key in ("mesh_path", "path"):
                raw_path = section.get(path_key)
                if not isinstance(raw_path, (str, Path)):
                    continue
                marker_path = Path(raw_path)
                if not marker_path.is_absolute():
                    marker_path = scenario_root / marker_path
                section[path_key] = str(marker_path.resolve(strict=False))

        if marker_config:
            merged_marker_config: dict[str, Any] = {
                "default": {
                    "shape": "sphere",
                    "center": True,
                }
            }
            for key, value in marker_config.items():
                if key == "default" and isinstance(value, Mapping):
                    merged_marker_config["default"].update(dict(value))
                else:
                    merged_marker_config[key] = value
            viz.node_marker_config = merged_marker_config

        for kind in ("tx", "rx"):
            size = _configured_size(kind)
            if size is None:
                size = DEFAULT_NODE_MARKER_SIZE_M
            setattr(viz, f"{kind}_marker_size", size)
            spin = getattr(viz, f"{kind}_marker_size_spin", None)
            if hasattr(spin, "setValue"):
                previous = spin.blockSignals(True) if hasattr(spin, "blockSignals") else None
                try:
                    spin.setValue(size)
                finally:
                    if previous is not None:
                        spin.blockSignals(previous)

    def _load_scene_xml(
        self,
        xml_path: Path,
        *,
        xml_root: Any = None,
        mesh_entries: list[dict[str, Any]] | None = None,
        cleanup_first: bool = True,
    ) -> None:
        """Load scenario XML through the composed scene controller."""
        load_prepared = getattr(self.visualizer.main_controller, "load_prepared_scene", None)
        if xml_root is not None and mesh_entries is not None and callable(load_prepared):
            load_prepared(
                xml_path,
                xml_root,
                mesh_entries,
                render_immediately=False,
                cleanup_first=cleanup_first,
            )
            return
        self.visualizer.main_controller.load_scene(
            xml_path,
            render_immediately=False,
            cleanup_first=cleanup_first,
        )

    def _load_scene_from_scenario(
        self,
        scenario: Any,
        app_config: Any,
        *,
        cleanup_first: bool = True,
    ) -> None:
        """Load scene XML from scenario specification."""
        scene_xml_path = self._resolve_scene_xml(scenario, app_config)
        self._load_scene_xml(scene_xml_path, cleanup_first=cleanup_first)
        logger.debug("Scene loaded from XML: %s", scene_xml_path)

    def _resolve_scene_xml(self, scenario: Any, app_config: Any) -> Path:
        """Resolve an existing scene XML path without loading or mutating it."""
        try:
            scene_source = scenario.scene_spec.get("source", "library")
            scene_id = scenario.scene_spec.get("id", "default")

            if scene_source == "library":
                if "/" in scene_id:
                    # Path like "etoile/etoile.xml"
                    scene_xml_path = app_config.scenes / scene_id
                    logger.debug("Looking for scene XML file: %s", scene_xml_path)

                    if scene_xml_path.exists() and scene_xml_path.is_file():
                        return scene_xml_path
                    else:
                        logger.warning("Scene XML file not found: %s", scene_xml_path)
                else:
                    # Scene name - look in scene directory
                    scene_dir = app_config.scenes / scene_id
                    logger.debug("Looking for scene in directory: %s", scene_dir)

                    xml_files = list(scene_dir.glob("*.xml"))
                    for xml_file in xml_files:
                        if xml_file.exists():
                            return xml_file

                    logger.warning("No XML scene files found in %s", scene_dir)
                    if scene_dir.exists():
                        logger.debug(
                            "Available files in scene directory: %s", list(scene_dir.iterdir())
                        )
            elif scene_source == "sionna":
                from shared.scenarios.parsers import resolve_sionna_scene_xml

                scene_xml_path = resolve_sionna_scene_xml(scene_id)
                if scene_xml_path:
                    return Path(scene_xml_path)
                logger.warning(
                    "Could not resolve Sionna scene '%s'; " "sionna package may not be installed",
                    scene_id,
                )
            elif scene_source == "osm":
                scene_xml_path = scenario.root / "scene.xml"
                if not scene_xml_path.exists():
                    try:
                        from shared.scenarios import load_scenario_configuration

                        resolved = load_scenario_configuration(
                            scenario.root,
                            project_root=self._resolve_project_root(),
                        )
                        scene_xml_path = resolved.scene_xml or scene_xml_path
                    except (ImportError, OSError, ValueError, KeyError) as exc:
                        logger.warning("Could not resolve generated scene: %s", exc)

                if scene_xml_path.exists() and scene_xml_path.is_file():
                    return scene_xml_path

                logger.warning("Generated scene XML not found: %s", scene_xml_path)
            else:
                # Local scenes - resolve relative to scenario root
                scene_path = scenario.root / scene_id if scene_id != "default" else scenario.root
                logger.debug("Looking for local scene at: %s", scene_path)

                if scene_path.is_file():
                    return scene_path

                if scene_path.is_dir():
                    xml_files = list(scene_path.glob("*.xml"))
                    for xml_file in xml_files:
                        return xml_file

                logger.warning("No scene found at %s", scene_path)

            raise FileNotFoundError(
                f"Could not resolve scene XML for source={scene_source!r}, id={scene_id!r}"
            )

        except (OSError, ValueError) as e:
            logger.warning("Could not resolve scene: %s", e)
            raise

    def _setup_frame_source(self, frame_source: Any) -> None:
        """Open the configured frame source."""
        if isinstance(frame_source, LiveGrpcSource):
            logger.info("Initializing live gRPC frame source (eager connect)")
            frame_source.open()
        else:
            # Lazily open other frame sources if they expose an open method
            if hasattr(frame_source, "open"):
                frame_source.open()

    def _update_animation_steps_from_scenario(self, scenario: Any, viz: Any) -> None:
        """Apply animation timing from the already parsed scenario model."""
        timeline = getattr(scenario, "timeline", None)
        if timeline is None:
            raise ValueError("Scenario timeline is required for visualizer playback")

        total_steps = int(timeline.steps)
        duration_s = float(timeline.duration_s)
        logger.debug("Timeline config: steps=%d, duration_s=%.2f", total_steps, duration_s)
        viz.total_animation_steps = total_steps
        viz._frame_duration = duration_s

        raytracing_spec = getattr(scenario, "raytracing", {}) or {}
        if not isinstance(raytracing_spec, Mapping):
            logger.warning("Ignoring non-mapping raytracing configuration")
            raytracing_spec = {}

        mesh_interval = raytracing_spec.get("mesh_update_interval_s")
        if mesh_interval is not None:
            viz._mesh_update_interval_s = mesh_interval

        if hasattr(viz, "ui_manager") and viz.ui_manager:
            viz.ui_manager.update_total_steps(total_steps)
            logger.debug("UI updated with total_steps: %d", total_steps)
