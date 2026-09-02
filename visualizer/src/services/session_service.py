"""Workspace snapshot persistence service.

This service saves renderer-neutral workspace intent such as camera position,
filters, color modes, animation state, and per-entry visibility/label choices.
Inspection-only object selection, highlight overlays, material display modes,
runtime PBR overrides, and native renderer resources are intentionally reset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PySide6.QtCore import QSignalBlocker

from shared.logging import get_logger

from ..io.frame_sources import make_frame_source
from ..scene.defaults import (
    DEFAULT_LABEL_FONT_SIZE,
    DEFAULT_LABEL_OFFSET_M,
    DEFAULT_NODE_MARKER_SIZE_M,
    DEFAULT_ORIENTATION_SCALE_M,
    DEFAULT_TRAJECTORY_LINE_WIDTH_PX,
    DEFAULT_TRAJECTORY_POINT_SIZE_PX,
    LABEL_FONT_SIZE_BOUNDS,
    LABEL_OFFSET_BOUNDS_M,
    MPC_LINE_WIDTH_BOUNDS_PX,
    MPC_POINT_SIZE_BOUNDS_PX,
    NODE_MARKER_SIZE_BOUNDS_M,
    ORIENTATION_SCALE_BOUNDS_M,
    TRAJECTORY_LINE_WIDTH_BOUNDS_PX,
    TRAJECTORY_POINT_SIZE_BOUNDS_PX,
)
from ..state import (
    BEAMFORMING_STATE_KEYS,
    MPC_ORDER_VALUES,
    MPC_TYPE_VALUES,
    AppState,
    MpcVisibility,
    get_beamforming_state_defaults,
    strip_beamforming_state,
)
from ..types.camera_state import CameraState
from ..utils.antenna_utils import spacing_m_to_wavelengths
from .coverage_service import (
    DEFAULT_COVERAGE_HEIGHT_ANIMATION_SPEED,
    DEFAULT_COVERAGE_INTERPOLATION,
    DEFAULT_COVERAGE_ISOLINE_COUNT,
    DEFAULT_COVERAGE_OPACITY,
)
from .object_identity import (
    ensure_node_entry_identity,
    ensure_scene_entry_identity,
    ensure_target_entry_identity,
)

if TYPE_CHECKING:
    from visualizer.visualizer import OrchavVisualizer

logger = get_logger("orchav.session")

# Session file format version
SESSION_VERSION = "6.0"
CAMERA_SESSION_FORMAT = "orbit-v1"
_LEGACY_AUTOSAVE_RE = re.compile(r".+_autosave_\d{8}_\d{6}$", re.IGNORECASE)

_NODE_APPEARANCE_ATTRIBUTE_SPECS = {
    "tx_marker_size_m": (
        "tx_marker_size",
        DEFAULT_NODE_MARKER_SIZE_M,
        NODE_MARKER_SIZE_BOUNDS_M,
    ),
    "rx_marker_size_m": (
        "rx_marker_size",
        DEFAULT_NODE_MARKER_SIZE_M,
        NODE_MARKER_SIZE_BOUNDS_M,
    ),
    "label_font_size": (
        "label_font_size",
        DEFAULT_LABEL_FONT_SIZE,
        LABEL_FONT_SIZE_BOUNDS,
    ),
    "label_offset_x_m": (
        "label_offset_x",
        DEFAULT_LABEL_OFFSET_M[0],
        LABEL_OFFSET_BOUNDS_M,
    ),
    "label_offset_y_m": (
        "label_offset_y",
        DEFAULT_LABEL_OFFSET_M[1],
        LABEL_OFFSET_BOUNDS_M,
    ),
    "label_offset_z_m": (
        "label_offset_z",
        DEFAULT_LABEL_OFFSET_M[2],
        LABEL_OFFSET_BOUNDS_M,
    ),
    "orientation_scale_m": (
        "orientation_scale",
        DEFAULT_ORIENTATION_SCALE_M,
        ORIENTATION_SCALE_BOUNDS_M,
    ),
}
_TRAJECTORY_APPEARANCE_SPECS = {
    "trajectory_line_width_px": (
        "trajectory_line_width",
        DEFAULT_TRAJECTORY_LINE_WIDTH_PX,
        TRAJECTORY_LINE_WIDTH_BOUNDS_PX,
    ),
    "trajectory_point_size_px": (
        "trajectory_point_size",
        DEFAULT_TRAJECTORY_POINT_SIZE_PX,
        TRAJECTORY_POINT_SIZE_BOUNDS_PX,
    ),
}
_NODE_APPEARANCE_WIDGET_KEYS = {
    "tx_marker_size_m": "tx_marker_size_spin",
    "rx_marker_size_m": "rx_marker_size_spin",
    "label_font_size": "label_font_size_spin",
    "label_offset_x_m": "x_offset_spinbox",
    "label_offset_y_m": "y_offset_spinbox",
    "label_offset_z_m": "z_offset_spinbox",
    "orientation_scale_m": "orientation_scale_spin",
    "trajectory_line_width_px": "trajectory_line_width_spin",
    "trajectory_point_size_px": "trajectory_point_size_spin",
}


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshotSummary:
    """Validated metadata used to identify a workspace snapshot in the UI."""

    path: Path
    scenario_root: Path
    scenario_name: str
    created_at: datetime
    frame: int
    is_autosave: bool


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """One validated snapshot document decoded from disk exactly once."""

    summary: WorkspaceSnapshotSummary
    payload: dict[str, Any]

    @property
    def path(self) -> Path:
        """Return the snapshot file used to create this document."""
        return self.summary.path

    @property
    def scenario_root(self) -> Path:
        """Return the canonical scenario root referenced by the snapshot."""
        return self.summary.scenario_root

    @property
    def scenario_name(self) -> str:
        """Return the scenario name used in startup progress messages."""
        return self.summary.scenario_name

    @property
    def frame(self) -> int:
        """Return the validated saved frame."""
        return self.summary.frame

    @property
    def camera(self) -> Optional[dict[str, Any]]:
        """Return a supported startup camera payload, when present."""
        camera = self.payload.get("camera")
        if isinstance(camera, dict) and camera.get("format") == CAMERA_SESSION_FORMAT:
            return copy.deepcopy(camera)
        return None


@dataclass(frozen=True, slots=True)
class _WorkspaceRestoreCheckpoint:
    """Renderer-neutral state used to best-effort undo a failed restore."""

    scenario_root: Path
    camera: dict[str, Any]
    app_state: dict[str, Any]
    rendering: dict[str, Any]
    entry_state: dict[str, dict[str, bool]]
    frame: int
    scene_only: bool


@dataclass(frozen=True, slots=True)
class _PreparedAppStateRestore:
    """Validated application state ready for one transactional apply."""

    state: AppState
    node_coloring_mode: str | None


def _normalize_path(value: object) -> Optional[Path]:
    """Return an absolute path for a stored or command-line path value."""
    if value is None:
        return None
    try:
        raw_path = os.fspath(value)
    except TypeError:
        return None
    if isinstance(raw_path, str) and not raw_path.strip():
        return None
    try:
        return Path(raw_path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        try:
            return Path(raw_path).expanduser()
        except (OSError, RuntimeError, TypeError, ValueError):
            return None


def normalize_scenario_root(value: object) -> Optional[Path]:
    """Canonicalize a scenario directory or YAML path to its absolute root."""
    path = _normalize_path(value)
    if path is None:
        return None
    if path.suffix.lower() in {".yaml", ".yml"}:
        return path.parent
    return path


def _snapshot_autosave_kind(path: Path, payload: dict[str, Any]) -> Optional[bool]:
    """Classify known snapshot kinds without guessing about unknown metadata."""
    kind = payload.get("snapshot_kind")
    if kind == "autosave":
        return True
    if kind == "manual":
        return False
    if kind is not None:
        return None
    # Version-6 snapshots created before ``snapshot_kind`` used this strict
    # timestamped filename. Keep recognizing them so the rolling retention
    # policy can retire old application-created autosaves safely.
    return bool(_LEGACY_AUTOSAVE_RE.fullmatch(path.stem))


def _snapshot_frame(payload: dict[str, Any]) -> Optional[int]:
    """Return a valid saved frame, or ``None`` for incomplete metadata."""
    animation = payload.get("animation")
    if not isinstance(animation, dict) or "current_frame" not in animation:
        return None
    try:
        frame = int(animation["current_frame"])
    except (TypeError, ValueError):
        return None
    return frame if frame >= 0 else None


def _read_workspace_payload(path: Path) -> Optional[tuple[Path, dict[str, Any]]]:
    """Decode one snapshot JSON object and return its normalized path."""
    normalized_path = _normalize_path(path)
    if normalized_path is None:
        return None
    try:
        with normalized_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return normalized_path, payload


def read_workspace_snapshot(path: Path) -> Optional[WorkspaceSnapshot]:
    """Decode and validate one snapshot for listing or automatic resume."""
    loaded = _read_workspace_payload(path)
    if loaded is None:
        return None
    normalized_path, payload = loaded
    if str(payload.get("version")) != SESSION_VERSION:
        return None

    scenario_root = normalize_scenario_root(payload.get("scenario_path"))
    if scenario_root is None or not scenario_root.name:
        return None
    raw_created_at = payload.get("created_at")
    try:
        created_at = datetime.fromisoformat(str(raw_created_at))
    except (TypeError, ValueError):
        return None

    frame = _snapshot_frame(payload)
    if frame is None:
        return None

    is_autosave = _snapshot_autosave_kind(normalized_path, payload)
    if is_autosave is None:
        return None
    return WorkspaceSnapshot(
        summary=WorkspaceSnapshotSummary(
            path=normalized_path,
            scenario_root=scenario_root,
            scenario_name=scenario_root.name,
            created_at=created_at,
            frame=frame,
            is_autosave=is_autosave,
        ),
        payload=payload,
    )


def read_workspace_summary(path: Path) -> Optional[WorkspaceSnapshotSummary]:
    """Read validated workspace metadata, returning ``None`` for invalid files."""
    snapshot = read_workspace_snapshot(path)
    return snapshot.summary if snapshot is not None else None


class SessionService:
    """Manage workspace snapshot save/load operations.

    Snapshots are stored in ``~/.orchav/sessions/`` as JSON files. Each one
    captures supported renderer-neutral workspace intent for
    restoration through the normal application services.
    """

    def __init__(self, visualizer: OrchavVisualizer):
        """Initialize session service.

        Args:
            visualizer: Parent visualizer instance
        """
        self.viz = visualizer
        self.session_dir = Path.home() / ".orchav" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_enabled = True
        logger.info(f"SessionService initialized. Session directory: {self.session_dir}")

    @staticmethod
    def _validated_alpha(alpha: Any) -> float | None:
        """Return a finite normalized transparency value, or ``None``."""
        if isinstance(alpha, bool):
            return None
        try:
            value = float(alpha)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value

    @staticmethod
    def _validated_bounded_float(
        value: Any,
        bounds: tuple[float, float],
    ) -> float | None:
        """Return one finite numeric value inside explicit safety bounds."""
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        minimum, maximum = bounds
        if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
            return None
        return numeric

    def save_session(self, path: Optional[Path] = None, name: Optional[str] = None) -> Path:
        """Save the current workspace state to a JSON snapshot.

        Args:
            path: Full destination path (overrides ``name``).
            name: Snapshot filename without the ``.json`` extension. When
                omitted with ``path``, updates the scenario autosave.

        Returns:
            Path to the saved workspace snapshot.
        """
        scenario_root = normalize_scenario_root(getattr(self.viz, "current_scenario_path", None))
        if scenario_root is None:
            raise ValueError("Load a scenario before saving a workspace snapshot")

        is_autosave = path is None and name is None
        if path is None:
            if is_autosave:
                path = self._autosave_path(scenario_root)
            else:
                assert name is not None
                if not name.endswith(".json"):
                    name = f"{name}.json"
                path = self.session_dir / name
        path = Path(path).expanduser()

        # Collect session data
        session_data = {
            "version": SESSION_VERSION,
            "created_at": datetime.now().isoformat(),
            "snapshot_kind": "autosave" if is_autosave else "manual",
            "scenario_path": str(scenario_root),
            "camera": self._get_camera_state(),
            "app_state": self._get_app_state(),
            "animation": self._get_animation_state(),
            "rendering": self._get_rendering_state(),
            "entry_state": self._get_entry_state_snapshot(),
        }

        # Write atomically so an interrupted application exit cannot leave a
        # truncated snapshot that looks loadable on the next launch.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(session_data, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove temporary snapshot %s", temporary_path)

        if is_autosave:
            self._prune_older_autosaves(scenario_root, keep_path=path)

        logger.info(f"Session saved to {path}")
        return path

    def load_session(
        self,
        source: Path | WorkspaceSnapshot,
        skip_camera: bool = False,
    ) -> bool:
        """Restore workspace state from a JSON snapshot.

        Args:
            source: Snapshot path, or a snapshot already decoded during startup.
            skip_camera: When True, skip restoring camera state (e.g. when
                the camera was already applied during boot via pre-read).

        Returns:
            True if loaded successfully, False otherwise
        """
        if bool(getattr(self.viz, "_shutdown_started", False)):
            logger.warning("Workspace restore rejected because application shutdown has begun")
            return False
        if bool(getattr(self.viz, "_session_restore_in_progress", False)):
            logger.warning("Workspace restore rejected because another restore is active")
            return False

        self.viz._session_restore_in_progress = True
        try:
            return self._load_session_transaction(source, skip_camera=skip_camera)
        finally:
            self.viz._session_restore_in_progress = False

    def _load_session_transaction(
        self,
        source: Path | WorkspaceSnapshot,
        *,
        skip_camera: bool,
    ) -> bool:
        """Execute one restore while the entry-level exclusion guard is held."""
        if isinstance(source, WorkspaceSnapshot):
            path = source.path
            session_data = source.payload
        else:
            loaded = _read_workspace_payload(source)
            if loaded is None:
                logger.error("Failed to decode workspace snapshot: %s", source)
                return False
            path, session_data = loaded

        version = str(session_data.get("version", "1.0"))
        if version != SESSION_VERSION:
            logger.warning(
                "Unsupported session version: file=%s expected=%s",
                version,
                SESSION_VERSION,
            )
            return False

        saved_frame = _snapshot_frame(session_data)
        if saved_frame is None:
            logger.error("Workspace snapshot has no valid saved frame: %s", path)
            return False

        validation_error = self._restore_payload_validation_error(session_data)
        if validation_error is not None:
            logger.error("Invalid workspace snapshot %s: %s", path, validation_error)
            return False
        try:
            prepared_app_state = self._prepare_app_state_restore(
                session_data["app_state"],
                frame=saved_frame,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.error("Invalid workspace app state in %s: %s", path, exc)
            return False

        # Validate the referenced domain before changing live-preview, scene,
        # application, or renderer state. A missing scenario is actionable to
        # callers and must not silently apply a snapshot to the wrong scene.
        saved_scenario = normalize_scenario_root(session_data.get("scenario_path"))
        if saved_scenario is None:
            logger.error("Workspace snapshot has no valid scenario path: %s", path)
            return False
        scenario_yaml = saved_scenario / "scenario.yaml"
        if not scenario_yaml.is_file():
            raise FileNotFoundError(
                f"Workspace snapshot references missing scenario: {scenario_yaml}"
            )

        current_scenario = normalize_scenario_root(getattr(self.viz, "current_scenario_path", None))
        scenario_differs = current_scenario is None or not self._paths_match(
            saved_scenario,
            current_scenario,
        )
        checkpoint = (
            self._capture_restore_checkpoint(current_scenario)
            if current_scenario is not None
            else None
        )
        camera_applied_during_open = False
        if scenario_differs:
            frame_preflight = self._prevalidate_local_saved_frame(saved_scenario, saved_frame)
            if frame_preflight is False:
                logger.error(
                    "Workspace saved frame %d is unavailable in scenario %s; "
                    "the active scenario was preserved",
                    saved_frame,
                    saved_scenario,
                )
                return False
            opened, camera_applied_during_open = self._open_scenario_for_restore(
                saved_scenario,
                None if skip_camera else session_data.get("camera"),
            )
            if not opened:
                self._rollback_failed_restore(checkpoint, reapply_if_active=False)
                return False
            if bool(getattr(self.viz, "_scene_only_mode", False)):
                logger.error(
                    "Workspace scenario reopened without frame data; frame %d was not restored",
                    saved_frame,
                )
                self._rollback_failed_restore(checkpoint, reapply_if_active=False)
                return False

        if not bool(getattr(self.viz, "_scene_only_mode", False)):
            frame_source = getattr(self.viz, "frame_source", None)
            has_frame = getattr(frame_source, "has_frame", None)
            if callable(has_frame):
                try:
                    frame_available = bool(has_frame(saved_frame))
                except Exception:  # source boundary: availability failure must stay transactional
                    frame_available = False
                if not frame_available:
                    logger.error(
                        "Workspace saved frame %d is unavailable; snapshot state was not applied",
                        saved_frame,
                    )
                    self._rollback_failed_restore(checkpoint, reapply_if_active=False)
                    return False

        if bool(getattr(self.viz, "_shutdown_started", False)):
            logger.warning("Workspace restore cancelled during application shutdown")
            return False

        restore_succeeded = False
        try:
            # The scenario open path can schedule its initial frame, and UI
            # restoration can also request a coalesced update. The saved frame
            # is the sole frame transaction for a frame-backed restore.
            self._cancel_pending_frame_work()
            self._reset_live_preview_state()
            logger.info(
                "Loading workspace snapshot from %s (created: %s)",
                path,
                session_data.get("created_at", "unknown"),
            )
            self._reset_transient_appearance_state()

            # Restore app state first.
            self._restore_app_state(
                session_data["app_state"],
                frame=saved_frame,
                prepared=prepared_app_state,
            )

            if "rendering" in session_data:
                self._restore_rendering_state(session_data["rendering"])

            # Entry flags are semantic application intent. Restore them before
            # frame hydration so normal node/target synchronization publishes
            # the correct renderer snapshots on its first pass.
            entry_state_deltas = self._restore_entry_state(session_data["entry_state"])

            # Static scene objects do not participate in animation hydration.
            # Refresh all entry visibility through semantic services so this
            # layer never needs renderer presence or visibility APIs.
            self._refresh_entry_state(entry_state_deltas)

            # Restore camera after semantic objects and entry visibility.
            if "camera" in session_data and not skip_camera and not camera_applied_during_open:
                self._restore_camera_state(session_data["camera"])

            self._sync_ui_with_state()

            if bool(getattr(self.viz, "_shutdown_started", False)):
                logger.warning("Workspace restore cancelled during application shutdown")
                return False

            # Signals emitted by a few semantic UI controls may have queued an
            # update while state was restored. Cancel it before issuing the one
            # intentional saved-frame transaction.
            self._cancel_pending_frame_work()
            if bool(getattr(self.viz, "_scene_only_mode", False)):
                restore_succeeded = self._schedule_scene_only_refresh()
            else:
                # Frame data, not transient renderer edits, owns object poses.
                if not self._restore_animation_frame(saved_frame):
                    logger.error("Workspace frame %d could not be restored", saved_frame)
                else:
                    restore_succeeded = True

        except Exception as e:  # restore boundary: every failure must trigger checkpoint rollback
            logger.error("Error restoring workspace state: %s", e, exc_info=True)

        if not restore_succeeded:
            self._rollback_failed_restore(checkpoint, reapply_if_active=True)
            return False

        logger.info("Workspace snapshot loaded successfully from %s", path)
        return True

    @staticmethod
    def _restore_payload_validation_error(session_data: dict[str, Any]) -> str | None:
        """Return the first malformed required v6 restore section, if any."""
        for section in ("app_state", "animation", "rendering", "entry_state"):
            if section not in session_data:
                return f"missing required {section!r} section"
            if not isinstance(session_data[section], dict):
                return f"{section!r} section must be a mapping"

        coverage = session_data["rendering"].get("coverage")
        if coverage is not None and not isinstance(coverage, dict):
            return "'rendering.coverage' must be a mapping"

        node_appearance = session_data["rendering"].get("node_appearance")
        if node_appearance is not None:
            if not isinstance(node_appearance, dict):
                return "'rendering.node_appearance' must be a mapping"
            specs = {
                **_NODE_APPEARANCE_ATTRIBUTE_SPECS,
                **_TRAJECTORY_APPEARANCE_SPECS,
            }
            for key, value in node_appearance.items():
                spec = specs.get(key)
                if spec is None:
                    continue
                bounds = spec[2]
                if SessionService._validated_bounded_float(value, bounds) is None:
                    return (
                        f"'rendering.node_appearance.{key}' must be a finite number "
                        f"between {bounds[0]} and {bounds[1]}"
                    )

        for entry_id, intent in session_data["entry_state"].items():
            if not isinstance(intent, dict):
                return f"entry_state[{entry_id!r}] must be a mapping"
            for field in ("visible", "show_label"):
                if field in intent and not isinstance(intent[field], bool):
                    return f"entry_state[{entry_id!r}].{field} must be boolean"

        camera = session_data.get("camera")
        if camera is not None and not isinstance(camera, dict):
            return "'camera' section must be a mapping when present"
        if isinstance(camera, dict) and camera.get("format") == CAMERA_SESSION_FORMAT:
            try:
                CameraState.from_dict(
                    {
                        "eye": camera["eye"],
                        "lookat": camera["lookat"],
                        "up": camera["up"],
                        "fov": camera["fov"],
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                return f"invalid orbit camera payload: {exc}"
        return None

    def _open_scenario_for_restore(
        self,
        scenario_root: Path,
        camera_payload: object,
    ) -> tuple[bool, bool]:
        """Open one snapshot scenario without scheduling an initial frame."""
        if bool(getattr(self.viz, "_shutdown_started", False)):
            logger.warning("Workspace scenario open skipped during application shutdown")
            return False, False
        open_scenario = getattr(self.viz, "open_scenario", None)
        if not callable(open_scenario):
            logger.error(
                "Workspace scenario differs from the current scenario and open_scenario() "
                "is unavailable: %s",
                scenario_root,
            )
            return False, False

        valid_camera = (
            isinstance(camera_payload, dict)
            and camera_payload.get("format") == CAMERA_SESSION_FORMAT
        )

        try:
            outcome = open_scenario(
                str(scenario_root),
                pending_camera=camera_payload if valid_camera else None,
                autorun_initial_frame=False,
            )
        except Exception as exc:  # application boundary: preserve the prior workspace on failure
            logger.error(
                "Failed to open scenario for workspace restore (%s): %s",
                scenario_root,
                exc,
            )
            return False, False
        if not bool(getattr(outcome, "succeeded", False)):
            logger.error(
                "Scenario open failed during workspace restore: %s",
                getattr(outcome, "message", None) or scenario_root,
            )
            return False, False
        current_scenario = normalize_scenario_root(getattr(self.viz, "current_scenario_path", None))
        if current_scenario is None or not self._paths_match(current_scenario, scenario_root):
            logger.error(
                "Scenario open did not activate the workspace scenario: expected=%s actual=%s",
                scenario_root,
                current_scenario,
            )
            return False, False
        logger.info("Loaded workspace scenario context: %s", scenario_root)
        return True, valid_camera

    def _prevalidate_local_saved_frame(
        self,
        scenario_root: Path,
        saved_frame: int,
    ) -> bool | None:
        """Check a file-backed saved frame without replacing the active scenario.

        ``None`` means the composed visualizer does not expose enough loader
        infrastructure for an isolated check. This preserves compatibility with
        small visualizer test doubles and non-local source modes; those paths retain
        the post-open availability check.
        """
        loader = getattr(self.viz, "scenario_loader_service", None)
        preflight_scenario = getattr(loader, "preflight_scenario", None)
        if not callable(preflight_scenario):
            return None

        try:
            preflight = preflight_scenario(str(scenario_root))
        except Exception as exc:  # preflight boundary: never replace state after a failed check
            logger.error("Workspace scenario preflight failed for %s: %s", scenario_root, exc)
            return False
        if preflight is None:
            return False

        scenario = getattr(preflight, "scenario", None)
        if scenario is None or str(getattr(scenario, "data_mode", "")) != "files":
            return None

        frame_source = None
        try:
            frame_source = make_frame_source(scenario)
            has_frame = getattr(frame_source, "has_frame", None)
            if not callable(has_frame):
                return None
            return bool(has_frame(saved_frame))
        except Exception as exc:  # isolated source boundary: failed checks are unavailable frames
            logger.error(
                "Could not prevalidate workspace frame %d for %s: %s",
                saved_frame,
                scenario_root,
                exc,
            )
            return False
        finally:
            close_source = getattr(frame_source, "close", None)
            if callable(close_source):
                try:
                    close_source()
                except Exception as exc:  # cleanup must not mask the availability result
                    logger.warning(
                        "Could not close workspace preflight source for %s: %s",
                        scenario_root,
                        exc,
                    )

    def _capture_restore_checkpoint(
        self,
        scenario_root: Path,
    ) -> _WorkspaceRestoreCheckpoint:
        """Capture the durable renderer-neutral state needed for rollback."""

        def _capture(name: str, operation: Any, default: Any) -> Any:
            try:
                return copy.deepcopy(operation())
            except Exception as exc:  # checkpoint capture must not block the requested restore
                logger.warning("Could not capture prior workspace %s: %s", name, exc)
                return copy.deepcopy(default)

        try:
            frame = max(0, int(getattr(getattr(self.viz, "app_state", None), "step", 0)))
        except (TypeError, ValueError):
            frame = 0

        return _WorkspaceRestoreCheckpoint(
            scenario_root=scenario_root,
            camera=_capture("camera", self._get_camera_state, {}),
            app_state=_capture("app state", self._get_app_state, {}),
            rendering=_capture("rendering state", self._get_rendering_state, {}),
            entry_state=_capture("entry state", self._get_entry_state_snapshot, {}),
            frame=frame,
            scene_only=bool(getattr(self.viz, "_scene_only_mode", False)),
        )

    def _rollback_failed_restore(
        self,
        checkpoint: _WorkspaceRestoreCheckpoint | None,
        *,
        reapply_if_active: bool,
    ) -> bool:
        """Best-effort reopen and reapply the workspace active before restore."""
        if checkpoint is None:
            return False
        if bool(getattr(self.viz, "_shutdown_started", False)):
            logger.warning("Workspace rollback skipped during application shutdown")
            return False

        current_scenario = normalize_scenario_root(getattr(self.viz, "current_scenario_path", None))
        prior_is_active = current_scenario is not None and self._paths_match(
            current_scenario,
            checkpoint.scenario_root,
        )
        if prior_is_active and not reapply_if_active:
            return True

        camera_applied_during_open = False
        try:
            if not prior_is_active:
                opened, camera_applied_during_open = self._open_scenario_for_restore(
                    checkpoint.scenario_root,
                    checkpoint.camera,
                )
                if not opened:
                    logger.error(
                        "Failed to reopen prior scenario after workspace restore failure: %s",
                        checkpoint.scenario_root,
                    )
                    return False

            # Small test doubles may not model the scenario workflow's mode reset.
            # The checkpoint remains the authority for the prior workspace mode.
            self.viz._scene_only_mode = checkpoint.scene_only
            self._cancel_pending_frame_work()
            self._reset_live_preview_state()
            self._reset_transient_appearance_state()

            previous_restore_flag = bool(getattr(self.viz, "_session_restore_in_progress", False))
            self.viz._session_restore_in_progress = True
            try:
                self._restore_app_state(checkpoint.app_state, frame=checkpoint.frame)
                self._restore_rendering_state(checkpoint.rendering)
                entry_state_deltas = self._restore_entry_state(checkpoint.entry_state)
                self._refresh_entry_state(entry_state_deltas)
                if checkpoint.camera and not camera_applied_during_open:
                    self._restore_camera_state(checkpoint.camera)
                self._sync_ui_with_state()
                self._cancel_pending_frame_work()
                if checkpoint.scene_only:
                    if not self._schedule_scene_only_refresh():
                        return False
                elif not self._restore_animation_frame(checkpoint.frame):
                    logger.error(
                        "Prior workspace scenario reopened, but frame %d could not be reapplied",
                        checkpoint.frame,
                    )
                    return False
            finally:
                self.viz._session_restore_in_progress = previous_restore_flag
        except Exception:  # rollback is a last-resort boundary and must not mask restore failure
            logger.exception(
                "Failed to roll back workspace restore to %s",
                checkpoint.scenario_root,
            )
            return False

        logger.info(
            "Rolled back failed workspace restore to %s at frame %d",
            checkpoint.scenario_root,
            checkpoint.frame,
        )
        return True

    def _cancel_pending_frame_work(self) -> None:
        """Cancel one coalesced frame update before deterministic restoration."""
        self.viz.cancel_scheduled_update()

    def _schedule_scene_only_refresh(self) -> bool:
        """Request one presentation unless application shutdown has begun."""
        if bool(getattr(self.viz, "_shutdown_started", False)):
            logger.warning("Scene-only workspace refresh skipped during application shutdown")
            return False
        self.viz.force_update_next_frame = True
        self.viz.schedule_update()
        return True

    def _reset_live_preview_state(self) -> None:
        """Drop transient interactive edits before restoring snapshot state."""
        live_preview = getattr(self.viz, "live_preview_service", None)
        reset = getattr(live_preview, "reset", None)
        if not callable(reset):
            return
        try:
            reset()
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("Failed to reset live preview before workspace restore: %s", exc)

    def _reset_transient_appearance_state(self) -> None:
        """Drop inspection-only selection, overlays, highlight, and PBR overrides."""
        selected = getattr(self.viz, "selected_objects", None)
        if hasattr(selected, "clear"):
            selected.clear()
        mode_service = getattr(self.viz, "material_mode_service", None)
        clear_modes = getattr(mode_service, "clear", None)
        if callable(clear_modes):
            clear_modes()
        pbr_service = getattr(self.viz, "material_pbr_service", None)
        overrides = getattr(pbr_service, "overrides", None)
        if hasattr(overrides, "clear"):
            overrides.clear()
        for entry in list(getattr(self.viz, "mesh_entries", []) or []) + list(
            getattr(self.viz, "target_entries", []) or []
        ):
            entry["highlighted"] = False
            binding = entry.get("_visual_material_binding")
            if getattr(getattr(binding, "source", None), "value", None) != "profile":
                entry.pop("_visual_material_binding", None)

    def _sync_ui_with_state(self) -> None:
        """Sync UI panel controls with the current app state.

        Restores widget values for all panels without triggering signals,
        so the visual state matches the restored AppState after a workspace load.
        """
        if not hasattr(self.viz, "app_state"):
            return

        app_state = self.viz.app_state

        if not (hasattr(self.viz, "ui_manager") and self.viz.ui_manager):
            return

        panels = self.viz.ui_manager.panels

        self._sync_context_panel(panels, app_state)
        self._sync_mpc_panel(panels, app_state)
        self._sync_nodes_panel(panels, app_state)
        self._sync_coverage_panel(panels, app_state)
        self._sync_materials_panel(panels, app_state)
        self._sync_beam_panel(panels, app_state)
        self._sync_performance_panel(panels, app_state)
        self._sync_render_panel(panels)
        update_paths_badge = getattr(self.viz.ui_manager, "update_paths_tab_badge", None)
        if callable(update_paths_badge):
            update_paths_badge()

        logger.debug("UI panels synced with restored state")

    # -- Per-panel sync helpers ------------------------------------------------

    @staticmethod
    def _set_widget(widget: object, value: object) -> None:
        """Set widget value with signals blocked."""
        from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QRadioButton, QSpinBox

        with QSignalBlocker(widget):
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value if value is not None else widget.minimum())
            elif isinstance(widget, QComboBox):
                if isinstance(value, int):
                    widget.setCurrentIndex(value)
                else:
                    widget.setCurrentText(str(value))
            elif isinstance(widget, QRadioButton):
                widget.setChecked(bool(value))

    @staticmethod
    def _sync_context_panel(panels: dict, state: object) -> None:
        """Sync persistent TX/RX scope and MPC master controls."""
        context = panels.get("context")
        sync = getattr(context, "sync_from_state", None)
        if callable(sync):
            sync(state)

    def _sync_mpc_panel(self, panels: dict, state: object) -> None:
        """Sync MPC panel widgets."""
        if "mpc" not in panels:
            return
        mpc = panels["mpc"]
        if not hasattr(mpc, "widgets"):
            return
        w = mpc.widgets

        # Paths-local visibility details. The MPC master lives in Context.
        visibility = state.mpc_visibility
        if "mpc_paths_cb" in w:
            self._set_widget(w["mpc_paths_cb"], visibility.paths)
        if "mpc_bounce_points_cb" in w:
            self._set_widget(w["mpc_bounce_points_cb"], visibility.bounce_points)

        # Top-K
        if "topk_render_cb" in w:
            self._set_widget(w["topk_render_cb"], state.topk_render_enabled)
        if "topk_render_max_spin" in w:
            self._set_widget(w["topk_render_max_spin"], int(state.topk_render_max_paths))
            w["topk_render_max_spin"].setEnabled(bool(state.topk_render_enabled))

        # Color mode radio buttons
        color_map = {
            "reflection_order": "reflection_order_rb",
            "delay": "delay_rb",
            "path_loss": "path_loss_rb",
            "material": "material_rb",
            "mpc_type": "mpc_type_rb",
            "reconstruction_type": "reconstruction_type_rb",
        }
        rb_name = color_map.get(state.color_mode)
        if rb_name and rb_name in w:
            self._set_widget(w[rb_name], True)

        # Distinct material colors
        if "distinct_material_colors_cb" in w:
            self._set_widget(w["distinct_material_colors_cb"], state.use_distinct_material_colors)

        # Empty allow-lists intentionally mean that no MPCs are visible.
        for order in MPC_ORDER_VALUES:
            key = f"order_{order}_cb"
            if key in w:
                self._set_widget(w[key], order in state.mpc_allowed_orders)

        # Interaction types include reconstructed/virtual paths (99).
        for type_val in MPC_TYPE_VALUES:
            key = f"type_{type_val}_cb"
            if key in w:
                self._set_widget(w[key], type_val in state.mpc_allowed_types)

        # Range filters (None -> widget min/max to effectively disable)
        range_filters = [
            ("delay_filter_min", state.delay_filter_min_ns),
            ("delay_filter_max", state.delay_filter_max_ns),
            ("power_filter_min", state.power_filter_min_db),
            ("power_filter_max", state.power_filter_max_db),
            ("aoa_az_filter_min", state.aoa_az_filter_min_deg),
            ("aoa_az_filter_max", state.aoa_az_filter_max_deg),
            ("aoa_el_filter_min", state.aoa_el_filter_min_deg),
            ("aoa_el_filter_max", state.aoa_el_filter_max_deg),
            ("aod_az_filter_min", state.aod_az_filter_min_deg),
            ("aod_az_filter_max", state.aod_az_filter_max_deg),
            ("aod_el_filter_min", state.aod_el_filter_min_deg),
            ("aod_el_filter_max", state.aod_el_filter_max_deg),
        ]
        for key, val in range_filters:
            if key in w:
                if val is None:
                    # None means "no bound" — set to the widget's natural extreme
                    is_max = key.endswith("_max")
                    val = w[key].maximum() if is_max else w[key].minimum()
                self._set_widget(w[key], val)

        # Aperture controls
        if "show_aoa_aperture_cb" in w:
            self._set_widget(w["show_aoa_aperture_cb"], state.show_aoa_aperture)
        if "show_aod_aperture_cb" in w:
            self._set_widget(w["show_aod_aperture_cb"], state.show_aod_aperture)
        if "aperture_radius_spin" in w:
            self._set_widget(w["aperture_radius_spin"], state.aperture_radius_m)
        if "show_global_angular_reference_cb" in w:
            self._set_widget(
                w["show_global_angular_reference_cb"],
                getattr(state, "show_global_angular_reference", False),
            )
        if "show_local_angular_reference_cb" in w:
            self._set_widget(
                w["show_local_angular_reference_cb"],
                getattr(state, "show_local_angular_reference", False),
            )
        if "mpc_interaction_markers_cb" in w:
            self._set_widget(
                w["mpc_interaction_markers_cb"],
                getattr(state, "show_mpc_type_markers", False),
            )
        if hasattr(mpc, "refresh_aperture_preview_state"):
            mpc.refresh_aperture_preview_state()
        if hasattr(mpc, "refresh_renderer_controls_state"):
            mpc.refresh_renderer_controls_state()

    def _sync_nodes_panel(self, panels: dict, state: object) -> None:
        """Sync Nodes panel widgets."""
        if "nodes" not in panels:
            return
        nodes = panels["nodes"]
        if not hasattr(nodes, "widgets"):
            return
        w = nodes.widgets

        # Node coloring mode radio buttons
        node_coloring_mode = getattr(self.viz, "node_coloring_mode", "per_type")
        if "per_node_type_rb" in w and "individual_nodes_rb" in w:
            if node_coloring_mode == "individual":
                self._set_widget(w["individual_nodes_rb"], True)
            else:
                self._set_widget(w["per_node_type_rb"], True)

        # Labels
        if "labels_cb" in w:
            self._set_widget(w["labels_cb"], state.show_labels)
        if "node_label_mode_combo" in w:
            combo = w["node_label_mode_combo"]
            mode = getattr(state, "node_label_mode", "role")
            idx = combo.findData(mode) if hasattr(combo, "findData") else -1
            with QSignalBlocker(combo):
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.setCurrentText("Device Name" if mode == "name" else "Role")
        if "target_labels_cb" in w:
            self._set_widget(w["target_labels_cb"], state.show_target_labels)

        appearance_values = {
            "tx_marker_size_spin": getattr(
                self.viz,
                "tx_marker_size",
                DEFAULT_NODE_MARKER_SIZE_M,
            ),
            "rx_marker_size_spin": getattr(
                self.viz,
                "rx_marker_size",
                DEFAULT_NODE_MARKER_SIZE_M,
            ),
            "label_font_size_spin": getattr(
                self.viz,
                "label_font_size",
                DEFAULT_LABEL_FONT_SIZE,
            ),
            "x_offset_spinbox": getattr(
                self.viz,
                "label_offset_x",
                DEFAULT_LABEL_OFFSET_M[0],
            ),
            "y_offset_spinbox": getattr(
                self.viz,
                "label_offset_y",
                DEFAULT_LABEL_OFFSET_M[1],
            ),
            "z_offset_spinbox": getattr(
                self.viz,
                "label_offset_z",
                DEFAULT_LABEL_OFFSET_M[2],
            ),
            "orientation_scale_spin": getattr(
                self.viz,
                "orientation_scale",
                DEFAULT_ORIENTATION_SCALE_M,
            ),
        }
        renderer = getattr(self.viz, "renderer", None)
        appearance_values.update(
            {
                "trajectory_line_width_spin": getattr(
                    renderer,
                    "trajectory_line_width",
                    DEFAULT_TRAJECTORY_LINE_WIDTH_PX,
                ),
                "trajectory_point_size_spin": getattr(
                    renderer,
                    "trajectory_point_size",
                    DEFAULT_TRAJECTORY_POINT_SIZE_PX,
                ),
            }
        )
        for widget_key, value in appearance_values.items():
            if widget_key in w:
                self._set_widget(w[widget_key], value)

        # Do not block 3D trajectory toggle signals: their handlers request the
        # shared trajectory coordinator and publish enabled renderer geometry.
        for key, val in [
            ("tx_trajectory_cb", state.show_tx_trajectory),
            ("rx_trajectory_cb", state.show_rx_trajectory),
            ("target_trajectory_cb", state.show_target_trajectory),
        ]:
            if key in w:
                w[key].setChecked(bool(val))

        # Trajectory color mode radio buttons
        traj_color_map = {
            "node_color": "trajectory_color_node_color_rb",
            "speed": "trajectory_color_speed_rb",
            "altitude": "trajectory_color_altitude_rb",
            "time": "trajectory_color_time_rb",
            "angular_speed": "trajectory_color_angular_speed_rb",
        }
        rb_name = traj_color_map.get(state.trajectory_color_mode)
        if rb_name and rb_name in w:
            self._set_widget(w[rb_name], True)

    def _sync_coverage_panel(self, panels: dict, state: object) -> None:
        """Sync Coverage panel widgets."""
        if "coverage" not in panels:
            return
        cov = panels["coverage"]
        if not hasattr(cov, "widgets"):
            return
        w = cov.widgets

        if "coverage_toggle" in w:
            self._set_widget(w["coverage_toggle"], state.show_coverage)
        if "coverage_height_combo" in w:
            self._set_widget(w["coverage_height_combo"], state.coverage_height_index)

    def _sync_beam_panel(self, panels: dict, state: object) -> None:
        """Sync Beam Pattern panel widgets."""
        if "beam_pattern" not in panels:
            return
        beam = panels["beam_pattern"]
        if not hasattr(beam, "widgets"):
            return
        w = beam.widgets

        if "beamforming_cb" in w:
            self._set_widget(w["beamforming_cb"], state.show_beamforming)

        # Mode radio buttons
        mode_map = {
            "frame": "mode_frame",
            "standalone": "mode_standalone",
        }
        optional_key = f"mode_optional_{state.standalone_beamforming_mode}"
        if optional_key in w:
            mode_map[state.standalone_beamforming_mode] = optional_key
        rb_name = mode_map.get(state.standalone_beamforming_mode)
        if rb_name and rb_name in w:
            self._set_widget(w[rb_name], True)

        # Standalone antenna parameters
        h_spacing_lambda = spacing_m_to_wavelengths(
            state.standalone_horizontal_spacing_m,
            state.standalone_carrier_frequency_ghz,
        )
        v_spacing_lambda = spacing_m_to_wavelengths(
            state.standalone_vertical_spacing_m,
            state.standalone_carrier_frequency_ghz,
        )
        standalone_map = {
            "standalone_rows": state.standalone_antenna_rows,
            "standalone_cols": state.standalone_antenna_cols,
            "standalone_freq": state.standalone_carrier_frequency_ghz,
            "standalone_h_spacing": h_spacing_lambda,
            "standalone_v_spacing": v_spacing_lambda,
            "standalone_azimuth": state.standalone_azimuth_deg,
            "standalone_elevation": state.standalone_elevation_deg,
        }
        for key, val in standalone_map.items():
            if key in w:
                self._set_widget(w[key], val)

        if "standalone_strategy" in w:
            strategy_text = {
                "svd": "SVD (Current MPCs)",
                "los": "LOS Steering",
                "manual": "Manual Steering",
            }.get(state.standalone_steering_strategy, "SVD (Current MPCs)")
            self._set_widget(w["standalone_strategy"], strategy_text)

        # Scale spinboxes
        if "beam_tx_scale_spin" in w:
            self._set_widget(w["beam_tx_scale_spin"], state.beamforming_tx_scale)
        if "beam_rx_scale_spin" in w:
            self._set_widget(w["beam_rx_scale_spin"], state.beamforming_rx_scale)

        # Display options
        if "beam_db_scale_cb" in w:
            self._set_widget(w["beam_db_scale_cb"], state.beamforming_db_scale)
        if "beam_dynamic_range" in w:
            self._set_widget(w["beam_dynamic_range"], state.beamforming_dynamic_range_db)
        if "beam_colormap" in w:
            self._set_widget(w["beam_colormap"], state.beamforming_colormap)
        if "beam_element_pattern" in w:
            self._set_widget(w["beam_element_pattern"], state.beamforming_element_pattern)
        if "beam_tx_element_pattern" in w:
            self._set_widget(w["beam_tx_element_pattern"], state.beamforming_tx_element_pattern)
        if "beam_rx_element_pattern" in w:
            self._set_widget(w["beam_rx_element_pattern"], state.beamforming_rx_element_pattern)
        update_beam_colorbar = getattr(beam, "update_beam_colorbar", None)
        if callable(update_beam_colorbar):
            update_beam_colorbar(
                show_beamforming=state.show_beamforming,
                db_scale=state.beamforming_db_scale,
                dynamic_range_db=state.beamforming_dynamic_range_db,
                colormap=state.beamforming_colormap,
            )

        # Context owns the selected pair and its display names. Delegate those
        # labels and their result status after restoring all beam parameters so
        # session sync cannot overwrite controller-derived text with raw IDs.
        beamforming_ui = getattr(self.viz, "beamforming_ui_controller", None)
        apply_selector_state = getattr(beamforming_ui, "apply_selector_state", None)
        if callable(apply_selector_state):
            apply_selector_state()

    def _sync_performance_panel(self, panels: dict, state: object) -> None:
        """Sync Performance panel widgets.

        Runtime performance diagnostics are sampled only while the panel is
        expanded, so workspace restore intentionally does not refresh them here.
        """
        if "performance" not in panels:
            return
        perf = panels["performance"]
        if not hasattr(perf, "widgets"):
            return

    def _sync_render_panel(self, panels: dict) -> None:
        """Sync Render panel widgets via the panel's own visualizer-aware hook."""
        render_panel = panels.get("render")
        if render_panel is None:
            return
        apply_runtime = getattr(render_panel, "apply_runtime_controls_from_state", None)
        if callable(apply_runtime):
            apply_runtime()
        sync_fn = getattr(render_panel, "_sync_from_visualizer", None)
        if callable(sync_fn):
            sync_fn()

    def _sync_materials_panel(self, panels: dict, _state: object) -> None:
        """Sync Materials panel RF X-Ray controls after workspace restore."""
        materials_panel = panels.get("materials")
        if materials_panel is None:
            return
        sync_fn = getattr(materials_panel, "_sync_rf_xray_controls", None)
        if callable(sync_fn):
            sync_fn()

    def _list_snapshot_paths(self, max_count: Optional[int] = None) -> list[Path]:
        """List snapshot files by modification time, newest first."""

        def _modified_at(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        sessions = sorted(self.session_dir.glob("*.json"), key=_modified_at, reverse=True)
        if max_count is not None:
            sessions = sessions[:max_count]
        return sessions

    def list_workspace_summaries(
        self,
        max_count: Optional[int] = None,
    ) -> list[WorkspaceSnapshotSummary]:
        """Return valid workspace summaries ordered newest file first."""
        if max_count is not None and max_count <= 0:
            return []
        summaries: list[WorkspaceSnapshotSummary] = []
        for path in self._list_snapshot_paths():
            summary = read_workspace_summary(path)
            if summary is None:
                continue
            summaries.append(summary)
            if max_count is not None and len(summaries) >= max_count:
                break
        return summaries

    def _paths_match(self, left: object, right: object) -> bool:
        """Compare two scenario references by their canonical root."""
        left_path = normalize_scenario_root(left)
        right_path = normalize_scenario_root(right)
        if left_path is None or right_path is None:
            return False
        return os.path.normcase(str(left_path)) == os.path.normcase(str(right_path))

    def auto_save_on_exit(self) -> Optional[Path]:
        """Update the rolling workspace snapshot when the visualizer closes."""
        if not self.auto_save_enabled:
            logger.debug("Auto-save disabled, skipping")
            return None
        if normalize_scenario_root(getattr(self.viz, "current_scenario_path", None)) is None:
            logger.debug("No scenario open; skipping workspace autosave")
            return None

        try:
            return self.save_session()
        except (OSError, IOError, PermissionError, ValueError) as e:
            logger.error(f"Auto-save failed: {e}", exc_info=True)
            return None

    # --- State Extraction Methods ---

    def _autosave_path(self, scenario_root: Path) -> Path:
        """Return the stable rolling autosave path for one scenario root."""
        slug = re.sub(r"[^a-z0-9]+", "-", scenario_root.name.lower()).strip("-")
        slug = (slug or "scenario")[:48]
        identity = os.path.normcase(str(scenario_root))
        path_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
        return self.session_dir / f"{slug}_autosave_{path_hash}.json"

    def _prune_older_autosaves(self, scenario_root: Path, *, keep_path: Path) -> None:
        """Keep the autosave just written and prune older recognized copies."""
        identity = os.path.normcase(str(scenario_root))
        keep_path = keep_path.resolve(strict=False)
        for path in self._list_snapshot_paths():
            summary = read_workspace_summary(path)
            if summary is None or not summary.is_autosave:
                continue
            if os.path.normcase(str(summary.scenario_root)) != identity:
                continue
            if summary.path == keep_path:
                continue
            try:
                summary.path.unlink()
                logger.info("Pruned older workspace autosave: %s", summary.path)
            except OSError as exc:
                logger.warning("Could not prune workspace autosave %s: %s", summary.path, exc)

    def _get_camera_state(self) -> dict:
        """Extract camera state from visualizer.

        Session camera state is renderer-agnostic. Backend matrices are
        intentionally excluded from the public camera contract.
        """
        state = {
            "format": CAMERA_SESSION_FORMAT,
            "mode": "overview",
            "eye": [0.0, 0.0, 10.0],
            "lookat": [0.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "fov": 60.0,
        }

        # Detect camera mode
        if hasattr(self.viz, "overview_mode_rb") and self.viz.overview_mode_rb.isChecked():
            state["mode"] = "overview"
        elif hasattr(self.viz, "follow_mode_rb") and self.viz.follow_mode_rb.isChecked():
            state["mode"] = "follow"
        elif hasattr(self.viz, "pov_mode_rb") and self.viz.pov_mode_rb.isChecked():
            state["mode"] = "pov"

        renderer = getattr(self.viz, "renderer", None)
        if renderer is not None and hasattr(renderer, "get_camera_state"):
            cam_state = renderer.get_camera_state()
            if isinstance(cam_state, CameraState):
                state["eye"] = list(cam_state.eye)
                state["lookat"] = list(cam_state.lookat)
                state["up"] = list(cam_state.up)
                state["fov"] = float(cam_state.fov_deg)
                logger.debug("Camera state saved via renderer CameraState")

        return state

    def _get_app_state(self) -> dict:
        """Extract app state (filters, selections, visibility toggles)."""
        if not hasattr(self.viz, "app_state"):
            return {}

        app_state = self.viz.app_state
        if hasattr(app_state, "to_dict"):
            payload = strip_beamforming_state(app_state.to_dict())
            payload.pop("step", None)
            payload["show_rf_xray"] = False
            payload["node_coloring_mode"] = getattr(self.viz, "node_coloring_mode", "per_type")
            return payload

        visibility = getattr(app_state, "mpc_visibility", None)
        if isinstance(visibility, MpcVisibility):
            visibility_payload = visibility.to_dict()
        elif isinstance(visibility, dict):
            visibility_payload = MpcVisibility.from_dict(visibility).to_dict()
        else:
            visibility_payload = MpcVisibility().to_dict()

        return {
            "mpc_visibility": visibility_payload,
            "show_coverage": (
                app_state.show_coverage if hasattr(app_state, "show_coverage") else False
            ),
            "show_rf_xray": False,
            "color_mode": app_state.color_mode if hasattr(app_state, "color_mode") else "material",
            "node_coloring_mode": getattr(self.viz, "node_coloring_mode", "per_type"),
        }

    def _get_animation_state(self) -> dict:
        """Extract the current frame; playback always resets on workspace load."""
        return {"current_frame": self.viz.app_state.step if hasattr(self.viz, "app_state") else 0}

    def _get_rendering_state(self) -> dict:
        """Extract rendering state (transparency, line widths, axes, etc).

        These are visual settings that live in the renderer and UI panels
        rather than in AppState.
        """
        state: dict = {}

        # Transparency values from visualizer instance
        state["building_alpha"] = getattr(self.viz, "current_building_alpha", 1.0)
        state["target_alpha"] = getattr(self.viz, "current_target_alpha", 1.0)
        state["coverage"] = self._get_coverage_rendering_state()
        state["node_appearance"] = self._get_node_appearance_state()

        # MPC appearance remains under its established top-level session keys,
        # but the Paths panel is now the sole UI owner.
        mpc_panel = self._get_panel("mpc")
        if mpc_panel is not None:
            mpc_widgets = getattr(mpc_panel, "widgets", {})
            point_size = mpc_widgets.get("point_size_spin")
            if point_size is not None:
                state["point_size"] = point_size.value()
            line_width = mpc_widgets.get("line_width_spin")
            if line_width is not None:
                state["mpc_line_width"] = line_width.value()

        # Read render panel widget values
        render_panel = self._get_panel("render")
        if render_panel is not None:
            widgets = getattr(render_panel, "widgets", {})
            # Edge line width (Open3D renderer)
            elw_slider = widgets.get("edge_line_width_slider")
            if elw_slider is not None:
                state["edge_line_width"] = elw_slider.value()

            # Background color preset
            bg_combo = widgets.get("bg_combo")
            if bg_combo is not None:
                state["background_preset"] = bg_combo.currentText()

            # Wireframe edges
            outline_cb = widgets.get("outline_cb")
            if outline_cb is not None:
                state["show_edges"] = outline_cb.isChecked()
            target_outline_cb = widgets.get("target_outline_cb")
            if target_outline_cb is not None:
                state["show_target_edges"] = target_outline_cb.isChecked()

            # Show axes (Open3D renderer)
            axes_cb = widgets.get("show_axes_cb")
            if axes_cb is not None:
                state["show_axes"] = axes_cb.isChecked()

        # Active tab index
        ui_mgr = getattr(self.viz, "ui_manager", None)
        if ui_mgr is not None:
            tab_widget = getattr(ui_mgr, "_tab_widget", None)
            if tab_widget is not None:
                state["active_tab"] = tab_widget.currentIndex()
            if hasattr(ui_mgr, "get_active_tab_label"):
                active_tab_label = ui_mgr.get_active_tab_label()
                if active_tab_label:
                    state["active_tab_label"] = active_tab_label

        return state

    def _get_node_appearance_state(self) -> dict[str, float]:
        """Extract user-entered node and trajectory appearance values."""
        panel = self._get_panel("nodes")
        widgets = getattr(panel, "widgets", {}) if panel is not None else {}
        renderer = getattr(self.viz, "renderer", None)
        state: dict[str, float] = {}

        for key, (attribute, default, bounds) in _NODE_APPEARANCE_ATTRIBUTE_SPECS.items():
            value = getattr(self.viz, attribute, default)
            widget = widgets.get(_NODE_APPEARANCE_WIDGET_KEYS[key])
            if widget is not None:
                value = widget.value()
            validated = self._validated_bounded_float(value, bounds)
            state[key] = default if validated is None else validated

        for key, (attribute, default, bounds) in _TRAJECTORY_APPEARANCE_SPECS.items():
            value = getattr(renderer, attribute, default)
            widget = widgets.get(_NODE_APPEARANCE_WIDGET_KEYS[key])
            if widget is not None:
                value = widget.value()
            validated = self._validated_bounded_float(value, bounds)
            state[key] = default if validated is None else validated

        return state

    def _get_coverage_rendering_state(self) -> dict[str, Any]:
        """Extract non-AppState coverage controls from their real UI/runtime owners."""
        panel = self._get_panel("coverage")
        widgets = getattr(panel, "widgets", {}) if panel is not None else {}
        controller = getattr(getattr(self.viz, "ui_controller", None), "coverage_controller", None)
        coverage_data = getattr(self.viz, "coverage_data", None)

        def _checked(key: str, fallback: bool) -> bool:
            widget = widgets.get(key)
            return bool(widget.isChecked()) if widget is not None else bool(fallback)

        def _value(key: str, fallback: Any) -> Any:
            widget = widgets.get(key)
            return widget.value() if widget is not None else fallback

        opacity_widget = widgets.get("coverage_opacity")
        opacity = (
            float(opacity_widget.value()) / 100.0
            if opacity_widget is not None
            else float(getattr(self.viz, "coverage_opacity", DEFAULT_COVERAGE_OPACITY))
        )
        metric_name = None
        if isinstance(coverage_data, dict):
            metric_name = coverage_data.get("metric_name")
        if not metric_name:
            metric_name = getattr(self.viz, "coverage_metric_name", None)

        threshold_value = _value(
            "coverage_threshold_value",
            getattr(self.viz, "coverage_threshold_value", None),
        )
        return {
            "opacity": opacity,
            "metric_name": metric_name,
            "interpolation": str(
                getattr(
                    self.viz,
                    "coverage_interpolation_method",
                    getattr(
                        controller, "coverage_interpolation_method", DEFAULT_COVERAGE_INTERPOLATION
                    ),
                )
            ),
            "threshold_enabled": _checked(
                "coverage_threshold_toggle",
                getattr(self.viz, "coverage_threshold_enabled", False),
            ),
            "threshold_value": (float(threshold_value) if threshold_value is not None else None),
            "threshold_mask_enabled": _checked(
                "coverage_threshold_mask_toggle",
                getattr(self.viz, "coverage_threshold_mask_enabled", False),
            ),
            "isolines_enabled": _checked(
                "coverage_isolines_toggle",
                getattr(self.viz, "coverage_isolines_enabled", False),
            ),
            "isoline_count": int(
                _value(
                    "coverage_isoline_count",
                    getattr(
                        self.viz,
                        "coverage_isoline_count",
                        DEFAULT_COVERAGE_ISOLINE_COUNT,
                    ),
                )
            ),
            "height_animation_speed": int(
                _value(
                    "coverage_height_speed",
                    getattr(
                        controller,
                        "height_animation_speed",
                        DEFAULT_COVERAGE_HEIGHT_ANIMATION_SPEED,
                    ),
                )
            ),
        }

    def _semantic_entries(self) -> list[tuple[str, dict[str, Any], bool]]:
        """Return canonical entries with stable IDs and label defaults.

        The boolean in each tuple is the domain's default per-entry label
        intent. Scene labels are opt-in; target and node labels are opt-out.
        """
        entries: list[tuple[str, dict[str, Any], bool]] = []
        for index, entry in enumerate(getattr(self.viz, "mesh_entries", []) or []):
            if isinstance(entry, dict):
                entries.append((ensure_scene_entry_identity(entry, index), entry, False))
        for index, entry in enumerate(getattr(self.viz, "target_entries", []) or []):
            if isinstance(entry, dict):
                entries.append((ensure_target_entry_identity(entry, index), entry, True))
        for kind in ("tx", "rx"):
            for index, entry in enumerate(getattr(self.viz, f"{kind}_entries", []) or []):
                if isinstance(entry, dict):
                    entries.append((ensure_node_entry_identity(entry, kind, index), entry, True))
        return entries

    def _get_entry_state_snapshot(self) -> dict[str, dict[str, bool]]:
        """Serialize application-owned visibility intent by stable entity ID."""
        return {
            entry_id: {
                "visible": bool(entry.get("visible", True)),
                "show_label": bool(entry.get("show_label", default_show_label)),
            }
            for entry_id, entry, default_show_label in self._semantic_entries()
        }

    def _restore_entry_state(
        self,
        entry_state: dict[str, Any],
    ) -> dict[str, dict[str, bool]]:
        """Restore changed semantic flags and return their per-entry deltas."""
        if not isinstance(entry_state, dict):
            return {}
        semantic_entries = self._semantic_entries()
        entries_by_id = {
            entry_id: (entry, default_show_label)
            for entry_id, entry, default_show_label in semantic_entries
        }
        deltas: dict[str, dict[str, bool]] = {}
        for raw_entry_id, raw_intent in entry_state.items():
            entry_id = str(raw_entry_id)
            resolved = entries_by_id.get(entry_id)
            if resolved is None or not isinstance(raw_intent, dict):
                continue
            entry, default_show_label = resolved
            changed_fields: dict[str, bool] = {}
            for field, default in (
                ("visible", True),
                ("show_label", default_show_label),
            ):
                restored = raw_intent.get(field)
                if not isinstance(restored, bool):
                    continue
                if bool(entry.get(field, default)) == restored:
                    continue
                entry[field] = restored
                changed_fields[field] = restored
            if changed_fields:
                deltas[entry_id] = changed_fields
        return deltas

    def _refresh_entry_state(
        self,
        deltas: dict[str, dict[str, bool]] | None = None,
    ) -> None:
        """Publish only changed entry intent through semantic services.

        Passing ``None`` retains a full-refresh path for direct callers. An
        explicit empty mapping means the restored entry state was already
        identical and therefore requires no service, renderer, or panel work.
        """
        semantic_entries = self._semantic_entries()
        if deltas is None:
            changes_by_id = {
                entry_id: {
                    "visible": bool(entry.get("visible", True)),
                    "show_label": bool(entry.get("show_label", default_show_label)),
                }
                for entry_id, entry, default_show_label in semantic_entries
            }
        else:
            changes_by_id = {
                str(entry_id): {
                    field: value
                    for field, value in changes.items()
                    if field in {"visible", "show_label"} and isinstance(value, bool)
                }
                for entry_id, changes in deltas.items()
                if isinstance(changes, dict)
            }
            changes_by_id = {
                entry_id: changes for entry_id, changes in changes_by_id.items() if changes
            }
            if not changes_by_id:
                return

        entries_by_id = {entry_id: entry for entry_id, entry, _default in semantic_entries}
        scene_changes: list[tuple[dict[str, Any], dict[str, bool]]] = []
        target_changes: list[tuple[dict[str, Any], dict[str, bool]]] = []
        node_changed = False
        for entry_id, changes in changes_by_id.items():
            entry = entries_by_id.get(entry_id)
            if entry is None:
                continue
            if entry_id.startswith("scene:"):
                scene_changes.append((entry, changes))
            elif entry_id.startswith("target:"):
                target_changes.append((entry, changes))
            elif entry_id.startswith("node:"):
                node_changed = True

        if not scene_changes and not target_changes and not node_changed:
            return

        appearance = getattr(self.viz, "object_appearance_service", None)
        if appearance is not None:
            visibility_entries = [
                entry
                for entry, changes in (*scene_changes, *target_changes)
                if "visible" in changes
            ]
            refresh_visibility = getattr(
                appearance,
                "refresh_object_visibility_batch",
                None,
            )
            if visibility_entries and callable(refresh_visibility):
                refresh_visibility(visibility_entries, update_renderer=False)

            # A visibility refresh already republishes effective scene-label
            # state. Only label-only changes need the independent label path.
            set_label_visibility = getattr(
                appearance,
                "set_building_label_visibility",
                None,
            )
            if callable(set_label_visibility):
                for entry, changes in scene_changes:
                    if "show_label" in changes and "visible" not in changes:
                        set_label_visibility(
                            entry,
                            bool(entry.get("show_label", False)),
                            update_renderer=False,
                        )

        node_service = getattr(self.viz, "node_service", None)
        update_nodes = getattr(node_service, "update_tx_rx_visibility", None)
        if node_changed and callable(update_nodes):
            update_nodes()

        # Target visibility snapshots include the label. Only target
        # label-only changes require the separate label synchronization pass.
        target_label_only_changed = any(
            "show_label" in changes and "visible" not in changes
            for _entry, changes in target_changes
        )
        update_target_labels = getattr(node_service, "update_target_label_visibility", None)
        if target_label_only_changed and callable(update_target_labels):
            update_target_labels()

        panels = getattr(getattr(self.viz, "ui_manager", None), "panels", {})
        obj_panel = panels.get("objects") if isinstance(panels, dict) else None
        sync_entries = getattr(obj_panel, "sync_all_entry_states", None)
        if callable(sync_entries):
            sync_entries()

    def _get_panel(self, name: str):
        """Get a UI panel by name, or None if not available."""
        if not hasattr(self.viz, "ui_manager") or self.viz.ui_manager is None:
            return None
        panels = getattr(self.viz.ui_manager, "panels", {})
        return panels.get(name)

    # --- State Restoration Methods ---

    def _restore_camera_state(self, camera_data: dict):
        """Restore camera state to visualizer.

        Camera snapshots use only renderer-agnostic orbit fields. Renderer
        matrix payloads from older schemas are intentionally unsupported.
        """
        if not hasattr(self.viz, "renderer") or self.viz.renderer is None:
            return

        renderer = self.viz.renderer
        if camera_data.get("format") != CAMERA_SESSION_FORMAT:
            logger.warning(
                "Skipping unsupported workspace camera format: %s",
                camera_data.get("format"),
            )
            return

        # Restore camera mode
        mode = camera_data.get("mode", "overview")
        if mode in {"follow", "pov"}:
            if not hasattr(renderer, "set_camera_state"):
                logger.warning(
                    "Renderer '%s' does not support '%s' camera mode restore; forcing overview",
                    getattr(renderer, "renderer_id", type(renderer).__name__),
                    mode,
                )
                mode = "overview"

        button_name = {
            "overview": "overview_mode_rb",
            "follow": "follow_mode_rb",
            "pov": "pov_mode_rb",
        }.get(mode)
        button = getattr(self.viz, button_name, None) if button_name else None
        if button is not None and hasattr(button, "setChecked"):
            if hasattr(button, "blockSignals"):
                with QSignalBlocker(button):
                    button.setChecked(True)
            else:
                button.setChecked(True)

        try:
            camera_state = CameraState.from_dict(
                {
                    "eye": camera_data["eye"],
                    "lookat": camera_data["lookat"],
                    "up": camera_data["up"],
                    "fov": camera_data["fov"],
                }
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping invalid workspace camera payload")
            return

        if hasattr(renderer, "set_camera_state") and renderer.set_camera_state(camera_state):
            logger.info("Camera restored from workspace state: eye=%s...", camera_state.eye[:2])
            return

        logger.warning("Renderer did not accept workspace camera state")

    def _prepare_app_state_restore(
        self,
        app_state_data: object,
        *,
        frame: Optional[int] = None,
    ) -> _PreparedAppStateRestore:
        """Parse one schema-v6 application-state section without mutating the app."""
        if not isinstance(app_state_data, dict):
            raise TypeError("app_state section must be a mapping")
        payload = dict(app_state_data)
        if frame is not None:
            payload["step"] = frame
        node_coloring_mode = payload.pop("node_coloring_mode", None)
        if node_coloring_mode not in {"per_type", "individual", None}:
            node_coloring_mode = "per_type"

        for key in BEAMFORMING_STATE_KEYS:
            payload.pop(key, None)
        payload.update(get_beamforming_state_defaults())
        payload["show_rf_xray"] = False
        payload.pop("rf_xray_show_bounces", None)

        return _PreparedAppStateRestore(
            state=AppState.from_dict(payload),
            node_coloring_mode=node_coloring_mode,
        )

    def _restore_app_state(
        self,
        app_state_data: object,
        *,
        frame: Optional[int] = None,
        prepared: _PreparedAppStateRestore | None = None,
    ) -> None:
        """Apply one validated AppState, propagating failures to the transaction."""
        set_state = getattr(self.viz, "set_state", None)
        if not callable(set_state):
            raise RuntimeError("Visualizer cannot apply workspace application state")

        prepared_state = prepared or self._prepare_app_state_restore(
            app_state_data,
            frame=frame,
        )
        restored = prepared_state.state
        changes = {field.name: getattr(restored, field.name) for field in fields(AppState)}
        set_state(**changes)
        node_coloring_mode = prepared_state.node_coloring_mode
        if node_coloring_mode in {"per_type", "individual"}:
            self.viz.node_coloring_mode = node_coloring_mode
            node_service = getattr(self.viz, "node_service", None)
            apply_coloring = getattr(node_service, "apply_node_coloring", None)
            if callable(apply_coloring):
                apply_coloring()
            update_legend = getattr(node_service, "update_node_coloring_legend", None)
            if callable(update_legend):
                update_legend()
        logger.info("Restored app state: %d fields", len(changes))

    def _restore_animation_frame(self, frame: int) -> bool:
        """Navigate to one validated saved frame and report pipeline success."""
        if bool(getattr(self.viz, "_shutdown_started", False)):
            logger.warning("Workspace frame restore skipped during application shutdown")
            return False
        display_index = frame
        get_display_index = getattr(self.viz, "get_animation_step_index", None)
        if callable(get_display_index):
            try:
                display_index = max(0, int(get_display_index(frame)))
            except (TypeError, ValueError, RuntimeError):
                logger.debug("Could not resolve display index for workspace frame %s", frame)

        # Ensure animation widgets can represent the restored frame; otherwise
        # Qt spin/slider values clamp to their old max (often 0/1 on startup).
        slider = getattr(self.viz, "step_slider", None)
        if slider is not None and hasattr(slider, "maximum") and hasattr(slider, "setMaximum"):
            try:
                slider_max = int(slider.maximum())
                if display_index > slider_max:
                    with QSignalBlocker(slider):
                        slider.setMaximum(display_index)
            except (TypeError, ValueError, RuntimeError, AttributeError):
                logger.debug("Could not resize step_slider range for restore frame %s", frame)

        frame_input = getattr(self.viz, "frame_input", None)
        if (
            frame_input is not None
            and hasattr(frame_input, "maximum")
            and hasattr(frame_input, "setMaximum")
        ):
            try:
                frame_input_max = int(frame_input.maximum())
                desired_max = display_index + 1  # UI uses 1-based frame numbers
                if desired_max > frame_input_max:
                    with QSignalBlocker(frame_input):
                        frame_input.setMaximum(desired_max)
            except (TypeError, ValueError, RuntimeError, AttributeError):
                logger.debug("Could not resize frame_input range for restore frame %s", frame)

        update_frame = getattr(self.viz, "update_frame", None)
        if not callable(update_frame):
            logger.warning("Workspace frame skipped: frame command is unavailable")
            return False
        try:
            # Force a true frame recompute during workspace hydration even when
            # state equality checks would otherwise short-circuit.
            self.viz.force_update_next_frame = True
            frame_completed = bool(update_frame(frame))
            if not frame_completed:
                logger.warning("Workspace frame pipeline did not complete for frame %d", frame)
                return False
            logger.info("Restored animation frame: %d", frame)
            return True
        except (ValueError, TypeError, RuntimeError, FileNotFoundError) as exc:
            logger.warning("Failed to restore frame %d: %s", frame, exc)
            return False

    def _restore_coverage_rendering_state(self, coverage_state: dict[str, Any]) -> None:
        """Delegate a saved coverage snapshot to its controller owner."""
        if not isinstance(coverage_state, dict):
            return
        controller = getattr(getattr(self.viz, "ui_controller", None), "coverage_controller", None)
        restore = getattr(controller, "restore_session_state", None)
        if callable(restore):
            restore(dict(coverage_state))
        else:
            logger.warning("Coverage workspace state skipped: controller is unavailable")

    def _restore_node_appearance_state(self, appearance_state: dict[str, Any]) -> None:
        """Restore validated node and trajectory appearance through semantic owners."""
        if not isinstance(appearance_state, dict):
            return

        restored: dict[str, float] = {}
        for key, (attribute, _default, bounds) in _NODE_APPEARANCE_ATTRIBUTE_SPECS.items():
            if key not in appearance_state:
                continue
            value = self._validated_bounded_float(appearance_state[key], bounds)
            if value is None:
                logger.warning("Skipping invalid node appearance value %s", key)
                continue
            setattr(self.viz, attribute, value)
            restored[key] = value

        renderer = getattr(self.viz, "renderer", None)
        for key, (attribute, _default, bounds) in _TRAJECTORY_APPEARANCE_SPECS.items():
            if key not in appearance_state:
                continue
            value = self._validated_bounded_float(appearance_state[key], bounds)
            if value is None:
                logger.warning("Skipping invalid node appearance value %s", key)
                continue
            restored[key] = value
            setter = getattr(renderer, f"set_{attribute}", None)
            if callable(setter):
                setter(value)
            elif renderer is not None:
                setattr(renderer, attribute, value)

        if not restored:
            return

        node_service = getattr(self.viz, "node_service", None)
        initialized = bool(getattr(self.viz, "vis_initialized", False))
        if initialized and node_service is not None:
            if "tx_marker_size_m" in restored:
                update_tx = getattr(node_service, "update_tx_marker_sizes", None)
                if callable(update_tx):
                    update_tx()
            if "rx_marker_size_m" in restored:
                update_rx = getattr(node_service, "update_rx_marker_sizes", None)
                if callable(update_rx):
                    update_rx()
            if "label_font_size" in restored:
                recreate_tx_rx = getattr(node_service, "recreate_tx_rx_labels", None)
                if callable(recreate_tx_rx):
                    recreate_tx_rx(restored["label_font_size"])
                recreate_targets = getattr(node_service, "recreate_target_labels", None)
                if callable(recreate_targets):
                    recreate_targets(restored["label_font_size"])
            if any(key.startswith("label_offset_") for key in restored):
                apply_offsets = getattr(node_service, "apply_label_offsets", None)
                if callable(apply_offsets):
                    apply_offsets()

        if "orientation_scale_m" in restored:
            controller = getattr(self.viz, "ui_controller", None)
            apply_orientation_scale = getattr(
                controller,
                "handle_orientation_scale_changed",
                None,
            )
            if callable(apply_orientation_scale):
                apply_orientation_scale(restored["orientation_scale_m"])

    def _restore_mpc_appearance_state(self, render_data: dict[str, Any]) -> None:
        """Restore established MPC size keys through the Paths-panel owner."""
        specs = (
            (
                "point_size",
                MPC_POINT_SIZE_BOUNDS_PX,
                "set_point_size",
            ),
            (
                "mpc_line_width",
                MPC_LINE_WIDTH_BOUNDS_PX,
                "set_line_width",
            ),
        )
        restored: dict[str, float] = {}
        for key, bounds, _setter_name in specs:
            if key not in render_data:
                continue
            value = self._validated_bounded_float(render_data[key], bounds)
            if value is None:
                logger.warning("Skipping invalid MPC appearance value %s", key)
                continue
            restored[key] = value
        if not restored:
            return

        mpc_panel = self._get_panel("mpc")
        restore = getattr(mpc_panel, "restore_session_state", None)
        if callable(restore):
            restore(restored)
            return

        renderer = getattr(self.viz, "renderer", None)
        for key, _bounds, setter_name in specs:
            if key not in restored:
                continue
            setter = getattr(renderer, setter_name, None)
            if callable(setter):
                setter(restored[key])

    def _restore_rendering_state(self, render_data: dict):
        """Restore rendering state (transparency, line widths, axes, etc)."""
        render_data = dict(render_data)
        for alpha_key in ("building_alpha", "target_alpha"):
            if alpha_key not in render_data:
                continue
            alpha = self._validated_alpha(render_data[alpha_key])
            if alpha is None:
                render_data.pop(alpha_key)
            else:
                render_data[alpha_key] = alpha

        coverage_state = render_data.get("coverage")
        if isinstance(coverage_state, dict):
            coverage_state = dict(coverage_state)
            if "opacity" not in coverage_state and "coverage_alpha" in render_data:
                coverage_state["opacity"] = render_data["coverage_alpha"]
        elif "coverage_alpha" in render_data:
            coverage_state = {"opacity": render_data["coverage_alpha"]}
        else:
            coverage_state = None
        if coverage_state is not None:
            self._restore_coverage_rendering_state(coverage_state)

        # Transparency is durable render-panel state, applied by its owner.
        scene_appearance = getattr(self.viz, "scene_appearance_service", None)
        if "building_alpha" in render_data and scene_appearance is not None:
            scene_appearance.set_building_transparency(render_data["building_alpha"])

        if "target_alpha" in render_data and scene_appearance is not None:
            scene_appearance.set_target_transparency(render_data["target_alpha"])

        self._restore_mpc_appearance_state(render_data)

        # Renderer-only controls belong to the render panel. It applies the
        # batch once and mirrors connected widgets with signals blocked.
        render_panel = self._get_panel("render")
        if render_panel is not None:
            restore = getattr(render_panel, "restore_session_state", None)
            if callable(restore):
                restore(dict(render_data))

        node_appearance = render_data.get("node_appearance")
        if isinstance(node_appearance, dict):
            node_appearance = dict(node_appearance)
        else:
            node_appearance = {}

        # Version-6 snapshots written before node appearance had a dedicated
        # block stored trajectory width under Rendering. Keep that input
        # compatible while the Nodes panel remains the sole runtime/UI owner.
        if "trajectory_line_width_px" not in node_appearance and "traj_line_width" in render_data:
            node_appearance["trajectory_line_width_px"] = render_data["traj_line_width"]
        if node_appearance:
            self._restore_node_appearance_state(node_appearance)

        # Tab availability and stable labels belong to the panel manager.
        ui_mgr = getattr(self.viz, "ui_manager", None)
        if ui_mgr is not None:
            restore_tab = getattr(ui_mgr, "restore_active_tab", None)
            if callable(restore_tab):
                restore_tab(
                    label=render_data.get("active_tab_label"),
                    index=render_data.get("active_tab"),
                )

        logger.debug("Restored rendering state: %d keys", len(render_data))
