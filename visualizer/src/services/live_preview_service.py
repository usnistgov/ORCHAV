"""Coordinate temporary actor edits for file and live-gRPC sources.

File-backed scenarios use the local preview worker and never change scenario
files. Live-gRPC scenarios send one acknowledged actor update when a gizmo drag
is committed; the connected generator remains authoritative. Remote frame sets
are read-only.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QTimer

from shared.frames import StandardMPCFrame
from shared.source_identity import (
    SourceIdentity,
    loaded_source_identity,
    source_bound_module_command,
)

from ..io.packed_frame_payload import (
    standard_frame_to_visual_frame,
    visual_frame_read_request,
)
from ..model import RenderObjectState, render_state_center
from ..renderers.protocol import renderer_capabilities
from ..scene.target_transforms import (
    build_sionna_rotation_matrix,
    closest_proper_rotation,
    orientation_metadata,
    sionna_orientation_from_rotation_matrix,
    sionna_orientation_from_transform,
)
from ..services.cache_service import CacheInvalidationScope, invalidate_visualizer_cache
from ..services.interactive_edit_session import (
    InteractiveEdit,
    InteractiveEditSession,
    InteractivePose,
)
from ..services.live_preview_payloads import encode_live_preview_arrays
from ..services.object_identity import normalize_token
from ..services.raytracing_settings_service import RaytracingSettingsService
from .base import BaseService

logger = logging.getLogger(__name__)

_DRAG_PREVIEW_INTERVAL_MS = 100
_WORKER_POLL_INTERVAL_MS = 50
_WORKER_SHUTDOWN_TIMEOUT_S = 1.0
_WORKER_STDERR_TAIL_LINES = 20
_EDIT_MODE_FILES = "files"
_EDIT_MODE_LIVE = "live_grpc"
_EDIT_MODE_REMOTE = "remote_hdf5"
_EDIT_MODE_UNAVAILABLE = "unavailable"


def _canonical_degrees(angle_radians: float) -> float:
    """Return one radian angle in the canonical ``[-180, 180)`` degree range."""
    degrees = float(np.degrees(angle_radians))
    wrapped = (degrees + 180.0) % 360.0 - 180.0
    if np.isclose(abs(wrapped), 180.0, rtol=0.0, atol=1e-10):
        return -180.0
    if np.isclose(wrapped, 0.0, rtol=0.0, atol=1e-12):
        return 0.0
    return wrapped


class LivePreviewService(BaseService):
    """Own temporary actor edits and route them to the active source mode."""

    def __init__(self, visualizer: Any, *, debounce_ms: int = _DRAG_PREVIEW_INTERVAL_MS) -> None:
        """Create the service without starting the expensive worker process."""
        super().__init__()
        self.visualizer = visualizer
        self._fallback_raytracing_settings = RaytracingSettingsService()
        self._drag_interval_ms = max(16, int(debounce_ms))
        self._enabled = False
        self._active_edit_mode: Optional[str] = None
        self._live_edit_in_flight = False
        self._live_committed_poses: dict[tuple[str, int], InteractivePose] = {}
        self._sequence = 0
        self._requested_sequence = 0
        self._solving = False
        self._last_error: Optional[str] = None
        self._source_identity = loaded_source_identity("visualizer")

        self._worker_process: Optional[subprocess.Popen[str]] = None
        self._worker_temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._worker_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._worker_stdout_thread: Optional[threading.Thread] = None
        self._worker_stderr_thread: Optional[threading.Thread] = None
        self._worker_ready = False
        self._worker_busy = False
        self._pending_profile: Optional[str] = None
        self._stderr_tail: list[str] = []
        self._runtime_scenario_root: Optional[Path] = None
        self._target_edit_baseline: Optional[dict[str, Any]] = None
        self._edit_session = InteractiveEditSession()

        self._drag_timer = QTimer()
        self._drag_timer.setSingleShot(True)
        self._drag_timer.timeout.connect(self._flush_drag_preview_request)

        self._poll_timer = QTimer()
        self._poll_timer.setInterval(_WORKER_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_worker_messages)

    @property
    def enabled(self) -> bool:
        """Whether transform callbacks and preview recomputation are active."""
        return self._enabled

    def is_available(self) -> bool:
        """Return whether the active source and renderer allow actor transforms."""
        renderer = getattr(self.visualizer, "renderer", None)
        return bool(
            self.edit_mode() in {_EDIT_MODE_FILES, _EDIT_MODE_LIVE}
            and renderer is not None
            and renderer_capabilities(renderer).transform_gizmo
            and callable(getattr(renderer, "begin_live_preview_transform_session", None))
            and callable(getattr(renderer, "end_live_preview_transform_session", None))
        )

    def edit_mode(self) -> str:
        """Return the edit policy selected by the active frame source."""
        from ..io.frame_sources import FileSource, LiveGrpcSource, RemoteHdf5Source

        source = getattr(self.visualizer, "frame_source", None)
        if isinstance(source, FileSource):
            return _EDIT_MODE_FILES
        if isinstance(source, LiveGrpcSource):
            return _EDIT_MODE_LIVE
        if isinstance(source, RemoteHdf5Source):
            return _EDIT_MODE_REMOTE

        scenario = getattr(self.visualizer, "scenario", None)
        configured = str(getattr(scenario, "data_mode", "") or "")
        if source is None and configured in {
            _EDIT_MODE_FILES,
            _EDIT_MODE_LIVE,
            _EDIT_MODE_REMOTE,
        }:
            return configured
        if source is None and scenario is not None:
            return _EDIT_MODE_FILES
        return _EDIT_MODE_UNAVAILABLE

    def set_enabled(self, enabled: bool) -> bool:
        """Enable or disable the transform session for the active source mode."""
        desired = bool(enabled)
        if desired and not self.is_available():
            mode = self.edit_mode()
            if mode == _EDIT_MODE_REMOTE:
                self._set_status("Remote HDF5 frame sets are read-only")
            else:
                self._set_status("Actor editing requires the pygfx renderer")
            self._sync_panel_controls()
            return False

        renderer = getattr(self.visualizer, "renderer", None)
        begin_session = getattr(renderer, "begin_live_preview_transform_session", None)
        end_session = getattr(renderer, "end_live_preview_transform_session", None)

        if desired:
            if not callable(begin_session) or not begin_session(self.handle_node_transform_event):
                self._enabled = False
                self.visualizer._live_preview_enabled = False
                self._set_status("Actor editing requires the pygfx renderer")
                self._sync_panel_controls()
                return False
            self._active_edit_mode = self.edit_mode()
            self._enabled = True
            self.visualizer._live_preview_enabled = True
            if self._active_edit_mode == _EDIT_MODE_FILES:
                if self._ensure_worker_started():
                    self._set_status("Local preview initializing...")
                else:
                    self._enabled = False
                    self._active_edit_mode = None
                    self.visualizer._live_preview_enabled = False
                    if callable(end_session):
                        end_session()
                    self._sync_panel_controls()
                    return False
            else:
                self._stop_worker()
                self._clear_preview_frame()
                self._set_status("Live session editing enabled; release the gizmo to apply")
        else:
            prior_mode = self._active_edit_mode
            self._enabled = False
            self._active_edit_mode = None
            self.visualizer._live_preview_enabled = False
            if callable(end_session):
                end_session()
            self._drag_timer.stop()
            self._stop_worker()
            self._clear_preview_frame()
            self._set_status(
                "Live session editing off" if prior_mode == _EDIT_MODE_LIVE else "Local preview off"
            )
        self._sync_panel_controls()
        return True

    def reset(self) -> None:
        """Return live preview to an inactive state for scenario/app teardown."""
        self._drag_timer.stop()
        self._enabled = False
        self._active_edit_mode = None
        self._live_edit_in_flight = False
        self._live_committed_poses.clear()
        self.visualizer._live_preview_enabled = False
        self._solving = False
        self._pending_profile = None
        self._runtime_scenario_root = None
        self._target_edit_baseline = None
        self._edit_session.clear()
        self.visualizer.current_target_positions = None
        self.visualizer.current_target_orientations = None
        renderer = getattr(self.visualizer, "renderer", None)
        end_session = getattr(renderer, "end_live_preview_transform_session", None)
        if callable(end_session):
            end_session()
        self._stop_worker()
        self._clear_preview_frame()
        self._sync_panel_controls()

    def stop(self) -> None:
        """Stop background preview resources through the service contract."""
        super().stop()
        self.reset()

    def handle_node_transform_event(self, event: dict[str, Any]) -> None:
        """Apply a renderer transform event and route its committed edit."""
        if not self._enabled:
            return
        kind = str(event.get("kind", "")).lower()
        if kind not in {"tx", "rx", "target"}:
            return
        try:
            index = int(event.get("index"))
            position = np.asarray(event.get("position"), dtype=np.float64).reshape(-1)[:3]
        except (TypeError, ValueError):
            return
        if position.size != 3 or not np.all(np.isfinite(position)):
            return

        phase = str(event.get("phase", "changed"))
        if kind == "target":
            if not self._apply_edited_target_pose(
                index,
                position,
                event,
                phase=phase,
                selected=phase == "selected",
            ):
                return
            label = self._target_status_label(index)
        else:
            label = f"{kind.upper()}{index + 1}"
            self._target_edit_baseline = None
            if phase == "selected":
                self._set_status(f"{label} selected")
                return
            if not self._apply_edited_position(kind, index, position):
                return

        if phase == "selected":
            self._set_status(f"{label} selected")
            return
        if phase == "committed":
            self._drag_timer.stop()
            if self._active_edit_mode == _EDIT_MODE_LIVE:
                self._commit_live_edit(kind, index, label)
            else:
                self._set_status(f"{label} edited - solving")
                self._request_solve("release")
            return

        if self._active_edit_mode == _EDIT_MODE_LIVE:
            self._set_status(f"{label} moved locally; release to apply")
        else:
            self._schedule_drag_preview_request()

    def recompute_now(self) -> None:
        """Request a release-quality local preview solve when files are active."""
        if not self._enabled:
            if not self.set_enabled(True):
                return
        if self._active_edit_mode == _EDIT_MODE_LIVE:
            self._set_status("Live edits are recomputed when the gizmo is released")
            return
        self._request_solve("release")

    def _commit_live_edit(self, kind: str, index: int, label: str) -> bool:
        """Send one completed gizmo edit to the controlling live generator."""
        edit = self._edit_session.get(kind, index)
        if edit is None:
            self._set_status(f"{label} has no valid edit to apply")
            return False
        if self._live_edit_in_flight:
            self._restore_rejected_live_edit(edit.kind, edit.index, edit)
            self._set_status("Finish the current live edit before applying another")
            return False

        self._live_edit_in_flight = True
        self._set_status(f"Applying {label} to the live generator...")
        try:
            success = self._send_live_pose(edit.kind, edit.index, edit.current)
        finally:
            self._live_edit_in_flight = False

        if success:
            self._live_committed_poses[(edit.kind, edit.index)] = edit.current
            self._set_status(f"{label} applied to the live generator")
            self._sync_panel_controls()
            return True

        self._restore_rejected_live_edit(edit.kind, edit.index, edit)
        self._set_status(f"{label} update rejected; restored the last accepted pose")
        self._sync_panel_controls()
        return False

    def _send_live_pose(self, kind: str, index: int, pose: InteractivePose) -> bool:
        """Apply one semantic actor pose through the acknowledged node RPC."""
        entry = self._live_entry(kind, index)
        scene_edit = getattr(self.visualizer, "scene_edit_service", None)
        apply_update = getattr(scene_edit, "edit_node_properties", None)
        if entry is None or not callable(apply_update):
            logger.error("Live actor edit service is unavailable for %s[%d]", kind, index)
            return False

        values: dict[str, Any] = {"position": list(pose.position)}
        if pose.orientation is not None:
            values["orientation"] = [_canonical_degrees(value) for value in pose.orientation]
        return bool(apply_update(entry, values))

    def _live_entry(self, kind: str, index: int) -> dict[str, Any] | None:
        """Resolve a gizmo identity to the canonical actor entry used by live RPCs."""
        kind = str(kind).lower()
        entries = getattr(self.visualizer, f"{kind}_entries", None)
        if not isinstance(entries, list) or index < 0 or index >= len(entries):
            return None
        entry = entries[index]
        if not isinstance(entry, dict):
            return None
        entry.setdefault("entry_type", kind)
        entry.setdefault("node_index", int(index))
        entry.setdefault("supports_position", True)
        if kind == "target":
            entry.setdefault("supports_orientation", True)
        else:
            entry.setdefault("node_name", f"{kind}_{index + 1}")
            entry.setdefault("supports_orientation", False)
        return entry

    def _restore_rejected_live_edit(
        self,
        kind: str,
        index: int,
        edit: InteractiveEdit,
    ) -> None:
        """Restore local actor state to the last pose accepted by the generator."""
        key = (str(kind).lower(), int(index))
        accepted = self._live_committed_poses.get(key, edit.original)
        self._restore_edit_pose(key[0], key[1], accepted)
        if accepted == edit.original:
            self._edit_session.discard(key[0], key[1])
            self._live_committed_poses.pop(key, None)
        else:
            self._edit_session.record(
                kind=key[0],
                index=key[1],
                position=accepted.position,
                orientation=accepted.orientation,
                baseline_position=edit.original.position,
                baseline_orientation=edit.original.orientation,
                identity_aliases=edit.identity_aliases,
            )
        if key[0] == "target":
            self._target_edit_baseline = None

    def dirty_edit_count(self) -> int:
        """Return the number of transient edited entities."""
        return self._edit_session.dirty_count()

    def edited_keys(self) -> tuple[tuple[str, int], ...]:
        """Return edited entity keys in stable order."""
        return self._edit_session.edited_keys()

    def reset_selected_edit(self) -> bool:
        """Reset the currently selected gizmo target, if it has a transient edit."""
        renderer = getattr(self.visualizer, "renderer", None)
        getter = getattr(renderer, "get_active_transform_target", None)
        active = getter() if callable(getter) else None
        if not isinstance(active, dict):
            self._set_status("No selected interactive edit")
            return False
        return self.reset_edit(str(active.get("kind", "")), int(active.get("index", -1)))

    def reset_all_edits(self) -> bool:
        """Reset every transient edit back to the loaded scenario/frame pose."""
        edits = list(self._edit_session.edits())
        if not edits:
            self._set_status("No interactive edits to reset")
            return False
        if (self._active_edit_mode or self.edit_mode()) == _EDIT_MODE_LIVE:
            changed = False
            for edit in edits:
                if not self._send_live_pose(edit.kind, edit.index, edit.original):
                    continue
                changed = self._restore_edit_pose(edit.kind, edit.index, edit.original) or changed
                self._edit_session.discard(edit.kind, edit.index)
                self._live_committed_poses.pop((edit.kind, edit.index), None)
            if changed:
                self._target_edit_baseline = None
                self._set_status("All live session edits reset")
            self._sync_panel_controls()
            return changed
        changed = False
        reset_target = False
        for edit in edits:
            changed = self._restore_edit_pose(edit.kind, edit.index, edit.original) or changed
            reset_target = reset_target or edit.kind == "target"
            self._edit_session.discard(edit.kind, edit.index)
        if reset_target:
            self._target_edit_baseline = None
        if changed:
            self._request_solve("release")
        self._set_status("All interactive edits reset")
        self._sync_panel_controls()
        return changed

    def reset_edit(self, kind: str, index: int) -> bool:
        """Reset one transient edit back to its captured baseline pose."""
        edit = self._edit_session.get(kind, index)
        if edit is None:
            self._set_status("No interactive edit to reset")
            return False
        if (self._active_edit_mode or self.edit_mode()) == _EDIT_MODE_LIVE:
            if not self._send_live_pose(edit.kind, edit.index, edit.original):
                self._set_status(f"{self._edit_label(edit.kind, edit.index)} reset rejected")
                return False
            if not self._restore_edit_pose(edit.kind, edit.index, edit.original):
                self._sync_panel_controls()
                return False
            self._edit_session.discard(edit.kind, edit.index)
            self._live_committed_poses.pop((edit.kind, edit.index), None)
            if edit.kind == "target":
                self._target_edit_baseline = None
            self._set_status(f"{self._edit_label(edit.kind, edit.index)} reset in live session")
            self._sync_panel_controls()
            return True

        edit = self._edit_session.discard(kind, index)
        assert edit is not None
        if edit.kind == "target":
            self._target_edit_baseline = None
        if not self._restore_edit_pose(edit.kind, edit.index, edit.original):
            self._sync_panel_controls()
            return False
        label = self._edit_label(edit.kind, edit.index)
        self._set_status(f"{label} reset")
        self._request_solve("release")
        self._sync_panel_controls()
        return True

    def _ensure_worker_started(self) -> bool:
        """Start or reuse the persistent subprocess that runs Sionna solves."""
        proc = self._worker_process
        if proc is not None and proc.poll() is None:
            if not self._poll_timer.isActive():
                self._poll_timer.start()
            return True

        self._stop_worker()
        self._worker_queue = queue.Queue()
        self._stderr_tail = []
        self._worker_ready = False
        self._worker_busy = False
        self._pending_profile = None
        self._worker_temp_dir = tempfile.TemporaryDirectory(prefix="orchav_live_preview_")

        try:
            self._worker_process = self._launch_worker_process()
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Could not start live preview worker")
            self._set_status(f"Preview failed: {exc}")
            self._cleanup_worker_temp_dir()
            return False

        self._start_worker_readers(self._worker_process)
        self._poll_timer.start()
        self._send_init_request()
        return True

    def _launch_worker_process(self) -> subprocess.Popen[str]:
        """Spawn the preview worker with unbuffered JSON-line stdio."""
        command = source_bound_module_command(
            "visualizer.src.services.live_preview_worker",
            "--stdio",
            identity=self._source_identity,
            anchor_package="visualizer",
            python_executable=sys.executable,
        )
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def _start_worker_readers(self, proc: subprocess.Popen[str]) -> None:
        """Start daemon readers that bridge worker output into Qt polling."""
        if proc.stdout is not None:
            self._worker_stdout_thread = threading.Thread(
                target=self._read_worker_stdout,
                args=(proc.stdout,),
                name="orchav-preview-stdout",
                daemon=True,
            )
            self._worker_stdout_thread.start()
        if proc.stderr is not None:
            self._worker_stderr_thread = threading.Thread(
                target=self._read_worker_stderr,
                args=(proc.stderr,),
                name="orchav-preview-stderr",
                daemon=True,
            )
            self._worker_stderr_thread.start()

    def _read_worker_stdout(self, stream: Any) -> None:
        """Decode JSON-line worker messages into the thread-safe queue."""
        try:
            for line in stream:
                text = str(line).strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    logger.debug("Ignoring non-protocol preview worker output: %s", text)
                    continue
                if isinstance(message, dict):
                    self._worker_queue.put(message)
        finally:
            self._worker_queue.put({"type": "stdout_closed"})

    def _read_worker_stderr(self, stream: Any) -> None:
        """Keep a bounded stderr tail for failure status messages."""
        for line in stream:
            text = str(line).strip()
            if not text:
                continue
            self._stderr_tail.append(text)
            if len(self._stderr_tail) > _WORKER_STDERR_TAIL_LINES:
                del self._stderr_tail[:-_WORKER_STDERR_TAIL_LINES]

    def _send_worker_message(self, message: dict[str, Any]) -> bool:
        """Write one protocol message to the worker, failing closed on pipes."""
        proc = self._worker_process
        if proc is None or proc.poll() is not None or proc.stdin is None:
            self._handle_worker_failure("Preview worker is not running")
            return False
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._handle_worker_failure(f"Preview worker pipe closed: {exc}")
            return False
        return True

    def _send_init_request(self) -> None:
        """Warm the worker runtime for the current scenario and step."""
        step = int(getattr(self.visualizer, "animation_step", 0) or 0)
        request = self._base_worker_request(step, quality="release")
        self._send_worker_message({"command": "init", "request": request})

    def _stop_worker(self) -> None:
        """Shut down the worker gracefully, then escalate to termination."""
        self._poll_timer.stop()
        proc = self._worker_process
        self._worker_process = None
        self._worker_ready = False
        self._worker_busy = False
        self._pending_profile = None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    proc.stdin.flush()
                    proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                proc.wait(timeout=_WORKER_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=_WORKER_SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._cleanup_worker_temp_dir()

    def _cleanup_worker_temp_dir(self) -> None:
        """Remove temporary pickle outputs owned by the current worker."""
        tmp = self._worker_temp_dir
        self._worker_temp_dir = None
        if tmp is not None:
            tmp.cleanup()

    def _schedule_drag_preview_request(self) -> None:
        """Debounce continuous drag updates into a lower-quality preview solve."""
        if not self._drag_timer.isActive():
            self._drag_timer.start(self._drag_interval_ms)

    def _flush_drag_preview_request(self) -> None:
        """Convert the pending drag debounce timer into a solve request."""
        self._request_solve("drag")

    def _request_solve(self, profile: str) -> None:
        """Queue or send a solve, preserving release quality over drag quality."""
        if not self._enabled:
            return
        profile = "release" if profile == "release" else "drag"
        if not self._ensure_worker_started():
            return
        if not self._worker_ready:
            self._pending_profile = self._merge_pending_profile(profile)
            return
        if self._worker_busy:
            self._pending_profile = self._merge_pending_profile(profile)
            return
        self._send_solve_request(profile)

    def _send_solve_request(self, profile: str) -> None:
        """Build and send a worker solve request for the current entity arrays."""
        step = int(getattr(self.visualizer, "animation_step", 0) or 0)
        tx_positions = self._position_array("tx")
        rx_positions = self._position_array("rx")
        if tx_positions.size == 0 or rx_positions.size == 0:
            self._set_status("Preview needs TX/RX positions")
            return
        target_positions = self._target_position_array()
        target_orientations = self._target_orientation_array(desired_count=len(target_positions))

        self._sequence += 1
        sequence = self._sequence
        self._requested_sequence = sequence
        output_path = self._worker_output_path(sequence)
        request = self._base_worker_request(step, quality=profile)
        request.update(
            encode_live_preview_arrays(
                tx_positions,
                rx_positions,
                target_positions,
                target_orientations,
            )
        )
        request.update({"sequence": int(sequence), "output_path": str(output_path)})
        if self._send_worker_message({"command": "solve", "request": request}):
            self._worker_busy = True
            self._solving = True
            label = "final" if profile == "release" else "live"
            self._set_status(f"Preview solving {label} with panel settings...")

    def _worker_output_path(self, sequence: int) -> Path:
        """Return the pickle path where the worker should store a result."""
        if self._worker_temp_dir is None:
            self._worker_temp_dir = tempfile.TemporaryDirectory(prefix="orchav_live_preview_")
        return Path(self._worker_temp_dir.name) / f"preview_{int(sequence)}.pkl"

    def _merge_pending_profile(self, profile: str) -> str:
        """Prefer a pending release solve over any drag-quality solve."""
        if profile == "release" or self._pending_profile == "release":
            return "release"
        return "drag"

    def _poll_worker_messages(self) -> None:
        """Drain worker messages and convert unexpected exits into UI errors."""
        processed = False
        while True:
            try:
                message = self._worker_queue.get_nowait()
            except queue.Empty:
                break
            processed = True
            self._handle_worker_message(message)

        proc = self._worker_process
        if proc is not None and proc.poll() is not None:
            code = proc.returncode
            if self._enabled:
                self._handle_worker_failure(f"Preview worker exited with code {code}")
            elif not processed:
                self._poll_timer.stop()

    def _handle_worker_message(self, message: dict[str, Any]) -> None:
        """Dispatch one protocol message from the preview worker."""
        message_type = str(message.get("type", ""))
        if message_type == "ready":
            try:
                child_identity = SourceIdentity.from_mapping(message.get("source_identity"))
            except (TypeError, ValueError) as exc:
                self._handle_worker_failure(f"Preview worker source identity is invalid: {exc}")
                return
            if not self._source_identity.matches(child_identity):
                self._handle_worker_failure(
                    "Preview worker source identity does not match the running visualizer"
                )
                return
            self._worker_ready = True
            self._set_status("Preview ready")
            self._send_pending_if_available()
            return
        if message_type == "result":
            if not self._worker_ready:
                self._handle_worker_failure(
                    "Preview worker sent a result before source identity was verified"
                )
                return
            self._handle_worker_result(message)
            return
        if message_type == "error":
            detail = str(message.get("error", "Preview worker error"))
            self._last_error = detail
            self._worker_busy = False
            self._solving = False
            self._set_status(f"Preview failed: {detail}")
            self._send_pending_if_available()
            return
        if message_type == "bye":
            self._poll_timer.stop()

    def _handle_worker_result(self, message: dict[str, Any]) -> None:
        """Apply only the newest worker result when no newer solve is pending."""
        self._worker_busy = False
        self._solving = False
        sequence = self._coerce_int(message.get("sequence"), default=-1)
        status = str(message.get("status", "ok"))
        if status != "ok":
            detail = str(message.get("error", "Preview worker failed"))
            self._last_error = detail
            self._set_status(f"Preview failed: {detail}")
            self._send_pending_if_available()
            return

        output = message.get("output_path")
        if (
            sequence == self._requested_sequence
            and self._pending_profile is None
            and isinstance(output, str)
        ):
            expected_output = self._worker_output_path(sequence)
            try:
                reported_output = Path(output).resolve()
                resolved_expected_output = expected_output.resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                self._handle_worker_failure(
                    f"Preview worker reported an invalid result path: {exc}"
                )
                return
            if reported_output != resolved_expected_output:
                self._handle_worker_failure("Preview worker reported an unexpected result path")
                return
            try:
                with resolved_expected_output.open("rb") as handle:
                    frame = pickle.load(handle)
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("Could not load live preview worker result")
                self._set_status(f"Preview failed: {exc}")
            else:
                if isinstance(frame, StandardMPCFrame):
                    self._apply_preview_frame(frame, sequence)
                else:
                    self._set_status("Preview failed: invalid worker frame")
        self._send_pending_if_available()

    def _send_pending_if_available(self) -> None:
        """Send a coalesced pending solve once the worker is idle and ready."""
        if not self._enabled or not self._worker_ready or self._worker_busy:
            return
        profile = self._pending_profile
        self._pending_profile = None
        if profile is not None:
            self._send_solve_request(profile)

    def _handle_worker_failure(self, message: str) -> None:
        """Record a worker failure, expose stderr context, and stop resources."""
        detail = self._stderr_summary()
        if detail:
            message = f"{message}: {detail}"
        self._last_error = message
        logger.warning("Live preview worker failure: %s", message)
        self._worker_ready = False
        self._worker_busy = False
        self._solving = False
        self._set_status(f"Preview failed: {message}")
        self._stop_worker()

    def _stderr_summary(self) -> str:
        """Return a short stderr excerpt suitable for status text."""
        if not self._stderr_tail:
            return ""
        text = "\n".join(self._stderr_tail[-5:])
        return text[-800:]

    def _base_worker_request(
        self,
        step: int,
        *,
        quality: str,
    ) -> dict[str, Any]:
        """Build request fields shared by init and solve commands."""
        scenario_root = self._scenario_root()
        project_root = getattr(self.visualizer, "current_project_root", None)
        self._runtime_scenario_root = scenario_root
        return {
            "step": int(step),
            "scenario_root": str(scenario_root),
            "project_root": str(project_root) if project_root else None,
            "solver_settings": self._solver_settings(str(quality)),
            "quality": str(quality),
        }

    def _solver_settings(self, quality: str) -> dict[str, Any]:
        """Return shared panel settings for release or scaled drag preview."""
        service = getattr(
            self.visualizer,
            "raytracing_settings_service",
            self._fallback_raytracing_settings,
        )
        method_name = "release_settings" if quality == "release" else "drag_settings"
        getter = getattr(service, method_name, None)
        if not callable(getter):
            getter = getattr(self._fallback_raytracing_settings, method_name)
        return dict(getter())

    def _scenario_root(self) -> Path:
        """Resolve the scenario root from current app state or scenario path."""
        scenario = getattr(self.visualizer, "scenario", None) or getattr(
            self.visualizer,
            "scenario_config",
            None,
        )
        root = getattr(scenario, "root", None)
        if root is not None:
            return Path(root)
        current = getattr(self.visualizer, "current_scenario_path", None)
        if current:
            current_path = Path(current)
            if current_path.is_absolute():
                return current_path if current_path.is_dir() else current_path.parent
            project_root = Path(getattr(self.visualizer, "current_project_root", Path.cwd()))
            resolved = project_root / current_path
            return resolved if resolved.is_dir() else resolved.parent
        raise RuntimeError("No scenario is loaded for live preview")

    def _apply_preview_frame(self, frame: StandardMPCFrame, sequence: int) -> None:
        """Project a preview frame into the visual payload and invalidate its cache."""
        frame = self._apply_interactive_target_pose_overrides(frame)
        step = int(frame.frame_index)
        request = visual_frame_read_request(include_sensing=frame.sensing is not None)
        payload = standard_frame_to_visual_frame(
            frame,
            request=request,
            points_dtype=getattr(
                getattr(self.visualizer, "mpc_core", None),
                "canon_points_dtype",
                np.dtype(np.float32),
            ),
        )
        source = dict(frame.provenance or {})
        payload["source_provider"] = str(source.get("provider", "live_preview"))
        payload["preview"] = bool(source.get("preview", True))
        self.visualizer._live_preview_frame = payload
        self.visualizer._live_preview_step = step
        self.visualizer._live_preview_sequence = int(sequence)
        cache_service = getattr(self.visualizer, "cache_service", None)
        invalidate_step = getattr(cache_service, "invalidate_canonical_step", None)
        if callable(invalidate_step):
            invalidate_step(step, reason="live_preview")
        else:
            invalidate_visualizer_cache(
                self.visualizer,
                CacheInvalidationScope.MPC_RENDER_SETTINGS,
                reason="live_preview",
            )
        self.visualizer.force_update_next_frame = True
        schedule = getattr(self.visualizer, "schedule_update", None)
        if callable(schedule):
            schedule()
        self._set_status("Preview applied")

    def _apply_interactive_target_pose_overrides(
        self,
        frame: StandardMPCFrame,
    ) -> StandardMPCFrame:
        """Return a frame whose target state includes dirty interactive edits."""
        target_edits = [edit for edit in self._edit_session.edits() if edit.kind == "target"]
        if not target_edits:
            return frame

        metadata = [dict(item) for item in frame.targets_metadata]
        target_positions = np.array(frame.target_positions_m, dtype=np.float64, copy=True)
        changed = False

        for edit in target_edits:
            aliases = set(edit.identity_aliases)
            if not aliases:
                aliases = self._canonical_target_identity_aliases(int(edit.index))
            metadata_index = self._target_metadata_index(metadata, aliases)
            if metadata_index is None:
                continue
            entry = metadata[metadata_index]
            position = [float(value) for value in edit.current.position[:3]]
            entry["current_position"] = position
            entry["position_valid"] = True
            if edit.current.orientation is not None:
                entry["orientation"] = [float(value) for value in edit.current.orientation[:3]]
            target_positions[metadata_index, :3] = np.asarray(position, dtype=np.float64)
            changed = True

        if not changed:
            return frame
        return replace(
            frame,
            target_positions_m=target_positions,
            targets_metadata=tuple(metadata),
        )

    def _clear_preview_frame(self) -> None:
        """Remove the active preview frame and refresh normal frame rendering."""
        self.visualizer._live_preview_frame = None
        self.visualizer._live_preview_step = None
        self.visualizer._live_preview_sequence = None
        cache_service = getattr(self.visualizer, "cache_service", None)
        if cache_service is not None:
            cache_service.invalidate(
                CacheInvalidationScope.MPC_RENDER_SETTINGS,
                reason="live_preview_disabled",
            )
        self.visualizer.force_update_next_frame = True
        schedule = getattr(self.visualizer, "schedule_update", None)
        if callable(schedule):
            schedule()

    def _apply_edited_position(self, kind: str, index: int, position: np.ndarray) -> bool:
        """Update semantic TX/RX state and publish complete node snapshots."""
        if index < 0:
            return False
        tx_positions = self._position_array("tx")
        rx_positions = self._position_array("rx")
        source_positions = tx_positions if kind == "tx" else rx_positions
        baseline_position = self._position_baseline(source_positions, kind, index)
        if kind == "tx":
            tx_positions = self._replace_position(tx_positions, kind, index, position)
        else:
            rx_positions = self._replace_position(rx_positions, kind, index, position)
        if not self._publish_node_positions(tx_positions, rx_positions):
            return False
        self._edit_session.record(
            kind=kind,
            index=index,
            position=position,
            baseline_position=baseline_position,
        )
        return True

    def _apply_edited_target_pose(
        self,
        index: int,
        object_position: np.ndarray,
        event: dict[str, Any],
        *,
        phase: str = "changed",
        selected: bool = False,
    ) -> bool:
        """Update transient target pose arrays from a validated renderer edit."""
        if index < 0:
            return False

        positions = self._target_position_array()
        orientations = self._target_orientation_array(
            desired_count=max(index + 1, len(positions)),
        )
        positions = self._ensure_target_position_count(positions, index + 1)
        orientations = self._ensure_target_orientation_count(orientations, index + 1)

        baseline = self._target_edit_baseline
        if selected or not baseline or int(baseline.get("index", -1)) != index:
            baseline = self._capture_target_edit_baseline(
                index,
                object_position,
                positions,
                orientations,
                self._event_transform_matrix(event),
            )
        if selected:
            return True

        transform = self._event_transform_matrix(event)
        rotation = (
            self._orthonormalize_rotation(transform[:3, :3]) if transform is not None else None
        )
        target_orientation = (
            self._rotation_matrix_to_sionna_orientation(rotation) if rotation is not None else None
        )
        if target_orientation is None:
            target_orientation = np.asarray(
                baseline["target_orientation"],
                dtype=np.float64,
            ).reshape(3)
        target_position = np.asarray(baseline["target_position"], dtype=np.float64).reshape(3)
        rotation_changed = self._target_rotation_changed(baseline, transform)
        if rotation_changed:
            target_position = np.asarray(positions[index, :3], dtype=np.float64).reshape(3)
        else:
            delta = np.asarray(object_position, dtype=np.float64).reshape(3) - np.asarray(
                baseline["object_position"],
                dtype=np.float64,
            ).reshape(3)
            target_position = target_position + delta

        positions[index, :] = target_position
        orientations[index, :] = target_orientation
        self.visualizer.current_target_positions = positions
        self.visualizer.current_target_orientations = orientations
        self._edit_session.record(
            kind="target",
            index=index,
            position=target_position,
            orientation=target_orientation,
            baseline_position=baseline["target_position"],
            baseline_orientation=baseline["target_orientation"],
            identity_aliases=self._canonical_target_identity_aliases(index),
        )
        self._sync_target_entry(index, target_position, target_orientation)
        self._publish_target_entry_snapshot(index)
        if phase == "committed":
            self._target_edit_baseline = {
                "index": int(index),
                "object_position": np.asarray(object_position, dtype=np.float64).reshape(3).copy(),
                "target_position": np.asarray(target_position, dtype=np.float64).reshape(3).copy(),
                "target_orientation": np.asarray(target_orientation, dtype=np.float64)
                .reshape(3)
                .copy(),
                "object_rotation": (
                    np.asarray(rotation, dtype=np.float64).copy()
                    if rotation is not None
                    else build_sionna_rotation_matrix(*target_orientation)
                ),
            }
        return True

    def _capture_target_edit_baseline(
        self,
        index: int,
        object_position: np.ndarray,
        positions: np.ndarray,
        orientations: np.ndarray,
        transform: np.ndarray | None,
    ) -> dict[str, Any]:
        """Remember the semantic target pose at gizmo selection time."""
        target_position = (
            positions[index, :3]
            if index < len(positions)
            else np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        )
        target_orientation = (
            orientations[index, :3]
            if index < len(orientations)
            else np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        )
        baseline = {
            "index": int(index),
            "object_position": np.asarray(object_position, dtype=np.float64).reshape(3).copy(),
            "target_position": np.asarray(target_position, dtype=np.float64).reshape(3).copy(),
            "target_orientation": np.asarray(
                target_orientation,
                dtype=np.float64,
            )
            .reshape(3)
            .copy(),
        }
        rotation = (
            self._orthonormalize_rotation(transform[:3, :3]) if transform is not None else None
        )
        if rotation is None:
            rotation = build_sionna_rotation_matrix(*baseline["target_orientation"])
        baseline["object_rotation"] = rotation
        self._target_edit_baseline = baseline
        return baseline

    def _position_baseline(self, positions: np.ndarray, kind: str, index: int) -> np.ndarray:
        """Return the original TX/RX position for edit-session reset."""
        if 0 <= index < len(positions):
            return np.asarray(positions[index, :3], dtype=np.float64).copy()
        candidate = self._semantic_node_position(kind, index)
        if candidate is not None:
            return candidate
        return np.zeros(3, dtype=np.float64)

    def _target_rotation_changed(
        self,
        baseline: dict[str, Any],
        transform: np.ndarray | None,
    ) -> bool:
        """Return True when a target gizmo edit is primarily rotational."""
        if transform is None:
            return False
        current = self._orthonormalize_rotation(transform[:3, :3])
        baseline_rotation = baseline.get("object_rotation")
        if current is None or baseline_rotation is None:
            return False
        return not np.allclose(
            np.asarray(current, dtype=np.float64),
            np.asarray(baseline_rotation, dtype=np.float64),
            atol=1e-6,
        )

    def _target_position_array(self) -> np.ndarray:
        """Return current target positions as a copied ``(N, 3)`` array."""
        values = self._target_entry_values("position")
        if values:
            return self._coerce_array(values)
        view_model = getattr(self.visualizer, "current_view_model", None)
        if view_model is not None and getattr(view_model, "target_positions", None) is not None:
            return self._coerce_array(getattr(view_model, "target_positions"))
        return self._coerce_array(getattr(self.visualizer, "current_target_positions", None))

    def _target_orientation_array(self, *, desired_count: int = 0) -> np.ndarray:
        """Return current target orientations as a copied ``(N, 3)`` array."""
        values = self._target_entry_values("orientation")
        if values:
            arr = self._coerce_array(values)
        else:
            view_model = getattr(self.visualizer, "current_view_model", None)
            if (
                view_model is not None
                and getattr(view_model, "target_orientations", None) is not None
            ):
                arr = self._coerce_array(getattr(view_model, "target_orientations"))
            else:
                arr = self._coerce_array(
                    getattr(self.visualizer, "current_target_orientations", None)
                )
        if desired_count > len(arr):
            arr = self._ensure_target_orientation_count(arr, desired_count)
        return arr

    def _target_entry_values(self, key: str) -> list[Any]:
        """Read one pose field from visible target entries."""
        values: list[Any] = []
        for entry in getattr(self.visualizer, "target_entries", []) or []:
            if key == "orientation":
                value = entry.get("orientation")
                if value is None:
                    value = entry.get("orientation_radians")
            else:
                value = entry.get(key)
            if value is None:
                values.append([0.0, 0.0, 0.0])
            else:
                values.append(value)
        return values

    def _canonical_target_identity_aliases(self, index: int) -> set[str]:
        """Return stable aliases for one persistent target-entry index."""
        entries = getattr(self.visualizer, "target_entries", []) or []
        if index < 0 or index >= len(entries):
            return set()
        return self._target_identity_aliases(entries[index])

    @staticmethod
    def _target_identity_aliases(target: Any) -> set[str]:
        """Return normalized stable/name aliases from a target record."""
        if not hasattr(target, "get"):
            return set()
        aliases: set[str] = set()
        for key in ("stable_target_id", "target_name", "node_name", "name", "id"):
            value = target.get(key)
            if value is not None and str(value).strip():
                aliases.add(normalize_token(value))
        return aliases

    def _target_metadata_index(
        self,
        metadata: list[Any],
        aliases: set[str],
    ) -> int | None:
        """Resolve one canonical target identity to a unique frame-array row."""
        if not aliases:
            return None
        matches = [
            index
            for index, entry in enumerate(metadata)
            if aliases.intersection(self._target_identity_aliases(entry))
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning(
                "Live preview target identity %s matched multiple frame rows; skipping override",
                sorted(aliases),
            )
        return None

    def _coerce_array(self, values: Any) -> np.ndarray:
        """Coerce arbitrary pose values into a copied ``(N, 3)`` array."""
        try:
            arr = np.asarray([] if values is None else values, dtype=np.float64)
        except (TypeError, ValueError):
            return np.empty((0, 3), dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if arr.ndim == 1 and arr.size >= 3:
            arr = arr.reshape((1, arr.size))
        if arr.ndim != 2 or arr.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)
        arr = arr[:, :3]
        if not np.all(np.isfinite(arr)):
            return np.empty((0, 3), dtype=np.float64)
        return np.array(arr, dtype=np.float64, copy=True)

    def _ensure_target_position_count(
        self, positions: np.ndarray, desired_count: int
    ) -> np.ndarray:
        """Expand target position rows from entries when an edited index is missing."""
        if len(positions) >= desired_count:
            return np.array(positions, dtype=np.float64, copy=True)
        expanded = np.zeros((desired_count, 3), dtype=np.float64)
        if len(positions):
            expanded[: len(positions), :] = positions[:, :3]
        entries = getattr(self.visualizer, "target_entries", []) or []
        for index in range(len(positions), min(desired_count, len(entries))):
            entry_pos = self._coerce_array(entries[index].get("position"))
            if len(entry_pos):
                expanded[index, :] = entry_pos[0, :3]
        return expanded

    def _ensure_target_orientation_count(
        self,
        orientations: np.ndarray,
        desired_count: int,
    ) -> np.ndarray:
        """Expand target orientation rows from entries when an edited index is missing."""
        if len(orientations) >= desired_count:
            return np.array(orientations, dtype=np.float64, copy=True)
        expanded = np.zeros((desired_count, 3), dtype=np.float64)
        if len(orientations):
            expanded[: len(orientations), :] = orientations[:, :3]
        entries = getattr(self.visualizer, "target_entries", []) or []
        for index in range(len(orientations), min(desired_count, len(entries))):
            value = entries[index].get("orientation")
            if value is None:
                value = entries[index].get("orientation_radians")
            entry_orientation = self._coerce_array(value)
            if len(entry_orientation):
                expanded[index, :] = entry_orientation[0, :3]
        return expanded

    def _sync_target_entry(
        self,
        index: int,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> None:
        """Mirror a transient target edit into the current target entry."""
        entries = getattr(self.visualizer, "target_entries", []) or []
        if index < 0 or index >= len(entries):
            return
        entry = entries[index]
        position_list = [float(value) for value in np.asarray(position, dtype=np.float64)[:3]]
        orientation_rad, orientation_deg = orientation_metadata(
            [float(value) for value in np.asarray(orientation, dtype=np.float64)[:3]]
        )
        entry["position"] = position_list
        entry["orientation"] = orientation_rad
        entry["orientation_radians"] = orientation_rad
        entry["orientation_degrees"] = orientation_deg
        entry["_target_position"] = position_list
        entry["_rotation_matrix"] = build_sionna_rotation_matrix(*orientation_rad)

    def _publish_target_entry_snapshot(self, index: int) -> bool:
        """Republish one edited target through the application target owner."""
        entries = getattr(self.visualizer, "target_entries", []) or []
        if index < 0 or index >= len(entries):
            return False
        target_service = getattr(self.visualizer, "target_service", None)
        sync_snapshot = getattr(target_service, "sync_target_entry_snapshot", None)
        if not callable(sync_snapshot):
            logger.error(
                "Live preview cannot publish target %d: TargetService is unavailable", index
            )
            return False
        return bool(sync_snapshot(entries[index]))

    def _publish_node_positions(
        self,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
    ) -> bool:
        """Publish semantic TX/RX arrays through the application node owner."""
        node_service = getattr(self.visualizer, "node_service", None)
        update_positions = getattr(node_service, "update_tx_rx_positions", None)
        if not callable(update_positions):
            logger.error("Live preview cannot publish TX/RX positions: NodeService is unavailable")
            return False
        update_positions(tx_positions, rx_positions)
        return True

    def _restore_edit_pose(self, kind: str, index: int, pose: InteractivePose) -> bool:
        """Restore one edited entity to its original pose."""
        kind = str(kind).lower()
        position = np.asarray(pose.position, dtype=np.float64)
        if kind in {"tx", "rx"}:
            tx_positions = self._position_array("tx")
            rx_positions = self._position_array("rx")
            if kind == "tx":
                tx_positions = self._replace_position(tx_positions, kind, index, position)
            else:
                rx_positions = self._replace_position(rx_positions, kind, index, position)
            return self._publish_node_positions(tx_positions, rx_positions)

        if kind != "target":
            return False
        positions = self._ensure_target_position_count(self._target_position_array(), index + 1)
        orientations = self._ensure_target_orientation_count(
            self._target_orientation_array(desired_count=index + 1),
            index + 1,
        )
        orientation = np.asarray(pose.orientation or (0.0, 0.0, 0.0), dtype=np.float64)
        positions[index, :] = position[:3]
        orientations[index, :] = orientation[:3]
        self.visualizer.current_target_positions = positions
        self.visualizer.current_target_orientations = orientations
        self._sync_target_entry(index, position, orientation)
        self._publish_target_entry_snapshot(index)
        return True

    def _edit_label(self, kind: str, index: int) -> str:
        """Return a short label for an edited entity."""
        kind = str(kind).lower()
        if kind == "target":
            return self._target_status_label(index)
        return f"{kind.upper()}{index + 1}"

    def _target_status_label(self, index: int) -> str:
        """Return a readable target label for status messages."""
        entries = getattr(self.visualizer, "target_entries", []) or []
        if 0 <= index < len(entries):
            entry = entries[index]
            return str(
                entry.get("display_name")
                or entry.get("target_name")
                or entry.get("name")
                or f"Target{index + 1}"
            )
        return f"Target{index + 1}"

    @staticmethod
    def _event_transform_matrix(event: dict[str, Any]) -> np.ndarray | None:
        """Decode a renderer transform matrix from one event payload."""
        try:
            matrix = np.asarray(event.get("transform"), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return None
        return matrix

    @classmethod
    def _orientation_from_transform(cls, matrix: np.ndarray | None) -> np.ndarray | None:
        """Extract Sionna yaw/pitch/roll radians from a renderer transform."""
        return None if matrix is None else sionna_orientation_from_transform(matrix)

    @staticmethod
    def _orthonormalize_rotation(values: np.ndarray) -> np.ndarray | None:
        """Return the closest proper rotation matrix, discarding scale."""
        return closest_proper_rotation(values)

    @staticmethod
    def _rotation_matrix_to_sionna_orientation(rotation: np.ndarray) -> np.ndarray:
        """Convert a ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` matrix to radians."""
        orientation = sionna_orientation_from_rotation_matrix(rotation)
        if orientation is None:
            return np.zeros(3, dtype=np.float64)
        return orientation

    def _position_array(self, kind: str) -> np.ndarray:
        """Return current TX or RX positions as a copied ``(N, 3)`` array."""
        attr = "current_tx_positions" if kind == "tx" else "current_rx_positions"
        values = getattr(self.visualizer, attr, None)
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            arr = np.empty((0, 3), dtype=np.float64)
        if arr.size == 0:
            arr = np.empty((0, 3), dtype=np.float64)
        elif arr.ndim == 1 and arr.size == 3:
            arr = arr.reshape((1, 3))
        elif arr.ndim == 2 and arr.shape[1] >= 3:
            arr = arr[:, :3]
        else:
            arr = np.empty((0, 3), dtype=np.float64)
        return np.array(arr, dtype=np.float64, copy=True)

    def _replace_position(
        self,
        positions: np.ndarray,
        kind: str,
        index: int,
        position: np.ndarray,
    ) -> np.ndarray:
        """Replace one node position, expanding from semantic app state if needed."""
        desired_count = max(index + 1, len(positions), self._known_node_count(kind))
        if len(positions) < desired_count:
            expanded = np.zeros((desired_count, 3), dtype=np.float64)
            if len(positions):
                expanded[: len(positions), :] = positions[:, :3]
            for i in range(len(positions), desired_count):
                candidate = self._semantic_node_position(kind, i)
                if candidate is not None:
                    expanded[i, :] = candidate
            positions = expanded
        positions[index, :] = np.asarray(position, dtype=np.float64).reshape(-1)[:3]
        return positions

    def _semantic_node_position(self, kind: str, index: int) -> np.ndarray | None:
        """Return a TX/RX position from application-owned entries or handles."""
        kind_norm = str(kind).lower()
        entries = getattr(self.visualizer, f"{kind_norm}_entries", []) or []
        if 0 <= index < len(entries):
            try:
                candidate = np.asarray(entries[index].get("position"), dtype=np.float64).reshape(-1)
            except (AttributeError, TypeError, ValueError):
                candidate = np.empty(0, dtype=np.float64)
            if candidate.size >= 3 and np.all(np.isfinite(candidate[:3])):
                return candidate[:3].copy()

        markers = getattr(self.visualizer, f"{kind_norm}_markers", []) or []
        if 0 <= index < len(markers) and isinstance(markers[index], RenderObjectState):
            candidate = np.asarray(render_state_center(markers[index]), dtype=np.float64).reshape(
                -1
            )
            if candidate.size >= 3 and np.all(np.isfinite(candidate[:3])):
                return candidate[:3].copy()
        return None

    def _known_node_count(self, kind: str) -> int:
        """Return the app's known TX/RX count, treating invalid values as zero."""
        raw = getattr(self.visualizer, f"num_{kind}", None)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def _set_status(self, message: str) -> None:
        """Mirror preview status to app state, nodes panel, and status bar."""
        text = str(message)
        dirty_count = self._edit_session.dirty_count()
        if dirty_count and not text.endswith("reset") and not text.startswith("Preview off"):
            suffix = "edit" if dirty_count == 1 else "edits"
            text = f"{text} ({dirty_count} {suffix})"
        self.visualizer._live_preview_status = text
        panel = getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get("nodes")
        update = getattr(panel, "update_live_preview_status", None)
        if callable(update):
            update(text)
        refresh = getattr(panel, "refresh_live_preview_state", None)
        if callable(refresh):
            refresh()
        set_status = getattr(self.visualizer, "_set_status_message", None)
        if callable(set_status):
            set_status(text, 3000)

    def _sync_panel_controls(self) -> None:
        """Ask the nodes panel to refresh live-preview controls when present."""
        panel = getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get("nodes")
        refresh = getattr(panel, "refresh_live_preview_state", None)
        if callable(refresh):
            refresh()

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        """Coerce a protocol integer while keeping caller-selected defaults."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)
