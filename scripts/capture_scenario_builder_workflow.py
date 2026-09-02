"""Run and capture the display-backed Scenario Builder release workflow.

This is an opt-in visual smoke tool.  It drives the real Qt workspace, pygfx
authoring viewport, compiler, persistence boundary, generator subprocess, and
generated-result preview.  Each checkpoint writes both the complete workspace
and the renderer-only viewport so UI state and geometry can be reviewed
independently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QObject, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog

from shared.scenarios.actors import (
    ActorRole,
    CircularMobilitySpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    LookAtOrientationSpec,
    SpinOrientationSpec,
    StationaryMobilitySpec,
    WaypointMobilitySpec,
)
from visualizer.src.app.lifecycle import shutdown_visualizer
from visualizer.src.authoring.domain import (
    AuthoringGroup,
    SceneReference,
    TimelineSettings,
)
from visualizer.src.authoring.feature import SCENARIO_BUILDER_ENV
from visualizer.src.authoring.generation import GenerationState
from visualizer.src.authoring.interaction import InteractionSession
from visualizer.src.authoring.mode_controller import WorkspaceMode
from visualizer.src.authoring.persistence import save_document
from visualizer.src.authoring.viewport_port import (
    AuthoringTool,
    OverlayVisibility,
    stable_renderer_id,
)
from visualizer.visualizer import OrchavVisualizer, ProgressReporter

CaptureSink = Callable[[dict[str, Any]], None]
StageSink = Callable[[str, str, Mapping[str, Any] | None], None]
_WORK_MARKER_FILENAME = ".orchav-capture-owned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the real pygfx Scenario Builder workflow and capture "
            "reviewable visual evidence."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New or empty directory for captures, manifest, scenario, and frames.",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Capture authoring states without launching the generator or playback.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Maximum wait for generation and generated-result playback.",
    )
    return parser.parse_args()


def _prepare_output_directory(value: str | Path) -> tuple[Path, Path]:
    output_dir = Path(value).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Capture output directory is not empty: {output_dir}. "
            "Choose a new directory so prior evidence is never overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = output_dir / "captures"
    capture_dir.mkdir()
    return output_dir, capture_dir


def _create_owned_work_scenario(
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, Path, str]:
    """Create a marker-owned scenario root directly beneath checkout tmp."""
    tmp_root = (repo_root / "tmp").resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="orchav-scenario-builder-", dir=tmp_root)).resolve()
    token = uuid4().hex
    (work_root / _WORK_MARKER_FILENAME).write_text(token, encoding="ascii")
    scenario_dir = work_root / "scenario"
    scenario_dir.mkdir()
    return work_root, scenario_dir, token


def _remove_owned_work_directory(
    work_root: Path,
    token: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Remove only a direct checkout-tmp child carrying the expected token."""
    resolved = work_root.resolve()
    tmp_root = (repo_root / "tmp").resolve()
    if resolved.parent != tmp_root:
        raise RuntimeError(f"Refusing to remove work directory outside {tmp_root}: {resolved}")
    marker = resolved / _WORK_MARKER_FILENAME
    if not marker.is_file() or marker.read_text(encoding="ascii") != token:
        raise RuntimeError(f"Refusing to remove unowned work directory: {resolved}")
    shutil.rmtree(resolved)


def _copy_relocatable_scenario_evidence(
    scenario_dir: Path,
    output_dir: Path,
) -> dict[str, str | None]:
    """Copy durable scenario evidence without retaining checkout-local paths."""
    source_yaml = scenario_dir / "scenario.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Saved scenario is missing: {source_yaml}")
    evidence_dir = output_dir / "scenario"
    evidence_dir.mkdir(exist_ok=True)
    shutil.copy2(source_yaml, evidence_dir / "scenario.yaml")
    frames_source = scenario_dir / "frames"
    frames_path: str | None = None
    if frames_source.is_dir():
        shutil.copytree(frames_source, evidence_dir / "frames", dirs_exist_ok=True)
        frames_path = "scenario/frames"
    return {
        "scenario_yaml": "scenario/scenario.yaml",
        "frames_directory": frames_path,
    }


class _UnexpectedModalGuard(QObject):
    """Reject and report any modal dialog raised by unattended capture."""

    def __init__(self, sink: Callable[[dict[str, str]], None]) -> None:
        super().__init__()
        self._sink = sink
        self._seen: dict[str, str] | None = None
        self._pending_dialog: QDialog | None = None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Show or not isinstance(watched, QDialog):
            return False
        if self._seen is not None or self._pending_dialog is not None:
            return False
        self._pending_dialog = watched
        QTimer.singleShot(0, self._capture_pending_dialog)
        return False

    def _capture_pending_dialog(self) -> None:
        """Inspect and reject a dialog after native show-event delivery completes."""

        watched = self._pending_dialog
        self._pending_dialog = None
        if watched is None:
            return
        text_parts: list[str] = []
        for accessor_name in ("text", "informativeText", "detailedText"):
            accessor = getattr(watched, accessor_name, None)
            if callable(accessor):
                value = str(accessor()).strip()
                if value:
                    text_parts.append(value)
        self._seen = {
            "title": str(watched.windowTitle()),
            "text": "\n".join(text_parts),
        }
        self._sink(dict(self._seen))
        reject = getattr(watched, "reject", None)
        if callable(reject):
            reject()
        else:
            watched.close()

    def raise_if_seen(self) -> None:
        """Fail the harness after Qt has safely returned from event delivery."""
        if self._seen is None:
            return
        title = self._seen["title"] or "<untitled>"
        detail = self._seen["text"] or "<no dialog text>"
        raise RuntimeError(f"Unexpected modal dialog {title!r}: {detail}")


_ACTIVE_MODAL_GUARD: _UnexpectedModalGuard | None = None


def _pump_events(app: QApplication, turns: int = 4) -> None:
    for _ in range(max(int(turns), 1)):
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if _ACTIVE_MODAL_GUARD is not None:
            _ACTIVE_MODAL_GUARD.raise_if_seen()


def _wait_until(
    app: QApplication,
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    label: str,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        _pump_events(app, 1)
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError(f"Timed out while waiting for {label}")


def _authoring_router(workspace: Any) -> Any:
    """Return the active renderer-lifetime authoring router."""

    port = getattr(getattr(workspace, "viewport", None), "port", None)
    router = getattr(port, "router", None)
    if (
        router is None
        or not bool(getattr(router, "active", False))
        or getattr(router, "session", None) is not InteractionSession.AUTHORING
    ):
        raise RuntimeError("the embedded authoring interaction router is not active")
    return router


def _route_native_event(workspace: Any, event: Any) -> None:
    """Send one native-shaped input through the installed renderer router."""

    router = _authoring_router(workspace)
    route = getattr(router, "_route_event", None)
    if not callable(route):
        raise TypeError("the authoring router has no native event entry point")
    route(event)


def _native_pointer_event(
    event_type: str,
    position: tuple[float, float],
    *,
    button: int = 0,
    buttons: tuple[int, ...] = (),
    target: Any = None,
    pick_info: Mapping[str, Any] | None = None,
) -> Any:
    """Build the minimal native event shape consumed by the pygfx router."""

    return SimpleNamespace(
        type=str(event_type),
        x=float(position[0]),
        y=float(position[1]),
        button=int(button),
        buttons=tuple(buttons),
        modifiers=(),
        dx=0.0,
        dy=0.0,
        target=target,
        pick_info=dict(pick_info or {}),
        stop_propagation=lambda: None,
    )


def _project_world_to_screen(
    router: Any,
    position: tuple[float, float, float],
) -> tuple[float, float]:
    """Project a finite world point with the active renderer camera matrices."""

    matrices = router.runtime.camera_matrices()
    if matrices is None:
        raise RuntimeError("authoring camera matrices are unavailable")
    projection_inverse, view_matrix = matrices
    try:
        projection = np.linalg.inv(np.asarray(projection_inverse, dtype=float))
        view = np.asarray(view_matrix, dtype=float)
        world = np.asarray((*position, 1.0), dtype=float)
        clip = projection @ view @ world
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise RuntimeError("authoring camera matrices cannot project a point") from exc
    if clip.shape != (4,) or not np.all(np.isfinite(clip)) or abs(float(clip[3])) <= 1e-12:
        raise RuntimeError("world point does not have a finite screen projection")
    ndc = clip[:3] / clip[3]
    width, height = router.runtime.logical_size
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("authoring viewport has no logical size")
    return (
        float((ndc[0] + 1.0) * width * 0.5),
        float((1.0 - ndc[1]) * height * 0.5),
    )


def _route_work_plane_pointer(
    workspace: Any,
    event_type: str,
    position: tuple[float, float, float],
    *,
    button: int = 0,
    buttons: tuple[int, ...] = (),
) -> None:
    """Route one empty-space pointer event and verify its work-plane hit."""

    plane_spin = getattr(workspace, "work_plane_z_spin", None)
    configured_plane = None if plane_spin is None else float(plane_spin.value())
    if configured_plane is None or not np.isclose(
        configured_plane,
        position[2],
        rtol=0.0,
        atol=1e-9,
    ):
        raise AssertionError(
            f"visible work plane is Z={configured_plane!r}, "
            f"but placement requested Z={position[2]!r}"
        )
    router = _authoring_router(workspace)
    screen_position = _project_world_to_screen(router, position)
    event = _native_pointer_event(
        event_type,
        screen_position,
        button=button,
        buttons=buttons,
    )
    hit = router.resolve_hit(event)
    if hit is None or not np.allclose(
        hit.world_position,
        position,
        rtol=0.0,
        atol=1e-5,
    ):
        raise AssertionError(
            f"work-plane projection resolved {None if hit is None else hit.world_position}, "
            f"expected {position}"
        )
    _route_native_event(workspace, event)


def _place_selected(
    workspace: Any,
    position: tuple[float, float, float],
    *,
    pointer_position: tuple[float, float] | None = None,
) -> None:
    del pointer_position
    _route_work_plane_pointer(
        workspace,
        "pointer_down",
        position,
        button=1,
        buttons=(1,),
    )


def _preview_placement(
    workspace: Any,
    position: tuple[float, float, float],
    *,
    pointer_position: tuple[float, float] | None = None,
) -> None:
    del pointer_position
    _route_work_plane_pointer(workspace, "pointer_move", position)


def _drag_waypoint(
    workspace: Any,
    actor_id: Any,
    *,
    ordinal: int,
    start: tuple[float, float, float],
    screen_delta: tuple[float, float] = (80.0, -55.0),
) -> tuple[float, float, float]:
    component = f"mobility_control_waypoint_{ordinal}"
    document_id = workspace.document.scenario.document_id
    object_id = stable_renderer_id(document_id, actor_id, component)
    workspace.set_tool(AuthoringTool.MOVE)
    port = workspace.viewport.port
    router = _authoring_router(workspace)
    semantic_objects = port.renderer_objects()
    if object_id not in semantic_objects:
        raise AssertionError(f"waypoint handle was not reconciled: {object_id}")
    renderer = getattr(port, "renderer", None)
    native = getattr(renderer, "_objects", {}).get(object_id)
    if native is None:
        raise AssertionError(f"waypoint handle has no native renderer object: {object_id}")
    start_screen = _project_world_to_screen(router, start)
    down = _native_pointer_event(
        "pointer_down",
        start_screen,
        button=1,
        buttons=(1,),
        target=native,
        pick_info={
            "world_object": native,
            "world_position": start,
            "vertex_index": 0,
        },
    )
    _route_native_event(workspace, down)

    end_screen = (
        start_screen[0] + float(screen_delta[0]),
        start_screen[1] + float(screen_delta[1]),
    )
    move = _native_pointer_event(
        "pointer_move",
        end_screen,
        button=1,
        buttons=(1,),
    )
    resolved = router._resolve_drag_hit(move)
    if resolved is None or np.allclose(
        resolved.world_position,
        start,
        rtol=0.0,
        atol=1e-6,
    ):
        raise AssertionError("native waypoint drag did not resolve a changed world position")
    _route_native_event(workspace, move)
    _route_native_event(
        workspace,
        _native_pointer_event(
            "pointer_up",
            end_screen,
            button=1,
        ),
    )
    return resolved.world_position


def _begin_gizmo_transform(
    workspace: Any,
    actor_id: Any,
    initial_position: tuple[float, float, float],
    changed_position: tuple[float, float, float],
) -> str:
    """Start a native gizmo gesture through the renderer-lifetime router."""

    workspace.set_tool(AuthoringTool.MOVE)
    port = workspace.viewport.port
    port.renderer_objects()
    if not port.show_transform_gizmo(actor_id):
        raise AssertionError("the persistent authoring gizmo could not attach")
    router = _authoring_router(workspace)
    object_id = stable_renderer_id(
        workspace.document.scenario.document_id,
        actor_id,
        "mobility_handles",
    )
    router._on_gizmo_transform(
        {
            "phase": "selected",
            "object_id": object_id,
            "transform": _translation_matrix(initial_position),
        }
    )
    router._on_gizmo_transform(
        {
            "phase": "changed",
            "object_id": object_id,
            "transform": _translation_matrix(changed_position),
        }
    )
    return object_id


def _commit_gizmo_transform(
    workspace: Any,
    object_id: str,
    position: tuple[float, float, float],
) -> None:
    """Commit the active native gizmo gesture through its router callback."""

    _authoring_router(workspace)._on_gizmo_transform(
        {
            "phase": "committed",
            "object_id": object_id,
            "transform": _translation_matrix(position),
        }
    )


def _workspace_inventory(workspace: Any) -> dict[str, Any]:
    document = workspace.document
    port = getattr(workspace.viewport, "port", None)
    router = getattr(port, "router", None)
    renderer_objects = port.renderer_objects() if port is not None else {}
    compilation = workspace.compilation
    issues = tuple(compilation.issues) if compilation is not None else ()
    selection = document.selected_subject if document is not None else None
    return {
        "document_revision": None if document is None else document.revision,
        "document_dirty": None if document is None else document.dirty,
        "actor_count": 0 if document is None else len(document.actors),
        "group_count": 0 if document is None else len(document.groups),
        "selected_kind": None if selection is None else selection.kind.value,
        "selected_id": None if selection is None else str(selection.id),
        "problem_count": len(issues),
        "error_count": sum(issue.severity.value == "error" for issue in issues),
        "renderer_object_count": len(renderer_objects),
        "handler_count": (
            None
            if router is None or not hasattr(router, "handler_count")
            else int(router.handler_count)
        ),
    }


def _actor_overlay(workspace: Any, actor_id: Any) -> Any:
    snapshot = workspace.viewport.port.current_snapshot()
    if snapshot is None:
        raise AssertionError("The authoring viewport has no overlay snapshot")
    return next(overlay for overlay in snapshot.actors if overlay.actor_id == actor_id)


def _translation_matrix(
    position: tuple[float, float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    x, y, z = position
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _image_evidence(path: Path) -> dict[str, Any]:
    """Return deterministic evidence that a captured image is decodable and nonblank."""

    image = QImage(str(path))
    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        raise AssertionError(f"capture is not a decodable image: {path}")
    sample = image.scaled(32, 32)
    colors = {
        int(sample.pixel(x, y)) for x in range(sample.width()) for y in range(sample.height())
    }
    visible_pixels = (
        sample.pixelColor(x, y) for x in range(sample.width()) for y in range(sample.height())
    )
    if not any(
        color.alpha() > 0 and (color.red(), color.green(), color.blue()) != (0, 0, 0)
        for color in visible_pixels
    ):
        raise AssertionError(f"capture is visually blank: {path}")
    content = path.read_bytes()
    if len(content) < 8:
        raise AssertionError(f"capture file is unexpectedly small: {path}")
    return {
        "width": image.width(),
        "height": image.height(),
        "byte_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "sampled_color_count": len(colors),
    }


def _assert_capture_images_differ(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    surface: str = "viewport",
) -> None:
    """Require two primary visual states to have different captured pixels."""

    key = f"{surface}_image_evidence"
    first_hash = str(first[key]["sha256"])
    second_hash = str(second[key]["sha256"])
    if first_hash == second_hash:
        raise AssertionError(
            f"{first['label']} and {second['label']} produced identical {surface} captures"
        )


def _renderer_component_present(workspace: Any, actor_id: Any, component: str) -> bool:
    """Return whether one stable actor overlay component reached the renderer port."""

    port = workspace.viewport.port
    object_id = stable_renderer_id(
        workspace.document.scenario.document_id,
        actor_id,
        component,
    )
    return object_id in port.renderer_objects()


def _assert_problem(
    compilation: Any,
    *,
    code: str,
    path: str,
    message: str,
) -> None:
    """Require one exact structured validation problem."""

    problems = {
        (str(issue.code), str(issue.path), str(issue.message)) for issue in compilation.issues
    }
    expected = (code, path, message)
    if expected not in problems:
        raise AssertionError(f"missing exact validation problem {expected!r}: {problems!r}")


def _capture(
    app: QApplication,
    visualizer: OrchavVisualizer,
    capture_dir: Path,
    label: str,
    *,
    workspace: Any | None,
    note: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if workspace is not None:
        workspace.fit_all()
    _pump_events(app, 8)

    workspace_path = capture_dir / f"{label}__workspace.png"
    pixmap = visualizer.grab()
    if pixmap.isNull() or not pixmap.save(str(workspace_path)):
        raise RuntimeError(f"Could not capture complete workspace for {label}")

    renderer = (
        getattr(getattr(getattr(workspace, "viewport", None), "port", None), "renderer", None)
        if workspace is not None
        else getattr(visualizer, "renderer", None)
    )
    viewport_path = capture_dir / f"{label}__viewport.png"
    export = getattr(renderer, "export_screenshot", None)
    if not callable(export) or not bool(export(str(viewport_path))):
        raise RuntimeError(f"Could not capture renderer viewport for {label}")

    workspace_evidence = _image_evidence(workspace_path)
    viewport_evidence = _image_evidence(viewport_path)
    payload: dict[str, Any] = {
        "label": label,
        "note": note,
        "workspace_image": workspace_path.relative_to(capture_dir.parent).as_posix(),
        "viewport_image": viewport_path.relative_to(capture_dir.parent).as_posix(),
        "workspace_image_evidence": workspace_evidence,
        "viewport_image_evidence": viewport_evidence,
        "semantic_evidence": dict(evidence or {}),
    }
    if workspace is not None:
        payload.update(_workspace_inventory(workspace))
    return payload


def _record_capture(capture: dict[str, Any], sink: CaptureSink) -> dict[str, Any]:
    """Publish one completed checkpoint to the incremental manifest."""

    sink(capture)
    return capture


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Persist the latest capture checkpoint inside the new evidence directory."""

    path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")


def _try_render_generated_frame(visualizer: Any) -> int | None:
    """Synchronously load and submit one available generated frame."""

    if not bool(getattr(visualizer, "vis_initialized", False)):
        return None
    available = tuple(int(step) for step in visualizer.get_available_animation_steps())
    if not available:
        return None
    requested = int(getattr(visualizer, "animation_step", available[0]))
    step = requested if requested in available else available[0]
    if not bool(visualizer.update_frame(step)):
        return None
    renderer = getattr(visualizer, "renderer", None)
    if renderer is None or getattr(renderer, "last_frame_packet", None) is None:
        return None
    return step


def _hdf5_frame_evidence(frames_directory: Path) -> dict[str, Any]:
    """Validate promoted HDF5 chunks and return their concrete file evidence."""

    chunks = tuple(sorted(frames_directory.glob("mpc_frames_*.h5")))
    if not chunks:
        raise AssertionError(f"no promoted HDF5 frame chunks in {frames_directory}")
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        content = chunk.read_bytes()
        if not content.startswith(b"\x89HDF\r\n\x1a\n"):
            raise AssertionError(f"generated frame chunk is not HDF5: {chunk}")
        records.append(
            {
                "path": chunk.name,
                "byte_size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "chunk_count": len(records),
        "chunks": records,
    }


def _undo_fingerprint(document: Any) -> dict[str, Any]:
    """Return observable undo identity and position for draft-retention checks."""

    adapter = document.undo_stack
    qt_stack = getattr(adapter, "stack", None)
    return {
        "adapter_identity": id(adapter),
        "stack_identity": None if qt_stack is None else id(qt_stack),
        "count": None if qt_stack is None else int(qt_stack.count()),
        "index": None if qt_stack is None else int(qt_stack.index()),
        "can_undo": bool(adapter.can_undo()),
        "can_redo": bool(adapter.can_redo()),
        "undo_text": str(adapter.undo_text()),
        "redo_text": str(adapter.redo_text()),
    }


def _select_overlay_options(workspace: Any) -> None:
    for combo, value in (
        (workspace.trajectory_visibility_combo, OverlayVisibility.ALL),
        (workspace.frame_samples_visibility_combo, OverlayVisibility.SELECTED),
        (workspace.control_rig_visibility_combo, OverlayVisibility.SELECTED),
        (workspace.orientation_axes_combo, OverlayVisibility.ALL),
        (workspace.look_at_rays_combo, OverlayVisibility.ALL),
    ):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)


def _configure_timeline(workspace: Any) -> None:
    workspace.document.set_timeline(
        TimelineSettings(
            steps=4,
            duration_s=1.0,
            quality=workspace.document.scenario.timeline.quality,
            export_path_metrics=True,
        )
    )


def _build_authoring_scenario(
    app: QApplication,
    visualizer: OrchavVisualizer,
    capture_dir: Path,
    capture_sink: CaptureSink,
) -> Any:
    controller = visualizer.workspace_mode_controller
    if not visualizer.new_authoring_scenario():
        raise RuntimeError("Scenario Builder did not enter authoring mode")
    workspace = controller.workspace
    if workspace is None or not workspace.viewport.available:
        error = None if workspace is None else workspace.viewport.initialization_error
        raise RuntimeError(f"Embedded pygfx authoring viewport is unavailable: {error}")

    _select_overlay_options(workspace)
    _configure_timeline(workspace)
    workspace.work_plane_z_spin.setValue(1.0)
    workspace.grid_snap_check.setChecked(False)
    workspace.refresh_now()
    expected_scene = SceneReference("library", "empty/empty.xml")
    if workspace.document.scenario.scene != expected_scene:
        raise AssertionError(
            f"new authoring scene is {workspace.document.scenario.scene!r}, "
            f"expected {expected_scene!r}"
        )
    empty_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "01_empty",
            workspace=workspace,
            note="New document on the default empty ORCHAV scene.",
            evidence={
                "scene_source": expected_scene.source,
                "scene_id": expected_scene.id,
            },
        ),
        capture_sink,
    )

    workspace._add_actor(ActorRole.TX)
    _preview_placement(workspace, (-8.0, -4.0, 1.0))
    placement_snapshot = workspace.viewport.port.current_snapshot()
    if (
        placement_snapshot is None
        or placement_snapshot.placement_ghost is None
        or not np.allclose(
            placement_snapshot.placement_ghost,
            (-8.0, -4.0, 1.0),
            rtol=0.0,
            atol=1e-5,
        )
    ):
        raise AssertionError("routed pointer hover did not publish the placement ghost")
    pending_placement_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "02_pending_placement",
            workspace=workspace,
            note="One-click placement ghost before the TX is committed.",
            evidence={"placement_ghost": list(placement_snapshot.placement_ghost)},
        ),
        capture_sink,
    )
    _assert_capture_images_differ(empty_capture, pending_placement_capture)
    _place_selected(workspace, (-8.0, -4.0, 1.0))
    tx = workspace.document.selected_actor
    if tx is None:
        raise AssertionError("TX placement did not retain selection")
    workspace.refresh_now()
    selected_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "03_selected",
            workspace=workspace,
            note="Placed and selected stationary TX.",
            evidence={
                "selected_actor_id": str(tx.id),
                "position_m": [-8.0, -4.0, 1.0],
            },
        ),
        capture_sink,
    )
    _assert_capture_images_differ(pending_placement_capture, selected_capture)

    circular_draft = CircularMobilitySpec(
        center_m=(-12.0, -4.0, 1.0),
        radius_m=4.0,
        start_angle_deg=0.0,
    )
    workspace.mobility_editor.set_mobility(circular_draft)
    workspace.mobility_editor.circular_radius_spin.setValue(4.5)
    _wait_until(
        app,
        lambda: workspace.has_pending_inspector_edits
        and workspace._candidate_compilation is not None,
        timeout_s=10.0,
        label="pending inspector candidate",
    )
    applied_tx = workspace.document.scenario.actor(tx.id)
    if applied_tx is None or not isinstance(applied_tx.mobility, StationaryMobilitySpec):
        raise AssertionError("A pending mobility draft changed the document")
    pending_overlay = _actor_overlay(workspace, tx.id)
    if not pending_overlay.mobility_draft_pending or not pending_overlay.pending_positions:
        raise AssertionError("pending circular mobility has no canonical ghost overlay")
    if not _renderer_component_present(workspace, tx.id, "pending_path"):
        raise AssertionError("pending circular path did not reach the renderer")
    pending_inspector_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "04_pending_inspector",
            workspace=workspace,
            note="Protected circular-motion draft with canonical ghost preview.",
            evidence={
                "applied_mobility": "stationary",
                "pending_mobility": "circular",
                "pending_path_points": len(pending_overlay.pending_positions),
            },
        ),
        capture_sink,
    )
    workspace.pending_reset_button.click()
    _pump_events(app)
    if workspace.has_pending_inspector_edits:
        raise AssertionError("Reset did not clear the pending inspector draft")

    workspace.mobility_editor.set_mobility(circular_draft)
    workspace.mobility_editor.circular_radius_spin.setValue(4.5)
    _wait_until(
        app,
        lambda: workspace.has_pending_inspector_edits
        and workspace._candidate_compilation is not None,
        timeout_s=10.0,
        label="recreated inspector candidate",
    )
    workspace.pending_apply_button.click()
    _pump_events(app)
    applied_tx = workspace.document.scenario.actor(tx.id)
    if (
        workspace.has_pending_inspector_edits
        or applied_tx is None
        or not isinstance(applied_tx.mobility, CircularMobilitySpec)
    ):
        raise AssertionError("Apply did not commit the pending inspector draft")
    workspace.refresh_now()
    if workspace.compilation is None:
        raise AssertionError("invalid circular checkpoint has no compilation")
    _assert_problem(
        workspace.compilation,
        code="actors.rx.required",
        path="actors.rx",
        message="At least one RX is required.",
    )
    if not _renderer_component_present(workspace, tx.id, "selection_path_halo"):
        raise AssertionError("selected circular path has no renderer halo")
    workspace.drawer.setCurrentWidget(workspace.problems_tree)
    invalid_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "05_invalid_circular",
            workspace=workspace,
            note="Circular TX is visible while validation reports the missing RX.",
            evidence={
                "problem_code": "actors.rx.required",
                "problem_path": "actors.rx",
                "selected_path_halo": True,
            },
        ),
        capture_sink,
    )
    _assert_capture_images_differ(pending_inspector_capture, invalid_capture)

    workspace._add_actor(ActorRole.RX)
    _place_selected(workspace, (-1.0, -4.0, 1.0), pointer_position=(300.0, 220.0))
    rx = workspace.document.selected_actor
    if rx is None:
        raise AssertionError("RX placement did not retain selection")
    workspace.document.select(tx.id)
    workspace.refresh_now()
    if workspace.compilation is None:
        raise AssertionError("valid circular checkpoint has no compilation")
    blocking_issues = tuple(
        issue for issue in workspace.compilation.issues if issue.severity.value == "error"
    )
    if blocking_issues:
        raise AssertionError(f"valid circular checkpoint has blocking issues: {blocking_issues!r}")
    if not _renderer_component_present(workspace, tx.id, "selection_path_halo"):
        raise AssertionError("valid selected circular path has no renderer halo")
    valid_circular_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "06_valid_circular",
            workspace=workspace,
            note="Valid selected circular TX after the required RX is placed.",
            evidence={
                "selected_actor_id": str(tx.id),
                "mobility": "circular",
                "blocking_problem_count": 0,
                "selected_path_halo": True,
            },
        ),
        capture_sink,
    )
    _assert_capture_images_differ(invalid_capture, valid_circular_capture)
    workspace.document.select(rx.id)
    authored_waypoints = (
        (-1.0, -4.0, 1.0),
        (1.0, 0.0, 3.0),
        (5.0, 4.0, 1.0),
    )
    workspace.mobility_editor.set_mobility(WaypointMobilitySpec(points_m=authored_waypoints))
    workspace._apply_mobility()
    workspace.refresh_now()
    _pump_events(app, 4)
    changed_waypoint = _drag_waypoint(
        workspace,
        rx.id,
        ordinal=1,
        start=authored_waypoints[1],
    )
    workspace.set_tool(AuthoringTool.SELECT)
    workspace.refresh_now()
    edited_rx = workspace.document.scenario.actor(rx.id)
    if (
        edited_rx is None
        or not isinstance(edited_rx.mobility, WaypointMobilitySpec)
        or not np.allclose(
            edited_rx.mobility.points_m[1],
            changed_waypoint,
            rtol=0.0,
            atol=1e-5,
        )
        or np.allclose(
            edited_rx.mobility.points_m[1],
            authored_waypoints[1],
            rtol=0.0,
            atol=1e-5,
        )
    ):
        raise AssertionError("native waypoint handle drag did not commit the edited point")
    if not _renderer_component_present(workspace, rx.id, "selection_path_halo"):
        raise AssertionError("selected waypoint path has no renderer halo")
    waypoint_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "07_waypoint",
            workspace=workspace,
            note="Generator-prepared waypoint route after direct handle dragging.",
            evidence={
                "edited_ordinal": 1,
                "edited_position_m": list(changed_waypoint),
                "selected_path_halo": True,
            },
        ),
        capture_sink,
    )

    workspace._add_actor(ActorRole.TARGET)
    _place_selected(workspace, (8.0, -2.0, 1.0), pointer_position=(460.0, 260.0))
    target = workspace.document.selected_actor
    if target is None:
        raise AssertionError("Target placement did not retain selection")
    target_index = workspace.target_asset_combo.findData("cube")
    if target_index < 0:
        raise RuntimeError("The cube catalog target is unavailable")
    workspace.target_asset_combo.setCurrentIndex(target_index)
    workspace.target_scale_spin.setValue(2.0)
    workspace._apply_target()
    target = workspace.document.scenario.actor(target.id)
    assert target is not None
    workspace.document.replace_actor(
        target.with_changes(
            orientation=LookAtOrientationSpec(actor=str(tx.id)),
        ),
        text=f"Point {target.name} at {tx.name}",
    )
    tx = workspace.document.scenario.actor(tx.id)
    assert tx is not None
    workspace.document.replace_actor(
        tx.with_changes(orientation=LookAtOrientationSpec(actor=str(target.id))),
        text=f"Point {tx.name} at {target.name}",
    )
    workspace.refresh_now()

    before_tx_orientation = _actor_overlay(workspace, tx.id).orientation_matrix
    target_overlay = _actor_overlay(workspace, target.id)
    initial_target_position = target_overlay.current_position
    if before_tx_orientation is None or initial_target_position is None:
        raise AssertionError("Prepared target and Look At overlays are incomplete")
    changed_target_position = (
        initial_target_position[0],
        initial_target_position[1] + 3.0,
        initial_target_position[2] + 2.0,
    )
    target_gizmo_id = _begin_gizmo_transform(
        workspace,
        target.id,
        initial_target_position,
        changed_target_position,
    )
    _pump_events(app)
    after_tx_orientation = _actor_overlay(workspace, tx.id).orientation_matrix
    if after_tx_orientation == before_tx_orientation:
        raise AssertionError("Transient target motion did not recompute Look At")
    target_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "08_target",
            workspace=workspace,
            note=(
                "Catalog target during a transient drag, with mutually recomputed "
                "look-at orientation evidence."
            ),
            evidence={
                "gizmo_routed": True,
                "transient_target_position_m": list(changed_target_position),
                "dependent_look_at_recomputed": True,
            },
        ),
        capture_sink,
    )
    _assert_capture_images_differ(waypoint_capture, target_capture)
    _commit_gizmo_transform(
        workspace,
        target_gizmo_id,
        changed_target_position,
    )
    workspace.set_tool(AuthoringTool.SELECT)

    workspace._add_actor(ActorRole.RX)
    _place_selected(workspace, (-1.0, -8.0, 1.0), pointer_position=(540.0, 310.0))
    second_rx = workspace.document.selected_actor
    if second_rx is None:
        raise AssertionError("Second RX placement did not retain selection")
    rx = workspace.document.scenario.actor(rx.id)
    assert rx is not None
    group = AuthoringGroup.create("WaypointFormation").with_changes(mobility=rx.mobility)
    first_member = rx.with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=-2.0),
        )
    )
    second_member = second_rx.with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=2.0),
        ),
        orientation=SpinOrientationSpec(axis="yaw", rate_deg_s=45.0),
    )
    workspace.document.add_group_with_members(group, (first_member, second_member))
    workspace.refresh_now()
    for member_id in (first_member.id, second_member.id):
        overlay = _actor_overlay(workspace, member_id)
        if overlay.group_origin_position is None or overlay.group_frame_matrix is None:
            raise AssertionError("group member overlay lacks its origin or prepared frame")
        for component in ("group_tether", "group_frame"):
            if not _renderer_component_present(workspace, member_id, component):
                raise AssertionError(f"group member {member_id} has no rendered {component}")
    group_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "09_group",
            workspace=workspace,
            note="Rigid two-RX formation with group path and member tethers.",
            evidence={
                "group_id": str(group.id),
                "member_ids": [str(first_member.id), str(second_member.id)],
                "member_tethers": 2,
                "member_frames": 2,
            },
        ),
        capture_sink,
    )
    _assert_capture_images_differ(target_capture, group_capture)
    return workspace


def _persist_unattended(workspace: Any, scenario_dir: Path) -> Path:
    """Save through persistence directly so errors propagate to the manifest."""
    saved = save_document(
        workspace.document,
        scenario_dir,
        compiler=getattr(workspace, "compiler", None),
        compile_lock=getattr(workspace, "compilation_lock", None),
    )
    workspace.append_generation_log(f"Saved {saved}")
    return saved


def _save_and_generate(
    app: QApplication,
    visualizer: OrchavVisualizer,
    workspace: Any,
    scenario_dir: Path,
    output_dir: Path,
    capture_dir: Path,
    capture_sink: CaptureSink,
    stage_sink: StageSink,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    mode_controller = visualizer.workspace_mode_controller
    document = workspace.document
    stage_sink("save", "started", None)
    saved_path = _persist_unattended(workspace, scenario_dir)
    relocated_evidence = _copy_relocatable_scenario_evidence(scenario_dir, output_dir)
    stage_sink(
        "save",
        "passed",
        {
            "saved_path": relocated_evidence["scenario_yaml"],
            "persistence_filename": saved_path.name,
            "document_revision": document.revision,
        },
    )
    saved_revision = document.revision
    saved_identity = id(document)
    saved_selection = document.selected_subject
    saved_undo = _undo_fingerprint(document)

    stage_sink("generation", "started", None)
    generation = visualizer.authoring_generation_controller
    if not generation.start(document):
        raise RuntimeError("Generation job did not start")
    _wait_until(
        app,
        lambda: generation.last_result is not None
        and generation.last_result.state is not GenerationState.RUNNING,
        timeout_s=timeout_s,
        label="Scenario Builder generation",
    )
    result = generation.last_result
    if result is None or not result.succeeded:
        raise RuntimeError(f"Scenario Builder generation failed: {result}")
    if result.stale or result.current_revision != result.launched_revision:
        raise AssertionError("unchanged draft produced a stale generation result")
    event_names = tuple(str(event.get("event")) for event in result.events)
    if "step_completed" not in event_names or event_names[-1:] != ("run_completed",):
        raise AssertionError(f"generation did not retain complete JSONL evidence: {event_names}")
    stdout_events = tuple(
        json.loads(line) for line in result.stdout_log if line.lstrip().startswith("{")
    )
    if tuple(str(event.get("event")) for event in stdout_events) != event_names:
        raise AssertionError("Generation Log JSONL lines do not match parsed job events")
    required_log_lines = tuple(
        f"[stdout] {line}"
        for line, event in zip(result.stdout_log, stdout_events, strict=True)
        if event.get("event") in {"step_completed", "run_completed"}
    )
    if not required_log_lines:
        raise AssertionError("Generation Log does not show structured stdout evidence")
    _wait_until(
        app,
        lambda: all(line in workspace.generation_log.toPlainText() for line in required_log_lines),
        timeout_s=min(timeout_s, 10.0),
        label="Generation Log JSONL delivery",
    )
    if workspace.generation_status.text() != "Generation succeeded":
        raise AssertionError(
            f"unexpected terminal generation status: {workspace.generation_status.text()!r}"
        )
    frame_evidence = _hdf5_frame_evidence(result.paths.final_frames)
    relocated_evidence = _copy_relocatable_scenario_evidence(scenario_dir, output_dir)
    workspace.drawer.setCurrentWidget(workspace.generation_log)
    generation_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "10_generation",
            workspace=workspace,
            note="Generation Log showing successful JSONL progress and promoted HDF5 output.",
            evidence={
                "generation_status": workspace.generation_status.text(),
                "jsonl_events": list(event_names),
                "stdout_lines": len(result.stdout_log),
                "stderr_lines": len(result.stderr_log),
                "hdf5_chunk_count": frame_evidence["chunk_count"],
                "stale": result.stale,
            },
        ),
        capture_sink,
    )
    stage_sink(
        "generation",
        "passed",
        {
            "frames_directory": relocated_evidence["frames_directory"],
            "hdf5_chunk_count": frame_evidence["chunk_count"],
            "jsonl_events": list(event_names),
        },
    )
    stage_sink("preview", "started", None)
    before_preview_router = _authoring_router(workspace)
    before_preview_handler_count = int(before_preview_router.handler_count)
    if before_preview_handler_count != 9:
        raise AssertionError(
            "the authoring router does not own exactly one complete "
            f"nine-event registration set: {before_preview_handler_count}"
        )
    before_preview_object_ids = frozenset(workspace.viewport.port.renderer_objects())

    mode_controller.preview_generated_result()
    rendered_step: list[int | None] = [None]

    def generated_frame_rendered() -> bool:
        if mode_controller.mode is not WorkspaceMode.VISUALIZATION:
            return False
        rendered_step[0] = _try_render_generated_frame(visualizer)
        return rendered_step[0] is not None

    _wait_until(
        app,
        generated_frame_rendered,
        timeout_s=timeout_s,
        label="loaded and renderer-accepted generated frame",
    )
    available_steps = tuple(int(step) for step in visualizer.get_available_animation_steps())
    if rendered_step[0] not in available_steps:
        raise AssertionError("renderer-accepted generated step is absent from the frame source")
    fit_generated_scene = getattr(visualizer.renderer, "reset_camera_bounds", None)
    if not callable(fit_generated_scene):
        raise AssertionError("generated-result renderer cannot fit the loaded scene")
    fit_generated_scene()
    _pump_events(app, 8)
    playback_capture = _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "11_generated_preview",
            workspace=None,
            note="Explicit playback after a generated HDF5 frame reached pygfx.",
            evidence={
                "rendered_step": rendered_step[0],
                "available_steps": list(available_steps),
                "renderer_packet_present": True,
                "hdf5_chunk_count": frame_evidence["chunk_count"],
            },
        ),
        capture_sink,
    )
    _assert_capture_images_differ(generation_capture, playback_capture, surface="workspace")
    stage_sink(
        "preview",
        "passed",
        {
            "rendered_step": rendered_step[0],
            "available_steps": list(available_steps),
        },
    )

    stage_sink("resume", "started", None)
    if not mode_controller.resume_document():
        raise RuntimeError("Could not return to the preserved authoring draft")
    resumed = mode_controller.workspace
    if resumed is None:
        raise RuntimeError("Authoring workspace was not reconstructed")
    _select_overlay_options(resumed)
    resumed.work_plane_z_spin.setValue(1.0)
    resumed.grid_snap_check.setChecked(False)
    resumed.refresh_now()
    _pump_events(app, 4)
    _wait_until(
        app,
        lambda: resumed.compilation is not None,
        timeout_s=timeout_s,
        label="resumed authoring compilation",
    )
    if id(resumed.document) != saved_identity:
        raise AssertionError("Generated preview replaced the authoring document")
    if resumed.document.revision != saved_revision:
        raise AssertionError("Generated preview changed the authoring revision")
    if resumed.document.selected_subject != saved_selection:
        raise AssertionError("Generated preview changed the authoring selection")
    resumed_undo = _undo_fingerprint(resumed.document)
    if resumed_undo != saved_undo:
        raise AssertionError(
            f"Generated preview changed undo history: {saved_undo!r} -> {resumed_undo!r}"
        )
    resumed_router = _authoring_router(resumed)
    resumed_handler_count = int(resumed_router.handler_count)
    if resumed_handler_count != before_preview_handler_count:
        raise AssertionError(
            "authoring router registration count changed across generated preview: "
            f"{before_preview_handler_count} -> {resumed_handler_count}"
        )
    resumed_object_ids = frozenset(resumed.viewport.port.renderer_objects())
    if resumed_object_ids != before_preview_object_ids:
        raise AssertionError(
            "authoring renderer object inventory changed across generated preview: "
            f"{sorted(before_preview_object_ids)!r} -> {sorted(resumed_object_ids)!r}"
        )
    stage_sink(
        "resume",
        "passed",
        {
            "draft_identity_preserved": True,
            "draft_revision_preserved": True,
            "draft_selection_preserved": True,
            "draft_undo_preserved": True,
            "renderer_object_ids_preserved": True,
        },
    )

    return {
        "scenario_yaml": "scenario/scenario.yaml",
        "frames_directory": "scenario/frames",
        "hdf5": frame_evidence,
        "jsonl_events": list(event_names),
        "rendered_step": rendered_step[0],
        "available_steps": list(available_steps),
        "launched_revision": result.launched_revision,
        "current_revision": result.current_revision,
        "stale": result.stale,
        "draft_identity_preserved": True,
        "draft_revision_preserved": True,
        "draft_selection_preserved": True,
        "draft_undo_preserved": True,
        "router_handler_count_before_preview": before_preview_handler_count,
        "router_handler_count_after_resume": resumed_handler_count,
        "renderer_object_count_before_preview": len(before_preview_object_ids),
        "renderer_object_count_after_resume": len(resumed_object_ids),
        "renderer_object_ids_preserved": True,
    }


def _capture_read_only(
    app: QApplication,
    visualizer: OrchavVisualizer,
    capture_dir: Path,
    capture_sink: CaptureSink,
) -> None:
    controller = visualizer.workspace_mode_controller
    fixture = REPO_ROOT / "tests" / "visualizer" / "authoring" / "fixtures" / "future_owned.yaml"
    if not controller.open_document(fixture):
        raise RuntimeError("Future-version fixture did not open in read-only mode")
    workspace = controller.workspace
    if workspace is None:
        raise RuntimeError("Read-only import has no workspace")
    result = controller._read_only_result
    expected_problem = (
        "compatibility.document_version",
        "visualizer.scenario_builder.document_version",
        "Unsupported Scenario Builder document version: 3.",
    )
    problems = {
        (str(issue.code), str(issue.path), str(issue.message))
        for issue in (() if result is None else result.issues)
    }
    if expected_problem not in problems:
        raise AssertionError(f"future document lacks its exact compatibility problem: {problems!r}")
    if (
        workspace.document is not None
        or not workspace._read_only
        or not workspace.read_only_label.isVisible()
        or workspace.read_only_label.text() != "READ ONLY"
    ):
        raise AssertionError("future document is not visibly locked read-only")
    workspace.drawer.setCurrentWidget(workspace.problems_tree)
    _record_capture(
        _capture(
            app,
            visualizer,
            capture_dir,
            "12_read_only",
            workspace=workspace,
            note="Future document version locked read-only at the exact marker path.",
            evidence={
                "read_only": True,
                "problem_code": expected_problem[0],
                "problem_path": expected_problem[1],
                "problem_message": expected_problem[2],
            },
        ),
        capture_sink,
    )


def main() -> None:
    global _ACTIVE_MODAL_GUARD

    args = parse_args()
    output_dir, capture_dir = _prepare_output_directory(args.output_dir)
    manifest_path = output_dir / "capture_manifest.json"
    manifest: dict[str, Any] = {
        "renderer": "pygfx",
        "viewport_mode": "embedded",
        "captures": [],
        "stages": [],
        "unexpected_modals": [],
        "generation": None,
        "passed": None,
    }
    active_stage: str | None = None

    def record_capture(capture: dict[str, Any]) -> None:
        manifest["captures"].append(capture)
        _write_manifest(manifest_path, manifest)

    def record_stage(
        name: str,
        status: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal active_stage
        if status == "started":
            if active_stage is not None:
                raise RuntimeError(
                    f"Cannot start capture stage {name!r}; {active_stage!r} is active"
                )
            active_stage = name
        elif active_stage == name:
            active_stage = None
        event: dict[str, Any] = {
            "name": name,
            "status": status,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        if evidence:
            event["evidence"] = dict(evidence)
        manifest["stages"].append(event)
        _write_manifest(manifest_path, manifest)

    def record_modal(modal: dict[str, str]) -> None:
        manifest["unexpected_modals"].append(modal)
        _write_manifest(manifest_path, manifest)

    _write_manifest(manifest_path, manifest)
    os.environ[SCENARIO_BUILDER_ENV] = "1"
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app: QApplication | None = None
    visualizer: OrchavVisualizer | None = None
    modal_guard: _UnexpectedModalGuard | None = None
    work_root: Path | None = None
    scenario_dir: Path | None = None
    work_token: str | None = None
    try:
        work_root, scenario_dir, work_token = _create_owned_work_scenario()
        manifest["working_directory"] = {
            "location": "checkout/tmp",
            "marker_owned": True,
            "removed": False,
        }
        _write_manifest(manifest_path, manifest)

        app = QApplication.instance() or QApplication(sys.argv)
        modal_guard = _UnexpectedModalGuard(record_modal)
        app.installEventFilter(modal_guard)
        _ACTIVE_MODAL_GUARD = modal_guard
        visualizer = OrchavVisualizer(
            progress=ProgressReporter(enabled=False),
            renderer_type="pygfx",
            viewport_mode="embedded",
        )
        visualizer.resize(args.width, args.height)
        visualizer.show()
        _pump_events(app, 4)
        visualizer._deferred_init(pending_camera=None)
        _pump_events(app, 4)
        workspace = _build_authoring_scenario(
            app,
            visualizer,
            capture_dir,
            record_capture,
        )
        if not args.skip_generation:
            generation_evidence = _save_and_generate(
                app,
                visualizer,
                workspace,
                scenario_dir,
                output_dir,
                capture_dir,
                record_capture,
                record_stage,
                timeout_s=args.timeout_s,
            )
            manifest["generation"] = generation_evidence
            _write_manifest(manifest_path, manifest)
        else:
            record_stage("save", "started")
            saved_path = _persist_unattended(workspace, scenario_dir)
            relocated = _copy_relocatable_scenario_evidence(scenario_dir, output_dir)
            record_stage(
                "save",
                "passed",
                {
                    "saved_path": relocated["scenario_yaml"],
                    "persistence_filename": saved_path.name,
                    "document_revision": workspace.document.revision,
                },
            )
            for stage_name in ("generation", "preview", "resume"):
                record_stage(
                    stage_name,
                    "skipped",
                    {"reason": "--skip-generation"},
                )
        record_stage("read_only", "started")
        _capture_read_only(app, visualizer, capture_dir, record_capture)
        record_stage("read_only", "passed", {"capture": "12_read_only"})
        manifest["passed"] = True
        _write_manifest(manifest_path, manifest)
    except BaseException as exc:
        if active_stage is not None:
            record_stage(
                active_stage,
                "failed",
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        manifest["passed"] = False
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_manifest(manifest_path, manifest)
        traceback.print_exc()
        raise
    finally:
        if app is not None and modal_guard is not None:
            app.removeEventFilter(modal_guard)
        _ACTIVE_MODAL_GUARD = None
        try:
            if visualizer is not None:
                shutdown_visualizer(visualizer, persist_state=False)
                visualizer.close()
            if app is not None:
                _pump_events(app, 2)
        finally:
            cleanup_error: Exception | None = None
            if work_root is not None and work_token is not None:
                try:
                    _remove_owned_work_directory(work_root, work_token)
                    manifest["working_directory"]["removed"] = True
                except Exception as exc:
                    cleanup_error = exc
                    manifest["cleanup_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
            _write_manifest(manifest_path, manifest)
            if cleanup_error is not None and manifest.get("passed") is True:
                raise cleanup_error


if __name__ == "__main__":
    main()
