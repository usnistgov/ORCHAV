"""Qt workspace behavior for the integrated Scenario Builder."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractSpinBox, QWidget
from shiboken6 import isValid

from generator.core.materials.target_materials import available_target_material_types
from shared.scenarios.actors import (
    ActorRole,
    AlignMotionOrientationSpec,
    CircularMobilitySpec,
    Figure8MobilitySpec,
    FixedOrientationSpec,
    GroupDeviationSpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    KeyframesOrientationSpec,
    LinearMobilitySpec,
    OrientationKeyframeSpec,
    RandomSamplingMobilitySpec,
    SampledMobilitySpec,
    SpinOrientationSpec,
    StationaryMobilitySpec,
    WaypointMobilitySpec,
)
from visualizer.src.authoring import workspace as workspace_module
from visualizer.src.authoring.assets import TargetPreviewAsset, prepared_actor_pose
from visualizer.src.authoring.compilation_scheduler import CompilationFailure
from visualizer.src.authoring.compiler import (
    ActorSamples,
    CompilationResult,
    GroupSamples,
    IssueSeverity,
    ScenarioCompiler,
    ValidationIssue,
)
from visualizer.src.authoring.document import ScenarioDocument
from visualizer.src.authoring.domain import (
    AuthoringActor,
    AuthoringGroup,
    AuthoringResource,
    AuthoringScenario,
    QualityPreset,
    ResourceKind,
    ScenarioSourceSnapshot,
    SceneReference,
    TargetAsset,
    TimelineSettings,
)
from visualizer.src.authoring.interaction import InteractionSession, PygfxInteractionRouter
from visualizer.src.authoring.mobility_control_rig import mobility_control_rig
from visualizer.src.authoring.model_capabilities import (
    EditAffordance,
    orientation_capability,
)
from visualizer.src.authoring.orientation_models import (
    actor_look_at_orientation,
    look_at_actor_id,
    orientation_kind,
)
from visualizer.src.authoring.persistence import scenario_from_mapping
from visualizer.src.authoring.viewport_port import (
    ActorVisualState,
    AuthoringTool,
    HitResult,
    KeyboardInput,
    OverlayVisibility,
    PointerInput,
    PointerPhase,
    PreviewProvenance,
    TrajectoryDisplayMode,
    TransformInput,
    TransformPhase,
    stable_renderer_id,
)
from visualizer.src.authoring.workspace import ScenarioAuthoringWorkspace
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload
from visualizer.visualizer import OrchavVisualizer


class _Compiler:
    project_root = Path(__file__).parents[3]
    group_offsets = staticmethod(ScenarioCompiler.group_offsets)

    def compile(self, _document, *, scenario_directory=None):
        return CompilationResult({}, "schema_version: 2\n", (), {})


class _FixedCompiler(_Compiler):
    def __init__(self, result):
        self.result = result

    def compile(self, _document, *, scenario_directory=None):
        return self.result


class _Port:
    def __init__(self):
        self.snapshots = []
        self.tool = AuthoringTool.SELECT
        self.drag_source = None
        self.drag_plane_z = None
        self.drag_constraint = None
        self.end_drag_calls = 0
        self.clear_gizmo_calls = 0
        self.show_gizmo_calls = []

    def reconcile(self, snapshot):
        self.snapshots.append(snapshot)

    def set_tool(self, tool):
        self.tool = tool

    def set_camera_mode(self, mode):
        self.camera_mode = mode

    def begin_drag_plane(self, z, source):
        self.drag_plane_z = z
        self.drag_source = source

    def begin_control_drag(self, constraint, source):
        self.drag_constraint = constraint
        self.begin_drag_plane(source.world_position[2], source)

    def end_drag(self):
        self.end_drag_calls += 1
        self.drag_plane_z = None
        self.drag_source = None

    def clear_transform_gizmo(self):
        self.clear_gizmo_calls += 1

    def show_transform_gizmo(self, actor_id):
        self.show_gizmo_calls.append(actor_id)
        return True

    def focus_actor(self, _actor_id):
        return True

    def fit_all(self):
        return True


class _Viewport(QWidget):
    input_received = Signal(object)

    def __init__(self, _visualizer, parent=None):
        super().__init__(parent)
        self.port = _Port()

    def close_viewport(self):
        return None


class _RouterRuntime:
    """Small semantic-gizmo runtime used for workspace/router integration."""

    def __init__(self) -> None:
        self.callback = None
        self.attached_object_id = None
        self.attachments = []
        self.poses = {}
        self.sync_calls = []
        self.gizmo = object()

    def add_event_handler(self, _callback, *_event_types) -> None:
        return None

    def remove_event_handler(self, _callback, *_event_types) -> None:
        return None

    def ensure_gizmo(self, *, authoring=True):
        return self.gizmo

    def hide_gizmo(self) -> None:
        self.callback = None
        self.attached_object_id = None

    def set_camera_mode(self, _mode):
        return True

    def attach_gizmo(self, object_id, callback) -> bool:
        pose = self.poses.get(object_id)
        if pose is None:
            return False
        self.callback = callback
        self.attached_object_id = object_id
        self.attachments.append(object_id)
        callback({"phase": "selected", "object_id": object_id, "transform": pose})
        return True

    def sync_gizmo_pose(self, object_id, transform) -> bool:
        self.sync_calls.append((object_id, np.asarray(transform, dtype=float).copy()))
        return object_id == self.attached_object_id


class _RouterRenderer:
    def __init__(self) -> None:
        self.runtime = _RouterRuntime()

    def scenario_authoring_runtime(self):
        return self.runtime


class _RouterPort(_Port):
    def __init__(self, sink) -> None:
        super().__init__()
        self.renderer = _RouterRenderer()
        self.runtime = self.renderer.runtime
        self.router = PygfxInteractionRouter(self.renderer)
        self.router.activate(InteractionSession.AUTHORING, sink)
        self.document_id = None

    def reconcile(self, snapshot):
        super().reconcile(snapshot)
        self.document_id = snapshot.document_id
        for actor in snapshot.actors:
            pose = (
                np.asarray(actor.orientation_matrix, dtype=float).copy()
                if actor.orientation_matrix is not None
                else np.eye(4, dtype=float)
            )
            position = actor.current_position or actor.positions[0]
            pose[:3, 3] = position
            object_id = stable_renderer_id(
                snapshot.document_id,
                actor.actor_id,
                "mobility_handles",
            )
            self.runtime.poses[object_id] = pose
            self.router.sync_transform_pose(object_id, pose)

    def set_tool(self, tool):
        super().set_tool(tool)
        self.router.set_tool(tool)

    def set_camera_mode(self, mode):
        return self.router.set_camera_mode(mode)

    def clear_transform_gizmo(self):
        super().clear_transform_gizmo()
        self.router.clear_transform_gizmo()

    def show_transform_gizmo(self, actor_id):
        self.show_gizmo_calls.append(actor_id)
        if self.document_id is None:
            return False
        return self.router.attach_transform_gizmo(
            stable_renderer_id(
                self.document_id,
                actor_id,
                "mobility_handles",
            )
        )

    def close(self) -> None:
        self.router.deactivate()


class _RouterViewport(QWidget):
    input_received = Signal(object)

    def __init__(self, _visualizer, parent=None):
        super().__init__(parent)
        self.port = _RouterPort(self.input_received.emit)

    def close_viewport(self):
        self.port.close()


_OWNED_WORKSPACES: list[ScenarioAuthoringWorkspace] = []


def _track_workspace(workspace: ScenarioAuthoringWorkspace) -> ScenarioAuthoringWorkspace:
    """Register one complete workspace for deterministic per-test teardown."""
    _OWNED_WORKSPACES.append(workspace)
    return workspace


@pytest.fixture(autouse=True)
def dispose_owned_workspaces():
    """Close authoring services and delete only roots created by the current test."""
    start = len(_OWNED_WORKSPACES)
    yield
    failures = []
    for workspace in reversed(_OWNED_WORKSPACES[start:]):
        visualizer = getattr(workspace, "visualizer", None)
        try:
            workspace.close_workspace()
        except Exception as exc:
            failures.append(exc)
        finally:
            for root in (workspace, visualizer):
                if not isinstance(root, QWidget) or not isValid(root):
                    continue
                root.close()
                root.deleteLater()
                QCoreApplication.sendPostedEvents(root, QEvent.Type.DeferredDelete)
    del _OWNED_WORKSPACES[start:]
    if failures:
        raise failures[0]


def _workspace(document: ScenarioDocument, compiler=None) -> ScenarioAuthoringWorkspace:
    return _track_workspace(
        ScenarioAuthoringWorkspace(
            QWidget(),
            document,
            compiler=compiler or _Compiler(),
            viewport_factory=_Viewport,
        )
    )


def _router_workspace(
    document: ScenarioDocument,
    compiler=None,
) -> ScenarioAuthoringWorkspace:
    return _track_workspace(
        ScenarioAuthoringWorkspace(
            QWidget(),
            document,
            compiler=compiler or _Compiler(),
            viewport_factory=_RouterViewport,
        )
    )


def _place(workspace, point, phase=PointerPhase.DOWN) -> None:
    workspace._on_viewport_input(PointerInput(phase, (10.0, 20.0), button=1))
    workspace._on_viewport_input(HitResult(point))


def test_workspace_exposes_scene_catalog_tree_and_work_plane_controls(
    qapp,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "available_sionna_scene_ids",
        lambda: ("box", "etoile"),
    )
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TARGET)
    workspace = _workspace(document)
    workspace.refresh_now()

    assert {
        workspace.scene_source_combo.itemData(index)
        for index in range(workspace.scene_source_combo.count())
    } == {"sionna", "library", "local"}
    assert workspace.scene_source_combo.currentData() == "library"
    assert workspace.scene_id_combo.currentText() == "empty/empty.xml"
    assert "ORCHAV library" in workspace.scene_id_combo.lineEdit().placeholderText()

    workspace.scene_source_combo.setCurrentIndex(workspace.scene_source_combo.findData("sionna"))
    assert workspace.scene_id_combo.count() == 2
    assert workspace.scene_id_combo.findText("box") >= 0
    assert workspace.scene_id_combo.findText("etoile") >= 0
    assert workspace.scene_id_combo.currentIndex() == -1
    assert "Choose a Sionna" in workspace.scene_id_combo.lineEdit().placeholderText()
    assert workspace.target_asset_combo.findData("cube") >= 0
    assert workspace.work_plane_check.isChecked()
    assert workspace.grid_snap_check.isChecked() is False
    assert workspace.camera_mode_combo.isEnabled()
    workspace.camera_mode_combo.setCurrentIndex(workspace.camera_mode_combo.findData("fly"))
    assert workspace.viewport.port.camera_mode == "fly"

    item = workspace.actor_tree.topLevelItem(0)
    item.setCheckState(2, Qt.Unchecked)
    item.setCheckState(3, Qt.Checked)
    assert document.scenario.actor(actor.id).visible is False
    assert document.scenario.actor(actor.id).locked is True

    workspace.scene_source_combo.setCurrentIndex(workspace.scene_source_combo.findData("library"))
    assert workspace.scene_id_combo.findText("empty/empty.xml") >= 0
    assert "ORCHAV library" in workspace.scene_id_combo.lineEdit().placeholderText()
    workspace.scene_id_combo.setEditText("empty/empty.xml")
    workspace._apply_scene()
    assert document.scenario.scene.source == "library"
    assert document.scenario.scene.id == "empty/empty.xml"

    workspace.grid_snap_check.setChecked(True)
    workspace.grid_snap_spacing_spin.setValue(0.5)
    workspace._refresh_viewport()
    snapshot = workspace.viewport.port.snapshots[-1]
    assert snapshot.work_plane_visible is True
    assert snapshot.grid_snap_m == 0.5

    prior_snapshot_count = len(workspace.viewport.port.snapshots)
    workspace._timeline_scrubbed(1)
    assert len(workspace.viewport.port.snapshots) == prior_snapshot_count + 1

    workspace._add_actor(ActorRole.TX)
    assert document.selected_actor.role is ActorRole.TX
    assert workspace._tool is AuthoringTool.PLACE
    workspace.close_workspace()


def test_default_timeline_playback_uses_authored_sample_spacing(qapp) -> None:
    workspace = _workspace(ScenarioDocument.new())

    assert workspace.timeline_slider.maximum() == 29
    assert workspace._play_timer.interval() == round(3_000 / 29)
    assert workspace.play_button.isEnabled()

    workspace._toggle_playback()
    assert workspace._play_timer.isActive()
    workspace.close_workspace()


def test_imported_custom_quality_is_preserved_until_explicit_replacement(qapp) -> None:
    tx = AuthoringActor.create(ActorRole.TX, "TX1")
    rx = AuthoringActor.create(ActorRole.RX, "RX1", position=(2.0, 0.0, 0.0))
    custom = {"max_depth": 4, "samples_per_src": 75_000}
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            timeline=TimelineSettings(
                steps=30,
                duration_s=3.0,
                quality=QualityPreset.HIGH,
            ),
            actors=(tx, rx),
            source_snapshot=ScenarioSourceSnapshot.from_mapping(
                {"raytracing": {"quality": {"custom": custom}}}
            ),
        )
    )
    workspace = _workspace(document)

    assert workspace.quality_combo.currentText() == ("high + custom overrides (preserved)")
    assert workspace.quality_combo.findData(QualityPreset.CUSTOM) == -1
    assert "not editable here" in workspace.preserved_settings_label.text()

    workspace.steps_spin.setValue(40)
    assert workspace.has_pending_timeline_edits
    document.select(tx.id)
    assert workspace.steps_spin.value() == 40
    assert document.scenario.timeline.steps == 30

    assert workspace._apply_timeline()
    assert document.scenario.timeline.steps == 40
    assert (
        document.scenario.source_snapshot.to_mapping()["raytracing"]["quality"]["custom"] == custom
    )

    workspace.quality_combo.setCurrentIndex(workspace.quality_combo.findData(QualityPreset.LOW))
    assert workspace.has_pending_timeline_edits
    assert "will replace" in workspace.timeline_pending_label.text()
    assert workspace._apply_timeline()
    assert document.scenario.timeline.quality is QualityPreset.LOW
    assert not document.scenario.source_snapshot.has_path("raytracing.quality.custom")

    document.undo()
    assert document.scenario.timeline.quality is QualityPreset.HIGH
    assert document.scenario.source_snapshot.has_path("raytracing.quality.custom")
    assert workspace.quality_combo.currentText() == ("high + custom overrides (preserved)")
    workspace.close_workspace()


def test_new_document_does_not_offer_empty_custom_quality(qapp) -> None:
    workspace = _workspace(ScenarioDocument.new())

    assert workspace.quality_combo.findData(QualityPreset.CUSTOM) == -1
    assert "custom" not in {
        workspace.quality_combo.itemText(index) for index in range(workspace.quality_combo.count())
    }
    workspace.close_workspace()


def test_timeline_draft_apply_discard_and_cancel_are_explicit(
    qapp,
    monkeypatch,
) -> None:
    document = ScenarioDocument.new()
    workspace = _workspace(document)

    workspace.steps_spin.setValue(45)
    assert workspace.has_pending_timeline_edits
    workspace._reset_pending_timeline()
    assert workspace.steps_spin.value() == 30
    assert not workspace.has_pending_timeline_edits

    workspace.steps_spin.setValue(40)
    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: workspace_module.QMessageBox.Apply,
    )
    assert workspace.commit_pending_edits()
    assert document.scenario.timeline.steps == 40

    workspace.steps_spin.setValue(50)
    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: workspace_module.QMessageBox.Discard,
    )
    assert workspace.commit_pending_edits()
    assert document.scenario.timeline.steps == 40
    assert workspace.steps_spin.value() == 40

    workspace.steps_spin.setValue(60)
    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: workspace_module.QMessageBox.Cancel,
    )
    assert not workspace.commit_pending_edits()
    assert document.scenario.timeline.steps == 40
    assert workspace.steps_spin.value() == 60
    assert workspace.has_pending_timeline_edits
    workspace._reset_pending_timeline()
    workspace.close_workspace()


def test_sampling_summary_distinguishes_waypoints_from_generator_frames(qapp) -> None:
    actor = (
        ScenarioDocument.new()
        .add_default_actor(ActorRole.RX)
        .with_changes(
            mobility=WaypointMobilitySpec(
                points_m=(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (2.0, 1.0, 0.0),
                    (3.0, 1.0, 0.0),
                )
            )
        )
    )
    document = ScenarioDocument(
        AuthoringScenario(
            timeline=TimelineSettings(steps=6, duration_s=3.0),
            actors=(actor,),
        )
    )
    document.select(actor.id)
    workspace = _workspace(document)

    assert "6 generator frame samples" in workspace.sampling_summary_label.text()
    assert "4 authored waypoints" in workspace.sampling_summary_label.text()
    assert "#e0a020" in workspace.sampling_summary_label.styleSheet()

    document.set_timeline(TimelineSettings(steps=7, duration_s=3.0))
    assert "7 generator frame samples" in workspace.sampling_summary_label.text()
    assert "do not all coincide" not in workspace.sampling_summary_label.text()
    assert workspace.sampling_summary_label.styleSheet() == ""
    workspace.close_workspace()


def test_tx_only_circular_preview_uses_generator_samples_before_rx_exists(qapp) -> None:
    actor = (
        ScenarioDocument.new()
        .add_default_actor(ActorRole.TX)
        .with_changes(
            mobility=CircularMobilitySpec(
                center_m=(1.0, 2.0, 3.0),
                radius_m=4.0,
                start_angle_deg=math.degrees(0.25),
                clockwise=False,
            )
        )
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    workspace = _workspace(document, ScenarioCompiler(_Compiler.project_root))
    workspace.refresh_now()

    assert workspace.compilation is not None
    assert workspace.compilation.valid is False
    assert any(issue.code == "actors.rx.required" for issue in workspace.compilation.issues)
    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert len(overlay.positions) == document.scenario.timeline.steps == 30
    assert overlay.positions == workspace.compilation.samples[actor.id].positions
    assert overlay.frame_samples == workspace.compilation.samples[actor.id].positions
    assert overlay.closed_trajectory is True
    workspace.close_workspace()


def test_async_edit_never_presents_previous_circular_samples_as_current(qapp) -> None:
    actor = (
        ScenarioDocument.new()
        .add_default_actor(ActorRole.TX)
        .with_changes(
            mobility=CircularMobilitySpec(
                center_m=(0.0, 0.0, 1.0),
                radius_m=2.0,
                start_angle_deg=0.0,
                clockwise=False,
            )
        )
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    old_samples = ActorSamples(
        positions=((2.0, 0.0, 1.0), (0.0, 2.0, 1.0)),
        orientations=((0.0, 0.0, 0.0),) * 2,
    )
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "schema_version: 2\n", (), {actor.id: old_samples})),
    )
    workspace.refresh_now()

    current = document.scenario.actor(actor.id)
    document.replace_actor(
        current.with_changes(
            mobility=CircularMobilitySpec(
                center_m=(0.0, 0.0, 1.0),
                radius_m=5.0,
                start_angle_deg=0.0,
                clockwise=False,
            )
        )
    )

    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.positions == ((5.0, 0.0, 1.0),)
    assert overlay.frame_samples == ()
    assert overlay.current_position == (5.0, 0.0, 1.0)
    assert overlay.preview_provenance is PreviewProvenance.AUTHORED_DRAFT
    workspace.close_workspace()


def test_async_edit_marks_previous_validation_evidence_as_pending(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    document.select(actor.id)
    previous_issue = ValidationIssue(
        severity=IssueSeverity.ERROR,
        code="test.previous",
        path="devices.tx.0.mobility",
        message="Previous draft problem",
        actor_id=actor.id,
    )
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "previous: true\n", (previous_issue,), {})),
    )
    workspace.refresh_now()
    assert "Previous draft problem" in workspace.inline_errors.text()

    current = document.scenario.actor(actor.id)
    document.replace_actor(
        current.with_changes(
            mobility=LinearMobilitySpec(
                start_m=(0.0, 0.0, 0.0),
                end_m=(2.0, 0.0, 0.0),
            )
        )
    )
    workspace._refresh_tree()

    assert workspace.yaml_preview.toPlainText() == "# Validating the current draft...\n"
    assert workspace.problems_tree.topLevelItemCount() == 1
    assert workspace.problems_tree.topLevelItem(0).text(2) == "Validation in progress"
    assert workspace.inline_errors.text() == ""
    assert workspace.actor_tree.topLevelItem(0).text(4) == "Validating"
    assert workspace.viewport.port.snapshots[-1].actors[0].status is ActorVisualState.PENDING
    workspace.close_workspace()


def test_path_change_invalidates_directory_relative_compiler_evidence(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    document.select(actor.id)
    samples = ActorSamples(
        positions=((0.0, 0.0, 0.0),),
        orientations=((0.0, 0.0, 0.0),),
    )
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "previous: true\n", (), {actor.id: samples})),
    )
    workspace.refresh_now()
    assert workspace._compilation_is_current() is True
    assert workspace.viewport.port.snapshots[-1].actors[0].frame_samples == samples.positions

    moved_path = Path(__file__).parent / "moved-scenario" / "scenario.yaml"
    document.mark_saved(moved_path)
    workspace._refresh_tree()

    assert workspace._compilation_is_current() is False
    assert workspace.actor_tree.topLevelItem(0).text(4) == "Validating"
    assert workspace.viewport.port.snapshots[-1].actors[0].frame_samples == ()
    workspace.close_workspace()


def test_background_compile_failure_clears_previous_canonical_evidence(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    samples = ActorSamples(
        positions=((0.0, 0.0, 0.0),),
        orientations=((0.0, 0.0, 0.0),),
    )
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "previous: true\n", (), {actor.id: samples})),
    )
    workspace.refresh_now()

    workspace._background_compile_failed(
        workspace._compile_request_token,
        CompilationFailure("builtins.RuntimeError", "preview exploded"),
    )

    assert workspace.compilation is None
    assert workspace._compiled_scenario is None
    assert workspace._prepared_sample_cache == {}
    assert "Compilation failed" in workspace.yaml_preview.toPlainText()
    assert workspace.viewport.port.snapshots[-1].actors[0].frame_samples == ()
    assert workspace.actor_tree.topLevelItem(0).text(4) == "Invalid"
    workspace.close_workspace()


def test_workspace_coalesces_generator_compilation_without_blocking_actor_creation(
    qapp,
    monkeypatch,
) -> None:
    class _SlowScenarioCompiler:
        project_root = _Compiler.project_root
        started = threading.Event()
        release = threading.Event()
        calls: list[int] = []

        def __init__(self, _project_root=None) -> None:
            return None

        def compile(self, scenario, *, scenario_directory=None):
            actor_count = len(scenario.actors)
            type(self).calls.append(actor_count)
            if not type(self).release.is_set():
                type(self).started.set()
                if not type(self).release.wait(timeout=2.0):
                    raise TimeoutError("test did not release background compilation")
            return CompilationResult({}, f"actors: {actor_count}\n", (), {})

    monkeypatch.setattr(workspace_module, "ScenarioCompiler", _SlowScenarioCompiler)
    document = ScenarioDocument.new()
    workspace = _track_workspace(
        ScenarioAuthoringWorkspace(
            QWidget(),
            document,
            compiler=_SlowScenarioCompiler(),
            viewport_factory=_Viewport,
        )
    )
    assert _SlowScenarioCompiler.started.wait(timeout=1.0)

    started_at = time.monotonic()
    document.add_default_actor(ActorRole.TX)
    document.add_default_actor(ActorRole.RX)
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert len(document.actors) == 2
    _SlowScenarioCompiler.release.set()
    deadline = time.monotonic() + 2.0
    scheduler = workspace._compile_scheduler
    assert scheduler is not None
    while scheduler.running and time.monotonic() < deadline:
        time.sleep(0.005)
    assert scheduler.running is False
    assert _SlowScenarioCompiler.calls == [0, 2]
    workspace.close_workspace()


def test_adaptive_orientation_editor_seeds_visible_pose_and_applies_once(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    document.replace_actor(
        actor.with_changes(
            mobility=LinearMobilitySpec(
                start_m=(0.0, 0.0, 1.0),
                end_m=(4.0, 0.0, 1.0),
            ),
            orientation=FixedOrientationSpec(
                yaw_deg=37.0,
                pitch_deg=-4.0,
                roll_deg=2.0,
            ),
        )
    )
    document.set_timeline(TimelineSettings(steps=2, duration_s=1.0))
    samples = ActorSamples(
        positions=((0.0, 0.0, 1.0), (4.0, 0.0, 1.0)),
        orientations=((37.0, -4.0, 2.0), (83.0, 6.0, -3.0)),
    )
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "schema_version: 2\n", (), {actor.id: samples})),
    )
    workspace.refresh_now()
    workspace._timeline_scrubbed(1)
    command_count = document.undo_stack.count
    original = document.scenario.actor(actor.id).orientation

    editor = workspace.orientation_editor
    assert editor.type_combo.count() == 6
    assert editor.fixed_angle_spins[0].stepType() == QAbstractSpinBox.AdaptiveDecimalStepType

    editor.type_combo.setCurrentIndex(editor.type_combo.findData("spin"))
    assert editor.spin_start_yaw_spin.value() == pytest.approx(83.0)
    assert editor.spin_pitch_spin.value() == pytest.approx(6.0)
    assert editor.spin_roll_spin.value() == pytest.approx(-3.0)
    editor.spin_rotations_spin.setValue(2.5)
    editor.apply_button.click()

    updated = document.scenario.actor(actor.id)
    expected = SpinOrientationSpec(
        axis="yaw",
        rate_deg_s=2.5,
        yaw_deg=83.0,
        pitch_deg=6.0,
        roll_deg=-3.0,
    )
    assert updated.orientation == expected
    assert document.undo_stack.count == command_count + 1
    document.undo()
    assert document.scenario.actor(actor.id).orientation == original
    document.redo()
    assert document.scenario.actor(actor.id).orientation == expected

    editor.type_combo.setCurrentIndex(editor.type_combo.findData("keyframes"))
    assert editor.orientation() == KeyframesOrientationSpec(
        keyframes=(
            OrientationKeyframeSpec(
                time_s=0.0,
                yaw_deg=83.0,
                pitch_deg=6.0,
                roll_deg=-3.0,
            ),
            OrientationKeyframeSpec(
                time_s=1.0,
                yaw_deg=83.0,
                pitch_deg=6.0,
                roll_deg=-3.0,
            ),
        )
    )
    workspace.close_workspace()


def test_selection_rebind_drops_stale_look_at_self_reference(qapp) -> None:
    document = ScenarioDocument.new()
    first = document.add_default_actor(ActorRole.TX)
    second = document.add_default_actor(ActorRole.RX)
    document.replace_actor(first.with_changes(orientation=actor_look_at_orientation(second.id)))
    document.select(first.id)
    workspace = _workspace(document)

    assert workspace.orientation_editor.look_at_combo.currentData() == second.id
    document.select(second.id)
    editor = workspace.orientation_editor
    choice_ids = {
        str(editor.look_at_combo.itemData(index)) for index in range(editor.look_at_combo.count())
    }
    assert str(first.id) in choice_ids
    editor.type_combo.setCurrentIndex(editor.type_combo.findData("look_at"))
    editor.look_at_target_mode_combo.setCurrentIndex(
        editor.look_at_target_mode_combo.findData("actor")
    )

    assert str(second.id) not in choice_ids
    first_index = next(
        index
        for index in range(editor.look_at_combo.count())
        if str(editor.look_at_combo.itemData(index)) == str(first.id)
    )
    editor.look_at_combo.setCurrentIndex(first_index)
    editor.apply_button.click()
    assert document.scenario.actor(second.id).orientation == actor_look_at_orientation(first.id)
    workspace.close_workspace()


def test_pending_look_at_keeps_uuid_and_recompiles_after_document_content_change(
    qapp,
) -> None:
    class _RecordingCompiler(_Compiler):
        def __init__(self) -> None:
            self.scenarios = []

        def compile(self, scenario, *, scenario_directory=None):
            self.scenarios.append(scenario)
            return CompilationResult({}, "schema_version: 2\n", (), {})

    owner = AuthoringActor.create(ActorRole.TX, "Observer")
    target = AuthoringActor.create(ActorRole.RX, "Reference")
    document = ScenarioDocument(AuthoringScenario(actors=(owner, target)))
    document.select(owner.id)
    compiler = _RecordingCompiler()
    workspace = _workspace(document, compiler)
    editor = workspace.orientation_editor
    target_index = next(
        index
        for index in range(editor.look_at_combo.count())
        if editor.look_at_combo.itemData(index) == target.id
    )
    editor.look_at_combo.setCurrentIndex(target_index)
    editor.type_combo.setCurrentIndex(editor.type_combo.findData("look_at"))
    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()
    assert workspace._candidate_scenario is not None

    document.rename_actor(target.id, "Renamed Reference")

    assert workspace.has_pending_inspector_edits
    assert editor.look_at_combo.currentData() == target.id
    assert look_at_actor_id(editor.orientation()) == target.id
    assert workspace._candidate_scenario is None
    assert workspace._candidate_compile_timer.isActive()

    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()
    candidate = compiler.scenarios[-1]
    assert candidate.actor(target.id).name == "Renamed Reference"
    assert look_at_actor_id(candidate.actor(owner.id).orientation) == target.id
    workspace.close_workspace()


def test_undo_and_redo_require_pending_inspector_resolution(qapp, monkeypatch) -> None:
    actor = AuthoringActor.create(ActorRole.TX, "Original")
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    document.rename_actor(actor.id, "Renamed")
    workspace = _workspace(document)
    workspace.orientation_editor.fixed_angle_spins[0].setValue(15.0)
    assert workspace.has_pending_inspector_edits

    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args: workspace_module.QMessageBox.Cancel,
    )
    workspace._undo()
    assert document.scenario.actor(actor.id).name == "Renamed"
    assert workspace.has_pending_inspector_edits

    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args: workspace_module.QMessageBox.Discard,
    )
    workspace._undo()
    assert document.scenario.actor(actor.id).name == "Original"
    assert not workspace.has_pending_inspector_edits

    workspace.orientation_editor.fixed_angle_spins[0].setValue(25.0)
    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args: workspace_module.QMessageBox.Cancel,
    )
    workspace._redo()
    assert document.scenario.actor(actor.id).name == "Original"
    assert workspace.has_pending_inspector_edits

    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args: workspace_module.QMessageBox.Discard,
    )
    workspace._redo()
    assert document.scenario.actor(actor.id).name == "Renamed"
    assert not workspace.has_pending_inspector_edits
    workspace.close_workspace()


def test_selection_passes_actor_and_group_role_context_to_mobility_editor(
    qapp,
    monkeypatch,
) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    group = document.add_default_group("Convoy")
    workspace = _workspace(document)
    observed: list[ActorRole | None] = []
    original = workspace.mobility_editor.set_actor_role

    def record(role: ActorRole | str | None) -> None:
        observed.append(None if role is None else ActorRole(role))
        original(role)

    monkeypatch.setattr(workspace.mobility_editor, "set_actor_role", record)

    document.select(actor.id)
    document.select_group(group.id)
    document.select(None)

    assert observed == [ActorRole.TX, ActorRole.TX, None]
    workspace.close_workspace()


def test_no_selection_and_read_only_import_clear_orientation_inspector(qapp) -> None:
    document = ScenarioDocument.new()
    first = document.add_default_actor(ActorRole.TX)
    second = document.add_default_actor(ActorRole.RX)
    document.replace_actor(first.with_changes(orientation=actor_look_at_orientation(second.id)))
    document.select(first.id)
    workspace = _workspace(document)
    assert workspace.orientation_editor.type_combo.currentData() == "look_at"

    document.select(None)
    assert workspace.name_edit.text() == ""
    assert workspace.orientation_editor.type_combo.currentData() == "fixed"
    assert workspace.orientation_editor.look_at_combo.count() == 0
    assert not workspace.orientation_editor.type_combo.isEnabled()

    document.select(first.id)
    workspace.show_read_only_import(
        SimpleNamespace(
            issues=(),
            raw_text="schema_version: 1\nunsupported: true\n",
            source_path=Path("read-only") / "scenario.yaml",
        )
    )
    assert workspace.document is None
    assert workspace.viewport.port.snapshots[-1].actors == ()
    assert workspace.viewport.port.snapshots[-1].scene_assets == ()
    assert workspace.viewport.port.snapshots[-1].work_plane_visible is False
    assert workspace.orientation_editor.type_combo.currentData() == "fixed"
    assert workspace.orientation_editor.look_at_combo.count() == 0
    assert not workspace.orientation_editor.type_combo.isEnabled()
    workspace.close_workspace()


def test_explicit_validation_shows_progress_then_success(qapp) -> None:
    observed_states = []

    class _ObservingCompiler(_Compiler):
        workspace = None

        def compile(self, _document, *, scenario_directory=None):
            observed_states.append(
                (
                    self.workspace.validation_status.text(),
                    self.workspace.validate_button.isEnabled(),
                )
            )
            return CompilationResult({}, "schema_version: 2\n", (), {})

    compiler = _ObservingCompiler()
    workspace = _workspace(ScenarioDocument.new(), compiler)
    compiler.workspace = workspace

    QTest.mouseClick(workspace.validate_button, Qt.LeftButton)

    assert observed_states == [("Validating current draft...", False)]
    assert workspace.validation_status.text() == "Validation passed: no problems found."
    assert workspace.validate_button.isEnabled()
    workspace.close_workspace()


def test_explicit_validation_reports_errors_and_warnings(qapp) -> None:
    issues = (
        ValidationIssue(
            IssueSeverity.ERROR,
            "test.error",
            "actors.rx",
            "Receiver is invalid",
        ),
        ValidationIssue(
            IssueSeverity.WARNING,
            "test.warning",
            "scene",
            "Scene warning",
        ),
    )
    workspace = _workspace(
        ScenarioDocument.new(),
        _FixedCompiler(CompilationResult({}, "schema_version: 2\n", issues, {})),
    )

    workspace.validate_current_draft()

    assert workspace.validation_status.text() == "Validation failed: 1 error, 1 warning."
    assert workspace.problems_tree.topLevelItemCount() == 2
    workspace.close_workspace()


def test_explicit_validation_distinguishes_generation_problems(qapp) -> None:
    generation_issue = ValidationIssue(
        IssueSeverity.ERROR,
        "test.generate",
        "raytracing.enabled",
        "Generation is unavailable",
    )
    result = CompilationResult(
        {},
        "schema_version: 2\n",
        (),
        {},
        generation_issues=(generation_issue,),
    )
    workspace = _workspace(ScenarioDocument.new(), _FixedCompiler(result))

    workspace.validate_current_draft()

    assert workspace.validation_status.text() == ("Validation passed, but Generate has 1 problem.")
    workspace.close_workspace()


def test_explicit_validation_surfaces_unexpected_compiler_failure(qapp) -> None:
    class _FailingCompiler(_Compiler):
        def compile(self, _document, *, scenario_directory=None):
            raise RuntimeError("compiler exploded")

    workspace = _workspace(ScenarioDocument.new(), _FailingCompiler())

    workspace.validate_current_draft()

    assert workspace.validation_status.text().endswith("builtins.RuntimeError: compiler exploded")
    assert workspace.problems_tree.topLevelItemCount() == 1
    assert workspace.problems_tree.topLevelItem(0).text(1) == "compiler"
    workspace.close_workspace()


def test_sampled_mobility_opens_as_a_locked_inspector_facet(qapp) -> None:
    scenario = scenario_from_mapping(
        {
            "schema_version": 2,
            "scene": {"source": "library", "id": "empty/empty.xml"},
            "timeline": {"steps": 3, "duration_s": 2.0},
            "raytracing": {"enabled": True},
            "actors": {
                "tx": [
                    {
                        "name": "TX1",
                        "mobility": {
                            "type": "sampled",
                            "positions_m": [[0, 0, 1], [1, 0, 1], [3, 0, 1]],
                        },
                    }
                ],
                "rx": [
                    {
                        "name": "RX1",
                        "mobility": {"type": "stationary", "position_m": [5, 0, 1]},
                    }
                ],
            },
        }
    )
    document = ScenarioDocument(scenario)
    tx = document.scenario.actor_by_name("TX1")
    assert tx is not None
    assert isinstance(tx.mobility, SampledMobilitySpec)

    workspace = _workspace(document)
    document.select(tx.id)

    assert workspace.mobility_editor.mobility() is tx.mobility
    assert workspace.mobility_editor.type_combo.currentData() == "sampled"
    assert workspace.mobility_editor.page_stack.currentWidget() is (
        workspace.mobility_editor.read_only_page
    )
    assert not workspace.mobility_editor.type_combo.isEnabled()
    assert not workspace.mobility_editor.apply_button.isEnabled()
    assert not workspace._mobility_translation_supported(tx.mobility)
    workspace.close_workspace()


def test_two_step_ten_second_timeline_uses_ten_second_interval(qapp) -> None:
    scenario = AuthoringScenario(timeline=TimelineSettings(steps=2, duration_s=10.0))
    workspace = _workspace(ScenarioDocument(scenario))

    assert workspace.timeline_slider.maximum() == 1
    assert workspace._play_timer.interval() == 10_000

    workspace._advance_playback()
    assert workspace.timeline_slider.value() == 1
    workspace.close_workspace()


def test_timeline_edit_updates_active_playback_interval(qapp) -> None:
    document = ScenarioDocument.new()
    workspace = _workspace(document)
    workspace._toggle_playback()
    assert workspace._play_timer.isActive()

    document.set_timeline(TimelineSettings(steps=5, duration_s=2.0))

    assert workspace.timeline_slider.maximum() == 4
    assert workspace._play_timer.interval() == 500
    assert workspace._play_timer.isActive()
    workspace.close_workspace()


def test_zero_duration_static_timeline_cannot_start_busy_playback(qapp) -> None:
    document = ScenarioDocument.new()
    document.add_default_actor(ActorRole.TX)
    document.add_default_actor(ActorRole.RX)
    document.set_timeline(TimelineSettings(steps=30, duration_s=0.0))
    workspace = _workspace(document)

    assert not workspace.play_button.isEnabled()
    workspace._toggle_playback()
    assert not workspace._play_timer.isActive()

    document.set_timeline(TimelineSettings(steps=1_000_000, duration_s=0.001))
    assert workspace._play_timer.interval() == 1
    assert workspace.play_button.isEnabled()
    assert (
        workspace._playback_interval_ms(TimelineSettings(steps=2, duration_s=1e308))
        == 2_147_483_647
    )
    workspace.close_workspace()


def test_target_material_inspector_uses_generator_registry_choices(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TARGET)
    workspace = _workspace(document)

    choices = tuple(
        str(workspace.target_material_combo.itemData(index))
        for index in range(workspace.target_material_combo.count())
    )
    assert choices == available_target_material_types()
    assert workspace.target_material_combo.isEditable() is False

    workspace.target_asset_combo.setCurrentIndex(workspace.target_asset_combo.findData("cube"))
    workspace.target_material_combo.setCurrentIndex(
        workspace.target_material_combo.findData("concrete")
    )
    workspace._apply_target()

    updated = document.scenario.actor(actor.id)
    assert updated is not None
    assert updated.target is not None
    assert updated.target.material == "concrete"
    workspace.close_workspace()


def test_mobility_type_change_keeps_stationary_world_position(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    document.replace_actor(
        actor.with_changes(mobility=StationaryMobilitySpec(position_m=(5.0, 5.0, 5.0)))
    )
    workspace = _workspace(document)

    workspace.mobility_editor.type_combo.setCurrentIndex(
        workspace.mobility_editor.type_combo.findData("circular")
    )

    preview = workspace.mobility_editor.mobility()
    assert isinstance(preview, CircularMobilitySpec)
    assert preview.center_m == (0.0, 5.0, 5.0)
    assert preview.radius_m == 5.0
    assert preview.start_angle_deg == 0.0
    workspace._apply_mobility()
    updated = document.scenario.actor(actor.id).mobility
    assert isinstance(updated, CircularMobilitySpec)
    assert (
        updated.center_m[0] + updated.radius_m,
        updated.center_m[1],
        updated.center_m[2],
    ) == (5.0, 5.0, 5.0)
    workspace.close_workspace()


def test_apply_mobility_exits_pending_placement_without_resetting_model(qapp) -> None:
    document = ScenarioDocument.new()
    workspace = _workspace(document)
    workspace._add_actor(ActorRole.RX)
    actor = document.selected_actor
    assert actor is not None
    assert workspace._tool is AuthoringTool.PLACE

    editor = workspace.mobility_editor
    editor.type_combo.setCurrentIndex(editor.type_combo.findData("linear"))
    editor.linear_end_spins[0].setValue(4.0)
    editor.apply_button.click()

    applied = document.scenario.actor(actor.id).mobility
    assert isinstance(applied, LinearMobilitySpec)
    assert workspace._tool is AuthoringTool.SELECT

    _place(workspace, (9.0, 8.0, 7.0))
    assert document.scenario.actor(actor.id).mobility == applied
    workspace.close_workspace()


def test_mobility_type_change_uses_visible_prepared_timeline_pose(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    actor = actor.with_changes(
        mobility=LinearMobilitySpec(
            start_m=(1.0, 1.0, 1.0),
            end_m=(9.0, 9.0, 9.0),
        )
    )
    document.replace_actor(actor)
    samples = ActorSamples(
        positions=((1.0, 1.0, 1.0), (9.0, 9.0, 9.0)),
        orientations=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "schema_version: 2\n", (), {actor.id: samples})),
    )
    workspace.refresh_now()
    workspace._timeline_scrubbed(1)

    workspace.mobility_editor.type_combo.setCurrentIndex(
        workspace.mobility_editor.type_combo.findData("stationary")
    )

    assert workspace.mobility_editor.mobility() == StationaryMobilitySpec(
        position_m=(9.0, 9.0, 9.0)
    )
    workspace.close_workspace()


def test_contextual_waypoint_draw_starts_at_current_pose(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    stationary = StationaryMobilitySpec(position_m=(5.0, 5.0, 5.0))
    document.replace_actor(actor.with_changes(mobility=stationary))
    workspace = _workspace(document)
    command_count = document.undo_stack.count

    workspace._start_waypoint_rebuild()
    assert document.scenario.actor(actor.id).mobility == stationary
    _place(workspace, (8.0, 5.0, 5.0))
    workspace._handle_key(KeyboardInput("Enter", True))

    assert document.scenario.actor(actor.id).mobility == WaypointMobilitySpec(
        points_m=((5.0, 5.0, 5.0), (8.0, 5.0, 5.0))
    )
    assert document.undo_stack.count == command_count + 1
    workspace.close_workspace()


def test_waypoint_drawing_is_an_explicit_modal_session(qapp) -> None:
    document = ScenarioDocument.new()
    document.add_default_actor(ActorRole.TX)
    actor = document.add_default_actor(ActorRole.RX)
    stationary = StationaryMobilitySpec(position_m=(5.0, 5.0, 5.0))
    document.replace_actor(actor.with_changes(mobility=stationary))
    workspace = _workspace(document)
    command_count = document.undo_stack.count
    menu_save = QAction(workspace)
    menu_save_as = QAction(workspace)
    workspace.visualizer.workspace_mode_controller = SimpleNamespace(
        authoring_document=document,
        mode=SimpleNamespace(value="authoring"),
        workspace=workspace,
    )
    workspace.visualizer.authoring_generation_controller = SimpleNamespace(running=False)
    workspace.visualizer.save_authoring_scenario_action = menu_save
    workspace.visualizer.save_authoring_scenario_as_action = menu_save_as
    workspace.visualizer._refresh_authoring_actions = lambda: (
        OrchavVisualizer._refresh_authoring_actions(workspace.visualizer)
    )
    workspace.visualizer._refresh_authoring_actions()
    assert menu_save.isEnabled()
    assert menu_save_as.isEnabled()

    workspace._toggle_playback()
    assert workspace._play_timer.isActive()

    workspace._start_waypoint_rebuild()

    assert not workspace._play_timer.isActive()
    assert workspace.play_button.text() == "Play"
    assert not workspace.waypoint_session_bar.isHidden()
    assert workspace.waypoint_session_bar.isEnabled()
    assert "Drawing waypoints for RX1" in workspace.waypoint_session_label.text()
    assert "1 point" in workspace.waypoint_session_label.text()
    assert not workspace.waypoint_remove_last_button.isEnabled()
    assert not workspace.waypoint_finish_button.isEnabled()
    assert workspace.waypoint_cancel_button.isEnabled()
    assert not workspace.actor_tree.isEnabled()
    assert not workspace.inspector_panel.isEnabled()
    assert not workspace.drawer.isTabEnabled(0)
    assert not workspace.mobility_editor.apply_button.isEnabled()
    assert not workspace.save_button.isEnabled()
    assert not workspace.save_as_button.isEnabled()
    assert not workspace.generate_button.isEnabled()
    assert not workspace.leave_authoring_button.isEnabled()
    assert not menu_save.isEnabled()
    assert not menu_save_as.isEnabled()

    workspace.set_tool(AuthoringTool.MOVE)
    assert workspace._tool is AuthoringTool.WAYPOINT
    assert document.scenario.actor(actor.id).mobility == stationary
    workspace._finish_waypoint_session()
    assert workspace._tool is AuthoringTool.WAYPOINT
    assert not workspace.waypoint_session_bar.isHidden()

    _place(workspace, (8.0, 5.0, 5.0))
    assert "2 points" in workspace.waypoint_session_label.text()
    assert workspace.waypoint_remove_last_button.isEnabled()
    assert workspace.waypoint_finish_button.isEnabled()

    workspace.waypoint_finish_button.click()

    assert workspace.waypoint_session_bar.isHidden()
    assert workspace._tool is AuthoringTool.SELECT
    assert workspace.actor_tree.isEnabled()
    assert workspace.inspector_panel.isEnabled()
    assert workspace.drawer.isTabEnabled(0)
    assert workspace.save_button.isEnabled()
    assert workspace.save_as_button.isEnabled()
    assert workspace.generate_button.isEnabled()
    assert workspace.leave_authoring_button.isEnabled()
    assert menu_save.isEnabled()
    assert menu_save_as.isEnabled()
    assert not workspace.has_pending_waypoint_session
    assert document.scenario.actor(actor.id).mobility == WaypointMobilitySpec(
        points_m=((5.0, 5.0, 5.0), (8.0, 5.0, 5.0))
    )
    assert document.undo_stack.count == command_count + 1
    workspace.close_workspace()


def test_waypoint_apply_and_drawing_preserve_catmull_rom_interpolation(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    document.replace_actor(
        actor.with_changes(
            mobility=WaypointMobilitySpec(
                points_m=((0.0, 0.0, 1.0), (4.0, 2.0, 1.0), (8.0, 0.0, 1.0)),
            )
        )
    )
    document.select(actor.id)
    workspace = _workspace(document)

    workspace.mobility_editor.waypoint_interpolation_combo.setCurrentIndex(
        workspace.mobility_editor.waypoint_interpolation_combo.findData("catmull_rom")
    )
    assert workspace._apply_mobility()
    applied = document.scenario.actor(actor.id)
    assert isinstance(applied.mobility, WaypointMobilitySpec)
    assert applied.mobility.interpolation == "catmull_rom"

    workspace._start_waypoint_rebuild()
    _place(workspace, (5.0, 3.0, 1.0))
    drawing = document.scenario.actor(actor.id)
    assert isinstance(drawing.mobility, WaypointMobilitySpec)
    assert drawing.mobility.interpolation == "catmull_rom"

    workspace._finish_waypoint_session()
    committed = document.scenario.actor(actor.id)
    assert isinstance(committed.mobility, WaypointMobilitySpec)
    assert committed.mobility.interpolation == "catmull_rom"
    workspace.close_workspace()


def test_generation_locks_return_to_visualization_button(qapp) -> None:
    workspace = _workspace(ScenarioDocument.new())
    assert workspace.leave_authoring_button.isEnabled()

    workspace.set_generation_running(True, launched_revision=3)
    assert not workspace.leave_authoring_button.isEnabled()

    workspace.set_generation_running(False)
    assert workspace.leave_authoring_button.isEnabled()
    workspace.close_workspace()


def test_applied_catmull_rom_trajectory_uses_prepared_curve_positions() -> None:
    control_points = ((0.0, 0.0, 1.0), (4.0, 4.0, 1.0), (8.0, 0.0, 1.0))
    prepared_curve = (
        (0.0, 0.0, 1.0),
        (1.0, 1.375, 1.0),
        (2.0, 2.75, 1.0),
        (3.0, 3.75, 1.0),
        (4.0, 4.0, 1.0),
        (5.0, 3.75, 1.0),
        (6.0, 2.75, 1.0),
        (7.0, 1.375, 1.0),
        (8.0, 0.0, 1.0),
    )

    catmull_rom = WaypointMobilitySpec(
        points_m=control_points,
        interpolation="catmull_rom",
    )
    linear = WaypointMobilitySpec(points_m=control_points, interpolation="linear")

    assert (
        ScenarioAuthoringWorkspace._trajectory_positions(catmull_rom, prepared_curve)
        == prepared_curve
    )
    assert (
        ScenarioAuthoringWorkspace._trajectory_positions(linear, prepared_curve) == control_points
    )


def test_waypoint_session_remove_cancel_and_apply_are_transaction_safe(qapp, monkeypatch) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    document.replace_actor(
        actor.with_changes(mobility=StationaryMobilitySpec(position_m=(4.0, 3.0, 2.0)))
    )
    workspace = _workspace(document)
    original = document.scenario
    command_count = document.undo_stack.count
    warnings: list[str] = []
    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(str(message)),
    )

    workspace._start_waypoint_rebuild()
    _place(workspace, (5.0, 3.0, 2.0))
    _place(workspace, (6.0, 3.0, 2.0))
    assert len(document.scenario.actor(actor.id).mobility.points_m) == 3

    workspace.waypoint_remove_last_button.click()
    assert document.scenario.actor(actor.id).mobility == WaypointMobilitySpec(
        points_m=((4.0, 3.0, 2.0), (5.0, 3.0, 2.0))
    )

    workspace._apply_mobility()
    assert warnings == []
    assert document.undo_stack.count == command_count

    workspace.waypoint_cancel_button.click()
    assert document.scenario == original
    assert document.undo_stack.count == command_count
    assert workspace._tool is AuthoringTool.SELECT
    assert not workspace.has_pending_waypoint_session
    assert workspace.save_button.isEnabled()
    assert workspace.save_as_button.isEnabled()
    assert workspace.generate_button.isEnabled()
    assert workspace.leave_authoring_button.isEnabled()
    workspace.close_workspace()


def test_waypoint_shortcuts_work_with_focus_outside_viewport(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    workspace = _workspace(document)
    workspace.show()
    workspace.activateWindow()
    qapp.processEvents()

    workspace._start_waypoint_rebuild()
    _place(workspace, (1.0, 0.0, 0.0))
    _place(workspace, (2.0, 0.0, 0.0))
    _place(workspace, (3.0, 0.0, 0.0))
    workspace.generation_log.setFocus()
    qapp.processEvents()

    workspace._dispatch_waypoint_shortcut("remove")
    workspace._handle_key(KeyboardInput("Backspace", True))
    assert len(document.scenario.actor(actor.id).mobility.points_m) == 3
    qapp.processEvents()

    QTest.keyClick(workspace.generation_log, Qt.Key_Backspace)
    qapp.processEvents()
    assert document.scenario.actor(actor.id).mobility == WaypointMobilitySpec(
        points_m=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    )

    QTest.keyClick(workspace.generation_log, Qt.Key_Return)
    qapp.processEvents()
    committed = document.scenario
    assert workspace._tool is AuthoringTool.SELECT
    assert workspace.waypoint_session_bar.isHidden()

    workspace._start_waypoint_rebuild()
    _place(workspace, (9.0, 0.0, 0.0))
    workspace.generation_log.setFocus()
    QTest.keyClick(workspace.generation_log, Qt.Key_Escape)
    qapp.processEvents()
    assert document.scenario == committed
    assert workspace._tool is AuthoringTool.SELECT
    workspace.close_workspace()


def test_orientation_overlay_preferences_and_cached_preparation(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    actor = actor.with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 0.0),
            end_m=(1.0, 0.0, 0.0),
        ),
        orientation=AlignMotionOrientationSpec(),
    )
    document.replace_actor(actor)
    document.select(actor.id)
    samples = ActorSamples(
        positions=((0.0, 0.0, 0.0),),
        orientations=((15.0, 0.0, 0.0),),
    )
    compiler = _FixedCompiler(CompilationResult({}, "schema_version: 2\n", (), {actor.id: samples}))
    workspace = _workspace(document, compiler)
    workspace.refresh_now()
    assert workspace.viewport.port.snapshots[-1].actors[0].orientation_matrix is not None
    assert (
        workspace.viewport.port.snapshots[-1].orientation_axes_visibility
        is OverlayVisibility.SELECTED
    )

    compiler.result = CompilationResult({}, "schema_version: 2\n", (), {})
    workspace.refresh_now()
    assert workspace.viewport.port.snapshots[-1].actors[0].orientation_matrix is not None

    workspace.orientation_axes_combo.setCurrentIndex(
        workspace.orientation_axes_combo.findData(OverlayVisibility.OFF)
    )
    assert (
        workspace.viewport.port.snapshots[-1].orientation_axes_visibility is OverlayVisibility.OFF
    )
    workspace.close_workspace()


def test_trajectory_frame_samples_and_control_rig_are_independent_overlays(qapp) -> None:
    actor = (
        ScenarioDocument.new()
        .add_default_actor(ActorRole.RX)
        .with_changes(
            mobility=WaypointMobilitySpec(
                points_m=(
                    (0.0, 0.0, 1.0),
                    (2.0, 0.0, 1.0),
                    (2.0, 2.0, 1.0),
                )
            )
        )
    )
    document = ScenarioDocument(
        AuthoringScenario(
            timeline=TimelineSettings(steps=4, duration_s=3.0),
            actors=(actor,),
        )
    )
    document.select(actor.id)
    samples = ActorSamples(
        positions=(
            (0.0, 0.0, 1.0),
            (1.5, 0.0, 1.0),
            (2.0, 0.5, 1.0),
            (2.0, 2.0, 1.0),
        ),
        orientations=((0.0, 0.0, 0.0),) * 4,
    )
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "schema_version: 2\n", (), {actor.id: samples})),
    )
    workspace.refresh_now()

    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.positions == actor.mobility.points_m
    assert overlay.frame_samples == samples.positions
    assert overlay.trajectory_visible is True
    assert overlay.frame_samples_visible is False
    assert overlay.mobility_control_rig is None
    assert overlay.preview_provenance is PreviewProvenance.GENERATOR_PREPARED

    workspace.frame_samples_visibility_combo.setCurrentIndex(
        workspace.frame_samples_visibility_combo.findData(OverlayVisibility.SELECTED)
    )
    assert workspace.viewport.port.snapshots[-1].actors[0].frame_samples_visible is True

    workspace.trajectory_visibility_combo.setCurrentIndex(
        workspace.trajectory_visibility_combo.findData(OverlayVisibility.OFF)
    )
    assert workspace.viewport.port.snapshots[-1].actors[0].trajectory_visible is False

    workspace.set_tool(AuthoringTool.MOVE)
    assert workspace.viewport.port.snapshots[-1].actors[0].mobility_control_rig is not None
    workspace.close_workspace()


def test_waypoint_session_shows_live_rubber_band_from_last_point(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    document.replace_actor(
        actor.with_changes(mobility=StationaryMobilitySpec(position_m=(1.0, 2.0, 3.0)))
    )
    workspace = _workspace(document)

    workspace._start_waypoint_rebuild()
    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (10.0, 10.0)))
    workspace._on_viewport_input(HitResult((4.0, 5.0, 3.0), component="work_plane"))

    snapshot = workspace.viewport.port.snapshots[-1]
    assert snapshot.placement_guide_start == (1.0, 2.0, 3.0)
    assert snapshot.placement_ghost == (4.0, 5.0, 3.0)
    workspace._cancel_waypoint_session_and_exit()
    workspace.close_workspace()


def test_target_asset_change_invalidates_prepared_orientation_cache(qapp) -> None:
    actor = (
        ScenarioDocument.new()
        .add_default_actor(ActorRole.TARGET)
        .with_changes(
            target=TargetAsset.from_catalog_id("cube"),
            orientation=FixedOrientationSpec(
                yaw_deg=25.0,
                pitch_deg=0.0,
                roll_deg=0.0,
            ),
        )
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    samples = ActorSamples(
        positions=((0.0, 0.0, 0.0),),
        orientations=((25.0, 0.0, 0.0),),
    )
    compiler = _FixedCompiler(CompilationResult({}, "schema_version: 2\n", (), {actor.id: samples}))
    workspace = _workspace(document, compiler)
    workspace.refresh_now()
    assert workspace._prepared_samples(actor) is samples

    changed = actor.with_changes(target=TargetAsset.from_catalog_id("nist_human_walking"))
    document.replace_actor(changed)
    compiler.result = CompilationResult({}, "schema_version: 2\n", (), {})
    workspace.compilation = compiler.result

    assert workspace._prepared_samples(document.scenario.actor(actor.id)) is None
    workspace.close_workspace()


def test_fixed_orientation_axis_does_not_require_whole_document_validation(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "schema_version: 2\n", (), {})),
    )
    workspace.refresh_now()

    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.actor_id == actor.id
    assert overlay.orientation_matrix is not None
    workspace.close_workspace()


def test_generation_result_and_reset_control_repeat_preview_availability(qapp) -> None:
    workspace = _workspace(ScenarioDocument.new())
    workspace.append_generation_log("[stdout] generated")
    result = SimpleNamespace(
        succeeded=True,
        stale=True,
        state=SimpleNamespace(value="succeeded"),
    )

    workspace.set_generation_result(result)

    assert workspace.preview_result_button.isEnabled() is True
    assert "older draft revision" in workspace.generation_status.text()
    assert workspace.generation_log.toPlainText() == "[stdout] generated"

    workspace.reset_generation_state()

    assert workspace.preview_result_button.isEnabled() is False
    assert workspace.cancel_generation_button.isEnabled() is False
    assert workspace.generation_status.text() == "No generation has been launched."
    assert workspace.generation_log.toPlainText() == ""
    assert workspace.generation_progress.value() == 0
    workspace.close_workspace()


def test_waypoint_sequence_commits_once_and_escape_restores_exact_state(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    workspace = _workspace(document)
    original = document.scenario
    command_count = document.undo_stack.count

    workspace.set_tool(AuthoringTool.WAYPOINT)
    _place(workspace, (1.0, 0.0, 0.0))
    _place(workspace, (2.0, 0.0, 0.0))
    assert document.undo_stack.count == command_count
    assert isinstance(document.scenario.actor(actor.id).mobility, WaypointMobilitySpec)

    workspace._handle_key(KeyboardInput("Backspace", True))
    assert isinstance(
        document.scenario.actor(actor.id).mobility,
        StationaryMobilitySpec,
    )
    workspace._handle_key(KeyboardInput("Escape", True))
    assert document.scenario == original
    assert document.undo_stack.count == command_count

    workspace.set_tool(AuthoringTool.WAYPOINT)
    _place(workspace, (3.0, 0.0, 0.0))
    _place(workspace, (4.0, 0.0, 0.0))
    workspace._handle_key(KeyboardInput("Enter", True))
    assert document.undo_stack.count == command_count + 1
    assert document.scenario.actor(actor.id).mobility == WaypointMobilitySpec(
        points_m=((3.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    )
    document.undo()
    assert isinstance(document.scenario.actor(actor.id).mobility, StationaryMobilitySpec)
    workspace.close_workspace()


def test_waypoint_double_click_adds_final_point_and_finishes(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    workspace = _workspace(document)
    command_count = document.undo_stack.count

    workspace.set_tool(AuthoringTool.WAYPOINT)
    _place(workspace, (1.0, 1.0, 0.0))
    _place(workspace, (2.0, 1.0, 0.0))
    _place(workspace, (2.0, 1.0, 0.0), PointerPhase.DOUBLE_CLICK)

    mobility = document.scenario.actor(actor.id).mobility
    assert mobility == WaypointMobilitySpec(points_m=((1.0, 1.0, 0.0), (2.0, 1.0, 0.0)))
    assert document.undo_stack.count == command_count + 1
    assert workspace._tool is AuthoringTool.SELECT
    workspace.close_workspace()


def test_direct_waypoint_drag_commits_once_and_escape_cancels(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    document.replace_actor(
        actor.with_changes(
            mobility=WaypointMobilitySpec(points_m=((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)))
        )
    )
    workspace = _workspace(document)
    workspace.refresh_now()
    workspace.set_tool(AuthoringTool.MOVE)
    snapshot = workspace.viewport.port.snapshots[-1]
    rig = snapshot.actors[0].mobility_control_rig
    assert rig is not None
    assert tuple(control.ordinal for control in rig.controls) == (0, 1)
    command_count = document.undo_stack.count
    original = document.scenario.actor(actor.id).mobility

    source = HitResult(
        world_position=(2.0, 0.0, 1.0),
        renderer_object_id="authoring:handle",
        actor_id=actor.id,
        component="mobility_control_waypoint_1",
    )
    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (10.0, 10.0), button=1))
    workspace._on_viewport_input(source)
    assert workspace.viewport.port.drag_constraint == "free"
    assert workspace.viewport.port.drag_plane_z == 1.0
    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (20.0, 20.0), buttons=(1,)))
    workspace._on_viewport_input(
        HitResult(
            world_position=(3.0, 4.0, 1.0),
            renderer_object_id=source.renderer_object_id,
            actor_id=actor.id,
            component="mobility_control_waypoint_1",
        )
    )
    workspace._on_viewport_input(PointerInput(PointerPhase.UP, (20.0, 20.0), button=1))
    workspace._on_viewport_input(
        HitResult(
            world_position=(4.0, 5.0, 1.0),
            renderer_object_id=source.renderer_object_id,
            actor_id=actor.id,
            component="mobility_control_waypoint_1",
        )
    )
    assert document.scenario.actor(actor.id).mobility == WaypointMobilitySpec(
        points_m=((0.0, 0.0, 1.0), (4.0, 5.0, 1.0))
    )
    assert document.undo_stack.count == command_count + 1
    document.undo()
    assert document.scenario.actor(actor.id).mobility == original

    workspace.set_tool(AuthoringTool.MOVE)
    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (10.0, 10.0), button=1))
    workspace._on_viewport_input(source)
    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (20.0, 20.0), buttons=(1,)))
    workspace._on_viewport_input(
        HitResult(
            world_position=(9.0, 9.0, 1.0),
            renderer_object_id=source.renderer_object_id,
            actor_id=actor.id,
            component="mobility_control_waypoint_1",
        )
    )
    workspace._handle_key(KeyboardInput("Escape", True))
    assert document.scenario.actor(actor.id).mobility == original
    # The first committed drag remains on the redo branch after ``undo``;
    # cancelling the second drag must not append another command.
    assert document.undo_stack.count == command_count + 1
    workspace.close_workspace()


@pytest.mark.parametrize(
    "derived_orientation",
    (
        AlignMotionOrientationSpec(),
        KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(time_s=0.0),
                OrientationKeyframeSpec(time_s=1.0, yaw_deg=45.0),
            )
        ),
        SpinOrientationSpec(
            axis="yaw",
            rate_deg_s=30.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        ),
    ),
    ids=lambda value: orientation_kind(value).value,
)
def test_edit_motion_attaches_semantic_gizmo_and_publishes_control_rig(
    qapp,
    derived_orientation,
) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    document.replace_actor(
        actor.with_changes(
            mobility=LinearMobilitySpec(
                start_m=(0.0, 0.0, 1.0),
                end_m=(8.0, 2.0, 1.0),
            ),
            orientation=FixedOrientationSpec(
                yaw_deg=15.0,
                pitch_deg=0.0,
                roll_deg=0.0,
            ),
        )
    )
    workspace = _workspace(document)
    workspace.refresh_now()

    workspace.set_tool(AuthoringTool.MOVE)

    assert workspace.viewport.port.show_gizmo_calls[-1] == actor.id
    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.positions == ((0.0, 0.0, 1.0), (8.0, 2.0, 1.0))
    assert overlay.mobility_control_rig is not None
    assert tuple(control.key for control in overlay.mobility_control_rig.controls) == (
        "start",
        "end",
    )
    assert overlay.transform_rotation_enabled is True
    assert overlay.transform_rotation_enabled is (
        EditAffordance.ROTATION_GIZMO
        in orientation_capability(
            orientation_kind(document.scenario.actor(actor.id).orientation).value
        ).direct_viewport_affordances
    )
    assert "XYZ gizmo" in workspace.tool_hint_label.text()

    current = document.scenario.actor(actor.id)
    document.replace_actor(current.with_changes(orientation=derived_orientation))
    workspace.refresh_now()
    derived_overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert derived_overlay.transform_rotation_enabled is (
        EditAffordance.ROTATION_GIZMO
        in orientation_capability(
            orientation_kind(derived_orientation).value
        ).direct_viewport_affordances
    )

    current = document.scenario.actor(actor.id)
    show_count = len(workspace.viewport.port.show_gizmo_calls)
    document.replace_actor(current.with_changes(locked=True))
    workspace.refresh_now()
    assert len(workspace.viewport.port.show_gizmo_calls) == show_count
    assert workspace.viewport.port.clear_gizmo_calls > 0
    workspace.close_workspace()


def _two_actor_figure8_document():
    tx = AuthoringActor.create(ActorRole.TX, "TX figure eight").with_changes(
        mobility=Figure8MobilitySpec(
            center_m=(10.0, 20.0, 3.0),
            size_m=8.0,
        ),
        orientation=FixedOrientationSpec(
            yaw_deg=10.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        ),
    )
    rx = AuthoringActor.create(ActorRole.RX, "RX figure eight").with_changes(
        mobility=Figure8MobilitySpec(
            center_m=(-30.0, 8.0, 2.0),
            size_m=6.0,
        ),
        orientation=FixedOrientationSpec(
            yaw_deg=-20.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        ),
    )
    tx_samples = ActorSamples(
        positions=(
            (10.0, 20.0, 3.0),
            (14.0, 24.0, 3.0),
            (10.0, 20.0, 3.0),
            (6.0, 16.0, 3.0),
            (10.0, 20.0, 3.0),
        ),
        orientations=((10.0, 0.0, 0.0),) * 5,
    )
    rx_samples = ActorSamples(
        positions=(
            (-30.0, 8.0, 2.0),
            (-27.0, 11.0, 2.0),
            (-30.0, 8.0, 2.0),
            (-33.0, 5.0, 2.0),
            (-30.0, 8.0, 2.0),
        ),
        orientations=((-20.0, 0.0, 0.0),) * 5,
    )
    document = ScenarioDocument(
        AuthoringScenario(
            actors=(tx, rx),
            timeline=TimelineSettings(steps=5, duration_s=4.0),
        )
    )
    document.select(rx.id)
    result = CompilationResult(
        {},
        "schema_version: 2\n",
        (),
        {
            tx.id: tx_samples,
            rx.id: rx_samples,
        },
    )
    return document, tx, rx, tx_samples, rx_samples, result


def test_edit_mobility_path_drag_translates_the_whole_trajectory(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    original = LinearMobilitySpec(
        start_m=(0.0, 0.0, 2.0),
        end_m=(10.0, 0.0, 2.0),
    )
    document.replace_actor(actor.with_changes(mobility=original))
    workspace = _workspace(document)
    workspace.set_tool(AuthoringTool.MOVE)
    command_count = document.undo_stack.count
    source = HitResult(
        world_position=(5.0, 0.0, 2.0),
        renderer_object_id="authoring:trajectory_hit",
        actor_id=actor.id,
        component="trajectory_hit",
    )

    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (10.0, 10.0), button=1))
    workspace._on_viewport_input(source)
    workspace._on_viewport_input(PointerInput(PointerPhase.UP, (20.0, 20.0), button=1))
    workspace._on_viewport_input(
        HitResult(
            world_position=(7.0, 3.0, 2.0),
            renderer_object_id=source.renderer_object_id,
            actor_id=actor.id,
            component="trajectory_hit",
        )
    )

    assert document.scenario.actor(actor.id).mobility == LinearMobilitySpec(
        start_m=(2.0, 3.0, 2.0),
        end_m=(12.0, 3.0, 2.0),
    )
    assert document.undo_stack.count == command_count + 1
    workspace.close_workspace()


def test_two_actor_figure8_path_drag_keeps_prepared_preview_and_identity(qapp) -> None:
    document, tx, rx, tx_samples, rx_samples, result = _two_actor_figure8_document()
    workspace = _workspace(document, _FixedCompiler(result))
    workspace.refresh_now()
    workspace.set_tool(AuthoringTool.MOVE)
    command_count = document.undo_stack.count
    source = HitResult(
        world_position=tx_samples.positions[0],
        renderer_object_id="authoring:trajectory_hit",
        actor_id=tx.id,
        component="trajectory_hit",
    )
    translation = (4.0, -3.0, 0.0)
    moved_position = tuple(
        value + offset for value, offset in zip(source.world_position, translation, strict=True)
    )
    moved_hit = HitResult(
        world_position=moved_position,
        renderer_object_id=source.renderer_object_id,
        actor_id=tx.id,
        component=source.component,
    )
    expected_positions = tuple(
        tuple(value + offset for value, offset in zip(position, translation, strict=True))
        for position in tx_samples.positions
    )

    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (10.0, 10.0), button=1))
    workspace._on_viewport_input(source)
    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (20.0, 20.0), buttons=(1,)))
    workspace._on_viewport_input(moved_hit)

    overlays = {
        overlay.actor_id: overlay for overlay in workspace.viewport.port.snapshots[-1].actors
    }
    assert overlays[tx.id].positions == expected_positions
    assert overlays[tx.id].frame_samples == expected_positions
    assert overlays[tx.id].current_position == expected_positions[0]
    assert overlays[rx.id].positions == rx_samples.positions
    assert document.scenario.actor(tx.id).mobility == Figure8MobilitySpec(
        center_m=(14.0, 17.0, 3.0),
        size_m=8.0,
    )
    assert document.scenario.actor(rx.id) == rx
    assert document.selected_actor_id == tx.id

    workspace._on_viewport_input(PointerInput(PointerPhase.UP, (20.0, 20.0), button=1))
    workspace._on_viewport_input(moved_hit)
    workspace._timeline_scrubbed(3)
    overlays = {
        overlay.actor_id: overlay for overlay in workspace.viewport.port.snapshots[-1].actors
    }
    assert overlays[tx.id].positions == expected_positions
    assert overlays[tx.id].current_position == expected_positions[3]
    assert overlays[rx.id].positions == rx_samples.positions
    assert document.undo_stack.count == command_count + 1
    workspace.close_workspace()


def test_form_group_preserves_start_positions_when_preview_is_on_last_frame(
    qapp,
    monkeypatch,
) -> None:
    document = ScenarioDocument.new()
    document.set_timeline(TimelineSettings(steps=5, duration_s=4.0))
    tx = document.add_default_actor(ActorRole.TX)
    rx_left = document.add_default_actor(ActorRole.RX)
    rx_right = document.add_default_actor(ActorRole.RX)
    group_motion = LinearMobilitySpec(
        start_m=(0.0, 0.0, 1.0),
        end_m=(10.0, 4.0, 1.0),
    )
    start_positions = {
        tx.id: group_motion.start_m,
        rx_left.id: (0.0, 3.0, 1.0),
        rx_right.id: (0.0, -2.0, 2.0),
    }
    document.replace_actor(tx.with_changes(mobility=group_motion))
    document.replace_actor(
        rx_left.with_changes(
            mobility=StationaryMobilitySpec(position_m=start_positions[rx_left.id])
        )
    )
    document.replace_actor(
        rx_right.with_changes(
            mobility=StationaryMobilitySpec(position_m=start_positions[rx_right.id])
        )
    )
    document.select(tx.id)
    initial = ScenarioCompiler().compile(document.scenario)
    assert initial.valid
    workspace = _workspace(document, _FixedCompiler(initial))
    workspace.refresh_now()
    workspace.timeline_slider.setValue(workspace.timeline_slider.maximum())
    assert workspace._play_step == workspace.timeline_slider.maximum()

    def accept_group_dialog(dialog) -> int:
        actor_list = dialog.findChild(workspace_module.QListWidget)
        primary_combo = dialog.findChild(workspace_module.QComboBox)
        use_primary_motion = dialog.findChild(workspace_module.QCheckBox)
        assert actor_list is not None
        assert primary_combo is not None
        assert use_primary_motion is not None
        for index in range(actor_list.count()):
            actor_list.item(index).setCheckState(Qt.Checked)
        primary_combo.setCurrentIndex(primary_combo.findData(str(tx.id)))
        use_primary_motion.setChecked(True)
        return workspace_module.QDialog.Accepted

    monkeypatch.setattr(workspace_module.QDialog, "exec", accept_group_dialog)
    workspace._form_group()

    assert len(document.groups) == 1
    assert document.groups[0].mobility == group_motion
    grouped = ScenarioCompiler().compile(document.scenario)
    assert grouped.valid
    displacement = tuple(
        end - start for start, end in zip(group_motion.start_m, group_motion.end_m, strict=True)
    )
    for actor_id, start_position in start_positions.items():
        samples = grouped.samples[actor_id]
        np.testing.assert_allclose(samples.positions[0], start_position)
        np.testing.assert_allclose(
            samples.positions[-1],
            tuple(value + delta for value, delta in zip(start_position, displacement, strict=True)),
        )

    workspace._apply_compilation(grouped)
    overlays = {
        overlay.actor_id: overlay for overlay in workspace.viewport.port.snapshots[-1].actors
    }
    for actor_id in start_positions:
        assert overlays[actor_id].positions == grouped.samples[actor_id].positions
        assert overlays[actor_id].frame_samples == grouped.samples[actor_id].positions
    workspace.close_workspace()


def test_group_member_path_drag_updates_only_the_local_group_offset(qapp) -> None:
    group = AuthoringGroup.create("Pair").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(4.0, 0.0, 1.0),
        )
    )
    tx = AuthoringActor.create(ActorRole.TX, "TX").with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=-1.0),
        )
    )
    rx = AuthoringActor.create(ActorRole.RX, "RX").with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=1.0),
        )
    )
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            actors=(tx, rx),
            groups=(group,),
        )
    )
    document.select(tx.id)
    compiled = ScenarioCompiler().compile(document.scenario)
    assert compiled.valid
    assert isinstance(compiled.group_samples[group.id], GroupSamples)
    workspace = _workspace(document, _FixedCompiler(compiled))
    workspace.refresh_now()
    selected_overlay = next(
        overlay
        for overlay in workspace.viewport.port.snapshots[-1].actors
        if overlay.actor_id == tx.id
    )
    assert selected_overlay.group_origin_position is not None
    assert selected_overlay.group_frame_matrix is not None
    workspace.set_tool(AuthoringTool.MOVE)
    command_count = document.undo_stack.count
    source = HitResult(
        world_position=(0.0, 1.0, 1.0),
        renderer_object_id="authoring:trajectory_hit",
        actor_id=tx.id,
        component="trajectory_hit",
    )

    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (10.0, 10.0), button=1))
    workspace._on_viewport_input(source)
    workspace._on_viewport_input(PointerInput(PointerPhase.UP, (20.0, 20.0), button=1))
    workspace._on_viewport_input(
        HitResult(
            world_position=(0.0, 3.0, 1.0),
            renderer_object_id=source.renderer_object_id,
            actor_id=tx.id,
            component="trajectory_hit",
        )
    )

    updated = document.scenario.actor(tx.id)
    assert updated is not None
    assert updated.mobility == GroupMemberMobilitySpec(
        group=str(group.id),
        offset_m=GroupOffsetSpec(right=-3.0),
    )
    assert document.scenario.actor(rx.id) == rx
    assert document.scenario.group(group.id) == group
    assert document.undo_stack.count == command_count + 1
    workspace.close_workspace()


def test_figure8_group_path_drag_keeps_prepared_preview_through_commit(qapp) -> None:
    group = AuthoringGroup.create("Figure-eight group").with_changes(
        mobility=Figure8MobilitySpec(
            center_m=(25.0, -10.0, 4.0),
            size_m=12.0,
        )
    )
    document = ScenarioDocument(
        AuthoringScenario(
            groups=(group,),
            timeline=TimelineSettings(steps=5, duration_s=4.0),
        )
    )
    document.select_group(group.id)
    compiled = ScenarioCompiler().compile(document.scenario)
    prepared = compiled.group_samples[group.id]
    workspace = _workspace(document, _FixedCompiler(compiled))
    workspace.refresh_now()
    workspace.set_tool(AuthoringTool.MOVE)
    translation = (-5.0, 7.0, 0.0)
    source = HitResult(
        world_position=prepared.positions[0],
        renderer_object_id="authoring:trajectory_hit",
        actor_id=group.id,
        component="trajectory_hit",
    )
    moved_hit = HitResult(
        world_position=tuple(
            value + offset
            for value, offset in zip(
                source.world_position,
                translation,
                strict=True,
            )
        ),
        renderer_object_id=source.renderer_object_id,
        actor_id=group.id,
        component=source.component,
    )
    expected_positions = tuple(
        tuple(value + offset for value, offset in zip(position, translation, strict=True))
        for position in prepared.positions
    )

    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (10.0, 10.0), button=1))
    workspace._on_viewport_input(source)
    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (20.0, 20.0), buttons=(1,)))
    workspace._on_viewport_input(moved_hit)
    overlay = next(
        item for item in workspace.viewport.port.snapshots[-1].actors if item.actor_id == group.id
    )
    assert overlay.positions == expected_positions

    workspace._on_viewport_input(PointerInput(PointerPhase.UP, (20.0, 20.0), button=1))
    workspace._on_viewport_input(moved_hit)
    workspace._timeline_scrubbed(3)
    overlay = next(
        item for item in workspace.viewport.port.snapshots[-1].actors if item.actor_id == group.id
    )
    assert overlay.positions == expected_positions
    assert overlay.current_position == expected_positions[3]
    assert document.scenario.group(group.id).mobility == Figure8MobilitySpec(
        center_m=(20.0, -3.0, 4.0),
        size_m=12.0,
    )
    workspace.close_workspace()


def test_linear_arrival_handle_changes_only_the_endpoint(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    original = LinearMobilitySpec(
        start_m=(1.0, 2.0, 3.0),
        end_m=(6.0, 2.0, 3.0),
    )
    document.replace_actor(actor.with_changes(mobility=original))
    workspace = _workspace(document)
    workspace.set_tool(AuthoringTool.MOVE)

    source = HitResult(
        world_position=original.end_m,
        renderer_object_id="authoring:mobility_control_end",
        actor_id=actor.id,
        component="mobility_control_end",
    )
    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (10.0, 10.0), button=1))
    workspace._on_viewport_input(source)
    workspace._on_viewport_input(PointerInput(PointerPhase.UP, (20.0, 20.0), button=1))
    workspace._on_viewport_input(
        HitResult(
            world_position=(9.0, 5.0, 3.0),
            renderer_object_id=source.renderer_object_id,
            actor_id=actor.id,
            component=source.component,
        )
    )

    assert document.scenario.actor(actor.id).mobility == LinearMobilitySpec(
        start_m=original.start_m,
        end_m=(9.0, 5.0, 3.0),
    )
    workspace.close_workspace()


def test_circular_radius_and_start_angle_handles_are_independent(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    document.replace_actor(
        actor.with_changes(
            mobility=CircularMobilitySpec(
                center_m=(1.0, 2.0, 3.0),
                radius_m=4.0,
                start_angle_deg=math.degrees(0.25),
                clockwise=False,
            )
        )
    )
    workspace = _workspace(document)
    workspace.set_tool(AuthoringTool.MOVE)

    radius_control = mobility_control_rig(document.scenario.actor(actor.id).mobility).control(
        "radius"
    )
    radius_source = HitResult(
        radius_control.position,
        "authoring:mobility_control_radius",
        actor.id,
        "mobility_control_radius",
    )
    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (1.0, 1.0), button=1))
    workspace._on_viewport_input(radius_source)
    assert workspace.viewport.port.drag_constraint == "radial"
    workspace._on_viewport_input(PointerInput(PointerPhase.UP, (2.0, 2.0), button=1))
    radius_axis = np.asarray(radius_control.position) - np.asarray(
        document.scenario.actor(actor.id).mobility.center_m
    )
    radius_axis /= np.linalg.norm(radius_axis)
    resized_position = np.asarray(document.scenario.actor(actor.id).mobility.center_m) + (
        radius_axis * 6.0
    )
    workspace._on_viewport_input(
        HitResult(
            tuple(resized_position),
            radius_source.renderer_object_id,
            actor.id,
            radius_source.component,
        )
    )
    resized = document.scenario.actor(actor.id).mobility
    assert isinstance(resized, CircularMobilitySpec)
    assert resized.radius_m == pytest.approx(6.0)
    assert resized.start_angle_deg == pytest.approx(math.degrees(0.25))

    angle_control = mobility_control_rig(resized).control("start_angle")
    angle_source = HitResult(
        angle_control.position,
        "authoring:mobility_control_start_angle",
        actor.id,
        "mobility_control_start_angle",
    )
    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (1.0, 1.0), button=1))
    workspace._on_viewport_input(angle_source)
    assert workspace.viewport.port.drag_constraint == "angular"
    workspace._on_viewport_input(PointerInput(PointerPhase.UP, (2.0, 2.0), button=1))
    workspace._on_viewport_input(
        HitResult(
            (1.0, -4.0, 3.0),
            angle_source.renderer_object_id,
            actor.id,
            angle_source.component,
        )
    )
    rotated = document.scenario.actor(actor.id).mobility
    assert isinstance(rotated, CircularMobilitySpec)
    assert rotated.radius_m == pytest.approx(6.0)
    assert rotated.start_angle_deg == pytest.approx(-90.0)
    workspace.close_workspace()


def test_motion_hover_highlights_wide_trajectory_target_and_semantic_handle(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    document.replace_actor(
        actor.with_changes(
            mobility=LinearMobilitySpec(
                start_m=(0.0, 0.0, 1.0),
                end_m=(4.0, 0.0, 1.0),
            )
        )
    )
    workspace = _workspace(document)
    workspace.set_tool(AuthoringTool.MOVE)

    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (10.0, 10.0)))
    workspace._on_viewport_input(
        HitResult(
            (2.0, 0.0, 1.0),
            "authoring:trajectory_hit",
            actor.id,
            "trajectory_hit",
        )
    )
    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.trajectory_hovered is True
    assert workspace.viewport.cursor().shape() is Qt.CursorShape.SizeAllCursor
    assert "entire motion" in workspace.tool_hint_label.text()

    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (11.0, 11.0)))
    workspace._on_viewport_input(
        HitResult(
            (4.0, 0.0, 1.0),
            "authoring:mobility_control_end",
            actor.id,
            "mobility_control_end",
        )
    )
    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.hovered_control_key == "end"
    assert "Arrival point" in workspace.tool_hint_label.text()

    workspace.work_plane_check.setChecked(False)
    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (40.0, 40.0)))
    qapp.processEvents()
    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.hovered_control_key is None
    assert overlay.trajectory_hovered is False
    assert workspace.viewport.cursor().shape() is Qt.CursorShape.ArrowCursor

    current = document.scenario.actor(actor.id)
    document.replace_actor(current.with_changes(locked=True))
    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.mobility_control_rig is None
    workspace.close_workspace()


def _target_preview_result(actor) -> CompilationResult:
    samples = ActorSamples(
        positions=((0.0, 0.0, 1.0), (10.0, 0.0, 1.0)),
        orientations=((185.0, 0.0, 0.0), (185.0, 0.0, 0.0)),
    )
    payload = MeshPayload(
        vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        triangles=np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    frames = tuple(
        TargetPreviewAsset(
            cache_key=f"target-step-{step}",
            actor_id=actor.id,
            step_index=step,
            mesh_path="catalog.ply",
            scale=1.0,
            mesh=payload,
            material=MaterialPayload(shader="pbr"),
            local_to_actor=np.eye(4),
        )
        for step in range(2)
    )
    return CompilationResult(
        {},
        "schema_version: 2\n",
        (),
        {actor.id: samples},
        runtime=SimpleNamespace(
            targets=(
                SimpleNamespace(
                    name=actor.name,
                    asset_front_yaw_offset_deg=180.0,
                ),
            )
        ),
        target_assets={actor.id: frames},
    )


def test_timeline_snapshot_uses_compiler_prepared_target_pose(qapp) -> None:
    actor = ScenarioDocument.new().add_default_actor(ActorRole.TARGET)
    actor = actor.with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(10.0, 0.0, 1.0),
        ),
        orientation=FixedOrientationSpec(yaw_deg=5.0, pitch_deg=0.0, roll_deg=0.0),
        target=TargetAsset.from_catalog_id("cube"),
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    workspace = _workspace(document, _FixedCompiler(_target_preview_result(actor)))
    workspace.refresh_now()

    workspace._timeline_scrubbed(1)

    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.current_position == (10.0, 0.0, 1.0)
    np.testing.assert_allclose(
        overlay.orientation_matrix,
        prepared_actor_pose((10.0, 0.0, 1.0), (185.0, 0.0, 0.0)),
    )
    assert overlay.target_asset.cache_key == "target-step-1"
    workspace.close_workspace()


def test_target_gizmo_translation_rotation_is_one_undo_command(qapp) -> None:
    actor = ScenarioDocument.new().add_default_actor(ActorRole.TARGET)
    actor = actor.with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(10.0, 0.0, 1.0),
        ),
        orientation=FixedOrientationSpec(yaw_deg=5.0, pitch_deg=0.0, roll_deg=0.0),
        target=TargetAsset.from_catalog_id("cube"),
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    workspace = _workspace(document, _FixedCompiler(_target_preview_result(actor)))
    workspace.refresh_now()
    workspace._timeline_scrubbed(1)
    command_count = document.undo_stack.count
    initial = prepared_actor_pose((10.0, 0.0, 1.0), (185.0, 0.0, 0.0))
    changed = prepared_actor_pose((12.0, 3.0, 1.5), (200.0, 0.0, 0.0))

    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.BEGIN, tuple(map(tuple, initial)))
    )
    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.UPDATE, tuple(map(tuple, changed)))
    )
    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.COMMIT, tuple(map(tuple, changed)))
    )

    updated = document.scenario.actor(actor.id)
    assert updated.mobility == LinearMobilitySpec(
        start_m=(2.0, 3.0, 1.5),
        end_m=(12.0, 3.0, 1.5),
    )
    assert updated.orientation.yaw_deg == pytest.approx(20.0)
    assert document.undo_stack.count == command_count + 1
    document.undo()
    assert document.scenario.actor(actor.id) == actor
    workspace.close_workspace()


def test_target_gizmo_round_trips_pitch_and_roll_through_asset_front_alignment(
    qapp,
) -> None:
    authored_angles = (25.0, 20.0, -15.0)
    actor = ScenarioDocument.new().add_default_actor(ActorRole.TARGET)
    actor = actor.with_changes(
        orientation=FixedOrientationSpec(
            yaw_deg=authored_angles[0],
            pitch_deg=authored_angles[1],
            roll_deg=authored_angles[2],
        ),
        target=TargetAsset.from_catalog_id("cube"),
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    workspace = _workspace(document, _FixedCompiler(_target_preview_result(actor)))
    workspace.refresh_now()

    authored_pose = prepared_actor_pose((0.0, 0.0, 0.0), authored_angles)
    alignment = prepared_actor_pose((0.0, 0.0, 0.0), (180.0, 0.0, 0.0))
    rendered = np.eye(4, dtype=float)
    rendered[:3, :3] = authored_pose[:3, :3] @ alignment[:3, :3]
    rendered[:3, 3] = (2.0, 3.0, 4.0)
    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.BEGIN, tuple(map(tuple, rendered)))
    )
    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.COMMIT, tuple(map(tuple, rendered)))
    )

    updated = document.scenario.actor(actor.id)
    assert updated.orientation.yaw_deg == pytest.approx(authored_angles[0])
    assert updated.orientation.pitch_deg == pytest.approx(authored_angles[1])
    assert updated.orientation.roll_deg == pytest.approx(authored_angles[2])
    assert workspace._transform_preview_samples is None
    workspace.close_workspace()


@pytest.mark.parametrize("role", (ActorRole.TX, ActorRole.RX, ActorRole.TARGET))
def test_router_workspace_gizmo_commits_after_scrub_and_active_reconcile(
    qapp,
    role,
) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(role)
    original = LinearMobilitySpec(
        start_m=(0.0, 0.0, 1.0),
        end_m=(10.0, 0.0, 1.0),
    )
    document.replace_actor(
        actor.with_changes(
            mobility=original,
            orientation=FixedOrientationSpec(
                yaw_deg=10.0,
                pitch_deg=0.0,
                roll_deg=0.0,
            ),
        )
    )
    workspace = _router_workspace(document)
    workspace.refresh_now()
    workspace.set_tool(AuthoringTool.MOVE)
    port = workspace.viewport.port
    runtime = port.runtime
    object_id = stable_renderer_id(
        document.scenario.document_id,
        actor.id,
        "mobility_handles",
    )
    assert runtime.attached_object_id == object_id

    # The attachment starts at the first prepared/control sample. Scrubbing
    # synchronizes the proxy and must also update the router's BEGIN baseline.
    workspace._timeline_scrubbed(1)
    changed = np.asarray(runtime.poses[object_id], dtype=float).copy()
    changed[0, 3] += 1.0
    callback = runtime.callback
    assert callback is not None
    command_count = document.undo_stack.count
    callback({"phase": "changed", "object_id": object_id, "transform": changed})

    # Exercise the real transient compile/reconcile path before pointer-up.
    qapp.processEvents()
    assert runtime.attached_object_id == object_id
    assert runtime.callback is callback
    callback({"phase": "committed", "object_id": object_id, "transform": changed})

    assert document.scenario.actor(actor.id).mobility == LinearMobilitySpec(
        start_m=(1.0, 0.0, 1.0),
        end_m=(11.0, 0.0, 1.0),
    )
    assert document.undo_stack.count == command_count + 1
    assert workspace._transform_actor_id is None
    document.undo()
    assert document.scenario.actor(actor.id).mobility == original
    workspace.close_workspace()


def test_two_actor_figure8_rotation_keeps_path_and_active_proxy_pose(qapp) -> None:
    document, tx, rx, tx_samples, rx_samples, result = _two_actor_figure8_document()
    document.select(tx.id)
    workspace = _router_workspace(document, _FixedCompiler(result))
    workspace.refresh_now()
    workspace.set_tool(AuthoringTool.MOVE)
    port = workspace.viewport.port
    runtime = port.runtime
    object_id = stable_renderer_id(
        document.scenario.document_id,
        tx.id,
        "mobility_handles",
    )
    callback = runtime.callback
    assert callback is not None
    sync_count = sum(call_object_id == object_id for call_object_id, _pose in runtime.sync_calls)
    changed = prepared_actor_pose(tx_samples.positions[0], (55.0, 0.0, 0.0))
    command_count = document.undo_stack.count

    callback({"phase": "changed", "object_id": object_id, "transform": changed})

    assert (
        sum(call_object_id == object_id for call_object_id, _pose in runtime.sync_calls)
        == sync_count
    )
    overlays = {
        overlay.actor_id: overlay for overlay in workspace.viewport.port.snapshots[-1].actors
    }
    assert overlays[tx.id].positions == tx_samples.positions
    assert overlays[tx.id].current_position == tx_samples.positions[0]
    np.testing.assert_allclose(overlays[tx.id].orientation_matrix, changed)
    assert overlays[rx.id].positions == rx_samples.positions
    assert document.scenario.actor(tx.id).orientation == FixedOrientationSpec(
        yaw_deg=55.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    assert document.scenario.actor(rx.id) == rx

    callback({"phase": "committed", "object_id": object_id, "transform": changed})
    workspace._timeline_scrubbed(3)
    overlays = {
        overlay.actor_id: overlay for overlay in workspace.viewport.port.snapshots[-1].actors
    }
    assert overlays[tx.id].positions == tx_samples.positions
    assert overlays[tx.id].current_position == tx_samples.positions[3]
    np.testing.assert_allclose(
        overlays[tx.id].orientation_matrix,
        prepared_actor_pose(tx_samples.positions[3], (55.0, 0.0, 0.0)),
    )
    assert overlays[rx.id].positions == rx_samples.positions
    assert document.scenario.actor(tx.id).mobility == tx.mobility
    assert document.scenario.actor(rx.id) == rx
    assert document.selected_actor_id == tx.id
    assert document.undo_stack.count == command_count + 1
    workspace.close_workspace()


def test_locked_target_never_attaches_semantic_gizmo(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TARGET)
    document.replace_actor(actor.with_changes(locked=True))
    workspace = _router_workspace(document)
    workspace.refresh_now()

    workspace.set_tool(AuthoringTool.MOVE)

    assert workspace.viewport.port.runtime.attachments == []
    assert workspace.viewport.port.runtime.attached_object_id is None
    workspace.close_workspace()


def test_tx_gizmo_transforms_without_compiled_target_assets(qapp) -> None:
    actor = ScenarioDocument.new().add_default_actor(ActorRole.TX)
    actor = actor.with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(10.0, 0.0, 1.0),
        ),
        orientation=FixedOrientationSpec(
            yaw_deg=10.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        ),
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    workspace = _workspace(document)
    workspace.refresh_now()
    command_count = document.undo_stack.count
    initial = prepared_actor_pose((0.0, 0.0, 1.0), (10.0, 0.0, 0.0))
    changed = prepared_actor_pose((2.0, 3.0, 4.0), (25.0, 0.0, 0.0))

    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.BEGIN, tuple(map(tuple, initial)))
    )
    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.UPDATE, tuple(map(tuple, changed)))
    )
    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.COMMIT, tuple(map(tuple, changed)))
    )

    updated = document.scenario.actor(actor.id)
    assert updated.mobility == LinearMobilitySpec(
        start_m=(2.0, 3.0, 4.0),
        end_m=(12.0, 3.0, 4.0),
    )
    assert updated.orientation.yaw_deg == pytest.approx(25.0)
    assert document.undo_stack.count == command_count + 1
    document.undo()
    assert document.scenario.actor(actor.id) == actor
    workspace.close_workspace()


@pytest.mark.parametrize(
    "orientation",
    (
        AlignMotionOrientationSpec(),
        KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(time_s=0.0),
                OrientationKeyframeSpec(time_s=1.0, yaw_deg=45.0, pitch_deg=5.0),
            )
        ),
        SpinOrientationSpec(
            axis="yaw",
            rate_deg_s=15.0,
            yaw_deg=10.0,
            pitch_deg=-3.0,
            roll_deg=2.0,
        ),
    ),
    ids=lambda value: orientation_kind(value).value,
)
def test_target_gizmo_translation_preserves_derived_orientation_mode(
    qapp,
    orientation,
) -> None:
    actor = ScenarioDocument.new().add_default_actor(ActorRole.TARGET)
    actor = actor.with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(10.0, 0.0, 1.0),
        ),
        orientation=orientation,
        target=TargetAsset.from_catalog_id("cube"),
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    workspace = _workspace(document, _FixedCompiler(_target_preview_result(actor)))
    workspace.refresh_now()
    initial = prepared_actor_pose((0.0, 0.0, 1.0), (185.0, 0.0, 0.0))
    changed = prepared_actor_pose((2.0, 3.0, 1.0), (240.0, 0.0, 0.0))

    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.BEGIN, tuple(map(tuple, initial)))
    )
    workspace._on_viewport_input(
        TransformInput(actor.id, TransformPhase.COMMIT, tuple(map(tuple, changed)))
    )

    updated = document.scenario.actor(actor.id)
    assert updated.mobility == LinearMobilitySpec(
        start_m=(2.0, 3.0, 1.0),
        end_m=(12.0, 3.0, 1.0),
    )
    assert updated.orientation == orientation
    workspace.close_workspace()


def test_pointer_hover_shows_placement_ghost_and_one_click_finishes(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    workspace = _workspace(document)
    workspace.set_tool(AuthoringTool.PLACE)

    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (5.0, 6.0)))
    workspace._on_viewport_input(HitResult((1.0, 2.0, 3.0)))
    assert workspace.viewport.port.snapshots[-1].placement_ghost == (1.0, 2.0, 3.0)

    _place(workspace, (4.0, 5.0, 6.0))
    assert document.scenario.actor(actor.id).mobility == StationaryMobilitySpec(
        position_m=(4.0, 5.0, 6.0)
    )
    assert workspace._tool is AuthoringTool.SELECT
    assert workspace.viewport.port.snapshots[-1].placement_ghost is None
    workspace.close_workspace()


def test_inspector_mobility_draft_previews_without_mutating_document_and_resets(
    qapp,
) -> None:
    class _DraftCompiler(_Compiler):
        def compile(self, scenario, *, scenario_directory=None):
            actor = scenario.actors[0]
            if isinstance(actor.mobility, LinearMobilitySpec):
                positions = (actor.mobility.start_m, actor.mobility.end_m)
            else:
                positions = (actor.mobility.position_m,)
            samples = ActorSamples(
                positions=positions,
                orientations=((0.0, 0.0, 0.0),) * len(positions),
                velocities_mps=((0.0, 0.0, 0.0),) * len(positions),
                forward_vectors=((1.0, 0.0, 0.0),) * len(positions),
            )
            return CompilationResult({}, "schema_version: 2\n", (), {actor.id: samples})

    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.TX)
    document.select(actor.id)
    workspace = _workspace(document, _DraftCompiler())
    workspace.refresh_now()
    original_revision = document.revision
    original_undo_count = document.undo_stack.count

    linear_index = workspace.mobility_editor.type_combo.findData("linear")
    workspace.mobility_editor.type_combo.setCurrentIndex(linear_index)

    assert workspace.has_pending_inspector_edits
    assert workspace._mobility_draft_pending
    assert not workspace.pending_inspector_banner.isHidden()
    assert not workspace.mobility_reset_button.isHidden()
    assert workspace.pending_apply_button.objectName() == "pendingInspectorApplyButton"
    assert document.scenario.actor(actor.id).mobility == StationaryMobilitySpec(
        position_m=(0.0, 0.0, 0.0)
    )
    assert document.revision == original_revision
    assert document.undo_stack.count == original_undo_count

    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()
    candidate_overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert candidate_overlay.pending_positions
    assert candidate_overlay.mobility_draft_pending
    assert document.revision == original_revision
    assert document.undo_stack.count == original_undo_count

    workspace.mobility_reset_button.click()
    assert not workspace.has_pending_inspector_edits
    assert workspace.pending_inspector_banner.isHidden()
    assert workspace.mobility_editor.type_combo.currentData() == "stationary"
    assert document.revision == original_revision
    workspace.close_workspace()


def test_pending_orientation_survives_document_refresh_and_applies_once(qapp) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    document.select(actor.id)
    workspace = _workspace(document)
    original_undo_count = document.undo_stack.count

    workspace.orientation_editor.fixed_angle_spins[0].setValue(37.0)
    assert workspace._orientation_draft_pending
    document.set_timeline(TimelineSettings(steps=31, duration_s=3.0))

    assert workspace.orientation_editor.fixed_angle_spins[0].value() == 37.0
    assert document.scenario.actor(actor.id).orientation == FixedOrientationSpec()
    workspace.orientation_editor.apply_button.click()
    assert document.scenario.actor(actor.id).orientation == FixedOrientationSpec(
        yaw_deg=37.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    assert not workspace.has_pending_inspector_edits
    # One timeline edit plus one orientation edit.
    assert document.undo_stack.count == original_undo_count + 2
    workspace.close_workspace()


def test_pending_group_mobility_survives_sync_and_seed_range_is_schema_bound(qapp) -> None:
    document = ScenarioDocument.new()
    group = document.add_default_group("G01")
    document.select_group(group.id)
    workspace = _workspace(document)

    workspace.mobility_editor.stationary_position_spins[0].setValue(12.5)
    assert workspace._mobility_draft_pending
    document.set_timeline(TimelineSettings(steps=32, duration_s=4.0))

    assert workspace.mobility_editor.stationary_position_spins[0].value() == 12.5
    assert document.scenario.group(group.id).mobility == StationaryMobilitySpec(
        position_m=(0.0, 0.0, 0.0)
    )
    assert workspace.group_deviation_seed.minimum() == 0
    assert workspace.group_deviation_seed.maximum() == workspace_module.MAX_RANDOM_SEED
    workspace.mobility_reset_button.click()
    assert not workspace.has_pending_inspector_edits
    workspace.close_workspace()


def test_pending_selection_cancel_keeps_owner_and_discard_allows_switch(
    qapp,
    monkeypatch,
) -> None:
    first = AuthoringActor.create(ActorRole.TX, "TX")
    second = AuthoringActor.create(ActorRole.RX, "RX")
    document = ScenarioDocument(AuthoringScenario(actors=(first, second)))
    document.select(first.id)
    workspace = _workspace(document)
    workspace.orientation_editor.fixed_angle_spins[0].setValue(20.0)

    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: workspace_module.QMessageBox.Cancel,
    )
    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (0.0, 0.0), button=1))
    workspace._on_viewport_input(
        HitResult((0.0, 0.0, 0.0), actor_id=second.id, component="mobility_handles")
    )
    assert document.selected_actor_id == first.id
    assert workspace.has_pending_inspector_edits

    monkeypatch.setattr(
        workspace_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: workspace_module.QMessageBox.Discard,
    )
    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (0.0, 0.0), button=1))
    workspace._on_viewport_input(
        HitResult((0.0, 0.0, 0.0), actor_id=second.id, component="mobility_handles")
    )
    assert document.selected_actor_id == second.id
    assert not workspace.has_pending_inspector_edits
    workspace.close_workspace()


def test_nonphysical_samples_use_observation_semantics_during_drag(qapp) -> None:
    mobility = RandomSamplingMobilitySpec(
        x_bounds_m=(-5.0, 5.0),
        y_bounds_m=(-5.0, 5.0),
        z_bounds_m=(1.0, 2.0),
        seed=7,
    )
    actor = AuthoringActor.create(ActorRole.RX, "Random RX").with_changes(mobility=mobility)
    document = ScenarioDocument(
        AuthoringScenario(
            actors=(actor,),
            timeline=TimelineSettings(steps=3, duration_s=2.0),
        )
    )
    document.select(actor.id)
    samples = ActorSamples(
        positions=((0.0, 0.0, 1.0), (4.0, -2.0, 1.5), (-3.0, 1.0, 2.0)),
        orientations=((0.0, 0.0, 0.0),) * 3,
        velocities_mps=((0.0, 0.0, 0.0),) * 3,
        forward_vectors=((1.0, 0.0, 0.0),) * 3,
        has_physical_velocity=False,
    )
    workspace = _workspace(
        document,
        _FixedCompiler(CompilationResult({}, "schema_version: 2\n", (), {actor.id: samples})),
    )
    workspace.refresh_now()

    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.trajectory_display is TrajectoryDisplayMode.OBSERVATIONS
    assert workspace.mobility_editor.average_speed_label.text() == "Average speed: —"
    moved = workspace._rigid_preview_samples(samples, translation=(2.0, 3.0, 0.0))
    assert moved.has_physical_velocity is False
    assert moved.velocities_mps is samples.velocities_mps
    assert moved.forward_vectors is samples.forward_vectors
    workspace.close_workspace()


def test_dragged_look_at_reference_reprepares_owner_orientation(qapp) -> None:
    target = AuthoringActor.create(
        ActorRole.RX,
        "Reference",
        position=(2.0, 0.0, 1.0),
    )
    owner = AuthoringActor.create(
        ActorRole.TX,
        "Observer",
        position=(0.0, 0.0, 1.0),
    ).with_changes(orientation=actor_look_at_orientation(target.id))
    scenario = AuthoringScenario(
        actors=(owner, target),
        timeline=TimelineSettings(steps=2, duration_s=1.0),
    )
    document = ScenarioDocument(scenario)
    document.select(target.id)
    owner_samples = ActorSamples(
        positions=((0.0, 0.0, 1.0),) * 2,
        orientations=((0.0, 0.0, 0.0),) * 2,
        velocities_mps=((0.0, 0.0, 0.0),) * 2,
        forward_vectors=((1.0, 0.0, 0.0),) * 2,
    )
    target_samples = ActorSamples(
        positions=((2.0, 0.0, 1.0),) * 2,
        orientations=((0.0, 0.0, 0.0),) * 2,
        velocities_mps=((0.0, 0.0, 0.0),) * 2,
        forward_vectors=((1.0, 0.0, 0.0),) * 2,
    )
    workspace = _workspace(
        document,
        _FixedCompiler(
            CompilationResult(
                {},
                "schema_version: 2\n",
                (),
                {owner.id: owner_samples, target.id: target_samples},
            )
        ),
    )
    workspace.refresh_now()
    initial = prepared_actor_pose((2.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    changed = prepared_actor_pose((0.0, 2.0, 1.0), (0.0, 0.0, 0.0))

    workspace._on_viewport_input(
        TransformInput(target.id, TransformPhase.BEGIN, tuple(map(tuple, initial)))
    )
    workspace._on_viewport_input(
        TransformInput(target.id, TransformPhase.COMMIT, tuple(map(tuple, changed)))
    )

    owner_overlay = next(
        overlay
        for overlay in workspace.viewport.port.snapshots[-1].actors
        if overlay.actor_id == owner.id
    )
    assert owner_overlay.orientation_matrix is not None
    np.testing.assert_allclose(
        np.asarray(owner_overlay.orientation_matrix)[:2, :2],
        ((0.0, -1.0), (1.0, 0.0)),
        atol=1e-6,
    )
    workspace.close_workspace()


def test_banner_apply_is_atomic_for_multiple_sections_and_failure_rolls_back(
    qapp,
    monkeypatch,
) -> None:
    document = ScenarioDocument.new()
    actor = document.add_default_actor(ActorRole.RX)
    document.select(actor.id)
    workspace = _workspace(document)
    linear_index = workspace.mobility_editor.type_combo.findData("linear")
    workspace.mobility_editor.type_combo.setCurrentIndex(linear_index)
    workspace.orientation_editor.fixed_angle_spins[0].setValue(42.0)
    before = document.scenario
    before_revision = document.revision
    before_undo_count = document.undo_stack.count
    monkeypatch.setattr(
        workspace.orientation_editor,
        "orientation",
        lambda: (_ for _ in ()).throw(ValueError("invalid orientation draft")),
    )
    monkeypatch.setattr(workspace_module.QMessageBox, "warning", lambda *_args: None)

    assert workspace._apply_all_pending_inspector_edits() is False
    assert document.scenario == before
    assert document.revision == before_revision
    assert document.undo_stack.count == before_undo_count
    assert workspace.has_pending_inspector_edits

    monkeypatch.undo()
    assert workspace._apply_all_pending_inspector_edits() is True
    updated = document.scenario.actor(actor.id)
    assert isinstance(updated.mobility, LinearMobilitySpec)
    assert updated.orientation == FixedOrientationSpec(
        yaw_deg=42.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    assert document.undo_stack.count == before_undo_count + 1
    assert not workspace.has_pending_inspector_edits
    workspace.close_workspace()


def test_target_and_group_settings_are_protected_pending_sections(qapp) -> None:
    target = AuthoringActor.create(ActorRole.TARGET, "Car").with_changes(
        target=TargetAsset.from_catalog_id("cube")
    )
    group = AuthoringGroup.create("G01")
    document = ScenarioDocument(
        AuthoringScenario(
            actors=(target,),
            groups=(group,),
        )
    )
    document.select(target.id)
    workspace = _workspace(document)

    workspace.target_scale_spin.setValue(2.5)
    assert workspace._target_draft_pending
    assert not workspace.target_reset_button.isHidden()
    assert document.scenario.actor(target.id).target.scale == 1.0
    workspace.target_reset_button.click()
    assert workspace.target_scale_spin.value() == 1.0
    assert not workspace.has_pending_inspector_edits

    document.select_group(group.id)
    workspace.group_deviation_enabled.setChecked(True)
    workspace.group_deviation_spins[0].setValue(0.25)
    assert workspace._group_settings_draft_pending
    assert not workspace.group_settings_reset_button.isHidden()
    assert document.scenario.group(group.id).deviation is None
    workspace.group_settings_reset_button.click()
    assert document.scenario.group(group.id).deviation is None
    assert not workspace.has_pending_inspector_edits
    workspace.close_workspace()


def test_pending_target_candidate_exposes_compiled_ghost_mesh(qapp) -> None:
    class _TargetDraftCompiler(_Compiler):
        def compile(self, scenario, *, scenario_directory=None):
            return _target_preview_result(scenario.actors[0])

    actor = AuthoringActor.create(ActorRole.TARGET, "Car").with_changes(
        target=TargetAsset.from_catalog_id("cube")
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    workspace = _workspace(document, _TargetDraftCompiler())
    workspace.refresh_now()

    workspace.target_scale_spin.setValue(3.0)
    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()

    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.pending_target_asset is not None
    assert document.scenario.actor(actor.id).target.scale == 1.0
    workspace.close_workspace()


def test_recreated_waypoint_and_keyframe_cell_editors_mark_pending(qapp) -> None:
    actor = AuthoringActor.create(ActorRole.RX, "RX").with_changes(
        mobility=WaypointMobilitySpec(points_m=((0.0, 0.0, 1.0), (5.0, 0.0, 1.0))),
        orientation=KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(time_s=0.0),
                OrientationKeyframeSpec(time_s=3.0, yaw_deg=20.0),
            )
        ),
    )
    document = ScenarioDocument(AuthoringScenario(actors=(actor,)))
    document.select(actor.id)
    workspace = _workspace(document)

    waypoint_x = workspace.mobility_editor.waypoint_table.cellWidget(0, 0)
    assert isinstance(waypoint_x, QAbstractSpinBox)
    waypoint_x.setProperty("value", 1.25)
    waypoint_x.setValue(1.25)
    assert workspace._mobility_draft_pending
    workspace.mobility_reset_button.click()

    keyframe_yaw = workspace.orientation_editor.keyframes_table.cellWidget(1, 1)
    assert isinstance(keyframe_yaw, QAbstractSpinBox)
    keyframe_yaw.setValue(45.0)
    assert workspace._orientation_draft_pending
    workspace.orientation_reset_button.click()

    # A normal selection sync destroys and recreates every table cell widget.
    document.select(None)
    document.select(actor.id)
    recreated = workspace.mobility_editor.waypoint_table.cellWidget(1, 0)
    recreated.setValue(7.5)
    assert workspace._mobility_draft_pending
    workspace.close_workspace()


def test_actor_sample_cache_key_tracks_resource_registration_source() -> None:
    actor = AuthoringActor.create(ActorRole.RX, "RX")
    first_source = Path(__file__)
    second_source = Path(workspace_module.__file__)
    first = AuthoringResource(
        ResourceKind.POSITION_SEQUENCE,
        first_source,
        "resources/positions.csv",
    )
    second = AuthoringResource(
        ResourceKind.POSITION_SEQUENCE,
        second_source,
        "resources/positions.csv",
    )
    first_scenario = AuthoringScenario(actors=(actor,), resources=(first,))
    second_scenario = AuthoringScenario(actors=(actor,), resources=(second,))

    assert ScenarioAuthoringWorkspace._sample_cache_key(
        actor,
        first_scenario,
    ) != ScenarioAuthoringWorkspace._sample_cache_key(
        actor,
        second_scenario,
    )


def test_actor_sample_cache_key_tracks_member_and_look_at_target_names() -> None:
    group = AuthoringGroup.create("Formation").with_changes(
        deviation=GroupDeviationSpec(max_right_m=1.0, seed=7)
    )
    member = AuthoringActor.create(ActorRole.TX, "Member A").with_changes(
        mobility=GroupMemberMobilitySpec(group=str(group.id))
    )
    target = AuthoringActor.create(ActorRole.RX, "Reference A")
    observer = AuthoringActor.create(ActorRole.TX, "Observer").with_changes(
        orientation=actor_look_at_orientation(target.id)
    )
    scenario = AuthoringScenario(
        actors=(member, target, observer),
        groups=(group,),
    )

    renamed_member = member.with_changes(name="Member B")
    member_scenario = scenario.replace_actor(renamed_member)
    assert ScenarioAuthoringWorkspace._sample_cache_key(
        member,
        scenario,
    ) != ScenarioAuthoringWorkspace._sample_cache_key(
        renamed_member,
        member_scenario,
    )

    renamed_target = target.with_changes(name="Reference B")
    target_scenario = scenario.replace_actor(renamed_target)
    assert ScenarioAuthoringWorkspace._sample_cache_key(
        observer,
        scenario,
    ) != ScenarioAuthoringWorkspace._sample_cache_key(
        observer,
        target_scenario,
    )


def test_selected_group_position_rejects_stale_compilation(qapp) -> None:
    group = AuthoringGroup.create("G01").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(10.0, 0.0, 1.0),
        )
    )
    original = AuthoringScenario(actors=(), groups=(group,))
    stale_samples = SimpleNamespace(
        positions=((0.0, 0.0, 1.0), (10.0, 0.0, 1.0)),
        has_physical_velocity=True,
        frame_transform=lambda **_kwargs: tuple(map(tuple, np.eye(4))),
    )
    result = CompilationResult(
        {},
        "schema_version: 2\n",
        (),
        {},
        group_samples={group.id: stale_samples},
    )
    document = ScenarioDocument(original)
    document.select_group(group.id)
    workspace = _workspace(document, _FixedCompiler(result))
    workspace.refresh_now()

    document.replace_group(
        group.with_changes(
            mobility=LinearMobilitySpec(
                start_m=(50.0, 5.0, 2.0),
                end_m=(60.0, 5.0, 2.0),
            )
        )
    )

    assert workspace._selected_current_position() == (50.0, 5.0, 2.0)
    workspace.close_workspace()


def test_group_member_translation_uses_semantic_current_group_samples(
    qapp,
    monkeypatch,
) -> None:
    group = AuthoringGroup.create("G01")
    member = AuthoringActor.create(ActorRole.RX, "Member").with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=1.0, forward=2.0, up=3.0),
        )
    )
    document = ScenarioDocument(
        AuthoringScenario(
            actors=(member,),
            groups=(group,),
        )
    )
    workspace = _workspace(document)
    stale = SimpleNamespace(
        group_samples={group.id: object()},
        group_offsets=lambda *_args, **_kwargs: (
            (0.0, 0.0, 0.0),
            (9.0, 9.0, 9.0),
        ),
    )
    workspace.compilation = stale
    current = SimpleNamespace(
        positions=((0.0, 0.0, 0.0),),
        has_physical_velocity=False,
        offsets_for_world_positions=lambda *_args, **_kwargs: (
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
        ),
    )
    monkeypatch.setattr(workspace, "_prepared_group_samples", lambda _group: current)

    updated = workspace._translate_subject_mobility(
        member,
        (4.0, 5.0, 6.0),
        reference_position=(0.0, 0.0, 0.0),
    )

    assert updated.offset_m == GroupOffsetSpec(right=2.0, forward=4.0, up=6.0)
    workspace.close_workspace()


def test_group_control_drag_keeps_members_on_exact_canonical_preview(qapp) -> None:
    group = AuthoringGroup.create("G01").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(4.0, 0.0, 1.0),
        )
    )
    tx = AuthoringActor.create(ActorRole.TX, "TX").with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=-1.0),
        )
    )
    rx = AuthoringActor.create(ActorRole.RX, "RX").with_changes(
        mobility=GroupMemberMobilitySpec(
            group=str(group.id),
            offset_m=GroupOffsetSpec(right=1.0),
        )
    )
    document = ScenarioDocument(
        AuthoringScenario(
            scene=SceneReference("library", "empty/empty.xml"),
            actors=(tx, rx),
            groups=(group,),
            timeline=TimelineSettings(steps=5, duration_s=4.0),
        )
    )
    document.select_group(group.id)
    compiled = ScenarioCompiler().compile(document.scenario)
    workspace = _workspace(document, _FixedCompiler(compiled))
    workspace.refresh_now()
    workspace.set_tool(AuthoringTool.MOVE)
    source = HitResult(
        world_position=(4.0, 0.0, 1.0),
        renderer_object_id="authoring:mobility_control_end",
        actor_id=group.id,
        component="mobility_control_end",
    )

    workspace._on_viewport_input(PointerInput(PointerPhase.DOWN, (1.0, 1.0), button=1))
    workspace._on_viewport_input(source)
    workspace._on_viewport_input(PointerInput(PointerPhase.MOVE, (2.0, 2.0), button=1))
    workspace._on_viewport_input(
        HitResult(
            world_position=(8.0, 3.0, 1.0),
            renderer_object_id=source.renderer_object_id,
            actor_id=group.id,
            component=source.component,
        )
    )

    expected = ScenarioCompiler().compile(document.scenario)
    overlays = {
        overlay.actor_id: overlay for overlay in workspace.viewport.port.snapshots[-1].actors
    }
    assert overlays[tx.id].positions == expected.samples[tx.id].positions
    assert overlays[rx.id].positions == expected.samples[rx.id].positions
    assert overlays[group.id].frame_samples == expected.group_samples[group.id].positions
    workspace._finish_mobility_drag(commit=False)
    workspace.close_workspace()


@pytest.mark.parametrize("grouped_target", (False, True), ids=("actor", "group_member"))
def test_pending_motion_previews_dependent_look_at_pose_and_ray(
    qapp,
    grouped_target,
) -> None:
    group = AuthoringGroup.create("G01").with_changes(
        mobility=LinearMobilitySpec(
            start_m=(0.0, 0.0, 1.0),
            end_m=(4.0, 0.0, 1.0),
        )
    )
    target = AuthoringActor.create(ActorRole.TX, "Target").with_changes(
        mobility=(
            GroupMemberMobilitySpec(
                group=str(group.id),
                offset_m=GroupOffsetSpec(),
            )
            if grouped_target
            else LinearMobilitySpec(
                start_m=(0.0, 0.0, 1.0),
                end_m=(4.0, 0.0, 1.0),
            )
        )
    )
    observer = AuthoringActor.create(
        ActorRole.RX,
        "Observer",
        position=(0.0, 5.0, 1.0),
    ).with_changes(orientation=actor_look_at_orientation(target.id))
    scenario = AuthoringScenario(
        actors=(target, observer),
        groups=(group,) if grouped_target else (),
        timeline=TimelineSettings(steps=2, duration_s=1.0),
    )
    document = ScenarioDocument(scenario)
    if grouped_target:
        document.select_group(group.id)
    else:
        document.select(target.id)

    class _LookAtDraftCompiler(_Compiler):
        def compile(self, candidate, *, scenario_directory=None):
            motion = (
                candidate.group(group.id).mobility
                if grouped_target
                else candidate.actor(target.id).mobility
            )
            assert isinstance(motion, LinearMobilitySpec)
            target_positions = (motion.start_m, motion.end_m)
            observer_positions = ((0.0, 5.0, 1.0),) * 2
            observer_orientations = tuple(
                (position[0] * 10.0, 0.0, 0.0) for position in target_positions
            )
            return CompilationResult(
                {},
                "schema_version: 2\n",
                (),
                {
                    target.id: ActorSamples(
                        positions=target_positions,
                        orientations=((0.0, 0.0, 0.0),) * 2,
                    ),
                    observer.id: ActorSamples(
                        positions=observer_positions,
                        orientations=observer_orientations,
                    ),
                },
            )

    workspace = _workspace(document, _LookAtDraftCompiler())
    workspace._play_step = 1
    workspace.refresh_now()
    workspace.mobility_editor.linear_end_spins[0].setValue(8.0)
    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()

    overlays = {
        overlay.actor_id: overlay for overlay in workspace.viewport.port.snapshots[-1].actors
    }
    assert overlays[target.id].pending_positions[-1] == (8.0, 0.0, 1.0)
    assert overlays[observer.id].pending_positions == ()
    assert overlays[observer.id].pending_orientation_matrix is not None
    assert overlays[observer.id].pending_look_at_position == (8.0, 0.0, 1.0)
    workspace.close_workspace()


def test_type_conversion_reconnects_dynamic_candidate_cell_editors(qapp) -> None:
    class _DynamicDraftCompiler(_Compiler):
        def compile(self, scenario, *, scenario_directory=None):
            actor = scenario.actors[0]
            if isinstance(actor.mobility, WaypointMobilitySpec):
                positions = actor.mobility.points_m
            else:
                positions = ((0.0, 0.0, 1.0),) * scenario.timeline.steps
            if isinstance(actor.orientation, KeyframesOrientationSpec):
                start = actor.orientation.keyframes[0]
                end = actor.orientation.keyframes[-1]
                orientations = (
                    (start.yaw_deg, start.pitch_deg, start.roll_deg),
                    (end.yaw_deg, end.pitch_deg, end.roll_deg),
                )
            else:
                orientations = ((0.0, 0.0, 0.0),) * len(positions)
            return CompilationResult(
                {},
                "schema_version: 2\n",
                (),
                {
                    actor.id: ActorSamples(
                        positions=positions,
                        orientations=orientations,
                    )
                },
            )

    actor = AuthoringActor.create(ActorRole.RX, "RX", position=(0.0, 0.0, 1.0))
    document = ScenarioDocument(
        AuthoringScenario(
            actors=(actor,),
            timeline=TimelineSettings(steps=2, duration_s=1.0),
        )
    )
    document.select(actor.id)
    workspace = _workspace(document, _DynamicDraftCompiler())
    workspace.refresh_now()

    waypoint_index = workspace.mobility_editor.type_combo.findData("waypoint")
    workspace.mobility_editor.type_combo.setCurrentIndex(waypoint_index)
    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()
    assert workspace._candidate_compilation is not None
    waypoint_x = workspace.mobility_editor.waypoint_table.cellWidget(1, 0)
    waypoint_x.setValue(6.0)
    assert waypoint_x.value() == 6.0
    assert workspace.mobility_editor.mobility().points_m[1][0] == 6.0
    assert workspace._candidate_compilation is None
    assert workspace._candidate_compile_timer.isActive()
    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()
    assert workspace._candidate_scenario.actor(actor.id).mobility.points_m[1][0] == 6.0
    assert workspace._candidate_compilation.samples[actor.id].positions[1][0] == 6.0
    assert workspace.viewport.port.snapshots[-1].actors[0].pending_positions[1][0] == 6.0

    workspace._reset_pending_mobility()
    keyframes_index = workspace.orientation_editor.type_combo.findData("keyframes")
    workspace.orientation_editor.type_combo.setCurrentIndex(keyframes_index)
    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()
    assert workspace._candidate_compilation is not None
    keyframe_yaw = workspace.orientation_editor.keyframes_table.cellWidget(1, 1)
    keyframe_yaw.setValue(65.0)
    assert workspace._candidate_compilation is None
    workspace._candidate_compile_timer.stop()
    workspace._compile_candidate_draft()
    overlay = workspace.viewport.port.snapshots[-1].actors[0]
    assert overlay.pending_orientation_matrix is not None
    workspace.close_workspace()
