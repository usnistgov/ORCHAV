"""Single-window Qt workspace for the Scenario Builder."""

from __future__ import annotations

import math
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from generator.core.scenario_actors import (
    PreparedMobility,
    Quaternion,
    Timeline,
    apply_asset_alignment,
    prepare_orientation,
    prepare_scenario,
)
from shared.scenarios.actors import (
    CircularMobilitySpec,
    FixedOrientationSpec,
    GroupDeviationSpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    KeyframesOrientationSpec,
    LinearMobilitySpec,
    LookAtOrientationSpec,
    MeshSequenceMobilitySpec,
    NetworkRouteMobilitySpec,
    SpinOrientationSpec,
    StationaryMobilitySpec,
    WaypointMobilitySpec,
)
from shared.scenarios.parsers import available_sionna_scene_ids
from shared.scenarios.paths import find_project_root
from shared.scenarios.yaml import validate_scenario_data

from ..scene.target_transforms import sionna_orientation_from_rotation_matrix
from ..widgets.form_layout import AlignedFormLayout
from .assets import prepared_actor_pose
from .compilation_scheduler import AuthoringCompilationScheduler, CompilationFailure
from .compiler import (
    ActorSamples,
    CompilationResult,
    GroupSamples,
    IssueSeverity,
    ScenarioCompiler,
    ScenePreviewResult,
    canonical_scenario_mapping,
)
from .document import DocumentEvent, DocumentEventKind, ScenarioDocument
from .domain import (
    ActorRole,
    AuthoringActor,
    AuthoringGroup,
    AuthoringResource,
    AuthoringScenario,
    QualityPreset,
    ResourceKind,
    SceneReference,
    TargetAsset,
    TimelineSettings,
    translate_mobility,
)
from .mobility_control_rig import (
    MobilityControlDescriptor,
    mobility_control_rig,
    update_mobility_from_rig_control,
)
from .mobility_models import MobilityNeedsContextError, convert_mobility
from .model_capabilities import mobility_capability
from .orientation_models import (
    convert_orientation,
    look_at_actor_id,
    orientation_kind,
    orientation_to_mapping,
)
from .persistence import resolve_authoring_resource
from .undo import QtUndoStackAdapter
from .viewport_port import (
    ActorOverlaySnapshot,
    ActorVisualState,
    AuthoringTool,
    HitResult,
    KeyboardInput,
    OverlaySnapshot,
    OverlayVisibility,
    PointerInput,
    PointerPhase,
    PreviewProvenance,
    SceneOverlayAsset,
    TargetOverlayAsset,
    TrajectoryDisplayMode,
    TransformInput,
    TransformPhase,
)

_QT_TIMER_MAX_INTERVAL_MS = 2_147_483_647
_PRESERVED_QUALITY_CHOICE = "preserved-imported-quality"


from .workspace_ui import AuthoringWorkspaceUI


@dataclass(frozen=True)
class _MotionDraftBaseline:
    """Editor state restored by Escape without discarding earlier pending edits."""

    scenario: AuthoringScenario
    subject_id: UUID | None
    is_group: bool
    mobility_pending: bool
    orientation_pending: bool
    preview_scenario: AuthoringScenario | None
    preview: CompilationResult | None


class ScenarioAuthoringWorkspace(AuthoringWorkspaceUI, QWidget):
    """Integrated actor tree, viewport, inspector, and workflow drawer."""

    dirty_changed = Signal(bool)
    title_changed = Signal(str)
    save_requested = Signal()
    save_as_requested = Signal()
    generate_requested = Signal()
    cancel_generation_requested = Signal()
    preview_generated_requested = Signal(object)
    leave_authoring_requested = Signal()
    inspect_recording_requested = Signal()
    create_variant_requested = Signal()

    def __init__(
        self,
        visualizer: Any,
        document: ScenarioDocument | None = None,
        parent: QWidget | None = None,
        *,
        compiler: ScenarioCompiler | None = None,
        viewport_factory: Callable[[Any, QWidget | None], QWidget] | None = None,
    ) -> None:
        super().__init__(parent)
        self.visualizer = visualizer
        if compiler is None:
            project_root = getattr(visualizer, "current_project_root", None)
            if document is not None and document.path is not None:
                document_root = find_project_root(document.path.parent)
                if (document_root / "libraries").is_dir():
                    project_root = document_root
            compiler = ScenarioCompiler(project_root)
        self.compiler = compiler
        self.document: ScenarioDocument | None = None
        self.compilation: CompilationResult | None = None
        self._early_scene: ScenePreviewResult | None = None
        self.camera_controls = None
        self._compile_progress = "Checking scenario settings and resources"
        self._compiled_scenario: AuthoringScenario | None = None
        self._compiled_directory: Path | None = None
        self._compilation_failure: CompilationFailure | None = None
        self._unsubscribe_document: Callable[[], None] | None = None
        self._syncing = False
        self._compile_scheduled = False
        self._explicit_validation_running = False
        self._tree_refresh_scheduled = False
        self._compile_request_token = 0
        self._compile_lock = threading.Lock()
        self._compile_scheduler: AuthoringCompilationScheduler | None = None
        self._candidate_compile_scheduler: AuthoringCompilationScheduler | None = None
        self._candidate_request_token = 0
        self._candidate_scenario: AuthoringScenario | None = None
        self._candidate_compilation: CompilationResult | None = None
        self._candidate_failure: CompilationFailure | None = None
        self._candidate_requested_scenario: AuthoringScenario | None = None
        self._pending_subject_id: UUID | None = None
        self._pending_subject_is_group = False
        self._mobility_draft_pending = False
        self._orientation_draft_pending = False
        self._target_draft_pending = False
        self._group_settings_draft_pending = False
        self._timeline_draft_pending = False
        self._dependent_drag_preview_samples: dict[UUID, ActorSamples] = {}
        self._last_pointer_phase: PointerPhase | None = None
        self._tool = AuthoringTool.SELECT
        self._waypoint_session_actor_id: UUID | None = None
        self._waypoint_base_count = 0
        self._pending_waypoints: list[tuple[float, float, float]] = []
        self._waypoint_baseline_actor: AuthoringActor | None = None
        self._waypoint_key_latch: set[str] = set()
        self._mobility_drag_actor_id: UUID | None = None
        self._mobility_drag_control: MobilityControlDescriptor | None = None
        self._mobility_drag_baseline_actor: AuthoringActor | AuthoringGroup | None = None
        self._mobility_drag_origin: tuple[float, float, float] | None = None
        self._mobility_drag_baseline_position: tuple[float, float, float] | None = None
        self._mobility_drag_baseline_samples: ActorSamples | None = None
        self._mobility_drag_preview_samples: ActorSamples | None = None
        self._mobility_drag_baseline_group_positions: (
            tuple[tuple[float, float, float], ...] | None
        ) = None
        self._mobility_drag_preview_group_positions: (
            tuple[tuple[float, float, float], ...] | None
        ) = None
        self._mobility_drag_baseline_group_physical: bool | None = None
        self._mobility_drag_baseline_group_member_samples: dict[UUID, ActorSamples] = {}
        self._mobility_drag_baseline_group_samples: GroupSamples | None = None
        self._mobility_drag_preview_group_samples: GroupSamples | None = None
        self._placement_ghost: tuple[float, float, float] | None = None
        self._hovered_actor_id: UUID | None = None
        self._hovered_component: str | None = None
        self._motion_hover_clear_pending = False
        self._transform_actor_id: UUID | None = None
        self._transform_baseline_actor: AuthoringActor | None = None
        self._transform_baseline_position: np.ndarray | None = None
        self._transform_baseline_rotation: np.ndarray | None = None
        self._transform_baseline_samples: ActorSamples | None = None
        self._transform_preview_samples: ActorSamples | None = None
        self._motion_draft_baseline: _MotionDraftBaseline | None = None
        self._read_only = False
        self._preview_result_available = False
        self._play_step = 0
        self._play_origin_step = 0
        self._play_origin_time = time.monotonic()
        self._prepared_sample_cache: dict[UUID, tuple[tuple[Any, ...], ActorSamples]] = {}
        self._prepared_group_position_cache: dict[
            UUID,
            tuple[
                tuple[Any, ...],
                tuple[tuple[float, float, float], ...],
            ],
        ] = {}
        self._prepared_group_physical_cache: dict[
            UUID,
            tuple[tuple[Any, ...], bool],
        ] = {}
        self._prepared_group_sample_cache: dict[
            UUID,
            tuple[tuple[Any, ...], GroupSamples],
        ] = {}
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)
        self._compile_status_timer = QTimer(self)
        self._compile_status_timer.setInterval(1000)
        self._compile_status_timer.timeout.connect(self._update_preview_status)
        self._compile_started_at = time.monotonic()
        self._candidate_compile_timer = QTimer(self)
        self._candidate_compile_timer.setSingleShot(True)
        self._candidate_compile_timer.setInterval(180)
        self._candidate_compile_timer.timeout.connect(self._compile_candidate_draft)
        self.setObjectName("scenarioAuthoringWorkspace")
        self._build_ui(viewport_factory)
        self._install_actions()
        if isinstance(self.compiler, ScenarioCompiler):
            project_root = self.compiler.project_root
            self._compile_scheduler = AuthoringCompilationScheduler(
                lambda: ScenarioCompiler(project_root),
                compile_lock=self._compile_lock,
                parent=self,
            )
            self._compile_scheduler.succeeded.connect(self._background_compile_succeeded)
            self._compile_scheduler.failed.connect(self._background_compile_failed)
            self._compile_scheduler.scene_ready.connect(self._background_scene_ready)
            self._compile_scheduler.progress_changed.connect(self._background_compile_progress)
            self._candidate_compile_scheduler = AuthoringCompilationScheduler(
                lambda: ScenarioCompiler(project_root),
                parent=self,
            )
            self._candidate_compile_scheduler.succeeded.connect(self._candidate_compile_succeeded)
            self._candidate_compile_scheduler.failed.connect(self._candidate_compile_failed)
        self.bind_document(document or ScenarioDocument.new(undo_stack=QtUndoStackAdapter()))

    @property
    def has_pending_waypoint_session(self) -> bool:
        """Return whether an explicit Finish/Cancel decision is still pending."""

        return self._waypoint_session_actor_id is not None

    @property
    def has_pending_inspector_edits(self) -> bool:
        """Return whether the inspector contains unapplied authored values."""

        return any(
            (
                self._mobility_draft_pending,
                self._orientation_draft_pending,
                self._target_draft_pending,
                self._group_settings_draft_pending,
            )
        )

    @property
    def has_pending_timeline_edits(self) -> bool:
        """Return whether timeline controls differ from the applied document."""

        return self._timeline_draft_pending

    @property
    def compilation_lock(self) -> AbstractContextManager[object]:
        """Return the guard shared by background, explicit, and save compilation."""

        return self._compile_lock

    # -- layout ---------------------------------------------------------

    def _viewport_appearance_changed(self) -> None:
        """Change only the authoring display, leaving the scenario draft untouched."""
        port = getattr(getattr(self, "viewport", None), "port", None)
        if callable(getattr(port, "set_node_color_mode", None)):
            port.set_node_color_mode(self.node_color_mode.currentData())

    def _appearance_preset_changed(self, *_args) -> None:
        port = getattr(getattr(self, "viewport", None), "port", None)
        apply = getattr(port, "set_appearance_preset", None)
        if callable(apply):
            try:
                if self.appearance_preset.currentText() == "Scenario appearance":
                    self._presented_document = None
                    self._apply_scenario_presentation()
                    notice = ""
                else:
                    notice = apply(self.appearance_preset.currentText())
            except (ValueError, RuntimeError, OSError) as exc:
                notice = str(exc)
            self.appearance_preset.setToolTip(notice or "Display preset for the Scenario Builder.")

    def show_problems(self) -> None:
        """Reveal the existing diagnostics without changing preview or selection."""

        self.details_toggle.setChecked(True)
        self.drawer.setCurrentWidget(self.problems_tree)
        self.problems_tree.setFocus(Qt.OtherFocusReason)

    def _connect_editor_draft_signals(self, editor: QWidget, kind: str) -> None:
        """Aggregate editor-owned controls into one workspace draft signal."""

        def mark(widget, *_args: object) -> None:
            if not self._syncing:
                widget.setProperty("pendingChange", True)
                widget.setStyleSheet("border-left: 2px solid #32c8bc;")
            self._mark_inspector_draft(kind)

        for widget_type, signal_name in (
            (QDoubleSpinBox, "valueChanged"),
            (QSpinBox, "valueChanged"),
            (QComboBox, "currentIndexChanged"),
            (QCheckBox, "toggled"),
            (QLineEdit, "textChanged"),
            (QTableWidget, "itemChanged"),
        ):
            for widget in editor.findChildren(widget_type):
                property_name = f"orchavDraftSignal:{kind}"
                if widget.property(property_name):
                    continue
                getattr(widget, signal_name).connect(lambda *args, field=widget: mark(field, *args))
                widget.setProperty(property_name, True)
        for button in editor.findChildren(QPushButton):
            property_name = f"orchavDraftButton:{kind}"
            if button.property(property_name):
                continue
            if button in {
                getattr(editor, "apply_button", None),
                getattr(editor, "draw_waypoints_button", None),
            }:
                button.setProperty(property_name, True)
                continue
            button.clicked.connect(
                lambda _checked=False, draft_kind=kind: QTimer.singleShot(
                    0,
                    lambda: self._refresh_dynamic_editor_draft_signals(
                        editor,
                        draft_kind,
                    ),
                )
            )
            button.setProperty(property_name, True)

    def _refresh_dynamic_editor_draft_signals(
        self,
        editor: QWidget,
        kind: str,
    ) -> None:
        """Connect newly inserted table editors, then mark their section."""

        self._connect_editor_draft_signals(editor, kind)
        self._mark_inspector_draft(kind)

    def _selected_subject(self) -> tuple[UUID, bool] | None:
        document = self.document
        if document is None:
            return None
        if document.selected_actor is not None:
            return document.selected_actor.id, False
        if document.selected_group is not None:
            return document.selected_group.id, True
        return None

    def _pending_matches_current_subject(self) -> bool:
        subject = self._selected_subject()
        return subject is not None and subject == (
            self._pending_subject_id,
            self._pending_subject_is_group,
        )

    def _mark_inspector_draft(self, kind: str) -> None:
        """Record an unapplied inspector edit and debounce canonical preview."""

        if self._syncing or self.document is None or self.document.read_only:
            return
        subject = self._selected_subject()
        if subject is None:
            return
        subject_id, is_group = subject
        if kind == "orientation" and is_group:
            return
        if kind == "target":
            actor = self.document.scenario.actor(subject_id)
            if is_group or actor is None or actor.role is not ActorRole.TARGET:
                return
        if kind == "group_settings" and not is_group:
            return
        if self.has_pending_inspector_edits and (
            subject_id != self._pending_subject_id or is_group != self._pending_subject_is_group
        ):
            return
        self._pending_subject_id = subject_id
        self._pending_subject_is_group = is_group
        if kind == "mobility":
            self._mobility_draft_pending = True
            try:
                self._pending_mobility_value = self.mobility_editor.mobility()
            except (TypeError, ValueError):
                pass
        elif kind == "orientation":
            self._orientation_draft_pending = True
        elif kind == "target":
            self._target_draft_pending = True
        elif kind == "group_settings":
            self._group_settings_draft_pending = True
        else:
            raise ValueError(f"unknown inspector draft kind: {kind}")
        token = self._next_candidate_token()
        if self._candidate_compile_scheduler is not None:
            self._candidate_compile_scheduler.invalidate(token)
        self._candidate_failure = None
        self._candidate_requested_scenario = None
        self._candidate_compile_timer.start()
        self._update_pending_inspector_ui()
        self._sync_motion_edit_action()
        self._refresh_viewport()

    def _update_pending_inspector_ui(self) -> None:
        pending = self.has_pending_inspector_edits
        self.pending_inspector_banner.setVisible(pending)
        for group, title, changed in (
            (self.mobility_group, "Trajectory", self._mobility_draft_pending),
            (self.orientation_group, "Orientation", self._orientation_draft_pending),
            (self.target_group, "Target", self._target_draft_pending),
            (self.group_settings, "Group jitter", self._group_settings_draft_pending),
        ):
            group.setTitle(title + (" *" if changed else ""))
        if not pending:
            for widget in self.inspector_panel.findChildren(QWidget):
                if widget.property("pendingChange"):
                    widget.setProperty("pendingChange", False)
                    widget.setStyleSheet("")
            self.pending_inspector_label.clear()
            return
        kinds = []
        if self._mobility_draft_pending:
            kinds.append("mobility")
        if self._orientation_draft_pending:
            kinds.append("orientation")
        if self._target_draft_pending:
            kinds.append("target configuration")
        if self._group_settings_draft_pending:
            kinds.append("group jitter")
        subject = None
        if self.document is not None and self._pending_subject_id is not None:
            subject = (
                self.document.scenario.group(self._pending_subject_id)
                if self._pending_subject_is_group
                else self.document.scenario.actor(self._pending_subject_id)
            )
        label = " and ".join(kinds).capitalize()
        subject_name = getattr(subject, "name", "selected subject")
        if self._candidate_failure is not None:
            detail = f" {self._candidate_failure.message} · Last valid preview retained."
        elif (
            self._candidate_compile_timer.isActive()
            or self._candidate_requested_scenario != self._candidate_scenario
        ):
            detail = (
                " Preparing preview; last valid preview retained."
                if self._candidate_compilation
                else " Preparing canonical candidate preview..."
            )
        elif self._candidate_compilation is not None:
            errors = tuple(
                issue
                for issue in self._candidate_compilation.issues
                if issue.severity is IssueSeverity.ERROR
                and (
                    issue.group_id == self._pending_subject_id
                    if self._pending_subject_is_group
                    else issue.actor_id == self._pending_subject_id
                )
            )
            detail = (
                f" Candidate has {len(errors)} validation error{'s' if len(errors) != 1 else ''}."
                if errors
                else " Canonical candidate preview is ready."
            )
        elif self._candidate_compile_timer.isActive():
            detail = " Preparing canonical candidate preview..."
        else:
            detail = " Enter valid values to preview the candidate."
        self.pending_inspector_label.setText(f"{label} for {subject_name} has changed.{detail}")

    def _candidate_scenario_from_editors(self) -> AuthoringScenario:
        document = self.document
        subject_id = self._pending_subject_id
        if document is None or subject_id is None:
            raise ValueError("no pending inspector subject")
        scenario = document.scenario
        resources = list(scenario.resources)
        if self._pending_subject_is_group:
            group = scenario.group(subject_id)
            if group is None:
                raise ValueError("pending group no longer exists")
            if self._mobility_draft_pending:
                mobility, additions = self._internalize_mobility_resource(
                    self.mobility_editor.mobility()
                )
                if isinstance(mobility, GroupMemberMobilitySpec):
                    raise ValueError("A group cannot reference another group.")
                scenario = scenario.replace_group(group.with_changes(mobility=mobility))
                resources.extend(additions)
            if self._group_settings_draft_pending:
                deviation = None
                if self.group_deviation_enabled.isChecked():
                    deviation = GroupDeviationSpec(
                        max_right_m=self.group_deviation_spins[0].value(),
                        max_forward_m=self.group_deviation_spins[1].value(),
                        max_up_m=self.group_deviation_spins[2].value(),
                        seed=self.group_deviation_seed.value(),
                    )
                current_group = scenario.group(subject_id)
                assert current_group is not None
                scenario = scenario.replace_group(current_group.with_changes(deviation=deviation))
        else:
            actor = scenario.actor(subject_id)
            if actor is None:
                raise ValueError("pending actor no longer exists")
            if self._mobility_draft_pending:
                mobility, additions = self._internalize_mobility_resource(
                    self.mobility_editor.mobility()
                )
                actor = actor.with_changes(mobility=mobility)
                resources.extend(additions)
            if self._orientation_draft_pending:
                actor = actor.with_changes(orientation=self.orientation_editor.orientation())
            if self._target_draft_pending:
                actor = actor.with_changes(**self._target_changes_from_controls(actor))
            scenario = scenario.replace_actor(actor)
        if resources:
            by_path = {resource.relative_path: resource for resource in resources}
            scenario = replace(scenario, resources=tuple(by_path.values()))
        return scenario

    def _next_candidate_token(self) -> int:
        self._candidate_request_token += 1
        return self._candidate_request_token

    def _compile_candidate_draft(self, *, immediate: bool = False) -> None:
        """Compile the editor candidate without mutating document or undo state."""

        if not self.has_pending_inspector_edits or not self._pending_matches_current_subject():
            return
        try:
            scenario = self._candidate_scenario_from_editors()
        except (RuntimeError, TypeError, ValueError) as exc:
            self._candidate_failure = CompilationFailure.from_exception(exc)
            self._update_pending_inspector_ui()
            self._refresh_viewport()
            return
        self._candidate_requested_scenario = scenario
        self._candidate_failure = None
        token = self._next_candidate_token()
        scheduler = self._candidate_compile_scheduler
        if scheduler is not None and not immediate:
            scheduler.request(
                scenario,
                self._current_scenario_directory(),
                token,
                subject_id=self._pending_subject_id,
            )
            self._update_pending_inspector_ui()
            return
        try:
            with self._compile_lock:
                if isinstance(self.compiler, ScenarioCompiler):
                    result = self.compiler.prepare_actor_preview(
                        scenario,
                        self._pending_subject_id,
                        scenario_directory=self._current_scenario_directory(),
                    )
                else:
                    result = self.compiler.compile(
                        scenario,
                        scenario_directory=self._current_scenario_directory(),
                    )
        except Exception as exc:  # noqa: BLE001 - compiler boundary
            self._candidate_compile_failed(token, CompilationFailure.from_exception(exc))
        else:
            self._candidate_compile_succeeded(token, result)

    def _candidate_compile_succeeded(self, token: object, result: object) -> None:
        if (
            token != self._candidate_request_token
            or not isinstance(result, CompilationResult)
            or not self.has_pending_inspector_edits
        ):
            return
        if result.valid:
            self._candidate_scenario = self._candidate_requested_scenario
            self._candidate_compilation = result
            self._candidate_failure = None
        else:
            self._candidate_failure = CompilationFailure(
                "ValidationError",
                "; ".join(i.message for i in result.issues if i.severity is IssueSeverity.ERROR),
            )
        self._update_pending_inspector_ui()
        self._sync_motion_edit_action()
        self._refresh_viewport()

    def _candidate_compile_failed(self, token: object, failure: object) -> None:
        if (
            token != self._candidate_request_token
            or not isinstance(failure, CompilationFailure)
            or not self.has_pending_inspector_edits
        ):
            return
        self._candidate_failure = failure
        self._update_pending_inspector_ui()
        self._refresh_viewport()

    def _begin_motion_draft(self, subject_id: UUID) -> AuthoringScenario | None:
        """Continue the inspector transaction without applying it on pointer-down."""
        document = self.document
        if document is None:
            return None
        if (
            self.has_pending_inspector_edits
            and subject_id != self._pending_subject_id
            and not self._resolve_pending_inspector_edits()
        ):
            return None
        if self.has_pending_inspector_edits:
            self._candidate_compile_timer.stop()
            try:
                scenario = self._candidate_scenario_from_editors()
            except (RuntimeError, TypeError, ValueError) as exc:
                self._candidate_failure = CompilationFailure.from_exception(exc)
                self._update_pending_inspector_ui()
                return None
            if scenario != self._candidate_scenario:
                self._compile_candidate_draft(immediate=True)
            if self._candidate_failure is not None:
                return None
        else:
            scenario = document.scenario
        self._motion_draft_baseline = _MotionDraftBaseline(
            scenario,
            self._pending_subject_id,
            self._pending_subject_is_group,
            self._mobility_draft_pending,
            self._orientation_draft_pending,
            self._candidate_scenario,
            self._candidate_compilation,
        )
        return scenario

    def _stage_motion_draft(self, subject: AuthoringActor | AuthoringGroup) -> None:
        """Publish direct manipulation to the same draft as typed inspector edits."""
        self._syncing = True
        try:
            self.mobility_editor.set_mobility(subject.mobility)
            if isinstance(subject, AuthoringActor):
                self.orientation_editor.set_orientation(subject.orientation)
        finally:
            self._syncing = False
        self._connect_editor_draft_signals(self.mobility_editor, "mobility")
        self._mark_inspector_draft("mobility")
        if isinstance(subject, AuthoringActor):
            committed = self.document.scenario.actor(subject.id)
            self._orientation_draft_pending = (
                self._orientation_draft_pending or subject.orientation != committed.orientation
            )
        self._candidate_compile_timer.stop()
        scenario = self._candidate_scenario_from_editors()
        previous = self._candidate_compilation or self.compilation
        samples = dict(previous.samples) if previous is not None else {}
        groups = dict(previous.group_samples) if previous is not None else {}
        samples.update(self._dependent_drag_preview_samples)
        if isinstance(subject, AuthoringGroup):
            if self._mobility_drag_preview_group_samples is not None:
                groups[subject.id] = self._mobility_drag_preview_group_samples
        else:
            preview = self._transform_preview_samples or self._mobility_drag_preview_samples
            if preview is not None and subject.id not in self._dependent_drag_preview_samples:
                samples[subject.id] = preview
            elif subject.id not in samples:
                try:
                    prepared, prepared_groups = self._prepare_transient_pose_samples(scenario)
                except (KeyError, RuntimeError, TypeError, ValueError):
                    # Incomplete actors (for example a target awaiting an asset)
                    # still expose their authored handles and pose.
                    pass
                else:
                    samples.update(prepared)
                    groups.update(prepared_groups)
        self._candidate_requested_scenario = scenario
        self._candidate_scenario = scenario
        self._candidate_compilation = CompilationResult(
            mapping=canonical_scenario_mapping(scenario),
            yaml_text="",
            issues=(),
            samples=samples,
            group_samples=groups,
            target_assets=previous.target_assets if previous is not None else {},
            fully_validated=False,
        )
        self._update_pending_inspector_ui()
        self._refresh_viewport()

    def _finish_motion_draft(self, *, accept: bool) -> None:
        """End a gesture; Apply remains the only document/undo commit boundary."""
        baseline = self._motion_draft_baseline
        self._motion_draft_baseline = None
        if baseline is None or accept:
            return
        self._invalidate_candidate_preview()
        scenario = baseline.scenario
        self._pending_subject_id = baseline.subject_id
        self._pending_subject_is_group = baseline.is_group
        self._mobility_draft_pending = baseline.mobility_pending
        self._orientation_draft_pending = baseline.orientation_pending
        self._candidate_scenario = baseline.preview_scenario
        self._candidate_compilation = baseline.preview
        self._candidate_requested_scenario = self._candidate_scenario
        selected = self._selected_subject()
        subject = (
            (scenario.group(selected[0]) if selected[1] else scenario.actor(selected[0]))
            if selected is not None
            else None
        )
        if subject is not None:
            self._pending_mobility_value = subject.mobility
            self._syncing = True
            try:
                self.mobility_editor.set_mobility(subject.mobility)
                if isinstance(subject, AuthoringActor):
                    self.orientation_editor.set_orientation(subject.orientation)
            finally:
                self._syncing = False
            self._connect_editor_draft_signals(self.mobility_editor, "mobility")
        self._update_pending_inspector_ui()

    def _draft_samples(self, actor: AuthoringActor) -> ActorSamples | None:
        if self._candidate_compilation is not None:
            samples = self._candidate_compilation.samples.get(actor.id)
            if samples is not None:
                return samples
        return self._prepared_samples(actor)

    def _invalidate_candidate_preview(self) -> None:
        self._candidate_compile_timer.stop()
        scheduler = self._candidate_compile_scheduler
        token = self._next_candidate_token()
        if scheduler is not None and not scheduler.closed:
            scheduler.invalidate(token)
        self._candidate_scenario = None
        self._candidate_compilation = None
        self._candidate_failure = None

    def _clear_pending_kind(self, kind: str, *, resync: bool) -> None:
        if kind == "mobility":
            self._mobility_draft_pending = False
        elif kind == "orientation":
            self._orientation_draft_pending = False
        elif kind == "target":
            self._target_draft_pending = False
        elif kind == "group_settings":
            self._group_settings_draft_pending = False
        else:
            raise ValueError(f"unknown inspector draft kind: {kind}")
        if not self.has_pending_inspector_edits:
            self._pending_subject_id = None
            self._pending_subject_is_group = False
            self._invalidate_candidate_preview()
        else:
            self._candidate_compile_timer.start()
        self._update_pending_inspector_ui()
        if resync:
            self._sync_document_controls()
            self._refresh_viewport()

    def _reset_pending_mobility(self) -> None:
        if not self._mobility_draft_pending or self.document is None:
            return
        subject = (
            self.document.scenario.group(self._pending_subject_id)
            if self._pending_subject_is_group
            else self.document.scenario.actor(self._pending_subject_id)
        )
        if subject is not None:
            self._syncing = True
            try:
                self.mobility_editor.set_mobility(subject.mobility)
            finally:
                self._syncing = False
        self._clear_pending_kind("mobility", resync=True)

    def _reset_pending_orientation(self) -> None:
        if not self._orientation_draft_pending or self.document is None:
            return
        actor = self.document.scenario.actor(self._pending_subject_id)
        if actor is not None:
            self._syncing = True
            try:
                self.orientation_editor.set_orientation(actor.orientation)
            finally:
                self._syncing = False
        self._clear_pending_kind("orientation", resync=True)

    def _reset_pending_target(self) -> None:
        if not self._target_draft_pending:
            return
        self._clear_pending_kind("target", resync=True)

    def _reset_pending_group_settings(self) -> None:
        if not self._group_settings_draft_pending:
            return
        self._clear_pending_kind("group_settings", resync=True)

    def _reset_all_pending_inspector_edits(self) -> None:
        if self._mobility_draft_pending:
            self._reset_pending_mobility()
        if self._orientation_draft_pending:
            self._reset_pending_orientation()
        if self._target_draft_pending:
            self._reset_pending_target()
        if self._group_settings_draft_pending:
            self._reset_pending_group_settings()

    def _apply_all_pending_inspector_edits(self) -> bool:
        """Commit every pending section for one subject as one undo command."""

        document = self.document
        subject_id = self._pending_subject_id
        if document is None or subject_id is None or document.read_only:
            return False
        if self._candidate_failure is not None:
            self._update_pending_inspector_ui()
            return False
        try:
            candidate = self._candidate_scenario_from_editors()
            # Applying before the debounce completes still crosses the canonical
            # preparation boundary. Only the edited reference component is built.
            result = self._candidate_compilation
            if isinstance(self.compiler, ScenarioCompiler):
                result = self.compiler.prepare_actor_preview(
                    candidate,
                    subject_id,
                    scenario_directory=self._current_scenario_directory(),
                )
                if not result.valid:
                    self._candidate_failure = CompilationFailure(
                        "ValidationError",
                        "; ".join(
                            issue.message
                            for issue in result.issues
                            if issue.severity is IssueSeverity.ERROR
                        ),
                    )
                    self._update_pending_inspector_ui()
                    return False
            if self._pending_subject_is_group:
                replacement = candidate.group(subject_id)
                if replacement is None:
                    raise ValueError("pending group no longer exists")
                document.replace_group_with_resources(
                    replacement,
                    candidate.resources,
                    text=f"Apply pending inspector edits to {replacement.name}",
                )
            else:
                replacement = candidate.actor(subject_id)
                if replacement is None:
                    raise ValueError("pending actor no longer exists")
                document.replace_actor_with_resources(
                    replacement,
                    candidate.resources,
                    text=f"Apply pending inspector edits to {replacement.name}",
                )
        except (RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid pending edits", str(exc))
            return False
        self._mobility_draft_pending = False
        self._orientation_draft_pending = False
        self._target_draft_pending = False
        self._group_settings_draft_pending = False
        self._pending_subject_id = None
        self._pending_subject_is_group = False
        self._invalidate_candidate_preview()
        if result is not None:
            for actor_id, samples in result.samples.items():
                actor = document.scenario.actor(actor_id)
                if actor is not None:
                    self._prepared_sample_cache[actor_id] = (
                        self._orientation_cache_key(actor),
                        samples,
                    )
            for group_id, samples in result.group_samples.items():
                group = document.scenario.group(group_id)
                if group is not None:
                    key = self._group_position_cache_key(group, document.scenario)
                    self._prepared_group_position_cache[group_id] = (key, samples.positions)
                    self._prepared_group_sample_cache[group_id] = (key, samples)
                    self._prepared_group_physical_cache[group_id] = (
                        key,
                        samples.has_physical_velocity,
                    )
        self._update_pending_inspector_ui()
        self._sync_document_controls()
        self._refresh_viewport()
        return True

    def _resolve_pending_inspector_edits(self) -> bool:
        """Prompt for pending fields before an operation can leave their owner."""

        if not self.has_pending_inspector_edits:
            return True
        choice = QMessageBox.question(
            self,
            "Pending Inspector Edits",
            "Apply the pending inspector values before continuing?",
            QMessageBox.Apply | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Apply,
        )
        if choice == QMessageBox.Apply:
            return self._apply_all_pending_inspector_edits()
        if choice == QMessageBox.Discard:
            self._reset_all_pending_inspector_edits()
            return True
        return False

    # -- document binding and compilation ------------------------------

    def bind_document(self, document: ScenarioDocument) -> None:
        """Replace the active document after resolving pending editor state."""

        if self.document is not None and self.document is not document:
            if not self._resolve_pending_inspector_edits():
                return
            if not self._resolve_pending_timeline_edits():
                return
        if hasattr(self, "osm_panel") and self.document is not document:
            self.osm_panel.reset()
            self._osm_build_context = None
        self._finish_target_transform(commit=False)
        self._finish_mobility_drag(commit=False)
        self._cancel_waypoint_session()
        self._reset_all_pending_inspector_edits()
        self._timeline_draft_pending = False
        if self._unsubscribe_document is not None:
            self._unsubscribe_document()
        self.document = document
        self._stop_playback(refresh=False)
        self._play_step = 0
        self.timeline_slider.setValue(0)
        self._osm_fit_pending = True
        self.sensing_inspector.set_mapping(document.scenario.source_snapshot.to_mapping())
        if self.compilation is not None and self.compilation.recording is not None:
            self.compilation.recording.close()
        self.compilation = None
        self._early_scene = None
        self._compiled_scenario = None
        self._compiled_directory = None
        self._compilation_failure = None
        self._prepared_sample_cache.clear()
        self._prepared_group_position_cache.clear()
        self._prepared_group_physical_cache.clear()
        self._prepared_group_sample_cache.clear()
        self._read_only = bool(document.read_only)
        self._unsubscribe_document = document.subscribe(self._document_changed)
        self._document_changed()

    def show_read_only_import(self, result: Any) -> None:
        """Present an unsupported import without inventing a mutable document."""

        if not self._resolve_pending_inspector_edits():
            return
        if not self._resolve_pending_timeline_edits():
            return
        self.osm_panel.reset()
        self._osm_build_context = None
        previous_document = self.document
        if self._compile_scheduler is not None:
            self._compile_scheduler.invalidate(self._next_compile_token())
        self._finish_target_transform(commit=False)
        self._finish_mobility_drag(commit=False)
        self._cancel_waypoint_session()
        self._reset_all_pending_inspector_edits()
        self._timeline_draft_pending = False
        self._prepared_sample_cache.clear()
        self._prepared_group_position_cache.clear()
        self._prepared_group_physical_cache.clear()
        self._prepared_group_sample_cache.clear()
        if self.compilation is not None and self.compilation.recording is not None:
            self.compilation.recording.close()
        self.compilation = None
        self._early_scene = None
        self._compiled_scenario = None
        self._compiled_directory = None
        self._compilation_failure = None
        self._read_only = True
        self.read_only_label.setVisible(True)
        self.read_only_label.setText("READ ONLY")
        self.recording_banner.hide()
        self.validation_status.setText("Validation unavailable: no editable draft is open.")
        self.document = None
        self._update_preview_status()
        self.preserved_settings_label.clear()
        self.preserved_settings_label.setVisible(False)
        self._update_timeline_pending_ui()
        if self._unsubscribe_document is not None:
            self._unsubscribe_document()
            self._unsubscribe_document = None
        self.actor_tree.clear()
        self._clear_actor_inspector()
        self.inline_errors.clear()
        self.problems_tree.clear()
        for issue in getattr(result, "issues", ()):
            self.problems_tree.addTopLevelItem(
                QTreeWidgetItem((str(issue.severity.value), str(issue.path), str(issue.message)))
            )
        raw_text = getattr(result, "raw_text", None)
        if raw_text is None:
            try:
                import yaml

                raw_text = yaml.safe_dump(getattr(result, "raw_mapping", {}), sort_keys=False)
            except Exception:
                raw_text = str(getattr(result, "raw_mapping", {}))
        self.yaml_preview.setPlainText(raw_text)
        self.sensing_inspector.set_mapping(getattr(result, "raw_mapping", {}))
        self.document_label.setText(Path(getattr(result, "source_path", "scenario.yaml")).name)
        self._set_editing_enabled(False)
        port = getattr(self.viewport, "port", None)
        if previous_document is not None and port is not None:
            port.reconcile(
                OverlaySnapshot(
                    document_id=previous_document.scenario.document_id,
                    revision=previous_document.revision,
                    actors=(),
                    scene_assets=(),
                    work_plane_visible=False,
                )
            )

    def _document_changed(self, event: DocumentEvent | None = None) -> None:
        document = self.document
        if document is None:
            return
        needs_compilation = event is None or event.kind is not DocumentEventKind.SELECTION
        refresh_candidate = needs_compilation and self.has_pending_inspector_edits
        if needs_compilation:
            self._compilation_failure = None
            self._show_compilation_pending()
        if refresh_candidate:
            self._invalidate_candidate_preview()
        path = document.path
        name = path.parent.name if path is not None else "Untitled Scenario"
        suffix = " *" if document.dirty else ""
        self.document_label.setText(name + suffix)
        self.title_changed.emit(name + suffix)
        self.dirty_changed.emit(document.dirty)
        self.read_only_label.setVisible(document.read_only)
        self.read_only_label.setText(
            "RECORDING" if document.scenario.recorded_poses else "READ ONLY"
        )
        self.recording_banner.setVisible(document.scenario.recorded_poses)
        self._set_editing_enabled(not self.document.read_only)
        self._sync_document_controls()
        self._update_waypoint_session_ui()
        if event is None:
            self._refresh_tree()
        else:
            self._schedule_tree_refresh()
        self._refresh_viewport()
        self._refresh_inline_errors()

        if refresh_candidate and self._pending_matches_current_subject():
            self._candidate_compile_timer.start()
            self._update_pending_inspector_ui()
        if not needs_compilation:
            if self.camera_controls is not None and document.selected_subject is not None:
                subject_id = document.selected_subject.id
                # The viewport's pointer event must finish before starting a
                # transition; otherwise its input handler cancels that new move.
                QTimer.singleShot(0, lambda: self._sync_selection_camera(document, subject_id))
            return
        if self._compile_scheduler is not None:
            self._schedule_background_compile()
        elif not self._compile_scheduled:
            self._compile_scheduled = True
            QTimer.singleShot(0, self._run_scheduled_compile)

    def _run_scheduled_compile(self) -> None:
        if self._compile_scheduled:
            self.refresh_now()

    def _schedule_tree_refresh(self) -> None:
        """Defer rebuilding Qt items until the originating item signal unwinds."""

        if self._tree_refresh_scheduled:
            return
        self._tree_refresh_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_tree_refresh)

    def _run_scheduled_tree_refresh(self) -> None:
        self._tree_refresh_scheduled = False
        self._refresh_tree()

    def refresh_now(self) -> None:
        """Synchronously compile the current immutable draft when explicitly requested."""

        self._compile_scheduled = False
        document = self.document
        if document is None:
            return
        token = self._next_compile_token()
        if self._compile_scheduler is not None:
            self._compile_scheduler.invalidate(token)
        directory = self._current_scenario_directory()
        with self._compile_lock:
            result = self.compiler.compile(document.scenario, scenario_directory=directory)
        self._apply_compilation(result)

    def validate_current_draft(self) -> None:
        """Run explicit validation with visible progress and outcome feedback."""

        document = self.document
        if document is None:
            self.validation_status.setText("Validation unavailable: no editable draft is open.")
            return
        generation = getattr(self.visualizer, "authoring_generation_controller", None)
        if self._explicit_validation_running or bool(getattr(generation, "running", False)):
            return

        self._explicit_validation_running = True
        self.validation_status.setText("Validating current draft...")
        self._update_preview_status(pending=True)
        self.validate_button.setEnabled(False)
        # Explicit validation is synchronous. Repaint these two small controls
        # before compilation so a slow generator-backed check has visible state.
        self.validation_status.repaint()
        self.preview_status_label.repaint()
        self.validate_button.repaint()
        try:
            self.refresh_now()
        except Exception as exc:  # compiler boundary: present failure in the workspace
            failure = CompilationFailure.from_exception(exc)
            self._background_compile_failed(self._compile_request_token, failure)
            self.validation_status.setText(
                f"Validation failed: {failure.exception_type}: {failure.message}"
            )
        else:
            if self.compilation is not None:
                self.validation_status.setText(self._explicit_validation_summary(self.compilation))
        finally:
            self._explicit_validation_running = False
            self._set_editing_enabled(not document.read_only)

    @staticmethod
    def _explicit_validation_summary(result: CompilationResult) -> str:
        """Return a compact pass/fail summary for one explicit validation."""

        if not result.fully_validated:
            if result.recording is not None:
                return "Saved recording: inspect its results or create an editable variant from the acquisition inputs."
            return (
                "Preview ready. Full validation runs before saving or generation."
                if result.valid
                else "Preview has problems; see the Problems drawer."
            )
        errors = sum(issue.severity is IssueSeverity.ERROR for issue in result.issues)
        warnings = sum(issue.severity is IssueSeverity.WARNING for issue in result.issues)
        generation_problems = len(result.generation_issues)

        def count_text(count: int, label: str) -> str:
            return f"{count} {label}{'' if count == 1 else 's'}"

        if errors:
            details = [count_text(errors, "error")]
            if warnings:
                details.append(count_text(warnings, "warning"))
            if generation_problems:
                details.append(count_text(generation_problems, "generation problem"))
            return f"Validation failed: {', '.join(details)}."
        if generation_problems:
            warning_text = f" with {count_text(warnings, 'warning')}" if warnings else ""
            return (
                f"Validation passed{warning_text}, but Generate has "
                f"{count_text(generation_problems, 'problem')}."
            )
        if warnings:
            return f"Validation passed with {count_text(warnings, 'warning')}."
        return "Validation passed: no problems found."

    def _next_compile_token(self) -> int:
        self._compile_request_token += 1
        return self._compile_request_token

    def _schedule_background_compile(self) -> None:
        """Queue only the newest immutable draft for generator-backed preparation."""

        document = self.document
        scheduler = self._compile_scheduler
        if document is None or scheduler is None:
            return
        directory = (
            document.path.parent if document.path is not None else Path(self.compiler.project_root)
        )
        # Texture policy must be in place before the worker resolves materials.
        self._apply_scenario_presentation()
        scheduler.request(document.scenario, directory, self._next_compile_token())

    def _background_compile_succeeded(
        self,
        token: object,
        result: object,
    ) -> None:
        """Publish a latest-only worker result on the workspace's Qt thread."""

        if (
            token != self._compile_request_token
            or not isinstance(result, CompilationResult)
            or self.document is None
        ):
            return
        self._apply_compilation(result)

    def _background_scene_ready(self, token: object, preview: object) -> None:
        """Show cached environment geometry while actor validation continues."""
        if (
            token != self._compile_request_token
            or not isinstance(preview, ScenePreviewResult)
            or self.document is None
        ):
            return
        self._early_scene = preview
        self._refresh_viewport()
        self._apply_scenario_presentation()
        self._update_preview_status(pending=True)

    def _background_compile_progress(self, token: object, message: object) -> None:
        if token == self._compile_request_token and self.document is not None:
            self._compile_progress = str(message)
            self._update_preview_status(pending=True)

    def _background_compile_failed(
        self,
        token: object,
        failure: object,
    ) -> None:
        """Surface an unexpected compiler failure without discarding the draft."""

        if (
            token != self._compile_request_token
            or not isinstance(failure, CompilationFailure)
            or self.document is None
        ):
            return
        self.compilation = None
        self._compiled_scenario = None
        self._compiled_directory = None
        self._compilation_failure = failure
        self._prepared_sample_cache.clear()
        self._prepared_group_position_cache.clear()
        self._prepared_group_physical_cache.clear()
        self._prepared_group_sample_cache.clear()
        self.yaml_preview.setPlainText(
            f"# Compilation failed\n# {failure.exception_type}: {failure.message}\n"
        )
        self.problems_tree.clear()
        self.problems_tree.addTopLevelItem(
            QTreeWidgetItem(
                (
                    "Error",
                    "compiler",
                    f"{failure.exception_type}: {failure.message}",
                )
            )
        )
        self.problems_tree.resizeColumnToContents(0)
        self.problems_tree.resizeColumnToContents(1)
        self._update_preview_status()
        self._refresh_tree()
        self._refresh_viewport()
        self._refresh_inline_errors()
        self._set_editing_enabled(not self.document.read_only)

    def _apply_scenario_presentation(self) -> None:
        document = self.document
        if (
            document is None
            or getattr(self, "_presented_document", None) == document.scenario.document_id
        ):
            return
        apply = getattr(getattr(self.viewport, "port", None), "apply_scenario_defaults", None)
        if not callable(apply):
            return
        defaults = document.scenario.source_snapshot.to_mapping().get("view_defaults") or {}
        if defaults.get("camera_pose"):
            self._osm_fit_pending = False
        try:
            apply(defaults, self._current_scenario_directory())
        except (OSError, ValueError, RuntimeError) as exc:
            self.append_generation_log(f"Scenario presentation: {exc}")
        self._presented_document = document.scenario.document_id
        if defaults.get("appearance_recipe"):
            with QSignalBlocker(self.appearance_preset):
                if self.appearance_preset.findText("Scenario appearance") < 0:
                    self.appearance_preset.addItem("Scenario appearance")
                self.appearance_preset.setCurrentText("Scenario appearance")

    def _apply_compilation(self, result: CompilationResult) -> None:
        """Apply canonical compiler evidence to the current document and viewport."""

        document = self.document
        if document is None:
            return
        previous = self.compilation
        self.compilation = result
        if (
            previous is not None
            and previous.recording is not None
            and previous.recording is not result.recording
        ):
            previous.recording.close()
        self._early_scene = None
        self._compiled_scenario = document.scenario
        self._compiled_directory = self._current_scenario_directory()
        self._compilation_failure = None
        for actor_id, samples in result.samples.items():
            actor = document.scenario.actor(actor_id)
            if actor is not None:
                self._prepared_sample_cache[actor_id] = (
                    self._orientation_cache_key(actor),
                    samples,
                )
        for group_id, samples in result.group_samples.items():
            group = document.scenario.group(group_id)
            if group is not None:
                self._prepared_group_position_cache[group_id] = (
                    self._group_position_cache_key(group, document.scenario),
                    tuple(samples.positions),
                )
                self._prepared_group_physical_cache[group_id] = (
                    self._group_position_cache_key(group, document.scenario),
                    samples.has_physical_velocity,
                )
                self._prepared_group_sample_cache[group_id] = (
                    self._group_position_cache_key(group, document.scenario),
                    samples,
                )
        self.yaml_preview.setPlainText(result.yaml_text)
        self.validation_status.setText(self._explicit_validation_summary(result))
        self._refresh_problems()
        self._refresh_tree()
        self._refresh_viewport()
        self._apply_scenario_presentation()
        self._refresh_inline_errors()
        self._set_editing_enabled(not document.read_only)

        if result.recording is not None:
            result.recording.start_trajectories(
                lambda: self._recorded_trajectories_changed(result.recording)
            )

    def _recorded_trajectories_changed(self, recording) -> None:
        """Ignore retired recordings and use one path snapshot in both views."""
        if self.compilation is None or self.compilation.recording is not recording:
            return
        self._refresh_viewport()
        self._update_preview_status()

    def _refresh_problems(self) -> None:
        self.problems_tree.clear()
        if self.compilation is None:
            return
        for issue in self.compilation.issues:
            self.problems_tree.addTopLevelItem(
                QTreeWidgetItem((issue.severity.value.title(), issue.path, issue.message))
            )
        for issue in getattr(self.compilation, "generation_issues", ()):
            self.problems_tree.addTopLevelItem(
                QTreeWidgetItem(("Generate", issue.path, issue.message))
            )
        self.problems_tree.resizeColumnToContents(0)
        self.problems_tree.resizeColumnToContents(1)
        self._update_preview_status()

    def _update_preview_status(self, *, pending: bool = False) -> None:
        """Summarize current compiler evidence without promising a rendered scene."""

        document = self.document
        if document is None:
            state = "read_only" if self._read_only else "unavailable"
            text = (
                "Read only · Preview validation unavailable" if self._read_only else "No draft open"
            )
        elif pending or (self._compilation_failure is None and not self._compilation_is_current()):
            state, text = "pending", f"Validating preview… · {self._compile_progress}"
            elapsed = time.monotonic() - self._compile_started_at
            if self._early_scene is not None and self._early_scene.scene_assets:
                text = f"Environment ready · {self._compile_progress}"
            if elapsed >= 1:
                text += f" · {elapsed:.0f} s"
            if self.compilation is not None:
                text += " · Previous geometry may still be shown"
        elif self._compilation_failure is not None:
            state, text = "error", "Preview validation failed · See Problems"
        else:
            result = self.compilation
            assert result is not None
            errors = sum(issue.severity is IssueSeverity.ERROR for issue in result.issues)
            warnings = sum(issue.severity is IssueSeverity.WARNING for issue in result.issues)
            if errors:
                state, text = "error", f"Preview has {errors} error{'s' if errors != 1 else ''}"
            elif document.scenario.recorded_poses:
                state, text = "read_only", "Recording ready"
                recording = result.recording
                loader = recording.trajectory_loader if recording is not None else None
                if recording is not None and recording.trajectory_error:
                    text += f" · Trajectory loading failed: {recording.trajectory_error}"
                elif loader is not None and loader.is_loading:
                    loaded, total = loader.progress
                    text += f" · Loading trajectories {loaded:,}/{total:,} frames"
            elif result.generation_issues:
                state, text = "generation_limited", "Preview ready · Generation limited"
            elif warnings:
                state, text = (
                    "warning",
                    f"Preview has {warnings} warning{'s' if warnings != 1 else ''}",
                )
            else:
                state, text = "valid", "Draft valid" if result.fully_validated else "Preview ready"
            prepared = len(result.samples)
            text += f" · {prepared}/{len(document.actors)} actors prepared"
            text += (
                f" · {len(result.scene_assets)} scene meshes"
                if result.scene_assets
                else " · No scene meshes loaded"
            )
        if document is not None and document.read_only:
            state, text = "read_only", f"Read only · {text}"
        if getattr(self.viewport, "initialization_error", None) is not None:
            state, text = "error", f"Viewport unavailable · {text}"
        self.preview_status_label.setProperty("validationState", state)
        self.preview_status_label.setText(text)
        self.preview_status_label.setToolTip(text)
        if state != "pending":
            self._compile_status_timer.stop()

    def _show_compilation_pending(self) -> None:
        """Replace superseded compiler evidence with an explicit pending state."""

        self._compile_started_at = time.monotonic()
        self._compile_progress = "Checking scenario settings and resources"
        self._compile_status_timer.start()
        if not self._explicit_validation_running:
            self.validation_status.setText("Validation pending for the current draft.")
        self.yaml_preview.setPlainText("# Validating the current draft...\n")
        self.problems_tree.clear()
        self.problems_tree.addTopLevelItem(QTreeWidgetItem(("Info", "", "Validation in progress")))
        self._update_preview_status(pending=True)

    def _refresh_tree(self) -> None:
        document = self.document
        if document is None:
            return
        selected_actor = document.selected_actor_id
        selected_group = document.selected_group_id
        self._syncing = True
        try:
            self.actor_tree.clear()
            for actor in document.actors:
                state = self._actor_status(actor.id)
                status = {
                    ActorVisualState.COMPLETE: "Complete",
                    ActorVisualState.PENDING: "Validating",
                    ActorVisualState.INCOMPLETE: "Needs endpoint",
                    ActorVisualState.INVALID: "Invalid",
                }[state]
                item = QTreeWidgetItem((actor.name, actor.role.value.upper(), "", "", status))
                if not document.read_only:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setData(0, Qt.UserRole, str(actor.id))
                item.setData(0, Qt.UserRole + 1, "actor")
                item.setCheckState(2, Qt.Checked if actor.visible else Qt.Unchecked)
                item.setCheckState(3, Qt.Checked if actor.locked else Qt.Unchecked)
                if not actor.visible:
                    item.setForeground(0, Qt.gray)
                self.actor_tree.addTopLevelItem(item)
                if actor.id == selected_actor:
                    item.setSelected(True)
            for group in document.groups:
                issues = (
                    self.compilation.issues_for_group(group.id)
                    if self._compilation_is_current() and self.compilation is not None
                    else ()
                )
                status = (
                    "Invalid"
                    if any(issue.severity is IssueSeverity.ERROR for issue in issues)
                    else ("Complete" if self._compilation_is_current() else "Validating")
                )
                item = QTreeWidgetItem((group.name, "GROUP", "", "", status))
                if not document.read_only:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setData(0, Qt.UserRole, str(group.id))
                item.setData(0, Qt.UserRole + 1, "group")
                item.setCheckState(2, Qt.Checked if group.visible else Qt.Unchecked)
                item.setCheckState(3, Qt.Checked if group.locked else Qt.Unchecked)
                if not group.visible:
                    item.setForeground(0, Qt.gray)
                self.actor_tree.addTopLevelItem(item)
                if group.id == selected_group:
                    item.setSelected(True)
        finally:
            self._syncing = False

    def _actor_status(self, actor_id: UUID) -> ActorVisualState:
        document = self.document
        if document is None:
            return ActorVisualState.INVALID
        if self._compilation_failure is not None:
            return ActorVisualState.INVALID
        actor = document.scenario.actor(actor_id)
        if actor is None:
            return ActorVisualState.INVALID
        if not self._compilation_is_current():
            return ActorVisualState.PENDING
        issues = self.compilation.issues_for_actor(actor_id) if self.compilation else ()
        if any(issue.severity is IssueSeverity.ERROR for issue in issues):
            return ActorVisualState.INVALID
        return ActorVisualState.COMPLETE

    def _compilation_is_current(self) -> bool:
        """Return whether visible compiler evidence belongs to the current draft."""

        return (
            self.document is not None
            and self.compilation is not None
            and self._compiled_scenario == self.document.scenario
            and self._compiled_directory == self._current_scenario_directory()
            and self._compilation_failure is None
        )

    def _current_scenario_directory(self) -> Path:
        """Return the directory against which the current draft resolves assets."""

        document = self.document
        if document is not None and document.path is not None:
            return document.path.parent.resolve()
        return Path(getattr(self.compiler, "project_root", Path.cwd())).resolve()

    def _orientation_cache_key(self, actor: AuthoringActor) -> tuple[Any, ...]:
        """Describe every authored value that can change prepared orientation."""

        document = self.document
        if document is None:
            return ()
        return (
            *self._sample_cache_key(actor, document.scenario),
            self._current_scenario_directory(),
        )

    @staticmethod
    def _sample_cache_key(
        actor: AuthoringActor,
        scenario: AuthoringScenario,
    ) -> tuple[Any, ...]:
        """Describe authored values that determine prepared samples."""

        timeline = scenario.timeline
        dependencies: list[Any] = []
        if isinstance(actor.mobility, GroupMemberMobilitySpec):
            try:
                group = scenario.group(actor.mobility.group)
            except ValueError:
                group = None
            if group is not None:
                dependencies.append((group.mobility, group.deviation))
        if isinstance(actor.orientation, LookAtOrientationSpec) and actor.orientation.actor:
            try:
                target_id = look_at_actor_id(actor.orientation)
            except ValueError:
                target_id = None
            target = scenario.actor(target_id) if target_id is not None else None
            if target is not None:
                dependencies.append((target.name, target.mobility))
                if isinstance(target.mobility, GroupMemberMobilitySpec):
                    try:
                        target_group = scenario.group(target.mobility.group)
                    except ValueError:
                        target_group = None
                    if target_group is not None:
                        dependencies.append((target_group.mobility, target_group.deviation))
        return (
            timeline.steps,
            timeline.duration_s,
            actor.name,
            actor.mobility,
            actor.orientation,
            actor.target,
            actor.representation,
            actor.geometry,
            actor.constant_rcs,
            actor.tr38901,
            scenario.resources,
            tuple(dependencies),
        )

    def _prepared_samples(self, actor: AuthoringActor) -> ActorSamples | None:
        """Return current samples or a still-semantic last valid preparation."""

        if self.compilation is not None and actor.id in getattr(
            self.compilation, "pose_sources", {}
        ):
            return self.compilation.pose_sources[actor.id].sample(self._play_step)
        if (
            self.compilation is not None
            and getattr(self.compilation, "recording", None) is not None
        ):
            try:
                return self.compilation.recording.sample(actor, self._play_step)
            except (OSError, ValueError, KeyError):
                return None

        expected_key = self._orientation_cache_key(actor)
        current = self.compilation.samples.get(actor.id) if self.compilation else None
        compiled_scenario = self._compiled_scenario
        compiled_actor = (
            compiled_scenario.actor(actor.id) if compiled_scenario is not None else None
        )
        if (
            current is not None
            and compiled_scenario is not None
            and compiled_actor is not None
            and (
                *self._sample_cache_key(compiled_actor, compiled_scenario),
                self._compiled_directory,
            )
            == expected_key
        ):
            return current
        cached = self._prepared_sample_cache.get(actor.id)
        if cached is None or cached[0] != expected_key:
            return None
        return cached[1]

    def _group_position_cache_key(
        self,
        group: AuthoringGroup,
        scenario: AuthoringScenario,
        *,
        scenario_directory: Path | None = None,
    ) -> tuple[Any, ...]:
        """Describe authored values that determine one prepared group path."""

        return (
            scenario.timeline.steps,
            scenario.timeline.duration_s,
            group.mobility,
            scenario.resources,
            scenario_directory or self._current_scenario_directory(),
        )

    def _prepared_group_positions(
        self,
        group: AuthoringGroup,
    ) -> tuple[tuple[float, float, float], ...] | None:
        """Return current, active-preview, or exact cached group positions."""

        document = self.document
        if document is None:
            return None
        expected_key = self._group_position_cache_key(group, document.scenario)
        current = (
            self.compilation.group_samples.get(group.id) if self.compilation is not None else None
        )
        compiled_scenario = self._compiled_scenario
        compiled_group = (
            compiled_scenario.group(group.id) if compiled_scenario is not None else None
        )
        if (
            current is not None
            and compiled_scenario is not None
            and compiled_group is not None
            and self._group_position_cache_key(
                compiled_group,
                compiled_scenario,
                scenario_directory=self._compiled_directory,
            )
            == expected_key
        ):
            return tuple(current.positions)
        cached = self._prepared_group_position_cache.get(group.id)
        if cached is None or cached[0] != expected_key:
            return None
        return cached[1]

    def _prepared_group_samples(self, group: AuthoringGroup) -> Any | None:
        """Return canonical group samples when they match the current draft."""

        document = self.document
        if document is None:
            return None
        expected_key = self._group_position_cache_key(group, document.scenario)
        samples = (
            self.compilation.group_samples.get(group.id) if self.compilation is not None else None
        )
        compiled_scenario = self._compiled_scenario
        compiled_group = (
            compiled_scenario.group(group.id) if compiled_scenario is not None else None
        )
        if (
            samples is not None
            and compiled_scenario is not None
            and compiled_group is not None
            and self._group_position_cache_key(
                compiled_group,
                compiled_scenario,
                scenario_directory=self._compiled_directory,
            )
            == expected_key
        ):
            return samples
        cached = self._prepared_group_sample_cache.get(group.id)
        if cached is None:
            return None
        return cached[1] if cached[0] == expected_key else None

    def _prepared_group_has_physical_velocity(
        self,
        group: AuthoringGroup,
    ) -> bool | None:
        """Return cached canonical path semantics for one group."""

        samples = self._prepared_group_samples(group)
        if samples is not None:
            return samples.has_physical_velocity
        document = self.document
        if document is None:
            return None
        expected_key = self._group_position_cache_key(group, document.scenario)
        cached = self._prepared_group_physical_cache.get(group.id)
        if cached is None or cached[0] != expected_key:
            return None
        return cached[1]

    @staticmethod
    def _translated_positions(
        positions: tuple[tuple[float, float, float], ...],
        translation: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], ...]:
        """Apply one rigid world-space translation to prepared positions."""

        return tuple(
            tuple(
                float(value + offset) for value, offset in zip(position, translation, strict=True)
            )
            for position in positions
        )

    @staticmethod
    def _rigid_preview_samples(
        samples: ActorSamples,
        *,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fixed_orientation: tuple[float, float, float] | None = None,
    ) -> ActorSamples:
        """Rigidly transform prepared samples without reimplementing mobility."""

        positions = ScenarioAuthoringWorkspace._translated_positions(
            samples.positions,
            translation,
        )
        orientations = samples.orientations
        if fixed_orientation is not None and orientations:
            orientation = tuple(float(value) for value in fixed_orientation)
            orientations = tuple(orientation for _ in orientations)
        return ActorSamples(
            positions=positions,
            orientations=orientations,
            velocities_mps=samples.velocities_mps,
            forward_vectors=samples.forward_vectors,
            has_physical_velocity=samples.has_physical_velocity,
        )

    @staticmethod
    def _prepared_mobility_from_samples(samples: ActorSamples) -> PreparedMobility:
        """Reconstruct the canonical mobility input retained by UI samples."""

        count = len(samples.positions)
        velocities = samples.velocities_mps or ((0.0, 0.0, 0.0),) * count
        forwards = samples.forward_vectors or ((1.0, 0.0, 0.0),) * count
        return PreparedMobility(
            positions_m=samples.positions,
            velocities_mps=velocities,
            forward_vectors=forwards,
            has_physical_velocity=samples.has_physical_velocity,
        )

    def _look_at_drag_previews(
        self,
        scenario: AuthoringScenario,
        moved_actor_id: UUID,
        moved_samples: ActorSamples | None,
    ) -> dict[UUID, ActorSamples]:
        """Re-evaluate LookAt owners affected by one transient actor move."""

        if moved_samples is None:
            return {}
        return self._look_at_drag_previews_for_samples(
            scenario,
            {moved_actor_id: moved_samples},
        )

    def _look_at_drag_previews_for_samples(
        self,
        scenario: AuthoringScenario,
        moved_samples: dict[UUID, ActorSamples],
    ) -> dict[UUID, ActorSamples]:
        """Re-evaluate LookAt for one or more canonically sampled moves."""

        if not moved_samples:
            return {}
        sample_by_id: dict[UUID, ActorSamples] = {}
        for actor in scenario.actors:
            override = moved_samples.get(actor.id)
            if override is not None:
                sample_by_id[actor.id] = override
                continue
            samples = self._draft_samples(actor)
            if samples is not None:
                sample_by_id[actor.id] = samples
        references = {
            actor.name: self._prepared_mobility_from_samples(sample_by_id[actor.id])
            for actor in scenario.actors
            if actor.id in sample_by_id
        }
        timeline = Timeline(
            scenario.timeline.steps,
            scenario.timeline.duration_s,
        )
        moved_actor_ids = frozenset(moved_samples)
        result: dict[UUID, ActorSamples] = dict(moved_samples)
        for actor in scenario.actors:
            if not isinstance(actor.orientation, LookAtOrientationSpec):
                continue
            try:
                target_id = (
                    look_at_actor_id(actor.orientation)
                    if actor.orientation.actor is not None
                    else None
                )
            except ValueError:
                continue
            if actor.id not in moved_actor_ids and target_id not in moved_actor_ids:
                continue
            samples = sample_by_id.get(actor.id)
            if samples is None:
                continue
            try:
                prepared = prepare_orientation(
                    orientation_to_mapping(actor.orientation, scenario.actors),
                    timeline,
                    self._prepared_mobility_from_samples(samples),
                    references=references,
                    path=f"actors.{actor.name}.orientation",
                )
                if actor.role is ActorRole.TARGET:
                    prepared = apply_asset_alignment(
                        prepared,
                        (self._target_front_yaw_offset(actor.name), 0.0, 0.0),
                    )
            except (KeyError, RuntimeError, TypeError, ValueError):
                continue
            result[actor.id] = ActorSamples(
                positions=samples.positions,
                orientations=tuple(
                    tuple(float(value) for value in angles) for angles in prepared.euler_deg
                ),
                velocities_mps=samples.velocities_mps,
                forward_vectors=samples.forward_vectors,
                has_physical_velocity=samples.has_physical_velocity,
            )
        return result

    def _prepare_transient_pose_samples(
        self,
        scenario: AuthoringScenario,
    ) -> tuple[dict[UUID, ActorSamples], dict[UUID, GroupSamples]]:
        """Prepare a pure pose snapshot for direct-manipulation feedback."""

        if isinstance(self.compiler, ScenarioCompiler):
            subject_id = self._mobility_drag_actor_id or self._transform_actor_id
            if subject_id is not None:
                with self._compile_lock:
                    result = self.compiler.prepare_actor_preview(
                        scenario,
                        subject_id,
                        scenario_directory=self._current_scenario_directory(),
                    )
                if not result.valid:
                    raise ValueError("; ".join(issue.message for issue in result.issues))
                return dict(result.samples), dict(result.group_samples)

        alignments = {
            actor.name: (self._target_front_yaw_offset(actor.name), 0.0, 0.0)
            for actor in scenario.actors
            if actor.role is ActorRole.TARGET and actor.target is not None
        }
        prepared = prepare_scenario(
            validate_scenario_data(canonical_scenario_mapping(scenario)),
            base_dir=self._current_scenario_directory(),
            asset_alignments=alignments,
        )
        by_name = {actor.name: actor for actor in prepared.actors}
        actor_samples = {
            actor.id: ActorSamples(
                positions=tuple(
                    tuple(float(value) for value in point)
                    for point in by_name[actor.name].positions_m
                ),
                orientations=tuple(
                    tuple(float(value) for value in angles)
                    for angles in by_name[actor.name].orientation.euler_deg
                ),
                velocities_mps=tuple(
                    tuple(float(value) for value in velocity)
                    for velocity in by_name[actor.name].mobility.velocities_mps
                ),
                forward_vectors=tuple(
                    tuple(float(value) for value in forward)
                    for forward in by_name[actor.name].mobility.forward_vectors
                ),
                has_physical_velocity=bool(by_name[actor.name].mobility.has_physical_velocity),
            )
            for actor in scenario.actors
            if actor.name in by_name
        }
        timeline = Timeline(
            scenario.timeline.steps,
            scenario.timeline.duration_s,
        )
        prepared_groups = {group.name: group for group in prepared.groups}
        group_samples = {
            group.id: GroupSamples(
                tuple(
                    tuple(float(value) for value in point)
                    for point in prepared_groups[group.name].mobility.positions_m
                ),
                prepared_groups[group.name].mobility,
                timeline,
            )
            for group in scenario.groups
            if group.name in prepared_groups
        }
        return actor_samples, group_samples

    def _refresh_viewport(self) -> None:
        document = self.document
        port = getattr(self.viewport, "port", None)
        if document is None or port is None:
            return
        self.osm_map_credit.setVisible(
            self.osm_panel.active
            or bool(document.scenario.scene and document.scenario.scene.source == "osm")
        )
        if self.osm_panel.active:
            port.clear_transform_gizmo()
            port.reconcile(
                OverlaySnapshot(
                    document_id=document.scenario.document_id,
                    revision=document.revision,
                    scene_assets=self.osm_panel.assets(),
                    work_plane_visible=False,
                )
            )
            return
        overlays = []
        compiled_scenario = self._compiled_scenario
        preview = self._early_scene
        if (
            preview is not None
            and preview.scenario_directory == self._current_scenario_directory()
            and preview.scenario.scene == document.scenario.scene
        ):
            compiled_scenario = preview.scenario
            available_scene_assets = preview.scene_assets
        elif (
            self.compilation is not None
            and compiled_scenario is not None
            and self._compiled_directory == self._current_scenario_directory()
            and compiled_scenario.scene == document.scenario.scene
        ):
            available_scene_assets = self.compilation.scene_assets
        else:
            available_scene_assets = ()
        scene_assets = tuple(
            SceneOverlayAsset(
                cache_key=asset.cache_key,
                name=asset.name,
                payload=asset.mesh,
                material=asset.material,
                material_type=asset.material_type,
                ground=asset.ground,
            )
            for asset in available_scene_assets
        )
        from .map_overlay import MAP_NAME

        has_map = any(asset.name == MAP_NAME for asset in scene_assets)
        if has_map:
            attribution = next(
                (asset.attribution for asset in available_scene_assets if asset.name == MAP_NAME),
                None,
            )
            if attribution:
                self.osm_map_credit.setText(attribution)
        self.map_overlay_check.setEnabled(has_map)
        self.map_overlay_check.setVisible(has_map)
        self.osm_map_credit.setVisible(
            has_map or bool(document.scenario.scene and document.scenario.scene.source == "osm")
        )
        if not self.map_overlay_check.isChecked():
            scene_assets = tuple(asset for asset in scene_assets if asset.name != MAP_NAME)
        pending_motion_actor_ids: set[UUID] = set()
        if self._pending_subject_id is not None and (
            self._mobility_draft_pending or self._group_settings_draft_pending
        ):
            if self._pending_subject_is_group:
                for pending_actor in document.actors:
                    if not isinstance(
                        pending_actor.mobility,
                        GroupMemberMobilitySpec,
                    ):
                        continue
                    try:
                        if UUID(str(pending_actor.mobility.group)) == self._pending_subject_id:
                            pending_motion_actor_ids.add(pending_actor.id)
                    except ValueError:
                        continue
            else:
                pending_motion_actor_ids.add(self._pending_subject_id)
        for actor in document.actors:
            if document.scenario.recorded_poses and (
                self.compilation is None or self.compilation.recording is None
            ):
                continue
            sample = self._prepared_samples(actor)
            frame_positions = tuple(sample.positions) if sample is not None else ()
            trajectory_positions = self._trajectory_positions(
                actor.mobility,
                frame_positions,
            )
            if document.scenario.recorded_poses:
                if not frame_positions:
                    continue
                trajectory_positions = self.compilation.recording.trajectory_positions.get(
                    actor.id, frame_positions
                )
            if (
                frame_positions
                and actor.target is not None
                and actor.target.source == "animated_asset"
            ):
                provider = self.compilation.pose_sources.get(actor.id)
                trajectory_positions = (
                    provider.trajectory_positions if provider is not None else frame_positions
                )
            pose_positions = frame_positions or trajectory_positions
            step = min(self._play_step, max(len(pose_positions) - 1, 0))
            current_position = pose_positions[step]
            selected = actor.id == document.selected_actor_id
            pending_group_member = False
            if (
                self._pending_subject_is_group
                and self._pending_subject_id is not None
                and isinstance(actor.mobility, GroupMemberMobilitySpec)
            ):
                try:
                    pending_group_member = (
                        UUID(str(actor.mobility.group)) == self._pending_subject_id
                    )
                except ValueError:
                    pass
            direct_candidate_actor = (
                self._pending_subject_id == actor.id and not self._pending_subject_is_group
            ) or pending_group_member
            pending_look_at_dependency = False
            if (
                pending_motion_actor_ids
                and isinstance(actor.orientation, LookAtOrientationSpec)
                and actor.orientation.actor is not None
            ):
                try:
                    pending_look_at_dependency = (
                        look_at_actor_id(actor.orientation) in pending_motion_actor_ids
                    )
                except ValueError:
                    pass
            candidate_actor = (
                self._candidate_scenario.actor(actor.id)
                if self._candidate_scenario is not None
                and (direct_candidate_actor or pending_look_at_dependency)
                else None
            )
            candidate_sample = (
                self._candidate_compilation.samples.get(actor.id)
                if candidate_actor is not None and self._candidate_compilation is not None
                else None
            )
            pending_positions = (
                tuple(candidate_sample.positions)
                if (
                    (self._mobility_draft_pending or self._group_settings_draft_pending)
                    and direct_candidate_actor
                    and candidate_sample is not None
                )
                else ()
            )
            if (
                not pending_positions
                and direct_candidate_actor
                and candidate_actor is not None
                and self._mobility_draft_pending
            ):
                pending_positions = self._trajectory_positions(candidate_actor.mobility, ())
            pending_step = min(self._play_step, max(len(pending_positions) - 1, 0))
            pending_current_position = (
                pending_positions[pending_step] if pending_positions else None
            )
            pending_control_rig = None
            if (
                self._mobility_draft_pending
                and not pending_group_member
                and candidate_actor is not None
            ):
                try:
                    pending_control_rig = mobility_control_rig(candidate_actor.mobility)
                except TypeError:
                    pass
            trajectory_visible = self._overlay_scope_applies(
                OverlayVisibility(self.trajectory_visibility_combo.currentData()),
                selected,
            )
            frame_samples_visible = self._overlay_scope_applies(
                OverlayVisibility(self.frame_samples_visibility_combo.currentData()),
                selected,
            ) or (self._play_timer.isActive() and selected)
            control_rig_visible = (
                self._tool is AuthoringTool.MOVE
                and not self._play_timer.isActive()
                and not actor.locked
                and not document.read_only
                and document.scenario.capability("mobility", actor.id).editable
                and self._overlay_scope_applies(
                    OverlayVisibility(self.control_rig_visibility_combo.currentData()),
                    selected,
                )
            )
            if control_rig_visible:
                try:
                    control_rig = mobility_control_rig((candidate_actor or actor).mobility)
                except TypeError:
                    control_rig = None
            else:
                control_rig = None
            orientation_matrix = None
            orientation_angles: tuple[float, float, float] | None = None
            if sample is not None and sample.orientations:
                orientation_step = min(step, len(sample.orientations) - 1)
                orientation_angles = sample.orientations[orientation_step]
            elif isinstance(actor.orientation, FixedOrientationSpec):
                orientation_angles = (
                    actor.orientation.yaw_deg,
                    actor.orientation.pitch_deg,
                    actor.orientation.roll_deg,
                )
            if orientation_angles is not None:
                pose = prepared_actor_pose(
                    current_position,
                    orientation_angles,
                )
                orientation_matrix = tuple(tuple(float(value) for value in row) for row in pose)
            pending_orientation_matrix = None
            if (
                candidate_sample is not None
                and candidate_sample.orientations
                and (
                    self._orientation_draft_pending
                    or self._mobility_draft_pending
                    or self._group_settings_draft_pending
                )
            ):
                candidate_orientation_step = min(
                    self._play_step,
                    len(candidate_sample.orientations) - 1,
                )
                candidate_pose_position = (
                    candidate_sample.positions[
                        min(self._play_step, len(candidate_sample.positions) - 1)
                    ]
                    if candidate_sample.positions
                    else current_position
                )
                candidate_pose = prepared_actor_pose(
                    candidate_pose_position,
                    candidate_sample.orientations[candidate_orientation_step],
                )
                pending_orientation_matrix = tuple(
                    tuple(float(value) for value in row) for row in candidate_pose
                )

            target_asset = None
            compiled_actor = (
                compiled_scenario.actor(actor.id) if compiled_scenario is not None else None
            )
            target_frames = (
                self.compilation.target_assets.get(actor.id, ())
                if self.compilation is not None
                and compiled_scenario is not None
                and self._compiled_directory == self._current_scenario_directory()
                and compiled_actor is not None
                and compiled_actor.target == actor.target
                and compiled_actor.target_representation_mapping()
                == actor.target_representation_mapping()
                and compiled_scenario.timeline == document.scenario.timeline
                else ()
            )
            if target_frames:
                try:
                    frame = target_frames[min(self._play_step, len(target_frames) - 1)]
                except (OSError, ValueError, KeyError) as exc:
                    self.inline_errors.setText(f"{actor.name}: frame {self._play_step}: {exc}")
                    continue
                target_asset = TargetOverlayAsset(
                    cache_key=frame.cache_key,
                    payload=frame.mesh,
                    material=frame.material,
                    scattering_points_local_m=frame.scattering_points_local_m,
                    scattering_point_ids=frame.scattering_point_ids,
                    representation=frame.representation,
                    metadata_source=frame.metadata_source,
                    local_to_actor=tuple(
                        tuple(float(value) for value in row) for row in frame.local_to_actor
                    ),
                )
            pending_target_asset = None
            candidate_target_frames = (
                self._candidate_compilation.target_assets.get(actor.id, ())
                if self._target_draft_pending
                and candidate_actor is not None
                and self._candidate_compilation is not None
                else ()
            )
            if candidate_target_frames:
                candidate_frame = candidate_target_frames[
                    min(step, len(candidate_target_frames) - 1)
                ]
                pending_target_asset = TargetOverlayAsset(
                    cache_key=candidate_frame.cache_key,
                    payload=candidate_frame.mesh,
                    material=candidate_frame.material,
                    scattering_points_local_m=candidate_frame.scattering_points_local_m,
                    scattering_point_ids=candidate_frame.scattering_point_ids,
                    representation=candidate_frame.representation,
                    metadata_source=candidate_frame.metadata_source,
                    local_to_actor=tuple(
                        tuple(float(value) for value in row)
                        for row in candidate_frame.local_to_actor
                    ),
                )

            look_at_position = None
            if isinstance(actor.orientation, LookAtOrientationSpec):
                target_id = (
                    look_at_actor_id(actor.orientation)
                    if actor.orientation.actor is not None
                    else None
                )
                target_actor = document.scenario.actor(target_id) if target_id is not None else None
                target_sample = (
                    self._prepared_samples(target_actor) if target_actor is not None else None
                )
                if target_sample is not None and target_sample.positions:
                    target_step = min(self._play_step, len(target_sample.positions) - 1)
                    look_at_position = target_sample.positions[target_step]
                elif target_actor is not None:
                    target_positions = self._trajectory_positions(target_actor.mobility, ())
                    look_at_position = target_positions[
                        min(self._play_step, len(target_positions) - 1)
                    ]
                elif actor.orientation.point_m is not None:
                    look_at_position = actor.orientation.point_m
            pending_look_at_position = None
            if (
                candidate_actor is not None
                and isinstance(candidate_actor.orientation, LookAtOrientationSpec)
                and (
                    self._orientation_draft_pending
                    or self._mobility_draft_pending
                    or self._group_settings_draft_pending
                )
            ):
                try:
                    candidate_target_id = (
                        look_at_actor_id(candidate_actor.orientation)
                        if candidate_actor.orientation.actor is not None
                        else None
                    )
                except ValueError:
                    candidate_target_id = None
                candidate_target_samples = (
                    self._candidate_compilation.samples.get(candidate_target_id)
                    if candidate_target_id is not None and self._candidate_compilation is not None
                    else None
                )
                if candidate_target_samples is not None and candidate_target_samples.positions:
                    pending_look_at_position = candidate_target_samples.positions[
                        min(
                            self._play_step,
                            len(candidate_target_samples.positions) - 1,
                        )
                    ]
                elif candidate_actor.orientation.point_m is not None:
                    pending_look_at_position = candidate_actor.orientation.point_m
            group_origin_position = None
            group_frame_matrix = None
            if isinstance(actor.mobility, GroupMemberMobilitySpec):
                try:
                    member_group = document.scenario.group(UUID(str(actor.mobility.group)))
                except ValueError:
                    member_group = None
                if member_group is not None and (
                    selected or member_group.id == document.selected_group_id
                ):
                    member_group_samples = self._prepared_group_samples(member_group)
                    if member_group_samples is not None and member_group_samples.positions:
                        group_step = min(
                            self._play_step,
                            len(member_group_samples.positions) - 1,
                        )
                        group_origin_position = member_group_samples.positions[group_step]
                        group_frame_matrix = member_group_samples.frame_transform(step=group_step)
            overlays.append(
                ActorOverlaySnapshot(
                    actor_id=actor.id,
                    role=actor.role.value,
                    name=actor.name,
                    positions=trajectory_positions,
                    frame_samples=frame_positions,
                    current_position=current_position,
                    mobility_control_rig=control_rig,
                    selected=selected,
                    visible=actor.visible,
                    locked=actor.locked,
                    trajectory_visible=trajectory_visible,
                    frame_samples_visible=frame_samples_visible,
                    closed_trajectory=(
                        isinstance(actor.mobility, CircularMobilitySpec)
                        and not document.scenario.recorded_poses
                    ),
                    trajectory_hovered=(
                        actor.id == self._hovered_actor_id
                        and self._hovered_component == "trajectory_hit"
                    ),
                    hovered_control_key=(
                        self._hovered_component.removeprefix("mobility_control_")
                        if control_rig is not None
                        and actor.id == self._hovered_actor_id
                        and self._hovered_component is not None
                        and self._hovered_component.startswith("mobility_control_")
                        else None
                    ),
                    preview_provenance=(
                        PreviewProvenance.RECORDED_FRAMES
                        if document.scenario.recorded_poses
                        else (
                            PreviewProvenance.GENERATOR_PREPARED
                            if sample is not None
                            else PreviewProvenance.AUTHORED_DRAFT
                        )
                    ),
                    trajectory_display=(
                        TrajectoryDisplayMode.OBSERVATIONS
                        if sample is not None
                        and not sample.has_physical_velocity
                        and not document.scenario.recorded_poses
                        else TrajectoryDisplayMode.PATH
                    ),
                    trajectory_geometry_key=(
                        f"recorded:{id(trajectory_positions)}"
                        if document.scenario.recorded_poses
                        else f"{repr(actor.mobility)}:{id(sample.positions) if sample else 'draft'}"
                    ),
                    frame_samples_geometry_key=(
                        str(id(sample.positions)) if sample is not None else ""
                    ),
                    status=self._actor_status(actor.id),
                    orientation_matrix=orientation_matrix,
                    pending_positions=pending_positions,
                    pending_trajectory_display=(
                        TrajectoryDisplayMode.OBSERVATIONS
                        if candidate_sample is not None
                        and not candidate_sample.has_physical_velocity
                        else TrajectoryDisplayMode.PATH
                    ),
                    pending_current_position=pending_current_position,
                    pending_look_at_position=pending_look_at_position,
                    pending_mobility_control_rig=pending_control_rig,
                    pending_orientation_matrix=pending_orientation_matrix,
                    pending_target_asset=pending_target_asset,
                    pending_geometry_key=(
                        str(id(candidate_sample.positions)) if candidate_sample is not None else ""
                    ),
                    mobility_draft_pending=(
                        (self._mobility_draft_pending or self._group_settings_draft_pending)
                        and direct_candidate_actor
                        and candidate_actor is not None
                    ),
                    orientation_draft_pending=(
                        self._orientation_draft_pending
                        and direct_candidate_actor
                        and candidate_actor is not None
                    ),
                    group_origin_position=group_origin_position,
                    group_frame_matrix=group_frame_matrix,
                    transform_rotation_enabled=isinstance(
                        actor.orientation,
                        FixedOrientationSpec,
                    ),
                    target_asset=target_asset,
                    look_at_position=look_at_position,
                )
            )
        for group in document.groups:
            prepared_positions = self._prepared_group_positions(group)
            prepared_group_samples = self._prepared_group_samples(group)
            group_has_physical_velocity = self._prepared_group_has_physical_velocity(group)
            frame_positions = prepared_positions or ()
            trajectory_positions = self._trajectory_positions(
                group.mobility,
                frame_positions,
            )
            pose_positions = frame_positions or trajectory_positions
            step = min(self._play_step, max(len(pose_positions) - 1, 0))
            selected = group.id == document.selected_group_id
            candidate_group = (
                self._candidate_scenario.group(group.id)
                if self._candidate_scenario is not None
                and self._pending_subject_id == group.id
                and self._pending_subject_is_group
                else None
            )
            candidate_group_samples = (
                self._candidate_compilation.group_samples.get(group.id)
                if candidate_group is not None and self._candidate_compilation is not None
                else None
            )
            pending_positions = (
                tuple(candidate_group_samples.positions)
                if (
                    (self._mobility_draft_pending or self._group_settings_draft_pending)
                    and candidate_group_samples is not None
                )
                else ()
            )
            pending_control_rig = None
            if self._mobility_draft_pending and candidate_group is not None:
                try:
                    pending_control_rig = mobility_control_rig(candidate_group.mobility)
                except TypeError:
                    pass
            control_rig = None
            if (
                self._tool is AuthoringTool.MOVE
                and not self._play_timer.isActive()
                and not group.locked
                and not document.read_only
                and self._overlay_scope_applies(
                    OverlayVisibility(self.control_rig_visibility_combo.currentData()),
                    selected,
                )
            ):
                try:
                    control_rig = mobility_control_rig((candidate_group or group).mobility)
                except TypeError:
                    pass
            issues = (
                self.compilation.issues_for_group(group.id)
                if self._compilation_is_current() and self.compilation is not None
                else ()
            )
            status = (
                ActorVisualState.INVALID
                if any(issue.severity is IssueSeverity.ERROR for issue in issues)
                else (
                    ActorVisualState.COMPLETE
                    if self._compilation_is_current()
                    else ActorVisualState.PENDING
                )
            )
            overlays.append(
                ActorOverlaySnapshot(
                    actor_id=group.id,
                    role="group",
                    name=group.name,
                    positions=trajectory_positions,
                    frame_samples=frame_positions,
                    current_position=pose_positions[step],
                    mobility_control_rig=control_rig,
                    selected=selected,
                    visible=group.visible,
                    locked=group.locked,
                    trajectory_visible=self._overlay_scope_applies(
                        OverlayVisibility(self.trajectory_visibility_combo.currentData()),
                        selected,
                    ),
                    frame_samples_visible=self._overlay_scope_applies(
                        OverlayVisibility(self.frame_samples_visibility_combo.currentData()),
                        selected,
                    ),
                    closed_trajectory=isinstance(
                        group.mobility,
                        CircularMobilitySpec,
                    ),
                    preview_provenance=(
                        PreviewProvenance.GENERATOR_PREPARED
                        if prepared_positions is not None
                        else PreviewProvenance.AUTHORED_DRAFT
                    ),
                    trajectory_display=(
                        TrajectoryDisplayMode.OBSERVATIONS
                        if group_has_physical_velocity is False
                        else TrajectoryDisplayMode.PATH
                    ),
                    trajectory_geometry_key=(
                        f"{repr(group.mobility)}:"
                        f"{id(prepared_group_samples.positions) if prepared_group_samples else 'draft'}"
                    ),
                    frame_samples_geometry_key=(
                        str(id(prepared_group_samples.positions))
                        if prepared_group_samples is not None
                        else ""
                    ),
                    pending_positions=pending_positions,
                    pending_trajectory_display=(
                        TrajectoryDisplayMode.OBSERVATIONS
                        if candidate_group_samples is not None
                        and not candidate_group_samples.has_physical_velocity
                        else TrajectoryDisplayMode.PATH
                    ),
                    pending_current_position=(
                        pending_positions[min(self._play_step, len(pending_positions) - 1)]
                        if pending_positions
                        else None
                    ),
                    pending_mobility_control_rig=pending_control_rig,
                    pending_geometry_key=(
                        str(id(candidate_group_samples.positions))
                        if candidate_group_samples is not None
                        else ""
                    ),
                    mobility_draft_pending=(
                        (self._mobility_draft_pending or self._group_settings_draft_pending)
                        and candidate_group is not None
                    ),
                    group_origin_position=(
                        prepared_group_samples.positions[
                            min(self._play_step, len(prepared_group_samples.positions) - 1)
                        ]
                        if selected
                        and prepared_group_samples is not None
                        and prepared_group_samples.positions
                        else None
                    ),
                    group_frame_matrix=(
                        prepared_group_samples.frame_transform(
                            step=min(
                                self._play_step,
                                len(prepared_group_samples.positions) - 1,
                            )
                        )
                        if selected
                        and prepared_group_samples is not None
                        and prepared_group_samples.positions
                        else None
                    ),
                    status=status,
                    transform_rotation_enabled=False,
                )
            )
        placement_guide_start = None
        if self._waypoint_session_actor_id is not None and self._pending_waypoints:
            placement_guide_start = self._pending_waypoints[-1]
        port.reconcile(
            OverlaySnapshot(
                document_id=document.scenario.document_id,
                revision=document.revision,
                scene_assets=scene_assets,
                actors=tuple(overlays),
                guides=self.builder_tools.guides(overlays),
                work_plane_z=self.work_plane_z_spin.value(),
                work_plane_visible=self.work_plane_check.isChecked(),
                grid_snap_m=(
                    self.grid_snap_spacing_spin.value()
                    if self.grid_snap_check.isChecked()
                    else None
                ),
                placement_ghost=self._placement_ghost,
                placement_guide_start=placement_guide_start,
                orientation_axes_visibility=OverlayVisibility(
                    self.orientation_axes_combo.currentData()
                ),
                look_at_visibility=OverlayVisibility(self.look_at_rays_combo.currentData()),
            )
        )
        if (
            self._osm_fit_pending
            and scene_assets
            and compiled_scenario is not None
            and compiled_scenario.source_snapshot == document.scenario.source_snapshot
        ):
            port.fit_all()
            self._osm_fit_pending = False
        self._sync_transform_gizmo()

        if self.camera_controls is not None:
            self.camera_controls.sync(port.current_snapshot())
        selected_overlay = next((actor for actor in overlays if actor.selected), None)
        self.trajectory_preview.setVisible(selected_overlay is not None)
        if selected_overlay is not None:
            context_key = (document.scenario.scene, id(available_scene_assets))
            if context_key != self.trajectory_preview.context_key:
                from .street_routes import scenario_graph

                graph = scenario_graph(document.scenario, self._current_scenario_directory())
                self.trajectory_preview.set_context(available_scene_assets, key=context_key)
                self.trajectory_preview.load_streets(graph)
            self.trajectory_preview.set_paths(
                selected_overlay.positions,
                selected_overlay.pending_positions,
                current=selected_overlay.current_position,
                proposed_current=selected_overlay.pending_current_position,
                last_valid=self._candidate_failure is not None,
            )
        self.frame_label.setText(f"{self._play_step + 1} / {document.scenario.timeline.steps}")

    @staticmethod
    def _overlay_scope_applies(scope: OverlayVisibility, selected: bool) -> bool:
        return scope is OverlayVisibility.ALL or (scope is OverlayVisibility.SELECTED and selected)

    def _sync_transform_gizmo(self) -> None:
        """Keep the persistent semantic gizmo attached to the editable selection."""

        document = self.document
        port = getattr(self.viewport, "port", None)
        if document is None or port is None:
            return
        actor = document.selected_actor
        if actor is not None and self._candidate_scenario is not None:
            actor = self._candidate_scenario.actor(actor.id) or actor
        show = getattr(port, "show_transform_gizmo", None)
        if (
            self._tool is AuthoringTool.MOVE
            and actor is not None
            and actor.visible
            and not actor.locked
            and document.scenario.capability("mobility", actor.id).editable
            and self._mobility_translation_supported(actor.mobility)
            and not document.read_only
            and callable(show)
            and show(actor.id)
        ):
            return
        clear = getattr(port, "clear_transform_gizmo", None)
        if callable(clear):
            clear()

    @staticmethod
    def _trajectory_positions(
        mobility: Any,
        prepared_positions: tuple[tuple[float, float, float], ...],
    ) -> tuple[tuple[float, float, float], ...]:
        """Return the exact authored route or canonical prepared circular path."""

        if isinstance(mobility, StationaryMobilitySpec):
            return (mobility.position_m,)
        if isinstance(mobility, LinearMobilitySpec):
            return (mobility.start_m, mobility.end_m)
        if isinstance(mobility, WaypointMobilitySpec):
            if mobility.interpolation == "catmull_rom" and prepared_positions:
                return prepared_positions
            return mobility.points_m
        if isinstance(mobility, CircularMobilitySpec):
            if prepared_positions:
                return prepared_positions
            start = mobility_control_rig(mobility).control("start_angle").position
            return (start,)
        if prepared_positions:
            return prepared_positions
        return ((0.0, 0.0, 0.0),)

    @staticmethod
    def _mobility_translation_supported(mobility: Any) -> bool:
        """Return whether viewport dragging can translate this mobility."""

        return mobility_capability(str(mobility.type)).supports_whole_path_translation

    # -- tree and inspector --------------------------------------------

    def _add_actor(self, role: ActorRole) -> None:
        if (
            self.document is not None
            and not self._read_only
            and self._resolve_pending_inspector_edits()
        ):
            self.document.add_default_actor(role)
            if role is ActorRole.TARGET:
                self.inspector_scroll.verticalScrollBar().setValue(0)
                self.target_asset_combo.setFocus(Qt.OtherFocusReason)
            # The newly selected actor consumes the very next primary click;
            # no modifier or secondary "Add Point" action is required.
            self.set_tool(AuthoringTool.PLACE)

    def _form_group(self) -> None:
        document = self.document
        if document is None or document.read_only:
            return
        if not self._resolve_pending_inspector_edits():
            return
        if len(document.actors) < 2:
            QMessageBox.information(
                self,
                "Form group",
                "Add at least two actors before forming a group.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Form Group")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Choose at least two actors:", dialog))
        reference_label = QLabel(
            "Formation offsets preserve actor positions at the start frame.",
            dialog,
        )
        reference_label.setWordWrap(True)
        layout.addWidget(reference_label)
        actor_list = QListWidget(dialog)
        for actor in document.actors:
            item = QListWidgetItem(f"{actor.name} ({actor.role.value.upper()})")
            item.setData(Qt.UserRole, str(actor.id))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if actor.id == document.selected_actor_id else Qt.Unchecked
            )
            actor_list.addItem(item)
        layout.addWidget(actor_list)

        options = AlignedFormLayout()
        primary_combo = QComboBox(dialog)
        for actor in document.actors:
            primary_combo.addItem(actor.name, str(actor.id))
        if document.selected_actor_id is not None:
            primary_combo.setCurrentIndex(primary_combo.findData(str(document.selected_actor_id)))
        use_primary_motion = QCheckBox("Use the primary actor's motion", dialog)
        use_primary_motion.setChecked(True)
        options.addRow("Primary actor", primary_combo)
        options.addRow("", use_primary_motion)
        layout.addLayout(options)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return

        selected_ids = tuple(
            UUID(str(actor_list.item(index).data(Qt.UserRole)))
            for index in range(actor_list.count())
            if actor_list.item(index).checkState() == Qt.Checked
        )
        if len(selected_ids) < 2:
            QMessageBox.warning(
                self,
                "Form group",
                "A group requires at least two actors.",
            )
            return
        selected = tuple(
            actor
            for actor_id in selected_ids
            if (actor := document.scenario.actor(actor_id)) is not None
        )
        primary = document.scenario.actor(primary_combo.currentData())
        if primary is None or primary.id not in selected_ids:
            primary = selected[0]

        formation_step = 0
        positions: list[tuple[float, float, float]] = []
        for actor in selected:
            samples = self._prepared_samples(actor)
            if samples is not None and samples.positions:
                positions.append(samples.positions[formation_step])
            else:
                positions.append(self._trajectory_positions(actor.mobility, ())[0])

        if use_primary_motion.isChecked():
            if isinstance(primary.mobility, GroupMemberMobilitySpec):
                QMessageBox.warning(
                    self,
                    "Form group",
                    "The primary actor already belongs to a group.",
                )
                return
            group_mobility = primary.mobility
        else:
            count = float(len(positions))
            centroid = tuple(
                sum(position[axis] for position in positions) / count for axis in range(3)
            )
            group_mobility = StationaryMobilitySpec(position_m=centroid)

        try:
            offsets = self.compiler.group_offsets(
                group_mobility,
                document.scenario.timeline.steps,
                document.scenario.timeline.duration_s,
                tuple(positions),
                step=formation_step,
                scenario_directory=self._current_scenario_directory(),
                resources=document.scenario.resources,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Form group", str(exc))
            return

        used_names = {group.name for group in document.groups}
        suffix = 1
        while f"Group{suffix}" in used_names:
            suffix += 1
        group = AuthoringGroup.create(f"Group{suffix}").with_changes(mobility=group_mobility)
        members = tuple(
            actor.with_changes(
                mobility=GroupMemberMobilitySpec(
                    group=str(group.id),
                    offset_m=GroupOffsetSpec(
                        right=offset[0],
                        forward=offset[1],
                        up=offset[2],
                    ),
                )
            )
            for actor, offset in zip(selected, offsets)
        )
        document.add_group_with_members(group, members)

    def _tree_selection_changed(self) -> None:
        if self._syncing or self.document is None:
            return
        if self._waypoint_session_actor_id is not None:
            # Drawing is an explicit modal task. Keep the session and its
            # actor selected until the user chooses Finish or Cancel.
            self._schedule_tree_refresh()
            return
        self._finish_target_transform(commit=False)
        self._finish_mobility_drag(commit=False)
        items = self.actor_tree.selectedItems()
        value = items[0].data(0, Qt.UserRole) if items else None
        kind = items[0].data(0, Qt.UserRole + 1) if items else None
        current = self._selected_subject()
        next_subject = (UUID(str(value)), kind == "group") if value is not None else None
        if (
            self.has_pending_inspector_edits
            and next_subject != current
            and not self._resolve_pending_inspector_edits()
        ):
            self._schedule_tree_refresh()
            return
        if kind == "group":
            self.document.select_group(value)
        else:
            self.document.select(value)

    def _tree_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        """Commit visibility and lock checkboxes as ordinary undoable edits."""

        if self._syncing or self.document is None or self.document.read_only:
            return
        subject_id = item.data(0, Qt.UserRole)
        kind = item.data(0, Qt.UserRole + 1)
        visible = item.checkState(2) == Qt.Checked
        locked = item.checkState(3) == Qt.Checked
        if kind == "group":
            group = self.document.scenario.group(subject_id) if subject_id else None
            if group is None or (visible == group.visible and locked == group.locked):
                return
            self.document.update_group(group.id, visible=visible, locked=locked)
            return
        actor = self.document.scenario.actor(subject_id) if subject_id else None
        if actor is None or (visible == actor.visible and locked == actor.locked):
            return
        self.document.update_actor(actor.id, visible=visible, locked=locked)

    @staticmethod
    def _has_imported_quality(scenario: AuthoringScenario) -> bool:
        return (
            scenario.timeline.quality is QualityPreset.CUSTOM
            or scenario.source_snapshot.has_path("raytracing.quality.custom")
        )

    def _update_preserved_settings_notice(
        self,
        scenario: AuthoringScenario,
    ) -> None:
        imported_custom = self._has_imported_quality(scenario)
        if scenario.source_snapshot.is_empty and not imported_custom:
            self.preserved_settings_label.clear()
            self.preserved_settings_label.setVisible(False)
            return

        message = (
            "Advanced settings from the imported YAML are preserved unchanged "
            "and are visible in YAML Preview, but are not editable here."
        )
        if imported_custom:
            message += (
                " Choosing a standard quality preset and applying Timeline "
                "replaces the preserved custom quality settings."
            )
        self.preserved_settings_label.setText(message)
        self.preserved_settings_label.setVisible(True)

    def _sync_quality_controls(self, scenario: AuthoringScenario) -> None:
        preserved_index = self.quality_combo.findData(_PRESERVED_QUALITY_CHOICE)
        if preserved_index >= 0:
            self.quality_combo.removeItem(preserved_index)

        timeline = scenario.timeline
        if self._has_imported_quality(scenario):
            if timeline.quality is QualityPreset.CUSTOM:
                label = "custom (preserved; medium base)"
            else:
                label = f"{timeline.quality.value} + custom overrides (preserved)"
            self.quality_combo.addItem(label, _PRESERVED_QUALITY_CHOICE)
            self.quality_combo.setCurrentIndex(self.quality_combo.count() - 1)
        else:
            self.quality_combo.setCurrentIndex(self.quality_combo.findData(timeline.quality))
        self._update_preserved_settings_notice(scenario)

    def _timeline_editor_state(self) -> tuple[TimelineSettings, bool]:
        document = self.document
        if document is None:
            raise ValueError("no authoring document")
        selected_quality = self.quality_combo.currentData()
        preserve_imported_quality = selected_quality == _PRESERVED_QUALITY_CHOICE
        quality = (
            document.scenario.timeline.quality
            if preserve_imported_quality
            else QualityPreset(selected_quality)
        )
        sample_rate = self.sample_rate_spin.value() if self.sample_rate_check.isChecked() else None
        imported_rate = document.scenario.timeline.sample_rate_hz
        if (
            sample_rate is not None
            and imported_rate is not None
            and sample_rate == round(imported_rate, self.sample_rate_spin.decimals())
        ):
            # A count/quality edit must not round an untouched imported clock
            # to the widget's display precision (for example 30000 / 1001 Hz).
            sample_rate = imported_rate
        timeline = TimelineSettings(
            steps=self.steps_spin.value(),
            duration_s=self.duration_spin.value(),
            quality=quality,
            export_path_metrics=self.path_metrics_check.isChecked(),
            sample_rate_hz=sample_rate,
        )
        replace_imported_quality = (
            self._has_imported_quality(document.scenario) and not preserve_imported_quality
        )
        return timeline, replace_imported_quality

    def _mark_timeline_draft(self, *_args: Any) -> None:
        document = self.document
        if self._syncing or document is None or document.read_only:
            return
        use_rate = self.sample_rate_check.isChecked()
        self.duration_spin.setEnabled(self.steps_spin.isEnabled() and not use_rate)
        self.sample_rate_spin.setEnabled(self.steps_spin.isEnabled() and use_rate)
        try:
            timeline, replace_imported_quality = self._timeline_editor_state()
        except (TypeError, ValueError):
            self._timeline_draft_pending = True
        else:
            if use_rate:
                with QSignalBlocker(self.duration_spin):
                    self.duration_spin.setMaximum(max(1_000_000.0, timeline.duration_s))
                    self.duration_spin.setValue(timeline.duration_s)
            self._timeline_draft_pending = (
                timeline != document.scenario.timeline or replace_imported_quality
            )
        self._update_timeline_pending_ui()

    def _update_timeline_pending_ui(self) -> None:
        pending = self._timeline_draft_pending
        document = self.document
        editable = (
            document is not None
            and not document.read_only
            and self._waypoint_session_actor_id is None
        )
        self.apply_timeline_button.setEnabled(editable and pending)
        self.reset_timeline_button.setEnabled(editable and pending)
        self.reset_timeline_button.setVisible(pending)
        self.timeline_pending_label.setVisible(pending)
        if not pending:
            self.timeline_pending_label.clear()
            return
        replacing_quality = False
        try:
            _timeline, replacing_quality = self._timeline_editor_state()
        except (TypeError, ValueError):
            pass
        message = "Timeline changes are pending. Apply or reset before saving or generating."
        if replacing_quality:
            message += " Applying will replace the imported custom quality settings."
        self.timeline_pending_label.setText(message)

    def _reset_pending_timeline(self) -> None:
        self._timeline_draft_pending = False
        self._sync_document_controls()

    def _resolve_pending_timeline_edits(self) -> bool:
        if not self._timeline_draft_pending:
            return True
        choice = QMessageBox.question(
            self,
            "Pending Timeline Edits",
            "Apply the pending timeline values before continuing?",
            QMessageBox.Apply | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Apply,
        )
        if choice == QMessageBox.Apply:
            return self._apply_timeline()
        if choice == QMessageBox.Discard:
            self._reset_pending_timeline()
            return True
        return False

    def _sync_document_controls(self) -> None:
        document = self.document
        if document is None:
            return
        self._syncing = True
        try:
            timeline = document.scenario.timeline
            if not self._timeline_draft_pending:
                self.steps_spin.setValue(timeline.steps)
                self.duration_spin.setMaximum(max(1_000_000.0, timeline.duration_s))
                self.duration_spin.setValue(timeline.duration_s)
                self.sample_rate_check.setChecked(timeline.sample_rate_hz is not None)
                if timeline.sample_rate_hz is not None:
                    self.sample_rate_spin.setRange(
                        min(1e-12, timeline.sample_rate_hz), max(1e15, timeline.sample_rate_hz)
                    )
                    self.sample_rate_spin.setValue(timeline.sample_rate_hz)
                self._sync_quality_controls(document.scenario)
                self.path_metrics_check.setChecked(timeline.export_path_metrics)
            else:
                self._update_preserved_settings_notice(document.scenario)
            self.timeline_slider.setMaximum(max(timeline.steps - 1, 0))
            self._update_playback_timing(timeline)
            scene = document.scenario.scene
            if not (self.osm_panel.active or self.osm_panel.building):
                source = scene.source if scene is not None else "sionna"
                source_index = self.scene_source_combo.findData(source)
                if source_index < 0:
                    self.scene_source_combo.addItem(f"{source} (preserved)", source)
                    source_index = self.scene_source_combo.count() - 1
                self.scene_source_combo.setCurrentIndex(source_index)
                self._populate_scene_ids(scene.id if scene is not None else "")
                if source == "osm":
                    self.osm_panel.sync_config(
                        document.scenario.source_snapshot.to_mapping()
                        .get("scene", {})
                        .get("osm", {})
                    )
            actor = document.selected_actor
            self.animation_notice.setVisible(
                document.scenario.recorded_poses
                or bool(actor and actor.target and actor.target.source == "animated_asset")
            )
            if document.scenario.recorded_poses:
                self.animation_notice.setText(
                    "Saved frames own these poses. Playback is read-only; use the acquisition inputs to author new motion."
                )
            elif actor and actor.target and actor.target.source == "animated_asset":
                self.animation_notice.setText(
                    "Animation trajectory (read-only). The path, heading and scale come from "
                    "the animation asset. Route editing is not supported for imported animations. "
                    "To change the route, prepare a new asset from the original poses and update "
                    "its path in the scenario YAML. Playback settings are preserved on save."
                )
                self.animation_notice.setToolTip(str(actor.target.path))
            group = document.selected_group
            self._update_sampling_summary(actor, timeline)
            enabled = (actor is not None or group is not None) and not document.read_only
            self.name_edit.setEnabled(enabled)
            self.mobility_editor.set_editing_enabled(enabled)
            self.orientation_editor.set_editing_enabled(enabled)
            set_group_choices = getattr(self.mobility_editor, "set_group_choices", None)
            if callable(set_group_choices):
                set_group_choices(((candidate.id, candidate.name) for candidate in document.groups))
            if group is not None:
                group_editable = document.scenario.capability("group", group.id).editable
                self.name_edit.setEnabled(enabled and group_editable)
                self.mobility_editor.set_editing_enabled(enabled and group_editable)
                preserve_mobility_draft = (
                    self._mobility_draft_pending
                    and self._pending_subject_is_group
                    and self._pending_subject_id == group.id
                )
                preserve_group_settings_draft = (
                    self._group_settings_draft_pending
                    and self._pending_subject_is_group
                    and self._pending_subject_id == group.id
                )
                member_roles = {
                    candidate.role
                    for candidate in document.actors
                    if isinstance(candidate.mobility, GroupMemberMobilitySpec)
                    and candidate.mobility.group == str(group.id)
                }
                self.mobility_editor.set_actor_role(
                    ActorRole.TARGET if member_roles == {ActorRole.TARGET} else ActorRole.TX
                )
                self.name_edit.setText(group.name)
                self.role_label.setText("GROUP")
                if not preserve_mobility_draft:
                    self.mobility_editor.set_mobility(group.mobility)
                samples = (
                    self.compilation.group_samples.get(group.id)
                    if self.compilation is not None
                    else None
                )
                speed = None
                if (
                    samples is not None
                    and samples.has_physical_velocity
                    and len(samples.positions) > 1
                    and timeline.duration_s > 0.0
                ):
                    speed = (
                        sum(
                            math.dist(start, end)
                            for start, end in zip(samples.positions, samples.positions[1:])
                        )
                        / timeline.duration_s
                    )
                self.mobility_editor.set_average_speed(speed)
                self.mobility_group.setVisible(True)
                if not preserve_group_settings_draft:
                    deviation = group.deviation
                    self.group_deviation_enabled.setChecked(deviation is not None)
                    values = (
                        (
                            deviation.max_right_m,
                            deviation.max_forward_m,
                            deviation.max_up_m,
                        )
                        if deviation is not None
                        else (0.0, 0.0, 0.0)
                    )
                    for spin, value in zip(self.group_deviation_spins, values):
                        spin.setValue(value)
                    self.group_deviation_seed.setValue(
                        deviation.seed if deviation is not None else 0
                    )
                self.group_settings.setVisible(True)
                self.orientation_group.setVisible(False)
                self.target_group.setVisible(False)
                return
            if actor is None:
                self._clear_actor_inspector()
                return
            preserve_mobility_draft = (
                self._mobility_draft_pending
                and not self._pending_subject_is_group
                and self._pending_subject_id == actor.id
            )
            preserve_orientation_draft = (
                self._orientation_draft_pending
                and not self._pending_subject_is_group
                and self._pending_subject_id == actor.id
            )
            preserve_target_draft = (
                self._target_draft_pending
                and not self._pending_subject_is_group
                and self._pending_subject_id == actor.id
            )
            self.mobility_editor.set_actor_role(actor.role)
            identity_editable = document.scenario.capability("identity", actor.id).editable
            mobility_editable = document.scenario.capability("mobility", actor.id).editable
            orientation_editable = document.scenario.capability(
                "orientation",
                actor.id,
            ).editable
            self.name_edit.setEnabled(enabled and identity_editable)
            self.mobility_editor.set_editing_enabled(enabled and mobility_editable)
            self.orientation_editor.set_editing_enabled(enabled and orientation_editable)
            self.name_edit.setText(actor.name)
            self.role_label.setText(actor.role.value.upper())
            if not preserve_mobility_draft:
                self.mobility_editor.set_mobility(actor.mobility)
            samples = self._prepared_samples(actor)
            speed = None
            if (
                samples is not None
                and samples.has_physical_velocity
                and len(samples.positions) > 1
                and timeline.duration_s > 0.0
            ):
                distance = sum(
                    math.dist(start, end)
                    for start, end in zip(samples.positions, samples.positions[1:])
                )
                speed = distance / timeline.duration_s
            self.mobility_editor.set_average_speed(speed)
            self.orientation_editor.set_look_at_choices(
                (
                    (candidate.id, candidate.name)
                    for candidate in document.actors
                    if candidate.id != actor.id
                ),
                preserve_selection=preserve_orientation_draft,
            )
            if not preserve_orientation_draft:
                self.orientation_editor.set_orientation(actor.orientation)
            self.mobility_group.setVisible(True)
            self.group_settings.setVisible(False)
            self.orientation_group.setVisible(True)
            self.target_group.setVisible(actor.role is ActorRole.TARGET)
            animated = actor.target is not None and actor.target.source == "animated_asset"
            self.mobility_group.setVisible(not animated and not document.scenario.recorded_poses)
            self.orientation_group.setVisible(not animated and not document.scenario.recorded_poses)
            self._set_target_editing_enabled(actor, enabled)
            if not preserve_target_draft:
                self.target_representation_editor.set_actor(actor)
                self._populate_target_asset_choices()
            self._update_target_asset_controls()
            if actor.target is not None and not preserve_target_draft:
                target_index = self.target_asset_combo.findData(actor.target.asset_id)
                if target_index < 0:
                    self.target_asset_combo.addItem(
                        (
                            f"{actor.target.asset_id} (imported)"
                            if actor.target.source != "catalog"
                            else f"{actor.target.asset_id} (unresolved)"
                        ),
                        actor.target.asset_id,
                    )
                    target_index = self.target_asset_combo.count() - 1
                self.target_asset_combo.setCurrentIndex(target_index)
                material_index = self.target_material_combo.findData(actor.target.material)
                if material_index < 0:
                    self.target_material_combo.addItem(
                        f"{actor.target.material} (unsupported)", actor.target.material
                    )
                    material_index = self.target_material_combo.count() - 1
                self.target_material_combo.setCurrentIndex(material_index)
                self.target_scale_spin.setValue(actor.target.scale)
                self.target_mesh_animation.setChecked(actor.target.mesh_animation)
                if animated:
                    self.role_label.setText("TARGET (animated asset)")
            elif not preserve_target_draft:
                self.target_asset_combo.setCurrentIndex(0)
                self.target_material_combo.setCurrentIndex(
                    self.target_material_combo.findData("glass")
                )
                self.target_scale_spin.setValue(1.0)
                self.target_mesh_animation.setChecked(True)
        finally:
            # Waypoint/keyframe table cell editors are recreated by set_*()
            # during selection sync, so discover the new widgets before
            # releasing the synchronization guard.
            self._connect_editor_draft_signals(self.mobility_editor, "mobility")
            self._connect_editor_draft_signals(self.orientation_editor, "orientation")
            self._syncing = False
            self._update_timeline_pending_ui()
            self._sync_motion_edit_action()
            self._update_tool_hint()

    def _clear_actor_inspector(self) -> None:
        """Reset actor-specific widgets without changing the document."""

        self.name_edit.clear()
        self.role_label.setText("—")
        self.mobility_editor.set_actor_role(None)
        self.mobility_editor.set_mobility(StationaryMobilitySpec(position_m=(0.0, 0.0, 0.0)))
        self.mobility_editor.set_average_speed(None)
        self.orientation_editor.set_look_at_choices((), preserve_selection=False)
        self.orientation_editor.set_orientation(FixedOrientationSpec())
        self.mobility_group.setVisible(True)
        self.group_settings.setVisible(False)
        self.orientation_group.setVisible(True)
        self.target_group.setVisible(False)

    def _update_sampling_summary(
        self,
        actor: AuthoringActor | None,
        timeline: TimelineSettings,
    ) -> None:
        """Explain authored controls versus discrete generator frame samples."""

        steps = int(timeline.steps)
        message = (
            f"{steps} generator frame sample{'s' if steps != 1 else ''} per actor. "
            "Trajectory shows the motion route; Frames are the exact generator samples "
            "followed during playback."
        )
        warning = False
        if actor is not None and isinstance(actor.mobility, WaypointMobilitySpec):
            point_count = len(actor.mobility.points_m)
            segment_count = max(point_count - 1, 0)
            boundaries_land_on_frames = segment_count <= 1 or (
                steps > 1 and (steps - 1) % segment_count == 0
            )
            if point_count > 1 and not boundaries_land_on_frames:
                message += (
                    f" {point_count} authored waypoints do not all coincide with these frame "
                    "times; increase Steps for a denser generated trajectory."
                )
                warning = True
        self.sampling_summary_label.setText(message)
        self.sampling_summary_label.setStyleSheet("color: #e0a020;" if warning else "")

    def _apply_name(self) -> None:
        if self._syncing or self.document is None:
            return
        name = self.name_edit.text().strip()
        if self.document.selected_actor is not None:
            self.document.rename_actor(self.document.selected_actor.id, name)
        elif self.document.selected_group is not None:
            self.document.rename_group(self.document.selected_group.id, name)

    def _selected_current_position(self) -> tuple[float, float, float]:
        """Return the selected actor pose at the visible timeline step."""

        document = self.document
        actor = document.selected_actor if document is not None else None
        group = document.selected_group if document is not None else None
        if self._candidate_compilation is not None and self._pending_matches_current_subject():
            samples = (
                self._candidate_compilation.group_samples.get(group.id)
                if group is not None
                else self._candidate_compilation.samples.get(actor.id)
            )
            if samples is not None and samples.positions:
                return samples.positions[min(self._play_step, len(samples.positions) - 1)]
        if group is not None:
            positions = self._prepared_group_positions(group)
            if positions:
                step = min(self._play_step, len(positions) - 1)
                return positions[step]
        if actor is None:
            return self._trajectory_positions(group.mobility, ())[0] if group else (0.0, 0.0, 0.0)
        samples = self._prepared_samples(actor)
        if samples is not None and samples.positions:
            step = min(self._play_step, len(samples.positions) - 1)
            return samples.positions[step]
        positions = self._trajectory_positions(actor.mobility, ())
        return positions[min(self._play_step, len(positions) - 1)]

    def _mobility_type_changed(self, token: str) -> None:
        """Prepare a position-preserving typed conversion for explicit apply."""

        if self._syncing or self.document is None:
            return
        actor = self.document.selected_actor
        group = self.document.selected_group
        current = (
            actor.mobility if actor is not None else group.mobility if group is not None else None
        )
        if self._mobility_draft_pending and self._pending_matches_current_subject():
            current = getattr(self, "_pending_mobility_value", current)
        if current is None:
            return
        try:
            converted = convert_mobility(
                current,
                token,
                self._selected_current_position(),
                duration_s=self.document.scenario.timeline.duration_s,
                seed=(actor.id if actor is not None else group.id).int & 0x7FFFFFFF,
            )
        except MobilityNeedsContextError:
            return
        self.mobility_editor.set_mobility(converted)
        self._connect_editor_draft_signals(self.mobility_editor, "mobility")
        self._mark_inspector_draft("mobility")

    def _apply_mobility(self) -> bool:
        if self.document is None:
            return False
        if self._waypoint_session_actor_id is not None:
            # The inspector is disabled while drawing, but keep this guard so
            # programmatic signal delivery cannot collide with the transient
            # document transaction.
            return False
        applied = False
        try:
            mobility = self.mobility_editor.mobility()
            mobility, resources = self._internalize_mobility_resource(mobility)
            actor = self.document.selected_actor
            group = self.document.selected_group
            if actor is not None:
                updated = actor.with_changes(mobility=mobility)
                self.document.replace_actor_with_resources(
                    updated,
                    resources,
                    text=f"Change {updated.name} mobility",
                )
                applied = True
            elif group is not None:
                if isinstance(mobility, GroupMemberMobilitySpec):
                    raise ValueError("A group cannot reference another group.")
                updated_group = group.with_changes(mobility=mobility)
                self.document.replace_group_with_resources(
                    updated_group,
                    resources,
                    text=f"Change {updated_group.name} mobility",
                )
                applied = True
        except (RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid mobility", str(exc))
            return False
        if applied and self._tool is AuthoringTool.PLACE:
            self.set_tool(AuthoringTool.SELECT)
        if applied and self._mobility_draft_pending:
            self._clear_pending_kind("mobility", resync=True)
        return applied

    def _internalize_mobility_resource(
        self,
        mobility: Any,
    ) -> tuple[Any, tuple[AuthoringResource, ...]]:
        field_name: str
        kind: ResourceKind
        raw: str | None
        if isinstance(mobility, NetworkRouteMobilitySpec):
            field_name = "graph_path"
            kind = ResourceKind.NETWORK_GRAPH
            raw = mobility.graph_path
        elif isinstance(mobility, MeshSequenceMobilitySpec):
            field_name = "positions_path"
            kind = ResourceKind.POSITION_SEQUENCE
            raw = mobility.positions_path
        else:
            return mobility, ()
        if not raw:
            return mobility, ()

        scenario_root = (
            self.document.path.parent
            if self.document is not None and self.document.path is not None
            else self._project_root()
        )
        resource = resolve_authoring_resource(
            raw,
            kind,
            scenario_root,
            (self.document.scenario.resources if self.document is not None else ()),
        )
        data = mobility.model_dump(mode="python")
        data[field_name] = resource.relative_path
        updated = type(mobility).model_validate(data)
        return updated, (resource,)

    def _apply_group_settings(self) -> bool:
        document = self.document
        group = document.selected_group if document is not None else None
        if document is None or group is None:
            return False
        try:
            deviation = None
            if self.group_deviation_enabled.isChecked():
                deviation = GroupDeviationSpec(
                    max_right_m=self.group_deviation_spins[0].value(),
                    max_forward_m=self.group_deviation_spins[1].value(),
                    max_up_m=self.group_deviation_spins[2].value(),
                    seed=self.group_deviation_seed.value(),
                )
            document.replace_group(
                group.with_changes(deviation=deviation),
                text=f"Edit {group.name} settings",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid group settings", str(exc))
            return False
        if self._group_settings_draft_pending:
            self._clear_pending_kind("group_settings", resync=True)
        return True

    def _start_waypoint_rebuild(self) -> None:
        """Start a contextual waypoint sequence anchored at the current pose."""

        document = self.document
        actor = document.selected_actor if document is not None else None
        if document is None or actor is None or actor.locked or document.read_only:
            return
        if not self._resolve_pending_inspector_edits():
            return
        if self._waypoint_session_actor_id is not None:
            return
        self._finish_target_transform(commit=False)
        self._finish_mobility_drag(commit=False)
        self._play_timer.stop()
        self.play_button.setText("Play")
        anchor = self._selected_current_position()
        document.begin_transient_edit(f"Draw {actor.name} waypoints")
        self._waypoint_session_actor_id = actor.id
        self._waypoint_base_count = 1
        self._pending_waypoints = [anchor]
        self._waypoint_baseline_actor = actor
        self.set_tool(AuthoringTool.WAYPOINT)
        self._update_waypoint_session_ui()

    def _selected_current_orientation(self) -> tuple[float, float, float]:
        """Return the selected actor's canonical orientation at the visible step."""

        document = self.document
        actor = document.selected_actor if document is not None else None
        if actor is None:
            return (0.0, 0.0, 0.0)
        samples = self._prepared_samples(actor)
        if samples is not None and samples.orientations:
            step = min(self._play_step, len(samples.orientations) - 1)
            return samples.orientations[step]
        orientation = actor.orientation
        if isinstance(orientation, FixedOrientationSpec):
            return orientation.yaw_deg, orientation.pitch_deg, orientation.roll_deg
        if isinstance(orientation, SpinOrientationSpec):
            return orientation.yaw_deg, orientation.pitch_deg, orientation.roll_deg
        if isinstance(orientation, KeyframesOrientationSpec) and orientation.keyframes:
            first = orientation.keyframes[0]
            return first.yaw_deg, first.pitch_deg, first.roll_deg
        return (0.0, 0.0, 0.0)

    def _orientation_type_changed(self, token: str) -> None:
        """Seed an adaptive orientation page from the visible prepared pose."""

        if self._syncing or self.document is None or self.document.selected_actor is None:
            return
        actor = self.document.selected_actor
        orientation = actor.orientation
        if token == orientation_kind(orientation).value:
            self.orientation_editor.set_orientation(orientation)
            return
        target_id = self.orientation_editor.look_at_combo.currentData()
        try:
            converted = convert_orientation(
                orientation,
                token,
                self._selected_current_orientation(),
                self._selected_current_position(),
                duration_s=self.document.scenario.timeline.duration_s,
                target_actor_id=(
                    target_id if token == "look_at" and target_id is not None else None
                ),
                random_seed=actor.id.int & 0x7FFFFFFF,
            )
        except ValueError as exc:
            self.orientation_editor.set_orientation(orientation)
            QMessageBox.warning(self, "Invalid orientation", str(exc))
            return
        self.orientation_editor.set_orientation(converted)
        self._connect_editor_draft_signals(self.orientation_editor, "orientation")
        self._mark_inspector_draft("orientation")

    def _apply_orientation(self) -> bool:
        if self.document is None or self.document.selected_actor is None:
            return False
        try:
            orientation = self.orientation_editor.orientation()
            actor = self.document.selected_actor.with_changes(orientation=orientation)
            self.document.replace_actor(actor, text=f"Change {actor.name} orientation")
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid orientation", str(exc))
            return False
        if self._orientation_draft_pending:
            self._clear_pending_kind("orientation", resync=True)
        return True

    def _apply_target(self) -> bool:
        if self.document is None or self.document.selected_actor is None:
            return False
        actor = self.document.selected_actor
        if actor.role is not ActorRole.TARGET:
            return False
        try:
            changes = self._target_changes_from_controls(actor)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid target", str(exc))
            return False
        self.document.replace_actor(actor.with_changes(**changes), text=f"Edit {actor.name}")
        if self._target_draft_pending:
            self._clear_pending_kind("target", resync=True)
        return True

    def _target_representation_changed(self) -> None:
        self._update_target_asset_controls()
        self._mark_inspector_draft("target")

    def _set_target_editing_enabled(self, actor: AuthoringActor, enabled: bool) -> None:
        """Derive every target control's state from the selected actor."""

        editable = enabled and actor.role is ActorRole.TARGET
        source = actor.target.source if actor.target is not None else "catalog"
        capability = self.document.scenario.capability("target_asset", actor.id)
        self.target_asset_combo.setEnabled(editable and capability.editable)
        self.target_asset_combo.setToolTip(
            capability.reason
            if not capability.editable
            else "Choose a target from the project asset catalog."
        )
        label = self.target_group.layout().labelForField(self.target_asset_combo)
        label.setText("Catalog asset" if source == "catalog" else "Imported asset")
        self.target_material_combo.setEnabled(editable)
        self.target_scale_spin.setEnabled(editable and source != "animated_asset")
        self.target_mesh_animation.setEnabled(editable and source not in {"file", "animated_asset"})
        self.target_representation_editor.setEnabled(editable and source != "animated_asset")

    def _update_target_asset_controls(self) -> None:
        mesh = self.target_representation_editor.representation.currentData() == "mesh_material"
        form = self.target_group.layout()
        for control in (
            self.target_asset_combo,
            self.target_material_combo,
            self.target_scale_spin,
            self.target_mesh_animation,
        ):
            control.setVisible(mesh)
            label = form.labelForField(control)
            if label is not None:
                label.setVisible(mesh)

    def _target_changes_from_controls(self, actor: AuthoringActor) -> dict[str, Any]:
        changes = self.target_representation_editor.actor_changes()
        if changes["representation"] == "mesh_material":
            changes["target"] = self._target_from_controls(actor)
        return changes

    def _target_from_controls(self, actor: AuthoringActor) -> TargetAsset | None:
        """Apply editable appearance without replacing an imported locator."""

        current = actor.target
        material = str(self.target_material_combo.currentData() or "glass")
        scale = self.target_scale_spin.value()
        mesh_animation = self.target_mesh_animation.isChecked()
        if current is not None and current.source != "catalog":
            if current.source == "animated_asset":
                return replace(current, material=material)
            return replace(
                current,
                material=material,
                scale=scale,
                mesh_animation=(
                    current.mesh_animation if current.source == "file" else mesh_animation
                ),
            )
        asset_id = str(self.target_asset_combo.currentData() or "").strip()
        return (
            None
            if not asset_id
            else TargetAsset.from_catalog_id(
                asset_id,
                material=material,
                scale=scale,
                mesh_animation=mesh_animation,
                mesh_end_behavior=current.mesh_end_behavior if current is not None else "loop",
            )
        )

    def _refresh_inline_errors(self) -> None:
        if self.document is None or not self._compilation_is_current():
            self.inline_errors.clear()
            return
        if self.document.selected_actor_id is not None:
            issues = self.compilation.issues_for_actor(self.document.selected_actor_id)
        elif self.document.selected_group_id is not None:
            issues = self.compilation.issues_for_group(self.document.selected_group_id)
        else:
            issues = ()
        self.inline_errors.setText("\n".join(f"{issue.path}: {issue.message}" for issue in issues))

    # -- timeline, scene, and viewport commands ------------------------

    def _apply_timeline(self) -> bool:
        if self.document is None or self.document.read_only:
            return False
        try:
            timeline, replace_imported_quality = self._timeline_editor_state()
            self.document.set_timeline(
                timeline,
                replace_source_quality=replace_imported_quality,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid timeline", str(exc))
            return False
        self._timeline_draft_pending = False
        self._sync_document_controls()
        return True

    def set_scene(self, source: str, scene_id: str) -> None:
        if self.document is not None:
            self.document.set_scene(SceneReference(source.strip(), scene_id.strip()))

    def _project_root(self) -> Path:
        return Path(getattr(self.compiler, "project_root", Path.cwd())).resolve()

    def _catalog_target_ids(self) -> tuple[str, ...]:
        target_root = self._project_root() / "libraries" / "targets"
        try:
            return tuple(sorted(path.name for path in target_root.iterdir() if path.is_dir()))
        except OSError:
            return ()

    def _populate_target_asset_choices(self) -> None:
        """Rebuild catalog choices without carrying another actor's imported locator."""
        self.target_asset_combo.clear()
        assets = self._catalog_target_ids()
        self.target_asset_combo.addItem(
            "Select catalog asset..." if assets else "No catalog assets found", ""
        )
        for asset_id in assets:
            self.target_asset_combo.addItem(asset_id, asset_id)

    def _scene_suggestions(self, source: str) -> tuple[str, ...]:
        if source == "library":
            scene_root = self._project_root() / "libraries" / "scenes"
            try:
                return tuple(
                    sorted(
                        path.relative_to(scene_root).as_posix()
                        for path in scene_root.rglob("*.xml")
                        if path.is_file()
                    )
                )
            except OSError:
                return ()
        if source == "sionna":
            return available_sionna_scene_ids()
        return ()

    def _populate_scene_ids(self, selected: str | None = None) -> None:
        source = str(self.scene_source_combo.currentData() or "sionna")
        placeholder = {
            "sionna": "Choose a Sionna built-in scene…",
            "library": "Choose an ORCHAV library scene…",
            "local": "Browse for a local XML scene…",
        }.get(source, "Choose a scene…")
        current = selected
        if current is None:
            current = self.scene_id_combo.currentText().strip()
        self.scene_id_combo.blockSignals(True)
        try:
            self.scene_id_combo.clear()
            self.scene_id_combo.addItems(self._scene_suggestions(source))
            if current:
                index = self.scene_id_combo.findText(current)
                if index < 0:
                    self.scene_id_combo.addItem(current)
                    index = self.scene_id_combo.count() - 1
                self.scene_id_combo.setCurrentIndex(index)
            else:
                self.scene_id_combo.setCurrentIndex(-1)
                self.scene_id_combo.setEditText("")
            line_edit = self.scene_id_combo.lineEdit()
            if line_edit is not None:
                line_edit.setPlaceholderText(placeholder)
        finally:
            self.scene_id_combo.blockSignals(False)
        self.scene_browse_button.setVisible(source == "local")

    def _scene_source_changed(self) -> None:
        source = str(self.scene_source_combo.currentData() or "")
        osm = source == "osm"
        self.osm_panel.setVisible(osm)
        self._osm_scroll.setVisible(osm)
        self.scene_id_combo.setVisible(not osm)
        self._scene_id_label.setVisible(not osm)
        self._scene_buttons.setVisible(not osm)
        preserved_extension = source not in {"sionna", "library", "local"}
        self.scene_id_combo.setEnabled(not preserved_extension)
        self.scene_apply_button.setEnabled(not preserved_extension)
        if not self._syncing:
            self._populate_scene_ids("")

    def _osm_view_changed(self) -> None:
        self._refresh_viewport()
        if self.document is not None:
            self._set_editing_enabled(not self.document.read_only)

    def _enter_osm_map(self, bounds) -> None:
        self._finish_target_transform(commit=False)
        self._finish_mobility_drag(commit=False)
        port = getattr(self.viewport, "port", None)
        if port is not None:
            port.set_tool(AuthoringTool.OSM_REGION)
            show = getattr(port, "show_ground_map", None)
            if callable(show):
                show(bounds)

    def _leave_osm_map(self) -> None:
        port = getattr(getattr(self, "viewport", None), "port", None)
        if port is not None:
            leave = getattr(port, "leave_ground_map", None)
            if callable(leave):
                leave()
            port.set_tool(self._tool)
        if self.document is not None and not self.osm_panel.building:
            self._sync_document_controls()

    def _osm_build_started(self) -> None:
        document = self.document
        self._osm_build_context = (
            (
                document,
                document.scenario.scene,
                document.scenario.source_snapshot.to_mapping().get("scene"),
            )
            if document is not None
            else None
        )

    def _install_osm_scene(self, config, scene_root) -> None:
        context = self._osm_build_context
        self._osm_build_context = None
        document = self.document
        if (
            context is None
            or document is not context[0]
            or document.read_only
            or document.scenario.scene != context[1]
            or document.scenario.source_snapshot.to_mapping().get("scene") != context[2]
        ):
            self.osm_panel.status.setText(
                "The active scene changed during the build. The result was not applied."
            )
            return
        try:
            self._osm_fit_pending = True
            document.install_osm_scene(config, scene_root, project_root=self._project_root())
        except (OSError, ValueError, PermissionError) as exc:
            self._osm_fit_pending = False
            self.osm_panel.status.setText(f"Could not apply scene: {exc}")
            return
        self.osm_panel.leave_map()

    def _browse_local_scene(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Local Scene XML",
            str(Path.cwd()),
            "Mitsuba Scene XML (*.xml)",
        )
        if path:
            self.scene_source_combo.setCurrentIndex(self.scene_source_combo.findData("local"))
            self._populate_scene_ids(path)

    def _apply_scene(self) -> None:
        if self.document is None or self.document.read_only:
            return
        source = str(self.scene_source_combo.currentData() or "").strip()
        scene_id = self.scene_id_combo.currentText().strip()
        self.document.set_scene(SceneReference(source, scene_id))

    def _viewport_settings_changed(self, *_args: Any) -> None:
        self.grid_snap_spacing_spin.setEnabled(self.grid_snap_check.isChecked())
        plane_state = (
            f"Work plane Z={self.work_plane_z_spin.value():g} m"
            if self.work_plane_check.isChecked()
            else "Work plane hidden"
        )
        snap_state = (
            f"Snap {self.grid_snap_spacing_spin.value():g} m"
            if self.grid_snap_check.isChecked()
            else "Snap off"
        )
        self.coordinate_label.setText(f"X: --   Y: --   Z: --   | {plane_state} | {snap_state}")
        self._refresh_viewport()

    def _camera_mode_changed(self, *_args: Any) -> None:
        if self.camera_controls is not None:
            self.camera_controls.ui_controller.handle_fly_mode_toggled(
                self.camera_mode_combo.currentData() == "fly"
            )
            return
        setter = getattr(getattr(self.viewport, "port", None), "set_camera_mode", None)
        if callable(setter):
            setter(str(self.camera_mode_combo.currentData()))

    def set_tool(self, tool: AuthoringTool) -> None:
        """Activate a tool while preserving explicit waypoint transactions."""

        next_tool = AuthoringTool(tool)
        if next_tool is AuthoringTool.MOVE and not self._tool_actions[next_tool].isEnabled():
            return
        if self._waypoint_session_actor_id is not None and next_tool is not AuthoringTool.WAYPOINT:
            # A tool change must never decide the fate of a pending waypoint
            # transaction. Finish and Cancel are explicit session commands.
            for candidate, action in self._tool_actions.items():
                action.setChecked(candidate is self._tool)
            return
        if self._tool is AuthoringTool.MOVE and next_tool is not AuthoringTool.MOVE:
            self._finish_mobility_drag(commit=False)
            self._dependent_drag_preview_samples.clear()
            self._finish_target_transform(commit=False)
            self._clear_motion_hover(refresh=False)
        if next_tool in {AuthoringTool.PLACE, AuthoringTool.MOVE, AuthoringTool.WAYPOINT}:
            self._stop_playback(refresh=False)
        if next_tool not in {AuthoringTool.PLACE, AuthoringTool.WAYPOINT}:
            self._placement_ghost = None
        self._tool = next_tool
        for candidate, action in self._tool_actions.items():
            action.setChecked(candidate is self._tool)
        port = getattr(self.viewport, "port", None)
        if port is not None:
            port.set_tool(self._tool)
        self._update_tool_hint()
        self._refresh_viewport()

    def _update_tool_hint(self) -> None:
        """Explain the active tool for the actual selection, including restrictions."""
        document = self.document
        subject = (document.selected_actor or document.selected_group) if document else None
        if self._tool is AuthoringTool.PLACE:
            hint = "Click once to place the new actor"
        elif self._tool is AuthoringTool.WAYPOINT:
            hint = "Click to add points; Enter finishes; Esc cancels"
        elif subject is None:
            hint = "Select an actor in the list or viewport"
        elif document.scenario.recorded_poses:
            hint = f"{subject.name} · Recorded motion"
        elif document.read_only:
            hint = f"{subject.name} · Read-only document"
        elif subject.locked:
            hint = f"{subject.name} · Unlock to edit motion"
        elif (
            isinstance(subject, AuthoringActor)
            and subject.target is not None
            and subject.target.source == "animated_asset"
        ):
            hint = f"{subject.name} · Animation owns motion"
        elif self._tool is AuthoringTool.SELECT or not self._mobility_translation_supported(
            subject.mobility
        ):
            hint = f"{subject.name} · Edit Mobility in the inspector"
        else:
            hint = f"{subject.name} · Edit handles, path or XYZ gizmo"
        self.tool_hint_label.setText(hint)

    def _sync_motion_edit_action(self) -> None:
        """Keep the inspector's viewport action consistent with motion capabilities."""
        document = self.document
        subject = (document.selected_actor or document.selected_group) if document else None
        if subject is not None and self._mobility_draft_pending:
            try:
                subject = subject.with_changes(mobility=self.mobility_editor.mobility())
            except (TypeError, ValueError):
                pass
        reason = ""
        if subject is None:
            reason = "Select an actor to edit its motion."
        elif document.read_only:
            reason = "This document is read-only."
        elif subject.locked:
            reason = "Unlock this actor in the list to edit its motion."
        else:
            facet = "mobility" if isinstance(subject, AuthoringActor) else "group"
            capability = document.scenario.capability(facet, subject.id)
            if not capability.editable:
                reason = capability.reason
            elif not self._mobility_translation_supported(subject.mobility):
                reason = (
                    "Edit route endpoints and speed in Mobility below; this route cannot be dragged."
                    if isinstance(subject.mobility, NetworkRouteMobilitySpec)
                    else "This motion has no viewport handles. Use the Mobility settings below."
                )
        busy = (
            self._waypoint_session_actor_id is not None
            or self.osm_panel.active
            or self.osm_panel.building
        )
        action = self._tool_actions[AuthoringTool.MOVE]
        action.setEnabled(not reason and not busy)
        action.setToolTip(reason or "Edit motion using handles, path dragging or the XYZ gizmo.")
        self.motion_edit_hint.setText(reason)
        self.motion_edit_hint.setVisible(bool(reason))
        if self._tool is AuthoringTool.MOVE and not action.isEnabled():
            self.set_tool(AuthoringTool.SELECT)

    def _on_viewport_input(self, value: Any) -> None:
        port = getattr(self.viewport, "port", None)
        if port is not None and self.osm_panel.handle_input(value, port):
            return
        if isinstance(value, TransformInput):
            self._handle_target_transform(value)
            return
        if isinstance(value, PointerInput):
            self._last_pointer_phase = value.phase
            if value.phase is PointerPhase.LEAVE:
                self._motion_hover_clear_pending = False
                changed = self._placement_ghost is not None or self._hovered_component is not None
                self._placement_ghost = None
                self._clear_motion_hover(refresh=False)
                if changed:
                    self._refresh_viewport()
            elif value.phase is PointerPhase.MOVE and self._tool is AuthoringTool.MOVE:
                self._motion_hover_clear_pending = True
                QTimer.singleShot(0, self._clear_unresolved_motion_hover)
            return
        if isinstance(value, KeyboardInput):
            self._handle_key(value)
            return
        if not isinstance(value, HitResult):
            return
        self._motion_hover_clear_pending = False
        x, y, z = value.world_position
        plane_state = (
            f"Work plane Z={self.work_plane_z_spin.value():g} m"
            if self.work_plane_check.isChecked()
            else "Work plane hidden"
        )
        snap_state = (
            f"Snap {self.grid_snap_spacing_spin.value():g} m"
            if self.grid_snap_check.isChecked()
            else "Snap off"
        )
        self.coordinate_label.setText(
            f"X: {x:.3f}   Y: {y:.3f}   Z: {z:.3f}   | {plane_state} | {snap_state}"
        )
        pointer_phase = self._last_pointer_phase
        if self.document is None:
            return
        if pointer_phase is PointerPhase.MOVE and self._tool in {
            AuthoringTool.PLACE,
            AuthoringTool.WAYPOINT,
        }:
            actor = self.document.selected_actor
            next_ghost = (
                value.world_position
                if actor is not None and not actor.locked and not self._read_only
                else None
            )
            if next_ghost != self._placement_ghost:
                self._placement_ghost = next_ghost
                self._refresh_viewport()
            self._last_pointer_phase = None
            return
        if self._tool is AuthoringTool.MOVE:
            if pointer_phase is PointerPhase.MOVE:
                self._update_motion_hover(value)
            self._handle_mobility_drag(value, pointer_phase)
            self._last_pointer_phase = None
            return
        if pointer_phase not in {PointerPhase.DOWN, PointerPhase.DOUBLE_CLICK}:
            return
        self._last_pointer_phase = None
        if self._tool is AuthoringTool.SELECT and value.actor_id is not None:
            current = self._selected_subject()
            next_subject = (
                value.actor_id,
                self.document.scenario.group(value.actor_id) is not None,
            )
            if (
                self.has_pending_inspector_edits
                and next_subject != current
                and not self._resolve_pending_inspector_edits()
            ):
                return
            if self.document.scenario.group(value.actor_id) is not None:
                self.document.select_group(value.actor_id)
            else:
                self.document.select(value.actor_id)
            return
        actor = self.document.selected_actor
        if actor is None or actor.locked or self._read_only:
            return
        if self._tool is AuthoringTool.WAYPOINT:
            self._begin_waypoint_session(actor)
            actor = self.document.scenario.actor(actor.id)
            if actor is None:
                return
            points = tuple(self._pending_waypoints)
            appended_points = (*points, value.world_position)
            if (
                pointer_phase is PointerPhase.DOUBLE_CLICK
                and points
                and math.dist(points[-1], value.world_position) <= 1e-6
            ):
                # Native double-click streams commonly include the second
                # pointer-down before the double-click event. Finish without
                # duplicating that final point.
                appended_points = points
            self._pending_waypoints = list(appended_points)
            if len(appended_points) >= 2:
                self.document.update_transient_actor(
                    actor.with_changes(
                        mobility=self._waypoint_mobility_with_points(
                            actor.mobility,
                            appended_points,
                        )
                    ),
                )
            if pointer_phase is PointerPhase.DOUBLE_CLICK:
                self._finish_waypoint_session()
        elif self._tool is AuthoringTool.PLACE:
            self.document.replace_actor(
                actor.with_changes(
                    mobility=StationaryMobilitySpec(position_m=value.world_position)
                ),
                text=f"Place {actor.name}",
            )
            self.set_tool(AuthoringTool.SELECT)

    def _handle_key(self, event: KeyboardInput) -> None:
        if not event.pressed:
            return
        key = event.key.lower()
        if key in {"escape", "esc"}:
            if self._waypoint_session_actor_id is not None:
                self._dispatch_waypoint_shortcut("cancel")
                return
            self._finish_target_transform(commit=False)
            self._finish_mobility_drag(commit=False)
            self.set_tool(AuthoringTool.SELECT)
        elif key in {"enter", "return"} and self._waypoint_session_actor_id is not None:
            self._dispatch_waypoint_shortcut("finish")
        elif key == "backspace" and self._waypoint_session_actor_id is not None:
            self._dispatch_waypoint_shortcut("remove")

    def _dispatch_waypoint_shortcut(self, command: str) -> None:
        """Deliver one key command once when Qt and the viewport both report it."""

        normalized = str(command)
        if normalized in self._waypoint_key_latch:
            return
        self._waypoint_key_latch.add(normalized)
        QTimer.singleShot(0, lambda: self._waypoint_key_latch.discard(normalized))
        if normalized == "finish":
            self._finish_waypoint_session()
        elif normalized == "cancel":
            self._cancel_waypoint_session_and_exit()
        elif normalized == "remove":
            self._remove_last_waypoint()

    def _begin_waypoint_session(self, actor: Any) -> None:
        """Start one pending sequence whose many clicks become one undo entry."""

        if self.document is None or self._waypoint_session_actor_id is not None:
            return
        self._play_timer.stop()
        self.play_button.setText("Play")
        self.document.begin_transient_edit(f"Edit {actor.name} waypoints")
        self._waypoint_session_actor_id = actor.id
        self._waypoint_baseline_actor = actor
        self._pending_waypoints = (
            list(actor.mobility.points_m)
            if isinstance(actor.mobility, WaypointMobilitySpec)
            else []
        )
        self._waypoint_base_count = len(self._pending_waypoints)
        self._update_waypoint_session_ui()

    @staticmethod
    def _waypoint_mobility_with_points(
        mobility: Any,
        points: tuple[tuple[float, float, float], ...],
    ) -> WaypointMobilitySpec:
        """Replace waypoint geometry without resetting interpolation or traversal."""

        if isinstance(mobility, WaypointMobilitySpec):
            return WaypointMobilitySpec(
                points_m=points,
                interpolation=mobility.interpolation,
                traversal=mobility.traversal,
            )
        return WaypointMobilitySpec(points_m=points)

    def _commit_waypoint_session(self) -> bool:
        if self.document is None or self._waypoint_session_actor_id is None:
            return False
        actor = self.document.scenario.actor(self._waypoint_session_actor_id)
        if actor is None or len(self._pending_waypoints) < 2:
            return False
        if not isinstance(actor.mobility, WaypointMobilitySpec):
            self.document.update_transient_actor(
                actor.with_changes(
                    mobility=self._waypoint_mobility_with_points(
                        actor.mobility,
                        tuple(self._pending_waypoints),
                    )
                )
            )
        self.document.commit_transient_edit()
        self._waypoint_session_actor_id = None
        self._waypoint_base_count = 0
        self._pending_waypoints = []
        self._waypoint_baseline_actor = None
        self._placement_ghost = None
        self._update_waypoint_session_ui()
        return True

    def _cancel_waypoint_session(self) -> bool:
        if self.document is not None and self._waypoint_session_actor_id is not None:
            self.document.cancel_transient_edit()
        active = self._waypoint_session_actor_id is not None
        self._waypoint_session_actor_id = None
        self._waypoint_base_count = 0
        self._pending_waypoints = []
        self._waypoint_baseline_actor = None
        self._placement_ghost = None
        self._update_waypoint_session_ui()
        return active

    def _finish_waypoint_session(self) -> None:
        """Commit the visible drawing task and return to ordinary selection."""

        if self._commit_waypoint_session():
            self.set_tool(AuthoringTool.SELECT)

    def _cancel_waypoint_session_and_exit(self) -> None:
        """Restore the pre-drawing document exactly and leave drawing mode."""

        if self._cancel_waypoint_session():
            self.set_tool(AuthoringTool.SELECT)

    def _remove_last_waypoint(self) -> None:
        """Remove only a point added by the current drawing session."""

        document = self.document
        actor_id = self._waypoint_session_actor_id
        if document is None or actor_id is None:
            return
        actor = document.scenario.actor(actor_id)
        if actor is None or len(self._pending_waypoints) <= self._waypoint_base_count:
            return
        self._pending_waypoints.pop()
        if len(self._pending_waypoints) >= 2:
            document.update_transient_actor(
                actor.with_changes(
                    mobility=self._waypoint_mobility_with_points(
                        actor.mobility,
                        tuple(self._pending_waypoints),
                    )
                ),
            )
        elif self._waypoint_baseline_actor is not None:
            document.update_transient_actor(self._waypoint_baseline_actor)
        self._refresh_viewport()

    def _update_waypoint_session_ui(self) -> None:
        """Synchronize the modal drawing bar, shortcuts, and blocked navigation."""

        actor_id = self._waypoint_session_actor_id
        document = self.document
        actor = document.scenario.actor(actor_id) if document is not None and actor_id else None
        active = actor is not None
        point_count = len(self._pending_waypoints) if active else 0
        name = actor.name if actor is not None else "actor"
        point_word = "point" if point_count == 1 else "points"
        finish_hint = "" if point_count >= 2 else " · Add another point to finish"
        self.waypoint_session_label.setText(
            f"Drawing waypoints for {name} · {point_count} {point_word} · "
            f"Click to add; Enter finishes; Esc cancels{finish_hint}"
        )
        removable = active and point_count > self._waypoint_base_count
        can_finish = active and point_count >= 2
        self.waypoint_remove_last_button.setEnabled(removable)
        self.waypoint_finish_button.setEnabled(can_finish)
        self.waypoint_finish_button.setToolTip(
            "" if can_finish else "Waypoint mobility requires at least two points."
        )
        self.waypoint_cancel_button.setEnabled(active)
        self.waypoint_session_bar.setVisible(active)
        for action in getattr(self, "_waypoint_shortcut_actions", ()):
            command = action.data()
            enabled = active
            if command == "remove":
                enabled = removable
            elif command == "finish":
                enabled = can_finish
            action.setEnabled(enabled)
        self._set_editing_enabled(bool(document is not None and not document.read_only))
        refresh_actions = getattr(self.visualizer, "_refresh_authoring_actions", None)
        if callable(refresh_actions):
            refresh_actions()

    def _update_motion_hover(self, hit: HitResult) -> None:
        """Highlight the easiest drag target and explain its semantic action."""

        document = self.document
        component = hit.component or ""
        actor = document.scenario.actor(hit.actor_id) if document and hit.actor_id else None
        group = document.scenario.group(hit.actor_id) if document and hit.actor_id else None
        if self._candidate_scenario is not None and hit.actor_id is not None:
            actor = self._candidate_scenario.actor(hit.actor_id) or actor
            group = self._candidate_scenario.group(hit.actor_id) or group
        subject = actor or group
        next_actor_id: UUID | None = None
        next_component: str | None = None
        hint = "Edit handles, drag the highlighted trajectory, or use the XYZ gizmo"
        cursor = Qt.CursorShape.ArrowCursor

        if subject is not None and document is not None and document.read_only:
            hint = "This authoring document is read-only"
            cursor = Qt.CursorShape.ForbiddenCursor
        elif subject is not None and subject.locked:
            hint = f"{subject.name} is locked"
            cursor = Qt.CursorShape.ForbiddenCursor
        elif subject is not None and component.startswith("mobility_control_"):
            control_key = component.removeprefix("mobility_control_")
            try:
                control = mobility_control_rig(subject.mobility).control(control_key)
            except (KeyError, TypeError, ValueError):
                control = None
            if control is not None:
                next_actor_id = subject.id
                next_component = component
                hint = f"{control.label}: {control.tooltip}"
                cursor = Qt.CursorShape.OpenHandCursor
        elif subject is not None and component == "trajectory_hit":
            if self._mobility_translation_supported(subject.mobility):
                next_actor_id = subject.id
                next_component = component
                hint = "Drag the highlighted trajectory to translate the entire motion"
                cursor = Qt.CursorShape.SizeAllCursor
            else:
                hint = (
                    "This resource-backed trajectory is positioned by its source data "
                    "and cannot be dragged"
                )
                cursor = Qt.CursorShape.ForbiddenCursor
        elif subject is not None and component == "mobility_handles":
            if self._mobility_translation_supported(subject.mobility):
                next_actor_id = subject.id
                next_component = component
                hint = "Drag the device marker or XYZ gizmo to translate the entire motion"
                cursor = Qt.CursorShape.SizeAllCursor
            else:
                hint = (
                    "This resource-backed trajectory is positioned by its source data "
                    "and cannot be dragged"
                )
                cursor = Qt.CursorShape.ForbiddenCursor

        changed = (
            next_actor_id != self._hovered_actor_id or next_component != self._hovered_component
        )
        self._hovered_actor_id = next_actor_id
        self._hovered_component = next_component
        self.tool_hint_label.setText(hint)
        self.viewport.setCursor(cursor)
        if changed:
            self._refresh_viewport()

    def _clear_unresolved_motion_hover(self) -> None:
        """Clear hover when a pointer move produced no typed hit result."""

        if not self._motion_hover_clear_pending:
            return
        self._motion_hover_clear_pending = False
        self._clear_motion_hover(refresh=True)

    def _clear_motion_hover(self, *, refresh: bool) -> None:
        changed = self._hovered_actor_id is not None or self._hovered_component is not None
        self._hovered_actor_id = None
        self._hovered_component = None
        viewport = getattr(self, "viewport", None)
        if viewport is not None:
            viewport.unsetCursor()
        if self._tool is AuthoringTool.MOVE and hasattr(self, "tool_hint_label"):
            self._update_tool_hint()
        if changed and refresh:
            self._refresh_viewport()

    def _translate_subject_mobility(
        self,
        subject: AuthoringActor | AuthoringGroup,
        world_offset: tuple[float, float, float],
        *,
        reference_position: tuple[float, float, float],
    ) -> Any:
        """Translate a mobility or a group member's canonical local offset."""

        mobility = subject.mobility
        if not (
            isinstance(subject, AuthoringActor) and isinstance(mobility, GroupMemberMobilitySpec)
        ):
            return translate_mobility(mobility, world_offset)

        document = self.document
        if document is None:
            raise RuntimeError("group-member translation requires an active document")
        group = document.scenario.group(mobility.group)
        if group is None:
            raise ValueError(f"Group {mobility.group!r} does not exist.")

        moved_position = tuple(
            float(value + delta)
            for value, delta in zip(reference_position, world_offset, strict=True)
        )
        step = min(self._play_step, max(document.scenario.timeline.steps - 1, 0))
        prepared_group = self._prepared_group_samples(group)
        if prepared_group is not None:
            projected = prepared_group.offsets_for_world_positions(
                (reference_position, moved_position),
                step=step,
            )
        else:
            projected = ScenarioCompiler.group_offsets(
                group.mobility,
                document.scenario.timeline.steps,
                document.scenario.timeline.duration_s,
                (reference_position, moved_position),
                step=step,
                scenario_directory=self._current_scenario_directory(),
                resources=document.scenario.resources,
            )
        local_delta = tuple(
            moved - original for original, moved in zip(projected[0], projected[1], strict=True)
        )
        offset = mobility.offset_m
        return GroupMemberMobilitySpec(
            group=mobility.group,
            offset_m=GroupOffsetSpec(
                right=offset.right + local_delta[0],
                forward=offset.forward + local_delta[1],
                up=offset.up + local_delta[2],
            ),
        )

    def _handle_mobility_drag(
        self,
        hit: HitResult,
        phase: PointerPhase | None,
    ) -> None:
        """Apply one semantic handle or whole-trajectory drag."""

        document = self.document
        if document is None or phase is None:
            return
        if phase is PointerPhase.DOWN:
            if hit.actor_id is None:
                return
            self._finish_mobility_drag(commit=False)
            scenario = self._begin_motion_draft(hit.actor_id)
            if scenario is None:
                return
            actor = scenario.actor(hit.actor_id)
            group = scenario.group(hit.actor_id)
            subject = actor or group
            if subject is None or subject.locked or document.read_only:
                return
            component = hit.component or ""
            control: MobilityControlDescriptor | None = None
            body_drag = component in {
                "trajectory_hit",
                "mobility_handles",
                "path",
                "points",
            }
            if component.startswith("mobility_control_"):
                control_key = component.removeprefix("mobility_control_")
                try:
                    control = mobility_control_rig(subject.mobility).control(control_key)
                except (KeyError, TypeError, ValueError):
                    control = None
            if body_drag and not self._mobility_translation_supported(subject.mobility):
                if group is not None:
                    document.select_group(group.id)
                else:
                    document.select(subject.id)
                self.tool_hint_label.setText(
                    "This resource-backed trajectory is positioned by its source data "
                    "and cannot be dragged"
                )
                return
            if control is None and not body_drag:
                if group is not None:
                    document.select_group(group.id)
                else:
                    document.select(subject.id)
                return

            self._stop_playback(refresh=False)
            baseline_samples = self._draft_samples(actor) if actor is not None else None
            self._dependent_drag_preview_samples.clear()
            baseline_group_positions = (
                self._prepared_group_positions(group) if group is not None else None
            )
            baseline_group_physical = (
                self._prepared_group_has_physical_velocity(group) if group is not None else None
            )
            baseline_group_samples = (
                self._prepared_group_samples(group) if group is not None else None
            )
            if group is not None and self._candidate_compilation is not None:
                candidate_samples = self._candidate_compilation.group_samples.get(group.id)
                if candidate_samples is not None:
                    baseline_group_samples = candidate_samples
                    baseline_group_positions = candidate_samples.positions
                    baseline_group_physical = candidate_samples.has_physical_velocity
            baseline_group_member_samples: dict[UUID, ActorSamples] = {}
            if group is not None:
                for candidate in document.actors:
                    if not isinstance(candidate.mobility, GroupMemberMobilitySpec):
                        continue
                    try:
                        belongs_to_group = UUID(str(candidate.mobility.group)) == group.id
                    except ValueError:
                        belongs_to_group = False
                    if not belongs_to_group:
                        continue
                    candidate_samples = self._draft_samples(candidate)
                    if candidate_samples is not None:
                        baseline_group_member_samples[candidate.id] = candidate_samples
            if group is not None:
                document.select_group(group.id)
            else:
                document.select(subject.id)
            self._mobility_drag_actor_id = subject.id
            self._mobility_drag_control = control
            self._mobility_drag_baseline_actor = subject
            self._mobility_drag_origin = hit.world_position
            self._mobility_drag_baseline_position = hit.world_position
            self._mobility_drag_baseline_samples = baseline_samples
            self._mobility_drag_preview_samples = (
                baseline_samples
                if control is None
                or not isinstance(
                    subject.mobility,
                    (
                        StationaryMobilitySpec,
                        LinearMobilitySpec,
                        WaypointMobilitySpec,
                    ),
                )
                else None
            )
            self._mobility_drag_baseline_group_positions = baseline_group_positions
            self._mobility_drag_preview_group_positions = (
                baseline_group_positions
                if control is None
                or not isinstance(
                    subject.mobility,
                    (
                        StationaryMobilitySpec,
                        LinearMobilitySpec,
                        WaypointMobilitySpec,
                    ),
                )
                else None
            )
            self._mobility_drag_baseline_group_physical = baseline_group_physical
            self._mobility_drag_baseline_group_member_samples = baseline_group_member_samples
            self._mobility_drag_baseline_group_samples = baseline_group_samples
            self._mobility_drag_preview_group_samples = baseline_group_samples
            port = getattr(self.viewport, "port", None)
            begin_control_drag = getattr(port, "begin_control_drag", None)
            begin_drag = getattr(port, "begin_drag_plane", None)
            if control is not None and callable(begin_control_drag):
                try:
                    begin_control_drag(control.constraint.value, hit)
                except (RuntimeError, TypeError, ValueError):
                    self._finish_mobility_drag(commit=False)
                return
            if not callable(begin_drag):
                self._finish_mobility_drag(commit=False)
                return
            try:
                drag_z = control.position[2] if control is not None else hit.world_position[2]
                begin_drag(drag_z, hit)
            except (RuntimeError, TypeError, ValueError):
                self._finish_mobility_drag(commit=False)
            return

        if (
            phase not in {PointerPhase.MOVE, PointerPhase.UP}
            or self._mobility_drag_actor_id is None
            or self._mobility_drag_baseline_actor is None
            or self._mobility_drag_origin is None
            or self._mobility_drag_baseline_position is None
            or hit.actor_id != self._mobility_drag_actor_id
        ):
            return
        baseline = self._mobility_drag_baseline_actor
        try:
            translation: tuple[float, float, float] | None = None
            if self._mobility_drag_control is not None:
                mobility = update_mobility_from_rig_control(
                    baseline.mobility,
                    self._mobility_drag_control,
                    hit.world_position,
                )
            else:
                translation = tuple(
                    float(value - origin)
                    for value, origin in zip(
                        hit.world_position,
                        self._mobility_drag_origin,
                        strict=True,
                    )
                )
                mobility = self._translate_subject_mobility(
                    baseline,
                    translation,
                    reference_position=self._mobility_drag_baseline_position,
                )
            updated = baseline.with_changes(mobility=mobility)
            if isinstance(updated, AuthoringGroup):
                baseline_group_positions = self._mobility_drag_baseline_group_positions
                if baseline_group_positions is not None and translation is not None:
                    self._mobility_drag_preview_group_positions = self._translated_positions(
                        baseline_group_positions,
                        translation,
                    )
                    baseline_group_samples = self._mobility_drag_baseline_group_samples
                    if baseline_group_samples is not None:
                        translated_group_mobility = PreparedMobility(
                            positions_m=self._mobility_drag_preview_group_positions,
                            velocities_mps=(
                                baseline_group_samples.prepared_mobility.velocities_mps
                            ),
                            forward_vectors=(
                                baseline_group_samples.prepared_mobility.forward_vectors
                            ),
                            has_physical_velocity=(baseline_group_samples.has_physical_velocity),
                        )
                        self._mobility_drag_preview_group_samples = GroupSamples(
                            self._mobility_drag_preview_group_positions,
                            translated_group_mobility,
                            baseline_group_samples.timeline,
                        )
                member_previews = self._mobility_drag_baseline_group_member_samples
                if translation is not None:
                    member_previews = {
                        actor_id: self._rigid_preview_samples(
                            samples,
                            translation=translation,
                        )
                        for actor_id, samples in member_previews.items()
                    }
                self._dependent_drag_preview_samples.clear()
                candidate_scenario = self._motion_draft_baseline.scenario.replace_group(updated)
                if translation is None:
                    try:
                        (
                            transient_actor_samples,
                            transient_group_samples,
                        ) = self._prepare_transient_pose_samples(candidate_scenario)
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        transient_actor_samples = member_previews
                    else:
                        prepared_group = transient_group_samples.get(updated.id)
                        if prepared_group is not None:
                            self._mobility_drag_preview_group_samples = prepared_group
                            self._mobility_drag_preview_group_positions = prepared_group.positions
                            self._mobility_drag_baseline_group_physical = (
                                prepared_group.has_physical_velocity
                            )
                    self._dependent_drag_preview_samples = transient_actor_samples
                else:
                    self._dependent_drag_preview_samples = self._look_at_drag_previews_for_samples(
                        candidate_scenario,
                        member_previews,
                    )
            else:
                baseline_samples = self._mobility_drag_baseline_samples
                if baseline_samples is not None and translation is not None:
                    self._mobility_drag_preview_samples = self._rigid_preview_samples(
                        baseline_samples,
                        translation=translation,
                    )
                self._dependent_drag_preview_samples.clear()
                candidate_scenario = self._motion_draft_baseline.scenario.replace_actor(updated)
                if translation is None:
                    try:
                        transient_actor_samples, _groups = self._prepare_transient_pose_samples(
                            candidate_scenario
                        )
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        transient_actor_samples = {}
                    self._mobility_drag_preview_samples = transient_actor_samples.get(updated.id)
                    self._dependent_drag_preview_samples = transient_actor_samples
                else:
                    self._dependent_drag_preview_samples = self._look_at_drag_previews(
                        candidate_scenario,
                        updated.id,
                        self._mobility_drag_preview_samples,
                    )
            self._stage_motion_draft(updated)
        except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
            self._finish_mobility_drag(commit=False)
            return
        if phase is PointerPhase.UP:
            self._finish_mobility_drag(commit=True)

    def _finish_mobility_drag(self, *, commit: bool) -> None:
        """End the handle gesture, retaining accepted changes until Apply."""

        active = self._mobility_drag_actor_id is not None
        if active:
            self._finish_motion_draft(accept=commit)
        port = getattr(getattr(self, "viewport", None), "port", None)
        end_drag = getattr(port, "end_drag", None)
        if callable(end_drag):
            end_drag()
        self._mobility_drag_actor_id = None
        self._mobility_drag_control = None
        self._mobility_drag_baseline_actor = None
        self._mobility_drag_origin = None
        self._mobility_drag_baseline_position = None
        self._mobility_drag_baseline_samples = None
        self._mobility_drag_preview_samples = None
        self._mobility_drag_baseline_group_positions = None
        self._mobility_drag_preview_group_positions = None
        self._mobility_drag_baseline_group_physical = None
        self._mobility_drag_baseline_group_member_samples = {}
        self._mobility_drag_baseline_group_samples = None
        self._mobility_drag_preview_group_samples = None
        self._dependent_drag_preview_samples.clear()
        if active:
            self._refresh_viewport()

    def _handle_target_transform(self, event: TransformInput) -> None:
        """Update the shared pending draft from an actor-gizmo gesture."""

        document = self.document
        if document is None or document.read_only:
            return
        matrix = np.asarray(event.matrix, dtype=np.float64)
        if event.phase is TransformPhase.BEGIN:
            # A renderer BEGIN is delivered from inside the active gizmo
            # callback. Cancelling an active document edit is safe here, but
            # detaching the gizmo would re-enter the router and erase the
            # incoming gesture before its UPDATE/COMMIT phases can arrive.
            if self._transform_actor_id is not None:
                self._finish_target_transform(commit=False, clear_gizmo=False)
            scenario = self._begin_motion_draft(event.actor_id)
            if scenario is None:
                return
            actor = scenario.actor(event.actor_id)
            if actor is None or actor.locked:
                return
            baseline_samples = self._draft_samples(actor)
            self._transform_actor_id = actor.id
            self._transform_baseline_actor = actor
            self._transform_baseline_position = matrix[:3, 3].copy()
            self._transform_baseline_rotation = matrix[:3, :3].copy()
            self._transform_baseline_samples = baseline_samples
            self._transform_preview_samples = baseline_samples
            return

        if event.actor_id != self._transform_actor_id:
            return
        if event.phase is TransformPhase.CANCEL:
            self._finish_target_transform(commit=False)
            return
        if event.phase not in {TransformPhase.UPDATE, TransformPhase.COMMIT}:
            return
        baseline = self._transform_baseline_actor
        baseline_position = self._transform_baseline_position
        if baseline is None or baseline_position is None:
            return
        delta = matrix[:3, 3] - baseline_position
        next_orientation = baseline.orientation
        if isinstance(baseline.orientation, FixedOrientationSpec) and not np.allclose(
            matrix[:3, :3], self._transform_baseline_rotation, rtol=0.0, atol=1e-10
        ):
            orientation = sionna_orientation_from_rotation_matrix(matrix[:3, :3])
            if orientation is None:
                self._finish_target_transform(commit=False)
                return
            orientation_degrees = tuple(float(value) for value in np.degrees(orientation))
            if baseline.role is ActorRole.TARGET:
                orientation_degrees = self._remove_target_front_alignment(
                    baseline.name,
                    orientation_degrees,
                )
            orientation_degrees = (
                (orientation_degrees[0] + 180.0) % 360.0 - 180.0,
                orientation_degrees[1],
                orientation_degrees[2],
            )
            next_orientation = FixedOrientationSpec(
                yaw_deg=float(orientation_degrees[0]),
                pitch_deg=float(orientation_degrees[1]),
                roll_deg=float(orientation_degrees[2]),
            )
        try:
            mobility = self._translate_subject_mobility(
                baseline,
                tuple(float(value) for value in delta),
                reference_position=tuple(float(value) for value in baseline_position),
            )
        except (IndexError, KeyError, RuntimeError, ValueError):
            self._finish_target_transform(commit=False)
            return
        updated = baseline.with_changes(mobility=mobility, orientation=next_orientation)
        baseline_samples = self._transform_baseline_samples
        if baseline_samples is not None:
            fixed_orientation = None
            if isinstance(next_orientation, FixedOrientationSpec):
                fixed_orientation = (
                    next_orientation.yaw_deg,
                    next_orientation.pitch_deg,
                    next_orientation.roll_deg,
                )
                if baseline.role is ActorRole.TARGET:
                    fixed_orientation = self._apply_target_front_alignment(
                        baseline.name,
                        fixed_orientation,
                    )
            self._transform_preview_samples = self._rigid_preview_samples(
                baseline_samples,
                translation=tuple(float(value) for value in delta),
                fixed_orientation=fixed_orientation,
            )
        self._dependent_drag_preview_samples.clear()
        candidate_scenario = self._motion_draft_baseline.scenario.replace_actor(updated)
        self._dependent_drag_preview_samples = self._look_at_drag_previews(
            candidate_scenario,
            updated.id,
            self._transform_preview_samples,
        )
        self._stage_motion_draft(updated)
        if event.phase is TransformPhase.COMMIT:
            self._finish_target_transform(commit=True)

    def _target_front_yaw_offset(self, actor_name: str) -> float:
        """Return the generator-resolved catalog front offset for one target."""

        document = self.document
        compiled_scenario = self._compiled_scenario
        if (
            document is None
            or compiled_scenario is None
            or self._compiled_directory != self._current_scenario_directory()
        ):
            return 0.0
        current_actor = next(
            (actor for actor in document.actors if actor.name == actor_name),
            None,
        )
        if current_actor is None:
            return 0.0
        compiled_actor = compiled_scenario.actor(current_actor.id)
        if compiled_actor is None or compiled_actor.target != current_actor.target:
            return 0.0
        runtime = self.compilation.runtime if self.compilation is not None else None
        for target in getattr(runtime, "targets", ()) if runtime is not None else ():
            if str(getattr(target, "name", "")) == actor_name:
                return float(getattr(target, "asset_front_yaw_offset_deg", 0.0) or 0.0)
        return 0.0

    def _apply_target_front_alignment(
        self,
        actor_name: str,
        authored_orientation: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Compose the catalog front rotation in the target's local frame."""

        authored = Quaternion.from_euler_deg(*authored_orientation)
        alignment = Quaternion.from_euler_deg(
            self._target_front_yaw_offset(actor_name),
            0.0,
            0.0,
        )
        return (authored * alignment).to_euler_deg()

    def _remove_target_front_alignment(
        self,
        actor_name: str,
        rendered_orientation: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Recover authored orientation from a renderer-facing target pose."""

        rendered = Quaternion.from_euler_deg(*rendered_orientation)
        alignment = Quaternion.from_euler_deg(
            self._target_front_yaw_offset(actor_name),
            0.0,
            0.0,
        )
        inverse_alignment = Quaternion(
            alignment.w,
            -alignment.x,
            -alignment.y,
            -alignment.z,
        )
        return (rendered * inverse_alignment).to_euler_deg()

    def _finish_target_transform(
        self,
        *,
        commit: bool,
        clear_gizmo: bool = True,
    ) -> None:
        """End the gizmo gesture, retaining accepted changes until Apply."""

        active = self._transform_actor_id is not None
        if active:
            self._finish_motion_draft(accept=commit)
        self._transform_actor_id = None
        self._transform_baseline_actor = None
        self._transform_baseline_position = None
        self._transform_baseline_rotation = None
        self._transform_baseline_samples = None
        self._transform_preview_samples = None
        self._dependent_drag_preview_samples.clear()
        if not commit and clear_gizmo:
            clear_viewport_gizmo = getattr(
                getattr(getattr(self, "viewport", None), "port", None),
                "clear_transform_gizmo",
                None,
            )
            if callable(clear_viewport_gizmo):
                clear_viewport_gizmo()

        if active:
            self._refresh_viewport()

    def _timeline_scrubbed(self, value: int) -> None:
        self._play_step = int(value)
        self._play_origin_step = self._play_step
        self._play_origin_time = time.monotonic()
        self._refresh_viewport()

    @staticmethod
    def _playback_interval_ms(timeline: TimelineSettings) -> int | None:
        """Return a safe Qt interval for consecutive authored samples."""

        try:
            steps = int(timeline.steps)
            duration_s = float(timeline.duration_s)
        except (TypeError, ValueError, OverflowError):
            return None
        if steps < 2 or not math.isfinite(duration_s) or duration_s <= 0.0:
            return None

        seconds_per_step = duration_s / (steps - 1)
        if not math.isfinite(seconds_per_step):
            return _QT_TIMER_MAX_INTERVAL_MS
        if seconds_per_step * 1_000.0 >= _QT_TIMER_MAX_INTERVAL_MS:
            return _QT_TIMER_MAX_INTERVAL_MS
        # QTimer accepts integer milliseconds. A positive sub-millisecond
        # authored interval must still avoid a zero-interval busy loop.
        return max(1, int(round(seconds_per_step * 1_000.0)))

    def _update_playback_timing(self, timeline: TimelineSettings) -> None:
        interval_ms = self._playback_interval_ms(timeline)
        if interval_ms is None:
            self._play_timer.stop()
            self.play_button.setText("Play")
            self.play_button.setEnabled(False)
            return
        self._play_timer.setInterval(max(16, interval_ms))
        self.play_button.setEnabled(True)

    def _toggle_playback(self) -> None:
        if self._play_timer.isActive():
            self._stop_playback(refresh=True)
        else:
            document = self.document
            if document is None or self._waypoint_session_actor_id is not None:
                return
            if self._tool is AuthoringTool.MOVE:
                self.set_tool(AuthoringTool.SELECT)
            self._update_playback_timing(document.scenario.timeline)
            if not self.play_button.isEnabled():
                return
            self._play_origin_step = self._play_step
            self._play_origin_time = time.monotonic()
            self._play_timer.start()
            self.play_button.setText("Pause")
            self._refresh_viewport()

    def _stop_playback(self, *, refresh: bool) -> None:
        was_active = self._play_timer.isActive()
        self._play_timer.stop()
        self.play_button.setText("Play")
        if was_active and refresh:
            self._refresh_viewport()

    def _advance_playback(self) -> None:
        maximum = self.timeline_slider.maximum()
        if maximum <= 0:
            self._play_timer.stop()
            self.play_button.setText("Play")
            return
        timeline = self.document.scenario.timeline if self.document is not None else None
        if timeline is None or timeline.duration_s <= 0:
            return
        elapsed = time.monotonic() - self._play_origin_time
        step = (self._play_origin_step + round(elapsed * maximum / timeline.duration_s)) % (
            maximum + 1
        )
        # Preserve the wall-clock anchor: scrubbing manually establishes a new
        # one, while timer delivery may skip samples to keep physical time.
        with QSignalBlocker(self.timeline_slider):
            self.timeline_slider.setValue(step)
        self._play_step = step
        self._refresh_viewport()

    def focus_selection(self) -> bool:
        if self.document is None or self.document.selected_subject is None:
            return False
        port = getattr(self.viewport, "port", None)
        if self.camera_controls is not None:
            self.camera_controls.select(self.document.selected_subject.id)
            return self.camera_controls.focus()
        return bool(port is not None and port.focus_actor(self.document.selected_subject.id))

    def _show_camera_controls(self) -> None:
        if self.camera_controls is not None:
            self.details_toggle.setChecked(True)
            self.drawer.setCurrentWidget(self.camera_scroll)

    def _sync_selection_camera(self, document, subject_id) -> None:
        if (
            self.camera_controls is not None
            and self.document is document
            and document.selected_subject is not None
            and document.selected_subject.id == subject_id
        ):
            self.camera_controls.select(
                subject_id,
                focus=self.focus_on_select.isChecked() and self._tool is AuthoringTool.SELECT,
            )

    def fit_all(self) -> bool:
        port = getattr(self.viewport, "port", None)
        if self.camera_controls is not None:
            tools = self.camera_controls.camera_controller.presentation()
            tools.prepare()
            tools.free_camera()
        return bool(port is not None and port.fit_all())

    def fit_actors(self) -> bool:
        port = getattr(self.viewport, "port", None)
        if self.camera_controls is not None:
            tools = self.camera_controls.camera_controller.presentation()
            tools.prepare()
            tools.free_camera()
        return bool(port is not None and port.fit_actors())

    def _install_actions(self) -> None:
        for sequence, callback in (
            (QKeySequence.Undo, self._undo),
            (QKeySequence.Redo, self._redo),
            (QKeySequence.Delete, self._delete_selection),
            (QKeySequence("F"), self.focus_selection),
        ):
            action = QAction(self)
            action.setShortcut(sequence)
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(callback)
            self.addAction(action)

        self._waypoint_shortcut_actions: list[QAction] = []
        for sequence, command in (
            (QKeySequence("Esc"), "cancel"),
            (QKeySequence("Return"), "finish"),
            (QKeySequence("Enter"), "finish"),
            (QKeySequence("Backspace"), "remove"),
        ):
            action = QAction(self)
            action.setShortcut(sequence)
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.setData(command)
            action.setEnabled(False)
            action.triggered.connect(
                lambda _checked=False, value=command: self._dispatch_waypoint_shortcut(value)
            )
            self.addAction(action)
            self._waypoint_shortcut_actions.append(action)

    def _undo(self) -> None:
        if self._transform_actor_id is not None:
            self._finish_target_transform(commit=False)
            return
        if self._mobility_drag_actor_id is not None:
            self._finish_mobility_drag(commit=False)
            return
        if self._waypoint_session_actor_id is not None:
            return
        if not self._resolve_pending_inspector_edits():
            return
        if not self._resolve_pending_timeline_edits():
            return
        if self.document is not None and self.document.can_undo:
            self.document.undo()

    def _redo(self) -> None:
        if self._transform_actor_id is not None:
            self._finish_target_transform(commit=False)
            return
        if self._mobility_drag_actor_id is not None:
            self._finish_mobility_drag(commit=False)
            return
        if self._waypoint_session_actor_id is not None:
            return
        if not self._resolve_pending_inspector_edits():
            return
        if not self._resolve_pending_timeline_edits():
            return
        if self.document is not None and self.document.can_redo:
            self.document.redo()

    def _delete_selection(self) -> None:
        if self._waypoint_session_actor_id is not None:
            return
        if not self._resolve_pending_inspector_edits():
            return
        self._finish_target_transform(commit=False)
        self._finish_mobility_drag(commit=False)
        self._cancel_waypoint_session()
        if self.document is None or self._read_only:
            return
        if self.document.selected_actor_id is not None:
            self.document.remove_actor(self.document.selected_actor_id)
        elif self.document.selected_group_id is not None:
            self.document.remove_group(self.document.selected_group_id)

    def _set_editing_enabled(self, enabled: bool) -> None:
        generation = getattr(self.visualizer, "authoring_generation_controller", None)
        drawing_waypoints = self._waypoint_session_actor_id is not None
        osm_busy = self.osm_panel.active or self.osm_panel.building
        ordinary_editing = enabled and not drawing_waypoints and not osm_busy
        self.osm_panel.setEnabled(enabled and not drawing_waypoints)
        inspector_editing = (
            ordinary_editing
            and self.document is not None
            and (
                self.document.selected_actor is not None or self.document.selected_group is not None
            )
        )
        workflow_enabled = ordinary_editing and not bool(getattr(generation, "running", False))
        self.street_actors_button.setEnabled(workflow_enabled)
        self.environment_library_button.setEnabled(workflow_enabled)
        self.builder_tools.set_editing_enabled(workflow_enabled)
        self.actor_panel.setEnabled(not drawing_waypoints)
        self.inspector_panel.setEnabled(not drawing_waypoints and not osm_busy)
        self.leave_authoring_button.setEnabled(
            not drawing_waypoints and not bool(getattr(generation, "running", False))
        )
        self.drawer.setTabEnabled(0, not drawing_waypoints)
        self.actor_tree.setEnabled(True)
        for button in self._add_actor_buttons:
            button.setEnabled(ordinary_editing)
        self._form_group_button.setEnabled(ordinary_editing)
        for action in self._tool_actions.values():
            action.setEnabled(ordinary_editing)
        self.scene_source_combo.setEnabled(ordinary_editing)
        preserved_scene_extension = str(self.scene_source_combo.currentData() or "") not in {
            "sionna",
            "library",
            "local",
        }
        self.scene_id_combo.setEnabled(ordinary_editing and not preserved_scene_extension)
        self.scene_apply_button.setEnabled(ordinary_editing and not preserved_scene_extension)
        self.scene_browse_button.setEnabled(ordinary_editing)
        self.steps_spin.setEnabled(ordinary_editing)
        use_rate = self.sample_rate_check.isChecked()
        self.sample_rate_check.setEnabled(ordinary_editing)
        self.sample_rate_spin.setEnabled(ordinary_editing and use_rate)
        self.duration_spin.setEnabled(ordinary_editing and not use_rate)
        self.quality_combo.setEnabled(ordinary_editing)
        self.path_metrics_check.setEnabled(ordinary_editing)
        self.apply_timeline_button.setEnabled(ordinary_editing and self._timeline_draft_pending)
        self.reset_timeline_button.setEnabled(ordinary_editing and self._timeline_draft_pending)
        self.mobility_editor.set_editing_enabled(inspector_editing)
        self.mobility_editor.set_waypoint_drawing_active(drawing_waypoints)
        self.orientation_editor.set_editing_enabled(inspector_editing)
        self.save_button.setEnabled(workflow_enabled)
        self.save_as_button.setEnabled(workflow_enabled)
        self.validate_button.setEnabled(workflow_enabled and not self._explicit_validation_running)
        generate_supported = self.compilation is None or not getattr(
            self.compilation, "generation_issues", ()
        )
        self.generate_button.setEnabled(workflow_enabled and generate_supported)
        self.preview_result_button.setEnabled(
            self._preview_result_available and not drawing_waypoints
        )
        if self.document is not None:
            actor = self.document.selected_actor
            group = self.document.selected_group
            if actor is not None:
                self.name_edit.setEnabled(
                    inspector_editing
                    and self.document.scenario.capability("identity", actor.id).editable
                )
                self.mobility_editor.set_editing_enabled(
                    inspector_editing
                    and self.document.scenario.capability("mobility", actor.id).editable
                )
                self.orientation_editor.set_editing_enabled(
                    inspector_editing
                    and self.document.scenario.capability(
                        "orientation",
                        actor.id,
                    ).editable
                )
                self._set_target_editing_enabled(actor, inspector_editing)
            elif group is not None:
                group_editable = self.document.scenario.capability(
                    "group",
                    group.id,
                ).editable
                self.name_edit.setEnabled(inspector_editing and group_editable)
                self.mobility_editor.set_editing_enabled(inspector_editing and group_editable)
        self._sync_motion_edit_action()

    def confirm_replace(self) -> bool:
        """Prompt before replacing or closing a dirty authoring document."""

        if not self._resolve_pending_inspector_edits():
            return False
        if not self._resolve_pending_timeline_edits():
            return False
        if self.document is None or not self.document.dirty:
            return True
        choice = QMessageBox.question(
            self,
            "Unsaved Scenario",
            "Discard unsaved Scenario Builder changes?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return choice == QMessageBox.Discard

    def commit_pending_edits(self) -> bool:
        """Resolve pending editor and transient values before snapshotting."""

        if not self._resolve_pending_inspector_edits():
            return False
        self._finish_target_transform(commit=True)
        self._finish_mobility_drag(commit=True)
        self._commit_waypoint_session()
        if self._tool is AuthoringTool.WAYPOINT:
            self.set_tool(AuthoringTool.SELECT)
        if not self._resolve_pending_timeline_edits():
            return False
        return True

    def append_generation_log(self, text: str) -> None:
        self.generation_log.appendPlainText(str(text))

    def reset_generation_state(self) -> None:
        """Clear presentation state before a new or unrelated generation session."""

        self._preview_result_available = False
        self.generation_log.clear()
        self.generation_status.setText("No generation has been launched.")
        self.generation_progress.setRange(0, 100)
        self.generation_progress.setValue(0)
        self.cancel_generation_button.setEnabled(False)
        self.preview_result_button.setEnabled(False)
        for attribute in (
            "_orchav_generation_replay_token",
            "_orchav_generation_replay_offset",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)

    def set_generation_running(
        self, running: bool, *, launched_revision: int | None = None
    ) -> None:
        """Reflect generation/save conflict state in the workflow controls."""

        writable = self.document is not None and not self.document.read_only
        workflow_enabled = writable and not running and self._waypoint_session_actor_id is None
        generate_supported = self.compilation is None or not getattr(
            self.compilation, "generation_issues", ()
        )
        self.generate_button.setEnabled(workflow_enabled and generate_supported)
        self.validate_button.setEnabled(workflow_enabled and not self._explicit_validation_running)
        self.save_button.setEnabled(workflow_enabled)
        self.save_as_button.setEnabled(workflow_enabled)
        self.leave_authoring_button.setEnabled(
            not running and self._waypoint_session_actor_id is None
        )
        self.cancel_generation_button.setEnabled(running)
        self.preview_result_button.setEnabled(
            self._preview_result_available
            and not running
            and self._waypoint_session_actor_id is None
        )
        if running:
            self.generation_progress.setRange(0, 0)
            self.generation_status.setText(
                f"Generating saved revision {launched_revision}…"
                if launched_revision is not None
                else "Generating…"
            )
        refresh_actions = getattr(self.visualizer, "_refresh_authoring_actions", None)
        if callable(refresh_actions):
            refresh_actions()

    def set_generation_progress(self, progress: Any) -> None:
        """Present one structured JSONL progress update."""

        total = max(int(progress.total_steps), 1)
        completed = max(0, min(int(progress.completed_steps), total))
        self.generation_progress.setRange(0, total)
        self.generation_progress.setValue(completed)
        self.generation_status.setText(
            f"Generated {completed}/{total} steps — {float(progress.elapsed_s):.1f} s"
        )

    def set_generation_result(self, result: Any) -> None:
        """Present terminal state, staleness, and preview availability."""

        self.set_generation_running(False)
        succeeded = bool(getattr(result, "succeeded", False))
        self._preview_result_available = succeeded
        stale = bool(getattr(result, "stale", False))
        state = getattr(getattr(result, "state", None), "value", str(getattr(result, "state", "")))
        suffix = " — result is from an older draft revision" if stale else ""
        self.generation_status.setText(f"Generation {state}{suffix}")
        self.preview_result_button.setEnabled(succeeded and self._waypoint_session_actor_id is None)

    def close_workspace(self) -> None:
        """Stop authoring work before releasing subscriptions and the renderer."""

        self._compile_status_timer.stop()
        if (
            self.compilation is not None
            and getattr(self.compilation, "recording", None) is not None
        ):
            self.compilation.recording.close()
        if self.camera_controls is not None:
            self.camera_controls.close_camera()
            self.camera_controls = None
        self.osm_panel.close_jobs()
        self._play_timer.stop()
        self._candidate_compile_timer.stop()
        if self._compile_scheduler is not None:
            self._compile_scheduler.close()
        if self._candidate_compile_scheduler is not None:
            self._candidate_compile_scheduler.close()
        close_compiler = getattr(self.compiler, "close", None)
        if callable(close_compiler):
            close_compiler()
        self._finish_target_transform(commit=False)
        self._finish_mobility_drag(commit=False)
        self._cancel_waypoint_session()
        if self._unsubscribe_document is not None:
            self._unsubscribe_document()
            self._unsubscribe_document = None
        close_viewport = getattr(self.viewport, "close_viewport", None)
        if callable(close_viewport):
            close_viewport()
