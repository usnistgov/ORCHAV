"""Environment reuse, street placement and geometry inspection UI."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.scenarios.actors import AlignMotionOrientationSpec, NetworkRouteMobilitySpec

from .domain import ActorRole, AuthoringActor, TargetAsset
from .mobility_models import translate_mobility
from .sensing_guides import line_guide, sensing_guides
from .spatial_checks import SpatialScene


class BuilderTools(QWidget):
    """Display-only inspection state; durable edits go through the document."""

    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self._scene = None
        self._scene_assets = None
        self._checked = None
        self._issues = ()
        self._actors = ()
        self.edits_enabled = True
        self.edit_actions = []
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        for label, callback in (
            ("Street actors…", self.street_actors),
            ("Environment library…", self.environments),
            ("Check current frame", self.check_positions),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            actions.addWidget(button)
            if callback != self.check_positions:
                self.edit_actions.append(button)
        self.show_sensing = QCheckBox("Sensing guides (read-only)")
        self.show_sensing.toggled.connect(lambda _: workspace._refresh_viewport())
        actions.addWidget(self.show_sensing)
        actions.addStretch()
        layout.addLayout(actions)
        self.sensing_note = QLabel(
            "Show sensor boresights and explicitly configured sensing limits. No sensing settings are changed."
        )
        self.sensing_note.setWordWrap(True)
        layout.addWidget(self.sensing_note)
        self.summary = QLabel(
            "Check positions at the current timeline frame. Checks cover environment bounds, ground clearance and building interiors; they are not swept-volume collision tests."
        )
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.issues = QTreeWidget()
        self.issues.setHeaderLabels(("Actor", "Placement issue"))
        self.issues.itemSelectionChanged.connect(self._select_issue)
        layout.addWidget(self.issues, 1)
        fixes = QHBoxLayout()
        self.snap = QPushButton("Snap to ground")
        self.clear = QPushButton("Move to nearby clear ground")
        self.snap.clicked.connect(lambda: self.correct(clear=False))
        self.clear.clicked.connect(lambda: self.correct(clear=True))
        self.snap.setToolTip(
            "Translate the selected actor's whole motion so its current mesh base rests on ground."
        )
        self.clear.setToolTip(
            "Translate editable motion to a nearby position outside building interiors. Graph-owned and animated motion must be edited through their source."
        )
        fixes.addWidget(self.snap)
        fixes.addWidget(self.clear)
        fixes.addStretch()
        layout.addLayout(fixes)

    def set_editing_enabled(self, enabled):
        self.edits_enabled = enabled
        for button in self.edit_actions:
            button.setEnabled(enabled)
        if not enabled:
            self.snap.setEnabled(False)
            self.clear.setEnabled(False)

    def _key(self):
        ws = self.workspace
        return (
            (ws.document.scenario.document_id, ws.document.revision, ws._play_step)
            if ws.document
            else None
        )

    def guides(self, actors):
        self._actors = tuple(actors)
        ws = self.workspace
        guides = []
        if self._checked is not None and self._checked != self._key():
            self.summary.setText("Placement checks are out of date. Check the current frame again.")
            self._issues = ()
            self.issues.clear()
            self._checked = None
        if self.show_sensing.isChecked():
            try:
                result, note = sensing_guides(
                    ws.document.scenario.source_snapshot.to_mapping(), actors
                )
            except (ValueError, TypeError, KeyError) as exc:
                result, note = (), f"Sensing guides are unavailable: {exc}"
            guides.extend(result)
            self.sensing_note.setText(note)
        for issue in self._issues:
            p = np.asarray(issue.position)
            lines = [(p - np.eye(3)[axis], p + np.eye(3)[axis]) for axis in range(3)]
            guides.append(
                line_guide(
                    f"placement:{issue.actor_id}:{issue.code}", lines, (1.0, 0.25, 0.15, 1.0)
                )
            )
        editable = bool(
            self.edits_enabled
            and ws.document
            and not ws.document.read_only
            and ws.document.selected_actor
        )
        self.snap.setEnabled(editable and self._checked == self._key())
        self.clear.setEnabled(editable and self._checked == self._key())
        return tuple(guides)

    @staticmethod
    def _bottom(actor):
        if actor.target_asset is None:
            return actor.current_position[2]
        pose = np.array(
            actor.orientation_matrix if actor.orientation_matrix is not None else np.eye(4),
            dtype=float,
        )
        pose[:3, 3] = actor.current_position
        pose = pose @ np.asarray(actor.target_asset.local_to_actor)
        vertices = actor.target_asset.payload.vertices
        return float((vertices @ pose[2, :3] + pose[2, 3]).min())

    def check_positions(self):
        ws = self.workspace
        ws.details_toggle.setChecked(True)
        ws.drawer.setCurrentWidget(self)
        if ws.osm_panel.active or ws.osm_panel.building:
            self.summary.setText(
                "Return to the scene after the OSM build before checking placement."
            )
            return
        if not ws._compilation_is_current() or ws.compilation is None:
            self.summary.setText("Wait for the scene preview to finish before checking placement.")
            return
        assets = ws.compilation.scene_assets
        if self._scene_assets is not assets:
            self._scene = SpatialScene(assets)
            self._scene_assets = assets
        self._issues = tuple(
            issue
            for actor in self._actors
            if actor.visible and actor.current_position is not None
            for issue in self._scene.check(
                actor.actor_id, actor.name, actor.current_position, bottom_z=self._bottom(actor)
            )
        )
        self._checked = self._key()
        self.issues.clear()
        for issue in self._issues:
            item = QTreeWidgetItem((issue.actor_name, issue.message))
            item.setData(0, Qt.UserRole, issue.actor_id)
            self.issues.addTopLevelItem(item)
        self.issues.resizeColumnToContents(0)
        self.summary.setText(
            f"Frame {ws._play_step + 1}: {len(self._issues)} placement issues. "
            + (
                "No labelled ground surface; ground clearance could not be checked. "
                if not self._scene.ground
                else ""
            )
            + (
                "No labelled building meshes; interiors could not be checked."
                if not self._scene.buildings
                else "Select a row to focus an actor. Corrections move its entire editable motion."
            )
        )
        ws._refresh_viewport()

    def _select_issue(self):
        item = self.issues.currentItem()
        if item is not None and self.workspace.document is not None:
            self.workspace.document.select(item.data(0, Qt.UserRole))
            self.workspace.focus_selection()

    def correct(self, *, clear):
        ws = self.workspace
        if not self.edits_enabled:
            return
        if (
            self._checked != self._key()
            or ws.document is None
            or ws.document.selected_actor is None
        ):
            return
        if not ws.commit_pending_edits() or self._checked != self._key():
            return
        actor = ws.document.selected_actor
        overlay = next((a for a in self._actors if a.actor_id == actor.id), None)
        if overlay is None or overlay.current_position is None:
            return
        try:
            if actor.locked or not ws.document.scenario.capability("mobility", actor.id).editable:
                raise ValueError("This actor's motion is locked or owned by its imported asset.")
            position = np.asarray(overlay.current_position)
            point = self._scene.corrected_position(
                position, bottom_z=self._bottom(overlay), clear=clear
            )
            mobility = translate_mobility(actor.mobility, tuple(point - position))
            ws.document.update_actor(actor.id, mobility=mobility)
        except (ValueError, RuntimeError, PermissionError) as exc:
            QMessageBox.warning(self, "Placement correction unavailable", str(exc))

    def street_actors(self, *, edit_selected=False):
        ws = self.workspace
        if not self.edits_enabled:
            return
        if ws.document is None or ws.document.read_only or not ws.commit_pending_edits():
            return
        from .street_dialog import StreetPlacementDialog
        from .street_routes import scenario_graph

        selected = ws.document.selected_actor
        if selected and (
            selected.locked or not ws.document.scenario.capability("mobility", selected.id).editable
        ):
            selected = None
        dialog = StreetPlacementDialog(
            scenario_graph(ws.document.scenario, ws._current_scenario_directory()),
            ws._catalog_target_ids(),
            self,
            selected_actor=selected,
        )
        if edit_selected and selected is not None:
            dialog.action.setCurrentIndex(dialog.action.findData("selected"))
        if dialog.exec() != QDialog.Accepted or not dialog.routes:
            return
        try:
            actors, resources = [], ()
            used = {a.name for a in ws.document.actors} | {g.name for g in ws.document.groups}
            for route in dialog.routes:
                mobility = NetworkRouteMobilitySpec(
                    travel_mode=dialog.travel_mode.currentData(),
                    graph_path=str(dialog.network.path.resolve()),
                    start_node=route.start,
                    end_node=route.end,
                    altitude_m=dialog.altitude.value(),
                    traversal={"type": "constant_speed", "speed_mps": dialog.speed.value()},
                )
                if dialog.action.currentData() == "selected":
                    # The inspector owns the transaction and its single undo step.
                    # Keep the graph's source path until Apply internalizes it.
                    ws.mobility_editor.set_mobility(mobility)
                    ws._mark_inspector_draft("mobility")
                    return
                mobility, resources = ws._internalize_mobility_resource(mobility)
                prefix = {"car": "Vehicle", "pedestrian": "Pedestrian", "bike": "Cyclist"}[
                    dialog.travel_mode.currentData()
                ]
                index = 1
                while f"{prefix}{index}" in used:
                    index += 1
                name = f"{prefix}{index}"
                used.add(name)
                target = TargetAsset.from_catalog_id(
                    dialog.asset.currentData(),
                    mesh_animation=dialog.travel_mode.currentData() == "pedestrian",
                )
                actor = AuthoringActor.create(ActorRole.TARGET, name, target=target)
                actors.append(
                    replace(
                        actor,
                        mobility=mobility,
                        orientation=AlignMotionOrientationSpec(allow_pitch=False),
                    )
                )
            ws.document.add_actors_with_resources(actors, resources)
        except (OSError, ValueError, RuntimeError, PermissionError) as exc:
            QMessageBox.warning(self, "Could not place actors", str(exc))

    def environments(self):
        ws = self.workspace
        if not self.edits_enabled:
            return
        if ws.document is None or not ws.commit_pending_edits():
            return
        from .environment_dialog import EnvironmentLibraryDialog

        dialog = EnvironmentLibraryDialog(ws)
        if dialog.exec() != QDialog.Accepted or dialog.entry is None:
            return
        try:
            entry = dialog.entry
            dependencies = dialog.library.dependencies(entry)
            ws._osm_fit_pending = True
            ws.document.use_environment(
                entry.metadata["scene"], dependencies, builder=entry.metadata.get("builder")
            )
        except (OSError, ValueError, PermissionError) as exc:
            ws._osm_fit_pending = False
            QMessageBox.warning(self, "Could not use environment", str(exc))
