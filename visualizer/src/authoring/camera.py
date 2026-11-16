"""Adapt draft actors to the existing camera controller and camera panels."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..controllers.camera_controller import CameraController
from ..controllers.camera_ui import CameraUIActions
from ..panels.advanced_camera_panel import AdvancedCameraPanel
from ..panels.camera_panel import CameraControlPanel
from ..scene.target_transforms import sionna_orientation_from_transform


class DraftCameraQuery:
    """Expose the currently displayed draft poses through the camera query API."""

    def __init__(self, owner):
        self.owner = owner

    def actor(self, selection=None):
        selection = selection or self.owner.target_focus_dropdown.currentData() or {}
        identity = selection.get("stable_target_id")
        return next((a for a in self.owner.actors if str(a.actor_id) == identity), None)

    def resolve_target_selection_position(self, selection):
        actor = self.actor(selection)
        return actor.current_position if actor is not None else None

    def get_focus_position(self):
        return self.resolve_target_selection_position(
            self.owner.target_focus_dropdown.currentData()
        )

    def get_entity_position_orientation_and_info(self):
        actor = self.actor()
        if actor is None or actor.current_position is None:
            return None, None, None
        orientation = (
            sionna_orientation_from_transform(actor.orientation_matrix)
            if actor.orientation_matrix is not None
            else None
        )
        return (
            actor.current_position,
            orientation,
            {
                "type": "target",
                "index": self.owner.actors.index(actor),
                "stable_target_id": str(actor.actor_id),
            },
        )

    def get_focus_motion_direction(self):
        actor = self.actor()
        workspace = self.owner.workspace
        if actor is None or workspace.document is None:
            return None
        source = workspace.document.scenario.actor(actor.actor_id)
        samples = workspace._prepared_samples(source) if source is not None else None
        if samples is None or len(samples.positions) < 2:
            return None
        step = min(workspace._play_step, len(samples.positions) - 1)
        previous = max(0, step - 1)
        following = min(len(samples.positions) - 1, step + 1)
        return np.subtract(samples.positions[following], samples.positions[previous])

    def compute_scene_bounds(self, *, scope="whole"):
        return self.owner.port.preview_bounds(include_scene=scope == "whole")

    def viewport_aspect(self):
        widget = self.owner.port.widget
        return widget.width() / max(1, widget.height())


class BuilderCamera(QWidget):
    """Own camera-only state; all navigation behavior comes from shared services."""

    supports_recorded_results = False

    def __init__(self, workspace, port):
        super().__init__(workspace)
        self.workspace, self.port = workspace, port
        self.renderer = port.renderer
        self.vis = self.renderer
        self.vis_initialized = True
        self.app_state = SimpleNamespace(
            camera_mode="overview", pov_axis="motion", pov_hidden_node=None, fly_mode=False
        )
        self.current_scenario_path = None
        self.actors = ()
        self.node_service = self
        self.camera_controller = CameraController(self, DraftCameraQuery(self))
        self.ui_controller = CameraUIActions(self)
        self.basic = CameraControlPanel(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.basic.create_panel())
        for name, widget in self.basic.widgets.items():
            setattr(self, name, widget)
        self.advanced = AdvancedCameraPanel(self)
        self.advanced_widget = self.advanced.create_panel()
        layout.addWidget(self.advanced_widget)
        layout.addStretch()
        self.advanced_widget.hide()
        self.advanced_camera_cb.setToolTip(
            "Show motion, framing, saved views and camera sequences."
        )
        self.advanced_camera_cb.toggled.connect(self.advanced_widget.setVisible)
        self.advanced_camera_cb.toggled.connect(
            lambda _: self.basic.sync_from_state(self.app_state)
        )
        self.live_figure_btn.hide()
        self.camera_target_view_btn.hide()
        self.fly_mode_cb.show()
        self.fly_mode_cb.toggled.connect(self.ui_controller.handle_fly_mode_toggled)
        for mode in ("overview", "follow", "pov"):
            getattr(self, f"{mode}_mode_rb").toggled.connect(
                lambda checked, mode=mode: self.ui_controller.handle_camera_mode_changed(
                    mode, checked
                )
            )
        for key, view in (
            ("top", "top"),
            ("side", "side"),
            ("iso", "isometric"),
            ("front", "front"),
        ):
            getattr(self, f"view_{key}_btn").clicked.connect(
                lambda _, view=view: self.overview(view)
            )
        self.reset_camera_btn.clicked.connect(lambda: self.overview("isometric"))
        self.camera_fov_spin.valueChanged.connect(self.camera_controller.set_field_of_view)
        self.target_focus_dropdown.currentIndexChanged.connect(self._tracking_changed)
        self.pov_axis_combo.currentIndexChanged.connect(
            lambda _: self.ui_controller.handle_pov_axis_combo_changed(
                self.pov_axis_combo.currentData()
            )
        )
        for index, button in enumerate(self.camera_preset_buttons, start=1):
            button.clicked.connect(lambda _, index=index: self._preset(index))

    def set_state(self, **changes):
        for key, value in changes.items():
            setattr(self.app_state, key, value)
        if self.app_state.camera_mode != "overview":
            self.app_state.fly_mode = False
        if "fly_mode" in changes or "camera_mode" in changes:
            mode = "fly" if self.app_state.fly_mode else "orbit"
            self.port.set_camera_mode(mode)
            with QSignalBlocker(self.fly_mode_cb):
                self.fly_mode_cb.setChecked(self.app_state.fly_mode)
            with QSignalBlocker(self.workspace.camera_mode_combo):
                self.workspace.camera_mode_combo.setCurrentIndex(
                    self.workspace.camera_mode_combo.findData(mode)
                )
        self.basic.sync_from_state(self.app_state)
        self.advanced.sync_from_state(self.app_state)
        if "pov_hidden_node" in changes or "camera_mode" in changes:
            self.sync_pov_entity_visibility(())

    def sync_pov_entity_visibility(self, _refs):
        hidden = self.app_state.pov_hidden_node
        actor_id = None
        if self.app_state.camera_mode == "pov" and hidden is not None:
            index = hidden[1]
            if 0 <= index < len(self.actors):
                actor_id = self.actors[index].actor_id
        self.port.set_pov_hidden_actor(actor_id)
        return True

    def sync(self, snapshot):
        """Refresh identities and follow/POV from the same poses shown in the viewport."""
        identity = str(snapshot.document_id)
        if self.current_scenario_path != identity:
            self.camera_controller.reset_for_recording()
            self.set_state(camera_mode="overview", pov_hidden_node=None)
            self.current_scenario_path = identity
        self.actors = snapshot.actors
        selected = self.target_focus_dropdown.currentData()
        choices = [
            (a.name, {"type": "target", "stable_target_id": str(a.actor_id), "index": i})
            for i, a in enumerate(self.actors)
        ]
        if choices != [
            (self.target_focus_dropdown.itemText(i), self.target_focus_dropdown.itemData(i))
            for i in range(self.target_focus_dropdown.count())
        ]:
            with QSignalBlocker(self.target_focus_dropdown):
                self.target_focus_dropdown.clear()
                for label, value in choices:
                    self.target_focus_dropdown.addItem(label, value)
                if selected is not None:
                    wanted = selected.get("stable_target_id")
                    index = next(
                        (
                            i
                            for i, (_, value) in enumerate(choices)
                            if value["stable_target_id"] == wanted
                        ),
                        -1,
                    )
                    self.target_focus_dropdown.setCurrentIndex(index)
            if self.target_focus_dropdown.currentIndex() < 0:
                self.camera_controller.presentation().stop()
                self.set_state(camera_mode="overview", pov_hidden_node=None)
        if self.app_state.camera_mode == "follow":
            self.camera_controller.update_follow_camera_focus()
        elif self.app_state.camera_mode == "pov":
            self.camera_controller.set_pov_camera()
        self.basic.refresh_field_of_view()

    def select(self, actor_id: UUID, *, focus=False):
        for index in range(self.target_focus_dropdown.count()):
            if self.target_focus_dropdown.itemData(index)["stable_target_id"] == str(actor_id):
                self.target_focus_dropdown.setCurrentIndex(index)
                if focus:
                    self.focus()
                return

    def focus(self):
        point = self.camera_controller.scene_query.get_focus_position()
        if point is None:
            return False
        self.camera_controller.presentation().focus(point)
        return True

    def overview(self, view):
        self.camera_controller.presentation().free_camera()
        return self.camera_controller.set_overview_view(view)

    def _tracking_changed(self, _index):
        self.ui_controller.handle_target_focus_changed(self.target_focus_dropdown.currentText())

    def _preset(self, index):
        if self.camera_preset_save_btn.isChecked():
            self.camera_controller.save_camera_preset(index)
            self.camera_preset_save_btn.setChecked(False)
            self.camera_preset_buttons[index - 1].setToolTip(f"Go to saved view {index}")
        else:
            self.camera_controller.load_camera_preset(index)

    def close_camera(self):
        self.camera_controller.presentation().close()
        self.port.set_pov_hidden_actor(None)
