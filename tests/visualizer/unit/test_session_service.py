import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QSlider, QSpinBox

from visualizer.src.controllers.coverage_controller import CoverageController
from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.materials.appearance import (
    MaterialDisplayMode,
    VisualMaterialBinding,
    VisualMaterialSource,
)
from visualizer.src.model import RenderObjectState, Transform
from visualizer.src.panels.coverage_panel import CoverageMapPanel
from visualizer.src.panels.render_panel import RenderPanel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.coverage_service import CoverageService
from visualizer.src.services.material_modes import MaterialModeService
from visualizer.src.services.session_service import (
    CAMERA_SESSION_FORMAT,
    SESSION_VERSION,
    SessionService,
    WorkspaceSnapshotSummary,
    normalize_scenario_root,
    read_workspace_snapshot,
    read_workspace_summary,
)
from visualizer.src.state import (
    MPC_ORDER_VALUES,
    MPC_TYPE_VALUES,
    MpcVisibility,
    create_initial_state,
    update_state,
)
from visualizer.src.types.camera_state import CameraState
from visualizer.src.types.render_payloads import MeshPayload


class _DummyRenderer:
    capabilities = RendererCapabilities(transparency=True)

    def __init__(self):
        self.coverage_alpha_calls: list[float] = []

    def set_coverage_transparency(self, alpha: float) -> None:
        self.coverage_alpha_calls.append(float(alpha))


class _DummyVisualizer:
    def __init__(self):
        self.renderer = _DummyRenderer()
        self.building_alpha_calls: list[float] = []
        self.target_alpha_calls: list[float] = []
        self.current_scenario_path = None
        self.app_state = SimpleNamespace()
        self.coverage_restore = Mock()
        self.scene_appearance_service = SimpleNamespace(
            set_building_transparency=lambda alpha: self.building_alpha_calls.append(float(alpha)),
            set_target_transparency=lambda alpha: self.target_alpha_calls.append(float(alpha)),
        )
        self.ui_controller = SimpleNamespace(
            coverage_controller=SimpleNamespace(
                restore_session_state=self.coverage_restore,
            )
        )


class _FakeButton(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.checked = False

    def setChecked(self, checked: bool) -> None:
        self.checked = bool(checked)


class _CameraRenderer:
    def __init__(self) -> None:
        self.set_camera_state_calls: list[CameraState] = []

    def set_camera_state(self, state: CameraState) -> bool:
        self.set_camera_state_calls.append(state)
        return True


class _RoundTripRenderer:
    def __init__(self, camera_state: CameraState) -> None:
        self.objects = {}
        self.camera_state = camera_state
        self.set_camera_state_calls: list[CameraState] = []

    def ensure_object(self, obj):
        self.objects[obj.id] = obj
        return True

    def get_camera_state(self) -> CameraState:
        return self.camera_state

    def set_camera_state(self, state: CameraState) -> bool:
        self.set_camera_state_calls.append(state)
        return True


def test_restore_rendering_state_applies_default_building_and_target_alpha() -> None:
    viz = _DummyVisualizer()
    service = SessionService(viz)

    service._restore_rendering_state(
        {
            "building_alpha": 1.0,
            "target_alpha": 1.0,
            "coverage_alpha": 0.4,
        }
    )

    assert viz.building_alpha_calls == [1.0]
    assert viz.target_alpha_calls == [1.0]
    viz.coverage_restore.assert_called_once_with({"opacity": 0.4})


def test_restore_rendering_state_applies_non_default_alpha_overrides() -> None:
    viz = _DummyVisualizer()
    service = SessionService(viz)

    service._restore_rendering_state(
        {
            "building_alpha": 0.65,
            "target_alpha": 0.35,
        }
    )

    assert viz.building_alpha_calls == [0.65]
    assert viz.target_alpha_calls == [0.35]


def test_restore_rendering_state_ignores_invalid_alpha_values() -> None:
    viz = _DummyVisualizer()
    service = SessionService(viz)

    service._restore_rendering_state(
        {
            "building_alpha": "not-a-number",
            "target_alpha": None,
        }
    )

    assert viz.building_alpha_calls == []
    assert viz.target_alpha_calls == []


def test_restore_rendering_state_migrates_legacy_trajectory_width() -> None:
    """Older snapshots restore through the authoritative node-appearance owner."""
    viz = _DummyVisualizer()
    viz.renderer.set_trajectory_line_width = Mock()
    service = SessionService(viz)

    service._restore_rendering_state({"traj_line_width": 5})

    viz.renderer.set_trajectory_line_width.assert_called_once_with(5.0)


def test_get_rendering_state_saves_actual_coverage_panel_controls(qapp) -> None:
    opacity = QSlider(Qt.Horizontal)
    opacity.setRange(10, 100)
    opacity.setValue(64)
    threshold_toggle = QCheckBox()
    threshold_toggle.setChecked(True)
    threshold_value = QDoubleSpinBox()
    threshold_value.setRange(-200.0, 200.0)
    threshold_value.setValue(7.5)
    threshold_mask = QCheckBox()
    threshold_mask.setChecked(True)
    isolines = QCheckBox()
    isolines.setChecked(True)
    isoline_count = QSpinBox()
    isoline_count.setRange(2, 12)
    isoline_count.setValue(8)
    speed = QSlider(Qt.Horizontal)
    speed.setRange(1, 10)
    speed.setValue(6)
    panel = SimpleNamespace(
        widgets={
            "coverage_opacity": opacity,
            "coverage_threshold_toggle": threshold_toggle,
            "coverage_threshold_value": threshold_value,
            "coverage_threshold_mask_toggle": threshold_mask,
            "coverage_isolines_toggle": isolines,
            "coverage_isoline_count": isoline_count,
            "coverage_height_speed": speed,
        }
    )
    viz = SimpleNamespace(
        current_building_alpha=1.0,
        current_target_alpha=1.0,
        coverage_data={"metric_name": "sinr_db"},
        coverage_metric_name="sinr_db",
        coverage_opacity=1.0,
        coverage_interpolation_method="linear",
        coverage_threshold_enabled=False,
        coverage_threshold_value=None,
        coverage_threshold_mask_enabled=False,
        coverage_isolines_enabled=False,
        coverage_isoline_count=6,
        ui_controller=SimpleNamespace(
            coverage_controller=SimpleNamespace(height_animation_speed=3)
        ),
        ui_manager=SimpleNamespace(panels={"coverage": panel}),
    )

    state = SessionService(viz)._get_rendering_state()["coverage"]

    assert state == {
        "opacity": 0.64,
        "metric_name": "sinr_db",
        "interpolation": "linear",
        "threshold_enabled": True,
        "threshold_value": 7.5,
        "threshold_mask_enabled": True,
        "isolines_enabled": True,
        "isoline_count": 8,
        "height_animation_speed": 6,
    }


def test_restore_rendering_state_syncs_complete_coverage_state(qapp) -> None:
    opacity = QSlider(Qt.Horizontal)
    opacity.setRange(10, 100)
    metric_combo = QComboBox()
    metric_combo.addItem("Path loss (dB)", "best_path_loss_db")
    metric_combo.addItem("SINR (dB)", "sinr_db")
    interpolation = QComboBox()
    interpolation.addItems(["Raw", "Smooth", "Smooth+"])
    threshold_toggle = QCheckBox()
    threshold_value = QDoubleSpinBox()
    threshold_value.setRange(-200.0, 200.0)
    threshold_mask = QCheckBox()
    isolines = QCheckBox()
    isoline_count = QSpinBox()
    isoline_count.setRange(2, 12)
    speed = QSlider(Qt.Horizontal)
    speed.setRange(1, 10)
    panel = SimpleNamespace(
        widgets={
            "coverage_opacity": opacity,
            "opacity_label": QLabel(),
            "coverage_metric_combo": metric_combo,
            "coverage_interpolation": interpolation,
            "coverage_threshold_toggle": threshold_toggle,
            "coverage_threshold_value": threshold_value,
            "coverage_threshold_mask_toggle": threshold_mask,
            "coverage_isolines_toggle": isolines,
            "coverage_isoline_count": isoline_count,
            "coverage_height_speed": speed,
            "coverage_height_speed_label": QLabel(),
        },
        update_coverage_status=Mock(),
    )
    coverage_data = {
        "available_metrics": ["best_path_loss_db", "sinr_db"],
        "metric_name": "best_path_loss_db",
        "metric_layers": {
            "best_path_loss_db": np.asarray([[[80.0]]], dtype=np.float32),
            "sinr_db": np.asarray([[[12.0]]], dtype=np.float32),
        },
        "tx_names": [],
    }
    viz = SimpleNamespace(
        renderer=_DummyRenderer(),
        coverage_service=CoverageService(),
        coverage_data=coverage_data,
        ui_manager=SimpleNamespace(panels={"coverage": panel}),
        force_update_next_frame=False,
    )
    controller = CoverageController(
        SimpleNamespace(visualizer=viz, coverage_service=viz.coverage_service)
    )
    viz.ui_controller = SimpleNamespace(coverage_controller=controller)
    restored_controls = Mock()
    panel.restore_session_controls = restored_controls

    SessionService(viz)._restore_rendering_state(
        {
            "coverage": {
                "opacity": 0.82,
                "metric_name": "sinr_db",
                "interpolation": "cubic",
                "threshold_enabled": True,
                "threshold_value": 8.5,
                "threshold_mask_enabled": True,
                "isolines_enabled": True,
                "isoline_count": 9,
                "height_animation_speed": 7,
            }
        }
    )

    assert coverage_data["metric_name"] == "sinr_db"
    assert viz.coverage_opacity == 0.82
    assert viz.coverage_interpolation_method == "cubic"
    assert viz.coverage_threshold_enabled is True
    assert viz.coverage_threshold_value == 8.5
    assert viz.coverage_threshold_mask_enabled is True
    assert viz.coverage_isolines_enabled is True
    assert viz.coverage_isoline_count == 9
    assert controller.coverage_interpolation_method == "cubic"
    assert controller.height_animation_speed == 7
    assert panel.update_coverage_status.call_args.kwargs == {"supports_transparency": True}
    restored_controls.assert_called_once_with(
        opacity=0.82,
        threshold_enabled=True,
        threshold_value=8.5,
        mask_enabled=True,
        isolines_enabled=True,
        isoline_count=9,
        interpolation="cubic",
        height_animation_speed=7,
    )
    assert viz.renderer.coverage_alpha_calls == [0.82]
    assert viz.force_update_next_frame is True


def test_coverage_panel_session_sync_blocks_command_signals(qapp) -> None:
    panel = CoverageMapPanel.__new__(CoverageMapPanel)
    opacity = QSlider(Qt.Horizontal)
    opacity.setRange(10, 100)
    speed = QSlider(Qt.Horizontal)
    speed.setRange(1, 10)
    interpolation = QComboBox()
    interpolation.addItems(["Raw", "Smooth", "Smooth+"])
    threshold_toggle = QCheckBox()
    threshold_value = QDoubleSpinBox()
    threshold_value.setRange(-200.0, 200.0)
    threshold_mask = QCheckBox()
    isolines = QCheckBox()
    isoline_count = QSpinBox()
    isoline_count.setRange(2, 12)
    panel.widgets = {
        "coverage_opacity": opacity,
        "opacity_label": QLabel(),
        "coverage_height_speed": speed,
        "coverage_height_speed_label": QLabel(),
        "coverage_interpolation": interpolation,
        "coverage_threshold_toggle": threshold_toggle,
        "coverage_threshold_value": threshold_value,
        "coverage_threshold_mask_toggle": threshold_mask,
        "coverage_isolines_toggle": isolines,
        "coverage_isoline_count": isoline_count,
    }
    panel._threshold_metric = "sinr_db"
    emitted: list[object] = []
    for signal in (
        opacity.valueChanged,
        speed.valueChanged,
        interpolation.currentTextChanged,
        threshold_toggle.toggled,
        threshold_value.valueChanged,
        threshold_mask.toggled,
        isolines.toggled,
        isoline_count.valueChanged,
    ):
        signal.connect(emitted.append)

    panel.restore_session_controls(
        opacity=0.82,
        threshold_enabled=True,
        threshold_value=8.5,
        mask_enabled=True,
        isolines_enabled=True,
        isoline_count=9,
        interpolation="cubic",
        height_animation_speed=7,
    )

    assert emitted == []
    assert opacity.value() == 82
    assert panel.widgets["opacity_label"].text() == "82%"
    assert speed.value() == 7
    assert panel.widgets["coverage_height_speed_label"].text() == "0.29 s"
    assert interpolation.currentText() == "Smooth+"
    assert threshold_toggle.isChecked() is True
    assert threshold_value.value() == 8.5
    assert threshold_mask.isChecked() is True
    assert isolines.isChecked() is True
    assert isoline_count.value() == 9


def test_render_panel_session_restore_applies_one_blocked_batch(qapp) -> None:
    panel = RenderPanel.__new__(RenderPanel)
    background = QComboBox()
    background.addItems(panel._BG_PRESETS)
    outline = QCheckBox()
    target_outline = QCheckBox()
    axes = QCheckBox()

    def _pair(minimum: int, maximum: int) -> tuple[QSlider, QSpinBox]:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        return slider, spin

    edge_slider, edge_spin = _pair(1, 10)
    building_alpha = QSlider(Qt.Horizontal)
    building_alpha.setRange(0, 100)
    target_alpha = QSlider(Qt.Horizontal)
    target_alpha.setRange(0, 100)
    panel.widgets = {
        "bg_combo": background,
        "outline_cb": outline,
        "target_outline_cb": target_outline,
        "show_axes_cb": axes,
        "edge_line_width_slider": edge_slider,
        "edge_line_width_spin": edge_spin,
        "building_alpha_slider": building_alpha,
        "target_alpha_slider": target_alpha,
    }
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(
            line_width=True,
            wireframe=True,
            trajectories=True,
            axes=True,
        ),
        set_line_width=Mock(),
        set_point_size=Mock(),
        set_edge_line_width=Mock(),
        set_trajectory_line_width=Mock(),
        show_axes=Mock(),
    )
    scene_appearance = SimpleNamespace(
        set_edge_visibility=Mock(),
        set_background_preset=Mock(),
    )
    target_service = SimpleNamespace(set_target_edge_visibility=Mock())
    panel.parent = SimpleNamespace(
        renderer=renderer,
        scene_appearance_service=scene_appearance,
        target_service=target_service,
        ui_manager=SimpleNamespace(panels={}),
    )
    emitted: list[str] = []
    for widget in panel.widgets.values():
        signal = getattr(widget, "valueChanged", None)
        if signal is None:
            signal = getattr(widget, "toggled", None)
        if signal is None:
            signal = getattr(widget, "currentTextChanged", None)
        if signal is not None:
            signal.connect(lambda *_args: emitted.append("command"))

    panel.restore_session_state(
        {
            "background_preset": "Dark Gray",
            "show_edges": True,
            "show_target_edges": True,
            "show_axes": True,
            "point_size": 8,
            "mpc_line_width": 4,
            "edge_line_width": 3,
            "building_alpha": 0.65,
            "target_alpha": 0.35,
        }
    )

    assert emitted == []
    assert background.currentText() == "Dark Gray"
    assert outline.isChecked() is True
    assert target_outline.isChecked() is True
    assert axes.isChecked() is True
    assert edge_slider.value() == edge_spin.value() == 3
    assert building_alpha.value() == 65
    assert target_alpha.value() == 35
    renderer.set_point_size.assert_not_called()
    renderer.set_line_width.assert_not_called()
    renderer.set_edge_line_width.assert_called_once_with(3.0)
    renderer.set_trajectory_line_width.assert_not_called()
    renderer.show_axes.assert_called_once_with(True)
    scene_appearance.set_edge_visibility.assert_called_once_with(True)
    target_service.set_target_edge_visibility.assert_called_once_with(True)
    assert scene_appearance.set_background_preset.call_args.args[0] == "Dark Gray"


def test_session_service_preserves_mpc_size_keys_through_paths_owner(qapp) -> None:
    point_size = QDoubleSpinBox()
    point_size.setRange(0.1, 1000.0)
    point_size.setValue(8.5)
    line_width = QDoubleSpinBox()
    line_width.setRange(0.1, 1000.0)
    line_width.setValue(4.25)
    paths_restore = Mock()
    render_restore = Mock()

    viz = _DummyVisualizer()
    viz.ui_manager = SimpleNamespace(
        panels={
            "mpc": SimpleNamespace(
                widgets={
                    "point_size_spin": point_size,
                    "line_width_spin": line_width,
                },
                restore_session_state=paths_restore,
            ),
            "render": SimpleNamespace(
                widgets={},
                restore_session_state=render_restore,
            ),
        }
    )
    service = SessionService(viz)

    saved = service._get_rendering_state()

    assert saved["point_size"] == 8.5
    assert saved["mpc_line_width"] == 4.25

    service._restore_rendering_state(
        {
            "point_size": 12.75,
            "mpc_line_width": 6.5,
        }
    )

    paths_restore.assert_called_once_with(
        {
            "point_size": 12.75,
            "mpc_line_width": 6.5,
        }
    )
    render_restore.assert_called_once()


def test_get_camera_state_writes_orbit_only_payload() -> None:
    renderer = _DummyRenderer()
    renderer.get_camera_state = lambda: CameraState(
        eye=(1.0, 2.0, 3.0),
        lookat=(4.0, 5.0, 6.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=45.0,
    )
    viz = _DummyVisualizer()
    viz.renderer = renderer
    service = SessionService(viz)

    camera = service._get_camera_state()

    assert camera == {
        "format": CAMERA_SESSION_FORMAT,
        "mode": "overview",
        "eye": [1.0, 2.0, 3.0],
        "lookat": [4.0, 5.0, 6.0],
        "up": [0.0, 0.0, 1.0],
        "fov": 45.0,
    }


def test_restore_camera_state_uses_neutral_orbit_payload() -> None:
    renderer = _CameraRenderer()
    overview = _FakeButton()
    viz = SimpleNamespace(
        renderer=renderer,
        overview_mode_rb=overview,
        follow_mode_rb=_FakeButton(),
        pov_mode_rb=_FakeButton(),
    )
    service = SessionService(viz)

    service._restore_camera_state(
        {
            "format": CAMERA_SESSION_FORMAT,
            "mode": "overview",
            "eye": [1.0, 2.0, 3.0],
            "lookat": [4.0, 5.0, 6.0],
            "up": [0.0, 0.0, 1.0],
            "fov": 50.0,
        }
    )

    assert overview.checked is True
    assert overview.signalsBlocked() is False
    assert len(renderer.set_camera_state_calls) == 1
    restored = renderer.set_camera_state_calls[0]
    assert restored.eye == (1.0, 2.0, 3.0)
    assert restored.lookat == (4.0, 5.0, 6.0)


def test_restore_animation_frame_reports_pipeline_failure() -> None:
    viz = SimpleNamespace(
        force_update_next_frame=False,
        update_frame=Mock(return_value=False),
    )
    service = SessionService(viz)

    assert service._restore_animation_frame(7) is False

    viz.update_frame.assert_called_once_with(7)
    assert viz.force_update_next_frame is True


def test_restore_animation_frame_sizes_widgets_by_sparse_display_index(qapp) -> None:
    slider = QSlider()
    slider.setMaximum(0)
    frame_input = QSpinBox()
    frame_input.setMaximum(1)
    viz = SimpleNamespace(
        force_update_next_frame=False,
        step_slider=slider,
        frame_input=frame_input,
        get_animation_step_index=lambda frame: {100: 0, 200: 1}[frame],
        update_frame=Mock(return_value=True),
    )
    service = SessionService(viz)

    assert service._restore_animation_frame(200) is True

    assert slider.maximum() == 1
    assert frame_input.maximum() == 2
    viz.update_frame.assert_called_once_with(200)


def test_animation_session_state_excludes_transient_playback() -> None:
    viz = SimpleNamespace(app_state=create_initial_state(step=7))
    service = SessionService.__new__(SessionService)
    service.viz = viz

    assert service._get_animation_state() == {"current_frame": 7}
    assert "step" not in service._get_app_state()


def test_session_restore_then_mpc_toggle_uses_restored_app_state() -> None:
    source_viz = SimpleNamespace(
        app_state=create_initial_state(
            step=2,
            mpc_allowed_orders=frozenset({0}),
            mpc_allowed_types=frozenset({99}),
        ),
        node_coloring_mode="per_type",
    )
    source_service = SessionService.__new__(SessionService)
    source_service.viz = source_viz
    serialized_state = source_service._get_app_state()
    assert "step" not in serialized_state
    assert serialized_state["mpc_allowed_types"] == [99]

    viz = SimpleNamespace(
        app_state=create_initial_state(),
        node_coloring_mode="per_type",
        schedule_update=Mock(),
    )

    def set_state(**changes) -> None:
        viz.app_state = update_state(viz.app_state, **changes)

    viz.set_state = set_state
    service = SessionService.__new__(SessionService)
    service.viz = viz
    service._restore_app_state(serialized_state, frame=17)

    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    controller._invalidate_cache = Mock()
    controller.handle_mpc_order_filter_changed(1, True)
    controller.handle_mpc_type_filter_changed(1, True)

    assert viz.app_state.mpc_allowed_orders == frozenset({0, 1})
    assert viz.app_state.mpc_allowed_types == frozenset({1, 99})
    assert viz.app_state.step == 17
    assert not hasattr(viz, "mpc_allowed_orders")
    assert not hasattr(viz, "mpc_allowed_types")


def test_mpc_panel_sync_treats_empty_allow_lists_as_none() -> None:
    widgets = {
        **{f"order_{value}_cb": QCheckBox() for value in MPC_ORDER_VALUES},
        **{f"type_{value}_cb": QCheckBox() for value in MPC_TYPE_VALUES},
    }
    for widget in widgets.values():
        widget.setChecked(True)

    state = create_initial_state(
        mpc_allowed_orders=frozenset(),
        mpc_allowed_types=frozenset(),
    )
    service = SessionService.__new__(SessionService)
    service._sync_mpc_panel({"mpc": SimpleNamespace(widgets=widgets)}, state)

    assert all(not widget.isChecked() for widget in widgets.values())


def test_context_panel_sync_restores_scope_and_mpc_master_without_signals() -> None:
    from visualizer.src.panels.global_context_panel import GlobalContextPanel

    state = create_initial_state(
        selected_tx=5,
        selected_rx=3,
        mpc_layer_enabled=False,
        show_mpc_paths=True,
        show_mpc_bounce_points=False,
    )
    parent = SimpleNamespace(app_state=state, _layout_profile="auto")
    context = GlobalContextPanel(parent)
    group = context.create_panel()
    tx_dropdown = context.widgets["tx_dropdown"]
    rx_dropdown = context.widgets["rx_dropdown"]
    master = context.widgets["mpc_layer_cb"]
    tx_dropdown.addItem("TX1", 0)
    tx_dropdown.addItem("TX6", 5)
    rx_dropdown.addItem("RX1", 0)
    rx_dropdown.addItem("RX4", 3)

    emissions = []
    tx_dropdown.currentIndexChanged.connect(lambda index: emissions.append(("tx", index)))
    rx_dropdown.currentIndexChanged.connect(lambda index: emissions.append(("rx", index)))
    master.stateChanged.connect(lambda value: emissions.append(("mpc", value)))

    SessionService._sync_context_panel({"context": context}, state)

    assert tx_dropdown.currentData() == 5
    assert rx_dropdown.currentData() == 3
    assert not master.isChecked()
    assert emissions == []
    group.deleteLater()


def test_entry_state_snapshot_uses_stable_semantic_entity_ids() -> None:
    viz = SimpleNamespace(
        mesh_entries=[
            {
                "name": "Building A",
                "visible": False,
                "show_label": True,
            }
        ],
        target_entries=[
            {
                "target_name": "Pedestrian One",
                "visible": True,
                "show_label": False,
            }
        ],
        tx_entries=[{"visible": False, "show_label": False}],
        rx_entries=[{"visible": True, "show_label": True}],
    )

    snapshot = SessionService(viz)._get_entry_state_snapshot()

    assert snapshot == {
        "scene:building_a": {"visible": False, "show_label": True},
        "target:pedestrian_one": {"visible": True, "show_label": False},
        "node:tx_0": {"visible": False, "show_label": False},
        "node:rx_0": {"visible": True, "show_label": True},
    }
    assert all(set(intent) == {"visible", "show_label"} for intent in snapshot.values())


def test_restore_entry_state_updates_only_known_boolean_semantic_intent() -> None:
    target_entry = {
        "target_name": "Pedestrian One",
        "visible": True,
        "show_label": True,
    }
    tx_entry = {"visible": True, "show_label": True}
    viz = SimpleNamespace(
        mesh_entries=[],
        target_entries=[target_entry],
        tx_entries=[tx_entry],
        rx_entries=[],
    )
    service = SessionService(viz)

    deltas = service._restore_entry_state(
        {
            "target:pedestrian_one": {
                "visible": False,
                "show_label": False,
                "transform": [[1.0, 0.0, 0.0, 99.0]],
                "material": {"color": [1.0, 0.0, 0.0]},
            },
            "node:tx_0": {"visible": False, "show_label": "invalid"},
            "target:missing": {"visible": False, "show_label": False},
        }
    )

    assert target_entry["visible"] is False
    assert target_entry["show_label"] is False
    assert "transform" not in target_entry
    assert "material" not in target_entry
    assert tx_entry["visible"] is False
    assert tx_entry["show_label"] is True
    assert deltas == {
        "target:pedestrian_one": {
            "visible": False,
            "show_label": False,
        },
        "node:tx_0": {"visible": False},
    }


def test_refresh_entry_state_no_arg_retains_full_refresh_path() -> None:
    scene_entry = {"name": "Building", "visible": False, "show_label": True}
    target_entry = {
        "target_name": "Pedestrian",
        "entry_type": "target",
        "visible": True,
        "show_label": False,
    }
    tx_entry = {"visible": True, "show_label": True}
    appearance = SimpleNamespace(
        refresh_object_visibility_batch=Mock(return_value=True),
        set_building_label_visibility=Mock(),
    )
    node_service = SimpleNamespace(
        update_tx_rx_visibility=Mock(),
        update_target_label_visibility=Mock(),
    )
    object_panel = SimpleNamespace(sync_all_entry_states=Mock())
    viz = SimpleNamespace(
        mesh_entries=[scene_entry],
        target_entries=[target_entry],
        tx_entries=[tx_entry],
        rx_entries=[],
        object_appearance_service=appearance,
        node_service=node_service,
        ui_manager=SimpleNamespace(panels={"objects": object_panel}),
    )

    SessionService(viz)._refresh_entry_state()

    appearance.refresh_object_visibility_batch.assert_called_once_with(
        [scene_entry, target_entry],
        update_renderer=False,
    )
    appearance.set_building_label_visibility.assert_not_called()
    node_service.update_tx_rx_visibility.assert_called_once_with()
    node_service.update_target_label_visibility.assert_not_called()
    object_panel.sync_all_entry_states.assert_called_once_with()


def test_identical_entry_state_restore_performs_zero_sync_work() -> None:
    scene_entry = {"name": "Building", "visible": True, "show_label": False}
    target_entry = {"target_name": "Pedestrian", "visible": False, "show_label": True}
    node_entry = {"visible": True, "show_label": False}
    appearance = SimpleNamespace(
        refresh_object_visibility_batch=Mock(return_value=True),
        set_building_label_visibility=Mock(),
    )
    node_service = SimpleNamespace(
        update_tx_rx_visibility=Mock(),
        update_target_label_visibility=Mock(),
    )
    object_panel = SimpleNamespace(sync_all_entry_states=Mock())
    viz = SimpleNamespace(
        mesh_entries=[scene_entry],
        target_entries=[target_entry],
        tx_entries=[node_entry],
        rx_entries=[],
        object_appearance_service=appearance,
        node_service=node_service,
        ui_manager=SimpleNamespace(panels={"objects": object_panel}),
    )
    service = SessionService(viz)

    deltas = service._restore_entry_state(
        {
            "scene:building": {"visible": True, "show_label": False},
            "target:pedestrian": {"visible": False, "show_label": True},
            "node:tx_0": {"visible": True, "show_label": False},
        }
    )
    service._refresh_entry_state(deltas)

    assert deltas == {}
    appearance.refresh_object_visibility_batch.assert_not_called()
    appearance.set_building_label_visibility.assert_not_called()
    node_service.update_tx_rx_visibility.assert_not_called()
    node_service.update_target_label_visibility.assert_not_called()
    object_panel.sync_all_entry_states.assert_not_called()


def test_refresh_entry_state_routes_only_changed_subsets() -> None:
    hidden_scene = {"name": "Hidden Building", "visible": False, "show_label": True}
    relabeled_scene = {"name": "Relabeled Building", "visible": True, "show_label": False}
    hidden_target = {
        "target_name": "Hidden Target",
        "entry_type": "target",
        "visible": False,
        "show_label": True,
    }
    relabeled_target = {
        "target_name": "Relabeled Target",
        "entry_type": "target",
        "visible": True,
        "show_label": False,
    }
    tx_entry = {"visible": False, "show_label": False}
    appearance = SimpleNamespace(
        refresh_object_visibility_batch=Mock(return_value=True),
        set_building_label_visibility=Mock(),
    )
    node_service = SimpleNamespace(
        update_tx_rx_visibility=Mock(),
        update_target_label_visibility=Mock(),
    )
    object_panel = SimpleNamespace(sync_all_entry_states=Mock())
    viz = SimpleNamespace(
        mesh_entries=[hidden_scene, relabeled_scene],
        target_entries=[hidden_target, relabeled_target],
        tx_entries=[tx_entry],
        rx_entries=[],
        object_appearance_service=appearance,
        node_service=node_service,
        ui_manager=SimpleNamespace(panels={"objects": object_panel}),
    )

    SessionService(viz)._refresh_entry_state(
        {
            "scene:hidden_building": {"visible": False},
            "scene:relabeled_building": {"show_label": False},
            "target:hidden_target": {"visible": False},
            "target:relabeled_target": {"show_label": False},
            "node:tx_0": {"visible": False, "show_label": False},
        }
    )

    appearance.refresh_object_visibility_batch.assert_called_once_with(
        [hidden_scene, hidden_target],
        update_renderer=False,
    )
    appearance.set_building_label_visibility.assert_called_once_with(
        relabeled_scene,
        False,
        update_renderer=False,
    )
    node_service.update_tx_rx_visibility.assert_called_once_with()
    node_service.update_target_label_visibility.assert_called_once_with()
    object_panel.sync_all_entry_states.assert_called_once_with()


def test_label_only_entry_deltas_do_not_enter_visibility_batch() -> None:
    scene_entry = {"name": "Building", "visible": True, "show_label": True}
    target_entry = {
        "target_name": "Pedestrian",
        "entry_type": "target",
        "visible": True,
        "show_label": False,
    }
    appearance = SimpleNamespace(
        refresh_object_visibility_batch=Mock(return_value=True),
        set_building_label_visibility=Mock(),
    )
    node_service = SimpleNamespace(
        update_tx_rx_visibility=Mock(),
        update_target_label_visibility=Mock(),
    )
    viz = SimpleNamespace(
        mesh_entries=[scene_entry],
        target_entries=[target_entry],
        tx_entries=[],
        rx_entries=[],
        object_appearance_service=appearance,
        node_service=node_service,
        ui_manager=None,
    )

    SessionService(viz)._refresh_entry_state(
        {
            "scene:building": {"show_label": True},
            "target:pedestrian": {"show_label": False},
        }
    )

    appearance.refresh_object_visibility_batch.assert_not_called()
    appearance.set_building_label_visibility.assert_called_once_with(
        scene_entry,
        True,
        update_renderer=False,
    )
    node_service.update_tx_rx_visibility.assert_not_called()
    node_service.update_target_label_visibility.assert_called_once_with()


def test_session_round_trip_uses_frame_pose_not_transient_render_transform(tmp_path) -> None:
    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text("schema_version: 1\n")
    session_path = tmp_path / "session.json"

    payload = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    transient_pose = Transform.from_translation([91.0, 92.0, 93.0])
    semantic_frame_pose = Transform.from_translation([7.0, 8.0, 9.0])
    marker = RenderObjectState(
        id="node:tx_0::marker",
        payload=payload,
        world_transform=transient_pose,
        metadata={"type": "node_marker", "kind": "tx", "index": 0},
    )
    saved_camera = CameraState(
        eye=(10.0, 11.0, 12.0),
        lookat=(1.0, 2.0, 3.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=45.0,
    )
    renderer = _RoundTripRenderer(saved_camera)
    reset_live_preview = Mock()
    restored_frames: list[int] = []

    viz = SimpleNamespace(
        renderer=renderer,
        current_scenario_path=scenario_yaml,
        app_state=create_initial_state(step=4),
        mesh_entries=[],
        target_entries=[],
        target_labels=[],
        tx_markers=[marker],
        tx_labels=[],
        tx_entries=[{"visible": False, "show_label": False}],
        rx_markers=[],
        rx_labels=[],
        rx_entries=[],
        coverage_mesh=None,
        current_building_alpha=1.0,
        current_target_alpha=1.0,
        node_coloring_mode="per_type",
        live_preview_service=SimpleNamespace(reset=reset_live_preview),
        _scene_only_mode=False,
        _session_restore_in_progress=False,
        cancel_scheduled_update=Mock(),
        ui_manager=None,
    )

    def set_state(**changes) -> None:
        viz.app_state = update_state(viz.app_state, **changes)

    def update_frame(frame: int) -> bool:
        restored_frames.append(int(frame))
        assert viz.tx_entries[0]["visible"] is False
        assert viz.tx_entries[0]["show_label"] is False
        viz.app_state = update_state(viz.app_state, step=int(frame))
        marker.world_transform = semantic_frame_pose
        renderer.ensure_object(marker.to_render_object())
        return True

    viz.set_state = set_state
    viz.update_frame = update_frame

    service = SessionService(viz)
    service.save_session(path=session_path)

    serialized = json.loads(session_path.read_text())
    assert SESSION_VERSION == "6.0"
    assert serialized["version"] == SESSION_VERSION
    assert serialized["animation"] == {"current_frame": 4}
    assert "visual_state" not in serialized
    marker_state = serialized["entry_state"]["node:tx_0"]
    assert marker_state == {"visible": False, "show_label": False}

    marker.world_transform = Transform.from_translation([-1.0, -2.0, -3.0])
    viz.app_state = update_state(viz.app_state, step=0)
    viz.tx_entries[0]["visible"] = True
    viz.tx_entries[0]["show_label"] = True
    renderer.objects.clear()

    assert service.load_session(session_path) is True

    reset_live_preview.assert_called_once_with()
    assert restored_frames == [4]
    assert viz.app_state.step == 4
    np.testing.assert_allclose(marker.world_transform.matrix, semantic_frame_pose.matrix)
    assert not np.array_equal(marker.world_transform.matrix, transient_pose.matrix)
    assert len(renderer.set_camera_state_calls) == 1
    restored_camera = renderer.set_camera_state_calls[0]
    assert restored_camera.eye == saved_camera.eye
    assert restored_camera.lookat == saved_camera.lookat


def test_get_app_state_forces_rf_xray_inactive() -> None:
    viz = _DummyVisualizer()
    viz.app_state = create_initial_state(
        show_rf_xray=True,
        rf_xray_mode="material_properties",
        rf_xray_property="conductivity",
        rf_xray_opacity=1.0,
    )
    service = SessionService(viz)

    payload = service._get_app_state()

    assert payload["show_rf_xray"] is False
    assert payload["rf_xray_mode"] == "material_properties"
    assert payload["rf_xray_property"] == "conductivity"
    assert payload["rf_xray_opacity"] == 1.0


def test_get_app_state_fallback_serializes_actual_mpc_visibility() -> None:
    viz = _DummyVisualizer()
    viz.app_state = SimpleNamespace(
        step=7,
        color_mode="delay",
        show_coverage=False,
        mpc_visibility=MpcVisibility(
            enabled=False,
            paths=True,
            bounce_points=False,
        ),
    )
    service = SessionService(viz)

    payload = service._get_app_state()

    assert payload["mpc_visibility"] == {
        "enabled": False,
        "paths": True,
        "bounce_points": False,
    }


def test_restore_app_state_forces_rf_xray_inactive() -> None:
    class _StatefulVisualizer(_DummyVisualizer):
        def __init__(self) -> None:
            super().__init__()
            self.app_state = create_initial_state()

        def set_state(self, **kwargs) -> None:
            self.app_state = update_state(self.app_state, **kwargs)

    viz = _StatefulVisualizer()
    service = SessionService(viz)
    payload = create_initial_state(
        show_rf_xray=True,
        rf_xray_mode="material_properties",
        rf_xray_property="thickness",
    ).to_dict()
    service._restore_app_state(payload)

    assert viz.app_state.show_rf_xray is False
    assert viz.app_state.rf_xray_mode == "material_properties"
    assert viz.app_state.rf_xray_property == "thickness"


def test_sync_ui_with_state_refreshes_materials_rf_xray_controls() -> None:
    calls = []
    viz = _DummyVisualizer()
    viz.app_state = create_initial_state(show_rf_xray=False)
    viz.ui_manager = SimpleNamespace(
        panels={
            "materials": SimpleNamespace(_sync_rf_xray_controls=lambda: calls.append("materials"))
        }
    )
    service = SessionService(viz)

    service._sync_ui_with_state()

    assert calls == ["materials"]


def _write_session_payload(
    path,
    *,
    scenario_path,
    camera=None,
    version=SESSION_VERSION,
    created_at="2026-06-08T00:00:00",
    frame=0,
    snapshot_kind=None,
) -> None:
    payload = {
        "version": version,
        "created_at": created_at,
        "scenario_path": str(scenario_path),
        "camera": camera or {},
        "app_state": create_initial_state().to_dict(),
        "animation": {"current_frame": frame},
        "rendering": {},
        "entry_state": {},
    }
    if snapshot_kind is not None:
        payload["snapshot_kind"] = snapshot_kind
    path.write_text(json.dumps(payload))


def test_load_session_resets_transient_live_preview_state(tmp_path) -> None:
    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text("schema_version: 1\n")
    session_path = tmp_path / "session.json"
    _write_session_payload(session_path, scenario_path=scenario_yaml)

    reset_live_preview = Mock()
    viz = SimpleNamespace(
        renderer=_DummyRenderer(),
        current_scenario_path=scenario_yaml,
        live_preview_service=SimpleNamespace(reset=reset_live_preview),
        app_state=create_initial_state(),
        _scene_only_mode=True,
        _session_restore_in_progress=False,
        cancel_scheduled_update=Mock(),
        schedule_update=Mock(),
        ui_manager=None,
    )
    viz.set_state = lambda **changes: setattr(
        viz,
        "app_state",
        update_state(viz.app_state, **changes),
    )
    service = SessionService(viz)

    assert service.load_session(session_path, skip_camera=True) is True
    reset_live_preview.assert_called_once_with()


def test_session_reset_drops_inspection_state_but_keeps_scenario_profile(tmp_path) -> None:
    manual = {
        "highlighted": True,
        "_visual_material_binding": VisualMaterialBinding(
            source=VisualMaterialSource.MANUAL,
            material_type="brick",
            overrides={"roughness": 0.2},
        ),
    }
    profile_binding = VisualMaterialBinding(
        source=VisualMaterialSource.PROFILE,
        material_type="skin",
        preset="Skin",
    )
    profile = {"highlighted": True, "_visual_material_binding": profile_binding}
    follow_override = {
        "highlighted": True,
        "_visual_material_binding": VisualMaterialBinding(),
    }
    modes = MaterialModeService()
    modes.set_mode("brick", MaterialDisplayMode.HIDDEN)
    viz = SimpleNamespace(
        selected_objects={"scene:wall::mesh"},
        material_mode_service=modes,
        material_pbr_service=SimpleNamespace(overrides={"brick": {"roughness": 0.2}}),
        mesh_entries=[manual, profile, follow_override],
        target_entries=[],
    )
    service = SessionService(viz)
    service.session_dir = tmp_path

    service._reset_transient_appearance_state()

    assert viz.selected_objects == set()
    assert modes.modes == {}
    assert viz.material_pbr_service.overrides == {}
    assert all(entry["highlighted"] is False for entry in viz.mesh_entries)
    assert "_visual_material_binding" not in manual
    assert "_visual_material_binding" not in follow_override
    assert profile["_visual_material_binding"] is profile_binding


def test_scenario_root_identity_treats_folder_and_yaml_as_equivalent(tmp_path) -> None:
    scenario_root = tmp_path / "Munich Scenario"
    scenario_root.mkdir()
    scenario_yaml = scenario_root / "scenario.yaml"
    scenario_yaml.write_text("schema_version: 1\n")

    assert normalize_scenario_root(scenario_root) == scenario_root.resolve()
    assert normalize_scenario_root(scenario_yaml) == scenario_root.resolve()
    service = SessionService(SimpleNamespace())
    assert service._paths_match(scenario_root, scenario_yaml) is True


def test_workspace_summary_reads_typed_metadata_and_skips_invalid_files(tmp_path) -> None:
    scenario_root = tmp_path / "etoile_target_context"
    scenario_root.mkdir()
    scenario_yaml = scenario_root / "scenario.yaml"
    scenario_yaml.write_text("schema_version: 1\n")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    manual = sessions / "etoile_workspace.json"
    _write_session_payload(
        manual,
        scenario_path=scenario_yaml,
        created_at="2026-07-18T14:30:00",
        frame=17,
        snapshot_kind="manual",
    )
    unreadable = sessions / "broken.json"
    unreadable.write_text("{not-json")
    invalid_utf8 = sessions / "invalid_utf8.json"
    invalid_utf8.write_bytes(b"\xff\xfe")
    incomplete = sessions / "incomplete.json"
    _write_session_payload(incomplete, scenario_path=scenario_root)
    incomplete_payload = json.loads(incomplete.read_text())
    incomplete_payload.pop("animation")
    incomplete.write_text(json.dumps(incomplete_payload), encoding="utf-8")
    unknown = sessions / "unknown.json"
    _write_session_payload(
        unknown,
        scenario_path=scenario_root,
        snapshot_kind="future-kind",
    )
    os.utime(unreadable, None)

    summary = read_workspace_summary(manual)

    assert summary == WorkspaceSnapshotSummary(
        path=manual.resolve(),
        scenario_root=scenario_root.resolve(),
        scenario_name="etoile_target_context",
        created_at=datetime(2026, 7, 18, 14, 30),
        frame=17,
        is_autosave=False,
    )
    service = SessionService(SimpleNamespace())
    service.session_dir = sessions
    assert service.list_workspace_summaries(max_count=1) == [summary]
    assert read_workspace_summary(unreadable) is None
    assert read_workspace_summary(invalid_utf8) is None
    assert read_workspace_summary(incomplete) is None
    assert read_workspace_summary(unknown) is None


def test_save_session_requires_a_loaded_scenario(tmp_path) -> None:
    service = SessionService(SimpleNamespace(current_scenario_path=None))
    service.session_dir = tmp_path

    with pytest.raises(ValueError, match="Load a scenario"):
        service.save_session()

    assert list(tmp_path.glob("*.json")) == []


def test_exit_autosave_quietly_skips_when_no_scenario_is_open(tmp_path) -> None:
    service = SessionService(SimpleNamespace(current_scenario_path=None))
    service.session_dir = tmp_path

    assert service.auto_save_on_exit() is None
    assert list(tmp_path.glob("*.json")) == []


def test_autosave_is_atomic_deterministic_and_prunes_only_recognized_duplicates(
    tmp_path,
) -> None:
    scenario_root = tmp_path / "Munich Filtering Dense"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    other_root = tmp_path / "Etoile"
    other_root.mkdir()
    (other_root / "scenario.yaml").write_text("schema_version: 1\n")
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    viz = SimpleNamespace(current_scenario_path=scenario_root)
    service = SessionService(viz)
    service.session_dir = sessions

    old_autosave = sessions / "munich_autosave_20260717_120000.json"
    _write_session_payload(
        old_autosave,
        scenario_path=scenario_root / "scenario.yaml",
        created_at="2026-07-17T12:00:00",
    )
    # Retention is anchored to the file just written, not wall-clock ordering.
    os.utime(old_autosave, (2_000_000_000.0, 2_000_000_000.0))
    other_autosave = sessions / "etoile_autosave_20260717_120000.json"
    _write_session_payload(other_autosave, scenario_path=other_root)
    manual = service.save_session(name="keep_autosave_notes")
    legacy_named_manual = sessions / "notes_autosave.json"
    _write_session_payload(legacy_named_manual, scenario_path=scenario_root)
    unreadable = sessions / "damaged_autosave_20260717_120000.json"
    unreadable.write_text("not-json")
    unknown = sessions / "unknown_kind.json"
    _write_session_payload(
        unknown,
        scenario_path=scenario_root,
        snapshot_kind="unknown",
    )

    first = service.save_session()
    second = service.save_session()

    assert first == second
    assert first.name.startswith("munich-filtering-dense_autosave_")
    assert first.exists()
    assert json.loads(first.read_text())["scenario_path"] == str(scenario_root.resolve())
    assert json.loads(first.read_text())["snapshot_kind"] == "autosave"
    assert not old_autosave.exists()
    assert other_autosave.exists()
    assert manual.exists()
    assert legacy_named_manual.exists()
    assert read_workspace_summary(legacy_named_manual).is_autosave is False
    assert json.loads(manual.read_text())["snapshot_kind"] == "manual"
    assert unreadable.exists()
    assert unknown.exists()
    assert list(sessions.glob("*.tmp")) == []
    matching_autosaves = [
        summary
        for summary in service.list_workspace_summaries()
        if summary.is_autosave and summary.scenario_root == scenario_root.resolve()
    ]
    assert [summary.path for summary in matching_autosaves] == [first.resolve()]


def _workspace_restore_visualizer(scenario_path, *, scene_only=False):
    renderer = _CameraRenderer()
    viz = SimpleNamespace(
        renderer=renderer,
        current_scenario_path=scenario_path,
        app_state=create_initial_state(),
        mesh_entries=[],
        target_entries=[],
        tx_entries=[],
        rx_entries=[],
        selected_objects=set(),
        node_coloring_mode="per_type",
        ui_manager=None,
        _scene_only_mode=scene_only,
        _session_restore_in_progress=False,
        force_update_next_frame=False,
        cancel_scheduled_update=Mock(),
        schedule_update=Mock(),
        update_frame=Mock(return_value=True),
        _flush_update=Mock(),
    )

    def set_state(**changes) -> None:
        viz.app_state = update_state(viz.app_state, **changes)

    viz.set_state = set_state
    return viz


@pytest.mark.parametrize("missing_section", ["app_state", "rendering", "entry_state"])
def test_workspace_restore_requires_complete_v6_state_sections(
    tmp_path,
    missing_section,
) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "incomplete_workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root, frame=7)
    payload = json.loads(snapshot.read_text())
    payload.pop(missing_section)
    snapshot.write_text(json.dumps(payload))
    viz = _workspace_restore_visualizer(scenario_root)
    prior_state = viz.app_state

    assert SessionService(viz).load_session(snapshot) is False

    assert viz.app_state is prior_state
    viz.cancel_scheduled_update.assert_not_called()
    viz.update_frame.assert_not_called()
    viz.schedule_update.assert_not_called()
    assert viz._session_restore_in_progress is False


def test_malformed_workspace_app_state_fails_before_live_state_mutation(tmp_path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "malformed_workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root, frame=7)
    payload = json.loads(snapshot.read_text())
    payload["app_state"] = {"not_an_app_state_field": True}
    snapshot.write_text(json.dumps(payload))
    viz = _workspace_restore_visualizer(scenario_root)
    prior_state = viz.app_state

    assert SessionService(viz).load_session(snapshot) is False

    assert viz.app_state is prior_state
    viz.cancel_scheduled_update.assert_not_called()
    viz.update_frame.assert_not_called()


def test_app_state_apply_failure_rolls_back_prior_workspace(tmp_path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root, frame=17)
    payload = json.loads(snapshot.read_text())
    payload["app_state"] = create_initial_state(color_mode="power").to_dict()
    snapshot.write_text(json.dumps(payload))

    viz = _workspace_restore_visualizer(scenario_root)
    viz.app_state = update_state(viz.app_state, step=3, color_mode="delay")
    apply_calls = 0

    def set_state(**changes) -> None:
        nonlocal apply_calls
        apply_calls += 1
        if apply_calls == 1:
            raise RuntimeError("state owner rejected target workspace")
        viz.app_state = update_state(viz.app_state, **changes)

    viz.set_state = set_state

    assert SessionService(viz).load_session(snapshot) is False

    assert apply_calls == 2
    assert viz.app_state.step == 3
    assert viz.app_state.color_mode == "delay"
    viz.update_frame.assert_called_once_with(3)
    assert viz._session_restore_in_progress is False


def test_failed_same_scenario_restore_reapplies_default_alpha(tmp_path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root, frame=17)
    payload = json.loads(snapshot.read_text())
    payload["rendering"] = {"building_alpha": 0.4, "target_alpha": 0.3}
    snapshot.write_text(json.dumps(payload))

    viz = _workspace_restore_visualizer(scenario_root)
    viz.current_building_alpha = 1.0
    viz.current_target_alpha = 1.0
    building_calls: list[float] = []
    target_calls: list[float] = []

    def set_building_alpha(alpha: float) -> None:
        viz.current_building_alpha = float(alpha)
        building_calls.append(float(alpha))

    def set_target_alpha(alpha: float) -> None:
        viz.current_target_alpha = float(alpha)
        target_calls.append(float(alpha))

    viz.scene_appearance_service = SimpleNamespace(
        set_building_transparency=set_building_alpha,
        set_target_transparency=set_target_alpha,
    )
    viz.update_frame = Mock(side_effect=[False, True])

    assert SessionService(viz).load_session(snapshot) is False

    assert building_calls == [0.4, 1.0]
    assert target_calls == [0.3, 1.0]
    assert viz.current_building_alpha == 1.0
    assert viz.current_target_alpha == 1.0


def test_workspace_restore_rejects_reentrant_entry_without_clearing_owner_guard(
    tmp_path,
) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root)
    viz = _workspace_restore_visualizer(scenario_root)
    viz._session_restore_in_progress = True

    assert SessionService(viz).load_session(snapshot) is False

    assert viz._session_restore_in_progress is True
    viz.cancel_scheduled_update.assert_not_called()
    viz.update_frame.assert_not_called()


def test_workspace_restore_rejects_entry_after_shutdown_begins(tmp_path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root)
    viz = _workspace_restore_visualizer(scenario_root, scene_only=True)
    viz._shutdown_started = True

    assert SessionService(viz).load_session(snapshot) is False

    assert viz._session_restore_in_progress is False
    viz.cancel_scheduled_update.assert_not_called()
    viz.schedule_update.assert_not_called()


def test_scene_only_restore_does_not_schedule_when_shutdown_starts_during_apply(
    tmp_path,
) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root)
    viz = _workspace_restore_visualizer(scenario_root, scene_only=True)
    apply_state = viz.set_state

    def begin_shutdown(**changes) -> None:
        apply_state(**changes)
        viz._shutdown_started = True

    viz.set_state = begin_shutdown

    assert SessionService(viz).load_session(snapshot) is False

    viz.schedule_update.assert_not_called()
    assert viz._session_restore_in_progress is False


def test_missing_workspace_scenario_fails_before_mutating_live_state(tmp_path) -> None:
    current_root = tmp_path / "current"
    current_root.mkdir()
    (current_root / "scenario.yaml").write_text("schema_version: 1\n")
    missing_root = tmp_path / "missing"
    snapshot = tmp_path / "missing_workspace.json"
    _write_session_payload(snapshot, scenario_path=missing_root, frame=17)
    viz = _workspace_restore_visualizer(current_root)
    viz.live_preview_service = SimpleNamespace(reset=Mock())
    viz.open_scenario = Mock()
    service = SessionService(viz)

    with pytest.raises(FileNotFoundError, match="missing.*scenario.yaml"):
        service.load_session(snapshot)

    viz.open_scenario.assert_not_called()
    viz.live_preview_service.reset.assert_not_called()
    viz.cancel_scheduled_update.assert_not_called()
    viz.update_frame.assert_not_called()
    viz.schedule_update.assert_not_called()
    assert viz._session_restore_in_progress is False


def test_invalid_workspace_encoding_or_missing_frame_fails_before_restore(tmp_path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    viz = _workspace_restore_visualizer(scenario_root)
    service = SessionService(viz)

    invalid_utf8 = tmp_path / "invalid_utf8.json"
    invalid_utf8.write_bytes(b"\xff\xfe")
    assert service.load_session(invalid_utf8) is False

    incomplete = tmp_path / "incomplete.json"
    _write_session_payload(incomplete, scenario_path=scenario_root)
    payload = json.loads(incomplete.read_text())
    payload.pop("animation")
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    assert service.load_session(incomplete) is False

    viz.cancel_scheduled_update.assert_not_called()
    viz.update_frame.assert_not_called()


def test_predecoded_workspace_remains_authoritative_after_file_is_removed(tmp_path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot_path = tmp_path / "workspace.json"
    _write_session_payload(
        snapshot_path,
        scenario_path=scenario_root,
        frame=17,
        snapshot_kind="autosave",
    )
    snapshot = read_workspace_snapshot(snapshot_path)
    assert snapshot is not None
    snapshot_path.unlink()

    viz = _workspace_restore_visualizer(scenario_root)
    service = SessionService(viz)

    assert service.load_session(snapshot) is True
    viz.update_frame.assert_called_once_with(17)


def test_failed_scenario_open_is_not_treated_as_workspace_restore(tmp_path) -> None:
    current_root = tmp_path / "current"
    current_root.mkdir()
    (current_root / "scenario.yaml").write_text("schema_version: 1\n")
    saved_root = tmp_path / "saved"
    saved_root.mkdir()
    (saved_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "saved_workspace.json"
    _write_session_payload(snapshot, scenario_path=saved_root, frame=17)
    viz = _workspace_restore_visualizer(current_root)
    viz.open_scenario = Mock(return_value=None)
    service = SessionService(viz)

    assert service.load_session(snapshot) is False

    viz.open_scenario.assert_called_once_with(
        str(saved_root.resolve()),
        pending_camera=None,
        autorun_initial_frame=False,
    )
    viz.cancel_scheduled_update.assert_not_called()
    viz.update_frame.assert_not_called()


def test_failed_saved_frame_pipeline_reports_workspace_restore_failure(tmp_path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root, frame=17)
    viz = _workspace_restore_visualizer(scenario_root)
    viz.update_frame = Mock(side_effect=[False, True])
    service = SessionService(viz)

    assert service.load_session(snapshot) is False

    assert viz.update_frame.call_args_list == [call(17), call(0)]
    viz.schedule_update.assert_not_called()


def test_different_local_workspace_prevalidates_frame_before_replacing_active_scenario(
    tmp_path,
    monkeypatch,
) -> None:
    current_root = tmp_path / "current"
    current_root.mkdir()
    (current_root / "scenario.yaml").write_text("schema_version: 1\n")
    saved_root = tmp_path / "saved"
    saved_root.mkdir()
    (saved_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "saved_workspace.json"
    _write_session_payload(snapshot, scenario_path=saved_root, frame=17)

    preflight = Mock(
        return_value=SimpleNamespace(
            scenario=SimpleNamespace(data_mode="files"),
        )
    )
    source = SimpleNamespace(
        has_frame=Mock(return_value=False),
        close=Mock(),
    )
    viz = _workspace_restore_visualizer(current_root)
    viz.scenario_loader_service = SimpleNamespace(preflight_scenario=preflight)
    viz.open_scenario = Mock()
    service = SessionService(viz)
    monkeypatch.setattr(
        "visualizer.src.services.session_service.make_frame_source",
        Mock(return_value=source),
    )

    assert service.load_session(snapshot) is False

    assert normalize_scenario_root(viz.current_scenario_path) == current_root.resolve()
    preflight.assert_called_once_with(str(saved_root.resolve()))
    source.has_frame.assert_called_once_with(17)
    source.close.assert_called_once_with()
    viz.open_scenario.assert_not_called()
    viz.cancel_scheduled_update.assert_not_called()


def test_reopened_scene_only_workspace_fails_and_reopens_prior_scenario(tmp_path) -> None:
    current_root = tmp_path / "current"
    current_root.mkdir()
    (current_root / "scenario.yaml").write_text("schema_version: 1\n")
    saved_root = tmp_path / "saved"
    saved_root.mkdir()
    (saved_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "saved_workspace.json"
    _write_session_payload(snapshot, scenario_path=saved_root, frame=17)

    viz = _workspace_restore_visualizer(current_root)
    open_calls: list[Path] = []

    def open_scenario(path: str, **_kwargs) -> SimpleNamespace:
        root = Path(path).resolve()
        open_calls.append(root)
        viz.current_scenario_path = root
        viz._scene_only_mode = root == saved_root.resolve()
        viz.frame_source = None
        return SimpleNamespace(succeeded=True, scenario_root=root)

    viz.open_scenario = open_scenario
    service = SessionService(viz)

    assert service.load_session(snapshot) is False

    assert open_calls == [saved_root.resolve(), current_root.resolve()]
    assert normalize_scenario_root(viz.current_scenario_path) == current_root.resolve()
    assert viz._scene_only_mode is False
    viz.update_frame.assert_called_once_with(0)


def test_post_open_apply_failure_reopens_and_reapplies_prior_workspace(tmp_path) -> None:
    current_root = tmp_path / "current"
    current_root.mkdir()
    (current_root / "scenario.yaml").write_text("schema_version: 1\n")
    saved_root = tmp_path / "saved"
    saved_root.mkdir()
    (saved_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "saved_workspace.json"
    _write_session_payload(snapshot, scenario_path=saved_root, frame=17)
    payload = json.loads(snapshot.read_text())
    payload["rendering"] = {"trigger_failure": True}
    snapshot.write_text(json.dumps(payload))

    viz = _workspace_restore_visualizer(current_root)
    viz.app_state = update_state(viz.app_state, step=3, color_mode="delay")
    open_calls: list[Path] = []
    update_calls: list[tuple[Path, int]] = []

    def open_scenario(path: str, **_kwargs) -> SimpleNamespace:
        root = Path(path).resolve()
        open_calls.append(root)
        viz.current_scenario_path = root
        viz._scene_only_mode = False
        viz.app_state = create_initial_state()
        viz.frame_source = SimpleNamespace(has_frame=lambda _frame: True)
        return SimpleNamespace(succeeded=True, scenario_root=root)

    def update_frame(frame: int) -> bool:
        root = normalize_scenario_root(viz.current_scenario_path)
        assert root is not None
        update_calls.append((root, frame))
        viz.set_state(step=frame)
        return True

    viz.open_scenario = open_scenario
    viz.update_frame = update_frame
    service = SessionService(viz)
    restore_rendering_state = service._restore_rendering_state

    def fail_target_rendering_state(rendering: dict) -> None:
        if rendering.get("trigger_failure"):
            raise RuntimeError("target rendering state rejected")
        restore_rendering_state(rendering)

    service._restore_rendering_state = fail_target_rendering_state

    assert service.load_session(snapshot) is False

    assert open_calls == [saved_root.resolve(), current_root.resolve()]
    assert normalize_scenario_root(viz.current_scenario_path) == current_root.resolve()
    assert viz.app_state.step == 3
    assert viz.app_state.color_mode == "delay"
    assert update_calls == [(current_root.resolve(), 3)]
    assert viz._session_restore_in_progress is False


def test_different_scenario_workspace_opens_once_then_updates_saved_frame_once(tmp_path) -> None:
    current_root = tmp_path / "current"
    current_root.mkdir()
    (current_root / "scenario.yaml").write_text("schema_version: 1\n")
    saved_root = tmp_path / "saved"
    saved_root.mkdir()
    (saved_root / "scenario.yaml").write_text("schema_version: 1\n")
    camera = {
        "format": CAMERA_SESSION_FORMAT,
        "mode": "overview",
        "eye": [1.0, 2.0, 3.0],
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "fov": 45.0,
    }
    snapshot = tmp_path / "saved_workspace.json"
    _write_session_payload(snapshot, scenario_path=saved_root, camera=camera, frame=17)
    viz = _workspace_restore_visualizer(current_root)
    open_calls = []

    def open_scenario(
        path: str,
        pending_camera=None,
        autorun_initial_frame: bool = True,
    ) -> SimpleNamespace:
        open_calls.append((path, pending_camera, autorun_initial_frame))
        viz.current_scenario_path = path
        return SimpleNamespace(succeeded=True, scenario_root=path)

    viz.open_scenario = open_scenario
    service = SessionService(viz)

    assert service.load_session(snapshot) is True

    assert open_calls == [(str(saved_root.resolve()), camera, False)]
    viz.update_frame.assert_called_once_with(17)
    assert viz.force_update_next_frame is True
    assert len(viz.renderer.set_camera_state_calls) == 0
    assert viz.cancel_scheduled_update.call_count == 2
    viz.schedule_update.assert_not_called()
    viz._flush_update.assert_not_called()


def test_same_scenario_yaml_and_folder_restore_without_reopening(tmp_path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    scenario_yaml = scenario_root / "scenario.yaml"
    scenario_yaml.write_text("schema_version: 1\n")
    camera = {
        "format": CAMERA_SESSION_FORMAT,
        "mode": "overview",
        "eye": [1.0, 2.0, 3.0],
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "fov": 45.0,
    }
    snapshot = tmp_path / "workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root, camera=camera, frame=5)
    viz = _workspace_restore_visualizer(scenario_yaml)
    viz.open_scenario = Mock()
    service = SessionService(viz)

    assert service.load_session(snapshot) is True

    viz.open_scenario.assert_not_called()
    viz.update_frame.assert_called_once_with(5)
    assert len(viz.renderer.set_camera_state_calls) == 1
    viz.schedule_update.assert_not_called()
    viz._flush_update.assert_not_called()


def test_scene_only_workspace_schedules_one_refresh_without_frame_or_flush(tmp_path) -> None:
    scenario_root = tmp_path / "scene_only"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 1\n")
    snapshot = tmp_path / "scene_workspace.json"
    _write_session_payload(snapshot, scenario_path=scenario_root, frame=12)
    viz = _workspace_restore_visualizer(scenario_root, scene_only=True)
    service = SessionService(viz)

    assert service.load_session(snapshot, skip_camera=True) is True

    viz.update_frame.assert_not_called()
    viz.schedule_update.assert_called_once_with()
    viz._flush_update.assert_not_called()
    assert viz.force_update_next_frame is True
