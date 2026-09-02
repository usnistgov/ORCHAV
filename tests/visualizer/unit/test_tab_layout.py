"""Tests for the tab-based panel layout with real Qt widgets."""

import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.session_service import SessionService
from visualizer.src.services.trajectory_load_service import TrajectoryLoadCoordinator
from visualizer.src.state import AppState, MpcVisibility, create_initial_state, update_state
from visualizer.src.utils.antenna_utils import spacing_wavelengths_to_m

# Keep the Qt class available while panel-module bindings are adjusted.
_REAL_QLABEL = QLabel


@pytest.fixture(autouse=True)
def mock_pyqt_widgets(monkeypatch, qapp):
    """Bind the real QLabel across loaded panel modules."""
    _ = qapp
    monkeypatch.setattr(QtWidgets, "QLabel", _REAL_QLABEL)
    # Panel modules bind QLabel at import time, so update loaded bindings directly.
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("visualizer.src.panels") and mod is not None:
            if hasattr(mod, "QLabel") and not isinstance(mod.QLabel, type):
                monkeypatch.setattr(mod, "QLabel", _REAL_QLABEL)
            elif hasattr(mod, "QLabel") and mod.QLabel is not _REAL_QLABEL:
                monkeypatch.setattr(mod, "QLabel", _REAL_QLABEL)
    yield


class _FakeVisualizer(QMainWindow):
    """Minimal stand-in for ORCHAV that satisfies panel constructors."""

    def __init__(self, renderer_type: str = "pygfx", layout_profile: str = "auto"):
        super().__init__()
        self._renderer_type = renderer_type
        self._layout_profile = layout_profile
        is_pygfx = renderer_type == "pygfx"
        self.renderer = SimpleNamespace(
            renderer_type=renderer_type,
            capabilities=RendererCapabilities(
                picking=is_pygfx,
                transform_gizmo=is_pygfx,
                rf_xray_overlay=is_pygfx,
                viewport_hud=is_pygfx,
            ),
        )
        self.trajectory_load_coordinator = TrajectoryLoadCoordinator()
        self.scenario_config = None
        self.app_state = create_initial_state()

    def set_state(self, **updates) -> None:
        """Apply panel state changes through the same immutable boundary as the app."""
        self.app_state = update_state(self.app_state, **updates)
        manager = getattr(self, "ui_manager", None)
        if manager is not None:
            manager.refresh_global_context(self.app_state)


# Reuse one tree within each test, then let pytest-qt dispose its native window.
_MGR = None
_FUNCTION_MANAGERS = []


def _dispose_manager(mgr, qapp) -> None:
    """Release one complete test panel tree without touching unrelated Qt widgets."""
    parent = getattr(mgr, "_test_parent_ref", None)

    mgr.cleanup()

    # On macOS, drain the timer deletions while the registries still own every
    # Python callback wrapper; clearing them first can race PySide slot
    # destruction.  A process-wide DeferredDelete drain is not needed on the
    # other Qt backends and can execute stale events left by an unrelated test
    # tree during a long shared-qapp run.
    if sys.platform == "darwin":
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()

    if parent is not None:
        coordinator = getattr(parent, "trajectory_load_coordinator", None)
        shutdown = getattr(coordinator, "shutdown", None)
        if callable(shutdown):
            shutdown(timeout=1.0)
        if getattr(parent, "ui_manager", None) is mgr:
            parent.ui_manager = None

    mgr._test_parent_ref = None
    mgr.parent = None

    mgr.panels.clear()
    mgr.widgets.clear()
    mgr.sections.clear()
    mgr.panel_sequence.clear()
    mgr.ctrl_panel = None
    mgr.ctrl_layout = None
    mgr._tab_widget = None


@pytest.fixture(autouse=True)
def dispose_test_managers(qtbot, qapp):
    """Clear managers before pytest-qt schedules native window deletion."""
    global _MGR
    start = len(_FUNCTION_MANAGERS)
    yield

    managers = list(reversed(_FUNCTION_MANAGERS[start:]))
    if _MGR is not None:
        managers.append(_MGR)

    seen = set()
    try:
        for mgr in managers:
            if id(mgr) in seen:
                continue
            seen.add(id(mgr))
            parent = getattr(mgr, "_test_parent_ref", None)
            if parent is not None:
                qtbot.addWidget(parent)
            _dispose_manager(mgr, qapp)
    finally:
        del _FUNCTION_MANAGERS[start:]
        _MGR = None


def _get_mgr():
    global _MGR
    if _MGR is None:
        from visualizer.src.app.panel_manager import UIPanelManager

        parent = _FakeVisualizer()
        mgr = UIPanelManager(parent, total_steps=10)
        mgr.create_all_panels()
        parent.ui_manager = mgr
        mgr._test_parent_ref = parent
        _MGR = mgr
    return _MGR


def _build_mgr(renderer_type: str = "pygfx", layout_profile: str = "auto"):
    from visualizer.src.app.panel_manager import UIPanelManager

    parent = _FakeVisualizer(renderer_type=renderer_type, layout_profile=layout_profile)
    mgr = UIPanelManager(parent, total_steps=10)
    mgr.create_all_panels()
    parent.ui_manager = mgr
    mgr._test_parent_ref = parent
    _FUNCTION_MANAGERS.append(mgr)
    return mgr


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in self._callbacks:
            callback(*args)


class _FakeButton:
    def __init__(self) -> None:
        self.clicked = _FakeSignal()

    def __bool__(self) -> bool:
        return True


def test_has_tab_widget():
    mgr = _get_mgr()
    assert hasattr(mgr, "_tab_widget")
    assert isinstance(mgr._tab_widget, QTabWidget)


def test_core_tabs_present():
    tabs = _get_mgr()._tab_widget
    tab_names = [tabs.tabText(i) for i in range(tabs.count())]
    expected_order = [
        "Scene",
        "Paths",
        "Coverage",
        "Analysis",
        "Edit",
        "Rendering",
        "Capture & Export",
        "Antennas",
        "System",
    ]
    for expected in expected_order:
        assert expected in tab_names, f"Tab {expected!r} not found in {tab_names}"
    assert tab_names == expected_order


def test_persistent_panels_not_in_tabs():
    """Animation, camera, and Context stay outside workflow tabs."""
    mgr = _get_mgr()
    assert "animation" not in mgr._tab_map
    assert "camera" not in mgr._tab_map
    assert "context" not in mgr._tab_map
    assert "context" not in mgr.sections
    tx_dropdown = mgr.panels["context"].widgets["tx_dropdown"]
    assert not mgr._tab_widget.isAncestorOf(tx_dropdown)


@pytest.mark.parametrize(
    ("layout_profile", "context_layout_type"),
    (("auto", QHBoxLayout), ("capture-workspace", QVBoxLayout)),
)
def test_persistent_controls_stay_compact_without_fixed_camera_width(
    layout_profile,
    context_layout_type,
):
    """Animation and camera share one row without forcing a 460 px camera."""
    mgr = _build_mgr(layout_profile=layout_profile)
    top_bar = mgr.ctrl_layout.itemAt(0).widget()
    context_group = mgr.ctrl_layout.itemAt(1).widget()
    camera_widget = top_bar.layout().itemAt(1).widget()

    assert isinstance(top_bar.layout(), QHBoxLayout)
    assert camera_widget.minimumWidth() != 460
    assert camera_widget.maximumWidth() != 460
    assert isinstance(context_group.layout(), context_layout_type)


def test_context_widgets_are_parent_compatibility_controls():
    """The persistent controls are the only TX/RX and MPC master widgets."""
    mgr = _build_mgr()
    parent = mgr._test_parent_ref
    mgr._connect_event_handlers = lambda _parent: None
    mgr.connect_widgets_to_parent(parent)
    context_widgets = mgr.panels["context"].widgets

    assert parent.tx_dropdown is context_widgets["tx_dropdown"]
    assert parent.rx_dropdown is context_widgets["rx_dropdown"]
    assert parent.mpc_layer_cb is context_widgets["mpc_layer_cb"]
    assert parent.viewport_hud_cb is context_widgets["viewport_hud_cb"]
    assert context_widgets["viewport_hud_cb"].text() == "HUD"
    assert context_widgets["viewport_hud_cb"].isChecked()
    assert "tx_dropdown" not in mgr.panels["nodes"].widgets
    assert "rx_dropdown" not in mgr.panels["nodes"].widgets
    assert "mpc_layer_cb" not in mgr.panels["mpc"].widgets


def test_paths_panel_exposes_compact_mpc_explorer_entry_point():
    """The scalable table opens separately instead of occupying the controls pane."""
    mgr = _build_mgr()
    button = mgr.panels["mpc"].widgets["mpc_explorer_btn"]

    assert button.text() == "Open MPC Explorer..."
    assert button.objectName() == "openMpcExplorerButton"
    assert not hasattr(mgr._test_parent_ref, "_mpc_explorer_session")


def test_context_hud_toggle_is_visible_but_disabled_without_renderer_support():
    """The persistent HUD switch remains discoverable on unsupported backends."""
    mgr = _build_mgr(renderer_type="open3d")
    hud = mgr.panels["context"].widgets["viewport_hud_cb"]

    assert not hud.isHidden()
    assert not hud.isEnabled()
    assert not hud.isChecked()
    assert "does not provide" in hud.toolTip()


def test_panels_exist_in_sections():
    """Core panels should have sections inside their tabs."""
    mgr = _get_mgr()
    for key in (
        "nodes",
        "mpc",
        "materials",
        "statistics",
        "statistics_graphs",
        "rf_xray",
        "interactive_preview",
        "scene_style",
        "scene_view",
        "lighting",
        "figure_capture",
        "export",
        "data_source",
    ):
        assert key in mgr.sections, f"Panel {key!r} missing from sections"


def test_rendering_tab_does_not_keep_unmounted_render_proxy_panels():
    """Render sub-panels should exist only when mounted in the tab schema."""
    mgr = _get_mgr()

    assert "appearance" not in mgr.panels
    assert "line_point" not in mgr.panels


def test_transport_buttons_wire_to_animation_controller() -> None:
    """PanelManager should not depend on root visualizer animation delegates."""
    from visualizer.src.app.panel_manager import UIPanelManager

    calls = []

    class _AnimationController:
        def toggle_animation(self, direction=None) -> None:
            calls.append(("toggle", direction))

        def play_backward(self) -> None:
            calls.append(("backward",))

        def previous_frame(self) -> None:
            calls.append(("previous",))

        def next_frame(self) -> None:
            calls.append(("next",))

        def reset_animation(self) -> None:
            calls.append(("reset",))

    parent = SimpleNamespace(
        ui_controller=SimpleNamespace(),
        animation_controller=_AnimationController(),
        play_btn=_FakeButton(),
        reverse_play_btn=_FakeButton(),
        prev_btn=_FakeButton(),
        next_btn=_FakeButton(),
        reset_btn=_FakeButton(),
    )
    mgr = UIPanelManager(parent, total_steps=10)

    mgr._connect_event_handlers(parent)

    parent.play_btn.clicked.emit(True)
    parent.reverse_play_btn.clicked.emit(True)
    parent.prev_btn.clicked.emit(True)
    parent.next_btn.clicked.emit(True)
    parent.reset_btn.clicked.emit(True)

    assert calls == [
        ("toggle", 1),
        ("backward",),
        ("previous",),
        ("next",),
        ("reset",),
    ]


def test_antennas_panel_single_pair_workflow_widgets():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    antennas_idx = mgr.find_tab_index_by_label("Antennas")
    scroll = tabs.widget(antennas_idx)
    panel = mgr.panels["beam_pattern"]
    widgets = panel.widgets

    subgroup_titles = {
        group.title() for group in scroll.findChildren(QtWidgets.QGroupBox) if group.title()
    }
    assert {"Pattern Source", "Shared TX/RX Parameters", "Display"}.issubset(subgroup_titles)
    assert "Steering" not in subgroup_titles

    assert widgets["beam_tx_selector"].text() == "N/A"
    assert widgets["beam_rx_selector"].text() == "N/A"
    assert not hasattr(widgets["beam_rx_selector"], "currentTextChanged")

    assert widgets["standalone_h_spacing"].value() == pytest.approx(0.5)
    assert widgets["standalone_v_spacing"].value() == pytest.approx(0.5)
    assert widgets["standalone_h_spacing"].suffix().strip() == "lambda"
    assert widgets["standalone_v_spacing"].suffix().strip() == "lambda"

    assert widgets["standalone_strategy"].currentText() == "SVD (Current MPCs)"
    assert "current MPC channel" in widgets["standalone_strategy"].toolTip()
    for key in (
        "standalone_rows",
        "standalone_cols",
        "standalone_freq",
        "standalone_h_spacing",
        "standalone_v_spacing",
        "standalone_azimuth",
        "standalone_elevation",
        "beam_azimuth_spin",
        "beam_elevation_spin",
        "beam_tx_scale_spin",
        "beam_rx_scale_spin",
        "beam_dynamic_range",
    ):
        assert widgets[key].keyboardTracking() is False
    assert widgets["standalone_rows"].maximum() == 32
    assert widgets["standalone_cols"].maximum() == 32
    assert widgets["beam_azimuth_spin"].maximum() == 180
    assert widgets["beam_elevation_spin"].maximum() == 91
    assert "8 million" in widgets["beam_complexity_note"].text()
    assert panel._angle_container is not None
    assert panel._angle_container.isHidden()
    assert widgets["beam_colorbar_container"].isHidden()

    widgets["standalone_strategy"].setCurrentText("Manual Steering")
    assert not panel._angle_container.isHidden()


def test_antennas_panel_beam_colorbar_updates_from_display_controls():
    mgr = _build_mgr()
    panel = mgr.panels["beam_pattern"]
    widgets = panel.widgets

    panel.update_beam_colorbar(
        show_beamforming=True,
        db_scale=False,
        dynamic_range_db=40.0,
        colormap="jet",
    )
    jet_key = widgets["beam_colorbar_gradient"].pixmap().cacheKey()

    assert not widgets["beam_colorbar_container"].isHidden()
    assert widgets["beam_colorbar_min_label"].text() == "0"
    assert widgets["beam_colorbar_max_label"].text() == "1"
    assert widgets["beam_dynamic_range"].isEnabled() is False

    panel.update_beam_colorbar(
        show_beamforming=True,
        db_scale=True,
        dynamic_range_db=55.0,
        colormap="viridis",
    )

    assert widgets["beam_colorbar_gradient"].pixmap().cacheKey() != jet_key
    assert widgets["beam_colorbar_min_label"].text() == "-55 dB"
    assert widgets["beam_colorbar_max_label"].text() == "0 dB"
    assert widgets["beam_dynamic_range"].isEnabled() is True


def test_antennas_read_only_pair_labels_update_after_panel_connection():
    from visualizer.src.controllers.beamforming_ui_controller import BeamformingUIController

    mgr = _build_mgr()
    parent = mgr._test_parent_ref
    parent.app_state = create_initial_state(
        selected_tx="all",
        selected_rx=1,
        beamforming_tx_node="auto",
        beamforming_rx_node="auto",
    )
    parent.available_tx = [0]
    parent.available_rx = [0, 1]
    parent._latest_beamforming_info = None
    parent._latest_beamforming_pairs = []
    parent._beamforming_tx_nodes = []
    parent._beamforming_rx_nodes = []
    parent.set_state = lambda **updates: setattr(
        parent,
        "app_state",
        update_state(parent.app_state, **updates),
    )

    widgets = mgr.panels["beam_pattern"].widgets
    parent.beam_tx_selector = widgets["beam_tx_selector"]
    parent.beam_rx_selector = widgets["beam_rx_selector"]
    parent.beam_status_label = widgets["beam_status_label"]
    parent.beam_gain_label = widgets["beam_gain_label"]
    BeamformingUIController(parent).apply_selector_state()

    assert parent.beam_tx_selector.text() == "All TX"
    assert parent.beam_rx_selector.text() == "RX2"
    assert parent.app_state.beamforming_tx_node == "auto"
    assert parent.app_state.beamforming_rx_node == "rx_2"
    assert parent.beam_status_label.text() == "Hidden"


def test_rendering_and_capture_tabs_use_task_focused_sections():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    rendering_idx = mgr.find_tab_index_by_label("Rendering")
    scroll = tabs.widget(rendering_idx)
    page = scroll.widget()
    layout = page.layout()

    titles = []
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if widget is not None and hasattr(widget, "_title"):
            titles.append(widget._title)

    assert titles == ["Scene Style", "Scene View", "Lighting", "Viewport HUD"]
    assert mgr.sections["scene_style"].is_expanded() is True
    assert mgr.sections["scene_view"].is_expanded() is False
    assert mgr.sections["lighting"].is_expanded() is False
    assert mgr.sections["viewport_hud"].is_expanded() is False

    capture_idx = mgr.find_tab_index_by_label("Capture & Export")
    capture_page = tabs.widget(capture_idx).widget()
    capture_titles = [
        item.widget()._title
        for item in (capture_page.layout().itemAt(i) for i in range(capture_page.layout().count()))
        if item.widget() is not None and hasattr(item.widget(), "_title")
    ]
    assert capture_titles == ["Figure Capture", "Export"]
    assert mgr.sections["figure_capture"].is_expanded() is True
    assert mgr.sections["export"].is_expanded() is False


def test_viewport_hud_rendering_section_follows_renderer_capability():
    pygfx_mgr = _build_mgr(renderer_type="pygfx")
    assert pygfx_mgr.panels["viewport_hud"] is not None
    assert "viewport_hud" in pygfx_mgr.sections

    open3d_mgr = _build_mgr(renderer_type="open3d")
    assert open3d_mgr.panels["viewport_hud"] is None
    assert "viewport_hud" not in open3d_mgr.sections

    rendering_idx = open3d_mgr.find_tab_index_by_label("Rendering")
    rendering_page = open3d_mgr._tab_widget.widget(rendering_idx).widget()
    titles = [
        item.widget()._title
        for item in (
            rendering_page.layout().itemAt(i) for i in range(rendering_page.layout().count())
        )
        if item.widget() is not None and hasattr(item.widget(), "_title")
    ]
    assert titles == ["Scene Style", "Scene View", "Lighting"]


def test_preview_and_rf_xray_sections_are_relocated_without_duplication():
    mgr = _build_mgr()
    tabs = mgr._tab_widget

    assert mgr.get_panel_tab_label("interactive_preview") == "Edit"
    assert mgr.get_panel_tab_label("rf_xray") == "Analysis"
    assert mgr.get_panel_tab_label("figure_capture") == "Capture & Export"
    assert mgr.get_panel_tab_label("export") == "Capture & Export"
    assert mgr.sections["trajectory"]._title == "Trajectory Analysis"

    preview_widget = mgr.panels["nodes"].widgets["live_preview_cb"]
    edit_tab = tabs.widget(mgr.find_tab_index_by_label("Edit"))
    scene_tab = tabs.widget(mgr.find_tab_index_by_label("Scene"))
    assert edit_tab.isAncestorOf(preview_widget)
    assert not scene_tab.isAncestorOf(preview_widget)

    rf_widget = mgr.panels["materials"].widgets["rf_xray_toggle"]
    analysis_tab = tabs.widget(mgr.find_tab_index_by_label("Analysis"))
    assert analysis_tab.isAncestorOf(rf_widget)
    assert not scene_tab.isAncestorOf(rf_widget)


def test_edit_tab_stays_available_while_renderer_features_are_gated_inside_it():
    pygfx_mgr = _build_mgr(renderer_type="pygfx")
    assert pygfx_mgr.find_tab_index_by_label("Edit") >= 0
    assert "interactive_preview" in pygfx_mgr.sections
    assert "rf_xray" in pygfx_mgr.sections

    open3d_mgr = _build_mgr(renderer_type="open3d")
    assert open3d_mgr.find_tab_index_by_label("Edit") >= 0
    assert "interactive_preview" in open3d_mgr.sections
    assert "rf_xray" not in open3d_mgr.sections


def test_collapsed_statistics_graphs_build_on_first_expand():
    mgr = _build_mgr()
    section = mgr.sections["statistics_graphs"]
    stats_panel = mgr.panels["statistics"]

    assert section.is_expanded() is False
    assert getattr(section, "_lazy_content_created", True) is False
    assert stats_panel._graphs_panel_created is False

    section.expand()

    assert getattr(section, "_lazy_content_created", False) is True
    assert stats_panel._graphs_panel_created is True


def test_rendering_reorg_preserves_stateful_widget_keys():
    mgr = _build_mgr()
    render_widgets = mgr.panels["render"].widgets
    mpc_widgets = mgr.panels["mpc"].widgets

    for key in (
        "bg_combo",
        "building_alpha_slider",
        "target_alpha_slider",
        "edge_line_width_slider",
        "paper_mode_cb",
        "aa_combo",
        "ibl_intensity_spin",
    ):
        assert key in render_widgets

    assert "point_size_slider" not in render_widgets
    assert "line_width_slider" not in render_widgets
    assert "point_size_spin" not in render_widgets
    assert "line_width_spin" not in render_widgets
    assert "point_size_spin" in mpc_widgets
    assert "line_width_spin" in mpc_widgets

    assert "opacity" in render_widgets["building_alpha_slider"].toolTip().lower()
    assert "opacity" in render_widgets["target_alpha_slider"].toolTip().lower()
    assert "transparency" not in render_widgets["building_alpha_slider"].toolTip().lower()
    assert "transparency" not in render_widgets["target_alpha_slider"].toolTip().lower()
    shader_combo = render_widgets["shader_combo"]
    shader_items = [shader_combo.itemText(i) for i in range(shader_combo.count())]
    assert shader_items == ["Standard", "Unlit", "Normals"]


def test_viewport_hud_detail_controls_defer_master_visibility_to_context():
    mgr = _build_mgr()
    panel = mgr.panels["render"]
    combo = panel.widgets["viewport_hud_mode_combo"]

    assert [combo.itemData(i) for i in range(combo.count())] == ["compact", "detailed"]
    assert "viewport_hud_toggle_btn" not in panel.widgets
    assert "selected-material color swatches" in panel.widgets["viewport_hud_filters_cb"].toolTip()

    state = SimpleNamespace(
        viewport_hud_enabled=False,
        viewport_hud_mode="detailed",
        viewport_hud_show_status=True,
        viewport_hud_show_legends=True,
        viewport_hud_show_filters=True,
        viewport_hud_show_annotations=True,
    )
    panel.parent.app_state = state
    panel._sync_viewport_hud_controls()

    assert combo.currentData() == "detailed"
    assert all(
        panel.widgets[f"viewport_hud_{key}_cb"].isEnabled()
        for key in ("status", "legends", "filters", "annotations")
    )

    state.viewport_hud_enabled = True
    panel._sync_viewport_hud_controls()

    assert all(
        panel.widgets[f"viewport_hud_{key}_cb"].isEnabled()
        for key in ("status", "legends", "filters", "annotations")
    )


def test_paper_mode_disables_shadows_without_owning_mpc_sizes():
    from visualizer.src.panels.render_panel import RenderPanel

    calls = []

    class Renderer:
        renderer_type = "pygfx"
        capabilities = RendererCapabilities(
            scene_shader=True,
            shadow_toggle=True,
            ibl=True,
            skybox=True,
        )

        def set_scene_shader(self, value):
            calls.append(("shader", value))

        def show_skybox(self, value):
            calls.append(("skybox", value))

        def set_ibl_intensity(self, value):
            calls.append(("ibl", value))

        def set_shadow_enabled(self, value):
            calls.append(("shadow", value))

        def set_line_width(self, value):
            calls.append(("line_width", value))

        def set_point_size(self, value):
            calls.append(("point_size", value))

    parent = SimpleNamespace(renderer=Renderer())
    parent.scene_appearance_service = SimpleNamespace(
        set_edge_visibility=lambda value: calls.append(("outline", value)),
        set_background_preset=lambda name, color: calls.append(("background", name)),
    )

    panel = RenderPanel(parent)
    group = panel.create_panel()
    panel.widgets["shadows_cb"].setChecked(True)

    panel.widgets["paper_mode_cb"].setChecked(True)

    assert "line_width_spin" not in panel.widgets
    assert "point_size_spin" not in panel.widgets
    assert panel.widgets["shadows_cb"].isChecked() is False
    assert ("shadow", False) in calls

    group.deleteLater()


def test_paper_mode_restores_previous_render_settings():
    from visualizer.src.panels.render_panel import RenderPanel

    calls = []

    class Renderer:
        renderer_type = "pygfx"
        capabilities = RendererCapabilities(
            scene_shader=True,
            shadow_toggle=True,
            ibl=True,
            skybox=True,
        )

        def set_scene_shader(self, value):
            calls.append(("shader", value))

        def show_skybox(self, value):
            calls.append(("skybox", value))

        def set_ibl_intensity(self, value):
            calls.append(("ibl", value))

        def set_shadow_enabled(self, value):
            calls.append(("shadow", value))

        def set_line_width(self, value):
            calls.append(("line_width", value))

        def set_point_size(self, value):
            calls.append(("point_size", value))

    parent = SimpleNamespace(renderer=Renderer())
    parent.scene_appearance_service = SimpleNamespace(
        set_edge_visibility=lambda value: calls.append(("outline", value)),
        set_background_preset=lambda name, color: calls.append(("background", name)),
    )

    panel = RenderPanel(parent)
    group = panel.create_panel()
    panel.widgets["bg_combo"].setCurrentText("Slate")
    panel.widgets["shader_combo"].setCurrentText("Normals")
    panel.widgets["skybox_cb"].setChecked(True)
    panel.widgets["shadows_cb"].setChecked(True)
    panel.widgets["outline_cb"].setChecked(False)
    panel.widgets["ibl_intensity_spin"].setValue(12000)

    panel.widgets["paper_mode_cb"].setChecked(True)
    assert panel.widgets["paper_mode_cb"].isChecked() is True
    assert panel.widgets["bg_combo"].currentText() == "White"
    assert panel.widgets["shader_combo"].currentText() == "Unlit"
    assert panel.widgets["ibl_intensity_spin"].value() == 5000

    panel.widgets["paper_mode_cb"].setChecked(False)

    assert panel.widgets["bg_combo"].currentText() == "Slate"
    assert panel.widgets["shader_combo"].currentText() == "Normals"
    assert panel.widgets["skybox_cb"].isChecked() is True
    assert panel.widgets["shadows_cb"].isChecked() is True
    assert panel.widgets["outline_cb"].isChecked() is False
    assert panel.widgets["ibl_intensity_spin"].value() == 12000
    assert ("background", "Slate") in calls
    assert ("shader", "normals") in calls
    assert ("skybox", True) in calls
    assert ("shadow", True) in calls
    assert ("outline", False) in calls

    group.deleteLater()


def test_rendering_scene_style_groups_edge_width_with_edges():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    rendering_idx = mgr.find_tab_index_by_label("Rendering")
    scroll = tabs.widget(rendering_idx)
    subgroups = scroll.findChildren(QtWidgets.QGroupBox)
    subgroup_titles = {group.title() for group in subgroups}

    assert "Opacity" in subgroup_titles
    assert "Edges" in subgroup_titles
    assert "Edge Width" not in subgroup_titles

    panel = mgr.panels["render"]
    edges_group = next(group for group in subgroups if group.title() == "Edges")
    edges_row = edges_group.layout().itemAt(0).layout()
    edge_width = panel.widgets["edge_line_width_container"]

    assert edges_row.indexOf(edge_width) > edges_row.indexOf(panel.widgets["outline_cb"])


def test_rendering_edge_thickness_is_pygfx_only():
    mgr = _build_mgr()
    parent = mgr._test_parent_ref
    panel = mgr.panels["render"]
    container = panel.widgets["edge_line_width_container"]

    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(trajectories=True),
    )
    panel._sync_from_visualizer()

    assert container.isHidden()

    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(trajectories=True, wireframe=True),
        set_wireframe=lambda checked: None,
    )
    panel._sync_from_visualizer()

    assert not container.isHidden()


def test_rendering_edge_thickness_callback_skips_open3d_renderer():
    mgr = _build_mgr()
    parent = mgr._test_parent_ref
    panel = mgr.panels["render"]
    calls: list[float] = []

    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(),
        set_edge_line_width=lambda value: calls.append(value),
    )
    panel._on_edge_line_width_spin_changed(4)

    assert calls == []

    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(wireframe=True),
        set_wireframe=lambda checked: None,
        set_edge_line_width=lambda value: calls.append(value),
    )
    panel._on_edge_line_width_spin_changed(5)

    assert calls == [5.0]


def test_scene_view_controls_follow_renderer_capabilities():
    mgr = _build_mgr()
    parent = mgr._test_parent_ref
    panel = mgr.panels["render"]
    show_axes = panel.widgets["show_axes_cb"]
    screen_labels = panel.widgets["screen_space_labels_cb"]
    culling = panel.widgets["culling_cb"]

    parent.renderer = SimpleNamespace(
        renderer_type="open3d",
        capabilities=RendererCapabilities(frustum_culling=True, axes=True),
        show_axes=lambda *_: None,
    )
    panel._sync_from_visualizer()

    assert not show_axes.isHidden()
    assert screen_labels.isHidden()
    assert not culling.isHidden()

    parent.renderer = SimpleNamespace(
        renderer_type="pygfx",
        capabilities=RendererCapabilities(axes=True, screen_space_labels=True),
        show_axes=lambda *_: None,
    )
    panel._sync_from_visualizer()

    assert not show_axes.isHidden()
    assert not screen_labels.isHidden()
    assert culling.isHidden()


def test_lighting_groups_follow_renderer_capabilities():
    mgr = _build_mgr()
    parent = mgr._test_parent_ref
    panel = mgr.panels["render"]

    def light_rig_state():
        return {
            "headlight_enabled": True,
            "headlight_intensity": 1.2,
            "key_azimuth_deg": -123.0,
            "key_elevation_deg": -48.0,
            "key_intensity": 3.0,
            "fill_azimuth_deg": 53.0,
            "fill_elevation_deg": -45.0,
            "fill_intensity": 1.0,
        }

    parent.renderer = SimpleNamespace(
        renderer_type="open3d",
        capabilities=RendererCapabilities(
            scene_shader=True,
            ibl=True,
            shadow_toggle=True,
            open3d_settings_panel=True,
            skybox=True,
        ),
        get_ibl_intensity=lambda: 30000,
        get_ibl_name=lambda: "default",
        show_skybox=lambda *_: None,
    )
    panel._sync_from_visualizer()

    assert not panel.widgets["shader_group"].isHidden()
    assert not panel.widgets["environment_group"].isHidden()
    assert panel.widgets["direct_lights_group"].isHidden()
    assert not panel.widgets["advanced_lighting_group"].isHidden()
    assert not panel.widgets["o3d_settings_cb"].isHidden()
    assert "color_fidelity_cb" not in panel.widgets
    assert "ibl_enabled_cb" not in panel.widgets

    parent.renderer = SimpleNamespace(
        renderer_type="pygfx",
        capabilities=RendererCapabilities(
            ibl=True,
            shadow_toggle=True,
            skybox=True,
            direct_lighting=True,
        ),
        get_ibl_intensity=lambda: 30000,
        get_ibl_name=lambda: "default",
        show_skybox=lambda *_: None,
        get_light_rig_state=light_rig_state,
        set_headlight_enabled=lambda *_: None,
        set_headlight_intensity=lambda *_: None,
        set_key_light_angles=lambda *_: None,
        set_fill_light_angles=lambda *_: None,
        set_key_light_intensity=lambda *_: None,
        set_fill_light_intensity=lambda *_: None,
    )
    panel._sync_from_visualizer()

    assert panel.widgets["shader_group"].isHidden()
    assert not panel.widgets["environment_group"].isHidden()
    assert not panel.widgets["direct_lights_group"].isHidden()
    assert not panel.widgets["advanced_lighting_group"].isHidden()
    assert panel.widgets["o3d_settings_cb"].isHidden()


def test_screen_space_label_callback_skips_open3d_renderer():
    mgr = _build_mgr()
    parent = mgr._test_parent_ref
    panel = mgr.panels["render"]
    parent.app_state = AppState(
        step=0,
        selected_tx=0,
        selected_rx=0,
        mpc_visibility=MpcVisibility(),
        mpc_allowed_orders=frozenset(),
        mpc_allowed_types=frozenset(),
        color_mode="reflection_order",
        show_labels=True,
        sync_target_position=True,
        label_screen_space=True,
    )
    parent.node_service = SimpleNamespace(recreate_tx_rx_labels=lambda *_: None)
    parent.label_font_size = 0.3

    parent.renderer = SimpleNamespace(
        renderer_type="open3d",
        capabilities=RendererCapabilities(),
    )
    panel._on_screen_space_labels_toggled(False)

    assert parent.app_state.label_screen_space is True

    parent.renderer = SimpleNamespace(
        renderer_type="pygfx",
        capabilities=RendererCapabilities(screen_space_labels=True),
        update_renderer=lambda: None,
    )
    panel._on_screen_space_labels_toggled(False)

    assert parent.app_state.label_screen_space is False


def test_legacy_authoring_panel_is_not_registered_for_any_renderer():
    for renderer_type in ("pygfx", "open3d"):
        mgr = _build_mgr(renderer_type=renderer_type)
        tab_names = [mgr._tab_widget.tabText(i) for i in range(mgr._tab_widget.count())]
        assert "Authoring" not in tab_names
        assert "trajectory_builder" not in mgr.sections
        assert "trajectory_builder" not in mgr.panels


def test_set_panel_visible_hides_section():
    mgr = _get_mgr()
    section = mgr.sections["mpc"]
    mgr.set_panel_visible("mpc", False)
    assert section.isHidden()
    mgr.set_panel_visible("mpc", True)
    assert not section.isHidden()


def test_coverage_has_a_dedicated_data_gated_tab():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    coverage_index = mgr.find_tab_index_by_label("Coverage")

    assert coverage_index == mgr.find_tab_index_by_label("Paths") + 1
    assert mgr._tab_map["coverage"] == "Coverage"
    if hasattr(tabs, "isTabVisible"):
        assert not tabs.isTabVisible(coverage_index)
    else:
        assert not tabs.isTabEnabled(coverage_index)

    mgr.set_coverage_data_available(True)
    if hasattr(tabs, "isTabVisible"):
        assert tabs.isTabVisible(coverage_index)
    assert tabs.isTabEnabled(coverage_index)
    assert not mgr.sections["coverage"].isHidden()

    # The legacy section API remains compatible with current service callers.
    mgr.set_panel_visible("coverage", False)
    if hasattr(tabs, "isTabVisible"):
        assert not tabs.isTabVisible(coverage_index)
    else:
        assert not tabs.isTabEnabled(coverage_index)


def test_scene_only_coverage_load_activates_coverage_tab():
    mgr = _build_mgr()
    mgr.set_frame_data_available(False)

    context = mgr.panels["context"].widgets
    assert not context["tx_dropdown"].isEnabled()
    assert not context["rx_dropdown"].isEnabled()
    assert not context["mpc_layer_cb"].isEnabled()
    assert context["viewport_hud_cb"].isEnabled()
    assert not context["frame_status_label"].isHidden()

    mgr.set_coverage_data_available(True)

    assert mgr.get_active_tab_label() == "Coverage"
    analysis_index = mgr.find_tab_index_by_label("Analysis")
    if hasattr(mgr._tab_widget, "isTabVisible"):
        assert mgr._tab_widget.isTabVisible(analysis_index)
    else:
        assert mgr._tab_widget.isTabEnabled(analysis_index)


def test_scene_only_mode_hides_frame_dependent_tabs_and_nodes():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    context = mgr.panels["context"].widgets

    mgr.set_frame_data_available(False)

    assert not context["tx_dropdown"].isEnabled()
    assert not context["rx_dropdown"].isEnabled()
    assert not context["mpc_layer_cb"].isEnabled()
    assert context["viewport_hud_cb"].isEnabled()
    assert not context["frame_status_label"].isHidden()

    for label in ("Paths", "Analysis", "Antennas"):
        idx = mgr.find_tab_index_by_label(label)
        assert idx >= 0
        if hasattr(tabs, "isTabVisible"):
            assert not tabs.isTabVisible(idx)
        else:
            assert not tabs.isTabEnabled(idx)
    assert mgr.sections["nodes"].isHidden()

    for label in ("Scene", "Rendering", "System"):
        idx = mgr.find_tab_index_by_label(label)
        assert idx >= 0
        if hasattr(tabs, "isTabVisible"):
            assert tabs.isTabVisible(idx)
        assert tabs.isTabEnabled(idx)

    mgr.set_frame_data_available(True)

    assert context["tx_dropdown"].isEnabled()
    assert context["rx_dropdown"].isEnabled()
    assert context["mpc_layer_cb"].isEnabled()
    assert context["frame_status_label"].isHidden()

    for label in ("Paths", "Analysis", "Antennas"):
        idx = mgr.find_tab_index_by_label(label)
        if hasattr(tabs, "isTabVisible"):
            assert tabs.isTabVisible(idx)
        assert tabs.isTabEnabled(idx)
    assert not mgr.sections["nodes"].isHidden()


def test_scene_only_mode_moves_active_tab_to_available_tab():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    paths_idx = mgr.find_tab_index_by_label("Paths")
    tabs.setCurrentIndex(paths_idx)

    mgr.set_frame_data_available(False)

    assert mgr.get_active_tab_label() != "Paths"
    assert tabs.isTabEnabled(tabs.currentIndex())
    if hasattr(tabs, "isTabVisible"):
        assert tabs.isTabVisible(tabs.currentIndex())


def test_ctrl_panel_is_widget():
    assert isinstance(_get_mgr().ctrl_panel, QWidget)


def test_paths_badge_reflects_range_filters():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    filter_status = mgr.panels["context"].widgets["filter_status_label"]
    paths_idx = mgr.find_tab_index_by_label("Paths")
    assert tabs.tabText(paths_idx) == "Paths"
    assert filter_status.isHidden()

    delay_min = mgr.panels["mpc"].widgets["delay_filter_min"]
    delay_min.setValue(delay_min.minimum() + 1.0)
    mgr.update_paths_tab_badge()
    assert tabs.tabText(paths_idx) == "Paths (filtered)"
    assert not filter_status.isHidden()

    delay_min.setValue(delay_min.minimum())
    mgr.update_paths_tab_badge()
    assert tabs.tabText(paths_idx) == "Paths"
    assert filter_status.isHidden()


def test_paths_badge_reflects_material_filters():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    paths_idx = mgr.find_tab_index_by_label("Paths")
    mpc_panel = mgr.panels["mpc"]
    filter_status = mgr.panels["context"].widgets["filter_status_label"]

    mpc_panel.set_materials(["brick", "glass"], checked={"brick", "glass"})
    mgr.update_paths_tab_badge()
    assert tabs.tabText(paths_idx) == "Paths"
    assert filter_status.isHidden()

    from PySide6.QtCore import Qt

    first_item = mpc_panel.widgets["materials_model"].item(0)
    first_item.setCheckState(Qt.Unchecked)
    mgr.update_paths_tab_badge()
    assert tabs.tabText(paths_idx) == "Paths (filtered)"
    assert not filter_status.isHidden()


def test_panel_active_query_uses_tab_widget_identity_not_display_label():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    system_idx = mgr.find_tab_index_by_label("System")
    tabs.setCurrentIndex(system_idx)

    assert mgr.get_panel_tab_label("performance") == "System"
    assert mgr.is_panel_in_active_tab("performance") is True

    tabs.setTabText(system_idx, "Runtime")
    assert mgr.is_panel_in_active_tab("performance") is True

    paths_idx = mgr.find_tab_index_by_label("Paths")
    tabs.setCurrentIndex(paths_idx)
    assert mgr.is_panel_in_active_tab("performance") is False


def test_session_service_saves_normalized_active_tab_label():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    paths_idx = mgr.find_tab_index_by_label("Paths")
    tabs.setTabText(paths_idx, "Paths (filtered)")
    tabs.setCurrentIndex(paths_idx)

    viz = SimpleNamespace(
        ui_manager=mgr,
        current_building_alpha=1.0,
        current_target_alpha=1.0,
    )
    service = SessionService.__new__(SessionService)
    service.viz = viz
    render_state = service._get_rendering_state()

    assert render_state["active_tab"] == paths_idx
    assert render_state["active_tab_label"] == "Paths"


def test_session_service_restores_tab_by_label_when_indices_shift():
    mgr = _build_mgr()
    tabs = mgr._tab_widget
    system_idx = mgr.find_tab_index_by_label("System")
    tabs.insertTab(system_idx, QWidget(), "Dummy")
    tabs.setCurrentIndex(0)

    viz = SimpleNamespace(
        ui_manager=mgr,
        current_building_alpha=1.0,
        current_target_alpha=1.0,
    )
    service = SessionService.__new__(SessionService)
    service.viz = viz
    service._restore_rendering_state({"active_tab": system_idx, "active_tab_label": "System"})

    assert mgr.get_active_tab_label() == "System"


def test_tab_restore_uses_index_only_when_no_saved_label_exists():
    mgr = _build_mgr(renderer_type="open3d")
    tabs = mgr._tab_widget
    scene_idx = mgr.find_tab_index_by_label("Scene")
    stale_idx = mgr.find_tab_index_by_label("Rendering")
    tabs.setCurrentIndex(scene_idx)

    restored = mgr.restore_active_tab(label="Edit & Preview", index=stale_idx)

    assert restored is False
    assert mgr.get_active_tab_label() == "Scene"

    restored = mgr.restore_active_tab(label=None, index=stale_idx)

    assert restored is True
    assert mgr.get_active_tab_label() == "Rendering"


def test_session_service_syncs_beam_spacing_widgets_as_wavelengths():
    mgr = _build_mgr()
    state = create_initial_state(
        standalone_carrier_frequency_ghz=60.0,
        standalone_horizontal_spacing_m=spacing_wavelengths_to_m(0.75, 60.0),
        standalone_vertical_spacing_m=spacing_wavelengths_to_m(0.25, 60.0),
    )

    service = SessionService.__new__(SessionService)
    selector_sync_calls = []
    service.viz = SimpleNamespace(
        ui_manager=mgr,
        beamforming_ui_controller=SimpleNamespace(
            apply_selector_state=lambda: selector_sync_calls.append(True)
        ),
    )

    service._sync_beam_panel(mgr.panels, state)

    widgets = mgr.panels["beam_pattern"].widgets
    assert widgets["standalone_h_spacing"].value() == pytest.approx(0.75)
    assert widgets["standalone_v_spacing"].value() == pytest.approx(0.25)
    assert selector_sync_calls == [True]
