"""Automatic saved-cell shells and fail-closed scene/selection lifecycles."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QCheckBox, QLabel

from tests.visualizer.unit.test_experimental_range_geometry import _request
from tests.visualizer.unit.test_sensing_dashboard_recording import _request as dashboard_request
from visualizer.src.experiments.range_shell import RangeShellController
from visualizer.src.sensing.selection import inspection_cell
from visualizer.src.sensing.sensing_window import SensingWindow

pytestmark = pytest.mark.usefixtures("sensing_figure_preferences")


class Overlay:
    def __init__(self):
        self.rendered = []
        self.current = None

    def range_locus(self, locus, *, cutaway):
        self.current = (locus, cutaway)
        self.rendered.append(self.current)

    def clear(self):
        self.current = None


@pytest.fixture
def shell(qtbot):
    request, selection = _request()
    enabled, cutaway = QCheckBox(), QCheckBox()
    enabled.setChecked(True)
    status = QLabel()
    dashboard = SimpleNamespace(
        _last_request=request,
        _linked_selection=selection,
        range_shell_enabled=enabled,
        range_shell_cutaway=cutaway,
        range_shell_status=status,
        isVisible=lambda: True,
    )
    controller = RangeShellController(
        SimpleNamespace(
            renderer=SimpleNamespace(renderer_id="pygfx"),
            app_state=SimpleNamespace(extension_state={"sensing": {"show_detections": True}}),
        )
    )
    controller.overlay = Overlay()
    controller.accepted_step = request.effective_display_step
    controller.dashboard = dashboard
    return controller, dashboard


def test_shell_automatically_follows_exact_bin_and_cutaway_without_source_mutation(shell):
    controller, dashboard = shell
    before = deepcopy(dashboard._last_request.display_frame)
    controller.sync()
    assert controller.overlay.current is not None
    original = controller.identity
    request = dashboard._last_request
    dashboard._linked_selection = replace(
        dashboard._linked_selection, cell=inspection_cell(request, 2, 1)
    )
    controller.sync()
    assert controller.identity != original
    assert "bin 2" in dashboard.range_shell_status.text()
    dashboard.range_shell_cutaway.setChecked(True)
    controller.sync()
    assert controller.overlay.current[1] is True
    for field in ("range_axis_m", "doppler_axis_hz", "range_doppler_power_by_pair"):
        np.testing.assert_array_equal(
            before["sensing"][field], request.display_frame["sensing"][field]
        )
    controller.sync()
    assert len(controller.overlay.rendered) == 3  # An unchanged request draws nothing new.


@pytest.mark.parametrize(
    "change",
    [
        "clear",
        "closed",
        "disabled",
        "frame",
        "source",
        "pair",
        "recipe",
        "catalog",
        "range_grid",
        "out_of_range",
        "variant",
    ],
)
def test_stale_unsupported_or_closed_selection_clears_shell(shell, change):
    controller, dashboard = shell
    controller.sync()
    request, selected = dashboard._last_request, dashboard._linked_selection
    if change == "clear":
        dashboard._linked_selection = None
    elif change == "closed":
        dashboard.isVisible = lambda: False
    elif change == "disabled":
        dashboard.range_shell_enabled.setChecked(False)
    elif change == "frame":
        controller.accepted_step = 8
    elif change == "source":
        dashboard._last_request = replace(request, source_identity=("other",))
    elif change == "pair":
        dashboard._linked_selection = replace(selected, scope=("x", "y", "z"))
    elif change == "recipe":
        dashboard._linked_selection = replace(selected, cell=replace(selected.cell, recipe="old"))
    elif change == "catalog":
        dashboard._linked_selection = replace(
            selected, cell=replace(selected.cell, catalog_identity=())
        )
    elif change == "range_grid":
        dashboard._last_request.display_frame["sensing"]["range_axis_m"] *= 2
    elif change == "out_of_range":
        dashboard._linked_selection = replace(selected, cell=replace(selected.cell, range_bin=999))
    elif change == "variant":
        dashboard._last_request = replace(request, variant="raw")
    controller.sync()
    assert controller.overlay.current is None
    assert controller.identity is None


def test_explicit_shell_works_while_detection_overlays_are_hidden(shell):
    controller, dashboard = shell
    controller.viz.app_state.extension_state["sensing"]["show_detections"] = False
    controller.sync()
    assert controller.overlay.current is not None
    assert "range-bin geometry" in dashboard.range_shell_status.text()
    assert not controller.viz.app_state.extension_state["sensing"]["show_detections"]
    dashboard.range_shell_enabled.setChecked(False)
    controller.sync()
    assert controller.overlay.current is None


def test_new_numerical_support_replaces_shell_and_retirement_clears_it(shell):
    controller, dashboard = shell
    controller.sync()
    first = controller.identity
    dashboard._last_request.display_frame["sensing"]["range_edges_m"] = np.array(
        [10.0, 15.0, 25.0, 40.0]
    )
    controller.sync()
    assert controller.identity != first
    controller.close()
    assert controller.dashboard is None and controller.overlay.current is None


def test_scene_acceptance_unblocks_pending_selection(shell):
    controller, dashboard = shell
    controller.accepted_step = 999
    controller.sync()
    assert controller.overlay.current is None
    controller.present(dashboard._last_request.effective_display_step)
    assert controller.overlay.current is not None


def test_rd_heatmap_click_selects_exact_cell_and_outside_click_clears(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    request = replace(dashboard_request(), source_identity=("recording",))
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    window.update_from_request(request)
    emissions = []
    window.range_shell_changed.connect(lambda: emissions.append(window._linked_selection))
    rd = request.rd_result
    x = rd.range_axis_m[1]
    y = window._doppler_to_velocity(rd.doppler_axis_hz, request.carrier_freq_hz)[2]
    point = window.rd_plot.getViewBox().mapViewToScene(QPointF(x, y))
    event = SimpleNamespace(
        button=lambda: Qt.MouseButton.LeftButton,
        scenePos=lambda: point,
        accept=lambda: None,
        isAccepted=lambda: False,
    )
    window._select_rd_cell(event)
    assert window._linked_selection.cell.range_bin == 1
    assert window._linked_selection.cell.doppler_bin == 2
    assert len(window._selected_rd_cells.points()) == 1
    assert emissions
    assert not window.range_shell_enabled.isChecked()
    assert not window.selection_tools_toggle.isChecked()
    assert window.range_shell_cutaway.text() == "Horizontal cutaway"
    emissions.clear()
    window.range_shell_cutaway.setChecked(True)
    window.range_shell_enabled.setChecked(True)
    assert len(emissions) == 2
    window.set_linked_selection(None, notify=True)
    assert window._linked_selection is None
    window.close()


def test_frame_shell_is_explicit_and_uses_validated_world_bounds(shell):
    controller, dashboard = shell
    frames = []
    controller.viz.renderer.set_overview_camera = (
        lambda mode, bounds: frames.append((mode, bounds)) or True
    )
    controller.sync()
    assert frames == []
    assert controller.frame()
    assert frames[0][0] == "isometric"
    np.testing.assert_allclose(frames[0][1][0], controller.locus.center)
    dashboard._linked_selection = None
    assert not controller.frame()
    assert len(frames) == 1


@pytest.mark.parametrize("hertz", [False, True])
@pytest.mark.parametrize("detection", [False, True])
def test_second_click_in_same_saved_cell_clears_linked_displays(qtbot, hertz, detection):
    window = SensingWindow()
    qtbot.addWidget(window)
    request = replace(dashboard_request(), source_identity=("recording",))
    request.display_frame["sensing"]["detections"] = [
        dict(range_m=1.0, doppler_hz=0.0, velocity_m_s=0.0, range_bin=1, doppler_bin=2)
    ]
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    # Keep the whole cell visible: the offset below can land exactly on the
    # automatic zoom boundary and round outside the viewport after a resize.
    window._zoom_data_cb.setChecked(False)
    window.update_from_request(request)
    window._doppler_hz_cb.setChecked(hertz)
    rd = request.rd_result
    power = rd.power_linear.copy()
    changed = []
    window.selection_changed.connect(changed.append)

    def click(offset=0.0):
        qtbot.wait(1)  # The inspector changes the plot's geometry when it opens.
        y = (
            rd.doppler_axis_hz
            if hertz
            else window._doppler_to_velocity(rd.doppler_axis_hz, request.carrier_freq_hz)
        )
        pos = window.rd_plot.getViewBox().mapViewToScene(
            QPointF(rd.range_axis_m[1] + offset, y[2] + offset * (y[3] - y[2]))
        )
        qtbot.mouseClick(
            window.rd_plot.viewport(),
            Qt.MouseButton.LeftButton,
            pos=window.rd_plot.mapFromScene(pos),
        )

    if detection:
        window.select_detection_index(0)
        window.select_detection_index(0)  # Programmatic synchronization is idempotent.
    else:
        click()
    assert window._linked_selection is not None
    assert not window.cell_explanation.isVisible()
    window.selection_tools_toggle.setChecked(True)
    assert window.cell_explanation.isVisible()
    assert len(window._selected_rd_cells.points()) == 1
    click(0.2)  # A different screen coordinate within the same numerical bin.
    assert window._linked_selection is None
    assert changed[-1] is None
    assert not window.selection_tools.isVisible()
    assert not window.selection_tools_toggle.isVisible()
    assert not window.cell_explanation.isVisible()
    assert len(window._selected_rd_cells.points()) == 0
    assert not window.detection_table.selectionModel().hasSelection()
    assert window.rd_det_scatter.points()[0].pen().color().alphaF() == 1.0
    window.update_from_request(request)
    assert window._cell_selection_cleared()
    assert window._linked_selection is None
    click()
    assert window._linked_selection.cell.range_bin == 1
    click()  # Hit the yellow selected-cell marker itself, which consumes scene clicks.
    assert window._linked_selection is None
    if detection:
        # Clicking the report marker also clears an existing coordinate probe.
        window._select_inspection_coordinate(1, 2)
        window._select_detection_point(
            None, [SimpleNamespace(data=lambda: 0)], SimpleNamespace(accept=lambda: None)
        )
        assert window._linked_selection is None
    np.testing.assert_array_equal(rd.power_linear, power)
    window.update_from_request(replace(dashboard_request(1), source_identity=("recording",)))
    assert window._cell_selection_cleared()
    window.close()
