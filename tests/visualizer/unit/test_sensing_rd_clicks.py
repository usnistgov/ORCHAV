"""Ground-truth context must not consume numerical range–Doppler selection."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt

from tests.visualizer.unit.test_sensing_dashboard_recording import _request
from visualizer.src.sensing.sensing_window import SensingWindow


@pytest.mark.parametrize("hertz", [False, True])
@pytest.mark.parametrize("overlap_detection", [False, True])
def test_ground_truth_marker_allows_underlying_saved_cell_or_detection_pick(
    qtbot, monkeypatch, hertz, overlap_detection
):
    request = _request()
    request = replace(
        request,
        source_identity=("recording",),
        carrier_freq_hz=request.display_frame["sensing"]["config_resolved"]["carrier_freq_hz"],
    )
    rd = request.rd_result
    saved_power = rd.power_linear.copy()
    window = SensingWindow()
    qtbot.addWidget(window)
    window.resize(1200, 1000)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    window._zoom_data_cb.setChecked(False)
    window._show_gt_cb.setChecked(True)
    window._show_det_cb.setChecked(True)
    window._doppler_hz_cb.setChecked(hertz)
    velocity = window._doppler_to_velocity(rd.doppler_axis_hz, request.carrier_freq_hz)
    # A truth position is continuous; it need not coincide with a cell center.
    # Coincident reports keep their own intentional higher-priority hit target.
    offset = 0.0 if overlap_detection else 0.2
    truth_range = rd.range_axis_m[1] + offset
    truth_velocity = velocity[2] + offset * (velocity[3] - velocity[2])
    monkeypatch.setattr(
        window, "_current_gt_range_speed", lambda: ([truth_range], [truth_velocity])
    )
    if overlap_detection:
        request.display_frame["sensing"]["detections"] = [
            dict(
                range_m=float(rd.range_axis_m[1]),
                doppler_hz=float(rd.doppler_axis_hz[2]),
                velocity_m_s=float(velocity[2]),
                range_bin=1,
                doppler_bin=2,
            )
        ]
    window.update_from_request(request)
    qtbot.waitUntil(lambda: window.rd_plot.isVisible() and window.rd_plot.width() > 300)
    previous, unchanged = None, 0

    def stable():
        nonlocal previous, unchanged
        transform = window.rd_plot.getViewBox().childTransform()
        geometry = (transform.m11(), transform.m22(), transform.dx(), transform.dy())
        unchanged = unchanged + 1 if geometry == previous else 0
        previous = geometry
        return unchanged >= 4 and not window.rd_plot._legend_timer.isActive()

    qtbot.waitUntil(stable)
    assert window.rd_gt_scatter.isVisible()
    assert len(window.rd_gt_scatter.points()) == 1
    if overlap_detection:
        assert window.rd_det_scatter.isVisible()
        assert len(window.rd_det_scatter.points()) == 1

    def click(button=Qt.MouseButton.LeftButton):
        y = (
            float(
                window._velocity_to_doppler_hz(np.array([truth_velocity]), request.carrier_freq_hz)[
                    0
                ]
            )
            if hertz
            else truth_velocity
        )
        assert window.rd_gt_scatter.points()[0].pos().x() == pytest.approx(truth_range)
        assert window.rd_gt_scatter.points()[0].pos().y() == pytest.approx(y)
        scene = window.rd_plot.getViewBox().mapViewToScene(QPointF(truth_range, y))
        position = window.rd_plot.mapFromScene(scene)
        qtbot.mouseClick(window.rd_plot.viewport(), button, pos=position)

    click()
    selection = window._linked_selection
    assert selection is not None
    assert selection.cell.range_bin == 1
    assert selection.cell.doppler_bin == 2
    assert selection.detection_index == (0 if overlap_detection else None)
    assert selection.cell.origin == ("detection" if overlap_detection else "cell")
    assert len(window._selected_rd_cells.points()) == 1
    previous, unchanged = None, 0
    qtbot.waitUntil(stable)
    click()
    assert window._linked_selection is None
    assert len(window._selected_rd_cells.points()) == 0
    click(Qt.MouseButton.RightButton)
    assert window._linked_selection is None
    window.rd_plot.getPlotItem().getViewBox().menu.close()
    np.testing.assert_array_equal(rd.power_linear, saved_power)
