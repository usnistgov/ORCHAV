"""Range–velocity overlays stay legible through selection and theme changes."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt

from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
from visualizer.src.app.theme import get_theme_manager
from visualizer.src.sensing.dashboard_data import result_context
from visualizer.src.sensing.display_style import detection_color
from visualizer.src.sensing.sensing_window import SensingWindow

pytestmark = pytest.mark.usefixtures("sensing_figure_preferences")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_bin_selection_keeps_detections_visible_above_heatmap(qtbot, monkeypatch, theme):
    manager = get_theme_manager()
    original_theme = manager.mode
    window = SensingWindow()
    request = _request()
    detections = [
        dict(range_m=1.0, velocity_m_s=0.0, doppler_hz=0.0, range_bin=1, doppler_bin=2),
        dict(
            range_m=2.0,
            velocity_m_s=0.0,
            doppler_hz=0.0,
            range_bin=2,
            doppler_bin=2,
            track_id=2,
        ),
    ]
    request.display_frame["sensing"]["detections"] = detections
    power = request.rd_result.power_linear.copy()
    monkeypatch.setattr(window, "_current_gt_range_speed", lambda: ([1.5], [0.0]))
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    window._show_gt_cb.setChecked(True)
    try:
        manager.set_mode(theme, persist=False)
        window.update_from_request(request)

        def check_markers():
            for point, detection in zip(window.rd_det_scatter.points(), detections):
                assert point.brush().color().name() == detection_color(detection)
                assert point.pen().color().name() != point.brush().color().name()
                assert point.pen().color().alphaF() == 1.0
                assert point.brush().color().alphaF() == 1.0
            for scatter in (window.rd_det_scatter, window.rd_gt_scatter):
                for point in scatter.points():
                    pos = window.rd_plot.getViewBox().mapViewToScene(QPointF(point.pos()))
                    stack = window.rd_plot.scene().items(pos)
                    assert stack.index(scatter) < stack.index(window.rd_image)
            for _, label in window.rd_plot.getPlotItem().legend.items:
                assert manager.current_theme.text_primary in label.item.toHtml()

        qtbot.waitUntil(check_markers)
        window._select_inspection_coordinate(1, 2)
        qtbot.waitUntil(check_markers)
        assert len(window._selected_rd_cells.points()) == 1
        window.update_from_request(request)
        qtbot.waitUntil(check_markers)

        # Selecting an estimate still emphasizes its identity over other estimates.
        window.select_detection_index(0)
        selected, other = window.rd_det_scatter.points()
        assert selected.brush().color().alphaF() == 1.0
        assert other.brush().color().alphaF() < 0.3
        assert selected.size() > other.size()
        window.set_linked_selection(None)
        qtbot.waitUntil(check_markers)
        np.testing.assert_array_equal(request.rd_result.power_linear, power)
    finally:
        window.close()
        manager.set_mode(original_theme, persist=False)


def test_profile_detection_lines_follow_saved_ranges_without_changing_power_or_zoom(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    request = _request()
    request.display_frame["sensing"]["detections"] = [
        dict(range_m=0.75, velocity_m_s=0.0),
        dict(range_m=0.75, velocity_m_s=1.0),
        dict(range_m=1.8, velocity_m_s=-1.0, track_id=2),
    ]
    window.update_from_request(request)
    samples = tuple(array.copy() for array in window.rp_curve.getData())
    ranges = window.rp_plot.viewRange()
    assert not window._profile_detection_lines
    window._profile_detections_cb.setChecked(True)
    lines = window._profile_detection_lines
    assert [line.value() for line in lines] == [0.75, 1.8]
    assert all(line.pen.style() == Qt.PenStyle.DashLine for line in lines)
    for actual, expected in zip(window.rp_curve.getData(), samples):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(window.rp_plot.viewRange(), ranges)
    window._show_det_cb.setChecked(False)
    assert not any(line.isVisible() for line in lines)
    window._show_det_cb.setChecked(True)
    assert all(line.isVisible() for line in lines)
    request.display_frame["sensing"]["detections"] = [dict(range_m=1.2)]
    window.update_from_request(request)
    assert lines[0].isVisible() and lines[0].value() == 1.2
    assert not lines[1].isVisible()
    window.reset_recording()
    assert not any(line.isVisible() for line in lines)


@pytest.mark.parametrize("recording", [False, True])
def test_detection_colors_match_maps_profiles_and_histories_through_seek_and_selection(
    qtbot, recording
):
    requests, records = [], []
    for step, row in enumerate(_history().records):
        request = _request(step)
        sensing = request.display_frame["sensing"]
        detections = [
            dict(range_m=0.5, velocity_m_s=0.0, track_id=1),
            dict(range_m=1.0, velocity_m_s=0.5, track_id=2),
            dict(range_m=1.5, velocity_m_s=-0.5),
        ]
        sensing["detections"] = detections
        requests.append(request)
        records.append(
            replace(
                row,
                detections=tuple(detections),
                identity=(
                    *result_context(sensing),
                    sensing["temporal"]["acquisition"]["acquisition_id"],
                ),
            )
        )
    history = replace(_history(), records=tuple(records))
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    window._profile_detections_cb.setChecked(True)

    def check_history(*, selected=False):
        colors = [detection_color(d) for d in detections]
        for scatter in (window._gt_range_det_scatter, window._gt_speed_det_scatter):
            for index, point in enumerate(scatter.points()):
                for color in (point.pen().color(), point.brush().color()):
                    assert color.name() == colors[index % 3]
                    assert color.alphaF() == pytest.approx(
                        0.18 if selected and index % 3 != 0 else 1.0, abs=0.01
                    )

    for step in (0, 1, 4, 1):
        window.update_from_request(
            replace(
                requests[step],
                recording_managed=recording,
                recording_history=history if recording else None,
            )
        )
        check_history()
        for point, line, detection in zip(
            window.rd_det_scatter.points(), window._profile_detection_lines, detections, strict=True
        ):
            assert (
                point.brush().color().name()
                == line.pen.color().name()
                == detection_color(detection)
            )
    if recording:
        window.select_detection_index(0)
        check_history(selected=True)
        window._select_inspection_coordinate(1, 2)
        check_history()
        window.select_detection_index(0)
        window.set_linked_selection(None)
        check_history()
        scatters = (window._gt_range_det_scatter, window._gt_speed_det_scatter)
        sprite_counts = [len(scatter.fragmentAtlas) for scatter in scatters]
        for _ in range(8):
            window.update_from_request(window._last_request)
            window.select_detection_index(0)
            window.set_linked_selection(None)
        # Replaying and highlighting the same identities must not accumulate
        # newly rasterized copies of identical marker styles.
        assert [len(scatter.fragmentAtlas) for scatter in scatters] == sprite_counts
        check_history()
    window.reset_recording()
    assert window._detection_history_colors == []
