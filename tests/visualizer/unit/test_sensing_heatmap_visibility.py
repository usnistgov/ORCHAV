"""Optional map drawing retains sensing overlays and explicitly pinned figures."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, QSettings
from PySide6.QtWidgets import QVBoxLayout, QWidget

from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
from tests.visualizer.unit.test_sensing_window_typed_angular import _angular_payload
from visualizer.src.sensing.sensing_window import SensingWindow


@pytest.fixture
def window(qtbot):
    settings = QSettings()
    key = "visualizer/sensing_heatmaps"
    previous = settings.value(key)
    options_key = "visualizer/sensing_figure_options_open"
    previous_options = settings.value(options_key)
    settings.setValue(options_key, False)
    settings.setValue(key, True)
    window = SensingWindow()
    window.resize(1200, 900)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    yield window
    for key in tuple(window._live_figures):
        window.release_live_figure(key)
    window.close()
    key = "visualizer/sensing_heatmaps"
    if previous is None:
        settings.remove(key)
    else:
        settings.setValue(key, previous)
    if previous_options is None:
        settings.remove(options_key)
    else:
        settings.setValue(options_key, previous_options)


def _no_work(*_args, **_kwargs):
    pytest.fail("hidden heatmap was rebuilt")


def test_current_hidden_map_keeps_markers_profile_and_selection(window, monkeypatch):
    request = _request(0, _history())
    request.display_frame["sensing"]["detections"] = [
        {"range_m": 1.0, "velocity_m_s": 0.0, "doppler_hz": 0.0}
    ]
    window.update_from_request(request)
    window.select_detection_index(0)
    selection = window._linked_selection
    assert selection is not None
    profile = window.rp_curve.getData()
    original_power = request.rd_result.power_linear.copy()
    monkeypatch.setattr(window.rd_image, "setImage", _no_work)
    monkeypatch.setattr(window.rd_cfar_overlay, "setImage", _no_work)
    # The profile still needs its own numerical conversion; only RD levels must stop.
    original_levels = window._levels

    def levels(kind, *args, **kwargs):
        assert kind != "rd"
        return original_levels(kind, *args, **kwargs)

    monkeypatch.setattr(window, "_levels", levels)
    window._show_heatmaps_cb.setChecked(False)
    assert window._last_request is request
    assert window._linked_selection == selection
    assert not window.rd_image.isVisible()
    assert not window.rd_colorbar.isVisible()
    np.testing.assert_array_equal(window.rd_det_scatter.getData()[0], [1.0])
    np.testing.assert_array_equal(window.rp_curve.getData()[1], profile[1])
    np.testing.assert_array_equal(request.rd_result.power_linear, original_power)


def test_angular_hidden_maps_skip_mesh_work_keep_spectrum_and_detection_projection(
    window, monkeypatch
):
    window._set_view_mode("angular")
    payload = _angular_payload()
    payload["detections"] = [
        {"range_m": 1.0, "doppler_hz": 0.0, "aoa_az_deg": 20.0, "aoa_el_deg": 0.0}
    ]
    window._show_heatmaps_cb.setChecked(False)
    monkeypatch.setattr(window.range_azimuth_image, "setData", _no_work)
    monkeypatch.setattr(window.doppler_azimuth_image, "setData", _no_work)
    window._update_angular_products(payload, "primary", "", 28e9)
    assert not window.range_azimuth_image.isVisible()
    assert not window.doppler_azimuth_image.isVisible()
    assert window.angular_spectrum_image.isVisible()
    np.testing.assert_array_equal(window.range_azimuth_det_scatter.getData()[0], [1.0])
    assert len(window.doppler_azimuth_det_scatter.getData()[0]) == 1
    assert window._angular_shape == (2, 3, 2, 3)


@pytest.mark.parametrize("key", ["rd_plot", "range_azimuth_plot"])
def test_pinned_map_overrides_dashboard_preference(window, key):
    request = _request(0, _history())
    if key == "range_azimuth_plot":
        frame = {"sensing": _angular_payload()}
        request = replace(request, display_frame=frame, state_frame=frame)
        window._set_view_mode("angular")
    window.update_from_request(request)
    window._show_heatmaps_cb.setChecked(False)
    host = QWidget()
    host.setLayout(QVBoxLayout())
    host.show()
    wrapper = window.acquire_live_figure(key, host)
    host.layout().addWidget(wrapper)
    wrapper.show()
    window.hide()
    window.update_from_request(request)
    image = window.rd_image if key == "rd_plot" else window.range_azimuth_image
    assert image.isVisible()
    assert window._heatmap_enabled(getattr(window, key))
    window.release_live_figure(key)
    host.close()


def test_heatmaps_reenable_from_current_request_and_persist_preference(window):
    request = _request(0, _history())
    window.update_from_request(request)
    window._show_heatmaps_cb.setChecked(False)
    other = SensingWindow()
    assert not other._show_heatmaps_cb.isChecked()
    other.close()
    changes = []
    window.view_changed.connect(lambda: changes.append(True))
    window._show_heatmaps_cb.setChecked(True)
    assert window.rd_image.isVisible()
    np.testing.assert_allclose(
        window.rd_image.image, 10 * np.log10(request.rd_result.power_linear.T)
    )
    assert window._last_request is request
    assert not changes, "display option must not submit a new reader request"


@pytest.mark.parametrize("width", [460, 1280])
def test_expanded_display_settings_fit_without_clipping_controls(window, qtbot, width):
    window.resize(width, 950)
    window.update_from_request(_request(0, _history()))
    window.figure_options.expand()
    qtbot.wait(20)
    assert window._advanced_widget.isVisible()

    def controls_fit():
        settings = window._advanced_widget
        for control in (window._rd_dynamic_range_box, window._show_heatmaps_cb):
            assert control.isVisible()
            bounds = QRect(control.mapTo(settings, QPoint()), control.size())
            assert settings.rect().contains(bounds)
        assert (
            window._show_heatmaps_cb.height() >= window._show_heatmaps_cb.minimumSizeHint().height()
        )

    qtbot.waitUntil(controls_fit)


@pytest.mark.parametrize("mode", ["angular", "diagnostics"])
def test_hidden_truth_histories_are_not_drawn_but_result_history_still_advances(
    window, monkeypatch, mode
):
    window._set_view_mode(mode)
    assert not window.gt_range_plot.isVisible()
    assert not window.gt_speed_plot.isVisible()
    monkeypatch.setattr(window, "_render_gt_curves", _no_work)
    first = _request(0)
    second = _request(1)
    for request in (first, second):
        request.display_frame["sensing"]["detections"] = [{"range_m": 1.0, "velocity_m_s": 0.0}]
    window.update_from_request(first)
    window.update_from_request(second)
    assert window._last_history_reference_time_s == 0.25
    assert window._last_history_result_identity is not None
    assert window._detection_history["times_s"] == [0.0, 0.25]
