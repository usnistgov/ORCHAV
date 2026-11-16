"""Pinned sensing figures share plots and read only their visible products."""

from concurrent.futures import Future
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QVBoxLayout, QWidget

from tests.visualizer.unit.test_sensing_dashboard_recording import _frame, _history, _request
from tests.visualizer.unit.test_sensing_window_typed_angular import _angular_payload
from visualizer.src.sensing.dashboard_data import (
    DashboardSession,
    _dashboard_product_names,
    _request_product_key,
    load_dashboard_request,
    load_recording_history,
    result_context,
)
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.services.sensing_service import SensingService


@pytest.fixture
def figures(qtbot):
    window = SensingWindow()
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    window.update_from_request(_request(0, _history()))
    host = QWidget()
    host.setLayout(QVBoxLayout())
    host.show()
    yield window, host
    for key in tuple(window._live_figures):
        window.release_live_figure(key)
    host.close()
    window.close()


def _pin(window, host, key):
    wrapper = window.acquire_live_figure(key, host)
    host.layout().addWidget(wrapper)
    wrapper.show()
    return wrapper


def test_plot_moves_once_and_survives_tab_change_return_and_zoom(figures):
    window, host = figures
    plot = window.rd_plot
    plot.setRange(xRange=(0.2, 1.7), yRange=(-0.1, 0.1), padding=0)
    # A user gesture emits this after pyqtgraph updates the axes.
    plot.getViewBox().sigRangeChangedManually.emit([True, True])
    before = plot.viewRange()
    wrapper = _pin(window, host, "rd_plot")
    assert plot.parentWidget() is wrapper
    assert window._grid.indexOf(window._live_figures["rd_plot"].placeholder) >= 0
    with pytest.raises(ValueError, match="already open"):
        window.acquire_live_figure("rd_plot", host)
    window._set_view_mode("tracking")
    assert plot.parentWidget() is wrapper and plot.isVisible()
    np.testing.assert_allclose(plot.viewRange(), before)
    window.release_live_figure("rd_plot")
    assert window.rd_plot is plot
    window._set_view_mode("analysis")
    assert window._grid.indexOf(plot) >= 0


def test_hidden_dashboard_updates_only_pinned_rd_and_reports_binding(figures, monkeypatch):
    window, host = figures
    wrapper = _pin(window, host, "rd_plot")
    window.hide()
    window.refresh_live_figure_visibility()
    assert window.has_visible_live_figures()
    assert window.request_view_mode() == "live"
    assert window.live_figure_products() == ("range_doppler",)
    assert not window.wants_recording_history()
    for method in ("_update_range_profile", "_update_angular_products", "_render_recording"):
        monkeypatch.setattr(
            window, method, lambda *args, **kwargs: pytest.fail("hidden figure updated")
        )
    request = replace(
        _request(2, _history()), state_step=3, display_step=2, display_from_cache=True
    )
    window.update_from_request(request)
    assert window.rd_plot.isVisible()
    assert window.rd_image.image is not None
    assert "Result 0.500 s" in window._live_figures["rd_plot"].caption.text()
    assert "held for scene 3" in window._live_figures["rd_plot"].caption.text()
    wrapper.hide()
    assert not window.has_visible_live_figures()
    assert window.live_figure_request_suffix() == ()


@pytest.mark.parametrize("range_origin", ("automatic", "manual", "refitted"))
def test_profile_pin_follows_results_unless_the_current_range_was_inspected(figures, range_origin):
    window, host = figures
    plot = window.rp_plot
    plot.setXRange(0.0, 2.0, padding=0)
    plot.setYRange(-50.0, 60.0, padding=0)
    if range_origin != "automatic":
        plot.getViewBox().sigRangeChangedManually.emit([True, True])
    if range_origin == "refitted":
        plot.setYRange(-40.0, 50.0, padding=0)
    before = plot.viewRange()[1]
    _pin(window, host, "rp_plot")
    window.hide()
    window.update_from_request(_request(4))
    if range_origin == "manual":
        np.testing.assert_allclose(plot.viewRange()[1], before)
    else:
        assert not np.allclose(plot.viewRange()[1], before)

    # An intentional gesture while the figure is pinned still holds its view.
    plot.setYRange(-45.0, 55.0, padding=0)
    plot.getViewBox().sigRangeChangedManually.emit([True, True])
    window.update_from_request(_request(5))
    np.testing.assert_allclose(plot.viewRange()[1], [-45.0, 55.0])


def test_two_pins_share_history_without_updating_hidden_speed_plot(figures, monkeypatch):
    window, host = figures
    _pin(window, host, "rd_plot")
    _pin(window, host, "gt_range_plot")
    window.hide()
    monkeypatch.setattr(
        window._gt_speed_det_scatter, "setData", lambda *a, **k: pytest.fail("hidden speed chart")
    )
    window.update_from_request(_request(3, _history()))
    assert window.rd_plot.isVisible() and window.gt_range_plot.isVisible()
    assert window.wants_recording_history()
    assert window.request_view_mode() == "tracking"
    assert window._show_det_cb.text() == "Detections"


@pytest.mark.parametrize("figure", ["rd_plot", "gt_range_plot"])
def test_motion_cursor_shortcut_requires_no_visible_live_figures(figures, figure):
    window, host = figures
    history = _history()
    window._set_view_mode("micro-doppler")
    window.update_from_request(_request(0, history))
    assert window.can_move_recording_cursor(history)
    wrapper = _pin(window, host, figure)
    assert not window.can_move_recording_cursor(history)
    window.update_from_request(_request(2, history))
    assert "Result 0.500 s" in window._live_figures[figure].caption.text()
    wrapper.hide()
    assert window.can_move_recording_cursor(history)


def test_missing_result_clears_pinned_visual_but_keeps_demand_and_catalog(figures):
    window, host = figures
    _pin(window, host, "rd_plot")
    choices = window.available_live_figures()
    window.hide()
    window.set_unavailable("No result bound to scene 3", scene_frame_id=3)
    assert not window.rd_plot.isVisible()
    assert window.has_visible_live_figures()
    assert window.available_live_figures() == choices
    assert "No result" in window._live_figures["rd_plot"].caption.text()
    window.update_from_request(_request(4))
    assert window.rd_plot.isVisible()


@pytest.mark.parametrize("mode", ["summary", "micro-doppler"])
def test_pinned_map_does_not_retain_previous_truth_in_other_workspaces(figures, mode):
    window, host = figures
    _pin(window, host, "rd_plot")
    window._set_view_mode(mode)
    window._target_history = {"previous": {"times_s": [0.0], "range": [19.0], "speed": [2.0]}}
    window.update_from_request(_request(4, _history()))
    assert len(window.rd_gt_scatter.getData()[0]) == 0


def test_catalog_hides_unrecorded_maps_and_declares_exact_extra_products(figures):
    window, _host = figures
    window.update_from_request(
        replace(_request(), rd_result=None, available_product_kinds=frozenset())
    )
    choices = dict(window.available_live_figures())
    assert "rd_plot" not in choices and "rp_plot" not in choices
    assert "range_azimuth_plot" not in choices
    assert "gt_range_plot" in choices
    assert _dashboard_product_names("summary", ("range_doppler_angle",)) == (
        "config_resolved",
        "range_doppler_angle",
    )
    key = (3, "analysis", "primary", False, "auto", ("range_doppler_angle",))
    # A pinned angle figure extends the same retained result; its readiness
    # still determines whether the worker must prepare the projection.
    assert _request_product_key(key) == key[:5]


def test_profile_pin_prefers_saved_profile_over_loading_rd(figures):
    window, host = figures
    window.update_from_request(
        replace(_request(), available_product_kinds=frozenset({"range_profile", "range_doppler"}))
    )
    _pin(window, host, "rp_plot")
    window.hide()
    assert window.live_figure_products() == ("range_profile",)
    window.set_unavailable("No result bound to scene 3", scene_frame_id=3)
    assert window.live_figure_products() == ("range_profile",)
    window.update_from_request(
        replace(_request(), available_product_kinds=frozenset({"range_doppler"}))
    )
    assert window.live_figure_products() == ("range_doppler",)


def test_angle_pin_updates_only_visible_projection(figures, monkeypatch):
    window, host = figures
    frame = {**_frame(), "sensing": _angular_payload()}
    request = replace(
        _request(),
        display_frame=frame,
        state_frame=frame,
        available_product_kinds=frozenset({"range_doppler_angle"}),
    )
    window.update_from_request(request)
    _pin(window, host, "range_azimuth_plot")
    window.hide()
    for owner, method in (
        (window, "_update_angular_spectrum"),
        (window.doppler_azimuth_image, "setData"),
        (window.rd_image, "setImage"),
    ):
        monkeypatch.setattr(owner, method, lambda *a, **k: pytest.fail("hidden chart updated"))
    window.update_from_request(request)
    assert window.range_azimuth_image.z is not None
    assert window.live_figure_products() == ("range_doppler_angle",)


def test_release_retires_wrapper_without_deleting_plot(figures, qtbot):
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import isValid

    window, host = figures
    plot = window.rd_plot
    wrapper = _pin(window, host, "rd_plot")
    window.release_live_figure("rd_plot")
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(wrapper)
    assert isValid(plot)


def test_history_pin_reads_estimates_without_saved_spectral_matrices(monkeypatch):
    frames = [_frame(0), _frame(1)]
    service = SensingService(SimpleNamespace())
    monkeypatch.setattr(service, "dashboard_result_rows", lambda: list(enumerate(frames)))
    calls = []
    monkeypatch.setattr(
        service,
        "dashboard_load_row",
        lambda index, products: calls.append(products) or frames[index],
    )
    monkeypatch.setattr(service, "dashboard_scene_frame", lambda _: {})
    monkeypatch.setattr(service, "dashboard_world_detections", lambda _: [])
    history = load_recording_history(
        service, result_context(frames[0]["sensing"]), "primary", Event(), False
    )
    assert calls == [("config_resolved",)] * 2
    assert len(history.records) == 2
    assert all(r.spectrum is None and r.micro_power is None for r in history.records)


def test_metadata_view_with_rd_pin_materializes_only_requested_saved_products(monkeypatch):
    frame = _frame()
    service = SensingService(SimpleNamespace())
    captured = []
    monkeypatch.setattr(service, "dashboard_scene_frame", lambda _: frame)
    monkeypatch.setattr(service, "dashboard_product_kinds", lambda _: frozenset({"range_doppler"}))

    def resolve(*args, **kwargs):
        captured.append(kwargs["product_names"])
        return SimpleNamespace(
            display_frame=frame,
            bound_scene_frame_id=0,
            stale_for_current_scene=False,
            storage_frame_id=0,
            replay_event=None,
        )

    monkeypatch.setattr(service, "resolve_display_sensing_frame", resolve)
    request = load_dashboard_request(
        service, 0, "summary", "primary", False, "auto", ("range_doppler",)
    )
    assert captured == [("config_resolved", "range_doppler")]
    assert request.rd_result is not None


def test_reopened_history_cache_upgrades_spectral_data_for_normal_dashboard(monkeypatch):
    provider = SimpleNamespace()
    first = DashboardSession(provider)
    first.history = _history()
    first.history_key = (*result_context(_frame()["sensing"]), "primary")
    first.history_inventory = ()
    first._history_signal_products = False
    first.close()
    reopened = DashboardSession(provider, previous=first)
    submitted = []
    monkeypatch.setattr(reopened.service, "dashboard_inventory_token", lambda: ())
    monkeypatch.setattr(
        reopened._executor,
        "submit",
        lambda *args, **kwargs: submitted.append((args, kwargs)) or Future(),
    )
    try:
        reopened.request_history(_frame()["sensing"], "primary", include_signal_products=False)
        assert not submitted
        reopened.request_history(_frame()["sensing"], "primary")
        assert len(submitted) == 1
        assert submitted[0][0][-1] is True
        assert submitted[0][1]["cache_policy"] is reopened.cache_policy
        assert reopened.history is None
    finally:
        reopened.close()
