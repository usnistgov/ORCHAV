"""Target framing uses persisted truth, preserving plots, results and selection."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget

from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
from tests.visualizer.unit.test_sensing_window_micro_doppler_mode import (
    _truth_result_frame_for_targets,
)
from visualizer.src.sensing.dashboard_data import HistoryRecord, RecordingHistory
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.truth_zoom import _bounds
from visualizer.src.sensing.update_request import SensingUpdateRequest

pytestmark = pytest.mark.usefixtures("sensing_figure_preferences")


def request(step=0, rows=None, *, source=("recording",)):
    rows = rows if rows is not None else [("walker", 25.0 + step, 2.0), ("car", 140.0, -5.0)]
    frame = _truth_result_frame_for_targets(
        scene_frame_id=step, reference_time_s=step * 0.25, target_rows=rows
    )
    truth = frame["sensing"]
    axis, doppler = np.arange(201.0), np.arange(-16.0, 17.0)
    power = np.ones((len(doppler), len(axis)))
    power[20, 25:28] = 20
    frame["sensing"] = build_visualizer_sensing_payload(
        {
            "range_doppler_power_by_pair": power[None, ...],
            "range_axis_m": axis,
            "doppler_axis_hz": doppler,
            "config_resolved": {"carrier_freq_hz": 3e8},
            **{
                key: truth[key] for key in ("target_catalog", "target_state", "truth", "detections")
            },
        },
        scene_frame_id=step,
        sample_time_s=step * 0.25,
    )
    frame["sensing"]["temporal"]["truth"] = truth["temporal"]["truth"]
    rd = SimpleNamespace(
        power_linear=power, range_axis_m=axis, doppler_axis_hz=doppler, metadata={}
    )
    return SensingUpdateRequest(
        display_frame=frame,
        state_frame=frame,
        state_step=step,
        rd_result=rd,
        carrier_freq_hz=3e8,
        source_identity=source,
    )


@pytest.fixture
def dashboard(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window._set_view_mode("analysis")
    return window


def choose(window, identity):
    index = window.truth_zoom.combo.findData(identity)
    assert index >= 0
    window.truth_zoom.combo.setCurrentIndex(index)
    QApplication.processEvents()


def midpoint(plot, dimension=0):
    return np.mean(plot.viewRange()[dimension])


def test_target_zoom_follows_identity_and_units_without_changing_samples_or_selection(dashboard):
    first = request()
    dashboard.update_from_request(first)
    dashboard.select_detection_index(0)
    selected = dashboard._linked_selection
    powers = first.rd_result.power_linear.copy()
    profile = tuple(a.copy() for a in dashboard.rp_curve.getData())
    choose(dashboard, "walker")
    assert midpoint(dashboard.rd_plot) == pytest.approx(25)
    assert midpoint(dashboard.rd_plot, 1) == pytest.approx(2)
    np.testing.assert_allclose(dashboard.rd_plot.viewRange()[0], dashboard.rp_plot.viewRange()[0])
    assert dashboard.rd_plot.viewRange()[0][1] < 140
    assert dashboard._linked_selection == selected
    np.testing.assert_array_equal(first.rd_result.power_linear, powers)
    for before, after in zip(profile, dashboard.rp_curve.getData(), strict=True):
        np.testing.assert_array_equal(before, after)

    # Reordering the target catalog must not follow a different row.
    dashboard.update_from_request(request(4, [("car", 140, -5), ("walker", 29, 3)]))
    assert dashboard.truth_zoom.target_id == "walker"
    assert midpoint(dashboard.rd_plot) == pytest.approx(29)
    assert midpoint(dashboard.rd_plot, 1) == pytest.approx(3)
    dashboard._doppler_hz_cb.setChecked(True)
    expected = dashboard._velocity_to_doppler_hz(np.array([3.0]), 3e8)[0]
    assert midpoint(dashboard.rd_plot, 1) == pytest.approx(expected)
    # Reverse seek frames that acquisition's persisted truth, never an outer pose.
    dashboard.update_from_request(first)
    assert midpoint(dashboard.rd_plot) == pytest.approx(25)
    assert midpoint(dashboard.rd_plot) != first.display_frame["target_pos"][0, 0]


def test_zoom_off_and_data_zoom_restore_normal_axes(dashboard):
    dashboard.update_from_request(request())
    QApplication.processEvents()
    normal = dashboard.rd_plot.viewRange()[0]
    choose(dashboard, "car")
    assert midpoint(dashboard.rd_plot) == pytest.approx(140)
    choose(dashboard, None)
    QApplication.processEvents()
    np.testing.assert_allclose(dashboard.rd_plot.viewRange()[0], normal)
    choose(dashboard, "walker")
    dashboard._zoom_data_cb.setChecked(True)
    assert dashboard.truth_zoom.target_id is None
    assert dashboard.truth_zoom.combo.currentData() is None


def test_truth_zoom_is_scoped_to_supported_plots_and_resets_on_new_source(dashboard):
    dashboard.figure_options.expand()
    dashboard.update_from_request(request())
    assert dashboard.truth_zoom.combo.isVisible()
    assert set(dashboard.truth_zoom.menus) == {
        "rd_plot",
        "rp_plot",
        "gt_range_plot",
        "gt_speed_plot",
        "tracking_plot",
    }
    choose(dashboard, "walker")
    dashboard._set_view_mode("micro-doppler")
    assert dashboard.truth_zoom.combo.isHidden()
    dashboard._set_view_mode("analysis")
    dashboard.update_from_request(request(source=("other recording",)))
    assert dashboard.truth_zoom.target_id is None
    choose(dashboard, "walker")
    dashboard.set_unavailable("No result", scene_frame_id=99)
    assert not dashboard.truth_zoom.combo.isEnabled()
    dashboard.update_from_request(request(source=("other recording",)))
    assert dashboard.truth_zoom.target_id == "walker"


def test_unavailable_velocity_is_not_replaced_with_zero(dashboard):
    value = request()
    for target in value.display_frame["sensing"]["truth"]["targets"]:
        if target["target_id"] == "walker":
            obs = target["pair_observables"][0]
            obs.update(
                velocity_valid=False,
                velocity_derivation_source="unavailable",
                undefined_reason="No authored velocity",
            )
            for key in (
                "path_closing_rate_m_s",
                "monostatic_equivalent_velocity_m_s",
                "bistatic_bisector_velocity_m_s",
            ):
                obs[key] = None
    dashboard.update_from_request(value)
    y_before = dashboard.rd_plot.viewRange()[1]
    choose(dashboard, "walker")
    assert midpoint(dashboard.rd_plot) == pytest.approx(25)
    np.testing.assert_allclose(dashboard.rd_plot.viewRange()[1], y_before, atol=1e-4)
    assert dashboard.rd_plot.getViewBox().autoRangeEnabled()[1]
    assert _bounds([None, np.nan], 1.0) is None


def test_history_and_spatial_tracking_fit_only_selected_truth_prefix(dashboard):
    records = []
    for step in range(5):
        records.append(
            HistoryRecord(
                time_s=step * 0.25,
                identity=("clock", "run", "pair", str(step)),
                spectrum=None,
                doppler_hz=None,
                range_m=None,
                range_power=None,
                detections=(
                    {
                        "track_id": 3,
                        "range_m": 25 + step,
                        "velocity_m_s": 2 + step,
                        "world_track_position": (25 + step, 10 + step, 5 + step),
                        "source_kind": "saved_world_track",
                    },
                ),
                truth={
                    "target_ids": ("walker", "car"),
                    "target_ranges": (25 + step, 140),
                    "target_speeds": (2 + step, -5),
                },
                truth_positions=(
                    ("walker", (25 + step, 10 + step, 5 + step)),
                    ("car", (140, 100, 30)),
                ),
                cadence_s=0.25,
            )
        )
    history = RecordingHistory(tuple(records))
    value = replace(request(2), recording_managed=True, recording_history=history)
    dashboard._initial_view_selected = True
    dashboard.update_from_request(value)
    dashboard._set_view_mode("tracking")
    dashboard.update_from_request(value)
    choose(dashboard, "walker")
    assert midpoint(dashboard.gt_range_plot, 1) == pytest.approx(26)
    assert midpoint(dashboard.gt_speed_plot, 1) == pytest.approx(3)
    assert midpoint(dashboard.tracking_plot, 0) == pytest.approx(26)
    assert midpoint(dashboard.tracking_plot, 1) == pytest.approx(11)
    assert dashboard.gt_range_plot.viewRange()[0][1] < 1.0
    assert dashboard.tracking_plot.viewRange()[0][1] < 140
    dashboard.tracking_projection.setCurrentText("XZ")
    assert midpoint(dashboard.tracking_plot, 1) == pytest.approx(6)
    dashboard._show_det_cb.setChecked(False)
    assert midpoint(dashboard.gt_range_plot, 1) == pytest.approx(26)
    assert midpoint(dashboard.tracking_plot, 1) == pytest.approx(6)
    assert history.records == tuple(records)


def test_live_figure_menu_keeps_truth_zoom_available(dashboard, qtbot):
    dashboard.update_from_request(request())
    host = QWidget()
    qtbot.addWidget(host)
    dashboard.acquire_live_figure("rd_plot", host)
    menu = dashboard.truth_zoom.menus["rd_plot"]
    dashboard.truth_zoom._populate_menu(menu)
    next(action for action in menu.actions() if action.text() == "Walker").trigger()
    assert dashboard.truth_zoom.target_id == "walker"
    dashboard.release_live_figure("rd_plot")
