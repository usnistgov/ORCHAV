"""Saved states stay separate from measurements, sources and inferred coasts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.sensing.tracking_plot_data import prepare_tracking_plot_data
from visualizer.src.sensing.tracking_presentation import (
    covariance_ellipse,
    state_uncertainty,
    tracking_snapshot,
)


def native_sensing():
    return {
        "temporal": {
            "clock": {"clock_id": "clock"},
            "result": {"processing_run_id": "run", "reference_time_s": 1.0},
            "acquisition": {"pair_id": "pair", "acquisition_id": "cpi-1"},
        },
        "detections": [
            {
                "track_id": 7,
                "track_confirmed": True,
                "range_m": 11.0,
                "velocity_m_s": 2.0,
                "track_range_m": 10.8,
                "track_velocity_m_s": 1.8,
            }
        ],
        "detection_filter_metadata": {"tracker": {"enabled": True}},
    }


def test_hidden_inspector_reuses_snapshot_without_rebuilding_track_choices(qtbot, monkeypatch):
    from visualizer.src.sensing import tracking_inspector

    widget = tracking_inspector.TrackingInspector()
    qtbot.addWidget(widget)
    sensing = native_sensing()
    snapshot = tracking_snapshot(sensing)
    calls = []

    def detailed_snapshot(*args, **kwargs):
        assert kwargs.get("include_decisions"), "Shared snapshot should be reused"
        calls.append(True)
        return tracking_snapshot(*args, **kwargs)

    monkeypatch.setattr(tracking_inspector, "tracking_snapshot", detailed_snapshot)
    widget.update_result(None, 1.0, "analysis", sensing=sensing, snapshot=snapshot)
    widget.select_track(7)
    assert widget.selected_track()["estimated_range_m"] == 10.8
    assert widget.tracks.count() == 0
    widget.update_result(None, 1.0, "tracking", sensing=sensing, snapshot=snapshot)
    assert widget.tracks.count() == 0  # Choices are inside the collapsed details.
    assert calls == []
    widget.toggle.setChecked(True)
    assert widget.tracks.currentData() == 7
    assert widget.tracks.count() == 2
    assert calls == [True]


def with_decisions():
    sensing = native_sensing()
    sensing["detection_filter_metadata"]["tracker"]["decisions"] = {
        "schema_version": 1,
        "update_time_s": 1.0,
        "state_labels": ["range_m", "closing_velocity_m_s"],
        "tracks": [
            {
                "track_id": 7,
                "mean": [10.8, 1.8],
                "confirmed": True,
                "misses": 0,
                "hits": 5,
                "covariance": [[4.0, 0.2], [0.2, 0.25]],
            },
            {
                "track_id": 8,
                "mean": [5.0, 0.0],
                "confirmed": True,
                "misses": 2,
                "hits": 4,
                "covariance": None,
            },
        ],
        "assignments": [{"track_id": 7, "detection_index": 3}],
        "candidates": [
            {
                "track_id": 7,
                "detection_index": 3,
                "admissible": True,
                "cost": 0.2,
                "probability": None,
            }
        ],
    }
    return sensing


def test_native_states_and_input_indices_are_not_emitted_rows():
    sensing = with_decisions()
    original = deepcopy(sensing)
    snapshot = tracking_snapshot(sensing)
    observed, coast = snapshot.states
    assert snapshot.complete_states
    assert observed["estimated_range_m"] == 10.8
    assert observed["evidence"]["detection_index"] == 0
    assert snapshot.associations[0]["detection_index"] == 3
    assert snapshot.associations[0]["outcome"] == "associated"
    assert coast["status"] == "coasting" and coast["evidence"] is None
    assert sensing == original


def test_emitted_outputs_do_not_claim_complete_population_or_covariance():
    snapshot = tracking_snapshot(native_sensing())
    assert not snapshot.complete_states
    assert len(snapshot.states) == 1
    assert snapshot.states[0]["radial_velocity_m_s"] == 1.8
    assert state_uncertainty(snapshot.states[0]) == ()
    other = native_sensing()
    other["temporal"]["result"]["processing_run_id"] = "other run"
    assert tracking_snapshot(other).source_key != snapshot.source_key


def test_cartesian_output_does_not_relabel_horizontal_range_or_doppler_as_radial_state():
    sensing = native_sensing()
    sensing["detections"][0].update(
        track_x_m=6.0,
        track_y_m=8.0,
        track_z_m=2.0,
        track_vx_m_s=1.0,
        track_vy_m_s=0.0,
        track_vz_m_s=0.0,
    )
    state = tracking_snapshot(sensing).states[0]
    assert state["estimated_range_m"] is None
    assert state["radial_velocity_m_s"] is None
    assert state["position_m"] == [6, 8, 2]
    assert state["position_frame"] == "receiver local"


def test_history_estimates_use_filtered_values_and_preserve_coast_without_measurement():
    sensing = with_decisions()
    row = SimpleNamespace(
        time_s=1.0,
        detections=sensing["detections"],
        tracking_states=tracking_snapshot(sensing).states,
        truth_positions=(),
        truth=None,
    )
    data = prepare_tracking_plot_data(SimpleNamespace(records=[row]))
    np.testing.assert_allclose(data.detections.values, [[11, 2]])
    np.testing.assert_allclose(data.estimates["7"].values, [[10.8, 1.8]])
    np.testing.assert_allclose(data.estimates["8"].values, [[5, 0]])
    np.testing.assert_allclose(data.statuses["8"].values, [[1, 1]])
    assert len(data.estimates["7"].through(0.9)[0]) == 0


def test_projection_and_covariance_keep_cartesian_coordinate_semantics():
    row = SimpleNamespace(
        time_s=1.0,
        detections=(),
        truth_positions=(),
        truth=None,
        tracking_states=({"track_id": 1, "position_frame": "world", "position_m": [1, 2, 3]},),
    )
    series = prepare_tracking_plot_data(SimpleNamespace(records=[row])).tracks["1"]
    np.testing.assert_allclose(series.projected("xz").values, [[1, 3]])
    np.testing.assert_allclose(series.projected("yz").values, [[2, 3]])
    state = {
        "state_labels": ("x_m", "y_m", "z_m"),
        "mean": [1, 2, 3],
        "covariance": np.diag([4, 9, 16]).tolist(),
    }
    ellipse = covariance_ellipse(state, "xz")
    np.testing.assert_allclose(ellipse.min(axis=0), [-1, -1])
    np.testing.assert_allclose(ellipse.max(axis=0), [3, 7])
    assert state_uncertainty(state) == (("x_m", 2), ("y_m", 3), ("z_m", 4))
    assert covariance_ellipse(tracking_snapshot(native_sensing()).states[0]) is None


class SavedReader:
    inspection_id = "saved-run"
    times = np.array([1.0])
    path = None

    def __getitem__(self, key):
        return {"sensor": {"sensor": {"position_m": [0, 0, 0]}}}

    def epoch_index(self, time_s):
        return 0

    def epoch(self, index):
        return {
            "tracks": [
                {
                    "track_id": 1,
                    "position_m": [10, 0, 0],
                    "velocity_m_s": [-2, 1, 0],
                    "confirmed": False,
                    "predicted_only": True,
                    "sources": [{"sensor_id": "sensor", "native_source": {"detection_index": 0}}],
                }
            ]
        }

    def decisions(self, index):
        pytest.fail("Summary/browser must not read indexed diagnostic files")


def test_saved_states_require_exact_epoch_preserve_tentative_coast_and_skip_decision_reads():
    binding = SimpleNamespace(reader=SavedReader(), sensor_id="sensor")
    snapshot = tracking_snapshot(None, saved_tracking=binding, time_s=1.0)
    state = snapshot.states[0]
    assert state["confirmed"] is False and state["predicted_only"] is True
    assert state["evidence"] is None
    assert state["radial_velocity_m_s"] == 2.0
    assert not tracking_snapshot(None, saved_tracking=binding, time_s=1.1).states


def test_inspector_exposes_native_association_and_uncertainty_without_json(qtbot):
    from visualizer.src.sensing.tracking_inspector import TrackingInspector

    widget = TrackingInspector()
    qtbot.addWidget(widget)
    widget.update_result(None, 1.0, "tracking", sensing=with_decisions())
    widget.select_track(7)
    assert not widget.toggle.isChecked()
    widget.toggle.setChecked(True)
    assert widget.state_table.topLevelItemCount() >= 8
    assert widget.associations.rowCount() == 1
    assert widget.uncertainty.topLevelItemCount() == 2
    assert widget.evidence.isEnabled()
    widget.select_track(8)
    assert not widget.evidence.isEnabled()
    widget.select_track(None)
    assert widget.selected_track() is None
    assert widget.tracks.currentData() is None
    assert not widget.evidence.isEnabled()
    assert widget.state_table.topLevelItemCount() == 1
    assert widget.state_table.topLevelItem(0).text(0) == "Selection"


def test_history_retains_states_and_browser_metadata_without_detector_arrays(monkeypatch):
    from threading import Event

    from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
    from visualizer.src.sensing.dashboard_data import load_recording_history, result_context
    from visualizer.src.services.sensing_service import SensingService

    payload = with_decisions()
    payload.pop("temporal")
    sensing = build_visualizer_sensing_payload(payload, sample_time_s=1.0)
    frame = {"sensing": sensing}
    service = SensingService(SimpleNamespace())
    monkeypatch.setattr(service, "dashboard_result_rows", lambda: [(0, frame)])
    monkeypatch.setattr(service, "dashboard_history_products", lambda *_: (frame, None, None))
    monkeypatch.setattr(service, "dashboard_scene_frame", lambda _: {})
    monkeypatch.setattr(service, "dashboard_world_detections", lambda _: [])
    history = load_recording_history(service, result_context(sensing), "primary", Event())
    record = history.records[0]
    assert len(record.tracking_states) == 2
    assert record.tracking_states_complete
    assert record.tracking_states[1]["predicted_only"] is True
    assert "temporal" not in record.sensing_metadata
    assert "detection_filter_metadata" not in record.sensing_metadata
    assert record.sensing_metadata["config_resolved"] == {
        "antenna_rows": sensing["config_resolved"].get("antenna_rows")
    }
    assert record.sensing_metadata["detections"][0]["range_m"] == 11.0
    assert all(
        set(product) <= {"kind", "variant", "processing"}
        for product in record.sensing_metadata["products"].values()
    )


def test_motion_browser_retains_world_track_population_and_native_detection_indices():
    from visualizer.src.sensing.workspace_shell import WorkspaceShellMixin

    accepted = []
    shell = SimpleNamespace(
        _last_request=SimpleNamespace(
            source_identity=("recording",),
            saved_tracking=SimpleNamespace(reader=SavedReader(), sensor_id="sensor"),
        ),
        result_browser=SimpleNamespace(
            update_result=lambda *args, **kwargs: accepted.append((args, kwargs))
        ),
    )
    record = SimpleNamespace(
        time_s=1.0,
        identity=("clock", "run", "pair", "cpi-1"),
        detections=(),
        tracking_states=tracking_snapshot(with_decisions()).states,
        sensing_metadata=native_sensing(),
    )
    WorkspaceShellMixin._update_browser_cursor(shell, record)
    detections, tracks = accepted[-1][0]
    assert detections[0]["track_id"] == 7
    assert [state["track_id"] for state in tracks] == [1]
    assert tracks[0]["source_kind"] == "saved_world_track"
    assert tracks[0]["predicted_only"] is True
    record.time_s = 1.1
    WorkspaceShellMixin._update_browser_cursor(shell, record)
    assert accepted[-1][0][1] == ()


def test_recorded_marker_brushes_keep_coasting_and_reuse_sprite_identity(qtbot):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualizer.src.sensing.tracking_views import _recorded_symbol_brushes

    curve = pg.PlotDataItem()
    first = _recorded_symbol_brushes(curve, "#123456", [False, True, False])
    assert first[0] is first[2]
    assert [brush.style() for brush in first] == [
        Qt.BrushStyle.SolidPattern,
        Qt.BrushStyle.NoBrush,
        Qt.BrushStyle.SolidPattern,
    ]
    assert first[0].color().name() == "#123456"
    shorter = _recorded_symbol_brushes(curve, "#123456", [True])
    assert len(shorter) == 1 and shorter[0] is first[1]
    curve.setOpacity(0.18)
    assert first[0].color().alpha() == 255
    recolored = _recorded_symbol_brushes(curve, "#654321", [False, True])
    assert recolored[0].color().name() == "#654321"
    assert all(a is not b for a, b in zip(recolored, first, strict=False))
    replacement = _recorded_symbol_brushes(pg.PlotDataItem(), "#654321", [False, True])
    assert all(a is not b for a, b in zip(replacement, recolored, strict=True))


def test_track_paths_keep_coasting_gaps_current_marker_and_rewind(qtbot):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox, QComboBox

    from visualizer.src.sensing.tracking_plot_data import TimedValues
    from visualizer.src.sensing.tracking_views import TrackingViewsMixin

    class View(TrackingViewsMixin):
        def select_track(self, *_):
            pass

        def _get_target_label(self, identity):
            return identity

    view = View()
    view.tracking_projection = QComboBox()
    view.tracking_projection.addItems(["XY", "XZ", "YZ"])
    view.tracking_plot = pg.PlotWidget()
    qtbot.addWidget(view.tracking_projection)
    qtbot.addWidget(view.tracking_plot)
    view._show_det_cb, view._show_gt_cb = QCheckBox(), QCheckBox()
    view._show_det_cb.setChecked(True)
    view._show_gt_cb.setChecked(True)
    view._tracking_curves = {}
    view._tracking_truth_curves = {}
    view._tracking_coast_curves = {}
    times = np.array([1.0, 2.0, 3.0])
    prepared = SimpleNamespace(
        tracks={"7": TimedValues(times, np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))},
        truths={},
        statuses={"7": TimedValues(times, np.array([[0, 1], [1, 1], [0, 1]]))},
        spatial_dimensions=3,
    )
    history = SimpleNamespace(tracking_plot_data=prepared)
    view._draw_track_paths(history, 3.0)
    curve = view._tracking_curves["7"]
    np.testing.assert_array_equal(curve.opts["connect"], [False, False, False])
    np.testing.assert_array_equal(
        view._tracking_coast_curves["7"].opts["connect"], [True, True, False]
    )
    np.testing.assert_array_equal(curve.opts["symbolSize"], [8, 8, 18])
    assert curve.opts["symbolBrush"][1].style() == Qt.BrushStyle.NoBrush
    observed = curve.opts["symbolBrush"][0]
    view._draw_track_paths(history, 1.0)
    np.testing.assert_array_equal(curve.getData(), [[1], [2]])
    np.testing.assert_array_equal(curve.opts["symbolSize"], [18])
    assert curve.opts["symbolBrush"] == [observed]
    assert curve.opts["symbolBrush"][0] is observed


def test_tracking_visibility_does_not_restore_estimates_after_rewind(qtbot):
    import pyqtgraph as pg
    from PySide6.QtWidgets import QCheckBox

    from visualizer.src.sensing.tracking_plot_data import TimedValues
    from visualizer.src.sensing.tracking_views import TrackingViewsMixin

    class View(TrackingViewsMixin):
        def _selected_view_mode(self):
            return "tracking"

        def select_track(self, *_):
            pass

    view = View()
    view._last_request = None
    view._show_det_cb = QCheckBox()
    view._show_det_cb.setChecked(True)
    view.tracking_measurements = QCheckBox()
    view.tracking_measurements.setChecked(True)
    view._track_estimate_curves = {}
    view._track_estimate_coasts = {}
    view._tracking_coast_curves = {}
    view.gt_range_plot, view.gt_speed_plot = pg.PlotWidget(), pg.PlotWidget()
    view._gt_range_det_scatter, view._gt_speed_det_scatter = (
        pg.ScatterPlotItem(),
        pg.ScatterPlotItem(),
    )
    for plot, scatter in (
        (view.gt_range_plot, view._gt_range_det_scatter),
        (view.gt_speed_plot, view._gt_speed_det_scatter),
    ):
        qtbot.addWidget(plot)
        plot.addLegend()
        plot.addItem(scatter)
    prepared = SimpleNamespace(
        estimates={"7": TimedValues(np.array([1.0]), np.array([[10, 2]]))}, statuses={}
    )
    view._draw_track_estimates(prepared, 2.0)
    assert view._track_estimate_curves[("7", 0)].isVisible()
    view._draw_track_estimates(prepared, 0.0)
    view._tracking_plot_labels()
    assert not view._track_estimate_curves[("7", 0)].isVisible()
    assert not view._track_estimate_coasts[("7", 0)].isVisible()


@pytest.mark.parametrize("saved_world", [False, True])
def test_track_highlight_covers_radial_spatial_and_coasting_then_clears(saved_world):
    from visualizer.src.sensing.tracking_views import TrackingViewsMixin

    class Curve:
        _sensing_base_opacity = 0.35

        def setOpacity(self, opacity):
            self.opacity = opacity

    curves = [Curve() for _ in range(8)]
    view = SimpleNamespace(
        _last_request=SimpleNamespace(saved_tracking=object() if saved_world else None),
        tracking_inspector=SimpleNamespace(selected_id=7 if saved_world else None),
        _linked_selection=None if saved_world else SimpleNamespace(track_id=7),
        _tracking_curves={"7": curves[0], "8": curves[1]},
        _tracking_coast_curves={"7": curves[2], "8": curves[3]},
        _track_estimate_curves={("7", 0): curves[4], ("8", 0): curves[5]},
        _track_estimate_coasts={("7", 0): curves[6], ("8", 0): curves[7]},
    )
    TrackingViewsMixin._highlight_track_selection(view)
    assert all(curve.opacity == 0.35 for curve in curves[::2])
    assert all(curve.opacity == pytest.approx(0.35 * 0.18) for curve in curves[1::2])
    view._linked_selection = None
    view.tracking_inspector.selected_id = None
    TrackingViewsMixin._highlight_track_selection(view)
    assert all(curve.opacity == 0.35 for curve in curves)


def test_synthetic_world_z_does_not_enable_3d_projection(qtbot):
    from PySide6.QtWidgets import QComboBox

    from visualizer.src.sensing.tracking_views import TrackingViewsMixin

    row = SimpleNamespace(
        time_s=1.0,
        detections=(
            {
                "track_id": 7,
                "track_x_m": 1,
                "track_y_m": 2,
                "z_m": 3,
                "world_track_position": [1, 2, 10],
            },
        ),
        truth_positions=(),
        truth=None,
        tracking_states=(
            {
                "track_id": 7,
                "world_position_m": [1, 2, 10],
                "position_frame": "receiver local",
                "position_m": [1, 2],
                "state_labels": ("x_m", "y_m", "vx_m_s", "vy_m_s"),
            },
        ),
    )
    data = prepare_tracking_plot_data(SimpleNamespace(records=[row]))
    assert data.spatial_dimensions == 2
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["XY", "XZ", "YZ"])
    combo.setCurrentIndex(2)
    view = SimpleNamespace(tracking_projection=combo)
    TrackingViewsMixin._set_tracking_projection_capability(view, data)
    assert combo.currentText() == "XY"
    assert not combo.model().item(1).isEnabled()
    row.tracking_states[0]["state_labels"] = ("x_m", "y_m", "z_m")
    data = prepare_tracking_plot_data(SimpleNamespace(records=[row]))
    assert data.spatial_dimensions == 3
    TrackingViewsMixin._set_tracking_projection_capability(view, data)
    assert combo.model().item(1).isEnabled() and combo.model().item(2).isEnabled()


def test_disabled_input_kind_does_not_open_inspection(qtbot):
    from PySide6.QtWidgets import QComboBox

    from visualizer.src.sensing.workspace_shell import WorkspaceShellMixin

    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Frequency response", "frequency")
    combo.model().item(0).setEnabled(False)
    opened = []
    shell = SimpleNamespace(
        input_kind_box=combo, _open_result_detail=lambda *args: opened.append(args)
    )
    WorkspaceShellMixin._choose_input_kind(shell)
    assert opened == []
    combo.model().item(0).setEnabled(True)
    WorkspaceShellMixin._choose_input_kind(shell)
    assert opened == [("diagnostics", "frequency")]


def test_motion_browser_keeps_known_empty_tracking_population():
    from visualizer.src.sensing.workspace_shell import WorkspaceShellMixin

    accepted = []
    shell = SimpleNamespace(
        _last_request=SimpleNamespace(source_identity=("recording",), saved_tracking=None),
        result_browser=SimpleNamespace(
            update_result=lambda *args, **kwargs: accepted.append(kwargs)
        ),
    )
    record = SimpleNamespace(
        time_s=1.0,
        identity=("clock", "run", "pair", "cpi"),
        sensing_metadata={},
        detections=(),
        tracking_states=(),
        tracking_states_complete=True,
    )
    WorkspaceShellMixin._update_browser_cursor(shell, record)
    assert accepted[-1]["complete_track_states"] is True
    record.tracking_states_complete = False
    WorkspaceShellMixin._update_browser_cursor(shell, record)
    assert accepted[-1]["complete_track_states"] is False
