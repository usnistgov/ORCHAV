"""Cross-boundary regressions for the optional recorded-data workspace."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from tests.visualizer.unit.mpc_explorer_test_data import make_large_los_catalog
from tests.visualizer.unit.test_experimental_range_geometry import _request
from visualizer.src.experiments.path_analysis import (
    SPEED_OF_LIGHT_M_PER_NS,
    delay_angle_histogram,
    select_paths,
)
from visualizer.src.experiments.service import ExperimentService, _analyze
from visualizer.src.experiments.window import ExperimentWindow
from visualizer.src.renderers.pygfx import experiments as adapter


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _catalog(count):
    catalog = make_large_los_catalog(count)
    canonical = catalog.canonical_data
    canonical.path_tx.fill(0)
    canonical.tx_id.fill(0)
    canonical.path_delays.fill(1)
    canonical.path_aoa_az = np.zeros(count, dtype=np.float32)
    return catalog


def _settings(**changes):
    settings = dict(
        pair=(0, 0),
        scope="all",
        weight_mode="count",
        angle="aoa_azimuth",
        angular_bin_width_deg=10.0,
        delay_bin_width_ns=10.0,
    )
    settings.update(changes)
    return settings


def _overlay():
    """Exercise the adapter's actual bounded path selection without a GPU."""
    from tests.visualizer.unit.test_cinematic_pulses import renderer

    overlay = adapter.ExperimentOverlay(renderer())
    overlay._replace = lambda _name, _value: None
    overlay.pulse_positions = None
    overlay.direction_ids = None
    overlay.pulses = lambda positions: setattr(overlay, "pulse_positions", positions.copy())
    overlay.directions = lambda _catalog, ids, **_kw: setattr(
        overlay, "direction_ids", np.asarray(ids).copy()
    )
    return overlay


@pytest.fixture
def service(app, qtbot):
    service = ExperimentService(SimpleNamespace())
    service.window = ExperimentWindow()
    qtbot.addWidget(service.window)
    service.window.show()
    from tests.visualizer.unit.test_cinematic_pulses import renderer

    service.overlay = adapter.ExperimentOverlay(renderer())
    service.overlay.pulse_positions = None
    service.overlay.direction_ids = None
    service.overlay.directions = lambda _catalog, ids, **_kw: setattr(
        service.overlay, "direction_ids", np.asarray(ids).copy()
    )
    original_pulses = service.overlay.cinematic_pulses

    def capture_pulses(sample, **kwargs):
        service.overlay.pulse_positions = sample.leading.copy()
        original_pulses(sample, **kwargs)

    service.overlay.cinematic_pulses = capture_pulses
    yield service
    service.close()
    explorer = getattr(service.viz, "_mpc_explorer_session", None)
    if explorer is not None:
        explorer.shutdown()


def test_nonuniform_terminal_angle_bin_is_painted_and_clicked_at_exact_edges(app, qtbot):
    catalog = _catalog(6)
    catalog.canonical_data.path_aoa_az[:] = [-150, -80, -10, 60, 130, 175]
    data = select_paths(catalog, tx_id=0, rx_id=0)
    histogram = delay_angle_histogram(data, angular_bin_width_deg=70)
    window = ExperimentWindow()
    qtbot.addWidget(window)
    try:
        window.set_analysis(data, histogram, reset_gate=True, clock_end=1)
        window.tabs.setCurrentIndex(1)
        window.show()
        app.processEvents()
        expected_x, expected_y = np.meshgrid(histogram.angle_edges_deg, histogram.delay_edges_ns)
        np.testing.assert_array_equal(window.image.x, expected_x)
        np.testing.assert_array_equal(window.image.y, expected_y)
        np.testing.assert_array_equal(window.image.z, histogram.values)
        assert np.diff(window.image.x[0]).tolist() == [70, 70, 70, 70, 70, 10]

        clicked = []
        window.bin_selected.connect(lambda delay, angle: clicked.append((delay, angle)))
        for angle in (150, 175):
            point = window.angle_plot.plotItem.vb.mapViewToScene(QPointF(angle, 1))
            window._image_click(SimpleNamespace(scenePos=lambda point=point: point))
        assert clicked == [(0, 4), (0, 5)]
        np.testing.assert_array_equal(histogram.path_ids_for_bin(*clicked[0]), [4])
        np.testing.assert_array_equal(histogram.path_ids_for_bin(*clicked[1]), [5])
        window.clear_analysis()
        assert window.image.z is None
    finally:
        window.close()


def test_gate_beyond_initial_display_cap_gets_its_own_geometric_clock(service):
    catalog = _catalog(1501)
    canonical = catalog.canonical_data
    canonical.path_delays[-1] = 100
    canonical.points[-1, 0] = 100
    before_points = canonical.points.copy()
    before_delays = canonical.path_delays.copy()
    service.catalog = catalog
    service.data, service.histogram, initial_end = _analyze(catalog, _settings())
    assert initial_end == pytest.approx(1 / SPEED_OF_LIGHT_M_PER_NS)
    service._clock_end = initial_end
    service.window.set_clock_extent(initial_end)
    service.window.set_clock(initial_end)
    service.window.play.setChecked(True)
    service.play(True)

    service.gate(99, 101)

    np.testing.assert_array_equal(service.path_ids, [1500])
    assert service._clock_end == pytest.approx(100 / SPEED_OF_LIGHT_M_PER_NS)
    assert service.window._clock_end == service._clock_end
    assert service._clock_time == 0
    assert service.window.clock.value() == 0
    assert not service._clock.isActive() and not service.window.play.isChecked()
    service.clock(service._clock_end)
    np.testing.assert_allclose(service.overlay.pulse_positions, [[100, 0, 0]])
    assert service.window.clock.value() == 1000
    np.testing.assert_array_equal(canonical.points, before_points)
    np.testing.assert_array_equal(canonical.path_delays, before_delays)


def test_bin_guides_and_pulses_use_exact_drawn_support_without_changing_metrics(
    service, monkeypatch
):
    monkeypatch.setattr(adapter, "MAX_SEGMENTS", 1)
    catalog = _catalog(3)
    service.catalog = catalog
    service.data, service.histogram, service._clock_end = _analyze(catalog, _settings())
    angle_bin = int(service.histogram.angle_bin_indices[0])

    service.select_bin(0, angle_bin)

    np.testing.assert_array_equal(service.path_ids, [0, 1, 2])
    np.testing.assert_array_equal(service.overlay.direction_ids, [0, 1, 2])
    assert service.data.path_count == 3 and service.histogram.path_count == 3
    assert "3 contributing paths; 3 drawn" in service.window.matching_label.toolTip()
    service.clock(service._clock_end)
    assert service.overlay.pulse_positions.shape == (3, 3)
    np.testing.assert_allclose(service.overlay.pulse_positions, [[1, 0, 0]] * 3)


def test_adapter_returns_only_finite_paths_inside_exact_budget(monkeypatch):
    monkeypatch.setattr(adapter, "MAX_SEGMENTS", 2)
    catalog = _catalog(4)
    catalog.canonical_data.points[3, 0] = np.nan
    overlay = _overlay()
    np.testing.assert_array_equal(overlay.paths(catalog, np.arange(4)), [0, 2])


def test_clear_overlays_resets_highlight_control_and_clock(service):
    service.path_ids = np.asarray([123], dtype=np.int32)
    service.window.set_matching_paths("Matching paths: 1")
    service._clock_time = 0.6
    service.window.set_clock(0.6)

    service.clear_overlays()

    assert service.path_ids.size == 0
    assert service.window.matching_label.text() == "Matching paths: none"
    assert service._clock_time == 0 and service.window.clock.value() == 0
    assert "clock 0.000 ns" in service.window.clock.toolTip()


@pytest.mark.parametrize("mode,factor", [("monostatic", 2.0), ("bistatic", 1.0)])
@pytest.mark.parametrize("stale", [False, True])
def test_sensing_cell_opens_exact_pair_delay_gate_and_rejects_stale_completion(
    service, mode, factor, stale
):
    request, selection = _request(mode=mode, tx=(0, 0, 0), rx=(0, 0, 0))
    request.display_frame["sensing"]["pair_axis"][0].update(tx_device_index=0, rx_device_index=0)
    dashboard = QWidget()
    dashboard._last_request, dashboard._linked_selection = request, selection
    dashboard.cell_explanation = SimpleNamespace(paths_note=QLabel())
    catalog = _catalog(4)
    canonical = catalog.canonical_data
    # Range support is [16, 27) m. One matching delay belongs to another TX.
    canonical.path_delays[:] = np.array([10, 20, 30, 20]) * factor / SPEED_OF_LIGHT_M_PER_NS
    canonical.path_tx[-1] = 1
    canonical.tx_id[canonical.path_id == 3] = 1
    saved_delays = canonical.path_delays.copy()
    saved_points = canonical.points.copy()
    service.viz.schedule_update = lambda: None
    packet = SimpleNamespace(canonical_data=canonical, segment_mask=None)
    view_model = SimpleNamespace(
        path_mask=None, mpc_visibility=SimpleNamespace(effective_paths=True)
    )
    service.present(request.effective_display_step, view_model, packet)
    service.window.gate_changed.connect(service.gate)
    service.open_cell_paths(dashboard)
    service._debounce.stop()
    assert service.window.settings()["pair"] == (0, 0)
    assert service.window.settings()["scope"] == "all"
    assert (
        "Doppler, beamforming and interference are not tested"
        in service.window.selection_context.text()
    )
    if stale:
        dashboard._linked_selection = replace(selection, acquisition_id="retired")
    service._completed(service._serial, _analyze(service.catalog, service.window.settings()), None)
    if stale:
        assert service.window.selection_context.isHidden()
        assert service.path_ids.size == 0
        assert "1 delay-matched" not in dashboard.cell_explanation.paths_note.text()
    else:
        np.testing.assert_allclose(
            service.window.gate.getRegion(), np.array([16, 27]) * factor / SPEED_OF_LIGHT_M_PER_NS
        )
        np.testing.assert_array_equal(service.path_ids, [1])
        np.testing.assert_array_equal(service._explorer._matching_path_ids, [1])
        assert "Sensing delay-matched candidates" in service.window.matching_label.text()
        assert "1 delay-matched paths" in dashboard.cell_explanation.paths_note.text()
    np.testing.assert_array_equal(canonical.path_delays, saved_delays)
    np.testing.assert_array_equal(canonical.points, saved_points)
    dashboard.close()


@pytest.mark.parametrize("phase", ["pending", "completed", "manual"])
def test_deselect_retires_only_paths_owned_by_sensing_cell(service, monkeypatch, phase):
    request, selection = _request()
    request.display_frame["sensing"]["pair_axis"][0].update(tx_device_index=0, rx_device_index=0)
    dashboard = QWidget()
    dashboard._last_request, dashboard._linked_selection = request, selection
    dashboard.cell_explanation = SimpleNamespace(paths_note=QLabel())
    monkeypatch.setattr(service.range_shell, "sync", lambda _dashboard=None: None)
    service.viz.schedule_update = lambda: None
    catalog = _catalog(2)
    catalog.canonical_data.path_delays[:] = [20, 30] / np.array(SPEED_OF_LIGHT_M_PER_NS)
    service.present(
        request.effective_display_step,
        SimpleNamespace(path_mask=None, mpc_visibility=SimpleNamespace(effective_paths=True)),
        SimpleNamespace(canonical_data=catalog.canonical_data, segment_mask=None),
    )
    service.window.gate_changed.connect(service.gate)
    service.open_cell_paths(dashboard)
    service._debounce.stop()
    serial = service._serial
    result = _analyze(service.catalog, service.window.settings())
    if phase != "pending":
        service._completed(serial, result, None)
        assert service.path_ids.size == 1
        assert not service.window.selection_context.isHidden()
        if phase == "manual":
            service.gate(0, 1000)
            assert service.window.selection_context.isHidden()
    dashboard._linked_selection = None
    service.sync_sensing_selection(dashboard)
    if phase == "manual":
        assert service.path_ids.size == 2
        assert service.data is result[0]
    else:
        assert service.path_ids.size == 0
        assert service.data is None and service.histogram is None
        assert service.window.matching_label.text() == "Matching paths: none"
        assert service.window.image.z is None
        assert not service.window.gate.isVisible()
        assert service.window.selection_context.isHidden()
        assert not service._debounce.isActive()
        service._completed(serial, result, None)
        assert service.data is None and service.path_ids.size == 0
        # A fresh explicit inspection can still use the accepted catalog.
        dashboard._linked_selection = selection
        service.open_cell_paths(dashboard)
        service._debounce.stop()
        service._completed(service._serial, result, None)
        assert service.path_ids.size == 1
        assert service.window.gate.isVisible()
    dashboard.close()


def test_latest_analysis_cancels_queued_work_and_ignores_running_stale_result(service, monkeypatch):
    catalog = _catalog(2)
    service.catalog = catalog
    service.window.set_pairs([(0, 0)])
    futures = []
    deliveries = []

    def submit(*_args, **_kwargs):
        future = Future()
        futures.append(future)
        return future

    monkeypatch.setattr(service._executor, "submit", submit)
    monkeypatch.setattr(
        service.window, "set_analysis", lambda *args, **_kw: deliveries.append(args)
    )
    service._submit()
    assert futures[0].set_running_or_notify_cancel()
    service._options_changed()
    service._submit()
    service._options_changed()
    service._submit()
    assert futures[1].cancelled()

    stale_result = _analyze(catalog, _settings())
    futures[0].set_result(stale_result)
    assert deliveries == [] and service.data is None
    latest_result = _analyze(catalog, _settings())
    futures[2].set_result(latest_result)
    assert len(deliveries) == 1
    assert deliveries[0][0] is latest_result[0]
    assert service.data is latest_result[0]


def test_new_accepted_frame_clears_old_analysis_and_rejects_its_completion(service):
    old_catalog = _catalog(2)
    old_result = _analyze(old_catalog, _settings())
    service.catalog = old_catalog
    service.data, service.histogram, service._clock_end = old_result
    service.path_ids = np.asarray([0], dtype=np.int32)
    service.window.set_matching_paths("Matching paths: 1")
    service._cell_gate = ("previous selection",)
    old_serial = service._serial
    new_catalog = _catalog(3)
    packet = SimpleNamespace(canonical_data=new_catalog.canonical_data, segment_mask=None)
    view_model = SimpleNamespace(
        path_mask=None, mpc_visibility=SimpleNamespace(effective_paths=True)
    )

    service.present(9, view_model, packet)

    assert service.step == 9 and service.packet is packet
    assert service.catalog.canonical_data is packet.canonical_data
    assert service.data is None and service.histogram is None
    assert service.path_ids.size == 0 and service._cell_gate is None
    assert service.window.matching_label.text() == "Matching paths: none"
    assert service.window.image.z is None
    service._completed(old_serial, old_result, None)
    assert service.data is None and service.histogram is None


def test_rejected_catalog_cannot_resubmit_previous_frame_after_debounce(service, monkeypatch):
    service.catalog = _catalog(2)
    service.window.set_pairs([(0, 0)])
    service._debounce.start()
    submissions = []
    monkeypatch.setattr(service._executor, "submit", lambda *args: submissions.append(args))
    packet = SimpleNamespace(canonical_data=SimpleNamespace(points=None), segment_mask=None)
    view_model = SimpleNamespace(
        path_mask=None, mpc_visibility=SimpleNamespace(effective_paths=True)
    )

    # The extension dispatch boundary catches expected data-validation errors;
    # a later queued timer must still be unable to use a previous catalog.
    try:
        service.present(9, view_model, packet)
    except ValueError:
        pass

    assert service.catalog is None
    assert not service._debounce.isActive()
    service._submit()
    assert submissions == []
