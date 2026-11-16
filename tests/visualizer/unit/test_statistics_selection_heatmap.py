"""Numerical, cache and GUI contracts for recorded link/interval exploration."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from shared.frames import FrameProjection
from tests.visualizer.fixtures.semantic_mpc import BASELINE_SEMANTIC_FRAME
from tests.visualizer.unit.test_scenario_statistics_accumulator import (
    _metric_projection,
    _semantic_projection,
)
from tests.visualizer.unit.test_scenario_statistics_service import _ProjectionProvider, _services
from visualizer.src.app.theme import get_theme_manager
from visualizer.src.metrics.scenario_statistics import (
    SCENARIO_STATISTICS_REQUEST,
    ScenarioStatisticsAccumulator,
)
from visualizer.src.metrics.statistics_analysis import (
    build_delay_time_heatmap,
    select_statistics_pair,
)
from visualizer.src.panels.statistics_panel import StatisticsPanel
from visualizer.src.panels.statistics_scope import StatisticsScopeBar


def _recording():
    projections = []
    for frame, timestamp, delay in ((3, 0.3, 1.2), (5, 0.5, 2.2), (8, 0.8, 3.2)):
        source = _metric_projection(
            delays_ns=[delay, delay + 0.1], path_loss_db=[20, 20], frame_index=frame
        )
        projections.append(
            FrameProjection.from_request(
                replace(source.frame, timestamp_s=timestamp), SCENARIO_STATISTICS_REQUEST
            )
        )
    return ScenarioStatisticsAccumulator().collect_from_projections(projections)


def test_heatmap_sums_linear_power_and_retains_gaps_and_timestamps():
    stats = _recording()
    power = stats["delay_time_power_db"]
    np.testing.assert_allclose(power[np.isfinite(power)], -20 + 10 * np.log10(2))
    assert stats["delay_time_axis"] == "Time (s)"
    np.testing.assert_allclose(stats["frame_times_s"], [0.3, 0.5, 0.8])
    assert np.any(np.isnan(power).all(axis=1))
    assert stats["delay_time_observation_counts"].sum() == 3
    assert stats["frame_indices"] == [3, 5, 8]


def test_heatmap_bounded_time_bins_average_power_not_db_and_preserve_weak_bins():
    count = 1024
    stats = build_delay_time_heatmap(
        list(range(count)),
        [],
        list(range(count)),
        [np.array([0.0, 2.0])] * count,
        [np.array([0.0, 200.0]) if i % 2 == 0 else np.array([10.0, 210.0]) for i in range(count)],
    )
    assert stats["delay_time_power_db"].shape == (512, 3)
    np.testing.assert_array_equal(stats["delay_time_observation_counts"], 2)
    np.testing.assert_allclose(stats["delay_time_power_db"][:, 0], 10 * np.log10(0.55))
    np.testing.assert_allclose(stats["delay_time_power_db"][:, 2], 10 * np.log10(0.55) - 200)
    assert np.isnan(stats["delay_time_power_db"][:, 1]).all()
    assert stats["delay_time_axis"] == "Frame"


def test_pair_selection_preserves_offsets_ids_and_asymmetric_validity():
    source = _semantic_projection(BASELINE_SEMANTIC_FRAME, 4)
    for row, pair in enumerate(source.frame.tx_rx_pairs):
        selected = select_statistics_pair(source.frame, tuple(map(int, pair)))
        lo, hi = source.frame.pair_path_offsets[row : row + 2]
        np.testing.assert_array_equal(selected.delays_ns, source.frame.delays_ns[lo:hi])
        np.testing.assert_array_equal(
            selected.metric_valid_bits, source.frame.metric_valid_bits[lo:hi]
        )
        stats = ScenarioStatisticsAccumulator(pair=tuple(map(int, pair))).collect_from_projections(
            [source]
        )
        assert stats["total_mpcs"] == hi - lo
        assert stats["tx_rx_pairs"] == [tuple(pair)]
        assert sum(stats["pair_visibility_counts"].values()) == 1
    absent = ScenarioStatisticsAccumulator(pair=(999, 999)).collect_from_projections([source])
    assert absent["total_mpcs"] == 0
    assert absent["tx_rx_pairs"] == []
    assert sum(absent["pair_visibility_counts"].values()) == 0


def test_subset_never_reads_or_overwrites_whole_run_cache(tmp_path):
    cache, service, scenario = _services(tmp_path)
    provider = _ProjectionProvider(generation_id="a", frame_set_id="a", frame_indices=(0, 2, 4))
    whole = service.collect(provider, scenario=scenario)
    subset = service.collect(provider, scenario=scenario, steps=(2,), pair=(0, 0))
    assert subset.stats["frame_indices"] == [2]
    assert subset.stats["selection_scoped"]
    assert not subset.from_cache
    restored = service.collect(provider, scenario=scenario)
    assert restored.from_cache
    np.testing.assert_array_equal(restored.stats["frame_indices"], whole.stats["frame_indices"])
    np.testing.assert_allclose(
        restored.stats["delay_time_power_db"], whole.stats["delay_time_power_db"], equal_nan=True
    )
    assert service.collect(provider, scenario=scenario, steps=()).stats["total_frames"] == 0


def test_scope_seconds_inclusive_selection_and_frame_fallback(qtbot):
    bar = StatisticsScopeBar()
    qtbot.addWidget(bar)
    stats = _recording()
    bar.bind(stats, enabled=True)
    bar.pair.setCurrentIndex(1)
    bar.interval.setChecked(True)
    bar.start.setValue(0.5)
    bar.end.setValue(0.8)
    with qtbot.waitSignal(bar.selection_requested) as signal:
        bar.apply.click()
    assert signal.args == [(0, 0), (5, 8)]
    bar.bind(dict(stats, frame_times_s=[]), enabled=True)
    assert bar.interval.text() == "Frames"
    assert bar.start.decimals() == 0
    assert bar.start.value() == 3
    assert bar.end.value() == 8


def test_heatmap_lazy_theme_export_and_nearest_recorded_frame(qtbot):
    selected = []
    panel = StatisticsPanel(SimpleNamespace(update_frame=selected.append))
    qtbot.addWidget(panel.create_panel())
    root = panel.create_graphs_panel()
    qtbot.addWidget(root)
    root.resize(1000, 900)
    panel._graphs_visible = True
    panel._apply_stats_result(_recording(), status_text="Ready")
    root.show()
    QApplication.processEvents()
    card = panel.widgets["delay_time_chart"]
    assert card.plot is None
    panel._graph_tabs.setCurrentIndex(6)
    QApplication.processEvents()
    assert card.plot is not None
    count = card.render_count
    view = card.plot.getViewBox()
    from PySide6.QtCore import QPointF

    position = view.mapViewToScene(QPointF(0.51, 2.2))
    card._click(
        SimpleNamespace(
            double=lambda: False, button=lambda: Qt.LeftButton, scenePos=lambda: position
        )
    )
    assert selected == [5]
    manager = get_theme_manager()
    old_mode = manager.mode
    try:
        for mode in ("dark", "light"):
            manager.set_mode(mode, persist=False)
            QApplication.processEvents()
            assert card.render_count == count
            assert card._colorbar.axis.labelText == "Path gain (dB)"
    finally:
        manager.set_mode(old_mode, persist=False)
    figure = panel._create_publication_figure(**panel._chart_specs["delay_time_chart"])
    figure.canvas.draw()
    assert figure.axes[0].get_xlabel() == "Time (s)"
    for _ in range(5):
        panel._apply_stats_result(_recording(), status_text="Selection updated")
        QApplication.processEvents()
        colorbars = [
            item for item in card.plot.scene().items() if type(item).__name__ == "ColorBarItem"
        ]
        images = [
            item for item in card.plot.getPlotItem().items if type(item).__name__ == "ImageItem"
        ]
        assert len(colorbars) == len(images) == 1
    panel.cleanup()


def test_reset_reuses_full_result_and_does_not_publish_subset_to_trajectory(qtbot):
    forwarded = []
    trajectory = SimpleNamespace(set_per_frame_stats=forwarded.append)
    panel = StatisticsPanel(
        SimpleNamespace(ui_manager=SimpleNamespace(panels={"trajectory": trajectory}))
    )
    qtbot.addWidget(panel.create_panel())
    root = panel.create_graphs_panel()
    qtbot.addWidget(root)
    full = _recording()
    panel._apply_stats_result(full, status_text="Full")
    subset = dict(full, selection_scoped=True, selection_pair=(0, 0))
    panel._apply_stats_result(subset, status_text="Subset")
    assert len(forwarded) == 1
    panel._reset_statistics_scope()
    assert panel.stats is full
    assert panel._selection_pair is None
    panel.cleanup()


def test_apply_runs_background_subset_and_reset_requires_no_reads(qtbot, tmp_path):
    _, service, scenario = _services(tmp_path)
    provider = _ProjectionProvider(generation_id="ui", frame_set_id="ui", frame_indices=(0, 2, 4))
    service.collect(provider, scenario=scenario)
    panel = StatisticsPanel(SimpleNamespace(scenario=scenario, scenario_statistics_service=service))
    qtbot.addWidget(panel.create_panel())
    root = panel.create_graphs_panel()
    qtbot.addWidget(root)
    assert panel.set_statistics_source(provider)
    qtbot.waitUntil(lambda: panel.stats is not None)
    bar = panel._scope_bar
    assert bar.isEnabled()
    bar.pair.setCurrentIndex(1)
    bar.interval.setChecked(True)
    bar.start.setValue(2)
    bar.end.setValue(2)
    bar.apply.click()
    qtbot.waitUntil(lambda: bool(panel.stats.get("selection_scoped")))
    assert panel.stats["frame_indices"] == [2]
    assert panel.stats["total_mpcs"] == 1
    reads = len(provider.projection_requests)
    bar.reset.click()
    np.testing.assert_array_equal(panel.stats["frame_indices"], [0, 2, 4])
    assert len(provider.projection_requests) == reads
    panel.cleanup()


@pytest.mark.parametrize("first", ["panel", "graphs"])
def test_independent_statistics_roots_release_callbacks_and_preserve_live_sibling(qtbot, first):
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import isValid

    cancellations = []
    service = SimpleNamespace(cancel_collection=lambda: cancellations.append("cancelled"))
    closed_metrics = []
    metrics = SimpleNamespace(close=lambda: closed_metrics.append("closed"))
    panel = StatisticsPanel(
        SimpleNamespace(scenario_statistics_service=service, metrics_service=metrics)
    )
    roots = {"panel": panel.create_panel(), "graphs": panel.create_graphs_panel()}
    for root in roots.values():
        qtbot.addWidget(root)
    panel._statistics_provider = object()
    panel._metrics_host = object()  # A host allocated lazily after root creation.
    panel._apply_stats_result(_recording(), status_text="Ready")
    panel._set_graphs_status("Graphs updated.", transient=True)
    timer = panel._graphs_status_clear_timer
    manager = get_theme_manager()
    previous = manager.mode
    try:
        roots[first].deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        sibling = "graphs" if first == "panel" else "panel"
        assert isValid(roots[sibling])
        assert cancellations == []
        assert closed_metrics == (["closed"] if first == "panel" else [])
        if first == "graphs":
            assert not isValid(timer)
        manager.set_mode("dark" if previous != "dark" else "light", persist=False)
        updated = _recording()
        panel._apply_stats_result(updated, status_text="Updated")
        assert panel.stats is updated
        if sibling == "panel":
            assert panel.widgets["total_frames"].text() == "3"
        roots[sibling].deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert cancellations == ["cancelled"]
        assert closed_metrics == ["closed"]
        panel.cleanup()
        assert cancellations == ["cancelled"]
        manager.set_mode(previous, persist=False)
    finally:
        panel.cleanup()
        manager.set_mode(previous, persist=False)
