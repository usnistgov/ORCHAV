"""Current-result metadata and shared selection remain independent of dense products."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt

from visualizer.src.sensing.current_result_summary import (
    CurrentResultSummary,
    result_cards,
    result_context_label,
)
from visualizer.src.sensing.result_browser import SensingResultBrowser


def _sensing():
    return {
        "detections": [],
        "detection_filter_metadata": {
            "final_kept": 0,
            "tracker": {
                "enabled": True,
                "raw": 2,
                "kept": 0,
                "active_tracks": 2,
                "confirmed_tracks": 0,
            },
            "range_gate": {"enabled": False, "rejected": 0},
            "ghost_filter": {"enabled": True, "rejected": 1},
        },
        "products": {
            "rd": {
                "kind": "range_doppler",
                "variant": "primary",
                "processing": {"slow_time_mean_removal_applied": True},
            }
        },
    }


def test_confirmation_summary_does_not_mistake_withheld_reports_for_failed_detection():
    cards = result_cards(_sensing())
    assert cards["Detection"].lines == ("0 published", "2 entering tracking")
    assert cards["Tracking"].lines == ("2 active · 0 confirmed", "2 tentative")
    assert cards["Filtering"].lines == ("Mean removal · on", "Multipath filter · 1 rejected")


def test_unavailable_metadata_and_pretracking_rows_are_not_reported_as_zero_outputs():
    missing = result_cards({})
    assert all(card.lines == ("Not recorded",) for card in missing.values())
    pretracking = result_cards(
        {"detections": [{}, {}], "_display_detection_collection": "detections_before_tracking"}
    )
    assert pretracking["Detection"].lines == ("Not recorded",)
    disabled = result_cards({"config_resolved": {"track_confirmation_enabled": False}})
    assert disabled["Tracking"].lines == ("Disabled",)


def test_summary_uses_metadata_and_optional_aligned_scores_only(qtbot):
    panel = CurrentResultSummary()
    qtbot.addWidget(panel)
    sensing = _sensing()
    sensing["range_doppler_by_pair"] = object()  # No array interpretation is allowed.
    panel.update_result(sensing, result_label="CPI 2 · 0.1 s · Held result")
    assert panel.context.text() == ""  # The caller already presents the result header.
    assert panel.cards["Evaluation"].isHidden()
    assert all(not hasattr(card, "details") for card in panel.cards.values())
    scores = {"reason": None, "default_summary": {"total_tp": 0, "total_fp": 0, "total_fn": 2}}
    panel.update_result(sensing, evaluation=scores)
    assert panel.cards["Evaluation"].value.text() == "0 matched · 0 unmatched\n2 missed targets"
    scores["reason"] = "Ground truth is not aligned with this result"
    panel.update_result(sensing, evaluation=scores)
    assert panel.cards["Evaluation"].value.text() == "Unavailable"
    scores["scope_label"] = "Tolerance ±0.6 m / ±0.25 m/s"
    panel.update_result(sensing, evaluation=scores)
    assert panel.cards["Evaluation"].value.text().endswith("Tolerance ±0.6 m / ±0.25 m/s")


def test_world_summary_distinguishes_known_empty_epoch_from_missing_epoch():
    snapshot = SimpleNamespace(
        source_key=("saved_world_track", "bundle", "sensor"), states=(), complete_states=True
    )
    assert result_cards({}, tracking_snapshot=snapshot)["Tracking"].lines == (
        "0 active · 0 confirmed",
        "Saved world tracks",
    )
    snapshot.complete_states = False
    assert result_cards({}, tracking_snapshot=snapshot)["Tracking"].lines == (
        "Unavailable",
        "Saved world tracks",
    )


def test_invalid_evaluation_cohort_and_missing_truth_never_display_zero_scores():
    zero_counts = {"total_tp": 0, "total_fp": 0, "total_fn": 0}
    missing_truth = {"truth": None, "default_summary": zero_counts}
    invalid_cohort = {
        "default_summary": {**zero_counts, "reason": "Track confirmation state unavailable"}
    }
    for evaluation in (missing_truth, invalid_cohort):
        assert result_cards({}, evaluation)["Evaluation"].lines == ("Unavailable",)
    assert result_cards({}, zero_counts)["Evaluation"].lines == (
        "0 matched · 0 unmatched",
        "0 missed targets",
    )


def test_result_header_uses_saved_acquisition_span_and_preserves_held_time():
    sensing = {
        "temporal": {
            "acquisition": {"start_time_s": 0.064, "end_time_s": 0.127},
            "result": {"reference_time_s": 0.064},
        },
        "config_resolved": {"processing": {"strategy": "coherent_cpi_fft"}},
    }
    assert result_context_label(sensing, "Held result · 0.064 s") == (
        "Held result · 0.064 s · CPI 0.064–0.127 s"
    )
    assert result_context_label(sensing, "Held result · 0.064 s", compact=True) == (
        "CPI 0.064–0.127 s"
    )
    sensing["config_resolved"]["processing"]["strategy"] = "sequential_stft"
    assert result_context_label(sensing).endswith("Acquisition 0.064–0.127 s")
    sensing["temporal"]["acquisition"]["end_time_s"] = 0.064
    assert result_context_label(sensing) == "Result · 0.064 s"
    sensing["temporal"]["acquisition"]["start_time_s"] = 1.0
    assert result_context_label(sensing) == "Result · 0.064 s"


def _browser(qtbot):
    browser = SensingResultBrowser()
    qtbot.addWidget(browser)
    browser.resize(850, 250)
    browser.show()
    return browser


def _click(qtbot, table, proxy, row):
    index = proxy.index(row, 0)
    qtbot.mouseClick(
        table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualRect(index).center()
    )


def test_sorted_detection_rows_select_original_observation_and_can_clear(qtbot):
    browser = _browser(qtbot)
    browser.update_result(
        [{"range_m": 9, "track_id": 7}, {"range_m": 2, "track_id": 8}],
        context_key=("source", "clock", "run", "pair"),
        acquisition_id="a",
    )
    browser.table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
    assert browser.model.data(browser.model.index(0, 0)) == "Detection 1"
    assert browser.model.headerData(4, Qt.Orientation.Horizontal) == "Track"
    with qtbot.waitSignal(browser.detection_selected) as selected:
        _click(qtbot, browser.table, browser.proxy, 0)
    assert selected.args == [1]
    with qtbot.waitSignal(browser.selection_cleared):
        _click(qtbot, browser.table, browser.proxy, 0)
    assert not browser.table.selectionModel().hasSelection()
    assert browser._selection_key is None


def test_keyboard_selection_links_sorted_rows_once_and_programmatic_sync_stays_silent(qtbot):
    browser = _browser(qtbot)
    browser.update_result(
        [{"range_m": 9}, {"range_m": 2}],
        [{"track_id": 7}, {"track_id": 8}],
        context_key=("source", "clock", "run", "pair"),
        acquisition_id="a",
    )
    detections, tracks = [], []
    browser.detection_selected.connect(detections.append)
    browser.track_selected.connect(tracks.append)
    browser.table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
    browser.select_detection(1)
    assert detections == []
    browser.table.setFocus()
    qtbot.keyClick(browser.table, Qt.Key.Key_Down)
    assert detections == [0]
    browser.select_detection(1)
    assert detections == [0]
    browser.tabs.setCurrentIndex(1)
    browser.select_track(7)
    assert tracks == []
    browser.track_table.setFocus()
    qtbot.keyClick(browser.track_table, Qt.Key.Key_Down)
    assert tracks == [8]
    browser.select_track(7)
    assert tracks == [8]


def test_observation_selection_is_scoped_to_source_acquisition_and_collection(qtbot):
    browser = _browser(qtbot)
    rows = [{"detection_id": "d", "range_m": 2, "track_id": 7}]
    context = ("source", "clock", "run", "pair")
    browser.update_result(rows, context_key=context, acquisition_id="a")
    browser.select_detection(0)
    browser.update_result(rows, context_key=context, acquisition_id="a")
    assert browser.table.selectionModel().hasSelection()
    browser.update_result(rows, context_key=context, acquisition_id="b")
    assert not browser.table.selectionModel().hasSelection()
    browser.select_detection(0)
    browser.update_result(rows, context_key=("new source", *context[1:]), acquisition_id="b")
    assert not browser.table.selectionModel().hasSelection()
    browser.select_detection(0)
    browser.update_result(
        rows,
        context_key=("new source", *context[1:]),
        acquisition_id="b",
        detection_collection="detections_before_tracking",
    )
    assert not browser.table.selectionModel().hasSelection()


def test_real_coasting_state_has_no_detection_and_retains_selection_across_acquisitions(qtbot):
    browser = _browser(qtbot)
    state = {
        "track_id": 3,
        "source_kind": "native",
        "source_key": ("run",),
        "confirmed": True,
        "predicted_only": True,
        "estimated_range_m": 4.2,
        "radial_velocity_m_s": 1.5,
    }
    context = ("source", "clock", "run", "pair")
    browser.update_result([], [state], context_key=context, acquisition_id="a")
    browser.tabs.setCurrentIndex(1)
    assert browser.model.rowCount() == 0
    assert browser.track_model.data(browser.track_model.index(0, 1)) == "Confirmed · coasting"
    with qtbot.waitSignal(browser.track_selected) as selected:
        _click(qtbot, browser.track_table, browser.track_proxy, 0)
    assert selected.args == [3]
    browser.update_result([], [state], context_key=context, acquisition_id="b")
    assert browser.track_table.selectionModel().hasSelection()
    browser.update_result(
        [], [{**state, "source_kind": "saved_world_track"}], context_key=context, acquisition_id="b"
    )
    assert not browser.track_table.selectionModel().hasSelection()


def test_request_browser_update_does_not_access_dense_products_or_evaluation(qtbot):
    browser = _browser(qtbot)
    sensing = _sensing()
    sensing["detections"] = [{"range_m": 2}]
    sensing["temporal"] = {
        "clock": {"clock_id": "clock"},
        "result": {"processing_run_id": "run"},
        "acquisition": {"pair_id": "pair", "acquisition_id": "a"},
    }
    request = SimpleNamespace(display_frame={"sensing": sensing}, source_identity=("source",))
    browser.update_from_request(request, tracks_note="No active tracks", complete_track_states=True)
    assert browser.model.rowCount() == 1
    assert browser._context_key == (("source",), "clock", "run", "pair")
    assert not browser.table.selectionModel().hasSelection()
    assert browser.tabs.tabText(1) == "Tracks (0)"
    assert browser.track_status.text() == "No active tracks"


def test_browser_does_not_call_missing_track_states_an_empty_population(qtbot):
    browser = _browser(qtbot)
    browser.update_result([], tracks_note="Tracking states were not recorded.")
    browser.tabs.setCurrentIndex(1)
    assert browser.tabs.tabText(1) == "Tracks"
    assert browser.track_status.isVisible()
    browser.update_result([], complete_track_states=True, tracks_note="No active tracks")
    assert browser.tabs.tabText(1) == "Tracks (0)"
    assert browser.track_status.text() == "No active tracks"


def test_request_browser_hides_unobservable_elevation_for_one_row_array(qtbot):
    browser = _browser(qtbot)
    sensing = {
        "config_resolved": {"antenna_rows": 1},
        "detections": [{"range_m": 2, "aoa_az_deg": 12, "aoa_el_deg": 0}],
    }
    request = SimpleNamespace(display_frame={"sensing": sensing}, source_identity=("source",))
    browser.update_from_request(request)
    headers = [
        browser.model.headerData(i, Qt.Orientation.Horizontal)
        for i in range(browser.model.columnCount())
    ]
    assert "Azimuth (deg)" in headers
    assert "Elevation (deg)" not in headers
    sensing["config_resolved"]["antenna_rows"] = 2
    browser.update_from_request(request)
    headers = [
        browser.model.headerData(i, Qt.Orientation.Horizontal)
        for i in range(browser.model.columnCount())
    ]
    assert "Elevation (deg)" in headers


def test_track_source_identifiers_stay_in_tooltip(qtbot):
    browser = _browser(qtbot)
    source_id = "long-recording-identity"
    browser.update_result(
        [], [{"track_id": 7, "source_kind": "saved_world_track", "source_key": (source_id,)}]
    )
    index = browser.track_model.index(0, 0)
    assert browser.track_model.data(index) == "7"
    tooltip = browser.track_model.data(index, Qt.ItemDataRole.ToolTipRole)
    assert tooltip.startswith("Saved world track") and source_id in tooltip


def test_current_summary_fits_two_columns_in_embedded_panel(qtbot):
    panel = CurrentResultSummary()
    qtbot.addWidget(panel)
    panel.update_result(_sensing(), evaluation={"total_tp": 10, "total_fp": 0, "total_fn": 0})
    panel.resize(340, 250)
    panel.show()
    qtbot.waitUntil(lambda: panel._columns == 2)
    assert panel.grid.getItemPosition(panel.grid.indexOf(panel.cards["Filtering"]))[:2] == (0, 1)
    assert panel.grid.getItemPosition(panel.grid.indexOf(panel.cards["Evaluation"]))[:2] == (1, 1)
    assert panel.minimumSizeHint().width() <= 340
    assert not hasattr(panel.cards["Detection"], "details")
    panel.resize(1000, 200)
    qtbot.waitUntil(lambda: panel._columns == 4)


def test_playback_preserves_table_models_sorting_and_user_column_widths(qtbot):
    browser = _browser(qtbot)
    browser.update_result([{"range_m": 9}, {"range_m": 2}], [{"track_id": 7}])
    browser.table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
    browser.table.setColumnWidth(1, 245)
    resets = []
    schema_changes = []
    browser.model.modelReset.connect(lambda: resets.append("detection"))
    browser.track_model.modelReset.connect(lambda: resets.append("track"))
    browser.model.headerDataChanged.connect(lambda *_: schema_changes.append(True))
    browser.update_result([{"range_m": 1}, {"range_m": 6}], [{"track_id": 8}])
    assert resets == []
    assert browser.model.data(browser.model.index(0, 1)) == "1.00"
    assert browser.track_model.data(browser.track_model.index(0, 0)) == "8"
    assert browser.proxy.mapToSource(browser.proxy.index(0, 0)).row() == 0
    assert browser.table.columnWidth(1) == 245
    browser.update_result([{"range_m": 1, "aoa_az_deg": 3}, {"range_m": 6}])
    assert schema_changes
    assert browser.model.headerData(4, Qt.Orientation.Horizontal) == "Azimuth (deg)"


@pytest.mark.parametrize("tracks", [False, True])
def test_stable_table_rows_repaint_changed_numeric_values(qtbot, tracks):
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtGui import QStandardItemModel

    from visualizer.src.sensing.result_browser import DetectionTableModel, TrackTableModel

    model_type = TrackTableModel if tracks else DetectionTableModel
    assert model_type.data is QStandardItemModel.data
    assert model_type.headerData is QStandardItemModel.headerData
    browser = _browser(qtbot)
    table = browser.track_table if tracks else browser.table
    model = browser.track_model if tracks else browser.model
    proxy = browser.track_proxy if tracks else browser.proxy
    column = 2 if tracks else 1
    if tracks:
        browser.tabs.setCurrentIndex(1)

    class PaintCounter(QObject):
        count = 0

        def eventFilter(self, watched, event):  # noqa: N802
            if event.type() == QEvent.Type.Paint:
                self.count += 1
            return False

    paints = PaintCounter(table)
    table.viewport().installEventFilter(paints)

    def update(value):
        if tracks:
            browser.update_result([], [{"track_id": 7, "estimated_range_m": value}])
        else:
            browser.update_result([{"range_m": value}])

    update(1.0)
    qtbot.waitUntil(lambda: paints.count > 0)
    before = table.viewport().grab(table.visualRect(proxy.index(0, column))).toImage()
    paints.count = 0
    source_roles, proxy_roles = [], []
    model.dataChanged.connect(lambda _first, _last, roles: source_roles.append(list(roles)))
    proxy.dataChanged.connect(lambda _first, _last, roles: proxy_roles.append(list(roles)))
    update(123.45)
    qtbot.waitUntil(lambda: paints.count > 0)
    assert proxy.data(proxy.index(0, column)) == "123.45"
    assert source_roles == proxy_roles == [[]]  # An empty role list invalidates all roles.
    after = table.viewport().grab(table.visualRect(proxy.index(0, column))).toImage()
    assert before != after


def test_hidden_tables_keep_current_selection_context_without_reformatting_on_show(
    qtbot, monkeypatch
):
    from visualizer.src.sensing import result_browser

    browser = SensingResultBrowser()
    qtbot.addWidget(browser)
    formats = []
    original = result_browser._format

    def record_format(*args, **kwargs):
        formats.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(result_browser, "_format", record_format)
    browser.update_result([{"range_m": 1}], acquisition_id="old")
    browser.select_detection(0)
    browser.update_result([{"range_m": 7}], acquisition_id="new")
    prepared_formats = list(formats)
    assert prepared_formats
    assert browser._selection_key is None
    assert browser.model.rowCount() == 1
    browser.show()
    qtbot.waitExposed(browser)
    assert browser.model.data(browser.model.index(0, 1)) == "7.00"
    assert formats == prepared_formats


def test_current_summary_only_reflows_when_card_topology_changes(qtbot, monkeypatch):
    panel = CurrentResultSummary()
    qtbot.addWidget(panel)
    panel.resize(500, 250)
    panel.update_result(_sensing())
    calls = []
    original = panel.grid.addWidget

    def add_widget(*args):
        calls.append(args)
        original(*args)

    monkeypatch.setattr(panel.grid, "addWidget", add_widget)
    sensing = _sensing()
    sensing["detections"] = [{"range_m": 2}]
    sensing["detection_filter_metadata"]["final_kept"] = 1
    panel.update_result(sensing)
    assert calls == []
    assert panel.cards["Detection"].value.text().startswith("1 published")
    panel.update_result(sensing, evaluation={"total_tp": 1, "total_fp": 0, "total_fn": 0})
    assert len(calls) == 4


def test_native_detection_roles_preserve_units_sorting_colors_and_schema_changes(qtbot):
    from PySide6.QtCore import QModelIndex

    from visualizer.src.sensing.display_style import detection_color
    from visualizer.src.sensing.result_browser import DetectionTableModel

    browser = _browser(qtbot)
    model = browser.model
    detections = [
        {
            "detection_id": "saved-row",
            "range_m": 9.125,
            "radial_velocity_m_s": -2.25,
            "snr_db": float("nan"),
            "aoa_az_deg": 12.34,
        }
    ]
    browser.update_result(detections)
    assert isinstance(model, DetectionTableModel)
    assert model._detections == detections
    assert model._extra_columns == [("aoa_az_deg", "Azimuth (deg)")]
    assert model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole) == "saved-row"
    assert model.data(model.index(0, 0), Qt.ItemDataRole.ForegroundRole).name() == detection_color(
        detections[0]
    )
    assert model.data(model.index(0, 1), Qt.ItemDataRole.UserRole) == 9.125
    assert model.data(model.index(0, 2)) == "-2.25"
    assert model.data(model.index(0, 3)) == "—"
    assert model.data(model.index(0, 3), Qt.ItemDataRole.UserRole) == float("-inf")
    assert model.data(model.index(0, 4)) == "12.3"
    assert model.data(model.index(0, 4), Qt.ItemDataRole.ToolTipRole) == "Receiver-local direction"
    assert model.rowCount(QModelIndex()) == 1
    assert model.rowCount(model.index(0, 0)) == model.columnCount(model.index(0, 0)) == 0
    assert not model.flags(model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable

    changes = []
    model.dataChanged.connect(
        lambda first, last, roles: changes.append((first.row(), last.column()))
    )
    browser.update_result([{"range_m": 3.0, "track_id": 7}])
    assert changes == [(0, 4)]
    assert model.columnCount() == 5  # A different optional field occupies the same native column.
    assert model.headerData(4, Qt.Orientation.Horizontal) == "Track"
    assert model.data(model.index(0, 4)) == "7"
    assert model.data(model.index(0, 4), Qt.ItemDataRole.UserRole) == 7.0
    assert model.data(model.index(0, 4), Qt.ItemDataRole.ToolTipRole) is None
    browser.update_result([])
    assert model.rowCount() == 0 and model.columnCount() == 4


def test_native_track_rows_batch_roles_and_clear_previous_position(qtbot):
    browser = _browser(qtbot)
    model = browser.track_model
    state = {
        "track_id": 7,
        "confirmed": True,
        "predicted_only": True,
        "estimated_range_m": 12.5,
        "radial_velocity_m_s": -1.0,
        "position_m": [1, 2, 3],
        "position_frame": "world",
        "source_kind": "saved_world_track",
        "source_key": ("recording", "tracker"),
    }
    browser.update_result([], [state])
    assert model.states == [state]
    assert model.data(model.index(0, 1)) == "Confirmed · coasting"
    assert model.data(model.index(0, 2), Qt.ItemDataRole.UserRole) == 12.5
    assert model.data(model.index(0, 4)) == "1.00, 2.00, 3.00"
    assert model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole) == (
        "Saved world track\nSource: recording / tracker"
    )
    changes = []
    model.dataChanged.connect(lambda *_: changes.append(True))
    browser.update_result([], [{"track_id": 9}])
    assert changes == [True]
    assert model.data(model.index(0, 0)) == "9"
    assert model.data(model.index(0, 1)) == "—"
    assert model.data(model.index(0, 4)) == "—"
    assert model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole) == "Saved track"
    assert model.data(model.index(0, 1), Qt.ItemDataRole.ForegroundRole) is None
