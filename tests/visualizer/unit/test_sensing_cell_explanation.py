"""Selected-cell evidence stays bound to saved products and source identities."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_experimental_range_geometry import _request
from tests.visualizer.unit.test_sensing_stage_dashboard import _typed_sensing, _with_detection
from visualizer.src.sensing.dashboard_data import load_dashboard_request
from visualizer.src.sensing.selection import inspection_cell, select_estimate
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.stage_inspector import build_detection_stage_bundle
from visualizer.src.sensing.update_request import SensingUpdateRequest


@pytest.fixture
def window(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    yield window
    window.close()


def test_preview_changes_with_range_while_velocity_changes_only_cell_values(window):
    request, selection = _request(mode="monostatic", tx=(0, 0, 0), rx=(0, 0, 0))
    rd = request.rd_result
    rd.power_linear = np.arange(1.0, 10.0).reshape(3, 3)
    before = rd.power_linear.copy()
    card = window.cell_explanation
    card.update_cell(request, selection)
    outline = card.locus.equator.copy()
    text = card.values.text()
    next_cell = replace(selection, cell=inspection_cell(request, 1, 2))
    card.update_cell(request, next_cell)
    np.testing.assert_array_equal(card.locus.equator, outline)
    assert card.values.text() != text
    assert "Displayed power 9.0 dB" in card.values.text()
    card.update_cell(request, replace(selection, cell=inspection_cell(request, 2, 2)))
    assert not np.array_equal(card.locus.equator, outline)
    np.testing.assert_array_equal(rd.power_linear, before)
    card.update_cell(request, None)
    assert card.isHidden() and card.locus is None


@pytest.mark.parametrize("change", ["source", "acquisition", "recipe", "variant", "catalog"])
def test_explanation_does_not_reuse_a_retired_coordinate(window, change):
    request, selection = _request()
    request.rd_result.power_linear = np.ones((3, 3))
    window.cell_explanation.update_cell(request, selection)
    if change == "source":
        request = replace(request, source_identity=("new-source",))
    elif change == "acquisition":
        selection = replace(selection, acquisition_id="other-acquisition")
    elif change == "recipe":
        selection = replace(selection, cell=replace(selection.cell, recipe="another-recipe"))
    elif change == "variant":
        request = replace(request, variant="raw")
    else:
        selection = replace(selection, cell=replace(selection.cell, catalog_identity=()))
    window.cell_explanation.update_cell(request, selection)
    assert window.cell_explanation.isHidden()
    assert window.cell_explanation.locus is None


def test_saved_detector_decision_is_explicit_and_other_recipe_is_unavailable(window):
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    request = SensingUpdateRequest(
        display_frame={"sensing": sensing},
        state_frame={},
        state_step=0,
        detection_stage_bundle=bundle,
        variant="primary",
        source_identity=("source",),
    )
    selection = replace(
        select_estimate(sensing),
        source=request.source_identity,
        cell=inspection_cell(request, 2, 1),
    )
    card = window.cell_explanation
    card.update_cell(request, selection)
    assert "Retained as a published detection" in card.decision.text()
    assert "Detector input" in card.decision.text() and "threshold" in card.decision.text()
    assert card.detector_button.isEnabled()
    other = replace(bundle, event=replace(bundle.event, processing_recipe_id="other-recipe"))
    card.update_cell(replace(request, detection_stage_bundle=other), selection)
    assert "unavailable" in card.decision.text()
    assert not card.detector_button.isEnabled()


def test_unavailable_recording_hides_previous_cell_preview(window):
    request, selection = _request()
    request.rd_result.power_linear = np.ones((3, 3))
    window.cell_explanation.update_cell(request, selection)
    assert not window.cell_explanation.isHidden()
    window.set_unavailable("No result here", scene_frame_id=19)
    assert window.cell_explanation.isHidden()
    assert window.cell_explanation.locus is None


def test_only_open_explanation_loads_detector_evidence(monkeypatch):
    from tests.visualizer.unit.test_sensing_dashboard_recording import _request as saved_request

    request = saved_request()
    request.display_frame["_sensing_display_provenance"] = {"storage_frame_id": 0, "pinned": False}
    calls = []
    selection = SimpleNamespace(
        display_frame=request.display_frame,
        bound_scene_frame_id=0,
        stale_for_current_scene=False,
        replay_event=None,
        storage_frame_id=0,
    )
    service = SimpleNamespace(
        dashboard_result_rows=lambda: (),
        recording_cache=None,
        dashboard_scene_frame=lambda step: request.display_frame,
        resolve_display_sensing_frame=lambda *args, **kwargs: selection,
        dashboard_products=lambda *args: (request.rp_result, request.rd_result),
        load_detection_stage_bundle=lambda selected: calls.append(selected) or "bundle",
        dashboard_product_kinds=lambda step: frozenset(),
    )
    ordinary = load_dashboard_request(service, 0, "analysis", "primary", False, "auto")
    assert ordinary.detection_stage_bundle is None and calls == []
    explained = load_dashboard_request(service, 0, "analysis-inspection", "primary", False, "auto")
    assert explained.detection_stage_bundle == "bundle" and calls == [selection]
