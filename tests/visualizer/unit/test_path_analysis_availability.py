"""Data-dependent analysis controls must not offer empty or all-zero products."""

import numpy as np
import pytest

from tests.visualizer.unit.test_experimental_integration_review import _settings
from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.path_analysis import PathWeightMode
from visualizer.src.experiments.service import _analyze
from visualizer.src.experiments.window import ExperimentWindow
from visualizer.src.metrics.mpc_path_catalog import MpcPathCatalog


@pytest.mark.parametrize("loss_kind", ["missing", "estimated", "unknown"])
def test_unavailable_power_falls_back_to_count_and_missing_angles_disable_tab(qtbot, loss_kind):
    catalog = _catalog(
        [10, 20],
        losses=[np.nan, np.nan] if loss_kind == "missing" else [60, 70],
        estimated_loss=[True, True] if loss_kind == "estimated" else None,
        known_loss=loss_kind != "unknown",
    )
    data, histogram, end = _analyze(catalog, _settings(weight_mode="power"))
    assert data.weight_mode is PathWeightMode.COUNT
    np.testing.assert_array_equal(data.weights, [1, 1])
    window = ExperimentWindow()
    qtbot.addWidget(window)
    changes = []
    window.options_changed.connect(lambda: changes.append(True))
    window.set_analysis(data, histogram, reset_gate=True, clock_end=end)
    assert window.tabs.isTabEnabled(0)
    assert not window.tabs.isTabEnabled(1)
    assert not window.weight.model().item(window.weight.findData("power")).isEnabled()
    assert window.settings()["weight_mode"] == "count"
    assert not changes
    assert "Relative power unavailable" in window.availability_note.text()


def test_angle_fallback_uses_rows_with_both_delay_and_weight_and_restores_tab(qtbot):
    catalog = _catalog([np.nan, 10, 20], losses=[60, 70, 80], angles=[0, np.nan, np.nan])
    catalog.canonical_data.path_aod_az = np.array([np.nan, 40, 50], dtype=np.float32)
    data, histogram, end = _analyze(catalog, _settings())
    assert histogram.angle == "aod_azimuth"
    window = ExperimentWindow()
    qtbot.addWidget(window)
    window.set_analysis(data, histogram, reset_gate=True, clock_end=end)
    assert not window.angle.model().item(window.angle.findData("aoa_azimuth")).isEnabled()
    assert window.settings()["angle"] == "aod_azimuth"
    assert window.tabs.isTabEnabled(1)
    window.tabs.setCurrentIndex(1)
    window.clear_analysis()
    assert not window.tabs.isTabEnabled(0) and not window.tabs.isTabEnabled(1)
    window.set_analysis(data, histogram, reset_gate=True, clock_end=end)
    assert window.tabs.currentIndex() == 1
    assert window.weight.model().item(window.weight.findData("power")).isEnabled()


def test_no_delays_disables_both_tabs_and_frame_change_restores_available_plot(qtbot):
    window = ExperimentWindow()
    qtbot.addWidget(window)
    for delays, available in (([np.nan], False), ([30], True), ([], False)):
        data, histogram, end = _analyze(_catalog(delays), _settings())
        window.set_analysis(data, histogram, reset_gate=True, clock_end=end)
        assert window.tabs.isTabEnabled(0) == available
        assert not window.tabs.isTabEnabled(1)
        assert window.play.isEnabled() == available
        assert window.gate.isVisible() == available


def test_estimated_delays_are_usable_but_explicitly_identified(qtbot):
    catalog = _catalog([10, 20], losses=[np.nan, np.nan])
    catalog.canonical_data.path_delay_is_estimated[:] = True
    data, histogram, end = _analyze(catalog, _settings())
    window = ExperimentWindow()
    qtbot.addWidget(window)
    window.set_analysis(data, histogram, reset_gate=True, clock_end=end)
    assert window.tabs.isTabEnabled(0)
    assert "geometry estimates" in window.availability_note.text()
    assert "estimates" in window.arrival_plot.getAxis("bottom").labelText


def test_power_weights_remain_relative_and_angles_exclude_ineligible_rows(qtbot):
    catalog = _catalog(
        [10, 20, 30],
        losses=[60, 70, 0],
        angles=[np.nan, np.nan, 50],
        estimated_loss=[False, False, True],
    )
    data, histogram, end = _analyze(catalog, _settings(weight_mode="power"))
    np.testing.assert_allclose(data.weights, [1, 0.1, 0])
    window = ExperimentWindow()
    qtbot.addWidget(window)
    window.set_analysis(data, histogram, reset_gate=True, clock_end=end)
    assert window.settings()["weight_mode"] == "power"
    assert window.tabs.isTabEnabled(0)
    assert not window.tabs.isTabEnabled(1)


@pytest.mark.parametrize("selection", ["pair", "scope"])
def test_pair_or_scope_change_rechecks_availability(qtbot, selection):
    canonical = _catalog([10, 20], losses=[60, np.nan], angles=[40, np.nan]).canonical_data
    if selection == "pair":
        canonical.path_tx[:] = [0, 1]
    catalog = MpcPathCatalog(canonical, filtered_path_mask=np.array([False, True]))
    initial = _settings(weight_mode="power")
    unavailable = _settings(
        weight_mode="power",
        **({"pair": (1, 0)} if selection == "pair" else {"scope": "filtered"}),
    )
    window = ExperimentWindow()
    qtbot.addWidget(window)
    for settings, supported in ((initial, True), (unavailable, False), (initial, True)):
        data, histogram, end = _analyze(catalog, settings)
        window.set_analysis(data, histogram, reset_gate=True, clock_end=end)
        assert window.tabs.isTabEnabled(1) == supported
        assert window.weight.model().item(window.weight.findData("power")).isEnabled() == supported
        assert window.settings()["weight_mode"] == ("power" if supported else "count")
