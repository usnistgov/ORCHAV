"""Independent boundaries for bounded recorded-geometry presentation."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_cinematic_pulses import renderer
from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.cinematic_pulses import CinematicPulseGeometry, cinematic_subset
from visualizer.src.experiments.service import ExperimentService
from visualizer.src.experiments.window import ExperimentWindow
from visualizer.src.renderers.pygfx.experiments import ExperimentOverlay


@pytest.fixture
def service(qtbot):
    viz = SimpleNamespace(renderer=renderer(), animation_running=False)
    obj = ExperimentService(viz)
    obj.window = ExperimentWindow()
    qtbot.addWidget(obj.window)
    obj.window.show()
    obj.overlay = ExperimentOverlay(viz.renderer)
    obj.window.cinematic_changed.connect(obj._cinematic_changed)
    obj.window.play_changed.connect(obj.play)
    yield obj
    obj.close()


def test_all_default_preserves_longest_path_and_prepares_paused(service):
    points = [[(0, 0, 0), (1, 0, 0)]] * 10 + [[(0, 0, 0), (100, 0, 0)]]
    service.catalog = _catalog(list(range(11)), losses=list(range(11)), points=points)
    assert not hasattr(service.window, "cinematic")
    assert service.window.cinematic_limit.currentText() == "All paths"
    service._emphasize(np.arange(11), "test")
    full_end = service._clock_end
    assert not service._clock.isActive()
    service.window.cinematic_limit.setCurrentIndex(service.window.cinematic_limit.findData(10))
    assert service._clock_end < full_end
    service.window.cinematic_limit.setCurrentIndex(0)
    assert service._clock_end == full_end
    assert not service._clock.isActive()
    np.testing.assert_array_equal(service._selected_population, np.arange(11))


def test_all_and_top_limits_use_full_population(service):
    count = 1502
    service.catalog = _catalog(list(range(count)), losses=list(range(count))[::-1])
    service._emphasize(np.arange(count), "test")
    assert len(service.path_ids) == count
    assert service._cinematic_geometry.sample(1).leading.shape == (count, 3)
    assert "1502/1502" in service.window.cinematic_note.text()
    service.window.cinematic_limit.setCurrentIndex(service.window.cinematic_limit.findData(10))
    np.testing.assert_array_equal(service._cinematic_geometry.ids, np.arange(1501, 1491, -1))
    assert len(service._selected_population) == count
    assert "10/1502" in service.window.cinematic_note.text()


def test_mixed_loss_provenance_and_ties_are_deterministic():
    catalog = _catalog(
        [1, 2, 3, 4], losses=[50, -100, 50, 1], estimated_loss=[False, True, False, True]
    )
    ids, _ = cinematic_subset(catalog, [3, 2, 1, 0], 10)
    np.testing.assert_array_equal(ids, [0, 2, 1, 3])


@pytest.mark.parametrize("path_budget,point_budget", [(1, 100), (100, 3)])
def test_batch_budgets_keep_all_paths_and_exact_positions(monkeypatch, path_budget, point_budget):
    monkeypatch.setattr(CinematicPulseGeometry, "BATCH_PATHS", path_budget)
    monkeypatch.setattr(CinematicPulseGeometry, "BATCH_POINTS", point_budget)
    catalog = _catalog([1, 2, 3], points=[[(0, 0, 0), (10, 0, 0)]] * 3)
    ids, _ = cinematic_subset(catalog, [2, 1, 0], None)
    geometry = CinematicPulseGeometry(catalog, ids)
    assert len(geometry._batches) == 3
    assert geometry.omitted_count == 0
    assert len(geometry.ids) == 3
    sample = geometry.sample(1)
    np.testing.assert_allclose(sample.leading, [[0.299792458, 0, 0]] * 3)
    assert sample.trail.shape == (30, 3)


def test_geometry_exceeds_former_point_cap_without_omissions():
    long = [(float(i), 0, 0) for i in range(12000)]
    geometry = CinematicPulseGeometry(_catalog([1, 2], points=[long, long]), [0, 1])
    assert geometry.omitted_count == 0
    assert geometry.ids.tolist() == [0, 1]
    assert geometry.sample(1).trail.shape == (20, 3)


def test_native_clipping_changes_apply_without_clock_tick(qapp):
    import os

    if os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1":
        pytest.skip("Native renderer opt-in")
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    native = PygfxRenderer(SimpleNamespace())
    try:
        native.initialize_visualizer(width=128, height=128)
        catalog = _catalog([1], points=[[(0, 0, 0), (1, 0, 0), (1, 1, 0)]])
        catalog._canonical_data.itype[1] = 1
        overlay = ExperimentOverlay(native)
        geometry = CinematicPulseGeometry(catalog, [0])
        overlay.cinematic_routes(catalog, [0])
        overlay.cinematic_pulses(geometry.sample(1))
        planes = ((1.0, 0.0, 0.0, 0.5),)
        native.set_clipping_planes(planes)
        for layer in overlay.layers.values():
            np.testing.assert_allclose(layer.material.clipping_planes, planes)
        overlay.clear()
        assert not native._presentation_materials
    finally:
        native.close()


def test_limit_change_preserves_playing_state(service):
    service.catalog = _catalog(list(range(20)))
    service._emphasize(np.arange(20), "test")
    service.window.play.setChecked(True)
    assert service._clock.isActive()
    service.window.cinematic_limit.setCurrentIndex(service.window.cinematic_limit.findData(10))
    assert service._clock.isActive()
    assert len(service._cinematic_geometry.ids) == 10
    assert len(service._selected_population) == 20
