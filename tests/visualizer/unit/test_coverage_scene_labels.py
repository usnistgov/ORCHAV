"""Coverage annotations share actor labels without rebuilding unchanged objects."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tests.visualizer.unit.test_coverage_volume import Renderer
from visualizer.src.coverage.scene_labels import CoverageSceneLabels
from visualizer.src.renderers.protocol import RendererCapabilities


def make_labels():
    renderer = Renderer()
    renderer.capabilities = RendererCapabilities(volume_rendering=True, label_callouts=True)
    viz = SimpleNamespace(
        renderer=renderer,
        app_state=SimpleNamespace(
            tx_device_names=["A"], tx_labels=["Base station"], show_labels=True
        ),
        current_tx_positions=np.array([[10, 20, 8.0]]),
        tx_labels=[object()],
        tx_entries=[{}],
    )
    labels = CoverageSceneLabels(
        viz, dict(tx_names=["A", "B"], tx_positions=[[10, 20, 8], [30, 40, 12]])
    )
    return viz, labels


def test_matching_actor_reused_and_unchanged_sync_does_no_renderer_work(monkeypatch):
    viz, labels = make_labels()
    labels.sync()
    assert labels.active_ids == {"analysis:coverage_site:B:point", "analysis:coverage_site:B:label"}
    snapshots = labels.objects
    viz.renderer.ensure_object = Mock(side_effect=AssertionError("Unchanged renderer write"))
    monkeypatch.setattr(np, "allclose", Mock(side_effect=AssertionError("Unchanged matching")))
    for _ in range(10):
        labels.sync()
    assert labels.objects is snapshots


def test_missing_and_moved_actors_use_recorded_positions_and_shared_preferences():
    viz, labels = make_labels()
    viz.current_tx_positions[0, 2] = 18
    viz.label_font_size = 0.7
    viz.label_offset_z = 2
    labels.sync()
    label = viz.renderer.objects["analysis:coverage_site:A:label"]
    assert label.payload.text == "Base station (coverage)"
    assert label.payload.font_size == 0.7
    np.testing.assert_equal(label.transform.matrix[:3, 3], [10, 20, 10])
    viz.app_state.label_callout = dict(mode="callout", tx_marker_size=31)
    labels.sync()
    label = viz.renderer.objects[label.id]
    assert label.metadata["label_callout"]["marker_size"] == 31
    assert not viz.renderer.objects["analysis:coverage_site:A:point"].visibility.visible
    viz.app_state.show_labels = False
    labels.sync()
    assert not viz.renderer.objects
    viz.app_state.show_labels = True
    labels.sync()
    assert labels.active_ids
    labels.clear()
    assert not viz.renderer.objects


def test_late_actor_replaces_only_coverage_fallback_and_per_node_hide_is_respected():
    viz, labels = make_labels()
    viz.renderer.objects["tx_label_0"] = viz.tx_labels[0]
    labels.sync()
    viz.app_state.tx_device_names.append("B")
    viz.tx_labels.append(object())
    viz.current_tx_positions = np.array([[10, 20, 8], [30, 40, 12]])
    labels.sync()
    assert not labels.active_ids
    assert "tx_label_0" in viz.renderer.objects
    viz.current_tx_positions[0, 2] = 18
    viz.tx_entries[0]["show_label"] = False
    labels.sync()
    assert not labels.active_ids
    viz.tx_entries[0]["show_label"] = True
    labels.sync()
    assert "analysis:coverage_site:A:label" in labels.active_ids
    viz.app_state.selected_tx = 1
    labels.sync()
    assert not labels.active_ids
