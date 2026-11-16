"""Selection cues remain visible and optional without changing saved estimates."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pygfx as gfx
import pytest

from tests.visualizer.unit.test_sensing_dashboard_recording import _frame
from visualizer.src.renderers.pygfx.sensing import PygfxSensingAdapter
from visualizer.src.sensing.fusion_inspection import FusionSelection
from visualizer.src.sensing.selection import select_estimate
from visualizer.src.services.sensing_service import SensingSceneLayer
from visualizer.src.types.render_payloads import LineSetPayload, MaterialPayload, PointCloudPayload


class _Renderer:
    """Native pygfx materials/buffers without starting a GPU canvas."""

    def __init__(self):
        self._gfx = gfx
        self._objects, self._materials, self.snapshots = {}, {}, {}
        self.visualizer = SimpleNamespace(
            app_state=SimpleNamespace(extension_state={"sensing": {"show_detections": True}})
        )

    def ensure_object(self, snapshot):
        payload = snapshot.payload
        self.snapshots[snapshot.id] = snapshot
        self._materials[snapshot.id] = snapshot.material_payload
        geometry = gfx.Geometry(positions=np.array(payload.points, dtype=np.float32, copy=True))
        if payload.colors is not None:
            geometry.colors = gfx.Buffer(np.array(payload.colors, dtype=np.float32, copy=True))
        material = SimpleNamespace(opacity=1.0, pick_write=False)
        self._objects[snapshot.id] = SimpleNamespace(
            material=material, geometry=geometry, add_event_handler=lambda *args: None
        )
        return True

    def remove_object(self, name):
        self._objects.pop(name, None)
        self.snapshots.pop(name, None)
        return True

    def _object_has_color_buffer(self, obj):
        return getattr(obj.geometry, "colors", None) is not None

    def _apply_clipping_to_material(self, material):
        pass


def _scene(*, count=2, rings=False):
    frame = _frame()
    frame["sensing"]["detections"] = [{"range_m": float(i + 2)} for i in range(count)]
    if rings:
        angle = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        points = np.concatenate(
            [
                np.array([np.cos(angle) * (i + 2), np.sin(angle) * (i + 2), np.zeros(8)]).T
                for i in range(count)
            ]
        )
        edges = [(8 * i + j, 8 * i + (j + 1) % 8) for i in range(count) for j in range(8)]
        payload = LineSetPayload(
            points, np.asarray(edges), np.tile([0.2, 0.5, 0.8, 1], (count * 8, 1))
        )
    else:
        payload = PointCloudPayload(
            np.array([[float(i), 2, 3] for i in range(count)], dtype=np.float32),
            np.tile([0.2, 0.5, 0.8, 1], (count, 1)),
        )
    layer = SensingSceneLayer(
        "sensing_detections",
        payload,
        MaterialPayload(),
        metadata={"detection_indices": tuple(range(count))},
    )
    renderer = _Renderer()
    renderer.ensure_object(layer.to_render_object())
    adapter = PygfxSensingAdapter(renderer)
    adapter.bind_selection(frame, [layer])
    return renderer, adapter, frame, layer


@pytest.mark.parametrize("count", [1, 10])
def test_detection_selection_has_gold_outline_even_when_no_other_marker_can_be_dimmed(count):
    renderer, adapter, frame, layer = _scene(count=count)
    colors = layer.payload.colors.copy()
    selection = select_estimate(frame["sensing"], detection_index=0)
    adapter.refresh_selection(selection)
    outline = renderer.snapshots["sensing_selection_points"]
    np.testing.assert_array_equal(outline.payload.points, layer.payload.points[:1])
    native = renderer._objects[outline.id]
    assert native.material.edge_width == 2.5
    assert native.material.color.a == 0
    assert native.material.edge_color.a == 1
    assert not native.material.pick_write and not native.material.depth_test
    np.testing.assert_array_equal(layer.payload.colors, colors)
    adapter.refresh_selection(None)
    assert outline.id not in renderer._objects
    np.testing.assert_allclose(renderer._objects[layer.name].geometry.colors.data, colors)


def test_selected_range_only_ring_is_emphasized_without_inventing_position():
    renderer, adapter, frame, layer = _scene(rings=True)
    selection = select_estimate(frame["sensing"], detection_index=1)
    source = layer.payload.points.copy()
    adapter.refresh_selection(selection)
    assert "sensing_selection_points" not in renderer._objects
    ring = renderer.snapshots["sensing_selection_range"]
    np.testing.assert_allclose(ring.payload.points, source[8:])
    assert len(ring.payload.lines) == 8
    assert not renderer._objects[ring.id].material.depth_test
    assert adapter.selected_position(selection) is None
    np.testing.assert_array_equal(layer.payload.points, source)


@pytest.mark.parametrize("option", ["highlight_selection", "show_detections"])
def test_disabling_selection_cues_restores_base_markers_and_enabling_restores_cue(option):
    renderer, adapter, frame, layer = _scene()
    selection = select_estimate(frame["sensing"], detection_index=0)
    adapter.refresh_selection(selection)
    renderer.visualizer.app_state.extension_state["sensing"][option] = False
    adapter.refresh_selection(selection)
    assert not adapter._selection_outline_names
    np.testing.assert_allclose(
        renderer._objects[layer.name].geometry.colors.data, layer.payload.colors
    )
    renderer.visualizer.app_state.extension_state["sensing"][option] = True
    adapter.refresh_selection(selection)
    assert "sensing_selection_points" in renderer._objects
    adapter.clear_selection_bindings()
    assert "sensing_selection_points" not in renderer._objects


def test_base_overlay_resync_cannot_erase_same_selection_emphasis():
    renderer, adapter, frame, layer = _scene()
    selection = select_estimate(frame["sensing"], detection_index=0)
    adapter.highlight_selection(frame, [layer], selection)
    renderer.ensure_object(layer.to_render_object())
    adapter.bind_selection(frame, [layer])
    adapter.highlight_selection(frame, [layer], selection)
    assert renderer._objects[layer.name].geometry.colors.data[1, 3] == pytest.approx(0.18)
    assert "sensing_selection_points" in renderer._objects


def test_switching_detection_keeps_reused_outline_visible_and_hollow(monkeypatch):
    renderer, adapter, frame, layer = _scene()
    create = renderer.ensure_object

    def update(snapshot):
        native = renderer._objects.get(snapshot.id)
        if native is None:
            return create(snapshot)
        # Match the real renderer's in-place geometry and generic material update:
        # payload alpha becomes opacity, while color receives an opaque alpha.
        renderer.snapshots[snapshot.id] = snapshot
        renderer._materials[snapshot.id] = snapshot.material_payload
        native.geometry.positions.data[:] = snapshot.payload.points
        native.material.color = (*snapshot.material_payload.base_color[:3], 1.0)
        native.material.opacity = snapshot.material_payload.base_color[3]
        return True

    monkeypatch.setattr(renderer, "ensure_object", update)
    outline = None
    for index in (0, 1, 0, 1):
        adapter.refresh_selection(select_estimate(frame["sensing"], detection_index=index))
        native = renderer._objects["sensing_selection_points"]
        if outline is not None:
            assert native is outline
        outline = native
        np.testing.assert_array_equal(
            native.geometry.positions.data, layer.payload.points[index : index + 1]
        )
        assert native.material.opacity == 1
        assert native.material.color.a == 0
        assert native.material.edge_color.a == 1
        assert not native.material.depth_test


def test_untracked_detection_never_outlines_ground_truth_or_unrelated_points():
    renderer, adapter, frame, layer = _scene()
    truth = SensingSceneLayer(
        "sensing_truth_0",
        PointCloudPayload(np.array([[100, 200, 300]], dtype=np.float32)),
        MaterialPayload(),
        metadata={"track_id": None},
    )
    renderer.ensure_object(truth.to_render_object())
    adapter.bind_selection(frame, [layer, truth])
    adapter.refresh_selection(select_estimate(frame["sensing"], detection_index=0))
    np.testing.assert_array_equal(
        renderer.snapshots["sensing_selection_points"].payload.points,
        layer.payload.points[:1],
    )


@pytest.mark.parametrize("change", ["scope", "acquisition", "collection", "cell"])
def test_unrelated_selection_does_not_emphasize_current_estimates(change):
    renderer, adapter, frame, _ = _scene()
    selection = select_estimate(frame["sensing"], detection_index=0)
    kwargs = (
        dict(scope=("other", "run", "pair"))
        if change == "scope"
        else (
            dict(acquisition_id="other")
            if change == "acquisition"
            else (
                dict(detection_collection="detections_before_tracking")
                if change == "collection"
                else dict(detection_index=None)
            )
        )
    )
    adapter.refresh_selection(replace(selection, **kwargs))
    assert not adapter._selection_outline_names


def test_saved_world_track_highlight_follows_identity_within_run_and_handles_empty_native_frame():
    renderer = _Renderer()
    selection = FusionSelection("saved run", 5, 0)
    layers = [
        SensingSceneLayer(
            f"sensing_saved_track_point_{tid}",
            PointCloudPayload(np.array([[tid, 2, 3]], dtype=np.float32)),
            MaterialPayload(),
            metadata={"track_id": tid, "fusion_selection": FusionSelection("saved run", tid, 3)},
        )
        for tid in (5, 8)
    ]
    for layer in layers:
        renderer.ensure_object(layer.to_render_object())
    adapter = PygfxSensingAdapter(renderer)
    adapter.bind_selection({}, layers)
    adapter.refresh_selection(selection)
    np.testing.assert_array_equal(
        renderer.snapshots["sensing_selection_points"].payload.points, [[5, 2, 3]]
    )
    assert renderer._objects[layers[1].name].material.opacity == pytest.approx(0.18)
    np.testing.assert_array_equal(adapter.selected_position(selection), [5, 2, 3])
    adapter.refresh_selection(replace(selection, run_id="different run"))
    assert not adapter._selection_outline_names
    assert renderer._objects[layers[1].name].material.opacity == 1
