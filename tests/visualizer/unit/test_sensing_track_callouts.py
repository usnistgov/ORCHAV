"""Track callouts share actor styling and remain anchored across playback."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.renderers.pygfx.sensing import PygfxSensingAdapter
from visualizer.src.services.sensing_service import SensingSceneLayer
from visualizer.src.state import create_initial_state
from visualizer.src.types.label_style import LabelCalloutStyle
from visualizer.src.types.render_payloads import MaterialPayload, PointCloudPayload


@pytest.mark.parametrize("mode", ["callout", "card"])
def test_native_track_style_moves_reverses_and_cleans_up(mode):
    class Renderer:
        capabilities = RendererCapabilities(label_callouts=True)

    renderer = Renderer()
    renderer.visualizer = SimpleNamespace(
        app_state=create_initial_state(
            label_callout=LabelCalloutStyle(mode=mode, background="ink")
        ),
        renderer=renderer,
    )
    renderer._objects = {}
    snapshots, removed = [], []
    renderer.ensure_object = snapshots.append
    renderer.remove_object = removed.append
    adapter = PygfxSensingAdapter(renderer)

    def show(position):
        layer = SensingSceneLayer(
            name="sensing_track_point_7",
            payload=PointCloudPayload(points=np.asarray([position], dtype=np.float32)),
            material=MaterialPayload(base_color=(0.2, 0.8, 0.5, 1)),
        )
        adapter._update_sensing_track_labels(
            {"sensing": {"detections": [{"track_id": 7, "range_m": 19.25}]}}, [layer]
        )
        return snapshots[-1]

    first = show([1, 2, 3])
    moved = show([4, 5, 6])
    assert first.metadata["label_connector_origin"] == [1, 2, 3]
    assert moved.metadata["label_connector_origin"] == [4, 5, 6]
    assert moved.metadata["label_callout"]["mode"] == mode
    assert moved.metadata["pickable"] is False
    assert "19.2 m" in moved.payload.text
    assert moved.payload.outline_thickness == 0
    renderer.visualizer.app_state = replace(
        renderer.visualizer.app_state, label_callout=LabelCalloutStyle()
    )
    plain = show([4, 5, 6])
    assert "label_callout" not in plain.metadata
    assert plain.payload.outline_thickness > 0
    adapter._update_sensing_track_labels({}, [])
    assert removed == ["sensing_track_label_7"]
    assert not adapter._sensing_label_states


@pytest.mark.parametrize("distance", [None, float("nan"), float("inf"), "unknown"])
def test_track_label_omits_unavailable_range(distance):
    assert PygfxSensingAdapter._sensing_track_label_text(7, {"range_m": distance}) == "T7"
