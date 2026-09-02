"""Tests for visualizer object selection state."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visualizer.src.app.selection_manager import SelectionManager
from visualizer.src.model import RenderObjectState
from visualizer.src.types.render_payloads import MeshPayload


def _mesh_payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )


def test_scene_selection_uses_stable_token_for_render_state() -> None:
    calls: list[tuple[str, bool]] = []
    entry = {
        "mesh": RenderObjectState(id="scene:wall::mesh", payload=_mesh_payload()),
        "name": "Wall",
        "stable_mesh_id": "wall",
        "visible": True,
    }
    viz = SimpleNamespace(
        mesh_entries=[entry],
        tx_markers=[],
        rx_markers=[],
        selected_objects=set(),
        object_appearance_service=SimpleNamespace(
            refresh_entry_material=lambda selected: calls.append(
                (
                    selected["name"],
                    selected["object_key"] in viz.selected_objects,
                )
            )
        ),
        ui_controller=None,
        selection_info_label=None,
        app_state=SimpleNamespace(),
    )
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    manager = SelectionManager(viz, logger)

    manager.toggle_object_selection(
        {
            "geometry": entry["mesh"],
            "selection_token": manager._scene_selection_token(entry),
            "type": "building",
            "name": "Wall",
            "entry": entry,
        }
    )

    assert len(viz.selected_objects) == 1
    selected_token = next(iter(viz.selected_objects))
    assert isinstance(selected_token, str)
    assert selected_token == entry["object_key"]
    assert calls == [("Wall", True)]

    manager.toggle_object_selection(
        {
            "geometry": entry["mesh"],
            "selection_token": selected_token,
            "type": "building",
            "name": "Wall",
            "entry": entry,
        }
    )

    assert viz.selected_objects == set()
    assert calls == [("Wall", True), ("Wall", False)]
