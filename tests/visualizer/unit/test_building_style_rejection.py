"""An oversized render-only facade must not freeze every subsequent material sync."""

import logging
from dataclasses import replace

import numpy as np

from visualizer.src.materials import haussmann_geometry
from visualizer.src.materials.building_style import apply_building_style
from visualizer.src.model import RenderObjectState
from visualizer.src.types.render_payloads import MeshPayload


def test_budget_rejection_is_reused_and_invalidates_on_source_or_recipe_change(monkeypatch, caplog):
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    monkeypatch.setattr(haussmann_geometry, "MAX_RENDER_VERTICES", 3)
    source = MeshPayload(
        vertices=np.array([[0.0, 0, 0], [9, 0, 0], [9, 0, 20], [0, 0, 20]]),
        triangles=np.array([[0, 1, 2], [0, 2, 3]]),
    )
    entry = dict(
        name="building",
        mesh=RenderObjectState(id="wall", payload=source),
        _osm_building_faces=2,
        _osm_building_parts=[[0, 2]],
    )
    props = {"color": [0.7, 0.7, 0.7]}
    calls = []
    original = haussmann_geometry.facade_payload

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(haussmann_geometry, "facade_payload", counted)

    def apply(**kwargs):
        return apply_building_style(
            entry,
            props,
            enabled=kwargs.get("enabled", True),
            eligible=True,
            mode="varied",
            theme="haussmann",
        )

    with caplog.at_level(logging.WARNING, logger="orchav.building_style"):
        for _ in range(5):
            assert apply() is props
            assert entry["mesh"].payload is source
    assert len(calls) == 1
    assert sum("subdivision budget" in r.message for r in caplog.records) == 1
    apply(enabled=False)
    apply()
    assert len(calls) == 1
    changed = replace(source, vertices=source.vertices + [0.5, 0, 0])
    entry["mesh"].replace_payload(changed)
    apply()
    assert len(calls) == 2
    # An in-place classification edit must not reuse the old failure snapshot.
    entry["_osm_building_parts"][:] = [[0, 1], [1, 1]]
    apply()
    assert len(calls) == 3
    entry["_osm_facade_exclusions"] = ((0, 1),)
    apply()
    assert len(calls) == 4


def test_transient_io_error_is_not_remembered_as_geometry_failure(monkeypatch):
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    source = MeshPayload(
        vertices=np.array([[0.0, 0, 0], [9, 0, 0], [0, 0, 20]]), triangles=np.array([[0, 1, 2]])
    )
    entry = dict(
        name="wall",
        mesh=RenderObjectState(id="wall", payload=source),
        _osm_building_faces=1,
        _osm_building_parts=[[0, 1]],
    )
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise OSError("temporary storage failure")

    monkeypatch.setattr(haussmann_geometry, "facade_payload", unavailable)
    for _ in range(2):
        apply_building_style(
            entry, {}, enabled=True, eligible=True, mode="varied", theme="haussmann"
        )
    assert len(calls) == 2
    assert "_building_style_rejection" not in entry
