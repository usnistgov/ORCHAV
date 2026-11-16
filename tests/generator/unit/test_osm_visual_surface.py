"""Visual surface retention must not silently change radio material selection."""

from types import SimpleNamespace

import pytest

pytest.importorskip("osmnx")

from generator.osm import scene_builder  # noqa: E402


@pytest.mark.parametrize("already_retained", [False, True])
def test_capture_surface_preserves_original_rf_contract(monkeypatch, already_retained):
    tags = ["highway", "width"] + (["surface"] if already_retained else [])
    settings = SimpleNamespace(useful_tags_way=tags)
    edge = {"highway": "footway", "surface": "paving_stones"}
    graph = SimpleNamespace(edges=lambda **kwargs: [(1, 2, edge)])

    def fetch(*args, **kwargs):
        assert "surface" in settings.useful_tags_way
        return graph

    monkeypatch.setattr(
        scene_builder, "ox", SimpleNamespace(settings=settings, graph_from_point=fetch)
    )
    assert scene_builder._fetch_street_graph((1, 2), 100) is graph
    assert settings.useful_tags_way is tags
    assert edge["_visual_surface"] == "paving_stones"
    assert ("surface" in edge) == already_retained


def test_capture_surface_restores_settings_after_fetch_failure(monkeypatch):
    tags = ["highway"]
    settings = SimpleNamespace(useful_tags_way=tags)

    def fetch(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(
        scene_builder, "ox", SimpleNamespace(settings=settings, graph_from_point=fetch)
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        scene_builder._fetch_street_graph((1, 2), 100)
    assert settings.useful_tags_way is tags
