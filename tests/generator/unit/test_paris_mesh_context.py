"""Confirm that the context draw path preserves authentic articulated geometry."""

import numpy as np
import pytest

from generator.figures.paris_mesh_context import context_layers
from generator.figures.temporal_mesh import _actor_layers, load_temporal_inputs
from tests.generator.unit.test_temporal_mesh_figures import (
    authored as authored,
)
from tests.generator.unit.test_temporal_mesh_figures import (
    temporal_animation as temporal_animation,
)


@pytest.fixture
def context_inputs(temporal_animation):
    scenario, _ = temporal_animation
    _, _, actors, _, _, _, _ = load_temporal_inputs(scenario)
    points = np.concatenate([mesh for actor in actors for mesh in actor["meshes"]]).reshape(-1, 3)
    low, high = points.min(0) - 0.2, points.max(0) + 0.2
    bounds = tuple(value for pair in zip(low, high) for value in pair)
    # An environment face intersects all six cutaway limits. Its vertices must
    # be clipped independently of the unchanged target endpoints.
    vertices = np.array([low - 10, high + 10, [low[0] - 10, high[1] + 10, high[2] + 10]])
    scene = [("building", vertices, np.array([[0, 1, 2]]))]
    return scene, actors, bounds


def test_city_and_street_layers_contain_identical_endpoint_meshes(context_inputs):
    scene, actors, street_bounds = context_inputs
    endpoints = np.concatenate([_actor_layers(a)[0] for a in actors])
    for bounds in (None, street_bounds):
        _, faces, colors = context_layers(scene, actors, bounds)
        assert np.array_equal(faces[-len(endpoints) :], endpoints)
        assert np.all(colors[-len(endpoints) :, 3] == 1)


def test_street_cutaway_preserves_every_target_pose(context_inputs):
    _, actors, bounds = context_inputs
    limits = np.asarray(bounds).reshape(3, 2)
    for actor in actors:
        vertices = np.concatenate(actor["meshes"]).reshape(-1, 3)
        assert np.all(vertices >= limits[:, 0])
        assert np.all(vertices <= limits[:, 1])


def test_environment_geometry_is_clipped_to_the_labeled_window(context_inputs):
    scene, _, bounds = context_inputs
    background, faces, _ = context_layers(scene, [], bounds)
    limits = np.asarray(bounds).reshape(3, 2)
    assert len(faces) > 0
    for triangles in [faces] + [layer[1] for layer in background]:
        assert np.all(triangles >= limits[:, 0] - 1e-9)
        assert np.all(triangles <= limits[:, 1] + 1e-9)


def test_context_and_temporal_exports_record_authenticated_animation(
    temporal_animation, context_inputs, tmp_path, monkeypatch
):
    import generator.figures.paris_mesh_context as context
    from generator.figures.temporal_mesh import render_temporal

    scenario, manifest = temporal_animation
    _, _, bounds = context_inputs
    monkeypatch.setattr(context, "SCENARIO", scenario)
    monkeypatch.setattr(context, "STREET_BOUNDS", bounds)
    temporal = render_temporal(scenario, tmp_path / "temporal-export", dpi=20)
    street = context.render_paris_context(tmp_path / "context-export", dpi=20)
    for receipt in (temporal, street):
        actor = receipt["actors"]["Reflector"]
        assert actor["animated"]
        assert actor["orientation_deg"] is None
        assert actor["extent_definition"] == "initial_world_aabb"
        assert len(actor["animated_samples"]) == len(actor["root_affines"]) == 3
        assert str(manifest) in receipt["source_files"]
        assert set(receipt["outputs"])
    assert street["renderer_sha256"]
    assert "source_revision" not in street
