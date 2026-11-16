"""Private map-generated scene copies retain the same portable visual assets."""

import pytest

from tests.visualizer.authoring.test_visual_surface_dependencies import (
    _assert_copied_scene_retains_classification,
    _assert_copy_retains_visual_sidecar,
)
from tests.visualizer.authoring.test_visual_surface_dependencies import (
    surface_scene as _surface_scene,
)
from tests.visualizer.authoring.test_visual_surface_dependencies import (
    visual_scenario as _visual_scenario,
)

surface_scene = _surface_scene
visual_scenario = _visual_scenario


def test_osm_copy_retains_visual_sidecar_and_only_referenced_maps(visual_scenario):
    _assert_copy_retains_visual_sidecar(visual_scenario, "osm")


@pytest.mark.parametrize("metadata_name", ["building_visuals.json", "osm_metadata.json"])
def test_osm_copied_scene_retains_classification(surface_scene, metadata_name):
    _assert_copied_scene_retains_classification(surface_scene, "osm", metadata_name)
