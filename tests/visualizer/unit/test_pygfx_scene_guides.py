"""World guides use static environment bounds without feeding back into them."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pygfx as gfx
import pytest

from visualizer.src.renderers.pygfx.scene_guides import (
    PygfxSceneGuidesMixin,
    _coordinate_text,
    _coordinate_ticks,
)


class Owner(PygfxSceneGuidesMixin):
    def __init__(self):
        self._gfx = gfx
        self._scene = gfx.Scene()
        self._static_group = gfx.Group()
        self._scene.add(self._static_group)
        self._objects = {}
        self._reverse_objects = {}
        self._initialized = True
        self.request_redraw = Mock()

    @staticmethod
    def _is_scene_mesh_name(name):
        return name.startswith("scene:")

    def add(self, name="scene:building::mesh", *, position=(1000, -200, 30), size=(20, 10, 4)):
        mesh = gfx.Mesh(gfx.box_geometry(*size), gfx.MeshBasicMaterial())
        mesh.local.position = position
        self._static_group.add(mesh)
        self._objects[name] = mesh
        self._reverse_objects[id(mesh)] = name
        return mesh


def test_off_origin_grid_covers_whole_static_environment_at_its_floor():
    owner = Owner()
    owner.add()
    owner.add("target:moving::mesh", position=(1e6, 2e6, 3e6))
    owner.set_ground_grid_visible(True)
    state = owner.get_environment_dimensions_state()
    np.testing.assert_allclose(state["bounds"], [[990, -205, 28], [1010, -195, 32]])
    assert state["extent_m"] == (20, 10, 4)
    grid_bounds = owner._ground_grid_obj.get_world_bounding_box()
    assert np.all(grid_bounds[0, :2] <= np.array(state["bounds"])[0, :2])
    assert np.all(grid_bounds[1, :2] >= np.array(state["bounds"])[1, :2])
    np.testing.assert_allclose(grid_bounds[:, 2], 28.01)
    np.testing.assert_allclose(grid_bounds.mean(axis=0)[:2], [1000, -200])


def test_whole_environment_bounds_include_hidden_static_meshes_only():
    owner = Owner()
    owner.add()
    hidden = owner.add("scene:hidden::mesh", position=(1100, -200, 30))
    hidden.visible = False
    owner.set_environment_dimensions_visible(True)
    assert owner.get_environment_dimensions_state()["extent_m"] == (120, 10, 4)
    before = owner.get_environment_dimensions_state()["bounds"]
    owner.show_axes(True)
    owner.set_ground_grid_visible(True)
    assert owner.get_environment_dimensions_state()["bounds"] == before
    assert set(owner._objects) == {"scene:building::mesh", "scene:hidden::mesh"}


def test_steady_playback_never_reduces_vertices_or_rebuilds_native_guides():
    owner = Owner()
    source = owner.add()
    target = owner.add("target:moving::mesh")
    owner.set_ground_grid_visible(True)
    owner.set_environment_dimensions_visible(True)
    owner.show_axes(True)
    objects = (owner._ground_grid_obj, owner._environment_dimensions_group, owner._scene_axes_group)
    source.get_geometry_bounding_box = Mock(
        side_effect=AssertionError("unchanged geometry must use cached bounds")
    )
    for step in range(5):
        target.local.position = (step, step, step)
        owner._sync_scene_guides()
        assert (
            owner._ground_grid_obj,
            owner._environment_dimensions_group,
            owner._scene_axes_group,
        ) == objects


def test_late_scene_population_and_what_if_transform_refresh_the_same_grid():
    owner = Owner()
    owner.set_ground_grid_visible(True)
    grid = owner._ground_grid_obj
    assert owner.get_environment_dimensions_state()["bounds"] is None
    source = owner.add()
    owner._sync_scene_guides()
    assert owner._ground_grid_obj is grid
    np.testing.assert_allclose(grid.world.position[:2], [1000, -200])
    source.local.position = (1200, -300, 20)
    owner._sync_scene_guides()
    assert owner._ground_grid_obj is grid
    np.testing.assert_allclose(grid.world.position[:2], [1200, -300])
    np.testing.assert_allclose(
        owner.get_environment_dimensions_state()["bounds"], [[1190, -305, 18], [1210, -295, 22]]
    )


def test_vertex_revisions_geometry_replacement_and_parent_transform_refresh_bounds():
    owner = Owner()
    source = owner.add(position=(0, 0, 0))
    owner.set_environment_dimensions_visible(True)
    # Complete the initial native buffer upload before editing the live mesh:
    # update_full coalesces writes while an initial full upload is pending.
    source.geometry.positions._gfx_get_chunk_descriptions()
    source.geometry.positions.data[:, 0] *= 2
    source.geometry.positions.update_full()
    owner._sync_scene_guides()
    assert owner.get_environment_dimensions_state()["extent_m"] == (40, 10, 4)
    source.geometry = gfx.box_geometry(2, 4, 6)
    owner._static_group.local.position = (50, 60, 70)
    owner._sync_scene_guides()
    np.testing.assert_allclose(
        owner.get_environment_dimensions_state()["bounds"], [[49, 58, 67], [51, 62, 73]]
    )


def test_density_changes_only_rebuild_grid_and_stays_bounded():
    owner = Owner()
    owner.add()
    owner.set_ground_grid_visible(True)
    grid = owner._ground_grid_obj
    assert owner.get_ground_grid_divisions() == 20
    owner.set_ground_grid_divisions(50)
    assert grid.parent is None
    replacement = owner._ground_grid_obj
    assert len(replacement.geometry.positions.data) == 4 * 51
    owner.set_ground_grid_divisions(50)
    assert owner._ground_grid_obj is replacement
    for invalid in (True, 1, 501, 12.5, float("nan")):
        with pytest.raises(ValueError, match="divisions"):
            owner.set_ground_grid_divisions(invalid)
    assert owner.get_ground_grid_divisions() == 50


def test_native_axes_and_dimension_labels_are_screen_space_and_do_not_pick():
    owner = Owner()
    owner.add()
    owner.show_axes(True)
    owner.set_environment_dimensions_visible(True)
    helper = owner._scene_axes_group.children[0]
    assert isinstance(helper, gfx.AxesHelper)
    np.testing.assert_array_equal(helper.world.position, [0, 0, 0])
    labels = owner._scene_guide_labels()
    assert len(labels) == 6
    assert all(label.screen_space and label.font_size == 15 for label in labels)
    assert all(not label.material.pick_write and not label.material.depth_write for label in labels)
    assert not helper.material.pick_write
    assert all(not child.material.pick_write for child in helper.children)
    owner.show_axes(False)
    assert not owner.get_axes_visible() and not owner._scene_axes_group.visible


def test_disabled_guides_and_getters_do_not_inspect_environment():
    owner = Owner()
    owner._update_scene_guide_bounds = Mock(
        side_effect=AssertionError("optional guides are disabled")
    )
    owner._sync_scene_guides()
    assert owner.get_environment_dimensions_state() == {
        "enabled": False,
        "style": "size_labels",
        "bounds": None,
        "extent_m": None,
    }
    assert not owner.get_ground_grid_visible() and not owner.get_axes_visible()
    owner.set_ground_grid_divisions(40)
    assert getattr(owner, "_ground_grid_obj", None) is None


def test_clear_removes_helpers_and_forgets_previous_source_bounds():
    owner = Owner()
    owner.add()
    owner.set_ground_grid_visible(True)
    owner.set_environment_dimensions_visible(True)
    owner.show_axes(True)
    guides = (owner._ground_grid_obj, owner._environment_dimensions_group, owner._scene_axes_group)
    owner._clear_scene_guides()
    assert all(group.parent is None for group in guides)
    assert tuple(owner._scene.children) == (owner._static_group,)
    assert owner.get_environment_dimensions_state() == {
        "enabled": False,
        "style": "size_labels",
        "bounds": None,
        "extent_m": None,
    }
    assert not owner.get_ground_grid_visible() and not owner.get_axes_visible()


@pytest.mark.parametrize(
    "low,high", [(-32, 83), (990, 1010), (-0.35, 0.55), (-250, 250), (0, 1), (1234560, 1234590)]
)
def test_paper_ticks_are_sparse_nice_world_coordinates_and_include_real_zero(low, high):
    ticks, step = _coordinate_ticks(low, high)
    assert 4 <= len(ticks) <= 7
    assert np.all(ticks >= low - 1e-10) and np.all(ticks <= high + 1e-10)
    magnitude = 10 ** np.floor(np.log10(step))
    assert any(np.isclose(step / magnitude, unit) for unit in (1, 2, 5))
    assert bool(np.any(ticks == 0)) == (low <= 0 <= high)
    for value in ticks:
        assert float(_coordinate_text(value, step).replace(",", "")) == pytest.approx(value)


def test_paper_axes_keep_actual_coordinates_and_reuse_geometry_until_bounds_change():
    owner = Owner()
    owner.add()
    owner.set_environment_dimensions_style("paper_axes")
    assert getattr(owner, "_environment_dimensions_group", None) is None
    owner.set_environment_dimensions_visible(True)
    group = owner._environment_dimensions_group
    assert owner.get_environment_dimensions_state()["style"] == "paper_axes"
    assert isinstance(group.children[0], gfx.Line)
    assert len(group.children[0].geometry.positions.data) == 24  # Full 12-edge bounding frame.
    ticks = [
        label for label in owner._scene_guide_labels() if hasattr(label, "_orchav_coordinate_value")
    ]
    assert 12 <= len(ticks) <= 21
    for label in ticks:
        index = "XYZ".index(label._orchav_coordinate_axis)
        assert label.local.position[index] == label._orchav_coordinate_value
        assert label._orchav_coordinate_value != 0  # This environment is offset on all axes.
    owner._sync_scene_guides()
    assert owner._environment_dimensions_group is group
    owner.set_environment_dimensions_style("size_labels")
    assert group.parent is None
    assert len(owner._scene_guide_labels()) == 3
    with pytest.raises(ValueError, match="style"):
        owner.set_environment_dimensions_style("other")


def test_paper_label_decluttering_is_camera_cached_and_never_moves_coordinate_anchors():
    owner = Owner()
    owner.add(position=(0, 0, 0))
    owner._camera = gfx.PerspectiveCamera(60, 1, depth_range=(0.1, 500))
    owner._camera.local.position = (5, -60, 25)
    owner._camera.look_at((0, 0, 0))
    owner._canvas = SimpleNamespace(get_logical_size=lambda: (240, 240))
    owner.set_environment_dimensions_style("paper_axes")
    owner.set_environment_dimensions_visible(True)
    labels = owner._scene_guide_labels()
    positions = [tuple(label.local.position) for label in labels]
    signature = owner._paper_dimension_layout_signature
    assert any(label.visible for label in labels)
    assert any(not label.visible for label in labels)
    owner._sync_scene_guides()
    assert owner._paper_dimension_layout_signature == signature
    assert [tuple(label.local.position) for label in labels] == positions
    owner._camera.local.position = (25, -60, 30)
    owner._sync_scene_guides()
    assert owner._paper_dimension_layout_signature != signature
    assert [tuple(label.local.position) for label in labels] == positions


@pytest.mark.parametrize("width", [1000, 1200])
def test_city_overview_keeps_zero_ticks_separate_from_axis_captions(width):
    owner = Owner()
    owner.add(position=(5.7, -12.6, 29), size=(1070, 1171, 58))
    owner._camera = gfx.PerspectiveCamera(56, width / 900, depth_range=(0.1, 6000))
    owner._camera.world.reference_up = (0, 0, 1)
    owner._camera.local.position = (942.5, -1417.8, 1288.1)
    owner._camera.look_at((-41.14, -59.44, 14.5))
    owner._canvas = SimpleNamespace(get_logical_size=lambda: (width, 900))
    owner.set_environment_dimensions_style("paper_axes")
    owner.set_environment_dimensions_visible(True)
    zeros = {
        label._orchav_coordinate_axis: label
        for label in owner._scene_guide_labels()
        if getattr(label, "_orchav_coordinate_value", None) == 0
    }
    assert set(zeros) == set("XYZ")
    assert all(label.visible for label in zeros.values())


def test_guide_cards_have_only_background_and_border_and_reuse_geometry_on_orbit():
    owner = Owner()
    owner.add(position=(0, 0, 0))
    owner._camera = gfx.PerspectiveCamera(60, 1, depth_range=(0.1, 500))
    owner._camera.local.position = (5, -60, 25)
    owner._camera.look_at((0, 0, 0))
    owner._canvas = SimpleNamespace(get_logical_size=lambda: (800, 600))
    owner.set_environment_dimensions_visible(True)
    labels = owner._scene_guide_labels()
    cards = []
    for label in labels:
        assert label.material.weight_offset == 75
        assert len(label.children) == 1
        plate = label._orchav_guide_plate
        assert label.children[0] is plate
        assert len(plate.children) == 1 and isinstance(plate.children[0], gfx.Line)
        assert plate.children[0].material.thickness == 1
        cards.append((plate.geometry, plate.children[0].geometry))
    owner._camera.local.position = (25, -60, 30)
    owner._sync_scene_guides()
    assert cards == [
        (label._orchav_guide_plate.geometry, label._orchav_guide_plate.children[0].geometry)
        for label in labels
    ]
