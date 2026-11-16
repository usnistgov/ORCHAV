"""Cutaway contours share static geometry and follow the existing plane contract."""

from unittest.mock import Mock

import numpy as np
import pygfx as gfx
import pytest

from visualizer.src.renderers.pygfx.section_contours import (
    PygfxSectionContoursMixin,
    _crosses_plane,
)


class Owner(PygfxSectionContoursMixin):
    def __init__(self):
        self._gfx = gfx
        self._scene = gfx.Scene()
        self._static_group = gfx.Group()
        self._scene.add(self._static_group)
        self._reverse_objects = {}
        self._clipping_planes = ((0, 0, -1, -2),)
        self.request_redraw = Mock()

    def _is_scene_mesh_name(self, name):
        return name.startswith("scene_")

    def add(self, name, z=2):
        source = gfx.Mesh(gfx.box_geometry(4, 4, 4), gfx.MeshBasicMaterial())
        source.local.z = z
        self._static_group.add(source)
        self._reverse_objects[id(source)] = name
        return source


def test_contours_share_only_intersecting_scene_meshes_and_convert_plane_sign():
    owner = Owner()
    building = owner.add("scene_building")
    owner.add("scene_remote", z=30)
    owner.add("target_car")
    vertices = building.geometry.positions.data.copy()
    owner.set_section_contours(True)
    assert owner.get_section_contour_state()["mesh_count"] == 1
    contour = next(iter(owner._section_contours.values()))
    assert contour.geometry is building.geometry
    assert contour.material.plane == (0, 0, -1, 2)
    assert not contour.material.pick_write and not contour.material.depth_write
    assert not contour.cast_shadow and not contour.receive_shadow
    np.testing.assert_array_equal(building.geometry.positions.data, vertices)
    np.testing.assert_allclose(contour.world.matrix, building.world.matrix)


def test_unchanged_draw_reuses_contours_without_rechecking_bounds(monkeypatch):
    owner = Owner()
    source = owner.add("scene_building")
    owner.set_section_contours(True)
    owner.request_redraw.reset_mock()
    owner.set_section_contours(True)
    owner.request_redraw.assert_not_called()
    first = next(iter(owner._section_contours.values()))
    bounds = Mock(side_effect=AssertionError("unchanged scene recomputed bounds"))
    monkeypatch.setattr(source, "get_world_bounding_box", bounds)
    for _ in range(4):
        owner._sync_section_contours()
    assert next(iter(owner._section_contours.values())) is first
    bounds.assert_not_called()


def test_other_cutaway_planes_clip_contours_and_controls_update_in_place():
    owner = Owner()
    owner.add("scene_building")
    owner._clipping_planes += ((1, 0, 0, 0),)
    owner.set_section_contours(True, width=3)
    assert len(owner._section_contours) == 2
    by_index = {key[1]: obj for key, obj in owner._section_contours.items()}
    assert by_index[0].material.clipping_planes == [(1, 0, 0, 0)]
    assert by_index[1].material.clipping_planes == [(0, 0, -1, -2)]
    owner.set_section_contours(True, width=4)
    assert by_index[0].material.thickness == 4
    owner._clipping_planes = ((0, 0, 1, 2),)
    owner._sync_section_contours()
    assert len(owner._section_contours) == 1
    assert by_index[0].material.plane == (0, 0, 1, -2)
    assert not by_index[0].material.clipping_planes
    assert by_index[1].parent is None


def test_geometry_pose_and_visibility_changes_reconcile_without_stale_lines():
    owner = Owner()
    source = owner.add("scene_building")
    owner.set_section_contours(True)
    source.geometry = gfx.box_geometry(2, 2, 6)
    source.local.x = 8
    owner._sync_section_contours()
    contour = next(iter(owner._section_contours.values()))
    assert contour.geometry is source.geometry
    np.testing.assert_allclose(contour.world.matrix, source.world.matrix)
    source.local.z = 30
    owner._sync_section_contours()
    assert not owner._section_contours
    source.local.z = 2
    owner._sync_section_contours()
    assert len(owner._section_contours) == 1
    owner._static_group.visible = False
    owner._sync_section_contours()
    assert not owner._section_contours
    owner._static_group.visible = True
    owner._sync_section_contours()
    # Consume the initial upload, as a rendered scene does. Native buffers
    # coalesce edits while a full upload is already pending.
    source.geometry.positions._gfx_get_chunk_descriptions()
    source.geometry.positions.data[:, 2] += 20
    source.geometry.positions.update_full()
    owner._sync_section_contours()
    assert not owner._section_contours


def test_off_reset_and_removed_planes_release_optional_objects():
    owner = Owner()
    owner.add("scene_building")
    owner._sync_section_contours()
    assert len(owner._scene.children) == 1
    owner.set_section_contours(True)
    group = owner._section_contour_group
    owner.set_section_contours(False)
    assert group.parent is None and owner.get_section_contour_state()["mesh_count"] == 0
    owner.set_section_contours(True)
    owner._clipping_planes = ()
    owner._sync_section_contours()
    assert owner._section_contour_group is None
    assert owner.get_section_contour_state()["enabled"]
    owner._clear_section_contours()
    assert not owner.get_section_contour_state()["enabled"]


def test_flat_coplanar_mesh_and_invalid_width_are_rejected():
    assert not _crosses_plane([[0, 0, 0], [10, 10, 0]], (0, 0, 1, 0))
    assert _crosses_plane([[0, 0, 0], [10, 10, 8]], (0, 0, 1, 4))
    assert not _crosses_plane([[0, 0, 0], [10, 10, 8]], (0, 0, 1, 14))
    owner = Owner()
    for width in (0, 7, np.nan):
        with pytest.raises(ValueError, match="width"):
            owner.set_section_contours(True, width=width)
