"""Selected actor shells follow geometry without owning extra frame buffers."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pygfx as gfx

from visualizer.src.panels.nodes_panel import NodesSelectionPanel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.renderers.pygfx.target_silhouette import PygfxTargetSilhouetteMixin


class Owner(PygfxTargetSilhouetteMixin):
    def __init__(self):
        self._gfx = gfx
        self._scene = gfx.Scene()
        self._objects = {}
        self.request_redraw = Mock()


def test_shell_shares_buffers_and_tracks_pose_replacement_visibility(monkeypatch):
    monkeypatch.setattr(
        "visualizer.src.renderers.pygfx.target_silhouette.silhouette_material_class",
        lambda: lambda: gfx.MeshBasicMaterial(pick_write=False, depth_write=False),
    )
    owner = Owner()
    source = gfx.Mesh(gfx.box_geometry(), gfx.MeshBasicMaterial())
    owner._objects["car"] = source
    owner._scene.add(source)
    owner.set_target_silhouette("car")
    shell = owner._target_silhouette
    assert shell.geometry is source.geometry
    assert not shell.cast_shadow and not shell.material.pick_write
    assert not shell.material.depth_write

    source.geometry = gfx.sphere_geometry()
    source.local.position = (2, 3, 4)
    owner._sync_target_silhouette()
    assert owner._target_silhouette is shell
    assert shell.geometry is source.geometry
    np.testing.assert_allclose(shell.world.matrix, source.world.matrix)

    source.visible = False
    owner._sync_target_silhouette()
    assert not shell.visible
    source.visible = True
    source.material.wireframe = True
    owner._sync_target_silhouette()
    assert not shell.visible
    source.material.wireframe = False
    owner._sync_target_silhouette()
    assert shell.visible
    source.material.opacity = 0.4
    owner._sync_target_silhouette()
    assert not shell.visible
    source.material.opacity = 1.0
    source.material.depth_write = False
    owner._sync_target_silhouette()
    assert not shell.visible
    source.material.depth_write = True
    owner._sync_target_silhouette()
    assert shell.visible

    owner.set_target_silhouette(None)
    assert shell.parent is None
    assert owner._target_silhouette is None
    assert len(owner._scene.children) == 1


def test_hidden_parent_missing_target_and_clear_do_not_leave_a_shell(monkeypatch):
    monkeypatch.setattr(
        "visualizer.src.renderers.pygfx.target_silhouette.silhouette_material_class",
        lambda: gfx.MeshBasicMaterial,
    )
    owner = Owner()
    group = gfx.Group()
    source = gfx.Mesh(gfx.box_geometry(), gfx.MeshBasicMaterial())
    group.add(source)
    owner._scene.add(group)
    owner._objects["car"] = source
    owner.set_target_silhouette("car")
    group.visible = False
    owner._sync_target_silhouette()
    assert not owner._target_silhouette.visible
    owner.set_target_silhouette("missing")
    assert owner._target_silhouette is None
    group.visible = True
    owner.set_target_silhouette("car")
    owner._clear_target_silhouette()
    assert owner._target_silhouette_id is None
    assert owner._target_silhouette is None
    assert len(owner._scene.children) == 1


def test_target_selector_moves_emphasis_and_disabling_clears_it(qtbot):
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(target_silhouette=True),
        set_target_silhouette=Mock(),
    )
    viz = SimpleNamespace(
        renderer=renderer,
        target_entries=[
            {"name": "Car", "mesh": SimpleNamespace(id="target::Car::mesh")},
            {"name": "Drone", "mesh": SimpleNamespace(id="target::Drone::mesh")},
        ],
    )
    panel = NodesSelectionPanel(viz)
    group = panel._create_target_group()
    qtbot.addWidget(group)
    panel.populate_target_dropdown()
    toggle = panel.widgets["target_silhouette_cb"]
    toggle.setChecked(True)
    renderer.set_target_silhouette.assert_called_with("target::Car::mesh")
    panel.widgets["target_dropdown"].setCurrentIndex(1)
    renderer.set_target_silhouette.assert_called_with("target::Drone::mesh")
    toggle.setChecked(False)
    renderer.set_target_silhouette.assert_called_with(None)


def test_closed_window_releases_shell_before_renderer_reuse(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setattr(
        "visualizer.src.renderers.pygfx.target_silhouette.silhouette_material_class",
        lambda: gfx.MeshBasicMaterial,
    )
    renderer = PygfxRenderer(visualizer=None)
    renderer._gfx = gfx
    renderer._scene = gfx.Scene()
    source = gfx.Mesh(gfx.box_geometry(), gfx.MeshBasicMaterial())
    renderer._objects["car"] = source
    renderer._scene.add(source)
    renderer.set_target_silhouette("car")
    shell = renderer._target_silhouette
    renderer._qt_window_closed = True
    renderer.close()
    assert shell.parent is None
    assert renderer._target_silhouette is None
    assert renderer._target_silhouette_id is None
