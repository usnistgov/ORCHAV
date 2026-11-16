"""Draft camera integration with the real shared camera coordinator."""

from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PySide6.QtWidgets import QComboBox, QWidget

from tests.visualizer.unit.test_camera_presentation import rig  # noqa: F401
from visualizer.src.authoring.camera import BuilderCamera
from visualizer.src.authoring.viewport_port import ActorOverlaySnapshot, OverlaySnapshot


def test_builder_reuses_camera_focus_follow_pov_and_saved_views(rig, qapp):  # noqa: F811
    _tools, renderer, _position = rig
    workspace = QWidget()
    workspace.camera_mode_combo = QComboBox()
    workspace.camera_mode_combo.addItem("Orbit", "orbit")
    workspace.camera_mode_combo.addItem("Fly", "fly")
    port = SimpleNamespace(
        renderer=renderer,
        widget=workspace,
        hidden=None,
        set_camera_mode=lambda mode: True,
        set_pov_hidden_actor=lambda identity: setattr(port, "hidden", identity),
        preview_bounds=lambda **_: (np.zeros(3), np.ones(3) * 20),
    )
    camera = BuilderCamera(workspace, port)
    identity = uuid4()
    actor = ActorOverlaySnapshot(
        uuid4(),
        "target",
        "Moving car",
        ((0, 0, 0), (100, 0, 0)),
        current_position=(30, 0, 0),
        orientation_matrix=tuple(map(tuple, np.eye(4))),
    )
    workspace.document = SimpleNamespace(scenario=SimpleNamespace(actor=lambda _: actor))
    workspace._prepared_samples = lambda _: SimpleNamespace(positions=((0, 0, 0), (100, 0, 0)))
    workspace._play_step = 0
    try:
        camera.sync(OverlaySnapshot(identity, 0, actors=(actor,)))
        camera.select(actor.actor_id, focus=True)
        tools = camera.camera_controller.presentation()
        assert tools.transition is not None
        tools.tick(tools.duration)
        np.testing.assert_allclose(renderer.get_camera_state().lookat, (30, 0, 0), atol=1e-5)
        tools.save_view("Car")
        camera.ui_controller.handle_camera_mode_changed("follow", True)
        moved = replace(actor, current_position=(40, 0, 0))
        camera.sync(OverlaySnapshot(identity, 0, actors=(moved,)))
        tools.tick(tools.duration)
        np.testing.assert_allclose(renderer.get_camera_state().lookat, (40, 0, 0), atol=1e-5)
        camera.ui_controller.handle_camera_mode_changed("pov", True)
        assert camera.app_state.camera_mode == "pov"
        assert port.hidden == actor.actor_id
        tools.tick(tools.duration)
        np.testing.assert_allclose(renderer.get_camera_state().eye, (40, 0, 0), atol=1e-5)
        tools.input_event(SimpleNamespace(type="pointer_down"))
        assert tools.transition is None
        assert camera.app_state.camera_mode == "overview"
        assert port.hidden is None
        tools.recall_view("Car")
        tools.tick(tools.duration)
        np.testing.assert_allclose(renderer.get_camera_state().lookat, (30, 0, 0), atol=1e-5)
        camera.sync(OverlaySnapshot(uuid4(), 0, actors=()))
        assert tools.views == {} and tools.history == []
        assert camera.app_state.camera_mode == "overview"
    finally:
        camera.close_camera()
        workspace.close()
