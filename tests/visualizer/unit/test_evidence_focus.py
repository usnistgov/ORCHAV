"""Presentation-only focus never changes physical geometry or scientific data."""

from types import SimpleNamespace as NS

import pytest

from visualizer.src.services.evidence_focus import EvidenceFocus


class Renderer:
    renderer_id = "pygfx"

    def __init__(self):
        self._objects = {}
        self.hud = {}
        self.redraws = 0

    def request_redraw(self):
        self.redraws += 1

    def set_hud_overlay(self, name, **kwargs):
        self.hud[name] = kwargs

    def clear_hud_overlay(self, name):
        self.hud.pop(name, None)


def setup():
    renderer = Renderer()
    objects = {
        name: NS(material=NS(opacity=0.8), visible=True)
        for name in (
            "mpc_lines",
            "mpc_points",
            "sensing_detections",
            "building",
            "sensing_track_point_7",
            "sensing_track_point_8",
            "sensing_uncertainty_7",
            "sensing_selection_points",
        )
    }
    renderer._objects = objects
    viz = NS(renderer=renderer, animation_running=False)
    return EvidenceFocus(viz), renderer, objects


def test_background_dims_only_and_restore_is_exact():
    focus, renderer, objects = setup()
    assert focus.start(identity=(2, 63), selected_track_id=7, result_time_s=1.25)
    for name in ("mpc_lines", "mpc_points", "sensing_detections", "sensing_track_point_8"):
        assert objects[name].material.opacity == pytest.approx(0.12)
    for name in (
        "building",
        "sensing_track_point_7",
        "sensing_uncertainty_7",
        "sensing_selection_points",
    ):
        assert objects[name].material.opacity == 0.8
    assert "1.25 s" in renderer.hud[focus.HUD]["html"]
    focus.stop()
    assert all(obj.material.opacity == 0.8 for obj in objects.values())
    assert all(obj.visible for obj in objects.values())
    assert not renderer.hud
    redraws = renderer.redraws
    focus.stop()
    focus.refresh((2, 63))
    assert renderer.redraws == redraws


def test_shared_physical_material_is_never_modified():
    focus, _, objects = setup()
    objects["mpc_lines"].material = objects["building"].material
    focus.start(identity=(2, 63))
    assert objects["building"].material.opacity == 0.8


def test_later_user_appearance_edit_survives_restore():
    focus, _, objects = setup()
    focus.start(identity=(2, 63))
    objects["mpc_lines"].material.opacity = 0.6
    focus.stop()
    assert objects["mpc_lines"].material.opacity == 0.6


@pytest.mark.parametrize("change", ["frame", "source", "renderer", "play", "close"])
def test_context_changes_restore_and_stop(change):
    focus, renderer, objects = setup()
    focus.start(identity=(2, 63))
    identity = (2, 63)
    if change == "frame":
        identity = (2, 64)
    elif change == "source":
        identity = (3, 63)
    elif change == "renderer":
        focus.viz.renderer = Renderer()
    elif change == "play":
        focus.viz.animation_running = True
    else:
        renderer._qt_window_closed = True
    focus.refresh(identity)
    assert not focus.active
    assert objects["mpc_lines"].material.opacity == 0.8


def test_no_enable_while_playing_or_unknown_frame():
    focus, renderer, _ = setup()
    assert not focus.start(identity=(2, None))
    focus.viz.animation_running = True
    assert not focus.start(identity=(2, 63))
    assert renderer.redraws == 0


def test_repeated_refresh_does_not_compound_dimming():
    focus, renderer, objects = setup()
    focus.start(identity=(2, 63))
    redraws = renderer.redraws
    for _ in range(5):
        focus.refresh((2, 63))
    assert renderer.redraws == redraws
    assert objects["mpc_lines"].material.opacity == pytest.approx(0.12)
    objects["mpc_lines"] = NS(material=NS(opacity=0.5), visible=True)
    focus.refresh((2, 63))
    assert objects["mpc_lines"].material.opacity == pytest.approx(0.075)
    focus.stop()
    assert objects["mpc_lines"].material.opacity == 0.5


def test_native_material_restore_and_shared_protection():
    import numpy as np
    import pygfx as gfx

    focus, renderer, _ = setup()
    material = gfx.LineMaterial(color=(1, 0, 0, 0.8))
    route = gfx.Line(
        gfx.Geometry(positions=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)), material
    )
    renderer._objects = {"mpc_lines": route}
    original = material.opacity
    focus.start(identity=(2, 63))
    assert material.opacity == pytest.approx(original * focus.DIM)
    focus.stop()
    assert material.opacity == pytest.approx(original)


def test_controls_playback_and_deleted_control_cleanup(qtbot):
    from PySide6.QtWidgets import QCheckBox
    from shiboken6 import delete

    from visualizer.src.services.paused_presentation import playback_changed

    focus, renderer, objects = setup()
    checkbox = QCheckBox()
    focus.control = checkbox
    viz = focus.viz
    viz._evidence_focus = focus
    viz.ui_manager = NS(panels={"sensing": NS(widgets={"evidence_focus": checkbox})})
    focus.start(identity=(2, 63))
    assert checkbox.isChecked()
    playback_changed(viz, True)
    assert not checkbox.isChecked() and not checkbox.isEnabled()
    assert objects["mpc_lines"].material.opacity == 0.8
    playback_changed(viz, False)
    assert checkbox.isEnabled() and not checkbox.isChecked()
    delete(checkbox)
    focus.stop()
    playback_changed(viz, True)


def test_native_focus_renders_restores_and_renderer_close(qapp):
    import os

    import numpy as np

    from visualizer.src.model import RenderObject
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.types.render_payloads import LineSetPayload, MaterialPayload

    if os.getenv("ORCHAV_RUN_PYGFX_TESTS") != "1":
        pytest.skip("Set ORCHAV_RUN_PYGFX_TESTS=1 for native renderer")
    viz = NS(animation_running=False)
    renderer = PygfxRenderer(viz)
    viz.renderer = renderer
    focus = viz._evidence_focus = EvidenceFocus(viz)
    try:
        renderer.initialize_visualizer(width=128, height=128)
        renderer.ensure_object(
            RenderObject(
                "mpc_lines",
                LineSetPayload(
                    points=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                    lines=np.array([[0, 1]], dtype=np.int32),
                ),
                MaterialPayload(base_color=(1, 0, 0, 1)),
            )
        )
        material = renderer._objects["mpc_lines"].material
        original = material.opacity
        renderer._container.resize(640, 480)
        renderer.set_hud_overlay(
            "cinematic_sensing_reveal",
            html="ANGLE — recorded bin",
            visible=True,
            role="status",
            corner="top_left",
            priority=20,
        )
        focus.start(identity=(2, 63), result_time_s=1.25)
        renderer._reposition_hud_overlays()
        reveal_badge = renderer._hud_overlay_labels["cinematic_sensing_reveal"]
        focus_badge = renderer._hud_overlay_labels[focus.HUD]
        assert not reveal_badge.geometry().intersects(focus_badge.geometry())
        assert focus_badge.y() > reveal_badge.y()
        renderer._renderer.render(renderer._scene, renderer._camera)
        assert renderer._renderer.snapshot().shape[-1] == 4
        assert material.opacity == pytest.approx(original * focus.DIM)
        renderer.close()
        assert not focus.active
        assert material.opacity == pytest.approx(original)
    finally:
        renderer.close()


def test_other_covariance_dimmed_without_track_id_prefix_collision():
    focus, renderer, _ = setup()
    names = (
        "sensing_uncertainty_7",
        "sensing_uncertainty_7_outline",
        "sensing_uncertainty_70_center",
    )
    renderer._objects = {name: NS(material=NS(opacity=0.5)) for name in names}
    focus.start(identity=(2, 63), selected_track_id=7)
    assert renderer._objects[names[0]].material.opacity == 0.5
    assert renderer._objects[names[1]].material.opacity == 0.5
    assert renderer._objects[names[2]].material.opacity == pytest.approx(0.075)
    focus.stop()
    assert all(obj.material.opacity == 0.5 for obj in renderer._objects.values())


def test_replaced_result_identity_stops_focus():
    focus, _, objects = setup()
    focus.start(identity=(2, 63, 11))
    focus.refresh((2, 63, 12))
    assert not focus.active
    assert objects["mpc_lines"].material.opacity == 0.8


def test_selected_track_requires_exact_native_source_scope_and_acquisition():
    from dataclasses import replace

    from tests.visualizer.unit.test_tracking_uncertainty_scene import frame
    from visualizer.src.sensing.selection import SensingSelection
    from visualizer.src.services.evidence_focus import selected_focus_track

    request = NS(source_identity=("provider",), effective_display_step=63, display_frame=frame())
    selection = SensingSelection(
        ("clock", "run", "pair"), "a", track_id=7, detection_index=0, source=("provider",)
    )
    assert selected_focus_track(selection, request, ("provider",), 63) == 7
    for changed in (
        replace(selection, source=("other",)),
        replace(selection, scope=("clock", "other", "pair")),
        replace(selection, acquisition_id="old"),
    ):
        assert selected_focus_track(changed, request, ("provider",), 63) is None
    assert selected_focus_track(selection, request, ("provider",), 64) is None


def test_selected_fused_track_requires_verified_source_run_and_exact_epoch():
    from dataclasses import replace

    from visualizer.src.sensing.fusion_inspection import FusionSelection
    from visualizer.src.services.evidence_focus import selected_focus_track

    class Reader(dict):
        inspection_id = "run"

    reader = Reader(sources={"sensor": {"lineage": [{"scene_frame_id": 63}]}})
    scene = NS(
        source_key=("provider", "result-path"), binding=NS(reader=reader, sensor_id="sensor")
    )
    selection = FusionSelection("run", 7, 0)
    assert selected_focus_track(selection, None, ("provider",), 63, scene) == 7
    assert (
        selected_focus_track(replace(selection, run_id="old"), None, ("provider",), 63, scene)
        is None
    )
    assert selected_focus_track(selection, None, ("other",), 63, scene) is None
    assert selected_focus_track(selection, None, ("provider",), 64, scene) is None


def test_pending_preparation_stops_focus_without_frame_reads(monkeypatch):
    from visualizer.src.sensing import runtime

    focus, renderer, objects = setup()
    viz = focus.viz
    viz._evidence_focus = focus
    focus.start(identity=(0, 63, id(None)))
    monkeypatch.setattr(runtime, "refresh_tracking_uncertainty", lambda *args, **kwargs: None)
    runtime._sync_frame(viz, NS(preparation_pending=True), {}, 63)
    assert not focus.active
    assert objects["mpc_lines"].material.opacity == 0.8
