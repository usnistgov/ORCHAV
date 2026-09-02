from types import SimpleNamespace

from visualizer.src.renderers.pygfx.picking import PygfxPickingMixin
from visualizer.src.state import AppState, create_initial_state


class _HoverProbe(PygfxPickingMixin):
    def __init__(self):
        self._hover_info_mode = "essential"
        self._reverse_objects = {}
        self._pick_metadata = {}
        self._materials = {}
        self._positions = {}
        self._last_hover_identity = None
        self.last_frame_packet = None
        self.hidden = 0
        self.shown = []
        self.repositioned = 0
        self.visualizer = SimpleNamespace()

    def _hide_tooltip(self):
        self.hidden += 1
        self._last_hover_identity = None

    def _show_tooltip(self, text, event):
        self.shown.append(text)

    def _reposition_tooltip(self, event):
        self.repositioned += 1


def _event_for(target, pick_info=None):
    return SimpleNamespace(target=target, x=10, y=20, pick_info=pick_info or {})


def test_app_state_defaults_to_essential_hover_info():
    state = create_initial_state()

    assert state.hover_info_mode == "essential"
    assert AppState.from_dict(state.to_dict()).hover_info_mode == "essential"
    assert AppState.from_dict({**state.to_dict(), "hover_info_mode": "all"}).hover_info_mode == (
        "inspect_all"
    )


def test_essential_hover_suppresses_scene_meshes_but_keeps_tx():
    probe = _HoverProbe()
    scene_obj = object()
    tx_obj = object()
    probe._reverse_objects[id(scene_obj)] = "scene:merged_abcd::mesh"
    probe._reverse_objects[id(tx_obj)] = "node:tx_0::marker"
    probe._positions["node:tx_0::marker"] = (1.0, 2.0, 3.0)

    probe._on_pointer_move(_event_for(scene_obj))
    probe._on_pointer_move(_event_for(tx_obj))

    assert probe.shown == ["TX1\n(1.0, 2.0, 3.0)"]
    assert probe.hidden == 1


def test_off_hover_suppresses_all_tooltips():
    probe = _HoverProbe()
    mpc_obj = object()
    probe._reverse_objects[id(mpc_obj)] = "mpc_lines"
    probe.set_hover_info_mode("off")

    probe._on_pointer_move(_event_for(mpc_obj))

    assert probe.shown == []
    assert probe.hidden >= 1


def test_inspect_all_hover_formats_merged_scene_group():
    probe = _HoverProbe()
    scene_obj = object()
    probe._reverse_objects[id(scene_obj)] = "scene:merged_abcd::mesh"
    probe._materials["scene:merged_abcd::mesh"] = {"material_name": "itu_concrete"}
    probe._pick_metadata["scene:merged_abcd::mesh"] = {
        "type": "scene_merged",
        "mesh_count": 3,
        "mesh_ids": [1, 2, 3],
    }
    probe.set_hover_info_mode("inspect_all")

    probe._on_pointer_move(_event_for(scene_obj))

    assert probe.shown == ["Scene group (3 meshes)\nMaterial: itu_concrete"]


def test_unified_mpc_line_hover_updates_between_segments_and_reuses_one_segment():
    probe = _HoverProbe()
    mpc_obj = object()
    probe._reverse_objects[id(mpc_obj)] = "mpc_lines"

    probe._on_pointer_move(_event_for(mpc_obj, {"vertex_index": 0}))
    probe._on_pointer_move(_event_for(mpc_obj, {"vertex_index": 2}))
    probe._on_pointer_move(_event_for(mpc_obj, {"vertex_index": 3}))

    assert probe.shown == ["MPC segment #0", "MPC segment #1"]
    assert probe.repositioned == 1


def test_mpc_point_hover_uses_direct_point_index_and_point_metadata():
    probe = _HoverProbe()
    point_obj = object()
    probe._reverse_objects[id(point_obj)] = "mpc_points"
    probe.last_frame_packet = SimpleNamespace(
        mpc_bounce_itypes=[1, 8],
    )

    probe._on_pointer_move(_event_for(point_obj, {"vertex_index": 0}))
    probe._on_pointer_move(_event_for(point_obj, {"vertex_index": 1}))

    assert probe._pick_metadata["mpc_points"] == {"type": "mpc_points"}
    assert probe.shown == [
        "Specular | interaction point #0",
        "Diffraction | interaction point #1",
    ]
