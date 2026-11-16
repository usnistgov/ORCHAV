"""Direct volume hits keep native XYZ indexing and reject stale/drag gestures."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.renderers.pygfx.picking import PygfxPickingMixin
from visualizer.src.types.render_payloads import VolumePayload


@pytest.fixture
def picker():
    picker = PygfxPickingMixin()
    payload = VolumePayload(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        np.ones((2, 4)),
        (0, 23),
        mode="mip",
    )
    native = SimpleNamespace(_orchav_volume_payload=payload)
    picker._reverse_objects = {id(native): "sensing:outcome_volume"}
    picker._kinds = {"sensing:outcome_volume": "volume"}
    calls, registrations = [], []
    picker._renderer = SimpleNamespace(
        add_event_handler=lambda *args: registrations.append(args),
        remove_event_handler=lambda *args: registrations.remove(args),
    )
    picker.set_volume_cell_selection_callback(lambda *args: calls.append(args))
    return picker, native, calls, registrations


def event(native, kind, index=(3, 2, 1), **kwargs):
    return SimpleNamespace(
        type=kind,
        pick_info={"world_object": native, "index": index},
        target=native,
        x=kwargs.pop("x", 10),
        y=10,
        button=kwargs.pop("button", 1),
        modifiers=kwargs.pop("modifiers", ()),
        pointer_id=kwargs.pop("pointer_id", 1),
        **kwargs,
    )


def test_native_corner_index_is_xyz_and_source_stays_read_only(picker):
    p, native, calls, _ = picker
    for index in [(0, 0, 0), (3, 2, 1)]:
        p._route_volume_cell_event(event(native, "pointer_down", index))
        p._route_volume_cell_event(event(native, "pointer_up", index))
    assert calls == [("sensing:outcome_volume", i, "mip") for i in [(0, 0, 0), (3, 2, 1)]]
    assert not native._orchav_volume_payload.values.flags.writeable
    assert native._orchav_volume_payload.values[1, 2, 3] == 23


@pytest.mark.parametrize(
    "index", [(-1, 0, 0), (4, 0, 0), (0, 3, 0), (0, 0, 2), (1.2, 0, 0), (np.nan, 0, 0)]
)
def test_invalid_native_index_never_clamps_to_another_cell(picker, index):
    p, native, calls, _ = picker
    p._route_volume_cell_event(event(native, "pointer_down", index))
    p._route_volume_cell_event(event(native, "pointer_up", index))
    assert not calls


@pytest.mark.parametrize(
    "interrupt", ["drag", "leave", "new_payload", "new_cell", "modified", "long"]
)
def test_camera_or_stale_gesture_never_selects(picker, interrupt, monkeypatch):
    p, native, calls, _ = picker
    p._route_volume_cell_event(event(native, "pointer_down"))
    end = event(native, "pointer_up")
    if interrupt == "drag":
        p._route_volume_cell_event(event(native, "pointer_move", x=20))
    elif interrupt == "leave":
        p._route_volume_cell_event(event(native, "pointer_leave"))
    elif interrupt == "new_payload":
        native._orchav_volume_payload = replace(native._orchav_volume_payload, opacity=0.5)
    elif interrupt == "new_cell":
        end.pick_info["index"] = (2, 2, 1)
    elif interrupt == "modified":
        end.modifiers = ("Shift",)
    else:
        monkeypatch.setattr("visualizer.src.renderers.pygfx.picking.time.monotonic", lambda: 1e30)
    p._route_volume_cell_event(end)
    assert not calls


def test_disable_and_reenable_does_not_duplicate_handlers(picker):
    p, native, calls, registrations = picker
    p._route_volume_cell_event(event(native, "pointer_down"))
    p.set_volume_cell_selection_callback(None)
    p._route_volume_cell_event(event(native, "pointer_up"))
    p.set_volume_cell_selection_callback(lambda *args: calls.append(args))
    assert len(registrations) == 1 and not calls


def test_scoped_inspectors_do_not_unregister_each_other(picker):
    p, native, calls, _ = picker
    sensing, coverage = [], []
    p.set_volume_cell_selection_callback(None)
    p.set_volume_cell_selection_callback(
        lambda *args: sensing.append(args), object_ids=("sensing:outcome_volume",)
    )
    p.set_volume_cell_selection_callback(
        lambda *args: coverage.append(args), object_ids=("analysis:coverage_volume",)
    )
    p.set_volume_cell_selection_callback(None, object_ids=("analysis:coverage_volume",))
    p._route_volume_cell_event(event(native, "pointer_down"))
    p._route_volume_cell_event(event(native, "pointer_up"))
    assert len(sensing) == 1 and not coverage and not calls


def test_renderer_session_cleanup_drops_panel_references_and_rebinds(picker):
    p, native, calls, registrations = picker
    removed = []
    p._renderer.remove_event_handler = lambda *args: removed.append(args)
    p.clear_volume_cell_selection_callbacks()
    assert not p._volume_cell_selection_callbacks and len(removed) == 1
    p._renderer = SimpleNamespace(add_event_handler=lambda *args: registrations.append(args))
    p.set_volume_cell_selection_callback(lambda *args: calls.append(args))
    assert len(registrations) == 2
    p._route_volume_cell_event(event(native, "pointer_down"))
    p._route_volume_cell_event(event(native, "pointer_up"))
    assert len(calls) == 1


def test_pending_callbacks_follow_backend_replacement_and_retirement():
    picker = PygfxPickingMixin()
    active_a, active_b = [], []

    def backend(active):
        return SimpleNamespace(
            add_event_handler=lambda *args: active.append(args),
            remove_event_handler=lambda *args: active.remove(args),
        )

    picker.set_volume_cell_selection_callback(lambda *a: None, object_ids=("volume",))
    assert getattr(picker, "_volume_pick_event_backend", None) is None
    picker._renderer = backend(active_a)
    picker._bind_volume_pick_events()
    picker._bind_volume_pick_events()
    assert len(active_a) == 1
    picker._renderer = backend(active_b)
    picker._bind_volume_pick_events()
    assert not active_a and len(active_b) == 1
    picker.set_volume_cell_selection_callback(None, object_ids=("volume",))
    assert not active_a and not active_b
    picker._renderer = backend(active_a)
    picker._bind_volume_pick_events()
    assert not active_a
