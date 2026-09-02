"""Tests for viewport MPC click selection and canonical pick mapping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visualizer.src.renderers.pygfx import picking as picking_module
from visualizer.src.renderers.pygfx.picking import PygfxPickingMixin


class _PickingProbe(PygfxPickingMixin):
    def __init__(self) -> None:
        self.target = object()
        self._reverse_objects = {id(self.target): "mpc_lines"}
        self._mpc_path_selection_callback = None
        self._mpc_path_click_candidate = None
        self._mpc_pick_segment_map_packet = None
        self._mpc_pick_canonical_segment_indices = np.empty((0,), dtype=np.int32)
        self._mpc_pick_identity_mapping = False
        canon = SimpleNamespace(
            segment_path_id=np.asarray((0, 1, 2), dtype=np.int32),
            segment_start_indices=np.asarray((0, 2, 4), dtype=np.int32),
            path_id=np.asarray((0, 0, 1, 1, 2, 2), dtype=np.int32),
        )
        self.last_frame_packet = SimpleNamespace(
            canonical_data=canon,
            mpc_lines=np.asarray(((2, 3),), dtype=np.int32),
            segment_mask=np.asarray((False, True, False), dtype=bool),
        )

    def event(
        self,
        event_type: str,
        *,
        x: float = 10.0,
        y: float = 20.0,
        target=None,
        button: int = 1,
        modifiers=(),
        pointer_id: int = 4,
    ):
        return SimpleNamespace(
            type=event_type,
            x=x,
            y=y,
            target=self.target if target is None else target,
            button=button,
            modifiers=modifiers,
            pointer_id=pointer_id,
            pick_info={"vertex_index": 0},
        )


def test_disabled_viewport_selection_does_not_inspect_event_or_frame() -> None:
    probe = _PickingProbe()

    class _UnreadableEvent:
        @property
        def type(self):
            raise AssertionError("disabled route inspected the event")

    probe.route_mpc_path_selection_event(_UnreadableEvent())

    assert probe._mpc_path_click_candidate is None
    assert probe._mpc_pick_segment_map_packet is None


def test_short_click_publishes_filtered_segment_canonical_path(monkeypatch) -> None:
    probe = _PickingProbe()
    selected = []
    clock = iter((10.0, 10.2))
    monkeypatch.setattr(picking_module.time, "monotonic", lambda: next(clock))
    probe.set_mpc_path_selection_callback(
        lambda path_id, packet_identity: selected.append((path_id, packet_identity))
    )
    mapping = np.asarray((1,), dtype=np.int32)
    assert probe.set_mpc_pick_segment_mapping(
        id(probe.last_frame_packet),
        mapping,
    )

    probe.route_mpc_path_selection_event(probe.event("pointer_down"))
    probe.route_mpc_path_selection_event(probe.event("pointer_up", x=12.0, y=21.0))

    assert selected == [(1, id(probe.last_frame_packet))]
    assert probe._mpc_path_click_candidate is None
    assert probe._mpc_pick_segment_map_packet is probe.last_frame_packet
    assert probe._mpc_pick_canonical_segment_indices is mapping
    assert probe._mpc_pick_canonical_segment_indices.dtype == np.int32
    assert np.shares_memory(probe._mpc_pick_canonical_segment_indices, mapping)
    np.testing.assert_array_equal(probe._mpc_pick_canonical_segment_indices, (1,))


def test_mapping_install_rejects_inputs_that_would_require_a_gui_thread_copy() -> None:
    probe = _PickingProbe()

    assert not probe.set_mpc_pick_segment_mapping(
        id(probe.last_frame_packet),
        [1],
    )
    assert not probe.set_mpc_pick_segment_mapping(
        id(probe.last_frame_packet),
        np.asarray((1,), dtype=np.int64),
    )
    noncontiguous = np.arange(4, dtype=np.int32)[::2]
    assert not noncontiguous.flags.c_contiguous
    probe.last_frame_packet.mpc_lines = np.zeros((2, 2), dtype=np.int32)
    assert not probe.set_mpc_pick_segment_mapping(
        id(probe.last_frame_packet),
        noncontiguous,
    )
    assert probe._mpc_pick_segment_map_packet is None


def test_camera_drag_does_not_select_path(monkeypatch) -> None:
    probe = _PickingProbe()
    selected = []
    monkeypatch.setattr(picking_module.time, "monotonic", lambda: 1.0)
    probe.set_mpc_path_selection_callback(
        lambda path_id, packet_identity: selected.append((path_id, packet_identity))
    )
    assert probe.set_mpc_pick_segment_mapping(
        id(probe.last_frame_packet),
        np.asarray((1,), dtype=np.int32),
    )

    probe.route_mpc_path_selection_event(probe.event("pointer_down"))
    probe.route_mpc_path_selection_event(probe.event("pointer_move", x=30.0, y=20.0))
    probe.route_mpc_path_selection_event(probe.event("pointer_up", x=30.0, y=20.0))

    assert selected == []
    assert probe._mpc_path_click_candidate is None


def test_long_modified_and_mismatched_clicks_are_rejected(monkeypatch) -> None:
    probe = _PickingProbe()
    selected = []
    clock = iter((2.0, 3.0))
    monkeypatch.setattr(picking_module.time, "monotonic", lambda: next(clock))
    probe.set_mpc_path_selection_callback(
        lambda path_id, packet_identity: selected.append((path_id, packet_identity))
    )
    assert probe.set_mpc_pick_segment_mapping(
        id(probe.last_frame_packet),
        np.asarray((1,), dtype=np.int32),
    )

    probe.route_mpc_path_selection_event(probe.event("pointer_down"))
    probe.route_mpc_path_selection_event(probe.event("pointer_up"))
    probe.route_mpc_path_selection_event(probe.event("pointer_down", modifiers=("shift",)))
    probe.route_mpc_path_selection_event(probe.event("pointer_up", pointer_id=99))

    assert selected == []


def test_disabling_selection_clears_gesture_and_releases_pick_cache(monkeypatch) -> None:
    probe = _PickingProbe()
    monkeypatch.setattr(picking_module.time, "monotonic", lambda: 1.0)
    probe.set_mpc_path_selection_callback(lambda _path_id, _packet_identity: None)
    assert probe.set_mpc_pick_segment_mapping(
        id(probe.last_frame_packet),
        np.asarray((1,), dtype=np.int32),
    )
    probe.route_mpc_path_selection_event(probe.event("pointer_down"))

    assert probe._mpc_path_click_candidate is not None
    assert probe._mpc_pick_segment_map_packet is probe.last_frame_packet

    probe.set_mpc_path_selection_callback(None)

    assert probe._mpc_path_click_candidate is None
    assert probe._mpc_pick_segment_map_packet is None
    assert probe._mpc_pick_canonical_segment_indices.size == 0


def test_packet_change_during_click_rejects_stale_mapping(monkeypatch) -> None:
    probe = _PickingProbe()
    selected = []
    monkeypatch.setattr(picking_module.time, "monotonic", lambda: 1.0)
    probe.set_mpc_path_selection_callback(
        lambda path_id, packet_identity: selected.append((path_id, packet_identity))
    )
    assert probe.set_mpc_pick_segment_mapping(
        id(probe.last_frame_packet),
        np.asarray((1,), dtype=np.int32),
    )

    probe.route_mpc_path_selection_event(probe.event("pointer_down"))
    old_packet = probe.last_frame_packet
    probe.last_frame_packet = SimpleNamespace(**vars(old_packet))
    probe.route_mpc_path_selection_event(probe.event("pointer_up"))

    assert selected == []


def test_click_waits_for_worker_prepared_mapping_without_cold_segment_scan(
    monkeypatch,
) -> None:
    probe = _PickingProbe()
    selected = []
    monkeypatch.setattr(picking_module.time, "monotonic", lambda: 1.0)
    probe.set_mpc_path_selection_callback(
        lambda path_id, packet_identity: selected.append((path_id, packet_identity))
    )

    probe.route_mpc_path_selection_event(probe.event("pointer_down"))
    probe.route_mpc_path_selection_event(probe.event("pointer_up"))

    assert selected == []
    assert probe._mpc_path_click_candidate is None
    assert probe._mpc_pick_segment_map_packet is None


def test_worker_can_install_identity_mapping_without_allocating_an_arange(
    monkeypatch,
) -> None:
    probe = _PickingProbe()
    probe.last_frame_packet.segment_mask = np.ones((1,), dtype=bool)
    selected = []
    monkeypatch.setattr(picking_module.time, "monotonic", lambda: 1.0)
    probe.set_mpc_path_selection_callback(
        lambda path_id, packet_identity: selected.append((path_id, packet_identity))
    )

    assert probe.set_mpc_pick_segment_mapping(id(probe.last_frame_packet), None)
    probe.route_mpc_path_selection_event(probe.event("pointer_down"))
    probe.route_mpc_path_selection_event(probe.event("pointer_up"))

    assert selected == [(0, id(probe.last_frame_packet))]
    assert probe._mpc_pick_canonical_segment_indices.size == 0
