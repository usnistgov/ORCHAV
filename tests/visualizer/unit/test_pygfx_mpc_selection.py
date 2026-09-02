"""Native-object tests for the pygfx selected-MPC overlay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

gfx = pytest.importorskip("pygfx")

from visualizer.src.renderers.mpc_path_inspection import MpcPathInspectionSnapshot
from visualizer.src.renderers.pygfx.mpc import PygfxMpcMixin
from visualizer.src.renderers.pygfx.mpc_selection import (
    MPC_SELECTION_BOUNCE_POINTS_NAME,
    MPC_SELECTION_DIRECTION_NAME,
    MPC_SELECTION_HALO_NAME,
    MPC_SELECTION_PATH_NAME,
    MPC_SELECTION_PREFIX,
    MPC_SELECTION_PULSE_NAME,
    PygfxMpcSelectionMixin,
)
from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
from visualizer.src.renderers.pygfx.runtime import PygfxRuntimeMixin


class _SelectionRenderer(PygfxMpcSelectionMixin):
    def __init__(self) -> None:
        self._gfx = gfx
        self._initialized = True
        self._qt_window_closed = False
        self._canvas = SimpleNamespace(_orchav_closed=False, _is_closed=False)
        self._scene = gfx.Scene()
        self._objects = {}
        self._name_to_handle = {}
        self._handle_to_name = {}
        self._kinds = {}
        self._topology = {}
        self._reverse_objects = {}
        self._pick_metadata = {}
        self._next_handle = 1
        self.remove_failures: dict[str, int] = {}
        self.redraw_count = 0
        self._initialize_mpc_path_inspection_state()

    def _allocate_handle(self) -> int:
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def request_redraw(self) -> None:
        self.redraw_count += 1

    def remove_named_geometry(self, name: str) -> bool:
        failures = self.remove_failures.get(name, 0)
        if failures > 0:
            self.remove_failures[name] = failures - 1
            return False
        obj = self._objects.get(name)
        if obj is None:
            return False
        self._scene.remove(obj)
        handle = self._name_to_handle.pop(name)
        self._handle_to_name.pop(handle, None)
        self._objects.pop(name, None)
        self._kinds.pop(name, None)
        self._topology.pop(name, None)
        self._reverse_objects.pop(id(obj), None)
        self._pick_metadata.pop(name, None)
        return True


def _snapshot(*, bounce_count: int = 2) -> MpcPathInspectionSnapshot:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (4.0, 2.0, 0.0),
        ),
        dtype=np.float32,
    )
    if bounce_count == 0:
        points = points[[0, -1]]
    return MpcPathInspectionSnapshot(
        frame_token=("scenario", 3),
        canonical_path_id=7,
        points=points,
        bounce_interaction_types=(np.asarray((1, 8), dtype=np.int32) if bounce_count else None),
        bounce_colors=(
            np.asarray(((1.0, 0.3, 0.1), (0.2, 0.6, 1.0)), dtype=np.float32)
            if bounce_count
            else None
        ),
    )


def test_selected_path_builds_stable_objects_and_selected_labels_only() -> None:
    renderer = _SelectionRenderer()

    assert renderer.set_mpc_path_inspection(_snapshot()) is True

    expected = {
        MPC_SELECTION_HALO_NAME,
        MPC_SELECTION_PATH_NAME,
        MPC_SELECTION_DIRECTION_NAME,
        MPC_SELECTION_PULSE_NAME,
        MPC_SELECTION_BOUNCE_POINTS_NAME,
        f"{MPC_SELECTION_PREFIX}bounce_1",
        f"{MPC_SELECTION_PREFIX}bounce_2",
    }
    assert renderer._mpc_selection_names == expected
    assert {name for name in renderer._objects if name.startswith(MPC_SELECTION_PREFIX)} == expected
    assert renderer._objects[f"{MPC_SELECTION_PREFIX}bounce_1"]._text_blocks[0]._input == (
        "text",
        "1",
    )
    assert renderer._objects[f"{MPC_SELECTION_PREFIX}bounce_2"]._text_blocks[0]._input == (
        "text",
        "2",
    )
    for name in expected:
        material = renderer._objects[name].material
        assert material.pick_write is False
        assert material.depth_write is False


def test_flow_updates_only_preallocated_pulse_and_leaves_bulk_mpc_untouched() -> None:
    renderer = _SelectionRenderer()
    bulk_geometry = gfx.Geometry(positions=np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.float32))
    bulk = gfx.Line(bulk_geometry, gfx.LineMaterial())
    renderer._objects["mpc_lines"] = bulk
    bulk_positions_before = bulk.geometry.positions.data.copy()
    bulk_revision_before = bulk.geometry.positions.rev

    assert renderer.set_mpc_path_inspection(_snapshot()) is True
    pulse = renderer._objects[MPC_SELECTION_PULSE_NAME]
    pulse_object_id = id(pulse)
    pulse_buffer_id = id(pulse.geometry.positions)
    pulse_positions_before = pulse.geometry.positions.data.copy()
    redraw_before = renderer.redraw_count

    assert renderer.update_mpc_path_flow(0.5) is True

    assert id(renderer._objects[MPC_SELECTION_PULSE_NAME]) == pulse_object_id
    assert id(renderer._objects[MPC_SELECTION_PULSE_NAME].geometry.positions) == pulse_buffer_id
    assert not np.array_equal(pulse.geometry.positions.data, pulse_positions_before)
    assert renderer.redraw_count == redraw_before + 1
    assert bulk.geometry.positions.rev == bulk_revision_before
    np.testing.assert_array_equal(bulk.geometry.positions.data, bulk_positions_before)
    assert renderer._objects["mpc_lines"] is bulk


def test_closed_or_unavailable_canvas_rejects_selection_and_flow_updates() -> None:
    renderer = _SelectionRenderer()
    snapshot = _snapshot()
    renderer._canvas = None

    assert renderer.set_mpc_path_inspection(snapshot) is False
    assert renderer._mpc_selection_names == set()

    renderer._canvas = SimpleNamespace(_orchav_closed=False, _is_closed=False)
    assert renderer.set_mpc_path_inspection(snapshot) is True
    pulse_positions = renderer._objects[MPC_SELECTION_PULSE_NAME].geometry.positions.data.copy()
    redraw_before = renderer.redraw_count
    renderer._qt_window_closed = True

    assert renderer.update_mpc_path_flow(0.5) is False
    np.testing.assert_array_equal(
        renderer._objects[MPC_SELECTION_PULSE_NAME].geometry.positions.data,
        pulse_positions,
    )
    assert renderer.redraw_count == redraw_before


def test_failed_selection_cleanup_hides_retains_and_retries_native_ownership() -> None:
    renderer = _SelectionRenderer()
    assert renderer.set_mpc_path_inspection(_snapshot()) is True
    renderer.remove_failures[MPC_SELECTION_HALO_NAME] = 1

    assert renderer.clear_mpc_path_inspection() is False
    assert renderer._mpc_selection_names == {MPC_SELECTION_HALO_NAME}
    assert renderer._objects[MPC_SELECTION_HALO_NAME].visible is False
    assert renderer._mpc_selection_snapshot is not None

    assert renderer.clear_mpc_path_inspection() is True
    assert renderer._mpc_selection_names == set()
    assert renderer._mpc_selection_snapshot is None
    assert renderer._mpc_selection_pulse_geometry is None


def test_frame_transition_restores_rejected_selection_and_commits_accepted_removal() -> None:
    renderer = _SelectionRenderer()
    snapshot = _snapshot()
    assert renderer.set_mpc_path_inspection(snapshot) is True

    renderer._begin_mpc_path_inspection_frame_transition()
    assert renderer._mpc_selection_names == set()
    assert renderer._mpc_selection_frame_snapshot is snapshot

    assert PygfxRuntimeMixin._complete_mpc_path_inspection_transition(renderer, False) is False
    assert renderer._mpc_selection_snapshot is snapshot
    assert renderer._mpc_selection_names

    renderer._begin_mpc_path_inspection_frame_transition()
    assert PygfxRuntimeMixin._complete_mpc_path_inspection_transition(renderer, True) is True
    assert renderer._mpc_selection_names == set()
    assert renderer._mpc_selection_snapshot is None
    assert renderer._mpc_selection_frame_snapshot is None


def test_apply_frame_suspends_old_selection_before_bulk_geometry_mutation() -> None:
    renderer = _SelectionRenderer()
    snapshot = _snapshot()
    assert renderer.set_mpc_path_inspection(snapshot) is True
    old_packet = SimpleNamespace(stats_text="old")
    new_packet = SimpleNamespace(stats_text="new")
    renderer.last_frame_packet = old_packet
    selection_visible_during_bulk_apply: list[bool] = []
    renderer._apply_mpc_lines = lambda _packet: (
        selection_visible_during_bulk_apply.append(bool(renderer._mpc_selection_names)) or False
    )
    renderer._apply_mpc_points = lambda _packet: True
    renderer._apply_coverage_data_diff = lambda _old, _new: True
    renderer._apply_rf_xray_overlay = lambda _packet: False
    renderer._apply_unsupported_features = lambda _packet: True
    renderer._apply_stats_diff = lambda _old, _new: None
    renderer._record_profile_metric = lambda *_args: None
    renderer._ensure_ground_grid_current = lambda: None
    renderer._update_mpc_hud_overlays = lambda _packet: None

    assert PygfxRenderer.apply_frame(renderer, new_packet) is False

    assert selection_visible_during_bulk_apply == [False]
    assert renderer.last_frame_packet is old_packet
    assert renderer._mpc_selection_snapshot is snapshot
    assert renderer._mpc_selection_names


def test_replacement_and_clear_remove_old_bounce_labels_without_bulk_changes() -> None:
    renderer = _SelectionRenderer()
    bulk = SimpleNamespace(material=object())
    renderer._objects["mpc_lines"] = bulk

    assert renderer.set_mpc_path_inspection(_snapshot()) is True
    assert renderer.set_mpc_path_inspection(_snapshot(bounce_count=0)) is True

    selection_names = {name for name in renderer._objects if name.startswith(MPC_SELECTION_PREFIX)}
    assert selection_names == {
        MPC_SELECTION_HALO_NAME,
        MPC_SELECTION_PATH_NAME,
        MPC_SELECTION_DIRECTION_NAME,
        MPC_SELECTION_PULSE_NAME,
    }
    assert renderer._objects["mpc_lines"] is bulk

    assert renderer.clear_mpc_path_inspection() is True
    assert not any(name.startswith(MPC_SELECTION_PREFIX) for name in renderer._objects)
    assert renderer._objects["mpc_lines"] is bulk
    assert renderer.update_mpc_path_flow(0.25) is False


def test_pulse_trail_clamps_pre_entry_particles_at_tx_instead_of_wrapping_to_rx() -> None:
    renderer = _SelectionRenderer()
    snapshot = _snapshot()
    assert renderer.set_mpc_path_inspection(snapshot) is True
    pulse_positions = renderer._objects[MPC_SELECTION_PULSE_NAME].geometry.positions.data

    np.testing.assert_allclose(
        pulse_positions,
        np.tile(snapshot.points[0], (len(pulse_positions), 1)),
    )

    assert renderer.update_mpc_path_flow(0.01) is True
    assert not np.any(np.all(np.isclose(pulse_positions, snapshot.points[-1]), axis=1))


def test_selected_bounce_markers_reuse_bulk_interaction_marker_semantics() -> None:
    marker_int = {
        "custom": 10,
        "circle": 11,
        "triangle_up": 12,
        "diamond": 13,
        "plus": 14,
        "square": 15,
        "cross": 16,
    }
    interaction_types = np.asarray((0, 1, 2, 4, 8, 99, 1234), dtype=np.int32)

    selected = PygfxMpcSelectionMixin._selected_bounce_marker_codes(
        interaction_types,
        marker_int,
    )
    bulk = PygfxMpcMixin._interaction_marker_codes(interaction_types, marker_int)

    np.testing.assert_array_equal(selected, bulk)


def test_degenerate_selected_path_keeps_static_pulse_without_arrow() -> None:
    renderer = _SelectionRenderer()
    snapshot = MpcPathInspectionSnapshot(
        frame_token=1,
        canonical_path_id=0,
        points=np.asarray(((1, 2, 3), (1, 2, 3)), dtype=np.float32),
    )

    assert renderer.set_mpc_path_inspection(snapshot) is True
    assert MPC_SELECTION_DIRECTION_NAME not in renderer._objects
    assert renderer.update_mpc_path_flow(0.75) is True
    np.testing.assert_allclose(
        renderer._objects[MPC_SELECTION_PULSE_NAME].geometry.positions.data,
        np.tile(np.asarray((1, 2, 3), dtype=np.float32), (8, 1)),
    )
