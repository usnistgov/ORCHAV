from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.mpc_selection_service import (
    MPC_FLOW_LOOP_SECONDS,
    MpcSelectionIdentity,
    MpcSelectionService,
)


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Renderer:
    capabilities = RendererCapabilities(
        mpc_path_inspection=True,
        picking=True,
    )

    def __init__(
        self,
        *,
        accept_overlay: bool = True,
        clear_results: tuple[bool, ...] = (),
    ) -> None:
        self.accept_overlay = accept_overlay
        self.available = True
        self.clear_results = list(clear_results)
        self.callback = None
        self.callback_values: list[Any] = []
        self.snapshots = []
        self.active_snapshot = None
        self.flow_phases: list[float] = []
        self.pick_mappings: list[tuple[int, Any]] = []
        self.clear_count = 0
        self.bulk_mpc_revision = ("unchanged", 7)

    def set_mpc_path_selection_callback(self, callback) -> None:
        self.callback = callback
        self.callback_values.append(callback)

    def mpc_path_inspection_available(self) -> bool:
        return self.available

    def set_mpc_path_inspection(self, snapshot) -> bool:
        self.snapshots.append(snapshot)
        if self.accept_overlay:
            self.active_snapshot = snapshot
        return self.accept_overlay

    def has_mpc_path_inspection(self, snapshot=None) -> bool:
        return self.active_snapshot is not None and (
            snapshot is None or snapshot is self.active_snapshot
        )

    def update_mpc_path_flow(self, phase: float) -> bool:
        self.flow_phases.append(phase)
        return True

    def clear_mpc_path_inspection(self) -> bool:
        self.clear_count += 1
        if self.clear_results:
            cleared = self.clear_results.pop(0)
            if not cleared:
                return False
        self.active_snapshot = None
        return True

    def set_mpc_pick_segment_mapping(self, packet_identity: int, mapping: Any) -> bool:
        self.pick_mappings.append((packet_identity, mapping))
        return True


class _UnsupportedRenderer(_Renderer):
    capabilities = RendererCapabilities()


class _Catalog:
    def __init__(
        self,
        *,
        path_zero_points: np.ndarray | None = None,
        filtered: tuple[bool, bool] = (True, False),
        rendered: tuple[bool, bool] = (True, False),
    ) -> None:
        self._points = (
            np.asarray(
                (
                    path_zero_points
                    if path_zero_points is not None
                    else [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
                ),
                dtype=np.float32,
            ),
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [3.0, 2.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        self._interactions = (
            np.empty((0,), dtype=np.int32),
            np.asarray([1, 8], dtype=np.int32),
        )
        self._materials = (
            np.empty((0,), dtype=np.int16),
            np.asarray([4, 7], dtype=np.int16),
        )
        self._filtered = filtered
        self._rendered = rendered
        self.path_count = 2
        self.canonical_data = SimpleNamespace(
            path_delay_is_estimated=np.asarray([False, True]),
            path_loss_is_estimated=np.asarray([False, False]),
        )
        self._columns = {
            "tx": np.asarray([3, 3], dtype=np.int16),
            "rx": np.asarray([5, 6], dtype=np.int16),
            "path_loss_db": np.asarray([72.0, 91.5], dtype=np.float32),
            "delay_ns": np.asarray([10.0, 25.0], dtype=np.float32),
            "interactions": np.asarray([0, 2], dtype=np.int16),
        }
        self.rendered_segment_indices = np.asarray((0, 2), dtype=np.int32)

    def path_points(self, path_id: int) -> np.ndarray:
        return self._points[path_id]

    def interaction_sequence(self, path_id: int) -> np.ndarray:
        return self._interactions[path_id]

    def material_sequence(self, path_id: int) -> np.ndarray:
        return self._materials[path_id]

    @staticmethod
    def material_name(material_id: int) -> str:
        return {4: "Concrete", 7: "Glass"}.get(material_id, "")

    def is_filtered(self, path_id: int) -> bool:
        return self._filtered[path_id]

    def is_rendered(self, path_id: int) -> bool:
        return self._rendered[path_id]

    def column(self, name: str) -> np.ndarray:
        return self._columns[name]


def _service(
    qapp,
    *,
    renderer: _Renderer | None = None,
    clock: _Clock | None = None,
) -> tuple[MpcSelectionService, _Renderer, _Clock]:
    del qapp
    active_renderer = renderer or _Renderer()
    active_clock = clock or _Clock()
    palette = np.zeros((9, 3), dtype=np.float32)
    palette[1] = (0.1, 0.2, 0.3)
    palette[5] = (0.8, 0.7, 0.6)
    visualizer = SimpleNamespace(
        renderer=active_renderer,
        mpc_core=SimpleNamespace(_type_palette=palette),
    )
    return (
        MpcSelectionService(visualizer, clock=active_clock),
        active_renderer,
        active_clock,
    )


def test_selects_full_canonical_path_outside_current_rendered_set(qapp) -> None:
    service, renderer, _clock = _service(qapp)
    token = ("scenario-a", 4, 100)
    catalog = _Catalog()
    selection_events = []
    detail_events = []
    service.selectionChanged.connect(selection_events.append)
    service.detailsChanged.connect(detail_events.append)

    service.set_presented_frame(token, catalog, object(), frame_changed=True)
    service.select_path(token, 1, origin="table")

    assert service.selection == MpcSelectionIdentity(token, 1)
    assert len(renderer.snapshots) == 1
    snapshot = renderer.snapshots[0]
    np.testing.assert_array_equal(snapshot.points, catalog.path_points(1))
    np.testing.assert_array_equal(
        snapshot.bounce_interaction_types,
        np.asarray([1, 8], dtype=np.int32),
    )
    np.testing.assert_allclose(
        snapshot.bounce_colors,
        np.asarray([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32),
    )
    assert snapshot.bounce_labels == ("1", "2")
    assert service.details is not None
    assert service.details.render_status == "outside current rendered set"
    assert service.details.is_filtered is False
    assert service.details.is_rendered is False
    assert service.details.material_names == ("Concrete", "Glass")
    assert service.details.interaction_labels == ("Specular", "Diffraction")
    assert service.details.delay_is_estimated is True
    assert service.flow_timer_active
    assert selection_events == [MpcSelectionIdentity(token, 1)]
    assert detail_events == [service.details]

    service.shutdown()


def test_changed_token_clears_selection_but_same_token_refresh_preserves_it(qapp) -> None:
    service, renderer, _clock = _service(qapp)
    token = ("scenario-a", 1, 10)
    initial = _Catalog()
    selection_events = []
    detail_events = []
    service.selectionChanged.connect(selection_events.append)
    service.detailsChanged.connect(detail_events.append)
    service.set_presented_frame(token, initial)
    service.select_path(token, 0, origin="table")
    flow_started_at = service._flow_started_at
    initial_snapshot = renderer.snapshots[-1]

    refreshed = _Catalog(
        rendered=(False, False),
    )
    service.set_presented_frame(
        token,
        refreshed,
        frame_changed=True,
    )

    assert service.selection == MpcSelectionIdentity(token, 0)
    assert service.catalog is refreshed
    assert renderer.snapshots == [initial_snapshot]
    assert service._flow_started_at == flow_started_at

    # Mask-derived details update after the query worker has warmed the new
    # catalog, without replacing immutable selected geometry.
    service.refresh_selected_details()
    assert service.details is not None
    assert service.details.geometric_length_m == 1.0
    assert service.details.render_status == "outside current rendered set"
    assert selection_events == [MpcSelectionIdentity(token, 0)]
    assert len(detail_events) == 2

    replacement_token = ("scenario-b", 1, 11)
    replacement = _Catalog()
    service.set_presented_frame(replacement_token, replacement, frame_changed=False)

    assert service.selection is None
    assert service.details is None
    assert not service.flow_timer_active
    assert service.catalog is replacement
    assert selection_events[-1] is None
    assert detail_events[-1] is None
    assert renderer.clear_count == 1

    service.shutdown()


def test_same_canonical_frame_rebuilds_selection_after_renderer_frame_transition(qapp) -> None:
    service, renderer, clock = _service(qapp)
    token = ("scenario-a", 1, 12)
    catalog = _Catalog()
    service.set_presented_frame(token, catalog)
    service.select_path(token, 1, origin="table")
    first_snapshot = renderer.snapshots[-1]
    first_flow_start = service._flow_started_at

    # A different render packet for the same canonical frame (for example a
    # filter change) transactionally removes the old native overlay before it
    # is submitted. The service retains selection identity and reinstalls it.
    renderer.active_snapshot = None
    clock.value += 2.0
    service.set_presented_frame(token, catalog, frame_changed=False)

    assert service.selection == MpcSelectionIdentity(token, 1)
    assert len(renderer.snapshots) == 2
    assert renderer.snapshots[0] is first_snapshot
    assert renderer.snapshots[1] is first_snapshot
    assert renderer.active_snapshot is first_snapshot
    assert service._flow_started_at > first_flow_start
    assert service.flow_timer_active
    service.shutdown()


def test_viewport_and_table_route_through_the_same_selection_method(qapp) -> None:
    service, renderer, _clock = _service(qapp)
    token = ("scenario", 2, 20)
    accepted_packet = object()
    service.set_presented_frame(token, _Catalog(), accepted_packet)
    assert callable(renderer.callback)

    renderer.callback(1, id(accepted_packet))
    assert service.selected_path_id == 1
    assert service.details is not None
    assert service.details.origin == "viewport"

    service.select_path(token, 1, origin="table")
    assert service.selected_path_id == 1
    assert service.details is not None
    assert service.details.origin == "table"

    service.clear_presented_frame(reason="Explorer hidden")
    assert renderer.callback is None
    assert service.catalog is None
    assert not service.is_active


def test_stale_frame_identity_cannot_select_a_path(qapp) -> None:
    service, renderer, _clock = _service(qapp)
    current = ("scenario", 3, 30)
    service.set_presented_frame(current, _Catalog())

    service.select_path(("scenario", 2, 20), 1, origin="table")

    assert service.selection is None
    assert renderer.snapshots == []
    assert not service.flow_timer_active
    service.shutdown()


def test_viewport_rejects_packet_not_accepted_by_explorer_lifecycle(qapp) -> None:
    service, renderer, _clock = _service(qapp)
    token = ("scenario", 4, 40)
    accepted_packet = object()
    rejected_packet = object()
    service.set_presented_frame(token, _Catalog(), accepted_packet)

    renderer.callback(1, id(rejected_packet))

    assert service.selection is None
    assert renderer.snapshots == []

    renderer.callback(1, id(accepted_packet))
    assert service.selection == MpcSelectionIdentity(token, 1)
    service.shutdown()


def test_worker_prepared_pick_mapping_requires_current_catalog_and_packet(qapp) -> None:
    service, renderer, _clock = _service(qapp)
    token = ("scenario", 5, 50)
    catalog = _Catalog()
    accepted_packet = object()
    service.set_presented_frame(token, catalog, accepted_packet)

    assert not service.prepare_viewport_mapping(
        ("stale", 5, 50),
        catalog,
        accepted_packet,
    )
    assert not service.prepare_viewport_mapping(token, _Catalog(), accepted_packet)
    assert service.prepare_viewport_mapping(token, catalog, accepted_packet)

    assert len(renderer.pick_mappings) == 1
    packet_identity, mapping = renderer.pick_mappings[0]
    assert packet_identity == id(accepted_packet)
    assert mapping is catalog.rendered_segment_indices
    np.testing.assert_array_equal(mapping, catalog.rendered_segment_indices)
    service.shutdown()


def test_flow_tick_updates_only_the_small_renderer_overlay(qapp) -> None:
    clock = _Clock(10.0)
    service, renderer, _clock = _service(qapp, clock=clock)
    token = ("scenario", 7, 70)
    service.set_presented_frame(token, _Catalog())
    service.select_path(token, 1, origin="table")
    bulk_revision = renderer.bulk_mpc_revision

    clock.value += MPC_FLOW_LOOP_SECONDS / 2.0
    service._on_flow_timeout()

    assert renderer.flow_phases == [0.5]
    assert renderer.bulk_mpc_revision == bulk_revision
    assert len(renderer.snapshots) == 1

    renderer.update_mpc_path_flow = lambda _phase: False
    service._on_flow_timeout()
    assert not service.flow_timer_active
    service.shutdown()


def test_unavailable_renderer_rejects_overlay_and_stops_an_existing_flow_timer(qapp) -> None:
    service, renderer, _clock = _service(qapp)
    token = ("scenario", 7, 71)
    service.set_presented_frame(token, _Catalog())
    renderer.available = False

    service.select_path(token, 1, origin="table")

    assert service.selection == MpcSelectionIdentity(token, 1)
    assert renderer.snapshots == []
    assert not service.flow_timer_active

    renderer.available = True
    service.select_path(token, 1, origin="table")
    assert service.flow_timer_active
    assert len(renderer.snapshots) == 1

    renderer.available = False
    service._on_flow_timeout()
    assert renderer.flow_phases == []
    assert not service.flow_timer_active
    service.shutdown()


def test_failed_renderer_cleanup_retains_owner_and_retries_on_next_lifecycle_step(qapp) -> None:
    renderer = _Renderer(clear_results=(False, True))
    service, renderer, _clock = _service(qapp, renderer=renderer)
    token = ("scenario", 7, 72)
    service.set_presented_frame(token, _Catalog())
    service.select_path(token, 1, origin="table")

    service.clear_selection()

    assert service.selection is None
    assert renderer.clear_count == 1
    assert service._overlay_renderer is renderer

    service.clear_presented_frame(reason="Explorer hidden")

    assert renderer.clear_count == 2
    assert service._overlay_renderer is None
    service.shutdown()


def test_zero_length_path_and_unsupported_renderer_never_start_animation(qapp) -> None:
    zero_path = np.asarray(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    service, renderer, _clock = _service(qapp)
    token = ("scenario", 8, 80)
    service.set_presented_frame(token, _Catalog(path_zero_points=zero_path))
    service.select_path(token, 0, origin="table")

    assert len(renderer.snapshots) == 1
    assert not service.flow_timer_active
    service.shutdown()

    unsupported = _UnsupportedRenderer()
    service, renderer, _clock = _service(qapp, renderer=unsupported)
    service.set_presented_frame(token, _Catalog())
    assert renderer.callback is None
    service.select_path(token, 1, origin="table")

    assert service.selection == MpcSelectionIdentity(token, 1)
    assert service.details is not None
    assert renderer.snapshots == []
    assert not service.flow_timer_active
    service.shutdown()


def test_shutdown_is_idempotent_and_releases_every_reference(qapp) -> None:
    service, renderer, _clock = _service(qapp)
    token = ("scenario", 9, 90)
    accepted_packet = object()
    service.set_presented_frame(token, _Catalog(), accepted_packet)
    service.select_path(
        token,
        1,
        origin="viewport",
        packet_identity=id(accepted_packet),
    )

    service.shutdown()
    service.shutdown()

    assert renderer.callback is None
    assert renderer.clear_count == 1
    assert service.selection is None
    assert service.details is None
    assert service.catalog is None
    assert service.frame_token is None
    assert service.presented_packet_identity is None
    assert not service.flow_timer_active
    assert not service.is_active
