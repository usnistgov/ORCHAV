"""Tests for aperture visualization service integration behavior."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.aperture_service import ApertureService
from visualizer.src.types.render_payloads import TextLabelPayload


class _FakeRenderer:
    def __init__(self) -> None:
        self.renderer_id = "pygfx"
        self.capabilities = RendererCapabilities(aperture_preview=True)
        self.added: list[tuple[str, object]] = []
        self.ensured_objects = []
        self.removed: list[str] = []
        self.active_names: set[str] = set()
        self.update_count = 0
        self.batch_depth = 0
        self.batch_events: list[str] = []
        self.operation_batch_depths: list[int] = []
        self.fail_ensure_once: set[str] = set()
        self.fail_remove_once: set[str] = set()

    @contextmanager
    def batch_updates(self):
        self.batch_events.append("enter")
        self.batch_depth += 1
        try:
            yield
        finally:
            self.batch_depth -= 1
            self.batch_events.append("exit")

    def ensure_named_geometry(self, name: str, geometry: object, **_kwargs) -> bool:
        self.added.append((name, geometry))
        self.active_names.add(name)
        return True

    def ensure_object(self, obj) -> bool:
        self.operation_batch_depths.append(self.batch_depth)
        self.ensured_objects.append(obj)
        self.added.append((obj.id, obj.payload))
        if obj.id in self.fail_ensure_once:
            self.fail_ensure_once.remove(obj.id)
            return False
        self.active_names.add(obj.id)
        return True

    def remove_object(self, name: str) -> bool:
        self.operation_batch_depths.append(self.batch_depth)
        self.removed.append(name)
        if name in self.fail_remove_once:
            self.fail_remove_once.remove(name)
            return False
        if name not in self.active_names:
            return False
        self.active_names.remove(name)
        return True

    def request_redraw(self) -> None:
        self.update_count += 1


class _FakeCacheService:
    def get_frame(self, _step: int) -> dict[str, list[tuple[float, float, float]]]:
        return {
            "tx_orientations": [(0.0, 0.0, 0.0)],
            "rx_orientations": [(0.0, 0.0, 0.0)],
        }


def _make_viz(**state_overrides):
    state = SimpleNamespace(
        step=0,
        show_aoa_aperture=True,
        show_aod_aperture=False,
        show_global_angular_reference=False,
        show_local_angular_reference=False,
        selected_tx="all",
        selected_rx=0,
        aperture_radius_m=1.0,
        aoa_az_filter_min_deg=-45.0,
        aoa_az_filter_max_deg=45.0,
        aoa_el_filter_min_deg=None,
        aoa_el_filter_max_deg=None,
        aod_az_filter_min_deg=-45.0,
        aod_az_filter_max_deg=45.0,
        aod_el_filter_min_deg=None,
        aod_el_filter_max_deg=None,
    )
    for key, value in state_overrides.items():
        setattr(state, key, value)

    return SimpleNamespace(
        app_state=state,
        vis_initialized=True,
        renderer=_FakeRenderer(),
        cache_service=_FakeCacheService(),
        current_rx_positions=[np.array([0.0, 0.0, 0.0])],
        current_tx_positions=[np.array([1.0, 0.0, 0.0])],
    )


def test_aperture_service_uses_named_renderer_geometry():
    viz = _make_viz()
    service = ApertureService(viz)

    service.update_apertures()

    assert {name for name, _geometry in viz.renderer.added} == {
        "aperture_aoa_0_patch",
        "aperture_aoa_0",
    }
    assert viz.renderer.update_count == 1

    service.clear_all()

    assert viz.renderer.removed == ["aperture_aoa_0", "aperture_aoa_0_patch"]


def test_aperture_update_and_clear_are_each_one_renderer_batch():
    viz = _make_viz()
    service = ApertureService(viz)

    assert service.update_apertures() is True
    assert viz.renderer.batch_events == ["enter", "exit"]
    assert viz.renderer.operation_batch_depths
    assert all(depth == 1 for depth in viz.renderer.operation_batch_depths)

    viz.renderer.batch_events.clear()
    viz.renderer.operation_batch_depths.clear()
    assert service.clear_all() is True

    assert viz.renderer.batch_events == ["enter", "exit"]
    assert viz.renderer.operation_batch_depths
    assert all(depth == 1 for depth in viz.renderer.operation_batch_depths)


def test_aperture_service_skips_all_node_selection():
    viz = _make_viz(selected_rx="all")
    service = ApertureService(viz)

    service.update_apertures()

    assert viz.renderer.added == []


def test_aperture_service_draws_full_range_preview_from_natural_bounds():
    viz = _make_viz(
        aoa_az_filter_min_deg=None,
        aoa_az_filter_max_deg=None,
        aoa_el_filter_min_deg=None,
        aoa_el_filter_max_deg=None,
    )
    service = ApertureService(viz)

    service.update_apertures()

    assert {name for name, _geometry in viz.renderer.added} == {
        "aperture_aoa_0_patch",
        "aperture_aoa_0",
    }
    geometry = next(geometry for name, geometry in viz.renderer.added if name == "aperture_aoa_0")
    assert len(geometry.points) > 0


def test_aperture_service_update_removes_aperture_when_unchecked():
    viz = _make_viz()
    service = ApertureService(viz)

    service.update_apertures()
    viz.app_state.show_aoa_aperture = False
    service.update_apertures()

    assert viz.renderer.removed == ["aperture_aoa_0", "aperture_aoa_0_patch"]
    assert viz.renderer.active_names == set()


def test_aperture_service_update_removes_aod_aperture_when_unchecked():
    viz = _make_viz(show_aoa_aperture=False, show_aod_aperture=True, selected_tx=0)
    service = ApertureService(viz)

    service.update_apertures()
    viz.app_state.show_aod_aperture = False
    service.update_apertures()

    assert viz.renderer.removed == ["aperture_aod_0", "aperture_aod_0_patch"]
    assert viz.renderer.active_names == set()


def test_aperture_service_draws_global_reference_at_selected_nodes():
    viz = _make_viz(
        show_aoa_aperture=False,
        selected_tx=0,
        selected_rx=0,
        show_global_angular_reference=True,
    )
    service = ApertureService(viz)

    service.update_apertures()

    names = [name for name, _geometry in viz.renderer.added]
    assert "angular_reference_global_tx_0" in names
    assert "angular_reference_global_rx_0" in names


def test_aperture_reference_labels_use_neutral_object_contract():
    viz = _make_viz(
        show_aoa_aperture=False,
        selected_tx=0,
        selected_rx="all",
        show_global_angular_reference=True,
    )
    service = ApertureService(viz)

    service.update_apertures()

    labels = [
        render_object
        for render_object in viz.renderer.ensured_objects
        if isinstance(render_object.payload, TextLabelPayload)
    ]
    assert len(labels) == 6
    assert all(render_object.visible for render_object in labels)
    assert {render_object.payload.text for render_object in labels} == {
        "G 0",
        "G 90",
        "G 180",
        "G -90",
        "G +El",
        "G -El",
    }


def test_aperture_service_removes_reference_when_unchecked():
    viz = _make_viz(
        show_aoa_aperture=False,
        selected_tx=0,
        selected_rx="all",
        show_local_angular_reference=True,
    )
    service = ApertureService(viz)

    service.update_apertures()
    viz.app_state.show_local_angular_reference = False
    service.update_apertures()

    assert set(viz.renderer.removed) == {
        "angular_reference_local_tx_0",
        *{f"angular_reference_local_tx_0_label_{index}" for index in range(6)},
    }
    assert viz.renderer.active_names == set()


def test_aperture_service_skips_non_pygfx_renderer():
    viz = _make_viz()
    viz.renderer.renderer_id = "open3d"
    viz.renderer.capabilities = RendererCapabilities()
    service = ApertureService(viz)

    service.update_apertures()

    assert viz.renderer.added == []


def test_aperture_service_identical_update_is_backend_noop():
    viz = _make_viz()
    service = ApertureService(viz)

    assert service.update_apertures() is True
    first_ensure_count = len(viz.renderer.ensured_objects)
    assert service.update_apertures() is True

    assert first_ensure_count == 2
    assert len(viz.renderer.ensured_objects) == first_ensure_count
    assert viz.renderer.removed == []
    assert viz.renderer.update_count == 1


def test_failed_aperture_removal_is_retained_for_identical_retry():
    viz = _make_viz()
    service = ApertureService(viz)
    assert service.update_apertures() is True
    viz.app_state.show_aoa_aperture = False
    viz.renderer.fail_remove_once.add("aperture_aoa_0")

    assert service.update_apertures() is False
    assert "aperture_aoa_0" in service._active_aoa_geometries

    assert service.update_apertures() is True
    assert "aperture_aoa_0" not in service._active_aoa_geometries
    assert viz.renderer.removed.count("aperture_aoa_0") == 2


def test_failed_initial_aperture_ensure_retries_same_desired_snapshot():
    viz = _make_viz()
    viz.renderer.fail_ensure_once.add("aperture_aoa_0")
    service = ApertureService(viz)

    assert service.update_apertures() is False
    assert "aperture_aoa_0" not in service._active_aoa_geometries

    assert service.update_apertures() is True
    assert "aperture_aoa_0" in service._active_aoa_geometries
    assert [obj.id for obj in viz.renderer.ensured_objects].count("aperture_aoa_0") == 2
