"""Contract tests for the renderer-neutral public protocol.

The shared persistent-object surface is deliberately declarative. Backend
geometry registries and named-object helpers are implementation details and do
not belong in this suite.

A lightweight ``MockProtocolRenderer`` keeps the contract tests independent of
GPU and display availability. Real backends retain their own focused behavior
tests.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Callable, Generator, Optional

import numpy as np
import pytest

from visualizer.src.model import RenderObject, Transform
from visualizer.src.pipeline.core import FrameRenderPacket
from visualizer.src.renderers.protocol import (
    RendererCapabilities,
    RendererProtocol,
    renderer_capabilities,
)
from visualizer.src.types.camera_state import CameraState
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    MeshPayload,
)

# ---------------------------------------------------------------------------
# Backend availability checks
# ---------------------------------------------------------------------------


def _pygfx_available() -> bool:
    try:
        import pygfx  # noqa: F401
        from PySide6 import QtWidgets as _QtWidgets  # noqa: F401
        from rendercanvas.qt import RenderCanvas as _RenderCanvas  # noqa: F401

        return True
    except Exception:
        return False


def _pygfx_tests_enabled() -> bool:
    return os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") == "1"


# ---------------------------------------------------------------------------
# Mock renderer that fulfils the protocol (reference implementation)
# ---------------------------------------------------------------------------


class MockProtocolRenderer:
    """Minimal renderer implementing RendererProtocol for contract testing.

    Stores declarative object state in plain dictionaries without exposing a
    backend-shaped geometry registry.
    """

    renderer_id: str = "mock"
    capabilities = RendererCapabilities(
        pbr=True,
        shadow_toggle=True,
        camera_lookat=True,
        transparency=True,
        trajectories=True,
        screenshot_export=True,
        material_clearcoat=True,
        material_emissive=True,
        material_anisotropy=True,
        material_transmission=True,
        material_volume_thickness=True,
        material_normal_map=True,
    )

    def __init__(self) -> None:
        self._objects: dict[str, RenderObject] = {}
        self._visible: dict[str, bool] = {}
        self._materials: dict[str, MaterialPayload | dict[str, Any]] = {}
        self._transforms: dict[str, np.ndarray] = {}
        self._camera: Optional[CameraState] = None
        self._initialized: bool = False

    # -- lifecycle ----------------------------------------------------------

    def initialize_visualizer(
        self,
        window_name: str = "ORCHAV",
        width: int = 1024,
        height: int = 768,
        left: int = -1,
        top: int = -1,
        suppress_default_camera: bool = False,
    ) -> Any:
        self._initialized = True
        return None

    def close(self) -> None:
        self._objects.clear()
        self._visible.clear()
        self._materials.clear()
        self._transforms.clear()
        self._initialized = False

    def update_renderer(self) -> None:
        pass

    def request_redraw(self) -> None:
        pass

    def refresh_viewport_hud(self) -> None:
        pass

    def defer_until_next_render_turn(self, callback: Callable[[], None]) -> bool:
        del callback
        return False

    def begin_frame_update(self) -> None:
        pass

    def end_frame_update(self) -> bool:
        return True

    def get_runtime_stats(self) -> dict[str, Any]:
        return {
            "event_pump_calls": 0,
            "redraw_requests": 0,
            "present_attempts": 0,
            "present_successes": 0,
            "avg_present_interval_ms": None,
            "avg_update_to_present_ms": None,
            "avg_draw_ms": None,
            "present_jitter_ms": None,
            "idle_loop_active": False,
        }

    @contextmanager
    def batch_updates(self) -> Generator[None, None, None]:
        yield

    # -- declarative objects ------------------------------------------------

    def ensure_object(self, obj: RenderObject) -> bool:
        self._objects[obj.id] = obj
        self._visible[obj.id] = bool(obj.visible)
        if obj.material_payload is None:
            self._materials.pop(obj.id, None)
        else:
            self._materials[obj.id] = obj.material_payload
        self._transforms[obj.id] = np.asarray(obj.transform_matrix, dtype=float)
        return True

    def update_mesh_vertex_stream(self, obj: RenderObject) -> bool:
        del obj
        return False

    def remove_object(self, object_id: str) -> bool:
        self._objects.pop(object_id, None)
        self._visible.pop(object_id, None)
        self._materials.pop(object_id, None)
        self._transforms.pop(object_id, None)
        return True

    def set_visible(self, object_id: str, visible: bool) -> bool:
        if object_id not in self._objects:
            return False
        self._visible[object_id] = bool(visible)
        return True

    def set_material(self, object_id: str, material: MaterialPayload | dict[str, Any]) -> bool:
        if object_id not in self._objects:
            return False
        self._materials[object_id] = material
        return True

    def set_transform(self, object_id: str, transform: Transform | np.ndarray) -> bool:
        if object_id not in self._objects:
            return False
        matrix = transform.matrix if isinstance(transform, Transform) else np.asarray(transform)
        self._transforms[object_id] = np.asarray(matrix, dtype=float)
        return True

    # -- camera -------------------------------------------------------------

    def get_camera_state(self) -> Optional[CameraState]:
        return self._camera

    def set_camera_state(self, state: CameraState) -> bool:
        self._camera = state
        return True

    def set_overview_camera(
        self,
        view: str,
        bounds: Any,
        fov: float = 60.0,
        distance: Optional[float] = None,
    ) -> bool:
        center, _extent = bounds
        center = np.asarray(center, dtype=float)
        dist = 10.0 if distance is None else float(distance)
        return self.set_camera_state(
            CameraState(
                eye=(float(center[0]), float(center[1] - dist), float(center[2] + dist)),
                lookat=(float(center[0]), float(center[1]), float(center[2])),
                up=(0.0, 0.0, 1.0),
                fov_deg=float(fov),
            )
        )

    def focus_camera(self, target_position: Any) -> bool:
        return self.update_follow_camera(target_position)

    def set_pov_camera(
        self,
        position: Any,
        orientation: Any,
        axis: str,
        *,
        defer_redraw: bool = False,
    ) -> bool:
        eye = np.asarray(position, dtype=float)
        return self.set_camera_state(
            CameraState(
                eye=(float(eye[0]), float(eye[1]), float(eye[2])),
                lookat=(float(eye[0] + 1.0), float(eye[1]), float(eye[2])),
                up=(0.0, 0.0, 1.0),
                fov_deg=60.0,
            )
        )

    def update_follow_camera(self, target_position: Any) -> bool:
        target = np.asarray(target_position, dtype=float)
        state = self._camera
        if state is None:
            offset = np.array([0.0, -10.0, 10.0], dtype=float)
            up = (0.0, 0.0, 1.0)
            fov = 60.0
        else:
            offset = np.asarray(state.eye, dtype=float) - np.asarray(state.lookat, dtype=float)
            up = state.up
            fov = state.fov_deg
        eye = target + offset
        return self.set_camera_state(
            CameraState(
                eye=(float(eye[0]), float(eye[1]), float(eye[2])),
                lookat=(float(target[0]), float(target[1]), float(target[2])),
                up=up,
                fov_deg=fov,
            )
        )

    def reset_follow_state(self) -> None:
        return None

    def set_shadow_enabled(self, enabled: bool) -> bool:
        return True

    # -- frame application (no-op for mock) ---------------------------------

    def apply_frame(self, packet: FrameRenderPacket) -> bool:
        del packet
        return True

    def set_trajectory_line_width(self, width: float) -> bool:
        del width
        return self.capabilities.line_width

    def set_trajectory_point_size(self, size: float) -> bool:
        del size
        return self.capabilities.trajectories


# ---------------------------------------------------------------------------
# Payload factory helpers
# ---------------------------------------------------------------------------


def _make_mesh() -> MeshPayload:
    """Simple 2-triangle quad."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
    return MeshPayload(vertices=vertices, triangles=triangles, vertex_colors=colors)


def _make_material() -> MaterialPayload:
    return MaterialPayload(
        base_color=(0.8, 0.2, 0.1, 1.0),
        roughness=0.4,
        metallic=0.0,
    )


def _make_camera() -> CameraState:
    return CameraState(
        eye=(10.0, 10.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=60.0,
    )


# ---------------------------------------------------------------------------
# Parameterized renderer fixture
# ---------------------------------------------------------------------------


def _make_pygfx_renderer():
    """Create a PygfxRenderer if dependencies are available."""
    if not _pygfx_available():
        pytest.skip("pygfx dependencies are not installed")

    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    r = PygfxRenderer(SimpleNamespace())
    try:
        r.initialize_visualizer(width=64, height=64)
    except Exception as exc:
        pytest.skip(f"pygfx unavailable in this environment: {exc}")
    return r


_BACKEND_FACTORIES = {
    "mock": lambda: MockProtocolRenderer(),
}
if _pygfx_available() and _pygfx_tests_enabled():
    _BACKEND_FACTORIES["pygfx"] = _make_pygfx_renderer


@pytest.fixture(params=list(_BACKEND_FACTORIES.keys()))
def renderer(request):
    """Yield a renderer implementing the protocol.

    Currently parametrized over:
    - mock: always available, reference implementation
    - pygfx: enabled only for the explicit pygfx runtime test profile
    """
    backend = request.param
    factory = _BACKEND_FACTORIES[backend]
    try:
        r = factory()
    except ImportError:
        pytest.skip(f"{backend} dependencies not available")
    yield r
    # Cleanup
    try:
        r.close()
    except (RuntimeError, AttributeError):
        pass


# ===================================================================
# Contract Tests
# ===================================================================


class TestRendererIdentity:
    """Verify renderer identity and capability flags."""

    def test_renderer_id_is_nonempty_string(self, renderer):
        assert isinstance(renderer.renderer_id, str)
        assert len(renderer.renderer_id) > 0

    def test_capability_map_is_explicit_and_typed(self, renderer):
        capabilities = renderer_capabilities(renderer)
        assert capabilities is renderer.capabilities
        for field in fields(RendererCapabilities):
            value = getattr(capabilities, field.name)
            if field.name.startswith("static_mesh_batch_"):
                assert isinstance(value, int)
            else:
                assert isinstance(value, bool), field.name

    def test_runtime_stats_contract(self, renderer):
        stats = renderer.get_runtime_stats()
        assert isinstance(stats, dict)
        for key in (
            "event_pump_calls",
            "redraw_requests",
            "present_attempts",
            "present_successes",
            "avg_present_interval_ms",
            "avg_update_to_present_ms",
            "avg_draw_ms",
            "present_jitter_ms",
            "idle_loop_active",
        ):
            assert key in stats

    def test_render_turn_deferral_contract(self, renderer):
        callbacks = []
        accepted = renderer.defer_until_next_render_turn(lambda: callbacks.append(True))
        assert isinstance(accepted, bool)
        if not accepted:
            assert callbacks == []

    def test_viewport_hud_refresh_contract(self, renderer):
        assert renderer.refresh_viewport_hud() is None

    def test_atomic_frame_submission_contract(self, renderer):
        renderer.begin_frame_update()
        assert isinstance(renderer.end_frame_update(), bool)

    def test_shadow_toggle_contract(self, renderer):
        if renderer_capabilities(renderer).shadow_toggle:
            assert renderer.set_shadow_enabled(True) is True
            assert renderer.set_shadow_enabled(False) is True

    def test_non_none_renderer_requires_explicit_capability_map(self):
        with pytest.raises(TypeError, match="RendererCapabilities"):
            renderer_capabilities(SimpleNamespace())


class TestRenderObjectApi:
    """Test the mandatory declarative persistent-object lifecycle."""

    def test_protocol_exposes_declarative_object_surface_only(self):
        for method in (
            "ensure_object",
            "update_mesh_vertex_stream",
            "remove_object",
            "set_visible",
            "set_material",
            "set_transform",
        ):
            assert hasattr(RendererProtocol, method)

        for backend_private_method in (
            "ensure_named_geometry",
            "remove_named_geometry",
            "has_named_geometry",
            "get_named_geometry_names",
            "is_named_visible",
            "set_named_visibility",
            "set_named_material",
            "set_named_transform",
            "get_named_position",
            "remap_external_geometry_name",
        ):
            assert not hasattr(RendererProtocol, backend_private_method)

    def test_ensure_object_lifecycle(self, renderer):
        obj = RenderObject(
            id="node:tx_0::marker",
            payload=_make_mesh(),
            material=_make_material(),
            transform=Transform.from_translation([2.0, 3.0, 4.0]),
            visibility=True,
        )

        assert renderer.ensure_object(obj) is True
        assert renderer.set_visible(obj.id, False) is True
        assert renderer.set_material(obj.id, {"base_color": [1.0, 1.0, 1.0, 1.0]}) is True
        assert (
            renderer.set_transform(
                obj.id,
                Transform.from_translation([5.0, 6.0, 7.0]),
            )
            is True
        )
        assert renderer.remove_object(obj.id) is True
        assert renderer.remove_object(obj.id) is True

    def test_ensure_object_updates_existing_id(self, renderer):
        first = RenderObject(id="scene:wall::mesh", payload=_make_mesh())
        replacement = RenderObject(
            id=first.id,
            payload=MeshPayload(
                vertices=np.array(
                    [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
                    dtype=np.float32,
                ),
                triangles=np.array([[0, 1, 2]], dtype=np.int32),
            ),
        )

        assert renderer.ensure_object(first) is True
        assert renderer.ensure_object(replacement) is True
        assert renderer.remove_object(first.id) is True

    def test_property_updates_reject_unknown_object(self, renderer):
        assert renderer.set_visible("missing", False) is False
        assert renderer.set_material("missing", _make_material()) is False
        assert renderer.set_transform("missing", Transform.identity()) is False


class TestCamera:
    """Test camera state round-trip."""

    def test_camera_roundtrip(self, renderer):
        cam = _make_camera()
        set_ok = renderer.set_camera_state(cam)
        assert set_ok is True

        got = renderer.get_camera_state()
        assert got is not None
        assert np.allclose(got.eye, cam.eye, atol=1e-2)
        assert np.allclose(got.lookat, cam.lookat, atol=1e-2)
        assert np.allclose(got.up, cam.up, atol=1e-2)
        assert abs(got.fov_deg - cam.fov_deg) < 2.0

    def test_camera_initially_none(self, renderer):
        """Before any set_camera_state, the result is backend-defined (None or default)."""
        # Just verify it doesn't crash
        renderer.get_camera_state()


class TestObjectProperties:
    """Test public transform, visibility, and material operations."""

    def test_set_visibility(self, renderer):
        obj = RenderObject(id="scene:visible::mesh", payload=_make_mesh())
        assert renderer.ensure_object(obj) is True
        assert renderer.set_visible(obj.id, False) is True
        assert renderer.set_visible(obj.id, True) is True
        assert renderer.remove_object(obj.id) is True

    def test_set_transform(self, renderer):
        obj = RenderObject(id="scene:transform::mesh", payload=_make_mesh())
        assert renderer.ensure_object(obj) is True
        assert (
            renderer.set_transform(
                obj.id,
                Transform.from_translation([7.0, 8.0, 9.0]),
            )
            is True
        )
        assert renderer.remove_object(obj.id) is True

    def test_set_material_payload_and_mapping(self, renderer):
        obj = RenderObject(id="scene:material::mesh", payload=_make_mesh())
        assert renderer.ensure_object(obj) is True
        assert renderer.set_material(obj.id, _make_material()) is True
        assert (
            renderer.set_material(
                obj.id,
                {"base_color": [0.5, 0.5, 0.5, 1.0], "roughness": 0.3},
            )
            is True
        )
        assert renderer.remove_object(obj.id) is True


class TestBatchUpdates:
    """Test batch_updates context manager."""

    def test_batch_context_manager(self, renderer):
        """batch_updates should be usable as a context manager without error."""
        first = RenderObject(id="scene:batch_1::mesh", payload=_make_mesh())
        second = RenderObject(id="scene:batch_2::mesh", payload=_make_mesh())
        with renderer.batch_updates():
            assert renderer.ensure_object(first) is True
            assert renderer.ensure_object(second) is True
        assert renderer.remove_object(first.id) is True
        assert renderer.remove_object(second.id) is True


class TestEdgeCases:
    """Exercise generic renderer operations outside object synchronization."""

    def test_many_objects(self, renderer):
        """Synchronize and remove a moderate object batch through the protocol."""
        object_ids = []
        for index in range(50):
            object_id = f"scene:batch_{index}::mesh"
            object_ids.append(object_id)
            assert renderer.ensure_object(RenderObject(id=object_id, payload=_make_mesh())) is True
        for object_id in object_ids:
            assert renderer.remove_object(object_id) is True

    def test_request_redraw_no_crash(self, renderer):
        renderer.request_redraw()

    def test_update_renderer_no_crash(self, renderer):
        renderer.update_renderer()
