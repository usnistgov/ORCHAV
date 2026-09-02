from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel as _RealQLabel

from visualizer.src.backends.pygfx_scene_helpers import (
    _writable_pygfx_buffer_array,
    apply_texture_policy_to_material_payload,
    build_lines_geometry,
    build_points_geometry,
    payload_to_pygfx_lines,
)
from visualizer.src.metrics.mpc_canon import CanonicalStepData
from visualizer.src.pipeline.core import ViewModel
from visualizer.src.renderers.pygfx.canvas import (
    build_pygfx_effect_passes,
    create_wgpu_renderer,
    display_renderer_kwargs,
    export_renderer_kwargs,
)
from visualizer.src.renderers.pygfx.capture import PygfxCaptureMixin
from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin
from visualizer.src.renderers.pygfx.lighting import (
    DEFAULT_PYGFX_IBL_INTENSITY,
    PygfxIBLManager,
)
from visualizer.src.renderers.pygfx.mpc import (
    INTERACTION_MARKER_SPECS,
    UNKNOWN_INTERACTION_MARKER_SPEC,
    PygfxMpcMixin,
    interaction_marker_spec,
)
from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin
from visualizer.src.renderers.pygfx.picking import PygfxPickingMixin
from visualizer.src.renderers.pygfx.rf_xray import (
    RF_XRAY_BOUNCES_NAME,
    RF_XRAY_COLORBAR_OVERLAY_ID,
    RF_XRAY_LEGEND_OVERLAY_ID,
    PygfxRFXRayMixin,
)
from visualizer.src.renderers.pygfx.scene_controls import PygfxSceneControlsMixin
from visualizer.src.renderers.pygfx.surface_overlays import PygfxSurfaceOverlayMixin
from visualizer.src.scene.surface_payloads import BeamformingSurface
from visualizer.src.services.rf_xray_analysis_service import (
    RFXRayAnalysisSnapshot,
    RFXRayLegendEntry,
    RFXRayMaterialUsage,
)
from visualizer.src.state import MpcVisibility
from visualizer.src.types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
    SurfaceColorSource,
)


def test_mpc_staging_buffer_is_detached_and_writable_on_first_allocation():
    owner = SimpleNamespace()

    staging = PygfxMpcMixin._ensure_buffer(owner, "_staging", (4, 3), np.float32)

    assert staging.flags.c_contiguous
    assert staging.flags.owndata
    assert staging.flags.writeable
    np.take(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.array([0, 1, 2, 3]),
        axis=0,
        out=staging,
    )


def test_mpc_staging_buffer_detaches_from_read_only_reused_buffer():
    owner = SimpleNamespace()
    first = PygfxMpcMixin._ensure_buffer(owner, "_staging", (4, 3), np.float32)
    first.setflags(write=False)

    reused = PygfxMpcMixin._ensure_buffer(owner, "_staging", (4, 3), np.float32)

    assert reused is not first
    assert reused.flags.c_contiguous
    assert reused.flags.owndata
    assert reused.flags.writeable
    np.take(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.array([0, 1, 2, 3]),
        axis=0,
        out=reused,
    )


def test_mpc_capacity_payload_keeps_reused_staging_buffers_writable():
    owner = SimpleNamespace()
    points = PygfxMpcMixin._ensure_buffer(
        owner,
        "_mpc_segment_points_buf",
        (4, 3),
        np.float32,
    )
    colors = PygfxMpcMixin._ensure_buffer(
        owner,
        "_mpc_segment_colors_buf",
        (4, 3),
        np.float32,
    )
    payload = LineSetPayload(
        points=points,
        lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
        colors=colors,
    )

    assert payload.points.flags.writeable is False
    assert payload.colors is not None
    assert payload.colors.flags.writeable is False
    assert points.flags.writeable
    assert colors.flags.writeable

    reused_points = PygfxMpcMixin._ensure_buffer(
        owner,
        "_mpc_segment_points_buf",
        (4, 3),
        np.float32,
    )
    reused_colors = PygfxMpcMixin._ensure_buffer(
        owner,
        "_mpc_segment_colors_buf",
        (4, 3),
        np.float32,
    )

    assert reused_points is points
    assert reused_colors is colors
    np.take(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.array([0, 1, 2, 3]),
        axis=0,
        out=reused_points,
    )
    reused_colors[0::2] = 1.0
    reused_colors[1::2] = 0.5


def test_standalone_pygfx_line_builder_detaches_immutable_payload_buffers():
    payload = LineSetPayload(
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        lines=np.empty((0, 2), dtype=np.int32),
        colors=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )
    gfx = SimpleNamespace(
        Geometry=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    )

    geometry = build_lines_geometry(gfx, payload)

    assert geometry.kwargs["positions"].flags.owndata
    assert geometry.kwargs["positions"].flags.writeable
    assert geometry.kwargs["colors"].flags.owndata
    assert geometry.kwargs["colors"].flags.writeable
    assert not np.shares_memory(geometry.kwargs["positions"], payload.points)
    assert not np.shares_memory(geometry.kwargs["colors"], payload.colors)


def test_standalone_pygfx_point_builder_detaches_immutable_payload_buffers():
    payload = PointCloudPayload(
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
        colors=np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    gfx = SimpleNamespace(
        Geometry=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    )

    geometry = build_points_geometry(gfx, payload)

    assert geometry.kwargs["positions"].flags.owndata
    assert geometry.kwargs["positions"].flags.writeable
    assert geometry.kwargs["colors"].flags.owndata
    assert geometry.kwargs["colors"].flags.writeable
    assert not np.shares_memory(geometry.kwargs["positions"], payload.points)
    assert not np.shares_memory(geometry.kwargs["colors"], payload.colors)


def test_standalone_pygfx_buffer_helper_reuses_owned_writable_array():
    owned = np.empty((4, 3), dtype=np.float32)

    assert _writable_pygfx_buffer_array(owned, np.float32) is owned


class _FakeBloomPass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeScene:
    def __init__(self):
        self.environment = None
        self.children = []

    def add(self, child):
        self.children.append(child)

    def remove(self, child):
        self.children.remove(child)


class _FakeBackgroundSkyboxMaterial:
    def __init__(self, texture=None, *, map=None, map_interpolation=None):
        self.texture = texture if texture is not None else map
        self.map_interpolation = map_interpolation


class _FakeBackground:
    def __init__(self, _, material):
        self.material = material


class _FakeTextureMap:
    def __init__(self, texture):
        self.texture = texture


class _FakeRenderer:
    def __init__(self, target, **kwargs):
        self.target = target
        self.kwargs = kwargs
        self.effect_passes = ()


class _FakeGfx:
    Scene = _FakeScene
    Background = _FakeBackground
    BackgroundSkyboxMaterial = _FakeBackgroundSkyboxMaterial
    TextureMap = _FakeTextureMap

    class renderers:
        WgpuRenderer = _FakeRenderer


class _FakeCoverageMaterial:
    def __init__(self, color=(0.8, 0.8, 0.8, 1.0), color_mode="auto"):
        self.color = color
        self.color_mode = color_mode
        self.opacity = color[3]
        self.side = "front"
        self.alpha_mode = "auto"
        self.depth_write = True
        self.depth_test = False
        self.env_map = "inherited"
        self.env_map_intensity = 1.0


class _FakeCoverageGfx:
    MeshBasicMaterial = _FakeCoverageMaterial


class _FakeMeshBasicMaterial:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.color = kwargs.get("color")
        self.opacity = None
        self.color_mode = kwargs.get("color_mode")
        self.map = None


class _FakeMeshStandardMaterial(_FakeMeshBasicMaterial):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.color = kwargs.get("color")
        self.opacity = None
        self.color_mode = kwargs.get("color_mode")
        self.map = None
        self.roughness = None
        self.metalness = None
        self.metallic = None


class _FakeMeshPhysicalMaterial(_FakeMeshStandardMaterial):
    pass


class _FakeMeshMaterialGfx:
    MeshBasicMaterial = _FakeMeshBasicMaterial
    MeshStandardMaterial = _FakeMeshStandardMaterial
    MeshPhysicalMaterial = _FakeMeshPhysicalMaterial
    MeshPhongMaterial = _FakeMeshStandardMaterial


class _TooltipOwner(PygfxPickingMixin):
    def __init__(self, packet):
        self.last_frame_packet = packet


class _FakeRFXRayRenderer(PygfxRFXRayMixin):
    def __init__(self):
        self.visualizer = SimpleNamespace()
        self._objects = {"scene:wall::mesh": object()}
        self._kinds = {"scene:wall::mesh": "mesh"}
        self._materials = {"scene:wall::mesh": MaterialPayload(base_color=(0.2, 0.3, 0.4, 1.0))}
        self._geometry_color_sources = {
            "scene:wall::mesh": SurfaceColorSource.VERTEX,
        }
        self._point_size = 5.0
        self._line_width = 2.0
        self.ensured = {}
        self.removed = []
        self.hud = {}
        self._initialize_rf_xray_state()

    def has_named_geometry(self, name):
        return name in self._objects or name in self.ensured

    def set_named_material(self, name, material):
        self._materials[name] = material
        return True

    def ensure_named_geometry(self, name, payload, material=None, visible=True):
        self.ensured[name] = (payload, material, visible)
        return True

    def remove_named_geometry(self, name):
        self.removed.append(name)
        self.ensured.pop(name, None)
        return True

    def _colorbar_overlay_html(self, label, value_range):
        return f"{label}:{value_range[0]:.1f}-{value_range[1]:.1f}"

    def _set_hud_overlay(self, overlay_id, **kwargs):
        self.hud[overlay_id] = kwargs

    def _clear_hud_overlay(self, overlay_id):
        self.hud.pop(overlay_id, None)


class _FakeBeamformingMaterial:
    def __init__(self, owner):
        self._owner = owner
        self.color_mode = "uniform"
        self._side = "front"
        self.roughness = None
        self.metalness = None

    @property
    def side(self):
        return self._side

    @side.setter
    def side(self, value):
        if self._owner.side_failures:
            self._owner.side_failures -= 1
            raise ValueError("native side update failed")
        self._side = value


class _FakeBeamformingOverlayRenderer(PygfxSurfaceOverlayMixin):
    def __init__(self):
        self.visualizer = SimpleNamespace()
        self._applied_beamforming_surfaces = {}
        self._beamforming_owned_names = set()
        self.ensure_calls = []
        self.material_calls = []
        self.removed = []
        self.redraws = 0
        self.ensure_failures = 0
        self.material_failures = 0
        self.side_failures = 0
        self.remove_failures = {}
        self._names = set()
        self._objects = {}

    def ensure_named_geometry(self, name, payload, **kwargs):
        self.ensure_calls.append((name, payload, dict(kwargs)))
        if self.ensure_failures:
            self.ensure_failures -= 1
            return False
        self._names.add(name)
        self._objects.setdefault(
            name,
            SimpleNamespace(material=_FakeBeamformingMaterial(self)),
        )
        return True

    def set_named_material(self, name, material):
        self.material_calls.append((name, material))
        if self.material_failures:
            self.material_failures -= 1
            return False
        native = self._objects[name].material
        native.roughness = material.roughness
        native.metalness = material.metallic
        return True

    def request_redraw(self):
        self.redraws += 1

    def has_named_geometry(self, name):
        return name in self._names

    def remove_named_geometry(self, name):
        self.removed.append(name)
        failures = self.remove_failures.get(name, 0)
        if failures:
            self.remove_failures[name] = failures - 1
            return False
        self._names.discard(name)
        self._objects.pop(name, None)
        return True


class _FakeMpcVisibilityRenderer:
    """Minimal pygfx MPC owner for visibility/removal helper tests."""

    def __init__(self) -> None:
        self.visualizer = SimpleNamespace(app_state=SimpleNamespace())
        self.names = {"mpc_lines", "mpc_points"}
        self.removed: list[str] = []
        self.visibility: list[tuple[str, bool]] = []
        self.remove_failures: dict[str, int] = {}
        self._mpc_lines_source_sig = object()
        self._mpc_points_source_sig = object()

    def has_named_geometry(self, name: str) -> bool:
        return name in self.names

    def remove_named_geometry(self, name: str) -> bool:
        remaining_failures = self.remove_failures.get(name, 0)
        if remaining_failures:
            self.remove_failures[name] = remaining_failures - 1
            return False
        self.names.discard(name)
        self.removed.append(name)
        return True

    def set_named_visibility(self, name: str, visible: bool) -> bool:
        self.visibility.append((name, visible))
        return True

    def _remove_mpc_geometry_if_present(self, name: str) -> bool:
        if not self.has_named_geometry(name):
            return True
        return self.remove_named_geometry(name)

    def _record_profile_metric(self, *_args) -> None:
        pass

    def _set_marker_legend_visible(self, _visible: bool) -> None:
        pass


class _FakeUnifiedMpcLineRenderer(PygfxMpcMixin):
    """Minimal owner that captures the production unified MPC line payload."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.payloads: dict[str, LineSetPayload] = {}
        self.pick_metadata: dict[str, dict] = {}
        self._mpc_lines_source_sig = None

    def has_named_geometry(self, name: str) -> bool:
        return name in self.names

    def remove_named_geometry(self, name: str) -> bool:
        self.names.discard(name)
        self.payloads.pop(name, None)
        return True

    def ensure_named_geometry(self, name: str, payload: LineSetPayload, **_kwargs) -> bool:
        self.names.add(name)
        self.payloads[name] = payload
        return True

    def _register_pick_metadata(self, name: str, metadata: dict) -> None:
        self.pick_metadata[name] = metadata

    def _record_profile_metric(self, *_args) -> None:
        pass

    def _record_profile_array_bytes(self, *_args) -> None:
        pass

    def _record_profile_bytes(self, *_args) -> None:
        pass


class _FakePointsMaterial:
    pass


class _FakePointsMarkerMaterial:
    def __init__(self, **kwargs):
        self.size = kwargs.get("size")
        self.marker_mode = kwargs.get("marker_mode")


class _FakeMarkerBuffer:
    def __init__(self, data):
        self.data = np.array(data, copy=True)
        self.update_count = 0

    def update_full(self):
        self.update_count += 1


class _FakeMpcMarkerGfx:
    PointsMaterial = _FakePointsMaterial
    PointsMarkerMaterial = _FakePointsMarkerMaterial

    def __init__(self):
        self.buffer_creations = 0

    def Buffer(self, data):
        self.buffer_creations += 1
        return _FakeMarkerBuffer(data)


class _FakeMpcMarkerRenderer(PygfxMpcMixin):
    """Minimal owner that exercises the marker-buffer cache contract."""

    def __init__(self):
        self._gfx = _FakeMpcMarkerGfx()
        self._objects = {
            "mpc_points": SimpleNamespace(
                material=_FakePointsMaterial(),
                geometry=SimpleNamespace(markers=None),
            )
        }
        self.visualizer = SimpleNamespace(app_state=SimpleNamespace(show_mpc_type_markers=True))
        self._point_size = 5.0
        self._mpc_marker_cache_key = None
        self._mpc_marker_codes_buf = None
        self.push_count = 0
        self.legend_visibility = []

    def _apply_clipping_to_material(self, _material):
        pass

    def _apply_named_visual_overrides(self, _name):
        pass

    def _build_points_material(self, *, has_vertex_colors):
        del has_vertex_colors
        return _FakePointsMaterial()

    def _push_buffer(self, buffer, data, *, label):
        assert label == "mpc_markers"
        self.push_count += 1
        buffer.data[:] = data
        buffer.update_full()

    def _set_marker_legend_visible(self, visible):
        self.legend_visibility.append(bool(visible))


def _make_pygfx_beamforming_surface(
    surface_id: str = "beamforming:tx_0_pair_0:mesh",
    *,
    offset: float = 0.0,
) -> BeamformingSurface:
    return BeamformingSurface(
        id=surface_id,
        payload=MeshPayload(
            vertices=np.asarray(
                [
                    [offset, 0.0, 0.0],
                    [offset + 0.5, 0.0, 0.0],
                    [offset, 0.5, 0.0],
                ],
                dtype=np.float32,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
            normals=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
            vertex_colors=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            color_source=SurfaceColorSource.VERTEX,
        ),
    )


def test_pygfx_beamforming_create_noop_and_payload_update():
    renderer = _FakeBeamformingOverlayRenderer()
    surface = _make_pygfx_beamforming_surface()
    packet = SimpleNamespace(beamforming_meshes=(surface,))

    assert renderer._apply_beamforming(packet) is True

    name, payload, kwargs = renderer.ensure_calls[0]
    native_material = renderer._objects[surface.id].material
    assert name == surface.id
    assert payload is surface.payload
    assert "preserve_vertex_colors" not in kwargs
    assert payload.color_source is SurfaceColorSource.VERTEX
    assert kwargs["visible"] is True
    assert native_material.color_mode == "vertex"
    assert native_material.side == "both"
    assert native_material.roughness == pytest.approx(0.5)
    assert native_material.metalness == pytest.approx(0.0)

    assert renderer._apply_beamforming(packet) is True
    assert len(renderer.ensure_calls) == 1
    assert len(renderer.material_calls) == 1

    replacement = _make_pygfx_beamforming_surface(offset=2.0)
    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=(replacement,))) is True
    assert len(renderer.ensure_calls) == 2
    assert renderer.ensure_calls[-1][1] is replacement.payload
    assert renderer._applied_beamforming_surfaces[surface.id].payload is replacement.payload


@pytest.mark.parametrize("failure", ["ensure", "material", "side"])
def test_pygfx_beamforming_failed_native_update_is_retried(failure):
    renderer = _FakeBeamformingOverlayRenderer()
    surface = _make_pygfx_beamforming_surface()
    setattr(renderer, f"{failure}_failures", 1)
    packet = SimpleNamespace(beamforming_meshes=(surface,))

    assert renderer._apply_beamforming(packet) is False
    assert surface.id not in renderer._applied_beamforming_surfaces

    assert renderer._apply_beamforming(packet) is True
    assert renderer._applied_beamforming_surfaces[surface.id].payload is surface.payload


def test_pygfx_beamforming_failed_stale_removal_is_retried():
    renderer = _FakeBeamformingOverlayRenderer()
    kept = _make_pygfx_beamforming_surface("beamforming:kept:mesh")
    stale = _make_pygfx_beamforming_surface("beamforming:stale:mesh", offset=2.0)
    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=(kept, stale)))
    renderer.remove_failures[stale.id] = 1
    packet = SimpleNamespace(beamforming_meshes=(kept,))

    assert renderer._apply_beamforming(packet) is False
    assert stale.id in renderer._applied_beamforming_surfaces
    assert renderer.has_named_geometry(stale.id)

    assert renderer._apply_beamforming(packet) is True
    assert stale.id not in renderer._applied_beamforming_surfaces
    assert not renderer.has_named_geometry(stale.id)


def test_pygfx_partial_beamforming_material_failure_is_removed_when_hidden():
    renderer = _FakeBeamformingOverlayRenderer()
    surface = _make_pygfx_beamforming_surface()
    renderer.material_failures = 1

    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=(surface,))) is False
    assert surface.id in renderer._beamforming_owned_names
    assert surface.id not in renderer._applied_beamforming_surfaces
    assert renderer.has_named_geometry(surface.id)

    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=())) is True
    assert surface.id not in renderer._beamforming_owned_names
    assert not renderer.has_named_geometry(surface.id)


def test_pygfx_identical_packet_path_retries_beamforming():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    packet = object()
    calls: list[str] = []
    owner = SimpleNamespace(
        _initialized=True,
        last_frame_packet=packet,
        _apply_coverage_data=lambda _packet: (calls.append("coverage") or True),
        _apply_beamforming=lambda _packet: (calls.append("beamforming") or True),
        _apply_rf_xray_overlay=lambda _packet: False,
        request_redraw=lambda: calls.append("redraw"),
    )

    assert PygfxRenderer.apply_frame(owner, packet) is True

    assert calls == ["coverage", "beamforming"]


def test_pygfx_apply_frame_retries_failed_overlay_before_committing_packet():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    old_packet = SimpleNamespace(stats_text="old")
    new_packet = SimpleNamespace(stats_text="new")
    coverage_results = iter((False, True))
    coverage_calls: list[object] = []

    def apply_coverage(_old, packet) -> bool:
        coverage_calls.append(packet)
        return next(coverage_results)

    owner = SimpleNamespace(
        _initialized=True,
        last_frame_packet=old_packet,
        _apply_mpc_lines=lambda _packet: True,
        _apply_mpc_points=lambda _packet: True,
        _apply_coverage_data_diff=apply_coverage,
        _apply_rf_xray_overlay=lambda _packet: False,
        _apply_unsupported_features=lambda _packet: True,
        _apply_stats_diff=lambda _old, _new: None,
        _record_profile_metric=lambda *_args: None,
        _ensure_ground_grid_current=lambda: None,
        _update_mpc_hud_overlays=lambda _packet: None,
    )

    assert PygfxRenderer.apply_frame(owner, new_packet) is False
    assert owner.last_frame_packet is old_packet

    assert PygfxRenderer.apply_frame(owner, new_packet) is True
    assert coverage_calls == [new_packet, new_packet]
    assert owner.last_frame_packet is new_packet


def test_pygfx_apply_frame_retries_failed_mpc_before_committing_packet():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    old_packet = SimpleNamespace(stats_text="old")
    new_packet = SimpleNamespace(stats_text="new")
    line_results = iter((False, True))
    line_calls: list[object] = []

    def apply_lines(packet) -> bool:
        line_calls.append(packet)
        return next(line_results)

    owner = SimpleNamespace(
        _initialized=True,
        last_frame_packet=old_packet,
        _apply_mpc_lines=apply_lines,
        _apply_mpc_points=lambda _packet: True,
        _apply_coverage_data_diff=lambda _old, _new: True,
        _apply_rf_xray_overlay=lambda _packet: False,
        _apply_unsupported_features=lambda _packet: True,
        _apply_stats_diff=lambda _old, _new: None,
        _record_profile_metric=lambda *_args: None,
        _ensure_ground_grid_current=lambda: None,
        _update_mpc_hud_overlays=lambda _packet: None,
    )

    assert PygfxRenderer.apply_frame(owner, new_packet) is False
    assert owner.last_frame_packet is old_packet

    assert PygfxRenderer.apply_frame(owner, new_packet) is True
    assert line_calls == [new_packet, new_packet]
    assert owner.last_frame_packet is new_packet


def test_pygfx_native_remove_failure_preserves_bookkeeping_for_retry():
    name = "beamforming:retry:mesh"
    native = object()

    class _FlakyParent:
        def __init__(self):
            self.children = [native]
            self.fail_once = True

        def remove(self, child):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("native scene removal failed")
            self.children.remove(child)

    parent = _FlakyParent()
    owner = SimpleNamespace(
        _name_to_handle={name: 1},
        _handle_to_name={1: name},
        _objects={name: native},
        _kinds={name: "mesh"},
        _topology={name: object()},
        _hidden=set(),
        _edge_geometry_names=set(),
        _geometry_color_sources={name: SurfaceColorSource.VERTEX},
        _materials={name: MaterialPayload()},
        _material_apply_signatures={name: object()},
        _transforms={name: np.eye(4)},
        _positions={name: (0.0, 0.0, 0.0)},
        _geometry_upload_center={name: np.zeros(3)},
        _geometry_texcoords_available={name: False},
        _pick_metadata={name: {}},
        _reverse_objects={id(native): name},
        _normal_line_overlays={},
        _render_object_snapshots={name: object()},
        _uncertain_mesh_index_buffers={name},
        _static_group=None,
        _scene=parent,
        _unregister_label_layout=lambda _name: None,
        _is_scene_mesh_name=lambda _name: False,
        _external_remove_name=lambda _name: None,
    )

    assert PygfxGeometryMixin.remove_named_geometry(owner, name) is False
    assert owner._name_to_handle[name] == 1
    assert owner._objects[name] is native
    assert name in owner._render_object_snapshots
    assert name in owner._uncertain_mesh_index_buffers
    assert parent.children == [native]

    assert PygfxGeometryMixin.remove_named_geometry(owner, name) is True
    assert name not in owner._name_to_handle
    assert name not in owner._objects
    assert name not in owner._render_object_snapshots
    assert name not in owner._uncertain_mesh_index_buffers
    assert parent.children == []


def test_pygfx_clear_retains_failed_beamforming_ownership_for_retry():
    name = "beamforming:retry_reset:mesh"
    removal_attempts = 0
    names = {name: 1}

    def remove_named_geometry(object_name: str) -> bool:
        nonlocal removal_attempts
        removal_attempts += 1
        if removal_attempts == 1:
            return False
        names.pop(object_name, None)
        return True

    owner = SimpleNamespace(
        _ground_grid_visible=False,
        _remove_ground_grid=lambda: None,
        _set_marker_legend_visible=lambda _visible: None,
        _update_mpc_hud_overlays=lambda _packet: None,
        _name_to_handle=names,
        remove_named_geometry=remove_named_geometry,
        _payload_cache={},
        _render_object_snapshots={},
        _dirty_render_object_geometry={"removed"},
        _uncertain_mesh_index_buffers={"removed"},
        _vertex_stream_incompatible_transitions=OrderedDict(
            (("removed", OrderedDict([(((1,), (2,)), None)])),)
        ),
        _vertex_stream_rebuild_names={"removed"},
        _mpc_lines_source_sig=object(),
        _mpc_points_source_sig=object(),
        _last_coverage_signature="coverage",
        _applied_coverage_state=object(),
        _applied_beamforming_surfaces={name: object()},
        _beamforming_owned_names={name},
        _edge_geometry_names=set(),
        _clear_mpc_buffers=lambda: None,
    )

    PygfxSceneControlsMixin.clear(owner)

    assert name in owner._name_to_handle
    assert name in owner._beamforming_owned_names
    assert name in owner._applied_beamforming_surfaces
    assert owner._vertex_stream_incompatible_transitions == OrderedDict()
    assert owner._vertex_stream_rebuild_names == set()

    PygfxSceneControlsMixin.clear(owner)

    assert name not in owner._name_to_handle
    assert name not in owner._beamforming_owned_names
    assert name not in owner._applied_beamforming_surfaces


def test_mpc_tooltip_uses_path_metrics_and_segment_order():
    canon = CanonicalStepData(
        points=np.zeros((6, 3), dtype=np.float32),
        lines=np.array([[4, 5]], dtype=np.int32),
        order=np.array([0, 0, 0, 0, 9, 9], dtype=np.uint8),
        itype=np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8),
        delay=np.zeros(6, dtype=np.float32),
        loss=np.zeros(6, dtype=np.float32),
        path_id=np.array([0, 0, 0, 0, 2, 2], dtype=np.int32),
        path_delays=np.array([1.0, 2.0, 12.5], dtype=np.float32),
        path_losses=np.array([60.0, 70.0, 80.25], dtype=np.float32),
        segment_order=np.array([3], dtype=np.uint8),
        segment_path_id=np.array([2], dtype=np.int32),
    )
    view_model = SimpleNamespace(
        canonical_data=canon,
        mpc_lines=np.array([[4, 5]], dtype=np.int32),
        mpc_line_itypes=np.array([1], dtype=np.uint8),
    )
    event = SimpleNamespace(pick_info={"vertex_index": 0})

    text = _TooltipOwner(view_model)._format_mpc_tooltip("mpc_lines", event)

    assert text == "Specular | order 3 | delay 12.50 ns | loss 80.25 dB"


def test_mpc_tooltip_maps_filtered_segment_to_canonical_point_fallbacks():
    canon = CanonicalStepData(
        points=np.zeros((6, 3), dtype=np.float32),
        lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
        order=np.array([1, 1, 2, 2, 7, 7], dtype=np.uint8),
        itype=np.array([0, 0, 4, 4, 8, 8], dtype=np.uint8),
        delay=np.zeros(6, dtype=np.float32),
        loss=np.zeros(6, dtype=np.float32),
        path_id=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        path_delays=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        path_losses=np.array([50.0, 60.0, 70.0], dtype=np.float32),
    )
    packet = SimpleNamespace(
        canonical_data=canon,
        mpc_lines=np.array([[0, 1]], dtype=np.int32),
        mpc_line_itypes=None,
        segment_mask=np.array([False, False, True]),
    )
    event = SimpleNamespace(pick_info={"vertex_index": 0})

    text = _TooltipOwner(packet)._format_mpc_tooltip("mpc_lines", event)

    assert text == "Diffraction | order 7 | delay 30.00 ns | loss 70.00 dB"


def test_interaction_marker_semantics_cover_canonical_and_unknown_types():
    assert interaction_marker_spec(0) is None
    assert {
        spec.interaction_type: (spec.label, spec.marker_name) for spec in INTERACTION_MARKER_SPECS
    } == {
        1: ("Specular", "circle"),
        2: ("Diffuse", "triangle_up"),
        4: ("Refraction", "diamond"),
        8: ("Diffraction", "plus"),
        99: ("Virtual", "square"),
    }
    assert interaction_marker_spec(1234) is UNKNOWN_INTERACTION_MARKER_SPEC


def test_interaction_marker_codes_hide_los_and_preserve_virtual_and_unknown():
    marker_int = {
        "custom": 10,
        "circle": 11,
        "triangle_up": 12,
        "diamond": 13,
        "plus": 14,
        "square": 15,
        "cross": 16,
    }
    itypes = np.array([0, 1, 2, 4, 8, 99, 255], dtype=np.uint8)

    codes = PygfxMpcMixin._interaction_marker_codes(itypes, marker_int)

    assert codes.tolist() == [10, 11, 12, 13, 14, 15, 16]


def test_mpc_marker_apply_reuses_buffer_and_skips_unchanged_revisions():
    renderer = _FakeMpcMarkerRenderer()
    first_packet = SimpleNamespace(
        mpc_bounce_points=np.zeros((3, 3), dtype=np.float32),
        mpc_bounce_colors=None,
        mpc_bounce_itypes=np.array([1, 2, 99], dtype=np.uint8),
        mpc_point_revision=("mpc-point-v1", "revision-1"),
    )

    renderer._apply_mpc_point_markers(first_packet)
    marker_buffer = renderer._objects["mpc_points"].geometry.markers
    first_codes = marker_buffer.data.copy()

    renderer._apply_mpc_point_markers(first_packet)

    assert renderer._objects["mpc_points"].geometry.markers is marker_buffer
    assert renderer._gfx.buffer_creations == 1
    assert renderer.push_count == 0
    assert marker_buffer.update_count == 0

    changed_packet = SimpleNamespace(
        mpc_bounce_points=first_packet.mpc_bounce_points,
        mpc_bounce_colors=None,
        mpc_bounce_itypes=np.array([8, 4, 255], dtype=np.uint8),
        mpc_point_revision=("mpc-point-v1", "revision-2"),
    )
    renderer._apply_mpc_point_markers(changed_packet)

    assert renderer._objects["mpc_points"].geometry.markers is marker_buffer
    assert renderer._gfx.buffer_creations == 1
    assert renderer.push_count == 1
    assert marker_buffer.update_count == 1
    assert not np.array_equal(marker_buffer.data, first_codes)


def test_interaction_marker_legend_uses_the_same_canonical_semantics():
    html = PygfxOverlayMixin._marker_legend_html()

    assert "Interaction Markers" in html
    for spec in (*INTERACTION_MARKER_SPECS, UNKNOWN_INTERACTION_MARKER_SPEC):
        assert spec.label in html
        assert spec.html_symbol in html
    assert "LoS" not in html
    assert "Reserved" not in html


def test_semantic_mpc_type_hud_legend_tracks_packet_types_and_cache_signature():
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(
        mpc_core=SimpleNamespace(_type_palette=np.zeros((9, 3), dtype=np.float32))
    )
    first_packet = SimpleNamespace(mpc_line_itype_codes=(1, 99, 255))
    second_packet = SimpleNamespace(mpc_line_itype_codes=(8,))

    html = renderer._semantic_mpc_legend_html("mpc_type", first_packet)

    assert "Specular" in html
    assert "Virtual" in html
    assert "#ff9933" in html
    assert "Unknown" in html
    assert "#808080" in html
    assert "Diffuse" not in html
    assert "Diffraction" not in html
    assert renderer._semantic_mpc_legend_cache_signature(
        "mpc_type", first_packet
    ) != renderer._semantic_mpc_legend_cache_signature("mpc_type", second_packet)
    assert renderer._semantic_mpc_legend_html("mpc_type", second_packet).count("Diffraction") == 1
    assert (
        renderer._semantic_mpc_legend_html(
            "mpc_type",
            SimpleNamespace(mpc_line_itype_codes=()),
        )
        == ""
    )


def test_material_color_mode_hud_legend_uses_all_current_colors_without_filter():
    state = SimpleNamespace(
        viewport_hud_enabled=True,
        viewport_hud_mode="compact",
        viewport_hud_show_status=True,
        viewport_hud_show_legends=True,
        viewport_hud_show_filters=True,
        viewport_hud_show_annotations=True,
        use_distinct_material_colors=False,
    )
    rows = [
        (
            f"material_{index}",
            np.array([index / 10.0, 0.25, 0.5], dtype=np.float32),
        )
        for index in range(6)
    ]
    calls = []
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(
        app_state=state,
        mpc_allowed_materials=None,
        mpc_core=SimpleNamespace(
            material_legend_items=lambda use_distinct, *, active_only: (
                calls.append((use_distinct, active_only)) or rows
            )
        ),
    )

    html = renderer._semantic_mpc_legend_html("material")

    assert "MPC Material Legend" in html
    assert "material_0" in html
    assert "material_3" in html
    assert "material_4" not in html
    assert "+2 more" in html
    assert calls == [(False, False)]


def test_material_color_mode_hud_signature_tracks_distinct_colors_and_filter_scope():
    state = SimpleNamespace(
        viewport_hud_enabled=True,
        viewport_hud_mode="compact",
        viewport_hud_show_status=True,
        viewport_hud_show_legends=True,
        viewport_hud_show_filters=True,
        viewport_hud_show_annotations=True,
        use_distinct_material_colors=False,
    )

    def _items(use_distinct, *, active_only):
        value = 0.8 if use_distinct else 0.2
        label = "selected" if active_only else "all"
        return [(label, np.array([value, 0.3, 0.4], dtype=np.float32))]

    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(
        app_state=state,
        mpc_allowed_materials=None,
        mpc_core=SimpleNamespace(material_legend_items=_items),
    )

    base_signature = renderer._semantic_mpc_legend_cache_signature("material")
    state.use_distinct_material_colors = True
    distinct_signature = renderer._semantic_mpc_legend_cache_signature("material")
    renderer.visualizer.mpc_allowed_materials = {"selected"}
    filtered_signature = renderer._semantic_mpc_legend_cache_signature("material")

    assert base_signature != distinct_signature
    assert distinct_signature != filtered_signature
    assert "all" in base_signature[-1][0]
    assert "selected" in filtered_signature[-1][0]


def test_trajectory_hud_and_geometry_share_the_canonical_viridis_lut():
    from visualizer.src.utils.colors import ensure_viridis_lut
    from visualizer.src.utils.trajectory_colors import map_scalar_to_colors

    state = SimpleNamespace(
        viewport_hud_enabled=True,
        viewport_hud_mode="compact",
        viewport_hud_show_status=True,
        viewport_hud_show_legends=True,
        viewport_hud_show_filters=True,
        viewport_hud_show_annotations=True,
    )
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(app_state=state)
    renderer._visible_trajectory_kinds = {"tx"}
    renderer._trajectory_hud_color_mode = "speed"
    renderer._trajectory_hud_scalar_range = (0.0, 1.0)
    captured = {}

    def _colorbar_html(label, value_range, *, lut=None):
        captured["label"] = label
        captured["range"] = value_range
        captured["lut"] = lut
        return "trajectory-colorbar"

    renderer._colorbar_overlay_html = _colorbar_html
    renderer._set_hud_overlay = lambda overlay_id, **kwargs: captured.update(
        overlay_id=overlay_id,
        overlay=kwargs,
    )
    renderer._clear_hud_overlay = lambda _overlay_id: None

    renderer._update_trajectory_hud_overlay()

    canonical_lut = ensure_viridis_lut()
    mapped = map_scalar_to_colors(
        np.asarray([0.0, 1.0], dtype=np.float64),
        scalar_range=(0.0, 1.0),
    )
    assert captured["label"] == "Trajectory Speed (m/frame)"
    assert captured["range"] == (0.0, 1.0)
    assert captured["lut"] is canonical_lut
    np.testing.assert_allclose(mapped, canonical_lut[[0, 255]])
    assert captured["overlay_id"] == "trajectory_colorbar"
    assert captured["overlay"]["html"] == "trajectory-colorbar"


def test_material_filter_swatch_hud_uses_canonical_colors_and_bounds_rows():
    state = SimpleNamespace(
        viewport_hud_enabled=True,
        viewport_hud_mode="compact",
        viewport_hud_show_status=True,
        viewport_hud_show_legends=False,
        viewport_hud_show_filters=True,
        viewport_hud_show_annotations=True,
        use_distinct_material_colors=True,
    )
    rows = [
        (f"material_{index:02d}", np.array([index / 20.0, 0.2, 0.4], dtype=np.float32))
        for index in range(14)
    ]
    calls = []
    mpc_core = SimpleNamespace(
        material_legend_items=lambda use_distinct, *, active_only: (
            calls.append((use_distinct, active_only)) or rows
        )
    )
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(
        app_state=state,
        mpc_allowed_materials={label for label, _color in rows},
        mpc_core=mpc_core,
    )
    overlays = {}
    cleared = []
    renderer._set_hud_overlay = lambda overlay_id, **kwargs: overlays.__setitem__(
        overlay_id, kwargs
    )
    renderer._clear_hud_overlay = lambda overlay_id: cleared.append(overlay_id)

    renderer._update_filter_hud_overlay()

    compact = overlays["material_filter_swatches"]
    assert compact["role"] == "filter_chips"
    assert "material_03" in compact["html"]
    assert "material_04" not in compact["html"]
    assert "+10 more" in compact["html"]
    assert calls[-1] == (True, True)

    state.viewport_hud_mode = "detailed"
    overlays.clear()
    renderer._update_filter_hud_overlay()

    detailed = overlays["material_filter_swatches"]
    assert detailed["role"] == "filters"
    assert "material_11" in detailed["html"]
    assert "material_12" not in detailed["html"]
    assert "+2 more" in detailed["html"]

    renderer.visualizer.mpc_allowed_materials = None
    renderer._update_filter_hud_overlay()
    assert "material_filter_swatches" in cleared


def test_material_filter_swatch_hud_hides_when_colors_cannot_be_resolved():
    state = SimpleNamespace(
        viewport_hud_enabled=True,
        viewport_hud_mode="compact",
        viewport_hud_show_status=True,
        viewport_hud_show_legends=True,
        viewport_hud_show_filters=True,
        viewport_hud_show_annotations=True,
        use_distinct_material_colors=False,
    )
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(
        app_state=state,
        mpc_allowed_materials={"unresolved"},
        mpc_core=SimpleNamespace(material_legend_items=lambda _use_distinct, *, active_only: []),
    )
    cleared = []
    renderer._clear_hud_overlay = lambda overlay_id: cleared.append(overlay_id)
    renderer._set_hud_overlay = lambda *_args, **_kwargs: None

    renderer._update_filter_hud_overlay()

    assert "material_filter_swatches" in cleared


def test_viewport_hud_refresh_reconciles_feature_owned_legends_across_policy_changes():
    state = SimpleNamespace(
        viewport_hud_enabled=True,
        viewport_hud_mode="compact",
        viewport_hud_show_status=True,
        viewport_hud_show_legends=False,
        viewport_hud_show_filters=True,
        viewport_hud_show_annotations=True,
        show_mpc_type_markers=True,
        mpc_visibility=MpcVisibility(),
    )
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(app_state=state)
    renderer.last_frame_packet = SimpleNamespace()
    renderer._hud_suppressed = False
    renderer._mpc_marker_legend_requested = True
    events = []
    renderer._hud_overlay_specs = {
        "optional_sensing_legend": {
            "visible": True,
            "role": "legend",
        }
    }
    renderer._hud_overlay_labels = {
        "optional_sensing_legend": SimpleNamespace(
            setVisible=lambda visible: events.append(("optional_visible", bool(visible)))
        )
    }
    renderer._reposition_hud_overlays = lambda: events.append(("reposition", None))
    renderer._update_mpc_hud_overlays = lambda _packet: None
    renderer._update_trajectory_hud_overlay = lambda: None
    renderer._set_hud_overlay = lambda overlay_id, **_kwargs: events.append(("set", overlay_id))
    renderer._clear_hud_overlay = lambda overlay_id: events.append(("clear", overlay_id))
    renderer._clear_all_hud_overlays = lambda: events.append(("clear_all", None))
    renderer._refresh_rf_xray_hud_overlay = lambda: events.append(("refresh_rf_xray", None))
    renderer._hide_tooltip = lambda: None
    renderer.request_redraw = lambda: None

    renderer.refresh_viewport_hud()
    assert ("clear", "mpc_marker_legend") in events
    assert ("refresh_rf_xray", None) in events
    assert renderer._hud_overlay_specs["optional_sensing_legend"]["visible"] is False
    assert ("optional_visible", False) in events
    assert renderer._mpc_marker_legend_requested is True

    events.clear()
    state.viewport_hud_enabled = False
    renderer.refresh_viewport_hud()
    assert events == [("clear_all", None)]
    assert renderer._mpc_marker_legend_requested is True

    events.clear()
    state.viewport_hud_enabled = True
    state.viewport_hud_show_legends = True
    renderer.refresh_viewport_hud()
    assert ("set", "mpc_marker_legend") in events
    assert ("refresh_rf_xray", None) in events


def test_marker_legend_intent_survives_sparse_frames_and_low_level_hides():
    state = SimpleNamespace(
        viewport_hud_enabled=True,
        viewport_hud_mode="compact",
        viewport_hud_show_status=True,
        viewport_hud_show_legends=True,
        viewport_hud_show_filters=True,
        viewport_hud_show_annotations=True,
        show_mpc_type_markers=True,
        mpc_visibility=MpcVisibility(),
    )
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(app_state=state)
    renderer._mpc_marker_legend_requested = False
    visibility_events = []
    renderer._set_hud_overlay = lambda overlay_id, **_kwargs: visibility_events.append(
        ("show", overlay_id)
    )
    renderer._clear_hud_overlay = lambda overlay_id: visibility_events.append(("hide", overlay_id))

    populated_packet = SimpleNamespace(mpc_bounce_points=np.ones((2, 3)))
    sparse_packet = SimpleNamespace(mpc_bounce_points=np.empty((0, 3)))
    renderer._sync_marker_legend_from_state(populated_packet)
    renderer._set_marker_legend_visible(False)
    renderer._sync_marker_legend_from_state(sparse_packet)

    assert renderer._mpc_marker_legend_requested is True
    assert visibility_events == [
        ("show", "mpc_marker_legend"),
        ("hide", "mpc_marker_legend"),
        ("show", "mpc_marker_legend"),
    ]

    renderer._sync_marker_legend_from_state(None)
    assert renderer._mpc_marker_legend_requested is False
    assert visibility_events[-1] == ("hide", "mpc_marker_legend")


def test_unified_mpc_lines_keep_virtual_and_unknown_interaction_types_visible():
    renderer = _FakeUnifiedMpcLineRenderer()
    packet = SimpleNamespace(
        mpc_visibility=MpcVisibility(),
        mpc_points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mpc_lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
        mpc_colors=None,
        mpc_line_itypes=np.array([1, 99, 255], dtype=np.uint8),
        mpc_line_revision=("virtual-and-unknown",),
    )

    assert renderer._apply_mpc_lines(packet) is True

    assert renderer.names == {"mpc_lines"}
    assert renderer.payloads["mpc_lines"].lines.tolist() == [[0, 1], [2, 3], [4, 5]]


def test_pygfx_mpc_source_signatures_use_frame_packet_revisions():
    view_model = ViewModel(
        tx_positions=np.empty((0, 3), dtype=np.float32),
        rx_positions=np.empty((0, 3), dtype=np.float32),
        tx_orientations=np.empty((0, 3), dtype=np.float32),
        rx_orientations=np.empty((0, 3), dtype=np.float32),
        mpc_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        mpc_lines=np.array([[0, 1]], dtype=np.int32),
        mpc_colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        colorbar=None,
        stats_text="",
        mpc_visibility=MpcVisibility(),
        target_positions=np.empty((0, 3), dtype=np.float32),
        target_orientations=np.empty((0, 3), dtype=np.float32),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
    )

    packet = view_model.to_render_packet()
    assert PygfxMpcMixin._mpc_line_source_signature(packet) == packet.mpc_line_revision
    assert PygfxMpcMixin._mpc_point_source_signature(packet) == packet.mpc_point_revision
    assert not hasattr(PygfxMpcMixin, "_mpc_array_signature")


def test_pygfx_mpc_visibility_never_uses_path_vertices_as_bounces() -> None:
    renderer = _FakeMpcVisibilityRenderer()
    bounces_requested_without_payload = SimpleNamespace(
        mpc_visibility=MpcVisibility(paths=False, bounce_points=True),
        mpc_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        mpc_lines=np.array([[0, 1]], dtype=np.int32),
        mpc_colors=np.ones((1, 3), dtype=np.float32),
        mpc_bounce_points=None,
        mpc_bounce_colors=None,
    )

    PygfxMpcMixin._apply_mpc_lines(renderer, bounces_requested_without_payload)
    PygfxMpcMixin._apply_mpc_points(renderer, bounces_requested_without_payload)

    assert renderer.removed == ["mpc_lines", "mpc_points"]


def test_pygfx_mpc_master_disable_hides_existing_bounce_points() -> None:
    renderer = _FakeMpcVisibilityRenderer()
    disabled = SimpleNamespace(
        mpc_visibility=MpcVisibility(enabled=False),
        mpc_points=np.empty((0, 3), dtype=np.float32),
        mpc_lines=np.empty((0, 2), dtype=np.int32),
        mpc_colors=np.empty((0, 3), dtype=np.float32),
        mpc_bounce_points=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
        mpc_bounce_colors=np.ones((1, 3), dtype=np.float32),
    )

    PygfxMpcMixin._apply_mpc_points(renderer, disabled)

    assert renderer.visibility == [("mpc_points", False)]


def test_pygfx_mpc_native_remove_failure_is_reported_and_retryable() -> None:
    renderer = _FakeMpcVisibilityRenderer()
    renderer.remove_failures["mpc_lines"] = 1
    packet = SimpleNamespace(
        mpc_visibility=MpcVisibility(paths=False),
        mpc_points=np.empty((0, 3), dtype=np.float32),
        mpc_lines=np.empty((0, 2), dtype=np.int32),
        mpc_colors=np.empty((0, 3), dtype=np.float32),
    )
    original_signature = renderer._mpc_lines_source_sig

    assert PygfxMpcMixin._apply_mpc_lines(renderer, packet) is False
    assert renderer._mpc_lines_source_sig is original_signature
    assert "mpc_lines" in renderer.names

    assert PygfxMpcMixin._apply_mpc_lines(renderer, packet) is True
    assert renderer._mpc_lines_source_sig is None
    assert "mpc_lines" not in renderer.names


def test_rf_xray_overlay_applies_material_and_restores_base_material():
    renderer = _FakeRFXRayRenderer()
    base = renderer._materials["scene:wall::mesh"]
    snapshot = RFXRayAnalysisSnapshot(
        enabled=True,
        mode="material_map",
        signature=("on",),
        geometry_colors={"scene:wall::mesh": (1.0, 0.1, 0.0, 0.8)},
    )

    assert renderer._apply_rf_xray_materials(snapshot) is True
    overlaid = renderer._materials["scene:wall::mesh"]
    assert overlaid.base_color == (1.0, 0.1, 0.0, 0.8)
    assert renderer._geometry_color_sources["scene:wall::mesh"] is SurfaceColorSource.MATERIAL

    assert renderer._clear_rf_xray_overlay() is True
    assert renderer._materials["scene:wall::mesh"] is base
    assert renderer._geometry_color_sources["scene:wall::mesh"] is SurfaceColorSource.VERTEX


def test_rf_xray_overlay_removes_stale_bounce_geometry_without_recreating():
    renderer = _FakeRFXRayRenderer()
    renderer.ensured[RF_XRAY_BOUNCES_NAME] = (object(), object(), True)
    snapshot = RFXRayAnalysisSnapshot(
        enabled=True,
        mode="mpc_usage",
        signature=("mpc_usage",),
        geometry_colors={},
        bounce_points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        bounce_colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    renderer.visualizer.rf_xray_analysis_service = SimpleNamespace(
        build_snapshot=lambda _view_model: snapshot
    )

    assert renderer._apply_rf_xray_overlay(SimpleNamespace()) is True
    assert RF_XRAY_BOUNCES_NAME in renderer.removed
    assert RF_XRAY_BOUNCES_NAME not in renderer.ensured


def test_rf_xray_material_map_publishes_material_legend_overlay():
    renderer = _FakeRFXRayRenderer()
    snapshot = RFXRayAnalysisSnapshot(
        enabled=True,
        mode="material_map",
        signature=("material_map",),
        legend_entries=(
            RFXRayLegendEntry(
                material_key="concrete",
                display_name="Concrete",
                color=(0.6, 0.6, 0.6, 1.0),
            ),
            RFXRayLegendEntry(
                material_key="glass",
                display_name="Glass",
                color=(0.2, 0.4, 0.9, 1.0),
            ),
        ),
    )

    renderer._update_rf_xray_hud_overlay(snapshot)

    overlay = renderer.hud[RF_XRAY_LEGEND_OVERLAY_ID]
    assert overlay["visible"] is True
    assert overlay["role"] == "legend"
    assert "RF X-Ray Material Map" in overlay["html"]
    assert "Concrete" in overlay["html"]
    assert "Glass" in overlay["html"]


def test_rf_xray_hud_refresh_reuses_snapshot_despite_signature_cache():
    renderer = _FakeRFXRayRenderer()
    snapshot = RFXRayAnalysisSnapshot(
        enabled=True,
        mode="material_map",
        signature=("material_map",),
        legend_entries=(
            RFXRayLegendEntry(
                material_key="concrete",
                display_name="Concrete",
                color=(0.6, 0.6, 0.6, 1.0),
            ),
        ),
    )
    renderer.visualizer.rf_xray_analysis_service = SimpleNamespace(
        build_snapshot=lambda _packet: snapshot
    )

    renderer._apply_rf_xray_overlay(SimpleNamespace())
    renderer.hud.clear()

    assert renderer._apply_rf_xray_overlay(SimpleNamespace()) is False
    assert renderer.hud == {}

    renderer._refresh_rf_xray_hud_overlay()
    assert "Concrete" in renderer.hud[RF_XRAY_LEGEND_OVERLAY_ID]["html"]


def test_rf_xray_mpc_usage_publishes_relative_colorbar_overlay():
    renderer = _FakeRFXRayRenderer()
    snapshot = RFXRayAnalysisSnapshot(
        enabled=True,
        mode="mpc_usage",
        signature=("mpc_usage",),
        usage=(
            RFXRayMaterialUsage(
                material_key="concrete",
                display_name="Concrete",
                family="building",
                weight=1.0,
                normalized_score=1.0,
            ),
        ),
    )

    renderer._update_rf_xray_hud_overlay(snapshot)

    overlay = renderer.hud[RF_XRAY_COLORBAR_OVERLAY_ID]
    assert overlay["visible"] is True
    assert overlay["role"] == "legend"
    assert overlay["corner"] == "top_right"
    assert "RF X-Ray MPC Usage (relative)" in overlay["html"]

    renderer._update_rf_xray_hud_overlay(
        RFXRayAnalysisSnapshot(enabled=True, mode="material_map", signature=("map",))
    )
    assert RF_XRAY_COLORBAR_OVERLAY_ID not in renderer.hud


def test_rf_xray_material_properties_publishes_selected_property_colorbar():
    renderer = _FakeRFXRayRenderer()
    snapshot = RFXRayAnalysisSnapshot(
        enabled=True,
        mode="material_properties",
        signature=("material_properties",),
        scalar_property="scattering_coefficient",
        scalar_property_label="Scattering coefficient",
        scalar_range=(0.05, 0.4),
        usage=(
            RFXRayMaterialUsage(
                material_key="concrete",
                display_name="Concrete",
                family="building",
                property_value=0.4,
            ),
        ),
    )

    renderer._update_rf_xray_hud_overlay(snapshot)

    overlay = renderer.hud[RF_XRAY_COLORBAR_OVERLAY_ID]
    assert overlay["visible"] is True
    assert "RF X-Ray: Scattering coefficient" in overlay["html"]
    assert "0.1-0.4" in overlay["html"]


def test_canvas_schedule_config_defaults_to_display_refresh(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", raising=False)
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", raising=False)

    kwargs, config = PygfxRenderer._canvas_schedule_config(display_refresh_hz=144.0)

    assert kwargs == {
        "update_mode": "ondemand",
        "max_fps": 144.0,
        "vsync": True,
    }
    assert config == {
        "update_mode": "ondemand",
        "max_fps": 144.0,
        "vsync": True,
        "uses_display_refresh": True,
    }


def test_canvas_schedule_config_ondemand_uses_60_fps_by_default(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", "ondemand")
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", raising=False)

    kwargs, config = PygfxRenderer._canvas_schedule_config()

    assert kwargs == {
        "update_mode": "ondemand",
        "max_fps": 60.0,
        "vsync": True,
    }
    assert config == {**kwargs, "uses_display_refresh": True}


@pytest.mark.parametrize("value", ["0", "-1"])
def test_canvas_schedule_config_max_fps_env_can_select_fastest(monkeypatch, value):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", raising=False)
    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", value)

    kwargs, config = PygfxRenderer._canvas_schedule_config()

    assert kwargs == {
        "update_mode": "fastest",
        "vsync": False,
    }
    assert config == {
        "update_mode": "fastest",
        "max_fps": None,
        "vsync": False,
        "uses_display_refresh": False,
    }


def test_canvas_schedule_config_invalid_mode_uses_ondemand_scheduler(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", "bad-value")
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", raising=False)

    kwargs, config = PygfxRenderer._canvas_schedule_config()

    assert kwargs == {
        "update_mode": "ondemand",
        "max_fps": 60.0,
        "vsync": True,
    }
    assert config == {
        **kwargs,
        "uses_display_refresh": True,
    }


def test_recent_present_cadence_discards_idle_and_playback_state_gaps():
    from visualizer.src.renderers.pygfx.runtime import PygfxRuntimeMixin

    runtime = PygfxRuntimeMixin()
    runtime._recent_present_intervals_s = [1.0 / 60.0]
    runtime._last_present_was_animating = False

    runtime._record_recent_present_interval(3.0, animating=False)
    assert runtime._recent_present_intervals_s == []

    runtime._record_recent_present_interval(1.0 / 60.0, animating=False)
    assert runtime._recent_present_intervals_s == [1.0 / 60.0]

    runtime._record_recent_present_interval(0.25, animating=True)
    assert runtime._recent_present_intervals_s == []

    runtime._record_recent_present_interval(0.2, animating=True)
    assert runtime._recent_present_intervals_s == [0.2]


def test_pygfx_nested_batch_propagates_exception_and_preserves_pending_redraw():
    from visualizer.src.renderers.pygfx.runtime import PygfxRuntimeMixin

    redraws = []
    runtime = PygfxRuntimeMixin()
    runtime._batch_mode = False
    runtime._batch_redraw_pending = False
    runtime._initialized = True
    runtime._qt_window_closed = False
    runtime.update_renderer = lambda: redraws.append("redraw")

    with pytest.raises(RuntimeError, match="nested update failed"):
        with runtime.batch_updates():
            runtime.request_redraw()
            with runtime.batch_updates():
                runtime.request_redraw()
                raise RuntimeError("nested update failed")

    assert runtime._batch_mode is False
    assert runtime._batch_redraw_pending is False
    assert redraws == ["redraw"]


def test_canvas_schedule_config_manual_disables_background_scheduler(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", "manual")
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", raising=False)

    kwargs, config = PygfxRenderer._canvas_schedule_config()

    assert kwargs == {"update_mode": "manual", "vsync": False}
    assert config == {
        "update_mode": "manual",
        "max_fps": None,
        "vsync": False,
        "uses_display_refresh": False,
    }


def test_canvas_schedule_config_invalid_max_fps_falls_back_to_60(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", "ondemand")
    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", "bad-value")

    kwargs, config = PygfxRenderer._canvas_schedule_config()

    assert kwargs == {
        "update_mode": "ondemand",
        "max_fps": 60.0,
        "vsync": True,
    }
    assert config == {**kwargs, "uses_display_refresh": False}


@pytest.mark.parametrize(
    ("platform_name", "value", "expected"),
    [
        ("linux", None, "auto"),
        ("darwin", None, "auto"),
        ("win32", None, "screen"),
        ("linux", "invalid", "auto"),
        ("win32", "invalid", "screen"),
        ("win32", "screen", "screen"),
        ("win32", "bitmap", "bitmap"),
        ("win32", "auto", "auto"),
        ("linux", " BITMAP ", "bitmap"),
    ],
)
def test_canvas_present_method_policy(monkeypatch, platform_name, value, expected):
    import visualizer.src.renderers.pygfx.renderer as renderer_module
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setattr(renderer_module.sys, "platform", platform_name)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    if value is None:
        monkeypatch.delenv("ORCHAV_PYGFX_PRESENT_METHOD", raising=False)
    else:
        monkeypatch.setenv("ORCHAV_PYGFX_PRESENT_METHOD", value)

    assert PygfxRenderer._read_canvas_present_method() == expected


def test_canvas_present_method_uses_bitmap_for_windows_qt_offscreen(monkeypatch):
    import visualizer.src.renderers.pygfx.renderer as renderer_module
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setattr(renderer_module.sys, "platform", "win32")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("ORCHAV_PYGFX_PRESENT_METHOD", raising=False)

    assert PygfxRenderer._read_canvas_present_method() == "bitmap"


def test_filter_supported_canvas_kwargs_respects_constructor_signature():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class LimitedWidget:
        def __init__(self, *, parent=None, max_fps=30.0):
            self.parent = parent
            self.max_fps = max_fps

    class FlexibleWidget:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    kwargs = {"update_mode": "ondemand", "max_fps": 60.0, "vsync": True}

    assert PygfxRenderer._filter_supported_canvas_kwargs(LimitedWidget, kwargs) == {"max_fps": 60.0}
    assert PygfxRenderer._filter_supported_canvas_kwargs(FlexibleWidget, kwargs) == kwargs


def test_create_canvas_widget_uses_display_refresh_scheduler_by_default(monkeypatch):
    import visualizer.src.renderers.pygfx.renderer as renderer_module
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setattr(renderer_module.sys, "platform", "linux")
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", raising=False)
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", raising=False)
    monkeypatch.delenv("ORCHAV_PYGFX_PRESENT_METHOD", raising=False)
    parent = object()

    class FakeWidget:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._container = parent
    renderer._WgpuWidget = FakeWidget

    canvas = renderer._create_canvas_widget()

    assert canvas.kwargs == {
        "parent": parent,
        "update_mode": "ondemand",
        "max_fps": 60.0,
        "vsync": True,
    }
    assert renderer._canvas_update_mode == "ondemand"
    assert renderer._canvas_max_fps == 60.0
    assert renderer._canvas_vsync is True
    assert renderer._canvas_uses_display_refresh is True
    assert renderer._canvas_schedule_kwargs == {
        "update_mode": "ondemand",
        "max_fps": 60.0,
        "vsync": True,
    }
    assert renderer._canvas_schedule_applied is True
    assert renderer._canvas_present_method_requested == "auto"


def test_create_canvas_widget_uses_screen_by_default_on_windows(monkeypatch):
    import visualizer.src.renderers.pygfx.renderer as renderer_module
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setattr(renderer_module.sys, "platform", "win32")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("ORCHAV_PYGFX_PRESENT_METHOD", raising=False)
    parent = object()

    class FakeWidget:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._container = parent
    renderer._WgpuWidget = FakeWidget

    canvas = renderer._create_canvas_widget()

    assert canvas.kwargs["present_method"] == "screen"
    assert renderer._canvas_present_method_requested == "screen"


def test_create_canvas_widget_unlimited_omits_rendercanvas_max_fps(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", raising=False)
    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", "0")
    monkeypatch.setenv("ORCHAV_PYGFX_PRESENT_METHOD", "bitmap")
    parent = object()

    class FakeWidget:
        def __init__(
            self,
            *,
            parent=None,
            update_mode=None,
            max_fps=None,
            vsync=None,
            present_method=None,
        ):
            self.kwargs = {
                "parent": parent,
                "update_mode": update_mode,
                "max_fps": max_fps,
                "vsync": vsync,
                "present_method": present_method,
            }

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._container = parent
    renderer._WgpuWidget = FakeWidget

    canvas = renderer._create_canvas_widget()

    assert canvas.kwargs == {
        "parent": parent,
        "update_mode": "fastest",
        "max_fps": None,
        "vsync": False,
        "present_method": "bitmap",
    }
    assert renderer._canvas_update_mode == "fastest"
    assert renderer._canvas_max_fps is None
    assert renderer._canvas_vsync is False
    assert renderer._canvas_schedule_kwargs == {
        "update_mode": "fastest",
        "vsync": False,
    }
    assert renderer._canvas_present_method_requested == "bitmap"


def test_create_canvas_widget_falls_back_when_scheduler_kwargs_are_rejected(monkeypatch):
    import visualizer.src.renderers.pygfx.renderer as renderer_module
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setattr(renderer_module.sys, "platform", "linux")
    monkeypatch.setenv("ORCHAV_PYGFX_CANVAS_SCHEDULER", "ondemand")
    monkeypatch.delenv("ORCHAV_PYGFX_CANVAS_MAX_FPS", raising=False)
    monkeypatch.delenv("ORCHAV_PYGFX_PRESENT_METHOD", raising=False)
    parent = object()

    class RejectingWidget:
        def __init__(
            self,
            *,
            parent=None,
            update_mode=None,
            max_fps=None,
            vsync=None,
            present_method=None,
        ):
            if (
                update_mode is not None
                or max_fps is not None
                or vsync is not None
                or present_method is not None
            ):
                raise TypeError("scheduler kwargs unsupported")
            self.parent = parent

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._container = parent
    renderer._WgpuWidget = RejectingWidget

    canvas = renderer._create_canvas_widget()

    assert canvas.parent is parent
    assert renderer._canvas_update_mode == "native"
    assert renderer._canvas_max_fps is None
    assert renderer._canvas_vsync is None
    assert renderer._canvas_schedule_kwargs == {}
    assert renderer._canvas_schedule_applied is False
    assert renderer._canvas_present_method_requested == "auto"


def test_create_canvas_widget_auto_omits_present_method(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setenv("ORCHAV_PYGFX_PRESENT_METHOD", "auto")
    parent = object()

    class FlexibleWidget:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._container = parent
    renderer._WgpuWidget = FlexibleWidget

    canvas = renderer._create_canvas_widget()

    assert "present_method" not in canvas.kwargs
    assert renderer._canvas_present_method_requested == "auto"


@pytest.mark.parametrize(
    ("present_to_screen", "expected"),
    [(True, "screen"), (False, "bitmap"), (None, "unresolved")],
)
def test_record_canvas_present_method(present_to_screen, expected):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._canvas = SimpleNamespace(_present_to_screen=present_to_screen)

    renderer._record_canvas_present_method()

    assert renderer._canvas_present_method == expected


def _make_initial_present_renderer():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._redraw_requests = 0
    renderer._draw_callbacks_received = 0
    renderer._render_successes = 0
    renderer._render_failures = 0
    renderer._initial_present_attempted = False
    renderer._initial_present_succeeded = None
    renderer._initial_present_duration_ms = None
    renderer._initial_present_error = None
    renderer._session_generation = 0
    renderer._canvas_draw_callback = None
    return renderer


def test_interactive_canvas_presents_empty_frame_before_returning():
    renderer = _make_initial_present_renderer()
    calls = []

    class Canvas:
        def request_draw(self, callback=None):
            calls.append(("request", callback))

        def force_draw(self):
            calls.append(("force", None))
            renderer._draw_callbacks_received += 1
            renderer._render_successes += 1

    renderer._canvas = Canvas()

    assert renderer._start_canvas_presentation(force_initial_present=True) is True

    assert calls == [("request", renderer._canvas_draw_callback), ("force", None)]
    assert renderer._redraw_requests == 1
    assert renderer._initial_present_attempted is True
    assert renderer._initial_present_succeeded is True
    assert renderer._initial_present_duration_ms is not None
    assert renderer._initial_present_error is None


def test_initial_present_callback_failure_is_recorded_and_retried():
    renderer = _make_initial_present_renderer()
    calls = []

    class Canvas:
        def request_draw(self, callback=None):
            calls.append(("request", callback))

        def force_draw(self):
            calls.append(("force", None))
            renderer._draw_callbacks_received += 1
            renderer._render_failures += 1

    renderer._canvas = Canvas()

    assert renderer._start_canvas_presentation(force_initial_present=True) is False

    assert calls == [
        ("request", renderer._canvas_draw_callback),
        ("force", None),
        ("request", None),
    ]
    assert renderer._initial_present_succeeded is False
    assert renderer._initial_present_error == "draw callback reported a render failure"


def test_unpaintable_interactive_canvas_defers_without_recording_failure():
    renderer = _make_initial_present_renderer()
    calls = []

    class Canvas:
        def request_draw(self, callback=None):
            calls.append(("request", callback))

        def force_draw(self):
            calls.append(("force", None))

    renderer._canvas = Canvas()

    assert renderer._start_canvas_presentation(force_initial_present=True) is False

    assert calls == [
        ("request", renderer._canvas_draw_callback),
        ("force", None),
        ("request", None),
    ]
    assert renderer._redraw_requests == 2
    assert renderer._initial_present_attempted is True
    assert renderer._initial_present_succeeded is None
    assert renderer._initial_present_duration_ms is not None
    assert renderer._initial_present_error is None


def test_cli_driven_canvas_keeps_first_present_deferred():
    renderer = _make_initial_present_renderer()
    calls = []
    renderer._canvas = SimpleNamespace(
        request_draw=lambda callback=None: calls.append(callback),
        force_draw=lambda: calls.append("force"),
    )

    assert renderer._start_canvas_presentation(force_initial_present=False) is False

    assert calls == [renderer._canvas_draw_callback]
    assert renderer._redraw_requests == 1
    assert renderer._initial_present_attempted is False
    assert renderer._initial_present_succeeded is None
    assert renderer._initial_present_duration_ms is None


def test_stale_canvas_draw_callback_cannot_render_new_session():
    renderer = _make_initial_present_renderer()
    callbacks = []
    renderer._canvas = SimpleNamespace(
        request_draw=lambda callback=None: callbacks.append(callback)
    )
    renderer._animate = Mock()
    renderer._session_generation = 4

    renderer._start_canvas_presentation(force_initial_present=False)
    renderer._session_generation = 5
    callbacks[0]()

    renderer._animate.assert_not_called()


def test_failed_initial_present_records_failure_and_queues_retry():
    renderer = _make_initial_present_renderer()
    calls = []

    class Canvas:
        def request_draw(self, callback=None):
            calls.append(("request", callback))

        def force_draw(self):
            calls.append(("force", None))
            raise RuntimeError("surface unavailable")

    renderer._canvas = Canvas()

    assert renderer._start_canvas_presentation(force_initial_present=True) is False

    assert calls == [
        ("request", renderer._canvas_draw_callback),
        ("force", None),
        ("request", None),
    ]
    assert renderer._redraw_requests == 2
    assert renderer._initial_present_attempted is True
    assert renderer._initial_present_succeeded is False
    assert renderer._initial_present_duration_ms is not None
    assert renderer._initial_present_error == "RuntimeError: surface unavailable"


def test_screen_renderer_creation_falls_back_to_bitmap_on_type_error(monkeypatch):
    import visualizer.src.renderers.pygfx.renderer as renderer_module
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    lifecycle_calls = []

    class Canvas:
        def __init__(self, name, present_to_screen):
            self.name = name
            self._present_to_screen = present_to_screen

        def hide(self):
            lifecycle_calls.append((self.name, "hide"))

        def setParent(self, parent):
            lifecycle_calls.append((self.name, "setParent", parent))

        def close(self):
            lifecycle_calls.append((self.name, "close"))

        def deleteLater(self):
            lifecycle_calls.append((self.name, "deleteLater"))

        def setFocusPolicy(self, policy):
            lifecycle_calls.append((self.name, "focus", policy))

        def show(self):
            lifecycle_calls.append((self.name, "show"))

    class Layout:
        def removeWidget(self, widget):
            lifecycle_calls.append(("layout", "remove", widget.name))

        def addWidget(self, widget):
            lifecycle_calls.append(("layout", "add", widget.name))

        def activate(self):
            lifecycle_calls.append(("layout", "activate"))

    class Container:
        def focusPolicy(self):
            return "strong-focus"

        def setFocusProxy(self, widget):
            lifecycle_calls.append(("container", "focus-proxy", widget.name))

    class App:
        def processEvents(self):
            lifecycle_calls.append(("app", "process-events"))

    screen_canvas = Canvas("screen", True)
    bitmap_canvas = Canvas("bitmap", False)
    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._canvas = screen_canvas
    renderer._canvas_widget = screen_canvas
    renderer._container = Container()
    renderer._clear_color = (0.0, 0.0, 0.0, 1.0)
    renderer._canvas_present_method_requested = "screen"
    renderer._canvas_present_fallback_reason = None
    renderer._qt_window_closed = False

    def create_canvas_widget(*, present_method=None):
        assert present_method == "bitmap"
        renderer._canvas_present_method_requested = "bitmap"
        return bitmap_canvas

    renderer._create_canvas_widget = create_canvas_widget

    def create_renderer(_gfx, target, *, clear_color, configure_effects):
        assert clear_color == renderer._clear_color
        assert configure_effects is False
        if target is screen_canvas:
            raise TypeError("screen surface unavailable")
        return "bitmap-renderer"

    monkeypatch.setattr(renderer_module, "create_wgpu_renderer", create_renderer)

    result = renderer._create_interactive_wgpu_renderer(
        object(),
        Layout(),
        app=App(),
    )
    renderer._record_canvas_present_method()

    assert result == "bitmap-renderer"
    assert renderer._canvas is bitmap_canvas
    assert renderer._canvas_widget is bitmap_canvas
    assert renderer._canvas_present_method_requested == "screen"
    assert renderer._canvas_present_method == "bitmap"
    assert renderer._canvas_present_fallback_reason == "TypeError: screen surface unavailable"
    assert getattr(screen_canvas, "_orchav_closed") is True
    assert ("layout", "remove", "screen") in lifecycle_calls
    assert ("layout", "add", "bitmap") in lifecycle_calls
    assert ("bitmap", "show") in lifecycle_calls
    assert ("layout", "activate") in lifecycle_calls
    assert ("container", "focus-proxy", "bitmap") in lifecycle_calls
    assert ("app", "process-events") in lifecycle_calls
    assert lifecycle_calls.index(("layout", "add", "bitmap")) < lifecycle_calls.index(
        ("bitmap", "show")
    )
    assert lifecycle_calls.index(("bitmap", "show")) < lifecycle_calls.index(("layout", "activate"))
    assert lifecycle_calls.index(("layout", "activate")) < lifecycle_calls.index(
        ("app", "process-events")
    )


def test_screen_change_updates_display_refresh_cap():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _Canvas:
        def __init__(self) -> None:
            self.calls = []

        def set_update_mode(self, mode, **kwargs) -> None:
            self.calls.append((mode, kwargs))

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._container = None
    renderer._canvas = _Canvas()
    renderer._canvas_uses_display_refresh = True
    renderer._canvas_refresh_rate_hz = 60.0
    renderer._canvas_max_fps = 60.0
    screen = SimpleNamespace(refreshRate=lambda: 144.0)

    renderer._apply_screen_refresh_rate(screen)

    assert renderer._canvas.calls == [("ondemand", {"min_fps": 0.0, "max_fps": 144.0})]
    assert renderer._canvas_refresh_rate_hz == 144.0
    assert renderer._canvas_max_fps == 144.0


class _LifecycleSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def disconnect(self, callback) -> None:
        self.callbacks.remove(callback)

    def emit(self, *args) -> None:
        for callback in tuple(self.callbacks):
            callback(*args)


def test_stale_destroyed_hooks_cannot_close_a_new_renderer_session():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer(SimpleNamespace())
    host_destroyed = _LifecycleSignal()
    canvas_destroyed = _LifecycleSignal()
    renderer._container = SimpleNamespace(destroyed=host_destroyed)
    renderer._canvas_widget = SimpleNamespace(destroyed=canvas_destroyed)
    renderer._session_generation = 7

    renderer._install_qt_lifecycle_hooks()
    stale_callbacks = tuple(renderer._qt_destroyed_callbacks)

    renderer._session_generation = 8
    renderer._qt_window_closed = False
    for callback in stale_callbacks:
        callback()

    assert renderer._qt_window_closed is False
    renderer._disconnect_qt_lifecycle_hooks()
    assert host_destroyed.callbacks == []
    assert canvas_destroyed.callbacks == []


def test_rejected_app_close_keeps_detached_renderer_session_open():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    close_calls = []
    scheduled = []
    renderer = PygfxRenderer(SimpleNamespace(close=lambda: close_calls.append("close") or False))
    renderer._session_generation = 5
    renderer._schedule_qt_callback = scheduled.append

    renderer._request_app_close_from_render_window(expected_generation=5)
    renderer._request_app_close_from_render_window(expected_generation=5)

    assert len(scheduled) == 1
    assert renderer._qt_window_closed is False
    assert renderer._qt_app_close_requested is True

    scheduled[0]()

    assert close_calls == ["close"]
    assert renderer._qt_window_closed is False
    assert renderer._qt_app_close_requested is False


def test_screen_refresh_hook_disconnects_and_ignores_stale_generation():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    signal = _LifecycleSignal()
    renderer = PygfxRenderer(SimpleNamespace())
    renderer._container = SimpleNamespace(
        windowHandle=lambda: SimpleNamespace(screenChanged=signal)
    )
    renderer._canvas = SimpleNamespace(
        set_update_mode=lambda *_args, **_kwargs: None,
    )
    renderer._canvas_uses_display_refresh = True
    renderer._canvas_refresh_rate_hz = 60.0
    renderer._canvas_max_fps = 60.0
    renderer._session_generation = 3

    renderer._install_screen_refresh_hook()
    stale_callback = renderer._qt_screen_changed_callback
    renderer._session_generation = 4
    stale_callback(SimpleNamespace(refreshRate=lambda: 144.0))

    assert renderer._canvas_refresh_rate_hz == 60.0
    renderer._disconnect_screen_refresh_hook()
    assert signal.callbacks == []


def test_renderer_session_reset_clears_controller_ibl_and_timing_state():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer(SimpleNamespace())
    cached_texture = object()
    renderer._ibl_manager._texture_cache["cached"] = cached_texture
    renderer._ibl_manager._tracked_materials.append(object())
    renderer._ibl_manager._background = object()
    renderer._active_controller_type = "fly"
    renderer._ibl_loaded = True
    renderer._skybox_visible = True
    renderer._qt_window_closed = True
    renderer._render_attempts = 12
    renderer._render_successes = 9
    renderer._recent_present_intervals_s.append(1.0 / 60.0)
    renderer._draw_durations.append(0.25)
    renderer._tick_count = 17

    generation = renderer._begin_renderer_session()

    assert generation == 1
    assert renderer._active_controller_type == "orbit"
    assert renderer._ibl_loaded is False
    assert renderer._skybox_visible is False
    assert renderer._ibl_manager._tracked_materials == []
    assert renderer._ibl_manager._background is None
    assert renderer._ibl_manager._texture_cache["cached"] is cached_texture
    assert renderer._qt_window_closed is False
    assert renderer._render_attempts == 0
    assert renderer._render_successes == 0
    assert renderer._recent_present_intervals_s == []
    assert renderer._draw_durations == []
    assert renderer._tick_count == 0
    assert renderer._canvas_present_method_requested == renderer._implicit_canvas_present_method()


def test_failed_renderer_initialization_releases_partial_embedded_canvas(monkeypatch):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    calls = []

    class _Layout:
        def removeWidget(self, widget) -> None:
            calls.append(("remove", widget))

    class _Widget:
        def __init__(self, name) -> None:
            self.name = name
            self._layout = _Layout()

        def isVisible(self) -> bool:
            return True

        def layout(self):
            return self._layout

        def setFocusProxy(self, widget) -> None:
            calls.append(("focus", widget))

        def close(self) -> None:
            calls.append((self.name, "close"))

        def deleteLater(self) -> None:
            calls.append((self.name, "deleteLater"))

    renderer = PygfxRenderer(SimpleNamespace())
    host = _Widget("host")
    canvas = _Widget("canvas")

    def _fail_after_canvas_creation(*_args, **_kwargs):
        renderer._owns_container = False
        renderer._container = host
        renderer._canvas = canvas
        renderer._canvas_widget = canvas
        renderer._renderer = object()
        raise RuntimeError("wgpu initialization failed")

    monkeypatch.setattr(
        renderer,
        "_initialize_visualizer_session",
        _fail_after_canvas_creation,
    )

    with pytest.raises(RuntimeError, match="wgpu initialization failed"):
        renderer.initialize_visualizer(host_parent=host)

    assert renderer.vis_initialized is False
    assert renderer._container is None
    assert renderer._canvas is None
    assert renderer._renderer is None
    assert ("remove", canvas) in calls
    assert ("focus", None) in calls
    assert ("canvas", "close") in calls
    assert ("canvas", "deleteLater") in calls
    assert ("host", "close") not in calls


def test_renderer_close_clears_scene_specific_layout_and_geometry_caches():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer(SimpleNamespace())
    renderer._label_anchor_groups[(0, 0, 0)] = {"label"}
    renderer._label_anchor_key_by_name["label"] = (0, 0, 0)
    renderer._label_anchor_by_name["label"] = np.zeros(3)
    renderer._geometry_payload_cache_keys[1] = "mesh"
    renderer._normal_line_overlays["mesh"] = object()
    renderer._uncertain_mesh_index_buffers.add("mesh")
    renderer._vertex_stream_array_tokens[1] = (lambda: None, 1)
    renderer._vertex_stream_next_array_token = 1
    renderer._vertex_stream_incompatible_transitions["mesh"] = OrderedDict(
        [((("old",), ("new",)), None)]
    )
    renderer._vertex_stream_rebuild_names.add("mesh")

    renderer.close()

    assert renderer._label_anchor_groups == {}
    assert renderer._label_anchor_key_by_name == {}
    assert renderer._label_anchor_by_name == {}
    assert renderer._geometry_payload_cache_keys == {}
    assert renderer._normal_line_overlays == {}
    assert renderer._uncertain_mesh_index_buffers == set()
    assert renderer._vertex_stream_array_tokens == OrderedDict()
    assert renderer._vertex_stream_next_array_token == 0
    assert renderer._vertex_stream_incompatible_transitions == OrderedDict()
    assert renderer._vertex_stream_rebuild_names == set()


class _FakeLineMaterial:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeLineSegmentMaterial:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAxesMaterial:
    def __init__(self, *, thickness=2.0):
        self.color = (1.0, 1.0, 1.0, 1.0)
        self.color_mode = "vertex"
        self.thickness = thickness
        self.pick_write = False
        self.opacity = 1.0


class _FakeAxesGeometry:
    def __init__(self):
        self.colors = SimpleNamespace(data=np.ones((6, 4), dtype=np.float32))


class _FakeGeometry:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeLine:
    def __init__(self, geometry, material):
        self.geometry = geometry
        self.material = material


class _FakeAxesHelper:
    def __init__(self, *, size=1.0, thickness=2):
        self.size = size
        self.thickness = thickness
        self.visible = True
        self.material = _FakeAxesMaterial(thickness=thickness)
        self.geometry = _FakeAxesGeometry()


class _FakeLineGfx:
    AxesHelper = _FakeAxesHelper
    Geometry = _FakeGeometry
    Line = _FakeLine
    LineMaterial = _FakeLineMaterial
    LineSegmentMaterial = _FakeLineSegmentMaterial


class _FakeCaptureCanvas:
    def request_draw(self):
        return None

    def snapshot(self):
        return np.zeros((2, 2, 4), dtype=np.uint8)


class _FakeCaptureRenderer:
    def __init__(self):
        self.render_calls = []

    def render(self, scene, camera, *, flush=True):
        self.render_calls.append((scene, camera, flush))

    def snapshot(self):
        image = np.zeros((2, 2, 4), dtype=np.uint8)
        image[:, :, 0] = 64
        image[:, :, 3] = 255
        return image


class _CaptureHarness(PygfxCaptureMixin):
    def __init__(self):
        self._initialized = True
        self._canvas = _FakeCaptureCanvas()
        self._renderer = _FakeCaptureRenderer()
        self._scene = object()
        self._camera = object()
        self._height = 2
        self._width = 2
        self._canvas_widget = None
        self._container = None
        self.updated = False
        self.headlight_updates = 0

    def update_renderer(self):
        self.updated = True

    def _update_headlight_pose(self):
        self.headlight_updates += 1


def test_pygfx_default_ibl_intensity_is_subtle_for_material_colors():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _AppState:
        label_screen_space = True

    class _Viz:
        app_state = _AppState()

    renderer = PygfxRenderer(_Viz())

    assert DEFAULT_PYGFX_IBL_INTENSITY == pytest.approx(2000.0)
    assert renderer.get_ibl_intensity() == pytest.approx(2000.0)


def test_pygfx_constructor_does_not_create_raw_wgpu_prewarm_thread():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _AppState:
        label_screen_space = True

    renderer = PygfxRenderer(SimpleNamespace(app_state=_AppState()))

    assert not hasattr(renderer, "_wgpu_prewarm_thread")
    assert not hasattr(renderer, "_start_wgpu_prewarm")


def test_pygfx_runtime_stats_report_canvas_presentation():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _AppState:
        label_screen_space = True

    class _Viz:
        app_state = _AppState()

    renderer = PygfxRenderer(_Viz())
    renderer._canvas = SimpleNamespace()
    renderer._renderer = SimpleNamespace()
    renderer._canvas_present_method_requested = "screen"
    renderer._canvas_present_method = "bitmap"
    renderer._canvas_present_fallback_reason = "TypeError: screen surface unavailable"
    renderer._initial_present_attempted = True
    renderer._initial_present_succeeded = False
    renderer._initial_present_duration_ms = 12.3456
    renderer._initial_present_error = "RuntimeError: initial draw failed"

    stats = renderer.get_runtime_stats()

    assert stats["canvas_present_method_requested"] == "screen"
    assert stats["canvas_present_method"] == "bitmap"
    assert stats["canvas_present_fallback_reason"] == "TypeError: screen surface unavailable"
    assert stats["initial_present_attempted"] is True
    assert stats["initial_present_succeeded"] is False
    assert stats["initial_present_duration_ms"] == 12.346
    assert stats["initial_present_error"] == "RuntimeError: initial draw failed"


def test_pygfx_capture_prefers_renderer_snapshot_over_black_canvas():
    capture = _CaptureHarness()
    capture._canvas.snapshot = lambda: pytest.fail(
        "canvas fallbacks must not run after a valid renderer snapshot"
    )
    capture._canvas.draw = lambda: pytest.fail(
        "canvas fallbacks must not run after a valid renderer snapshot"
    )
    capture._canvas.force_draw = lambda: pytest.fail(
        "canvas fallbacks must not run after a valid renderer snapshot"
    )
    capture._canvas.get_context = lambda *_args: pytest.fail(
        "canvas fallbacks must not run after a valid renderer snapshot"
    )

    image = capture.export_screenshot_to_array()

    assert capture.updated is False
    assert capture.headlight_updates == 1
    assert capture._renderer.render_calls == [(capture._scene, capture._camera, True)]
    assert image.shape == (2, 2, 3)
    assert int(image.max()) == 64


def test_pygfx_capture_stops_at_first_visible_canvas_fallback():
    capture = _CaptureHarness()
    events = []
    fallback_minimap_states = []
    redraws = []
    capture._minimap_enabled = True
    capture._renderer.snapshot = lambda: np.zeros((2, 2, 4), dtype=np.uint8)
    capture.update_renderer = lambda: fallback_minimap_states.append(capture._minimap_enabled)
    capture.request_redraw = lambda: redraws.append(True)

    def _canvas_snapshot():
        events.append("snapshot")
        image = np.zeros((2, 2, 4), dtype=np.uint8)
        image[:, :, 1] = 72
        image[:, :, 3] = 255
        return image

    capture._canvas.snapshot = _canvas_snapshot
    capture._canvas.draw = lambda: pytest.fail(
        "later canvas fallbacks must not run after visible pixels are found"
    )
    capture._canvas.force_draw = lambda: pytest.fail(
        "later canvas fallbacks must not run after visible pixels are found"
    )
    capture._canvas.get_context = lambda *_args: pytest.fail(
        "context fallbacks must not run after visible pixels are found"
    )

    image = capture.export_screenshot_to_array()

    assert events == ["snapshot"]
    assert fallback_minimap_states == [False]
    assert capture._minimap_enabled is True
    assert redraws == [True]
    assert int(image[:, :, 1].max()) == 72


def test_pygfx_capture_composites_hud_over_renderer_snapshot():
    capture = _CaptureHarness()
    captured_bases = []

    def _composite(base):
        captured_bases.append(base.copy())
        image = base.copy()
        image[:, :, 1] = 96
        return image

    def _forbid_qt_grab():
        raise AssertionError("HUD capture must not use QWidget.grab() as its base")

    capture._composite_visible_qt_overlays = _composite
    capture._capture_qt_widget_frame = _forbid_qt_grab

    image = capture.export_screenshot_to_array(include_hud=True)

    assert len(captured_bases) == 1
    assert int(captured_bases[0][:, :, 0].max()) == 64
    assert int(image[:, :, 1].max()) == 96
    assert capture._renderer.render_calls == [(capture._scene, capture._camera, True)]


def test_pygfx_capture_renders_gpu_minimap_only_when_hud_is_requested():
    capture = _CaptureHarness()
    capture._minimap_enabled = True
    events = []

    def _render(scene, camera, *, flush=True):
        assert scene is capture._scene
        assert camera is capture._camera
        events.append(("scene", bool(flush)))

    def _snapshot():
        events.append(("snapshot",))
        image = np.zeros((2, 2, 4), dtype=np.uint8)
        image[:, :, 0] = 64
        image[:, :, 3] = 255
        return image

    def _render_minimap():
        events.append(("minimap",))
        return True

    capture._renderer.render = _render
    capture._renderer.snapshot = _snapshot
    capture._render_minimap = _render_minimap
    capture._composite_visible_qt_overlays = lambda base: base

    capture.export_screenshot_to_array(include_hud=False)
    assert events == [("scene", True), ("snapshot",)]
    assert capture._minimap_enabled is True

    events.clear()
    capture.export_screenshot_to_array(include_hud=True)
    assert events == [("scene", False), ("minimap",), ("snapshot",)]
    assert capture._minimap_enabled is True


def test_pygfx_qt_widget_capture_reads_current_pyside6_memoryview():
    from PySide6.QtGui import QColor, QImage

    source = QImage(3, 2, QImage.Format_RGBA8888)
    source.fill(QColor(12, 34, 56, 255))

    class _Pixmap:
        def isNull(self):
            return False

        def toImage(self):
            return source

    capture = _CaptureHarness()
    capture._canvas_widget = SimpleNamespace(grab=lambda: _Pixmap())

    image = capture._capture_qt_widget_frame()

    assert image is not None
    assert image.shape == (2, 3, 3)
    assert image[0, 0].tolist() == [12, 34, 56]


def test_pygfx_hud_compositor_does_not_dispatch_pending_qt_events(qapp):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QWidget

    container = QWidget()
    container.resize(4, 3)
    capture = _CaptureHarness()
    capture._container = container
    capture._hud_overlay_labels = {}
    capture._tooltip_label = None
    pending = []
    QTimer.singleShot(0, lambda: pending.append("timer"))

    try:
        image = capture._composite_visible_qt_overlays(np.zeros((3, 4, 3), dtype=np.uint8))
        assert image.shape == (3, 4, 3)
        assert pending == []
    finally:
        qapp.processEvents()
        container.close()

    assert pending == ["timer"]


def test_pygfx_hud_capture_composites_real_qt_widgets_at_gpu_scale():
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    container = QWidget()
    container.resize(6, 4)

    hud = _RealQLabel(container)
    hud.setStyleSheet("background: rgba(255, 0, 0, 128);")
    hud.setGeometry(2, 1, 2, 2)
    hud.show()

    tooltip = _RealQLabel(container)
    tooltip.setStyleSheet("background: rgba(0, 255, 0, 255);")
    tooltip.setGeometry(0, 0, 1, 1)
    tooltip.show()

    container.show()
    app.processEvents()

    gpu_frame = np.zeros((8, 12, 4), dtype=np.uint8)
    gpu_frame[:, :, :3] = [12, 34, 200]
    gpu_frame[:, :, 3] = 255
    capture = _CaptureHarness()
    capture._renderer.snapshot = lambda: gpu_frame.copy()
    capture._container = container
    capture._hud_overlay_labels = {"test": hud}
    capture._tooltip_label = tooltip

    try:
        assert container.size().toTuple() == (6, 4)
        assert hud.isVisible()
        hud_rgba = capture._render_qt_widget_rgba(hud, width=4, height=4)
        assert hud_rgba is not None
        assert int(hud_rgba[:, :, 3].max()) == 128
        preview = capture._composite_visible_qt_overlays(gpu_frame[:, :, :3])
        assert preview[2, 4].tolist() == [134, 17, 100]
        image = capture.export_screenshot_to_array(include_hud=True)
        scaled = capture.export_screenshot_to_array(
            resolution_scale=0.5,
            include_hud=True,
        )
    finally:
        container.close()

    assert image.shape == (8, 12, 3)
    assert scaled.shape == (4, 6, 3)
    # Container-relative (2, 1, 2, 2) maps to (4, 2, 4, 4) in the 2x GPU
    # snapshot. Pixels elsewhere preserve the authoritative GPU readback.
    assert image[1, 4].tolist() == [12, 34, 200]
    assert image[6, 4].tolist() == [12, 34, 200]
    assert image[2, 4].tolist() == [134, 17, 100]
    assert image[5, 7].tolist() == [134, 17, 100]
    # The visible tooltip is captured too and remains above persistent HUDs.
    assert image[0, 0].tolist() == [0, 255, 0]
    assert image[1, 1].tolist() == [0, 255, 0]


def test_pygfx_clean_or_hidden_hud_capture_preserves_gpu_pixels():
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    container = QWidget()
    container.resize(4, 3)
    hud = _RealQLabel(container)
    hud.setStyleSheet("background: rgba(255, 0, 0, 255);")
    hud.setGeometry(1, 1, 2, 1)
    container.show()
    hud.show()
    app.processEvents()

    gpu_frame = np.zeros((3, 4, 4), dtype=np.uint8)
    gpu_frame[:, :, :3] = [9, 33, 77]
    gpu_frame[:, :, 3] = 255
    capture = _CaptureHarness()
    capture._renderer.snapshot = lambda: gpu_frame.copy()
    capture._container = container
    capture._hud_overlay_labels = {"test": hud}
    capture._tooltip_label = None

    try:
        clean = capture.export_screenshot_to_array()
        hud.hide()
        app.processEvents()
        hidden = capture.export_screenshot_to_array(include_hud=True)
    finally:
        container.close()

    np.testing.assert_array_equal(clean, gpu_frame[:, :, :3])
    np.testing.assert_array_equal(hidden, gpu_frame[:, :, :3])


def test_pygfx_screenshot_export_rejects_unavailable_or_black_readback(tmp_path):
    capture = _CaptureHarness()
    capture._initialized = False
    uninitialized_path = tmp_path / "uninitialized.png"

    assert capture.export_screenshot(str(uninitialized_path)) is False
    assert not uninitialized_path.exists()

    capture._initialized = True
    capture._renderer.snapshot = lambda: np.zeros((2, 2, 4), dtype=np.uint8)
    black_path = tmp_path / "black.png"

    assert capture.export_screenshot(str(black_path)) is False
    assert not black_path.exists()


def test_pygfx_screenshot_export_writes_valid_readback(tmp_path):
    capture = _CaptureHarness()
    output = tmp_path / "capture.png"

    assert capture.export_screenshot(str(output)) is True
    assert output.is_file()


def test_pygfx_payload_center_filters_nonfinite_vertices():
    payload = MeshPayload(
        vertices=np.asarray(
            [
                [np.inf, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        ),
        triangles=np.empty((0, 3), dtype=np.int32),
    )

    with np.errstate(all="raise"):
        center = PygfxGeometryMixin._compute_payload_center(payload)

    np.testing.assert_allclose(center, [2.0, 3.0, 4.0])


def test_pygfx_payload_center_returns_zero_for_all_nonfinite_vertices():
    payload = MeshPayload(
        vertices=np.asarray(
            [
                [np.inf, 0.0, 0.0],
                [np.nan, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
        triangles=np.empty((0, 3), dtype=np.int32),
    )

    with np.errstate(all="raise"):
        center = PygfxGeometryMixin._compute_payload_center(payload)

    np.testing.assert_allclose(center, [0.0, 0.0, 0.0])


def test_export_renderer_kwargs_respects_offscreen_env(monkeypatch):
    monkeypatch.setenv("ORCHAV_PYGFX_EXPORT_PIXEL_SCALE", "1.5")
    monkeypatch.setenv("ORCHAV_PYGFX_EXPORT_PIXEL_FILTER", "linear")
    monkeypatch.setenv("ORCHAV_PYGFX_EXPORT_PPAA", "default")

    kwargs = export_renderer_kwargs()

    assert kwargs == {
        "pixel_scale": 1.5,
        "pixel_filter": "linear",
        "ppaa": "default",
    }


def test_display_renderer_kwargs_default_to_native_resolution(monkeypatch):
    monkeypatch.delenv("ORCHAV_PYGFX_PIXEL_SCALE", raising=False)

    assert display_renderer_kwargs() == {"pixel_scale": 1.0}


def test_display_renderer_kwargs_allow_explicit_supersampling(monkeypatch):
    monkeypatch.setenv("ORCHAV_PYGFX_PIXEL_SCALE", "1.5")

    assert display_renderer_kwargs() == {"pixel_scale": 1.5}


def test_build_pygfx_effect_passes_warns_for_fog(monkeypatch, caplog):
    monkeypatch.setenv("ORCHAV_PYGFX_ENABLE_FOG", "1")
    monkeypatch.delenv("ORCHAV_PYGFX_ENABLE_BLOOM", raising=False)

    passes = build_pygfx_effect_passes(_FakeGfx, clear_color=(0.8, 0.8, 0.8, 1.0))

    assert passes == ()
    assert "whites out dense scenes" in caplog.text


def test_build_pygfx_effect_passes_builds_bloom(monkeypatch):
    monkeypatch.setenv("ORCHAV_PYGFX_ENABLE_BLOOM", "1")
    monkeypatch.setenv("ORCHAV_PYGFX_BLOOM_STRENGTH", "0.1")
    monkeypatch.setenv("ORCHAV_PYGFX_BLOOM_MAX_MIP_LEVELS", "4")
    monkeypatch.setenv("ORCHAV_PYGFX_BLOOM_FILTER_RADIUS", "0.02")
    monkeypatch.setenv("ORCHAV_PYGFX_BLOOM_USE_KARIS_AVERAGE", "1")
    monkeypatch.setattr(
        "visualizer.src.renderers.pygfx.canvas.load_physical_bloom_pass",
        lambda: _FakeBloomPass,
    )

    passes = build_pygfx_effect_passes(_FakeGfx, clear_color=(0.8, 0.8, 0.8, 1.0))

    assert len(passes) == 1
    assert isinstance(passes[0], _FakeBloomPass)
    assert passes[0].kwargs == {
        "bloom_strength": 0.1,
        "max_mip_levels": 4,
        "filter_radius": 0.02,
        "use_karis_average": True,
    }


def test_create_wgpu_renderer_applies_offscreen_kwargs(monkeypatch):
    monkeypatch.setenv("ORCHAV_PYGFX_EXPORT_PIXEL_SCALE", "2.0")
    monkeypatch.setenv("ORCHAV_PYGFX_EXPORT_PPAA", "0")

    renderer = create_wgpu_renderer(
        _FakeGfx,
        target="canvas",
        clear_color=(0.8, 0.8, 0.8, 1.0),
        offscreen=True,
    )

    assert isinstance(renderer, _FakeRenderer)
    assert renderer.kwargs == {"pixel_scale": 2.0, "ppaa": False}
    assert renderer.effect_passes == ()


def test_create_wgpu_renderer_uses_native_resolution_for_interactive_canvas(monkeypatch):
    monkeypatch.delenv("ORCHAV_PYGFX_PIXEL_SCALE", raising=False)

    renderer = create_wgpu_renderer(
        _FakeGfx,
        target="canvas",
        clear_color=(0.8, 0.8, 0.8, 1.0),
    )

    assert isinstance(renderer, _FakeRenderer)
    assert renderer.kwargs == {"pixel_scale": 1.0}
    assert renderer.effect_passes == ()


def test_create_wgpu_renderer_can_defer_effect_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "visualizer.src.renderers.pygfx.canvas._refresh_renderer_effect_passes",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    renderer = create_wgpu_renderer(
        _FakeGfx,
        target="canvas",
        clear_color=(0.8, 0.8, 0.8, 1.0),
        configure_effects=False,
    )

    assert isinstance(renderer, _FakeRenderer)
    assert calls == []


def test_coverage_transparency_preserves_vertex_color_material():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    redraws = []
    coverage_name = PygfxRenderer.COVERAGE_MESH_NAME
    renderer._gfx = _FakeCoverageGfx
    renderer._objects = {
        coverage_name: SimpleNamespace(material=SimpleNamespace(color_mode="auto"))
    }
    renderer._kinds = {coverage_name: "mesh"}
    renderer._clipping_planes = ()
    renderer._ibl_manager = SimpleNamespace(_tracked_materials=[])
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_coverage_transparency(0.9) is True

    mat = renderer._objects[coverage_name].material
    assert isinstance(mat, _FakeCoverageMaterial)
    assert mat.color_mode == "vertex"
    assert mat.side == "both"
    assert mat.opacity == pytest.approx(0.9)
    assert mat.color == (1.0, 1.0, 1.0, 0.9)
    assert mat.alpha_mode == "weighted_blend"
    assert mat.depth_write is False
    assert mat.depth_test is True
    assert mat.env_map is None
    assert mat.env_map_intensity == 0.0
    assert redraws == [True]

    renderer._ibl_manager._tracked_materials.append(mat)
    assert renderer.set_coverage_transparency(1.0) is True
    mat = renderer._objects[coverage_name].material
    assert mat.color_mode == "vertex"
    assert mat.opacity == pytest.approx(1.0)
    assert mat.alpha_mode == "auto"
    assert mat.depth_write is True
    assert mat.env_map is None
    assert mat.env_map_intensity == 0.0
    assert renderer._ibl_manager._tracked_materials == []
    assert redraws == [True, True]


def test_pygfx_coverage_upload_preserves_vertex_colors():
    class _CoverageRenderer(PygfxSurfaceOverlayMixin):
        COVERAGE_MESH_NAME = "coverage_mesh"
        COVERAGE_ISOLINES_NAME = "coverage_isolines"

        def __init__(self):
            self.calls = []
            self.material_alphas = []
            self.isoline_payloads = []
            self.native_names = set()
            self._last_coverage_signature = None

        def ensure_named_geometry(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            self.native_names.add(args[0])
            return True

        def _apply_coverage_material_state(self, alpha, *, request_redraw):
            self.material_alphas.append((alpha, request_redraw))
            return True

        def _apply_coverage_isolines_payload(self, payload):
            self.isoline_payloads.append(payload)
            return True

        def has_named_geometry(self, name):
            return name in self.native_names

        def remove_named_geometry(self, name):
            self.native_names.discard(name)
            return True

    renderer = _CoverageRenderer()
    view_model = SimpleNamespace(
        show_coverage=True,
        coverage_vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        coverage_triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        coverage_colors=np.asarray(
            [[0.1, 0.2, 0.9], [0.2, 0.7, 0.3], [0.9, 0.1, 0.1]],
            dtype=np.float32,
        ),
        coverage_signature="coverage-color-regression",
        coverage_opacity=0.65,
    )

    renderer._apply_coverage_data(view_model)

    assert len(renderer.calls) == 1
    args, kwargs = renderer.calls[0]
    assert args[0] == "coverage_mesh"
    assert kwargs["visible"] is True
    assert "preserve_vertex_colors" not in kwargs
    assert args[1].color_source is SurfaceColorSource.VERTEX
    assert args[1].vertex_colors is not None
    np.testing.assert_allclose(args[1].vertex_colors, view_model.coverage_colors)
    assert renderer.material_alphas == [(pytest.approx(0.65), False)]
    assert renderer.isoline_payloads == [None]
    assert renderer._last_coverage_signature == "coverage-color-regression"


def test_payload_to_pygfx_lines_uses_line_material_for_line_strips():
    payload = LineSetPayload(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        lines=np.empty((0, 2), dtype=np.int32),
    )

    line = payload_to_pygfx_lines(_FakeLineGfx, payload, line_width=3.5)

    assert isinstance(line.material, _FakeLineMaterial)
    assert line.material.kwargs["thickness"] == pytest.approx(3.5)
    assert "indices" not in line.geometry.kwargs


def test_payload_to_pygfx_lines_uses_segment_material_for_indexed_lines():
    payload = LineSetPayload(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [9.0, 9.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        lines=np.array([[0, 1], [3, 1]], dtype=np.int32),
    )

    line = payload_to_pygfx_lines(_FakeLineGfx, payload, line_width=2.0)

    assert isinstance(line.material, _FakeLineSegmentMaterial)
    np.testing.assert_array_equal(
        line.geometry.kwargs["positions"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert "indices" not in line.geometry.kwargs


def test_pygfx_renderer_builds_line_strips_with_line_material():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._gfx = _FakeLineGfx
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    payload = LineSetPayload(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        lines=np.empty((0, 2), dtype=np.int32),
    )

    obj, kind = renderer._build_world_object(payload)

    assert kind == "lines"
    assert isinstance(obj.material, _FakeLineMaterial)
    assert "indices" not in obj.geometry.kwargs


def test_pygfx_line_geometry_detaches_writable_backend_buffers():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._gfx = SimpleNamespace(Geometry=_FakeGeometry)
    payload = LineSetPayload(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        lines=np.empty((0, 2), dtype=np.int32),
        colors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )

    geometry = renderer._build_lines_geometry(payload, line_strip=True)

    assert payload.points.flags.writeable is False
    assert payload.colors is not None and payload.colors.flags.writeable is False
    assert geometry.kwargs["positions"].flags.writeable is True
    assert geometry.kwargs["colors"].flags.writeable is True
    assert not np.shares_memory(geometry.kwargs["positions"], payload.points)
    assert not np.shares_memory(geometry.kwargs["colors"], payload.colors)


def test_pygfx_backend_array_reuses_owned_writable_memory():
    owned = np.ones((4, 3), dtype=np.float32)

    result = PygfxGeometryMixin._writable_backend_array(owned)

    assert result is owned


def test_pygfx_buffer_layout_signature_tracks_optional_mesh_buffers():
    renderer = PygfxGeometryMixin()
    vertices = np.zeros((4, 3), dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    without_attributes = MeshPayload(vertices=vertices, triangles=triangles)
    baseline = renderer._get_buffer_layout_signature(without_attributes)

    payloads_with_optional_buffer = (
        MeshPayload(
            vertices=vertices,
            triangles=triangles,
            normals=np.ones((4, 3), dtype=np.float32),
        ),
        MeshPayload(
            vertices=vertices,
            triangles=triangles,
            vertex_colors=np.ones((4, 3), dtype=np.float32),
        ),
        MeshPayload(
            vertices=vertices,
            triangles=triangles,
            triangle_uvs=np.zeros((4, 2), dtype=np.float32),
        ),
    )

    for payload in payloads_with_optional_buffer:
        assert renderer._get_buffer_layout_signature(payload) != baseline


def test_pygfx_buffer_preparation_keeps_non_uv_mesh_updates_zero_copy():
    renderer = PygfxGeometryMixin()
    payload = MeshPayload(
        vertices=np.zeros((4, 3), dtype=np.float32),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        normals=np.ones((4, 3), dtype=np.float32),
    )

    buffers = renderer._prepare_geometry_buffers(payload)

    assert buffers is not None
    assert np.shares_memory(buffers["positions"], payload.vertices)
    assert np.shares_memory(buffers["indices"], payload.triangles)
    assert payload.normals is not None
    assert np.shares_memory(buffers["normals"], payload.normals)


class _TrackedGeometryBuffer:
    def __init__(self, data: np.ndarray, *, fail_updates: int = 0):
        self.data = np.array(data, copy=True)
        self.fail_updates = fail_updates
        self.update_full_calls = 0

    def update_full(self) -> None:
        self.update_full_calls += 1
        if self.fail_updates:
            self.fail_updates -= 1
            raise RuntimeError("simulated GPU upload failure")


class _SetDataOnlyGeometryBuffer:
    def __init__(self):
        self.uploaded: np.ndarray | None = None
        self.update_full_calls = 0

    def set_data(self, data: np.ndarray) -> None:
        self.uploaded = np.array(data, copy=True)

    def update_full(self) -> None:
        self.update_full_calls += 1


class _MeshBufferUpdateHarness(PygfxGeometryMixin):
    def __init__(self):
        self._uncertain_mesh_index_buffers: set[str] = set()

    def _record_profile_metric(self, *_args, **_kwargs) -> None:
        pass

    def _record_profile_bytes(self, *_args, **_kwargs) -> None:
        pass


def _tracked_geometry(buffers: dict[str, np.ndarray]) -> SimpleNamespace:
    return SimpleNamespace(
        **{name: _TrackedGeometryBuffer(values) for name, values in buffers.items()}
    )


def test_pygfx_mesh_update_skips_exactly_matching_prepared_indices():
    renderer = _MeshBufferUpdateHarness()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    initial = MeshPayload(
        vertices=np.zeros((4, 3), dtype=np.float32),
        triangles=triangles,
    )
    animated = MeshPayload(
        vertices=np.ones((4, 3), dtype=np.float32),
        triangles=triangles.copy(),
    )
    initial_buffers = renderer._prepare_geometry_buffers(initial)
    animated_buffers = renderer._prepare_geometry_buffers(animated)
    assert initial_buffers is not None
    assert animated_buffers is not None
    geometry = _tracked_geometry(initial_buffers)

    renderer._update_buffers_mesh(
        geometry,
        animated,
        name="target:walker::mesh",
        buffers=animated_buffers,
    )

    assert geometry.positions.update_full_calls == 1
    assert geometry.indices.update_full_calls == 0
    np.testing.assert_array_equal(geometry.positions.data, animated_buffers["positions"])


def test_pygfx_mesh_update_uploads_same_shape_different_indices():
    renderer = _MeshBufferUpdateHarness()
    initial = MeshPayload(
        vertices=np.zeros((4, 3), dtype=np.float32),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    changed = MeshPayload(
        vertices=np.ones((4, 3), dtype=np.float32),
        triangles=np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32),
    )
    initial_buffers = renderer._prepare_geometry_buffers(initial)
    changed_buffers = renderer._prepare_geometry_buffers(changed)
    assert initial_buffers is not None
    assert changed_buffers is not None
    assert initial_buffers["indices"].shape == changed_buffers["indices"].shape
    geometry = _tracked_geometry(initial_buffers)

    renderer._update_buffers_mesh(
        geometry,
        changed,
        name="target:retopologized::mesh",
        buffers=changed_buffers,
    )

    assert geometry.indices.update_full_calls == 1
    np.testing.assert_array_equal(geometry.indices.data, changed_buffers["indices"])


def test_pygfx_mesh_update_compares_uv_seam_expanded_native_indices():
    renderer = _MeshBufferUpdateHarness()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    triangle_uvs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.2],
            [0.8, 0.8],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    initial = MeshPayload(
        vertices=np.zeros((4, 3), dtype=np.float32),
        triangles=triangles,
        triangle_uvs=triangle_uvs,
    )
    animated = MeshPayload(
        vertices=np.ones((4, 3), dtype=np.float32),
        triangles=triangles.copy(),
        triangle_uvs=triangle_uvs.copy(),
    )
    initial_buffers = renderer._prepare_geometry_buffers(initial)
    animated_buffers = renderer._prepare_geometry_buffers(animated)
    assert initial_buffers is not None
    assert animated_buffers is not None
    assert initial_buffers["positions"].shape != initial.vertices.shape
    assert not np.array_equal(initial_buffers["indices"], initial.triangles)
    geometry = _tracked_geometry(initial_buffers)

    renderer._update_buffers_mesh(
        geometry,
        animated,
        name="target:textured::mesh",
        buffers=animated_buffers,
    )

    assert geometry.indices.update_full_calls == 0
    assert geometry.positions.update_full_calls == 1
    assert geometry.texcoords.update_full_calls == 1


def test_pygfx_mesh_update_uploads_indices_when_native_identity_is_unknown():
    renderer = _MeshBufferUpdateHarness()
    payload = MeshPayload(
        vertices=np.zeros((3, 3), dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )
    buffers = renderer._prepare_geometry_buffers(payload)
    assert buffers is not None
    geometry = _tracked_geometry(buffers)
    geometry.indices = _SetDataOnlyGeometryBuffer()

    renderer._update_buffers_mesh(
        geometry,
        payload,
        name="target:unknown::mesh",
        buffers=buffers,
    )

    assert geometry.indices.update_full_calls == 1
    np.testing.assert_array_equal(geometry.indices.uploaded, buffers["indices"])


def test_pygfx_mesh_index_upload_failure_forces_retry_even_after_cpu_copy():
    renderer = _MeshBufferUpdateHarness()
    name = "target:retry::mesh"
    initial = MeshPayload(
        vertices=np.zeros((4, 3), dtype=np.float32),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    changed = MeshPayload(
        vertices=np.ones((4, 3), dtype=np.float32),
        triangles=np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32),
    )
    initial_buffers = renderer._prepare_geometry_buffers(initial)
    changed_buffers = renderer._prepare_geometry_buffers(changed)
    assert initial_buffers is not None
    assert changed_buffers is not None
    geometry = _tracked_geometry(initial_buffers)
    geometry.indices.fail_updates = 1

    with pytest.raises(RuntimeError, match="simulated GPU upload failure"):
        renderer._update_buffers_mesh(
            geometry,
            changed,
            name=name,
            buffers=changed_buffers,
        )

    # _push_buffer copied the desired indices before update_full() failed.
    np.testing.assert_array_equal(geometry.indices.data, changed_buffers["indices"])
    assert name in renderer._uncertain_mesh_index_buffers

    renderer._update_buffers_mesh(
        geometry,
        changed,
        name=name,
        buffers=changed_buffers,
    )

    assert geometry.indices.update_full_calls == 2
    assert name not in renderer._uncertain_mesh_index_buffers


def test_pygfx_buffer_layout_signature_tracks_optional_buffer_shapes():
    renderer = PygfxGeometryMixin()
    vertices = np.zeros((4, 3), dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    rgb = MeshPayload(
        vertices=vertices,
        triangles=triangles,
        vertex_colors=np.ones((4, 3), dtype=np.float32),
    )
    rgba = MeshPayload(
        vertices=vertices,
        triangles=triangles,
        vertex_colors=np.ones((4, 4), dtype=np.float32),
    )

    assert renderer._get_buffer_layout_signature(rgb) != renderer._get_buffer_layout_signature(rgba)


def test_pygfx_buffer_layout_signature_tracks_uv_seam_expansion():
    renderer = PygfxGeometryMixin()
    vertices = np.zeros((4, 3), dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    shared_uvs = MeshPayload(
        vertices=vertices,
        triangles=triangles,
        triangle_uvs=np.zeros((6, 2), dtype=np.float32),
    )
    split_uvs = MeshPayload(
        vertices=vertices,
        triangles=triangles,
        triangle_uvs=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    assert renderer._get_buffer_layout_signature(
        shared_uvs
    ) != renderer._get_buffer_layout_signature(split_uvs)


def test_pygfx_point_geometry_detaches_writable_backend_buffers():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._gfx = SimpleNamespace(Geometry=_FakeGeometry)
    payload = PointCloudPayload(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
        colors=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )

    geometry = renderer._build_points_geometry(payload)

    assert payload.points.flags.writeable is False
    assert payload.colors is not None and payload.colors.flags.writeable is False
    assert geometry.kwargs["positions"].flags.writeable is True
    assert geometry.kwargs["colors"].flags.writeable is True
    assert not np.shares_memory(geometry.kwargs["positions"], payload.points)
    assert not np.shares_memory(geometry.kwargs["colors"], payload.colors)


def test_pygfx_renderer_expands_segment_colored_indexed_lines():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._gfx = _FakeLineGfx
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    payload = LineSetPayload(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        lines=np.array([[0, 1], [1, 2]], dtype=np.int32),
        colors=np.array([[0.2, 0.5, 1.0], [0.2, 0.5, 1.0]], dtype=np.float32),
    )

    obj, kind = renderer._build_world_object(payload, name="aperture_aoa_0")

    assert kind == "lines"
    assert isinstance(obj.material, _FakeLineSegmentMaterial)
    assert obj.material.kwargs["color_mode"] == "vertex"
    assert obj.geometry.kwargs["positions"].shape == (4, 3)
    assert obj.geometry.kwargs["colors"].shape == (4, 3)
    assert "indices" not in obj.geometry.kwargs
    assert renderer._get_buffer_sizes(payload) == (4, 4)


def test_pygfx_renderer_expands_vertex_colored_indexed_lines():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._gfx = _FakeLineGfx
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    payload = LineSetPayload(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [9.0, 9.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        lines=np.array([[0, 1], [3, 1]], dtype=np.int32),
        colors=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    obj, kind = renderer._build_world_object(payload, name="scene_outline_wall")

    assert kind == "lines"
    assert isinstance(obj.material, _FakeLineSegmentMaterial)
    np.testing.assert_array_equal(
        obj.geometry.kwargs["positions"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        obj.geometry.kwargs["colors"],
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert "indices" not in obj.geometry.kwargs


def test_pygfx_renderer_builds_orientation_frames_with_native_axes_helper():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.scene.orientation_frame_payloads import make_orientation_frame_payload

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._gfx = _FakeLineGfx
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None

    obj, kind = renderer._build_world_object(
        make_orientation_frame_payload(3.0, thickness=4.0),
        name="node:tx_0::orientation_frame",
    )

    assert kind == "orientation_frame"
    assert isinstance(obj, _FakeAxesHelper)
    assert obj.size == pytest.approx(3.0)
    assert obj.thickness == pytest.approx(4.0)


def test_set_named_material_preserves_orientation_frame_vertex_colors():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "node:tx_0::orientation_frame"
    frame = _FakeAxesHelper(size=3.0, thickness=4.0)
    frame.material.color_mode = "auto"
    frame.material.thickness = 1.0
    frame.material.pick_write = True
    clipped = []
    redraws = []
    renderer._objects = {name: frame}
    renderer._materials = {}
    renderer._kinds = {name: "orientation_frame"}
    renderer._geometry_color_sources = {}
    renderer._ibl_loaded = False
    renderer._clipping_planes = ()
    renderer._record_frame_update_metric = lambda *_args, **_kwargs: None
    renderer._apply_clipping_to_material = lambda mat: clipped.append(mat)
    renderer.request_redraw = lambda: redraws.append(True)

    assert (
        renderer.set_named_material(
            name,
            MaterialPayload(
                base_color=(0.2, 0.4, 0.6, 1.0),
                line_width=6.0,
            ),
        )
        is True
    )

    assert frame.material.color_mode == "vertex"
    assert frame.material.color == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert frame.material.thickness == pytest.approx(6.0)
    assert frame.material.pick_write is False
    assert clipped == [frame.material]
    assert redraws == [True]


def test_pygfx_label_layout_separates_co_located_tx_rx_labels():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    transforms = {}
    renderer.visualizer = SimpleNamespace(
        label_offset_x=1.5,
        label_offset_y=0.0,
        label_offset_z=1.0,
    )
    renderer._geometry_upload_center = {}
    renderer._positions = {}
    renderer.has_named_geometry = lambda _name: True

    def _set_named_transform(name, transform):
        transforms[name] = np.asarray(transform, dtype=np.float32).copy()
        return True

    renderer.set_named_transform = _set_named_transform

    anchor = np.array([0.0, -4.8, 1.5], dtype=np.float32)
    tx_label = "node:tx_0::label"
    rx_label = "node:rx_0::label"

    offset = np.array([1.5, 0.0, 1.0], dtype=np.float32)
    assert renderer._register_and_layout_label(tx_label, anchor, offset)
    assert renderer._register_and_layout_label(rx_label, anchor, offset)

    tx_pos = np.array(renderer._positions[tx_label])
    rx_pos = np.array(renderer._positions[rx_label])
    assert tx_pos[0] == pytest.approx(1.5)
    assert rx_pos[0] == pytest.approx(1.5)
    assert tx_pos[1] < rx_pos[1]
    assert tx_pos[2] < rx_pos[2]
    assert not np.allclose(transforms[tx_label][:3, 3], transforms[rx_label][:3, 3])


def test_pygfx_label_layout_skips_unchanged_anchor_group_relayout():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._label_anchor_groups = {}
    renderer._label_anchor_key_by_name = {}
    renderer._label_anchor_by_name = {}
    renderer._label_offset_by_name = {}
    renderer.has_named_geometry = lambda _name: True
    layout_calls = []
    renderer._layout_label_group = lambda key: layout_calls.append(key) or True

    anchor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    offset = np.array([0.5, 0.0, 1.0], dtype=np.float32)

    assert renderer._register_and_layout_label("node:tx_0::label", anchor, offset)
    assert len(layout_calls) == 1
    assert renderer._register_and_layout_label(
        "node:tx_0::label",
        anchor.copy(),
        offset.copy(),
    )
    assert len(layout_calls) == 1


def test_pygfx_text_label_render_object_is_idempotent():
    from visualizer.src.model import make_text_label_state
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _TextMaterial:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Text:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.material = kwargs["material"]
            self.visible = True
            self.local = SimpleNamespace(matrix=np.eye(4, dtype=np.float32))

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._gfx = SimpleNamespace(Text=_Text, TextMaterial=_TextMaterial)
    renderer._initialized = True
    renderer._scene = _FakeScene()
    renderer._name_to_handle = {}
    renderer._handle_to_name = {}
    renderer._objects = {}
    renderer._kinds = {}
    renderer._reverse_objects = {}
    renderer._geometry_upload_center = {}
    renderer._render_object_snapshots = {}
    renderer._materials = {}
    renderer._next_handle = 1
    renderer._register_pick_metadata = lambda *_args, **_kwargs: None

    def _set_material(name, material):
        renderer._materials[name] = material
        return True

    renderer.set_named_material = _set_material
    renderer._apply_render_object_transform = lambda _obj: True
    renderer.set_named_visibility = lambda *_args: True

    label = make_text_label_state(
        "node:tx_0::label",
        "TX1",
        [1.0, 0.0, 0.0],
        position=[1.0, 2.0, 3.0],
    )

    assert renderer.ensure_object(label.to_render_object()) is True
    assert renderer.ensure_object(label.to_render_object()) is True
    assert len(renderer._scene.children) == 1
    native = renderer._objects[label.id]
    assert native.kwargs["text"] == "TX1"
    assert native.kwargs["screen_space"] is True


def test_pygfx_render_object_refreshes_metadata_without_geometry_upload():
    from visualizer.src.model import RenderObject
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._name_to_handle = {}
    renderer._materials = {}
    renderer._material_apply_signatures = {}
    renderer._render_object_snapshots = {}
    renderer._dirty_render_object_geometry = set()
    renderer._pick_metadata = {}
    renderer._apply_render_object_transform = lambda _obj: True
    renderer.set_named_visibility = lambda *_args: True
    uploads = []

    def _ensure_named_geometry(*, name, geometry, **_kwargs):
        uploads.append(geometry)
        renderer._name_to_handle[name] = 1
        return True

    renderer.ensure_named_geometry = _ensure_named_geometry
    payload = PointCloudPayload(points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))

    initial = RenderObject(id="authoring:target", payload=payload, metadata={"step": 0})
    updated = RenderObject(id="authoring:target", payload=payload, metadata={"step": 1})
    cleared = RenderObject(id="authoring:target", payload=payload)

    assert renderer.ensure_object(initial) is True
    assert renderer.ensure_object(updated) is True
    assert renderer._pick_metadata["authoring:target"] == {"step": 1}
    assert renderer.ensure_object(cleared) is True
    assert "authoring:target" not in renderer._pick_metadata
    assert uploads == [payload]


def test_render_object_pickable_metadata_controls_native_pick_writes():
    from visualizer.src.model import RenderObject

    object_id = "authoring:guide"
    native_material = SimpleNamespace(
        pick_write=True,
        depth_write=True,
        depth_compare="<",
    )
    native = SimpleNamespace(material=native_material, render_order=0)
    owner = SimpleNamespace(
        _pick_metadata={},
        _objects={object_id: native},
    )
    payload = PointCloudPayload(
        points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    )

    PygfxGeometryMixin._sync_render_object_metadata(
        owner,
        RenderObject(
            id=object_id,
            payload=payload,
            metadata={
                "component": "control_guide",
                "pickable": False,
                "render_order": 10,
                "depth_write": False,
            },
        ),
    )
    assert native_material.pick_write is False
    assert native_material.depth_write is False
    assert native_material.depth_compare == "<"
    assert native.render_order == 10

    PygfxGeometryMixin._sync_render_object_metadata(
        owner,
        RenderObject(
            id=object_id,
            payload=payload,
            metadata={
                "component": "control_handle",
                "pickable": True,
                "render_order": 20,
                "depth_write": False,
                "depth_compare": "<=",
            },
        ),
    )
    assert native_material.pick_write is True
    assert native_material.depth_write is False
    assert native_material.depth_compare == "<="
    assert native.render_order == 20


def test_pygfx_renderer_builds_explicit_line_strips_without_name_heuristics():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._gfx = _FakeLineGfx
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    payload = LineSetPayload(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        lines=np.array([[0, 1], [1, 2]], dtype=np.int32),
        line_strip=True,
    )

    obj, kind = renderer._build_world_object(payload, name="trajectory_1")

    assert kind == "lines"
    assert isinstance(obj.material, _FakeLineMaterial)
    assert "indices" not in obj.geometry.kwargs


def test_set_named_visibility_syncs_actual_object_when_hidden_state_is_stale():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "node:tx_0::label"
    text_obj = SimpleNamespace(visible=False)
    redraws = []
    renderer._name_to_handle = {name: 1}
    renderer._objects = {name: text_obj}
    renderer._hidden = set()
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.is_named_visible(name) is True
    assert renderer.set_named_visibility(name, True) is True

    assert text_obj.visible is True
    assert redraws == [True]


def test_set_named_visibility_failure_preserves_cache_and_redraw_state():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _RejectVisibility:
        @property
        def visible(self):
            return True

        @visible.setter
        def visible(self, _value):
            raise RuntimeError("native visibility rejected")

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "node:tx_0::label"
    redraws = []
    renderer._name_to_handle = {name: 1}
    renderer._objects = {name: _RejectVisibility()}
    renderer._hidden = set()
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_named_visibility(name, False) is False
    assert renderer.is_named_visible(name) is True
    assert redraws == []


def _make_transform_renderer(name, local):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._objects = {name: SimpleNamespace(local=local)}
    renderer._transforms = {}
    renderer._positions = {}
    return renderer


def test_set_named_transform_applies_matrix_before_caching():
    name = "target:pedestrian::mesh"
    local = SimpleNamespace(matrix=None, position=None)
    renderer = _make_transform_renderer(name, local)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = (1.0, 2.0, 3.0)

    assert renderer.set_named_transform(name, transform) is True
    np.testing.assert_allclose(local.matrix, transform)
    np.testing.assert_allclose(renderer._transforms[name], transform)
    assert renderer._positions[name] == pytest.approx((1.0, 2.0, 3.0))


def test_set_named_transform_does_not_drop_small_delta_at_large_coordinates():
    name = "target:pedestrian::mesh"
    old_transform = np.eye(4, dtype=np.float32)
    old_transform[:3, 3] = (100_000.0, 200_000.0, 300_000.0)
    local = SimpleNamespace(matrix=old_transform.copy(), position=None)
    renderer = _make_transform_renderer(name, local)
    renderer._transforms[name] = old_transform.copy()
    renderer._positions[name] = tuple(float(value) for value in old_transform[:3, 3])
    desired = old_transform.copy()
    desired[0, 3] += 0.5

    assert renderer.set_named_transform(name, desired) is True
    np.testing.assert_array_equal(local.matrix, desired)
    np.testing.assert_array_equal(renderer._transforms[name], desired)


def test_set_named_transform_uses_position_fallback_for_translation_only():
    name = "node:rx_0::label"
    local = SimpleNamespace(position=None)
    renderer = _make_transform_renderer(name, local)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = (4.0, 5.0, 6.0)

    assert renderer.set_named_transform(name, transform) is True
    assert local.position == pytest.approx((4.0, 5.0, 6.0))
    np.testing.assert_allclose(renderer._transforms[name], transform)
    assert renderer._positions[name] == pytest.approx((4.0, 5.0, 6.0))


def test_set_named_transform_rejects_rotation_when_only_position_is_available():
    name = "target:pedestrian::mesh"
    local = SimpleNamespace(position=None)
    renderer = _make_transform_renderer(name, local)
    transform = np.eye(4, dtype=np.float32)
    transform[:2, :2] = ((0.0, -1.0), (1.0, 0.0))

    assert renderer.set_named_transform(name, transform) is False
    assert local.position is None
    assert renderer._transforms == {}
    assert renderer._positions == {}


def test_set_named_transform_failure_preserves_transform_and_position_caches():
    class _RejectTransform:
        @property
        def matrix(self):
            return None

        @matrix.setter
        def matrix(self, _value):
            raise RuntimeError("matrix rejected")

        @property
        def position(self):
            return None

        @position.setter
        def position(self, _value):
            raise RuntimeError("position rejected")

    name = "target:pedestrian::mesh"
    renderer = _make_transform_renderer(name, _RejectTransform())
    old_transform = np.eye(4, dtype=np.float32)
    old_position = (7.0, 8.0, 9.0)
    renderer._transforms[name] = old_transform.copy()
    renderer._positions[name] = old_position
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = (1.0, 2.0, 3.0)

    assert renderer.set_named_transform(name, transform) is False
    np.testing.assert_array_equal(renderer._transforms[name], old_transform)
    assert renderer._positions[name] == old_position


def test_set_named_material_requests_redraw_without_canvas():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.types.render_payloads import MaterialPayload

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "scene:wall_1::mesh"
    material = SimpleNamespace(
        color=None,
        opacity=None,
        roughness=None,
        metalness=None,
        metallic=None,
    )
    redraws = []
    renderer._objects = {name: SimpleNamespace(material=material)}
    renderer._materials = {}
    renderer._gfx = SimpleNamespace(MeshBasicMaterial=None, MeshPhysicalMaterial=None)
    renderer._kinds = {name: "mesh"}
    renderer._geometry_color_sources = {}
    renderer._ibl_loaded = False
    renderer._coerce_material = lambda _payload: MaterialPayload(
        base_color=(0.2, 0.4, 0.6, 1.0),
        roughness=0.3,
        metallic=0.1,
    )
    renderer._record_frame_update_metric = lambda *_args, **_kwargs: None
    renderer._apply_material_alpha_state = lambda *_args, **_kwargs: None
    renderer._load_texture_binding = lambda *_args, **_kwargs: None
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_named_material(name, {"color": [0.2, 0.4, 0.6]}) is True

    assert material.color == pytest.approx((0.2, 0.4, 0.6, 1.0))
    assert redraws == [True]


def test_set_named_material_uses_uniform_mode_for_mesh_color_buffer():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.types.render_payloads import MaterialPayload

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "scene:wall_1::mesh"
    material = SimpleNamespace(
        color=None,
        color_mode="vertex",
        opacity=None,
        roughness=None,
        metalness=None,
        metallic=None,
    )
    geometry = SimpleNamespace(colors=SimpleNamespace(data=np.ones((3, 3), dtype=np.float32)))
    redraws = []
    renderer._objects = {name: SimpleNamespace(material=material, geometry=geometry)}
    renderer._materials = {}
    renderer._gfx = SimpleNamespace(
        MeshBasicMaterial=None,
        MeshPhysicalMaterial=None,
        MeshStandardMaterial=None,
    )
    renderer._kinds = {name: "mesh"}
    renderer._geometry_color_sources = {}
    renderer._ibl_loaded = False
    renderer._coerce_material = lambda _payload: MaterialPayload(
        base_color=(0.2, 0.4, 0.6, 1.0),
        roughness=0.3,
        metallic=0.1,
    )
    renderer._record_frame_update_metric = lambda *_args, **_kwargs: None
    renderer._apply_material_alpha_state = lambda *_args, **_kwargs: None
    renderer._load_texture_binding = lambda *_args, **_kwargs: None
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_named_material(name, {"color": [0.2, 0.4, 0.6]}) is True

    assert material.color_mode == "uniform"
    assert material.color == pytest.approx((0.2, 0.4, 0.6, 1.0))
    assert redraws == [True]


def test_set_named_material_multiplier_preserves_vertex_color_buffer():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "target:person::mesh"
    material = SimpleNamespace(
        color=None,
        color_mode="uniform",
        opacity=None,
        roughness=None,
        metalness=None,
        metallic=None,
    )
    colors = np.asarray(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.4, 0.2]],
        dtype=np.float32,
    )
    geometry = SimpleNamespace(colors=SimpleNamespace(data=colors.copy()))
    renderer._objects = {name: SimpleNamespace(material=material, geometry=geometry)}
    renderer._materials = {}
    renderer._material_apply_signatures = {}
    renderer._gfx = SimpleNamespace(
        MeshBasicMaterial=None,
        MeshPhysicalMaterial=None,
        MeshStandardMaterial=None,
    )
    renderer._kinds = {name: "mesh"}
    renderer._geometry_color_sources = {name: SurfaceColorSource.VERTEX}
    renderer._geometry_texcoords_available = {name: False}
    renderer._ibl_loaded = False
    renderer._clipping_planes = ()
    renderer._coerce_material = lambda payload: payload
    renderer._record_frame_update_metric = lambda *_args, **_kwargs: None
    renderer._apply_material_alpha_state = lambda *_args, **_kwargs: None
    renderer._load_texture_binding = lambda *_args, **_kwargs: None
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    renderer.request_redraw = lambda: None

    highlighted = MaterialPayload(color_multiplier=(1.0, 0.3, 0.3))
    assert renderer.set_named_material(name, highlighted) is True

    assert material.color_mode == "vertex"
    assert material.color == pytest.approx((1.0, 0.3, 0.3, 1.0))
    np.testing.assert_array_equal(geometry.colors.data, colors)


def test_textured_multiplier_reuses_pygfx_texture_cache(monkeypatch, tmp_path):
    from PIL import Image as PILImage

    from visualizer.src.model import RenderObject
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    texture_path = tmp_path / "albedo.png"
    PILImage.fromarray(
        np.full((2, 2, 4), 255, dtype=np.uint8),
        mode="RGBA",
    ).save(texture_path)
    payload = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        triangle_uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "scene:textured::mesh"
    material = SimpleNamespace(
        color=None,
        color_mode="uniform",
        opacity=None,
        roughness=None,
        metalness=None,
        metallic=None,
        map=None,
    )
    geometry = SimpleNamespace(texcoords=SimpleNamespace(data=np.ones((3, 2))))
    native_object = SimpleNamespace(material=material, geometry=geometry)
    texture_allocations = []

    def _make_texture(data, *, dim):
        texture = SimpleNamespace(data=data, dim=dim)
        texture_allocations.append(texture)
        return texture

    renderer._objects = {}
    renderer._name_to_handle = {}
    renderer._materials = {}
    renderer._material_apply_signatures = {}
    renderer._gfx = SimpleNamespace(
        MeshBasicMaterial=None,
        MeshPhysicalMaterial=None,
        MeshStandardMaterial=None,
        TextureMap=None,
        Texture=_make_texture,
    )
    renderer._kinds = {}
    renderer._geometry_color_sources = {}
    renderer._geometry_texcoords_available = {}
    renderer._texture_cache = {}
    renderer._texture_source_identities = {}
    renderer._render_object_snapshots = {}
    renderer._dirty_render_object_geometry = set()
    renderer._ibl_loaded = False
    renderer._clipping_planes = ()
    renderer._coerce_material = lambda payload: payload
    renderer._record_frame_update_metric = lambda *_args, **_kwargs: None
    renderer._apply_material_alpha_state = lambda *_args, **_kwargs: None
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    renderer.request_redraw = lambda: None
    renderer._apply_render_object_transform = lambda _obj: True
    renderer.set_named_visibility = lambda *_args: True
    geometry_uploads = []

    def _ensure_named_geometry(*, name, geometry, material, **_kwargs):
        geometry_uploads.append(geometry)
        renderer._objects[name] = native_object
        renderer._name_to_handle[name] = 1
        renderer._kinds[name] = "mesh"
        renderer._geometry_color_sources[name] = SurfaceColorSource.MATERIAL
        renderer._geometry_texcoords_available[name] = True
        return renderer.set_named_material(name, material)

    renderer.ensure_named_geometry = _ensure_named_geometry

    initial = RenderObject(
        id=name,
        payload=payload,
        material=MaterialPayload(texture_path=str(texture_path)),
    )
    assert renderer.ensure_object(initial) is True
    cached_texture = material.map
    assert cached_texture is not None
    assert texture_allocations == [cached_texture]
    assert geometry_uploads == [payload]

    from visualizer.src.materials.texture_assets import clear_decoded_texture_cache

    clear_decoded_texture_cache()

    monkeypatch.setattr(
        PILImage,
        "open",
        lambda *_args, **_kwargs: pytest.fail("cached highlight must not read the texture"),
    )

    highlighted = RenderObject(
        id=name,
        payload=payload,
        material=MaterialPayload(
            texture_path=str(texture_path),
            color_multiplier=(1.0, 0.3, 0.3),
        ),
    )
    assert renderer.ensure_object(highlighted) is True

    assert geometry_uploads == [payload]
    assert native_object.geometry is geometry
    assert texture_allocations == [cached_texture]
    assert material.color_mode == "auto"
    assert material.map is cached_texture
    assert material.color == pytest.approx((1.0, 0.3, 0.3, 1.0))


def test_set_named_material_uses_auto_mode_for_textured_mesh(monkeypatch, tmp_path):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.types.render_payloads import MaterialPayload

    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    texture_path = tmp_path / "albedo.png"
    texture_path.write_bytes(b"texture")

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "target:car::mesh"
    material = SimpleNamespace(
        color=None,
        color_mode="uniform",
        opacity=None,
        roughness=None,
        metalness=None,
        metallic=None,
        map=None,
    )
    redraws = []
    renderer._objects = {
        name: SimpleNamespace(
            material=material,
            geometry=SimpleNamespace(texcoords=SimpleNamespace(data=np.ones((3, 2)))),
        )
    }
    renderer._materials = {}
    renderer._gfx = SimpleNamespace(
        MeshBasicMaterial=None,
        MeshPhysicalMaterial=None,
        MeshStandardMaterial=None,
    )
    renderer._kinds = {name: "mesh"}
    renderer._geometry_color_sources = {}
    renderer._ibl_loaded = False
    renderer._coerce_material = lambda _payload: MaterialPayload(
        base_color=(1.0, 1.0, 1.0, 1.0),
        roughness=0.3,
        metallic=0.1,
        texture_path=str(texture_path),
    )
    renderer._record_frame_update_metric = lambda *_args, **_kwargs: None
    renderer._apply_material_alpha_state = lambda *_args, **_kwargs: None
    renderer._load_texture_binding = lambda *_args, **_kwargs: "texture-binding"
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_named_material(name, {"texture_path": str(texture_path)}) is True

    assert material.color_mode == "auto"
    assert material.map == "texture-binding"
    assert redraws == [True]


def test_set_named_material_drops_textures_without_uvs(monkeypatch, tmp_path):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.types.render_payloads import MaterialPayload

    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    texture_path = tmp_path / "albedo.png"
    texture_path.write_bytes(b"texture")
    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "scene:ground::mesh"
    material = SimpleNamespace(
        color=None,
        color_mode="auto",
        opacity=None,
        roughness=None,
        metalness=None,
        metallic=None,
        map="stale",
    )
    renderer._objects = {name: SimpleNamespace(material=material, geometry=SimpleNamespace())}
    renderer._materials = {}
    renderer._gfx = SimpleNamespace(
        MeshBasicMaterial=None,
        MeshPhysicalMaterial=None,
        MeshStandardMaterial=None,
    )
    renderer._kinds = {name: "mesh"}
    renderer._geometry_color_sources = {}
    renderer._ibl_loaded = False
    renderer._coerce_material = lambda _payload: MaterialPayload(
        base_color=(0.22, 0.22, 0.254, 0.75),
        roughness=0.3,
        metallic=0.1,
        texture_path=str(texture_path),
    )
    renderer._record_frame_update_metric = lambda *_args, **_kwargs: None
    renderer._apply_material_alpha_state = lambda *_args, **_kwargs: None
    renderer._load_texture_binding = lambda *_args, **_kwargs: "texture-binding"
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    renderer.request_redraw = lambda: None

    assert renderer.set_named_material(name, {"texture_path": str(texture_path)}) is True

    assert material.map is None
    assert material.color_mode == "uniform"
    assert material.color == pytest.approx((0.22, 0.22, 0.254, 0.75))


def test_pygfx_texture_policy_keeps_maps_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    paths = {}
    for key in ("albedo", "normal", "roughness", "ao", "metallic"):
        path = tmp_path / f"{key}.png"
        path.write_bytes(b"texture")
        paths[key] = str(path)

    material = apply_texture_policy_to_material_payload(
        MaterialPayload(
            base_color=(0.22, 0.22, 0.254, 0.75),
            texture_path=paths["albedo"],
            normal_map_path=paths["normal"],
            roughness_map_path=paths["roughness"],
            ao_map_path=paths["ao"],
            metallic_map_path=paths["metallic"],
        ),
        context="test",
    )

    assert material.texture_path == paths["albedo"]
    assert material.base_color == pytest.approx((1.0, 1.0, 1.0, 0.75))
    assert material.normal_map_path == paths["normal"]
    assert material.roughness_map_path == paths["roughness"]
    assert material.ao_map_path == paths["ao"]
    assert material.metallic_map_path == paths["metallic"]


def test_pygfx_texture_policy_strips_maps_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    paths = {}
    for key in ("albedo", "normal", "roughness", "ao", "metallic"):
        path = tmp_path / f"{key}.png"
        path.write_bytes(b"texture")
        paths[key] = str(path)

    material = apply_texture_policy_to_material_payload(
        MaterialPayload(
            base_color=(0.22, 0.22, 0.254, 0.75),
            texture_path=paths["albedo"],
            normal_map_path=paths["normal"],
            roughness_map_path=paths["roughness"],
            ao_map_path=paths["ao"],
            metallic_map_path=paths["metallic"],
        ),
        context="test",
    )

    assert material.base_color == pytest.approx((0.22, 0.22, 0.254, 0.75))
    assert material.texture_path is None
    assert material.normal_map_path is None
    assert material.roughness_map_path is None
    assert material.ao_map_path is None
    assert material.metallic_map_path is None


def test_set_triangle_uvs_updates_native_scene_mesh_and_invalidates_cache():
    o3d = pytest.importorskip("open3d")

    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._payload_cache = {}

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=np.int32))
    renderer._payload_cache[id(mesh)] = (("mesh", 3, 1, 0, 0, 0), object())

    uvs = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    renderer.set_triangle_uvs(mesh, uvs)

    assert mesh.has_triangle_uvs()
    np.testing.assert_allclose(np.asarray(mesh.triangle_uvs), uvs)
    assert id(mesh) not in renderer._payload_cache


def test_set_geometry_vertices_invalidates_external_payload_cache():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    mesh = SimpleNamespace(vertices=np.zeros((3, 3), dtype=np.float64))
    calls = []

    renderer._payload_cache = {id(mesh): (("mesh", 3, 1, 0, 0, 0), object())}
    renderer._external_name_for_geometry = lambda geometry: "mesh" if geometry is mesh else None
    renderer.update_geometry_in_visualizer = lambda geometry: calls.append(geometry)

    next_vertices = np.asarray(
        [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    renderer.set_geometry_vertices(mesh, next_vertices)

    np.testing.assert_allclose(mesh.vertices, next_vertices)
    assert id(mesh) not in renderer._payload_cache
    assert calls == [mesh]


def test_remap_external_geometry_name_moves_mapping_to_new_object():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    old_geometry = object()
    new_geometry = object()
    renderer._external_geometry_names = {id(old_geometry): "target:old::mesh"}
    renderer._payload_cache = {
        id(old_geometry): (("mesh", 3, 1, 0, 0, 0), object()),
        id(new_geometry): (("mesh", 3, 1, 0, 0, 0), object()),
    }
    renderer._geometry_payload_cache_keys = {id(old_geometry): "target_runtime/old"}

    assert renderer.remap_external_geometry_name(
        old_geometry=old_geometry,
        new_geometry=new_geometry,
        name="target:new::mesh",
    )

    assert id(old_geometry) not in renderer._external_geometry_names
    assert renderer._external_geometry_names[id(new_geometry)] == "target:new::mesh"
    assert id(old_geometry) not in renderer._payload_cache
    assert id(new_geometry) not in renderer._payload_cache
    assert id(old_geometry) not in renderer._geometry_payload_cache_keys


def test_set_ibl_reports_load_failure_without_redraw():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    redraws = []

    renderer._deferred_default_ibl_name = "neutral_outdoor"
    renderer._deferred_ibl_load_scheduled = True
    renderer._load_ibl_into_scene = lambda name: False
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_ibl("missing") is False
    assert renderer._deferred_default_ibl_name is None
    assert renderer._deferred_ibl_load_scheduled is False
    assert redraws == []


def test_stale_deferred_ibl_callback_cannot_mutate_new_session(monkeypatch):
    from PySide6 import QtCore

    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    callbacks = []
    monkeypatch.setattr(
        QtCore,
        "QTimer",
        SimpleNamespace(singleShot=lambda _delay, callback: callbacks.append(callback)),
    )
    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._defer_default_ibl = True
    renderer._ibl_loaded = False
    renderer._deferred_default_ibl_name = "neutral_outdoor"
    renderer._deferred_ibl_load_scheduled = False
    renderer._session_generation = 8
    renderer._load_deferred_default_ibl = Mock()

    renderer._schedule_deferred_default_ibl_load()
    renderer._session_generation = 9
    callbacks[0]()

    renderer._load_deferred_default_ibl.assert_not_called()


def test_set_ibl_extracts_legacy_ktx_name_and_redraws_on_success():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    loaded = []
    redraws = []

    renderer._deferred_default_ibl_name = None
    renderer._deferred_ibl_load_scheduled = False
    renderer._load_ibl_into_scene = lambda name: loaded.append(name) or True
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_ibl("/assets/neutral_outdoor_ibl.ktx") is True
    assert loaded == ["neutral_outdoor"]
    assert redraws == [True]


def test_ibl_intensity_does_not_scale_direct_lights():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    ambient = SimpleNamespace(intensity=1.5)
    key = SimpleNamespace(intensity=3.0)
    fill = SimpleNamespace(intensity=1.0)
    head = SimpleNamespace(intensity=1.2)
    ibl_values = []
    redraws = []

    renderer._ambient_light = ambient
    renderer._key_light = key
    renderer._fill_light = fill
    renderer._head_light = head
    renderer._base_ambient_intensity = 1.5
    renderer._base_key_intensity = 3.0
    renderer._base_fill_intensity = 1.0
    renderer._headlight_intensity = 1.2
    renderer._headlight_enabled = True
    renderer._ibl_manager = SimpleNamespace(set_intensity=ibl_values.append)
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_ibl_intensity(80000.0) is True

    assert ibl_values == [pytest.approx(80000.0 / 30000.0)]
    assert ambient.intensity == pytest.approx(1.5)
    assert key.intensity == pytest.approx(3.0)
    assert fill.intensity == pytest.approx(1.0)
    assert head.intensity == pytest.approx(1.2)
    assert redraws == [True]


def test_pygfx_color_fidelity_mode_swaps_and_restores_mesh_material():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.types.render_payloads import MaterialPayload

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    name = "scene:wall_1::mesh"
    obj = SimpleNamespace(material=_FakeMeshStandardMaterial())
    redraws = []

    renderer._objects = {name: obj}
    renderer._materials = {
        name: MaterialPayload(base_color=(0.2, 0.4, 0.6, 1.0), roughness=0.3, metallic=0.1)
    }
    renderer._gfx = _FakeMeshMaterialGfx
    renderer._kinds = {name: "mesh"}
    renderer._geometry_color_sources = {}
    renderer._ibl_loaded = False
    renderer._batch_mode = False
    renderer._batch_redraw_pending = False
    renderer._initialized = False
    renderer._unlit_mode_enabled = False
    renderer._coerce_material = lambda payload: payload
    renderer._record_frame_update_metric = lambda *_args, **_kwargs: None
    renderer._apply_material_alpha_state = lambda *_args, **_kwargs: None
    renderer._load_texture_binding = lambda *_args, **_kwargs: None
    renderer._apply_clipping_to_material = lambda *_args, **_kwargs: None
    renderer.request_redraw = lambda: redraws.append(True)

    assert renderer.set_color_fidelity_mode(True) is True
    assert type(obj.material) is _FakeMeshBasicMaterial
    assert obj.material.color == pytest.approx((0.2, 0.4, 0.6, 1.0))

    assert renderer.set_color_fidelity_mode(False) is True
    assert type(obj.material) is _FakeMeshStandardMaterial
    assert obj.material.color == pytest.approx((0.2, 0.4, 0.6, 1.0))

    assert redraws


def test_coverage_colorbar_overlay_html_includes_metric_height_and_no_data():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)

    html = renderer._coverage_colorbar_overlay_html(
        {
            "metric_name": "best_path_loss_db",
            "value_min": 71.25,
            "value_max": 133.75,
            "selected_height_value": 1.5,
            "no_data_fraction": 0.25,
        }
    )

    assert "Coverage: Best path loss (dB)" in html
    assert "71.2" in html
    assert "133.8" in html
    assert "Height: 1.50 m" in html
    assert "No data (transparent / hidden): 25.0%" in html
    assert "logarithmic" not in html
    assert "#333333" not in html


def test_coverage_colorbar_identifies_logarithmic_linear_rf_colors():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)

    html = renderer._coverage_colorbar_overlay_html(
        {
            "metric_name": "path_gain_linear/TX1",
            "value_min": 1.0e-12,
            "value_max": 1.0e-6,
            "color_scale": "logarithmic",
        }
    )

    assert "Coverage: Path gain (TX1)" in html
    assert "1e-12" in html
    assert "1e-06" in html
    assert "Color scale: logarithmic" in html


@pytest.mark.parametrize(
    ("metric_name", "low_color", "high_color"),
    [
        ("best_path_loss_db", "#006837", "#a50026"),
        ("sinr_db", "#a50026", "#006837"),
    ],
)
def test_coverage_colorbar_gradient_matches_metric_direction(
    metric_name,
    low_color,
    high_color,
):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)

    html = renderer._coverage_colorbar_overlay_html(
        {
            "metric_name": metric_name,
            "value_min": 0.0,
            "value_max": 1.0,
        }
    )

    assert html.index(f"color:{low_color}") < html.index(f"color:{high_color}")
    assert "font-family" not in html.lower()


def test_serving_tx_coverage_overlay_uses_categories_without_no_service():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)

    html = renderer._coverage_colorbar_overlay_html(
        {
            "metric_name": "serving_tx",
            "tx_names": ["gNB A", "gNB B"],
            "tx_count": 2,
            "value_min": 0.0,
            "value_max": 1.0,
            "selected_height_value": 1.5,
            "no_data_fraction": 0.5,
        }
    )

    assert "Coverage: Serving TX" in html
    assert "gNB A" in html
    assert "gNB B" in html
    assert "No service" not in html
    assert "-1" not in html
    assert "No data" not in html
    assert "Height: 1.50 m" in html


def test_coverage_hud_overlay_updates_and_clears():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    overlays = []
    cleared = []
    renderer._set_hud_overlay = lambda overlay_id, **kwargs: overlays.append((overlay_id, kwargs))
    renderer._clear_hud_overlay = lambda overlay_id: cleared.append(overlay_id)

    view_model = SimpleNamespace(
        show_coverage=True,
        coverage_vertices=np.ones((4, 3), dtype=np.float32),
        coverage_metadata={
            "metric_name": "path_loss_db/TX1",
            "value_min": 80.0,
            "value_max": 120.0,
            "selected_height_value": 10.0,
            "valid_cell_count": 1,
        },
    )

    assert renderer._update_coverage_hud_overlay(view_model) is True
    assert overlays[0][0] == "coverage_colorbar"
    assert overlays[0][1]["corner"] == "top_right"
    assert overlays[0][1]["priority"] == 15
    assert "Coverage: Path loss (TX1) (dB)" in overlays[0][1]["html"]

    assert (
        renderer._update_coverage_hud_overlay(
            SimpleNamespace(show_coverage=False, coverage_vertices=None)
        )
        is False
    )
    assert cleared[-1] == "coverage_colorbar"

    assert (
        renderer._update_coverage_hud_overlay(
            SimpleNamespace(
                show_coverage=True,
                coverage_vertices=np.ones((4, 3), dtype=np.float32),
                coverage_metadata={"valid_cell_count": 0},
            )
        )
        is False
    )
    assert cleared[-1] == "coverage_colorbar"

    assert (
        renderer._update_coverage_hud_overlay(
            SimpleNamespace(
                show_coverage=True,
                coverage_vertices=np.empty((0, 3), dtype=np.float32),
                coverage_metadata={"valid_cell_count": 0},
            )
        )
        is False
    )
    assert cleared[-1] == "coverage_colorbar"


def test_hud_overlay_repeated_state_is_a_qt_noop():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _Container:
        @staticmethod
        def width() -> int:
            return 640

    class _Label:
        def __init__(self) -> None:
            self.visible_values = []
            self.calls = {
                "style": 0,
                "text": 0,
                "adjust": 0,
                "visible": 0,
                "raise": 0,
                "move": 0,
            }

        def setStyleSheet(self, _value) -> None:
            self.calls["style"] += 1

        def setText(self, _value) -> None:
            self.calls["text"] += 1

        def adjustSize(self) -> None:
            self.calls["adjust"] += 1

        def setVisible(self, value) -> None:
            self.visible_values.append(bool(value))
            self.calls["visible"] += 1

        def raise_(self) -> None:
            self.calls["raise"] += 1

        def width(self) -> int:
            return 100

        def height(self) -> int:
            return 20

        def move(self, _x, _y) -> None:
            self.calls["move"] += 1

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._container = _Container()
    label = _Label()
    renderer._hud_overlay_labels = {"legend": label}
    renderer._hud_overlay_specs = {}
    redraws = []
    renderer.request_redraw = lambda: redraws.append(True)

    state = {
        "html": "<b>Legend</b>",
        "visible": True,
        "role": "legend",
        "corner": "top_right",
        "priority": 10,
    }
    renderer._set_hud_overlay("legend", **state)
    after_first = dict(label.calls)
    renderer._set_hud_overlay("legend", **state)

    assert label.calls == after_first
    assert len(redraws) == 1
    assert after_first == {
        "style": 1,
        "text": 1,
        "adjust": 1,
        "visible": 1,
        "raise": 1,
        "move": 1,
    }
    assert label.visible_values == [True]

    shorter_state = {**state, "html": "<b>Short</b>"}
    renderer._set_hud_overlay("legend", **shorter_state)
    after_shorter = dict(label.calls)
    assert label.visible_values[-2:] == [False, True]
    assert len(redraws) == 2

    renderer._set_hud_overlay("legend", **shorter_state)
    assert label.calls == after_shorter
    assert len(redraws) == 2

    setattr(label, "_orchav_hud_layout_hidden", True)
    renderer._set_hud_overlay("legend", **shorter_state)
    assert label.calls["move"] == 3
    assert label.calls["raise"] == 3
    assert label.calls["visible"] == 4
    assert len(redraws) == 3

    renderer._clear_hud_overlay("legend")
    after_clear = dict(label.calls)
    renderer._clear_hud_overlay("legend")
    assert label.calls == after_clear
    assert len(redraws) == 4


def test_active_ibl_skips_coverage_material():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    applied = []
    coverage_name = PygfxRenderer.COVERAGE_MESH_NAME
    scene_mat = SimpleNamespace(env_map=None)
    coverage_mat = SimpleNamespace(env_map=None)

    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._scene = object()
    renderer._skybox_visible = False
    renderer._objects = {
        "scene_mesh": SimpleNamespace(material=scene_mat),
        coverage_name: SimpleNamespace(material=coverage_mat),
    }
    renderer._materials = {}
    renderer._ibl_loaded = False
    renderer._sync_solid_background = lambda: True
    renderer._ibl_manager = SimpleNamespace(
        apply_to_scene=lambda scene: True,
        set_skybox_visible=lambda show, scene: True,
        apply_to_material=lambda mat: applied.append(mat),
    )

    assert renderer._apply_active_ibl_to_scene_materials() is True
    assert applied == [scene_mat]
    assert renderer._ibl_loaded is True


def test_ibl_manager_scene_environment_mode(monkeypatch):
    monkeypatch.setenv("ORCHAV_PYGFX_USE_SCENE_ENVIRONMENT", "1")

    mgr = PygfxIBLManager(_FakeGfx, Path("/tmp"))
    texture = object()
    mgr._texture_cache[mgr._current_name] = texture

    scene = _FakeScene()
    assert mgr.apply_to_scene(scene) is True
    assert isinstance(scene.environment, _FakeTextureMap)
    assert scene.environment.texture is texture
    assert len(scene.children) == 1

    material = SimpleNamespace(env_map="sentinel", env_map_intensity=0.0)
    mgr.apply_to_material(material)
    assert material.env_map is None
    assert material.env_map_intensity == 1.0


def test_ibl_manager_material_env_map_fallback(monkeypatch):
    monkeypatch.setenv("ORCHAV_PYGFX_USE_SCENE_ENVIRONMENT", "0")

    mgr = PygfxIBLManager(_FakeGfx, Path("/tmp"))
    texture = object()
    mgr._texture_cache[mgr._current_name] = texture

    scene = _FakeScene()
    assert mgr.apply_to_scene(scene) is True
    assert scene.environment is None
    assert len(scene.children) == 1

    material = SimpleNamespace(env_map=None, env_map_intensity=0.0)
    mgr.apply_to_material(material)
    assert material.env_map is texture
    assert material.env_map_intensity == 1.0
