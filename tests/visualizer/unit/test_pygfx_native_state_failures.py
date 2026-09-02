"""Failure/retry coverage for pygfx-owned native renderer state."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin
from visualizer.src.renderers.pygfx.labels import PygfxLabelMixin
from visualizer.src.renderers.pygfx.materials import PygfxMaterialMixin
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    SurfaceColorSource,
    TextLabelPayload,
)


class _FailOnceColorMaterial:
    def __init__(self) -> None:
        self._color = (1.0, 1.0, 1.0, 1.0)
        self.color_set_attempts = 0

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value) -> None:
        self.color_set_attempts += 1
        if self.color_set_attempts == 1:
            raise RuntimeError("transient native color failure")
        self._color = value


class _MaterialHarness(PygfxMaterialMixin):
    def __init__(self) -> None:
        self.name = "scene:test::mesh"
        self.native_material = _FailOnceColorMaterial()
        self._objects = {
            self.name: SimpleNamespace(material=self.native_material, geometry=None),
        }
        self._kinds = {self.name: "mesh"}
        self._materials = {
            self.name: MaterialPayload(base_color=(0.1, 0.2, 0.3, 1.0)),
        }
        self._material_apply_signatures = {}
        self._geometry_color_sources = {
            self.name: SurfaceColorSource.MATERIAL,
        }
        self._geometry_texcoords_available = {}
        self._unlit_mode_enabled = False
        self._ibl_loaded = False
        self._clipping_planes = ()
        self._gfx = SimpleNamespace(
            MeshBasicMaterial=None,
            MeshPhysicalMaterial=None,
            MeshStandardMaterial=None,
        )
        self.redraws = 0

    def _record_frame_update_metric(self, *_args, **_kwargs) -> None:
        pass

    def request_redraw(self) -> None:
        self.redraws += 1


def test_pygfx_material_failure_preserves_applied_cache_and_retries() -> None:
    renderer = _MaterialHarness()
    previous = renderer._materials[renderer.name]
    requested = MaterialPayload(base_color=(0.8, 0.4, 0.2, 1.0))
    payload = object()
    render_object = SimpleNamespace(
        id=renderer.name,
        payload=payload,
        material_payload=requested,
        metadata={},
        is_edge=False,
        transform_matrix=np.eye(4, dtype=np.float32),
        visible=True,
    )
    renderer._render_object_snapshots = {
        renderer.name: (payload, False, SurfaceColorSource.MATERIAL),
    }
    renderer._name_to_handle = {renderer.name: 1}
    renderer._apply_render_object_transform = lambda _obj: True
    renderer.set_named_visibility = lambda _name, _visible: True

    assert PygfxGeometryMixin.ensure_object(renderer, render_object) is False
    assert renderer._materials[renderer.name] == previous
    assert renderer.name not in renderer._material_apply_signatures
    assert renderer.native_material.color_set_attempts == 1
    assert renderer.redraws == 0

    assert PygfxGeometryMixin.ensure_object(renderer, render_object) is True
    assert renderer._materials[renderer.name].base_color == pytest.approx(requested.base_color)
    assert renderer.native_material.color_set_attempts == 2
    assert renderer.redraws == 1

    assert PygfxGeometryMixin.ensure_object(renderer, render_object) is True
    assert renderer.native_material.color_set_attempts == 2
    assert renderer.redraws == 1


class _FailOnceOpacityAfterColorMaterial:
    """Native material that fails after accepting the requested color."""

    def __init__(self) -> None:
        self._color = (1.0, 1.0, 1.0, 1.0)
        self._opacity = 1.0
        self.fail_opacity_once = False

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value) -> None:
        self._color = value

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value) -> None:
        if self.fail_opacity_once:
            self.fail_opacity_once = False
            raise RuntimeError("transient native opacity failure")
        self._opacity = value


def test_pygfx_material_failure_repairs_native_state_when_desired_reverts() -> None:
    renderer = _MaterialHarness()
    native = _FailOnceOpacityAfterColorMaterial()
    renderer.native_material = native
    renderer._objects[renderer.name].material = native
    original = renderer._materials[renderer.name]
    requested = MaterialPayload(base_color=(0.8, 0.4, 0.2, 0.6))
    payload = object()
    renderer._render_object_snapshots = {
        renderer.name: (payload, False, SurfaceColorSource.MATERIAL),
    }
    renderer._name_to_handle = {renderer.name: 1}
    renderer._apply_render_object_transform = lambda _obj: True
    renderer.set_named_visibility = lambda _name, _visible: True

    assert renderer.set_named_material(renderer.name, original) is True
    native.fail_opacity_once = True
    failed_update = SimpleNamespace(
        id=renderer.name,
        payload=payload,
        material_payload=requested,
        metadata={},
        is_edge=False,
        transform_matrix=np.eye(4, dtype=np.float32),
        visible=True,
    )
    assert PygfxGeometryMixin.ensure_object(renderer, failed_update) is False
    assert native.color == pytest.approx(requested.base_color)
    assert renderer.name not in renderer._material_apply_signatures

    reverted = SimpleNamespace(
        id=renderer.name,
        payload=payload,
        material_payload=original,
        metadata={},
        is_edge=False,
        transform_matrix=np.eye(4, dtype=np.float32),
        visible=True,
    )
    assert PygfxGeometryMixin.ensure_object(renderer, reverted) is True
    assert native.color == pytest.approx(original.base_color)
    assert renderer.name in renderer._material_apply_signatures


def test_pygfx_failed_texture_decode_clears_previous_maps(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    paths = {}
    for key in ("albedo", "normal", "roughness", "ao", "metallic"):
        path = tmp_path / f"replacement_{key}.png"
        path.write_bytes(b"corrupt texture")
        paths[key] = str(path)

    renderer = _MaterialHarness()
    stale_maps = {key: object() for key in paths}
    native = SimpleNamespace(
        color=(1.0, 1.0, 1.0, 1.0),
        opacity=1.0,
        map=stale_maps["albedo"],
        normal_map=stale_maps["normal"],
        normal_scale=(3.0, 3.0),
        roughness_map=stale_maps["roughness"],
        ao_map=stale_maps["ao"],
        metalness_map=stale_maps["metallic"],
    )
    renderer.native_material = native
    renderer._objects[renderer.name].material = native
    renderer._geometry_texcoords_available[renderer.name] = True
    renderer._load_texture_binding = lambda *_args, **_kwargs: None

    requested = MaterialPayload(
        texture_path=paths["albedo"],
        normal_map_path=paths["normal"],
        normal_map_strength=3.0,
        roughness_map_path=paths["roughness"],
        ao_map_path=paths["ao"],
        metallic_map_path=paths["metallic"],
    )

    assert renderer.set_named_material(renderer.name, requested) is True
    assert native.map is None
    assert native.normal_map is None
    assert native.normal_scale == (1.0, 1.0)
    assert native.roughness_map is None
    assert native.ao_map is None
    assert native.metalness_map is None


def test_pygfx_text_label_replacement_aborts_when_native_removal_fails() -> None:
    name = "target:walker::label"
    previous_native = object()
    removal_attempts: list[str] = []
    renderer = SimpleNamespace(
        _initialized=True,
        _scene=object(),
        _name_to_handle={name: 1},
        _objects={name: previous_native},
        _gfx=SimpleNamespace(),
    )

    def fail_remove(object_name: str) -> bool:
        removal_attempts.append(object_name)
        return False

    renderer.remove_named_geometry = fail_remove
    payload = TextLabelPayload(text="Walker", font_size=0.3)

    assert PygfxGeometryMixin._ensure_text_label_payload(renderer, name, payload) is False
    assert removal_attempts == [name]
    assert renderer._objects[name] is previous_native
    assert renderer._name_to_handle[name] == 1


class _FailOnceSceneRemove:
    def __init__(self, child: object) -> None:
        self.children = [child]
        self.attempts = 0

    def remove(self, child: object) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("transient native remove failure")
        self.children.remove(child)


class _TopologyReplacementHarness(PygfxGeometryMixin):
    def __init__(self) -> None:
        self.name = "beamforming:topology_retry:mesh"
        self.native = SimpleNamespace(visible=True, material=None)
        self._scene = _FailOnceSceneRemove(self.native)
        self._static_group = None
        self._initialized = True
        self._name_to_handle = {self.name: 1}
        self._handle_to_name = {1: self.name}
        self._objects = {self.name: self.native}
        self._kinds = {self.name: "mesh"}
        self._topology = {self.name: ("old", ())}
        self._materials = {}
        self._material_apply_signatures = {}
        self._transforms = {}
        self._positions = {}
        self._hidden = set()
        self._edge_geometry_names = set()
        self._geometry_color_sources = {}
        self._geometry_upload_center = {}
        self._geometry_texcoords_available = {}
        self._pick_metadata = {}
        self._reverse_objects = {id(self.native): self.name}
        self._external_geometry_names = {}
        self._normal_line_overlays = {}
        self._render_object_snapshots = {}
        self.create_calls = 0

    def _record_frame_update_metric(self, *_args, **_kwargs) -> None:
        pass

    def _prepare_geometry_buffers(self, _payload, *, line_strip=False):
        return {"line_strip": line_strip}

    def _get_buffer_layout_signature(self, _payload, *, line_strip=False, buffers=None):
        return ("new", (("line_strip", (int(line_strip),), "bool"),))

    def _unregister_label_layout(self, _name: str) -> None:
        pass

    def _apply_named_visual_overrides(self, _name: str, *, is_edge=False) -> bool:
        return False

    def _create_entity(self, name, _payload, *, buffers=None, layout_signature=None):
        self.create_calls += 1
        native = SimpleNamespace(visible=True, material=None)
        self._name_to_handle[name] = 2
        self._handle_to_name[2] = name
        self._objects[name] = native
        self._kinds[name] = "mesh"
        self._topology[name] = layout_signature
        self._reverse_objects[id(native)] = name
        self._scene.children.append(native)
        return 2


def test_pygfx_topology_replacement_aborts_on_native_remove_failure_and_retries() -> None:
    renderer = _TopologyReplacementHarness()
    payload = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )

    assert renderer.ensure_named_geometry(renderer.name, payload) is False
    assert renderer.create_calls == 0
    assert renderer._objects[renderer.name] is renderer.native
    assert renderer._name_to_handle[renderer.name] == 1

    assert renderer.ensure_named_geometry(renderer.name, payload) is True
    assert renderer.create_calls == 1
    assert renderer._objects[renderer.name] is not renderer.native
    assert renderer._name_to_handle[renderer.name] == 2


class _FailOnceCoverageMaterial:
    def __init__(self, **_kwargs) -> None:
        object.__setattr__(self, "_failure_attr", None)
        object.__setattr__(self, "_failures_remaining", 0)
        self.color_mode = "auto"
        self.side = "front"
        self.opacity = 1.0
        self.color = (1.0, 1.0, 1.0, 1.0)
        self.env_map = "inherited"
        self.env_map_intensity = 1.0
        self.alpha_mode = "auto"
        self.depth_write = True
        self.depth_test = False

    def fail_once_on(self, attr: str) -> None:
        object.__setattr__(self, "_failure_attr", attr)
        object.__setattr__(self, "_failures_remaining", 1)

    def __setattr__(self, name, value) -> None:
        if (
            name == getattr(self, "_failure_attr", None)
            and getattr(self, "_failures_remaining", 0) > 0
        ):
            object.__setattr__(self, "_failures_remaining", 0)
            raise RuntimeError(f"transient native {name} failure")
        object.__setattr__(self, name, value)


class _CoverageMaterialHarness(PygfxMaterialMixin):
    COVERAGE_MESH_NAME = "coverage_mesh"

    def __init__(self, failure_attr: str) -> None:
        material = _FailOnceCoverageMaterial()
        material.fail_once_on(failure_attr)
        self._objects = {
            self.COVERAGE_MESH_NAME: SimpleNamespace(material=material),
        }
        self._kinds = {self.COVERAGE_MESH_NAME: "mesh"}
        self._gfx = SimpleNamespace(MeshBasicMaterial=_FailOnceCoverageMaterial)
        self._clipping_planes = ()
        self._ibl_manager = SimpleNamespace(_tracked_materials=[])
        self.redraws = 0

    def request_redraw(self) -> None:
        self.redraws += 1


@pytest.mark.parametrize("failure_attr", ["opacity", "depth_write"])
def test_pygfx_coverage_required_property_failure_retries(failure_attr: str) -> None:
    renderer = _CoverageMaterialHarness(failure_attr)

    assert renderer.set_coverage_transparency(0.4) is False
    assert renderer.redraws == 0

    assert renderer.set_coverage_transparency(0.4) is True
    assert renderer.redraws == 1
    material = renderer._objects[renderer.COVERAGE_MESH_NAME].material
    assert material.opacity == pytest.approx(0.4)
    assert material.depth_write is False


class _LabelLayoutHarness(PygfxLabelMixin):
    def __init__(self) -> None:
        self._label_anchor_groups = {}
        self._label_anchor_key_by_name = {}
        self._label_anchor_by_name = {}
        self._label_offset_by_name = {}
        self._label_layout_dirty_groups = set()
        self._geometry_upload_center = {}
        self._positions = {}
        self.native_names = {"node:tx_0::label", "node:rx_0::label"}
        self.failure_name: str | None = None
        self.failures_remaining = 0
        self.transform_calls: list[str] = []

    def has_named_geometry(self, name: str) -> bool:
        return name in self.native_names

    def set_named_transform(self, name: str, _transform: np.ndarray) -> bool:
        self.transform_calls.append(name)
        if name == self.failure_name and self.failures_remaining > 0:
            self.failures_remaining -= 1
            return False
        return True


@pytest.mark.parametrize("failure_name", ["node:tx_0::label", "node:rx_0::label"])
def test_pygfx_label_layout_requires_every_transform_and_retries(failure_name: str) -> None:
    renderer = _LabelLayoutHarness()
    anchor = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    offset = np.asarray([0.5, 0.0, 1.0], dtype=np.float32)
    tx_name = "node:tx_0::label"
    rx_name = "node:rx_0::label"

    assert renderer._register_and_layout_label(tx_name, anchor, offset) is True
    renderer.failure_name = failure_name
    renderer.failures_remaining = 1

    assert renderer._register_and_layout_label(rx_name, anchor, offset) is False
    assert renderer._label_layout_dirty_groups

    assert renderer._register_and_layout_label(rx_name, anchor, offset) is True
    assert not renderer._label_layout_dirty_groups
    calls_after_retry = list(renderer.transform_calls)

    assert renderer._register_and_layout_label(rx_name, anchor, offset) is True
    assert renderer.transform_calls == calls_after_retry
    assert tx_name in renderer._positions
    assert rx_name in renderer._positions
