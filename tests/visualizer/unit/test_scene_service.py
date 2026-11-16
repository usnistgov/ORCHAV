import json
import logging
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.visualizer.fixtures.mock_factories import RecordingBatchRenderer
from visualizer.src.backends import pygfx_scene_helpers
from visualizer.src.materials.catalog import resolve_pbr_material
from visualizer.src.model import RenderObjectState, make_text_label_state
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.scene import io as scene_io
from visualizer.src.scene.geometry_helpers import (
    create_building_labels,
    ensure_scene_mesh_render_state,
)
from visualizer.src.services import scene_service
from visualizer.src.services.cache_service import CacheService
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.services.object_identity import make_target_entry_geometry_name
from visualizer.src.services.scene_batches import (
    PreparedSceneMember,
    SceneBatch,
    SceneBatchRegistry,
)
from visualizer.src.services.scene_service import SceneService
from visualizer.src.state import get_beamforming_state_defaults
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
    TextLabelPayload,
)


class _DummyTask:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _DummyProgress:
    def __init__(self):
        self.calls: list[str] = []

    def task(self, description: str):
        self.calls.append(description)
        return _DummyTask()


class _DummyUIController:
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def populate_controls(self):
        self.visualizer.populate_called = True

    def populate_material_filters(self):
        self.visualizer.materials_refreshed = True


class _HashableDummy:
    """A hashable dummy object for testing."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._hash = id(self)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other


def _mesh_payload(offset: float = 0.0) -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray(
            [
                [offset, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset, 1.0, 0.0],
            ],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )


def _scene_state(stable_id: str, offset: float = 0.0) -> RenderObjectState:
    return RenderObjectState(
        id=f"scene:{stable_id}::mesh",
        payload=_mesh_payload(offset),
        metadata={"type": "scene_mesh"},
    )


def _material_scene_entry(
    stable_id: str,
    *,
    offset: float = 0.0,
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict:
    """Build one canonical scene entry for ownership-transition tests."""
    return {
        "name": stable_id,
        "stable_mesh_id": stable_id,
        "entry_type": "mesh",
        "mesh": _scene_state(stable_id, offset),
        "visible": True,
        "material_type": "concrete",
        "pbr_properties": {
            "color": list(color),
            "roughness": 0.6,
            "metallic": 0.0,
            "reflectance": 0.4,
            "alpha": 1.0,
        },
    }


def _register_merged_group(
    service: SceneService,
    entries: list[dict],
    *,
    group_name: str = "scene:merged_test::mesh",
) -> str:
    """Install a coherent desired aggregate in a SceneService test fixture."""
    meshes = [entry["mesh"] for entry in entries]
    resolved = service._scene_entry_base_material(entries[0])
    material = resolved.payload
    member_ids = [id(mesh) for mesh in meshes]
    geometry = service._merge_scene_mesh_payloads(meshes)
    assert geometry is not None
    sources = tuple((mesh.id, mesh.payload) for mesh in meshes)
    batch = SceneBatch(
        name=group_name,
        material_signature=("concrete", material),
        member_mesh_ids=member_ids,
        baseline_geometry=geometry,
        baseline_sources=sources,
        presentation={
            "geometry": geometry,
            "geometry_sources": sources,
            "render_state": RenderObjectState(
                id=group_name,
                payload=geometry,
                material=material,
                metadata={"type": "scene_aggregate"},
            ),
        },
    )
    service._scene_batches.add_batch(batch)
    for entry, mesh, mesh_id in zip(entries, meshes, member_ids, strict=True):
        mesh.material = material
        service._scene_batches.register_entry(mesh_id, entry)
    return group_name


def test_scene_entry_render_state_wraps_cached_payload_without_copy() -> None:
    payload = _mesh_payload()
    entry = {
        "name": "Wall",
        "stable_mesh_id": "wall",
        "mesh": payload,
        "visible": False,
    }

    state = ensure_scene_mesh_render_state(entry, 0)

    assert state.id == "scene:wall::mesh"
    assert state.payload is payload
    assert state.visible is False
    assert entry["mesh"] is state
    assert ensure_scene_mesh_render_state(entry, 0) is state


def test_scene_snapshot_batches_mesh_and_outline_without_noop_present(
    dummy_visualizer,
) -> None:
    renderer = RecordingBatchRenderer()
    dummy_visualizer.renderer = renderer
    dummy_visualizer.outlines_enabled = True
    service = SceneService(dummy_visualizer)
    mesh = _scene_state("batched")
    outline = RenderObjectState(
        id="scene:batched::outline",
        payload=_mesh_payload(),
        is_edge=True,
    )
    entry = {
        "name": "Batched",
        "stable_mesh_id": "batched",
        "entry_type": "mesh",
        "mesh": mesh,
        "outline_geometry": outline,
        "visible": True,
    }

    def _sync_outline(_entry, *, visible, rebuild):
        assert visible is True
        assert rebuild is False
        return renderer.ensure_object(outline.to_render_object(effective_visible=visible))

    dummy_visualizer.scene_appearance_service = SimpleNamespace(
        sync_scene_entry_outline_snapshot=_sync_outline,
    )

    assert service.sync_scene_entry_snapshot(entry)

    assert {object_id for object_id, _depth in renderer.operation_depths} == {
        mesh.id,
        outline.id,
    }
    assert all(depth >= 1 for _object_id, depth in renderer.operation_depths)
    assert renderer.outer_batch_count == 1
    assert renderer.update_renderer_calls == 1

    assert service.sync_scene_entry_snapshot(entry)

    assert renderer.outer_batch_count == 2
    assert renderer.update_renderer_calls == 1


class _DummyRenderer:
    capabilities = RendererCapabilities()

    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.reset_called = False
        self.initialized = False
        self.ensured_geometries = []
        self.ensured_objects = []
        self.material_updates = []
        self.object_material_updates = []
        self.visibility_updates = []
        self.removed_objects = []

    @property
    def renderer_type(self):
        return "mock"

    def reset_state(self):
        self.reset_called = True

    def initialize_visualizer(
        self,
        title,
        width,
        height,
        left=-1,
        top=-1,
        suppress_default_camera=False,
        *,
        host_parent=None,
    ):
        assert host_parent is None
        self.visualizer.booted = True
        self.initialized = True
        return _HashableDummy()

    def set_background_color(self, color):
        self.background_color = list(color)

    def add_geometry_to_visualizer(self, geometry, reset_bounding_box=False):
        pass

    def remove_geometry_from_visualizer(self, geometry, reset_bounding_box=False):
        pass

    def update_geometry_in_visualizer(self, geometry):
        pass

    def ensure_named_geometry(self, **kwargs):
        self.ensured_geometries.append(kwargs)
        return True

    def set_named_material(self, name, material):
        self.material_updates.append((name, material))
        return True

    def ensure_object(self, obj):
        self.ensured_objects.append(obj)
        return True

    def set_material(self, name, material):
        self.object_material_updates.append((name, material))
        return True

    def set_visible(self, name, visible):
        self.visibility_updates.append((name, bool(visible)))
        return True

    def remove_object(self, name):
        self.removed_objects.append(name)
        return True

    def clear_scene_geometry(self):
        pass

    @contextmanager
    def batch_updates(self):
        yield


class _DummyVisualizer:
    def __init__(self):
        self.cleaned = False
        self.rendered = False
        self.booted = False
        self.materials_refreshed = False
        self.vis_initialized = False
        self.progress = _DummyProgress()
        self.populate_called = False
        self.load_called = False
        self.ui_controller = _DummyUIController(self)
        self.node_service = SimpleNamespace(update_tx_rx_visibility=lambda: None)
        self.selection_manager = SimpleNamespace(populate_dropdown=lambda: None)
        self.scene_appearance_service = SimpleNamespace(
            ensure_light_gray_background=lambda: None,
        )
        self.mesh_entries = []
        self.target_entries = []
        self.target_meshes = []
        self.vis = None
        self.selected_objects = set()
        self.tx_markers = []
        self.rx_markers = []
        self.tx_labels = []
        self.rx_labels = []
        self.available_tx = []
        self.available_rx = []
        self.tx_rx_data_loaded = False
        self.current_tx_positions = []
        self.current_rx_positions = []
        self.positions_loaded_step = None
        self.building_labels = []
        self.target_labels = []
        self.animation_service = SimpleNamespace(clear_preload_data=lambda **_: None)
        self.cache_service = CacheService(self)
        self.renderer = _DummyRenderer(self)
        self.vis = _DummyVisualizerWindow()
        self.show_building_labels = False
        self.frame_source = None
        self.use_preload_mode = False
        self.last_cam = None
        self._window_layout = None
        self._pending_camera = None

    def cleanup_previous_scene(self):
        self.cleaned = True

    def load_from_xml(self):
        self.load_called = True

    def populate_controls(self):
        self.ui_controller.populate_controls()

    def render_scene(self):
        self.rendered = True

    def detect_mpc_frames(self):
        pass

    def start_preloading(self):
        pass

    def reset_camera_to_overview(self):
        pass

    def geometry(self):
        return SimpleNamespace(left=0, width=1, top=0)

    def set_state(self, **changes):
        self.app_state = SimpleNamespace(**changes)


@pytest.fixture
def dummy_visualizer():
    viz = _DummyVisualizer()
    viz.app_state = None
    return viz


def test_scene_service_announces_texture_cache_build_once(dummy_visualizer, monkeypatch, caplog):
    messages: list[tuple[str, int]] = []
    dummy_visualizer._set_status_message = lambda text, timeout=0: messages.append((text, timeout))
    dummy_visualizer.renderer = SimpleNamespace(
        renderer_type="pygfx",
        capabilities=RendererCapabilities(mesh_buffer_cache=True),
    )
    monkeypatch.setattr(scene_io, "get_uv_cache_root", lambda: Path("/tmp/uvcache"))
    monkeypatch.setattr(
        pygfx_scene_helpers,
        "get_pygfx_mesh_buffer_cache_root",
        lambda: Path("/tmp/meshcache"),
    )

    service = SceneService(dummy_visualizer)
    service.scene_path = "/tmp/scenarios/osm/paris_west/scene.xml"

    orchav_logger = logging.getLogger("orchav")
    original_propagate = orchav_logger.propagate
    try:
        orchav_logger.propagate = True
        with caplog.at_level("INFO", logger="orchav.scene"):
            service._maybe_announce_texture_cache_build(reason="UV")
            service._maybe_announce_texture_cache_build(reason="pygfx mesh")
    finally:
        orchav_logger.propagate = original_propagate

    assert messages == [
        (
            "Building scene texture caches. First textured launch can take longer; "
            "later launches will be faster.",
            12000,
        )
    ]
    assert "Trigger: cold UV cache" in caplog.text
    assert str(Path("/tmp/uvcache")) in caplog.text
    assert str(Path("/tmp/meshcache")) in caplog.text


def test_texture_discovery_ignores_generic_material_identity(dummy_visualizer):
    service = SceneService(dummy_visualizer)

    assert (
        service._discover_scene_texture_defaults(
            {"material_type": "wood", "material_id": "mat-itu_wood"},
            auto_generate_uvs=True,
            texture_cache={"wood": "/tmp/wood.png"},
        )
        == {}
    )
    assert (
        service._discover_scene_texture_defaults(
            {"material_type": "wood", "material_id": "wood"},
            auto_generate_uvs=True,
            texture_cache={"wood": "/tmp/wood.png"},
        )
        == {}
    )


def test_texture_discovery_returns_only_authored_material_id_default(
    dummy_visualizer,
    tmp_path,
):
    service = SceneService(dummy_visualizer)
    texture_path = tmp_path / "custom_wood.png"
    texture_path.write_bytes(b"texture")

    defaults = service._discover_scene_texture_defaults(
        {"material_type": "wood", "material_id": "mat-custom_wood"},
        auto_generate_uvs=True,
        texture_cache={"custom_wood": str(texture_path)},
    )

    assert defaults == {"texture_path": str(texture_path)}


def test_prepare_scene_member_resolves_material_once_for_either_owner_path(
    dummy_visualizer,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    texture_path = tmp_path / "brick.png"
    texture_path.write_bytes(b"texture")
    resolved = resolve_pbr_material(
        [0.8, 0.8, 0.8],
        {"texture_path": str(texture_path), "roughness": 0.7},
    )
    resolve_entry = Mock(return_value=resolved)
    dummy_visualizer.material_pbr_service = SimpleNamespace(
        resolve_entry_material=resolve_entry,
    )
    service = SceneService(dummy_visualizer)
    state = _scene_state("wall")
    entry = _material_scene_entry("wall")
    entry["mesh"] = state
    entry["material_id"] = "mat-brick"
    service._ensure_scene_member_uvs = Mock()
    service._wrap_pygfx_scene_mesh_payload = Mock(side_effect=lambda state, **_kwargs: state)

    members = service._prepare_scene_members(
        [(state, entry)],
        has_pbr=True,
        auto_generate_uvs=True,
        texture_cache={"brick": str(texture_path)},
    )

    assert len(members) == 1
    assert members[0].resolved_material is resolved
    assert members[0].material is resolved.payload
    resolve_entry.assert_called_once_with(
        entry,
        discovered_texture_defaults={"texture_path": str(texture_path)},
    )
    service._ensure_scene_member_uvs.assert_called_once_with(state, entry, resolved)


def test_scene_material_resolution_preserves_entry_alpha_until_global_override(
    dummy_visualizer,
):
    dummy_visualizer.material_pbr_service = MaterialPBRService(dummy_visualizer)
    service = SceneService(dummy_visualizer)
    entry = {
        "material_type": "glass",
        "pbr_properties": {
            "color": [0.8, 0.9, 1.0],
            "roughness": 0.05,
            "metallic": 0.0,
            "reflectance": 0.9,
            "alpha": 0.25,
        },
    }

    assert service._scene_entry_base_material(entry).payload.base_color[3] == pytest.approx(0.25)

    dummy_visualizer.current_building_alpha = 0.55
    assert service._scene_entry_base_material(entry).payload.base_color[3] == pytest.approx(0.55)


def test_scene_service_loads_scene(tmp_path, dummy_visualizer):
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene></scene>")

    service = SceneService(dummy_visualizer)

    service.load_scene(str(xml_path))

    assert service.scene_path == str(xml_path)
    assert dummy_visualizer.load_called is False
    assert dummy_visualizer.populate_called
    assert dummy_visualizer.booted
    assert dummy_visualizer.vis_initialized
    assert dummy_visualizer.materials_refreshed


def test_scene_service_configures_compass_from_metadata_without_street_graph_and_clears_it(
    tmp_path, dummy_visualizer
):
    geo = tmp_path / "geographic"
    geo.mkdir()
    xml_path = geo / "scene.xml"
    xml_path.write_text("<scene></scene>")
    (geo / "osm_metadata.json").write_text(
        json.dumps(
            {
                "utm_crs": "EPSG:32631",
                "origin_utm": [451052.283, 5413413.766, 0],
            }
        )
    )
    dummy_visualizer.renderer.configure_compass = Mock(return_value=True)
    sync_controls = Mock()
    dummy_visualizer.ui_manager = SimpleNamespace(
        panels={"render": SimpleNamespace(sync_scene_guides=sync_controls)}
    )
    service = SceneService(dummy_visualizer)
    service.load_scene(str(xml_path), render_immediately=False)
    assert not service.street_names.available  # No graph was created.
    configure = dummy_visualizer.renderer.configure_compass
    assert configure.call_args.args[0] == pytest.approx((0.00877536, 0.99996150))
    assert configure.call_args.kwargs["reason"] == ""
    assert sync_controls.call_count == 2  # Cleanup and publication both update the UI.
    service.cleanup_previous_scene()
    configure.assert_called_with(None)
    assert sync_controls.call_count == 3
    local = tmp_path / "cartesian.xml"
    local.write_text("<scene></scene>")
    service.load_scene(str(local), render_immediately=False)
    assert configure.call_args.args[0] is None
    assert "no geographic" in configure.call_args.kwargs["reason"]
    assert sync_controls.call_count == 5


def test_scene_service_can_skip_cleanup_when_scenario_workflow_already_cleaned(
    tmp_path, dummy_visualizer
):
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene></scene>")
    service = SceneService(dummy_visualizer)
    service.cleanup_previous_scene = Mock()
    service.load_from_xml = Mock()

    service.load_scene(
        str(xml_path),
        render_immediately=False,
        cleanup_first=False,
    )

    service.cleanup_previous_scene.assert_not_called()
    service.load_from_xml.assert_called_once_with()


def test_scene_service_installs_preflighted_payload_without_rebuilding(
    tmp_path, dummy_visualizer, monkeypatch
):
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene></scene>")
    xml_root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    monkeypatch.setattr(
        scene_service,
        "build_scene_from_root",
        lambda *_args: pytest.fail("prepared payload must not be rebuilt"),
    )
    service = SceneService(dummy_visualizer)

    service.load_prepared_scene(
        str(xml_path),
        xml_root,
        [],
        render_immediately=False,
        cleanup_first=False,
    )

    assert service.scene_path == str(xml_path)
    assert dummy_visualizer.xml_root is xml_root
    assert dummy_visualizer.mesh_entries == []
    assert dummy_visualizer.populate_called is True


def test_scene_service_parse_failure_does_not_poison_retry(tmp_path, dummy_visualizer, monkeypatch):
    xml_path = tmp_path / "scene_invalid.xml"
    xml_path.write_text("<scene></scene>")

    original_load = scene_service.XMLSceneHandler.load_xml_scene
    attempts = 0

    def _fail_once(path: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("parse-fail")
        return original_load(path)

    monkeypatch.setattr(
        scene_service.XMLSceneHandler,
        "load_xml_scene",
        staticmethod(_fail_once),
    )

    service = SceneService(dummy_visualizer)
    with pytest.raises(ValueError, match="parse-fail"):
        service.load_scene(str(xml_path))

    assert service.scene_path is None
    assert dummy_visualizer.xml_path is None
    assert dummy_visualizer.xml_root is None
    assert dummy_visualizer.mesh_entries == []
    assert dummy_visualizer.populate_called is False
    assert dummy_visualizer.booted is False

    service.load_scene(str(xml_path))

    assert attempts == 2
    assert service.scene_path == str(xml_path)
    assert dummy_visualizer.populate_called is True
    assert dummy_visualizer.booted is True


def test_scene_service_geometry_failure_is_not_committed(tmp_path, dummy_visualizer, monkeypatch):
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene></scene>")

    def _fail_build(_root, _path):
        raise RuntimeError("mesh-build-fail")

    monkeypatch.setattr(scene_service, "build_scene_from_root", _fail_build)
    service = SceneService(dummy_visualizer)

    with pytest.raises(RuntimeError, match="mesh-build-fail"):
        service.load_scene(str(xml_path))

    assert service.scene_path is None
    assert dummy_visualizer.xml_path is None
    assert dummy_visualizer.xml_root is None
    assert dummy_visualizer.mesh_entries == []
    assert dummy_visualizer.populate_called is False
    assert dummy_visualizer.booted is False


class _DummyVisualizerWindow:
    def __init__(self):
        self.scene = SimpleNamespace(set_background=lambda *args, **kwargs: None)

    def add_geometry(self, *args, **kwargs):
        return None

    def remove_geometry(self, *args, **kwargs):
        return None


# ======================================================================
# Mesh-merging tests
# ======================================================================


class TestShouldMergeMeshes:
    """Tests for _should_merge_meshes decision logic."""

    def test_default_does_not_merge_below_threshold(self, dummy_visualizer):
        """Small scenes stay individual unless the scenario opts into merging."""
        dummy_visualizer.renderer.capabilities = RendererCapabilities(
            static_mesh_batching=True,
            static_mesh_batch_object_threshold=4,
        )
        service = SceneService(dummy_visualizer)
        assert not service._should_merge_meshes(
            [_material_scene_entry(f"mesh-{index}") for index in range(4)]
        )

    def test_capable_renderer_auto_merges_only_for_meaningful_reduction(
        self,
        dummy_visualizer,
    ):
        dummy_visualizer.renderer.capabilities = RendererCapabilities(
            static_mesh_batching=True,
            static_mesh_batch_object_threshold=4,
        )
        service = SceneService(dummy_visualizer)
        compatible = [_material_scene_entry(f"mesh-{index}") for index in range(5)]
        distinct = [_material_scene_entry(f"mesh-{index}") for index in range(5)]
        for index, entry in enumerate(distinct):
            entry["material_type"] = f"material-{index}"

        assert service._should_merge_meshes(compatible)
        assert not service._should_merge_meshes(distinct)

    def test_explicit_true_forces_merge(self, dummy_visualizer):
        """merge_scene_meshes: true in YAML overrides auto threshold."""
        dummy_visualizer.scenario = SimpleNamespace(
            view_defaults={"merge_scene_meshes": True},
        )
        service = SceneService(dummy_visualizer)
        # Even below threshold, explicit True forces merging
        assert service._should_merge_meshes([_material_scene_entry("mesh")])

    def test_explicit_false_disables_merge(self, dummy_visualizer):
        """merge_scene_meshes: false in YAML disables merging."""
        dummy_visualizer.renderer.capabilities = RendererCapabilities(
            static_mesh_batching=True,
            static_mesh_batch_object_threshold=2,
        )
        dummy_visualizer.scenario = SimpleNamespace(
            view_defaults={"merge_scene_meshes": False},
        )
        service = SceneService(dummy_visualizer)
        # Even above threshold, explicit False disables
        assert not service._should_merge_meshes(
            [_material_scene_entry(f"mesh-{index}") for index in range(10)]
        )


class TestMaterialBatchSignature:
    """Tests for complete typed material and membership signatures."""

    def test_batch_name_is_deterministic_and_collision_safe_for_membership(self):
        signature = ("concrete", MaterialPayload(roughness=0.5))

        first = SceneService._scene_batch_name_for_owner_ids(signature, ["scene:a", "scene:b"])
        repeated = SceneService._scene_batch_name_for_owner_ids(signature, ["scene:a", "scene:b"])
        different_members = SceneService._scene_batch_name_for_owner_ids(
            signature, ["scene:a", "scene:c"]
        )

        assert first == repeated
        assert first != different_members

    def test_member_chunks_respect_renderer_member_and_triangle_limits(
        self,
        dummy_visualizer,
    ):
        dummy_visualizer.renderer.capabilities = RendererCapabilities(
            static_mesh_batching=True,
            static_mesh_batch_member_limit=2,
            static_mesh_batch_triangle_limit=2,
        )
        service = SceneService(dummy_visualizer)
        resolved = resolve_pbr_material([0.5, 0.5, 0.5], {"roughness": 0.5})
        members = [
            PreparedSceneMember(
                state=_scene_state(f"mesh-{index}"),
                entry={"name": f"mesh-{index}"},
                visible=True,
                material_type="concrete",
                resolved_material=resolved,
            )
            for index in range(5)
        ]

        chunks = service._batch_member_chunks(members)

        assert [len(chunk) for chunk in chunks] == [2, 2, 1]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("clearcoat", 0.7),
            ("clearcoat_roughness", 0.3),
            ("anisotropy", 0.6),
            ("emissive_color", [0.2, 0.4, 0.8]),
            ("emissive_intensity", 2.5),
            ("transmission", 0.8),
            ("glass_thickness", 0.9),
            ("absorption_color", [0.7, 0.8, 0.9]),
            ("normal_map_strength", 1.8),
            ("uv_scale_meters", 4.0),
            ("uv_repeat_scale", [2.0, 3.0]),
            ("shader_variant", "defaultLitSSR"),
        ],
    )
    def test_advanced_material_field_changes_batch_signature(self, field, value):
        base = {
            "color": [0.5, 0.5, 0.5],
            "roughness": 0.5,
            "metallic": 0.0,
            "reflectance": 0.5,
            "alpha": 1.0,
        }
        changed = dict(base)
        changed[field] = value

        first = resolve_pbr_material(base["color"], base).payload
        second = resolve_pbr_material(changed["color"], changed).payload
        assert ("concrete", first) != ("concrete", second)


class TestMergeTracking:
    """Tests for merge-group bookkeeping."""

    def test_initial_state_empty(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        assert service._merged_groups == {}
        assert service._mesh_id_to_group == {}
        assert service._merged_meshes == {}
        assert not service._merge_enabled


def test_visual_surface_mesh_cache_keys_distinguish_partition_and_mapping(
    dummy_visualizer, tmp_path
):
    """Same-file fragments must never reuse each other's expanded geometry/UVs."""
    source = tmp_path / "wall.ply"
    source.write_bytes(b"source mesh revision")
    service = SceneService(dummy_visualizer)
    entry = {"rel_path": str(source), "transform_state": {"scale": 1.0}}
    props = {"uv_scale_meters": 2.0}
    legacy_key = service._pygfx_mesh_buffer_cache_key(entry, effective_props=props)
    assert legacy_key is not None
    entry["_visual_surface_key"] = None
    assert service._pygfx_mesh_buffer_cache_key(entry, effective_props=props) == legacy_key

    keys = []
    for identity in ("mapping-v1/front", "mapping-v1/complement", "mapping-v2/front"):
        entry["_visual_surface_key"] = identity
        key = service._pygfx_mesh_buffer_cache_key(entry, effective_props=props)
        assert key == service._pygfx_mesh_buffer_cache_key(entry, effective_props=props)
        keys.append(key)
    assert len(set([legacy_key, *keys])) == 4


def test_scene_batch_registry_rejects_membership_collision_atomically() -> None:
    registry = SceneBatchRegistry()
    material = MaterialPayload()
    registry.add_batch(
        SceneBatch(
            name="scene:batch-a::mesh",
            material_signature=("default", material),
            member_mesh_ids=[1],
        )
    )

    with pytest.raises(ValueError, match="already belongs"):
        registry.add_batch(
            SceneBatch(
                name="scene:batch-b::mesh",
                material_signature=("default", material),
                member_mesh_ids=[1, 2],
            )
        )

    assert set(registry.batches) == {"scene:batch-a::mesh"}
    assert registry.mesh_to_batch == {1: "scene:batch-a::mesh"}


def test_render_individual_has_one_canonical_render_state_handoff(dummy_visualizer):
    service = SceneService(dummy_visualizer)
    state = RenderObjectState(
        id="scene:wall_a::mesh",
        payload=_mesh_payload(),
    )
    entry = {
        "name": "Wall A",
        "stable_mesh_id": "wall_a",
        "mesh": state,
        "visible": True,
    }
    dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
    dummy_visualizer.renderer.ensure_named_geometry = Mock(return_value=True)
    resolved = resolve_pbr_material(
        [0.4, 0.5, 0.6],
        {
            "roughness": 0.7,
            "metallic": 0.0,
            "reflectance": 0.4,
            "alpha": 1.0,
        },
    )

    service._render_individual(
        [
            PreparedSceneMember(
                state=state,
                entry=entry,
                visible=True,
                material_type="concrete",
                resolved_material=resolved,
            )
        ]
    )

    dummy_visualizer.renderer.ensure_object.assert_called_once()
    assert dummy_visualizer.renderer.ensure_object.call_args.args[0].id == state.id
    dummy_visualizer.renderer.ensure_named_geometry.assert_not_called()


def test_initial_scene_visibility_never_resurrects_manually_hidden_entry(
    dummy_visualizer,
):
    service = SceneService(dummy_visualizer)
    state = _scene_state("hidden")
    entry = {"mesh": state, "entry_type": "mesh", "visible": False}

    assert not service._effective_initial_scene_visibility(state, entry)


class TestMergedHighlight:
    def test_resolved_highlight_refreshes_payload_for_non_merged_mesh(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        mesh = _scene_state("wall_b")
        entry = {
            "mesh": mesh,
            "name": "Wall B",
            "stable_mesh_id": "wall_b",
            "color": [0.6, 0.6, 0.6],
            "pbr_properties": {"texture_path": "/tmp/wall_albedo.png"},
        }

        dummy_visualizer.renderer.capabilities = RendererCapabilities(pbr=True)
        dummy_visualizer.renderer.set_material = Mock(return_value=True)
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.set_named_material = Mock(return_value=True)

        entry["highlighted"] = True
        resolved = service._resolved_scene_appearance(entry)
        assert service.sync_scene_resolved_appearance(entry, resolved) is True
        assert entry["highlighted"] is True
        assert mesh.material.base_color == (0.6, 0.6, 0.6, 1.0)
        snapshot = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert snapshot.material_payload.base_color == (1.0, 0.0, 0.0, 1.0)
        assert snapshot.material_payload.texture_path is None
        dummy_visualizer.renderer.set_material.assert_not_called()
        dummy_visualizer.renderer.set_named_material.assert_not_called()

    def test_resolved_highlight_uses_common_material_cache_for_render_state(self, dummy_visualizer):
        state = RenderObjectState(
            id="scene:wall_b::mesh",
            payload=_mesh_payload(),
        )
        entry = {
            "mesh": state,
            "name": "Wall B",
            "geometry_name": state.id,
            "visible": True,
            "color": [0.6, 0.6, 0.6],
        }
        service = SceneService(dummy_visualizer)
        dummy_visualizer.renderer.capabilities = RendererCapabilities(pbr=True)
        dummy_visualizer.renderer.set_material = Mock(return_value=True)
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.set_named_material = Mock(return_value=True)

        entry["highlighted"] = True
        resolved = service._resolved_scene_appearance(entry)
        assert service.sync_scene_resolved_appearance(entry, resolved) is True

        assert state.material.base_color == (0.6, 0.6, 0.6, 1.0)
        snapshot = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert snapshot.material_payload.base_color == (1.0, 0.0, 0.0, 1.0)
        dummy_visualizer.renderer.set_material.assert_not_called()
        dummy_visualizer.renderer.set_named_material.assert_not_called()

    def test_rebuild_merged_group_skips_highlighted_members(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        mesh_a = _scene_state("wall_a")
        mesh_b = _scene_state("wall_b", offset=2.0)

        dummy_visualizer.mesh_entries = [
            {
                "mesh": mesh_a,
                "visible": True,
                "highlighted": True,
                "stable_mesh_id": "wall_a",
            },
            {"mesh": mesh_b, "visible": True, "stable_mesh_id": "wall_b"},
        ]
        group_name = _register_merged_group(service, dummy_visualizer.mesh_entries)

        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.set_visible = Mock(return_value=True)

        service.rebuild_merged_group_from_state(group_name)

        aggregate = next(
            call.args[0]
            for call in dummy_visualizer.renderer.ensure_object.call_args_list
            if call.args[0].id == group_name
        )
        assert isinstance(aggregate.payload, MeshPayload)
        np.testing.assert_allclose(aggregate.payload.vertices, mesh_b.payload.vertices)
        assert "scene:wall_a::mesh" in service._individual_scene_owner_ids
        dummy_visualizer.renderer.set_visible.assert_not_called()

    def test_material_wide_highlight_keeps_one_merged_owner(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = [
            _material_scene_entry("wall_a"),
            _material_scene_entry("wall_b", offset=2.0),
        ]
        dummy_visualizer.mesh_entries = entries
        group_name = _register_merged_group(service, entries)
        original_payload = service._merged_meshes[group_name]["render_state"].payload
        for entry in entries:
            entry["highlighted"] = True

        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)

        assert service.rebuild_merged_group_from_state(group_name)

        dummy_visualizer.renderer.ensure_object.assert_called_once()
        aggregate = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert aggregate.id == group_name
        assert aggregate.payload is original_payload
        assert aggregate.material_payload.base_color == (1.0, 0.0, 0.0, 1.0)
        assert aggregate.material_payload.color_multiplier == (1.0, 1.0, 1.0)
        assert not service._individual_scene_owner_ids
        dummy_visualizer.renderer.remove_object.assert_not_called()

    def test_material_wide_highlight_then_hide_updates_only_aggregate(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = [
            _material_scene_entry("wall_a"),
            _material_scene_entry("wall_b", offset=2.0),
        ]
        dummy_visualizer.mesh_entries = entries
        group_name = _register_merged_group(service, entries)
        for entry in entries:
            entry["highlighted"] = True

        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)
        assert service.rebuild_merged_group_from_state(group_name)

        dummy_visualizer.renderer.ensure_object.reset_mock()
        for entry in entries:
            entry["visible"] = False

        assert service.rebuild_merged_group_from_state(group_name)

        dummy_visualizer.renderer.ensure_object.assert_not_called()
        dummy_visualizer.renderer.remove_object.assert_called_once_with(group_name)

    def test_material_wide_highlight_scales_with_merged_owners(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = [
            _material_scene_entry(f"wall_{index}", offset=float(index)) for index in range(250)
        ]
        dummy_visualizer.mesh_entries = entries
        group_name = _register_merged_group(service, entries)
        original_payload = service._merged_meshes[group_name]["render_state"].payload
        for entry in entries:
            entry["highlighted"] = True

        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)

        assert service.rebuild_merged_group_from_state(group_name)

        dummy_visualizer.renderer.ensure_object.assert_called_once()
        aggregate = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert aggregate.id == group_name
        assert aggregate.payload is original_payload
        assert not service._individual_scene_owner_ids
        dummy_visualizer.renderer.remove_object.assert_not_called()

    def test_visibility_batch_scales_with_unique_render_owners(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = []
        group_names = [f"scene:merged_{index}::mesh" for index in range(5)]
        for index in range(250):
            state = _scene_state(f"wall_{index}", offset=float(index))
            entry = {
                "name": f"Wall {index}",
                "entry_type": "mesh",
                "mesh": state,
                "visible": index % 2 == 0,
            }
            entries.append((entry, bool(entry["visible"])))
            service._mesh_id_to_group[id(state)] = group_names[index % len(group_names)]

        service.rebuild_merged_group_from_state = Mock(return_value=True)
        service._sync_individual_scene_snapshot = Mock(return_value=True)

        assert service.sync_scene_entry_visibility_batch(entries)

        assert service.rebuild_merged_group_from_state.call_count == len(group_names)
        assert {
            call.args[0] for call in service.rebuild_merged_group_from_state.call_args_list
        } == set(group_names)
        service._sync_individual_scene_snapshot.assert_not_called()

    def test_appearance_batch_rebuilds_each_merged_owner_once(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        group_names = [f"scene:merged_{index}::mesh" for index in range(5)]
        resolved_entries = []
        for index in range(250):
            state = _scene_state(f"wall_{index}", offset=float(index))
            entry = {
                "name": f"Wall {index}",
                "entry_type": "mesh",
                "mesh": state,
                "visible": True,
            }
            service._mesh_id_to_group[id(state)] = group_names[index % len(group_names)]
            resolved_entries.append(
                (
                    entry,
                    SimpleNamespace(
                        visible=index % 2 == 0,
                        highlighted=index % 3 == 0,
                        material=MaterialPayload(),
                    ),
                )
            )

        service.rebuild_merged_group_from_state = Mock(return_value=True)
        service._refresh_scene_material_entries = Mock(return_value=True)

        assert service.sync_scene_resolved_appearance_batch(
            resolved_entries,
            materials_changed=False,
        )

        assert service.rebuild_merged_group_from_state.call_count == len(group_names)
        assert {
            call.args[0] for call in service.rebuild_merged_group_from_state.call_args_list
        } == set(group_names)
        service._refresh_scene_material_entries.assert_not_called()

    def test_pbr_batch_updates_large_merged_group_once(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = [
            _material_scene_entry(f"wall_{index}", offset=float(index)) for index in range(250)
        ]
        dummy_visualizer.mesh_entries = entries
        group_name = _register_merged_group(service, entries)
        original_payload = service._merged_meshes[group_name]["render_state"].payload

        resolved_entries = []
        for entry in entries:
            entry["pbr_properties"]["roughness"] = 0.91
            material = service._scene_entry_base_material(entry).payload
            entry["mesh"].material = material
            resolved_entries.append(
                (
                    entry,
                    SimpleNamespace(
                        visible=True,
                        highlighted=False,
                        material=material,
                    ),
                )
            )

        dummy_visualizer.renderer.set_material = Mock(return_value=True)
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        service._refresh_scene_material_entries = Mock(return_value=True)

        assert service.sync_scene_resolved_appearance_batch(
            resolved_entries,
            materials_changed=True,
        )

        dummy_visualizer.renderer.ensure_object.assert_called_once()
        aggregate = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert aggregate.id == group_name
        assert aggregate.material_payload.roughness == pytest.approx(0.91)
        dummy_visualizer.renderer.set_material.assert_not_called()
        service._refresh_scene_material_entries.assert_not_called()
        assert service._merged_meshes[group_name]["render_state"].payload is original_payload
        assert not service._individual_scene_owner_ids

    def test_pbr_batch_generates_uvs_before_individual_owner_publish(
        self,
        dummy_visualizer,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
        monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
        texture_path = tmp_path / "brick.png"
        texture_path.write_bytes(b"texture")
        resolved_material = resolve_pbr_material(
            [0.5, 0.5, 0.5],
            {"texture_path": str(texture_path), "uv_scale_meters": 4.0},
        )
        generated_uvs = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        monkeypatch.setattr(
            scene_io,
            "load_or_generate_box_projection_uvs",
            Mock(return_value=generated_uvs),
        )
        finalize_uv_cache = Mock()
        monkeypatch.setattr(scene_service, "finalize_uv_cache_stores", finalize_uv_cache)

        service = SceneService(dummy_visualizer)
        entry = _material_scene_entry("textured_wall")
        entry["mesh"].material = resolved_material.payload
        service._refresh_scene_material_entries = Mock(return_value=True)
        appearance = SimpleNamespace(
            visible=True,
            highlighted=False,
            material=resolved_material.payload,
        )

        assert service.sync_scene_resolved_appearance_batch(
            [(entry, appearance)],
            materials_changed=True,
        )

        assert np.array_equal(entry["mesh"].payload.triangle_uvs, generated_uvs)
        published = dummy_visualizer.renderer.ensured_objects[-1]
        assert published.id == entry["mesh"].id
        assert published.payload is entry["mesh"].payload
        finalize_uv_cache.assert_called_once_with()

    def test_pbr_batch_generates_uvs_before_merged_owner_rebuild(
        self,
        dummy_visualizer,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
        monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
        entries = [
            _material_scene_entry("wall_a"),
            _material_scene_entry("wall_b", offset=2.0),
        ]
        dummy_visualizer.mesh_entries = entries
        service = SceneService(dummy_visualizer)
        group_name = _register_merged_group(service, entries)

        texture_path = tmp_path / "brick.png"
        texture_path.write_bytes(b"texture")
        resolved_material = resolve_pbr_material(
            [0.5, 0.5, 0.5],
            {"texture_path": str(texture_path)},
        )
        generated_uvs = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        generate_uvs = Mock(return_value=generated_uvs)
        monkeypatch.setattr(
            scene_io,
            "load_or_generate_box_projection_uvs",
            generate_uvs,
        )
        finalize_uv_cache = Mock()
        monkeypatch.setattr(scene_service, "finalize_uv_cache_stores", finalize_uv_cache)

        resolved_entries = []
        for entry in entries:
            entry["mesh"].material = resolved_material.payload
            resolved_entries.append(
                (
                    entry,
                    SimpleNamespace(
                        visible=True,
                        highlighted=False,
                        material=resolved_material.payload,
                    ),
                )
            )
        service._refresh_scene_material_entries = Mock(return_value=True)

        assert service.sync_scene_resolved_appearance_batch(
            resolved_entries,
            materials_changed=True,
        )

        assert generate_uvs.call_count == 2
        published = dummy_visualizer.renderer.ensured_objects[-1]
        assert published.id == group_name
        assert published.payload.triangle_uvs is not None
        assert published.payload.triangle_uvs.shape == (6, 2)
        assert not service._individual_scene_owner_ids
        service._refresh_scene_material_entries.assert_not_called()
        finalize_uv_cache.assert_called_once_with()

    def test_pbr_batch_keeps_full_highlight_on_merged_owner(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = [
            _material_scene_entry("wall_a"),
            _material_scene_entry("wall_b", offset=2.0),
        ]
        dummy_visualizer.mesh_entries = entries
        _register_merged_group(service, entries)

        resolved_entries = []
        for entry in entries:
            entry["pbr_properties"]["metallic"] = 0.72
            material = service._scene_entry_base_material(entry).payload
            entry["mesh"].material = material
            highlighted = replace(material, base_color=(1.0, 0.0, 0.0, 1.0))
            resolved_entries.append(
                (
                    entry,
                    SimpleNamespace(
                        visible=True,
                        highlighted=True,
                        material=highlighted,
                    ),
                )
            )

        dummy_visualizer.renderer.set_material = Mock(return_value=True)
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        service._refresh_scene_material_entries = Mock(return_value=True)

        assert service.sync_scene_resolved_appearance_batch(
            resolved_entries,
            materials_changed=True,
        )

        dummy_visualizer.renderer.ensure_object.assert_called_once()
        aggregate = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        displayed = aggregate.material_payload
        assert displayed.base_color == (1.0, 0.0, 0.0, 1.0)
        assert displayed.metallic == pytest.approx(0.72)
        dummy_visualizer.renderer.set_material.assert_not_called()
        service._refresh_scene_material_entries.assert_not_called()
        assert not service._individual_scene_owner_ids

    def test_partial_pbr_batch_uses_regrouping_path(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = [
            _material_scene_entry("wall_a"),
            _material_scene_entry("wall_b", offset=2.0),
        ]
        dummy_visualizer.mesh_entries = entries
        _register_merged_group(service, entries)
        changed = entries[0]
        changed["pbr_properties"]["roughness"] = 0.91
        material = service._scene_entry_base_material(changed).payload
        changed["mesh"].material = material
        resolved_entries = [
            (
                changed,
                SimpleNamespace(
                    visible=True,
                    highlighted=False,
                    material=material,
                ),
            )
        ]
        service._refresh_scene_material_entries = Mock(return_value=True)

        assert service.sync_scene_resolved_appearance_batch(
            resolved_entries,
            materials_changed=True,
        )

        service._refresh_scene_material_entries.assert_called_once_with([changed])

    def test_visibility_batch_syncs_each_unmerged_owner_once(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = [
            {
                "name": "Wall A",
                "entry_type": "mesh",
                "mesh": _scene_state("wall_a"),
                "visible": False,
            },
            {
                "name": "Wall B",
                "entry_type": "mesh",
                "mesh": _scene_state("wall_b"),
                "visible": True,
            },
        ]
        service._sync_individual_scene_snapshot = Mock(return_value=True)
        service.rebuild_merged_group_from_state = Mock(return_value=True)

        assert service.sync_scene_entry_visibility_batch([(entries[0], False), (entries[1], True)])

        assert service._sync_individual_scene_snapshot.call_count == 2
        service.rebuild_merged_group_from_state.assert_not_called()

    def test_material_change_repartitions_member_without_duplicate_geometry(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        changed_mesh = _scene_state("changed")
        neighbor_mesh = _scene_state("neighbor", offset=2.0)
        changed_entry = {
            "mesh": changed_mesh,
            "visible": True,
            "stable_mesh_id": "changed",
            "material_type": "concrete",
        }
        neighbor_entry = {
            "mesh": neighbor_mesh,
            "visible": True,
            "stable_mesh_id": "neighbor",
            "material_type": "concrete",
        }
        neighbor_id = id(neighbor_mesh)

        dummy_visualizer.mesh_entries = [changed_entry, neighbor_entry]
        group_name = _register_merged_group(service, [changed_entry, neighbor_entry])
        changed_id = id(changed_mesh)
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)

        changed_material = MaterialPayload(base_color=(0.1, 0.2, 0.9, 1.0))
        assert service.sync_scene_entry_snapshot(
            changed_entry,
            material=changed_material,
            effective_visible=True,
        )

        assert service._merged_groups[group_name] == [neighbor_id]
        assert changed_id not in service._mesh_id_to_group
        aggregate = next(
            call.args[0]
            for call in dummy_visualizer.renderer.ensure_object.call_args_list
            if call.args[0].id == group_name
        )
        np.testing.assert_allclose(aggregate.payload.vertices, neighbor_mesh.payload.vertices)
        ensured = next(
            call.args[0]
            for call in dummy_visualizer.renderer.ensure_object.call_args_list
            if call.args[0].id == changed_mesh.id
        )
        assert ensured.id == changed_mesh.id
        assert ensured.material_payload == changed_material
        # The neighbor was aggregate-owned, never an individual native object.
        # Ownership tracking avoids a redundant backend removal probe.
        dummy_visualizer.renderer.remove_object.assert_not_called()

    def test_reset_all_repartitions_override_merged_members_by_authored_material(
        self,
        dummy_visualizer,
        tmp_path,
        monkeypatch,
    ):
        """Removing a shared override restores every member's authored material."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        service = SceneService(dummy_visualizer)
        material_service = MaterialPBRService(dummy_visualizer)
        dummy_visualizer.scene_service = service
        dummy_visualizer.material_pbr_service = material_service
        dummy_visualizer.object_appearance_service = SimpleNamespace(
            refresh_entry_appearance_batch=lambda batch: service._refresh_scene_material_entries(
                list(batch)
            )
        )

        smooth = _material_scene_entry("smooth", color=(0.8, 0.2, 0.1))
        smooth["pbr_properties"]["roughness"] = 0.15
        rough = _material_scene_entry("rough", offset=2.0, color=(0.1, 0.3, 0.8))
        rough["pbr_properties"]["roughness"] = 0.9
        entries = [smooth, rough]
        dummy_visualizer.mesh_entries = entries
        material_service.overrides["concrete"] = {
            "color": [0.4, 0.4, 0.4],
            "roughness": 0.5,
            "metallic": 0.0,
            "reflectance": 0.4,
            "alpha": 1.0,
        }

        group_name = _register_merged_group(service, entries)
        service._merge_enabled = True
        override_material = smooth["mesh"].material
        assert rough["mesh"].material == override_material
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.set_material = Mock(return_value=True)
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)

        assert material_service.reset_all()

        smooth_material = service._scene_entry_base_material(smooth).payload
        rough_material = service._scene_entry_base_material(rough).payload
        assert smooth_material != rough_material
        assert smooth["mesh"].material == smooth_material
        assert rough["mesh"].material == rough_material
        assert id(smooth["mesh"]) not in service._mesh_id_to_group
        assert service._mesh_id_to_group[id(rough["mesh"])] == group_name
        assert service._merged_groups[group_name] == [id(rough["mesh"])]
        assert service._merged_meshes[group_name]["render_state"].material == rough_material

    def test_rebuild_highlight_split_preserves_albedo_texture_with_tint(
        self, dummy_visualizer, tmp_path, monkeypatch
    ):
        service = SceneService(dummy_visualizer)
        mesh = _scene_state("wall_a")
        albedo = tmp_path / "wall_albedo.png"
        albedo.write_bytes(b"texture")
        monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
        monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
        entry = {
            "mesh": mesh,
            "visible": True,
            "highlighted": True,
            "stable_mesh_id": "wall_a",
            "pbr_properties": {"texture_path": str(albedo)},
        }
        group_name = "scene:merged_test::mesh"

        dummy_visualizer.mesh_entries = [entry]
        _register_merged_group(service, [entry], group_name=group_name)

        ensured_objects = []
        dummy_visualizer.renderer.ensure_object = lambda obj: ensured_objects.append(obj) or True

        service.rebuild_merged_group_from_state(group_name)

        assert len(ensured_objects) == 1
        aggregate = ensured_objects[0]
        assert aggregate.id == group_name
        assert aggregate.material_payload.texture_path == str(albedo)
        assert aggregate.material_payload.color_multiplier == (1.0, 0.3, 0.3)

    def test_rebuild_highlight_split_does_not_repeat_visibility_after_ensure(
        self, dummy_visualizer
    ):
        service = SceneService(dummy_visualizer)
        regular_mesh = _scene_state("regular")
        highlighted_mesh = _scene_state("highlighted", offset=2.0)
        regular_entry = {
            "mesh": regular_mesh,
            "visible": True,
            "stable_mesh_id": "regular",
        }
        highlighted_entry = {
            "mesh": highlighted_mesh,
            "visible": True,
            "highlighted": True,
            "stable_mesh_id": "highlighted",
        }
        group_name = "scene:merged_test::mesh"
        dummy_visualizer.mesh_entries = [regular_entry, highlighted_entry]
        _register_merged_group(
            service,
            [regular_entry, highlighted_entry],
            group_name=group_name,
        )
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)

        service.rebuild_merged_group_from_state(group_name)

        assert {
            call.args[0].id for call in dummy_visualizer.renderer.ensure_object.call_args_list
        } == {group_name, "scene:highlighted::mesh"}

    def test_initial_group_ensure_failure_retains_owner_and_retry_converges(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        entries = [
            _material_scene_entry("wall_a"),
            _material_scene_entry("wall_b", offset=2.0),
        ]
        dummy_visualizer.mesh_entries = entries
        service._merge_enabled = True
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)
        dummy_visualizer.renderer.ensure_object = Mock(side_effect=(False, True))
        material = service._scene_entry_base_material(entries[0]).payload

        assert not service.sync_scene_entry_snapshot(entries[0], material=material)
        group_name = service._mesh_id_to_group[id(entries[0]["mesh"])]
        assert service._mesh_id_to_group[id(entries[1]["mesh"])] == group_name
        assert group_name in service._pending_merged_group_syncs
        first_snapshot = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert first_snapshot.id == group_name

        assert service.sync_scene_entry_snapshot(entries[0], material=material)
        assert group_name not in service._pending_merged_group_syncs
        assert dummy_visualizer.renderer.ensure_object.call_count == 2
        second_snapshot = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert second_snapshot.id == group_name
        assert second_snapshot.payload is first_snapshot.payload

    @pytest.mark.parametrize("failed_cleanup", ["mesh", "outline"])
    def test_old_owner_cleanup_failure_blocks_new_owner_until_identical_retry(
        self, dummy_visualizer, failed_cleanup
    ):
        service = SceneService(dummy_visualizer)
        entry = _material_scene_entry("wall")
        dummy_visualizer.mesh_entries = [entry]
        group_name = _register_merged_group(service, [entry])
        remove_group_results = iter((False, True)) if failed_cleanup == "mesh" else None

        def remove_object(name):
            if name != group_name or remove_group_results is None:
                return True
            return next(remove_group_results)

        remove_outline = (
            Mock(side_effect=(False, True))
            if failed_cleanup == "outline"
            else Mock(return_value=True)
        )
        dummy_visualizer.scene_appearance_service = SimpleNamespace(
            remove_merged_outline_geometry=remove_outline,
        )
        dummy_visualizer.renderer.remove_object = Mock(side_effect=remove_object)
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        changed = MaterialPayload(base_color=(0.1, 0.2, 0.9, 1.0))

        assert not service.sync_scene_entry_snapshot(entry, material=changed)
        assert group_name in service._merged_groups
        assert service._pending_scene_entry_sources[id(entry["mesh"])] == group_name
        dummy_visualizer.renderer.ensure_object.assert_not_called()

        assert service.sync_scene_entry_snapshot(entry, material=changed)
        assert group_name not in service._merged_groups
        assert id(entry["mesh"]) not in service._pending_scene_entry_sources
        dummy_visualizer.renderer.ensure_object.assert_called_once()

    def test_aggregate_material_failure_never_falls_back_to_individual_owner(
        self, dummy_visualizer
    ):
        service = SceneService(dummy_visualizer)
        entries = [
            _material_scene_entry("wall_a"),
            _material_scene_entry("wall_b", offset=2.0),
        ]
        dummy_visualizer.mesh_entries = entries
        group_name = _register_merged_group(service, entries)
        for entry in entries:
            entry["pbr_properties"]["color"] = [0.1, 0.2, 0.9]
        changed = service._scene_entry_base_material(entries[0]).payload
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)
        dummy_visualizer.renderer.set_material = Mock(side_effect=(False, True))
        dummy_visualizer.renderer.ensure_object = Mock(side_effect=(False, True))

        assert not service.sync_scene_entry_snapshot(entries[0], material=changed)
        assert group_name in service._pending_merged_group_syncs
        fallback = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert fallback.id == group_name

        assert service.sync_scene_entry_snapshot(entries[0], material=changed)
        assert group_name not in service._pending_merged_group_syncs
        assert all(service._mesh_id_to_group[id(entry["mesh"])] == group_name for entry in entries)
        assert dummy_visualizer.renderer.ensure_object.call_count == 2
        retry = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert retry.id == group_name
        assert retry.payload is fallback.payload

    def test_aggregate_highlight_syncs_merged_outline_without_mutating_base_material(
        self, dummy_visualizer
    ):
        service = SceneService(dummy_visualizer)
        entry = _material_scene_entry("wall", color=(0.2, 0.3, 0.4))
        dummy_visualizer.mesh_entries = [entry]
        group_name = _register_merged_group(service, [entry])
        entry["highlighted"] = True
        dummy_visualizer.outlines_enabled = True
        dummy_visualizer.renderer.capabilities = RendererCapabilities(pbr=True)
        dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)
        sync_merged_outline = Mock(return_value=True)
        dummy_visualizer.scene_appearance_service = SimpleNamespace(
            sync_merged_outline_geometry=sync_merged_outline,
            sync_scene_entry_outline_snapshot=Mock(return_value=True),
        )

        assert service.rebuild_merged_group_from_state(group_name)

        base_material = service._scene_entry_base_material(entry).payload
        assert entry["mesh"].material == base_material
        aggregate = dummy_visualizer.renderer.ensure_object.call_args.args[0]
        assert aggregate.id == group_name
        assert aggregate.material_payload.base_color == (1.0, 0.0, 0.0, 1.0)
        sync_merged_outline.assert_called_once_with(
            group_name,
            service._merged_meshes[group_name],
            visible=True,
            rebuild=False,
        )

    def test_cleanup_clears_merge_state(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        service._scene_batches.add_batch(
            SceneBatch(
                name="merged_group_0",
                material_signature=("default", MaterialPayload()),
                member_mesh_ids=[1, 2, 3],
            )
        )
        service._merge_enabled = True

        service.cleanup_previous_scene()

        assert service._merged_groups == {}
        assert service._mesh_id_to_group == {}
        assert service._merged_meshes == {}
        assert not service._merge_enabled

    def test_cleanup_resets_beamforming_state(self, dummy_visualizer):
        service = SceneService(dummy_visualizer)
        clear_result_metadata = Mock()
        dummy_visualizer.beamforming_ui_controller = SimpleNamespace(
            clear_result_metadata=clear_result_metadata
        )
        dummy_visualizer.app_state = SimpleNamespace(
            step=7,
            show_beamforming=True,
            beamforming_tx_node="tx_9",
            beamforming_rx_node="rx_7",
            standalone_beamforming_mode="optional",
            standalone_steering_strategy="svd",
        )

        service.cleanup_previous_scene()

        defaults = get_beamforming_state_defaults()
        assert dummy_visualizer.app_state.step == 0
        for key, value in defaults.items():
            assert getattr(dummy_visualizer.app_state, key) == value
        clear_result_metadata.assert_called_once_with()

    def test_cleanup_clears_aperture_tracking_after_renderer_reset(self, dummy_visualizer):
        """A hard renderer reset must also retire cached aperture ownership."""
        events = []
        dummy_visualizer.renderer.reset_state = Mock(
            side_effect=lambda: events.append("renderer_reset")
        )
        clear_apertures = Mock(side_effect=lambda: events.append("aperture_clear") or True)
        dummy_visualizer.aperture_service = SimpleNamespace(clear_all=clear_apertures)
        service = SceneService(dummy_visualizer)

        service.cleanup_previous_scene()

        assert events == ["renderer_reset", "aperture_clear"]
        clear_apertures.assert_called_once_with()

    def test_cleanup_removes_render_states_through_common_object_api(self, dummy_visualizer):
        scene_mesh = RenderObjectState(
            id="scene:wall::mesh",
            payload=_mesh_payload(),
        )
        target_mesh = RenderObjectState(
            id="target:walker::mesh",
            payload=_mesh_payload(offset=2.0),
        )
        cached_target = RenderObjectState(
            id="target:cached::mesh",
            payload=_mesh_payload(offset=4.0),
        )
        native_point = RenderObjectState(
            id="target:walker::scattering_point_0",
            payload=PointCloudPayload(points=np.zeros((1, 3))),
        )
        dummy_visualizer.mesh_entries = [{"name": "Wall", "mesh": scene_mesh}]
        dummy_visualizer.target_entries = [
            {
                "name": "Walker",
                "target_name": "walker",
                "mesh": target_mesh,
                "_native_scattering_points": (native_point,),
            }
        ]
        dummy_visualizer.target_meshes = {
            "walker": target_mesh,
            "cached": cached_target,
        }
        dummy_visualizer.vis_initialized = True
        dummy_visualizer.renderer.remove_object = Mock(return_value=True)
        dummy_visualizer.renderer.remove_named_geometry = Mock(return_value=True)
        service = SceneService(dummy_visualizer)
        service._individual_scene_owner_ids.add(scene_mesh.id)

        service.cleanup_previous_scene()

        removed_ids = {
            call.args[0] for call in dummy_visualizer.renderer.remove_object.call_args_list
        }
        assert removed_ids == {scene_mesh.id, target_mesh.id, cached_target.id, native_point.id}
        dummy_visualizer.renderer.remove_named_geometry.assert_not_called()


def test_render_scene_preserves_declarative_camera_without_overview_reset(
    dummy_visualizer, monkeypatch
):
    calls = {"bounds": 0, "overview": 0}
    dummy_visualizer.renderer.reset_camera_bounds = lambda: calls.__setitem__(
        "bounds", calls["bounds"] + 1
    )
    dummy_visualizer.reset_camera_to_overview = lambda: calls.__setitem__(
        "overview", calls["overview"] + 1
    )
    monkeypatch.setattr(scene_service, "create_building_labels", lambda _viz: None)
    monkeypatch.setattr(scene_service, "create_target_labels", lambda _viz: None)
    monkeypatch.setattr(
        scene_service,
        "QTimer",
        SimpleNamespace(singleShot=lambda *_args, **_kwargs: None),
    )

    service = SceneService(dummy_visualizer)
    service.render_scene()

    assert calls == {"bounds": 1, "overview": 0}
    anchor = dummy_visualizer._empty_scene_anchor
    assert isinstance(anchor, RenderObjectState)
    assert anchor.id == "scene:empty_anchor::sphere"
    assert any(obj.id == anchor.id for obj in dummy_visualizer.renderer.ensured_objects)


def test_empty_scene_anchor_removal_retries_through_common_object_api(dummy_visualizer):
    service = SceneService(dummy_visualizer)
    anchor = RenderObjectState(
        id="scene:empty_anchor::sphere",
        payload=_mesh_payload(),
        metadata={"type": "empty_scene_anchor"},
    )
    dummy_visualizer._empty_scene_anchor = anchor
    dummy_visualizer.renderer.remove_object = Mock(side_effect=(False, True))

    service.remove_empty_scene_anchor()
    assert dummy_visualizer._empty_scene_anchor is anchor

    service.remove_empty_scene_anchor()
    assert dummy_visualizer._empty_scene_anchor is None
    assert [item.args[0] for item in dummy_visualizer.renderer.remove_object.call_args_list] == [
        anchor.id,
        anchor.id,
    ]


def test_repeated_render_retires_existing_aggregate_before_rebuilding_maps(
    dummy_visualizer,
    monkeypatch,
):
    """A second render cannot orphan an aggregate whose grouping ID changes."""
    entries = [
        _material_scene_entry("wall_a"),
        _material_scene_entry("wall_b", offset=2.0),
    ]
    dummy_visualizer.mesh_entries = entries
    service = SceneService(dummy_visualizer)
    old_group = _register_merged_group(service, entries)
    service._merge_enabled = True
    service._should_merge_meshes = Mock(return_value=False)
    dummy_visualizer.renderer.remove_object = Mock(return_value=True)
    dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
    monkeypatch.setattr(scene_service, "create_building_labels", lambda _viz: None)
    monkeypatch.setattr(scene_service, "create_target_labels", lambda _viz: None)
    monkeypatch.setattr(
        scene_service,
        "QTimer",
        SimpleNamespace(singleShot=lambda *_args, **_kwargs: None),
    )

    service.render_scene()

    dummy_visualizer.renderer.remove_object.assert_any_call(old_group)
    assert old_group not in service._merged_meshes
    assert service._merged_groups == {}
    assert service._individual_scene_owner_ids == {entry["mesh"].id for entry in entries}


def test_failed_repeated_render_retirement_preserves_owner_for_retry(dummy_visualizer):
    """A failed removal aborts rebuilding without forgetting the old owner."""
    entry = _material_scene_entry("wall")
    dummy_visualizer.mesh_entries = [entry]
    service = SceneService(dummy_visualizer)
    service._individual_scene_owner_ids.add(entry["mesh"].id)
    dummy_visualizer.renderer.remove_object = Mock(return_value=False)
    dummy_visualizer.renderer.ensure_object = Mock(return_value=True)

    with pytest.raises(RuntimeError, match="previous renderer owners remain"):
        service.render_scene()

    dummy_visualizer.renderer.remove_object.assert_called_once_with(entry["mesh"].id)
    dummy_visualizer.renderer.ensure_object.assert_not_called()
    assert service._individual_scene_owner_ids == {entry["mesh"].id}


def test_initial_scene_geometry_sync_failure_is_not_reported_as_success(dummy_visualizer):
    """A rejected initial owner must fail the scenario-open transaction."""
    entry = _material_scene_entry("wall")
    dummy_visualizer.mesh_entries = [entry]
    dummy_visualizer.renderer.ensure_object = Mock(return_value=False)
    service = SceneService(dummy_visualizer)

    with pytest.raises(RuntimeError, match="Initial scene geometry synchronization failed"):
        service.render_scene()

    assert entry["mesh"].id in service._individual_scene_owner_ids


def test_repeated_empty_render_reuses_anchor_then_nonempty_render_retires_it(
    dummy_visualizer,
    monkeypatch,
):
    """The camera anchor has one stable owner and disappears before real scene data."""
    monkeypatch.setattr(scene_service, "create_building_labels", lambda _viz: None)
    monkeypatch.setattr(scene_service, "create_target_labels", lambda _viz: None)
    monkeypatch.setattr(
        scene_service,
        "QTimer",
        SimpleNamespace(singleShot=lambda *_args, **_kwargs: None),
    )
    service = SceneService(dummy_visualizer)

    service.render_scene()
    anchor = dummy_visualizer._empty_scene_anchor
    service.render_scene()

    assert dummy_visualizer._empty_scene_anchor is anchor
    assert sum(obj.id == anchor.id for obj in dummy_visualizer.renderer.ensured_objects) == 2

    dummy_visualizer.mesh_entries = [_material_scene_entry("wall")]
    service.render_scene()

    assert dummy_visualizer._empty_scene_anchor is None
    assert anchor.id in dummy_visualizer.renderer.removed_objects


def test_render_scene_records_startup_breakdown_for_initial_boot(dummy_visualizer, monkeypatch):
    recorded: dict[str, dict[str, float]] = {}
    dummy_visualizer._scene_boot_start = 1.0
    dummy_visualizer._scene_boot_logged = False
    dummy_visualizer.set_startup_detail_timing = lambda name, timings: recorded.setdefault(
        name, dict(timings)
    )
    monkeypatch.setattr(scene_service, "create_building_labels", lambda _viz: None)
    monkeypatch.setattr(scene_service, "create_target_labels", lambda _viz: None)
    monkeypatch.setattr(
        scene_service,
        "QTimer",
        SimpleNamespace(singleShot=lambda *_args, **_kwargs: None),
    )

    service = SceneService(dummy_visualizer)
    service.render_scene()

    assert "render_initial_scene_breakdown_ms" in recorded
    breakdown = recorded["render_initial_scene_breakdown_ms"]
    assert "upload_scene_geometry_ms" in breakdown
    assert "fit_camera_bounds_ms" in breakdown
    assert "update_node_visibility_ms" in breakdown


def test_render_scene_skips_building_label_creation_when_not_requested(
    dummy_visualizer, monkeypatch
):
    calls: list[str] = []
    dummy_visualizer.mesh_entries = [
        {
            "name": "B1",
            "stable_mesh_id": "b1",
            "mesh": _scene_state("b1"),
            "show_label": False,
        }
    ]
    monkeypatch.setattr(scene_service, "create_building_labels", lambda _viz: calls.append("bldg"))
    monkeypatch.setattr(scene_service, "create_target_labels", lambda _viz: None)
    monkeypatch.setattr(
        scene_service,
        "QTimer",
        SimpleNamespace(singleShot=lambda *_args, **_kwargs: None),
    )

    service = SceneService(dummy_visualizer)
    service.render_scene()

    assert calls == []


def test_create_building_labels_positions_mesh_payload() -> None:
    viz = SimpleNamespace(
        building_labels=[],
        mesh_entries=[
            {
                "name": "Wall",
                "mesh": _mesh_payload(),
                "color": [0.2, 0.3, 0.4],
            }
        ],
    )

    create_building_labels(viz)

    assert len(viz.building_labels) == 1
    label = viz.building_labels[0]
    assert isinstance(label, RenderObjectState)
    assert label.id == "bldg_label_0"
    assert isinstance(label.payload, TextLabelPayload)
    assert label.payload.text == "Wall"
    assert label.material.base_color == (0.2, 0.3, 0.4, 1.0)
    np.testing.assert_allclose(
        label.world_transform.translation,
        [1.0 / 3.0, 1.0 / 3.0, 2.0],
    )


def test_render_scene_builds_building_labels_when_requested(dummy_visualizer, monkeypatch):
    calls: list[str] = []
    dummy_visualizer.mesh_entries = [
        {
            "name": "B1",
            "stable_mesh_id": "b1",
            "mesh": _scene_state("b1"),
            "show_label": True,
        }
    ]
    monkeypatch.setattr(scene_service, "create_building_labels", lambda _viz: calls.append("bldg"))
    monkeypatch.setattr(scene_service, "create_target_labels", lambda _viz: None)
    monkeypatch.setattr(
        scene_service,
        "QTimer",
        SimpleNamespace(singleShot=lambda *_args, **_kwargs: None),
    )

    service = SceneService(dummy_visualizer)
    service.render_scene()

    assert calls == ["bldg"]


def test_render_scene_startup_labels_follow_parent_visibility(dummy_visualizer, monkeypatch):
    building = RenderObjectState(
        id="scene:building_a::mesh",
        payload=_mesh_payload(),
    )
    target = RenderObjectState(
        id="target:target_a::mesh",
        payload=_mesh_payload(offset=2.0),
    )
    building_entry = {
        "name": "Building A",
        "stable_mesh_id": "building_a",
        "mesh": building,
        "entry_type": "mesh",
        "visible": False,
        "show_label": True,
    }
    target_entry = {
        "name": "Target A",
        "target_name": "target_a",
        "mesh": target,
        "entry_type": "target",
        "visible": True,
        "_frame_visible": False,
        "show_label": True,
    }
    dummy_visualizer.mesh_entries = [building_entry]
    dummy_visualizer.target_entries = [target_entry]
    dummy_visualizer.app_state = SimpleNamespace(
        camera_mode="overview",
        show_target_labels=True,
    )
    dummy_visualizer.renderer.ensure_object = Mock(return_value=True)
    dummy_visualizer.target_service = SimpleNamespace(
        sync_target_entry_snapshot=lambda entry: dummy_visualizer.renderer.ensure_object(
            entry["mesh"].to_render_object(
                effective_visible=bool(
                    entry.get("visible", True) and entry.get("_frame_visible", True)
                )
            )
        )
    )

    def _create_building_labels(viz):
        viz.building_labels = [
            make_text_label_state(
                "bldg_label_0",
                "Building A",
                [0.5, 0.5, 0.5],
            )
        ]

    def _create_target_labels(viz):
        viz.target_labels = [
            make_text_label_state(
                make_target_entry_geometry_name(target_entry, "label"),
                "Target A",
                [0.8, 0.2, 0.2],
            )
        ]

    monkeypatch.setattr(scene_service, "create_building_labels", _create_building_labels)
    monkeypatch.setattr(scene_service, "create_target_labels", _create_target_labels)
    monkeypatch.setattr(
        scene_service,
        "QTimer",
        SimpleNamespace(singleShot=lambda *_args, **_kwargs: None),
    )

    SceneService(dummy_visualizer).render_scene()

    labels = {
        call.args[0].id: call.args[0]
        for call in dummy_visualizer.renderer.ensure_object.call_args_list
        if call.args[0].id.endswith("::label") or call.args[0].id.startswith("bldg_label_")
    }
    assert labels["bldg_label_0"].visible is False
    assert labels["target:target_a::label"].visible is False
    target_mesh_snapshots = [
        call.args[0]
        for call in dummy_visualizer.renderer.ensure_object.call_args_list
        if call.args[0].id == target.id
    ]
    assert len(target_mesh_snapshots) == 1
    assert target_mesh_snapshots[0].visible is False
