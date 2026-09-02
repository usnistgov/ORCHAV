"""Tests for scene/assembly.py — renderer-agnostic scene building helpers."""

import numpy as np
import pytest

from visualizer.src.scene.assembly import (
    CameraOrbit,
    build_texture_cache,
    compute_camera_orbit,
    make_sphere_payload,
    mesh_entry_to_payload,
    resolve_texture,
    target_metadata_to_payload,
    view_model_to_mpc_payloads,
)
from visualizer.src.state import MpcVisibility
from visualizer.src.types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
)


def _triangle_mesh_payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=float),
    )


def _box_mesh_payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )


def _write_target_ply(path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "element face 4",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0 0 0 1 255 0 0",
                "2 0 0 0 0 1 0 255 0",
                "0 4 0 0 0 1 0 0 255",
                "0 0 6 0 0 1 255 255 255",
                "3 0 1 2",
                "3 0 1 3",
                "3 0 2 3",
                "3 1 2 3",
            ]
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CameraOrbit
# ---------------------------------------------------------------------------


@pytest.mark.headless
class TestCameraOrbit:
    def test_eye_at_zero_azimuth_zero_elevation(self):
        orbit = CameraOrbit(center=np.zeros(3), distance=10.0, azimuth_deg=0.0, elevation_deg=0.0)
        eye = orbit.eye_position()
        np.testing.assert_allclose(eye, [10.0, 0.0, 0.0], atol=1e-10)

    def test_eye_top_down(self):
        orbit = CameraOrbit(center=np.zeros(3), distance=50.0, azimuth_deg=0.0, elevation_deg=90.0)
        eye = orbit.eye_position()
        assert abs(eye[2] - 50.0) < 1e-8
        assert abs(eye[0]) < 1e-8
        assert abs(eye[1]) < 1e-8

    def test_center_offset(self):
        center = np.array([10.0, 20.0, 5.0])
        orbit = CameraOrbit(center=center, distance=100.0, azimuth_deg=45.0, elevation_deg=30.0)
        eye = orbit.eye_position()
        actual_dist = np.linalg.norm(eye - center)
        np.testing.assert_allclose(actual_dist, 100.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Texture helpers
# ---------------------------------------------------------------------------


@pytest.mark.headless
class TestTextureHelpers:
    def test_resolve_texture_by_type(self):
        cache = {"concrete": "/path/to/concrete.png"}
        assert resolve_texture("concrete", None, cache) == "/path/to/concrete.png"

    def test_resolve_texture_can_disable_type_fallback(self):
        cache = {"wood": "/path/to/wood.png"}
        assert (
            resolve_texture(
                "wood",
                "mat-itu_wood",
                cache,
                allow_material_type_fallback=False,
            )
            is None
        )

    def test_resolve_texture_ignores_generic_material_id(self):
        cache = {"wood": "/path/to/wood.png"}
        assert (
            resolve_texture(
                "wood",
                "wood",
                cache,
                allow_material_type_fallback=False,
            )
            is None
        )

    def test_resolve_texture_ignores_generic_mat_material_id(self):
        cache = {"wood": "/path/to/wood.png"}
        assert (
            resolve_texture(
                "wood",
                "mat-wood",
                cache,
                allow_material_type_fallback=False,
            )
            is None
        )

    def test_resolve_texture_by_id(self):
        cache = {"mymat": "/path/to/mymat.png", "concrete": "/other.png"}
        assert resolve_texture("concrete", "mat-mymat", cache) == "/path/to/mymat.png"

    def test_resolve_texture_miss(self):
        cache = {"concrete": "/path/to/concrete.png"}
        assert resolve_texture("glass", None, cache) is None

    def test_build_texture_cache(self, tmp_path):
        tex_dir = tmp_path / "libraries" / "textures"
        tex_dir.mkdir(parents=True)
        (tex_dir / "concrete.png").write_text("fake")
        (tex_dir / "glass.jpg").write_text("fake")
        (tex_dir / "readme.txt").write_text("skip")

        cache = build_texture_cache(tmp_path)
        assert "concrete" in cache
        assert "glass" in cache
        assert "readme" not in cache


# ---------------------------------------------------------------------------
# Sphere payload
# ---------------------------------------------------------------------------


@pytest.mark.headless
class TestSpherePayload:
    def test_shape_and_type(self):
        payload = make_sphere_payload([0, 0, 0], radius=1.0, color=[1, 0, 0])
        assert isinstance(payload, MeshPayload)
        assert payload.vertices.ndim == 2
        assert payload.vertices.shape[1] == 3
        assert payload.triangles.ndim == 2
        assert payload.triangles.shape[1] == 3
        assert payload.normals is not None
        assert payload.vertex_colors is not None
        assert payload.vertex_colors.shape == payload.vertices.shape

    def test_position_offset(self):
        pos = np.array([10.0, 20.0, 30.0])
        payload = make_sphere_payload(pos, radius=2.0, color=[0, 1, 0])
        center = payload.vertices.mean(axis=0)
        np.testing.assert_allclose(center, pos, atol=0.1)

    def test_radius_scaling(self):
        p1 = make_sphere_payload([0, 0, 0], radius=1.0, color=[1, 1, 1])
        p2 = make_sphere_payload([0, 0, 0], radius=5.0, color=[1, 1, 1])
        extent1 = p1.vertices.max(axis=0) - p1.vertices.min(axis=0)
        extent2 = p2.vertices.max(axis=0) - p2.vertices.min(axis=0)
        ratio = extent2.mean() / extent1.mean()
        np.testing.assert_allclose(ratio, 5.0, atol=0.1)


# ---------------------------------------------------------------------------
# ViewModel -> MPC payloads
# ---------------------------------------------------------------------------


@pytest.mark.headless
class TestViewModelToPayloads:
    def _make_mock_vm(self, n_points=10, n_lines=5, *, bounce_points=True):
        """Create a minimal ViewModel-like object."""

        class MockVM:
            pass

        vm = MockVM()
        vm.mpc_points = np.random.rand(n_points, 3).astype(np.float32)
        vm.mpc_lines = np.column_stack([np.arange(n_lines), np.arange(1, n_lines + 1)]).astype(
            np.int32
        )
        vm.mpc_colors = np.random.rand(n_lines, 3).astype(np.float32)
        bounce_colors = np.random.rand(n_points, 3).astype(np.float32)
        vm.mpc_bounce_points = vm.mpc_points[1:-1]
        vm.mpc_bounce_colors = bounce_colors[1:-1]
        vm.mpc_visibility = MpcVisibility(bounce_points=bounce_points)
        return vm

    def test_returns_payloads(self):
        vm = self._make_mock_vm()
        lines, points = view_model_to_mpc_payloads(vm)
        assert isinstance(lines, LineSetPayload)
        assert isinstance(points, PointCloudPayload)
        assert lines.points.shape == (10, 3)
        assert lines.lines.shape == (5, 2)

    def test_no_bounces(self):
        vm = self._make_mock_vm(bounce_points=False)
        lines, points = view_model_to_mpc_payloads(vm)
        assert lines is not None
        assert points is None

    def test_none_view_model(self):
        lines, points = view_model_to_mpc_payloads(None)
        assert lines is None
        assert points is None

    def test_empty_lines(self):
        vm = self._make_mock_vm(n_points=0, n_lines=0)
        vm.mpc_lines = np.empty((0, 2), dtype=np.int32)
        vm.mpc_colors = np.empty((0, 3), dtype=np.float32)
        lines, points = view_model_to_mpc_payloads(vm)
        assert lines is None
        assert points is None


# ---------------------------------------------------------------------------
# Camera orbit computation
# ---------------------------------------------------------------------------


@pytest.mark.headless
class TestComputeCameraOrbit:
    def test_default_values(self):
        orbit = compute_camera_orbit([], np.empty((0, 3)), np.empty((0, 3)))
        assert orbit.azimuth_deg == 45.0
        assert orbit.elevation_deg == 30.0
        assert orbit.distance == 100.0

    def test_custom_values(self):
        orbit = compute_camera_orbit(
            [],
            np.empty((0, 3)),
            np.empty((0, 3)),
            azimuth=90.0,
            elevation=60.0,
            distance=200.0,
            center=[1, 2, 3],
        )
        assert orbit.azimuth_deg == 90.0
        assert orbit.elevation_deg == 60.0
        assert orbit.distance == 200.0
        np.testing.assert_allclose(orbit.center, [1, 2, 3])

    def test_auto_distance_from_bboxes(self):
        bboxes = [
            (np.array([0, 0, 0]), np.array([-50, -50, 0]), np.array([50, 50, 10])),
        ]
        orbit = compute_camera_orbit(bboxes, np.empty((0, 3)), np.empty((0, 3)))
        assert orbit.distance > 50.0


# ---------------------------------------------------------------------------
# mesh_entry_to_payload
# ---------------------------------------------------------------------------


@pytest.mark.headless
class TestMeshEntryToPayload:
    def test_invisible_entry(self):
        entry = {"mesh": None, "visible": False}
        assert mesh_entry_to_payload(entry, {}) is None

    def test_no_mesh(self):
        entry = {"mesh": None, "visible": True}
        assert mesh_entry_to_payload(entry, {}) is None

    def test_valid_entry(self):
        entry = {
            "mesh": _triangle_mesh_payload(),
            "visible": True,
            "name": "test_sphere",
            "material_type": "default",
            "material_id": None,
        }
        result = mesh_entry_to_payload(entry, {}, index=0)
        assert result is not None
        name, payload, material = result
        assert "test_sphere" in name
        assert isinstance(payload, MeshPayload)
        assert isinstance(material, MaterialPayload)
        assert payload.vertices.shape[1] == 3

    def test_solid_material_does_not_use_type_texture_fallback(self):
        entry = {
            "mesh": _box_mesh_payload(),
            "visible": True,
            "name": "wood_wall",
            "material_type": "wood",
            "material_id": "mat-itu_wood",
        }

        result = mesh_entry_to_payload(entry, {"wood": "/path/to/wood.png"}, index=0)

        assert result is not None
        _name, _payload, material = result
        assert material.texture_path is None

    def test_material_id_texture_fallback_still_applies(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
        monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
        texture_path = tmp_path / "custom_wood.png"
        texture_path.write_bytes(b"texture")
        entry = {
            "mesh": _box_mesh_payload(),
            "visible": True,
            "name": "custom_wall",
            "material_type": "wood",
            "material_id": "mat-custom_wood",
        }

        result = mesh_entry_to_payload(
            entry,
            {"custom_wood": str(texture_path)},
            index=0,
        )

        assert result is not None
        _name, _payload, material = result
        assert material.texture_path == str(texture_path)

    def test_emissive_intensity_uses_entry_color_by_default(self, monkeypatch):
        from visualizer.src.materials.catalog import ITU_TO_PBR

        monkeypatch.setitem(
            ITU_TO_PBR,
            "doc_glow",
            {
                "color": [0.0, 0.0, 0.0],
                "roughness": 0.5,
                "metallic": 0.0,
                "reflectance": 0.5,
                "alpha": 1.0,
                "emissive_intensity": 3.0,
            },
        )

        entry = {
            "mesh": _triangle_mesh_payload(),
            "visible": True,
            "name": "glow_sphere",
            "material_type": "doc_glow",
            "material_id": None,
            "color": [0.2, 0.4, 0.9],
        }
        result = mesh_entry_to_payload(entry, {}, index=0)
        assert result is not None
        _name, _payload, material = result
        assert material.emissive_color == pytest.approx((0.2, 0.4, 0.9), abs=1e-5)
        assert material.emissive_intensity == pytest.approx(3.0, abs=1e-5)


# ---------------------------------------------------------------------------
# target_metadata_to_payload
# ---------------------------------------------------------------------------


@pytest.mark.headless
class TestTargetMetadataToPayload:
    def test_loads_target_mesh_without_renderer_native_objects(self, tmp_path):
        _write_target_ply(tmp_path / "target.ply")
        meta = {
            "name": "walker",
            "mesh_file": "target.ply",
            "mesh_directory": "",
            "current_position": [10.0, 20.0, 5.0],
            "scale": 2.0,
            "orientation": [90.0, 0.0, 0.0],
        }

        result = target_metadata_to_payload(meta, tmp_path, tmp_path)

        assert result is not None
        name, payload, material = result
        assert name == "walker"
        assert isinstance(payload, MeshPayload)
        assert isinstance(material, MaterialPayload)
        assert not hasattr(payload, "get_center")
        np.testing.assert_allclose(
            (payload.vertices.min(axis=0) + payload.vertices.max(axis=0)) / 2.0,
            [10.0, 20.0, 5.0],
            atol=1e-10,
        )
        assert payload.vertex_colors is not None
        np.testing.assert_allclose(payload.vertex_colors[0], [1.0, 0.0, 0.0])
        assert payload.normals is not None
        np.testing.assert_allclose(payload.normals[0], [0.0, 0.0, 1.0])

    def test_missing_target_mesh_returns_none(self, tmp_path):
        meta = {
            "name": "missing",
            "mesh_file": "missing.ply",
            "mesh_directory": "",
        }

        assert target_metadata_to_payload(meta, tmp_path, tmp_path) is None
