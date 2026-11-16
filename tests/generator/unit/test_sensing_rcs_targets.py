"""Native sensing factories, pose ownership and actual engine metadata."""

import importlib
import math
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from generator.core.target.rcs import (
    NativeSensingCapabilityError,
    NativeTargetManager,
    create_target_manager,
)
from generator.core.target.representation import NativeTargetConfig
from shared.scenarios.actors import (
    ActorsSpec,
    ConstantRCSSpec,
    FixedOrientationSpec,
    NativeBoxGeometrySpec,
    NativeMeshGeometrySpec,
    TimelineSpec,
    TR38901RCSSpec,
)


class _Scene:
    def __init__(self):
        self.objects = {}
        self.calls = []

    def get(self, name):
        return self.objects.get(name)

    def edit(self, *, add=(), remove=()):
        self.calls.append((tuple(add), tuple(remove)))
        for target in remove:
            self.objects.pop(target.name, None)
        for target in add:
            self.objects[target.name] = target


def _config(**changes):
    values = {
        "name": "target",
        "representation": "constant_rcs",
        "geometry": NativeBoxGeometrySpec(length_m=1, width_m=2, height_m=3),
        "constant_rcs": ConstantRCSSpec(sigma_m2=0.25),
        "orientation": FixedOrientationSpec(yaw_deg=90),
        "_initial_position": (10, 20, 30),
        "_initial_velocity": (2, 0, 0),
    }
    values.update(changes)
    return NativeTargetConfig(**values)


@pytest.fixture
def native_engine():
    mi = pytest.importorskip("mitsuba")
    if mi.variant() is None:
        mi.set_variant(os.environ.get("ORCHAV_NATIVE_RCS_VARIANT", "llvm_ad_mono_polarized"))
    rcs = pytest.importorskip("sionna.rt.rcs")
    if not hasattr(rcs, "ConstantRCSSensingTarget"):
        pytest.skip("Sionna RT native sensing targets are unavailable")
    return mi, rcs


def test_factory_keeps_mesh_manager_unchanged(monkeypatch):
    calls = []
    fake_module = ModuleType("generator.core.target.manager")
    fake_module.TargetManager = lambda *args, **kwargs: calls.append((args, kwargs)) or "mesh"
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    config = SimpleNamespace(name="mesh")
    scene = _Scene()
    assert create_target_manager(config, scene, {"metal": 1}) == "mesh"
    assert calls == [((config, scene), {"material_overrides": {"metal": 1}})]
    native = create_target_manager(_config(), scene, {"metal": 1})
    assert isinstance(native, NativeTargetManager)
    with pytest.raises(ValueError, match="NativeTargetConfig"):
        create_target_manager(SimpleNamespace(representation="constant_rcs"), scene)


def test_native_configuration_imports_do_not_load_engine():
    code = """
import sys
from generator.core.target.rcs import NativeTargetManager
from generator.core.target.representation import NativeTargetConfig
from shared.scenarios.model import NativeRcsSolverModel
assert not any(name == package or name.startswith(package + '.')
               for package in ('mitsuba', 'drjit', 'sionna')
               for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=30)


def test_missing_native_capability_is_a_clear_error(monkeypatch):
    module = importlib.import_module("generator.core.target.rcs")
    monkeypatch.setattr(module, "version", lambda package: "2.2.0")
    original = module.importlib.import_module

    def imports(name):
        if name == "sionna.rt.rcs":
            raise ImportError("no native sensing")
        if name == "mitsuba":
            return SimpleNamespace()
        return original(name)

    monkeypatch.setattr(module.importlib, "import_module", imports)
    with pytest.raises(NativeSensingCapabilityError, match="Sionna RT 2.2"):
        NativeTargetManager(_config(), _Scene()).create_target()


@pytest.mark.parametrize("installed", ["2.0.1", "2.1.0", "2.2.0rc1", "invalid"])
def test_native_version_gate_runs_before_engine_import(monkeypatch, installed):
    module = importlib.import_module("generator.core.target.rcs")
    monkeypatch.setattr(module, "version", lambda package: installed)
    imports = []
    monkeypatch.setattr(module.importlib, "import_module", lambda name: imports.append(name))
    with pytest.raises(NativeSensingCapabilityError, match=r"orchav\[native-rcs\]"):
        NativeTargetManager(_config(), _Scene()).create_target()
    assert imports == []


def test_native_version_gate_requires_distribution_metadata(monkeypatch):
    module = importlib.import_module("generator.core.target.rcs")

    def missing(package):
        raise module.PackageNotFoundError(package)

    monkeypatch.setattr(module, "version", missing)
    with pytest.raises(NativeSensingCapabilityError, match="Sionna RT 2.2"):
        NativeTargetManager(_config(), _Scene()).create_target()


def test_native_capability_gate_checks_solver_and_polarization(monkeypatch):
    module = importlib.import_module("generator.core.target.rcs")
    monkeypatch.setattr(module, "version", lambda package: "2.2.0")
    fake_rcs = SimpleNamespace(
        ConstantRCSSensingTarget=lambda: None, TR38901SensingTarget=lambda: None
    )
    fake_mi = SimpleNamespace(variant=lambda: "llvm_ad_mono_polarized")
    monkeypatch.setattr(
        module.importlib, "import_module", lambda name: fake_mi if name == "mitsuba" else fake_rcs
    )
    with pytest.raises(NativeSensingCapabilityError, match="RCSSolver"):
        module._native_engine()
    fake_rcs.RCSSolver = lambda: None
    fake_mi.variant = lambda: "llvm_ad_rgb"
    with pytest.raises(NativeSensingCapabilityError, match="mono_polarized"):
        module._native_engine()
    fake_mi.variant = lambda: "llvm_ad_mono_polarized"
    assert module._native_engine() == (fake_mi, fake_rcs)


def test_native_geometry_config_requires_resolved_static_mesh(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        _config(geometry=NativeMeshGeometrySpec(path="target.ply"))
    with pytest.raises(ValueError, match="also define"):
        _config(resolved_mesh_path=tmp_path / "target.ply")
    with pytest.raises(ValueError, match="only ConstantRCSSpec"):
        _config(tr38901=TR38901RCSSpec(object_type="human"))


def test_mesh_preview_does_not_invent_aabb_or_points(tmp_path):
    config = _config(
        geometry=NativeMeshGeometrySpec(path="target.ply", scale=2),
        resolved_mesh_path=tmp_path / "target.ply",
    )
    preview = config.preview_metadata()
    assert preview["geometry"] == {
        "type": "mesh",
        "path": "target.ply",
        "scaling": (2, 2, 2),
    }
    assert preview["point_positions_source"] == "awaiting_native_mesh"
    assert "scattering_points_world_m" not in preview


def test_canonical_runtime_preserves_native_mobility_and_orientation(tmp_path):
    from generator.core.scenario_actors.runtime import prepare_actor_runtime

    scenario = SimpleNamespace(
        root=tmp_path,
        timeline=TimelineSpec(steps=3, duration_s=2),
        actors=ActorsSpec.model_validate(
            {
                "targets": [
                    {
                        "name": "native",
                        "representation": "constant_rcs",
                        "geometry": {"type": "box"},
                        "constant_rcs": {"sigma_m2": 0.1},
                        "mobility": {"type": "linear", "start_m": [1, 2, 3], "end_m": [5, 2, 3]},
                        "orientation": {"type": "fixed", "yaw_deg": 30},
                    }
                ]
            }
        ),
    )
    runtime = prepare_actor_runtime(scenario)
    config = runtime.targets[0]
    assert isinstance(config, NativeTargetConfig)
    assert config.initial_position == (1, 2, 3)
    assert config.initial_velocity == (2, 0, 0)
    assert config.mobility.prepared_positions() == [(1, 2, 3), (3, 2, 3), (5, 2, 3)]
    assert config.orientation.orientations()[0] == pytest.approx((30, 0, 0))
    assert not config.switch_meshes and not config.use_ply_position


@pytest.mark.optional_runtime
def test_native_constant_has_one_proxy_and_updates_pose(native_engine):
    scene = _Scene()
    manager = NativeTargetManager(_config(), scene)
    target = manager.create_target()
    assert list(scene.objects) == ["target"]
    assert target is manager.update_mesh_for_frame(20)
    assert len(scene.calls) == 1
    assert target.radio_material.__class__.__name__ == "AbsorberRadioMaterial"
    initial = manager.metadata_snapshot()
    assert initial["constant_rcs"]["sigma_m2"] == 0.25
    assert initial["constant_rcs"]["xpr_db"] is None
    assert initial["scattering_points_local_m"] == ((0, 0, 0),)
    assert initial["scattering_points_world_m"] == ((10, 20, 30),)
    assert initial["scattering_point_ids"] == (0,)
    assert initial["orientation"] == pytest.approx((math.pi / 2, 0, 0))
    assert initial["velocity_m_s"] == (2, 0, 0)
    manager.apply_position_snapshot((4, 5, 6))
    manager.apply_orientation_snapshot((0, 0, 0))
    manager.apply_velocity_snapshot((0, 3, 0))
    manager.apply_scale_snapshot(2)
    moved = manager.metadata_snapshot()
    assert moved["current_position"] == (4, 5, 6)
    assert moved["velocity_m_s"] == (0, 3, 0)
    assert moved["geometry"]["length_m"] == 2
    assert moved["geometry"]["width_m"] == 4
    assert moved["geometry"]["height_m"] == 6
    assert initial["current_position"] == (10, 20, 30)
    target.sigma = 0.5
    target.xpr_db = 3
    updated = manager.metadata_snapshot()
    assert updated["constant_rcs"] == {"sigma_m2": 0.5, "xpr_db": 3}
    assert initial["constant_rcs"]["sigma_m2"] == 0.25


@pytest.mark.optional_runtime
@pytest.mark.parametrize(
    "object_type,model_type",
    [
        ("uav-small-size", 1),
        ("human", 1),
        ("human", 2),
        ("uav-large-size", 2),
        ("vehicle-single-sp", 2),
        ("vehicle-multi-sp", 2),
        ("agv-single-sp", 2),
        ("agv-multi-sp", 2),
    ],
)
def test_all_presets_and_preview_points_match_native_engine(native_engine, object_type, model_type):
    import numpy as np

    config = _config(
        representation="tr38901",
        constant_rcs=None,
        tr38901=TR38901RCSSpec(object_type=object_type, model_type=model_type),
        geometry=NativeBoxGeometrySpec(),
    )
    manager = NativeTargetManager(config, _Scene())
    manager.create_target()
    actual, preview = manager.metadata_snapshot(), config.preview_metadata()
    assert len(actual["scattering_point_ids"]) == config.tr38901.num_scattering_points
    for field in ("scattering_points_local_m", "scattering_points_world_m"):
        np.testing.assert_allclose(actual[field], preview[field], rtol=1e-6, atol=1e-6)
    for dimension in ("length_m", "width_m", "height_m"):
        assert actual["geometry"][dimension] == pytest.approx(preview["geometry"][dimension])
    assert actual["tr38901"]["random_sigma_s"] is False
    assert actual["tr38901"]["random_xpr"] is False
    assert actual["tr38901"]["random_phases"] is False


@pytest.mark.optional_runtime
def test_failed_scene_edit_removes_partial_native_candidate(native_engine):
    scene = _Scene()
    original_edit = scene.edit

    def failing_edit(*, add=(), remove=()):
        original_edit(add=add, remove=remove)
        if add:
            raise RuntimeError("edit failed")

    scene.edit = failing_edit
    manager = NativeTargetManager(_config(), scene)
    with pytest.raises(Exception, match="creation failed"):
        manager.create_target()
    assert manager.target_object is None
    assert scene.objects == {}


@pytest.mark.optional_runtime
def test_static_mesh_proxy_has_effective_native_dimensions(native_engine, tmp_path):
    mesh_path = tmp_path / "proxy.ply"
    vertices = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    faces = [
        (0, 2, 3),
        (0, 3, 1),
        (4, 5, 7),
        (4, 7, 6),
        (0, 1, 5),
        (0, 5, 4),
        (2, 6, 7),
        (2, 7, 3),
        (0, 4, 6),
        (0, 6, 2),
        (1, 3, 7),
        (1, 7, 5),
    ]
    mesh_path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\n"
        "property float z\nelement face 12\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        + "".join(f"{x} {y} {z}\n" for x, y, z in vertices)
        + "".join(f"3 {a} {b} {c}\n" for a, b, c in faces),
        encoding="utf-8",
    )
    config = _config(
        geometry=NativeMeshGeometrySpec(path="proxy.ply", scale=2),
        resolved_mesh_path=mesh_path,
    )
    scene = _Scene()
    manager = NativeTargetManager(config, scene)
    manager.create_target()
    snapshot = manager.metadata_snapshot()
    assert len(scene.objects) == 1
    assert snapshot["geometry"] == {
        "type": "mesh",
        "path": "proxy.ply",
        "scaling": (2, 2, 2),
        "length_m": 2,
        "width_m": 2,
        "height_m": 2,
    }
    assert snapshot["scattering_points_world_m"] == ((10, 20, 30),)


@pytest.mark.optional_runtime
def test_native_target_rejects_duplicate_scene_identity(native_engine):
    scene = _Scene()
    existing = SimpleNamespace(name="target")
    scene.edit(add=[existing])
    manager = NativeTargetManager(_config(), scene)
    with pytest.raises(Exception, match="already contains"):
        manager.create_target()
    assert scene.get("target") is existing
    assert manager.target_object is None


@pytest.mark.optional_runtime
def test_asymmetric_static_mesh_rotation_and_scale_preserve_actor_center(native_engine, tmp_path):
    path = tmp_path / "asymmetric.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
        "property float z\nelement face 4\nproperty list uchar int vertex_indices\n"
        "end_header\n0 0 0\n2 0 0\n0 1 0\n0 0 1\n"
        "3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n",
        encoding="utf-8",
    )
    config = _config(
        geometry=NativeMeshGeometrySpec(path="asymmetric.ply", scale=1.5),
        resolved_mesh_path=path,
        orientation=FixedOrientationSpec(yaw_deg=45),
    )
    manager = NativeTargetManager(config, _Scene())
    manager.create_target()
    assert manager.get_current_position() == pytest.approx(config.initial_position, abs=1e-5)
    assert manager.metadata_snapshot()["scattering_points_world_m"][0] == pytest.approx(
        config.initial_position, abs=1e-5
    )
    manager.apply_position_snapshot((2, 3, 4))
    for orientation in ((90, 0, 0), (40, 20, 10), (-45, -15, 30)):
        manager.apply_orientation_snapshot(orientation)
        assert manager.get_current_position() == pytest.approx((2, 3, 4), abs=1e-5)
        manager.apply_scale_snapshot(2)
        snapshot = manager.metadata_snapshot()
        assert snapshot["current_position"] == pytest.approx((2, 3, 4), abs=1e-5)
        assert snapshot["scattering_points_world_m"][0] == pytest.approx((2, 3, 4), abs=1e-5)
