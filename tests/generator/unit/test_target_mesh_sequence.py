from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

mi = pytest.importorskip("mitsuba")
if not hasattr(mi, "variants") or "llvm_ad_mono_polarized" not in mi.variants():
    pytest.skip("Usable Mitsuba runtime required", allow_module_level=True)
try:
    mi.set_variant("llvm_ad_mono_polarized")
except Exception as exc:
    pytest.skip(f"Usable Mitsuba runtime required: {exc}", allow_module_level=True)

import generator.core.target as target_package
from generator.core.materials.defaults import (
    DEFAULT_TARGET_MATERIAL_THICKNESS,
    DEFAULT_TARGET_SCATTERING_COEFFICIENT,
)
from generator.core.materials.target_materials import (
    apply_target_material_overrides,
    target_material_override_key_candidates,
)
from generator.core.orientation import FixedOrientationSpec, LookAtOrientationSpec
from generator.core.target import TargetConfig, TargetManager


class _FakeScene:
    def __init__(self) -> None:
        self.added = []
        self.removed = []

    def edit(self, *, add=None, remove=None) -> None:
        if add:
            self.added.extend(add)
        if remove:
            self.removed.extend(remove)


class _FakeMaterial:
    name = "fake-material"


class _FakeRadioMaterial:
    def __init__(self) -> None:
        self.name = "mat-itu_metal_MunichDrone"
        self.scattering_coefficient = 0.0
        self.scattering_pattern = "isotropic"
        self.thickness = DEFAULT_TARGET_MATERIAL_THICKNESS


class _FakeSceneObject:
    def __init__(self, *, fname: str, name: str, radio_material) -> None:
        self.fname = fname
        self.name = name
        self.radio_material = radio_material
        self.position = None
        self.scaling = None
        self.orientation = None


def _write_mesh_sequence(mesh_dir: Path, count: int = 5) -> list[str]:
    mesh_paths: list[str] = []
    for idx in range(count):
        mesh_path = mesh_dir / f"mesh_{idx:05d}.obj"
        mesh_path.write_text("# fake mesh\n", encoding="utf-8")
        mesh_paths.append(str(mesh_path))
    return mesh_paths


def _build_config(
    mesh_dir: Path,
    *,
    mesh_start_index: int = 0,
    mesh_frame_stride: int = 1,
    mesh_end_behavior: str = "loop",
) -> TargetConfig:
    return TargetConfig(
        name="Target",
        mobility=None,
        mesh_pattern="mesh_*.obj",
        mesh_directory=str(mesh_dir),
        resolved_mesh_directory=mesh_dir.resolve(),
        _initial_position=(0.0, 0.0, 0.0),
        mesh_start_index=mesh_start_index,
        mesh_frame_stride=mesh_frame_stride,
        mesh_end_behavior=mesh_end_behavior,
    )


def test_target_manager_applies_start_index_and_stride(tmp_path: Path) -> None:
    _write_mesh_sequence(tmp_path, count=5)
    manager = TargetManager(
        _build_config(tmp_path, mesh_start_index=3, mesh_frame_stride=2),
        scene=_FakeScene(),
    )

    assert manager.current_mesh_idx == 3
    assert manager.config.mesh_end_behavior == "loop"
    assert [manager._mesh_index_for_update_call(i) for i in range(4)] == [3, 0, 2, 4]


def test_target_manager_can_hold_final_mesh_with_start_and_stride(tmp_path: Path) -> None:
    _write_mesh_sequence(tmp_path, count=5)
    manager = TargetManager(
        _build_config(
            tmp_path,
            mesh_start_index=3,
            mesh_frame_stride=2,
            mesh_end_behavior="hold_last",
        ),
        scene=_FakeScene(),
    )

    assert manager.current_mesh_idx == 3
    assert [manager._mesh_index_for_update_call(i) for i in range(4)] == [3, 4, 4, 4]


def test_target_manager_hold_last_clamps_start_beyond_sequence(tmp_path: Path) -> None:
    _write_mesh_sequence(tmp_path, count=5)
    manager = TargetManager(
        _build_config(
            tmp_path,
            mesh_start_index=8,
            mesh_end_behavior="hold_last",
        ),
        scene=_FakeScene(),
    )

    assert manager.current_mesh_idx == 4
    assert [manager._mesh_index_for_update_call(i) for i in range(3)] == [4, 4, 4]


def test_target_package_exports_only_public_classes() -> None:
    assert target_package.__all__ == ["TargetConfig", "TargetManager"]
    assert not hasattr(target_package, "_prepare_mesh_path_for_mitsuba")
    assert not hasattr(target_package, "_write_float_vertex_color_ply")


def test_create_material_uses_named_target_defaults(monkeypatch) -> None:
    class _FakeITUMaterial:
        def __init__(self, *, name, itu_type, thickness, color) -> None:
            self.name = name
            self.itu_type = itu_type
            self.thickness = thickness
            self.color = color
            self.scattering_coefficient = None

    monkeypatch.setattr("sionna.rt.radio_materials.ITURadioMaterial", _FakeITUMaterial)

    manager = TargetManager.__new__(TargetManager)
    manager.material_type = "metal"
    manager.config = SimpleNamespace(name="MunichDrone")
    manager.material_overrides = {}

    manager.create_material()

    assert manager.target_material.thickness == DEFAULT_TARGET_MATERIAL_THICKNESS
    assert manager.target_material.scattering_coefficient == DEFAULT_TARGET_SCATTERING_COEFFICIENT


def test_create_target_uses_configured_start_mesh(tmp_path: Path, monkeypatch) -> None:
    mesh_paths = _write_mesh_sequence(tmp_path, count=5)
    scene = _FakeScene()

    monkeypatch.setattr("generator.core.target.manager.SceneObject", _FakeSceneObject)

    def _fake_create_material(self) -> None:
        self.target_material = _FakeMaterial()

    monkeypatch.setattr(TargetManager, "create_material", _fake_create_material)

    manager = TargetManager(
        _build_config(tmp_path, mesh_start_index=3, mesh_frame_stride=1),
        scene=scene,
    )
    target = manager.create_target()

    assert target is not None
    assert target.fname == mesh_paths[3]
    assert scene.added
    assert scene.added[0].fname == mesh_paths[3]


def test_target_material_override_candidates_are_generic_only() -> None:
    assert target_material_override_key_candidates("metal") == [
        "metal",
        "itu_metal",
        "mat-itu_metal",
    ]


def test_target_material_overrides_ignore_target_specific_aliases() -> None:
    target_material = _FakeRadioMaterial()
    material_overrides = {
        "metal": {
            "scattering_coefficient": DEFAULT_TARGET_SCATTERING_COEFFICIENT,
            "thickness": 0.2,
        },
        "itu_metal_MunichDrone": {"scattering_coefficient": 0.99},
    }

    apply_target_material_overrides(
        target_material,
        material_type="metal",
        target_name="MunichDrone",
        material_overrides=material_overrides,
    )

    assert target_material.scattering_coefficient == DEFAULT_TARGET_SCATTERING_COEFFICIENT
    assert target_material.thickness == 0.2


def test_target_config_validates_mesh_sequence_controls(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mesh_start_index"):
        _build_config(tmp_path, mesh_start_index=-1)

    with pytest.raises(ValueError, match="mesh_frame_stride"):
        _build_config(tmp_path, mesh_frame_stride=0)

    with pytest.raises(ValueError, match="mesh_end_behavior"):
        _build_config(tmp_path, mesh_end_behavior="reverse")


def test_target_config_records_asset_alignment_without_mutating_authored_orientation(
    tmp_path: Path,
) -> None:
    authored = FixedOrientationSpec(yaw_deg=15.0, pitch_deg=2.0, roll_deg=3.0)
    config = TargetConfig(
        name="Target",
        mobility=None,
        mesh_pattern="mesh_*.obj",
        mesh_directory=str(tmp_path),
        resolved_mesh_directory=tmp_path.resolve(),
        orientation=authored,
        asset_front_yaw_offset_deg=90.0,
        _initial_position=(0.0, 0.0, 0.0),
    )

    assert config.orientation is authored
    assert config.orientation == FixedOrientationSpec(
        yaw_deg=15.0,
        pitch_deg=2.0,
        roll_deg=3.0,
    )
    assert config.asset_front_yaw_offset_deg == 90.0

    config.apply_asset_front_yaw_offset(45.0)
    assert config.orientation is authored
    assert config.asset_front_yaw_offset_deg == 45.0


@pytest.mark.parametrize(
    ("orientation", "expected"),
    (
        (FixedOrientationSpec(yaw_deg=10.0), (30.0, 0.0, 0.0)),
        (
            LookAtOrientationSpec(point_m=(1.0, 0.0, 0.0), yaw_offset_deg=15.0),
            (35.0, 0.0, 0.0),
        ),
    ),
    ids=("fixed", "point-look-at"),
)
def test_target_manager_initial_preview_uses_canonical_orientation_and_alignment(
    tmp_path: Path,
    monkeypatch,
    orientation,
    expected,
) -> None:
    config = TargetConfig(
        name="Target",
        mobility=None,
        mesh_pattern="mesh_*.obj",
        mesh_directory=str(tmp_path),
        resolved_mesh_directory=tmp_path.resolve(),
        orientation=orientation,
        asset_front_yaw_offset_deg=20.0,
        _initial_position=(0.0, 0.0, 0.0),
    )
    manager = TargetManager.__new__(TargetManager)
    manager.config = config
    target = SimpleNamespace(orientation=None)
    monkeypatch.setattr(
        "generator.core.target.manager.orientation_to_point3f_with_engine_radians",
        lambda angles: (tuple(angles), tuple(angles)),
    )

    manager._apply_initial_orientation_preview(target, mi.Point3f(0.0, 0.0, 0.0))

    assert target.orientation == pytest.approx(expected)


def test_target_manager_requires_existing_mesh_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="mesh directory not found"):
        TargetManager(_build_config(missing_dir), scene=_FakeScene())


def test_target_manager_requires_matching_mesh_files(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    with pytest.raises(FileNotFoundError, match="has no meshes matching"):
        TargetManager(_build_config(tmp_path), scene=_FakeScene())
