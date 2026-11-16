"""Tests for renderer-neutral material catalog helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_material_catalog_import_does_not_import_open3d() -> None:
    """Importing material presets should not pull in an Open3D backend."""
    code = (
        "import sys; "
        "import visualizer.src.materials.catalog as catalog; "
        "assert catalog.ITU_TO_PBR['default']['alpha'] == 1.0; "
        "assert 'open3d' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_working_checkout_supplies_installed_visual_assets(tmp_path) -> None:
    project = Path(__file__).resolve().parents[3]
    checkout = tmp_path / "assets"
    (checkout / "scenarios").mkdir(parents=True)
    for relative in (
        "ibl/neutral_outdoor.hdr",
        "ibl/neutral_outdoor_ibl.ktx",
        "ibl/neutral_outdoor_skybox.ktx",
        "textures/pbr/concrete/albedo.png",
        "textures/pbr/water_ripples/albedo.png",
        "textures/pbr/water_ripples/normal.png",
        "textures/pbr/water_ripples/roughness.png",
    ):
        target = checkout / "libraries" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / "libraries" / relative, target)

    code = """
from pathlib import Path
from types import SimpleNamespace
import sys
from visualizer.src.authoring.dependencies import ScenarioDependencyCollector
from visualizer.src.materials.catalog import material_preset
from visualizer.src.materials.surface_defaults import geographic_surface_properties
from visualizer.src.renderers.pygfx.renderer import _default_ibl_dir
from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager
from visualizer.src.renderers.open3d.lighting_controls import Open3DLightingControlsMixin
from visualizer.src.services.scene_service import TEXTURE_LIBRARY_PATH

root = Path.cwd()
assert Path(material_preset("concrete")["texture_path"]).samefile(
    root / "libraries/textures/pbr/concrete/albedo.png")
water = geographic_surface_properties("water.ply", "water")
for key, name in (("texture_path", "albedo"), ("normal_map_path", "normal"),
                  ("roughness_map_path", "roughness")):
    assert Path(water[key]).samefile(root / f"libraries/textures/pbr/water_ripples/{name}.png")
ibl = PygfxIBLManager(SimpleNamespace(Scene=lambda: SimpleNamespace()), _default_ibl_dir())
assert ibl._resolve_hdr_path("neutral_outdoor").samefile(root / "libraries/ibl/neutral_outdoor.hdr")
assert Path(Open3DLightingControlsMixin()._resolve_default_ibl()).samefile(
    root / "libraries/ibl/neutral_outdoor_ibl.ktx")
assert TEXTURE_LIBRARY_PATH == root / "libraries/textures"
assert ScenarioDependencyCollector().library_root == root / "libraries"
assert ScenarioDependencyCollector(Path(sys.argv[1])).project_root == Path(sys.argv[1])
"""
    subprocess.run(
        [sys.executable, "-c", code, str(project)],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(project)},
        check=True,
    )


def test_material_catalog_normalizes_scene_material_names() -> None:
    from visualizer.src.materials.catalog import normalize_material_type_name

    assert normalize_material_type_name("mat-itu_concrete") == "concrete"
    assert normalize_material_type_name("mat_itu_glass") == "glass"
    assert normalize_material_type_name("itu-wood") == "wood"
    assert normalize_material_type_name("") == "default"


def test_material_catalog_infers_known_type_from_material_id() -> None:
    from visualizer.src.materials.catalog import infer_material_type_from_id, material_id_stem

    assert infer_material_type_from_id("mat-itu_ceiling_board") == "ceiling_board"
    assert infer_material_type_from_id("concrete_material") == "concrete"
    assert infer_material_type_from_id("mat-custom") == "default"
    assert material_id_stem("mat-ground_asphalt") == "ground_asphalt"


def test_material_preset_returns_copy_and_default_fallback() -> None:
    from visualizer.src.materials.catalog import material_preset

    concrete = material_preset("mat-itu_concrete")
    concrete["roughness"] = 0.0

    assert material_preset("concrete")["roughness"] == 0.8
    assert material_preset("unknown")["color"] == [0.8, 0.6, 0.5]
