"""Tests for renderer-neutral material catalog helpers."""

from __future__ import annotations

import subprocess
import sys


def test_material_catalog_import_does_not_import_open3d() -> None:
    """Importing material presets should not pull in an Open3D backend."""
    code = (
        "import sys; "
        "import visualizer.src.materials.catalog as catalog; "
        "assert catalog.ITU_TO_PBR['default']['alpha'] == 1.0; "
        "assert 'open3d' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


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
