"""Prepare optical material maps for the existing Paris scene and BMW skin.

Limestone microrelief is authored in metres; BMW finish masks follow the named
source-material palette. Albedo images, RF meshes and recordings are untouched.
Run with the checkout's Python environment, or through native_appearance.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from visualizer.src.materials.surface_detail import relief_normal, relief_occlusion

VERSION = "paris-optical-detail-v1"
# Slot order belongs to the retained CC0 BMW palette, not to colour inference.
BMW_SLOTS = (
    "CarShellNew",
    "Plate",
    "MatteBlack",
    "Tailight",
    "Chrome",
    "Glass",
    "BlackPlastic",
    "HeadlightMatte",
    "Interior",
    "HeadlightGlass",
    "HeadlightChrome",
    "HeadLightAngelEye",
    "HeadLightGlass",
    "ChromeMatte",
    "Mirror",
    "Material",
    "TireRubber",
    "Brake",
    "interiorRoof",
)


def limestone_maps(size: int = 512) -> dict[str, np.ndarray]:
    """Tileable fine mineral relief; no inferred joints from the albedo image."""
    y, x = np.mgrid[:size, :size] / size
    rng = np.random.default_rng(27421)
    relief = np.zeros((size, size), dtype=np.float32)
    grain = np.zeros_like(relief)
    for _ in range(28):
        fx, fy = rng.integers(14, 72, size=2)
        wave = np.sin(2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 2 * np.pi))
        relief += wave.astype(np.float32) * 0.000018
        grain += wave.astype(np.float32) / 28
    spacing = (2.0 / size, 3.2 / size)
    roughness = np.clip(0.84 + grain * 0.18, 0.75, 0.92)
    return {
        "normal_map_path": relief_normal(relief, spacing),
        "roughness_map_path": np.repeat(np.rint(roughness[..., None] * 255).astype(np.uint8), 3, 2),
        "ao_map_path": relief_occlusion(relief, spacing),
    }


def copper_maps(size: int = 512) -> dict[str, np.ndarray]:
    """Subtle patina roughness and sheet waviness on the existing four-metre UV tile.

    These optical fields describe a weathered finish, not measured corrosion or
    reconstructed seams. The authored copper albedo remains untouched.
    """
    y, x = np.mgrid[:size, :size] / size
    rng = np.random.default_rng(27422)
    relief = np.asarray(0.0008 * np.sin(2 * np.pi * (3 * x + y)), dtype=np.float32)
    patina = np.zeros_like(relief)
    for _ in range(24):
        fx, fy = rng.integers(6, 37, size=2)
        wave = np.sin(2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 2 * np.pi))
        relief += wave.astype(np.float32) * 0.000035
        patina += wave.astype(np.float32) / 24
    roughness = np.clip(0.78 + patina * 0.24, 0.68, 0.88)
    return {
        "normal_map_path": relief_normal(relief, (4.0 / size, 4.0 / size)),
        "roughness_map_path": np.repeat(np.rint(roughness[..., None] * 255).astype(np.uint8), 3, 2),
    }


def bmw_finish_maps(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Keep paint, chrome, glazing and rubber optically distinct in one mesh."""
    height, width = shape
    if width % len(BMW_SLOTS):
        raise ValueError("BMW albedo no longer matches the source material palette")
    tile = width // len(BMW_SLOTS)
    maps = {
        key: np.full((height, width, 3), 255, np.uint8)
        for key in ("roughness_map_path", "metallic_map_path", "clearcoat_roughness_map_path")
    }
    for slot, name in enumerate(BMW_SLOTS):
        roughness, metalness, coat_roughness = 0.72, 0.0, 1.0
        if name == "CarShellNew":
            roughness, metalness, coat_roughness = 0.32, 0.12, 0.13
        elif name in {"Chrome", "HeadlightChrome", "Mirror"}:
            roughness, metalness = 0.19, 0.95
        elif name in {"ChromeMatte", "Brake"}:
            roughness, metalness = 0.43, 0.85
        elif name in {"Glass", "HeadlightGlass", "HeadLightGlass", "Tailight"}:
            roughness, coat_roughness = 0.14, 0.09
        elif name == "TireRubber":
            roughness = 0.94
        for pixels, value in zip(
            maps.values(), (roughness, metalness, coat_roughness), strict=True
        ):
            pixels[:, slot * tile : (slot + 1) * tile] = round(value * 255)
    return maps


def _save_maps(folder: Path, prefix: str, maps: dict[str, np.ndarray]) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, pixels in maps.items():
        path = folder / f"{prefix}_{key}.png"
        if not path.is_file():
            Image.fromarray(pixels).save(path)
        result[key] = path.name
    return result


def prepare_surface_detail(directory: Path = HERE) -> dict:
    """Write reproducible visual sidecars only; source triangle files are read-only."""
    report = {"version": VERSION, "visual_only": True, "rf_geometry_changed": False}
    folder = directory / "assets/vehicles/bmw"
    sidecar = folder / "surface.json"
    if sidecar.is_file():
        config = json.loads(sidecar.read_text())
        with Image.open(folder / config["texture"]) as image:
            shape = (image.height, image.width)
        with np.load(folder / config["uv_map"], allow_pickle=False) as uv:
            slots = uv["triangle_uvs"][:, 0] * len(BMW_SLOTS) - 0.5
            if not np.allclose(slots, np.rint(slots), atol=2e-6):
                raise ValueError("BMW UVs no longer reference the expected material slots")
        names = _save_maps(folder, VERSION, bmw_finish_maps(shape))
        config["material"] = dict(
            roughness=1.0, metallic=1.0, clearcoat=0.7, clearcoat_roughness=1.0, **names
        )
        sidecar.write_text(json.dumps(config, indent=2) + "\n")
        report["bmw"] = dict(
            palette_slots=BMW_SLOTS,
            albedo_sha256=hashlib.sha256((folder / config["texture"]).read_bytes()).hexdigest(),
            uv_sha256=hashlib.sha256((folder / config["uv_map"]).read_bytes()).hexdigest(),
            optical_glass="Opaque source glazing with specular finish; no refraction",
            maps=names,
        )
    scene = directory / "scene_clean"
    visual_path = scene / "visual_surfaces.json"
    if visual_path.is_file():
        document = json.loads(visual_path.read_text())
        names = _save_maps(scene / "textures", VERSION, limestone_maps())
        copper_names = _save_maps(scene / "textures", f"{VERSION}-copper", copper_maps())
        matched = 0
        copper_triangles = 0
        for entry in document["meshes"].values():
            for surface in entry.get("surfaces", []):
                material = surface.get("material", {})
                if Path(material.get("texture_path", "")).name == "paris_limestone_paving.png":
                    material.update({key: f"textures/{value}" for key, value in names.items()})
                    material["roughness"] = 1.0
                    matched += 1
                elif (
                    surface.get("id") == "opera_copper"
                    and Path(material.get("texture_path", "")).name
                    == "opera_verdigris_copper_albedo.png"
                ):
                    material.update(
                        {key: f"textures/{value}" for key, value in copper_names.items()}
                    )
                    material["roughness"] = 1.0
                    copper_triangles += len(surface.get("triangle_indices", []))
        visual_path.write_text(json.dumps(document, separators=(",", ":")) + "\n")
        report["limestone"] = dict(
            matched_surfaces=matched,
            tile_extent_m=[3.2, 2.0],
            relief="28 deterministic periodic mineral waves, 18 micrometres amplitude each",
            maps=names,
        )
        report["copper"] = dict(
            bound_triangles=copper_triangles,
            approximate_tile_extent_m=[4.0, 4.0],
            relief="0.8 mm broad sheet waviness and 35 micrometre patina waves; no authored seams",
            maps=copper_names,
        )
    (directory / "surface_detail_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="Prepared run directory")
    args = parser.parse_args()
    print(json.dumps(prepare_surface_detail(args.scenario.resolve()), indent=2))
