"""Generate a simulation-free, app-openable ORCHAV material inspection scene."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pygfx as gfx
import yaml

ROOT = Path(__file__).resolve().parents[3]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
from shared.frames import FrameSetWriter, standard_mpc_frame_from_pair_data


def write_ply(path, geometry):
    vertices = np.asarray(geometry.positions.data)
    faces = np.asarray(geometry.indices.data).reshape(-1, 3)
    with path.open("w") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
        )
        np.savetxt(stream, vertices, fmt="%.8g")
        np.savetxt(stream, np.column_stack((np.full(len(faces), 3), faces)), fmt="%d")


def generate(destination=None):
    root = Path(destination or Path(__file__).parent).resolve()
    root.mkdir(parents=True, exist_ok=True)
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    write_ply(assets / "sphere.ply", gfx.sphere_geometry(0.86, 48, 32))
    write_ply(assets / "floor.ply", gfx.box_geometry(15, 14, 0.12))
    write_ply(assets / "panel.ply", gfx.plane_geometry(1.45, 1.25))
    radiance = np.full((64, 128, 3), 0.18, np.float32)
    radiance[10:23, 42:65] = 4.0
    radiance[20:38, 96:108] = 1.0
    maximum = radiance.max(axis=-1)
    mantissa, exponent = np.frexp(maximum)
    rgbe = np.zeros((*maximum.shape, 4), np.uint8)
    rgbe[..., :3] = np.minimum(radiance * (mantissa * 256 / maximum)[..., None], 255).astype(
        np.uint8
    )
    rgbe[..., 3] = (exponent + 128).astype(np.uint8)
    (assets / "reference.hdr").write_bytes(
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 64 +X 128\n" + rgbe.tobytes()
    )
    xml = ET.Element("scene", version="3.0.0")
    manifest = []

    def add(name, mesh, position, color=(0.67, 0.7, 0.74), **properties):
        material = ET.SubElement(xml, "bsdf", type="diffuse", id=f"mat-{name}")
        ET.SubElement(material, "string", name="type", value="default")
        ET.SubElement(material, "rgb", name="reflectance", value=" ".join(map(str, color)))
        for key, value in properties.items():
            ET.SubElement(material, "float", name=key, value=str(value))
        shape = ET.SubElement(xml, "shape", type="ply", id=name)
        ET.SubElement(shape, "string", name="filename", value=f"assets/{mesh}.ply")
        transform = ET.SubElement(shape, "transform", name="to_world")
        ET.SubElement(
            transform, "translate", x=str(position[0]), y=str(position[1]), z=str(position[2])
        )
        ET.SubElement(shape, "ref", name="bsdf", id=f"mat-{name}")
        manifest.append({"name": name, "position_m": position, "color_srgb": color, **properties})

    add("floor", "floor", (0, 0, -0.16), (0.28, 0.29, 0.31), roughness=0.95)
    for row, (name, extra) in enumerate(
        (
            ("dielectric", {}),
            ("metal", {"metallic": 1.0}),
            ("clearcoat", {"clearcoat": 0.4, "clearcoat_roughness": 0.15}),
        )
    ):
        for column, roughness in enumerate((0.08, 0.25, 0.55, 0.9)):
            add(
                f"{name}_roughness_{roughness}",
                "sphere",
                (-3.8 + column * 2.6, 3.9 - row * 2.8, 0.78),
                roughness=roughness,
                **extra,
            )
    for column, reflectance in enumerate((0.0, 0.5, 1.0)):
        add(
            f"reflectance_{reflectance}",
            "sphere",
            (-4.8 + column * 1.9, -4.5, 0.43),
            (0.3, 0.34, 0.4),
            roughness=0.18,
            reflectance=reflectance,
        )
    for column, alpha in enumerate((0.25, 0.5, 0.75)):
        add(
            f"alpha_{alpha}",
            "panel",
            (1 + column * 1.7, -4.5, 0.35),
            (0.2, 0.7, 0.9),
            alpha=alpha,
            roughness=0.6,
        )
    ET.indent(xml)
    ET.ElementTree(xml).write(root / "scene.xml", encoding="utf-8", xml_declaration=True)
    scenario = {
        "schema_version": 2,
        "scene": {"source": "local", "id": "scene.xml"},
        "data": {"mode": "files", "files": {"format": "hdf5", "directory": "frames"}},
        "raytracing": {"enabled": False, "export_path_metrics": False},
        "timeline": {"steps": 1, "duration_s": 0.0},
        "actors": {
            "tx": [
                {"name": "Tx", "mobility": {"type": "stationary", "position_m": [-7, 4.2, 2.6]}}
            ],
            "rx": [{"name": "Rx", "mobility": {"type": "stationary", "position_m": [7, 4.2, 2.6]}}],
        },
        "view_defaults": {
            "resolution": [1280, 900],
            "camera_view": "front",
            "camera_dist": 24,
            "merge_scene_meshes": False,
            "mpc_visibility": {"enabled": False, "paths": False, "bounce_points": False},
        },
        "generator_summary": {"enabled": False},
    }
    (root / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=False))
    frame = standard_mpc_frame_from_pair_data(
        frame_index=0,
        timestamp_s=0.0,
        tx_rx_pairs=np.array([[0, 0]]),
        tx_positions=[[-7, 4.2, 2.6]],
        rx_positions=[[7, 4.2, 2.6]],
        tx_names=["Tx"],
        rx_names=["Rx"],
        vertices_by_pair=[np.empty((0, 0, 3))],
        interactions_by_pair=[np.empty((0, 0), dtype=np.uint8)],
        provenance={
            "kind": "controlled_visual_fixture",
            "simulation": False,
            "description": "Empty paths; no measured or simulated RF claims",
        },
    )
    with FrameSetWriter.for_scenario(root) as writer:
        writer.append(frame)
        writer.finalize(provenance={"kind": "controlled_visual_fixture", "simulation": False})
    (root / "fixture_manifest.json").write_text(
        json.dumps(
            {"kind": "controlled_visual_fixture", "simulation": False, "objects": manifest},
            indent=2,
        )
    )
    return root


if __name__ == "__main__":
    generate()
