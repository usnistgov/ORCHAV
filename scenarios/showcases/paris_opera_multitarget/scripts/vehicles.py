"""Static, uncompressed CC0 vehicle GLB decoder for this showcase.

Adapted from the existing licensed vehicle inspection utility. Geometry and PBR
textures are preserved; glTF Y-up is converted to the scene's Z-up metre frame.
"""

import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parents[1] / "assets/vehicles"


def read_vehicle(name, assets=ASSETS):
    """Convert the static GLB to ORCHAV payloads, grouped by source material."""
    from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload

    folder = Path(assets) / name
    raw = (folder / "model.glb").read_bytes()
    magic, version, size = struct.unpack_from("<4sII", raw)
    assert (magic, version, size) == (b"glTF", 2, len(raw))
    json_size, json_type = struct.unpack_from("<II", raw, 12)
    assert json_type == 0x4E4F534A
    doc = json.loads(raw[20 : 20 + json_size])
    bin_size, bin_type = struct.unpack_from("<II", raw, 20 + json_size)
    assert bin_type == 0x004E4942
    binary = memoryview(raw)[28 + json_size : 28 + json_size + bin_size]
    assert not doc.get("animations") and not doc.get("skins")

    def accessor(index):
        a = doc["accessors"][index]
        assert "sparse" not in a and not a.get("normalized", False)
        view = doc["bufferViews"][a["bufferView"]]
        assert view.get("buffer", 0) == 0
        dtype = np.dtype({5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]])
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride", dtype.itemsize * width)
        return np.ndarray(
            (a["count"], width),
            dtype=dtype,
            buffer=binary,
            offset=offset,
            strides=(stride, dtype.itemsize),
        ).copy()

    texture_dir = folder / "textures"
    texture_dir.mkdir(exist_ok=True)
    textures = []
    for index, source in enumerate(doc["images"]):
        assert source["mimeType"] == "image/png"
        view = doc["bufferViews"][source["bufferView"]]
        start = view.get("byteOffset", 0)
        path = texture_dir / f"image_{index:02d}.png"
        path.write_bytes(binary[start : start + view["byteLength"]])
        textures.append(path)

    # Right-handed glTF Y-up -> ORCHAV Z-up, with source metres preserved.
    zup = np.eye(4)
    zup[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    grouped = defaultdict(list)
    components = []

    def visit(index, parent):
        node = doc["nodes"][index]
        if "matrix" in node:
            local = np.asarray(node["matrix"]).reshape(4, 4).T
        else:
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(
                node.get("rotation", [0, 0, 0, 1])
            ).as_matrix() @ np.diag(node.get("scale", [1, 1, 1]))
            local[:3, 3] = node.get("translation", [0, 0, 0])
        world = parent @ local
        if "mesh" in node:
            for primitive in doc["meshes"][node["mesh"]]["primitives"]:
                assert primitive.get("mode", 4) == 4 and not primitive.get("targets")
                attributes = primitive["attributes"]
                positions = accessor(attributes["POSITION"])
                positions = positions @ world[:3, :3].T + world[:3, 3]
                normals = accessor(attributes["NORMAL"]) @ np.linalg.inv(world[:3, :3])
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                uv = accessor(attributes["TEXCOORD_0"])
                faces = (
                    (
                        accessor(primitive["indices"]).ravel()
                        if "indices" in primitive
                        else np.arange(len(positions))
                    )
                    .reshape(-1, 3)
                    .astype(np.int32)
                )
                if np.linalg.det(world[:3, :3]) < 0:
                    faces = faces[:, [0, 2, 1]]
                material_index = primitive["material"]
                material = doc["materials"][material_index]
                texture = material["pbrMetallicRoughness"]["baseColorTexture"]
                transform = texture.get("extensions", {}).get("KHR_texture_transform", {})
                assert transform.get("texCoord", texture.get("texCoord", 0)) == 0
                uv *= transform.get("scale", [1, 1])
                angle = transform.get("rotation", 0)
                uv = uv @ np.array(
                    [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
                )
                uv += transform.get("offset", [0, 0])
                grouped[material_index].append((positions, normals, uv, faces))
                components.append(
                    dict(
                        name=node.get("name"),
                        triangles=len(faces),
                        bounds=[positions.min(0).tolist(), positions.max(0).tolist()],
                    )
                )
        for child in node.get("children", []):
            visit(child, world)

    for node in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        visit(node, zup)
    all_positions = np.concatenate([part[0] for parts in grouped.values() for part in parts])
    lower, upper = all_positions.min(0), all_positions.max(0)
    translation = np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]])
    result = []
    for index, parts in grouped.items():
        positions, normals, uv, faces = [], [], [], []
        count = 0
        for p, n, t, f in parts:
            positions.append(p - translation)
            normals.append(n)
            uv.append(t)
            faces.append(f + count)
            count += len(p)
        positions, normals, uv, faces = map(np.concatenate, (positions, normals, uv, faces))
        source = doc["materials"][index]
        pbr = source["pbrMetallicRoughness"]
        extensions = source.get("extensions", {})
        coat = extensions.get("KHR_materials_clearcoat", {})
        texture = textures[doc["textures"][pbr["baseColorTexture"]["index"]]["source"]]
        factor = pbr.get("baseColorFactor", [1, 1, 1, 1])
        assert factor == [1, 1, 1, 1], "Unexpected albedo multiplier; handle in texture preparation"
        payload = MeshPayload(
            vertices=positions.astype(np.float32),
            triangles=faces,
            normals=normals.astype(np.float32),
            triangle_uvs=uv[faces].reshape(-1, 2).astype(np.float32),
        )
        # Native pygfx alpha gives clear glazing a readable display fallback;
        # source GLB and textures are not altered. Optical refraction is absent.
        glazing = source["name"] in ("glass.001", "Original clear glazing")
        material = MaterialPayload(
            base_color=(1, 1, 1, 0.28 if glazing else 1),
            texture_path=str(texture),
            metallic=pbr.get("metallicFactor", 1),
            roughness=pbr.get("roughnessFactor", 1),
            clearcoat=coat.get("clearcoatFactor", 0),
            clearcoat_roughness=coat.get("clearcoatRoughnessFactor", 0),
            transmission=extensions.get("KHR_materials_transmission", {}).get(
                "transmissionFactor", 0
            ),
        )
        result.append((source["name"], payload, material))
    info = dict(
        source=json.loads((folder / "provenance.json").read_text()),
        dimensions_m=(upper - lower).tolist(),
        triangle_count=sum(len(p.triangles) for _, p, _ in result),
        material_count=len(result),
        component_count=len(components),
        components=components,
        coordinate_transform="glTF (x,y,z) to ORCHAV (x,-z,y); centre XY and ground Z; metres unchanged",
        material_mapping="Source albedo, UV transforms, normals, metallic, roughness and clearcoat preserved. Clear glazing uses 0.28 alpha in the display material; source GLB unchanged. Pygfx does not reproduce source refractive transmission.",
    )
    return result, np.array([lower - translation, upper - translation]), info
