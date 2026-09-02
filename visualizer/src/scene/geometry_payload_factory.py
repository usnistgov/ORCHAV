"""Geometry payload factory helpers.

Services and controllers use this module to build renderer-neutral payloads.
Backend-native geometry conversion belongs inside renderer packages.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..types.render_payloads import (
    LineSetPayload,
    MeshPayload,
    PointCloudPayload,
    SurfaceColorSource,
)


def _as_color(color: Optional[Iterable[float]]) -> Optional[np.ndarray]:
    """Normalize optional RGB-like inputs to a 3-channel float array."""
    if color is None:
        return None
    values = np.asarray(list(color), dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.size < 3:
        raise ValueError("Color must provide at least 3 channels")
    return values[:3]


def _vertex_colors(vertex_count: int, color: Optional[Iterable[float]]) -> Optional[np.ndarray]:
    """Return per-vertex color rows for constant-color mesh payloads."""
    rgb = _as_color(color)
    if rgb is None:
        return None
    return np.tile(rgb, (int(vertex_count), 1))


def _compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> Optional[np.ndarray]:
    """Compute averaged vertex normals from triangle winding."""
    if len(vertices) == 0 or len(triangles) == 0:
        return None
    normals = np.zeros_like(vertices, dtype=float)
    for tri in np.asarray(triangles, dtype=np.int64):
        if np.any(tri < 0) or np.any(tri >= len(vertices)):
            continue
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length > 0.0:
            normals[tri] += normal / length
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    if not np.any(valid):
        return None
    normals[valid] = normals[valid] / lengths[valid, None]
    return normals


def make_sphere_payload(
    radius: float,
    color: Optional[Iterable[float]] = None,
    resolution: int = 10,
) -> MeshPayload:
    """Create a UV-sphere mesh payload."""
    if radius <= 0.0:
        raise ValueError("Sphere radius must be positive")
    rings = max(3, int(resolution))
    segments = max(8, rings * 2)

    vertices: list[list[float]] = [[0.0, 0.0, float(radius)]]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        z = float(radius * np.cos(phi))
        r_xy = float(radius * np.sin(phi))
        for segment in range(segments):
            theta = 2.0 * np.pi * segment / segments
            vertices.append([r_xy * np.cos(theta), r_xy * np.sin(theta), z])
    bottom_index = len(vertices)
    vertices.append([0.0, 0.0, -float(radius)])

    triangles: list[list[int]] = []
    first_ring = 1
    for segment in range(segments):
        nxt = (segment + 1) % segments
        triangles.append([0, first_ring + segment, first_ring + nxt])

    ring_count = rings - 1
    for ring in range(ring_count - 1):
        start = first_ring + ring * segments
        next_start = start + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            a = start + segment
            b = start + nxt
            c = next_start + segment
            d = next_start + nxt
            triangles.append([a, c, b])
            triangles.append([b, c, d])

    last_ring = first_ring + (ring_count - 1) * segments
    for segment in range(segments):
        nxt = (segment + 1) % segments
        triangles.append([bottom_index, last_ring + nxt, last_ring + segment])

    vertices_arr = np.asarray(vertices, dtype=float)
    normals = vertices_arr / np.linalg.norm(vertices_arr, axis=1)[:, None]
    return MeshPayload(
        vertices=vertices_arr,
        triangles=np.asarray(triangles, dtype=np.int32),
        normals=normals,
        vertex_colors=_vertex_colors(len(vertices_arr), color),
    )


def make_box_payload(
    width: float,
    height: float,
    depth: float,
    color: Optional[Iterable[float]] = None,
) -> MeshPayload:
    """Create an axis-aligned box payload anchored at the origin."""
    if width <= 0.0 or height <= 0.0 or depth <= 0.0:
        raise ValueError("Box dimensions must be positive")

    w, h, d = float(width), float(height), float(depth)
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [w, 0.0, 0.0],
            [w, h, 0.0],
            [0.0, h, 0.0],
            [0.0, 0.0, d],
            [w, 0.0, d],
            [w, h, d],
            [0.0, h, d],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return MeshPayload(
        vertices=vertices,
        triangles=triangles,
        normals=_compute_vertex_normals(vertices, triangles),
        vertex_colors=_vertex_colors(len(vertices), color),
    )


def load_mesh_payload(path: str | Path) -> MeshPayload:
    """Load an OBJ or PLY mesh file into a renderer-neutral payload."""
    mesh_path = Path(path)
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        return _load_obj_payload(mesh_path)
    if suffix == ".ply":
        return _load_ply_payload(mesh_path)

    with mesh_path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip().lower()
    if first_line == "ply":
        return _load_ply_payload(mesh_path)
    return _load_obj_payload(mesh_path)


def _parse_obj_index(raw: str, count: int) -> int:
    """Convert OBJ positive or relative indices into zero-based indices."""
    index = int(raw)
    if index > 0:
        return index - 1
    if index < 0:
        return count + index
    raise ValueError("OBJ indices are 1-based; index 0 is invalid")


def _load_obj_payload(path: Path) -> MeshPayload:
    """Load OBJ vertices, normals, colors, and polygon faces into a mesh payload."""
    vertices: list[list[float]] = []
    vertex_colors: list[list[float]] = []
    texcoords_src: list[list[float]] = []
    normals_src: list[list[float]] = []
    normals_by_vertex: dict[int, np.ndarray] = {}
    triangles: list[list[int]] = []
    triangle_uvs: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    vertex_colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
            elif parts[0] == "vt" and len(parts) >= 3:
                texcoords_src.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "vn" and len(parts) >= 4:
                normals_src.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                face_vertices: list[int] = []
                face_uvs: list[Optional[int]] = []
                face_normals: list[Optional[int]] = []
                for token in parts[1:]:
                    fields = token.split("/")
                    vertex_index = _parse_obj_index(fields[0], len(vertices))
                    uv_index = None
                    if len(fields) >= 2 and fields[1]:
                        uv_index = _parse_obj_index(fields[1], len(texcoords_src))
                    normal_index = None
                    if len(fields) >= 3 and fields[2]:
                        normal_index = _parse_obj_index(fields[2], len(normals_src))
                    face_vertices.append(vertex_index)
                    face_uvs.append(uv_index)
                    face_normals.append(normal_index)
                for offset in range(1, len(face_vertices) - 1):
                    tri = [face_vertices[0], face_vertices[offset], face_vertices[offset + 1]]
                    triangles.append(tri)
                    tri_uvs = [face_uvs[0], face_uvs[offset], face_uvs[offset + 1]]
                    if all(uv_index is not None for uv_index in tri_uvs):
                        triangle_uvs.extend([texcoords_src[int(uv_index)] for uv_index in tri_uvs])
                    for vertex_index, normal_index in zip(
                        tri,
                        [face_normals[0], face_normals[offset], face_normals[offset + 1]],
                    ):
                        if normal_index is not None and vertex_index not in normals_by_vertex:
                            normals_by_vertex[vertex_index] = np.asarray(
                                normals_src[normal_index], dtype=float
                            )

    vertices_arr = np.asarray(vertices, dtype=float)
    triangles_arr = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    if vertices_arr.size == 0:
        raise ValueError(f"OBJ mesh has no vertices: {path}")

    normals = None
    if len(normals_by_vertex) == len(vertices_arr):
        normals = np.zeros_like(vertices_arr)
        for index, normal in normals_by_vertex.items():
            normals[index] = normal
    if normals is None:
        normals = _compute_vertex_normals(vertices_arr, triangles_arr)

    colors = None
    if len(vertex_colors) == len(vertices_arr):
        colors = np.asarray(vertex_colors, dtype=float)
        if np.any(colors > 1.0):
            colors = colors / 255.0

    return MeshPayload(
        vertices=vertices_arr,
        triangles=triangles_arr,
        normals=normals,
        vertex_colors=colors,
        triangle_uvs=(
            np.asarray(triangle_uvs, dtype=float)
            if len(triangle_uvs) == len(triangles_arr) * 3
            else None
        ),
        cache_key=str(path),
    )


_PLY_NUMPY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_PLY_STRUCT_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _read_ply_header(path: Path) -> tuple[list[str], int]:
    """Read a PLY header and return the byte offset of the payload body."""
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header is missing end_header: {path}")
            stripped = line.decode("ascii", errors="replace").strip()
            header.append(stripped)
            if stripped == "end_header":
                return header, handle.tell()


def _parse_ply_header(header: list[str]) -> tuple[str, list[dict[str, object]]]:
    """Parse PLY format and element/property descriptors from header lines."""
    ply_format = ""
    elements: list[dict[str, object]] = []
    current: Optional[dict[str, object]] = None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            ply_format = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            properties = current["properties"]
            if not isinstance(properties, list):
                continue
            if len(parts) >= 5 and parts[1] == "list":
                properties.append(
                    {
                        "kind": "list",
                        "count_type": parts[2],
                        "item_type": parts[3],
                        "name": parts[4],
                    }
                )
            elif len(parts) >= 3:
                properties.append({"kind": "scalar", "type": parts[1], "name": parts[2]})
    return ply_format, elements


def _load_ply_payload(path: Path) -> MeshPayload:
    """Dispatch PLY loading by declared ASCII or binary endianness."""
    header, data_start = _read_ply_header(path)
    ply_format, elements = _parse_ply_header(header)
    if ply_format == "ascii":
        return _load_ascii_ply_payload(path)
    if ply_format == "binary_little_endian":
        return _load_binary_ply_payload(path, elements, data_start, "<")
    if ply_format == "binary_big_endian":
        return _load_binary_ply_payload(path, elements, data_start, ">")
    raise ValueError(f"Unsupported PLY mesh format '{ply_format or 'unknown'}': {path}")


def _ply_numpy_dtype(type_name: str, endian: str) -> np.dtype:
    """Return a NumPy dtype for one scalar PLY property type."""
    dtype_code = _PLY_NUMPY_DTYPES.get(type_name)
    if dtype_code is None:
        raise ValueError(f"Unsupported PLY property type: {type_name}")
    if dtype_code.endswith("1"):
        return np.dtype(dtype_code)
    return np.dtype(f"{endian}{dtype_code}")


def _ply_struct_format(type_name: str, endian: str) -> str:
    """Return a ``struct`` format string for one scalar PLY property type."""
    format_code = _PLY_STRUCT_FORMATS.get(type_name)
    if format_code is None:
        raise ValueError(f"Unsupported PLY property type: {type_name}")
    return f"{endian}{format_code}"


def _read_binary_scalar(handle, type_name: str, endian: str) -> int | float:
    """Read one scalar value from a binary PLY stream."""
    fmt = _ply_struct_format(type_name, endian)
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unexpected EOF while reading binary PLY data")
    return struct.unpack(fmt, data)[0]


def _skip_binary_scalar(handle, type_name: str, endian: str) -> None:
    """Advance past one scalar value in a binary PLY stream."""
    fmt = _ply_struct_format(type_name, endian)
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unexpected EOF while skipping binary PLY data")


def _read_binary_list(handle, count_type: str, item_type: str, endian: str) -> np.ndarray:
    """Read one counted list property from a binary PLY stream."""
    count = int(_read_binary_scalar(handle, count_type, endian))
    dtype = _ply_numpy_dtype(item_type, endian)
    byte_count = int(count) * dtype.itemsize
    data = handle.read(byte_count)
    if len(data) != byte_count:
        raise ValueError("Unexpected EOF while reading binary PLY list data")
    return np.frombuffer(data, dtype=dtype, count=count)


def _read_fixed_triangle_face_block(
    handle,
    *,
    path: Path,
    properties: list[object],
    count: int,
    endian: str,
) -> Optional[np.ndarray]:
    """Vectorized fast path for binary PLY triangle face blocks."""
    if len(properties) != 1:
        return None
    prop = properties[0]
    if not isinstance(prop, dict) or prop.get("kind") != "list":
        return None
    if str(prop.get("name", "")) not in {"vertex_indices", "vertex_index"}:
        return None

    count_type = str(prop.get("count_type", ""))
    item_type = str(prop.get("item_type", ""))
    count_dtype = _ply_numpy_dtype(count_type, endian)
    item_dtype = _ply_numpy_dtype(item_type, endian)
    record_dtype = np.dtype([("count", count_dtype), ("indices", item_dtype, (3,))])
    byte_count = record_dtype.itemsize * int(count)
    start = handle.tell()
    try:
        remaining = path.stat().st_size - start
    except OSError:
        remaining = byte_count
    if remaining < byte_count:
        return None

    raw = handle.read(byte_count)
    if len(raw) != byte_count:
        handle.seek(start)
        return None

    records = np.frombuffer(raw, dtype=record_dtype, count=int(count))
    if not np.all(records["count"] == 3):
        handle.seek(start)
        return None
    return records["indices"].astype(np.int32, copy=False)


def _load_binary_ply_payload(
    path: Path,
    elements: list[dict[str, object]],
    data_start: int,
    endian: str,
) -> MeshPayload:
    """Load a binary PLY mesh while preserving optional normals and colors."""
    vertices_arr: Optional[np.ndarray] = None
    normals_arr: Optional[np.ndarray] = None
    colors_arr: Optional[np.ndarray] = None
    triangles: list[list[int]] = []
    triangles_arr: Optional[np.ndarray] = None

    with path.open("rb") as handle:
        handle.seek(data_start)
        for element in elements:
            name = str(element.get("name", ""))
            count = int(element.get("count", 0))
            properties = element.get("properties", [])
            if not isinstance(properties, list):
                continue

            if name == "vertex":
                if any(prop.get("kind") == "list" for prop in properties if isinstance(prop, dict)):
                    raise ValueError(f"Binary PLY vertex list properties are unsupported: {path}")
                dtype_fields = []
                for index, prop in enumerate(properties):
                    if not isinstance(prop, dict):
                        continue
                    prop_name = str(prop.get("name", f"prop_{index}"))
                    prop_type = str(prop.get("type", ""))
                    dtype_fields.append((prop_name, _ply_numpy_dtype(prop_type, endian)))
                dtype = np.dtype(dtype_fields)
                raw = handle.read(dtype.itemsize * count)
                if len(raw) != dtype.itemsize * count:
                    raise ValueError(f"Unexpected EOF while reading binary PLY vertices: {path}")
                vertex_data = np.frombuffer(raw, dtype=dtype, count=count)
                required = {"x", "y", "z"}
                if not required.issubset(vertex_data.dtype.names or ()):
                    raise ValueError(f"PLY vertex properties must include x, y, z: {path}")
                vertices_arr = np.column_stack(
                    [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
                ).astype(float, copy=False)
                if {"nx", "ny", "nz"}.issubset(vertex_data.dtype.names or ()):
                    normals_arr = np.column_stack(
                        [vertex_data["nx"], vertex_data["ny"], vertex_data["nz"]]
                    ).astype(float, copy=False)
                if {"red", "green", "blue"}.issubset(vertex_data.dtype.names or ()):
                    colors_arr = np.column_stack(
                        [vertex_data["red"], vertex_data["green"], vertex_data["blue"]]
                    ).astype(float, copy=False)
            elif name == "face":
                fixed_triangles = _read_fixed_triangle_face_block(
                    handle,
                    path=path,
                    properties=properties,
                    count=count,
                    endian=endian,
                )
                if fixed_triangles is not None:
                    triangles_arr = fixed_triangles
                    continue

                for _ in range(count):
                    face_indices: Optional[np.ndarray] = None
                    for prop in properties:
                        if not isinstance(prop, dict):
                            continue
                        if prop.get("kind") == "list":
                            values = _read_binary_list(
                                handle,
                                str(prop.get("count_type", "")),
                                str(prop.get("item_type", "")),
                                endian,
                            )
                            if face_indices is None and str(prop.get("name", "")) in {
                                "vertex_indices",
                                "vertex_index",
                            }:
                                face_indices = values.astype(np.int64, copy=False)
                        else:
                            _skip_binary_scalar(handle, str(prop.get("type", "")), endian)
                    if face_indices is None:
                        continue
                    indices = [int(value) for value in face_indices]
                    for offset in range(1, len(indices) - 1):
                        triangles.append([indices[0], indices[offset], indices[offset + 1]])
            else:
                _skip_binary_ply_element(handle, properties, count, endian)

    if vertices_arr is None or vertices_arr.size == 0:
        raise ValueError(f"PLY mesh has no vertices: {path}")

    if triangles_arr is None:
        triangles_arr = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    if normals_arr is None:
        normals_arr = _compute_vertex_normals(vertices_arr, triangles_arr)
    if colors_arr is not None and np.any(colors_arr > 1.0):
        colors_arr = colors_arr / 255.0

    return MeshPayload(
        vertices=vertices_arr,
        triangles=triangles_arr,
        normals=normals_arr,
        vertex_colors=colors_arr,
        cache_key=str(path),
    )


def _skip_binary_ply_element(
    handle,
    properties: list[object],
    count: int,
    endian: str,
) -> None:
    """Skip non-mesh PLY elements while consuming scalar and list payloads."""
    for _ in range(count):
        for prop in properties:
            if not isinstance(prop, dict):
                continue
            if prop.get("kind") == "list":
                values = _read_binary_list(
                    handle,
                    str(prop.get("count_type", "")),
                    str(prop.get("item_type", "")),
                    endian,
                )
                # Force eager consumption of the view for clarity; the bytes
                # have already been read, so no further action is needed.
                _ = values
            else:
                _skip_binary_scalar(handle, str(prop.get("type", "")), endian)


def _load_ascii_ply_payload(path: Path) -> MeshPayload:
    """Load an ASCII PLY mesh with optional normals and RGB vertex colors."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header: list[str] = []
        for line in handle:
            stripped = line.strip()
            header.append(stripped)
            if stripped == "end_header":
                break
        else:
            raise ValueError(f"PLY header is missing end_header: {path}")

        if not any(line == "format ascii 1.0" for line in header):
            raise ValueError(f"Only ASCII PLY meshes are supported by neutral loader: {path}")

        vertex_count = 0
        face_count = 0
        vertex_props: list[str] = []
        current_element: Optional[str] = None
        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element" and len(parts) >= 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                elif current_element == "face":
                    face_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex" and len(parts) >= 3:
                vertex_props.append(parts[-1])

        prop_index = {name: index for index, name in enumerate(vertex_props)}
        required = {"x", "y", "z"}
        if not required.issubset(prop_index):
            raise ValueError(f"PLY vertex properties must include x, y, z: {path}")

        vertices: list[list[float]] = []
        normals: list[list[float]] = []
        colors: list[list[float]] = []
        has_normals = {"nx", "ny", "nz"}.issubset(prop_index)
        has_colors = {"red", "green", "blue"}.issubset(prop_index)

        for _ in range(vertex_count):
            values = handle.readline().split()
            vertices.append(
                [
                    float(values[prop_index["x"]]),
                    float(values[prop_index["y"]]),
                    float(values[prop_index["z"]]),
                ]
            )
            if has_normals:
                normals.append(
                    [
                        float(values[prop_index["nx"]]),
                        float(values[prop_index["ny"]]),
                        float(values[prop_index["nz"]]),
                    ]
                )
            if has_colors:
                colors.append(
                    [
                        float(values[prop_index["red"]]),
                        float(values[prop_index["green"]]),
                        float(values[prop_index["blue"]]),
                    ]
                )

        triangles: list[list[int]] = []
        for _ in range(face_count):
            values = handle.readline().split()
            if not values:
                continue
            count = int(values[0])
            indices = [int(value) for value in values[1 : 1 + count]]
            for offset in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[offset], indices[offset + 1]])

    vertices_arr = np.asarray(vertices, dtype=float)
    triangles_arr = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    if vertices_arr.size == 0:
        raise ValueError(f"PLY mesh has no vertices: {path}")

    normals_arr = np.asarray(normals, dtype=float) if len(normals) == len(vertices_arr) else None
    if normals_arr is None:
        normals_arr = _compute_vertex_normals(vertices_arr, triangles_arr)

    colors_arr = np.asarray(colors, dtype=float) if len(colors) == len(vertices_arr) else None
    if colors_arr is not None and np.any(colors_arr > 1.0):
        colors_arr = colors_arr / 255.0

    return MeshPayload(
        vertices=vertices_arr,
        triangles=triangles_arr,
        normals=normals_arr,
        vertex_colors=colors_arr,
        cache_key=str(path),
    )


def make_lines_payload(
    points: np.ndarray,
    lines: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> LineSetPayload:
    """Create LineSetPayload from numpy arrays."""
    return LineSetPayload(
        points=np.asarray(points, dtype=float),
        lines=np.asarray(lines, dtype=np.int32),
        colors=None if colors is None else np.asarray(colors, dtype=float),
    )


def make_pointcloud_payload(
    points: np.ndarray, colors: Optional[np.ndarray] = None
) -> PointCloudPayload:
    """Create PointCloudPayload from numpy arrays."""
    return PointCloudPayload(
        points=np.asarray(points, dtype=float),
        colors=None if colors is None else np.asarray(colors, dtype=float),
    )


def extract_wireframe_payload(mesh_payload: MeshPayload) -> LineSetPayload:
    """Extract unique mesh edges as a LineSetPayload."""
    tris = np.asarray(mesh_payload.triangles, dtype=np.int32)
    if tris.size == 0:
        return LineSetPayload(
            points=np.asarray(mesh_payload.vertices, dtype=float),
            lines=np.empty((0, 2), dtype=np.int32),
        )
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return LineSetPayload(
        points=np.asarray(mesh_payload.vertices, dtype=float),
        lines=edges.astype(np.int32, copy=False),
    )


def merge_mesh_payloads(payloads: list[MeshPayload]) -> MeshPayload:
    """Merge multiple MeshPayloads into one, offsetting triangle indices."""
    if not payloads:
        return MeshPayload(
            vertices=np.empty((0, 3), dtype=float),
            triangles=np.empty((0, 3), dtype=np.int32),
        )

    all_verts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    offset = 0
    for payload in payloads:
        vertices = np.asarray(payload.vertices, dtype=float)
        triangles = np.asarray(payload.triangles, dtype=np.int32)
        all_verts.append(vertices)
        all_tris.append(triangles + offset)
        offset += len(vertices)

    normals = None
    if all(payload.normals is not None for payload in payloads):
        normals = np.vstack([np.asarray(payload.normals, dtype=float) for payload in payloads])

    colors = None
    if all(payload.vertex_colors is not None for payload in payloads):
        colors = np.vstack([np.asarray(payload.vertex_colors, dtype=float) for payload in payloads])

    uvs = None
    if all(payload.triangle_uvs is not None for payload in payloads):
        uvs = np.vstack([np.asarray(payload.triangle_uvs, dtype=float) for payload in payloads])

    return MeshPayload(
        vertices=np.vstack(all_verts),
        triangles=np.vstack(all_tris),
        normals=normals,
        vertex_colors=colors,
        triangle_uvs=uvs,
        color_source=(
            payloads[0].color_source
            if all(payload.color_source is payloads[0].color_source for payload in payloads)
            else SurfaceColorSource.MATERIAL
        ),
    )
