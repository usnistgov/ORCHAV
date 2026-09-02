"""Deterministic visualizer debug inventory helpers.

The helpers here collect renderer, scene, camera, target, node, and visual-state
snapshots without mutating the live visualizer. Debug scripts use the payloads
to compare frame captures, named-geometry registries, and object lifecycle
state across renderer backends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..services.object_identity import make_node_geometry_name, make_target_entry_geometry_name
from ..types.camera_state import CameraState

_ORIGIN_EPSILON = 1e-3


def _as_vec3(value: Any) -> Optional[np.ndarray]:
    """Coerce a value into a 3D float vector when possible."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3:
        return None
    return arr[:3]


def _extract_bounds(bounds: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extract min/max scene bounds from Open3D-style or dataclass objects."""
    if bounds is None:
        return None
    if hasattr(bounds, "get_min_bound") and hasattr(bounds, "get_max_bound"):
        min_bound = _as_vec3(bounds.get_min_bound())
        max_bound = _as_vec3(bounds.get_max_bound())
    else:
        min_bound = _as_vec3(getattr(bounds, "min_bound", None))
        max_bound = _as_vec3(getattr(bounds, "max_bound", None))
    if min_bound is None or max_bound is None:
        return None
    return min_bound, max_bound


def deterministic_camera_from_bounds(
    bounds: Any,
    *,
    preset: str = "auto_iso",
    fov_deg: float = 55.0,
) -> Optional[CameraState]:
    """Return a stable camera framing for a scene bounding box."""
    extracted = _extract_bounds(bounds)
    if extracted is None:
        return None
    min_bound, max_bound = extracted
    center = 0.5 * (min_bound + max_bound)
    extent = np.maximum(max_bound - min_bound, 1e-3)
    radius = float(max(extent.max(), 1.0))

    preset_norm = str(preset).strip().lower()
    if preset_norm == "topdown":
        eye = center + np.array([0.0, 0.0, 3.2 * radius], dtype=float)
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        eye = center + np.array([-1.8, -1.4, 1.1], dtype=float) * radius
        up = np.array([0.0, 0.0, 1.0], dtype=float)

    return CameraState(
        eye=(float(eye[0]), float(eye[1]), float(eye[2])),
        lookat=(float(center[0]), float(center[1]), float(center[2])),
        up=(float(up[0]), float(up[1]), float(up[2])),
        fov_deg=float(fov_deg),
    )


def deterministic_camera_from_points(
    points: list[Any],
    *,
    preset: str = "target_focus",
    fov_deg: float = 50.0,
) -> Optional[CameraState]:
    """Return a stable camera framing for selected focus points."""
    pts = []
    for point in points:
        arr = _as_vec3(point)
        if arr is not None:
            pts.append(arr)
    if not pts:
        return None
    pts_arr = np.asarray(pts, dtype=float)
    min_bound = np.min(pts_arr, axis=0)
    max_bound = np.max(pts_arr, axis=0)
    center = 0.5 * (min_bound + max_bound)
    extent = np.maximum(max_bound - min_bound, 1e-3)
    radius = float(max(extent.max(), 2.0))

    preset_norm = str(preset).strip().lower()
    if preset_norm == "origin_probe":
        center = np.zeros(3, dtype=float)
        radius = max(radius, 4.0)
        eye = center + np.array([-1.2, -1.2, 0.9], dtype=float) * radius
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    elif preset_norm == "topdown":
        eye = center + np.array([0.0, 0.0, 2.4 * radius], dtype=float)
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        eye = center + np.array([-1.15, -0.95, 0.75], dtype=float) * radius
        up = np.array([0.0, 0.0, 1.0], dtype=float)

    return CameraState(
        eye=(float(eye[0]), float(eye[1]), float(eye[2])),
        lookat=(float(center[0]), float(center[1]), float(center[2])),
        up=(float(up[0]), float(up[1]), float(up[2])),
        fov_deg=float(fov_deg),
    )


def collect_focus_points(viz: Any, *, scope: str = "all") -> list[np.ndarray]:
    """Collect target and optional node positions for deterministic framing."""
    renderer = viz.renderer
    points: list[np.ndarray] = []

    for entry in getattr(viz, "target_entries", []):
        geometry_name = make_target_entry_geometry_name(entry, "mesh")
        named_pos = _renderer_named_position(renderer, geometry_name)
        arr = _as_vec3(named_pos if named_pos is not None else entry.get("position"))
        if arr is not None:
            points.append(arr)

    if str(scope).strip().lower() == "targets":
        return points

    for kind, markers in (
        ("tx", getattr(viz, "tx_markers", [])),
        ("rx", getattr(viz, "rx_markers", [])),
    ):
        for idx, marker in enumerate(markers):
            geometry_name = make_node_geometry_name(kind, idx, "marker")
            named_pos = _renderer_named_position(renderer, geometry_name)
            arr = _as_vec3(named_pos)
            if arr is None and hasattr(marker, "get_center"):
                try:
                    arr = _as_vec3(marker.get_center())
                except Exception:
                    arr = None
            if arr is not None:
                points.append(arr)

    return points


def load_camera_state(path: str | Path) -> CameraState:
    """Load a serialized camera state JSON file."""
    payload = json.loads(Path(path).read_text())
    try:
        return CameraState.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid camera state file: {path}") from exc


def _named_geometry_inventory(renderer: Any) -> dict[str, Any]:
    """Summarize renderer named-geometry registries and hidden objects."""
    names = set(_renderer_named_geometry_names(renderer))
    hidden = {
        name
        for name in names
        if getattr(renderer, "is_named_visible", lambda _name: None)(name) is False
    }

    external_names = sorted(
        {
            str(name)
            for name in getattr(renderer, "_external_geometry_names", {}).values()
            if isinstance(name, str)
        }
    )
    target_names = sorted(name for name in names if name.startswith("target:"))
    node_names = sorted(name for name in names if name.startswith("node:"))
    scene_count = sum(1 for name in names if name.startswith("scene:"))

    other_names = sorted(
        name
        for name in names
        if not name.startswith("target:")
        and not name.startswith("node:")
        and not name.startswith("scene:")
    )
    return {
        "named_count": len(names),
        "scene_named_count": scene_count,
        "target_named_names": target_names,
        "node_named_names": node_names,
        "other_named_names": other_names,
        "hidden_names": sorted(hidden),
        "external_geom_names": external_names,
    }


def _renderer_named_position(renderer: Any, name: str) -> Optional[list[float]]:
    """Return a named geometry position from renderer APIs, if available."""
    if not hasattr(renderer, "get_named_position"):
        return None
    pos = renderer.get_named_position(name)
    arr = _as_vec3(pos)
    if arr is None:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _renderer_has_named_geometry(renderer: Any, name: str) -> bool:
    """Return whether a renderer registry contains a named geometry."""
    fn = getattr(renderer, "has_named_geometry", None)
    if callable(fn):
        try:
            return bool(fn(name))
        except Exception:
            return False
    return False


def _renderer_named_geometry_names(renderer: Any) -> tuple[str, ...]:
    """Return renderer-owned names through the public inspection contract."""
    fn = getattr(renderer, "get_named_geometry_names", None)
    if not callable(fn):
        return ()
    try:
        return tuple(str(name) for name in fn())
    except Exception:
        return ()


def _external_name_for_geometry(renderer: Any, geometry: Any) -> Optional[str]:
    """Return the renderer's object-id based name for an external geometry."""
    fn = getattr(renderer, "_external_name_for_geometry", None)
    if callable(fn):
        try:
            name = fn(geometry)
        except Exception:
            return None
        return None if name is None else str(name)
    mapping = getattr(renderer, "_external_geometry_names", None)
    if isinstance(mapping, dict):
        name = mapping.get(id(geometry))
        return None if name is None else str(name)
    return None


def _true_external_duplicate_name(renderer: Any, geometry: Any) -> Optional[str]:
    """Return active ``external_geom_*`` duplicates for a geometry object."""
    name = _external_name_for_geometry(renderer, geometry)
    if not name or not str(name).startswith("external_geom_"):
        return None
    return str(name)


def _named_alias_duplicate_name(renderer: Any, geometry: Any, stable_name: str) -> Optional[str]:
    """Return object-identity aliases that coexist with a stable name."""
    fallback_name = f"geometry_{id(geometry)}"
    if fallback_name == stable_name:
        return None
    if not _renderer_has_named_geometry(renderer, stable_name):
        return None
    if not _renderer_has_named_geometry(renderer, fallback_name):
        return None
    return fallback_name


def _position_from_entry(entry: dict[str, Any]) -> Optional[list[float]]:
    """Return the first recognized position vector from an entry dictionary."""
    for key in ("position", "pos", "center"):
        arr = _as_vec3(entry.get(key))
        if arr is not None:
            return [float(arr[0]), float(arr[1]), float(arr[2])]
    return None


def _target_summary(viz: Any, renderer: Any) -> list[dict[str, Any]]:
    """Summarize target entries, renderer names, transforms, and materials."""
    targets: list[dict[str, Any]] = []
    # Cache named-object inspections to report mesh world position and
    # bounding-box extent alongside the transform-translation position.  When
    # the transform translation is correct but the mesh is visually invisible,
    # the likely cause is collapsed vertex buffers or an off-scene world pose.
    objects = getattr(renderer, "_objects", {})
    transforms = getattr(renderer, "_transforms", {})
    materials = getattr(renderer, "_materials", {})
    hidden = getattr(renderer, "_hidden", set())
    for entry in getattr(viz, "target_entries", []):
        geometry_name = make_target_entry_geometry_name(entry, "mesh")
        label_name = make_target_entry_geometry_name(entry, "label")
        mesh = entry.get("mesh")
        external_name = _true_external_duplicate_name(renderer, mesh)
        named_alias_name = _named_alias_duplicate_name(renderer, mesh, geometry_name)
        named_position = _renderer_named_position(renderer, geometry_name)
        position = _position_from_entry(entry)
        origin_visible = bool(
            named_position is not None
            and np.linalg.norm(np.asarray(named_position, dtype=float)) <= _ORIGIN_EPSILON
        )
        # Inspect the actual pygfx object
        pygfx_obj_info: dict[str, Any] = {}
        obj = objects.get(geometry_name) if isinstance(objects, dict) else None
        if obj is not None:
            obj_info: dict[str, Any] = {"type": type(obj).__name__}
            geom = getattr(obj, "geometry", None)
            if geom is not None:
                obj_info["has_geometry"] = True
                positions_buf = getattr(geom, "positions", None)
                if positions_buf is not None and hasattr(positions_buf, "data"):
                    data = positions_buf.data
                    if data is not None:
                        try:
                            arr = np.asarray(data)
                            obj_info["n_vertices"] = int(arr.shape[0])
                            if arr.size > 0:
                                obj_info["local_min"] = [float(v) for v in arr.min(axis=0)]
                                obj_info["local_max"] = [float(v) for v in arr.max(axis=0)]
                        except Exception:
                            pass
                texcoords_buf = getattr(geom, "texcoords", None)
                if texcoords_buf is not None and hasattr(texcoords_buf, "data"):
                    try:
                        texcoords_arr = np.asarray(texcoords_buf.data)
                        obj_info["texcoords_shape"] = [int(v) for v in texcoords_arr.shape]
                    except Exception:
                        pass
            world = getattr(obj, "world", None)
            if world is not None:
                pos_val = getattr(world, "position", None)
                if pos_val is not None:
                    try:
                        obj_info["world_position"] = [float(v) for v in pos_val]
                    except Exception:
                        pass
            local = getattr(obj, "local", None)
            if local is not None:
                scale_val = getattr(local, "scale", None)
                if scale_val is not None:
                    try:
                        obj_info["local_scale"] = [float(v) for v in scale_val]
                    except Exception:
                        pass
            obj_info["visible"] = bool(getattr(obj, "visible", True))
            pygfx_obj_info = obj_info
        material_info: dict[str, Any] = {}
        mat_payload = materials.get(geometry_name) if isinstance(materials, dict) else None
        if mat_payload is not None:
            try:
                material_info = {
                    "base_color": getattr(mat_payload, "base_color", None),
                    "texture_path": getattr(mat_payload, "texture_path", None),
                    "normal_map_path": getattr(mat_payload, "normal_map_path", None),
                    "roughness_map_path": getattr(mat_payload, "roughness_map_path", None),
                    "ao_map_path": getattr(mat_payload, "ao_map_path", None),
                    "metallic_map_path": getattr(mat_payload, "metallic_map_path", None),
                }
            except Exception:
                pass
        if obj is not None:
            obj_mat = getattr(obj, "material", None)
            if obj_mat is not None:
                object_material: dict[str, Any] = {"type": type(obj_mat).__name__}
                for attr in (
                    "color_mode",
                    "color",
                    "opacity",
                    "roughness",
                    "metalness",
                    "metallic",
                ):
                    try:
                        value = getattr(obj_mat, attr)
                    except Exception:
                        continue
                    try:
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        elif not isinstance(value, (str, int, float, bool, type(None))):
                            value = list(value)
                    except Exception:
                        value = str(value)
                    object_material[attr] = value
                for attr in (
                    "map",
                    "normal_map",
                    "roughness_map",
                    "ao_map",
                    "metalness_map",
                ):
                    try:
                        value = getattr(obj_mat, attr)
                    except Exception:
                        continue
                    object_material[f"has_{attr}"] = value is not None
                    texture = getattr(value, "texture", None)
                    if texture is not None:
                        object_material[f"{attr}_texture_type"] = type(texture).__name__
                material_info["object_material"] = object_material
        hidden_flag = geometry_name in hidden if isinstance(hidden, set) else False
        transform_info = None
        if isinstance(transforms, dict) and geometry_name in transforms:
            try:
                tmat = np.asarray(transforms[geometry_name])
                transform_info = {
                    "translation": [float(tmat[0, 3]), float(tmat[1, 3]), float(tmat[2, 3])],
                    "det": float(np.linalg.det(tmat[:3, :3])),
                }
            except Exception:
                transform_info = None
        targets.append(
            {
                "target_name": entry.get("target_name") or entry.get("name"),
                "object_id": entry.get("object_id"),
                "geometry_name": geometry_name,
                "label_name": label_name,
                "frame_visible": bool(entry.get("_frame_visible", True)),
                "named_exists": bool(
                    getattr(renderer, "has_named_geometry", lambda _n: False)(geometry_name)
                ),
                "named_visible": getattr(renderer, "is_named_visible", lambda _n: None)(
                    geometry_name
                ),
                "named_position": named_position,
                "entry_position": position,
                "external_name": external_name,
                "named_alias_name": named_alias_name,
                "origin_named_visible": origin_visible,
                "pygfx_object": pygfx_obj_info,
                "material": material_info,
                "hidden_flag": hidden_flag,
                "transform": transform_info,
            }
        )
    return targets


def _node_summary(viz: Any, renderer: Any, kind: str) -> list[dict[str, Any]]:
    """Summarize TX or RX marker naming, visibility, and positions."""
    markers = getattr(viz, f"{kind}_markers", [])
    summary: list[dict[str, Any]] = []
    for idx, marker in enumerate(markers):
        geometry_name = make_node_geometry_name(kind, idx, "marker")
        label_name = make_node_geometry_name(kind, idx, "label")
        named_position = _renderer_named_position(renderer, geometry_name)
        center = None
        if hasattr(marker, "get_center"):
            try:
                center_arr = _as_vec3(marker.get_center())
            except Exception:
                center_arr = None
            if center_arr is not None:
                center = [float(center_arr[0]), float(center_arr[1]), float(center_arr[2])]
        external_name = _true_external_duplicate_name(renderer, marker)
        named_alias_name = _named_alias_duplicate_name(renderer, marker, geometry_name)
        origin_visible = bool(
            named_position is not None
            and np.linalg.norm(np.asarray(named_position, dtype=float)) <= _ORIGIN_EPSILON
        )
        summary.append(
            {
                "index": idx,
                "geometry_name": geometry_name,
                "label_name": label_name,
                "named_exists": bool(
                    getattr(renderer, "has_named_geometry", lambda _n: False)(geometry_name)
                ),
                "named_visible": getattr(renderer, "is_named_visible", lambda _n: None)(
                    geometry_name
                ),
                "named_position": named_position,
                "marker_center": center,
                "external_name": external_name,
                "named_alias_name": named_alias_name,
                "origin_named_visible": origin_visible,
            }
        )
    return summary


def _pygfx_named_object_summary(renderer: Any, name: str) -> dict[str, Any]:
    """Return pygfx object, geometry-buffer, and material details for a name."""
    objects = getattr(renderer, "_objects", {})
    materials = getattr(renderer, "_materials", {})
    obj = objects.get(name) if isinstance(objects, dict) else None
    item: dict[str, Any] = {"geometry_name": name, "exists": obj is not None}
    if obj is None:
        return item
    item["object_type"] = type(obj).__name__
    item["visible"] = bool(getattr(obj, "visible", True))
    geom = getattr(obj, "geometry", None)
    if geom is not None:
        geom_info: dict[str, Any] = {}
        for attr in ("positions", "indices", "texcoords", "colors", "normals"):
            buf = getattr(geom, attr, None)
            if buf is None or not hasattr(buf, "data"):
                continue
            try:
                arr = np.asarray(buf.data)
                geom_info[f"{attr}_shape"] = [int(v) for v in arr.shape]
                if attr == "texcoords" and arr.size:
                    geom_info["texcoords_min"] = float(np.nanmin(arr))
                    geom_info["texcoords_max"] = float(np.nanmax(arr))
            except Exception:
                pass
        item["geometry"] = geom_info
    payload = materials.get(name) if isinstance(materials, dict) else None
    if payload is not None:
        item["payload_material"] = {
            "base_color": getattr(payload, "base_color", None),
            "texture_path": getattr(payload, "texture_path", None),
            "normal_map_path": getattr(payload, "normal_map_path", None),
            "roughness_map_path": getattr(payload, "roughness_map_path", None),
            "ao_map_path": getattr(payload, "ao_map_path", None),
            "metallic_map_path": getattr(payload, "metallic_map_path", None),
        }
    mat = getattr(obj, "material", None)
    if mat is not None:
        material_info: dict[str, Any] = {"type": type(mat).__name__}
        for attr in ("color_mode", "color", "opacity", "roughness", "metalness", "metallic"):
            try:
                value = getattr(mat, attr)
            except Exception:
                continue
            try:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    value = list(value)
            except Exception:
                value = str(value)
            material_info[attr] = value
        for attr in ("map", "normal_map", "roughness_map", "ao_map", "metalness_map"):
            try:
                value = getattr(mat, attr)
            except Exception:
                continue
            material_info[f"has_{attr}"] = value is not None
            texture = getattr(value, "texture", None)
            if texture is not None:
                material_info[f"{attr}_texture_type"] = type(texture).__name__
        item["object_material"] = material_info
    return item


def _scene_mesh_summary(renderer: Any) -> list[dict[str, Any]]:
    """Return summaries for renderer named scene meshes."""
    names: set[str] = set()
    for registry_name in ("_name_to_handle", "_named_geometry"):
        registry = getattr(renderer, registry_name, None)
        if isinstance(registry, dict):
            names.update(str(name) for name in registry.keys() if str(name).startswith("scene:"))
    return [_pygfx_named_object_summary(renderer, name) for name in sorted(names)]


def _unknown_named_geometries(renderer: Any) -> list[str]:
    """Return named geometries whose prefix is unexpected.

    The renderer's named-geometry registry should only hold ``target:*``,
    ``node:*``, ``scene:*``, or known overlay names (``mpc_lines``,
    ``mpc_points``, ``axes``, etc.).  Anything else — especially
    ``external_geom_<id>`` entries — means the renderer has accumulated a
    scene entity outside the stable naming contract.  These are classic
    ghosts: the old object was not cleaned up when the logical target or
    scene mesh moved to a new identity.
    """
    names: set[str] = set()
    if hasattr(renderer, "_name_to_handle"):
        names.update(str(n) for n in getattr(renderer, "_name_to_handle", {}).keys())
    if hasattr(renderer, "_named_geometry"):
        names.update(str(n) for n in getattr(renderer, "_named_geometry", {}).keys())

    known_prefixes = ("target:", "node:", "scene:")
    known_exact = {
        "mpc_lines",
        "mpc_points",
        "axes",
        "ground_grid",
        "coverage_mesh",
        "coverage_outline",
        "scene_edge_lines",
        "orchav_axes",
    }
    unknown: list[str] = []
    for name in sorted(names):
        if name.startswith(known_prefixes):
            continue
        if name in known_exact:
            continue
        # Filter out hud_* / marker_legend_* / similar well-known overlay
        # prefixes.  These are safe to list; add more as they appear.
        if (
            name.startswith("hud_")
            or name.startswith("mpc_hud_")
            or name.startswith("marker_legend_")
            or name.startswith("orientation_")
            or name.startswith("cutaway_")
            or name.startswith("sensing_")
            or name.startswith("beam_")
            or name.startswith("debug_")
        ):
            continue
        unknown.append(name)
    return unknown


def _orphan_target_names(viz: Any, renderer: Any) -> list[str]:
    """Return target:* named geometries not backed by any live target entry.

    For every ``target:*::mesh`` name in the renderer's named-geometry
    registry, confirm it corresponds to the stable name produced by one of
    the current ``viz.target_entries``.  Any name that does NOT match is an
    orphan — a stale scene entity left behind from a previous lifecycle.
    """
    expected_mesh_names: set[str] = set()
    expected_label_names: set[str] = set()
    expected_sphere_names: set[str] = set()
    expected_outline_names: set[str] = set()
    for entry in getattr(viz, "target_entries", []):
        expected_mesh_names.add(make_target_entry_geometry_name(entry, "mesh"))
        expected_label_names.add(make_target_entry_geometry_name(entry, "label"))
        expected_sphere_names.add(make_target_entry_geometry_name(entry, "sphere"))
        expected_outline_names.add(make_target_entry_geometry_name(entry, "outline"))
        extra = entry.get("_outline_geometry_name")
        if extra:
            expected_outline_names.add(str(extra))

    names: set[str] = set()
    if hasattr(renderer, "_name_to_handle"):
        names.update(str(n) for n in getattr(renderer, "_name_to_handle", {}).keys())
    if hasattr(renderer, "_named_geometry"):
        names.update(str(n) for n in getattr(renderer, "_named_geometry", {}).keys())

    orphans: list[str] = []
    for name in sorted(names):
        if not name.startswith("target:"):
            continue
        if (
            name in expected_mesh_names
            or name in expected_label_names
            or name in expected_sphere_names
            or name in expected_outline_names
        ):
            continue
        orphans.append(name)
    return orphans


def _orphan_node_names(viz: Any, renderer: Any) -> list[str]:
    """Return node:* named geometries not backed by the current tx/rx markers."""
    n_tx = len(getattr(viz, "tx_markers", []))
    n_rx = len(getattr(viz, "rx_markers", []))
    expected: set[str] = set()
    for kind, count in (("tx", n_tx), ("rx", n_rx)):
        for idx in range(count):
            expected.add(make_node_geometry_name(kind, idx, "marker"))
            expected.add(make_node_geometry_name(kind, idx, "label"))

    names: set[str] = set()
    if hasattr(renderer, "_name_to_handle"):
        names.update(str(n) for n in getattr(renderer, "_name_to_handle", {}).keys())
    if hasattr(renderer, "_named_geometry"):
        names.update(str(n) for n in getattr(renderer, "_named_geometry", {}).keys())

    orphans: list[str] = []
    for name in sorted(names):
        if not name.startswith("node:"):
            continue
        if name in expected:
            continue
        orphans.append(name)
    return orphans


def _stale_external_object_mappings(viz: Any, renderer: Any) -> list[dict[str, Any]]:
    """Find entries in _external_geometry_names whose Python id no longer matches
    any live target mesh, outline, or TX/RX marker object.

    A stale entry means the renderer thinks the named geometry belongs to a
    Python object the app has forgotten about.  That old object may still be
    referenced by someone else (for example, ``target_asset_cache``), and a
    later object-id based remove/update operation would act on the wrong entity.
    """
    mapping = getattr(renderer, "_external_geometry_names", None)
    if not isinstance(mapping, dict) or not mapping:
        return []

    live_ids: set[int] = set()
    for entry in getattr(viz, "target_entries", []):
        mesh = entry.get("mesh")
        if mesh is not None:
            live_ids.add(id(mesh))
        outline = entry.get("outline_geometry")
        if outline is not None:
            live_ids.add(id(outline))
    for label in getattr(viz, "target_labels", []):
        if label is not None:
            live_ids.add(id(label))
    for marker in getattr(viz, "tx_markers", []):
        live_ids.add(id(marker))
    for marker in getattr(viz, "rx_markers", []):
        live_ids.add(id(marker))
    for label in getattr(viz, "tx_labels", []):
        if label is not None:
            live_ids.add(id(label))
    for label in getattr(viz, "rx_labels", []):
        if label is not None:
            live_ids.add(id(label))

    stale: list[dict[str, Any]] = []
    for obj_id, name in mapping.items():
        name_str = str(name)
        # Only flag entries pointing at target/node names — scene meshes and
        # internal overlay objects legitimately reuse ids across frames.
        if not (name_str.startswith("target:") or name_str.startswith("node:")):
            continue
        if int(obj_id) not in live_ids:
            stale.append({"object_id": int(obj_id), "name": name_str})
    return stale


def _pygfx_scene_object_census(renderer: Any) -> Optional[dict[str, Any]]:
    """Walk the pygfx scene graph and count visible mesh/line/point objects.

    Returns None for non-pygfx renderers or when the scene is not yet built.
    When available, returns counts split by ``type(obj).__name__`` plus a
    cross-check against ``_name_to_handle``.
    """
    scene = getattr(renderer, "_scene", None)
    if scene is None or not hasattr(scene, "traverse"):
        return None

    counts_by_type: dict[str, int] = {}
    visible_by_type: dict[str, int] = {}
    tracked_object_ids: set[int] = set()

    for tracked in getattr(renderer, "_objects", {}).values():
        tracked_object_ids.add(id(tracked))

    untracked: list[str] = []

    def _visit(obj: Any) -> None:
        """Accumulate object counts while pygfx traverses the scene graph."""
        type_name = type(obj).__name__
        # Only count leaf render objects — skip Scene, Group, and similar.
        if type_name in {"Scene", "Group", "WorldObject"}:
            return
        counts_by_type[type_name] = counts_by_type.get(type_name, 0) + 1
        if getattr(obj, "visible", True):
            visible_by_type[type_name] = visible_by_type.get(type_name, 0) + 1
        if id(obj) not in tracked_object_ids:
            # Scene may hold lights, axes, etc. that are not named-geometry —
            # only flag the drawable types that usually come through named API.
            if type_name in {"Mesh", "Points", "Line"}:
                untracked.append(type_name)

    try:
        scene.traverse(_visit)
    except Exception:
        return None

    return {
        "counts_by_type": {k: counts_by_type[k] for k in sorted(counts_by_type)},
        "visible_by_type": {k: visible_by_type[k] for k in sorted(visible_by_type)},
        "named_handle_count": len(_renderer_named_geometry_names(renderer)),
        "untracked_drawable_count": len(untracked),
    }


def _renderer_state_summary(viz: Any) -> dict[str, Any]:
    """Summarize stable IDs owned by the active renderer."""
    renderer = getattr(viz, "renderer", None)
    if renderer is None:
        return {"object_count": 0, "target_keys": [], "node_keys": []}
    keys = sorted(_renderer_named_geometry_names(renderer))
    return {
        "object_count": len(keys),
        "target_keys": [key for key in keys if key.startswith("target:")],
        "node_keys": [key for key in keys if key.startswith("node:")],
    }


def collect_debug_inventory(viz: Any, *, label: str, step: Optional[int]) -> dict[str, Any]:
    """Collect a serializable diagnostic snapshot for one visualizer state."""
    renderer = viz.renderer
    bounds = getattr(renderer, "compute_scene_bounds", lambda: None)()
    bounds_pair = _extract_bounds(bounds)
    camera_state = None
    if hasattr(renderer, "get_camera_state"):
        candidate = renderer.get_camera_state()
        camera_state = candidate if isinstance(candidate, CameraState) else None

    targets = _target_summary(viz, renderer)
    tx_nodes = _node_summary(viz, renderer, "tx")
    rx_nodes = _node_summary(viz, renderer, "rx")
    named = _named_geometry_inventory(renderer)
    origin_targets = [item["geometry_name"] for item in targets if item["origin_named_visible"]]
    origin_nodes = [
        item["geometry_name"] for item in tx_nodes + rx_nodes if item["origin_named_visible"]
    ]
    target_external_dups = [item["external_name"] for item in targets if item["external_name"]]
    node_external_dups = [
        item["external_name"] for item in tx_nodes + rx_nodes if item["external_name"]
    ]
    target_named_alias_dups = [
        item["named_alias_name"] for item in targets if item["named_alias_name"]
    ]
    node_named_alias_dups = [
        item["named_alias_name"] for item in tx_nodes + rx_nodes if item["named_alias_name"]
    ]
    orphan_target_names = _orphan_target_names(viz, renderer)
    orphan_node_names = _orphan_node_names(viz, renderer)
    stale_external = _stale_external_object_mappings(viz, renderer)
    unknown_named = _unknown_named_geometries(renderer)
    # For each unknown, identify its pygfx object type (Mesh, Line, Points)
    # so ghost reports distinguish between dead-mesh leaks and overlay leaks.
    unknown_named_types: dict[str, str] = {}
    objects = getattr(renderer, "_objects", {})
    for name in unknown_named:
        obj = objects.get(name)
        if obj is not None:
            unknown_named_types[name] = type(obj).__name__
    # For each ghost, find its inverse external mapping
    ghost_external_info: list[dict[str, Any]] = []
    external_map_raw2 = getattr(renderer, "_external_geometry_names", {}) or {}
    for name in unknown_named:
        extracted_id_str = name.rsplit("_", 1)[-1] if name.startswith("external_geom_") else None
        if extracted_id_str is None:
            continue
        try:
            extracted_id = int(extracted_id_str)
        except ValueError:
            continue
        mapping = external_map_raw2.get(extracted_id)
        ghost_external_info.append(
            {
                "name": name,
                "extracted_id": extracted_id,
                "external_map_entry": mapping,
            }
        )
    # For each target, record the current mesh id and its external mapping entry
    # so ghost reports can correlate ghost ids back to specific targets.
    target_mesh_ids: list[dict[str, Any]] = []
    external_map_raw = getattr(renderer, "_external_geometry_names", {}) or {}
    for entry in getattr(viz, "target_entries", []):
        mesh = entry.get("mesh")
        if mesh is None:
            continue
        target_mesh_ids.append(
            {
                "target_name": entry.get("target_name") or entry.get("name"),
                "mesh_id": id(mesh),
                "external_map_name": external_map_raw.get(id(mesh), None),
            }
        )
    pygfx_census = _pygfx_scene_object_census(renderer)

    payload: dict[str, Any] = {
        "label": label,
        "step": step,
        "renderer": getattr(renderer, "renderer_type", type(renderer).__name__),
        "camera": None if camera_state is None else camera_state.to_dict(),
        "scene_bounds": None,
        "runtime_stats": (
            renderer.get_runtime_stats() if hasattr(renderer, "get_runtime_stats") else {}
        ),
        "named_geometry": named,
        "scene_meshes": _scene_mesh_summary(renderer),
        "targets": targets,
        "nodes": {"tx": tx_nodes, "rx": rx_nodes},
        "renderer_state": _renderer_state_summary(viz),
        "ghost_checks": {
            "origin_target_meshes": origin_targets,
            "origin_node_markers": origin_nodes,
            "target_external_duplicates": target_external_dups,
            "node_external_duplicates": node_external_dups,
            "target_named_alias_duplicates": target_named_alias_dups,
            "node_named_alias_duplicates": node_named_alias_dups,
            "orphan_target_named": orphan_target_names,
            "orphan_node_named": orphan_node_names,
            "stale_external_mappings": stale_external,
            "unknown_named_geometries": unknown_named,
            "unknown_named_types": unknown_named_types,
            "ghost_external_info": ghost_external_info,
            "target_mesh_ids": target_mesh_ids,
            "pygfx_scene_census": pygfx_census,
        },
    }
    if bounds_pair is not None:
        min_bound, max_bound = bounds_pair
        payload["scene_bounds"] = {
            "min": [float(v) for v in min_bound],
            "max": [float(v) for v in max_bound],
            "extent": [float(v) for v in (max_bound - min_bound)],
        }
    return payload
