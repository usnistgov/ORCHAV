"""Standalone pygfx geometry and material builder functions.

These free functions convert backend-neutral payload dataclasses into pygfx
scene-graph objects (Mesh, Line, Points).  They have no dependency on the
PygfxRenderer class or Qt, so they can be used in headless / notebook contexts.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from shared.logging import get_logger

from ..cache import pygfx_mesh_buffers as _mesh_cache
from ..diagnostics.cache_telemetry import (
    record_cache_event,
    set_cache_inventory,
)
from ..materials.texture_assets import load_decoded_texture, texture_asset_identity
from ..materials.texture_policy import resolve_texture_policy, warn_for_texture_policy
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
)

logger = get_logger("orchav.pygfx_helpers")


_PYGFX_MESH_CACHE_MODE_SPLIT_ALL = _mesh_cache.PYGFX_MESH_CACHE_MODE_SPLIT_ALL
_PYGFX_MESH_CACHE_MODE_REINDEX = _mesh_cache.PYGFX_MESH_CACHE_MODE_REINDEX
_PYGFX_MESH_BUFFER_MEMORY_CACHE = _mesh_cache._PYGFX_MESH_BUFFER_MEMORY_CACHE
_PRUNED_PYGFX_MESH_CACHE_ROOTS = _mesh_cache._PRUNED_ROOTS


# Low-level array helper


def ensure_contiguous(arr: Any, dtype: np.dtype) -> np.ndarray:
    """Return *arr* as a C-contiguous ndarray of *dtype*, avoiding a copy when possible."""
    a = np.asarray(arr)
    if a.dtype == dtype and a.flags["C_CONTIGUOUS"]:
        return a
    return np.ascontiguousarray(a, dtype=dtype)


def _writable_pygfx_buffer_array(arr: Any, dtype: np.dtype) -> np.ndarray:
    """Return an owned, writable C array for a pygfx buffer boundary."""
    contiguous = ensure_contiguous(arr, dtype)
    if contiguous.flags.owndata and contiguous.flags.writeable:
        return contiguous
    return np.array(contiguous, dtype=dtype, order="C", copy=True)


get_pygfx_mesh_buffer_cache_root = _mesh_cache.get_pygfx_mesh_buffer_cache_root
resolve_pygfx_mesh_buffer_cache_path = _mesh_cache.resolve_pygfx_mesh_buffer_cache_path
get_pygfx_mesh_buffer_cache_info = _mesh_cache.get_pygfx_mesh_buffer_cache_info
clear_pygfx_mesh_buffer_cache = _mesh_cache.clear_pygfx_mesh_buffer_cache
set_pygfx_mesh_buffer_cache_event_hook = _mesh_cache.set_pygfx_mesh_buffer_cache_event_hook
reset_pygfx_mesh_buffer_cache_metrics = _mesh_cache.reset_pygfx_mesh_buffer_cache_metrics


def _load_cached_mesh_payload_buffers(
    payload: MeshPayload,
    *,
    cache_identity: str,
) -> dict[str, np.ndarray] | None:
    """Apply a validated persisted seam-splitting plan to *payload*."""
    if payload.cache_key is None or payload.triangle_uvs is None:
        return None

    positions = ensure_contiguous(payload.vertices, np.float32)
    indices = ensure_contiguous(payload.triangles, np.int32)
    normals = (
        ensure_contiguous(payload.normals, np.float32) if payload.normals is not None else None
    )
    colors = (
        ensure_contiguous(payload.vertex_colors, np.float32)
        if payload.vertex_colors is not None
        else None
    )
    uvs = np.asarray(payload.triangle_uvs, dtype=np.float32)
    if uvs.ndim != 2 or uvs.shape[1] != 2 or len(uvs) != len(indices) * 3:
        return None

    plan = _mesh_cache.load_pygfx_mesh_cache_plan(
        payload.cache_key,
        cache_identity=cache_identity,
        vertex_count=len(positions),
        triangle_count=len(indices),
        uv_count=len(uvs),
    )
    if plan is None:
        return None

    flat_indices = indices.reshape(-1)
    n_vertices = len(positions)
    mode = plan["mode"]
    if mode == _PYGFX_MESH_CACHE_MODE_SPLIT_ALL:
        positions = ensure_contiguous(positions[flat_indices], np.float32)
        expanded_normals = _expand_vertex_attribute_for_faces(
            normals,
            n_vertices=n_vertices,
            flat_indices=flat_indices,
        )
        if expanded_normals is not None:
            normals = expanded_normals
        expanded_colors = _expand_vertex_attribute_for_faces(
            colors,
            n_vertices=n_vertices,
            flat_indices=flat_indices,
        )
        if expanded_colors is not None:
            colors = expanded_colors
        indices = np.arange(len(flat_indices), dtype=np.int32).reshape(-1, 3)
        texcoords = ensure_contiguous(uvs, np.float32)
    else:
        unique_corner_indices = plan["unique_corner_indices"]
        positions = ensure_contiguous(
            positions[flat_indices[unique_corner_indices]],
            np.float32,
        )
        gathered_normals = _gather_reindexed_attribute(
            normals,
            n_vertices=n_vertices,
            flat_indices=flat_indices,
            unique_corner_indices=unique_corner_indices,
        )
        if gathered_normals is not None:
            normals = gathered_normals
        gathered_colors = _gather_reindexed_attribute(
            colors,
            n_vertices=n_vertices,
            flat_indices=flat_indices,
            unique_corner_indices=unique_corner_indices,
        )
        if gathered_colors is not None:
            colors = gathered_colors
        indices = plan["indices"]
        texcoords = ensure_contiguous(uvs[unique_corner_indices], np.float32)

    buffers: dict[str, np.ndarray] = {
        "positions": positions,
        "indices": indices,
        "texcoords": texcoords,
    }
    if normals is not None:
        buffers["normals"] = normals
    if colors is not None:
        buffers["colors"] = colors
    return buffers


def _copy_pygfx_buffer_kwargs(kwargs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return an owned copy of pygfx buffer kwargs.

    pygfx geometry/buffer objects may keep references to the numpy arrays they
    are constructed from and later mutate those arrays during in-place updates.
    Returning owned copies prevents those writes from aliasing back into
    Open3D-backed payload arrays or the process-local mesh buffer cache.
    """
    copied: dict[str, np.ndarray] = {}
    for key, value in kwargs.items():
        arr = np.asarray(value)
        copied[key] = arr.copy() if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
    return copied


def _payload_has_texcoords(payload: MeshPayload) -> bool:
    """Return whether *payload* can produce a pygfx texcoords buffer."""
    if payload.triangle_uvs is None:
        return False
    uvs = np.asarray(payload.triangle_uvs)
    if uvs.ndim != 2 or uvs.shape[1] != 2:
        return False
    return len(uvs) in {len(payload.vertices), len(payload.triangles) * 3}


def _array_layout_token(values: Any | None) -> str:
    """Return a compact shape/dtype token without hashing array contents."""
    if values is None:
        return "none"
    array = np.asarray(values)
    shape = "x".join(str(dimension) for dimension in array.shape)
    return f"{array.dtype.str}:{shape}"


def _memory_mesh_cache_key(payload: MeshPayload, *, cache_identity: str) -> str:
    """Return a process-local cache key that includes renderable attributes."""
    return (
        f"{cache_identity}"
        f"|vertices={_array_layout_token(payload.vertices)}"
        f"|triangles={_array_layout_token(payload.triangles)}"
        f"|normals={_array_layout_token(payload.normals)}"
        f"|colors={_array_layout_token(payload.vertex_colors)}"
        f"|texcoords={_array_layout_token(payload.triangle_uvs)}"
    )


def _cached_buffers_match_payload(payload: MeshPayload, cached: dict[str, np.ndarray]) -> bool:
    """Return whether cached pygfx buffers expose the attributes this payload needs."""
    positions = cached.get("positions")
    if positions is None:
        return False
    try:
        n_positions = len(positions)
    except TypeError:
        return False
    expected = {
        "normals": payload.normals is not None,
        "colors": payload.vertex_colors is not None,
        "texcoords": _payload_has_texcoords(payload),
    }
    for name, should_have in expected.items():
        value = cached.get(name)
        if (value is not None) != should_have:
            return False
        if value is not None and len(value) != n_positions:
            return False
    return True


def _load_memory_cached_mesh_payload_buffers(
    payload: MeshPayload,
    *,
    cache_identity: str,
) -> dict[str, np.ndarray] | None:
    """Return a validated process-local prepared-buffer cache hit."""
    memory_key = _memory_mesh_cache_key(payload, cache_identity=cache_identity)
    cached = _mesh_cache.load_prepared_pygfx_mesh_buffers(memory_key)
    if cached is None:
        return None
    if not _cached_buffers_match_payload(payload, cached):
        _mesh_cache.invalidate_prepared_pygfx_mesh_buffers(memory_key)
        return None
    return cached


def _store_memory_cached_mesh_payload_buffers(
    payload: MeshPayload,
    kwargs: dict[str, np.ndarray],
    *,
    cache_identity: str,
) -> None:
    """Store prepared buffers through the byte-budgeted cache owner."""
    memory_key = _memory_mesh_cache_key(payload, cache_identity=cache_identity)
    _mesh_cache.store_prepared_pygfx_mesh_buffers(
        memory_key,
        kwargs,
        take_ownership=True,
    )


def _store_cached_mesh_payload_buffers(
    payload: MeshPayload,
    cache_plan: dict[str, Any],
    *,
    cache_identity: str,
) -> None:
    """Persist one seam-splitting plan through the cache lifecycle owner."""
    if payload.cache_key is None or payload.triangle_uvs is None:
        return
    _mesh_cache.store_pygfx_mesh_cache_plan(
        payload.cache_key,
        cache_identity=cache_identity,
        vertex_count=len(payload.vertices),
        triangle_count=len(payload.triangles),
        uv_count=len(payload.triangle_uvs),
        plan=cache_plan,
    )


# Geometry builders


def _expand_vertex_attribute_for_faces(
    attr: np.ndarray | None,
    *,
    n_vertices: int,
    flat_indices: np.ndarray,
) -> np.ndarray | None:
    """Expand a vertex-aligned attribute to face-corner rows when needed."""
    if attr is None:
        return None
    arr = np.asarray(attr)
    if arr.shape[0] == n_vertices:
        return ensure_contiguous(arr[flat_indices], np.float32)
    if arr.shape[0] == flat_indices.shape[0]:
        return ensure_contiguous(arr, np.float32)
    return None


def _gather_reindexed_attribute(
    attr: np.ndarray | None,
    *,
    n_vertices: int,
    flat_indices: np.ndarray,
    unique_corner_indices: np.ndarray,
) -> np.ndarray | None:
    """Gather a vertex/corner attribute for a seam-split indexed mesh."""
    if attr is None:
        return None
    arr = np.asarray(attr)
    if arr.shape[0] == n_vertices:
        return ensure_contiguous(arr[flat_indices[unique_corner_indices]], np.float32)
    if arr.shape[0] == flat_indices.shape[0]:
        return ensure_contiguous(arr[unique_corner_indices], np.float32)
    return None


def mesh_payload_to_pygfx_buffers(payload: MeshPayload) -> dict[str, np.ndarray]:
    """Convert a mesh payload into pygfx-ready buffer kwargs.

    Open3D stores UVs as face-corner rows (``N_triangles * 3``). pygfx
    expects texcoords aligned with the position buffer, so face-varying UVs
    require duplicated indexed vertices to preserve distinct values at shared
    corners instead of overwriting one face's UV with another.
    """
    cache_identity = (
        _mesh_cache.pygfx_mesh_cache_identity(payload.cache_key) if payload.cache_key else None
    )
    if cache_identity is not None:
        cached = _load_memory_cached_mesh_payload_buffers(
            payload,
            cache_identity=cache_identity,
        )
        if cached is not None:
            return cached
    if cache_identity is not None and payload.triangle_uvs is not None:
        cached = _load_cached_mesh_payload_buffers(
            payload,
            cache_identity=cache_identity,
        )
        if cached is not None:
            _store_memory_cached_mesh_payload_buffers(
                payload,
                cached,
                cache_identity=cache_identity,
            )
            return _copy_pygfx_buffer_kwargs(cached)

    positions = ensure_contiguous(payload.vertices, np.float32)
    indices = ensure_contiguous(payload.triangles, np.int32)
    normals = (
        ensure_contiguous(payload.normals, np.float32) if payload.normals is not None else None
    )
    colors = (
        ensure_contiguous(payload.vertex_colors, np.float32)
        if payload.vertex_colors is not None
        else None
    )
    texcoords: np.ndarray | None = None
    cache_plan: dict[str, Any] | None = None

    if payload.triangle_uvs is not None:
        uvs = np.asarray(payload.triangle_uvs, dtype=np.float32)
        if uvs.ndim == 2 and uvs.shape[1] == 2:
            if len(uvs) == len(indices) * 3:
                flat_indices = indices.reshape(-1)
                n_vertices = len(positions)
                keys = np.empty(
                    len(flat_indices),
                    dtype=[("vertex", np.int32), ("u", np.float32), ("v", np.float32)],
                )
                keys["vertex"] = flat_indices
                keys["u"] = uvs[:, 0]
                keys["v"] = uvs[:, 1]
                _, unique_corner_indices, inverse = np.unique(
                    keys,
                    return_index=True,
                    return_inverse=True,
                )
                if len(unique_corner_indices) == len(flat_indices):
                    positions = ensure_contiguous(positions[flat_indices], np.float32)
                    expanded_normals = _expand_vertex_attribute_for_faces(
                        normals,
                        n_vertices=n_vertices,
                        flat_indices=flat_indices,
                    )
                    if expanded_normals is not None:
                        normals = expanded_normals
                    expanded_colors = _expand_vertex_attribute_for_faces(
                        colors,
                        n_vertices=n_vertices,
                        flat_indices=flat_indices,
                    )
                    if expanded_colors is not None:
                        colors = expanded_colors
                    indices = np.arange(len(flat_indices), dtype=np.int32).reshape(-1, 3)
                    texcoords = ensure_contiguous(uvs, np.float32)
                    cache_plan = {"mode": _PYGFX_MESH_CACHE_MODE_SPLIT_ALL}
                else:
                    order = np.argsort(unique_corner_indices)
                    unique_corner_indices = unique_corner_indices[order]
                    remap = np.empty(len(order), dtype=np.int32)
                    remap[order] = np.arange(len(order), dtype=np.int32)
                    inverse = remap[inverse]

                    positions = ensure_contiguous(
                        positions[flat_indices[unique_corner_indices]],
                        np.float32,
                    )
                    gathered_normals = _gather_reindexed_attribute(
                        normals,
                        n_vertices=n_vertices,
                        flat_indices=flat_indices,
                        unique_corner_indices=unique_corner_indices,
                    )
                    if gathered_normals is not None:
                        normals = gathered_normals
                    gathered_colors = _gather_reindexed_attribute(
                        colors,
                        n_vertices=n_vertices,
                        flat_indices=flat_indices,
                        unique_corner_indices=unique_corner_indices,
                    )
                    if gathered_colors is not None:
                        colors = gathered_colors
                    indices = ensure_contiguous(inverse.reshape(-1, 3), np.int32)
                    texcoords = ensure_contiguous(uvs[unique_corner_indices], np.float32)
                    cache_plan = {
                        "mode": _PYGFX_MESH_CACHE_MODE_REINDEX,
                        "unique_corner_indices": ensure_contiguous(
                            unique_corner_indices,
                            np.int32,
                        ),
                        "indices": indices,
                    }
            elif len(uvs) == len(positions):
                texcoords = ensure_contiguous(uvs, np.float32)

    kwargs: dict[str, np.ndarray] = {
        "positions": positions,
        "indices": indices,
    }
    if normals is not None:
        kwargs["normals"] = normals
    if colors is not None:
        kwargs["colors"] = colors
    if texcoords is not None:
        kwargs["texcoords"] = texcoords
    if cache_identity is not None:
        _store_memory_cached_mesh_payload_buffers(
            payload,
            kwargs,
            cache_identity=cache_identity,
        )
    if cache_identity is not None and cache_plan is not None:
        _store_cached_mesh_payload_buffers(
            payload,
            cache_plan,
            cache_identity=cache_identity,
        )
    return _copy_pygfx_buffer_kwargs(kwargs)


def build_mesh_geometry(gfx: Any, payload: MeshPayload) -> Any:
    """Convert a ``MeshPayload`` into a ``pygfx.Geometry``."""
    return gfx.Geometry(**mesh_payload_to_pygfx_buffers(payload))


def build_lines_geometry(gfx: Any, payload: LineSetPayload) -> Any:
    """Convert a ``LineSetPayload`` into a ``pygfx.Geometry``.

    Indexed payloads are expanded to disjoint endpoint pairs before they
    are sent to ``LineSegmentMaterial``. This preserves Open3D ``LineSet``
    semantics where every row in ``lines`` is an independent segment.
    """
    positions = ensure_contiguous(payload.points, np.float32)
    lines = ensure_contiguous(payload.lines, np.int32)
    colors = ensure_contiguous(payload.colors, np.float32) if payload.colors is not None else None

    if len(lines) > 0:
        n_segs = len(lines)
        seg_points = np.empty((n_segs * 2, 3), dtype=np.float32)
        seg_points[0::2] = positions[lines[:, 0]]
        seg_points[1::2] = positions[lines[:, 1]]

        if colors is not None:
            n_cols = colors.shape[-1]
            seg_colors = np.empty((n_segs * 2, n_cols), dtype=np.float32)
            if len(colors) == len(positions):
                seg_colors[0::2] = colors[lines[:, 0]]
                seg_colors[1::2] = colors[lines[:, 1]]
            else:
                # Open3D LineSet colors are per segment.
                seg_colors[0::2] = colors[:n_segs]
                seg_colors[1::2] = colors[:n_segs]
            return gfx.Geometry(
                positions=_writable_pygfx_buffer_array(seg_points, np.float32),
                colors=_writable_pygfx_buffer_array(seg_colors, np.float32),
            )

        return gfx.Geometry(positions=_writable_pygfx_buffer_array(seg_points, np.float32))

    if colors is not None:
        return gfx.Geometry(
            positions=_writable_pygfx_buffer_array(positions, np.float32),
            colors=_writable_pygfx_buffer_array(colors, np.float32),
        )
    return gfx.Geometry(positions=_writable_pygfx_buffer_array(positions, np.float32))


def build_points_geometry(gfx: Any, payload: PointCloudPayload) -> Any:
    """Convert a ``PointCloudPayload`` into a ``pygfx.Geometry``."""
    kwargs: dict[str, Any] = {
        "positions": _writable_pygfx_buffer_array(payload.points, np.float32),
    }
    if payload.colors is not None:
        kwargs["colors"] = _writable_pygfx_buffer_array(payload.colors, np.float32)
    return gfx.Geometry(**kwargs)


# Material builders


import os as _os

# Runtime flag: when set, the pygfx renderer uses an unlit material path
# (``MeshBasicMaterial``) so ``base_color`` passes straight through to the
# framebuffer. This is the "chip = displayed color" workflow the Open3D
# renderer exposes via ``defaultUnlit`` and pygfx has been missing. Using
# an env var for the first iteration so the feature is testable without
# UI plumbing; a Render-panel checkbox can follow once the visual is
# confirmed.
_PYGFX_UNLIT_MODE: bool = _os.environ.get("ORCHAV_PYGFX_UNLIT") == "1"


def _is_pygfx_unlit_mode_enabled() -> bool:
    """Return True when pygfx should render meshes with ``MeshBasicMaterial``."""
    return _PYGFX_UNLIT_MODE


def apply_texture_policy_to_material_payload(
    material: MaterialPayload,
    *,
    context: str | None = None,
) -> MaterialPayload:
    """Return *material* with inactive maps stripped and albedo color policy applied."""

    try:
        alpha = float(material.base_color[3])
    except (TypeError, ValueError, IndexError):
        alpha = 1.0
    policy = resolve_texture_policy(
        {
            "texture_path": material.texture_path,
            "normal_map_path": material.normal_map_path,
            "roughness_map_path": material.roughness_map_path,
            "ao_map_path": material.ao_map_path,
            "metallic_map_path": material.metallic_map_path,
            "alpha": alpha,
        },
        color=material.base_color,
        alpha=alpha,
        context=context or "pygfx material",
    )
    warn_for_texture_policy(policy, log=logger)
    return replace(
        material,
        base_color=policy.renderer_base_color,
        texture_path=policy.active_maps["texture_path"],
        normal_map_path=policy.active_maps["normal_map_path"],
        roughness_map_path=policy.active_maps["roughness_map_path"],
        ao_map_path=policy.active_maps["ao_map_path"],
        metallic_map_path=policy.active_maps["metallic_map_path"],
    )


def _apply_mesh_material_alpha_state(mat: Any, material: MaterialPayload) -> None:
    """Map backend-neutral alpha to pygfx's opacity and blend/depth state."""
    alpha = max(0.0, min(1.0, float(material.base_color[3])))
    transparent = alpha < 0.999 or "transparen" in material.shader.lower()

    for attr, value in (
        ("opacity", alpha),
        ("alpha_mode", "weighted_blend" if transparent else "auto"),
        ("depth_write", not transparent),
        ("depth_test", True),
    ):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, value)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass


def make_mesh_material(
    gfx: Any,
    material: MaterialPayload,
    *,
    has_vertex_colors: bool = False,
    ibl_manager: Any | None = None,
) -> Any:
    """Create a pygfx mesh material from a ``MaterialPayload``.

    Returns a ``MeshPhysicalMaterial`` when the payload requests advanced PBR
    (clearcoat / anisotropy / emissive); otherwise returns a
    ``MeshStandardMaterial`` for the fast path. Falls back to
    ``MeshPhongMaterial`` if neither is available.

    When ``ORCHAV_PYGFX_UNLIT=1`` is set, returns ``MeshBasicMaterial``
    instead so ``base_color`` displays unmodulated by any lights, IBL, or
    shadows — useful for the "chip = displayed color" workflow that
    matches Filament's ``defaultUnlit`` on Open3D.

    Wave B fields (``transmission`` / ``thickness``) are silently discarded
    because pygfx 0.15.3 has no equivalent on ``MeshPhysicalMaterial``.
    """
    material = apply_texture_policy_to_material_payload(material, context="pygfx mesh material")
    effective_color = (
        (1.0, 1.0, 1.0, material.base_color[3]) if has_vertex_colors else material.base_color
    )

    if _is_pygfx_unlit_mode_enabled():
        basic_kwargs: dict[str, Any] = {"color": effective_color}
        if has_vertex_colors:
            basic_kwargs["color_mode"] = "vertex"
        try:
            mat = gfx.MeshBasicMaterial(**basic_kwargs)
        except TypeError:
            basic_kwargs.pop("color_mode", None)
            mat = gfx.MeshBasicMaterial(**basic_kwargs)
        if material.texture_path is not None and hasattr(mat, "map"):
            tex = _load_pygfx_texture(gfx, material.texture_path)
            if tex is not None:
                mat.map = tex
        _apply_mesh_material_alpha_state(mat, material)
        return mat

    kwargs: dict[str, Any] = {
        "color": effective_color,
        "roughness": material.roughness,
        "metalness": material.metallic,
    }
    if has_vertex_colors:
        kwargs["color_mode"] = "vertex"

    use_physical = material.has_advanced_pbr and hasattr(gfx, "MeshPhysicalMaterial")
    if use_physical:
        kwargs["clearcoat"] = material.clearcoat
        kwargs["clearcoat_roughness"] = material.clearcoat_roughness
        # Anisotropy is intentionally NOT passed: pygfx 0.15.3 can compile
        # an invalid WGSL shader for MeshPhysicalMaterial.anisotropy > 0
        # when no normal map is bound. The
        # Open3D renderer applies it via base_anisotropy as usual.
        if any(c > 0.0 for c in material.emissive_color):
            er, eg, eb = material.emissive_color
            kwargs["emissive"] = (er, eg, eb, 1.0)
        kwargs["emissive_intensity"] = max(0.0, float(material.emissive_intensity))
        material_cls = gfx.MeshPhysicalMaterial
    else:
        material_cls = gfx.MeshStandardMaterial

    try:
        mat = material_cls(**kwargs)
    except TypeError:
        kwargs.pop("color_mode", None)
        try:
            mat = material_cls(**kwargs)
        except (TypeError, RuntimeError, AttributeError):
            mat = gfx.MeshPhongMaterial(color=material.base_color, shininess=20)
            _apply_mesh_material_alpha_state(mat, material)
            return mat
    except (RuntimeError, AttributeError):
        mat = gfx.MeshPhongMaterial(color=material.base_color, shininess=20)
        _apply_mesh_material_alpha_state(mat, material)
        return mat

    reflectance = max(0.0, min(1.0, float(material.reflectance)))
    for attr in ("reflectivity", "specular_intensity"):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, reflectance)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    _apply_mesh_material_alpha_state(mat, material)

    # PBR maps use the shared decoded-asset cache so notebook/static and
    # desktop rendering share CPU pixels while retaining separate native
    # texture handles. Missing or unreadable files remain best-effort.
    _apply_pbr_maps_to_pygfx_material(gfx, mat, material)

    if ibl_manager is not None:
        try:
            ibl_manager.apply_to_material(mat)
        except (AttributeError, RuntimeError):
            pass
    return mat


# Module-level native texture cache shared by notebook/static callers. CPU
# decoding belongs to ``materials.texture_assets``; this cache owns only gfx
# resources, so unrelated pygfx module instances never share native handles.
_PYGFX_TEXTURE_CACHE: dict[tuple[str, int], Any] = {}
_PYGFX_TEXTURE_CACHE_BYTES: dict[tuple[str, int], int] = {}
_PYGFX_TEXTURE_SOURCE_IDENTITIES: dict[tuple[str, int], str] = {}
_PYGFX_TEXTURE_NEGATIVE_IDENTITIES: set[tuple[str, int]] = set()
_PYGFX_TEXTURE_CACHE_LOCK = threading.RLock()
_DEFAULT_PYGFX_NATIVE_TEXTURE_CACHE_MAX_BYTES = 256 * 1024 * 1024


def _pygfx_native_texture_cache_max_bytes() -> int:
    raw = _os.environ.get("ORCHAV_PYGFX_NATIVE_TEXTURE_CACHE_MAX_BYTES")
    try:
        return (
            _DEFAULT_PYGFX_NATIVE_TEXTURE_CACHE_MAX_BYTES
            if raw is None
            else max(0, int(float(raw)))
        )
    except (OverflowError, TypeError, ValueError):
        return _DEFAULT_PYGFX_NATIVE_TEXTURE_CACHE_MAX_BYTES


def _pygfx_native_texture_cache_nbytes_unlocked() -> int:
    return sum(_PYGFX_TEXTURE_CACHE_BYTES.values())


def _touch_pygfx_native_texture_unlocked(cache_key: tuple[str, int]) -> Any | None:
    texture = _PYGFX_TEXTURE_CACHE.pop(cache_key, None)
    if texture is not None:
        _PYGFX_TEXTURE_CACHE[cache_key] = texture
    return texture


def _evict_pygfx_native_textures_unlocked() -> tuple[int, int]:
    max_bytes = _pygfx_native_texture_cache_max_bytes()
    evicted = 0
    evicted_bytes = 0
    while _PYGFX_TEXTURE_CACHE and _pygfx_native_texture_cache_nbytes_unlocked() > max_bytes:
        cache_key = next(iter(_PYGFX_TEXTURE_CACHE))
        _PYGFX_TEXTURE_CACHE.pop(cache_key, None)
        removed_bytes = _PYGFX_TEXTURE_CACHE_BYTES.pop(cache_key, 0)
        identity, gfx_id = cache_key
        for source_key, source_identity in tuple(_PYGFX_TEXTURE_SOURCE_IDENTITIES.items()):
            if source_key[1] == gfx_id and source_identity == identity:
                _PYGFX_TEXTURE_SOURCE_IDENTITIES.pop(source_key, None)
        evicted += 1
        evicted_bytes += removed_bytes
    return evicted, evicted_bytes


def get_pygfx_native_texture_cache_info() -> dict[str, int]:
    """Return notebook/static pygfx native-texture cache inventory."""
    with _PYGFX_TEXTURE_CACHE_LOCK:
        entries = len(_PYGFX_TEXTURE_CACHE)
        byte_count = _pygfx_native_texture_cache_nbytes_unlocked()
        failed_identities = len(_PYGFX_TEXTURE_NEGATIVE_IDENTITIES)
    set_cache_inventory(
        "pygfx_native_texture",
        entries=entries,
        byte_count=byte_count,
    )
    return {
        "entries": entries,
        "bytes": byte_count,
        "max_bytes": _pygfx_native_texture_cache_max_bytes(),
        "failed_sources": 0,
        "failed_identities": failed_identities,
    }


def clear_pygfx_native_texture_cache() -> dict[str, int]:
    """Release notebook/static native textures and negative-cache state."""
    with _PYGFX_TEXTURE_CACHE_LOCK:
        entries = len(_PYGFX_TEXTURE_CACHE)
        failures = len(_PYGFX_TEXTURE_NEGATIVE_IDENTITIES)
        byte_count = _pygfx_native_texture_cache_nbytes_unlocked()
        _PYGFX_TEXTURE_CACHE.clear()
        _PYGFX_TEXTURE_CACHE_BYTES.clear()
        _PYGFX_TEXTURE_SOURCE_IDENTITIES.clear()
        _PYGFX_TEXTURE_NEGATIVE_IDENTITIES.clear()
    record_cache_event(
        "pygfx_native_texture",
        "clear",
        count=entries,
        byte_count=byte_count,
    )
    set_cache_inventory("pygfx_native_texture", entries=0, byte_count=0)
    return {"entries": entries, "bytes": byte_count, "failures": failures}


def _load_pygfx_texture(gfx: Any, path: str) -> Any:
    """Return a gfx-native texture backed by the shared decoded asset cache."""
    identity_result = texture_asset_identity(path)
    if identity_result is None:
        try:
            source_path = str(Path(path).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            source_path = str(path)
        asset = load_decoded_texture(path)
        if asset is None:
            record_cache_event("pygfx_native_texture", "failure")
            return None
    else:
        identity, resolved_path = identity_result
        source_path = str(resolved_path)
        cache_key = (identity, id(gfx))
        with _PYGFX_TEXTURE_CACHE_LOCK:
            cached = _touch_pygfx_native_texture_unlocked(cache_key)
            if cached is not None:
                record_cache_event("pygfx_native_texture", "hit")
                return cached
            if cache_key in _PYGFX_TEXTURE_NEGATIVE_IDENTITIES:
                record_cache_event("pygfx_native_texture", "negative_hit")
                return None
        asset = load_decoded_texture(resolved_path)
        if asset is None:
            with _PYGFX_TEXTURE_CACHE_LOCK:
                _PYGFX_TEXTURE_NEGATIVE_IDENTITIES.add(cache_key)
            record_cache_event("pygfx_native_texture", "failure")
            return None

    source_key = (source_path, id(gfx))
    cache_key = (asset.identity, id(gfx))
    with _PYGFX_TEXTURE_CACHE_LOCK:
        cached = _touch_pygfx_native_texture_unlocked(cache_key)
        if cached is not None:
            record_cache_event("pygfx_native_texture", "hit")
            return cached
        if cache_key in _PYGFX_TEXTURE_NEGATIVE_IDENTITIES:
            record_cache_event("pygfx_native_texture", "negative_hit")
            return None

    try:
        tex = gfx.Texture(np.array(asset.rgba, copy=True), dim=2)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        with _PYGFX_TEXTURE_CACHE_LOCK:
            _PYGFX_TEXTURE_NEGATIVE_IDENTITIES.add(cache_key)
        record_cache_event("pygfx_native_texture", "failure")
        return None

    with _PYGFX_TEXTURE_CACHE_LOCK:
        previous_identity = _PYGFX_TEXTURE_SOURCE_IDENTITIES.get(source_key)
        if previous_identity is not None and previous_identity != asset.identity:
            _PYGFX_TEXTURE_CACHE.pop((previous_identity, id(gfx)), None)
            _PYGFX_TEXTURE_CACHE_BYTES.pop((previous_identity, id(gfx)), None)
            _PYGFX_TEXTURE_NEGATIVE_IDENTITIES.discard((previous_identity, id(gfx)))
        _PYGFX_TEXTURE_CACHE[cache_key] = tex
        _PYGFX_TEXTURE_CACHE_BYTES[cache_key] = asset.nbytes
        _PYGFX_TEXTURE_SOURCE_IDENTITIES[source_key] = asset.identity
        evicted, evicted_bytes = _evict_pygfx_native_textures_unlocked()
        entries = len(_PYGFX_TEXTURE_CACHE)
        byte_count = _pygfx_native_texture_cache_nbytes_unlocked()
    if evicted:
        record_cache_event(
            "pygfx_native_texture",
            "eviction",
            count=evicted,
            byte_count=evicted_bytes,
        )
    record_cache_event(
        "pygfx_native_texture",
        "miss",
        byte_count=asset.nbytes,
    )
    set_cache_inventory(
        "pygfx_native_texture",
        entries=entries,
        byte_count=byte_count,
    )
    return tex


def _apply_pbr_maps_to_pygfx_material(gfx: Any, mat: Any, material: MaterialPayload) -> None:
    """Load and assign PBR texture maps to a pygfx mesh material.

    Shared between ``make_mesh_material`` (notebook path) and the
    renderer's own ``set_named_material`` (desktop path) to keep the
    two ``MaterialPayload → pygfx material`` code paths in sync. The shared
    decoder and module-level native cache avoid repeated work per mesh.
    """
    map_paths: list[tuple[Optional[str], str]] = [
        (material.texture_path, "map"),
        (material.normal_map_path, "normal_map"),
        (material.roughness_map_path, "roughness_map"),
        (material.ao_map_path, "ao_map"),
        (material.metallic_map_path, "metalness_map"),
    ]
    for path, attr in map_paths:
        if path is None or not hasattr(mat, attr):
            continue
        tex = _load_pygfx_texture(gfx, path)
        if tex is not None:
            setattr(mat, attr, tex)
            if attr == "normal_map" and hasattr(mat, "normal_scale"):
                normal_strength = max(0.0, min(4.0, float(material.normal_map_strength)))
                try:
                    mat.normal_scale = (normal_strength, normal_strength)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass


def make_line_material(
    gfx: Any,
    *,
    has_vertex_colors: bool = False,
    line_width: float = 2.0,
    line_strip: bool = False,
) -> Any:
    """Create a pygfx line material.

    Indexed ``LineSetPayload`` objects use ``LineSegmentMaterial``. Payloads
    without explicit line indices are continuous line strips and need
    ``LineMaterial`` so consecutive positions are joined.
    """
    kwargs: dict[str, Any] = {
        "color": (0.6, 0.6, 0.6, 1.0),
        "thickness": line_width,
    }
    if has_vertex_colors:
        kwargs["color_mode"] = "vertex"
    if line_strip and hasattr(gfx, "LineMaterial"):
        try:
            return gfx.LineMaterial(**kwargs)
        except TypeError:
            kwargs.pop("color_mode", None)
            return gfx.LineMaterial(**kwargs)

    try:
        return gfx.LineSegmentMaterial(**kwargs)
    except TypeError:
        kwargs.pop("color_mode", None)
        try:
            return gfx.LineSegmentMaterial(**kwargs)
        except (TypeError, RuntimeError, AttributeError):
            return gfx.LineMaterial(**kwargs)


def make_point_material(
    gfx: Any,
    *,
    has_vertex_colors: bool = False,
    point_size: float = 5.0,
) -> Any:
    """Create a ``PointsMaterial``."""
    kwargs: dict[str, Any] = {
        "color": (0.9, 0.9, 0.9, 1.0),
        "size": point_size,
    }
    if has_vertex_colors:
        kwargs["color_mode"] = "vertex"
    try:
        return gfx.PointsMaterial(**kwargs)
    except TypeError:
        kwargs.pop("color_mode", None)
        return gfx.PointsMaterial(**kwargs)


# High-level convenience: payload → pygfx world-object


def payload_to_pygfx_mesh(
    gfx: Any,
    mesh_payload: MeshPayload,
    material: MaterialPayload,
    *,
    ibl_manager: Any | None = None,
) -> Any:
    """Build a ``pygfx.Mesh`` from payload dataclasses."""
    geom = build_mesh_geometry(gfx, mesh_payload)
    has_colors = mesh_payload.vertex_colors is not None
    mat = make_mesh_material(gfx, material, has_vertex_colors=has_colors, ibl_manager=ibl_manager)
    return gfx.Mesh(geom, mat)


def payload_to_pygfx_lines(
    gfx: Any,
    lines_payload: LineSetPayload,
    *,
    line_width: float = 2.0,
) -> Any:
    """Build a ``pygfx.Line`` from a ``LineSetPayload``."""
    geom = build_lines_geometry(gfx, lines_payload)
    has_colors = lines_payload.colors is not None
    mat = make_line_material(
        gfx,
        has_vertex_colors=has_colors,
        line_width=line_width,
        line_strip=len(lines_payload.lines) == 0,
    )
    return gfx.Line(geom, mat)


def payload_to_pygfx_points(
    gfx: Any,
    points_payload: PointCloudPayload,
    *,
    point_size: float = 5.0,
) -> Any:
    """Build a ``pygfx.Points`` from a ``PointCloudPayload``."""
    geom = build_points_geometry(gfx, points_payload)
    has_colors = points_payload.colors is not None
    mat = make_point_material(gfx, has_vertex_colors=has_colors, point_size=point_size)
    return gfx.Points(geom, mat)
