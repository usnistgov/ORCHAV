"""Visual-only UV skins shared by every pose of a fixed-topology target.

The sidecar binds texture coordinates to triangle corners, never to world
positions. It therefore works across articulated poses without changing the
recording, mesh vertices, transforms, or electromagnetic materials.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..materials.texture_policy import TEXTURE_MAP_KEYS
from ..types.render_payloads import MeshPayload, SurfaceColorSource

_SURFACE_SCALARS = frozenset(
    {"roughness", "metallic", "reflectance", "clearcoat", "clearcoat_roughness"}
)


@dataclass(frozen=True)
class TargetSurface:
    """Validated appearance data, loaded once and shared by pose builders."""

    texture_path: str
    triangles: np.ndarray
    triangle_uvs: np.ndarray
    vertex_count: int
    fingerprint: str
    material: tuple[tuple[str, Any], ...] = ()

    def apply(self, payload: MeshPayload) -> MeshPayload:
        """Attach UVs only when vertex count and ordered topology match."""
        if len(payload.vertices) != self.vertex_count or not np.array_equal(
            payload.triangles, self.triangles
        ):
            raise ValueError("Target surface topology does not match the animated mesh")
        return replace(
            payload,
            triangle_uvs=self.triangle_uvs,
            vertex_colors=None,
            color_source=SurfaceColorSource.MATERIAL,
            cache_key=(
                f"{payload.cache_key}:surface:{self.fingerprint}" if payload.cache_key else None
            ),
        )


def load_target_surface(path: Path) -> TargetSurface:
    """Read a JSON sidecar and numeric NPZ UVs relative to its location."""
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema") != "orchav.target_surface.v1":
        raise ValueError("Expected an orchav.target_surface.v1 sidecar")
    texture_path = (path.parent / config["texture"]).resolve(strict=True)
    uv_path = (path.parent / config["uv_map"]).resolve(strict=True)
    properties = config.get("material", {})
    if not isinstance(properties, dict) or set(properties) - (
        _SURFACE_SCALARS | (set(TEXTURE_MAP_KEYS) - {"texture_path"})
    ):
        raise ValueError("Target surface material contains unsupported fields")
    material = {}
    for key, value in properties.items():
        if key in _SURFACE_SCALARS:
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Target surface {key} must be finite in [0, 1]")
            material[key] = float(value)
        else:
            if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
                raise ValueError(f"Target surface {key} must be a relative path")
            map_path = (path.parent / value).resolve(strict=True)
            if not map_path.is_file():
                raise ValueError(f"Target surface {key} must reference a file")
            material[key] = str(map_path)
    with np.load(uv_path, allow_pickle=False) as data:
        triangles = data["triangles"]
        uv = np.asarray(data["triangle_uvs"], dtype=np.float32)
        count = data["vertex_count"]
        if count.shape != () or count.dtype.kind not in "iu" or int(count) < 1:
            raise ValueError("Target surface vertex_count must be a positive integer")
        vertex_count = int(count)
        if (
            triangles.ndim != 2
            or triangles.shape[1] != 3
            or triangles.dtype.kind not in "iu"
            or not len(triangles)
            or np.any(triangles < 0)
            or np.any(triangles >= vertex_count)
        ):
            raise ValueError("Target surface triangles must index the declared vertices")
        if uv.shape != (len(triangles) * 3, 2) or not np.all(np.isfinite(uv)):
            raise ValueError("Target surface requires finite UVs for every triangle corner")
    # Immutable storage lets worker threads and mesh payloads share these arrays.
    triangles = np.frombuffer(triangles.tobytes(), dtype=triangles.dtype).reshape(-1, 3)
    uv = np.frombuffer(uv.tobytes(), dtype=np.float32).reshape(-1, 2)
    digest = hashlib.sha256(raw + triangles.tobytes() + uv.tobytes()).hexdigest()
    return TargetSurface(
        str(texture_path), triangles, uv, vertex_count, digest, tuple(material.items())
    )


def target_surface_for_profile(viz: Any, name: str, material_type: str) -> TargetSurface | None:
    """Resolve the selected profile's sidecar using the scenario path policy."""
    service = getattr(viz, "visual_profile_service", None)
    resolve = getattr(service, "resolve_surface_path", None)
    path = resolve(name, material_type, "target") if callable(resolve) else None
    if not isinstance(path, str) or not path:
        return None
    policy = getattr(viz, "current_scenario_policy", None)
    resolved = policy.resolve_path(path) if policy is not None else Path(path).resolve()
    return load_target_surface(resolved)
