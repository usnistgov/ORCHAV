"""Backend-neutral geometry and material payload dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np


class SurfaceColorSource(str, Enum):
    """Own the RGB detail used by one mesh surface.

    ``MATERIAL`` means the material (a uniform base color or an external
    albedo texture) owns surface color. ``VERTEX`` means authored vertex
    colors are intrinsic geometry detail and must survive material changes.
    """

    MATERIAL = "material"
    VERTEX = "vertex"


def _immutable_array(values: Any) -> np.ndarray:
    """Return a detached array snapshot backed by immutable storage.

    Render payload construction must never freeze or otherwise mutate a
    caller-owned array. Backend staging buffers are deliberately writable and
    may be reused after a payload is built from them. A bytes-backed snapshot
    makes the payload genuinely read-only while allowing snapshots that are
    already immutable to be reused without another copy.
    """
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("Render payload arrays must use a numeric dtype")
    if array.flags.c_contiguous and _has_immutable_storage(array):
        return array
    storage = array.tobytes(order="C")
    return np.frombuffer(storage, dtype=array.dtype).reshape(array.shape)


def _has_immutable_storage(array: np.ndarray) -> bool:
    """Return whether an array ultimately references a read-only byte buffer."""
    owner: Any = array
    visited: set[int] = set()
    while isinstance(owner, (np.ndarray, memoryview)):
        identity = id(owner)
        if identity in visited:
            return False
        visited.add(identity)
        if isinstance(owner, np.ndarray):
            owner = owner.base
            if owner is None:
                return False
            continue
        if not owner.readonly:
            return False
        owner = owner.obj
    if isinstance(owner, bytes):
        return True
    return False


def _immutable_optional_array(values: Any | None) -> np.ndarray | None:
    """Return a frozen array for optional payload buffers."""
    return None if values is None else _immutable_array(values)


@dataclass(frozen=True)
class MeshPayload:
    """Triangle mesh payload with numpy buffers."""

    vertices: np.ndarray
    triangles: np.ndarray
    normals: Optional[np.ndarray] = None
    vertex_colors: Optional[np.ndarray] = None
    triangle_uvs: Optional[np.ndarray] = None
    color_source: SurfaceColorSource = SurfaceColorSource.MATERIAL
    cache_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Freeze all mesh buffers at the renderer-neutral boundary."""
        object.__setattr__(self, "vertices", _immutable_array(self.vertices))
        object.__setattr__(self, "triangles", _immutable_array(self.triangles))
        object.__setattr__(self, "normals", _immutable_optional_array(self.normals))
        object.__setattr__(
            self,
            "vertex_colors",
            _immutable_optional_array(self.vertex_colors),
        )
        object.__setattr__(
            self,
            "triangle_uvs",
            _immutable_optional_array(self.triangle_uvs),
        )
        object.__setattr__(self, "color_source", SurfaceColorSource(self.color_source))


def mesh_payload_for_pbr_material(
    payload: MeshPayload,
    *,
    color_source: SurfaceColorSource | None = None,
    cache_key: Optional[str] = None,
) -> MeshPayload:
    """Return a mesh payload prepared for PBR material-color rendering.

    Scene meshes often carry stale vertex colors from mesh loading or XML
    preview tinting. PBR material paths should drop those colors unless the
    payload explicitly declares authored vertex colors as its color source.
    """
    resolved_source = (
        payload.color_source if color_source is None else SurfaceColorSource(color_source)
    )
    return MeshPayload(
        vertices=payload.vertices,
        triangles=payload.triangles,
        normals=payload.normals,
        vertex_colors=(
            payload.vertex_colors if resolved_source is SurfaceColorSource.VERTEX else None
        ),
        triangle_uvs=payload.triangle_uvs,
        color_source=resolved_source,
        cache_key=payload.cache_key if cache_key is None else cache_key,
    )


@dataclass(frozen=True)
class LineSetPayload:
    """Line set payload with numpy buffers."""

    points: np.ndarray
    lines: np.ndarray
    colors: Optional[np.ndarray] = None
    line_strip: bool = False

    def __post_init__(self) -> None:
        """Freeze all line-set buffers at the renderer-neutral boundary."""
        object.__setattr__(self, "points", _immutable_array(self.points))
        object.__setattr__(self, "lines", _immutable_array(self.lines))
        object.__setattr__(self, "colors", _immutable_optional_array(self.colors))


@dataclass(frozen=True)
class PointCloudPayload:
    """Point cloud payload with numpy buffers."""

    points: np.ndarray
    colors: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Freeze all point-cloud buffers at the renderer-neutral boundary."""
        object.__setattr__(self, "points", _immutable_array(self.points))
        object.__setattr__(self, "colors", _immutable_optional_array(self.colors))


@dataclass(frozen=True)
class OrientationFramePayload:
    """Semantic RGB coordinate-frame payload.

    Renderers should map this to their native axis-frame helper when one is
    available instead of degrading it to generic line geometry.
    """

    size: float
    thickness: float = 4.0


@dataclass(frozen=True, slots=True)
class TextLabelPayload:
    """Renderer-neutral text content and presentation intent.

    Position, visibility, and color deliberately stay in ``RenderObject``
    state so labels follow the same synchronization contract as every other
    persistent scene object.
    """

    text: str
    font_size: float = 0.3
    screen_space: bool = True
    outline_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    outline_thickness: float = 0.15

    def __post_init__(self) -> None:
        """Normalize values shared by both renderer implementations."""
        object.__setattr__(self, "text", str(self.text))
        size = float(self.font_size)
        if not np.isfinite(size) or size <= 0.0:
            raise ValueError("TextLabelPayload font_size must be a positive finite value")
        object.__setattr__(self, "font_size", size)
        object.__setattr__(self, "screen_space", bool(self.screen_space))
        outline = tuple(float(value) for value in self.outline_color)
        if len(outline) != 3 or not all(np.isfinite(outline)):
            raise ValueError("TextLabelPayload outline_color must contain three finite values")
        object.__setattr__(self, "outline_color", outline)
        thickness = float(self.outline_thickness)
        if not np.isfinite(thickness) or thickness < 0.0:
            raise ValueError("TextLabelPayload outline_thickness must be finite and non-negative")
        object.__setattr__(self, "outline_thickness", thickness)


@dataclass(frozen=True)
class MaterialPayload:
    """Backend-neutral material payload."""

    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    color_multiplier: tuple[float, float, float] = (1.0, 1.0, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0
    reflectance: float = 0.5
    shader: str = "lit"
    line_width: Optional[float] = None
    point_size: Optional[float] = None
    texture_path: Optional[str] = None
    # Advanced PBR fields supported by both renderers.
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.0
    anisotropy: float = 0.0
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_intensity: float = 0.0
    # Filament transmission fields. Renderers without an equivalent retain
    # the values but do not apply them.
    transmission: float = 0.0
    # Volumetric glass thickness for the Filament transmission shader.
    # Named ``glass_thickness`` to avoid colliding with the XML-derived
    # physical-material ``thickness`` metadata that flows through scene I/O
    # and is displayed in the materials panel as a read-only label.
    glass_thickness: float = 0.0
    # Tint that transmitted light picks up as it passes through the
    # material — Filament ``MaterialRecord.absorption_color``. Default
    # white means no tint. Pygfx has no equivalent in 0.15.3 and does not
    # apply the value.
    absorption_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # ``texture_path`` is the albedo map. The remaining maps add surface detail
    # without changing geometry; box projection supplies UVs when an asset has
    # no authored UV layout.
    normal_map_path: Optional[str] = None
    # Scalar multiplier for normal-map relief. 1.0 keeps the authored map
    # as-is; values above 1 strengthen the apparent embossing while still
    # remaining shading-only (no geometry displacement).
    normal_map_strength: float = 1.0
    roughness_map_path: Optional[str] = None
    ao_map_path: Optional[str] = None
    metallic_map_path: Optional[str] = None
    # Target tile size in world meters for box-projected UVs. E.g. 2.0
    # means the texture tiles once every 2 m along each axis.
    uv_scale_meters: float = 2.0
    # Optional pygfx TextureMap repeat multiplier for authored UV assets.
    # ``None`` keeps the source UV layout unchanged.
    uv_repeat_scale: Optional[tuple[float, float]] = None
    # Shader variant override. ``"defaultLitSSR"`` opts a material into
    # Filament's screen-space reflection shader on the Open3D renderer.
    # Pygfx 0.15.3 has no equivalent and does not apply this value.
    shader_variant: Optional[str] = None

    @property
    def has_advanced_pbr(self) -> bool:
        """Return whether pygfx needs its heavier physical-material shader.

        Used by the pygfx renderer to decide between MeshStandardMaterial
        (fast path) and MeshPhysicalMaterial (heavier shader).

        ``anisotropy`` is intentionally excluded: pygfx 0.15.3 can emit a
        ``getTangentFrame`` call without the corresponding function definition
        unless a normal map is bound, which crashes the shader compile.
        Anisotropy is therefore Open3D-renderer-only in this release. The
        Open3D renderer applies it via
        ``MaterialRecord.base_anisotropy`` regardless of this predicate.

        Transmission and glass thickness are also excluded because pygfx has
        no equivalent in 0.15.3.
        """
        return (
            self.clearcoat > 0.0
            or self.clearcoat_roughness > 0.0
            or self.emissive_intensity > 0.0
            or any(c > 0.0 for c in self.emissive_color)
        )


def material_payload_from_mapping(material: MaterialPayload | dict[str, Any]) -> MaterialPayload:
    """Return a neutral material payload from renderer-style material values."""
    if isinstance(material, MaterialPayload):
        return material

    def _rgba(value: Any, *, alpha: Any = None) -> tuple[float, float, float, float]:
        """Coerce color-like values to renderer-normalized RGBA floats."""
        try:
            values = list(value)
        except TypeError:
            values = [1.0, 1.0, 1.0, 1.0]
        if len(values) < 3:
            values = [1.0, 1.0, 1.0, 1.0]
        resolved_alpha = values[3] if len(values) >= 4 else alpha
        if resolved_alpha is None:
            resolved_alpha = 1.0
        return (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(resolved_alpha),
        )

    def _rgb(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
        """Coerce color-like values to RGB floats with a typed fallback."""
        try:
            values = list(value)
        except TypeError:
            return default
        if len(values) < 3:
            return default
        return (float(values[0]), float(values[1]), float(values[2]))

    def _opt_float(key: str) -> Optional[float]:
        """Read an optional numeric material field without inventing defaults."""
        value = material.get(key)
        return float(value) if value is not None else None

    def _opt_str(key: str) -> Optional[str]:
        """Read an optional path or shader field as a string when present."""
        value = material.get(key)
        return str(value) if value is not None else None

    uv_repeat = material.get("uv_repeat_scale")
    uv_repeat_scale: Optional[tuple[float, float]] = None
    if uv_repeat is not None:
        try:
            uv_repeat_scale = (float(uv_repeat[0]), float(uv_repeat[1]))
        except (TypeError, ValueError, IndexError):
            uv_repeat_scale = None

    color = material.get("base_color", material.get("color", [1.0, 1.0, 1.0, 1.0]))
    return MaterialPayload(
        base_color=_rgba(color, alpha=material.get("alpha")),
        color_multiplier=_rgb(
            material.get("color_multiplier", (1.0, 1.0, 1.0)),
            (1.0, 1.0, 1.0),
        ),
        roughness=float(material.get("roughness", 0.5)),
        metallic=float(material.get("metallic", 0.0)),
        reflectance=float(material.get("reflectance", 0.5)),
        shader=str(material.get("shader", "lit")),
        line_width=_opt_float("line_width"),
        point_size=_opt_float("point_size"),
        texture_path=_opt_str("texture_path"),
        clearcoat=float(material.get("clearcoat", 0.0)),
        clearcoat_roughness=float(material.get("clearcoat_roughness", 0.0)),
        anisotropy=float(material.get("anisotropy", 0.0)),
        emissive_color=_rgb(material.get("emissive_color", (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0)),
        emissive_intensity=float(material.get("emissive_intensity", 0.0)),
        transmission=float(material.get("transmission", 0.0)),
        glass_thickness=float(material.get("glass_thickness", 0.0)),
        absorption_color=_rgb(
            material.get("absorption_color", (1.0, 1.0, 1.0)),
            (1.0, 1.0, 1.0),
        ),
        normal_map_path=_opt_str("normal_map_path"),
        normal_map_strength=float(material.get("normal_map_strength", 1.0)),
        roughness_map_path=_opt_str("roughness_map_path"),
        ao_map_path=_opt_str("ao_map_path"),
        metallic_map_path=_opt_str("metallic_map_path"),
        uv_scale_meters=float(material.get("uv_scale_meters", 2.0)),
        uv_repeat_scale=uv_repeat_scale,
        shader_variant=_opt_str("shader_variant"),
    )


GeometryPayload = Union[MeshPayload, LineSetPayload, PointCloudPayload, OrientationFramePayload]
RenderPayload = Union[GeometryPayload, TextLabelPayload]
