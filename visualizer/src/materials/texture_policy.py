"""Shared texture policy for visualizer materials.

This module centralizes the rules that decide whether material texture maps are
active, whether the ordinary color picker should be editable, and which base
color factor renderers should receive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

from shared.logging import get_logger

logger = get_logger("orchav.materials.texture_policy")

ALBEDO_MAP_KEY = "texture_path"
TEXTURE_MAP_KEYS = (
    ALBEDO_MAP_KEY,
    "normal_map_path",
    "roughness_map_path",
    "ao_map_path",
    "metallic_map_path",
)

# Internal hand-off marker used when application code has already validated
# texture paths. Renderer conversion still reapplies launch enable/disable
# policy, but it must not repeat filesystem probes for the same material.
TEXTURE_POLICY_RESOLVED_KEY = "_texture_policy_resolved"

_WARNED_TEXTURE_POLICY_MESSAGES: set[str] = set()


@dataclass(frozen=True, slots=True)
class TexturePolicyResult:
    """Resolved texture state for one material/object."""

    textures_enabled: bool
    active_maps: Mapping[str, Optional[str]]
    color_editable: bool
    renderer_base_color: tuple[float, float, float, float]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Detach mutable inputs so one resolution is safe to share."""
        object.__setattr__(
            self,
            "active_maps",
            MappingProxyType(dict(self.active_maps)),
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def active_albedo_path(self) -> Optional[str]:
        """Return the active albedo texture path, if texture policy kept one."""
        return self.active_maps.get(ALBEDO_MAP_KEY)

    @property
    def has_active_maps(self) -> bool:
        """Return whether any renderer texture map survived policy resolution."""
        return any(self.active_maps.get(key) is not None for key in TEXTURE_MAP_KEYS)


def textures_globally_enabled() -> bool:
    """Return True when launch policy allows texture maps.

    Texture maps are opt-in so the default visualizer launch uses scalar PBR
    materials with predictable colors and lower startup cost. Set
    ``ORCHAV_ENABLE_TEXTURES=1`` or pass ``--enable-textures`` to activate
    albedo/detail maps. ``ORCHAV_DISABLE_TEXTURES`` forces maps off.
    """

    if os.environ.get("ORCHAV_DISABLE_TEXTURES") == "1":
        return False
    return os.environ.get("ORCHAV_ENABLE_TEXTURES") == "1"


def apply_texture_launch_policy(
    *,
    enable_textures: bool = False,
    disable_textures: bool = False,
) -> None:
    """Apply mutually exclusive launch flags to the process texture policy."""
    if enable_textures and disable_textures:
        raise ValueError("enable_textures and disable_textures are mutually exclusive")
    if enable_textures:
        os.environ["ORCHAV_ENABLE_TEXTURES"] = "1"
        os.environ.pop("ORCHAV_DISABLE_TEXTURES", None)
    elif disable_textures:
        os.environ["ORCHAV_DISABLE_TEXTURES"] = "1"
        os.environ.pop("ORCHAV_ENABLE_TEXTURES", None)


def _coerce_rgba(color: Any, alpha: Any = None) -> tuple[float, float, float, float]:
    """Coerce material color/alpha inputs into clamped renderer RGBA floats."""
    try:
        r = float(color[0])
        g = float(color[1])
        b = float(color[2])
        if alpha is None and len(color) >= 4:
            alpha_value = float(color[3])
        else:
            alpha_value = float(1.0 if alpha is None else alpha)
    except (TypeError, ValueError, IndexError):
        r, g, b = 0.7, 0.7, 0.7
        alpha_value = float(1.0 if alpha is None else alpha)
    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
        max(0.0, min(1.0, alpha_value)),
    )


@lru_cache(maxsize=1024)
def _texture_path_is_file(path_text: str) -> bool:
    """Return a cached existence check for one normalized texture path."""
    try:
        return Path(path_text).is_file()
    except OSError:
        return False


def clear_texture_path_validation_cache() -> None:
    """Forget texture existence probes after an external asset change."""
    _texture_path_is_file.cache_clear()


def _resolve_texture_path(
    raw_path: Any,
    *,
    base_path: Path | None,
    map_key: str,
    context: str | None,
    validate_paths: bool,
    warnings: list[str],
) -> Optional[str]:
    """Resolve a texture-map path and append policy warnings for missing files."""
    if raw_path is None:
        return None
    path_text = str(raw_path).strip()
    if not path_text:
        return None

    path = Path(path_text).expanduser()
    if base_path is not None and not path.is_absolute():
        path = base_path / path
    try:
        validation_path = str(path.resolve(strict=False))
    except OSError:
        validation_path = str(path.absolute())
    if validate_paths and not _texture_path_is_file(validation_path):
        label = f" for {context}" if context else ""
        warnings.append(f"Ignoring missing texture map '{map_key}'{label}: {path}")
        return None
    return str(path)


def resolve_texture_policy(
    props: Mapping[str, Any] | None,
    *,
    color: Any = None,
    alpha: Any = None,
    textures_enabled: bool | None = None,
    base_path: str | Path | None = None,
    context: str | None = None,
    validate_paths: bool = True,
) -> TexturePolicyResult:
    """Resolve the active texture maps and renderer color factor.

    Rules:
    - globally disabled textures strip every texture map;
    - an active albedo texture locks ordinary color editing and forces white
      renderer base color;
    - detail-only maps keep the material color editable;
    - missing paths are ignored with a warning and fall back to solid material.
    """

    props = props or {}
    enabled = textures_globally_enabled() if textures_enabled is None else bool(textures_enabled)
    rgba = _coerce_rgba(
        color if color is not None else props.get("color", (0.7, 0.7, 0.7)),
        props.get("alpha", alpha) if alpha is None else alpha,
    )

    active_maps: dict[str, Optional[str]] = {key: None for key in TEXTURE_MAP_KEYS}
    warnings: list[str] = []
    base = Path(base_path).expanduser() if base_path is not None else None

    if enabled:
        for key in TEXTURE_MAP_KEYS:
            active_maps[key] = _resolve_texture_path(
                props.get(key),
                base_path=base,
                map_key=key,
                context=context,
                validate_paths=validate_paths,
                warnings=warnings,
            )

    has_albedo = active_maps[ALBEDO_MAP_KEY] is not None
    renderer_base = (1.0, 1.0, 1.0, rgba[3]) if has_albedo else rgba
    return TexturePolicyResult(
        textures_enabled=enabled,
        active_maps=active_maps,
        color_editable=not has_albedo,
        renderer_base_color=renderer_base,
        warnings=tuple(warnings),
    )


def apply_texture_policy_to_props(
    props: Mapping[str, Any] | None,
    *,
    color: Any = None,
    alpha: Any = None,
    textures_enabled: bool | None = None,
    base_path: str | Path | None = None,
    context: str | None = None,
    validate_paths: bool = True,
) -> tuple[dict[str, Any], TexturePolicyResult]:
    """Return a copy of *props* with only active texture-map paths retained."""

    filtered = dict(props or {})
    paths_already_resolved = bool(filtered.pop(TEXTURE_POLICY_RESOLVED_KEY, False))
    policy = resolve_texture_policy(
        filtered,
        color=color if color is not None else filtered.get("color"),
        alpha=alpha if alpha is not None else filtered.get("alpha"),
        textures_enabled=textures_enabled,
        base_path=base_path,
        context=context,
        validate_paths=validate_paths and not paths_already_resolved,
    )
    for key in TEXTURE_MAP_KEYS:
        filtered[key] = policy.active_maps.get(key)
    if paths_already_resolved:
        filtered[TEXTURE_POLICY_RESOLVED_KEY] = True
    return filtered, policy


def warn_for_texture_policy(
    policy: TexturePolicyResult,
    *,
    log: Any = logger,
) -> None:
    """Emit each texture-policy warning at most once per process."""

    for message in policy.warnings:
        if message in _WARNED_TEXTURE_POLICY_MESSAGES:
            continue
        _WARNED_TEXTURE_POLICY_MESSAGES.add(message)
        log.warning(message)
