"""Renderer-neutral inventory and explicit clearing for static asset caches.

Frame, ViewModel, and renderer-frame caches are transient visualizer state and
remain owned by :class:`CacheService`.  This module coordinates reusable asset
layers with longer lifetimes: neutral scene payloads, generated UVs, decoded
pixels, backend preparation plans, and native texture objects.  Merely polling
inventory never clears a layer; deletion happens only through the explicit
clear function below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from shared.logging import get_logger

logger = get_logger("orchav.asset_cache_coordinator")


class AssetCacheStorage(str, Enum):
    """Physical ownership class for one asset-cache layer."""

    MEMORY = "memory"
    DISK = "disk"
    NATIVE = "native"


@dataclass(frozen=True, slots=True)
class AssetCacheLayer:
    """One cache layer with truthful, optional entry and byte budgets."""

    key: str
    label: str
    storage: AssetCacheStorage
    entries: int | None = None
    bytes: int | None = None
    max_entries: int | None = None
    max_bytes: int | None = None
    root: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly diagnostic record."""
        return {
            "label": self.label,
            "storage": self.storage.value,
            "entries": self.entries,
            "bytes": self.bytes,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "root": self.root,
        }


@dataclass(frozen=True, slots=True)
class StaticAssetCacheSnapshot:
    """Inventory snapshot across every reusable asset-cache owner."""

    layers: tuple[AssetCacheLayer, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return per-layer records plus storage-class aggregates."""
        aggregates: dict[str, dict[str, int]] = {}
        for storage in AssetCacheStorage:
            selected = tuple(layer for layer in self.layers if layer.storage is storage)
            known_entries = tuple(layer.entries for layer in selected if layer.entries is not None)
            known_bytes = tuple(layer.bytes for layer in selected if layer.bytes is not None)
            known_max_bytes = tuple(
                layer.max_bytes for layer in selected if layer.max_bytes is not None
            )
            aggregates[storage.value] = {
                "layers": len(selected),
                "entries": sum(known_entries),
                "bytes": sum(known_bytes),
                "max_bytes": sum(known_max_bytes),
                "unknown_entry_layers": sum(layer.entries is None for layer in selected),
                "unknown_byte_layers": sum(layer.bytes is None for layer in selected),
                "unbudgeted_layers": sum(layer.max_bytes is None for layer in selected),
            }
        return {
            "layers": {layer.key: layer.as_dict() for layer in self.layers},
            "aggregate": aggregates,
        }


_CACHE_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError)


def _read_cache_info(name: str, callback: Callable[[], Any]) -> dict[str, Any]:
    """Read one optional owner without breaking the diagnostics panel."""
    try:
        value = callback() or {}
    except _CACHE_ERRORS as exc:
        logger.debug("Unable to inspect %s cache: %s", name, exc)
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    """Normalize a non-negative diagnostic integer when one is available."""
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def collect_static_asset_cache_snapshot(renderer: Any = None) -> StaticAssetCacheSnapshot:
    """Inspect reusable asset caches without importing backend policy into services."""
    from ..materials.texture_assets import get_decoded_texture_cache_info
    from ..scene.io import get_persistent_uv_cache_info, get_scene_payload_cache_info

    layers: list[AssetCacheLayer] = []

    scene = _read_cache_info("scene payload", get_scene_payload_cache_info)
    layers.append(
        AssetCacheLayer(
            key="scene_payload_disk",
            label="Neutral scene payloads",
            storage=AssetCacheStorage.DISK,
            entries=_optional_int(scene.get("entries")),
            bytes=_optional_int(scene.get("bytes")),
            max_bytes=_optional_int(scene.get("max_bytes")),
            root=scene.get("root"),
        )
    )

    uv = _read_cache_info("generated UV", get_persistent_uv_cache_info)
    layers.append(
        AssetCacheLayer(
            key="generated_uv_disk",
            label="Generated UVs",
            storage=AssetCacheStorage.DISK,
            entries=_optional_int(uv.get("entries")),
            bytes=_optional_int(uv.get("bytes")),
            max_bytes=_optional_int(uv.get("max_bytes")),
            root=uv.get("root"),
        )
    )

    decoded = _read_cache_info("decoded texture", get_decoded_texture_cache_info)
    layers.append(
        AssetCacheLayer(
            key="decoded_texture_memory",
            label="Decoded texture pixels",
            storage=AssetCacheStorage.MEMORY,
            entries=_optional_int(decoded.get("entries")),
            bytes=_optional_int(decoded.get("bytes")),
            max_bytes=_optional_int(decoded.get("max_bytes")),
        )
    )

    try:
        from .pygfx_mesh_buffers import get_pygfx_mesh_buffer_cache_info
    except ImportError:
        mesh_buffers: dict[str, Any] = {}
    else:
        mesh_buffers = _read_cache_info(
            "pygfx mesh buffer",
            get_pygfx_mesh_buffer_cache_info,
        )
    memory = mesh_buffers.get("memory", {})
    disk = mesh_buffers.get("disk", {})
    layers.extend(
        (
            AssetCacheLayer(
                key="prepared_mesh_memory",
                label="Prepared mesh buffers",
                storage=AssetCacheStorage.MEMORY,
                entries=_optional_int(memory.get("entries")),
                bytes=_optional_int(memory.get("bytes")),
                max_bytes=_optional_int(memory.get("max_bytes")),
            ),
            AssetCacheLayer(
                key="prepared_mesh_disk",
                label="Prepared mesh plans",
                storage=AssetCacheStorage.DISK,
                entries=_optional_int(disk.get("files")),
                bytes=_optional_int(disk.get("bytes")),
                max_bytes=_optional_int(disk.get("max_bytes")),
                root=mesh_buffers.get("root"),
            ),
        )
    )

    try:
        from ..backends.pygfx_scene_helpers import get_pygfx_native_texture_cache_info
    except ImportError:
        shared_native: dict[str, Any] = {}
    else:
        shared_native = _read_cache_info(
            "shared native texture",
            get_pygfx_native_texture_cache_info,
        )
    layers.append(
        AssetCacheLayer(
            key="shared_native_textures",
            label="Shared native textures",
            storage=AssetCacheStorage.NATIVE,
            entries=_optional_int(shared_native.get("entries")),
            bytes=_optional_int(shared_native.get("bytes")),
            max_bytes=_optional_int(shared_native.get("max_bytes")),
        )
    )

    get_renderer_info = getattr(renderer, "get_native_asset_cache_info", None)
    renderer_native = (
        _read_cache_info("active renderer native", get_renderer_info)
        if callable(get_renderer_info)
        else {}
    )
    layers.append(
        AssetCacheLayer(
            key="renderer_native_textures",
            label="Active renderer native textures",
            storage=AssetCacheStorage.NATIVE,
            entries=_optional_int(renderer_native.get("entries")),
            bytes=_optional_int(renderer_native.get("bytes")),
            max_bytes=_optional_int(renderer_native.get("max_bytes")),
        )
    )
    return StaticAssetCacheSnapshot(tuple(layers))


def _clear_layer(name: str, callback: Callable[[], Any]) -> dict[str, Any]:
    """Run one explicit cache clear while isolating optional owner failures."""
    try:
        result = callback() or {}
    except _CACHE_ERRORS as exc:
        logger.warning("Unable to clear %s cache: %s", name, exc)
        return {"error": str(exc)}
    return dict(result) if isinstance(result, Mapping) else {}


def clear_static_asset_caches(
    renderer: Any = None,
    *,
    include_disk: bool = True,
) -> dict[str, dict[str, Any]]:
    """Explicitly release reusable asset layers and return per-owner counts."""
    from ..materials.texture_assets import clear_decoded_texture_cache

    cleared: dict[str, dict[str, Any]] = {
        "decoded_texture_memory": _clear_layer(
            "decoded texture",
            clear_decoded_texture_cache,
        )
    }

    try:
        from .pygfx_mesh_buffers import clear_pygfx_mesh_buffer_cache
    except ImportError:
        pass
    else:
        cleared["prepared_mesh"] = _clear_layer(
            "prepared mesh",
            lambda: clear_pygfx_mesh_buffer_cache(memory=True, disk=include_disk),
        )

    try:
        from ..backends.pygfx_scene_helpers import clear_pygfx_native_texture_cache
    except ImportError:
        pass
    else:
        cleared["shared_native_textures"] = _clear_layer(
            "shared native texture",
            clear_pygfx_native_texture_cache,
        )

    clear_renderer = getattr(renderer, "clear_native_asset_cache", None)
    if callable(clear_renderer):
        cleared["renderer_native_textures"] = _clear_layer(
            "active renderer native",
            clear_renderer,
        )

    if include_disk:
        from ..scene.io import clear_persistent_uv_cache, clear_scene_payload_cache

        cleared["scene_payload_disk"] = _clear_layer(
            "scene payload",
            clear_scene_payload_cache,
        )
        cleared["generated_uv_disk"] = _clear_layer(
            "generated UV",
            clear_persistent_uv_cache,
        )
    return cleared
