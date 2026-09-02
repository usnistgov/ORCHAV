"""Shared CPU texture decoding with backend-native resource ownership."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.logging import get_logger

from ..diagnostics.cache_telemetry import record_cache_event, set_cache_inventory

logger = get_logger("orchav.materials.texture_assets")

_DEFAULT_DECODED_TEXTURE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_LOCK = threading.RLock()
_DECODED_TEXTURE_CACHE: OrderedDict[str, "DecodedTextureAsset"] = OrderedDict()
_FAILED_TEXTURE_IDENTITIES: set[str] = set()


@dataclass(frozen=True, slots=True)
class DecodedTextureAsset:
    """Immutable RGBA texture pixels plus a source-revision identity."""

    path: str
    identity: str
    rgba: np.ndarray

    @property
    def nbytes(self) -> int:
        """Return decoded CPU byte ownership."""
        return int(self.rgba.nbytes)


def texture_asset_identity(texture_path: str | os.PathLike[str]) -> tuple[str, Path] | None:
    """Return a canonical path and source-revision identity for a texture."""
    path = Path(texture_path).expanduser().resolve(strict=False)
    try:
        stat = path.stat()
    except OSError:
        return None
    return (
        f"{path}|size={int(stat.st_size)}|mtime_ns={int(stat.st_mtime_ns)}"
        f"|ctime_ns={int(stat.st_ctime_ns)}",
        path,
    )


def _decoded_texture_cache_max_bytes() -> int:
    raw = os.environ.get("ORCHAV_DECODED_TEXTURE_CACHE_MAX_BYTES")
    try:
        return (
            max(0, int(float(raw))) if raw is not None else _DEFAULT_DECODED_TEXTURE_CACHE_MAX_BYTES
        )
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_DECODED_TEXTURE_CACHE_MAX_BYTES


def _cache_nbytes_unlocked() -> int:
    return sum(asset.nbytes for asset in _DECODED_TEXTURE_CACHE.values())


def load_decoded_texture(
    texture_path: str | os.PathLike[str],
) -> DecodedTextureAsset | None:
    """Decode one texture once and return an immutable RGBA asset."""
    identity_result = texture_asset_identity(texture_path)
    if identity_result is None:
        failure_key = str(Path(texture_path).expanduser().resolve(strict=False))
        with _LOCK:
            first_failure = failure_key not in _FAILED_TEXTURE_IDENTITIES
            _FAILED_TEXTURE_IDENTITIES.add(failure_key)
        record_cache_event("texture_decode", "missing")
        if first_failure:
            logger.warning("Texture source does not exist: %s", texture_path)
        return None
    identity, path = identity_result

    with _LOCK:
        cached = _DECODED_TEXTURE_CACHE.get(identity)
        if cached is not None:
            _DECODED_TEXTURE_CACHE.move_to_end(identity)
            record_cache_event("texture_decode", "hit", byte_count=cached.nbytes)
            return cached
        if identity in _FAILED_TEXTURE_IDENTITIES:
            record_cache_event("texture_decode", "negative_hit")
            return None

    start = time.perf_counter()
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        storage = rgba.tobytes(order="C")
        immutable = np.frombuffer(storage, dtype=np.uint8).reshape(rgba.shape)
        asset = DecodedTextureAsset(path=str(path), identity=identity, rgba=immutable)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        with _LOCK:
            _FAILED_TEXTURE_IDENTITIES.add(identity)
        record_cache_event(
            "texture_decode",
            "failure",
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )
        logger.warning("Failed to decode texture '%s': %s", path, exc)
        return None

    max_bytes = _decoded_texture_cache_max_bytes()
    with _LOCK:
        if max_bytes <= 0 or asset.nbytes > max_bytes:
            record_cache_event("texture_decode", "oversize", byte_count=asset.nbytes)
            return asset
        _DECODED_TEXTURE_CACHE[identity] = asset
        _DECODED_TEXTURE_CACHE.move_to_end(identity)
        while _DECODED_TEXTURE_CACHE and _cache_nbytes_unlocked() > max_bytes:
            _old_key, old_asset = _DECODED_TEXTURE_CACHE.popitem(last=False)
            record_cache_event("texture_decode", "eviction", byte_count=old_asset.nbytes)
        entries = len(_DECODED_TEXTURE_CACHE)
        byte_count = _cache_nbytes_unlocked()
    record_cache_event(
        "texture_decode",
        "miss",
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        byte_count=asset.nbytes,
    )
    set_cache_inventory(
        "texture_decode",
        entries=entries,
        byte_count=byte_count,
    )
    return asset


def get_decoded_texture_cache_info() -> dict[str, int]:
    """Return process-local decoded texture cache inventory and budget."""
    with _LOCK:
        entries = len(_DECODED_TEXTURE_CACHE)
        byte_count = _cache_nbytes_unlocked()
        failures = len(_FAILED_TEXTURE_IDENTITIES)
    set_cache_inventory("texture_decode", entries=entries, byte_count=byte_count)
    return {
        "entries": entries,
        "bytes": byte_count,
        "failures": failures,
        "max_bytes": _decoded_texture_cache_max_bytes(),
    }


def clear_decoded_texture_cache() -> dict[str, int]:
    """Release shared decoded pixels and negative-cache state."""
    with _LOCK:
        entries = len(_DECODED_TEXTURE_CACHE)
        byte_count = _cache_nbytes_unlocked()
        _DECODED_TEXTURE_CACHE.clear()
        _FAILED_TEXTURE_IDENTITIES.clear()
    record_cache_event(
        "texture_decode",
        "clear",
        count=entries,
        byte_count=byte_count,
    )
    set_cache_inventory("texture_decode", entries=0, byte_count=0)
    return {"entries": entries, "bytes": byte_count}
