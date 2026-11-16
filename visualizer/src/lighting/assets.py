"""Pinned HDR downloads and derived-cache identities for optional visual lighting.

Discovery and renderer loading are offline. Only ``download_environment`` makes
network requests, after an explicit UI action. Downloads are bounded, checked,
and atomically installed outside the source checkout.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import Callable
from urllib.request import Request, urlopen

import yaml


@dataclass(frozen=True)
class EnvironmentAsset:
    """One version-pinned source and its visual lighting recipe."""

    id: str
    label: str
    source: str
    url: str
    source_page: str
    authors: tuple[str, ...]
    license: str
    size: int
    sha256: str
    radiance_scale: float
    yaw_degrees: float
    lighting: dict[str, float]

    @property
    def cache_key(self) -> str:
        """Invalidate derived images when source bytes or preparation changes."""
        recipe = (self.sha256, self.radiance_scale, self.yaw_degrees)
        digest = hashlib.sha256(json.dumps(recipe).encode()).hexdigest()[:24]
        return f"{self.id}-{digest}"


@lru_cache(maxsize=1)
def environment_catalog() -> dict[str, EnvironmentAsset]:
    """Read the small packaged catalog without inspecting disk assets or networking."""
    data = yaml.safe_load(Path(__file__).with_name("environments.yaml").read_text())
    if data["version"] != 1:
        raise ValueError("Unsupported environment catalog version")
    assets = {}
    for entry in data["environments"]:
        asset = EnvironmentAsset(**{**entry, "authors": tuple(entry["authors"])})
        if not asset.url.startswith("https://") or len(asset.sha256) != 64:
            raise ValueError(f"Invalid environment catalog entry: {asset.id}")
        assets[asset.id] = asset
    return assets


def ibl_cache_dir() -> Path:
    """Return a per-user cache, optionally relocated with ORCHAV_IBL_CACHE_DIR."""
    override = os.environ.get("ORCHAV_IBL_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        return base / "ORCHAV/Cache/ibl"
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/orchav/ibl"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "orchav/ibl"


def source_path(asset: EnvironmentAsset) -> Path:
    """Return the managed original-file location; create nothing."""
    return ibl_cache_dir() / "sources" / f"{asset.source}-{asset.sha256[:16]}.hdr"


@lru_cache(maxsize=64)
def _matches_hash(path: str, size: int, mtime: int, ctime: int, digest: str) -> bool:
    # Stat identity keeps repeated GUI/renderer checks cheap, while modifications
    # invalidate verification. ctime catches replacement with a preserved mtime.
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == digest


def cached_environment(asset: EnvironmentAsset) -> Path | None:
    """Return an intact cached source, or None for missing/corrupt downloads."""
    path = source_path(asset)
    try:
        stat = path.stat()
        if stat.st_size != asset.size:
            return None
        # Some filesystems give rapid in-place writes the same timestamp. Do
        # not memoize verification until the source has been stable for a second.
        verify = _matches_hash
        if time.time_ns() - max(stat.st_mtime_ns, stat.st_ctime_ns) < 1_000_000_000:
            verify = _matches_hash.__wrapped__
        if verify(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, asset.sha256):
            return path
    except OSError:
        return None
    return None


class DownloadCancelled(Exception):
    """An explicit cancellation left the active lighting unchanged."""


def download_environment(
    asset: EnvironmentAsset,
    *,
    cancelled: Event | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Fetch one pinned source; preserve existing files on failure or cancellation.

    Call from a worker, not the GUI thread. A five-second socket timeout bounds
    a stalled read; cancellation is checked between chunks and before install.
    The known file size bounds disk use even when a server sends unexpected data.
    """
    cancel = cancelled or Event()
    if cancel.is_set():
        raise DownloadCancelled()
    existing = cached_environment(asset)
    if existing is not None:
        return existing
    target = source_path(asset)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{asset.id}-", suffix=".part", dir=target.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            request = Request(asset.url, headers={"User-Agent": "ORCHAV/0.2 optional-HDR-download"})
            with urlopen(request, timeout=5) as response:
                count = 0
                digest = hashlib.sha256()
                while True:
                    if cancel.is_set():
                        raise DownloadCancelled()
                    chunk = response.read(min(65536, asset.size - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > asset.size:
                        raise ValueError("Environment download exceeds its expected size")
                    output.write(chunk)
                    digest.update(chunk)
                    if progress:
                        progress(count, asset.size)
                if count != asset.size or digest.hexdigest() != asset.sha256:
                    raise ValueError("Environment download failed size/checksum verification")
        if cancel.is_set():
            raise DownloadCancelled()
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)
