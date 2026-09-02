"""Typed, bounded ownership for target mesh assets and runtime state.

Target animations commonly reference a directory of PLY files. The cache in
this module keeps only a small working set of those files, identifies each
asset by its canonical source path and file revision, and owns every derived
local-space representation needed by target transforms.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Optional

import numpy as np

from shared.logging import get_logger

from ..diagnostics.cache_telemetry import (
    measure_cache_operation,
    record_cache_event,
    set_cache_inventory,
)
from ..model import RenderObjectState
from ..scene.target_transforms import TargetGeometryMeta
from ..types.render_payloads import MeshPayload

logger = get_logger("orchav.target_asset_cache")

TargetLogicalKey = tuple[str, str]

# The byte budget is the primary memory guard. Sixty-four entries keep common
# short target animations (including the 37-frame Etoile pedestrian) warm
# across playback loops without allowing an unbounded collection of tiny
# assets.
DEFAULT_TARGET_ASSET_CACHE_ENTRIES = 64
DEFAULT_TARGET_ASSET_CACHE_BYTES = 256 * 1024 * 1024
DEFAULT_TARGET_ASSET_PREFETCH_WORKERS = 2
TARGET_ASSET_CACHE_NAMESPACE = "target_assets"


@dataclass(frozen=True, slots=True)
class TargetSourceRevision:
    """Filesystem revision used to reject stale target payloads."""

    size_bytes: int
    modified_ns: int
    changed_ns: int
    file_id: int


@dataclass(frozen=True, slots=True)
class TargetAssetKey:
    """Stable identity for one revision and target-specific binding."""

    canonical_path: str
    revision: TargetSourceRevision
    target_name: str
    mesh_filename: str


@dataclass(frozen=True, slots=True)
class TargetAssetSource:
    """Registered logical target frame and its canonical source path."""

    target_name: str
    mesh_filename: str
    canonical_path: str

    @property
    def logical_key(self) -> TargetLogicalKey:
        """Return the target-name/filename lookup key for this frame."""
        return (self.target_name, self.mesh_filename)

    def resolve(self) -> ResolvedTargetAssetSource:
        """Stat the source and return a revision-bound loading request."""
        source_path = Path(self.canonical_path)
        stat = source_path.stat()
        key = TargetAssetKey(
            canonical_path=os.path.normcase(str(source_path)),
            revision=TargetSourceRevision(
                size_bytes=int(stat.st_size),
                modified_ns=int(stat.st_mtime_ns),
                changed_ns=int(stat.st_ctime_ns),
                file_id=int(getattr(stat, "st_ino", 0)),
            ),
            target_name=self.target_name,
            mesh_filename=self.mesh_filename,
        )
        return ResolvedTargetAssetSource(
            target_name=self.target_name,
            mesh_filename=self.mesh_filename,
            canonical_path=str(source_path),
            key=key,
        )

    @classmethod
    def from_path(cls, target_name: str, path: str | Path) -> TargetAssetSource:
        """Register an absolute path without eagerly statting every frame.

        Directory discovery has already produced the sequence. The concrete
        file revision is intentionally captured only when that frame enters
        the initial load or small look-ahead window.
        """
        # ``Path.resolve(strict=False)`` still calls ``stat()`` on Windows.
        # Sequence registration may contain thousands of frames, so perform
        # lexical canonicalization here and leave the one concrete filesystem
        # revision probe to ``resolve()`` when a frame enters the working set.
        resolved = Path(
            os.path.normpath(
                os.path.abspath(os.path.expanduser(os.fspath(path))),
            )
        )
        return cls(
            target_name=str(target_name),
            mesh_filename=resolved.name,
            canonical_path=str(resolved),
        )


@dataclass(frozen=True, slots=True)
class ResolvedTargetAssetSource:
    """Target source whose filesystem revision has already been captured."""

    target_name: str
    mesh_filename: str
    canonical_path: str
    key: TargetAssetKey

    @property
    def logical_key(self) -> TargetLogicalKey:
        """Return the target-name/filename lookup key."""
        return (self.target_name, self.mesh_filename)


@dataclass(slots=True)
class TargetAsset:
    """One loaded target frame and all authoritative local-space baselines."""

    source: ResolvedTargetAssetSource
    mesh: RenderObjectState
    original_vertices: np.ndarray
    scaled_vertices: np.ndarray
    geometry_meta: TargetGeometryMeta
    has_vertex_texture: bool = False

    @property
    def key(self) -> TargetAssetKey:
        """Return the revision-bound cache key."""
        return self.source.key

    @property
    def logical_key(self) -> TargetLogicalKey:
        """Return the target-name/filename lookup key."""
        return self.source.logical_key

    @property
    def estimated_bytes(self) -> int:
        """Estimate unique NumPy storage retained by this asset."""
        arrays: list[np.ndarray] = [
            np.asarray(self.original_vertices),
            np.asarray(self.scaled_vertices),
        ]
        payload = self.mesh.payload
        if isinstance(payload, MeshPayload):
            for value in (
                payload.vertices,
                payload.triangles,
                payload.normals,
                payload.vertex_colors,
                payload.triangle_uvs,
            ):
                if value is not None:
                    arrays.append(np.asarray(value))

        # Views may expose the same backing allocation. Count each root array
        # once so cache budgeting does not punish zero-copy baselines.
        seen: set[int] = set()
        total = 0
        for array in arrays:
            root = array
            while isinstance(root.base, np.ndarray):
                root = root.base
            identity = id(root)
            if identity in seen:
                continue
            seen.add(identity)
            total += int(root.nbytes)
        return total


@dataclass(frozen=True, slots=True)
class TargetRuntimeState:
    """Last successfully assembled semantic state for one target."""

    position: tuple[float, ...]
    orientation: tuple[float, ...]
    mesh_filename: str
    position_valid: bool
    use_ply_position: bool
    runtime_visible: bool
    scale: float | None = None


TargetAssetLoader = Callable[[ResolvedTargetAssetSource], TargetAsset]


class TargetAssetCache:
    """Thread-safe target source registry, LRU, and runtime-state owner."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_TARGET_ASSET_CACHE_ENTRIES,
        max_bytes: int = DEFAULT_TARGET_ASSET_CACHE_BYTES,
        prefetch_workers: int = DEFAULT_TARGET_ASSET_PREFETCH_WORKERS,
    ) -> None:
        """Create an empty bounded cache with a small neutral-loader pool.

        Two workers let independent animated targets prepare their next mesh
        concurrently. Only neutral parsing runs in this pool; renderer state
        remains confined to the presentation thread. Cache entry and byte
        budgets still bound completed assets, while the worker count bounds
        in-flight payload construction.
        """
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = max(1, int(max_bytes))
        self._assets: OrderedDict[TargetAssetKey, TargetAsset] = OrderedDict()
        self._asset_bytes: dict[TargetAssetKey, int] = {}
        self._logical_to_key: dict[TargetLogicalKey, TargetAssetKey] = {}
        self._sources: dict[TargetLogicalKey, TargetAssetSource] = {}
        self._sequences: dict[str, tuple[TargetLogicalKey, ...]] = {}
        self._runtime_states: dict[str, TargetRuntimeState] = {}
        self._pinned: set[TargetAssetKey] = set()
        self._pending: dict[TargetAssetKey, tuple[int, Future[TargetAsset]]] = {}
        self._generation = 0
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._prefetch_submitted = 0
        self._prefetch_completed = 0
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(prefetch_workers)),
            thread_name_prefix="orchav-target-assets",
        )
        self._publish_inventory_locked()

    @property
    def runtime_states(self) -> dict[str, TargetRuntimeState]:
        """Return the typed runtime-state mapping owned by this cache."""
        return self._runtime_states

    @property
    def max_entries(self) -> int:
        """Return the current entry limit."""
        return self._max_entries

    def configure(
        self, *, max_entries: Optional[int] = None, max_bytes: Optional[int] = None
    ) -> None:
        """Update cache budgets and immediately enforce the new limits."""
        with self._lock:
            if max_entries is not None:
                self._max_entries = max(1, int(max_entries))
            if max_bytes is not None:
                self._max_bytes = max(1, int(max_bytes))
            self._evict_locked()
            self._publish_inventory_locked()

    def register_sequence(
        self,
        target_name: str,
        paths: Sequence[str | Path],
    ) -> tuple[TargetAssetSource, ...]:
        """Register the ordered animation sources for one target."""
        sources = tuple(TargetAssetSource.from_path(target_name, path) for path in paths)
        logical_keys = tuple(source.logical_key for source in sources)
        with self._lock:
            old_keys = set(self._sequences.get(str(target_name), ()))
            new_keys = set(logical_keys)
            for stale_key in old_keys - new_keys:
                self._sources.pop(stale_key, None)
                self._drop_logical_asset_locked(stale_key)
            for source in sources:
                existing = self._sources.get(source.logical_key)
                if existing is not None and existing.canonical_path != source.canonical_path:
                    self._drop_logical_asset_locked(source.logical_key)
                self._sources[source.logical_key] = source
            self._sequences[str(target_name)] = logical_keys
        return sources

    def register_source(self, target_name: str, path: str | Path) -> TargetAssetSource:
        """Register one source discovered outside the initial sequence."""
        source = TargetAssetSource.from_path(target_name, path)
        with self._lock:
            existing = self._sources.get(source.logical_key)
            if existing is not None and existing.canonical_path != source.canonical_path:
                self._drop_logical_asset_locked(source.logical_key)
            self._sources[source.logical_key] = source
            sequence = list(self._sequences.get(source.target_name, ()))
            if source.logical_key not in sequence:
                sequence.append(source.logical_key)
                self._sequences[source.target_name] = tuple(sequence)
        return source

    def source_for(self, target_name: str, mesh_filename: str) -> Optional[TargetAssetSource]:
        """Return a registered logical source without touching the LRU."""
        with self._lock:
            return self._sources.get((str(target_name), str(mesh_filename)))

    def get(self, target_name: str, mesh_filename: str) -> Optional[TargetAsset]:
        """Return a revision-valid asset and promote it in the LRU."""
        logical_key = (str(target_name), str(mesh_filename))
        with self._lock:
            source = self._sources.get(logical_key)
            existing_key = self._logical_to_key.get(logical_key)
            # Explicitly inserted typed assets need not have a registered
            # filesystem source (for example, generated in-memory targets).
            if source is None and existing_key is not None:
                asset = self._assets.get(existing_key)
                if asset is not None:
                    self._assets.move_to_end(existing_key)
                    self._hits += 1
                    record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "hit")
                    return asset
        if source is None:
            with self._lock:
                self._misses += 1
                record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "miss")
            return None
        try:
            resolved = source.resolve()
        except OSError:
            with self._lock:
                self._misses += 1
                record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "miss")
            return None
        with self._lock:
            stale_key = self._logical_to_key.get(logical_key)
            if stale_key is not None and stale_key != resolved.key:
                self._drop_key_locked(stale_key)
            asset = self._assets.get(resolved.key)
            if asset is None:
                self._misses += 1
                record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "miss")
                return None
            self._assets.move_to_end(resolved.key)
            self._hits += 1
            record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "hit")
            return asset

    def get_or_load(
        self,
        target_name: str,
        mesh_filename: str,
        loader: TargetAssetLoader,
    ) -> TargetAsset:
        """Return an asset, awaiting in-flight prefetch before loading inline."""
        asset = self.get(target_name, mesh_filename)
        if asset is not None:
            return asset
        source = self.source_for(target_name, mesh_filename)
        if source is None:
            raise KeyError(f"Unregistered target asset: {target_name}/{mesh_filename}")
        resolved = source.resolve()
        with self._lock:
            generation = self._generation
            ready_asset = self._assets.get(resolved.key)
            if ready_asset is not None:
                # A prefetch callback can publish after the initial ``get``
                # misses but before this lock is acquired. Recheck residency so
                # that narrow race does not parse the same source twice.
                self._assets.move_to_end(resolved.key)
                self._hits += 1
                record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "hit")
                return ready_asset
            pending_record = self._pending.get(resolved.key)
            pending = pending_record[1] if pending_record is not None else None
        if pending is not None:
            try:
                with measure_cache_operation(TARGET_ASSET_CACHE_NAMESPACE, "prefetch_wait"):
                    asset = pending.result()
            except CancelledError as exc:
                raise KeyError(
                    f"Target asset load was invalidated: {target_name}/{mesh_filename}"
                ) from exc
            if not self._put_if_current(asset, generation):
                raise KeyError(f"Target asset load was invalidated: {target_name}/{mesh_filename}")
            return asset
        with measure_cache_operation(TARGET_ASSET_CACHE_NAMESPACE, "load"):
            asset = loader(resolved)
        record_cache_event(
            TARGET_ASSET_CACHE_NAMESPACE,
            "load_payload",
            byte_count=asset.estimated_bytes,
        )
        if not self._put_if_current(asset, generation):
            raise KeyError(f"Target asset load was invalidated: {target_name}/{mesh_filename}")
        return asset

    def put(self, asset: TargetAsset) -> None:
        """Store a complete asset and enforce entry and byte budgets."""
        with self._lock:
            prior_key = self._logical_to_key.get(asset.logical_key)
            if prior_key is not None and prior_key != asset.key:
                self._drop_key_locked(prior_key)
            prior = self._assets.pop(asset.key, None)
            if prior is not None:
                self._bytes -= self._asset_bytes.pop(asset.key, 0)
            asset_bytes = asset.estimated_bytes
            self._assets[asset.key] = asset
            self._asset_bytes[asset.key] = asset_bytes
            self._logical_to_key[asset.logical_key] = asset.key
            self._bytes += asset_bytes
            self._evict_locked()
            self._publish_inventory_locked()

    def touch(self, asset: TargetAsset) -> None:
        """Promote an externally retained active asset when it remains cached."""
        with self._lock:
            if asset.key in self._assets:
                self._assets.move_to_end(asset.key)

    def pin(self, asset: TargetAsset) -> None:
        """Keep a currently displayed asset resident while evicting older frames."""
        with self._lock:
            self._pinned.add(asset.key)
            if self._assets.get(asset.key) is asset:
                self._assets.move_to_end(asset.key)
                return
            self.put(asset)

    def pin_handoff(
        self,
        asset: TargetAsset,
        previous: TargetAsset | None,
    ) -> None:
        """Atomically pin a replacement while releasing the prior live asset.

        A freshly loaded replacement may already have been evicted because the
        current asset was pinned when the byte limit was enforced. Pinning the
        replacement before reinsertion and releasing the old key in the same
        lock lets the cache evict the old frame instead of the frame being
        presented next.
        """
        with self._lock:
            self._pinned.add(asset.key)
            if previous is not None and previous.key != asset.key:
                self._pinned.discard(previous.key)
            self.put(asset)

    def unpin(self, asset: TargetAsset) -> None:
        """Release a no-longer-displayed frame back to normal LRU policy."""
        with self._lock:
            self._pinned.discard(asset.key)
            self._evict_locked()
            self._publish_inventory_locked()

    def prefetch_after(
        self,
        target_name: str,
        mesh_filename: str,
        *,
        count: int,
        loader: TargetAssetLoader,
        direction: int = 1,
    ) -> int:
        """Queue adjacent sequence assets in playback order, including loop wrap."""
        if count <= 0:
            return 0
        with self._lock:
            sequence = self._sequences.get(str(target_name), ())
        current_key = (str(target_name), str(mesh_filename))
        try:
            current_index = sequence.index(current_key)
        except ValueError:
            return 0

        step = -1 if int(direction) < 0 else 1
        candidate_count = min(int(count), max(0, len(sequence) - 1))
        logical_candidates = tuple(
            sequence[(current_index + step * offset) % len(sequence)]
            for offset in range(1, candidate_count + 1)
        )
        submitted = 0
        for logical_key in logical_candidates:
            source = self.source_for(*logical_key)
            if source is None:
                continue
            try:
                resolved = source.resolve()
            except OSError:
                continue
            with self._lock:
                cached = self._assets.get(resolved.key)
                if cached is not None:
                    continue
                if resolved.key in self._pending:
                    continue
                generation = self._generation
                future = self._executor.submit(self._prefetch_load, loader, resolved)
                self._pending[resolved.key] = (generation, future)
                self._prefetch_submitted += 1
                record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "prefetch_submit")
            future.add_done_callback(
                lambda completed, key=resolved.key, generation=generation: self._finish_prefetch(
                    key,
                    generation,
                    completed,
                )
            )
            submitted += 1
        return submitted

    def asset_for_logical_key(self, key: TargetLogicalKey) -> Optional[TargetAsset]:
        """Return the latest loaded record without filesystem validation."""
        with self._lock:
            asset_key = self._logical_to_key.get(key)
            return self._assets.get(asset_key) if asset_key is not None else None

    def logical_keys(self) -> tuple[TargetLogicalKey, ...]:
        """Return a stable snapshot of currently loaded logical keys."""
        with self._lock:
            return tuple(self._logical_to_key)

    def clear(self) -> int:
        """Atomically invalidate sources, assets, transforms, and runtime state."""
        with self._lock:
            removed = len(self._assets) + len(self._runtime_states) + len(self._pending)
            self._generation += 1
            pending = [future for _, future in self._pending.values()]
            self._pending.clear()
            self._assets.clear()
            self._asset_bytes.clear()
            self._logical_to_key.clear()
            self._sources.clear()
            self._sequences.clear()
            self._runtime_states.clear()
            self._pinned.clear()
            self._bytes = 0
            self._publish_inventory_locked()
        for future in pending:
            future.cancel()
        return int(removed)

    def clear_inactive_assets(self) -> dict[str, int]:
        """Release prefetched/inactive meshes while preserving live target state.

        The currently presented asset for each target is pinned.  An explicit
        asset-cache clear may discard every other loaded frame and cancel
        lookahead work, but it must not invalidate source registration,
        transforms, or the mesh already owned by the renderer.
        """
        with self._lock:
            self._generation += 1
            pending = [future for _, future in self._pending.values()]
            pending_count = len(pending)
            self._pending.clear()

            removed_entries = 0
            removed_bytes = 0
            for key in tuple(self._assets):
                if key in self._pinned:
                    continue
                asset = self._assets.pop(key)
                asset_bytes = self._asset_bytes.pop(key, 0)
                removed_entries += 1
                removed_bytes += asset_bytes
                self._bytes -= asset_bytes
                if self._logical_to_key.get(asset.logical_key) == key:
                    self._logical_to_key.pop(asset.logical_key, None)
            self._publish_inventory_locked()

        for future in pending:
            future.cancel()
        if removed_entries or pending_count:
            record_cache_event(
                TARGET_ASSET_CACHE_NAMESPACE,
                "clear_inactive",
                count=removed_entries + pending_count,
                byte_count=removed_bytes,
            )
        return {
            "entries": removed_entries,
            "bytes": removed_bytes,
            "pending": pending_count,
        }

    def close(self) -> None:
        """Invalidate the cache and stop its background loader."""
        self.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def telemetry(self) -> dict[str, int]:
        """Return bounded-cache counters for diagnostics."""
        with self._lock:
            return {
                "entries": len(self._assets),
                "bytes": int(self._bytes),
                "max_entries": int(self._max_entries),
                "max_bytes": int(self._max_bytes),
                "hits": int(self._hits),
                "misses": int(self._misses),
                "evictions": int(self._evictions),
                "pending": len(self._pending),
                "prefetch_submitted": int(self._prefetch_submitted),
                "prefetch_completed": int(self._prefetch_completed),
                "pinned": len(self._pinned),
            }

    def _finish_prefetch(
        self,
        key: TargetAssetKey,
        generation: int,
        future: Future[TargetAsset],
    ) -> None:
        """Publish a completed neutral asset only if its generation is current."""
        try:
            asset = future.result()
        except Exception as exc:  # worker failures are retried synchronously on demand
            with self._lock:
                current = self._pending.get(key)
                if current is not None and current[1] is future:
                    self._pending.pop(key, None)
            logger.debug("Target asset prefetch failed for %s: %s", key.canonical_path, exc)
            record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "prefetch_failure")
            return
        with self._lock:
            current = self._pending.get(key)
            if current is not None and current[1] is future:
                self._pending.pop(key, None)
            if generation != self._generation:
                return
        if not self._put_if_current(asset, generation):
            record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "prefetch_stale")
            return
        with self._lock:
            self._prefetch_completed += 1
            record_cache_event(TARGET_ASSET_CACHE_NAMESPACE, "prefetch_complete")

    @staticmethod
    def _prefetch_load(
        loader: TargetAssetLoader,
        source: ResolvedTargetAssetSource,
    ) -> TargetAsset:
        """Measure neutral background parsing without touching renderer state."""
        with measure_cache_operation(TARGET_ASSET_CACHE_NAMESPACE, "prefetch_load"):
            asset = loader(source)
        record_cache_event(
            TARGET_ASSET_CACHE_NAMESPACE,
            "prefetch_payload",
            byte_count=asset.estimated_bytes,
        )
        return asset

    def _evict_locked(self) -> None:
        """Evict least-recently-used assets until both budgets are satisfied."""
        while self._assets and (
            len(self._assets) > self._max_entries
            or (self._bytes > self._max_bytes and len(self._assets) > 1)
        ):
            evict_key = next(
                (candidate for candidate in self._assets if candidate not in self._pinned),
                None,
            )
            if evict_key is None:
                break
            key = evict_key
            asset = self._assets.pop(key)
            asset_bytes = self._asset_bytes.pop(key, 0)
            self._bytes -= asset_bytes
            if self._logical_to_key.get(asset.logical_key) == key:
                self._logical_to_key.pop(asset.logical_key, None)
            self._evictions += 1
            record_cache_event(
                TARGET_ASSET_CACHE_NAMESPACE,
                "eviction",
                byte_count=asset_bytes,
            )

    def _drop_logical_asset_locked(self, logical_key: TargetLogicalKey) -> None:
        """Remove the current revision for one logical target frame."""
        key = self._logical_to_key.get(logical_key)
        if key is not None:
            self._drop_key_locked(key)

    def _drop_key_locked(self, key: TargetAssetKey) -> None:
        """Remove one concrete revision and its logical index."""
        asset = self._assets.pop(key, None)
        if asset is None:
            return
        self._pinned.discard(key)
        self._bytes -= self._asset_bytes.pop(key, 0)
        if self._logical_to_key.get(asset.logical_key) == key:
            self._logical_to_key.pop(asset.logical_key, None)
        self._publish_inventory_locked()

    def _put_if_current(self, asset: TargetAsset, generation: int) -> bool:
        """Publish a load only while its source registration remains current."""
        with self._lock:
            if generation != self._generation:
                return False
            source = self._sources.get(asset.logical_key)
            if source is None:
                return False
            try:
                current_source = source.resolve()
            except OSError:
                return False
            if current_source.key != asset.key:
                return False
            # RLock makes this validation and publish one transaction relative
            # to scenario invalidation.
            self.put(asset)
            return True

    def _publish_inventory_locked(self) -> None:
        """Publish current bounded residency to shared cache diagnostics."""
        set_cache_inventory(
            TARGET_ASSET_CACHE_NAMESPACE,
            entries=len(self._assets),
            byte_count=max(0, int(self._bytes)),
        )
