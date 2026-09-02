"""Stable O(1) indexes for persistent visualizer entry collections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_BUCKET_NAMES = ("mesh_entries", "target_entries", "tx_entries", "rx_entries")
_ENTRY_TYPE_TO_BUCKET = {
    "mesh": "mesh_entries",
    "target": "target_entries",
    "tx": "tx_entries",
    "rx": "rx_entries",
}


def _remember_first(index: dict[Any, int], key: Any, position: int) -> None:
    """Record first-match semantics while tolerating unhashable metadata."""
    if key is None:
        return
    try:
        index.setdefault(key, position)
    except TypeError:
        return


@dataclass(slots=True)
class _BucketIndex:
    """Index snapshot for one visualizer-owned list-like collection."""

    collection: Any
    length: int
    by_identity: dict[int, int]
    by_object_key: dict[Any, int]
    by_object_id: dict[Any, int]
    by_mesh_identity: dict[int, int]
    by_name: dict[Any, int]

    @classmethod
    def build(cls, collection: Any) -> "_BucketIndex":
        """Build all canonical lookup maps in one linear pass."""
        by_identity: dict[int, int] = {}
        by_object_key: dict[Any, int] = {}
        by_object_id: dict[Any, int] = {}
        by_mesh_identity: dict[int, int] = {}
        by_name: dict[Any, int] = {}
        for position, entry in enumerate(collection):
            if not isinstance(entry, dict):
                continue
            by_identity.setdefault(id(entry), position)
            _remember_first(by_object_key, entry.get("object_key"), position)
            _remember_first(by_object_id, entry.get("object_id"), position)
            mesh = entry.get("mesh")
            if mesh is not None:
                by_mesh_identity.setdefault(id(mesh), position)
            _remember_first(by_name, entry.get("name"), position)
        return cls(
            collection=collection,
            length=len(collection),
            by_identity=by_identity,
            by_object_key=by_object_key,
            by_object_id=by_object_id,
            by_mesh_identity=by_mesh_identity,
            by_name=by_name,
        )

    def candidate(self, position: int | None) -> dict[str, Any] | None:
        """Return the current entry at *position*, if the snapshot is usable."""
        if position is None or position < 0 or position >= len(self.collection):
            return None
        candidate = self.collection[position]
        return candidate if isinstance(candidate, dict) else None


class CanonicalEntryIndex:
    """Resolve canonical entries and collection indexes without repeated scans."""

    def __init__(self, visualizer: Any) -> None:
        self._visualizer = visualizer
        self._buckets: dict[str, _BucketIndex] = {}

    def invalidate(self) -> None:
        """Drop snapshots after an in-place collection rewrite."""
        self._buckets.clear()

    @staticmethod
    def _sync_entry_keys(
        snapshot: _BucketIndex,
        position: int,
        entry: dict[str, Any],
    ) -> None:
        """Add stable keys assigned after the collection was first indexed."""
        snapshot.by_identity[id(entry)] = position
        _remember_first(snapshot.by_object_key, entry.get("object_key"), position)
        _remember_first(snapshot.by_object_id, entry.get("object_id"), position)
        mesh = entry.get("mesh")
        if mesh is not None:
            snapshot.by_mesh_identity.setdefault(id(mesh), position)
        _remember_first(snapshot.by_name, entry.get("name"), position)

    def _collection(self, bucket_name: str) -> Any:
        collection = getattr(self._visualizer, bucket_name, None)
        return collection if collection is not None else ()

    def _bucket(self, bucket_name: str) -> _BucketIndex:
        collection = self._collection(bucket_name)
        snapshot = self._buckets.get(bucket_name)
        if (
            snapshot is None
            or snapshot.collection is not collection
            or snapshot.length != len(collection)
        ):
            snapshot = _BucketIndex.build(collection)
            self._buckets[bucket_name] = snapshot
        return snapshot

    def _lookup(
        self,
        bucket_name: str,
        map_name: str,
        key: Any,
        matches: Callable[[dict[str, Any]], bool],
    ) -> dict[str, Any] | None:
        """Read one map and repair a detected same-length in-place rewrite."""
        if key is None:
            return None
        snapshot = self._bucket(bucket_name)
        try:
            position = getattr(snapshot, map_name).get(key)
        except TypeError:
            return None
        candidate = snapshot.candidate(position)
        if candidate is not None and matches(candidate):
            self._sync_entry_keys(snapshot, int(position), candidate)
            return candidate
        # A same-length in-place collection rewrite is invisible to list
        # identity/length checks. Rebuild once on a miss so newly inserted
        # stable keys are discoverable without requiring every list-mutating
        # caller to understand this index's lifecycle.
        self._buckets.pop(bucket_name, None)
        snapshot = self._bucket(bucket_name)
        try:
            position = getattr(snapshot, map_name).get(key)
            candidate = snapshot.candidate(position)
        except TypeError:
            return None
        if candidate is not None and matches(candidate):
            self._sync_entry_keys(snapshot, int(position), candidate)
            return candidate
        return None

    def refresh_entry(
        self,
        entry: dict[str, Any],
        *,
        entry_type: str | None = None,
    ) -> None:
        """Refresh keys assigned to one existing entry without rebuilding a bucket."""
        bucket_name = _ENTRY_TYPE_TO_BUCKET.get(str(entry_type or "").lower())
        bucket_names = (bucket_name,) if bucket_name is not None else _BUCKET_NAMES
        for name in bucket_names:
            snapshot = self._bucket(name)
            position = snapshot.by_identity.get(id(entry))
            if snapshot.candidate(position) is entry:
                self._sync_entry_keys(snapshot, int(position), entry)
                return

    def resolve(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Return the authoritative persistent entry for a transient reference."""
        for bucket_name in _BUCKET_NAMES:
            candidate = self._lookup(
                bucket_name,
                "by_identity",
                id(entry),
                lambda item: item is entry,
            )
            if candidate is not None:
                return candidate

        object_key = entry.get("object_key")
        if object_key is not None:
            for bucket_name in _BUCKET_NAMES:
                candidate = self._lookup(
                    bucket_name,
                    "by_object_key",
                    object_key,
                    lambda item: item.get("object_key") == object_key,
                )
                if candidate is not None:
                    return candidate

        object_id = entry.get("object_id")
        if object_id is not None:
            for bucket_name in _BUCKET_NAMES:
                candidate = self._lookup(
                    bucket_name,
                    "by_object_id",
                    object_id,
                    lambda item: item.get("object_id") == object_id,
                )
                if candidate is not None:
                    return candidate

        mesh = entry.get("mesh")
        if mesh is not None:
            for bucket_name in _BUCKET_NAMES:
                candidate = self._lookup(
                    bucket_name,
                    "by_mesh_identity",
                    id(mesh),
                    lambda item: item.get("mesh") is mesh,
                )
                if candidate is not None:
                    return candidate

        entry_type = str(entry.get("entry_type") or "").lower()
        node_index = entry.get("node_index")
        if entry_type in {"tx", "rx"} and node_index is not None:
            try:
                position = int(node_index)
            except (TypeError, ValueError):
                position = -1
            candidate = self._bucket(_ENTRY_TYPE_TO_BUCKET[entry_type]).candidate(position)
            if candidate is not None:
                return candidate

        name = entry.get("name")
        if name is not None:
            for bucket_name in _BUCKET_NAMES:
                candidate = self._lookup(
                    bucket_name,
                    "by_name",
                    name,
                    lambda item: item.get("name") == name,
                )
                if candidate is not None:
                    return candidate
        return entry

    def index_for_entry(
        self,
        entry: dict[str, Any],
        *,
        entry_type: str | None = None,
    ) -> int:
        """Return an entry's stable position in its persistent collection."""
        bucket_name = _ENTRY_TYPE_TO_BUCKET.get(str(entry_type or "").lower())
        bucket_names = (bucket_name,) if bucket_name is not None else _BUCKET_NAMES
        for name in bucket_names:
            snapshot = self._bucket(name)
            position = snapshot.by_identity.get(id(entry))
            candidate = snapshot.candidate(position)
            if candidate is entry:
                self._sync_entry_keys(snapshot, int(position), entry)
                return int(position)
            self._buckets.pop(name, None)
            snapshot = self._bucket(name)
            position = snapshot.by_identity.get(id(entry))
            if snapshot.candidate(position) is entry:
                self._sync_entry_keys(snapshot, int(position), entry)
                return int(position)
        return -1

    def index_for_collection(self, collection: Any, entry: dict[str, Any]) -> int:
        """Return an index for a known visualizer collection, with a safe fallback."""
        for bucket_name in _BUCKET_NAMES:
            snapshot = self._bucket(bucket_name)
            if snapshot.collection is collection:
                return self.index_for_entry(
                    entry,
                    entry_type=bucket_name.removesuffix("_entries"),
                )
        mesh = entry.get("mesh")
        name = entry.get("name")
        for position, candidate in enumerate(collection or ()):
            if not isinstance(candidate, dict):
                continue
            if candidate is entry or (mesh is not None and candidate.get("mesh") is mesh):
                return position
            if name is not None and candidate.get("name") == name:
                return position
        return -1

    def entry_type(self, entry: dict[str, Any]) -> str | None:
        """Return the persistent bucket type containing *entry*, if any."""
        for entry_type, bucket_name in _ENTRY_TYPE_TO_BUCKET.items():
            if self.index_for_entry(entry, entry_type=entry_type) >= 0:
                return entry_type
        return None
