"""Persisted statistics cache for scenario frame sets.

The cache stores large numeric arrays in ``npz`` entries and keeps scalar or
small structured metadata in JSON. Packed HDF5 v2 sources are keyed by the
manifest's ``generation_id`` and ``frame_set_id``. Older and non-HDF5 sources
fall back to a stat-based frame fingerprint when those durable identities are
not available.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from shared.logging import get_logger

from ..metrics.scenario_statistics import SCENARIO_STATISTICS_SCHEMA_VERSION
from .base import BaseService

logger = get_logger("orchav.statistics_cache")

_CACHE_IO_LOCKS_GUARD = threading.Lock()
_CACHE_IO_LOCKS: dict[str, threading.Lock] = {}


def _cache_io_lock(cache_path: Path) -> threading.Lock:
    """Return the process-wide lock for one normalized cache destination."""
    key = os.path.normcase(os.path.normpath(str(cache_path)))
    with _CACHE_IO_LOCKS_GUARD:
        return _CACHE_IO_LOCKS.setdefault(key, threading.Lock())


class StatisticsCacheService(BaseService):
    """Persist and reload scenario-level statistics across visualizer sessions."""

    CACHE_SCHEMA_VERSION = 4
    CACHE_FILENAME_PREFIX = ".orchav-stats-"
    _CACHE_PATH_HASH_BYTES = 16
    _ARRAY_KEYS = (
        "path_loss_values",
        "delay_values",
        "aoa_az_values",
        "aoa_el_values",
        "aod_az_values",
        "aod_el_values",
        "frame_indices",
        "mpc_evolution",
        "delay_spread_evolution",
        "direct_path_pair_share_evolution",
        "pair_aggregate_path_gain_db_values",
        "pair_rms_delay_spread_ns_values",
        "strongest_single_path_loss_evolution",
    )
    _ARRAY_DTYPES = {
        "path_loss_values": np.float32,
        "delay_values": np.float32,
        "aoa_az_values": np.float32,
        "aoa_el_values": np.float32,
        "aod_az_values": np.float32,
        "aod_el_values": np.float32,
        "frame_indices": np.int32,
        "mpc_evolution": np.int32,
        "delay_spread_evolution": np.float32,
        "direct_path_pair_share_evolution": np.float32,
        "pair_aggregate_path_gain_db_values": np.float32,
        "pair_rms_delay_spread_ns_values": np.float32,
        "strongest_single_path_loss_evolution": np.float32,
    }
    _INT_KEY_FIELDS = (
        "reflection_order_dist",
        "mpc_type_dist",
        "reflection_order_evolution_per_frame",
        "mpc_type_evolution_per_frame",
    )

    def __init__(self, visualizer: Any) -> None:
        """Bind persisted statistics-cache paths to the active visualizer."""
        super().__init__()
        self.visualizer = visualizer

    def load_cached_stats(
        self,
        scenario: Optional[Any] = None,
        *,
        provider: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        """Return cached statistics if the persisted cache matches the current frame set."""
        t0 = time.perf_counter()
        cache_path = self._cache_path(scenario)
        if cache_path is None or not cache_path.exists():
            logger.debug("Statistics cache load skipped: cache missing")
            return None

        current_cache_key = self._cache_key(scenario=scenario, provider=provider)
        if current_cache_key is None:
            logger.debug("Statistics cache load skipped: frame-set identity unavailable")
            return None

        try:
            with _cache_io_lock(cache_path):
                with np.load(cache_path, allow_pickle=False) as payload:
                    metadata = self._decode_metadata(payload)
                    if int(metadata.get("schema_version", 0)) != self.CACHE_SCHEMA_VERSION:
                        logger.info(
                            "Ignoring outdated statistics cache schema: %s",
                            cache_path,
                        )
                        logger.debug(
                            "Statistics cache schema mismatch after %.1f ms",
                            (time.perf_counter() - t0) * 1000,
                        )
                        return None
                    if metadata.get("cache_key") != current_cache_key:
                        logger.info(
                            "Statistics cache identity mismatch: %s",
                            cache_path,
                        )
                        logger.debug(
                            "Statistics cache identity mismatch after %.1f ms",
                            (time.perf_counter() - t0) * 1000,
                        )
                        return None

                    stats = metadata.get("stats", {})
                    if not isinstance(stats, dict):
                        return None

                    stats = dict(stats)
                    for key in self._ARRAY_KEYS:
                        if key in payload:
                            stats[key] = np.asarray(payload[key])

                    for key in self._INT_KEY_FIELDS:
                        value = stats.get(key)
                        if isinstance(value, dict):
                            stats[key] = {int(k): v for k, v in value.items()}

                    logger.debug(
                        "Loaded statistics cache in %.1f ms",
                        (time.perf_counter() - t0) * 1000,
                    )
                    return stats
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Could not load statistics cache %s: %s", cache_path, exc)
            logger.debug(
                "Statistics cache load failed after %.1f ms",
                (time.perf_counter() - t0) * 1000,
                exc_info=True,
            )
            return None

    def save_cached_stats(
        self,
        stats: dict[str, Any],
        scenario: Optional[Any] = None,
        *,
        provider: Optional[Any] = None,
    ) -> Optional[Path]:
        """Persist statistics for the current scenario and return the cache path."""
        t0 = time.perf_counter()
        cache_path = self._cache_path(scenario)
        cache_key = self._cache_key(scenario=scenario, provider=provider)
        if cache_path is None or cache_key is None:
            logger.debug("Statistics cache save skipped: destination or identity unavailable")
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        stats_payload = {}
        for key, value in stats.items():
            if key in self._ARRAY_KEYS:
                continue
            stats_payload[key] = self._to_jsonable(value)

        metadata = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cache_key": cache_key,
            "stats": stats_payload,
        }

        array_payload = {}
        for key in self._ARRAY_KEYS:
            dtype = self._ARRAY_DTYPES.get(key, np.float32)
            array_payload[key] = np.asarray(stats.get(key, np.array([], dtype=dtype)), dtype=dtype)
        metadata_json = np.array(json.dumps(metadata, separators=(",", ":")))

        tmp_path = self._temporary_cache_path(cache_path)
        try:
            np.savez(tmp_path, metadata_json=metadata_json, **array_payload)
            with _cache_io_lock(cache_path):
                os.replace(tmp_path, cache_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "Could not remove temporary statistics cache %s: %s",
                    tmp_path,
                    exc,
                )
        logger.info("Saved statistics cache: %s", cache_path)
        logger.debug(
            "Saved statistics cache in %.1f ms",
            (time.perf_counter() - t0) * 1000,
        )
        return cache_path

    def _cache_path(self, scenario: Optional[Any] = None) -> Optional[Path]:
        """Return the deterministic mutable-cache sibling for a frame directory."""
        frames_dir = self._frames_dir(scenario)
        if frames_dir is None:
            return None
        try:
            expanded_frames_dir = os.path.expanduser(str(frames_dir))
            absolute_frames_dir = os.path.abspath(os.path.normpath(expanded_frames_dir))
        except (OSError, RuntimeError):
            return None
        normalized_frames_dir = Path(absolute_frames_dir)
        normalized_identity = os.path.normcase(absolute_frames_dir)
        digest = hashlib.sha256(normalized_identity.encode("utf-8")).digest()
        token = (
            base64.urlsafe_b64encode(digest[: self._CACHE_PATH_HASH_BYTES])
            .decode("ascii")
            .rstrip("=")
        )
        return normalized_frames_dir.parent / (f"{self.CACHE_FILENAME_PREFIX}{token}.npz")

    @staticmethod
    def _temporary_cache_path(cache_path: Path) -> Path:
        """Return a unique same-directory path for one atomic cache save."""
        token = secrets.token_urlsafe(16)
        return cache_path.with_name(f"{cache_path.stem}.{token}.tmp.npz")

    def _frames_dir(self, scenario: Optional[Any] = None) -> Optional[Path]:
        """Resolve the frame directory from explicit or visualizer-held scenario state."""
        scenario_obj = scenario or getattr(self.visualizer, "scenario", None)
        if scenario_obj is None:
            return None
        frames_dir = getattr(scenario_obj, "frames_dir", None)
        return None if frames_dir is None else Path(frames_dir)

    def _frame_set_fingerprint(self, scenario: Optional[Any] = None) -> Optional[str]:
        """Hash frame chunk and authoritative-manifest size/mtime metadata."""
        frames_dir = self._frames_dir(scenario)
        if frames_dir is None or not frames_dir.exists():
            return None

        tracked_paths = sorted(frames_dir.glob("mpc_frames_*.h5"))

        if not tracked_paths:
            return None

        manifest_path = frames_dir / "frames_manifest.json"
        if manifest_path.exists():
            tracked_paths.append(manifest_path)

        hasher = hashlib.sha256()
        for path in sorted(tracked_paths, key=lambda p: p.name):
            stat = path.stat()
            hasher.update(path.name.encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    def _cache_key(
        self,
        *,
        scenario: Optional[Any] = None,
        provider: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        """Return the complete identity for one statistics result.

        Manifest identities avoid opening or stat-scanning HDF5 chunks during
        cache lookup. The fallback fingerprint keeps non-HDF5 providers
        functional when they do not expose a durable source identity.
        """

        provider_identity = self._provider_identity(provider)
        if provider_identity is not None:
            generation_id, frame_set_id = provider_identity
            legacy_fingerprint = None
        else:
            generation_id = None
            frame_set_id = None
            legacy_fingerprint = self._frame_set_fingerprint(scenario)
            if legacy_fingerprint is None:
                return None

        return {
            "cache_schema_version": self.CACHE_SCHEMA_VERSION,
            "algorithm_schema_version": SCENARIO_STATISTICS_SCHEMA_VERSION,
            "generation_id": generation_id,
            "frame_set_id": frame_set_id,
            "legacy_frame_set_fingerprint": legacy_fingerprint,
        }

    @staticmethod
    def _provider_identity(provider: Optional[Any]) -> Optional[tuple[str, str]]:
        """Read durable identities without loading a frame projection."""

        if provider is None:
            return None
        candidate = getattr(provider, "provider", None) or provider
        try:
            info = candidate.info
            generation_id = getattr(info, "generation_id", None)
            frame_set_id = getattr(info, "frame_set_id", None)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        if not generation_id or not frame_set_id:
            return None
        return str(generation_id), str(frame_set_id)

    @staticmethod
    def _decode_metadata(payload: Any) -> dict[str, Any]:
        """Decode the JSON metadata entry from an ``np.load`` payload."""
        raw = payload["metadata_json"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        """Convert NumPy values and nested containers into JSON-safe values."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): cls._to_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        return value
