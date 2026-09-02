"""Transactional publication and bounded reuse of fixed generator summaries.

Summary figures are disposable derived output, not part of the authoritative
frame-set manifest.  This module therefore gives ``<scenario>/summary`` its own
small lifecycle: a versioned cache key derived from normalized YAML can skip
unchanged work, regeneration uses one private sibling directory, and the prior
summary is replaced only after every requested figure has been built
successfully.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shared.frames.directory_ownership import ManagedFrameDirectoryLock, compact_uuid_token
from shared.logging import get_logger

logger = get_logger(__name__)

SUMMARY_HASH_MARKER = ".orchav-summary-yaml.sha256"
# Version 2 publishes every requested coverage figure family at every configured
# height and identifies each file by height. Bump this whenever code changes the
# names or required set of files in an otherwise unchanged summary tree.
SUMMARY_OUTPUT_CONTRACT_VERSION = 2


class SummaryPublicationError(RuntimeError):
    """Raised when a requested summary cannot be built or published safely."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _summary_cache_key(yaml_hash: str) -> str:
    """Bind normalized scenario YAML to the summary-output contract version."""
    payload = (f"orchav-summary-output-v{SUMMARY_OUTPUT_CONTRACT_VERSION}\0{yaml_hash}").encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _sensing_summary_requested(configuration: Any) -> bool:
    summary = _mapping(getattr(configuration, "generator_summary", None))
    create = set(summary.get("create", ()) or ())
    sensing = _mapping(getattr(configuration, "sensing", None))
    return (
        bool(summary.get("enabled", False))
        and "sensing" in create
        and bool(sensing.get("enabled", False))
    )


def _summary_products_requested(configuration: Any) -> bool:
    """Return whether this run requests any product below fixed ``summary/``."""
    summary = _mapping(getattr(configuration, "generator_summary", None))
    enabled = bool(summary.get("enabled", False))
    create = set(summary.get("create", ()) or ())
    generator_requested = enabled and bool(
        create
        & {
            "scene2d",
            "scene3d",
            "speed",
            "orientation",
            "angular_velocity",
        }
    )
    coverage = _mapping(getattr(configuration, "coverage_cfg", None))
    coverage_figure = _mapping(_mapping(coverage.get("save")).get("figure"))
    coverage_requested = bool(coverage.get("enabled", False)) and bool(
        coverage_figure.get("enabled", False)
    )
    return generator_requested or _sensing_summary_requested(configuration) or coverage_requested


def _is_indirect(path: Path) -> bool:
    """Return whether the exact output entry is a link or Windows junction."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction()) if callable(is_junction) else False
    except OSError:
        return True


def _entry_exists(path: Path) -> bool:
    """Return whether the exact directory entry exists, including broken links."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SummaryPublicationError(f"Could not inspect summary path {path}: {exc}") from exc
    return True


def _remove_owned_path(path: Path) -> None:
    """Remove a private transaction path without following indirect entries."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode) and not _is_indirect(path):
        shutil.rmtree(path)
    else:
        path.unlink()


class SummaryPublication:
    """Own one requested fixed-summary build and its explicit commit point."""

    def __init__(self, configuration: Any) -> None:
        self.requested = _summary_products_requested(configuration)
        self.sensing_requested = _sensing_summary_requested(configuration)
        root_value = getattr(configuration, "root", None)
        if self.requested and not isinstance(root_value, (str, os.PathLike)):
            raise SummaryPublicationError("Summary publication requires a scenario root")
        self.scenario_root = (
            Path(root_value).resolve() if isinstance(root_value, (str, os.PathLike)) else Path.cwd()
        )
        self.destination = self.scenario_root / "summary"
        summary = _mapping(getattr(configuration, "generator_summary", None))
        self.force = bool(summary.get("force", False))
        hash_value = getattr(configuration, "summary_yaml_hash", None)
        self.yaml_hash = hash_value if isinstance(hash_value, str) else ""
        if self.requested and len(self.yaml_hash) != 64:
            raise SummaryPublicationError(
                "Requested summary is missing its normalized scenario YAML identity"
            )
        self.cache_key = _summary_cache_key(self.yaml_hash)

        token = compact_uuid_token()
        self._staging = self.scenario_root / f".orchav-summary-s-{token}"
        self._backup = self.scenario_root / f".orchav-summary-b-{token}"
        self._lock: ManagedFrameDirectoryLock | None = None
        self._state = "new" if self.requested else "disabled"
        self._staging_owned = False

    @property
    def staging_directory(self) -> Path:
        """Return the private root that requested figure writers must use."""
        if self._state != "open":
            raise RuntimeError("Summary staging is available only during an active build")
        return self._staging

    @property
    def skipped(self) -> bool:
        """Whether a matching versioned cache marker reused the existing tree."""
        return self._state == "skipped"

    @property
    def active(self) -> bool:
        """Whether requested figures should currently write into staging."""
        return self._state == "open"

    def _marker_matches(self) -> bool:
        marker = self.destination / SUMMARY_HASH_MARKER
        try:
            value = marker.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError, UnicodeError):
            return False
        return value == self.cache_key

    def _log_reuse_warning(self) -> None:
        logger.warning(
            "Summary regeneration was automatically skipped because the normalized "
            "scenario YAML and summary-output contract are unchanged. XML, meshes, "
            "textures, target catalogs, Python-scripted objects, and other external "
            "inputs are not tracked; set generator_summary.force: true if any of them "
            "changed."
        )

    def begin(self) -> bool:
        """Reserve private staging and return whether figures must be generated."""
        if self._state == "disabled":
            return False
        if self._state != "new":
            raise RuntimeError(f"Cannot begin summary publication from state {self._state}")
        if not self.scenario_root.is_dir():
            raise SummaryPublicationError(
                f"Scenario root is not a real directory: {self.scenario_root}"
            )
        if _entry_exists(self.destination) and (
            not self.destination.is_dir() or _is_indirect(self.destination)
        ):
            raise SummaryPublicationError(
                f"Fixed summary output must be a real directory: {self.destination}"
            )
        lock = ManagedFrameDirectoryLock(self.destination)
        self._lock = lock
        try:
            lock.acquire()
            if not self.force and self._marker_matches():
                self._state = "skipped"
                self._log_reuse_warning()
                self._release_lock()
                return False
            if _entry_exists(self._staging) or _entry_exists(self._backup):
                raise FileExistsError("Private summary transaction path already exists")
            self._staging.mkdir()
            self._staging_owned = True
            self._state = "open"
            return True
        except BaseException:
            if lock.acquired:
                # We own the summary lock, so this failed requested rebuild
                # must not leave an old matching marker eligible for reuse.
                self._invalidate_marker()
            self.abort()
            raise

    def _release_lock(self) -> None:
        lock = self._lock
        self._lock = None
        if lock is None:
            return
        try:
            lock.release()
        except Exception as exc:
            logger.warning("Could not release summary publication lock: %s", exc)

    def _invalidate_marker(self) -> None:
        try:
            (self.destination / SUMMARY_HASH_MARKER).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not invalidate prior summary cache marker: %s", exc)

    def fail(self) -> None:
        """Discard staged output and ensure the prior summary is retried next time."""
        self.abort()

    def abort(self) -> None:
        """Discard staging and make an interrupted requested rebuild retry."""
        if self._state not in {"new", "open"}:
            return
        if self._state == "open":
            self._invalidate_marker()
        if self._staging_owned:
            try:
                _remove_owned_path(self._staging)
            except OSError as exc:
                logger.warning("Could not remove staged summary %s: %s", self._staging, exc)
            self._staging_owned = False
        self._release_lock()
        self._state = "aborted"

    def _promote(self) -> None:
        moved_previous = False
        try:
            if _entry_exists(self.destination):
                os.replace(self.destination, self._backup)
                moved_previous = True
            os.replace(self._staging, self.destination)
            self._staging_owned = False
        except BaseException as promotion_error:
            try:
                if _entry_exists(self._backup):
                    if _entry_exists(self.destination):
                        os.replace(self.destination, self._staging)
                        self._staging_owned = True
                    os.replace(self._backup, self.destination)
            except BaseException as rollback_error:
                raise SummaryPublicationError(
                    f"Summary promotion failed ({promotion_error}); rollback also failed "
                    f"({rollback_error}). Prior output may remain at {self._backup}."
                ) from promotion_error
            raise SummaryPublicationError(
                f"Summary promotion failed; the prior summary was retained: {promotion_error}"
            ) from promotion_error

        if moved_previous:
            try:
                _remove_owned_path(self._backup)
            except OSError as exc:
                logger.warning(
                    "Published the new summary but could not remove prior backup %s: %s",
                    self._backup,
                    exc,
                )

    def finalize(self) -> None:
        """Write the YAML marker last in staging and replace the complete tree."""
        if self._state in {"disabled", "skipped", "finalized"}:
            return
        if self._state != "open":
            raise RuntimeError(f"Cannot finalize summary publication from state {self._state}")
        try:
            (self._staging / SUMMARY_HASH_MARKER).write_text(
                self.cache_key + "\n",
                encoding="utf-8",
            )
            self._promote()
            self._state = "finalized"
        except BaseException:
            self._invalidate_marker()
            self.abort()
            raise
        finally:
            if self._state == "finalized":
                self._release_lock()


__all__ = [
    "SUMMARY_HASH_MARKER",
    "SUMMARY_OUTPUT_CONTRACT_VERSION",
    "SummaryPublication",
    "SummaryPublicationError",
]
