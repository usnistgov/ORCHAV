"""Bind one staged coverage map to a committed frame-set generation.

Coverage is derived from the same scene run as the MPC frames. This lifecycle
keeps the previous canonical map unchanged while frames are being assembled,
then publishes or removes that map only after the frame manifest commits.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from shared.coverage.schema import (
    COVERAGE_FRAME_GENERATION_ID_ATTR,
    COVERAGE_FRAME_SET_ID_ATTR,
    decode_hdf5_attr,
    validate_coverage_hdf5_contract,
)
from shared.frames.directory_ownership import compact_uuid_token
from shared.frames.manifest import FrameSetManifest
from shared.logging import get_logger

from .coverage_writer import save_coverage_map

logger = get_logger(__name__)


class CoveragePublicationError(RuntimeError):
    """Raised when canonical coverage cannot be published safely."""


def _entry_exists(path: Path) -> bool:
    """Return whether the exact entry exists, including an indirect entry."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CoveragePublicationError(f"Could not inspect coverage path {path}: {exc}") from exc
    return True


def _is_indirect(path: Path) -> bool:
    """Return whether the exact entry is a symbolic link or junction."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction()) if callable(is_junction) else False
    except OSError:
        return True


def _is_owned_coverage_map(path: Path) -> bool:
    """Return whether *path* is a real file using the ORCHAV coverage schema."""
    import h5py

    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or _is_indirect(path):
            return False
        with h5py.File(path, "r") as coverage_file:
            validate_coverage_hdf5_contract(
                int(coverage_file.attrs.get("coverage_schema_version", 0)),
                coverage_file.attrs.get("coverage_storage_layout"),
            )
    except (OSError, TypeError, ValueError):
        return False
    return True


class CoveragePublication:
    """Own staged coverage until its corresponding frame manifest commits."""

    def __init__(self, scenario_configuration: Any, *, generation_id: str) -> None:
        root_value = getattr(scenario_configuration, "root", None)
        if not isinstance(root_value, (str, os.PathLike)):
            raise CoveragePublicationError("Coverage publication requires a scenario root")
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise CoveragePublicationError("Coverage publication requires a generation identity")

        self.scenario_root = Path(root_value).resolve(strict=True)
        if not self.scenario_root.is_dir() or _is_indirect(self.scenario_root):
            raise CoveragePublicationError(
                f"Scenario root must be a real directory: {self.scenario_root}"
            )
        self.destination_directory = self.scenario_root / "coverage"
        self.destination = self.destination_directory / "coverage_maps.h5"
        self.generation_id = generation_id
        self._staging = self.scenario_root / (f".orchav-coverage-s-{compact_uuid_token()}.h5")
        self._staging_owned = False
        self._state = "open"

    @property
    def state(self) -> str:
        """Return the lifecycle state for diagnostics and tests."""
        return self._state

    @property
    def staging_path(self) -> Path:
        """Return the private file used by figures before frame commit."""
        if self._state not in {"open", "staged"}:
            raise RuntimeError(f"Coverage staging is unavailable in state {self._state}")
        return self._staging

    def stage(self, coverage_data: dict[str, Any], scenario_configuration: Any) -> str | None:
        """Write enabled coverage to the private run file."""
        if self._state != "open":
            raise RuntimeError(f"Cannot stage coverage from state {self._state}")
        if _entry_exists(self._staging):
            raise FileExistsError(f"Private coverage staging path exists: {self._staging}")

        try:
            result = save_coverage_map(
                coverage_data,
                scenario_configuration,
                output_path=self._staging,
                frame_generation_id=self.generation_id,
            )
            if result is None:
                self._state = "empty"
                return None
            if _entry_exists(self._staging):
                self._staging_owned = True
            if Path(result) != self._staging or not _is_owned_coverage_map(self._staging):
                raise CoveragePublicationError(
                    "Coverage writer did not create the expected private schema-v2 file"
                )
            self._state = "staged"
            return str(self._staging)
        except BaseException:
            if _entry_exists(self._staging):
                self._staging_owned = True
            self.abort()
            raise

    def _bind_frame_set(self, manifest: FrameSetManifest) -> None:
        """Complete the staged identity after the frame manifest is known."""
        import h5py

        with h5py.File(self._staging, "r+") as coverage_file:
            validate_coverage_hdf5_contract(
                int(coverage_file.attrs.get("coverage_schema_version", 0)),
                coverage_file.attrs.get("coverage_storage_layout"),
            )
            staged_generation = decode_hdf5_attr(
                coverage_file.attrs.get(COVERAGE_FRAME_GENERATION_ID_ATTR)
            )
            if staged_generation != manifest.generation_id:
                raise CoveragePublicationError(
                    "Staged coverage generation does not match the committed frame manifest"
                )
            coverage_file.attrs[COVERAGE_FRAME_SET_ID_ATTR] = manifest.frame_set_id
            coverage_file.flush()

    def _prepare_destination(self) -> None:
        if _entry_exists(self.destination_directory):
            if not self.destination_directory.is_dir() or _is_indirect(self.destination_directory):
                raise CoveragePublicationError(
                    "Fixed coverage output must be a real directory: "
                    f"{self.destination_directory}"
                )
        else:
            self.destination_directory.mkdir()

        if _entry_exists(self.destination) and not _is_owned_coverage_map(self.destination):
            raise CoveragePublicationError(
                "Refusing to replace a file not recognized as ORCHAV coverage output: "
                f"{self.destination}"
            )

    def _remove_prior_map(self) -> None:
        if not _entry_exists(self.destination):
            return
        if not _is_owned_coverage_map(self.destination):
            raise CoveragePublicationError(
                "Replacement frames do not include coverage, but the existing fixed "
                f"file is not recognized as ORCHAV coverage output: {self.destination}"
            )
        self.destination.unlink()
        logger.info("Removed coverage map from the prior frame generation: %s", self.destination)

    def finalize(self, manifest: FrameSetManifest) -> Path | None:
        """Publish or remove coverage after the matching frames commit."""
        if self._state == "finalized":
            return self.destination if self.destination.is_file() else None
        if self._state not in {"open", "empty", "staged"}:
            raise RuntimeError(f"Cannot finalize coverage publication from state {self._state}")
        if not isinstance(manifest, FrameSetManifest):
            raise TypeError("Coverage publication requires a committed FrameSetManifest")
        if manifest.generation_id != self.generation_id:
            self.abort()
            raise CoveragePublicationError(
                "Coverage publication generation does not match the committed frames"
            )

        try:
            if self._state == "staged":
                self._bind_frame_set(manifest)
                self._prepare_destination()
                os.replace(self._staging, self.destination)
                self._staging_owned = False
                self._state = "finalized"
                logger.info(
                    "Published coverage for frame set %s: %s",
                    manifest.frame_set_id[:12],
                    self.destination,
                )
                return self.destination

            self._remove_prior_map()
            self._state = "finalized"
            return None
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Discard only the private coverage file owned by this run."""
        if self._state in {"aborted", "finalized"}:
            return
        if self._staging_owned:
            try:
                self._staging.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove staged coverage %s: %s", self._staging, exc)
            self._staging_owned = False
        self._state = "aborted"


__all__ = ["CoveragePublication", "CoveragePublicationError"]
