"""Bounded filesystem staging for live scene XML updates.

The live generator owns every staging filename.  Payload validation and atomic
publication happen here; ``GeneratorService`` remains responsible for building
the staged candidate and deciding whether it becomes the active scene.
"""

from __future__ import annotations

import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from shared.logging import get_logger

logger = get_logger(__name__)

LIVE_SCENE_XML_MAX_BYTES = 4 * 1024 * 1024


class LiveSceneXmlStager:
    """Own temporary XML files used by one live generator service."""

    def __init__(self, *, max_payload_bytes: int = LIVE_SCENE_XML_MAX_BYTES) -> None:
        """Create a stager with a positive UTF-8 payload limit."""
        normalized_limit = int(max_payload_bytes)
        if normalized_limit <= 0:
            raise ValueError("max_payload_bytes must be positive")
        self.max_payload_bytes = normalized_limit
        self._service_token = uuid.uuid4().hex
        self._sequence = 0
        self._owned_paths: set[Path] = set()
        self._active_path: Path | None = None

    @property
    def active_path(self) -> Path | None:
        """Return the currently accepted staged XML path, if any."""
        return self._active_path

    def stage(self, xml_payload: str, *, source_scene_path: Path) -> Path:
        """Validate and atomically stage one candidate beside its source XML.

        A sibling file preserves Mitsuba's relative mesh and texture paths. The
        generated hidden filename is unrelated to all client-supplied fields.
        """
        if not isinstance(xml_payload, str) or not xml_payload.strip():
            raise ValueError("Live scene XML payload must not be empty")

        payload_bytes = xml_payload.encode("utf-8")
        if len(payload_bytes) > self.max_payload_bytes:
            raise ValueError(
                "Live scene XML payload is too large: "
                f"{len(payload_bytes)} bytes exceeds {self.max_payload_bytes} bytes"
            )

        try:
            root = ET.fromstring(payload_bytes)
        except ET.ParseError as exc:
            raise ValueError(f"Live scene XML is not well formed: {exc}") from exc
        root_name = str(root.tag).rsplit("}", 1)[-1]
        if root_name != "scene":
            raise ValueError(f"Live scene XML root must be 'scene', got {root_name!r}")

        source = Path(source_scene_path).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Active scene XML is not a file: {source}")

        self._sequence += 1
        candidate = source.parent / (f".orchav-live-{self._service_token}-{self._sequence:06d}.xml")
        partial = candidate.with_suffix(".partial")
        try:
            with partial.open("xb") as stream:
                stream.write(payload_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, candidate)
        except Exception:
            partial.unlink(missing_ok=True)
            candidate.unlink(missing_ok=True)
            raise

        self._owned_paths.add(candidate)
        return candidate

    def accept(self, candidate: Path) -> None:
        """Retain *candidate* as active and remove the preceding staged file."""
        resolved = Path(candidate).resolve()
        if resolved not in self._owned_paths:
            raise ValueError(f"Live scene candidate is not owned by this service: {resolved}")

        previous = self._active_path
        self._active_path = resolved
        if previous is not None and previous != resolved:
            self.discard(previous)

    def discard(self, candidate: Path) -> None:
        """Remove an owned candidate that was not accepted or is no longer active."""
        resolved = Path(candidate).resolve()
        if resolved not in self._owned_paths:
            return
        try:
            resolved.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove live scene staging file %s: %s", resolved, exc)
            return
        self._owned_paths.discard(resolved)
        if self._active_path == resolved:
            self._active_path = None

    def close(self) -> None:
        """Best-effort cleanup of every staging file owned by this service."""
        for path in tuple(self._owned_paths):
            self.discard(path)
