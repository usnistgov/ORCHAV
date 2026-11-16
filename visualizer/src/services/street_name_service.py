"""Optional, lazily loaded street annotations owned by the current scene."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from shared.logging import get_logger

from ..model import RenderObject, Transform
from ..renderers.protocol import renderer_capabilities
from ..scene.street_names import load_street_names, street_graph_path
from ..types.render_payloads import MaterialPayload, MeshPayload, TextLabelPayload

logger = get_logger("orchav.street_names")


class StreetNameService:
    """Cache the source and neutral labels until the scene changes.

    The option starts off for each scene. Toggling it never reloads geometry or
    parses the graph again, and no frame-update callback is registered.
    """

    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.enabled = False
        self.graph_path = None
        self.metadata = {}
        self.labels = None
        self.style_mode = "card"
        self._styled_labels = None
        self.error = ""

    @property
    def available(self):
        return bool(self.graph_path and self.graph_path.is_file() and not self.error)

    def set_scene(self, scene_path):
        self.clear()
        if scene_path is not None:
            try:
                self.graph_path, self.metadata = street_graph_path(Path(scene_path))
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                self.error = f"Street metadata is unavailable: {exc}"

    def clear(self):
        renderer = getattr(self.visualizer, "renderer", None)
        for label in self.labels or ():
            if renderer is not None:
                renderer.remove_object(label.id)
        self.enabled = False
        self.graph_path = None
        self.metadata = {}
        self.labels = None
        self._styled_labels = None
        self.error = ""

    def _load(self):
        candidates = load_street_names(self.graph_path, self.metadata)
        positions = np.array([candidate.position for candidate in candidates]).reshape(-1, 3)
        # Drape terrain scenes once at load. Bridge offsets remain relative to
        # that ground; no per-frame ray casting or GIS dependency is needed.
        for entry in getattr(self.visualizer, "mesh_entries", ()):
            if Path(entry.get("rel_path") or "").name != "ground.ply":
                continue
            payload = getattr(entry.get("mesh"), "payload", None)
            if isinstance(payload, MeshPayload) and len(payload.vertices) and len(positions):
                from scipy.spatial import cKDTree

                vertices = np.asarray(payload.vertices)
                _, nearest = cKDTree(vertices[:, :2]).query(positions[:, :2])
                positions[:, 2] += vertices[nearest, 2]
            break
        # Use the actual displayed road height when present. Nearby vertices
        # retain terrain/bridge elevations; compare against the graph's level
        # so coincident ground roads and bridges remain distinct.
        expected_height = positions[:, 2].copy()
        nearest_distance = np.full(len(positions), np.inf)
        for entry in getattr(self.visualizer, "mesh_entries", ()):
            path = Path(entry.get("rel_path") or "")
            if entry.get("geographic_layer") != "roads" and not path.stem.startswith("streets"):
                continue
            payload = getattr(entry.get("mesh"), "payload", None)
            if (
                not isinstance(payload, MeshPayload)
                or not len(payload.vertices)
                or not len(positions)
            ):
                continue
            from scipy.spatial import cKDTree

            vertices = np.asarray(payload.vertices)
            distance, nearest = cKDTree(vertices[:, :2]).query(
                positions[:, :2], k=min(8, len(vertices))
            )
            distance, nearest = distance.reshape(len(positions), -1), nearest.reshape(
                len(positions), -1
            )
            heights = vertices[nearest, 2]
            choice = np.argmin(np.abs(heights - expected_height[:, None]), axis=1)
            row = np.arange(len(positions))
            selected = distance[row, choice] < nearest_distance
            positions[selected, 2] = heights[row, choice][selected] + 0.02
            nearest_distance[selected] = distance[row, choice][selected]
        self.labels = tuple(
            RenderObject(
                id=f"street_name:{index}",
                payload=TextLabelPayload(candidate.name, font_size=0.24, outline_thickness=0.08),
                material=MaterialPayload(base_color=(0.97, 0.98, 1.0, 1.0)),
                transform=Transform.from_translation(position),
                metadata={
                    "pickable": False,
                    "depth_test": True,
                    "depth_write": False,
                    "alpha_mode": "blend",
                    "layout_anchor": tuple(position),
                    "label_anchor": "bottom-center",
                    "street_label": candidate.name,
                },
            )
            for index, (candidate, position) in enumerate(zip(candidates, positions))
        )
        if not self.labels:
            self.error = "No named streets with local coordinates are available in this scene."

    def set_style(self, mode):
        """Retain the street-only appearance across visibility and scene rebuilds."""
        mode = "box" if mode == "callout" else mode
        if mode not in ("plain", "box", "card"):
            raise ValueError(f"Unknown street-name style: {mode}")
        if mode == self.style_mode:
            return True
        self.style_mode = mode
        self._styled_labels = None
        return self.set_enabled(True) if self.enabled else True

    def _styled(self):
        if self._styled_labels is None:
            self._styled_labels = tuple(
                (
                    label
                    if self.style_mode == "plain"
                    else replace(
                        label,
                        payload=replace(label.payload, outline_thickness=0),
                        metadata={
                            **label.metadata,
                            "label_background_rgba": (0.035, 0.065, 0.10, 0.90),
                            "label_card": self.style_mode == "card",
                            "label_anchor_offset": 3 if self.style_mode == "card" else 5,
                        },
                    )
                )
                for label in self.labels or ()
            )
        return self._styled_labels

    def set_enabled(self, enabled):
        renderer = getattr(self.visualizer, "renderer", None)
        if enabled and (
            not self.available or not renderer_capabilities(renderer).screen_space_labels
        ):
            return False
        if enabled and self.labels is None:
            try:
                self._load()
            except (OSError, ValueError, ET.ParseError) as exc:
                self.error = f"Could not read street names: {exc}"
                logger.warning(self.error)
                return False
        self.enabled = bool(enabled and self.labels)
        success = True
        for label in self._styled() if self.labels is not None else ():
            success = (
                bool(renderer.ensure_object(replace(label, visibility=self.enabled))) and success
            )
        return success and (self.enabled == bool(enabled))
