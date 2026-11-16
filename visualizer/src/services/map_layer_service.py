"""Scene-scoped, on-demand OSM reference map with independent display controls."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..model import RenderObject, Transform
from ..scene.map_reference import ATTRIBUTION, mercator_bounds, read_map, tile_layout
from ..scene.map_tiles import TILE_URL
from ..types.render_payloads import MaterialPayload
from .osm_jobs import OsmJob, osm_cache_root


class MapLayerService(QObject):
    """Load on demand, reject stale jobs, and keep source visibility independent."""

    changed = Signal()
    OBJECT_ID = "geographic:osm_map"

    def __init__(self, visualizer, *, job_factory=OsmJob):
        super().__init__()
        self.viz = visualizer
        self.job_factory = job_factory
        self._job = None
        self._jobs = set()
        self._artifact = None
        self._mesh = None
        self._texture = None
        self._credit = ""
        self.enabled = False
        self.opacity, self.height = 0.85, 0.3
        self.available = False
        self.reason = "This scene has no OSM map georeferencing."
        self.status = ""
        self.metadata = {}
        self.saved_map = None

    def set_scene(self, scene_path):
        self.clear()
        self.opacity, self.height = 0.85, 0.3
        if scene_path:
            root = Path(scene_path).parent
            saved = root / "builder_map" / "map.json"
            self.saved_map = (
                saved if saved.is_file() and saved.with_name("map.png").is_file() else None
            )
            try:
                metadata_path = root / "osm_metadata.json"
                if metadata_path.is_file():
                    self.metadata = json.loads(metadata_path.read_text())
                    tile_layout(mercator_bounds(self.metadata))
                    self.available = True
                    self.reason = ""
            except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
                self.reason = f"Map unavailable: {exc}"
            if self.saved_map:
                self.available, self.reason = True, ""
        from ..renderers.protocol import renderer_capabilities

        if not renderer_capabilities(getattr(self.viz, "renderer", None)).viewport_hud:
            self.available = False
            self.reason = "OSM map display requires the pygfx renderer."
        self.changed.emit()

    def snapshot(self):
        return dict(enabled=self.enabled, opacity=self.opacity, height=self.height)

    def restore(self, values):
        """Restore display settings without changing the workspace's saved camera."""
        if not isinstance(values, dict):
            return
        for key, low, high in (("opacity", 0, 1), ("height", -10000, 10000)):
            value = values.get(key)
            if type(value) in (float, int) and math.isfinite(value) and low <= value <= high:
                setattr(self, key, float(value))
        # Older workspaces used a map-only Plan switch. Keep their map intent;
        # the schematic Plan appearance now has a separate owner.
        self.set_enabled(values.get("enabled") is True or values.get("plan") is True)

    def set_opacity(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Map opacity must be between 0 and 1.")
        self.opacity = value
        self._refresh_display()

    def set_height(self, value):
        value = float(value)
        if not math.isfinite(value) or not -10000 <= value <= 10000:
            raise ValueError("Map height must be between -10000 and 10000 metres.")
        self.height = value
        self._refresh_display()

    def _refresh_display(self):
        """Present interactive edits even when playback and the camera are idle."""
        self.publish()
        renderer = getattr(self.viz, "renderer", None)
        if renderer is not None:
            renderer.request_redraw()
        self.changed.emit()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled and self.available)
        if not self.enabled:
            self._cancel()
            self.status = ""
            self.publish()
        elif self._mesh is not None:
            self.publish()
        elif self._job is None:
            self._load()
        self.changed.emit()

    def _load(self):
        if self.saved_map is not None:
            try:
                self._mesh, self._texture, self._credit = read_map(self.saved_map)
                self.status = "Saved map reference"
                self.publish()
                return
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.status = f"Saved map unavailable: {exc}"
        try:
            # Validate before starting a process; missing coordinates never become (0, 0).
            tile_layout(mercator_bounds(self.metadata))
            credit = ATTRIBUTION
            if os.environ.get("ORCHAV_OSM_TILE_ATTRIBUTION"):
                credit += " · " + os.environ["ORCHAV_OSM_TILE_ATTRIBUTION"]
            job = self.job_factory(
                "scene_map",
                {
                    "metadata": self.metadata,
                    "cache": str(osm_cache_root() / "map-tiles"),
                    "tile_url": os.environ.get("ORCHAV_OSM_TILE_URL", TILE_URL),
                    "attribution": credit,
                },
                self,
            )
            self._job = job
            self._jobs.add(job)
            job.progress.connect(lambda text: self._progress(job, text))
            job.finished.connect(lambda result: self._finished(job, result))
            self.status = "Loading OSM map…"
            job.start()
        except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
            self._failed(str(exc))

    def _progress(self, job, text):
        if job is self._job:
            self.status = text
            self.changed.emit()

    def _failed(self, message):
        self.enabled = False
        self.status = f"Map unavailable: {message}. Toggle the map to retry."
        self.publish()
        self.changed.emit()

    def _finished(self, job, result):
        self._jobs.discard(job)
        if job is not self._job:
            shutil.rmtree(job.root, ignore_errors=True)
            job.deleteLater()
            return
        self._job = None
        try:
            if result.get("event") != "complete":
                raise ValueError(result.get("error", "Map loading failed"))
            detail = result["result"]
            self._mesh, self._texture, self._credit = read_map(detail["map_path"])
            self._artifact = job.root
            self.status = f"{detail['loaded_tiles']} tiles ready · {detail['cached_tiles']} cached"
            self.publish()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            shutil.rmtree(job.root, ignore_errors=True)
            self._failed(str(exc))
        finally:
            job.deleteLater()
        self.changed.emit()

    def publish(self):
        """Reuse one immutable mesh; sliders update only material and transform."""
        renderer = getattr(self.viz, "renderer", None)
        if renderer is None:
            return
        visible = self.enabled and self._mesh is not None
        if self._mesh is not None:
            success = renderer.ensure_object(
                RenderObject(
                    id=self.OBJECT_ID,
                    payload=self._mesh,
                    material=MaterialPayload(
                        shader="unlit",
                        shader_variant="reference_image",
                        texture_path=self._texture,
                        base_color=(1, 1, 1, self.opacity),
                    ),
                    transform=Transform.from_translation(
                        (0, 0, self.height - self._mesh.vertices[0, 2])
                    ),
                    visibility=visible,
                    metadata={
                        "pickable": False,
                        "cast_shadow": False,
                        "receive_shadow": False,
                        "depth_test": True,
                        "depth_write": False,
                        "alpha_mode": "blend",
                    },
                )
            )
            if visible and not success:
                self._failed("The renderer could not display the map")
                return
        attribution = getattr(renderer, "set_map_attribution", None)
        if callable(attribution):
            attribution(self._credit if visible and self.opacity > 0 else "")

    def _cancel(self):
        job, self._job = self._job, None
        if job is not None:
            job.cancel()

    def clear(self):
        self._cancel()
        renderer = getattr(self.viz, "renderer", None)
        if renderer is not None:
            if self._mesh is not None:
                renderer.remove_object(self.OBJECT_ID)
            credit = getattr(renderer, "set_map_attribution", None)
            if callable(credit):
                credit("")
        if self._artifact is not None:
            shutil.rmtree(self._artifact, ignore_errors=True)
        self._artifact = self._mesh = self._texture = None
        self._credit = self.status = ""
        self.enabled = self.available = False
        self.metadata, self.saved_map = {}, None
        self.reason = "This scene has no OSM map georeferencing."
        self.changed.emit()

    def close(self):
        self.clear()
        for job in tuple(self._jobs):
            job.close()
