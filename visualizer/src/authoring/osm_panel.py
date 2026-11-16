"""OpenStreetMap controls and transient map selection for Scenario Builder."""

from __future__ import annotations

import html
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..types.render_payloads import MaterialPayload, MeshPayload
from ..widgets.form_layout import AlignedFormLayout
from .osm_jobs import OsmJob, osm_cache_root
from .osm_region import GroundMap, parse_location, validate_bbox
from .viewport_port import HitResult, KeyboardInput, PointerInput, PointerPhase, SceneOverlayAsset


def _rectangle(bounds, z=0):
    w, s, e, n = bounds
    return np.asarray(((w, s, z), (e, s, z), (e, n, z), (w, n, z)), dtype=np.float32)


class OsmScenePanel(QWidget):
    view_changed = Signal()
    map_entered = Signal(object)
    map_left = Signal()
    scene_ready = Signal(object, object)
    build_started = Signal()

    def __init__(self, parent=None, *, job_factory=OsmJob):
        super().__init__(parent)
        self.job_factory = job_factory
        self.active = False
        self.ground = None
        self.bbox = None
        self._texture = None
        self._map_asset = None
        self._drag_start = None
        self._pointer = None
        self._map_job = self._build_job = None
        self._jobs = []
        self._extra_config = {}
        self._loaded_tiles = 0
        self._last_progress_line = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = AlignedFormLayout()
        self.location = QLineEdit("38.8895, -77.0250")
        self.location.setPlaceholderText("Latitude, longitude or OSM #map link")
        self.location.setToolTip(
            "Paste an OpenStreetMap link with #map=zoom/latitude/longitude, or enter latitude, longitude."
        )
        self.zoom = QSpinBox()
        self.zoom.setRange(13, 19)
        self.zoom.setFixedWidth(65)
        self.zoom.setValue(16)
        form.addRow("Location", self.location)
        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.zoom)
        self.show_button = QPushButton("Show map")
        self.show_button.clicked.connect(self.show_map)
        row.addWidget(self.show_button)
        form.addRow("Zoom", controls)
        layout.addLayout(form)
        pan = QHBoxLayout()
        for label, dx, dy in (("←", -1, 0), ("↑", 0, 1), ("↓", 0, -1), ("→", 1, 0)):
            button = QPushButton(label)
            button.setMinimumWidth(0)
            button.setMaximumWidth(48)
            button.setToolTip("Move the map and select a new region")
            button.clicked.connect(lambda _=False, x=dx, y=dy: self.pan(x, y))
            pan.addWidget(button)
        layout.addLayout(pan)
        self.instructions = QLabel("Drag a rectangle on the map, then build the selected region.")
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        self.selection_label = QLabel("No region selected")
        self.selection_label.setWordWrap(True)
        self.selection_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.selection_label)
        self.options_toggle = QCheckBox("Scene options")
        layout.addWidget(self.options_toggle)
        self.options = QWidget()
        options_layout = QVBoxLayout(self.options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        self.checks = {}
        for key, label, default in (
            ("vegetation", "Vegetation", True),
            ("green_vegetation", "Green vegetation", True),
            ("barriers", "Walls and fences", True),
            ("bridges", "Bridges", True),
            ("terrain", "Terrain elevation (extra download)", False),
            ("height_enrichment", "Building heights (extra download)", False),
        ):
            check = QCheckBox(label)
            check.setChecked(default)
            self.checks[key] = check
            options_layout.addWidget(check)
        self.timeout = QSpinBox()
        self.timeout.setRange(30, 600)
        self.timeout.setValue(180)
        self.timeout.setSuffix(" s acquisition limit")
        options_layout.addWidget(self.timeout)
        self.server = QLineEdit()
        self.server.setPlaceholderText("https://overpass-api.de/api (default)")
        self.server.setToolTip(
            "Optional Overpass API endpoint, including /api. Use your own provider for repeated large builds."
        )
        options_layout.addWidget(QLabel("Overpass server"))
        options_layout.addWidget(self.server)
        layout.addWidget(self.options)
        self.options.hide()
        self.options_toggle.toggled.connect(self.options.setVisible)
        self.keep_map = QCheckBox("Keep 2D map as a scene overlay")
        self.keep_map.setChecked(True)
        self.keep_map.toggled.connect(lambda _: self._selection_changed())
        self.keep_map.setToolTip(
            "Save the displayed map with the scenario for offline reference. The map lies on a flat plane; it does not change simulation geometry."
        )
        layout.addWidget(self.keep_map)
        self.build_button = QPushButton("Build selected region")
        self.build_button.clicked.connect(self.build)
        self.build_button.setEnabled(False)
        layout.addWidget(self.build_button)
        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel job")
        self.cancel_button.clicked.connect(self.cancel_jobs)
        self.cancel_button.setEnabled(False)
        self.back_button = QPushButton("Back to scene")
        self.back_button.clicked.connect(self.leave_map)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.back_button)
        layout.addLayout(buttons)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.details_toggle = QCheckBox("Show job details")
        layout.addWidget(self.details_toggle)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(250)
        self.details.setMaximumHeight(160)
        self.details.hide()
        self.details_toggle.toggled.connect(self.details.setVisible)
        layout.addWidget(self.details)
        self.attribution = QLabel(
            '<a href="https://www.openstreetmap.org/copyright">© OpenStreetMap contributors</a>'
        )
        self.attribution.setOpenExternalLinks(True)
        self.attribution.setWordWrap(True)
        layout.addWidget(self.attribution)
        provider_credit = os.environ.get("ORCHAV_OSM_TILE_ATTRIBUTION")
        if provider_credit:
            self.attribution.setText(self.attribution.text() + " · " + html.escape(provider_credit))

    @property
    def building(self):
        return self._build_job is not None

    def sync_config(self, config):
        if self.active or self.building:
            return
        self._extra_config = dict(config or {})
        self.bbox = self.ground = None
        self._selection_changed()
        self.server.setText(str(self._extra_config.get("osm_server") or ""))
        for key, check in self.checks.items():
            check.setChecked(
                bool(self._extra_config.get(key, key not in {"terrain", "height_enrichment"}))
            )
        self.timeout.setValue(int(self._extra_config.get("acquisition_timeout", 180)))
        bbox = self._extra_config.get("bbox")
        center = self._extra_config.get("center")
        if bbox:
            try:
                self.bbox = validate_bbox(bbox)
                s, w, n, e = self.bbox
                self.location.setText(f"{(s + n) / 2:.7f}, {(w + e) / 2:.7f}")
                extent = max(
                    (n - s) * 111320, (e - w) * 111320 * math.cos(math.radians((s + n) / 2))
                )
                self.zoom.setValue(
                    max(
                        13,
                        min(
                            19,
                            int(
                                math.log2(
                                    120225050 * math.cos(math.radians((s + n) / 2)) / (extent * 1.4)
                                )
                            ),
                        ),
                    )
                )
                self.ground = GroundMap((s + n) / 2, (w + e) / 2, self.zoom.value())
                self._selection_changed()
            except (TypeError, ValueError):
                self.bbox = None
        elif center:
            self.location.setText(f"{center[0]}, {center[1]}")

    def _new_job(self, operation, config):
        job = self.job_factory(operation, config, self)
        self._jobs.append(job)
        job.progress.connect(
            lambda text, owner=job: (
                self._show_progress(text) if owner in (self._map_job, self._build_job) else None
            )
        )
        return job

    def _show_progress(self, text):
        self.status.setText(text)
        message = text.split("\nElapsed:", 1)[0]
        if message != self._last_progress_line:
            self.details.appendPlainText(message)
            self._last_progress_line = message

    def show_map(self):
        if self.building:
            return
        try:
            lat, lon, zoom = parse_location(self.location.text())
            if zoom is not None:
                self.zoom.setValue(max(13, min(19, zoom)))
                self.location.setText(f"{lat:.7f}, {lon:.7f}")
            ground = GroundMap(lat, lon, self.zoom.value())
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if self._map_job:
            self._map_job.cancel()
        if self.ground != ground or self.bbox is None:
            self.bbox = ground.default_selection()
        self.ground = ground
        self._texture = self._map_asset = None
        self._loaded_tiles = 0
        self._drag_start = None
        self.active = True
        self._selection_changed()
        self.view_changed.emit()
        self.map_entered.emit(ground.bounds)
        self.status.setText("Loading the visible map… Drag to select the build region.")
        job = self._new_job(
            "map",
            {
                "map": asdict(ground),
                "cache": str(osm_cache_root() / "map-tiles"),
                "tile_url": os.environ.get(
                    "ORCHAV_OSM_TILE_URL", "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                ),
            },
        )
        self._map_job = job
        self._selection_changed()
        job.finished.connect(lambda result, owner=job: self._map_finished(owner, result))
        self.cancel_button.setEnabled(True)
        self.progress_bar.show()
        job.start()

    def _map_finished(self, job, result):
        if job is not self._map_job:
            return
        self._map_job = None
        self._selection_changed()
        self.progress_bar.setVisible(self.building)
        self.cancel_button.setEnabled(self.building)
        if result["event"] == "complete":
            details = result["result"]
            self._texture = details["texture"]
            self._map_asset = None
            count = details["loaded_tiles"]
            self._loaded_tiles = count
            self.status.setText(
                f"Map ready in {result['elapsed_s']:.1f} s. Drag a region, then build."
                if count == 9
                else f"{count}/9 map tiles available. Retry Show map, or build the selected coordinates."
            )
            self.view_changed.emit()
        else:
            self.status.setText(result.get("error", "Map unavailable. Retry Show map."))

    def pan(self, dx, dy):
        if self.ground is None or self.building:
            return
        w, s, e, n = self.ground.bounds
        lat, lon = self.ground.geographic(dx * (e - w) / 3, dy * (n - s) / 3)
        self.location.setText(f"{lat:.7f}, {lon:.7f}")
        self.show_map()

    def _selection_changed(self):
        if self.bbox is None:
            self.build_button.setEnabled(False)
            self.selection_label.setText("No valid region selected")
            return
        s, w, n, e = self.bbox
        width = math.radians(e - w) * 6378137 * math.cos(math.radians((s + n) / 2))
        height = math.radians(n - s) * 6378137
        self.selection_label.setText(
            f"{width:.0f} × {height:.0f} m · {width * height / 1e6:.3f} km²\nS {s:.6f}, W {w:.6f}\nN {n:.6f}, E {e:.6f}"
        )
        self.build_button.setEnabled(
            not self.building and not (self.keep_map.isChecked() and self._map_job is not None)
        )

    def handle_input(self, value, port):
        if not self.active:
            return False
        if (
            isinstance(value, KeyboardInput)
            and value.pressed
            and value.key.lower() in {"escape", "esc"}
        ):
            if self._drag_start is not None:
                self.bbox = self._drag_previous_bbox
                self._selection_changed()
                self.view_changed.emit()
            self._drag_start = None
            port.end_drag()
            self.status.setText("Selection drag cancelled.")
        elif isinstance(value, PointerInput):
            self._pointer = value
            if value.phase is PointerPhase.LEAVE:
                self._drag_start = None
                port.end_drag()
        elif isinstance(value, HitResult) and self._pointer is not None and not self.building:
            pointer = self._pointer
            self._pointer = None
            if pointer.phase is PointerPhase.DOWN and pointer.button == 1:
                self._drag_previous_bbox = self.bbox
                self._drag_start = value.world_position
                port.begin_drag_plane(0.0, value)
            elif self._drag_start is not None and pointer.phase in {
                PointerPhase.MOVE,
                PointerPhase.UP,
            }:
                try:
                    self.bbox = self.ground.selection(self._drag_start, value.world_position)
                    self.status.setText("Release, then build the selected region.")
                    self._selection_changed()
                    self.view_changed.emit()
                except ValueError as exc:
                    self.bbox = None
                    self.selection_label.setText("Invalid region — drag a smaller rectangle")
                    self._selection_changed()
                    self.status.setText(str(exc))
                    self.view_changed.emit()
                if pointer.phase is PointerPhase.UP:
                    self._drag_start = None
                    port.end_drag()
        return True

    def assets(self):
        if self.ground is None:
            return ()
        if self._map_asset is None:
            self._map_asset = SceneOverlayAsset(
                cache_key=f"osm-map:{self.ground}:{self._texture}",
                name="OpenStreetMap selection map",
                payload=MeshPayload(
                    vertices=_rectangle(self.ground.bounds),
                    triangles=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
                    triangle_uvs=np.asarray(((0, 1), (1, 1), (1, 0), (0, 0)), dtype=np.float32),
                ),
                material=MaterialPayload(
                    shader="unlit",
                    base_color=(1, 1, 1, 1),
                    texture_path=self._texture,
                    shader_variant="reference_image",
                ),
            )
        assets = [self._map_asset]
        if self.bbox:
            s, w, n, e = self.bbox
            x1, y1 = self.ground.world(s, w)
            x2, y2 = self.ground.world(n, e)
            thickness = max(1.0, (self.ground.bounds[2] - self.ground.bounds[0]) / 400)
            rectangles = (
                (x1, y1, x2, y1 + thickness),
                (x1, y2 - thickness, x2, y2),
                (x1, y1, x1 + thickness, y2),
                (x2 - thickness, y1, x2, y2),
            )
            vertices = np.concatenate([_rectangle(rect, 0.3) for rect in rectangles])
            faces = np.asarray(
                [
                    (4 * i + a, 4 * i + b, 4 * i + c)
                    for i in range(4)
                    for a, b, c in ((0, 1, 2), (0, 2, 3))
                ],
                dtype=np.int32,
            )
            assets.append(
                SceneOverlayAsset(
                    cache_key=f"osm-selection:{self.bbox}",
                    name="Selected OSM region",
                    payload=MeshPayload(vertices=vertices, triangles=faces),
                    material=MaterialPayload(shader="unlit", base_color=(0.02, 0.5, 1.0, 1.0)),
                )
            )
        return tuple(assets)

    def build(self):
        if self.bbox is None or self.building:
            return
        if self.keep_map.isChecked() and self._map_job is not None:
            self.status.setText(
                "The map is still loading. Wait, or turn off Keep 2D map to build from coordinates."
            )
            return
        if self._map_job:
            self._map_job.cancel()
            self._map_job = None
        config = dict(self._extra_config)
        for key in ("center", "url", "refresh_cache"):
            config.pop(key, None)
        config.update(bbox=list(validate_bbox(self.bbox)), acquisition_timeout=self.timeout.value())
        config.update({key: check.isChecked() for key, check in self.checks.items()})
        config.setdefault("ground_pad_m", 30.0)
        config.setdefault("overpass_timeout", 45)
        server = self.server.text().strip()
        if server:
            from urllib.parse import urlsplit

            parsed = urlsplit(server)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                self.status.setText("Enter an http:// or https:// Overpass server URL.")
                return
            config["osm_server"] = server
        else:
            config.pop("osm_server", None)
        worker_config = dict(config)
        if self.keep_map.isChecked() and self._texture and self._loaded_tiles:
            worker_config["_ground_map"] = {
                "texture": self._texture,
                "map": asdict(self.ground),
                "attribution": self.attribution.text(),
            }
        self.details.clear()
        self._last_progress_line = ""
        job = self._new_job("build", worker_config)
        self._build_job = job
        job.finished.connect(
            lambda result, owner=job, cfg=config: self._build_finished(owner, cfg, result)
        )
        self._selection_changed()
        self.show_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.options.setEnabled(False)
        self.keep_map.setEnabled(False)
        self.progress_bar.show()
        self.status.setText("Starting OSM build… Existing actors keep their local coordinates.")
        self.build_started.emit()
        self.view_changed.emit()
        job.start()

    def _build_finished(self, job, config, result):
        if job is not self._build_job:
            return
        self._build_job = None
        self.options.setEnabled(True)
        self.keep_map.setEnabled(True)
        self.progress_bar.setVisible(self._map_job is not None)
        self.show_button.setEnabled(True)
        self.cancel_button.setEnabled(self._map_job is not None)
        self._selection_changed()
        if result["event"] == "complete":
            statistics = result["result"].get("statistics", {})
            detail = " · ".join(
                f"{statistics[key]} {label}"
                for key, label in (
                    ("meshes", "meshes"),
                    ("layers", "layers"),
                    ("cache_hits", "cache hits"),
                    ("requests", "requests"),
                )
                if key in statistics
            )
            self._show_progress(
                f"Scene built in {result['elapsed_s']:.1f} s. {detail}\nSave to keep this scene."
            )
            for warning in result["result"].get("warnings", []):
                self.details.appendPlainText(warning)
                self.status.setText(self.status.text() + "\n" + warning)
            self.scene_ready.emit(config, Path(result["result"]["scene_root"]))
        else:
            self.status.setText(
                result.get("error", "Build failed. Retry to reuse successful downloads.")
            )
        self.view_changed.emit()

    def cancel_jobs(self):
        for job in (self._map_job, self._build_job):
            if job:
                job.cancel()
        self.status.setText("Cancelling…")
        self.cancel_button.setEnabled(False)

    def leave_map(self):
        if self._map_job:
            self._map_job.cancel()
            self._map_job = None
        self._drag_start = self._pointer = None
        self.active = False
        self.progress_bar.setVisible(self.building)
        self.cancel_button.setEnabled(self.building)
        self.map_left.emit()
        self.view_changed.emit()

    def reset(self):
        self.cancel_jobs()
        self._map_job = self._build_job = None
        self.leave_map()
        self.ground = self.bbox = self._texture = self._map_asset = None
        self._extra_config = {}
        self.status.clear()
        self.details.clear()
        self._loaded_tiles = 0
        self.options.setEnabled(True)
        self.keep_map.setEnabled(True)
        self.selection_label.setText("No region selected")
        self.show_button.setEnabled(True)
        self.build_button.setEnabled(False)

    def close_jobs(self):
        for job in self._jobs:
            job.close()
        self._jobs.clear()
