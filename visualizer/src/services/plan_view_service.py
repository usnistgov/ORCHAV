"""Offline schematic environment for the Plan appearance, independent of map tiles."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, Signal

from ..model import RenderObject
from ..scene.street_names import street_graph_path
from ..scene.street_plan import GRID_COLOR, STREET_COLOR, load_street_polylines, schematic_payloads
from ..types.render_payloads import MaterialPayload, MeshPayload


class PlanViewService(QObject):
    changed = Signal()
    streets_ready = Signal(int, object)
    _TRAJECTORY_KINDS = ("tx", "rx", "target")

    def __init__(self, visualizer):
        super().__init__()
        self.viz = visualizer
        self.enabled = False
        self.available = False
        self.status = ""
        self._generation = 0
        self._graph = None
        self._loading = False
        self._streets = None
        self._payloads = None
        self._previous_trajectories = None
        self._bounds = ((-50, -50), (50, 50))
        self.streets_ready.connect(self._prepared)

    def set_scene(self, path):
        self.clear()
        if path:
            try:
                graph, _ = street_graph_path(Path(path))
                self._graph = graph if graph.is_file() else None
            except (OSError, ValueError, TypeError) as exc:
                self.status = f"Street plan unavailable: {exc}"
        self.available = self._graph is not None
        self.changed.emit()

    def _prepare_grid(self):
        """Derive bounds only when Plan is first requested for this scene."""
        extents = []
        for entry in getattr(self.viz, "mesh_entries", ()):
            mesh = entry.get("mesh")
            payload = getattr(mesh, "payload", mesh)
            if isinstance(payload, MeshPayload) and len(payload.vertices):
                xy = payload.vertices[:, :2]
                extents.extend((xy.min(axis=0), xy.max(axis=0)))
        if extents:
            self._bounds = (np.min(extents, axis=0), np.max(extents, axis=0))
        self._payloads = schematic_payloads(self._bounds, self._streets or ())

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled and not self.available:
            return
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if enabled:
            if self._payloads is None:
                self._prepare_grid()
            self.publish()
            coordinator = getattr(self.viz, "trajectory_load_coordinator", None)
            if coordinator is not None and coordinator.supports_source(
                getattr(self.viz, "frame_source", None)
            ):
                self._previous_trajectories = self._trajectory_visibility()
                self._set_trajectory_visibility(dict.fromkeys(self._TRAJECTORY_KINDS, True))
            if self._streets is None and self._graph is not None and not self._loading:
                self._load_streets()
        else:
            self.publish()
            self._restore_trajectories()
        self.changed.emit()

    def _trajectory_visibility(self):
        return {
            kind: bool(getattr(self.viz.app_state, f"show_{kind}_trajectory", False))
            for kind in self._TRAJECTORY_KINDS
        }

    def _set_trajectory_visibility(self, values, *, render=True):
        """Use the normal trajectory loader and keep the Nodes controls in sync."""
        for kind, enabled in values.items():
            checkbox = getattr(self.viz, f"{kind}_trajectory_cb", None)
            if render and checkbox is not None and checkbox.isChecked() != enabled:
                # Use its normal signals, including dependent style controls.
                checkbox.setChecked(enabled)
                continue
            if checkbox is not None:
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(enabled)
            if render:
                getattr(self.viz.ui_controller, f"handle_{kind}_trajectory_toggled")(enabled)
            else:
                self.viz.set_state(**{f"show_{kind}_trajectory": enabled})

    def _restore_trajectories(self, *, render=True):
        if self._previous_trajectories is not None:
            self._set_trajectory_visibility(self._previous_trajectories, render=render)
            self._previous_trajectories = None

    def capture_trajectory_state(self):
        """Keep temporary Plan visibility separate from pre-Plan workspace preferences."""
        if self._previous_trajectories is None:
            return None
        return {
            "previous": dict(self._previous_trajectories),
            "current": self._trajectory_visibility(),
        }

    def restore_trajectory_state(self, data):
        if not self.enabled or self._previous_trajectories is None or not isinstance(data, dict):
            return
        for key in ("previous", "current"):
            values = data.get(key)
            if (
                not isinstance(values, dict)
                or set(values) != set(self._TRAJECTORY_KINDS)
                or any(type(value) is not bool for value in values.values())
            ):
                return
        self._previous_trajectories = dict(data["previous"])
        self._set_trajectory_visibility(data["current"])

    def _load_streets(self):
        generation, graph = self._generation, self._graph
        self._loading = True
        self.status = "Loading saved street plan…"

        def prepare():
            try:
                result = load_street_polylines(graph)
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                result = str(exc)
            try:
                self.streets_ready.emit(generation, result)
            except RuntimeError:
                pass  # Owner closed while this read-only graph load was finishing.

        threading.Thread(target=prepare, name="orchav-street-plan", daemon=True).start()

    def _prepared(self, generation, result):
        if generation != self._generation:
            return
        self._loading = False
        if isinstance(result, str) or not result:
            self._streets = ()
            self.available = False
            detail = result if isinstance(result, str) else "the saved graph has no streets"
            self.status = f"Street plan unavailable: {detail}"
            self.set_enabled(False)
            appearance = getattr(self.viz, "appearance_preset_service", None)
            if appearance is not None:
                appearance.mark_if_custom()
            self.changed.emit()
            return
        self._streets = result
        self.status = "Saved street plan · local X / Y (m)"
        self._payloads = schematic_payloads(self._bounds, self._streets)
        self.publish()
        self.changed.emit()

    def publish(self):
        renderer = getattr(self.viz, "renderer", None)
        if renderer is None:
            return
        # Keep the environment usable while loading or if the graph cannot load.
        active = self.enabled and bool(self._streets)
        if self._payloads is not None:
            for key, payload, color, width in zip(
                ("grid", "streets"), self._payloads, (GRID_COLOR, STREET_COLOR), (1, 1.5)
            ):
                if not len(payload.lines):
                    continue
                rgb = tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
                renderer.ensure_object(
                    RenderObject(
                        id=f"plan:{key}",
                        payload=payload,
                        visibility=active,
                        material=MaterialPayload(
                            shader="unlit", base_color=(*rgb, 1), line_width=width
                        ),
                        metadata={"pickable": False, "depth_write": False},
                    )
                )
        changed = []
        for entry in getattr(self.viz, "mesh_entries", ()):
            if entry.get("_plan_view_visible", True) != (not active):
                entry["_plan_view_visible"] = not active
                changed.append(entry)
        appearance = getattr(self.viz, "object_appearance_service", None)
        if changed and appearance is not None:
            appearance.refresh_object_visibility_batch(changed)

    def clear(self):
        self._generation += 1
        # Scene teardown already retired the trajectory source. Restore intent
        # without restarting workers against a source that is about to close.
        self._restore_trajectories(render=False)
        self.enabled = self.available = self._loading = False
        renderer = getattr(self.viz, "renderer", None)
        if renderer is not None and self._payloads is not None:
            for key in ("grid", "streets"):
                renderer.remove_object(f"plan:{key}")
        for entry in getattr(self.viz, "mesh_entries", ()):
            entry.pop("_plan_view_visible", None)
        self._graph = self._streets = self._payloads = None
        self._bounds = ((-50, -50), (50, 50))
        self.status = ""
        appearance = getattr(self.viz, "appearance_preset_service", None)
        if appearance is not None:
            appearance.mark_if_custom()
        self.changed.emit()

    close = clear
