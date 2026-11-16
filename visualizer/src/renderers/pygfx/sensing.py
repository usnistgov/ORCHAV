"""Private sensing annotation adapter for the pygfx renderer.

The sensing layer builder provides neutral point-cloud payloads. This mixin
adds pygfx-only presentation on top of those payloads: marker glyph materials
and screen-space track labels. Geometry ownership stays with the main sensing
layers; renderer-owned annotation labels are tracked separately for cleanup.
"""

from __future__ import annotations

import logging
from typing import Any
from weakref import proxy

import numpy as np

from ...model import RenderObjectState, Transform, make_text_label_state
from ...sensing.runtime import state_value
from ...sensing.selection import matches_detection, select_estimate, selection_scope
from ...services.label_callout_service import label_with_callout
from ...types.render_payloads import MaterialPayload, PointCloudPayload, TextLabelPayload

logger = logging.getLogger(__name__)


class PygfxSensingAdapter:
    """Apply private glyph, label, and HUD details through a pygfx renderer."""

    def __init__(self, renderer: Any) -> None:
        """Store the renderer and initialize adapter-owned semantic caches."""
        self._renderer = proxy(renderer)
        self._sensing_annotation_names: set[str] = set()
        self._sensing_label_states: dict[str, Any] = {}
        self._sensing_marker_specs: dict[str, tuple[Any, ...]] = {}
        self._pick_objects: dict[str, Any] = {}
        self._selection_layers: dict[str, Any] = {}
        self._selection_frame: dict = {}
        self._click_candidate = None
        self._highlight_key = None
        self._highlight_active = False
        self._selection_outline_names: set[str] = set()

    def bind_selection(self, frame: dict, layers: list[Any]) -> None:
        """Attach click handlers once per native object; retain compact row mappings."""
        # Base overlay synchronization can restore materials and color buffers
        # even when the saved layer objects have not changed.
        self._highlight_key = None
        self._selection_frame = frame
        self._selection_layers = {layer.name: layer for layer in layers}
        for name in set(self._pick_objects) - self._selection_layers.keys():
            self._pick_objects.pop(name)
        for layer in layers:
            meta = layer.metadata or {}
            if "track_id" not in meta and "detection_indices" not in meta:
                continue
            obj = self._objects.get(layer.name)
            if obj is None:
                continue
            # Neutral material updates can reset picking while retaining the
            # native object. Restore the flag without registering handlers twice.
            if not obj.material.pick_write:
                obj.material.pick_write = True
            if self._pick_objects.get(layer.name) is obj:
                continue
            obj.add_event_handler(
                self._on_sensing_pointer,
                "pointer_down",
                "pointer_move",
                "pointer_up",
                "pointer_leave",
            )
            self._pick_objects[layer.name] = obj

    def clear_selection_bindings(self) -> None:
        """Forget removed overlays so camera focus cannot use stale geometry."""
        self._clear_selection_outline()
        for obj in self._pick_objects.values():
            remove = getattr(obj, "remove_event_handler", None)
            if callable(remove):
                remove(
                    self._on_sensing_pointer,
                    "pointer_down",
                    "pointer_move",
                    "pointer_up",
                    "pointer_leave",
                )
        self._selection_frame = {}
        self._selection_layers.clear()
        self._pick_objects.clear()
        self._click_candidate = None
        self._highlight_key = None
        self._highlight_active = False

    def retire(self) -> None:
        """Remove all adapter-owned presentation when sensing is retired."""
        self.clear_sensing_scene_annotations()
        self.clear_selection_bindings()
        self._sensing_marker_specs.clear()
        self._clear_hud_overlay("sensing_scene_legend")

    def _on_sensing_pointer(self, event: Any) -> None:
        """Select a short click while leaving camera drag gestures unchanged."""
        from time import monotonic

        name = self._reverse_objects.get(id(event.target))
        layer = self._selection_layers.get(name)
        position = np.array([event.x, event.y])
        if event.type == "pointer_leave":
            self._click_candidate = None
        elif event.type == "pointer_down":
            self._click_candidate = None
            if event.button == 1 and not event.modifiers and layer is not None:
                self._click_candidate = (name, position, monotonic(), self._selection_frame)
        elif event.type == "pointer_move":
            candidate = self._click_candidate
            if candidate is not None and np.linalg.norm(position - candidate[1]) > 5:
                self._click_candidate = None
        elif event.type == "pointer_up":
            candidate, self._click_candidate = self._click_candidate, None
            if (
                candidate is None
                or candidate[0] != name
                or event.button != 1
                or event.modifiers
                or monotonic() - candidate[2] > 0.6
                or np.linalg.norm(position - candidate[1]) > 5
                or candidate[3] is not self._selection_frame
            ):
                return
            meta = layer.metadata or {}
            if "fusion_selection" in meta:
                window = getattr(self.visualizer, "fusion_window", None)
                if window is not None and not window.closed:
                    window.select_fusion(meta["fusion_selection"])
                else:
                    panel = self.visualizer.ui_manager.panels.get("sensing")
                    if panel is not None:
                        panel.select_scene_estimate(meta["fusion_selection"])
                return
            sensing = self._selection_frame["sensing"]
            if "track_id" in meta:
                selection = select_estimate(sensing, track_id=meta["track_id"])
            else:
                vertex = self._picked_vertex_index(event)
                if vertex is None:
                    return
                index = vertex // int(meta.get("pick_vertex_span", 1))
                indices = meta["detection_indices"]
                if index >= len(indices):
                    return
                selection = select_estimate(sensing, detection_index=indices[index])
            panel = self.visualizer.ui_manager.panels.get("sensing")
            if panel is not None:
                panel.select_scene_estimate(selection)

    def highlight_selection(self, frame: dict, layers: list[Any], selection: Any) -> None:
        """Emphasize one identity using small marker buffers; never reload products."""
        from ...sensing.fusion_inspection import FusionSelection

        state = getattr(getattr(self._renderer, "visualizer", None), "app_state", None)
        if not state_value(state, "show_detections") or not state_value(
            state, "highlight_selection"
        ):
            selection = None
        key = (tuple(id(layer) for layer in layers), selection, bool(frame.get("stale")))
        if key == self._highlight_key:
            return
        self._highlight_key = key
        sensing = frame.get("sensing") or {}
        fusion = isinstance(selection, FusionSelection)
        active = (
            any(
                self._same_saved_track((layer.metadata or {}).get("fusion_selection"), selection)
                for layer in layers
            )
            if fusion
            else bool(
                selection is not None
                and sensing
                and selection.scope == selection_scope(sensing)
                and selection.detection_collection
                == sensing.get("_display_detection_collection", "detections")
                and (
                    selection.detection_index is None
                    and selection.track_id is not None
                    or selection.detection_index is not None
                    and selection.acquisition_id
                    == sensing["temporal"]["acquisition"]["acquisition_id"]
                )
            )
        )
        emphasized = active or bool(frame.get("stale"))
        if not emphasized and not self._highlight_active:
            return
        self._highlight_active = emphasized
        points, rings = [], []
        for layer in layers:
            obj = self._objects.get(layer.name)
            if obj is None:
                continue
            meta = layer.metadata or {}
            alpha = 0.45 if frame.get("stale") else 1.0
            selected_track = (
                self._same_saved_track(meta.get("fusion_selection"), selection)
                if fusion
                else active
                and selection.track_id is not None
                and "track_id" in meta
                and "fusion_selection" not in meta
                and meta.get("track_id") == selection.track_id
            )
            if "track_id" in meta and active:
                alpha *= 1.0 if selected_track else 0.18
            if obj.material.opacity != alpha:
                obj.material.opacity = alpha
            selected_rows = [
                not fusion
                and active
                and matches_detection(selection, sensing, index, sensing["detections"][index])
                for index in meta.get("detection_indices", ())
            ]
            if "detection_indices" in meta and isinstance(layer.payload, PointCloudPayload):
                colors = np.array(layer.payload.colors, dtype=np.float32, copy=True)
                if active:
                    for vertex, chosen in enumerate(selected_rows):
                        if not chosen:
                            colors[vertex, 3] *= 0.18
                        else:
                            points.append(layer.payload.points[vertex])
                obj.geometry.colors = self._gfx.Buffer(colors)
            elif any(selected_rows):
                # A range-only detection is a ring, not a unique target point.
                count = len(layer.payload.points) // len(selected_rows)
                for index, chosen in enumerate(selected_rows):
                    if chosen:
                        rings.append(layer.payload.points[index * count : (index + 1) * count])
            if selected_track and isinstance(layer.payload, PointCloudPayload):
                points.extend(layer.payload.points)
        self._show_selection_outline(points, rings)

    @staticmethod
    def _same_saved_track(candidate, selected) -> bool:
        """A saved identity may follow epochs within its run, never another run."""
        return candidate is not None and (candidate.run_id, candidate.track_id, candidate.mode) == (
            selected.run_id,
            selected.track_id,
            selected.mode,
        )

    def _clear_selection_outline(self) -> None:
        for name in self._selection_outline_names:
            self.remove_object(name)
        self._selection_outline_names.clear()

    def _show_selection_outline(self, points, rings) -> None:
        """Gold cues preserve identity colors and the recorded geometry underneath."""
        from ...types.render_payloads import LineSetPayload

        color = (1.0, 0.88, 0.15, 1.0)
        desired = set()
        if points:
            name = "sensing_selection_points"
            desired.add(name)
            state = RenderObjectState(
                id=name,
                payload=PointCloudPayload(points=np.array(points, dtype=np.float32)),
                material=MaterialPayload(base_color=(*color[:3], 0.0), point_size=25.0),
                metadata={"pickable": False},
            )
            self.ensure_object(state.to_render_object())
            # Generic material updates apply the transparent fill as opacity
            # for the whole object. Restore the hollow marker after every sync,
            # including when the same native object moves to another detection.
            self._sensing_marker_specs.pop(name, None)
            self._apply_point_marker_material(
                name, marker_name="circle", point_size=25.0, edge_width=2.5, edge_color=color
            )
            native = self._objects.get(name)
            if native is not None:
                native.material.opacity = 1.0
                native.material.pick_write = False
                native.render_order = 1010
        if rings:
            name = "sensing_selection_range"
            desired.add(name)
            vertices, edges = [], []
            for ring in rings:
                offset = len(vertices)
                vertices.extend(ring)
                edges.extend((offset + i, offset + (i + 1) % len(ring)) for i in range(len(ring)))
            state = RenderObjectState(
                id=name,
                payload=LineSetPayload(
                    points=np.array(vertices, dtype=np.float32),
                    lines=np.array(edges, dtype=np.int32),
                ),
                material=MaterialPayload(base_color=color, line_width=4.0),
                metadata={"pickable": False},
            )
            self.ensure_object(state.to_render_object())
            native = self._objects.get(name)
            if native is not None:
                native.material.pick_write = False
                native.material.depth_test = False
                native.material.depth_write = False
                native.render_order = 1010
        for name in self._selection_outline_names - desired:
            self.remove_object(name)
        self._selection_outline_names = desired

    def refresh_selection(self, selection: Any) -> None:
        """Refresh only presentation after a plot click, without a scene reload."""
        if self._selection_layers:
            self._highlight_key = None
            self.highlight_selection(
                self._selection_frame, list(self._selection_layers.values()), selection
            )
            canvas = getattr(self, "_canvas", None)
            if canvas is not None:
                canvas.request_draw()

    def selected_position(self, selection: Any) -> Any:
        """Return a saved world anchor, leaving range-only detections unresolved."""
        from ...sensing.fusion_inspection import FusionSelection

        if isinstance(selection, FusionSelection):
            for layer in self._selection_layers.values():
                if isinstance(layer.payload, PointCloudPayload) and self._same_saved_track(
                    (layer.metadata or {}).get("fusion_selection"), selection
                ):
                    return layer.payload.points[0]
            return None
        sensing = self._selection_frame.get("sensing")
        if selection is None or not sensing or selection.scope != selection_scope(sensing):
            return None
        for name, layer in self._selection_layers.items():
            if not isinstance(layer.payload, PointCloudPayload):
                continue
            meta = layer.metadata or {}
            if (
                name.startswith("sensing_track_point_")
                and meta.get("track_id") == selection.track_id
            ):
                return layer.payload.points[0]
            for vertex, index in enumerate(meta.get("detection_indices", ())):
                if matches_detection(selection, sensing, index, sensing["detections"][index]):
                    return layer.payload.points[vertex]
        return None

    def __getattr__(self, name: str) -> Any:
        """Delegate renderer operations used by the annotation implementation."""
        return getattr(self._renderer, name)

    def apply_sensing_scene_annotations(
        self,
        raw_frame: dict[str, Any],
        layers: list[Any],
    ) -> None:
        """Apply sensing glyphs and optional track labels for the current frame."""
        app_state = getattr(self.visualizer, "app_state", None)
        if app_state is None or not bool(state_value(app_state, "show_detections")):
            self._sensing_marker_specs.clear()
            self.clear_sensing_scene_annotations()
            return

        if bool(state_value(app_state, "use_marker_glyphs")):
            active = {layer.name for layer in layers}
            self._sensing_marker_specs = {
                name: spec for name, spec in self._sensing_marker_specs.items() if name in active
            }
            self._apply_sensing_marker_glyphs(layers)
        else:
            self._sensing_marker_specs.clear()
            self._reset_sensing_marker_glyphs(layers)

        if bool(state_value(app_state, "show_track_labels")):
            self._update_sensing_track_labels(raw_frame, layers)
        else:
            self.clear_sensing_scene_annotations()

    def clear_sensing_scene_annotations(self) -> None:
        """Remove renderer-owned sensing labels."""
        for name in list(self._sensing_annotation_names):
            self.remove_object(name)
        self._sensing_annotation_names.clear()
        getattr(self, "_sensing_label_states", {}).clear()

    def update_scene_legend(self) -> None:
        """Refresh or clear the private scene legend HUD."""
        html = self._scene_legend_html()
        if html:
            self._set_hud_overlay(
                "sensing_scene_legend",
                html=html,
                visible=True,
                role="legend",
                corner="top_right",
                priority=5,
            )
        else:
            self._clear_hud_overlay("sensing_scene_legend")

    def _scene_legend_html(self) -> str:
        """Build the active private scene legend."""
        state = getattr(self.visualizer, "app_state", None)
        if state is None or not bool(state_value(state, "show_detections")):
            return ""
        mode = str(state_value(state, "scene_mode"))
        arrows = bool(state_value(state, "show_velocity_arrows"))
        glyphs = bool(state_value(state, "use_marker_glyphs"))

        confirmed = "◇ Tracks · trails: history" if glyphs else "Tracks · trails: history"
        tentative = "Muted: tentative · faded: historical"
        if arrows:
            confirmed += " · arrows: velocity"
        detections = "× Detections" if glyphs else "Detections"
        detections += " · rings: range only"
        rows_by_mode = {
            "detections": [detections],
            "tracks": [detections, confirmed, tentative],
            "evidence": [detections, "History cloud"],
            "full": [detections, confirmed, tentative, "History cloud"],
        }
        rows = rows_by_mode.get(mode, rows_by_mode["full"])
        if bool(state_value(state, "show_truth")):
            rows = [*rows, "+ Ground truth" if glyphs else "Ground truth"]
        lines = "".join(f"<tr><td>{row}</td></tr>" for row in rows)
        requirements = ""
        if self._selection_frame.get("stale"):
            timestamp = self._selection_frame.get("result_time_s")
            time_label = (
                f" at {timestamp:.3f} s"
                if isinstance(timestamp, (int, float)) and np.isfinite(timestamp)
                else ""
            )
            requirements = f"<div style='margin-bottom:4px;'>Held result{time_label} (faded)</div>"
        return (
            "<div style='font-weight:600; margin-bottom:4px;'>Sensing</div>"
            f"{requirements}"
            f"<table cellspacing='0' cellpadding='0'>{lines}</table>"
        )

    def _apply_sensing_marker_glyphs(self, layers: list[Any]) -> None:
        """Swap sensing point clouds to pygfx marker materials when available."""
        marker_cls = getattr(self._gfx, "PointsMarkerMaterial", None)
        if marker_cls is None:
            return
        for layer in layers:
            name = str(getattr(layer, "name", ""))
            payload = getattr(layer, "payload", None)
            if not name.startswith("sensing_") or not isinstance(payload, PointCloudPayload):
                continue
            marker_name, point_size, edge_width, edge_color = self._sensing_marker_style(
                name, getattr(layer, "material", None)
            )
            self._apply_point_marker_material(
                name,
                marker_name=marker_name,
                point_size=point_size,
                edge_width=edge_width,
                edge_color=edge_color,
            )

    def _reset_sensing_marker_glyphs(self, layers: list[Any]) -> None:
        """Restore sensing point clouds to plain point materials."""
        marker_cls = getattr(self._gfx, "PointsMarkerMaterial", None)
        if marker_cls is None:
            return
        for layer in layers:
            name = str(getattr(layer, "name", ""))
            payload = getattr(layer, "payload", None)
            if not name.startswith("sensing_") or not isinstance(payload, PointCloudPayload):
                continue
            obj = self._objects.get(name)
            if obj is None or not isinstance(getattr(obj, "material", None), marker_cls):
                continue
            try:
                obj.material = self._build_points_material(
                    has_vertex_colors=self._object_has_color_buffer(obj)
                )
            except Exception as exc:
                logger.debug("Failed to reset sensing marker material for '%s': %s", name, exc)
                continue
            material = self._materials.get(name)
            if material is not None:
                self.set_named_material(name, material)

    def _sensing_marker_style(
        self,
        name: str,
        material: MaterialPayload | None,
    ) -> tuple[str, float, float, tuple[float, float, float, float]]:
        """Choose marker glyph and edge styling from the sensing layer name."""
        requested_size = (
            float(material.point_size)
            if material is not None and material.point_size is not None
            else 8.0
        )
        if name.startswith(("sensing_track_point_", "sensing_saved_track_point_")):
            return "diamond", max(15.0, requested_size + 4.0), 1.5, (1.0, 1.0, 1.0, 0.95)
        if name.startswith("sensing_truth_"):
            return "plus", max(16.0, requested_size), 0.0, (1.0, 1.0, 1.0, 0.0)
        if name == "sensing_evidence":
            return "circle", max(5.0, requested_size), 0.0, (1.0, 1.0, 1.0, 0.0)
        return "cross", max(12.0, requested_size + 2.0), 0.0, (0.02, 0.02, 0.02, 0.0)

    def _apply_point_marker_material(
        self,
        name: str,
        *,
        marker_name: str,
        point_size: float,
        edge_width: float,
        edge_color: tuple[float, float, float, float],
    ) -> None:
        """Apply one pygfx PointsMarkerMaterial while preserving vertex colors."""
        obj = self._objects.get(name)
        if obj is None:
            return
        marker_cls = getattr(self._gfx, "PointsMarkerMaterial", None)
        if marker_cls is None:
            return
        has_colors = self._object_has_color_buffer(obj)
        material_payload = self._materials.get(name)
        base_color = (
            material_payload.base_color if material_payload is not None else (0.95, 0.95, 0.95, 1.0)
        )
        current = getattr(obj, "material", None)
        geom = getattr(obj, "geometry", None)
        positions = getattr(getattr(geom, "positions", None), "data", None)
        count = int(len(positions)) if positions is not None else 0
        style = (marker_name, point_size, edge_width, edge_color, base_color, has_colors, count)
        previous = self._sensing_marker_specs.get(name)
        if (
            previous is not None
            and previous[0] is current
            and previous[1] is geom
            and previous[2] == style
        ):
            return
        if not isinstance(current, marker_cls):
            kwargs: dict[str, Any] = {
                "color": base_color,
                "size": float(point_size),
                "pick_write": name != "sensing_evidence" and not name.startswith("sensing_truth_"),
                "marker": marker_name,
                "marker_mode": "vertex",
                "edge_width": float(edge_width),
                "edge_color": edge_color,
            }
            if has_colors:
                kwargs["color_mode"] = "vertex"
            try:
                current = marker_cls(**kwargs)
            except TypeError:
                kwargs.pop("color_mode", None)
                current = marker_cls(**kwargs)
            self._apply_clipping_to_material(current)
            try:
                obj.material = current
            except Exception as exc:
                logger.debug("Failed to apply sensing marker material for '%s': %s", name, exc)
                return
        else:
            for attr, value in (
                ("size", float(point_size)),
                ("marker", marker_name),
                ("marker_mode", "vertex"),
                ("edge_width", float(edge_width)),
                ("edge_color", edge_color),
                ("color", base_color),
                ("color_mode", "vertex" if has_colors else "auto"),
            ):
                try:
                    if hasattr(current, attr):
                        setattr(current, attr, value)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

        if count <= 0:
            return
        if name != "sensing_evidence":
            # These are result annotations at unchanged world coordinates.
            # In particular, center-based truth often lies inside its mesh.
            # Keep markers readable while trails retain ordinary occlusion.
            current.depth_test = False
            current.depth_write = False
            obj.render_order = 1000
        try:
            geom.markers = self._gfx.Buffer(
                np.full(count, self._marker_int(marker_name), dtype=np.int32)
            )
            self._sensing_marker_specs[name] = (current, geom, style)
        except Exception as exc:
            logger.debug("Failed to assign sensing markers for '%s': %s", name, exc)

    @staticmethod
    def _marker_int(marker_name: str) -> int:
        """Return pygfx marker enum integer, falling back to circle or zero."""
        try:
            from pygfx import MarkerInt

            return int(MarkerInt[marker_name])
        except (ImportError, KeyError, TypeError, ValueError):
            try:
                from pygfx import MarkerInt

                return int(MarkerInt["circle"])
            except (ImportError, KeyError, TypeError, ValueError):
                return 0

    def _update_sensing_track_labels(self, raw_frame: dict[str, Any], layers: list[Any]) -> None:
        """Create/update labels for track-point layers and remove stale labels."""
        detection_by_track = self._sensing_detection_by_track(raw_frame)
        label_states: dict[str, RenderObjectState] = getattr(
            self,
            "_sensing_label_states",
            {},
        )
        self._sensing_label_states = label_states
        current_names: set[str] = set()
        for layer in layers:
            name = str(getattr(layer, "name", ""))
            if not name.startswith("sensing_track_point_"):
                continue
            payload = getattr(layer, "payload", None)
            if not isinstance(payload, PointCloudPayload):
                continue
            points = np.asarray(payload.points, dtype=np.float32).reshape(-1, 3)
            if points.size == 0:
                continue
            try:
                track_id = int(name.rsplit("_", 1)[-1])
            except ValueError:
                continue
            label_name = f"sensing_track_label_{track_id}"
            position = np.asarray(points[0], dtype=np.float64) + np.array([0.0, 0.0, 0.55])
            text = self._sensing_track_label_text(track_id, detection_by_track.get(track_id, {}))
            if not (layer.metadata or {}).get("current", True):
                text += "  historical"
            color = self._sensing_layer_color(layer)
            state = label_states.get(label_name)
            if not isinstance(state, RenderObjectState) or not isinstance(
                state.payload,
                TextLabelPayload,
            ):
                state = make_text_label_state(
                    label_name,
                    text,
                    color,
                    font_size=0.28,
                    position=position,
                    outline_thickness=0.22,
                )
                label_states[label_name] = state
            else:
                if state.payload.text != text:
                    state.replace_payload(
                        TextLabelPayload(
                            text=text,
                            font_size=0.28,
                            screen_space=True,
                            outline_thickness=0.22,
                        )
                    )
                rgba = tuple(float(value) for value in color[:3]) + (1.0,)
                if state.material.base_color != rgba:
                    state.material = MaterialPayload(base_color=rgba, shader="unlit")
                state.world_transform = Transform.from_translation(position)
                state.visible = True
            state.metadata.update(layout_anchor=points[0].tolist(), pickable=False)
            snapshot = label_with_callout(state.to_render_object(), self.visualizer)
            self.ensure_object(snapshot)
            native = self._objects.get(label_name)
            if native is not None and "label_callout" not in snapshot.metadata:
                # Screen-space offset keeps labels clear of anchors/trails,
                # including a top-down camera where a world-Z offset vanishes.
                native.anchor = "bottom-left"
                native.anchor_offset = 10.0
            current_names.add(label_name)

        for stale_name in self._sensing_annotation_names - current_names:
            self.remove_object(stale_name)
            label_states.pop(stale_name, None)
        self._sensing_annotation_names = current_names

    @staticmethod
    def _sensing_detection_by_track(raw_frame: dict[str, Any]) -> dict[int, dict[str, Any]]:
        """Return the strongest detection record for each track ID."""
        detections = (raw_frame.get("sensing") or {}).get("detections") or []
        best: dict[int, dict[str, Any]] = {}
        best_snr: dict[int, float] = {}
        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            try:
                track_id = int(tid)
            except (TypeError, ValueError):
                continue
            try:
                snr = float(det.get("snr_db", -1e9))
            except (TypeError, ValueError):
                snr = -1e9
            if track_id not in best or snr >= best_snr.get(track_id, -1e9):
                best[track_id] = det
                best_snr[track_id] = snr
        return best

    @staticmethod
    def _sensing_track_label_text(track_id: int, det: dict[str, Any]) -> str:
        """Identify the track and its saved measured range when available."""
        parts = [f"T{track_id}"]
        distance = det.get("range_m")
        if isinstance(distance, (int, float)) and np.isfinite(distance):
            parts.append(f"{distance:.1f} m")
        if det and not bool(det.get("track_confirmed", True)):
            parts.append("tentative")
        return " · ".join(parts)

    @staticmethod
    def _sensing_layer_color(layer: Any) -> tuple[float, float, float]:
        """Return the RGB color used for labels attached to a sensing layer."""
        material = getattr(layer, "material", None)
        color = getattr(material, "base_color", None)
        if color is None:
            return (1.0, 1.0, 1.0)
        arr = np.asarray(color, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return (1.0, 1.0, 1.0)
        return (float(arr[0]), float(arr[1]), float(arr[2]))
