"""Transient selected-MPC path presentation for the pygfx backend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..mpc_path_inspection import MpcPathInspectionSnapshot
from .explanatory_overlay import configure_explanatory_overlay
from .mpc import interaction_marker_spec

logger = logging.getLogger(__name__)

MPC_SELECTION_PREFIX = "mpc_selection::"
MPC_SELECTION_HALO_NAME = f"{MPC_SELECTION_PREFIX}halo"
MPC_SELECTION_PATH_NAME = f"{MPC_SELECTION_PREFIX}path"
MPC_SELECTION_DIRECTION_NAME = f"{MPC_SELECTION_PREFIX}direction"
MPC_SELECTION_PULSE_NAME = f"{MPC_SELECTION_PREFIX}pulse"
MPC_SELECTION_BOUNCE_POINTS_NAME = f"{MPC_SELECTION_PREFIX}bounce_points"

_PULSE_COUNT = 8
_PULSE_TRAIL_FRACTION = 0.018
_ARC_EPSILON = 1.0e-12


class PygfxMpcSelectionMixin:
    """Own native pygfx objects for one transient selected MPC path."""

    def _initialize_mpc_path_inspection_state(self) -> None:
        """Initialize selection-overlay state without allocating native objects."""
        self._mpc_selection_names: set[str] = set()
        self._mpc_selection_snapshot: MpcPathInspectionSnapshot | None = None
        self._mpc_selection_pulse_geometry: Any = None
        self._mpc_selection_frame_snapshot: MpcPathInspectionSnapshot | None = None

    def mpc_path_inspection_available(self) -> bool:
        """Return whether native selection objects can be drawn safely."""
        if (
            not getattr(self, "_initialized", False)
            or getattr(self, "_qt_window_closed", False)
            or getattr(self, "_scene", None) is None
            or getattr(self, "_canvas", None) is None
        ):
            return False
        canvas = self._canvas
        return not bool(
            getattr(canvas, "_orchav_closed", False) or getattr(canvas, "_is_closed", False)
        )

    def has_mpc_path_inspection(
        self,
        snapshot: MpcPathInspectionSnapshot | None = None,
    ) -> bool:
        """Return whether the expected selected-path overlay is still installed."""
        current = getattr(self, "_mpc_selection_snapshot", None)
        if current is None or not getattr(self, "_mpc_selection_names", None):
            return False
        return snapshot is None or current is snapshot

    def set_mpc_path_inspection(self, snapshot: MpcPathInspectionSnapshot) -> bool:
        """Replace the transient selected-path overlay from a small snapshot."""
        if not isinstance(snapshot, MpcPathInspectionSnapshot):
            raise TypeError("snapshot must be an MpcPathInspectionSnapshot")
        if not self.mpc_path_inspection_available():
            return False
        if not self._clear_mpc_path_inspection(request_redraw=False):
            return False

        try:
            self._build_mpc_path_inspection(snapshot)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("PygfxRenderer: failed to create MPC selection overlay: %s", exc)
            self._clear_mpc_path_inspection(request_redraw=True)
            return False

        self._mpc_selection_snapshot = snapshot
        self.request_redraw()
        return True

    def clear_mpc_path_inspection(self) -> bool:
        """Remove every selected-path object without touching bulk MPC geometry."""
        return self._clear_mpc_path_inspection(request_redraw=True)

    def _clear_mpc_path_inspection(self, *, request_redraw: bool) -> bool:
        """Clear native objects, optionally scheduling one presentation."""
        names = tuple(
            sorted(
                getattr(self, "_mpc_selection_names", ()),
                key=lambda name: (not name.startswith(f"{MPC_SELECTION_PREFIX}bounce_"), name),
            )
        )
        success = True
        changed = False
        remaining: set[str] = set()
        for name in names:
            present = name in getattr(self, "_objects", {})
            if not present:
                continue
            try:
                removed = bool(self.remove_named_geometry(name))
            except Exception:
                removed = False
                logger.warning(
                    "PygfxRenderer: MPC selection cleanup raised for '%s'",
                    name,
                    exc_info=True,
                )
            changed = removed or changed
            if not removed and name in getattr(self, "_objects", {}):
                success = False
                remaining.add(name)
                changed = self._hide_mpc_selection_object(name) or changed
                logger.warning(
                    "PygfxRenderer: retained hidden MPC selection object '%s' "
                    "after cleanup failure; removal will be retried",
                    name,
                )

        self._mpc_selection_names = remaining
        if not remaining:
            self._mpc_selection_snapshot = None
            self._mpc_selection_pulse_geometry = None
        if changed and request_redraw:
            self.request_redraw()
        return success

    def _hide_mpc_selection_object(self, name: str) -> bool:
        """Hide one retained native object after a failed removal attempt."""
        obj = getattr(self, "_objects", {}).get(name)
        if obj is None:
            return False
        try:
            was_visible = bool(getattr(obj, "visible", True))
            obj.visible = False
        except (AttributeError, RuntimeError, TypeError):
            logger.warning(
                "PygfxRenderer: could not hide retained MPC selection object '%s'",
                name,
                exc_info=True,
            )
            return False
        return was_visible

    def _begin_mpc_path_inspection_frame_transition(self) -> None:
        """Suspend the old-frame overlay before mutating frame geometry."""
        snapshot = getattr(self, "_mpc_selection_snapshot", None)
        if snapshot is None:
            return
        if getattr(self, "_mpc_selection_frame_snapshot", None) is not None:
            logger.warning("PygfxRenderer: replacing an unfinished MPC selection transition")
        self._mpc_selection_frame_snapshot = snapshot
        self._clear_mpc_path_inspection(request_redraw=False)

    def _finish_mpc_path_inspection_frame_transition(self, *, presented: bool) -> None:
        """Commit overlay removal or restore it when frame presentation fails."""
        snapshot = getattr(self, "_mpc_selection_frame_snapshot", None)
        self._mpc_selection_frame_snapshot = None
        if snapshot is None:
            return

        if presented:
            # A transient native-removal failure may have left hidden objects
            # behind. Keep them owned and retry once while ensuring no stale
            # snapshot can continue animating over the accepted new frame.
            if getattr(self, "_mpc_selection_names", None):
                self._clear_mpc_path_inspection(request_redraw=False)
            self._mpc_selection_snapshot = None
            self._mpc_selection_pulse_geometry = None
            return

        if not self.mpc_path_inspection_available():
            logger.debug(
                "PygfxRenderer: MPC selection restore skipped because the canvas is unavailable"
            )
            return
        if not self.set_mpc_path_inspection(snapshot):
            logger.warning(
                "PygfxRenderer: could not restore MPC selection after rejected frame presentation"
            )

    def update_mpc_path_flow(self, phase: float) -> bool:
        """Move the preallocated pulse trail to one normalized arc-length phase."""
        snapshot = getattr(self, "_mpc_selection_snapshot", None)
        geometry = getattr(self, "_mpc_selection_pulse_geometry", None)
        if snapshot is None or geometry is None or not self.mpc_path_inspection_available():
            return False
        try:
            normalized_phase = float(phase)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(normalized_phase):
            return False

        positions = getattr(getattr(geometry, "positions", None), "data", None)
        if positions is None or np.asarray(positions).shape != (_PULSE_COUNT, 3):
            return False
        self._fill_pulse_positions(snapshot, normalized_phase, positions)

        position_buffer = geometry.positions
        if hasattr(position_buffer, "update_range"):
            position_buffer.update_range(0, _PULSE_COUNT)
        elif hasattr(position_buffer, "update_full"):
            position_buffer.update_full()
        else:
            return False
        self.request_redraw()
        return True

    def _build_mpc_path_inspection(self, snapshot: MpcPathInspectionSnapshot) -> None:
        """Create static overlay objects and one writable pulse buffer."""
        gfx = self._gfx
        path_points = np.array(snapshot.points, dtype=np.float32, copy=True, order="C")
        path_geometry = gfx.Geometry(positions=path_points)
        red, green, blue, alpha = snapshot.path_color

        halo_material = gfx.LineMaterial(
            color=(red, green, blue, min(0.22, alpha)),
            thickness=11.0,
            thickness_space="screen",
            aa=True,
            alpha_mode="add",
            depth_test=False,
            depth_write=False,
            pick_write=False,
        )
        self._apply_mpc_selection_clipping(halo_material)
        halo = gfx.Line(path_geometry, halo_material, name=MPC_SELECTION_HALO_NAME)
        halo.render_order = 20_000
        self._register_mpc_selection_object(MPC_SELECTION_HALO_NAME, halo, "lines")

        path_material = gfx.LineMaterial(
            color=(red, green, blue, alpha),
            thickness=3.5,
            thickness_space="screen",
            aa=True,
            alpha_mode="blend",
            depth_test=False,
            depth_write=False,
            pick_write=False,
        )
        self._apply_mpc_selection_clipping(path_material)
        path = gfx.Line(path_geometry, path_material, name=MPC_SELECTION_PATH_NAME)
        path.render_order = 20_010
        self._register_mpc_selection_object(MPC_SELECTION_PATH_NAME, path, "lines")

        direction_positions = self._direction_positions(snapshot)
        if direction_positions is not None:
            direction_material = gfx.LineArrowMaterial(
                color=(red, green, blue, 1.0),
                thickness=5.0,
                thickness_space="screen",
                aa=True,
                alpha_mode="blend",
                depth_test=False,
                depth_write=False,
                pick_write=False,
            )
            self._apply_mpc_selection_clipping(direction_material)
            direction = gfx.Line(
                gfx.Geometry(positions=direction_positions),
                direction_material,
                name=MPC_SELECTION_DIRECTION_NAME,
            )
            direction.render_order = 20_020
            self._register_mpc_selection_object(
                MPC_SELECTION_DIRECTION_NAME,
                direction,
                "lines",
            )

        pulse_positions = np.empty((_PULSE_COUNT, 3), dtype=np.float32)
        self._fill_pulse_positions(snapshot, 0.0, pulse_positions)
        pulse_colors = np.empty((_PULSE_COUNT, 4), dtype=np.float32)
        pulse_colors[:, :3] = (red, green, blue)
        pulse_colors[:, 3] = np.linspace(0.95, 0.08, _PULSE_COUNT, dtype=np.float32)
        pulse_sizes = np.linspace(18.0, 5.0, _PULSE_COUNT, dtype=np.float32)
        pulse_geometry = gfx.Geometry(
            positions=pulse_positions,
            colors=pulse_colors,
            sizes=pulse_sizes,
        )
        pulse_material = gfx.PointsGaussianBlobMaterial(
            size_mode="vertex",
            size_space="screen",
            color_mode="vertex",
            aa=True,
            alpha_mode="add",
            depth_test=False,
            depth_write=False,
            pick_write=False,
        )
        self._apply_mpc_selection_clipping(pulse_material)
        pulse = gfx.Points(
            pulse_geometry,
            pulse_material,
            name=MPC_SELECTION_PULSE_NAME,
        )
        pulse.render_order = 20_030
        self._register_mpc_selection_object(MPC_SELECTION_PULSE_NAME, pulse, "points")
        self._mpc_selection_pulse_geometry = pulse_geometry

        self._build_mpc_selection_bounces(snapshot)

    def _build_mpc_selection_bounces(self, snapshot: MpcPathInspectionSnapshot) -> None:
        """Create one marker object and selected-path-only bounce labels."""
        bounce_positions = np.array(
            snapshot.points[1:-1],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        bounce_count = len(bounce_positions)
        if bounce_count == 0:
            return

        gfx = self._gfx
        red, green, blue, _alpha = snapshot.path_color
        if snapshot.bounce_colors is None:
            bounce_colors = np.tile(
                np.asarray((red, green, blue, 1.0), dtype=np.float32),
                (bounce_count, 1),
            )
        else:
            bounce_colors = np.array(
                snapshot.bounce_colors,
                dtype=np.float32,
                copy=True,
                order="C",
            )
        bounce_sizes = np.full(bounce_count, 15.0, dtype=np.float32)
        geometry_kwargs: dict[str, np.ndarray] = {
            "positions": bounce_positions,
            "colors": bounce_colors,
            "sizes": bounce_sizes,
        }
        material_kwargs: dict[str, Any] = {
            "marker": "diamond",
            "marker_mode": "uniform",
            "size_mode": "vertex",
            "size_space": "screen",
            "color_mode": "vertex",
            "edge_color": (0.02, 0.04, 0.08, 1.0),
            "edge_width": 1.8,
            "aa": True,
            "alpha_mode": "blend",
            "depth_test": False,
            "depth_write": False,
            "pick_write": False,
        }
        interaction_types = snapshot.bounce_interaction_types
        marker_int = getattr(gfx, "MarkerInt", None)
        if interaction_types is not None and marker_int is not None:
            geometry_kwargs["markers"] = self._selected_bounce_marker_codes(
                interaction_types,
                marker_int,
            )
            material_kwargs["marker_mode"] = "vertex"

        bounce_material = gfx.PointsMarkerMaterial(**material_kwargs)
        self._apply_mpc_selection_clipping(bounce_material)
        bounces = gfx.Points(
            gfx.Geometry(**geometry_kwargs),
            bounce_material,
            name=MPC_SELECTION_BOUNCE_POINTS_NAME,
        )
        bounces.render_order = 20_040
        self._register_mpc_selection_object(
            MPC_SELECTION_BOUNCE_POINTS_NAME,
            bounces,
            "points",
        )

        for index, (position, text) in enumerate(
            zip(bounce_positions, snapshot.bounce_labels),
            start=1,
        ):
            name = f"{MPC_SELECTION_PREFIX}bounce_{index}"
            material = gfx.TextMaterial(
                color=(1.0, 1.0, 1.0, 1.0),
                outline_color=(0.01, 0.02, 0.05, 1.0),
                outline_thickness=0.18,
                aa=True,
                depth_test=False,
                depth_write=False,
                pick_write=False,
            )
            label = gfx.Text(
                text=text,
                font_size=15.0,
                screen_space=True,
                anchor="bottom-center",
                material=material,
                name=name,
            )
            label.local.position = tuple(float(value) for value in position)
            label.render_order = 20_050 + index
            self._register_mpc_selection_object(name, label, "text")

    def _register_mpc_selection_object(self, name: str, obj: Any, kind: str) -> None:
        """Attach one native object to the normal stable-name registry."""
        if name in self._objects or name in self._name_to_handle:
            raise RuntimeError(f"duplicate MPC selection object: {name}")
        self._scene.add(obj)
        handle = self._allocate_handle()
        self._name_to_handle[name] = handle
        self._handle_to_name[handle] = name
        self._objects[name] = obj
        self._kinds[name] = kind
        self._topology[name] = ("mpc-selection-native", kind)
        self._reverse_objects[id(obj)] = name
        self._pick_metadata[name] = {"type": "mpc_selection"}
        configure_explanatory_overlay(self, name, native_object=obj)
        self._mpc_selection_names.add(name)

    def _apply_mpc_selection_clipping(self, material: Any) -> None:
        """Apply the renderer's current clipping-plane state when available."""
        apply_clipping = getattr(self, "_apply_clipping_to_material", None)
        if callable(apply_clipping):
            apply_clipping(material)

    @staticmethod
    def _direction_positions(
        snapshot: MpcPathInspectionSnapshot,
    ) -> np.ndarray | None:
        """Return one in-segment TX-to-RX arrow near 70% path length."""
        lengths = snapshot.segment_lengths
        nonzero = np.flatnonzero(lengths > _ARC_EPSILON)
        if nonzero.size == 0 or snapshot.total_length <= _ARC_EPSILON:
            return None
        target = snapshot.total_length * 0.70
        segment_index = int(np.searchsorted(snapshot.cumulative_lengths[1:], target, side="right"))
        if segment_index >= len(lengths) or lengths[segment_index] <= _ARC_EPSILON:
            nearest = int(np.argmin(np.abs(nonzero - min(segment_index, len(lengths) - 1))))
            segment_index = int(nonzero[nearest])
        start = snapshot.points[segment_index].astype(np.float64, copy=False)
        vector = snapshot.segment_vectors[segment_index]
        positions = np.vstack((start + vector * 0.20, start + vector * 0.82))
        return np.ascontiguousarray(positions, dtype=np.float32)

    @staticmethod
    def _fill_pulse_positions(
        snapshot: MpcPathInspectionSnapshot,
        phase: float,
        out: np.ndarray,
    ) -> None:
        """Fill a preallocated trail by cumulative geometric arc length."""
        if snapshot.total_length <= _ARC_EPSILON:
            out[:] = snapshot.points[0]
            return

        head_fraction = float(np.remainder(float(phase), 1.0))
        fractions = np.clip(
            head_fraction - np.arange(_PULSE_COUNT, dtype=np.float64) * _PULSE_TRAIL_FRACTION,
            0.0,
            1.0,
        )
        distances = fractions * snapshot.total_length
        segment_indices = np.searchsorted(
            snapshot.cumulative_lengths[1:],
            distances,
            side="right",
        )
        np.clip(segment_indices, 0, len(snapshot.segment_lengths) - 1, out=segment_indices)
        starts = snapshot.cumulative_lengths[segment_indices]
        lengths = snapshot.segment_lengths[segment_indices]
        local = np.divide(
            distances - starts,
            lengths,
            out=np.zeros_like(distances),
            where=lengths > _ARC_EPSILON,
        )
        points = snapshot.points[segment_indices].astype(np.float64, copy=False)
        vectors = snapshot.segment_vectors[segment_indices]
        out[:] = points + local[:, None] * vectors

    @staticmethod
    def _selected_bounce_marker_codes(
        interaction_types: np.ndarray,
        marker_int: Any,
    ) -> np.ndarray:
        """Map interaction types to visible marker glyphs for physical bounces."""
        unknown = int(marker_int["cross"])
        markers = np.full(len(interaction_types), unknown, dtype=np.int32)
        for interaction_type in np.unique(interaction_types):
            spec = interaction_marker_spec(int(interaction_type))
            marker_name = "custom" if spec is None else spec.marker_name
            markers[interaction_types == interaction_type] = int(marker_int[marker_name])
        return markers
