"""Optional world guides sized from the complete static environment.

Guides are native scene children, outside the named scientific object registry.
They never enlarge framing bounds or take part in ray tracing; visible guides
remain inside the camera's depth-clipping envelope. Static geometry
and transform revisions invalidate one small bounds cache; moving actors and
ordinary playback do not cause vertex reductions or guide reconstruction.
"""

from __future__ import annotations

import numpy as np

from .callout_cards import card_padding, update_card_shape


def _coordinate_ticks(low: float, high: float) -> tuple[np.ndarray, float]:
    """Choose a short 1/2/5 sequence anchored to real world-coordinate zero."""
    span = high - low
    if span <= 0:
        return np.asarray([low], dtype=float), 1.0
    magnitude = 10.0 ** np.floor(np.log10(span / 5.0))
    options = []
    for factor in (0.1, 1.0, 10.0):
        for unit in (1.0, 2.0, 5.0):
            step = magnitude * factor * unit
            first = int(np.ceil((low - span * 1e-12) / step))
            last = int(np.floor((high + span * 1e-12) / step))
            options.append((abs(last - first + 1 - 5), -step, first, last))
    _, negative_step, first, last = min(options)
    step = -negative_step
    return np.arange(first, last + 1, dtype=float) * step, step


def _coordinate_text(value: float, step: float) -> str:
    decimals = max(0, min(8, int(-np.floor(np.log10(step)))))
    if abs(value) < step * 1e-10:
        value = 0.0
    return f"{value:,.{decimals}f}"


class PygfxSceneGuidesMixin:
    """Ground grid, labelled world axes and environment dimensions."""

    def _update_scene_guide_bounds(self):
        static = getattr(self, "_static_group", None)
        sources, signature = [], []
        for source in () if static is None else static.children:
            name = getattr(self, "_reverse_objects", {}).get(id(source))
            if (
                not name
                or not self._is_scene_mesh_name(name)
                or not isinstance(source, self._gfx.Mesh)
            ):
                continue
            geometry = source.geometry
            positions = getattr(geometry, "positions", None)
            if positions is None:
                continue
            sources.append(source)
            signature.append(
                (id(source), id(geometry), id(positions), positions.rev, source.world.last_modified)
            )
        signature = tuple(signature)
        if signature == getattr(self, "_scene_guide_bounds_signature", None):
            return getattr(self, "_scene_guide_bounds", None)
        bounds = []
        for source in sources:
            # pygfx caches these geometry-only bounds by position-buffer
            # revision. Child labels, headlights and overlays are excluded.
            local = source.get_geometry_bounding_box()
            if local is None:
                continue
            local = np.asarray(local, dtype=float)
            matrix = source.world.matrix
            if local.shape != (2, 3) or not np.isfinite(local).all():
                continue
            center = matrix[:3, :3] @ local.mean(axis=0) + matrix[:3, 3]
            radius = np.abs(matrix[:3, :3]) @ ((local[1] - local[0]) * 0.5)
            bounds.append((center - radius, center + radius))
        self._scene_guide_bounds = (
            np.stack(
                (np.min(np.asarray(bounds)[:, 0], axis=0), np.max(np.asarray(bounds)[:, 1], axis=0))
            )
            if bounds
            else None
        )
        self._scene_guide_bounds_signature = signature
        return self._scene_guide_bounds

    def get_ground_grid_visible(self) -> bool:
        return bool(getattr(self, "_ground_grid_visible", False))

    def set_ground_grid_visible(self, visible: bool) -> None:
        self._ground_grid_visible = bool(visible)
        self._sync_scene_guides()
        grid = getattr(self, "_ground_grid_obj", None)
        if grid is not None:
            grid.visible = bool(visible)
        self.request_redraw()

    def get_ground_grid_divisions(self) -> int:
        return int(getattr(self, "_ground_grid_divisions", 20))

    def set_ground_grid_divisions(self, divisions: int) -> bool:
        if (
            isinstance(divisions, bool)
            or not isinstance(divisions, (int, np.integer))
            or not 2 <= divisions <= 500
        ):
            raise ValueError("Ground grid divisions must be an integer between 2 and 500")
        if divisions == self.get_ground_grid_divisions():
            return True
        self._ground_grid_divisions = int(divisions)
        self._ground_grid_needs_rebuild = True
        self._sync_scene_guides()
        self.request_redraw()
        return True

    def _remove_ground_grid(self) -> None:
        grid = getattr(self, "_ground_grid_obj", None)
        if grid is not None and grid.parent is not None:
            grid.parent.remove(grid)
        self._ground_grid_obj = None
        self._ground_grid_signature = None

    def _build_ground_grid(self) -> None:
        """Build a unit helper once; bounds changes only update its transform."""
        if getattr(self, "_scene", None) is None:
            return
        self._remove_ground_grid()
        self._ground_grid_obj = self._gfx.GridHelper(
            size=1.0,
            divisions=self.get_ground_grid_divisions(),
            color1=(0.48, 0.53, 0.61, 0.75),
            color2=(0.38, 0.43, 0.50, 0.50),
            thickness=0.75,
        )
        self._ground_grid_obj.material.pick_write = False
        self._ground_grid_obj.material.depth_write = False
        self._scene.add(self._ground_grid_obj)
        self._ground_grid_needs_rebuild = False

    def _ensure_ground_grid_current(self) -> None:
        """Compatibility entry from frame application; all guides share bounds."""
        self._sync_scene_guides()

    def set_environment_dimensions_visible(self, visible: bool) -> bool:
        self._environment_dimensions_visible = bool(visible)
        self._sync_scene_guides()
        group = getattr(self, "_environment_dimensions_group", None)
        if group is not None:
            group.visible = bool(visible)
        self.request_redraw()
        return True

    def set_environment_dimensions_style(self, style: str) -> bool:
        if style not in {"size_labels", "paper_axes"}:
            raise ValueError("Environment dimensions style must be size_labels or paper_axes")
        if style == getattr(self, "_environment_dimensions_style", "size_labels"):
            return True
        self._environment_dimensions_style = style
        self._sync_scene_guides()
        self.request_redraw()
        return True

    def get_environment_dimensions_state(self) -> dict:
        active = (
            self.get_ground_grid_visible()
            or bool(getattr(self, "_environment_dimensions_visible", False))
            or self.get_axes_visible()
        )
        bounds = (
            self._update_scene_guide_bounds()
            if active
            else getattr(self, "_scene_guide_bounds", None)
        )
        return {
            "enabled": bool(getattr(self, "_environment_dimensions_visible", False)),
            "style": getattr(self, "_environment_dimensions_style", "size_labels"),
            "bounds": (
                None if bounds is None else tuple(tuple(float(v) for v in row) for row in bounds)
            ),
            "extent_m": None if bounds is None else tuple(float(v) for v in bounds[1] - bounds[0]),
        }

    def _guide_label(self, text, position, *, color="#223247", font_size=15):
        label = self._gfx.Text(
            text=text,
            font_size=font_size,
            screen_space=True,
            anchor="middle-center",
            material=self._gfx.TextMaterial(
                color=color,
                outline_thickness=0,
                weight_offset=75,
                depth_test=False,
                depth_write=False,
                pick_write=False,
                render_queue=4100,
            ),
        )
        label.local.position = position
        label._orchav_guide_text = text
        label._orchav_label_card = True
        plate = self._gfx.Mesh(
            self._gfx.plane_geometry(1, 1),
            self._gfx.MeshBasicMaterial(
                color=(0.96, 0.975, 0.995, 0.94),
                alpha_mode="blend",
                side="both",
                depth_test=False,
                depth_write=False,
                pick_write=False,
                render_queue=4098,
            ),
        )
        plate.add(
            self._gfx.Line(
                self._gfx.Geometry(
                    positions=np.asarray(
                        [
                            [-0.5, -0.5, 0],
                            [0.5, -0.5, 0],
                            [0.5, 0.5, 0],
                            [-0.5, 0.5, 0],
                            [-0.5, -0.5, 0],
                        ],
                        dtype=np.float32,
                    )
                ),
                self._gfx.LineMaterial(
                    color=(0.48, 0.56, 0.67, 0.85),
                    thickness=1.0,
                    alpha_mode="blend",
                    depth_test=False,
                    depth_write=False,
                    pick_write=False,
                    render_queue=4099,
                ),
            )
        )
        label.add(plate)
        label._orchav_guide_plate = plate
        return label

    def show_axes(self, show: bool, size: float = 1.0) -> bool:
        if not getattr(self, "_initialized", False):
            return False
        size = float(size)
        if not np.isfinite(size) or size <= 0:
            raise ValueError("World-axis size must be finite and positive")
        self._scene_axes_visible = bool(show)
        self._scene_axes_requested_size = size
        self._sync_scene_guides()
        group = getattr(self, "_scene_axes_group", None)
        if group is not None:
            group.visible = bool(show)
        self.request_redraw()
        return True

    def get_axes_visible(self) -> bool:
        return bool(getattr(self, "_scene_axes_visible", False))

    def _scene_guide_labels(self) -> tuple:
        """Return guide-owned text for compact card layout."""
        return tuple(
            child
            for field in ("_scene_axes_group", "_environment_dimensions_group")
            if (group := getattr(self, field, None)) is not None and group.visible
            for child in group.children
            if isinstance(child, self._gfx.Text)
        )

    def _default_axes_size(self, requested_size: float) -> float:
        requested = float(requested_size)
        if requested > 1:
            return requested
        bounds = self._update_scene_guide_bounds()
        if bounds is None:
            return 1.0
        return max(2.0, min(75.0, float(np.max(bounds[1] - bounds[0])) * 0.12))

    def _sync_scene_guides(self) -> None:
        grid_on = self.get_ground_grid_visible()
        dimensions_on = bool(getattr(self, "_environment_dimensions_visible", False))
        axes_on = bool(getattr(self, "_scene_axes_visible", False))
        if not (grid_on or dimensions_on or axes_on) or getattr(self, "_scene", None) is None:
            return
        bounds = self._update_scene_guide_bounds()
        if grid_on:
            if getattr(self, "_ground_grid_obj", None) is None or getattr(
                self, "_ground_grid_needs_rebuild", False
            ):
                self._build_ground_grid()
            low, high = ((-10, -10, 0), (10, 10, 0)) if bounds is None else bounds
            low, high = np.asarray(low), np.asarray(high)
            size = max(float(np.max((high - low)[:2])) * 1.04, 1.0)
            center = (low + high) * 0.5
            signature = (size, float(center[0]), float(center[1]), float(low[2]))
            if signature != getattr(self, "_ground_grid_signature", None):
                # Native GridHelper is XZ; rotate to world XY and center at the
                # actual environment (including off-origin imported scenes).
                matrix = np.asarray(
                    [
                        [size, 0, 0, center[0]],
                        [0, 0, -size, center[1]],
                        [0, 1, 0, low[2] + 0.01],
                        [0, 0, 0, 1],
                    ],
                    dtype=float,
                )
                self._ground_grid_obj.local.matrix = matrix
                self._ground_grid_signature = signature
            self._ground_grid_obj.visible = True
        if dimensions_on:
            self._sync_environment_dimensions(bounds)
            self._layout_paper_dimension_labels()
        if axes_on:
            size = self._default_axes_size(getattr(self, "_scene_axes_requested_size", 1.0))
            if (
                size != getattr(self, "_scene_axes_size", None)
                or getattr(self, "_scene_axes_group", None) is None
            ):
                self._remove_guide_group("_scene_axes_group")
                group = self._gfx.Group()
                helper = self._gfx.AxesHelper(size=size, thickness=2)
                colors = ("#ff776d", "#88dd8e", "#85acff")
                # pygfx 0.17's set_colors still accesses a retired private
                # geometry field. The public geometry/material buffers work.
                for index, color in enumerate(colors):
                    rgba = tuple(self._gfx.Color(color))
                    helper.geometry.colors.data[index * 2 : index * 2 + 2] = rgba
                    helper.children[index].material.color = rgba
                helper.geometry.colors.update_full()
                helper.material.pick_write = False
                for child in helper.children:
                    child.material.pick_write = False
                group.add(helper)
                for index, (axis, color) in enumerate(zip("XYZ", colors, strict=True)):
                    position = np.eye(3)[index] * size * 1.08
                    ink = ("#ad3731", "#2b7340", "#305a9b")[index]
                    group.add(self._guide_label(axis, position, color=ink))
                self._scene.add(group)
                self._scene_axes_group = group
                self._scene_axes_size = size
            self._scene_axes_group.visible = True
        self._layout_scene_guide_cards()

    def _layout_scene_guide_cards(self) -> None:
        """Billboard compact cards using the existing native label convention."""
        labels = self._scene_guide_labels()
        camera, canvas = getattr(self, "_camera", None), getattr(self, "_canvas", None)
        if not labels or camera is None or canvas is None:
            return
        size = np.asarray(canvas.get_logical_size(), dtype=float)
        if np.min(size) <= 0:
            return
        camera.set_view_size(*size)
        projection, camera_matrix = camera.projection_matrix, camera.world.matrix
        signature = (
            projection.tobytes(),
            camera_matrix.tobytes(),
            tuple(size),
            tuple(
                (id(label), label.visible, label.world.last_modified, label.font_size)
                for label in labels
            ),
        )
        if signature == getattr(self, "_scene_guide_card_signature", None):
            return
        view_projection = projection @ camera.world.inverse_matrix
        for label in labels:
            plate = label._orchav_guide_plate
            anchor = label.world.position
            clip = view_projection @ np.r_[anchor, 1]
            plate.visible = bool(label.visible and clip[3] > 0 and 0 <= clip[2] <= clip[3])
            if not plate.visible:
                continue
            label._update_object()
            bounds = label._aabb
            dimensions_px = bounds[1, :2] - bounds[0, :2] + 2 * card_padding(label)
            update_card_shape(self, label, plate, dimensions_px)
            units = 2 * clip[3] / (np.diag(projection)[:2] * size)
            center = (bounds[1, :2] + bounds[0, :2]) * 0.5 * units
            world = np.eye(4)
            world[:3, :3] = camera_matrix[:3, :3] @ np.diag([*(dimensions_px * units), 1])
            world[:3, 3] = anchor + camera_matrix[:3, :2] @ center
            plate.local.matrix = label.world.inverse_matrix @ world
        self._scene_guide_card_signature = signature

    def _sync_environment_dimensions(self, bounds) -> None:
        style = getattr(self, "_environment_dimensions_style", "size_labels")
        signature = (style, None if bounds is None else tuple(bounds.ravel()))
        group = getattr(self, "_environment_dimensions_group", None)
        if (
            signature == getattr(self, "_environment_dimensions_signature", None)
            and group is not None
        ):
            group.visible = True
            return
        self._remove_guide_group("_environment_dimensions_group")
        self._environment_dimensions_signature = signature
        self._paper_dimension_labels = ()
        self._paper_dimension_layout_signature = None
        self._scene_guide_card_signature = None
        if bounds is None:
            return
        if style == "paper_axes":
            self._build_paper_dimensions(bounds)
            return
        low, high = bounds
        extent = high - low
        gap = max(0.2, min(8.0, float(np.max(extent)) * 0.025))
        base = low - (gap, gap, 0)
        segments, label_positions = [], []
        for index in range(3):
            start, end = base.copy(), base.copy()
            start[index], end[index] = low[index], high[index]
            segments.extend((start, end))
            cross_axis = 1 if index == 0 else 0
            tick = np.eye(3)[cross_axis] * gap * 0.20
            segments.extend((start - tick, start + tick, end - tick, end + tick))
            label_positions.append((start + end) * 0.5 - np.eye(3)[cross_axis] * gap * 0.35)
        group = self._gfx.Group()
        group.add(
            self._gfx.Line(
                self._gfx.Geometry(positions=np.asarray(segments, dtype=np.float32)),
                self._gfx.LineSegmentMaterial(
                    color="#506077", thickness=1.25, aa=True, depth_write=False, pick_write=False
                ),
            )
        )
        for axis, value, position in zip("XYZ", extent, label_positions, strict=True):
            group.add(self._guide_label(f"{axis}  {value:,.1f} m", position))
        self._scene.add(group)
        self._environment_dimensions_group = group

    def _build_paper_dimensions(self, bounds) -> None:
        """Draw measured coordinate edges plus a light full bounding frame."""
        low, high = bounds
        extent = high - low
        # Separate captions from the coordinate-number lane even on city-sized
        # scenes. A fixed metre cap makes both lanes coincide at overview scale.
        gap = max(0.2, float(np.max(extent)) * 0.08)
        base = low - (gap, gap, 0)
        group = self._gfx.Group()
        corners = np.asarray(
            [[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]]
        )
        frame, seen = [], set()
        for index, corner in enumerate(corners):
            for bit in (1, 2, 4):
                if index & bit:
                    continue
                pair = (tuple(corner), tuple(corners[index | bit]))
                if pair[0] != pair[1] and pair not in seen:
                    frame.extend(pair)
                    seen.add(pair)
        if frame:
            group.add(
                self._gfx.Line(
                    self._gfx.Geometry(positions=np.asarray(frame, dtype=np.float32)),
                    self._gfx.LineSegmentMaterial(
                        color="#607187",
                        opacity=0.85,
                        thickness=1.25,
                        aa=True,
                        depth_write=False,
                        pick_write=False,
                    ),
                )
            )
        segments, labels = [], []
        for index, axis in enumerate("XYZ"):
            start, end = base.copy(), base.copy()
            start[index], end[index] = low[index], high[index]
            if extent[index] > 0:
                segments.extend((start, end))
            cross = np.eye(3)[1 if index == 0 else 0]
            ticks, step = _coordinate_ticks(low[index], high[index])
            for value in ticks:
                point = base.copy()
                point[index] = value
                segments.extend((point - cross * gap * 0.12, point + cross * gap * 0.12))
                label = self._guide_label(
                    _coordinate_text(value, step), point - cross * gap * 0.50, font_size=14
                )
                label._orchav_guide_priority = 1 if abs(value) < step * 1e-10 else 2
                label._orchav_coordinate_value = float(value)
                label._orchav_coordinate_axis = axis
                group.add(label)
                labels.append(label)
            label = self._guide_label(f"{axis} (m)", (start + end) * 0.5 - cross * gap * 2.25)
            label._orchav_guide_priority = 0
            group.add(label)
            labels.append(label)
        group.add(
            self._gfx.Line(
                self._gfx.Geometry(positions=np.asarray(segments, dtype=np.float32)),
                self._gfx.LineSegmentMaterial(
                    color="#506077", thickness=1.25, aa=True, depth_write=False, pick_write=False
                ),
            )
        )
        self._scene.add(group)
        self._environment_dimensions_group = group
        self._paper_dimension_labels = tuple(labels)

    def _layout_paper_dimension_labels(self) -> None:
        """Hide colliding numbers on oblique views without moving their anchors."""
        labels = getattr(self, "_paper_dimension_labels", ())
        camera, canvas = getattr(self, "_camera", None), getattr(self, "_canvas", None)
        if not labels or camera is None or canvas is None:
            return
        size = np.asarray(canvas.get_logical_size(), dtype=float)
        if np.min(size) <= 0:
            return
        camera.set_view_size(*size)
        projection = camera.projection_matrix
        signature = (
            camera.world.last_modified,
            projection.tobytes(),
            tuple(size),
            id(self._environment_dimensions_group),
        )
        if signature == getattr(self, "_paper_dimension_layout_signature", None):
            return
        matrix = projection @ camera.world.inverse_matrix
        occupied = []
        for label in sorted(labels, key=lambda item: item._orchav_guide_priority):
            clip = matrix @ np.r_[label.world.position, 1]
            if clip[3] <= 0 or not 0 <= clip[2] <= clip[3]:
                label.visible = False
                continue
            center = (clip[:2] / clip[3] + 1) * size * 0.5
            half = np.asarray(
                (
                    len(label._orchav_guide_text) * label.font_size * 0.33 + 4,
                    label.font_size * 0.7 + 3,
                )
            )
            rectangle = np.stack((center - half, center + half))
            visible = bool(np.all(rectangle[1] >= 0) and np.all(rectangle[0] <= size))
            if visible:
                visible = not any(
                    np.all(np.minimum(rectangle[1], other[1]) > np.maximum(rectangle[0], other[0]))
                    for other in occupied
                )
            label.visible = visible
            if visible:
                occupied.append(rectangle)
        self._paper_dimension_layout_signature = signature

    def _remove_guide_group(self, field) -> None:
        group = getattr(self, field, None)
        if group is not None and group.parent is not None:
            group.parent.remove(group)
        setattr(self, field, None)

    def _clear_scene_guides(self) -> None:
        """Remove presentation objects and caches before replacing the source."""
        self._remove_ground_grid()
        self._remove_guide_group("_scene_axes_group")
        self._remove_guide_group("_environment_dimensions_group")
        self._ground_grid_visible = False
        self._ground_grid_needs_rebuild = False
        self._scene_axes_visible = False
        self._environment_dimensions_visible = False
        self._scene_guide_bounds_signature = None
        self._scene_guide_bounds = None
        self._environment_dimensions_signature = None
        self._scene_axes_size = None
        self._paper_dimension_labels = ()
        self._paper_dimension_layout_signature = None
        self._scene_guide_card_signature = None
