"""Transient, bounded native pygfx layers for the experiment workspace."""

from __future__ import annotations

import weakref

import numpy as np

MAX_PATHS = 1500
MAX_SEGMENTS = 100_000


class ExperimentOverlay:
    """Own explanatory objects without touching bulk MPCs or persistent meshes."""

    def __init__(self, renderer):
        if getattr(renderer, "renderer_id", None) != "pygfx":
            raise ValueError("Visual experiments require the pygfx renderer")
        self.renderer = renderer
        self.group = None
        self.layers = {}
        self._interaction_label_specs = ()

    def _replace(self, name, obj):
        renderer = self.renderer
        scene = getattr(renderer, "_scene", None)
        if scene is None or getattr(renderer, "_qt_window_closed", False):
            return
        if self.group is None or self.group.parent is not scene:
            self.clear()
            self.group = renderer._gfx.Group()
            scene.add(self.group)
        previous = self.layers.pop(name, None)
        if previous is not None:
            self.group.remove(previous)
            if name.startswith("cinematic_"):
                self.renderer._presentation_materials.discard(previous.material)
        if obj is not None:
            if name.startswith("cinematic_"):
                if not hasattr(renderer, "_presentation_materials"):
                    renderer._presentation_materials = weakref.WeakSet()
                renderer._presentation_materials.add(obj.material)
            self.group.add(obj)
            self.layers[name] = obj
        renderer.request_redraw()

    def clear(self):
        for name, obj in self.layers.items():
            if name.startswith("cinematic_"):
                self.renderer._presentation_materials.discard(obj.material)
        if self.group is not None and self.group.parent is not None:
            self.group.parent.remove(self.group)
        self.group = None
        self.layers.clear()
        self.renderer.request_redraw()

    def paths(self, catalog, path_ids):
        """Show a disclosed canonical-ID prefix, never change analytical support."""
        gfx = self.renderer._gfx
        vertices = []
        used = 0
        shown = []
        for path_id in np.asarray(path_ids)[:MAX_PATHS]:
            points = catalog.path_points(int(path_id))
            segments = max(0, len(points) - 1)
            if used + segments > MAX_SEGMENTS:
                break
            if segments and np.all(np.isfinite(points)):
                vertices.append(np.stack((points[:-1], points[1:]), axis=1).reshape(-1, 3))
                used += segments
                shown.append(int(path_id))
        obj = None
        if vertices:
            obj = gfx.Line(
                gfx.Geometry(positions=np.concatenate(vertices).astype(np.float32)),
                gfx.LineSegmentMaterial(
                    color="#ffe66d", thickness=4.0, pick_write=False, depth_test=True
                ),
            )
        self._replace("paths", obj)
        return np.asarray(shown, dtype=np.int32)

    def pulses(self, positions):
        """Update only bounded pulse positions during the geometric travel clock."""
        gfx = self.renderer._gfx
        points = np.asarray(positions, dtype=np.float32)
        points = points[np.all(np.isfinite(points), axis=1)]
        old = self.layers.get("pulses")
        if old is not None and old.geometry.positions.data.shape == points.shape:
            old.geometry.positions.data[:] = points
            old.geometry.positions.update_full()
            self.renderer.request_redraw()
            return
        self._replace(
            "pulses",
            (
                gfx.Points(
                    gfx.Geometry(positions=points.copy()),
                    gfx.PointsMaterial(color="#ffffff", size=12, pick_write=False),
                )
                if len(points)
                else None
            ),
        )

    def clear_cinematic(self):
        if "paths" in self.layers:
            self.layers["paths"].visible = True
        for name in tuple(self.layers):
            if not name.startswith("cinematic_"):
                continue
            if name in self.layers:
                self.remove(name)

    def cinematic_routes(self, catalog, ids):
        """One bounded halo allocation per frozen selection, not per tick."""
        gfx = self.renderer._gfx
        vertices = []
        if "paths" in self.layers:
            self.layers["paths"].visible = False
        for path_id in ids:
            points = catalog.path_points(int(path_id))
            if len(points) > 1 and np.isfinite(points).all():
                vertices.extend(np.stack((points[:-1], points[1:]), axis=1).reshape(-1, 3))
        if vertices:
            mat = gfx.LineSegmentMaterial(
                color=(0.2, 0.65, 0.85, 0.13), thickness=7, pick_write=False, depth_write=False
            )
            self.renderer._apply_clipping_to_material(mat)
            self._replace(
                "cinematic_halo",
                gfx.Line(gfx.Geometry(positions=np.asarray(vertices, dtype=np.float32)), mat),
            )

    def set_cinematic_halo_opacity(self, opacity):
        """Update presentation opacity without rebuilding the recorded route buffer."""
        if not np.isfinite(opacity) or not 0 <= opacity <= 0.30:
            raise ValueError("Halo opacity must be between 0 and 0.30")
        halo = self.layers.get("cinematic_halo")
        if halo is not None:
            halo.material.color = (0.2, 0.65, 0.85, float(opacity))
            self.renderer.request_redraw()

    def clear_cinematic_pulses(self):
        """Keep the route context, but retire transient heads/trails/interactions."""
        for name in tuple(self.layers):
            if not name.startswith("cinematic_") or name == "cinematic_halo":
                continue
            if name in self.layers:
                self.remove(name)

    def cinematic_pulses(
        self, sample, *, interaction_types=None, type_markers=False, label_specs=()
    ):
        """Update preallocated positions/RGBA; no object rebuild at animation ticks."""
        gfx = self.renderer._gfx
        for name, positions, colors, size in (
            ("cinematic_head", sample.leading, sample.leading_colors, 13),
            ("cinematic_trail", sample.trail, sample.trail_colors, 7),
            ("cinematic_flash", sample.interactions, sample.interaction_colors, 23),
        ):
            old = self.layers.get(name)
            if old is not None and old.geometry.positions.data.shape == positions.shape:
                old.geometry.positions.data[:] = positions
                old.geometry.positions.update_full()
                old.geometry.colors.data[:] = colors
                old.geometry.colors.update_full()
                self.renderer._apply_clipping_to_material(old.material)
            elif len(positions):
                mat = gfx.PointsMaterial(
                    color_mode="vertex", size=size, pick_write=False, depth_write=False
                )
                self.renderer._apply_clipping_to_material(mat)
                self._replace(
                    name,
                    gfx.Points(gfx.Geometry(positions=positions.copy(), colors=colors.copy()), mat),
                )
            elif old is not None:
                self.remove(name)
        flash = self.layers.get("cinematic_flash")
        if flash is not None:
            from .mpc import PygfxMpcMixin

            typed = bool(type_markers and interaction_types is not None)
            if typed != isinstance(flash.material, gfx.PointsMarkerMaterial):
                self.renderer._presentation_materials.discard(flash.material)
                kwargs = dict(color_mode="vertex", size=23, pick_write=False, depth_write=False)
                if typed:
                    kwargs.update(
                        marker_mode="vertex",
                        edge_width=1,
                        edge_color="#20252b",
                        edge_color_mode="vertex",
                        custom_sdf="return size;",
                    )
                flash.material = (gfx.PointsMarkerMaterial if typed else gfx.PointsMaterial)(
                    **kwargs
                )
                self.renderer._presentation_materials.add(flash.material)
            if typed:
                edge = getattr(flash.geometry, "edge_colors", None)
                if edge is None or edge.data.shape != sample.interaction_colors.shape:
                    rgba = np.empty_like(sample.interaction_colors)
                    rgba[:, :3] = (0.125, 0.145, 0.169)
                    rgba[:, 3] = sample.interaction_colors[:, 3]
                    flash.geometry.edge_colors = gfx.Buffer(rgba)
                else:
                    edge.data[:, 3] = sample.interaction_colors[:, 3]
                    edge.update_full()
            if typed and getattr(flash, "_interaction_types_ref", None) is not interaction_types:
                codes = PygfxMpcMixin._interaction_marker_codes(interaction_types, gfx.MarkerInt)
                buffer = getattr(flash.geometry, "markers", None)
                if buffer is None or buffer.data.shape != codes.shape:
                    flash.geometry.markers = gfx.Buffer(codes)
                elif not np.array_equal(buffer.data, codes):
                    buffer.data[:] = codes
                    buffer.update_full()
                flash._interaction_types_ref = interaction_types
            self.renderer._apply_clipping_to_material(flash.material)
        # Labels share the exact marker visibility; never expose an unreached event.
        specs = tuple(label_specs)[:32]
        if specs != self._interaction_label_specs:
            for name in tuple(self.layers):
                if name.startswith("cinematic_label_"):
                    self.remove(name)
            self._interaction_label_specs = specs
        for slot, (index, text) in enumerate(specs):
            name = f"cinematic_label_{slot}"
            label = self.layers.get(name)
            if label is None:
                label = gfx.Text(
                    text=text,
                    font_size=13,
                    screen_space=True,
                    anchor="bottom-center",
                    material=gfx.TextMaterial(
                        color="white",
                        outline_color="#182028",
                        outline_thickness=0.25,
                        pick_write=False,
                        depth_write=False,
                        depth_test=False,
                        render_queue=4100,
                    ),
                )
                self._replace(name, label)
            rgba = sample.interaction_colors[index]
            label.local.position = sample.interactions[index]
            label.visible = bool(rgba[3] > 0)
            label.material.color = tuple(float(x) for x in rgba[:3])
            label.material.opacity = float(rgba[3])
            self.renderer._apply_clipping_to_material(label.material)
        halo = self.layers.get("cinematic_halo")
        if halo is not None:
            self.renderer._apply_clipping_to_material(halo.material)
        self.renderer.request_redraw()

    def directions(self, catalog, path_ids, *, departure=False):
        """Draw fixed display-length direction guides, not extra physical rays."""
        gfx = self.renderer._gfx
        segments = []
        for path_id in np.asarray(path_ids)[:MAX_PATHS]:
            points = catalog.path_points(int(path_id))
            if len(points) < 2 or not np.all(np.isfinite(points)):
                continue
            origin = points[0] if departure else points[-1]
            adjacent = points[1] if departure else points[-2]
            delta = adjacent - origin
            norm = np.linalg.norm(delta)
            if norm > 1e-10:
                segments.extend((origin, origin + 2.0 * delta / norm))
        self._replace(
            "directions",
            (
                gfx.Line(
                    gfx.Geometry(positions=np.asarray(segments, dtype=np.float32)),
                    gfx.LineSegmentMaterial(color="#49dfcf", thickness=3, pick_write=False),
                )
                if segments
                else None
            ),
        )

    def range_locus(self, locus, *, cutaway=False):
        """Draw translucent reference-pose geometry with no detection inference."""
        gfx = self.renderer._gfx
        group = gfx.Group()
        for vertices, triangles in locus.surfaces:
            material = gfx.MeshBasicMaterial(
                color=(0.16, 0.85, 0.90, 0.17),
                side="both",
                alpha_mode="weighted_blend",
                pick_write=False,
                depth_write=False,
            )
            if cutaway:
                material.clipping_planes = [(0.0, 0.0, 1.0, -float(locus.center[2]))]
            mesh = gfx.Mesh(
                gfx.Geometry(positions=vertices.astype(np.float32), indices=triangles), material
            )
            group.add(mesh)
        if len(locus.equator):
            group.add(
                gfx.Line(
                    gfx.Geometry(positions=locus.equator.astype(np.float32)),
                    gfx.LineMaterial(color="#fff3a1", thickness=3, pick_write=False),
                )
            )
        group.add(
            gfx.Points(
                gfx.Geometry(positions=locus.foci.astype(np.float32)),
                gfx.PointsMaterial(color="#ff9970", size=13, pick_write=False),
            )
        )
        self._replace("range", group)

    def angular_cell(self, cell):
        """Show recorded bin boundaries, without smoothing a measured field."""
        gfx = self.renderer._gfx
        group = gfx.Group()
        group.add(
            gfx.Mesh(
                gfx.Geometry(positions=cell.vertices, indices=cell.triangles),
                gfx.MeshBasicMaterial(
                    color=(0.16, 0.85, 0.90, 0.32),
                    side="both",
                    alpha_mode="weighted_blend",
                    pick_write=False,
                    depth_write=False,
                ),
            ),
            gfx.Line(
                gfx.Geometry(positions=cell.outline),
                gfx.LineMaterial(color="#fff3a1", thickness=3, pick_write=False),
            ),
        )
        self._replace("angle", group)

    def report_position(self, point):
        """Emphasize one saved spatial estimate while retaining scene occlusion."""
        gfx = self.renderer._gfx
        self._replace(
            "position",
            gfx.Points(
                gfx.Geometry(positions=np.asarray([point], dtype=np.float32)),
                gfx.PointsMarkerMaterial(
                    color="#ffe45c", size=20, marker="diamond", pick_write=False
                ),
            ),
        )

    def track_position(self, point, color):
        """A distinct saved-track anchor, not truth or another inferred position."""
        gfx = self.renderer._gfx
        material = gfx.PointsMarkerMaterial(color=color, size=22, marker="cross", pick_write=False)
        self.renderer._apply_clipping_to_material(material)
        self._replace(
            "track",
            gfx.Points(gfx.Geometry(positions=np.asarray([point], dtype=np.float32)), material),
        )

    def receive_response(self, obj):
        """Display the shared metadata-derived array mesh; radius is illustrative."""
        gfx = self.renderer._gfx
        payload = obj.payload
        self._replace(
            "response",
            gfx.Mesh(
                gfx.Geometry(
                    positions=payload.vertices,
                    indices=payload.triangles,
                    colors=payload.vertex_colors,
                ),
                gfx.MeshBasicMaterial(
                    color_mode="vertex",
                    opacity=obj.material.payload.base_color[3],
                    side="both",
                    alpha_mode="weighted_blend",
                    pick_write=False,
                    depth_write=False,
                ),
            ),
        )

    def remove(self, name):
        self._replace(name, None)
