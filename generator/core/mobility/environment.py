#!/usr/bin/env python3
"""Environment-aware meshgrid mobility patterns.

These classes start from ``MeshGridMobility`` and remove candidate points based
on scene mesh occupancy. Fast bounding-box filtering is cheap and conservative;
Open3D raycasting is more precise when available and requested.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.logging import get_logger

from ..utils import point_to_tuple
from .base import Position3
from .grid import MeshGridMobility

logger = get_logger(__name__)


class _EnvironmentMeshGridMobility(MeshGridMobility):
    """
    Base class for meshgrid mobility patterns that filter positions based on scene geometry.

    Supports multiple filtering strategies selectable via `filter_mode`:
        - 'bbox': Fast axis-aligned bounding boxes (default)
        - 'raycast': Accurate signed-distance queries using Open3D ray casting

    Subclasses implement `_point_passes_filter` to define whether a grid point should be kept.
    """

    FILTER_LABEL = "environment"

    def __init__(
        self,
        scene_geometry: list[Any] | None = None,
        min_distance_from_buildings: float = 0.0,
        use_scipy: bool = True,
        fallback_to_open3d: bool = True,
        filter_mode: str = "bbox",
        **kwargs,
    ):
        """
        Initialize environment-filtered meshgrid mobility.

        Args:
            scene_geometry: Sequence of mesh objects from scene loading (optional)
            min_distance_from_buildings: Minimum distance from building surfaces (meters)
            use_scipy: Whether to use SciPy for point-in-mesh testing
            fallback_to_open3d: Whether to fallback to Open3D if SciPy is unavailable
            filter_mode: Filtering strategy ('bbox' or 'raycast')
            **kwargs: All arguments from MeshGridMobility (x_bounds, y_bounds, etc.)
        """
        super().__init__(**kwargs)

        self.scene_geometry = scene_geometry or []
        self.min_distance_from_buildings = min_distance_from_buildings
        self.use_scipy = use_scipy
        self.fallback_to_open3d = fallback_to_open3d
        self.filter_mode_requested = filter_mode.lower()
        if self.filter_mode_requested not in ("bbox", "raycast"):
            raise ValueError("filter_mode must be one of: 'bbox', 'raycast'")

        self._building_meshes = []
        self._building_bboxes = []
        self._has_scipy = False
        self._has_open3d = False
        self._raycasting_scene = None

        self._check_available_libraries()
        self._process_scene_geometry()
        self._precompute_bounding_boxes()
        self._active_filter_mode = self.filter_mode_requested
        if self.filter_mode_requested == "raycast":
            self._raycasting_scene = self._build_raycast_scene()
            if self._raycasting_scene is None:
                # Raycasting needs Open3D tensor geometry; fall back to bbox
                # filtering so scenarios still generate with reduced precision.
                logger.warning(
                    "%s: Raycast filter requested but unavailable, falling back to bounding boxes",
                    self.__class__.__name__,
                )
                self._active_filter_mode = "bbox"
        else:
            self._active_filter_mode = "bbox"

        self._base_traversal_order = list(self._traversal_order)
        self._original_total_points = len(self.grid_points)
        self._filtered_grid_points, self._filtered_grid_indices = self._filter_points()
        self.total_points = len(self._filtered_grid_points)
        self._recompute_filtered_traversal_order()

        logger.info(
            "%s: %d filtered (%s) points from %d total",
            self.__class__.__name__,
            self.total_points,
            self.FILTER_LABEL,
            self._original_total_points,
        )

    def _check_available_libraries(self):
        """Check which collision detection libraries are available."""
        if self.use_scipy:
            try:
                from scipy.spatial import ConvexHull  # noqa: F401
                from scipy.spatial.distance import cdist  # noqa: F401

                self._has_scipy = True
            except ImportError:
                self._has_scipy = False

        if self.fallback_to_open3d or self.filter_mode_requested == "raycast":
            try:
                import open3d  # noqa: F401

                self._has_open3d = True
            except ImportError:
                self._has_open3d = False

        if not self._has_scipy and not self._has_open3d:
            raise ImportError(
                "Neither SciPy nor Open3D is available for collision detection. "
                "Please install one of them."
            )

    def _process_scene_geometry(self):
        """Process scene geometry to extract building meshes for collision detection."""
        if not self.scene_geometry:
            return

        for mesh_info in self.scene_geometry:
            try:
                mesh = mesh_info.get("mesh")
                if mesh is None:
                    continue

                if self._has_scipy:
                    vertices = self._extract_vertices(mesh)
                    triangles = self._extract_triangles(mesh)
                    if vertices is not None and len(vertices) > 0:
                        self._building_meshes.append(
                            {
                                "type": "scipy",
                                "vertices": vertices,
                                "triangles": triangles,
                                "name": mesh_info.get("name", "unknown"),
                            }
                        )
                elif self._has_open3d:
                    o3d_mesh = self._convert_to_open3d_mesh(mesh)
                    if o3d_mesh is not None:
                        self._building_meshes.append(
                            {
                                "type": "open3d",
                                "mesh": o3d_mesh,
                                "name": mesh_info.get("name", "unknown"),
                            }
                        )
            except (KeyError, AttributeError, TypeError) as e:
                logger.warning("Failed to process mesh %s: %s", mesh_info.get("name", "unknown"), e)

    def _precompute_bounding_boxes(self):
        """Pre-compute bounding boxes for all building meshes for fast collision detection."""
        self._building_bboxes = []

        for building in self._building_meshes:
            try:
                if building["type"] == "scipy":
                    vertices = building["vertices"]
                elif building["type"] == "open3d":
                    vertices = np.asarray(building["mesh"].vertices)
                else:
                    continue

                if len(vertices) == 0:
                    continue

                min_coords = np.min(vertices, axis=0) - self.min_distance_from_buildings
                max_coords = np.max(vertices, axis=0) + self.min_distance_from_buildings

                self._building_bboxes.append(
                    {"min": min_coords, "max": max_coords, "name": building["name"]}
                )
            except (KeyError, ValueError) as e:
                logger.warning(
                    "Failed to compute bounding box for %s: %s", building.get("name", "unknown"), e
                )

        logger.debug("Pre-computed %d building bounding boxes", len(self._building_bboxes))

    def _convert_to_open3d_mesh(self, mesh):
        """Convert a mesh to Open3D TriangleMesh format."""
        try:
            import open3d as o3d

            vertices = self._extract_vertices(mesh)
            if vertices is None or len(vertices) == 0:
                return None

            triangles = self._extract_triangles(mesh)
            if triangles is None or len(triangles) == 0:
                return None

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            o3d_mesh.remove_degenerate_triangles()
            o3d_mesh.remove_duplicated_triangles()
            o3d_mesh.remove_duplicated_vertices()
            o3d_mesh.remove_unreferenced_vertices()
            return o3d_mesh
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Failed to convert mesh to Open3D: %s", e)
            return None

    def _create_open3d_mesh_from_arrays(self, vertices: np.ndarray, triangles: np.ndarray | None):
        """Create an Open3D mesh from raw vertices/triangles arrays."""
        try:
            import open3d as o3d

            if vertices is None or len(vertices) == 0 or triangles is None or len(triangles) == 0:
                return None
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            return mesh
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Failed to build Open3D mesh from arrays: %s", e)
            return None

    def _build_raycast_scene(self):
        """Construct an Open3D raycasting scene for precise point-in-mesh testing."""
        if not self._has_open3d:
            return None
        try:
            import open3d as o3d

            scene = o3d.t.geometry.RaycastingScene()
            mesh_count = 0
            for building in self._building_meshes:
                legacy_mesh = None
                if building["type"] == "open3d":
                    legacy_mesh = building["mesh"]
                elif building["type"] == "scipy":
                    legacy_mesh = self._create_open3d_mesh_from_arrays(
                        building.get("vertices"), building.get("triangles")
                    )
                if legacy_mesh is None:
                    continue
                t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
                scene.add_triangles(t_mesh)
                mesh_count += 1
            if mesh_count == 0:
                return None
            logger.info(
                "%s: Raycasting scene built with %d meshes", self.__class__.__name__, mesh_count
            )
            return scene
        except (ImportError, RuntimeError) as e:
            logger.warning("Failed to build raycasting scene: %s", e)
            return None

    def _point_to_numpy(self, point: Any) -> np.ndarray:
        """Convert input point to numpy array."""
        return np.asarray(point_to_tuple(point), dtype=np.float64)

    def _compute_signed_distance(self, point: np.ndarray) -> float | None:
        """Compute signed distance to the nearest surface using raycasting scene."""
        if self._active_filter_mode != "raycast" or self._raycasting_scene is None:
            return None
        try:
            import open3d as o3d

            query_np = np.asarray(point, dtype=np.float32).reshape(1, 3)
            query_tensor = o3d.core.Tensor(query_np, dtype=o3d.core.Dtype.Float32)
            sdf_tensor = self._raycasting_scene.compute_signed_distance(query_tensor)
            if hasattr(sdf_tensor, "numpy"):
                sdf_np = sdf_tensor.numpy()
            else:
                sdf_np = np.asarray(sdf_tensor)
            sdf_np = np.asarray(sdf_np, dtype=np.float32).reshape(-1)
            if sdf_np.size == 0:
                return None
            return float(sdf_np[0])
        except (ImportError, RuntimeError) as e:
            logger.warning("Raycast signed-distance query failed: %s", e)
            self._raycasting_scene = None
            self._active_filter_mode = "bbox"
            return None

    def _extract_vertices(self, mesh):
        """Extract vertices from a mesh object."""
        try:
            if hasattr(mesh, "vertices"):
                vertices = np.asarray(mesh.vertices)
                if vertices.size > 0:
                    return vertices
            return None
        except (ValueError, TypeError):
            return None

    def _extract_triangles(self, mesh):
        """Extract triangles/faces from a mesh object."""
        try:
            faces = None
            if hasattr(mesh, "triangles"):
                faces = np.asarray(mesh.triangles)
            elif hasattr(mesh, "faces"):
                faces = np.asarray(mesh.faces)

            if faces is not None and faces.size > 0 and faces.shape[1] >= 3:
                return faces[:, :3]
            return None
        except (ValueError, TypeError, IndexError):
            return None

    def _filter_points(self):
        """Filter grid points using the subclass predicate."""
        filtered_points = []
        filtered_indices = []

        for idx, point in enumerate(self.grid_points):
            if self._point_passes_filter(point):
                filtered_points.append(point)
                filtered_indices.append(idx)

        return filtered_points, filtered_indices

    def _point_passes_filter(self, point: Any) -> bool:
        """Determine whether a point passes the subclass filter."""
        raise NotImplementedError("Subclasses must implement _point_passes_filter")

    def _recompute_filtered_traversal_order(self):
        """Rebuild traversal order so indices align with filtered grid points."""
        if not self._filtered_grid_points:
            self._traversal_order = []
            return

        index_map = {
            original_idx: filtered_idx
            for filtered_idx, original_idx in enumerate(self._filtered_grid_indices)
        }

        filtered_order = [index_map[idx] for idx in self._base_traversal_order if idx in index_map]

        if not filtered_order:
            filtered_order = list(range(len(self._filtered_grid_points)))

        self._traversal_order = filtered_order

    def _is_point_in_bounding_box_fast(
        self, point: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray
    ) -> bool:
        """Ultra-fast bounding box test using vectorized operations."""
        return bool(np.all(point >= min_coords) and np.all(point <= max_coords))

    def _point_intersects_building_bbox(self, point: np.ndarray) -> bool:
        """Check if a point lies within any building bounding box."""
        for bbox in self._building_bboxes:
            if self._is_point_in_bounding_box_fast(point, bbox["min"], bbox["max"]):
                return True
        return False

    def get_grid_info(self) -> dict[str, Any]:
        """Get information about the filtered grid."""
        base_info = super().get_grid_info()
        base_info.update(
            {
                "filtered_points": self.total_points,
                "total_grid_points": self._original_total_points,
                "building_meshes": len(self._building_meshes),
                "filter_label": self.FILTER_LABEL,
                "filter_mode_requested": self.filter_mode_requested,
                "filter_mode_active": self._active_filter_mode,
                "min_distance_from_buildings": self.min_distance_from_buildings,
                "collision_detection": {
                    "method_used": (
                        "raycast" if self._active_filter_mode == "raycast" else "fast_bounding_box"
                    ),
                    "building_bboxes": len(self._building_bboxes),
                    "scipy_available": self._has_scipy,
                    "open3d_available": self._has_open3d,
                    "raycast_scene": self._raycasting_scene is not None,
                },
            }
        )
        return base_info

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Get positions for the specified number of scene steps using filtered points."""
        if not self._filtered_grid_points:
            raise ValueError(f"No {self.FILTER_LABEL} grid points available")

        positions: list[Position3] = []
        for i in range(scene_steps):
            if self.auto_expand_scene_steps and i < self.total_points:
                point_idx = self._traversal_order[i]
            else:
                point_idx = self._traversal_order[i % self.total_points]
            positions.append(point_to_tuple(self._filtered_grid_points[point_idx]))
        return positions


class OutdoorMeshGridMobility(_EnvironmentMeshGridMobility):
    """
    Meshgrid mobility that keeps only outdoor (non-building) positions.

    Filtering strategy is controlled by `filter_mode` ('bbox' or 'raycast').
    """

    FILTER_LABEL = "outdoor"

    def _point_passes_filter(self, point: Any) -> bool:
        """Outdoor points must be outside building geometry and clearance buffers."""
        point_array = self._point_to_numpy(point)
        sdf = self._compute_signed_distance(point_array)
        if sdf is not None:
            if self.min_distance_from_buildings > 0:
                inside = (sdf < 0) or (sdf <= self.min_distance_from_buildings)
            else:
                inside = sdf < 0
        else:
            inside = self._point_intersects_building_bbox(point_array)
        return not inside

    def get_grid_info(self) -> dict[str, Any]:
        """Return filtered grid information with the outdoor point count."""
        info = super().get_grid_info()
        info["outdoor_points"] = info["filtered_points"]
        return info


class IndoorMeshGridMobility(_EnvironmentMeshGridMobility):
    """
    Meshgrid mobility that keeps positions located inside building volumes.

    Intended for indoor scenes (e.g., lobby.xml) where the entire environment
    represents a single interior, so all scene meshes are considered indoors.
    Supports `filter_mode` ('bbox' or 'raycast') for interior detection.
    """

    FILTER_LABEL = "indoor"

    def __init__(
        self,
        scene_geometry: list[Any] | None = None,
        min_distance_from_buildings: float = 0.0,
        use_scipy: bool = True,
        fallback_to_open3d: bool = True,
        filter_mode: str = "bbox",
        **kwargs,
    ):
        super().__init__(
            scene_geometry=scene_geometry,
            min_distance_from_buildings=min_distance_from_buildings,
            use_scipy=use_scipy,
            fallback_to_open3d=fallback_to_open3d,
            filter_mode=filter_mode,
            **kwargs,
        )

    def _point_passes_filter(self, point: Any) -> bool:
        """Indoor points must be inside building/interior geometry."""
        point_array = self._point_to_numpy(point)
        sdf = self._compute_signed_distance(point_array)
        if sdf is not None:
            if self.min_distance_from_buildings > 0:
                inside = sdf <= -self.min_distance_from_buildings
            else:
                inside = sdf < 0
        else:
            inside = self._point_intersects_building_bbox(point_array)
        return inside

    def get_grid_info(self) -> dict[str, Any]:
        """Return filtered grid information with the indoor point count."""
        info = super().get_grid_info()
        info["indoor_points"] = info["filtered_points"]
        return info
