"""Engine-independent configuration for rigid native sensing targets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from shared.scenarios.actors import (
    ConstantRCSSpec,
    FixedOrientationSpec,
    NativeBoxGeometrySpec,
    NativeMeshGeometrySpec,
    NativeTargetGeometrySpec,
    TR38901RCSSpec,
)

# Renderer-independent previews mirror the verified native preset geometry at
# NVlabs/sionna-rt 15b5ee0, rcs/tr38901/parameters.py and sensing_target.py.
# Runtime metadata always replaces these estimates with native accessors.
_PRESET_DIMENSIONS_M = {
    "uav-small-size": (0.3, 0.4, 0.2),
    "uav-large-size": (1.6, 1.5, 0.7),
    "human": (0.5, 0.5, 1.75),
    "vehicle-single-sp": (5.0, 2.0, 1.6),
    "vehicle-multi-sp": (5.0, 2.0, 1.6),
    "agv-single-sp": (0.5, 1.0, 0.5),
    "agv-multi-sp": (0.5, 1.0, 0.5),
}


@dataclass(frozen=True)
class NativeTargetConfig:
    """One native target and its sole absorbing, rigid geometry proxy.

    Mobility positions are the center of the target. Geometry is never an
    independently scattering asset and cannot carry articulated animation.
    The random model flags are seeded by the RCS solver, not the constructor.
    """

    name: str
    representation: Literal["constant_rcs", "tr38901"]
    geometry: NativeTargetGeometrySpec
    mobility: Any | None = None
    orientation: Any = field(default_factory=FixedOrientationSpec)
    constant_rcs: ConstantRCSSpec | None = None
    tr38901: TR38901RCSSpec | None = None
    resolved_mesh_path: Path | None = None
    _initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    switch_meshes: Literal[False] = field(default=False, init=False)
    use_ply_position: Literal[False] = field(default=False, init=False)
    animated_asset: None = field(default=None, init=False, repr=False)
    asset_front_yaw_offset_deg: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("native target name must be a non-empty string")
        if self.representation == "constant_rcs":
            if not isinstance(self.constant_rcs, ConstantRCSSpec) or self.tr38901 is not None:
                raise ValueError("constant_rcs target requires only ConstantRCSSpec")
        elif self.representation == "tr38901":
            if not isinstance(self.tr38901, TR38901RCSSpec) or self.constant_rcs is not None:
                raise ValueError("tr38901 target requires only TR38901RCSSpec")
        else:
            raise ValueError("native representation must be constant_rcs or tr38901")
        if isinstance(self.geometry, NativeMeshGeometrySpec):
            if self.resolved_mesh_path is None or not Path(self.resolved_mesh_path).is_absolute():
                raise ValueError("native mesh requires an absolute resolved_mesh_path")
            resolved = Path(self.resolved_mesh_path).resolve()
            if resolved.suffix.lower() not in {".ply", ".obj"}:
                raise ValueError("native mesh requires one static .ply or .obj proxy")
            object.__setattr__(self, "resolved_mesh_path", resolved)
        elif isinstance(self.geometry, NativeBoxGeometrySpec):
            if self.resolved_mesh_path is not None:
                raise ValueError("native box cannot also define a mesh path")
        else:
            raise ValueError(
                "native geometry must be NativeBoxGeometrySpec or NativeMeshGeometrySpec"
            )
        for key in ("_initial_position", "_initial_velocity"):
            raw = getattr(self, key)
            if isinstance(raw, (str, bytes)):
                raise ValueError(f"{key} must be a finite three-vector")
            values = tuple(float(component) for component in raw)
            if len(values) != 3 or not all(math.isfinite(component) for component in values):
                raise ValueError(f"{key} must be a finite three-vector")
            object.__setattr__(self, key, values)

    @property
    def initial_position(self) -> tuple[float, float, float]:
        """Return the center at the first prepared timeline sample."""
        return self._initial_position

    @property
    def initial_velocity(self) -> tuple[float, float, float]:
        """Return rigid-body velocity at the first prepared timeline sample."""
        return self._initial_velocity

    @property
    def scale(self) -> float:
        """Return uniform proxy scaling, already dimensioned for box geometry."""
        return self.geometry.scale if isinstance(self.geometry, NativeMeshGeometrySpec) else 1.0

    def preview_metadata(self) -> dict[str, Any]:
        """Describe native geometry without importing or initializing the engine.

        Box previews use the native preset dimensions/point placement. Mesh
        AABBs and their dependent points are intentionally deferred to the
        native runtime rather than inferred from an unrelated display asset.
        """
        geometry: dict[str, Any] = {
            "type": self.geometry.type,
            "scaling": (self.scale, self.scale, self.scale),
        }
        orientation = self.orientation
        if hasattr(orientation, "orientations"):
            angles = tuple(next(iter(orientation.orientations()), (0.0, 0.0, 0.0)))
        elif isinstance(orientation, FixedOrientationSpec):
            angles = (orientation.yaw_deg, orientation.pitch_deg, orientation.roll_deg)
        else:
            angles = (0.0, 0.0, 0.0)
        radians = tuple(math.radians(angle) for angle in angles)
        snapshot: dict[str, Any] = {
            "name": self.name,
            "representation": self.representation,
            "geometry": geometry,
            "current_position": self.initial_position,
            "orientation": radians,
            "velocity_m_s": self.initial_velocity,
            "metadata_source": "preset_preview",
            "upstream_version": None,
        }
        model = self.constant_rcs or self.tr38901
        assert model is not None
        snapshot[self.representation] = model.model_dump()
        if isinstance(self.geometry, NativeMeshGeometrySpec):
            geometry["path"] = self.geometry.path
            snapshot["point_positions_source"] = "awaiting_native_mesh"
            return snapshot
        defaults = (
            _PRESET_DIMENSIONS_M[self.tr38901.object_type]
            if self.tr38901 is not None
            else (0.3, 0.3, 0.3)
        )
        dimensions = tuple(
            default if getattr(self.geometry, key) is None else getattr(self.geometry, key)
            for key, default in zip(("length_m", "width_m", "height_m"), defaults)
        )
        geometry.update(zip(("length_m", "width_m", "height_m"), dimensions))
        length, width, height = dimensions
        local: tuple[tuple[float, float, float], ...]
        if self.tr38901 is not None and self.tr38901.num_scattering_points == 5:
            local = (
                (0.0, width / 2, 0.0),
                (-length / 2, 0.0, 0.0),
                (0.0, -width / 2, 0.0),
                (length / 2, 0.0, 0.0),
                (0.0, 0.0, height / 2),
            )
            if self.tr38901.object_type == "agv-multi-sp":
                local = (local[3], local[0], local[1], local[2], local[4])
        elif self.tr38901 is not None and self.tr38901.object_type == "human":
            local = ((0.0, 0.0, height * (1.5 / 1.75 - 0.5)),)
        else:
            local = ((0.0, 0.0, 0.0),)
        yaw, pitch, roll = radians
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rotation = (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
        world = tuple(
            tuple(
                self.initial_position[i] + sum(a * b for a, b in zip(row, point))
                for i, row in enumerate(rotation)
            )
            for point in local
        )
        snapshot.update(
            scattering_points_local_m=local,
            scattering_points_world_m=world,
            scattering_point_ids=tuple(range(len(local))),
            point_positions_source="preset",
        )
        return snapshot


__all__ = ["NativeTargetConfig"]
