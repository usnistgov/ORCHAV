"""Configuration model for mesh-backed target scene objects.

``TargetConfig`` is the YAML/API-side representation of a target. It records the
mesh sequence to load, the material convention to use, optional mobility and
orientation patterns, and playback controls such as start index and stride. The
live Sionna ``SceneObject`` is created later by ``TargetManager``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, cast

from shared.scenarios.actors import FixedOrientationSpec, OrientationSpec

from ..orientation.base import PreparedOrientationSource
from ..utils import point_to_tuple
from .metadata import load_target_asset_metadata

if TYPE_CHECKING:
    from ..mobility import MobilityPattern


# Treat sub-nanodegree deltas as unchanged so JSON/float round-trip noise does
# not perturb authored orientations or make no-op metadata refreshes look real.
TARGET_ASSET_FRONT_YAW_TOLERANCE_DEG = 1e-9


@dataclass
class TargetConfig:
    """Configuration for one mesh-backed target object in the scene.

    ``mesh_start_index`` and ``mesh_frame_stride`` select the sequence of mesh
    files used for animation. ``mesh_end_behavior`` controls whether the
    sequence loops or holds its final mesh. ``mesh_directory`` preserves the
    portable scenario-authored value for frame metadata, while
    ``resolved_mesh_directory`` is the absolute generator I/O location.
    ``use_ply_position`` means the mesh vertex coordinates carry the target's
    world-space trajectory; otherwise mobility controls target position.
    """

    name: str
    mobility: Optional["MobilityPattern"]
    mesh_pattern: str
    mesh_directory: str
    resolved_mesh_directory: Path
    scale: float = 1.0
    orientation: OrientationSpec | PreparedOrientationSource = field(
        default_factory=FixedOrientationSpec
    )
    material_type: str = "glass"
    switch_meshes: bool = True
    use_ply_position: bool = False
    mesh_start_index: int = 0
    mesh_frame_stride: int = 1
    mesh_end_behavior: Literal["loop", "hold_last"] = "loop"
    asset_front_yaw_offset_deg: Optional[float] = None
    _initial_position: Optional[Tuple[float, float, float]] = None
    _asset_front_yaw_offset_from_metadata: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate mesh playback fields and position ownership."""
        resolved_mesh_directory = Path(self.resolved_mesh_directory)
        if not resolved_mesh_directory.is_absolute():
            raise ValueError(f"TargetConfig {self.name}: resolved_mesh_directory must be absolute")
        self.resolved_mesh_directory = resolved_mesh_directory.resolve()
        self._initialize_asset_front_yaw_offset()

        try:
            self.mesh_start_index = int(self.mesh_start_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TargetConfig {self.name}: mesh_start_index must be an integer"
            ) from exc
        try:
            self.mesh_frame_stride = int(self.mesh_frame_stride)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TargetConfig {self.name}: mesh_frame_stride must be an integer"
            ) from exc
        if self.mesh_start_index < 0:
            raise ValueError(f"TargetConfig {self.name}: mesh_start_index must be >= 0")
        if self.mesh_frame_stride < 1:
            raise ValueError(f"TargetConfig {self.name}: mesh_frame_stride must be >= 1")
        if self.mesh_end_behavior not in ("loop", "hold_last"):
            raise ValueError(
                f"TargetConfig {self.name}: mesh_end_behavior must be 'loop' or 'hold_last'"
            )
        if self._initial_position is not None and self.mobility is not None:
            mobility = cast(Any, self.mobility)
            if hasattr(mobility, "start_pos"):
                mobility_start = point_to_tuple(mobility.start_pos)
                initial_position = point_to_tuple(self._initial_position)
                if mobility_start != initial_position:
                    raise ValueError(
                        f"TargetConfig {self.name}: mobility.start_pos {mobility_start} doesn't match initial_position {initial_position}"
                    )

    def _initialize_asset_front_yaw_offset(self) -> None:
        """Set initial target-front alignment from metadata or explicit config.

        ``None`` means the target should follow metadata in the resolved asset
        directory. A numeric value is an explicit scenario/API override and
        must not be replaced by metadata.
        """
        if self.asset_front_yaw_offset_deg is not None:
            self.apply_asset_front_yaw_offset(float(self.asset_front_yaw_offset_deg))
            return

        self._asset_front_yaw_offset_from_metadata = True
        metadata = load_target_asset_metadata(self.resolved_mesh_directory)
        self.apply_asset_front_yaw_offset(metadata.front_yaw_offset_deg)

    def apply_asset_front_yaw_offset(self, yaw_offset_deg: float) -> None:
        """Record the target-library front alignment for canonical preparation.

        Target meshes do not all model "front" along the same local axis. The
        metadata offset lets a mesh library declare that convention once, while
        preserving the immutable scenario-authored orientation. Actor-state
        preparation composes this local rotation in quaternion space.
        """

        self.asset_front_yaw_offset_deg = float(yaw_offset_deg)

    @property
    def initial_position(self):
        """Get the initial scene position for target creation.

        Mobility-owned targets derive this from the mobility pattern. Static
        targets need an explicit initial position unless their mesh sequence is
        later used directly for per-frame positions.
        """
        if self.mobility is None:
            if self._initial_position is None:
                raise ValueError(
                    f"TargetConfig {self.name}: stationary target requires explicit initial_position"
                )
            return self._initial_position
        mobility = cast(Any, self.mobility)
        if hasattr(mobility, "start_pos"):
            return point_to_tuple(mobility.start_pos)
        if hasattr(mobility, "prepare") and hasattr(mobility, "prepared_positions"):
            try:
                mobility.prepare(1, 1.0, getattr(mobility, "start_pos", None))
                positions = mobility.prepared_positions()
                if positions:
                    return point_to_tuple(positions[0])
            except (AttributeError, ValueError, TypeError, RuntimeError):
                pass
        if self._initial_position is not None:
            return point_to_tuple(self._initial_position)
        raise ValueError(
            f"TargetConfig {self.name}: {self.mobility.__class__.__name__} requires explicit initial_position"
        )
