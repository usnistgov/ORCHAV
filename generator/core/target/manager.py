"""Live scene-object management for mesh-backed targets.

``TargetManager`` is the runtime counterpart to ``TargetConfig``. It resolves
mesh paths, loads target metadata, creates the Sionna ``SceneObject``, and
applies prepared position/orientation/mesh snapshots during frame generation.
Mesh file compatibility helpers live in ``mesh.py`` so this module can focus on
scene ownership and per-frame updates.
"""

import os
from typing import Any, cast

import mitsuba as mi
from sionna.rt.scene_object import SceneObject

from shared.logging import get_logger
from shared.scenarios.actors import FixedOrientationSpec, LookAtOrientationSpec

from ..exceptions import ComputationError
from ..materials.target_materials import create_target_material
from ..orientation.base import (
    Orientation3,
    PreparedOrientationSource,
    orientation_to_tuple,
)
from ..scenario_actors import (
    PreparedOrientation,
    Quaternion,
    Timeline,
    apply_asset_alignment,
    prepare_orientation,
    prepare_sampled_mobility,
)
from ..sionna_integration import orientation_to_point3f_with_engine_radians, point3f
from ..utils import point_to_tuple
from .config import TARGET_ASSET_FRONT_YAW_TOLERANCE_DEG, TargetConfig
from .mesh import (
    _ply_header_has_faces,
    _prepare_mesh_path_for_mitsuba,
    mesh_sequence_index,
)

logger = get_logger(__name__)


class TargetManager:
    """Manage one mesh-backed target ``SceneObject`` and its updates."""

    def __init__(self, target_config: TargetConfig, scene, material_overrides=None):
        self.config = target_config
        self.scene = scene
        self.material_overrides = material_overrides or {}

        self.relative_mesh_directory = target_config.mesh_directory
        self.resolved_mesh_directory = target_config.resolved_mesh_directory

        self.meshes = self.load_mesh_sequence()
        mesh_end_behavior = getattr(self.config, "mesh_end_behavior", "loop")
        if self.meshes and self.config.mesh_start_index >= len(self.meshes):
            if mesh_end_behavior == "hold_last":
                logger.warning(
                    "mesh_start_index=%d exceeds mesh count=%d for target %s; "
                    "holding at final index %d",
                    self.config.mesh_start_index,
                    len(self.meshes),
                    self.config.name,
                    len(self.meshes) - 1,
                )
            else:
                logger.warning(
                    "mesh_start_index=%d exceeds mesh count=%d for target %s; wrapping to %d",
                    self.config.mesh_start_index,
                    len(self.meshes),
                    self.config.name,
                    self.config.mesh_start_index % len(self.meshes),
                )
        self.current_mesh_idx = self._mesh_index_for_update_call(0) if self.meshes else 0
        # Counts mesh-playback updates, which can be lower-rate than frame
        # updates when mesh_update_interval_s is configured.
        self._mesh_call_count = 0
        self.target_object = None

        self.material_type = target_config.material_type
        self.target_material = None

        logger.debug("Target material type for %s: %s", self.config.name, self.material_type)
        if self.meshes:
            logger.info(
                "Target mesh playback for %s starts at index %d with stride %d (%s)",
                self.config.name,
                self.current_mesh_idx,
                self.config.mesh_frame_stride,
                mesh_end_behavior,
            )

    def _mesh_index_for_update_call(self, mesh_call_count: int) -> int:
        """Return the mesh sequence index selected for a playback update count.

        ``mesh_start_index`` selects the first frame, ``mesh_frame_stride`` skips
        source mesh frames. The configured end behavior either preserves
        modulo wraparound or holds the final source mesh.
        """
        if not self.meshes:
            return 0
        return mesh_sequence_index(
            len(self.meshes),
            mesh_call_count,
            mesh_start_index=self.config.mesh_start_index,
            mesh_frame_stride=self.config.mesh_frame_stride,
            mesh_end_behavior=getattr(self.config, "mesh_end_behavior", "loop"),
        )

    def create_material(self) -> None:
        """Create the per-target ITU material used by the Sionna SceneObject."""
        self.target_material = create_target_material(
            material_type=self.material_type,
            target_name=self.config.name,
            material_overrides=self.material_overrides,
        )

    def load_mesh_sequence(self) -> list[str]:
        """Load all meshes matching the pattern.

        A target configuration is invalid if its mesh directory or pattern does
        not resolve to at least one mesh. Failing here prevents long ray-tracing
        runs from silently omitting targets.
        """
        import glob
        import os

        mesh_dir = self.resolved_mesh_directory

        logger.debug(f"Loading meshes for {self.config.name}")
        logger.debug(f"Directory (relative): {self.relative_mesh_directory}")
        logger.debug(f"Directory (resolved): {mesh_dir}")
        logger.debug(f"Pattern: {self.config.mesh_pattern}")

        if not os.path.exists(mesh_dir):
            logger.debug(f"Original relative path was: {self.relative_mesh_directory}")
            raise FileNotFoundError(
                f"Target '{self.config.name}' mesh directory not found: {mesh_dir} "
                f"(configured as {self.relative_mesh_directory!r})"
            )
        if not os.path.isdir(mesh_dir):
            raise NotADirectoryError(
                f"Target '{self.config.name}' mesh path is not a directory: {mesh_dir}"
            )

        # Find all files matching the pattern
        pattern = os.path.join(mesh_dir, self.config.mesh_pattern)
        logger.debug(f"Full pattern: {pattern}")

        mesh_files = glob.glob(pattern)
        # Mesh names define animation order; deterministic sorting keeps target
        # playback reproducible across filesystems.
        mesh_files.sort()

        logger.info(f"Loaded {len(mesh_files)} mesh files for {self.config.name}")

        if not mesh_files:
            try:
                all_files = os.listdir(mesh_dir)
                ply_files = [f for f in all_files if f.endswith(".ply")]
                obj_files = [f for f in all_files if f.endswith(".obj")]
                logger.debug(f"Directory contents: PLY={len(ply_files)}, OBJ={len(obj_files)}")
            except OSError as e:
                logger.error(f"Error listing directory: {e}")
            raise FileNotFoundError(
                f"Target '{self.config.name}' has no meshes matching {self.config.mesh_pattern!r} "
                f"in {mesh_dir}"
            )

        return mesh_files

    def _apply_initial_orientation_preview(self, target: SceneObject, pos: mi.Point3f) -> None:
        """Apply a creation-time preview when the orientation is self-contained.

        Actor-referenced look-at orientations depend on the complete scene and
        are applied by ``ActorStateManager``. An already-prepared time series
        exposes its first sample without being resampled. Self-contained schema
        orientations can be evaluated immediately.
        """
        orientation = self.config.orientation

        if isinstance(orientation, LookAtOrientationSpec) and orientation.actor is not None:
            logger.debug(
                "Deferring actor-referenced look-at orientation for target '%s' "
                "until actor context is prepared",
                self.config.name,
            )
            return

        if isinstance(orientation, PreparedOrientationSource):
            try:
                first_sample = next(iter(orientation.orientations()))
            except RuntimeError:
                logger.debug(
                    "Deferring unprepared orientation for target '%s' until its timeline is applied",
                    self.config.name,
                )
                return
            except StopIteration:
                logger.debug(
                    "Deferring empty prepared orientation for target '%s' until its timeline is applied",
                    self.config.name,
                )
                return
            prepared = PreparedOrientation(
                (Quaternion.from_euler_deg(*orientation_to_tuple(first_sample)),),
                asset_alignment_applied=bool(
                    getattr(orientation, "asset_alignment_applied", False)
                ),
            )
        else:
            if not isinstance(orientation, (FixedOrientationSpec, LookAtOrientationSpec)):
                logger.debug(
                    "Deferring timeline-dependent orientation for target '%s'",
                    self.config.name,
                )
                return
            timeline = Timeline(1, 1.0)
            mobility = prepare_sampled_mobility(
                (point_to_tuple(pos),),
                timeline,
                path=f"targets.{self.config.name}.mobility",
            )
            prepared = prepare_orientation(
                orientation,
                timeline,
                mobility,
                path=f"targets.{self.config.name}.orientation",
            )
        asset_yaw = float(self.config.asset_front_yaw_offset_deg or 0.0)
        if abs(asset_yaw) > TARGET_ASSET_FRONT_YAW_TOLERANCE_DEG and not (
            prepared.asset_alignment_applied
        ):
            prepared = apply_asset_alignment(
                prepared,
                (asset_yaw, 0.0, 0.0),
                path=f"targets.{self.config.name}.asset_alignment",
            )
        orientations = list(prepared.euler_deg)
        if orientations and any(abs(angle) > 1e-6 for angle in orientations[0]):
            target.orientation, engine_orientation = orientation_to_point3f_with_engine_radians(
                orientations[0]
            )
            logger.debug(
                "Applied initial orientation preview: %s (converted to radians: %.4f, %.4f, %.4f)",
                orientations[0],
                engine_orientation[0],
                engine_orientation[1],
                engine_orientation[2],
            )

    def create_target(self) -> SceneObject:
        """Create, configure, and register the required target object."""
        if not self.meshes:
            raise ComputationError(
                f"Target {self.config.name!r} cannot be created: no mesh is available "
                f"in {self.resolved_mesh_directory}"
            )

        mesh_path = self.meshes[self.current_mesh_idx]
        candidate = None
        candidate_add_attempted = False
        previous_target = self.target_object
        try:
            logger.debug(f"Loading mesh: {mesh_path}")

            if not os.path.exists(mesh_path):
                raise FileNotFoundError(mesh_path)

            if _ply_header_has_faces(mesh_path) is False:
                logger.warning(
                    f"PLY file '{mesh_path}' appears to be a point cloud (no faces). "
                    "SceneObject requires meshes with faces for raytracing. "
                    "Consider converting the point cloud to a mesh first."
                )

            if self.target_material is None:
                self.create_material()
                logger.debug(f"Material created: {type(self.target_material).__name__}")

            logger.debug("Creating SceneObject...")

            scene_mesh_path = _prepare_mesh_path_for_mitsuba(mesh_path)
            candidate = SceneObject(
                fname=scene_mesh_path, name=self.config.name, radio_material=self.target_material
            )

            logger.debug("SceneObject created successfully")

            candidate_add_attempted = True
            self.scene.edit(add=[candidate])
            logger.debug("Target added to scene successfully")

            if self.config.use_ply_position:
                logger.debug("Using position from PLY file")
                temp_pos = point3f((0.0, 0.0, 0.0))
                candidate.position = temp_pos
                logger.debug("Initialized target at origin before applying PLY transform")
                if abs(self.config.scale - 1.0) > 1e-6:
                    candidate.scaling = mi.Vector3f(
                        self.config.scale, self.config.scale, self.config.scale
                    )
                    logger.debug(f"Applied scaling: {self.config.scale:.2f}")
            else:
                logger.debug("Setting manual position")
                pos = point3f(self.config.initial_position)
                candidate.position = pos

                if abs(self.config.scale - 1.0) > 1e-6:
                    candidate.scaling = mi.Vector3f(
                        self.config.scale, self.config.scale, self.config.scale
                    )
                    logger.debug(f"Applied scaling: {self.config.scale:.2f}")

                self._apply_initial_orientation_preview(candidate, pos)

            logger.debug(f"Final position: {candidate.position}")
            logger.debug(f"Final scaling: {candidate.scaling}")
            logger.debug(f"Final orientation: {candidate.orientation}")

            self.target_object = candidate
            logger.info(
                f"Created target '{self.config.name}' with mesh: {os.path.basename(mesh_path)}"
            )
            logger.debug(f"Target object reference: {self.target_object}")
            logger.debug(f"Target object type: {type(self.target_object)}")
            logger.debug(f"Target object has position: {hasattr(self.target_object, 'position')}")
            if hasattr(self.target_object, "position"):
                logger.debug(f"Target object position: {self.target_object.position}")
            return candidate

        except Exception as exc:  # noqa: BLE001 - external scene failures require cleanup
            self.target_object = previous_target
            cleanup_error = None
            if candidate_add_attempted and candidate is not None:
                try:
                    getter = getattr(self.scene, "get", None)
                    try:
                        active = getter(self.config.name) if callable(getter) else candidate
                    except Exception:  # noqa: BLE001
                        active = candidate
                    if active is candidate:
                        self.scene.edit(remove=[candidate])
                except Exception as cleanup_exc:  # noqa: BLE001
                    cleanup_error = cleanup_exc
            message = f"Target {self.config.name!r} creation failed for mesh {mesh_path}: {exc}"
            if cleanup_error is not None:
                message += f"; candidate cleanup also failed: {cleanup_error}"
            raise ComputationError(message) from exc

    def apply_position_snapshot(self, new_position: tuple[float, float, float]) -> None:
        """Apply a prepared target position snapshot to the live scene object.

        Args:
            new_position: (x, y, z) world coordinates
        """
        if self.target_object is None:
            raise ComputationError(
                f"Target {self.config.name!r} position update failed: no active scene object exists"
            )
        try:
            self.target_object.position = point3f(new_position)
        except Exception as exc:  # noqa: BLE001 - preserve external setter context
            raise ComputationError(
                f"Target {self.config.name!r} position update failed for {new_position}: {exc}"
            ) from exc

    def apply_orientation_snapshot(self, orient_tuple: Orientation3) -> None:
        """Apply a prepared target orientation snapshot to the live scene object.

        Converts degrees to radians for Sionna RT. A missing object or rejected
        setter is a generation error because the target is required.
        """
        if self.target_object is None:
            raise ComputationError(
                f"Target {self.config.name!r} orientation update failed: "
                "no active scene object exists"
            )
        try:
            orient, engine_orientation = orientation_to_point3f_with_engine_radians(orient_tuple)
            self.target_object.orientation = orient
            logger.debug(
                "Updated orientation from tuple to: %s (converted to radians: %.4f, %.4f, %.4f)",
                orient_tuple,
                engine_orientation[0],
                engine_orientation[1],
                engine_orientation[2],
            )
        except Exception as exc:  # noqa: BLE001 - preserve external setter context
            raise ComputationError(
                f"Target {self.config.name!r} orientation update failed for {orient_tuple}: {exc}"
            ) from exc

    def apply_scale_snapshot(self, scale_value: float) -> None:
        """Apply a runtime scale without changing the scenario configuration."""
        try:
            scale_value = float(scale_value)
        except (TypeError, ValueError):
            logger.warning("Invalid scale value for target %s: %s", self.config.name, scale_value)
            return
        if scale_value <= 0:
            logger.warning(
                "Scale must be positive for target %s (got %s)", self.config.name, scale_value
            )
            return
        if self.target_object is not None:
            try:
                self.target_object.scaling = mi.Vector3f(scale_value, scale_value, scale_value)
                logger.debug("Updated scale for target %s to %.3f", self.config.name, scale_value)
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning(
                    "Could not apply scale %.3f to target %s: %s",
                    scale_value,
                    self.config.name,
                    exc,
                )

    def get_current_position(self) -> tuple[float, float, float]:
        """Get current target position"""
        if self.target_object is not None:
            try:
                pos = cast(Any, self.target_object.position)
                return (float(pos.x), float(pos.y), float(pos.z))
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Error getting position: {e}")
                return self.config.initial_position
        return self.config.initial_position

    def update_mesh_for_frame(
        self,
        frame_idx: int,
        *,
        expected_call_count: int | None = None,
    ) -> SceneObject:
        """Switch to the appropriate mesh for the given frame index.

        Advances through the mesh sequence, recreating the SceneObject when
        the mesh index changes while preserving position and scaling. The
        sequence either loops or holds its final mesh according to
        ``mesh_end_behavior``. Orientation is set later by the pipeline to
        avoid double application.

        ``expected_call_count`` lets the propagation layer drive mesh playback
        from a wall-clock cadence rather than from every simulation frame. That
        keeps animated geometry fixed across multi-step acquisition windows
        while preserving the intended mesh start/stride sequence.
        """
        if self.target_object is None:
            raise ComputationError(
                f"Target {self.config.name!r} mesh update failed at frame {frame_idx}: "
                "no active scene object exists"
            )

        if not self.meshes or len(self.meshes) <= 1:
            logger.debug(
                f"[TARGET] {self.config.name}: No mesh switching - {len(self.meshes)} meshes available"
            )
            return self.target_object

        call_count = (
            max(0, int(expected_call_count))
            if expected_call_count is not None
            else self._mesh_call_count
        )
        mesh_index = self._mesh_index_for_update_call(call_count)
        next_call_count = call_count + 1
        mesh_file_name = (
            os.path.basename(self.meshes[mesh_index])
            if mesh_index < len(self.meshes)
            else "unknown"
        )
        logger.info(
            "[TARGET] [MESH_UPDATE] %s: frame_idx=%d, mesh_call=%d, "
            "mesh_index=%d/%d, mesh_file=%s",
            self.config.name,
            frame_idx,
            call_count,
            mesh_index,
            len(self.meshes),
            mesh_file_name,
        )

        if mesh_index == self.current_mesh_idx:
            self._mesh_call_count = next_call_count
            logger.debug(
                f"[TARGET] {self.config.name}: No mesh switch needed - already at mesh index {self.current_mesh_idx}"
            )
            return self.target_object

        mesh_path = self.meshes[mesh_index]
        previous_target = self.target_object
        candidate = None
        scene_edit_attempted = False
        try:
            if self.target_material is None:
                self.create_material()

            scene_mesh_path = _prepare_mesh_path_for_mitsuba(mesh_path)
            candidate = SceneObject(
                fname=scene_mesh_path,
                name=self.config.name,
                radio_material=self.target_material,
            )

            previous_position = previous_target.position
            previous_scaling = previous_target.scaling
            previous_velocity = getattr(previous_target, "velocity", None)

            scene_edit_attempted = True
            self.scene.edit(add=[candidate], remove=[previous_target])

            if candidate.radio_material is not None:
                radio_material = cast(Any, candidate.radio_material)
                if hasattr(radio_material, "frequency_update"):
                    try:
                        radio_material.frequency_update()
                    except (RuntimeError, ValueError) as exc:
                        logger.warning(
                            "Target %s material frequency update was unavailable: %s",
                            self.config.name,
                            exc,
                        )

            if self.config.use_ply_position:
                candidate.scaling = previous_scaling
            else:
                candidate.position = previous_position
                candidate.scaling = previous_scaling
                if previous_velocity is not None and hasattr(candidate, "velocity"):
                    candidate.velocity = previous_velocity

            self.target_object = candidate
            self.current_mesh_idx = mesh_index
            self._mesh_call_count = next_call_count
            logger.info(
                "Target %s switched to mesh %s at frame %d",
                self.config.name,
                os.path.basename(mesh_path),
                frame_idx,
            )
            return candidate
        except Exception as exc:  # noqa: BLE001 - the external scene edit must roll back
            rollback_error = None
            if candidate is not None and scene_edit_attempted:
                try:
                    active = None
                    getter = getattr(self.scene, "get", None)
                    if callable(getter):
                        active = getter(self.config.name)
                    if active is candidate:
                        self.scene.edit(add=[previous_target], remove=[candidate])
                    elif active is None:
                        # A failed external edit can remove the old object
                        # before registering its replacement.
                        if callable(getter):
                            self.scene.edit(add=[previous_target])
                        else:
                            self.scene.edit(add=[previous_target], remove=[candidate])
                    elif active is not None and active is not previous_target:
                        self.scene.edit(add=[previous_target], remove=[active])
                except Exception as restore_exc:  # noqa: BLE001
                    rollback_error = restore_exc

            message = (
                f"Target {self.config.name!r} mesh update failed at frame {frame_idx} "
                f"for {mesh_path}: {exc}"
            )
            if rollback_error is not None:
                message += f"; rollback also failed: {rollback_error}"
            raise ComputationError(message) from exc
