"""Lazy Sionna RT 2.2 native sensing-target construction and pose updates."""

from __future__ import annotations

import importlib
import math
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

from shared.scenarios.actors import FixedOrientationSpec, NativeBoxGeometrySpec

from ..exceptions import ComputationError
from .representation import NativeTargetConfig

if TYPE_CHECKING:
    from .config import TargetConfig
    from .manager import TargetManager


class NativeSensingCapabilityError(ComputationError):
    """The installed engine does not expose the requested native sensing API."""


def _native_engine() -> tuple[Any, Any]:
    try:
        installed_version = version("sionna-rt")
        if Version(installed_version) < Version("2.2.0"):
            raise RuntimeError(f"installed sionna-rt {installed_version} is older than 2.2.0")
        mi = importlib.import_module("mitsuba")
        rcs = importlib.import_module("sionna.rt.rcs")
        for name in ("ConstantRCSSensingTarget", "TR38901SensingTarget", "RCSSolver"):
            if not callable(getattr(rcs, name, None)):
                raise ImportError(f"sionna.rt.rcs.{name} is unavailable")
        if "mono_polarized" not in str(mi.variant()):
            raise RuntimeError(
                "native RCS solving requires a mono_polarized Mitsuba variant "
                "(for CPU, use llvm_ad_mono_polarized)"
            )
        return mi, rcs
    except (ImportError, AttributeError, RuntimeError, OSError, InvalidVersion) as exc:
        raise NativeSensingCapabilityError(
            "Native constant_rcs/tr38901 targets require Sionna RT 2.2 native sensing "
            "support and an initialized Mitsuba variant. Install orchav[native-rcs] "
            f"in a compatible environment: {exc}"
        ) from exc


def create_target_manager(
    target_config: TargetConfig | NativeTargetConfig,
    scene: Any,
    material_overrides: Any = None,
) -> TargetManager | NativeTargetManager:
    """Select exactly one representation without altering the mesh-material lane."""
    if isinstance(target_config, NativeTargetConfig):
        return NativeTargetManager(target_config, scene)
    if getattr(target_config, "representation", "mesh_material") != "mesh_material":
        raise ValueError("native target representations require NativeTargetConfig")
    from .manager import TargetManager

    return TargetManager(target_config, scene, material_overrides=material_overrides)


def _vec3(value: Any, *, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a finite numeric three-vector")
    result = tuple(float(component) for component in value)
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} must be a finite numeric three-vector")
    return result


def _point_rows(points: Any) -> tuple[tuple[float, float, float], ...]:
    """Detach a native (3, N) point array into immutable JSON-compatible rows."""
    axes = []
    for axis in ("x", "y", "z"):
        values = getattr(points, axis)
        try:
            axes.append(tuple(float(component) for component in values))
        except TypeError:
            axes.append((float(values),))
    if len({len(axis) for axis in axes}) != 1:
        raise ValueError("native point array has inconsistent coordinate lengths")
    return tuple(zip(*axes))


def _single_point(point: Any) -> tuple[float, float, float]:
    rows = _point_rows(point)
    if len(rows) != 1:
        raise ValueError("expected one native pose vector")
    return rows[0]


def _scalar(value: Any) -> float:
    """Detach one ordinary or native scalar value."""
    try:
        return float(value)
    except TypeError:
        return float(value[0])


class NativeTargetManager:
    """Manage one native scattering model and its sole absorbing scene proxy."""

    def __init__(self, target_config: NativeTargetConfig, scene: Any) -> None:
        self.config = target_config
        self.scene = scene
        self.target_object: Any | None = None
        self.current_mesh_idx = 0

    def _active_target(self) -> Any:
        if self.target_object is None:
            raise ComputationError(f"Native target {self.config.name!r} has no active scene object")
        return self.target_object

    def create_target(self) -> Any:
        """Create and register one fully posed native target, without a second mesh."""
        if self.target_object is not None:
            raise ComputationError(f"Native target {self.config.name!r} is already created")
        mi, rcs = _native_engine()
        geometry = self.config.geometry
        kwargs: dict[str, Any] = {
            "name": self.config.name,
            "position": mi.Point3f(self.config.initial_position),
            "velocity": mi.Vector3f(self.config.initial_velocity),
        }
        if isinstance(geometry, NativeBoxGeometrySpec):
            for dimension in ("length", "width", "height"):
                value = getattr(geometry, f"{dimension}_m")
                if value is not None:
                    kwargs[dimension] = value
        else:
            mesh_path = self.config.resolved_mesh_path
            if mesh_path is None or not mesh_path.is_file():
                raise ComputationError(
                    f"Native target {self.config.name!r} static proxy is missing: {mesh_path}"
                )
            kwargs["fname"] = str(mesh_path)
        if self.config.constant_rcs is not None:
            kwargs.update(
                sigma=self.config.constant_rcs.sigma_m2, xpr_db=self.config.constant_rcs.xpr_db
            )
            constructor = rcs.ConstantRCSSensingTarget
        else:
            assert self.config.tr38901 is not None
            kwargs.update(self.config.tr38901.model_dump())
            constructor = rcs.TR38901SensingTarget

        candidate = None
        add_attempted = False
        try:
            getter = getattr(self.scene, "get", None)
            if callable(getter) and getter(self.config.name) is not None:
                raise ValueError(f"scene already contains {self.config.name!r}")
            orientation = self.config.orientation
            angles = None
            if hasattr(orientation, "orientations"):
                samples = orientation.orientations()
                angles = next(iter(samples), None)
            elif isinstance(orientation, FixedOrientationSpec):
                angles = (orientation.yaw_deg, orientation.pitch_deg, orientation.roll_deg)
            if angles is not None:
                from ..sionna_integration import orientation_to_point3f_with_engine_radians

                kwargs["orientation"], _ = orientation_to_point3f_with_engine_radians(angles)
            candidate = constructor(**kwargs)
            candidate.scaling = mi.Vector3f(self.config.scale)
            # Native mesh orientation/scaling can move an asymmetric proxy's
            # AABB center. Actor positions own the target center, so restore
            # that center after all initial shape transforms are applied.
            candidate.position = mi.Point3f(self.config.initial_position)
            add_attempted = True
            self.scene.edit(add=[candidate])
            self.target_object = candidate
            return candidate
        except Exception as exc:  # noqa: BLE001 - external scene edit needs cleanup
            cleanup_error = None
            if candidate is not None and add_attempted:
                try:
                    getter = getattr(self.scene, "get", None)
                    if not callable(getter) or getter(self.config.name) is candidate:
                        self.scene.edit(remove=[candidate])
                except Exception as cleanup_exc:  # noqa: BLE001
                    cleanup_error = cleanup_exc
            message = f"Native target {self.config.name!r} creation failed: {exc}"
            if cleanup_error is not None:
                message += f"; candidate cleanup also failed: {cleanup_error}"
            raise ComputationError(message) from exc

    def apply_position_snapshot(self, position: tuple[float, float, float]) -> None:
        """Set the center in world coordinates, preserving native point offsets."""
        mi, _ = _native_engine()
        self._active_target().position = mi.Point3f(_vec3(position, label="position"))

    def apply_orientation_snapshot(self, angles: tuple[float, float, float]) -> None:
        """Convert canonical degree angles to Sionna's engine convention."""
        from ..sionna_integration import orientation_to_point3f_with_engine_radians

        orientation, _ = orientation_to_point3f_with_engine_radians(
            _vec3(angles, label="orientation")
        )
        target = self._active_target()
        center = _single_point(target.position)
        target.orientation = orientation
        mi, _ = _native_engine()
        target.position = mi.Point3f(center)

    def apply_velocity_snapshot(self, velocity: tuple[float, float, float]) -> None:
        """Set rigid-body velocity explicitly for native target Doppler."""
        mi, _ = _native_engine()
        self._active_target().velocity = mi.Vector3f(_vec3(velocity, label="velocity"))

    def apply_scale_snapshot(self, scale_value: float) -> None:
        """Scale the sole proxy and its native scattering points together."""
        scale_value = float(scale_value)
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError("native target scale must be finite and positive")
        mi, _ = _native_engine()
        target = self._active_target()
        center = _single_point(target.position)
        target.scaling = mi.Vector3f(scale_value)
        target.position = mi.Point3f(center)

    def get_current_position(self) -> tuple[float, float, float]:
        """Return the actual current center or the configured initial center."""
        if self.target_object is None:
            return self.config.initial_position
        return _single_point(self.target_object.position)

    def update_mesh_for_frame(
        self, frame_idx: int, *, expected_call_count: int | None = None
    ) -> Any:
        """Keep native proxies rigid; only the shared actor pose changes per frame."""
        del frame_idx, expected_call_count
        return self._active_target()

    def metadata_snapshot(self) -> dict[str, Any]:
        """Detach actual model, geometry, pose and native point positions for one frame."""
        target = self._active_target()
        mi, _ = _native_engine()
        spst = target.scattering_model.spst
        local_unscaled = _point_rows(spst.lcs_positions)
        target_indices = mi.UInt([0] * len(local_unscaled))
        local = _point_rows(spst.scaled_lcs_positions(target.scaling, target_indices))
        world = _point_rows(
            spst.gcs_positions(target.position, target.orientation, target_indices, target.scaling)
        )
        geometry: dict[str, Any] = {
            "type": self.config.geometry.type,
            **{f"{key}_m": float(value) for key, value in target.dimensions.items()},
            "scaling": _single_point(target.scaling),
        }
        if self.config.geometry.type == "mesh":
            geometry["path"] = self.config.geometry.path
        try:
            upstream_version = version("sionna-rt")
        except PackageNotFoundError:
            upstream_version = "unknown"
        snapshot: dict[str, Any] = {
            "name": self.config.name,
            "representation": self.config.representation,
            "geometry": geometry,
            "scattering_points_local_m": local,
            "scattering_points_world_m": world,
            "scattering_point_ids": tuple(range(len(local))),
            "current_position": _single_point(target.position),
            "orientation": _single_point(target.orientation),
            "velocity_m_s": _single_point(target.velocity),
            "upstream_version": upstream_version,
            "metadata_source": "native_runtime",
        }
        model = self.config.constant_rcs or self.config.tr38901
        assert model is not None
        model_metadata = model.model_dump()
        if self.config.constant_rcs is not None:
            model_metadata.update(
                sigma_m2=_scalar(target.sigma),
                xpr_db=None if target.xpr_db is None else _scalar(target.xpr_db),
            )
        else:
            for flag in ("random_sigma_s", "random_xpr", "random_phases"):
                model_metadata[flag] = bool(getattr(target.scattering_model, flag))
        snapshot[self.config.representation] = model_metadata
        return snapshot


__all__ = ["NativeSensingCapabilityError", "NativeTargetManager", "create_target_manager"]
