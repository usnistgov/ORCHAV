"""Renderer-neutral quaternion math for actor orientations.

Quaternions are stored explicitly as ``w, x, y, z`` and represent active
right-handed rotations. Euler helpers use yaw/pitch/roll in degrees with the
conventional intrinsic Z/Y/X order and +X as the actor's forward axis. With
+Z upward, positive pitch rotates the forward axis toward -Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._adapters import vec3

_NORM_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Quaternion:
    """A normalized renderer-neutral quaternion in named ``w, x, y, z`` form."""

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        components = (float(self.w), float(self.x), float(self.y), float(self.z))
        if not all(math.isfinite(component) for component in components):
            raise ValueError("quaternion components must be finite")
        norm = math.sqrt(sum(component * component for component in components))
        if norm <= _NORM_EPSILON:
            raise ValueError("quaternion norm must be non-zero")
        object.__setattr__(self, "w", components[0] / norm)
        object.__setattr__(self, "x", components[1] / norm)
        object.__setattr__(self, "y", components[2] / norm)
        object.__setattr__(self, "z", components[3] / norm)

    @classmethod
    def identity(cls) -> "Quaternion":
        """Return the no-rotation quaternion."""

        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_euler_deg(
        cls,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> "Quaternion":
        """Build an intrinsic Z/Y/X yaw-pitch-roll rotation."""

        yaw = math.radians(float(yaw_deg)) * 0.5
        pitch = math.radians(float(pitch_deg)) * 0.5
        roll = math.radians(float(roll_deg)) * 0.5
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        return cls(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    @classmethod
    def from_axis_angle_deg(
        cls,
        axis: object,
        angle_deg: float,
    ) -> "Quaternion":
        """Build a quaternion rotating ``angle_deg`` about ``axis``."""

        ax, ay, az = vec3(axis, name="axis")
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm <= _NORM_EPSILON:
            raise ValueError("rotation axis must be non-zero")
        half_angle = math.radians(float(angle_deg)) * 0.5
        scale = math.sin(half_angle) / norm
        return cls(math.cos(half_angle), ax * scale, ay * scale, az * scale)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Compose this rotation with ``other`` in the actor-local frame."""

        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def dot(self, other: "Quaternion") -> float:
        """Return the four-dimensional dot product."""

        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def negated(self) -> "Quaternion":
        """Return the equivalent antipodal quaternion."""

        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def slerp(self, other: "Quaternion", fraction: float) -> "Quaternion":
        """Shortest-path spherical interpolation to ``other``."""

        t = min(max(float(fraction), 0.0), 1.0)
        end = other
        dot = self.dot(end)
        if dot < 0.0:
            end = end.negated()
            dot = -dot
        dot = min(max(dot, -1.0), 1.0)
        if dot > 0.9995:
            return Quaternion(
                self.w + t * (end.w - self.w),
                self.x + t * (end.x - self.x),
                self.y + t * (end.y - self.y),
                self.z + t * (end.z - self.z),
            )
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        first = math.sin((1.0 - t) * theta_0) / sin_theta_0
        second = math.sin(t * theta_0) / sin_theta_0
        return Quaternion(
            first * self.w + second * end.w,
            first * self.x + second * end.x,
            first * self.y + second * end.y,
            first * self.z + second * end.z,
        )

    def to_euler_deg(self) -> tuple[float, float, float]:
        """Return intrinsic Z/Y/X yaw, pitch, and roll in degrees."""

        sin_pitch = 2.0 * (self.w * self.y - self.z * self.x)
        if abs(sin_pitch) >= 1.0 - 1e-12:
            # Euler angles are not unique at gimbal lock. Keep a stable
            # canonical representation with zero roll and preserve heading.
            pitch = math.copysign(math.pi / 2.0, sin_pitch)
            yaw = 2.0 * math.atan2(self.z, self.w)
            roll = 0.0
        else:
            pitch = math.asin(sin_pitch)
            sin_roll_cos_pitch = 2.0 * (self.w * self.x + self.y * self.z)
            cos_roll_cos_pitch = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
            roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
            sin_yaw_cos_pitch = 2.0 * (self.w * self.z + self.x * self.y)
            cos_yaw_cos_pitch = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
            yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
        return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))
