"""Small-amplitude periodic mobility models.

These patterns model breathing, vibration, and pendulum-like motion for devices
or targets whose main pose is otherwise static. They are useful for sensing
micro-motion studies where centimeter-scale displacement matters.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..utils import point_to_tuple
from .base import MobilityPattern, Position3, _numeric_triple


class OscillatingMobility(MobilityPattern):
    """Periodic oscillation around a center point.

    Supported patterns:

    - ``sinusoidal``: simple harmonic motion ``A * sin(2*pi*f*t)``
    - ``breathing``: sinusoidal with a slow amplitude envelope
    - ``vibrating``: higher-frequency small-amplitude oscillation
    """

    yaml_type = "oscillating"

    def __init__(
        self,
        center: Any = (0, 0, 0),
        amplitude: float = 0.05,
        frequency_hz: float = 0.3,
        axis: tuple[float, float, float] = (1, 0, 0),
        pattern: str = "sinusoidal",
    ):
        super().__init__()
        self.center = center
        self.amplitude = float(amplitude)
        self.frequency_hz = float(frequency_hz)
        norm = math.sqrt(sum(v * v for v in axis))
        self.axis = tuple(v / norm for v in axis) if norm > 0 else (1, 0, 0)
        self.pattern = pattern
        self.start_pos = center

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "OscillatingMobility":
        """Build oscillating mobility from a YAML mobility block."""
        center = position or point_to_tuple(mobility_cfg.get("center", [0, 0, 0]))
        return cls(
            center=center,
            amplitude=float(mobility_cfg.get("amplitude", 0.05)),
            frequency_hz=float(mobility_cfg.get("frequency_hz", 0.3)),
            axis=_numeric_triple(mobility_cfg.get("axis", [1, 0, 0]), "axis"),
            pattern=str(mobility_cfg.get("pattern", "sinusoidal")),
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Return the peak tangential speed implied by the oscillation."""
        return float(self.amplitude * self.frequency_hz * 2 * math.pi)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate periodic offsets around the configured center point."""
        cx, cy, cz = point_to_tuple(self.center)

        ax, ay, az = self.axis
        t = np.linspace(0, scene_duration, scene_steps)
        f = self.frequency_hz
        A = self.amplitude

        if self.pattern == "breathing":
            envelope = 0.5 * (1 + np.cos(2 * np.pi * f * 0.1 * t))
            displacement = A * envelope * np.sin(2 * np.pi * f * t)
        elif self.pattern == "vibrating":
            displacement = A * np.sin(2 * np.pi * f * 5 * t)
        else:
            displacement = A * np.sin(2 * np.pi * f * t)

        return [
            (
                cx + float(d) * ax,
                cy + float(d) * ay,
                cz + float(d) * az,
            )
            for d in displacement
        ]


class PendulumMobility(MobilityPattern):
    """Pendulum swing in a plane.

    The bob traces an arc of length ``length`` from a ``pivot`` point,
    swinging in the specified ``plane`` (xy, xz, or yz).
    """

    yaml_type = "pendulum"

    def __init__(
        self,
        pivot: Any = (0, 0, 1.5),
        length: float = 0.6,
        max_angle_deg: float = 30.0,
        plane: str = "xz",
        frequency_hz: float = 1.0,
    ):
        super().__init__()
        self.pivot = pivot
        self.length = float(length)
        self.max_angle_deg = float(max_angle_deg)
        self.plane = plane.lower()
        self.frequency_hz = float(frequency_hz)
        self.start_pos = pivot

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "PendulumMobility":
        """Build pendulum mobility from a YAML mobility block."""
        pivot = position or point_to_tuple(mobility_cfg.get("pivot", [0, 0, 1.5]))
        return cls(
            pivot=pivot,
            length=float(mobility_cfg.get("length", 0.6)),
            max_angle_deg=float(mobility_cfg.get("max_angle_deg", 30)),
            plane=str(mobility_cfg.get("plane", "xz")),
            frequency_hz=float(mobility_cfg.get("frequency_hz", 1.0)),
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Return the peak tangential speed implied by the pendulum swing."""
        return float(
            self.length * math.radians(self.max_angle_deg) * self.frequency_hz * 2 * math.pi
        )

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate pendulum bob positions over the scene timeline."""
        px, py, pz = point_to_tuple(self.pivot)

        t = np.linspace(0, scene_duration, scene_steps)
        theta = np.radians(self.max_angle_deg) * np.sin(2 * np.pi * self.frequency_hz * t)
        L = self.length

        positions = []
        for angle in theta:
            if self.plane == "xz":
                x = px + L * math.sin(angle)
                y = py
                z = pz - L * math.cos(angle)
            elif self.plane == "yz":
                x = px
                y = py + L * math.sin(angle)
                z = pz - L * math.cos(angle)
            else:  # xy
                x = px + L * math.sin(angle)
                y = py + L * math.cos(angle)
                z = pz
            positions.append((float(x), float(y), float(z)))

        return positions
