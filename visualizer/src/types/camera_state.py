"""Renderer-agnostic camera state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraState:
    """Portable orbit camera state shared by controllers and renderers."""

    eye: tuple[float, float, float]
    lookat: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_deg: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "eye": list(self.eye),
            "lookat": list(self.lookat),
            "up": list(self.up),
            "fov": float(self.fov_deg),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CameraState":
        """Parse the current renderer-neutral serialized form.

        Renderer APIs exchange ``CameraState`` directly. This parser exists only
        at JSON boundaries such as sessions and deterministic debug captures.
        """
        if not isinstance(value, dict):
            raise TypeError("camera state must be a dictionary")

        def _vector(key: str) -> tuple[float, float, float]:
            raw = value[key]
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                raise ValueError(f"camera field '{key}' must contain three values")
            return (float(raw[0]), float(raw[1]), float(raw[2]))

        return cls(
            eye=_vector("eye"),
            lookat=_vector("lookat"),
            up=_vector("up"),
            fov_deg=float(value["fov"]),
        )
