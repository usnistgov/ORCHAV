"""Shared type models for renderer-neutral visualizer contracts.

The package exports stable camera and render-payload dataclasses used by
services, scene builders, renderers, and tests.
"""

from .camera_state import CameraState
from .render_payloads import (
    GeometryPayload,
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    OrientationFramePayload,
    PointCloudPayload,
    RenderPayload,
    TextLabelPayload,
)

__all__ = [
    "CameraState",
    "GeometryPayload",
    "MeshPayload",
    "LineSetPayload",
    "PointCloudPayload",
    "RenderPayload",
    "TextLabelPayload",
    "OrientationFramePayload",
    "MaterialPayload",
]
