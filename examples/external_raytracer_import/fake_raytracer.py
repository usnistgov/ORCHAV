"""Deterministic records produced by a stand-alone example ray tracer.

The record model belongs to the external producer. It uses ordinary Python
values and source-native units, with no knowledge of ORCHAV frame classes or
storage. A real integration can replace :func:`trace` with its own decoder or
solver API while leaving the ORCHAV-side adapter separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DeviceRecord:
    """One named radio device in the external coordinate system."""

    name: str
    position_m: Point3
    orientation_deg: Point3


@dataclass(frozen=True, slots=True)
class BounceRecord:
    """One physical interaction point reported for a non-LoS path."""

    position_m: Point3
    mechanism: str
    material_name: str
    material_family: str


@dataclass(frozen=True, slots=True)
class PathRecord:
    """One external path with source-native metric units."""

    transmitter: str
    receiver: str
    bounces: tuple[BounceRecord, ...]
    delay_s: float
    path_loss_db: float
    arrival_azimuth_rad: float
    arrival_elevation_rad: float
    departure_azimuth_rad: float
    departure_elevation_rad: float


@dataclass(frozen=True, slots=True)
class TargetRecord:
    """Dynamic target state accompanying one external frame."""

    name: str
    position_m: Point3
    velocity_m_s: Point3
    category: str


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One decoded external time step."""

    frame_index: int
    timestamp_s: float
    transmitters: tuple[DeviceRecord, ...]
    receivers: tuple[DeviceRecord, ...]
    paths: tuple[PathRecord, ...]
    targets: tuple[TargetRecord, ...]


def _devices() -> tuple[tuple[DeviceRecord, ...], tuple[DeviceRecord, ...]]:
    """Return the stable radio-device catalog used by both frames."""

    transmitters = (
        DeviceRecord("tx_west", (0.0, 0.0, 1.5), (0.0, 0.0, 0.0)),
        DeviceRecord("tx_east", (0.0, 8.0, 1.5), (180.0, 0.0, 0.0)),
    )
    receivers = (
        DeviceRecord("rx_west", (12.0, 0.0, 1.5), (180.0, 0.0, 0.0)),
        DeviceRecord("rx_east", (12.0, 8.0, 1.5), (0.0, 0.0, 0.0)),
    )
    return transmitters, receivers


def _paths(frame_index: int) -> tuple[PathRecord, ...]:
    """Return sparse-pair LoS and reflected paths for one time step."""

    metric_shift = 0.5 * frame_index
    angle_shift = math.radians(1.0 * frame_index)
    return (
        PathRecord(
            transmitter="tx_west",
            receiver="rx_east",
            bounces=(),
            delay_s=(45.0 + metric_shift) * 1e-9,
            path_loss_db=61.0 + metric_shift,
            arrival_azimuth_rad=math.radians(180.0) + angle_shift,
            arrival_elevation_rad=math.radians(0.0),
            departure_azimuth_rad=math.radians(34.0) + angle_shift,
            departure_elevation_rad=math.radians(0.0),
        ),
        PathRecord(
            transmitter="tx_west",
            receiver="rx_east",
            bounces=(
                BounceRecord(
                    position_m=(6.0, -2.0, 2.0),
                    mechanism="surface_reflection",
                    material_name="painted_concrete",
                    material_family="concrete",
                ),
            ),
            delay_s=(52.0 + metric_shift) * 1e-9,
            path_loss_db=69.0 + metric_shift,
            arrival_azimuth_rad=math.radians(152.0) + angle_shift,
            arrival_elevation_rad=math.radians(3.0),
            departure_azimuth_rad=math.radians(-18.0) + angle_shift,
            departure_elevation_rad=math.radians(2.0),
        ),
        PathRecord(
            transmitter="tx_east",
            receiver="rx_west",
            bounces=(
                BounceRecord(
                    position_m=(6.0, 10.0, 2.5),
                    mechanism="surface_reflection",
                    material_name="window_glass",
                    material_family="glass",
                ),
            ),
            delay_s=(48.0 + metric_shift) * 1e-9,
            path_loss_db=66.0 + metric_shift,
            arrival_azimuth_rad=math.radians(-150.0) - angle_shift,
            arrival_elevation_rad=math.radians(4.0),
            departure_azimuth_rad=math.radians(162.0) - angle_shift,
            departure_elevation_rad=math.radians(3.0),
        ),
    )


def trace() -> tuple[FrameRecord, ...]:
    """Return two deterministic decoded frames with a moving target."""

    transmitters, receivers = _devices()
    return (
        FrameRecord(
            frame_index=0,
            timestamp_s=0.0,
            transmitters=transmitters,
            receivers=receivers,
            paths=_paths(0),
            targets=(
                TargetRecord(
                    name="delivery_cart",
                    position_m=(6.0, 4.0, 0.75),
                    velocity_m_s=(1.0, 0.0, 0.0),
                    category="vehicle",
                ),
            ),
        ),
        FrameRecord(
            frame_index=1,
            timestamp_s=0.1,
            transmitters=transmitters,
            receivers=receivers,
            paths=_paths(1),
            targets=(
                TargetRecord(
                    name="delivery_cart",
                    position_m=(6.1, 4.0, 0.75),
                    velocity_m_s=(1.0, 0.0, 0.0),
                    category="vehicle",
                ),
            ),
        ),
    )


__all__ = [
    "BounceRecord",
    "DeviceRecord",
    "FrameRecord",
    "PathRecord",
    "Point3",
    "TargetRecord",
    "trace",
]
