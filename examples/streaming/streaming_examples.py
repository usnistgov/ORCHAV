#!/usr/bin/env python3
"""Low-level motion and orientation extensions for live gRPC workflows.

These examples implement the generator runtime interfaces directly for data
that arrives from a sensor, API, or simulation bridge. Scenario files and
scripted actor sets use the immutable specifications in ``shared.scenarios``.
"""

import math
import time
from typing import Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from generator.core.mobility.base import MobilityPattern
from generator.core.orientation import smoothing_time_from_step_fraction
from generator.core.scenario_actors import (
    Timeline,
    prepare_orientation,
    prepare_sampled_mobility,
)
from shared.logging import get_logger
from shared.scenarios.actors import (
    FixedOrientationSpec,
    LookAtOrientationSpec,
    SpinOrientationSpec,
)

logger = get_logger(__name__)

Orientation3 = tuple[float, float, float]


def _orientation_tuple(value, name: str = "orientation") -> Orientation3:
    """Normalize example streaming orientations to yaw/pitch/roll degree tuples."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly three values, got {arr.size}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


class StreamingOrientationSource:
    """Example prepared source for orientations computed one frame at a time.

    The class structurally implements ``PreparedOrientationSource`` without
    becoming part of the generator's stable API. Actor-state preparation passes
    the owner's sampled positions in ``context``; subclasses may consume those
    samples while still exposing the small ``prepare``/``orientations`` contract.
    """

    def __init__(self) -> None:
        self._scene_steps: int | None = None
        self._scene_duration: float | None = None
        self._self_positions: tuple[tuple[float, float, float], ...] = ()

    def get_orientation(
        self,
        step: int,
        total_steps: int,
        current_pos=None,
        previous_pos=None,
        scene_duration: float | None = None,
    ) -> Orientation3:
        """Return the orientation for one streaming step."""
        raise NotImplementedError("Example subclasses must implement get_orientation")

    def prepare(
        self,
        steps: int,
        duration: float | None = None,
        context: dict | None = None,
    ) -> None:
        """Store timeline parameters without precomputing the full orientation list."""
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"{self.__class__.__name__} requires at least one step")
        self._scene_steps = steps
        self._scene_duration = float(duration) if duration is not None else None
        positions = () if context is None else tuple(context.get("self_positions", ()))
        if positions and len(positions) != steps:
            raise ValueError(
                f"{self.__class__.__name__} received {len(positions)} owner positions "
                f"for {steps} steps"
            )
        self._self_positions = tuple(
            tuple(float(component) for component in position) for position in positions
        )

    def orientations(self) -> "StreamingOrientationIterator":
        """Return an iterator over streaming orientations."""
        if self._scene_steps is None:
            raise RuntimeError(f"{self.__class__.__name__} must be prepared before iteration")
        return StreamingOrientationIterator(self)


class StreamingOrientationIterator:
    """Iterator facade for local streaming orientation examples."""

    def __init__(self, orientation_source: StreamingOrientationSource) -> None:
        if orientation_source._scene_steps is None:
            raise RuntimeError("Streaming orientation must be prepared before iteration")
        self.orientation_source = orientation_source
        self.current_step = 0
        self.max_steps = orientation_source._scene_steps

    def __iter__(self) -> "StreamingOrientationIterator":
        return self

    def __next__(self) -> Orientation3:
        if self.current_step >= self.max_steps:
            raise StopIteration
        positions = self.orientation_source._self_positions
        current_pos = positions[self.current_step] if positions else None
        previous_pos = positions[max(0, self.current_step - 1)] if positions else None
        orientation = self.orientation_source.get_orientation(
            self.current_step,
            self.max_steps,
            current_pos=current_pos,
            previous_pos=previous_pos,
            scene_duration=self.orientation_source._scene_duration,
        )
        self.current_step += 1
        return _orientation_tuple(
            orientation,
            f"{self.orientation_source.__class__.__name__} step {self.current_step - 1}",
        )

    def __len__(self) -> int:
        return self.max_steps


class StepwiseMobility(MobilityPattern):
    """Example helper for sources that naturally expose one position per step."""

    def get_position(self, step: int) -> Tuple[float, float, float]:
        """Return the position for one step."""
        raise NotImplementedError("Example subclasses must implement get_position()")

    def get_positions(
        self,
        start_pos: Tuple[float, float, float],
        scene_steps: int,
        scene_duration: float,
    ) -> list[Tuple[float, float, float]]:
        """Build the full trajectory expected by the generator mobility contract."""
        del start_pos, scene_duration
        return [self.get_position(step) for step in range(scene_steps)]


class LiveSensorMobility(StepwiseMobility):
    """Stepwise mobility pattern that fetches positions from a live sensor"""

    def __init__(self, sensor_url: str, update_interval: float = 0.1):
        """
        Initialize live sensor mobility.

        Args:
            sensor_url: URL or identifier for the sensor data source
            update_interval: Minimum time between sensor updates (seconds)
        """
        super().__init__()
        self.sensor_url = sensor_url
        self.update_interval = update_interval
        self.last_update = 0
        self.cached_position = None
        self.position_history = []

    def get_position(self, step: int) -> Tuple[float, float, float]:
        """Get position from live sensor (with caching for performance)"""
        current_time = time.time()

        # Update position if enough time has passed
        if (current_time - self.last_update) >= self.update_interval:
            try:
                # In a real implementation, this would fetch from the sensor
                # For now, we'll simulate with a simple pattern
                pos = self._fetch_from_sensor()
                self.cached_position = pos
                self.position_history.append((current_time, pos))
                self.last_update = current_time

                # Keep only recent history (last 100 positions)
                if len(self.position_history) > 100:
                    self.position_history = self.position_history[-100:]

            except (OSError, ValueError, TypeError) as e:
                logger.warning(f"Failed to fetch sensor data: {e}")
                # Use last known position or default
                if self.cached_position is None:
                    self.cached_position = (0.0, 0.0, 0.0)

        return self.cached_position

    def _fetch_from_sensor(self) -> Tuple[float, float, float]:
        """Fetch position from sensor (implement with actual sensor interface)"""
        # This is a placeholder - implement actual sensor communication
        # For demo purposes, simulate a moving target
        t = time.time()
        x = 5.0 * math.sin(t * 0.5)
        y = 3.0 * math.cos(t * 0.3)
        z = 2.0 + 0.5 * math.sin(t * 0.2)
        return (x, y, z)


class APIMobility(StepwiseMobility):
    """Stepwise mobility pattern that fetches positions from an external API"""

    def __init__(
        self, api_endpoint: str, api_key: Optional[str] = None, request_timeout: float = 5.0
    ):
        """
        Initialize API mobility.

        Args:
            api_endpoint: API endpoint URL for position data
            api_key: Optional API key for authentication
            request_timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.last_position = (0.0, 0.0, 0.0)
        self.last_step = -1

    def get_position(self, step: int) -> Tuple[float, float, float]:
        """Get position from API (with step-based caching)"""
        # Only fetch if this is a new step
        if step != self.last_step:
            try:
                pos = self._fetch_from_api(step)
                self.last_position = pos
                self.last_step = step
            except (OSError, ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to fetch API data for step {step}: {e}")
                # Use last known position

        return self.last_position

    def _fetch_from_api(self, step: int) -> Tuple[float, float, float]:
        """Fetch position from API (implement with actual API client)"""
        # This is a placeholder - implement actual API communication
        # For demo purposes, simulate API response
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            query = urlencode({"step": step, "timestamp": time.time()})
            separator = "&" if "?" in self.api_endpoint else "?"
            request = Request(f"{self.api_endpoint}{separator}{query}", headers=headers)

            with urlopen(request, timeout=self.request_timeout) as response:
                import json

                data = json.loads(response.read().decode("utf-8"))
            return (data["x"], data["y"], data["z"])

        except (OSError, ValueError, TypeError, KeyError) as e:
            logger.error(f"API request failed: {e}")
            # Fallback to simulated data
            t = step * 0.1
            x = 2.0 * math.sin(t)
            y = 1.5 * math.cos(t * 1.2)
            z = 1.0 + 0.3 * math.sin(t * 0.8)
            return (x, y, z)


class PhysicsSimulationMobility(StepwiseMobility):
    """Stepwise mobility pattern that computes positions using real-time physics simulation"""

    def __init__(
        self,
        initial_position: Tuple[float, float, float] = (0, 0, 0),
        initial_velocity: Tuple[float, float, float] = (0, 0, 0),
        gravity: float = -9.81,
        damping: float = 0.99,
    ):
        """
        Initialize physics simulation mobility.

        Args:
            initial_position: Starting position (x, y, z)
            initial_velocity: Starting velocity (vx, vy, vz)
            gravity: Gravity acceleration (m/s²)
            damping: Velocity damping factor (0-1)
        """
        super().__init__()
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.gravity = gravity
        self.damping = damping
        self.dt = 0.1  # Time step for simulation
        self.last_step = -1

    def get_position(self, step: int) -> Tuple[float, float, float]:
        """Get position from physics simulation"""
        # Simulate physics up to the current step
        if step > self.last_step:
            self._simulate_physics(step - self.last_step)
            self.last_step = step

        return tuple(self.position)

    def _simulate_physics(self, steps: int):
        """Simulate physics for the given number of steps"""
        for _ in range(steps):
            # Apply gravity
            self.velocity[2] += self.gravity * self.dt

            # Update position
            self.position += self.velocity * self.dt

            # Apply damping
            self.velocity *= self.damping

            # Simple ground collision (z = 0)
            if self.position[2] < 0:
                self.position[2] = 0
                self.velocity[2] = -self.velocity[2] * 0.8  # Bounce with energy loss


class RandomWalkMobility(StepwiseMobility):
    """Stepwise mobility pattern that generates random walk movement"""

    def __init__(
        self,
        start_position: Tuple[float, float, float] = (0, 0, 0),
        step_size: float = 0.1,
        noise_level: float = 0.05,
    ):
        """
        Initialize random walk mobility.

        Args:
            start_position: Starting position (x, y, z)
            step_size: Maximum step size per update
            noise_level: Random noise level
        """
        super().__init__()
        self.position = np.array(start_position, dtype=float)
        self.step_size = step_size
        self.noise_level = noise_level
        self.last_step = -1

    def get_position(self, step: int) -> Tuple[float, float, float]:
        """Get position from random walk simulation"""
        # Generate random walk up to the current step
        if step > self.last_step:
            self._random_walk(step - self.last_step)
            self.last_step = step

        return tuple(self.position)

    def _random_walk(self, steps: int):
        """Generate random walk for the given number of steps"""
        for _ in range(steps):
            # Generate random direction
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)  # Normalize

            # Add random step
            step = direction * self.step_size * np.random.random()
            self.position += step

            # Add noise
            noise = np.random.randn(3) * self.noise_level
            self.position += noise


class StreamingLookAtOrientation(StreamingOrientationSource):
    """Prepared streaming source that delegates look-at math to the canonical kernel."""

    def __init__(
        self,
        target_mobility: StepwiseMobility,
        yaw_offset_deg: float = 0.0,
        pitch_offset_deg: float = 0.0,
        smooth_factor: float = 1.0,
    ):
        """
        Initialize streaming look-at orientation.

        Args:
            target_mobility: Target mobility pattern to look at
            yaw_offset_deg: Yaw offset in degrees
            pitch_offset_deg: Pitch offset in degrees
            smooth_factor: Smoothing factor for orientation changes (0-1)
        """
        super().__init__()
        self.target_mobility = target_mobility
        self.yaw_offset_deg = yaw_offset_deg
        self.pitch_offset_deg = pitch_offset_deg
        self.smooth_factor = float(smooth_factor)
        if not 0.0 <= self.smooth_factor <= 1.0:
            raise ValueError("smooth_factor must be between 0 and 1")
        self._canonical_orientations: tuple[Orientation3, ...] = ()

    def prepare(
        self,
        steps: int,
        duration: float | None = None,
        context: dict | None = None,
    ) -> None:
        """Sample the live endpoints, then evaluate canonical look-at quaternions."""

        super().prepare(steps, duration, context)
        assert self._scene_steps is not None
        duration_s = float(duration or 0.0)
        timeline = Timeline(self._scene_steps, duration_s)
        owner_positions = self._self_positions or ((0.0, 0.0, 0.0),) * self._scene_steps
        target_positions = tuple(
            _orientation_tuple(self.target_mobility.get_position(step), f"target position {step}")
            for step in range(self._scene_steps)
        )
        owner = prepare_sampled_mobility(
            owner_positions,
            timeline,
            path="streaming.orientation.owner_mobility",
        )
        target = prepare_sampled_mobility(
            target_positions,
            timeline,
            path="streaming.orientation.target_mobility",
        )
        smoothing_time_s = (
            0.0
            if self.smooth_factor == 0.0
            else smoothing_time_from_step_fraction(
                self.smooth_factor,
                steps=self._scene_steps,
                duration_s=duration_s,
            )
        )
        prepared = prepare_orientation(
            LookAtOrientationSpec(
                actor="_streaming_target",
                smoothing_time_s=smoothing_time_s,
                yaw_offset_deg=self.yaw_offset_deg,
                pitch_offset_deg=self.pitch_offset_deg,
            ),
            timeline,
            owner,
            references={"_streaming_target": target},
            path="streaming.orientation",
        )
        if self.smooth_factor == 0.0:
            yaw, pitch, roll = prepared.euler_deg[0]
            prepared = prepare_orientation(
                FixedOrientationSpec(
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    roll_deg=roll,
                ),
                timeline,
                owner,
                path="streaming.orientation",
            )
        self._canonical_orientations = prepared.euler_deg

    def get_orientation(
        self,
        step: int,
        total_steps: int,
        current_pos=None,
        previous_pos=None,
        scene_duration: float = None,
    ) -> Tuple[float, float, float]:
        """Return one sample prepared by the canonical kernel."""

        del total_steps, current_pos, previous_pos, scene_duration
        if not self._canonical_orientations:
            raise RuntimeError("StreamingLookAtOrientation must be prepared before use")
        return self._canonical_orientations[step]


class StreamingCircularOrientation(StreamingOrientationSource):
    """Prepared streaming source backed by the canonical spin model."""

    def __init__(
        self,
        rotation_speed_deg_per_step: float = 1.0,
        axis: str = "yaw",
        base_yaw: float = 0.0,
        base_pitch: float = 0.0,
        base_roll: float = 0.0,
    ):
        """
        Initialize streaming circular orientation.

        Args:
            rotation_speed_deg_per_step: Rotation speed in degrees per step
            axis: Rotation axis ('yaw', 'pitch', or 'roll')
            base_yaw: Base yaw angle in degrees
            base_pitch: Base pitch angle in degrees
            base_roll: Base roll angle in degrees
        """
        super().__init__()
        self.rotation_speed = rotation_speed_deg_per_step
        self.axis = axis.lower()
        self.base_yaw = base_yaw
        self.base_pitch = base_pitch
        self.base_roll = base_roll
        self._canonical_orientations: tuple[Orientation3, ...] = ()

    def prepare(
        self,
        steps: int,
        duration: float | None = None,
        context: dict | None = None,
    ) -> None:
        """Map the configured per-step rate onto a canonical time-based spin."""

        super().prepare(steps, duration, context)
        assert self._scene_steps is not None
        duration_s = float(duration or 0.0)
        rate_deg_s = (
            self.rotation_speed * (self._scene_steps - 1) / duration_s
            if self._scene_steps > 1 and duration_s > 0.0
            else 0.0
        )
        timeline = Timeline(self._scene_steps, duration_s)
        owner_positions = self._self_positions or ((0.0, 0.0, 0.0),) * self._scene_steps
        owner = prepare_sampled_mobility(
            owner_positions,
            timeline,
            physical_velocity=False,
            path="streaming.orientation.owner_mobility",
        )
        prepared = prepare_orientation(
            SpinOrientationSpec(
                axis=self.axis,
                rate_deg_s=rate_deg_s,
                yaw_deg=self.base_yaw,
                pitch_deg=self.base_pitch,
                roll_deg=self.base_roll,
            ),
            timeline,
            owner,
            path="streaming.orientation",
        )
        self._canonical_orientations = prepared.euler_deg

    def get_orientation(
        self,
        step: int,
        total_steps: int,
        current_pos=None,
        previous_pos=None,
        scene_duration: float = None,
    ) -> Tuple[float, float, float]:
        """Return one sample prepared by the canonical kernel."""

        del total_steps, current_pos, previous_pos, scene_duration
        if not self._canonical_orientations:
            raise RuntimeError("StreamingCircularOrientation must be prepared before use")
        return self._canonical_orientations[step]
