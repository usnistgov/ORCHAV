# Changelog

## 0.2.0 - 2026-11-16

- Introduces [ORCHAV Sensing](docs/sensing/README.md) for range, Doppler, and angle
  estimation, detection, tracking, and evaluation.
- Supports coherent acquisitions and sequential processing, with configurable
  bandwidth, noise, clutter filtering, and detection settings.
- Reprocesses saved observations and compares sensing methods without rerunning
  ray tracing.
- Adds a Sensing Dashboard with range-Doppler plots, detections, tracks,
  evaluation figures, and replay alongside the 3D scene.
- Includes [sensing examples](scenarios/sensing/README.md) for cubes, walking
  targets, vehicles, drones, sensor fusion, and urban scenes.
- Supports mesh/material targets and optional
  [Sionna RT 2.2](https://github.com/NVlabs/sionna-rt/releases/tag/v2.2.0) native
  RCS models.
- Adds multi-height radio coverage, background coverage generation, and scene
  appearance controls.
- Adds OpenStreetMap scene generation from a location or bounding box, including
  buildings, roads, and optional terrain and vegetation.
- Imports animated targets from SMPL/AMASS and Kimodo/SOMA data obtained
  separately by the user.

Requires Python 3.12. See [Installation](docs/getting_started/installation.md)
for setup and [Versions and Compatibility](docs/reference/compatibility.md)
for supported data formats.

## 0.1.0 - 2026-09-01

- Introduces the ORCHAV Generator, Visualizer, Shared Data Layer, and example
  scenarios.
- Provides documentation for installation, scenario workflows, visualization,
  data handling, and licensing.
- Includes scene, target, texture, and lighting assets for the examples.
