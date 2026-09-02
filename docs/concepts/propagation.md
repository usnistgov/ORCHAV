# Propagation Concepts

ORCHAV orchestrates [Sionna RT](https://nvlabs.github.io/sionna/)
propagation studies. The [Generator](../generator/README.md) prepares the scene,
actors, timeline, and solver settings. Sionna RT solves propagation paths, then
the Generator converts the results into ORCHAV frames for the
[Shared Data Layer](../shared/README.md), [Visualizer](../visualizer/README.md),
and analysis tools.

This page is a first-reader domain primer. Use the generated
[Scenario YAML Reference](../reference/scenario_yaml.md) for exact scenario
fields and the [Frame Data Reference](../shared/frame_reference.md) for frame
fields, shapes, units, and validation rules.

## Ray Tracing

When ray tracing is enabled, the Generator applies the current actor state and
solver settings, then asks Sionna RT to find valid paths through the scene.
Geometry, radio materials, enabled interactions, and the ray budget affect the
result. Larger budgets can discover more paths but cost more time and memory.

The `files` data mode writes manifest-driven HDF5 frames. Normal Visualizer
playback reads those frames without rerunning ray tracing. [Local What-if
Preview](../visualizer/interactive_recomputation.md#local-what-if-preview) is a
separate workflow that can temporarily recompute only the displayed step. See
[Simulation Configuration](../generator/configuration.md#ray-tracing-yaml-fields)
for tracing controls.

## Multipath Components

A multipath component, or MPC, is one propagation path between a transmitter
and receiver in one frame. One TX/RX pair can have a direct path, several paths
that interact with the scene, or no valid path.

```text
TX ------------------------------> RX       direct path, no bounce
TX ----------> wall -----------> RX       reflected path, one bounce
```

The compact frame contract assigns paths to TX/RX pairs with `tx_rx_pairs` and
`pair_path_offsets`, then assigns physical bounces with `bounce_offsets`. See
the [Frame Contract](../shared/frame_reference.md#frame-contract) for the exact
arrays and missing-value rules.

## Interaction Types

The `interactions` array stores one code per physical bounce. A direct
line-of-sight path has no physical bounce row. The Visualizer derives code `0`
when it classifies the direct path.

| Code | Mechanism | Meaning |
|---:|---|---|
| `0` | Line of sight | Derived segment classification, not a stored bounce. |
| `1` | Specular reflection | Mirror-like reflection from a surface. |
| `2` | Diffuse scattering | Rough-surface scattering into more directions. |
| `4` | Refraction | Transmission through a material boundary. |
| `8` | Diffraction | Propagation around an edge when supported by the backend and settings. |

One MPC can have an ordered sequence of several mechanisms. Inspect individual
paths in the [MPC Explorer](../visualizer/mpc_explorer.md).

## Path Metrics

ORCHAV frames contain path-aligned arrays for delay, path loss, angle of
arrival, and angle of departure, but the values are optional. The Generator
populates them when `raytracing.export_path_metrics: true` and the backend
supplies the values. An unavailable value is `NaN` with its bit cleared in
`metric_valid_bits`.

Delay uses nanoseconds, path loss uses decibels, and AoA/AoD use world-space
degrees. Path loss is attenuation, not received power in watts or dBm. RMS
delay spread is weighted by path gain and uses only paths with valid delay and
path loss. See [Statistics Helpers](../shared/statistics.md) for calculations
and [Visual Analysis](../visualizer/analysis.md) for inspection.

## Materials

Radio materials affect reflection, scattering, transmission, and attenuation.
They are separate from render colors and textures. ORCHAV can retain values
loaded with the scene, apply a material-family scattering preset, or apply
explicit overrides. See [Material
Overrides](../generator/configuration.md#material-overrides) and the included
[scene-material scenarios](../../scenarios/generator/README.md#propagation-and-materials).

## Diffuse Scattering

Diffuse paths require tracing quality with `diffuse_reflection` enabled and a
positive `scattering_coefficient` on a relevant material. A scattering pattern
controls the angular distribution after a diffuse event. It does not steer an
antenna. Compare [Scene Diffuse
Scattering](../../scenarios/generator/propagation_and_materials/scene_diffuse_scattering/README.md)
with [Target Diffuse
Scattering](../../scenarios/generator/targets/target_diffuse_scattering/README.md).

## Antennas

TX and RX arrays can have independent element patterns, polarization,
dimensions, and spacing. Standard quality presets use synthetic-array mode, so
array changes normally affect response and gain without changing traced path
geometry. Visualizer beam-pattern overlays are inspection-only and do not
change generated frames or Generator physics. See [Antenna
Configuration](../generator/configuration.md#antenna-arrays) and [Beam-Pattern
Visualization](../visualizer/beam_patterns.md).

## Coverage Maps

Coverage maps are separate grid datasets, not MPCs embedded in normal frames.
The solver samples virtual receiver locations, so a coverage-only scenario does
not need named receiver actors. ORCHAV stores canonical path gain and can derive
path loss, received power, SINR, serving-transmitter, and transmitter-margin
layers when their inputs are available.

Finer grids and larger solver budgets increase computation and output size.
See [Coverage Configuration](../generator/configuration.md#coverage-configuration),
[Coverage Inspection](../visualizer/analysis.md#coverage-inspection), and the
[Single-Transmitter Coverage scenario](../../scenarios/generator/coverage/single_tx/README.md).

## Motion Across Frames

Actor motion can change path geometry and available metrics from one frame to
the next. Core frames store actor positions and radio-node orientations, but
not per-path complex phase or Doppler. See [Mobility and
Orientation](../generator/mobility_and_orientation.md) and [Summary and Coverage
Figures](../generator/generated_figures.md) for trajectory diagnostics.

## Choose The Next Guide

| Goal | Guide |
|---|---|
| Configure tracing, materials, antennas, filtering, or coverage | [Simulation Configuration](../generator/configuration.md) |
| Look up an exact scenario field or default | [Scenario YAML Reference](../reference/scenario_yaml.md) |
| Inspect frame fields, units, and missing values | [Frame Data Reference](../shared/frame_reference.md) |
| Analyze path metrics in Python | [Statistics Helpers](../shared/statistics.md) |
| Explore paths and coverage interactively | [Visual Analysis](../visualizer/analysis.md) |
| Run a focused propagation example | [Generator Scenarios](../../scenarios/generator/README.md) |

---

Home: [Documentation](../README.md) | Related: [Glossary](../reference/glossary.md)
