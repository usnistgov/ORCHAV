# Scenarios

ORCHAV includes runnable scenarios for the first workflow, Generator features,
and Visualizer features.

## Recommended Path

1. Start with [Getting Started](getting_started/README.md) to learn the basic
   generation, inspection, and visualization workflow.
2. Choose the [Generator scenarios](generator/README.md) to learn authoring and
   frame, summary, or coverage generation. Choose the [Visualizer
   scenarios](visualizer/README.md) to continue the guided playback and
   analysis path.

Each group guide lists its included scenarios and recommends an order based on
the feature you want to learn.

## Scenario Contents

Scenario directories contain portable inputs: usually a
[`scenario.yaml`](../docs/generator/scenario_authoring.md#start-with-yaml) file, plus a
[`generate.py` Python driver](../docs/generator/scenario_authoring.md#python-scripted-actors)
when a scenario needs behavior outside the YAML schema, such as a calculated
trajectory, external preprocessing, or generated setup.
Generated outputs such as `frames/`, `summary/`, and rendered images are local
artifacts that can be recreated.

## Units And Coordinates

Unless a scenario or parameter name states otherwise, ORCHAV scenarios use
meters for positions, distances, dimensions, altitudes, radii, and waypoints.
Position vectors are `[x, y, z]` values in the scenario's local Cartesian scene
frame. Speeds are meters per second. Durations are seconds. Static yaw, pitch,
and roll values are degrees. Carrier frequencies and bandwidths are hertz.

Coverage-grid coordinates and heights are meters. Path loss is reported in
decibels.

## What Is Not Bundled

The runnable scenario directories include definitions and curated input assets,
not their generated [`frames/`](../docs/shared/frame_reference.md#hdf5-frame-layout),
[`summary/`](../docs/generator/generated_figures.md),
[`coverage/`](../docs/generator/configuration.md#coverage-configuration), video, or
cache output directories. Run the documented commands to create those products
locally.
Selected preview images used by these guides are maintained separately under
`docs/assets/`.

## Input Assets

The repository includes only curated target assets needed by included examples.
See [LICENSE](../LICENSE), [NOTICE](../NOTICE), the
[NIST software disclaimer](../NIST_SOFTWARE_DISCLAIMER.md), and
[Third-Party Notices](../THIRD_PARTY_NOTICES.md) for bundled asset notices.

---

Up: [Documentation](../docs/README.md) | Begin: [Getting Started](getting_started/README.md)
