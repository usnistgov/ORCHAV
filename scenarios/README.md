# Scenarios

Start with [Getting Started](getting_started/README.md) to generate, inspect, and open your first frame set. Then choose a catalog by task.

## Recommended Path

| Folder | What you can do |
|---|---|
| [getting_started/](getting_started/README.md) | Run Hello World and inspect the results in the Visualizer. |
| [generator/](generator/README.md) | Configure actors, propagation, materials, or coverage grids and generate results. |
| [visualizer/](visualizer/README.md) | Inspect propagation paths, follow channel metrics, and compare links during playback. |
| [sensing/](sensing/README.md) | Start with a stationary cube, then explore motion, detection, tracking, and more complex scenes. |

To analyze sensing results on a CPU, first generate an acquisition with a supported ray-tracing backend, then follow the [saved-acquisition workflow](../docs/sensing/reprocessing.md#run-processing). Generated sensing recordings are not bundled.


## Scenario Contents

Scenario directories contain portable inputs: usually a
[`scenario.yaml`](../docs/generator/scenario_authoring.md#start-with-yaml) file, plus a
[`generate.py` Python driver](../docs/generator/scenario_authoring.md#python-scripted-actors)
when a scenario needs behavior outside the YAML schema, such as a calculated
trajectory, external preprocessing, or generated setup.

## Units And Coordinates

Unless a scenario or parameter name states otherwise, ORCHAV scenarios use
meters for positions, distances, dimensions, altitudes, radii, and waypoints.
Position vectors are `[x, y, z]` values in the scenario's local Cartesian scene
frame. Speeds are meters per second. Durations are seconds. Static yaw, pitch,
and roll values are degrees. Carrier frequencies and bandwidths are hertz.

Coverage-grid coordinates and heights are meters. Path loss is reported in
decibels.

## Generated outputs

Each scenario guide gives the commands for generating its [frames](../docs/shared/frame_reference.md#hdf5-frame-layout), [summary figures](../docs/generator/generated_figures.md), or [coverage data](../docs/generator/configuration.md#coverage-configuration). Guides that include a recording or reference result explain how to use it.

## Input Assets

The repository includes only curated target assets needed by included examples.
See [LICENSE](../LICENSE), [NOTICE](../NOTICE), the
[NIST software disclaimer](../NIST_SOFTWARE_DISCLAIMER.md), and
[Third-Party Notices](../THIRD_PARTY_NOTICES.md) for bundled asset notices.

---

Up: [Documentation](../docs/README.md) | Begin: [Getting Started](getting_started/README.md)
