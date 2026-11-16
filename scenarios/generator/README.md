# Generator Scenarios

[Scenarios](../README.md) > Generator

After [Hello World](../getting_started/hello_world/README.md), use these
examples to define moving actors, compare propagation mechanisms, add targets,
and compute coverage. Choose a track by the part of the simulation you want to
change. Each guide explains its configuration, running commands, and results.

## Mobility and Orientation

Start here when learning how actor positions and orientations change over the
shared timeline.

| Scenario | What it demonstrates |
|---|---|
| [`actor_mobility/`](mobility_and_orientation/actor_mobility/README.md) | Stationary, linear, waypoint, circular, and grid-scan receivers. |
| [`actor_orientation/`](mobility_and_orientation/actor_orientation/README.md) | Fixed, look-at, and motion-aligned orientation. |

## Propagation and Materials

Use this track to isolate path mechanisms and material behavior in small static
scenes.

| Scenario | What it demonstrates |
|---|---|
| [`specular_reflection/`](propagation_and_materials/specular_reflection/README.md) | Line-of-sight and mirror-like reflections. |
| [`scene_diffuse_scattering/`](propagation_and_materials/scene_diffuse_scattering/README.md) | The diffuse solver switch, material presets, explicit overrides, and output-size effects. |
| [`refraction_and_diffraction/`](propagation_and_materials/refraction_and_diffraction/README.md) | Refraction and diffraction options supported by the active [Sionna RT](https://nvlabs.github.io/sionna/) solver. |

## Targets

Use this track after the actor mobility, orientation, and propagation examples.

| Scenario | What it demonstrates |
|---|---|
| [`mesh_targets/`](targets/mesh_targets/README.md) | Moving and stationary mesh-backed targets. |
| [`target_orientation/`](targets/target_orientation/README.md) | Target, TX, and RX orientation checks. |
| [`target_diffuse_scattering/`](targets/target_diffuse_scattering/README.md) | Diffuse interactions from a target surface. |

## Coverage

Coverage scenarios sample virtual receiver locations across a grid instead of
producing only named TX/RX frame streams.

| Scenario | What it demonstrates |
|---|---|
| [`single_tx/`](coverage/single_tx/README.md) | One-transmitter coverage at two heights. |
| [`multi_tx/`](coverage/multi_tx/README.md) | Per-transmitter layers, serving-TX selection, margin, and SINR. |
| [`paris_volume/`](coverage/paris_volume/README.md) | Inspect slices and volume sweeps at 13 heights around the Arc de Triomphe. |


## Geographic Scene Construction

[Paris Opéra from OSM](paris_opera_osm/README.md) acquires current map features
and constructs the scene locally from its YAML recipe. The optional OSM
installation and attribution steps are included; geographic data is not bundled.




## Running A Scenario

Run this example from the repository root. For other scenarios, check the scenario guide for prerequisites, preparation scripts, and output paths.

Validate the mobility example:

```bash
orchav-validate scenarios/generator/mobility_and_orientation/actor_mobility
```

Generate its recording:

```bash
orchav-generator scenarios/generator/mobility_and_orientation/actor_mobility
```

Inspect the first frame:

```bash
orchav-inspect scenarios/generator/mobility_and_orientation/actor_mobility --frame 0
```

Open the recording in the Visualizer:

```bash
orchav-visualizer --scenario scenarios/generator/mobility_and_orientation/actor_mobility
```

See [Scenario Authoring](../../docs/generator/scenario_authoring.md) to create a
new scenario or [Simulation
Configuration](../../docs/generator/configuration.md) for tracing and coverage
settings.

To change how an existing scene looks, use the Visualizer's
[Building Appearance exercise](../visualizer/building_appearance/README.md).

---

Up: [All Scenarios](../README.md) | Choose a track: [Actor Mobility](mobility_and_orientation/actor_mobility/README.md) | [Specular Reflection](propagation_and_materials/specular_reflection/README.md) | [Mesh Targets](targets/mesh_targets/README.md) | [Single-Transmitter Coverage](coverage/single_tx/README.md)
