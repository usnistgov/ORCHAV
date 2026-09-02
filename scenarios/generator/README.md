# Generator Scenarios

[Scenarios](../README.md) > Generator

These runnable YAML scenarios teach Generator features in independent learning
tracks. Start with the track that matches your task. The tracks are not one
mandatory sequence.

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

## Running A Scenario

From the repository root, replace the path below with any scenario in the
tables:

```bash
orchav-validate scenarios/generator/mobility_and_orientation/actor_mobility
orchav-generator scenarios/generator/mobility_and_orientation/actor_mobility
orchav-inspect scenarios/generator/mobility_and_orientation/actor_mobility --frame 0
orchav-visualizer --scenario scenarios/generator/mobility_and_orientation/actor_mobility
```

All scenarios in this catalog are authored directly in `scenario.yaml`.
Generated frames, coverage data, and figures are written inside the selected
scenario directory and are not committed.

See [Scenario Authoring](../../docs/generator/scenario_authoring.md) to create a
new scenario or [Simulation
Configuration](../../docs/generator/configuration.md) for tracing and coverage
settings.

---

Up: [All Scenarios](../README.md) | Choose a track: [Actor Mobility](mobility_and_orientation/actor_mobility/README.md) | [Specular Reflection](propagation_and_materials/specular_reflection/README.md) | [Mesh Targets](targets/mesh_targets/README.md) | [Single-Transmitter Coverage](coverage/single_tx/README.md)
