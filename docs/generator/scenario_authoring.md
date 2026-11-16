# Scenario Authoring

An ORCHAV scenario is a directory built around `scenario.yaml`. The YAML
selects a scene, defines one shared timeline, declares actors, and includes
only the simulation or output settings that differ from the defaults. ORCHAV
validates and prepares that configuration before asking
[Sionna RT](https://nvlabs.github.io/sionna/) to compute propagation paths.

## Choose An Authoring Surface

| Goal | Surface |
|---|---|
| YAML-only scenarios with declarative actors, motion, targets and outputs | `scenario.yaml` |
| Visual creation using the supported schema | [Scenario Builder](scenario_builder.md) |
| Calculated actors, preprocessing, generated cases, or custom equations | `generate.py` |

The Scenario Builder writes the same validated YAML used by command-line
generation. It preserves recognized settings outside its editable surface.

## Scenario Directory

```text
my_scenario/
|-- scenario.yaml       # required scenario definition
|-- README.md           # recommended explanation and running instructions
|-- generate.py         # optional Python-scripted actor construction
|-- actors.py           # optional actor factory shared with local preview
|-- frames/             # generated HDF5 frame set
|-- summary/            # optional diagnostic figures
`-- coverage/           # optional coverage-map data
```

Run the Generator to create the output directories. Open the frames in the [Visualizer](../visualizer/README.md) or [load them from Python](../shared/README.md).

## Start With YAML

The smallest generated scenario contains one stationary transmitter and one
stationary receiver. This example selects `ultra-low` so an introductory run
finishes quickly. Omit the quality block to use the normal `low` default.

Create `scenarios/my_scenario/` in your checkout and save the following as
`scenarios/my_scenario/scenario.yaml`.

```yaml
schema_version: 2

scene:
  source: sionna
  id: etoile

timeline:
  steps: 1
  duration_s: 0.0

raytracing:
  enabled: true
  quality:
    preset: ultra-low

actors:
  tx:
    - name: TX1
      mobility:
        type: stationary
        position_m: [-114.0, 37.0, 30.0]
  rx:
    - name: RX1
      mobility:
        type: stationary
        position_m: [-27.18, -12.61, 1.5]
```

Give each [actor](../reference/glossary.md#actor) a unique name. Use `actors.tx`
and `actors.rx` for radios, and `actors.targets` for target objects. Set positions
in metres in the mobility model. The [Scenario YAML Reference](../reference/scenario_yaml.md)
lists optional sections and defaults.

## Validate and generate

Run commands from the repository root.

Validate your scenario:

```bash
orchav-validate scenarios/my_scenario/
```

Generate its recording:

```bash
orchav-generator scenarios/my_scenario/
```

Generation writes reusable HDF5 frames under `frames/`. Continue with
[frame inspection](../shared/frame_inspection.md) or the
[first Visualizer session](../visualizer/first_session.md) to inspect the result.

Generation checks the scenario before computing paths. To diagnose an edit on its own, run `orchav-validate`: it reports invalid fields and unresolved references. The Generator and [Scenario Builder](scenario_builder.md) sample the same actor definitions on the scenario timeline. See [scenario validation](../reference/scenario_validation.md) for asset checks and strict mode.

## Add Motion

Every actor has one mobility model. Position is part of that model, so the
initial pose remains unambiguous for stationary and moving actors.

```yaml
timeline:
  steps: 30
  duration_s: 3.0

actors:
  rx:
    - name: WalkingRX
      mobility:
        type: linear
        start_m: [-10.0, 0.0, 1.5]
        end_m: [10.0, 0.0, 1.5]
      orientation:
        type: align_motion
        allow_pitch: false
```

This receiver traverses the complete line over the scenario duration and faces
its direction of travel. [Mobility and Orientation](mobility_and_orientation.md)
explains how to set the sampling interval, speed and heading.

## Add A Target

For an ordinary mesh target, `asset` supplies geometry and a radio material.
Its `mobility` and `orientation` can keep it fixed or move it rigidly
through a scene; neither requires a human-motion model. For a changing shape,
such as a walking person's moving limbs, the simplest input is a
[folder of PLY poses](motion/animated_targets.md#simple-example-a-folder-of-ply-poses).
An explicit timed [manifest](motion/animated_targets.md#alternative-a-timed-motion-manifest)
is optional for advanced motion data. Native Sionna RT 2.2
[RCS targets](../sensing/target_models.md#choose-a-representation) instead
use prescribed scattering-point responses rather than the mesh-material
surface response. These are separate choices of RF representation and motion.

```yaml
actors:
  targets:
    - name: ParkedCar
      asset:
        source: catalog
        id: car
        material_type: metal
        scale: 1.0
      mobility:
        type: stationary
        position_m: [5.0, -8.0, 0.79]
      orientation:
        type: fixed
        yaw_deg: 90.0
```

An asset may come from the included catalog, one mesh file, or a directory and
pattern for a mesh sequence. Keep scenario-local paths relative to the scenario
directory and shared-library paths relative to the repository root. The
[bundled UAV meshes](../../libraries/targets/uav/README.md) are another option
for target geometry. Their use does not imply calibrated radar cross section.

## Refer To Another Actor

Orientation and group relationships use globally unique names. For example, a
stationary transmitter can track a moving receiver:

```yaml
actors:
  tx:
    - name: TrackingTX
      mobility:
        type: stationary
        position_m: [0.0, 0.0, 10.0]
      orientation:
        type: look_at
        actor: WalkingRX
```

Validation rejects unresolved, ambiguous, and self references. Renaming an
actor in the Scenario Builder updates references before saving.

## Python-Scripted Actors

A scripted scenario loads the normal YAML configuration and supplies calculated
actors to the same Generator pipeline. Put actor construction in an adjacent
`actors.py` file exposing `build_actors(scenario) -> ActorsSpec`. The
[Hello World Scripted example](../../scenarios/getting_started/hello_world_scripted/README.md)
provides a complete factory for a calculated trajectory.

The `generate.py` entry point can then reuse that factory:

```python
from pathlib import Path

from generator import build_simulation_config, perform_pipeline
from shared.scenarios import load_scenario_configuration
from shared.scenarios.scripted import load_scripted_actors

scenario = load_scenario_configuration(Path(__file__).parent)
perform_pipeline(
    simulation_config=build_simulation_config(scenario),
    scenario_configuration=scenario,
    actors=load_scripted_actors(scenario),
)
```

The supplied actors replace the YAML actor collection for that invocation.
They retain its timeline and the same motion, reference and runtime validation.
The factory should return actors without writing outputs or mutating the
configuration.

[Local What-if Preview](../visualizer/interactive_recomputation.md#local-what-if-preview)
uses the same factory when YAML does not define both TX and RX actors. It does
not execute `generate.py`, so separate input preparation from actor construction.

## Local scene mesh transforms

Read this section when a local Mitsuba XML scene applies transforms directly
to PLY or OBJ shapes. Sionna RT and Mitsuba evaluate the complete scene
transform grammar for ray tracing. ORCHAV also reads those meshes for
Visualizer and Scenario Builder previews, summary overlays, and fallback scene
bounds. This lightweight mesh reader accepts the following ordered transform
subset:

```xml
<transform name="to_world">
  <scale value="2"/>
  <rotate x="1" angle="10"/>
  <rotate y="1" angle="20"/>
  <rotate z="1" angle="30"/>
  <translate x="1" y="2" z="3"/>
</transform>
```

Each operation may be omitted. Scale must be one positive uniform scalar.
Rotations use the positive X, Y, then Z canonical axes. Translation may use
X/Y/Z attributes or a three-value vector. Keep the written order as scale,
rotations, then translation.

Matrices, `lookat`, nonuniform or mirrored scale, arbitrary rotation axes,
repeated or reordered operations, and symbolic values are valid in the wider
Mitsuba grammar but cannot be represented by ORCHAV's lightweight reader. A
direct mesh using one of those forms is rejected for ORCHAV preview instead of
being displayed at an approximate position. Camera and sensor transforms are
not subject to this lightweight mesh restriction.

## Continue

- [Simulation Configuration](configuration.md) explains scene selection,
  ray-tracing settings, antennas, materials, path filtering, and coverage.
- [Mobility and Orientation](mobility_and_orientation.md) covers pose models and
  group motion.
- [Summary and Coverage Figures](generated_figures.md) covers standalone images
  written beside generated data.
- [Generator Scenarios](../../scenarios/generator/README.md) provides runnable
  examples organized by learning track.

---

Up: [Generator](README.md) | Next: [Simulation Configuration](configuration.md) | Related: [Scenario YAML Reference](../reference/scenario_yaml.md)
