# Quickstart

Activate the environment from [Installation](installation.md) and open a terminal in the root of the matching source checkout. Hello World uses one stationary transmitter and receiver. Generate its single frame, then open it in the Visualizer.

## 1. Generate Frames

```bash
orchav-generator scenarios/getting_started/hello_world/
```

The [scenario.yaml](../../scenarios/getting_started/hello_world/scenario.yaml)
defines the scene, radios, timeline and ray-tracing settings. The Generator
validates it, asks [Sionna RT](https://nvlabs.github.io/sionna/) to compute paths,
and saves one HDF5 frame under `scenarios/getting_started/hello_world/frames/`.

If generation fails or produces no frame set, see
[backend troubleshooting](../help/troubleshooting.md#mitsuba-and-drjit-backend-messages).

## 2. Open the Visualizer

```bash
orchav-visualizer --scenario scenarios/getting_started/hello_world
```

The default pygfx renderer shows the scene, transmitter, receiver and their
propagation paths. This scenario has one stationary frame, so playback does not
move the actors. Continue with [Your First Visualizer Session](../visualizer/first_session.md)
to explore its paths and controls.

Keep the generated `frames/` directory. Viewing it does not repeat ray tracing;
generate it again only after changing the scenario inputs.

## Optional Checks

Inspect the saved frame without opening the desktop application:

```bash
orchav-inspect scenarios/getting_started/hello_world --frame 0
```

The summary lists radios, path counts, available metrics and positions. See
[Frame Inspection](../shared/frame_inspection.md) for other views.

After editing a scenario, check its configuration and input paths separately:

```bash
orchav-validate scenarios/getting_started/hello_world
```

This check does not trace rays. `--strict` also treats referenced-path warnings
as errors, including a missing configured frame directory; it does not inspect
frame contents. See [Scenario Validation](../reference/scenario_validation.md).

## Choose Another Task

| Goal | Start here |
|------|------------|
| Understand the YAML you just ran | [Hello World walkthrough](../../scenarios/getting_started/hello_world/README.md) |
| Calculate actor motion in Python | [Hello World Scripted](../../scenarios/getting_started/hello_world_scripted/README.md) |
| Inspect a first sensing result | [Stationary Cube](../../scenarios/sensing/cube_radar/stationary/README.md) |
| Explore another feature | [Scenario catalog](../../scenarios/README.md) |

You can also list the included Generator scenarios from the terminal:

```bash
orchav-generator
```

---

Previous: [Installation](installation.md) | Home: [Documentation](../README.md) | Continue: [Your First Visualizer Session](../visualizer/first_session.md)
