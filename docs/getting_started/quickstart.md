# Quickstart

> **Prerequisite**: complete the [Installation](installation.md) steps first and
> activate your Python environment before running the commands below.

This walkthrough generates frames from the included Hello World scenario,
inspects them in the terminal, and opens them in the Visualizer:

```text
scenario.yaml -> ORCHAV Generator -> frames/
                                      |---> orchav-inspect
                                      `---> ORCHAV Visualizer
```

## 1. Validate the Scenario (Optional)

Run a fast configuration check before invoking
[Sionna RT](https://nvlabs.github.io/sionna/):

```bash
orchav-validate scenarios/getting_started/hello_world
```

The Generator also validates a scenario's configuration before generation
begins. Running this separate preflight is useful after editing a scenario
because it checks the YAML, scenario schema, and referenced input paths without
tracing rays.

## 2. Generate Frames

```bash
orchav-generator scenarios/getting_started/hello_world/
```

The Generator writes HDF5 frame chunks under `frames/`. These files are the
persisted simulation output: the Visualizer uses them for paths and actor
state while loading scene geometry from the scenario, and Python tools can
inspect the same frame data. See
the [Frame Data Reference](../shared/frame_reference.md) for how persisted HDF5 frames are loaded and how
data modes are selected.

This example is YAML-authored: `scenario.yaml` defines its scene, transmitter,
receiver, timeline, and ray-tracing settings.

The terminal may show messages such as:

```text
jitc_llvm_init(): LLVM API initialization failed ...
jit_registry_shutdown(): leaking ...
jit_malloc_shutdown(): leaked ...
```

These messages do not all mean generation failed. If the command exits
successfully, continue to the inspection step. If it exits with an error,
produces no new frame set, or the LLVM initialization failure appears when you
intended to run on the CPU, see
[Troubleshooting](../help/troubleshooting.md#mitsuba-and-drjit-backend-messages).

For a scenario that saves frames, `orchav-validate --strict` also checks that
the configured frame directory exists. It treats any referenced-path warning
as an error, but it does not inspect the saved frame contents. The next step
confirms that a frame can be loaded. See
[Scenario Validation](../reference/scenario_validation.md) for the exact checks.

## 3. Inspect the Generated Frame

```bash
orchav-inspect scenarios/getting_started/hello_world --frame 0
```

The inspector prints a compact summary of the generated HDF5 frame: available
frame indices, TX/RX counts, path counts, exported metrics, and node positions.
Use it when you want to check generated frame data without opening the desktop
Visualizer. See [Frame Inspection](../shared/frame_inspection.md) for path inputs,
`--list`, and `--json`.

## 4. Visualize the Scenario

```bash
orchav-visualizer --scenario scenarios/getting_started/hello_world
```

By default this launches the `pygfx` renderer. On Windows and Linux, you can
use `--renderer open3d` for the Open3D/Filament backend. On macOS, use pygfx;
an explicit Open3D request is rejected before Qt starts. See
[Renderers](../visualizer/renderers.md) for the backend differences and
platform behavior.

## 5. Continue With Hello World

Keep the generated `frames/` directory. The next pages reuse it. You only need
to generate it again after changing the scenario inputs.

1. Read the [Hello World scenario
   walkthrough](../../scenarios/getting_started/hello_world/README.md) for a
   guided explanation of its scene, YAML configuration, generated files, and
   expected result.
2. Follow [Your First Visualizer Session](../visualizer/first_session.md) to
   explore the same frame in the desktop interface.

For another workflow, choose a scenario from the table below. The scenario tree
is organized by workflow. Use the smallest scenario that exercises the feature
you need.

| Goal | Start here |
|------|------------|
| Learn the Visualizer interface | [Your First Visualizer Session](../visualizer/first_session.md) |
| Verify installation and YAML-only authoring | [`scenarios/getting_started/hello_world/`](../../scenarios/getting_started/hello_world/README.md) |
| Inspect a minimal Python-scripted example | [`scenarios/getting_started/hello_world_scripted/`](../../scenarios/getting_started/hello_world_scripted/README.md) |
| Try Generator features | [`scenarios/generator/`](../../scenarios/generator/README.md) |
| Test Visualizer loading, rendering, and data modes | [`scenarios/visualizer/`](../../scenarios/visualizer/README.md) |

You can also list the included Generator scenarios from the terminal:

```bash
orchav-generator
```

See [Included Scenarios](../../scenarios/README.md) for the full list.

> **Contributing:** use [Contributing](../../CONTRIBUTING.md#checks) for the
> development quality gates and test commands.

## More Detail

- [Hello World Scenario](../../scenarios/getting_started/hello_world/README.md)
- [Generator Guide](../generator/README.md)
- [Visualizer First Session](../visualizer/first_session.md)
- [Frame Inspection](../shared/frame_inspection.md)
- [Frame Data Reference](../shared/frame_reference.md)
- [Generated Figures](../generator/generated_figures.md)
- [Scenario Authoring](../generator/scenario_authoring.md)
- [Scenario Validation](../reference/scenario_validation.md)
- [Scenarios](../../scenarios/README.md)

---

Previous: [Installation](installation.md) | Home: [Documentation](../README.md) | Continue: [Hello World Scenario](../../scenarios/getting_started/hello_world/README.md) | Related: [Included Scenarios](../../scenarios/README.md)
