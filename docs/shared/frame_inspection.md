# Frame Inspection

`orchav-inspect` prints a terminal summary of generated ORCHAV HDF5 frames and
coverage files. Use it after `orchav-generator` when you want to check data
contents without opening the desktop Visualizer.

```bash
orchav-inspect scenarios/getting_started/hello_world --frame 0
```

For example, a generated Hello World recording reports the following summary
(excerpt). MPC counts can vary with the solver and sampling settings:

```text
Frame count: 1
Available frames: 0
Selected frame: 0
TX: 1, RX: 1, Targets: 0
Pairs: 1
Total MPCs: 6
Paths per pair: [6]
Metrics: delays_ns, path_loss_db, aoa_az_deg, aoa_el_deg, aod_az_deg, aod_el_deg
```

Check that the actor and pair counts match your scenario. The complete output
also lists interaction counts and node positions.

For frames with sensing data, the summary also lists the processing mode,
saved observations and spectra with their dimensions, target and detection
counts, and range/velocity resolution. Detector diagnostics are summarized
on one line.

## Inputs

The command accepts these input forms:

| Input | Example |
| --- | --- |
| Scenario directory | `scenarios/getting_started/hello_world` |
| Scenario YAML | `scenarios/getting_started/hello_world/scenario.yaml` |
| Frame directory | `scenarios/getting_started/hello_world/frames` |
| Individual chunk | `scenarios/getting_started/hello_world/frames/mpc_frames_00000-00000.h5` |

By default, a scenario directory is expected to contain a `frames/` subdirectory.
Use `--frames-subdir` only when inspecting an imported or read-only layout that
keeps its frame set somewhere else under the scenario root. Normal Generator
output always uses the canonical `frames/` child:

```bash
orchav-inspect scenarios/my_scenario --frames-subdir custom_frames --frame 10
```

## Common Uses

List available frame indices without loading a frame:

```bash
orchav-inspect scenarios/getting_started/hello_world --list
```

Inspect a specific frame:

```bash
orchav-inspect scenarios/getting_started/hello_world --frame 0
```

Emit structured output for scripts or CI diagnostics:

```bash
orchav-inspect scenarios/getting_started/hello_world --frame 0 --json
```

Show full sensing product descriptors and target identities:

```bash
orchav-inspect scenarios/sensing/cube_radar/stationary --frame 63 --verbose
```

`--verbose` includes product identifiers, storage keys, shapes, axes,
representations and units. `--json` always includes the full structured
metadata, regardless of `--verbose`.

Show the scattering coefficients recorded for the scene materials:

```bash
orchav-inspect scenarios/generator/propagation_and_materials/scene_diffuse_scattering/itu_preset --materials
```

`--materials` reads the resolved coefficients from the generated frame
manifest. Add `--json` when a script needs the same information as structured
data.

Inspect coverage data:

```bash
orchav-inspect scenarios/generator/coverage/single_tx --coverage
```

Coverage inspection reports the coverage schema version, grid shape, heights,
TX/RX metadata, solver settings, available metrics, and value ranges.

## Relationship to validation

[Scenario validation](../reference/scenario_validation.md) checks authored inputs
before generation. Frame inspection checks saved output. The
[Hello World walkthrough](../../scenarios/getting_started/hello_world/README.md)
contains the complete generation and visualization sequence.

Up: [Shared Data Layer](README.md) | Related: [Statistics](statistics.md) | [Frame Data Reference](frame_reference.md)
