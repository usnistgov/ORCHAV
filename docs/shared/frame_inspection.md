# Frame Inspection

`orchav-inspect` prints a terminal summary of generated ORCHAV HDF5 frames and
coverage files. Use it after `orchav-generator` when you want to check data
contents without opening the desktop Visualizer.

```bash
orchav-inspect scenarios/getting_started/hello_world --frame 0
```

Typical output includes the source directory, available frame indices, selected
frame, TX/RX/target counts, TX-RX pair count, total MPC count, per-pair path
counts, interaction-type counts, exported metric fields, node positions, and
the HDF5 chunk that served the frame.

## Inputs

The command accepts any of these paths:

```bash
orchav-inspect scenarios/getting_started/hello_world
orchav-inspect scenarios/getting_started/hello_world/scenario.yaml
orchav-inspect scenarios/getting_started/hello_world/frames
orchav-inspect scenarios/getting_started/hello_world/frames/mpc_frames_00000-00000.h5
```

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
orchav-inspect scenarios/generator/coverage/single_tx/coverage/coverage_maps.h5
```

Coverage inspection reports the coverage schema version, grid shape, heights,
TX/RX metadata, solver settings, available metrics, and value ranges.

## Relationship to Validation

`orchav-validate` checks whether `scenario.yaml` is structurally valid before
generation. `orchav-inspect` reads generated HDF5 output after generation.
Validation is optional. Use it when editing YAML or debugging configuration
errors.

```bash
orchav-validate scenarios/getting_started/hello_world
```

The generated-frame workflow is:

```bash
orchav-generator scenarios/getting_started/hello_world/
orchav-inspect scenarios/getting_started/hello_world --frame 0
orchav-visualizer --scenario scenarios/getting_started/hello_world
```

---

Up: [Shared Data Layer](README.md) | Related: [Statistics](statistics.md) | [Frame Data Reference](frame_reference.md)
