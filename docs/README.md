# ORCHAV Documentation

The repository [README](../README.md) introduces ORCHAV and its three main
components. Use this documentation home to follow the first-run path or go
directly to the workflow, concept, or reference you need.

## Start Here

Follow one path for a first successful run:

1. [Install ORCHAV](getting_started/installation.md).
2. [Generate, inspect, and visualize Hello World](getting_started/quickstart.md).
3. [Review the Hello World inputs and outputs](../scenarios/getting_started/hello_world/README.md).
4. [Learn the Visualizer with the generated Hello World frame](visualizer/first_session.md).
5. [Choose another included scenario](../scenarios/README.md).

The quickstart introduces scenario validation before generation. When a command
fails, go directly to [Troubleshooting](help/troubleshooting.md).

## Work With ORCHAV

### Generate Frames

The [Generator guide](generator/README.md) is the entry point for frame
production.

| Goal | Guide |
|------|-------|
| Create or modify a scenario | [Scenario Authoring](generator/scenario_authoring.md) |
| Configure ray tracing, antennas, materials, path filtering, or coverage | [Generator Configuration](generator/configuration.md) |
| Choose mobility, orientation, or groups | [Mobility and Orientation](generator/mobility_and_orientation.md) |
| Configure standalone diagnostic figures | [Generated Figures](generator/generated_figures.md) |
| Automate or override a run | [Generator CLI Reference](generator/cli_reference.md) |

### Visualize And Explore

The [Visualizer guide](visualizer/README.md) is the entry point for interactive
frame consumption.

| Goal | Guide |
|------|-------|
| Choose initial camera, coloring, visibility, panels, or node markers | [Visualizer Scenario Defaults](visualizer/scenario_defaults.md) |
| Filter paths or compare metrics and statistics | [Visual Analysis](visualizer/analysis.md) |
| Recompute the displayed frame interactively | [Interactive Recomputation](visualizer/interactive_recomputation.md) |
| Inspect individual MPCs | [MPC Explorer](visualizer/mpc_explorer.md) |
| Choose a renderer | [Renderers](visualizer/renderers.md) |
| Visually author supported scenario fields | [Scenario Builder](generator/scenario_builder.md) |
| Render images without the desktop application | [Headless Rendering](visualizer/headless_rendering.md) |
| Render local frames in Jupyter | [Notebook Visualization](visualizer/notebooks.md) |

### Store, Transport, And Analyze Frames

The [Shared Data Layer guide](shared/README.md) explains the contract between
frame [producers](reference/glossary.md#producer) and
[consumers](reference/glossary.md#consumer). It also helps you choose Local
HDF5 Playback, Live Generator, or Remote HDF5 Playback.

| Goal | Guide |
|------|-------|
| Understand the frame contract, HDF5 layout, selected-field reads, or how frames are retrieved | [Frame Data Reference](shared/frame_reference.md) |
| Inspect generated HDF5 frames | [Frame Inspection](shared/frame_inspection.md) |
| Load local HDF5 frames from Python | [Frame Data Reference](shared/frame_reference.md#minimal-hdf5-frame-provider-example) |
| Compute channel statistics | [Statistics](shared/statistics.md) |
| Create repeatable synthetic frames | [Synthetic Frame Generation](shared/synthetic_frames.md) |
| Import MPCs from external software | [External Ray-Tracer Import](shared/frame_reference.md#external-ray-tracer-import) |

## Concepts

- [Propagation Concepts](concepts/propagation.md)
- [Visualizer Caching](visualizer/caching.md)

## Reference

- [Glossary](reference/glossary.md)
- [Scenario YAML Reference](reference/scenario_yaml.md)
- [Scenario Validation](reference/scenario_validation.md)
- [Versions and Compatibility](reference/compatibility.md)
- [Application Configuration](reference/application_configuration.md)
- [Logging](reference/logging.md)

## Help, Reproducibility, And Development

- [Troubleshooting](help/troubleshooting.md)
- [SIMULTECH 2026 Reproducibility](reproducibility/simultech_2026.md)
- [Developer Architecture](development/architecture.md)
- [Development Coding Style](development/coding_style.md)
- [Contributing](../CONTRIBUTING.md)
- [Test Suite](../tests/README.md)
- [Project Changelog](../CHANGELOG.md)
- [Contributors](../CONTRIBUTORS.md)
- [Examples](../examples/README.md)
- [Included Human-Walking Target](../libraries/targets/nist_human_walking/README.md)
- [Texture Library](../libraries/textures/README.md)

Project licensing is defined by [LICENSE](../LICENSE), [NOTICE](../NOTICE), the
[NIST software disclaimer](../NIST_SOFTWARE_DISCLAIMER.md), and
[Third-Party Notices](../THIRD_PARTY_NOTICES.md).

---

Home: [Repository README](../README.md) | Begin: [Installation](getting_started/installation.md)
