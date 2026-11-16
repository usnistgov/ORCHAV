# ORCHAV Documentation

To run ORCHAV for the first time, [install it](getting_started/installation.md)
then follow the [Hello World quickstart](getting_started/quickstart.md) to generate
propagation paths and open them in the Visualizer. For a look at the application,
watch the [Visualizer demos](visualizer/demo_video.md).

## Choose a task

| Goal | Start here |
| --- | --- |
| Run a first simulation | [Installation](getting_started/installation.md), then [Quickstart](getting_started/quickstart.md). |
| Understand how ORCHAV is organized | [Generator](generator/README.md), [Shared Data Layer](shared/README.md), [Visualizer](visualizer/README.md) and [ORCHAV Sensing](sensing/README.md) explain each component and its place in the architecture. |
| Create or change a scenario | [Author in YAML or Python](generator/scenario_authoring.md), or use [Scenario Builder](generator/scenario_builder.md). |
| Inspect a channel in its scene | [First Visualizer session](visualizer/first_session.md), then [path inspection](visualizer/path_inspection.md) or [channel analysis](visualizer/analysis.md). |
| Study detection and tracking | Follow the [sensing examples](../scenarios/sensing/README.md), with [sensing guides](sensing/README.md) for concepts and settings. |
| Work with saved frames in Python | [Frame data reference](shared/frame_reference.md#minimal-hdf5-frame-provider-example) and [statistics helpers](shared/statistics.md). |
| Diagnose a problem | [Troubleshooting](help/troubleshooting.md). |

The [scenario catalog](../scenarios/README.md) owns complete runnable examples:
inputs, commands, expected results and output locations. The guides here explain
controls and choices that apply across those examples.

## Look up a detail

Use the [Scenario YAML Reference](reference/scenario_yaml.md) for fields and
defaults, [application configuration](reference/application_configuration.md)
for checkout settings, and [compatibility](reference/compatibility.md) for format
versions. The shared [glossary](reference/glossary.md) defines terms used across
components. [Propagation](reference/propagation.md) connects those terms to
path and coverage measurements.

## Reproduce or contribute

The [SIMULTECH 2026 guide](reproducibility/simultech_2026.md) maps the paper's
workflows to included examples. [Code examples](../examples/README.md) provide
small Python integrations.

Contributors can start with the [developer architecture](development/architecture.md),
[coding style](development/coding_style.md) and
[documentation style](development/documentation_style.md).
See [Contributing](../CONTRIBUTING.md) for checks and the
[changelog](../CHANGELOG.md) for release changes. Project credits are listed in
[Contributors](../CONTRIBUTORS.md).


---

Home: [Repository README](../README.md) | Begin: [Installation](getting_started/installation.md)
