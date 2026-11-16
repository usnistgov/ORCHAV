# Visualizer Scenarios

These examples show how to use the ORCHAV Visualizer as a
[consumer](../../docs/reference/glossary.md#consumer) of generated or synthetic
frame data. Complete [Your First Visualizer
Session](../../docs/visualizer/first_session.md) with Hello World before starting
this catalog.

## Core Learning Path

Follow these four scenarios in order:

| Step | Scenario | What it adds | Generation work |
|---:|---|---|---|
| 1 | [MPC Inspection](mpc_inspection/README.md) | Core 3D path inspection, coloring, filters, and the MPC Explorer. | One static frame. |
| 2 | [Metrics Evolution](metrics_evolution/README.md) | Current-frame channel metrics during a receiver trajectory. | A moving receiver across many frames. |
| 3 | [Statistics](statistics/README.md) | Whole-scenario statistics and companion coverage data. | Many frames plus a coverage grid. |
| 4 | [Multi-Device Trajectory](multi_device_trajectory/README.md) | Multiple receivers, a moving target, orientations, and per-link inspection. | Multiple moving actors across many frames. |

## Optional Workflows

Choose the workflow you need; these examples and guided exercises can be followed independently.

| Goal | Scenario |
|---|---|
| Compare local, live, and remote frame delivery | [Data Modes](data_modes/README.md) |
| Inspect antenna-array response overlays | [Beamforming](beamforming/README.md) |
| Render local frames in Jupyter | [Notebook Mode](notebook_mode/README.md) |
| Select a saved range–Doppler cell and inspect its 3D range shell and associated paths | [Sensing selection guided exercise](sensing_selection/README.md), using runnable fixtures from [Visualizer experiments](../../examples/visualizer_experiments/README.md) |
| Compare Detailed, Paper, Glow, and procedural facades on a small local scene | [Building Appearance](building_appearance/README.md) |


## Performance And Diagnostics

Use the [Synthetic MPC Benchmark](synthetic_mpc_benchmark/README.md) to measure
Visualizer performance with controlled random paths, without ray tracing.
For developers checking the source-cell selection behavior of coverage and
sensing volume views, the [direct source-cell inspection fixture](volume_cell_inspection/README.md)
uses **synthetic data and a native Qt harness**; it is not a radar experiment.

For saved products and detector controls, follow the [sensing dashboard tour](../../docs/visualizer/sensing_dashboard.md). To generate a first sensing recording, start with the [stationary cube](../sensing/cube_radar/stationary/README.md).

## Browse Scenarios

Up: [All scenarios](../README.md) | Begin: [MPC Inspection](mpc_inspection/README.md)
