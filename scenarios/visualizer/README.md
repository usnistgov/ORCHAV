# Visualizer Scenarios

[Scenarios](../README.md) > Visualizer

These runnable examples teach the ORCHAV Visualizer as a
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

Choose these by task. They are not additional required tutorial steps.

| Goal | Scenario |
|------|----------|
| Compare local, live, and remote frame delivery | [Data Modes](data_modes/README.md) |
| Inspect antenna-array response overlays | [Beamforming](beamforming/README.md) |
| Render local frames in Jupyter | [Notebook Mode](notebook_mode/README.md) |

## Performance And Diagnostics

These scenarios measure software behavior. Do not interpret their outputs as a
physical-channel study unless the individual page says otherwise.

| Goal | Scenario |
|------|----------|
| Generate controlled random MPC populations without [Sionna RT](https://nvlabs.github.io/sionna/) | [Synthetic MPC Benchmark](synthetic_mpc_benchmark/README.md) |

Synthetic workloads range from a quick smoke check to memory-intensive stress
runs. Begin with the default or quick-smoke command on the scenario page.

## Generated Data

Scenarios that run the Generator create reusable HDF5 frame sets under their
local `frames/` directory. The same frames can be reopened in the Visualizer,
inspected with `orchav-inspect`, or consumed through the
[Shared Data Layer](../../docs/shared/README.md).

The synthetic benchmark writes structurally valid frames without Sionna RT,
so use it for performance and cache testing rather than propagation
interpretation.

For workflows associated with a paper, use the dedicated
[Reproducibility documentation](../../docs/reproducibility/simultech_2026.md)
rather than treating the scenario catalog as a paper index.

## Browse Scenarios

> **Scenario path:** [All scenarios](../README.md) | Current: **Visualizer**
>
> Start the core path: [MPC Inspection](mpc_inspection/README.md)
