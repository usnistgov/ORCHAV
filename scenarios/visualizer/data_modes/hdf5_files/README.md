# Local HDF5 Playback

[Scenarios](../../../README.md) > [Visualizer](../../README.md) >
[Data Modes](../README.md) > Local HDF5 Playback

Generate frames once, then open the same local `frames/` directory repeatedly
in the Visualizer, inspect it with ORCHAV tools, or copy it to another checkout
or host when needed.

To run it, see [Running](#running).

## Scene Setup

![2D scene layout](../../../../docs/assets/scenarios/scenarios_visualizer_data_modes_hdf5_files_summary2d.png)

| Element | Configuration |
| --- | --- |
| `BaseStation` | Fixed transmitter above the Etoile scene |
| `MobileRx` | Receiver moving diagonally across the scene over 50 frames |
| `data.mode` | `files` (the default Local HDF5 Playback mode) |

## Scenario Configuration

`scenario.yaml` omits `data` because the `files` data-mode value is the default.
The resolved settings use HDF5, the scenario's `frames/` directory, and the
standard generated chunk names. Add a `data` block only when selecting Live
Generator or Remote HDF5 Playback, or when a read-only workflow needs a custom
frame location.

See the [Shared Data Layer guide](../../../../docs/shared/README.md) for the
comparison with `live_grpc` and `remote_hdf5`.

## Running

```bash
orchav-generator scenarios/visualizer/data_modes/hdf5_files
orchav-visualizer --scenario scenarios/visualizer/data_modes/hdf5_files
```

## What To Notice

- Open **System** and expand **Data Source**. It reports mode **File**, the
  local HDF5 path and chunks, and a total of 50 frames.
- Scrub from frame 0 to frame 49 and watch `MobileRx` move across the fixed
  frame sequence.
- Close the Visualizer and run the Visualizer command again. The saved frames
  reopen without keeping the Generator or
  [Sionna RT](https://nvlabs.github.io/sionna/) running.
- The Generator writes `frames/mpc_frames_00000-00049.h5` and the authoritative
  `frames/frames_manifest.json`.
- The same frames can be inspected with `orchav-inspect` or loaded from Python.

## Local What-if Preview

With pygfx and local Sionna RT dependencies, this example can use **Local
What-if Preview** for a temporary current-frame actor or solver change. Follow
[Interactive Recomputation](../../../../docs/visualizer/interactive_recomputation.md#local-what-if-preview)
for the controls and persistence boundary.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../../README.md) |
> [Visualizer](../../README.md) | [Data Modes](../README.md) | Current:
> **Local HDF5 Playback**
>
> Up: [Data Modes](../README.md) | Compare:
> [Live Generator](../live_grpc/README.md) |
> [Remote HDF5 Playback](../remote_hdf5/README.md)
