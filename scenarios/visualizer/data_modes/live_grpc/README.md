# Live Generator (gRPC)

[Scenarios](../../../README.md) > [Visualizer](../../README.md) >
[Data Modes](../README.md) > Live Generator (gRPC)

Live Generator (gRPC) connects the Visualizer to a separate running Generator
service and requests frames on demand. Use it when the Visualizer should send
tracing settings, TX/RX positions, or target transforms to that live session
for recomputation. `live_grpc` is the corresponding
[`data.mode`](../../../../docs/shared/README.md#choose-a-data-mode) and CLI value.

Unlike Local What-if Preview, this workflow keeps a Generator service running
and sends requests over gRPC. See
[Interactive Recomputation](../../../../docs/visualizer/interactive_recomputation.md)
for the shared persistence rules.

Install the [`grpc` optional
extra](../../../../docs/getting_started/installation.md#optional-extras) before
running this example.

To run it, see [Running](#running).

## Scene Setup

![2D scene layout](../../../../docs/assets/scenarios/scenarios_visualizer_data_modes_live_grpc_summary2d.png)

| Element | Configuration |
| --- | --- |
| `BaseStation` | Fixed transmitter above the Etoile scene |
| `MobileRx` | Receiver moving diagonally across the scene over 50 frames |
| `data.mode` | `live_grpc` (Live Generator over gRPC) |

## Scenario Configuration

```yaml
data:
  mode: live_grpc
```

With no endpoint override, both processes use
`grpc://localhost:50051`. To use another local port without editing YAML, pass
the same override to both processes:

```bash
orchav-generator SCENARIO --data-mode live_grpc --grpc-port 50053
orchav-visualizer --scenario SCENARIO --data-mode live_grpc --grpc-port 50053 --no-resume
```

For a trusted private network, also select a server listener explicitly with
`--bind-host HOST`. Otherwise, the Generator remains loopback-only. The client
port override retains the configured host. If the stream disconnects, reopen
the scenario after the Generator is available rather than expecting automatic
reconnection or request replay.

## Running

Open two terminals:

```bash
# Terminal 1: start the Generator server
orchav-generator scenarios/visualizer/data_modes/live_grpc

# Terminal 2: connect the Visualizer
orchav-visualizer --scenario scenarios/visualizer/data_modes/live_grpc
```

## What To Notice

- Open **System** and expand **Data Source**. It reports mode **Live gRPC** and
  shows the connection, buffer, and frame-timeline status.
- Scrub to a later frame and watch the latest-frame and buffer values update
  while the Generator remains running in the first terminal.
- No persistent `frames/` directory is written. Frames are computed, streamed,
  and cached in memory.
- One Visualizer controls the mutable live simulation at a time.

## What To Try

- Open **Edit** and apply a different ray-tracing quality preset.
- With the pygfx renderer, open **Edit**, enable **Live Actor Editing**, move
  the receiver with its gizmo, and release it to send one update to the
  Generator. The Object Management context menu also offers **Node
  Properties** for live actors.
- In **System** > **Data Source** > **Actions**, choose **Reset Live Frames**
  and request the frame again.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../../README.md) |
> [Visualizer](../../README.md) | [Data Modes](../README.md) | Current:
> **Live Generator (gRPC)**
>
> Up: [Data Modes](../README.md) | Compare:
> [Local HDF5 Playback](../hdf5_files/README.md) |
> [Remote HDF5 Playback](../remote_hdf5/README.md)
