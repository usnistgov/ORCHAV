# Data Mode Examples

[Scenarios](../../README.md) > [Visualizer](../README.md) > Data Modes

These three branches use the same Etoile scene, transmitter, receiver route,
and tracing settings. Only frame production, storage, and delivery change.

Use the [Shared Data Layer guide](../../../docs/shared/README.md) to decide
which mode fits a workflow. The pages here own only the runnable commands and
configuration for each example.

Local HDF5 Playback uses the base installation. Live Generator and Remote HDF5
Playback require the [`grpc` optional
extra](../../../docs/getting_started/installation.md#optional-extras).

## Choose An Example

| Mode | Example | Choose it to see |
|---|---|---|
| `files` | [Local HDF5 Playback](hdf5_files/README.md) | The Generator persists frames once. Local consumers reopen them later. |
| `live_grpc` | [Live Generator](live_grpc/README.md) | A running Generator produces requested frames and accepts supported session edits over gRPC. |
| `remote_hdf5` | [Remote HDF5 Playback](remote_hdf5/README.md) | A remote frame server reads a fixed HDF5 frame set for read-only playback over gRPC. |

Local What-if Preview is an optional `files`-mode feature, not another data
mode. See [Interactive Recomputation](../../../docs/visualizer/interactive_recomputation.md)
for how its local worker differs from Live Generator editing.

## Shared Scenario

| Element | Value |
|---|---|
| Scene | Built-in [Sionna RT](https://nvlabs.github.io/sionna/) `etoile` scene |
| TX | `BaseStation`, fixed at 30 m altitude |
| RX | `MobileRx`, 50-frame linear street-level route |
| Metrics | Delay, path loss, AoA, and AoD |

The Generator accepts `files` and `live_grpc`. The Visualizer additionally
accepts `remote_hdf5`, which reads an already generated frame set through the
remote frame server.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../README.md) | [Visualizer](../README.md) |
> Current: **Data Modes**
>
> Choose: [Local HDF5 Playback](hdf5_files/README.md) |
> [Live Generator](live_grpc/README.md) |
> [Remote HDF5 Playback](remote_hdf5/README.md)
