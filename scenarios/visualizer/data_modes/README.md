# Data Mode Examples

Compare three ways to load the same moving-receiver scenario: saved local
frames, frames computed live, or saved frames served over a network.

To run it, see [Running](#running).

## Choose A Mode

| Mode | Use it to |
|---|---|
| [Local HDF5 Playback](hdf5_files/README.md) (`files`) | Generate once and reopen saved frames. |
| [Live Generator](live_grpc/README.md) (`live_grpc`) | Request frames from a running Generator and try temporary edits. |
| [Remote HDF5 Playback](remote_hdf5/README.md) (`remote_hdf5`) | View saved frames while their HDF5 files stay on a server. |

All three use the built-in [Sionna RT](https://nvlabs.github.io/sionna/)
Etoile scene. `BaseStation` stays 30 m above the scene while `MobileRx` follows
a street-level route over 50 frames and 5 s.

## Running

Run the commands from the repository root. Choose one of the following setups.
The live and remote setups also need the gRPC extra:

```bash
python -m pip install -e ".[grpc]"
```

### Local frames

Generate local HDF5 frames:

```bash
orchav-generator scenarios/visualizer/data_modes/hdf5_files
```

Open the local recording:

```bash
orchav-visualizer --scenario scenarios/visualizer/data_modes/hdf5_files --no-resume
```

The frames are saved under this scenario's `frames/` directory. On later
visits, only the Visualizer command is needed.

### Live Generator

In the first terminal, start the Generator and leave it running:

```bash
orchav-generator scenarios/visualizer/data_modes/live_grpc
```

In a second terminal, connect the Visualizer:

```bash
orchav-visualizer --scenario scenarios/visualizer/data_modes/live_grpc --no-resume
```

The Generator computes requested frames and keeps them in memory. It does not
save a frame set. The [live walkthrough](live_grpc/README.md#try-a-live-edit)
shows how to move the receiver and recompute paths.

### Remote saved frames

First generate the recording that the server will read:

```bash
orchav-generator scenarios/visualizer/data_modes/remote_hdf5/generation
```

In the same terminal, start the frame server and leave it running:

```bash
python -m generator.io.grpc.file_server --frames-dir scenarios/visualizer/data_modes/remote_hdf5/generation --port 50052
```

In a second terminal, connect the Visualizer:

```bash
orchav-visualizer --scenario scenarios/visualizer/data_modes/remote_hdf5 --no-resume
```

This first run puts both processes on your machine. The server reads
`generation/frames/`; playback does not rerun ray tracing. See the
[remote walkthrough](remote_hdf5/README.md#connect-to-another-machine) to use
another machine.

## What To Inspect

1. Open **System → Data Source**. Check that the mode matches your setup:
   **File**, **Live gRPC**, or **Remote HDF5**.
2. Play or scrub the timeline and watch `MobileRx` move through the scene.
3. For a live or remote run, keep the server terminal open during playback.
   When finished, close the Visualizer and stop the server with **Ctrl+C**.

For storage and transport details, see the
[Shared Data Layer guide](../../../docs/shared/README.md#choose-a-data-mode).

## Browse Scenarios

Up: [Visualizer](../README.md) ·
[Local HDF5 Playback](hdf5_files/README.md) ·
[Live Generator](live_grpc/README.md) ·
[Remote HDF5 Playback](remote_hdf5/README.md)
