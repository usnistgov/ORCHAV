# Remote HDF5 Frame Generation

[Scenarios](../../../../README.md) > [Visualizer](../../../README.md) >
[Data Modes](../../README.md) > [Remote HDF5 Playback](../README.md) > Frame Generation

This child scenario generates the fixed HDF5 frame set served by the parent
remote-HDF5 example. It writes reusable chunks to `frames/`. It does not start
the frame-file server.

Install the [`grpc` optional
extra](../../../../../docs/getting_started/installation.md#optional-extras) on
the machine that will run the frame-file server.

To run it, see [Running](#running).

## Generated Frame Set

| Element | Value |
| --- | --- |
| Scene | Built-in [Sionna RT](https://nvlabs.github.io/sionna/) `etoile` scene |
| `BaseStation` | Fixed transmitter at `(0.0 m, 0.0 m, 30.0 m)` |
| `MobileRx` | Linear route from `(60.0 m, -40.0 m, 1.5 m)` to `(-60.0 m, 40.0 m, 1.5 m)` |
| Timeline | 50 frames over 5 seconds |
| Output | Reusable HDF5 chunks and `frames_manifest.json` under `generation/frames/` |

The parent workflow serves this ordinary
[HDF5 frame set](../../../../../docs/shared/frame_reference.md#hdf5-frame-layout) without
regenerating it.

The child YAML omits `data` because local HDF5 generation under `frames/` is
the default file-output workflow.

## Running

From the repository root:

```bash
orchav-generator scenarios/visualizer/data_modes/remote_hdf5/generation
python -m generator.io.grpc.file_server \
    --frames-dir scenarios/visualizer/data_modes/remote_hdf5/generation --port 50052
```

Then open the parent remote client from the Visualizer machine:

```bash
orchav-visualizer --scenario scenarios/visualizer/data_modes/remote_hdf5
```

The Generator owns `generation/frames/`. The parent scenario remains a
read-only network client and does not replace those files.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../../../README.md) |
> [Visualizer](../../../README.md) | [Data Modes](../../README.md) |
> [Remote HDF5 Playback](../README.md) | Current: **Frame Generation**
