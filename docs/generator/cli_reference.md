# Generator CLI Reference

`orchav-generator` runs YAML-authored scenarios through the ORCHAV Generator.
`python -m generator` provides the same command when working directly from a
source checkout.

```text
orchav-generator [OPTIONS] [SCENARIO]
```

`SCENARIO` must be either a directory containing `scenario.yaml` or that exact
`scenario.yaml` file. A Python-scripted scenario owns additional setup outside
the common YAML pipeline. Run its `generate.py` directly instead.

Run `orchav-generator` without a scenario to print the curated producer
catalog. The catalog gives one primary command for each entry:
`orchav-generator` for YAML scenarios and the `generate.py` driver for
Python-scripted scenarios.

## Options

| Option | Default | Behavior |
|---|---|---|
| `-h`, `--help` | -- | Print the command help and exit. |
| `--geometry-only` | off | For a `files` run, use minimum-budget frame ray tracing to check that the environment scene and actors load and that ordinary frame/topology output can be written. Coverage uses its own solver settings. |
| `--progress-format {text,jsonl}` | `text` | `text` keeps the interactive progress display and logs on stderr. `jsonl` writes versioned progress events to stdout and keeps logs on stderr for a supervising process. |
| `--data-mode {files,live_grpc}` | scenario value, or `files` when omitted from YAML | `files` writes the canonical scenario-local HDF5 frame set. `live_grpc` starts the on-demand Generator instead, and the process remains active until the server is stopped. |
| `--grpc-port PORT` | scenario port, or `50051` when unconfigured | Override the Live Generator port for this process. `PORT` must be from 1 through 65535, and the effective data mode must be `live_grpc`. |
| `--bind-host HOST` | `127.0.0.1` | Override the live server's listener interface. The effective data mode must be `live_grpc`. This does not change the host advertised to clients. |

CLI overrides apply to one process and do not edit `scenario.yaml`.
For Remote HDF5 Playback, first generate in `files` mode. Then start the
frame-file server described in the
[Frame Reference](../shared/frame_reference.md#remote-hdf5-server).

## Examples

Run a YAML scenario with its authored settings:

```bash
orchav-generator scenarios/getting_started/hello_world
```

Emit machine-readable progress while keeping logs separate:

```bash
orchav-generator scenarios/getting_started/hello_world \
  --progress-format jsonl
```

Start a Live Generator and choose its port for this run:

```bash
orchav-generator scenarios/visualizer/data_modes/live_grpc \
  --data-mode live_grpc --grpc-port 50052
```

Connect the Visualizer to the same port in another terminal:

```bash
orchav-visualizer \
  --scenario scenarios/visualizer/data_modes/live_grpc \
  --data-mode live_grpc --grpc-port 50052 --no-resume
```

A successful `files` run writes the HDF5 frame set under
`<scenario>/frames/`. See the
[HDF5 Frame Layout](../shared/frame_reference.md#hdf5-frame-layout) for its
on-disk organization. For Live Generator endpoint defaults and precedence, see
[Application Configuration](../reference/application_configuration.md).

---

Up: [Generator](README.md) | Related: [Scenario Authoring](scenario_authoring.md) | [Shared Data Layer](../shared/README.md)
