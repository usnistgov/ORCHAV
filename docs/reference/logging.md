# Logging

The Generator and Visualizer use the same logging settings. Increase the log
level to diagnose a problem, or save messages to a file for later inspection.

## Configuration

The applications configure logging automatically. Set the verbosity through
scenario YAML or an environment override.

### `debug_level` in scenario.yaml

The primary way to control verbosity is the `debug_level` field at the root
of `scenario.yaml`:

```yaml
debug_level: WARNING    # Only warnings and errors
```

| Level | What you see |
|-------|-------------|
| `DEBUG` | Detailed diagnostic output for troubleshooting |
| `INFO` | Service initialization and pipeline-stage messages |
| `WARNING` | Only warnings and errors. Normal text progress remains separate. |
| `ERROR` | Only errors |

The Generator and Visualizer default to `WARNING` when `debug_level` is not
set.

### Environment variables

Environment variables override `debug_level` from the YAML:

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHAV_LOG_LEVEL` | from YAML or `WARNING` | Log level (overrides `scenario.yaml`) |
| `ORCHAV_LOG_FORMAT` | `compact` | Console output format (see below) |
| `ORCHAV_LOG_FILE` | *(none)* | Path to a rotating log file (10 MB, 5 backups) |

Override the level for one Generator run in Bash:

```bash
ORCHAV_LOG_LEVEL=DEBUG orchav-generator scenarios/getting_started/hello_world/
```

Save Visualizer log messages to a file:

```bash
ORCHAV_LOG_LEVEL=INFO ORCHAV_LOG_FILE=orchav.log \
  orchav-visualizer --scenario scenarios/getting_started/hello_world
```

In PowerShell, set the environment variable before launching the Generator:

```powershell
$env:ORCHAV_LOG_LEVEL = "DEBUG"
orchav-generator scenarios/getting_started/hello_world/
```

To save a Visualizer log instead, select its level and output path:

```powershell
$env:ORCHAV_LOG_LEVEL = "INFO"
$env:ORCHAV_LOG_FILE = "orchav.log"
orchav-visualizer --scenario scenarios/getting_started/hello_world/
```

These variables persist in that PowerShell window. Clear them after the run:

```powershell
Remove-Item Env:ORCHAV_LOG_LEVEL, Env:ORCHAV_LOG_FILE -ErrorAction SilentlyContinue
```

The explicit `INFO` level makes normal startup and service messages visible in
the example file. At the default `WARNING` level, a successful quiet run can
leave the file empty. ORCHAV creates the log file but does not create missing
parent directories. Create any nested directory in the configured path before
launching the command.

### Format profiles

Set the console profile via `ORCHAV_LOG_FORMAT`. Rotating file logs always use
the verbose profile so they retain timestamps, thread names, and source
locations independently of the console choice.

| Profile | Format | Use case |
|---------|--------|----------|
| `compact` | `LEVEL name: message` | Default with clean terminal output |
| `minimal` | `LEVEL message` | Minimal noise |
| `verbose` | `HH:MM:SS LEVEL [thread] name:line func: message` | Debugging with timestamps and source locations |
| `json` | Structured JSON | Log aggregation and automated parsing |

## Third-party logger suppression

Noisy loggers from gRPC, h5py, Matplotlib, and PIL are
automatically set to `WARNING` regardless of the configured level. This
prevents thousands of framework messages from flooding the output when
running at `DEBUG` or `INFO`.

## Generator Progress Output

With the normal text progress format, the Generator writes its progress bar
and a one-line completion summary to stderr independently of the log level:

```text
Done. 50 steps, 1 TX, 1 RX (1 pair) in 00:12. Output: <frame-set summary>
```

Machine-readable progress uses JSONL events instead of the text progress bar
and summary. Neither form is controlled by `debug_level`.

## Runtime level changes

The Visualizer's **Performance > Diagnostics** control shows the effective
project log level and changes it for the current process. Python integrations
can use `set_log_level()` from [`shared.logging`](../../shared/logging/__init__.py).
This updates the `orchav.*` loggers without changing scenario YAML.

Home: [Documentation](../README.md) | Related: [Application Configuration](application_configuration.md)
