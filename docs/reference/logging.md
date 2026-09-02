# Logging

ORCHAV uses Python's `logging` module with a centralized configuration in
`shared/logging/`. Both the Generator and the Visualizer share the same
logging infrastructure.

## Configuration

Logging is configured via `configure_logging()`, called automatically when
the Generator pipeline starts or when the Visualizer loads a scenario. You
rarely need to call it directly.

### `debug_level` in scenario.yaml

The primary way to control verbosity is the `debug_level` field at the root
of `scenario.yaml`:

```yaml
debug_level: WARNING    # Only warnings and errors
```

| Level | What you see |
|-------|-------------|
| `DEBUG` | Detailed diagnostic output for troubleshooting |
| `INFO` | Step-by-step progress, service initialization, pipeline stages |
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

Examples for Bash:

```bash
# Override log level for a single run
ORCHAV_LOG_LEVEL=DEBUG orchav-generator scenarios/getting_started/hello_world/

# Log to file
ORCHAV_LOG_LEVEL=INFO ORCHAV_LOG_FILE=orchav.log \
  orchav-visualizer --scenario myscenario
```

Examples for PowerShell:

```powershell
$env:ORCHAV_LOG_LEVEL = "DEBUG"
orchav-generator scenarios/getting_started/hello_world/

$env:ORCHAV_LOG_LEVEL = "INFO"
$env:ORCHAV_LOG_FILE = "orchav.log"
orchav-visualizer --scenario myscenario

Remove-Item Env:ORCHAV_LOG_LEVEL, Env:ORCHAV_LOG_FILE -ErrorAction SilentlyContinue
```

The explicit `INFO` level makes normal startup and service messages visible in
the example file. At the default `WARNING` level, a successful quiet run can
leave the file empty. ORCHAV creates the log file but does not create missing
parent directories; create any nested directory in the configured path before
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
project log level and can change it for the current process. Python
integrations can call `set_log_level()` directly:

```python
from shared.logging import set_log_level

set_log_level("DEBUG")   # Switch to debug output
set_log_level("WARNING") # Back to quiet mode
```

This updates all loggers in the `orchav.*` namespace immediately.

## API reference

The logging API is exported from `shared.logging`:

| Function | Description |
|----------|-------------|
| `configure_logging(level, format_profile, include_third_party)` | One-time setup (idempotent) |
| `resolve_log_level(level)` | Resolve a requested level after applying `ORCHAV_LOG_LEVEL` precedence |
| `get_current_log_level_name()` | Return the effective project level as a display-ready name |
| `get_logger(name)` | Get a logger, remapped to `orchav.*` namespace |
| `set_log_level(level)` | Update all ORCHAV loggers at runtime |

---

Home: [Documentation](../README.md) | Related: [Shared Data Layer](../shared/README.md) | [Application Configuration](application_configuration.md)
