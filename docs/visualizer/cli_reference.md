# Visualizer CLI Reference

`orchav-visualizer` launches the desktop Visualizer, which is a
[consumer](../reference/glossary.md#consumer) of ORCHAV frames. Use this page
to choose a scenario, select a renderer, override the configured
[frame provider](../reference/glossary.md#frame-provider), or run a bounded
rendering or benchmark job. The command is equivalent to `python -m
visualizer`.

```bash
orchav-visualizer --scenario scenarios/getting_started/hello_world
```

The default `pygfx` renderer opens inside the main Visualizer window. When
launched without `--scenario`, the application can resume the newest valid
[workspace snapshot](workspace_snapshots.md). If no snapshot can be resumed,
use **File > Open Scenario...**.

## Interactive Launch Options

| Option | Default | Description |
|--------|---------|-------------|
| `--scenario PATH` | none | Scenario directory or `scenario.yaml` file. |
| `--data-mode {files,live_grpc,remote_hdf5}` | scenario YAML | Override the configured frame provider for this launch. Requires `--scenario`. |
| `--grpc-port PORT` | configured endpoint | Replace the configured client port for `live_grpc` or `remote_hdf5`. The configured host is retained. Requires `--scenario`. |
| `--renderer {pygfx,open3d}` | `pygfx` | Renderer backend. Use pygfx on macOS; both choices are available on Windows and Linux. |
| `--viewport-mode {auto,embedded,detached}` | `auto` | Select renderer hosting. Normal pygfx launches are embedded. Open3D uses its native window. |
| `--no-resume` | off | Skip automatic workspace restoration for this launch. |
| `--sensing-frames-only` | automatic, except with figures and 3D explicitly `off` | Require playback of only scene frames bound to saved sensing results. Available with ORCHAV Sensing; requires `--scenario` and local frame files. |
| `--sensing-figures {on-demand,summary,motion,prepare,off}` | saved loading choice | Choose initial sensing views and preparation for a local recording. See [loading choices](sensing_dashboard.md#loading-and-playback). |
| `--sensing-motion {all-ranges,saved-stft}` | `all-ranges` | With `--sensing-figures motion`, select the initial saved-RD overview or authored STFT. |
| `--sensing-3d {on,off}` | saved visibility; off for focused modes | Override initial **Show 3D sensing**. `off` skips 3D sensing preparation, including with figure preparation. Enable overlays later from the Sensing panel. |
| `--sensing-overview-cache-mb N` | `64` | Retained compact spectral overview budget in MiB. Affects recording-wide plot coverage. |
| `--sensing-history-cache-mb N` | `64` | Retained detailed detection/tracking history budget in MiB. |
| `--sensing-history-results N` | `4096` | Detailed-history result limit, independent of compact spectral coverage. |
| `--sensing-result-cache-mb N` | `512`; `2048` with `prepare` | RAM budget in MiB for individual figure requests. |
| `--sensing-result-cache-entries N` | `512`; `2048` with `prepare` | Maximum cached individual figure requests. |
| `--fusion-results PATH` | none | Open saved tracking or [multi-sensor fusion results](../sensing/detection_and_tracking.md#multi-sensor-fusion). |
| `--sensing-volume DIR` | none | Open a saved [sensing volume](volume_inspection.md#sensing-volumes) in pygfx. |
| `--enable-textures` | off | Enable bundled or scenario-provided albedo/detail texture maps. |
| `--disable-textures` | texture maps off | Explicitly disable texture maps when another setting enables them. |
| `--author` | off | Open the [Scenario Builder](../generator/scenario_builder.md), available by default with pygfx. |

See the [Shared Data Layer](../shared/README.md) for frame-provider behavior,
[Renderers](renderers.md) for the two rendering backends, and [Workspace
Snapshots](workspace_snapshots.md) for resume behavior.

Texture precedence and manual restoration are described in
[Workspace snapshots](workspace_snapshots.md#save-open-and-resume).

## Override The Frame Provider

CLI overrides remain in memory and are not written to `scenario.yaml` or a
workspace snapshot. Connect to a Live Generator on a non-default port:

```bash
orchav-visualizer --scenario path/to/scenario \
  --data-mode live_grpc --grpc-port 50053 --no-resume

```

Connect to a remote frame server on a non-default port:

```bash
orchav-visualizer --scenario path/to/scenario \
  --data-mode remote_hdf5 --grpc-port 50054 --no-resume
```

`--grpc-port` retains the endpoint host selected by scenario configuration. A
port override with effective `files` mode is rejected because no gRPC client is
involved.

## Load Only Sensing Frames

For a saved sensing recording, select its result-bound scene frames before
automatic preloading starts:

```bash
orchav-visualizer --scenario path/to/scenario \
  --data-mode files --sensing-frames-only
```

The Visualizer indexes saved result bindings and timestamps in the background,
then enables **Sensing → Play sensing frames only**. This includes results with
zero detections and handles irregular frame spacing; it does not assume a fixed
stride. **Real time** playback uses the recorded timestamps. Disable the checkbox
to restore the full timeline and preload all scene frames.

The option reduces frame decoding, cache population, and frame preparation.
Static geometry and textures still load normally; compressed HDF5 chunks may
contain additional rows even when only selected rows are decoded. Startup may
also read a frame for scene discovery or workspace restoration. Use `--no-resume`
to skip workspace restoration.

This option applies to the scenario opened at launch and requires saved results
with available scene bindings. Missing results produce a startup error without
falling back to full-frame preloading. Live sources, remote frame servers,
authoring, benchmarks, and batch rendering do not support this flag.

## Rendering And Capture Options

| Option | Default | Description |
|--------|---------|-------------|
| `--render-frames DIR` | none | Render every available frame as PNG to `DIR`, then exit. Requires `--scenario`. |
| `--layout-profile {auto,capture-workspace,capture-renderer}` | `auto` | Select ordinary placement or a deterministic recording layout when the display can fit it. |
| `--pygfx-present-method {screen,bitmap,auto}` | platform-aware | Override pygfx canvas presentation. Omission uses `screen` on Windows and `auto` elsewhere. |
| `--pygfx-adapter-name NAME` | environment/wgpu choice | Request a pygfx/wgpu adapter before renderer startup. |

For direct image production without the desktop application, use
[Headless Rendering](headless_rendering.md). For renderer selection and
graphics-environment guidance, use [Renderers](renderers.md) and
[Troubleshooting](../help/troubleshooting.md).

## Performance And Benchmark Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-performance` | off | Enable the high-memory cache profile for heavy playback. |
| `--pygfx-mpc-line-cache-mb MiB` | environment, otherwise `0` | Set the CPU expanded-line cache budget in MiB. Explicit `0` disables it, including under `--max-performance`. See [sizing and tradeoffs](caching.md#tuning-large-scenarios). |
| `--benchmark N` | `0` | Run benchmark mode for `N` frame transitions, write timing JSON, and exit. Requires `--scenario`. |
| `--benchmark-warmup W` | `5` | Warmup frames discarded before benchmark measurement. |
| `--benchmark-output PATH` | `benchmark_render_results.json` | Benchmark JSON destination. |
| `--benchmark-previsit-all-frames` | off | Visit all frames before timing to measure a warm-cache regime. |
| `--benchmark-present-mode {blocking,request}` | `blocking` | Wait for one renderer draw or only queue a draw request. |
| `--benchmark-state-json PATH` | off | Apply a benchmark-only `AppState` override. Requires `--benchmark`. |
| `--camera-debug` | off | Enable structured Follow/POV camera diagnostics. |

Use [Visualizer Caching](caching.md) for cache decisions and the
[Synthetic MPC Benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md)
for controlled workloads and result interpretation.

## Option Constraints

- `--render-frames` and `--benchmark` are mutually exclusive.
- Authoring cannot be combined with either mode.
- `--author --renderer open3d` is rejected before Qt starts.
- On macOS, any `--renderer open3d` launch is rejected before Qt starts. Use
  the default pygfx renderer.
- Explicit embedded mode is pygfx-only.
- Capture profiles and frame rendering require detached mode.
- Benchmark and batch-render launches do not restore interactive workspace
  snapshots.
- `--enable-textures` and `--disable-textures` are mutually exclusive.

## Batch Rendering

Batch-render a scenario:

```bash
orchav-visualizer --scenario scenarios/getting_started/hello_world \
  --render-frames tmp/orchav-frames
```

The Visualizer opens the scenario, renders every available frame to the
requested directory, then exits. For individual camera and frame selection,
use [Headless Rendering](headless_rendering.md).

## More Examples

Run a controlled renderer benchmark:

```bash
orchav-visualizer --renderer pygfx --benchmark 50 \
  --benchmark-warmup 10 \
  --benchmark-output tmp/pygfx-benchmark.json \
  --scenario scenarios/generator/mobility_and_orientation/actor_orientation
```

Start a clean desktop session with textures enabled:

```bash
orchav-visualizer --scenario path/to/scenario \
  --enable-textures --no-resume
```

Run `orchav-visualizer --help` for the exact options supported by the installed
version.

---

Up: [Visualizer](README.md) | Related: [Your First Visualizer Session](first_session.md) | [Renderers](renderers.md) | [Visualizer Scenario Defaults](scenario_defaults.md)
