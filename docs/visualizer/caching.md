# Loading and Performance

The Visualizer is a [frame consumer](../reference/glossary.md#consumer). It keeps
frames and reusable rendering work in memory so playback does not repeat frame
reads and MPC processing. GPU objects stay in the current process. For file-backed
recordings, the source data remains in the HDF5 frame set.

## When A Scenario Opens

For a file-backed [frame provider](../reference/glossary.md#frame-provider), the first
frame is loaded for display, then a background preloader reads the available
sequence. Backend-neutral MPC geometry and metrics are prepared as frames
arrive, and display-specific data is warmed during idle UI time. Playback can
start before preloading finishes. Live Generator (gRPC) frames are requested
on demand instead of being fully preloaded.

Changing a filter or color mode may require new display data, but the raw
frames and prepared MPC data can still be reused. Loading another scenario
clears scenario-dependent cache entries.

Opening a scenario also loads scene meshes, target models and textures, then
prepares its appearance. A large scene can therefore open slowly even when its
recording contains few paths. Startup time, preload completion and playback
rate measure different work.

## Choose What To Prepare

| Goal | Choice | Tradeoff |
|---|---|---|
| Start inspecting promptly | Begin playback while preloading continues. | Revisiting a frame can become faster once its display data is cached. |
| Inspect saved sensing results | Keep **Play sensing frames only** enabled. | The timeline visits result-bound scene frames instead of every recorded scene sample. |
| Inspect sensing plots with fewer first-use waits | Choose **Prepare before playback**. | Opening takes longer and uses more cache storage. |
| Preserve main-view responsiveness | Close unused extra views and plots, or use **Main priority** for extra views. | Fewer views update simultaneously. |
| Repeat dense path playback | Consider the memory settings below. | Larger caches use more RAM and help only when reused data fits. |

Extra cameras, live plots, dense path overlays and translucent surfaces all add
display work. Compare settings on the same scene, camera and viewport size.
The [synthetic benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md)
provides a controlled workload when you need measured results.

## Cache Layers And Defaults

| Layer | Reused data | Desktop default |
|-------|-------------|-----------------|
| Raw frames | Frames loaded from the active provider | 32 frames initially. Grows automatically to fit the detected sequence during full preloading. |
| Prepared MPC data | Backend-neutral MPC geometry and metrics | 2,048 MB |
| Display data | Filtered and colorized frame views | Up to 1,000 entries |
| Coverage meshes | Prepared coverage-map surfaces | 50 entries |
| pygfx expanded MPC lines | Renderer-ready line endpoint and color arrays | Disabled (`0` MiB) |

The raw-frame cache returns to its initial capacity when the scenario cache is
reset. Prepared-MPC and display caches evict older entries when their limits
are reached.

## Sensing Display Cache

For a local sensing recording, the dashboard reuses saved result bindings,
detections, tracks and plot overviews. With the default on-demand strategy,
opening a new view can trigger its first read of a range–Doppler map or angle
spectrum. The `--sensing-figures summary` and `--sensing-figures motion` choices
open a focused view and defer 3D sensing preparation until requested. See
[loading choices](sensing_dashboard.md#loading-and-playback) for these modes,
full figure preparation and opening with figures off.

To prepare a recording on the machine where it will be shown, run the following
command after generating or copying the local frames.

```bash
python -m visualizer.sensing_prepare path/to/scenario
```

This uses the same preparation pipeline as `--sensing-figures prepare`: supported
saved maps, Motion families, Run summary and inspection data, with 3D sensing
included by default. Add `--sensing-3d off` to prepare figures without 3D sensing.
Complex Inputs and explicit signal reconstruction remain on demand. Preparation
reports any limited spectral coverage; a completed disk bundle does not mean
every individual result is resident in RAM.

To prepare only Motion, choose the same view that you will open:

```bash
python -m visualizer.sensing_prepare path/to/scenario \
  --sensing-figures motion --sensing-motion saved-stft

orchav-visualizer --scenario path/to/scenario \
  --sensing-figures motion --sensing-motion saved-stft --sensing-3d off
```

Use `--sensing-motion all-ranges` in both commands for range-summed RD instead.
Motion preparation includes the visible range–time profiles and defers 3D sensing
unless `--sensing-3d on` is requested. It shares the ordinary component cache,
so opening the matching view reuses the prepared spectra. Preparing another view
reuses compatible components and reads only the missing families. The first
preparation still reads the relevant saved products; it moves that work before
the demonstration. Scene assets and the recording index also take time to open.

The cache belongs to that machine and recording location. After replacing
source files, reopen the recording so the dashboard can rebuild outdated
entries. To force a rebuild, close the Visualizer and remove the cache under
`$XDG_CACHE_HOME/orchav/sensing` (by default `~/.cache/orchav/sensing`).
The source frames stay in the scenario directory.

**System → Performance → Diagnostics → Cache and memory…** shows current
reader activity and memory use. The ordinary component cache has a 512 MiB
disk budget. Full upfront preparation uses a separate disk bundle.

The following positive-integer options apply at the next recording opening.
Both visualizer and offline preparation commands accept them. Explicit values
take precedence over the defaults for the chosen loading mode.

| Retained data | Default | Option |
|---|---|---|
| Compact spectral overviews | 64 MiB | `--sensing-overview-cache-mb` |
| Detailed detection/tracking history | 64 MiB and 4,096 results | `--sensing-history-cache-mb`, `--sensing-history-results` |
| Individual figure requests | 512 MiB and 512 entries; 2,048 MiB and 2,048 entries with `prepare` | `--sensing-result-cache-mb`, `--sensing-result-cache-entries` |

These are separate retention budgets. Source indexes, current results, temporary
reads, plot buffers and renderer allocations also use memory. Increasing an
individual-result cache improves reuse during playback; spectral plot coverage
is controlled by the overview budget. Detailed history can reach either its byte
limit or its result-count limit, so the notice identifies the relevant control.

For example, allow 128 MiB of compact spectral history:

```bash
orchav-visualizer --scenario path/to/scenario \
  --sensing-figures motion --sensing-overview-cache-mb 128
```

Complete compatible components are reused across views and budgets when they fit.
An incomplete component is rebuilt when its limit is increased. The figures show
their loaded/available counts and time span, with an explicit notice when limited.
Changing a cache limit does not change the saved signal values or processing recipe.

## Statistics Cache

Whole-scenario Statistics is derived from the active frame provider. For a local
or remote HDF5 provider with a durable frame-set identity, the Visualizer stores
numeric results in `.orchav-stats-<path-hash>.npz` beside the configured
`frames/` directory. This file is a rebuildable consumer-side artifact. The
Generator does not create it, and it is not part of the managed frame set.

The path hash identifies the normalized `frames/` location. A cache is reused
only while its recorded frame-set identity and any supplied generation identity
still match. Providers that support streaming or live overrides do not create or
reuse this file, even if local frame files exist beside the scenario.

It is safe to delete a statistics cache while the Visualizer is closed. The
next cacheable scan recreates it. **Compute Statistics** rescans the recording
that is currently open. If you replace its frame files, reopen the recording
first, then compute statistics if no matching cache exists. An already opened
provider retains its original recording snapshot.

## Reusable Scene And Texture Assets

Scene and target assets use caches with a different lifecycle from frame
playback. They retain parsed scene payloads, generated UVs, decoded texture
pixels, target-animation frames, and renderer preparation work. A path-filter
change therefore does not evict a decoded building texture, and clearing frame
data does not discard scene assets.

Changed, corrupt, or stale asset-cache entries are ignored and rebuilt.
Renderer-native GPU objects remain scoped to the renderer and device that
created them.

Open **System > Performance** for two separate reset operations:

- **Clear Frames** removes raw frames, prepared MPC data, display views,
  preload bookkeeping, and renderer frame state.
- **Clear Assets** removes inactive target lookahead, decoded/native textures,
  prepared meshes, generated UVs, and reusable scene payloads. Visible target
  meshes remain pinned until they are no longer displayed.

Use these actions for diagnostics or after changing an external asset. They
are not required during ordinary playback.

## Tuning Large Scenarios

The defaults require no configuration for ordinary inspection. For a dense
multi-frame workload, `--max-performance` raises the prepared-MPC cache to at
least 4,096 MiB and enables a 1,024 MiB pygfx line cache:

```bash
orchav-visualizer --scenario path/to/scenario --max-performance
```

Set only the pygfx line-cache budget with:

```bash
orchav-visualizer --scenario path/to/scenario \
  --pygfx-mpc-line-cache-mb 1024
```

The pygfx line cache is useful only when the selected budget can retain the
replay working set. An undersized cache can be slower because expanded arrays
are copied and evicted before reuse. The CLI values override `ORCHAV_PYGFX_MPC_LINE_CACHE_MB`. An explicit zero disables
the line cache even with `--max-performance`. The prepared-MPC budget can also be set
with the `VIZ_CANON_CACHE_MB` environment variable before launch.

See the [Visualizer CLI Reference](cli_reference.md) for these
options and the
[Synthetic MPC Benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md)
for a repeatable stress workload.

---

Up: [Visualizer](README.md) | Related: [Shared Data Layer](../shared/README.md) | [Synthetic MPC Benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md)
