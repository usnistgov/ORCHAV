# Visualizer Caching

The Visualizer is a [frame consumer](../reference/glossary.md#consumer). It keeps
frames and reusable rendering work in memory so playback does not repeat frame
reads and MPC processing. These caches are process-local. Packed HDF5 frame
sets remain the persistent data in file-backed modes.

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

## Cache Layers And Defaults

| Layer | Reused data | Desktop default |
|-------|-------------|-----------------|
| Raw frames | Frames loaded from the active provider | 32 frames initially. Grows automatically to fit the detected sequence during full preloading. |
| Prepared MPC data | Backend-neutral MPC geometry and metrics | 2,048 MB |
| Display data | Filtered and colorized frame views | Up to 1,000 entries |
| Coverage meshes | Prepared coverage-map surfaces | 50 entries |
| pygfx expanded MPC lines | Renderer-ready line endpoint and color arrays | Disabled (`0` MB) |

The raw-frame cache returns to its initial capacity when the scenario cache is
reset. Prepared-MPC and display caches evict older entries when their limits
are reached.

## Statistics Cache

Whole-scenario Statistics is derived from the active frame provider. For a local
or remote HDF5 provider with a durable frame-set identity, the Visualizer stores
numeric results in `.orchav-stats-<path-hash>.npz` beside the configured
`frames/` directory. This file is a rebuildable consumer-side artifact: the
Generator does not create it, and it is not part of the managed frame set.

The path hash identifies the normalized `frames/` location. A cache is reused
only while its recorded frame-set and generation identities still match. Live
Generator sessions have no durable frame-set identity, so their statistics are
computed without creating or reusing this file.

It is safe to delete a statistics cache while the Visualizer is closed. The
next cacheable scan recreates it. If the frame set changes while the Visualizer
remains open, choose **Compute Statistics** to force a fresh scan.

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
least 4,096 MB and enables a 1,024 MB pygfx line cache:

```bash
orchav-visualizer --scenario path/to/scenario --max-performance
```

Set only the pygfx line-cache budget with:

```bash
orchav-visualizer --renderer pygfx --scenario path/to/scenario \
  --pygfx-mpc-line-cache-mb 1024
```

The pygfx line cache is useful only when the selected budget can retain the
replay working set. An undersized cache can be slower because expanded arrays
are copied and evicted before reuse. The prepared-MPC budget can also be set
with the `VIZ_CANON_CACHE_MB` environment variable before launch.

See the [Visualizer CLI Reference](cli_reference.md) for these
options and the
[Synthetic MPC Benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md)
for a repeatable stress workload.

---

Up: [Visualizer](README.md) | Related: [Shared Data Layer](../shared/README.md) | [Synthetic MPC Benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md)
