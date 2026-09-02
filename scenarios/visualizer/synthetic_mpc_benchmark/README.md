# Synthetic MPC Benchmark

[Scenarios](../../README.md) > [Visualizer](../README.md) > Synthetic MPC Benchmark

This scenario uses a synthetic
[frame producer](../../../docs/reference/glossary.md#producer) for Visualizer
benchmarking without [Sionna RT](https://nvlabs.github.io/sionna/) ray tracing.
It constructs `StandardMPCFrame` objects and writes the normal manifest-driven
HDF5 frame set. The MPCs are random, but later playback uses the Shared Data Layer's
normal Local HDF5 Playback route and the same preparation, filtering,
rendering, preloading, and cache paths as ordinary generated output.

Use it for performance tests, not channel interpretation.

The template omits `raytracing`. Disabled ray tracing is the default, and
`generate.py` produces the synthetic frame contents directly. Synthetic
generation feeds the normal `files` data mode. It is not a separate data mode
or frame-delivery route.

To run it, see [Running](#running).

## Running

```bash
# Default: 10 frames, 10k MPCs per frame
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py

orchav-visualizer --scenario tmp/orchav_bench_10k
```

By default, the script creates a complete scratch scenario under
`tmp/orchav_bench_10k`,
including its own `scenario.yaml` and HDF5 chunks under `frames/`. The repository
ignores `tmp/`, so this default workflow leaves the checked-in scenario template
unchanged.

## Useful Workloads

```bash
# Quick smoke test
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
    --frames 2 --mpcs 10 --output tmp/orchav_bench_smoke

# Playback and preloading test
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
    --frames 100 --mpcs 100000 --output tmp/orchav_bench_100k

# Large-MPC stress test
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
    --frames 10 --mpcs 1000000 --max-interactions 5 \
    --output tmp/orchav_bench_1m

# Fixed-count comparison workload
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
    --frames 20 --mpcs 50000 --output tmp/orchav_bench_50k

# Varying-count playback workload, uniformly sampled between the bounds
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
    --frames 50 --mpcs-min 10000 --mpcs-max 100000 \
    --output tmp/orchav_bench_churn
```

## Renderer Benchmarking

Run the same generated frames against different renderers. Keep the viewport
size, display path, requested and resolved presentation method, benchmark
present mode, warmup, and cache regime fixed across comparisons. Because the
implicit pygfx method is platform-aware, pass `--pygfx-present-method`
explicitly for controlled cross-platform comparisons. VNC-backed results are
affected by the virtual-display presentation path and are not directly
comparable with local-desktop results as a measure of GPU performance.

```bash
python -m visualizer \
    --renderer pygfx \
    --scenario tmp/orchav_bench_10k \
    --benchmark 30 \
    --benchmark-warmup 5 \
    --benchmark-output tmp/orchav_visualizer_bench/pygfx.json \
    --no-resume
```

For the standard density pack:

```bash
python scripts/benchmarks/run_current_visualizer_benchmark_pack.py \
  --mpc-counts 100 1000 10000 100000 \
  --repeats 3 \
  --present-modes request blocking \
  --output-root tmp/orchav_current_visualizer_bench
```

The default renderer set is pygfx and Open3D/Filament on Windows and Linux,
and pygfx alone on macOS. On Apple Silicon, pygfx/wgpu uses Metal, Apple's
native graphics API. An explicit Open3D request is rejected on macOS. The
command uses the current local desktop. On Linux under X11 or VNC, add
`--display :1 --qt-platform xcb` only when those values match the active
session. Those two overrides are not needed for an ordinary local desktop run.
On Linux VNC, Open3D follows the display's GL path and may measure Mesa
llvmpipe. Run Open3D directly; Open3D through VirtualGL was not included in
v0.1 testing. Keep software Open3D numbers as functional or
environment-specific trend evidence, not L40S or V100 performance. Prefer
pygfx and record its wgpu adapter for GPU-backed VNC performance measurements.

## What To Test

- During manual playback, open **System** and expand **Performance**. Compare
  **Update work**, **Work average**, and **Renderer submit** as MPC density
  increases.
- In **Paths**, enable **Limit rendered MPCs** and change **Max MPCs** to
  compare Top-K drawing with the complete frame still loaded.
- Use the **Cache** and **Preload Status** fields under **Performance** to
  distinguish initial loading from warm playback.
- Use benchmark JSON for quantitative renderer and cache comparisons. Measure
  bounce-marker cost separately from line rendering.

## Current Trend Measurement

Use this scenario to measure renderer trends for your hardware, driver,
display, and dependency stack. The expected trend is monotonic growth in file
size, load cost, and visible-line rendering pressure as MPC density increases.

Treat benchmark numbers as environment-specific. Keep raw JSON, CSV summaries,
plots, and scratch-generated benchmark scenarios under the ignored `tmp/`
directory or another untracked scratch location unless a publication workflow
explicitly requires otherwise.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../README.md) | [Visualizer](../README.md) |
> Current: **Synthetic MPC Benchmark**
>
> Up: [Visualizer Scenarios](../README.md)
