# Synthetic Frame Generation

The synthetic frame generator is an alternative
[frame producer](../reference/glossary.md#producer) for controlled test data.
It constructs `StandardMPCFrame` objects without running
[Sionna RT](https://nvlabs.github.io/sionna/), then writes them through the
normal frame-set output path as manifest-driven HDF5. This is useful for:

- **Benchmarking** the Visualizer pipeline with repeatable input
- **Stress testing** at controlled MPC densities
- **Reproducing** timing measurements with deterministic random data
- **Developing** new Visualizer features against controllable frame sizes

## Quick Start

```bash
# Default: 10 frames, 10k MPCs each
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py

# Open in the Visualizer
orchav-visualizer --scenario tmp/orchav_bench_10k
```

The script creates a self-contained scratch scenario with `scenario.yaml` and
HDF5 frames. The repository ignores `tmp/`, so this workflow leaves the
checked-in scenario template unchanged.

## Configuration

All parameters are set via command-line flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--frames` | 10 | Number of frames to generate |
| `--mpcs` | 10000 | Fixed MPC count per frame |
| `--mpcs-min` | unset | Minimum MPC count for variable-count generation |
| `--mpcs-max` | unset | Maximum MPC count for variable-count generation |
| `--output` | `tmp/orchav_bench_10k` | Output scenario directory |
| `--seed` | 42 | Random seed for reproducibility |
| `--compression` | `lzf` | HDF5 compression (`gzip`, `lzf`, or `none`) |
| `--chunk-size` | 50 | Maximum frames requested per HDF5 chunk |
| `--max-interactions`, `--max-order` | 2 | Maximum synthetic interaction points per MPC |

### Examples

```bash
# 100 frames, 100k MPCs — preload and animation testing
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
  --frames 100 --mpcs 100000 --max-interactions 2 \
  --output tmp/orchav_bench_100k

# 10 frames, 1M MPCs — higher-order stress test
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
  --mpcs 1000000 --max-interactions 5 \
  --output tmp/orchav_bench_1m

# Custom output directory (self-contained scenario)
python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
  --mpcs 50000 --output tmp/orchav_bench_50k
orchav-visualizer --scenario tmp/orchav_bench_50k
```

## What Gets Generated

Each HDF5 frame contains:

- **1 TX / 1 RX pair** at fixed positions in meters near the etoile scene
  center
- **Random interaction vertices** (up to `--max-interactions` per path) in a
  200 m x 200 m x 40 m box
- **Interaction types** (reflection, diffuse, refraction, diffraction)
- **Materials** (concrete, glass, metal, plasterboard)
- **Metrics** (delay in nanoseconds, path loss in dB, AoA/AoD angles in
  degrees)
- **Compact pair/path/bounce offsets and arrays** matching ordinary generated
  frames, with `NaN` only for unavailable metric values

The result follows the same `files` data mode as Sionna RT-generated output.
The Visualizer reads the saved HDF5 frame set through its normal frame pipeline
and renderer. Synthetic generation is not a separate data mode or delivery
route.

## Benchmarking

Run the Visualizer in benchmark mode to measure frame latency. For renderer
comparisons, keep the viewport size, display path, requested and resolved
presentation method, benchmark present mode, warmup, and cache regime fixed.
Because the implicit pygfx method is platform-aware, pass
`--pygfx-present-method` explicitly for controlled cross-platform comparisons.
VNC-backed results are affected by the virtual-display presentation path and
are not directly comparable with local-desktop results as a measure of GPU
performance.

```bash
orchav-visualizer \
    --renderer pygfx \
    --benchmark 20 \
    --benchmark-warmup 5 \
    --benchmark-previsit-all-frames \
    --benchmark-present-mode request \
    --scenario tmp/orchav_bench_10k \
    --benchmark-output tmp/orchav_bench_10k/benchmark.json
```

The `--benchmark N` flag processes `N` frame transitions and writes per-frame
timing to `--benchmark-output`. The summary includes
`avg_total_before_end_ms` for the frame pipeline before end-of-frame present,
`avg_total_ms` for the full benchmark frame including present, and derived
throughput fields such as `avg_total_fps_equiv`. The `_fps_equiv` fields are
reciprocals of measured durations, not observed display FPS. Use
`--benchmark-present-mode request` for non-blocking update/request timing and
`--benchmark-present-mode blocking` for present-inclusive timing. Use
`--benchmark-previsit-all-frames` when you want a steady-state warm-cache
measurement over every available frame.

## Generation Cost

The script uses vectorized NumPy operations without per-path Python loops.
Runtime and frame-set size depend on the requested interaction count,
compression, storage device, and machine, so measure the intended workload
locally.

---

Up: [Shared Data Layer](README.md) | Related: [Frame Data Reference](frame_reference.md) | [Synthetic MPC Benchmark](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md)
