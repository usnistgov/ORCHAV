# SIMULTECH 2026 Reproducibility

ORCHAV v0.1 exposes the SIMULTECH 2026 Visualizer workflows through the normal
Visualizer scenarios. The scenarios regenerate frame data, Visualizer inputs,
summary figures, and benchmark trend inputs from the same scenario entry points
users run for visual inspection.

Fresh outputs may not match paper figures or benchmark values exactly. Ray
tracing can involve stochastic sampling, and results can also change with
hardware, drivers, [Sionna RT](https://nvlabs.github.io/sionna/) versions,
renderer backends, and ORCHAV
implementation changes. Use these workflows to reproduce the scenario setup and
check qualitative trends.

## SIMULTECH 2026 Mapping

| Paper workflow | Included scenario |
| --- | --- |
| Munich interactive propagation analysis | [`mpc_inspection`](../../scenarios/visualizer/mpc_inspection/README.md) |
| Florence temporal statistics with coverage | [`statistics`](../../scenarios/visualizer/statistics/README.md) |
| Etoile multi-device trajectory workflow | [`multi_device_trajectory`](../../scenarios/visualizer/multi_device_trajectory/README.md) |
| Visualizer MPC-density benchmark trends | [`synthetic_mpc_benchmark`](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md#current-trend-measurement) |

## Output Locations

Each workflow writes generated data to ignored output directories. Scenario
runs use their local `frames/` and `summary/` directories. The benchmark
commands below use `tmp/`, which keeps their larger comparison outputs separate
from the scenario source.

The commands below assume ORCHAV is installed in the active Python environment.
See [Installation](../getting_started/installation.md) for setup details.

## Visualizer Workflows

Run the one-frame Munich workflow:

```bash
orchav-generator scenarios/visualizer/mpc_inspection
orchav-visualizer --scenario scenarios/visualizer/mpc_inspection
```

Run the Florence temporal statistics and coverage workflow:

```bash
orchav-generator scenarios/visualizer/statistics
orchav-visualizer --scenario scenarios/visualizer/statistics
```

Run the Etoile multi-device trajectory workflow:

```bash
python scenarios/visualizer/multi_device_trajectory/generate.py
orchav-visualizer --scenario scenarios/visualizer/multi_device_trajectory
```

Each generated scenario writes reusable HDF5 frame chunks under `frames/`.
Those frames can be opened in the Visualizer, inspected with `orchav-inspect`,
or loaded from Python analysis code.

## Benchmark Trends

The Visualizer stress tests use synthetic HDF5 frame data rather than a full
Sionna RT solve. This gives controlled MPC counts while keeping the same frame
contract used by generated scenarios. These results are trend checks for the
current implementation, not fixed paper values.

Generate 100, 1,000, 10,000, and 100,000 MPC frame sets into scratch
directories:

```bash
for n in 100 1000 10000 100000; do
  python scenarios/visualizer/synthetic_mpc_benchmark/generate.py \
    --frames 60 \
    --mpcs "$n" \
    --max-interactions 2 \
    --output "tmp/orchav_synthetic_mpcs_$n" \
    --chunk-size 60
done
```

Open one generated dataset:

```bash
orchav-visualizer --scenario tmp/orchav_synthetic_mpcs_100000
```

Expected trend checks:

- HDF5 file size and write time should increase with MPC count.
- Visualizer frame time should increase with MPC density.
- Sparse, medium, and dense workloads should preserve their relative ordering
  even when absolute timings change.
- Hardware-dependent FPS should be compared by trend and order of magnitude,
  not exact equality.

Home: [Documentation](../README.md) | Related: [Included Scenarios](../../scenarios/README.md) | [Scenario Validation](../reference/scenario_validation.md)
