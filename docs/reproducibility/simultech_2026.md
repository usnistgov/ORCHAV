# SIMULTECH 2026 Reproducibility

Recreate the three propagation scenarios from the [SIMULTECH 2026 paper](https://doi.org/10.5220/0014911600004094), or measure how the current Visualizer handles different numbers of multipath components (MPCs). Ray sampling, ORCHAV and Sionna RT versions, drivers, and hardware can all affect the results.

## SIMULTECH 2026 Mapping

| Paper workflow | Included scenario |
| --- | --- |
| Munich interactive propagation analysis | [`mpc_inspection`](../../scenarios/visualizer/mpc_inspection/README.md) |
| Florence temporal statistics with coverage | [`statistics`](../../scenarios/visualizer/statistics/README.md) |
| Etoile multi-device trajectory workflow | [`multi_device_trajectory`](../../scenarios/visualizer/multi_device_trajectory/README.md) |
| Visualizer MPC-density benchmark trends | [`synthetic_mpc_benchmark`](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md#current-trend-measurement) |

Follow the linked scenario for its inputs, commands, expected views and output
locations. These examples use the current ORCHAV release. They let you revisit
the paper's workflows without claiming bitwise reproduction of its historical
software and hardware setup.

## Benchmark Trends

The benchmark uses synthetic frames to control the number of paths independently of the ray tracer. From a graphics-enabled session, run the same renderer and settings at four path counts:

```bash
python scripts/benchmarks/run_current_visualizer_benchmark_pack.py --renderers pygfx --mpc-counts 100 1000 10000 100000 --repeats 3 --output-root tmp/orchav_current_visualizer_bench
```

Open the timestamped run directory under `tmp/orchav_current_visualizer_bench/`. Compare frame timing across path counts in `aggregate_summary.csv`. Use `summary.csv` for individual runs and `summary.json` for their settings. The [synthetic benchmark guide](../../scenarios/visualizer/synthetic_mpc_benchmark/README.md#renderer-benchmarking) also shows how to inspect a dataset in the Visualizer.

Home: [Documentation](../README.md) | Related: [Included Scenarios](../../scenarios/README.md) | [Scenario Validation](../reference/scenario_validation.md)
