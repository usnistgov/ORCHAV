# Default sensing-run timings

The [run records](../assets/sensing_performance/curriculum_20261109.json)
identify commands, configuration hashes, output locations and elapsed seconds.
The [main sensing page](../../scenarios/sensing/README.md#timing-results)
describes the machine and how to read these measurements. The run records
contain the compact timing evidence.

## Measurement scope

A monotonic wall clock surrounds each complete checkout-module invocation.
Generation includes startup, asset validation/loading, acquisition, configured
sensing processing, frame writing and normal summary figures. Supplied
motion/scene assets are prepared beforehand. Separate postprocessing, fusion,
comparison figures and optional Paris analysis are independent invocations.
Their times are not inferred components of generation.

Each native generator has one available GPU. Independent jobs can overlap.
Wall-grid workers also use one job per GPU. CPU processing can overlap those
jobs. Documentation rendering and validation also overlapped some acquisitions.
The run records give device assignments, concurrency, and the source and export
revision of each invocation. OS caches are not cleared. Marching person uses its documented CMU `08/08_03` preparation.
Paris uses the [historical walking-library alternative](../../scenarios/showcases/paris_opera_multitarget/PREPARATION.md)
with 80 epochs and 800,000 rays. Its executed YAML hash identifies that binding
separately from the included template.

## Native configurations

| Configuration | Generation |
|---|---:|
| [showcases/paris_opera_multitarget](../../scenarios/showcases/paris_opera_multitarget/README.md#benchmark) | 2201.55 s |
| [sensing/targets_motion/marching_person/coherent](../../scenarios/sensing/targets_motion/marching_person/coherent/README.md#benchmark) | 439.78 s |
| [sensing/applications/drone_tracking/munich](../../scenarios/sensing/applications/drone_tracking/munich/README.md#benchmark) | 676.62 s |
| [sensing/applications/drone_tracking/munich/south](../../scenarios/sensing/applications/drone_tracking/munich/south/README.md#benchmark) | 671.87 s |
| [sensing/applications/drone_tracking/munich/two_drones](../../scenarios/sensing/applications/drone_tracking/munich/two_drones/README.md#benchmark) | 873.84 s |
| [sensing/applications/drone_tracking/munich/two_drones/south](../../scenarios/sensing/applications/drone_tracking/munich/two_drones/south/README.md#benchmark) | 908.71 s |
| [sensing/applications/wall_detection_volume](../../scenarios/sensing/applications/wall_detection_volume/README.md#benchmark) | 5.74 s |
| [sensing/cube_radar/azimuth](../../scenarios/sensing/cube_radar/azimuth/README.md#benchmark) | 20.63 s |
| [sensing/cube_radar/moving](../../scenarios/sensing/cube_radar/moving/README.md#benchmark) | 23.47 s |
| [sensing/cube_radar/moving_tracking](../../scenarios/sensing/cube_radar/moving_tracking/README.md#benchmark) | 16.60 s |
| [sensing/cube_radar/moving_tracking_3d](../../scenarios/sensing/cube_radar/moving_tracking_3d/README.md#benchmark) | 63.52 s |
| [sensing/cube_radar/oscillating_coherent](../../scenarios/sensing/cube_radar/oscillating_coherent/README.md#benchmark) | 23.49 s |
| [sensing/cube_radar/oscillating_sequential](../../scenarios/sensing/cube_radar/oscillating_sequential/README.md#benchmark) | 167.37 s |
| [sensing/cube_radar/stationary](../../scenarios/sensing/cube_radar/stationary/README.md#benchmark) | 5.27 s |
| [sensing/cube_radar/two_moving](../../scenarios/sensing/cube_radar/two_moving/README.md#benchmark) | 16.54 s |
| [sensing/foundations/RCS/06_vehicle_multi](../../scenarios/sensing/foundations/RCS/06_vehicle_multi/README.md#benchmark) | 11.03 s |
| [sensing/foundations/RCS/08_human_model1](../../scenarios/sensing/foundations/RCS/08_human_model1/README.md#benchmark) | 10.67 s |
| [sensing/foundations/static_and_moving_cubes](../../scenarios/sensing/foundations/static_and_moving_cubes/README.md#benchmark) | 7.50 s |
| [sensing/cube_radar/azimuth/elevation](../../scenarios/sensing/cube_radar/azimuth/elevation/README.md#benchmark) | 27.73 s |
| [sensing/targets_motion/marching_person/rigid](../../scenarios/sensing/targets_motion/marching_person/rigid/README.md#benchmark) | 19.68 s |

## Additional commands

| Invocation | Kind | Elapsed |
|---|---|---:|
| `filtering` | processing | 5.76 s |
| `window_boxcar` | processing | 5.05 s |
| `window_hann` | processing | 4.32 s |
| `clutter` | processing | 1.81 s |
| `munich_fusion` | processing | 17.71 s |
| `munich_two_fusion` | processing | 18.93 s |
| `oscillation_figures` | figures | 33.71 s |
| `paris_review` | optional analysis | 5.44 s |
| `wall_volume` | generation | 301.89 s |
| `wall_collect` | processing | 2.82 s |
| `wall_report` | processing | 2.40 s |

## Measure another run

Measure one complete generation command with a monotonic wall clock, keeping
its normal summary outputs enabled. Record the command, configuration identity,
software, device and concurrent workloads with the elapsed time. Report
separate processing commands separately from generation.

Up: [Sensing](README.md) · Related: [Sensing curriculum](../../scenarios/sensing/README.md)
