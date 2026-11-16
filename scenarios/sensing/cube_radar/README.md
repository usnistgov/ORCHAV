# Cube Radar Experiments

Start with one stationary reflector and add one sensing concept at a time.
These small scenes connect the configured acquisition to range, motion,
direction and persistent track identities.

| Step | Example | What it adds | Benchmark |
|---:|---|---|---|
| 1 | [Stationary cube](stationary/README.md) | Range estimation and detection. | [Stationary cube](stationary/README.md#benchmark) |
| 2 | [Filtering](stationary_filtering/README.md) | Reprocess one recording to compare detector decisions. | [Filtering](stationary_filtering/README.md#benchmark) |
| 3 | [Moving cube](moving/README.md) | Doppler and closing velocity. | [Moving cube](moving/README.md#benchmark) |
| 4 | [Oscillating cube](oscillating_comparison/README.md) | Compare coherent CPI and sequential STFT acquisition for changing radial motion. | [Oscillating cube](oscillating_comparison/README.md#benchmark) |
| 5 | [Radial tracking](moving_tracking/README.md) | Associate range and radial velocity over time. | [Radial tracking](moving_tracking/README.md#benchmark) |
| 6 | [Azimuth and elevation](azimuth/README.md) | Horizontal and planar arrays add directional observations. | [Angles](azimuth/README.md#benchmark) |
| 7 | [Cartesian tracking](moving_tracking_3d/README.md) | Associate and filter the planar array's spatial position observations. | [Cartesian tracking](moving_tracking_3d/README.md#benchmark) |
| 8 | [Two targets](two_moving/README.md) | Keep separate identities for two radial returns. | [Two targets](two_moving/README.md#benchmark) |

Radial tracking follows distance and motion toward or away from a sensor.
Cartesian tracking estimates positions in space and needs directional
observations, or other geometry that constrains those positions.

Additional exercises cover [range windowing](../foundations/windowing/README.md)
and [static clutter](../foundations/static_and_moving_cubes/README.md).
Each walkthrough includes a measured benchmark.

## Browse Scenarios

Up: [Sensing](../README.md) | Continue: [Marching person](../targets_motion/marching_person/README.md)
