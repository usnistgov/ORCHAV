# Sensing Scenarios

Begin with the [stationary cube](cube_radar/stationary/README.md), then follow
one progressive curriculum. Run commands from the repository root after
[installing ORCHAV](../../docs/getting_started/installation.md).

<a id="progressive-learning-path"></a>
<a id="core-tutorial-order"></a>
<a id="core-path"></a>
<a id="cube-radar-experiments"></a>

## Learning Path

| Example | What it demonstrates |
|---|---|
| [Cube radar](cube_radar/README.md) | Stationary → filtering → moving → oscillating (coherent/sequential) → radial tracking → azimuth/elevation → Cartesian tracking → two targets. |
| [Human mesh and marching person](targets_motion/marching_person/README.md) | Bundled rigid mesh, then articulated motion with coherent and sequential acquisition. Articulation requires a supplied motion asset. |
| [Native RCS targets](foundations/RCS/README.md) | Visualize and process native Sionna RT targets: human model 1 and a five-point car. |
| [Paris Opéra](../showcases/paris_opera_multitarget/README.md) | Five targets among buildings. Requires a supplied walking asset. |
| [Munich drone fusion](applications/drone_tracking/README.md) | Combine two viewpoints through building occlusion. |
| [Cube and wall volume](applications/wall_detection_volume/README.md) | Test independent target positions with and without an obstruction. |

<a id="choose-an-experiment"></a>
<a id="extended-targets-and-applications"></a>
<a id="continue-beyond-the-cubes"></a>
<a id="native-rcs-target-models"></a>
<a id="explore-more"></a>
<a id="optional-references"></a>

## Optional Exercises and References

- [Processing comparisons](foundations/README.md): frequency windows and static-clutter suppression on saved observations.
- [Sensing panel](../../docs/visualizer/sensing_dashboard.md): inspect maps, detector decisions and tracks.
- [Reusing an acquisition](../../docs/sensing/reprocessing.md#run-processing): which changes need retained complex observations and which need another acquisition.

Each walkthrough ends with a **Benchmark** section reporting a single measured run.

## Shared Volume Utilities

Volume experiments use shared modules for campaign preparation, acquisition,
progress reporting and storage: `volume_cli.py`, `volume_tools.py`,
`volume_progress.py` and `volume_storage.py`. They are helpers called by
scenario scripts, not separate commands for users to run.


## Timing results

Timing results give an idea of the cost on a machine with NVIDIA L40S GPUs and
Intel Xeon Platinum 8480+ CPUs, using ORCHAV 0.2.0, Python 3.12.16 and Sionna RT
2.2.0 (Mitsuba 3.9.1, Dr.Jit 1.5.0). Each is one complete invocation, including
startup, acquisition, configured sensing, frame writing and normal figures.
Separate processing commands have their own times. Independent jobs can run
concurrently, one per GPU; these observations are not performance guarantees.
See the [commands, settings and measurement details](../../docs/sensing/timings.md).

## Browse Scenarios

Up: [All scenarios](../README.md) | Related: [Sensing guide](../../docs/sensing/README.md)
