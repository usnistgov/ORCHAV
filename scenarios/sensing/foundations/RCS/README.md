# Native RCS Targets

Use Sionna RT 2.2's native radar cross section (RCS) targets in ORCHAV, visualize
their geometry and scattering points, and process their returns into range–Doppler
maps, detections and tracks. Two examples show the integration:

| Example | What to observe | Benchmark |
|---|---|---|
| [Human model 1](08_human_model1/README.md) | A single scattering point, prescribed RCS and rigid translation Doppler. | [Timing](08_human_model1/README.md#benchmark) |
| [Five-point car](06_vehicle_multi/README.md) | Several scattering points combine coherently; five points can produce fewer than five detections. | [Timing](06_vehicle_multi/README.md#benchmark) |

ORCHAV supports Sionna RT 2.2's constant-RCS model and all built-in TR 38.901
target presets: humans, small/large UAVs, single/multipoint vehicles and automated
guided vehicles. The [RCS configuration reference](../../../../docs/sensing/target_models.md#configuration)
lists valid model combinations, geometry, randomization and solver options,
with links to the upstream documentation. These two examples illustrate that
shared support; no separate example is needed for every preset.

## Shared Acquisition

Both examples use a monostatic radar at (0 m, 0 m, 1 m) in an empty scene.
Each target has an absorbing geometry proxy and native scattering points.
The proxy blocks propagation; the scattering model supplies its return.
The individual pages show the topology and exact trajectory.

| Parameter | Value | Meaning |
|---|---|---|
| `raytracing.carrier_frequency_hz` | 3.5 GHz | Carrier for propagation and Doppler conversion. |
| `sensing.acquisition.bandwidth_hz` | 200 MHz | Approximately 0.75 m range-resolution scale. |
| `sensing.acquisition.frequency_samples` | 128 | Frequency observations across the bandwidth. |
| `sensing.acquisition.samples_per_cpi` / `sample_rate_hz` | 64 / 500 Hz | Eight coherent processing intervals (CPIs), each 128 ms. |
| `sensing.preset` | `coherent_hann` | Finite-bandwidth range response and Hann weighting. |
| `sensing.processing.clutter.coherent_mode` | `unweighted_mean` | Remove the ordinary slow-time mean when rows vary; identical stationary rows are preserved. |
| `sensing.tracking.type` | `m_of_n` | Confirm reports after repeated range/velocity association. |
| `sensing.recording.profile` | `maps` | Retain maps, detections and tracks for inspection. |

Detection uses CFAR threshold scale 4, a 30 dB peak-relative filter and a
10–75 m range interval. Association gates are 2 m and 1 m/s. Native target
motion is rigid translation: these presets do not add limb or rotor motion.
See the [configuration reference](../../../../docs/sensing/target_models.md)
for the supported solver and acquisition conditions.

## Running

From the repository root, install the [native RCS extra](../../../../docs/getting_started/installation.md#native-rcs-targets),
then follow either example.

Generate the human-model recording:

```bash
orchav-generator scenarios/sensing/foundations/RCS/08_human_model1
```

Open its native target and sensing results:

```bash
orchav-visualizer --scenario scenarios/sensing/foundations/RCS/08_human_model1 --no-resume
```

Generation writes reusable `frames/` and standard SVG figures in
`summary/sensing/` beside the YAML. Select the target under **Objects** to
inspect its proxy and scattering point. Open **Sensing → Result graphs** for
maps, detector decisions and tracking history. A single receive element
measures range and radial velocity; it does not independently locate the
return in Cartesian space.


## Browse Scenarios

Up: [Sensing](../../README.md) | Related:
[Articulated mesh target](../../targets_motion/marching_person/README.md) ·
[RCS configuration](../../../../docs/sensing/target_models.md)
