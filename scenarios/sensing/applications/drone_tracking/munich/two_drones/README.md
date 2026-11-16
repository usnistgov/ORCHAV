# Two Drones: West Radar

Observe two drones rounding the Munich building from the west viewpoint.
DroneA flies at 10 m and DroneB returns at 14 m.
The [scenario.yaml](scenario.yaml) contains this sensor's scene, routes, and
sensing settings.

To run it, see [Running](#running).



## Scene Layout

The collocated transmitter and receiver are at (87 m, 182 m, 7 m)
in the built-in Munich scene. The 8 × 8 array supplies azimuth and elevation.
`DroneA` follows the YAML waypoints from [102, 179, 10] m to [120, 158, 10] m.
`DroneB` follows the YAML waypoints from [120, 158, 14] m to [102, 179, 14] m.
The [shared topology and fusion walkthrough](../../README.md#scene-layout)
places both radars and the obscuring building in the same world frame.

## Configuration Walkthrough

### Sensing settings

| Parameter | Value | Meaning |
|---|---|---|
| `sensing.preset` | coherent_hann | Coherent CPI, CFR/IFFT and Hann frequency/Doppler weighting; YAML fields override the [preset](../../../../../../docs/sensing/parameters.md#grouped-configuration). |
| `acquisition.bandwidth_hz` | 500 MHz | Frequency span governing the range-resolution scale. |
| `acquisition.frequency_samples` | 512 | Number of acquired frequency observations. |
| `acquisition.samples_per_cpi` / `sample_rate_hz` | 128 / 2000.0 Hz | Samples and rate defining one coherent observation interval. |
| `processing.clutter.coherent_mode` | windowed_mean | Subtract the Doppler-window-weighted slow-time mean to suppress stationary returns. |
| `recording.profile` | maps | Keep numeric maps and reports, without complex phase samples; see [profiles](../../../../../../docs/sensing/reprocessing.md#choose-a-profile). |
| `detection.mode` | beam_scan | Use receive-array directions when detecting returns. |
| `tracking.type` | cartesian3d | Cartesian position/velocity estimation. |

Acquisition, processing, detection, tracking and recording paths are relative
to `sensing`. A coherent processing interval (CPI) groups pulses for Doppler
estimation. See the [parameter reference](../../../../../../docs/sensing/parameters.md) for defaults and constraints.

## Running

From the repository root, generate this radar’s recording:

```bash
orchav-generator scenarios/sensing/applications/drone_tracking/munich/two_drones
```

Open the recording in the Visualizer:

```bash
orchav-visualizer --scenario scenarios/sensing/applications/drone_tracking/munich/two_drones --no-resume
```

Generation saves local HDF5 data in `frames/` and figures in `summary/sensing/`.
Open **Sensing → Result graphs** and use **Current result → Detections** and
**Tracking** to inspect this radar's observations.

## Browse Scenarios

Related: [South radar](south/README.md)

Continue with the [drone-fusion walkthrough](../../README.md#2-add-a-second-drone-returning-above-the-first) for the complete
comparison and commands to combine the radar recordings.

## Benchmark

| Invocation | Elapsed |
|---|---:|
| Generation | 873.84 s |
