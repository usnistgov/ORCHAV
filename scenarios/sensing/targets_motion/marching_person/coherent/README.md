# Marching Person: Coherent Configuration

The [marching-person example](../README.md) explains target shape, articulated
motion, input preparation, topology, settings and the paired comparison.
Use its shared prepared asset before running this mode.

To run it, see [Running](#running).

## Running

```bash
orchav-generator scenarios/sensing/targets_motion/marching_person/coherent
```

[scenario.yaml](scenario.yaml) writes `frames/` and `summary/` in this directory.
Its `maps` recording profile retains the numeric products used by the motion
comparison without retaining complex observations for reprocessing.

Open the saved motion history:

```bash
orchav-visualizer \
  --scenario scenarios/sensing/targets_motion/marching_person/coherent \
  --sensing-figures motion --sensing-motion saved-stft \
  --sensing-3d off --no-resume
```

Open **Sensing → Result graphs** for the saved STFT and range–time figures.
Choose **All ranges** for the range-summed RD comparison. Other figures and 3D
sensing load when requested. See [loading and playback](../../../../../docs/visualizer/sensing_dashboard.md#loading-and-playback)
for the available modes and memory limits.

## Browse Scenarios

Up: [Marching person](../README.md)

## Benchmark

| Invocation | Elapsed |
|---|---:|
| Generation | 439.78 s |
