# Marching Person: Sequential Configuration

The [marching-person example](../README.md) explains target shape, articulated
motion, input preparation, topology, settings and the paired comparison.
Use its shared prepared asset before running this mode.

To run it, see [Running](#running).

## Running

```bash
orchav-generator scenarios/sensing/targets_motion/marching_person/sequential
```

[scenario.yaml](scenario.yaml) writes `frames/` and `summary/` in this directory.
Its `maps` recording profile retains the numeric products used by the motion
comparison without retaining complex observations for reprocessing.

Open the saved motion history:

```bash
orchav-visualizer \
  --scenario scenarios/sensing/targets_motion/marching_person/sequential \
  --sensing-figures motion --sensing-motion saved-stft \
  --sensing-3d off --no-resume
```

Open **Sensing → Result graphs** for the saved STFT and range–time figures.
This mode reads STFT products and saved profiles, with other figures and 3D
sensing deferred. Choose **All ranges**, or use `--sensing-motion all-ranges`,
for the range-summed RD comparison; its first load reads every saved RD map.
The recording contains 125 sensing results: 256-pulse windows with a 64-pulse
hop, giving 75% overlap and a 16 ms result cadence. All 8,192 pulses still use
their own body poses; the prepared SMPL asset therefore needs 8,192 samples at
4,000 Hz. Generation retains all scene steps.

Before a demonstration, prepare the selected Motion figures:

```bash
python -m visualizer.sensing_prepare scenarios/sensing/targets_motion/marching_person/sequential \
  --sensing-figures motion --sensing-motion saved-stft --sensing-3d off
```

To inspect all supported saved figures and 3D sensing, prepare the full display:

```bash
python -m visualizer.sensing_prepare scenarios/sensing/targets_motion/marching_person/sequential \
  --sensing-figures all --sensing-3d on

orchav-visualizer \
  --scenario scenarios/sensing/targets_motion/marching_person/sequential \
  --sensing-figures prepare --sensing-3d on --no-resume
```

The first preparation indexes the recording and reads the selected products;
subsequent launches reuse those figures. Detection and tracking remain disabled
in this example. See the
[shared running instructions](../README.md#running) for the coherent/sequential
tradeoff and full preparation, and
[loading and playback](../../../../../docs/visualizer/sensing_dashboard.md#loading-and-playback)
for cache limits.

## Browse Scenarios

Up: [Marching person](../README.md)
