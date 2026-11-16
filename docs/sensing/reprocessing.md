# Save and reprocess sensing observations

<a id="reprocess-a-sensing-recording"></a>

Saved sensing observations let you compare sensing processing without rerunning
ray tracing. First choose the data to retain during generation, then use
`orchav-sensing` to apply another recipe. Each processing invocation writes a
derived recording to a new output directory.

A recording profile selects the sensing data written to local HDF5 frames.
Retention does not change the acquisition, detector or tracker.

<a id="sensing-recording-profiles"></a>

## Choose a profile

| Profile | Retained data | Typical use |
|---|---|---|
| `results` | Detections, pre-tracking measurements, tracks, truth, identities, timing and recipe metadata | Scene playback, evaluation and compatible tracker replay |
| `maps` | Results plus applicable numeric range, Doppler, angle and motion display products | Interactive inspection and compatible detection replay |
| `reprocessing` | Maps plus one canonical complex observation representation | Supported window, clutter and angular-processing comparisons |
| `full` | Reprocessing data plus detector stages and tracking diagnostics | Detailed processing and decision inspection |

Use `maps` to inspect saved results, or `reprocessing` for comparisons that need
retained complex observations. Diagnostic additions can be enabled independently:

```yaml
sensing:
  recording:
    profile: maps
    detection_stages: true
    tracking_diagnostics: true
```

This is the retention choice used by the [sensing-panel guide](../visualizer/sensing_dashboard.md).
It exposes intermediate decisions without retaining input phase. `full` enables
both additions on top of `reprocessing`.

## What can be changed afterwards?

| Action | Required data or action |
|---|---|
| Inspect saved detections, tracks or evaluation | Results and their time-aligned truth/metadata |
| Change plot limits or colors | Corresponding numeric display product |
| Repeat CFAR, filtering or clustering | Compatible range–Doppler or angular power products |
| Repeat tracking | Pre-tracking measurements with times and sensor metadata |
| Change Doppler window or clutter processing | Compatible complex slow-time observations and acquisition recipe |
| Change CFR frequency weighting or range-IFFT size | Complex frequency observations on the acquired grid |
| Refine an array angle | Compatible complex samples for each receive element |
| Inspect detector or tracker decisions | Explicitly retained stage artifacts or tracking diagnostics |
| Change geometry, motion, material, carrier or acquired bandwidth | New acquisition |

A profile requests retention, but cannot supply an unconfigured processor or
missing sensor capability. A single receive element has no independent array
angle estimate. Consumers check the actual stored products, not only the
profile name. See [reprocessing](#run-processing) for commands and configuration
inheritance.

<a id="start-with-an-existing-example"></a>

## Run processing

Use [stationary-cube filtering](../../scenarios/sensing/cube_radar/stationary_filtering/README.md)
for a first comparison. Its three recipes produce one, five or no detections
from the same saved power map. The [windowing exercise](../../scenarios/sensing/foundations/windowing/README.md)
compares frequency weighting using retained complex observations. Both examples
provide the acquisition, exact settings, output locations and figures.

For another recording, inspect the available input and proposed recipe first:

```bash
orchav-sensing path/to/scenario/frames --dry-run
```

Run processing into a fresh directory:

```bash
orchav-sensing path/to/scenario/frames --output-dir analysis/comparison
```

Open the derived recording:

```bash
orchav-visualizer --scenario analysis/comparison
```

An adjacent source `scenario.yaml` allows the postprocessor to write an
inspection scenario referencing the original geometry and target assets.
Those assets must remain available. Inputs without a scenario still produce
frames, but no scene binding is invented.

## Configuration inheritance

Native v4 recordings supply physical axes, sample times and source identity.
Omitted range and Doppler windows inherit their recorded values, as do
coherent-CPI edge and DC guards. **Detector and tracker settings use CLI
defaults or explicit overrides**, rather than inheriting the entire native
recipe. To isolate one change, first set the comparison recipe to the intended
baseline. Use `orchav-sensing --help` for available options.

## Named recipes

A recipe file runs several configurations against the same acquisition:

```bash
orchav-sensing path/to/scenario/frames \
  --recipes path/to/recipes.json \
  --output-dir analysis/comparison
```

The [filtering example](../../scenarios/sensing/cube_radar/stationary_filtering/README.md)
provides a complete recipe file. Shared `defaults` apply to every named recipe,
which overrides only its listed values. Each recipe gets its own result
recording. `comparison.json` records the comparison, and each result's `postprocessing_config.json` records the applied settings.

## Rebuild spectra and angles

The [retention table](#what-can-be-changed-afterwards) above identifies the
required input. Common options are:

| Change | Options and scope |
|---|---|
| Doppler window or clutter processing | `--rebuild-rd` with the chosen `--window-doppler` and clutter options. Requires compatible complex slow-time samples. |
| CFR frequency weighting or range-IFFT length | `--frequency-window` and `--range-ifft-size`. Reuses the acquired frequency grid, sample times and noise. |
| Detection across angular beams | `--array-detection-mode beam_scan`. Requires compatible angular products. |
| Refine a beam angle | Add `--beam-angle-oversampling` from 1 to 16. Requires coherent per-element complex samples or a compatible reconstruction. |

Zero-padding or beam oversampling reduces grid quantization. Physical bandwidth
and aperture still set resolving power. Reprocessing cannot change the acquired
geometry or bandwidth.

### Keep fewer result windows

For a densely overlapping recording, `--streaming --result-hop-samples 64`
selects complete saved windows every 64 acquired samples. The hop must be a
multiple of the source hop and no larger than the analysis window. This
reduces result cadence without changing the acquired pulses or window length:

```bash
orchav-sensing path/to/scenario/frames \
  --output-dir analysis/motion-75pct --streaming --result-hop-samples 64 \
  --rebuild-rd --coherent-clutter-mode none --skip-detection \
  --recording-profile maps
```

`--recording-profile maps` requires this selected-window streaming rebuild.
The example requires retained complex frequency observations and disables
clutter removal and detection. It writes maps and motion displays without
retaining complex observations in the derived recording. Source identities
and the selected output cadence remain recorded with each result.

## Inspect derived results

Compare the same time interval, scene binding and display scale in the
[sensing panel](../visualizer/sensing_dashboard.md). `--save-stage-artifacts`
retains detector decisions, and `--save-tracking-diagnostics` retains tracker
association and lifecycle records. Figure selection controls presentation
without changing the retained scientific result.

[Evaluation](evaluation.md) explains how to compare counts
and errors. [Custom tracker integration](custom_tracker.md) replaces tracking
while retaining the ordinary data and evaluation interfaces.

## Complex observations

CFR/IFFT acquisition can retain frequency observations and reconstruct delay
samples using the saved transform. Delay-domain acquisition retains delay
samples. Both retain sample times, validity, element order, noise metadata and
the processing recipe. The canonical representation must preserve the actual
observation, including the stage where noise was introduced.

Power products cannot recover discarded phase. A saved detection retains its
originally computed refined angle even when complex samples are omitted, but
new refinement needs those samples. Detector-stage and tracker-decision
retention are independent of complex-input retention.

## Motion display products

`maps` retains applicable numeric motion plots with a fixed display recipe.
Their limits and colors can change interactively. Changing the STFT window,
range gate or processing recipe requires compatible complex inputs.
Independent coherent CPIs are not treated as one continuous coherent recording.

`recording.display` accepts the motion-analysis recipe fields. Without an
override, it uses the acquired range span, a Hann window of the configured
Doppler length, and both retained-mean and removed-mean spectra. The default
hop is a full window for coherent acquisition and one quarter-window for
sequential acquisition, rounded down to at least one sample. When sequential
results use a larger hop, the display hop is rounded up to a multiple of that
result hop. An explicit display hop must also be a multiple of the result hop.

For a 128-sample coherent window:

```yaml
sensing:
  recording:
    profile: maps
    display:
      window_samples: 128
      hop_samples: 128
      window: hann
      mean: both
```

Short histories and gaps are reported as unavailable support. They are not
padded into an invented observation. `results` does not accept display-recipe
overrides. Standalone [summary figures](../generator/generated_figures.md)
are optional exports of numeric products, not a separate recording profile.

## Compatibility

`recording.profile` cannot be combined with `recording.complex_observations`
or the flat `raw_payload_policy`. Existing explicit policies remain readable.
Without a profile, the declared output and complex-observation controls apply.
Changing YAML does not convert an existing recording.

Local frames carry sensing products and their descriptors. Ordinary propagation
live and remote gRPC transport does not carry the sensing extension. To inspect
sensing on another machine, copy or mount the recording and required scene
assets. See the [frame contract](../shared/frame_reference.md#sensing-payloads).

Up: [Sensing](README.md) · Related: [Parameters](parameters.md), [Evaluation](evaluation.md)
