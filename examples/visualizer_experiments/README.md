# Visualizer experiments with generated ray paths

Generate one of these four small scenes to inspect how path delay, arrival angle, and sensor geometry appear in the Visualizer. Each scene uses procedural metal reflectors and the ORCHAV Generator with Sionna RT.

| Scene | What to inspect |
|---|---|
| [arrival_courtyard/](arrival_courtyard/scenario.yaml) | Compare arrival times for the direct path and two wall reflections. |
| [angular_corridor/](angular_corridor/scenario.yaml) | Separate arrivals with the same delay by their arrival direction. |
| [bistatic_range_locus/](bistatic_range_locus/scenario.yaml) | Inspect an ellipsoidal range shell with the transmitter and receiver apart. |
| [monostatic_range_locus/](monostatic_range_locus/scenario.yaml) | Inspect a spherical range shell with the transmitter and receiver together. |

## Generate and open

Use a Python 3.12 environment with the development and sensing dependencies. From the repository root, run the commands below for the arrival courtyard. Substitute another folder from the table to explore that scene.

Generate the arrival-courtyard recording:

```bash
orchav-generator examples/visualizer_experiments/arrival_courtyard/
```

Open it to inspect the recorded paths:

```bash
orchav-visualizer --scenario examples/visualizer_experiments/arrival_courtyard/scenario.yaml
```

Open **Paths → Appearance → Path tools → Path Analysis…**.
The window contains **Arrival time** and **Delay–angle**. In the two sensing
examples, open **Sensing → Result graphs → Current result → Detections** and
select a heatmap cell. **Explain selected cell → Inspect delay-matched paths…**
opens the same path tools with a range-derived gate.
Use **All paths** and **Path count** for the comparisons below, so renderer
limits, filtering, and unequal path power do not change the intended counts.
The overlays interpret the loaded frame; they do not rerun propagation.

## Arrival courtyard: different route lengths

At frame **0**, choose the `main_tx` → `main_rx` pair. The sensors are at
(-4, 0, 1.5) m and (4, 0, 1.5) m. Walls at y = 3 m and y = -6 m produce:

- Direct path: 8 m, approximately 26.685 ns
- Near-wall reflection: 10 m, approximately 33.356 ns
- Far-wall reflection: sqrt(208) m = 14.422 m, approximately 48.107 ns

On **Arrival time**, drag the shaded gate, or enter its bounds. A 30–36 ns
gate selects only the near-wall path; 45–51 ns selects only the far-wall path.
The highlighted geometry should follow the selected reflection.
With **Path count**, each individual path accounts for one third of the pair's
eligible paths. Widen the gate and press **Play geometric clock** to see travel
along the actual segments.

The geometric clock advances segment arc length divided by c. It is a
geometry-only timing illustration, not animated RF fields or a second CIR.
Recorded delay and geometric travel time happen to agree for these simple
free-space/specular paths; the visualizer does not force them to agree.

Switch to `main_tx` → `offset_rx`, whose RX is at (4, 2, 1.5) m. Its three delays
are approximately 27.506, 29.835, and 53.786 ns. Pair isolation means the first
pair's rows must not be included. Return to the main pair before comparing its
three route lengths.

## Angular corridor: one delay, two directions

At frame **0**, the sensors remain 8 m apart. Side walls at y = ±3 m give two
10 m reflected routes, and the end wall at x = 10 m gives a 20 m route.
The direct path is also retained.

Open **Delay–angle**, choose **Arrival azimuth**, and leave angular bins at
10° and delay bins at 10 ns. The two side-wall paths occupy the same delay
interval around 33.356 ns but different angular cells. Click each occupied
cell to highlight its own path and world-direction guide.

The exported arrival azimuths are approximately 143.130° and 216.870°
(equivalently -143.130° after wrapping). The direct path is at 180°, and the
end-wall return is at 0° and 66.713 ns. Departure azimuth is a separate
observable. Do not read arrival and departure guides as interchangeable.

These are numerical bins of recorded rays, not inferred scatterer clusters
or angular-resolution claims. The direction guides have a fixed display
length, not a measured path length. Changing bin widths changes the grouping,
not the scene or the channel.

## Range geometry: ellipsoid and sphere

The two range examples have one TX/RX pair, one static wall at y = 8 m,
no direct path, and one ordinary reflected ray. Both use:

- Eight scene samples at 1 kHz, spanning 0–0.007 s
- One Sionna RT solve supplying eight coherent CIR samples
- `coherent_cpi`, 128 frequency bins over 300 MHz, and an eight-point Doppler FFT
- No injected noise or clutter removal
- Detector output and persisted detection-stage artifacts enabled for exact cell inspection

Leave the main scene at frame **0** (or navigate back to it). Open the
**Sensing → Result graphs** section. The dashboard finds
the completed result stored in frame **7** and displays it at its recorded
reference pose, frame 0 at time 0.0 s. Pinning is not needed. Frames 0–6 have
no embedded sensing result; frame 7 is the publication row, not the scene
pose to use for this experiment.

Open **Sensing → Result graphs → Current result → Detections** and click the heatmap at the saved return: range bin
**17**, Doppler bin **4** for the bistatic example, or range bin **16**, Doppler
bin **4** for the monostatic example. Indices are zero-based. The range shell
appears in the main scene when **Sensing → Show 3D sensing** and **Range shell**
are enabled. Use **Cutaway** or **Frame shell** in that section to inspect it.
**Clear selection** beside the dashboard's selection removes the shell.
Framing is optional; ordinary cell changes do not move the camera. See the [step-by-step walkthrough](../../scenarios/visualizer/sensing_selection/README.md)
for native screenshots and lifecycle checks.

The recorded coordinate is not, by itself, a validated target detection. The
saved detector-stage view can also inspect cells when those artifacts exist.

For `bistatic_range_locus`, TX/RX are at (-3, 0, 1.5) m and (3, 0, 1.5) m.
The specular bounce is (0, 8, 1.5) m. Total path length is
2 sqrt(73) = **17.088 m**, corresponding to approximately **56.999 ns**.
The strongest recorded RD cell is Doppler index **4**, range index **17**:
**0 Hz**, approximately **16.988 m** of total bistatic path length.
The direct-two-leg locus has the TX and RX as foci. Use the horizontal cutaway
to see its elliptical section through the sensor midpoint.

For `monostatic_range_locus`, both sensors are at (0, 0, 1.5) m.
The reflected path totals **16 m**, approximately **53.370 ns**, while the
monostatic range coordinate is **8 m**. Its strongest cell is Doppler index
**4**, range index **16**: **0 Hz**, approximately **7.994 m** displayed range.
The corresponding locus is a sphere; the cutaway is circular.

Discrete cell centers can differ from the geometric values. Range sample
spacing is approximately 0.9993 m for bistatic total length and 0.4997 m for
monostatic range. The displayed shell uses numerical cell support, when
available, and must not be interpreted as a localization error bound,
detection probability, or unique target position. These examples use a
single planar reflection and do not validate general multipath localization,
absolute amplitude, detection performance, or moving-target Doppler.

## Rebuild inputs and verify recordings

Use [build_scenarios.py](build_scenarios.py) to recreate the inputs outside the checkout. After generating all four recordings, run the [recording checks](validate_recordings.py).

Build the procedural scene definitions in a separate directory:

```bash
python examples/visualizer_experiments/build_scenarios.py --output /tmp/visual-experiments
```

Generate the arrival courtyard:

```bash
orchav-generator /tmp/visual-experiments/arrival_courtyard/
```

Generate the angular corridor:

```bash
orchav-generator /tmp/visual-experiments/angular_corridor/
```

Generate the bistatic range-locus recording:

```bash
orchav-generator /tmp/visual-experiments/bistatic_range_locus/
```

Generate the monostatic range-locus recording:

```bash
orchav-generator /tmp/visual-experiments/monostatic_range_locus/
```

Validate all four recordings:

```bash
python -m examples.visualizer_experiments.validate_recordings --input /tmp/visual-experiments
```

Use `--input examples/visualizer_experiments` instead when you generated in the
checked-in example directories. The verifier reads the actual manifest-driven
recordings through `Hdf5Provider`, checks each pair's geometric route lengths,
the corridor's distinct arrival angles, final-frame sensing publication, one
solve/eight coherent samples, and the static zero-Doppler peak within half a
range sample of the expected geometric coordinate. It prints the measured
values as JSON and never creates or modifies scientific data.

With the viewer dependencies installed, also check the actual dashboard-to-range
binding without opening a window:

```bash
QT_QPA_PLATFORM=offscreen python -m examples.visualizer_experiments.validate_recordings --input /tmp/visual-experiments --check-viewer
```

This verifies that scene step 0 resolves the result stored in frame 7, exposes
the persisted detector-stage inspection, creates an exact recorded-cell
selection, and produces two range-support surfaces with the expected
reference-pose sensor foci. It is not a GUI interaction test.

The input-only tests run without Sionna generation:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest --no-cov -q tests/visualizer/unit/test_experiment_scenarios.py
```

### CPU backend

Follow the [Linux CPU setup and arithmetic check](../../docs/getting_started/installation.md#linux-cpu-ray-tracing),
then run the four Generator commands and both recording checks above. These
examples exercise actual propagation and saved-result binding beyond the
installation preflight.

Use writable `DRJIT_CACHE_DIR`, `XDG_CACHE_HOME` and `MPLCONFIGDIR` locations
when the environment needs explicit cache paths. The earlier Sionna RT 2.0.1
CPU run is historical evidence, separate from the documented 2.2.0 setup.
Generated recordings and summaries remain local outputs.
