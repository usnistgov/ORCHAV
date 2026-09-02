# Metrics Evolution

[Scenarios](../../README.md) > [Visualizer](../README.md) > Metrics Evolution

This compact Visualizer scenario makes per-frame channel changes easy to see.
One receiver moves directly toward a fixed transmitter over 100 frames above a
flat ground plane. [Rough-surface scattering from the
ground](../../../docs/concepts/propagation.md#diffuse-scattering) provides many
paths without the geometric complexity of an urban scene.

To run it, see [Running](#running).

## Scene Layout

All positions use meters in the local Cartesian scene frame.

| Element | Position or motion | Role |
| --- | --- | --- |
| `TX` | Fixed at `[-20, 0, 4]` | Transmitter |
| `ApproachingRX` | Linear motion from `[25, 0, 1.5]` to `[-15, 0, 1.5]` | Receiver approaching the transmitter |
| Ground | 60 m by 50 m plane at `z = 0` | Concrete radio material with an asphalt visual appearance |

The 20-second timeline contains 100 frames, including both motion endpoints.
The receiver moves at 2 m/s, and the direct TX/RX distance falls from about
45.1 m to 5.6 m. The corresponding free-space propagation delay is therefore
about 150 ns in frame 0 and 19 ns in frame 99.

## Configuration Walkthrough

The receiver uses the shared linear mobility model:

```yaml
timeline:
  steps: 100
  duration_s: 20.0

actors:
  rx:
    - name: ApproachingRX
      mobility:
        type: linear
        start_m: [25.0, 0.0, 1.5]
        end_m: [-15.0, 0.0, 1.5]
```

The local `ground/ground_60x50.xml` scene names its material
`mat-ground_asphalt`, which ORCHAV exposes as `ground_asphalt`. Its radio model
uses concrete properties. The custom quality profile inherits the medium
profile, where diffuse reflection is enabled. The scenario overrides only its
depth and ray budgets, then assigns a nonzero material scattering coefficient:

```yaml
raytracing:
  quality:
    preset: custom
    custom:
      max_depth: 1
      samples_per_src: 20000
      max_num_paths_per_src: 500000
  materials:
    ground_asphalt:
      scattering_coefficient: 0.3
      scattering_pattern: lambertian
```

The one-interaction limit keeps the example focused on the direct path and
single ground interactions. The 20,000-sample ray budget is large enough to
produce a useful diffuse population while keeping all 100 frames practical to
regenerate. [Sionna RT](https://nvlabs.github.io/sionna/) performs the
propagation solve. ORCHAV evaluates the mobility and stores the resulting frame
sequence.

For exact ray-tracing fields, see the
[Scenario YAML Reference](../../../docs/reference/scenario_yaml.md#ray-tracing). For the
actor-motion behavior, see [Mobility and
Orientation](../../../docs/generator/mobility_and_orientation.md#mobility-models).

## Running

From the repository root, validate the schema, generate the local frame set,
then run the strict path check:

```bash
orchav-validate scenarios/visualizer/metrics_evolution/scenario.yaml
orchav-generator scenarios/visualizer/metrics_evolution
orchav-validate --strict scenarios/visualizer/metrics_evolution/scenario.yaml
```

Strict validation checks that the generated `frames/` directory exists, so run
it after generation on a fresh checkout.

The Generator writes 100 HDF5 frames under `frames/`. Generated frames are
local artifacts and are not stored in the repository. Compare the endpoints
from the terminal:

```bash
orchav-inspect scenarios/visualizer/metrics_evolution --frame 0
orchav-inspect scenarios/visualizer/metrics_evolution --frame 99
```

Then open the scenario:

```bash
orchav-visualizer --scenario scenarios/visualizer/metrics_evolution
```

## What To Observe

The **Metrics** window follows the displayed frame and the active TX/RX and
path-filter selection. Its **Overview** tab includes a
[power delay profile (PDP)](../../../docs/visualizer/analysis.md#current-frame-metrics),
which groups selected path gains by arrival delay. The current-frame summaries
also report RMS delay spread, a single value for how widely those arrivals are
spread in time.

1. Press `M` or use **View -> Metrics Window**, then select the **Overview** tab.
2. Play the timeline or scrub from frame 0 toward frame 99.
3. Watch the delay distribution and **Power Delay Profile (1 ns resolution)**
   move toward shorter delays as the receiver approaches.
4. Select the **Channel** tab to compare the delay-power scatter plot and link
   summary at different frames.
5. Compare path count, path loss, and delay spread while the delay-colored MPCs
   update in the 3D view.

The earliest path delay should decrease steadily because the direct distance
decreases steadily. The power-weighted mean delay should generally move lower
as well. Diffuse MPC count, individual diffuse paths, received strength, and
RMS delay spread can fluctuate from frame to frame. They are not expected to be
strictly monotonic.

## Representative Output

A generation with Sionna RT 2.0.x and the checked-in seed produced 714 MPCs in
each frame: one direct path, one specular ground path, and 712 diffuse ground
paths. Selected frames showed:

| Frame | RX x (m) | Earliest delay (ns) | Power-weighted mean delay (ns) | RMS delay spread (ns) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 25.000 | 150.335 | 150.542 | 1.119 |
| 25 | 14.899 | 116.709 | 116.952 | 1.640 |
| 50 | 4.798 | 83.136 | 83.332 | 1.613 |
| 75 | -5.303 | 49.728 | 49.777 | 1.282 |
| 99 | -15.000 | 18.647 | 18.938 | 1.489 |

The strongest retained path loss improved from about 94.47 dB to 76.34 dB.
Exact diffuse counts and powers can vary with the Sionna RT version and
hardware, but the geometric endpoint delays should remain stable.

For the frame data contract used by both views, see
[Frame Data Reference](../../../docs/shared/frame_reference.md).

## Browse Scenarios

> **Scenario path:** [All scenarios](../../README.md) | [Visualizer](../README.md) |
> Current: **Metrics Evolution**
>
> Previous: [MPC Inspection](../mpc_inspection/README.md) | Next:
> [Statistics](../statistics/README.md)
