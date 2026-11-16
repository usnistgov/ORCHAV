# Statistics

This scenario demonstrates whole-sequence Statistics in the Visualizer. It
uses the [Sionna RT](https://nvlabs.github.io/sionna/) `florence` scene, which
represents Florence's historic city center with dense streets and piazzas. A
street-level receiver
follows a 50-frame route while a rooftop transmitter stays fixed. The Analysis
tab summarizes the complete frame sequence, and the scenario also saves a
coverage map for the same area.

To run it, see [Running](#running).

## Scene Layout

![Scenario topology](../../../docs/assets/scenarios/scenarios_visualizer_statistics_topology.png)

![Coverage map](../../../docs/assets/scenarios/scenarios_visualizer_statistics_coverage_map.png)

| Element | Role |
| --- | --- |
| `TX1` | Stationary rooftop transmitter in the Florence scene |
| `RX1` | Receiver following a street-level waypoint route |
| `florence` | Sionna RT-provided scene representing Florence's historic city center |
| Coverage grid | Street-level virtual receiver samples at 1.5 m |

## Scenario Configuration

See [scenario.yaml](scenario.yaml) for the complete configuration.

```yaml
timeline:
  steps: 50
  duration_s: 50.0

raytracing:
  enabled: true
  quality:
    preset: custom

coverage:
  enabled: true
  grid:
    resolution_m: [2.0, 2.0]
    heights_m: [1.5]

generator_summary:
  enabled: true
  create: [topology, speed]
```

The custom ray-tracing and coverage profiles inherit the medium solver
settings, then override the interaction depth, ray budget, refraction, and
diffraction settings. The full YAML also sets two material scattering
coefficients and explicitly enables coverage figures with a scene overlay.
Frame output uses a local manifest-driven HDF5 frame set, while the coverage
data is saved separately to `coverage/coverage_maps.h5`.

The receiver route is declared in YAML with `mobility.type: waypoint`.
`frames/` contains the generated HDF5 frame set, `coverage/coverage_maps.h5`
contains the saved coverage-map data, and `summary/` contains standalone
diagnostic figures for the route, speed profile, and coverage map.

## Running

Generate the frames from the repository root:

```bash
orchav-generator scenarios/visualizer/statistics
```

Inspect the saved data:

```bash
orchav-inspect scenarios/visualizer/statistics --coverage
```

Then open the scenario:

```bash
orchav-visualizer --scenario scenarios/visualizer/statistics
```

## What To Notice

- When the Visualizer opens the frame set, the **Analysis** tab's
  **Statistics** section checks for a matching cache. On a cache miss, choose
  **Compute Statistics**. Automatic computation on a cache miss is optional
  and off by default.
- The **Graphs** section summarizes the full 50-frame route. MPC count, path
  loss, delay, interaction-mechanism and interaction-order trends change as `RX1` moves through the
  streets.
- The saved coverage map provides a broader street-level path-loss view for the
  same Florence scene.

## What To Try

- Choose **Compute Statistics** if no matching cache was loaded. Wait for the
  status to report that the values are ready, then expand **Graphs**. The same
  button can force another scan of the currently open recording. For material
  contacts in the current frame, use **Metrics → Materials** instead.
- Open **Coverage** and enable **Show coverage map**. This scenario has one
  coverage layer, at 1.5 m. The coverage graphs in **Analysis** > **Graphs**
  follow the selected coverage metric.
- For spatial context, open **Scene** > **Nodes**, enable **Trajectory** in the
  **RX** section, and compare the route with the whole-sequence graphs.

## Browse Scenarios

Up: [Visualizer scenarios](../README.md) | Previous: [Metrics Evolution](../metrics_evolution/README.md) | Next: [Multi-Device Trajectory](../multi_device_trajectory/README.md)
