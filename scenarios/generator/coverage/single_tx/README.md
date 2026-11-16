# Single-Transmitter Coverage Map

This scenario generates coverage maps over the built-in
[Sionna RT](https://nvlabs.github.io/sionna/) Etoile scene. It is a static
radio-map workflow: one transmitter is placed above the scene and the Generator
samples virtual receiver locations at two heights instead of tracing a small
set of named receivers.

To run it, see [Running](#running).

## Scene Layout

![Scenario topology](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_topology.png)

| Element | Role |
| --- | --- |
| `BaseStation` | Single transmitter at 30 m altitude |
| Coverage grid | Virtual receiver samples over the Etoile scene |
| Coverage heights | 1.5 m and 30 m |

## Configuration Walkthrough

See [scenario.yaml](scenario.yaml) for the complete configuration.

The scenario is intentionally small:

```yaml
scene:
  source: sionna
  id: etoile

actors:
  tx:
    - name: BaseStation
      mobility:
        type: stationary
        position_m: [0, 0, 30]
      orientation:
        type: fixed
        pitch_deg: -45

coverage:
  enabled: true
  grid:
    resolution_m: [5, 5]
    heights_m: [1.5, 30.0]
```

The coverage block uses automatic scene bounds and the default `medium`
coverage solver. An explicit seed is supplied to the coverage solver.

## Running

Generate the frames:

```bash
orchav-generator scenarios/generator/coverage/single_tx
```

Open the scenario in the Visualizer:

```bash
orchav-visualizer --scenario scenarios/generator/coverage/single_tx
```

## Outputs

Running the scenario writes the coverage dataset to `coverage/coverage_maps.h5`.
The PNG maps, metric guides, and distributions shown below are generated under
`summary/coverage/`, once for each configured height. The accompanying
`frames/` directory contains the single topology frame and its manifest. It
does not duplicate the coverage grid.

| Output | 1.5 m | 30 m |
| --- | --- | --- |
| Quick-look map | ![Coverage at 1.5 m](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_coverage_height-01_1.5m.png) | ![Coverage at 30 m](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_coverage_height-02_30m.png) |
| Metric guide | ![Coverage metrics at 1.5 m](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_metrics_height-01_1.5m.png) | ![Coverage metrics at 30 m](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_metrics_height-02_30m.png) |
| Distribution | ![Coverage distributions at 1.5 m](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_distributions_height-01_1.5m.png) | ![Coverage distributions at 30 m](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_distributions_height-02_30m.png) |

Use the map views to compare where coverage is strong or weak across the scene.
Use the distribution plots to compare aggregate path loss, RSS, and SINR across
the sampled grid. Map and metric-guide colors use limits shared across both
heights, so their colors can be compared directly.

The output names include both the one-based height order and physical height,
such as `coverage_maps_height-01_1.5m.png` and
`coverage_maps_height-02_30m.png`.

Standalone 3D views use the same saved coverage values:

| 3D view at 1.5 m | 3D view at 30 m |
| --- | --- |
| ![3D coverage at 1.5 meters](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_coverage_3d.png) | ![3D coverage at 30 meters](../../../../docs/assets/scenarios/scenarios_generator_coverage_single_tx_coverage_3d_height-02_30m.png) |

To export a standalone 3D coverage slice:

```bash
python -m generator.tools.coverage_figure --scenario scenarios/generator/coverage/single_tx --output coverage_3d.png
```

Use `--height-index 1` for the 30 m slice. This command runs without opening the
Visualizer or rerunning the solver. See [Standalone 3D Coverage
Slices](../../../../docs/generator/generated_figures.md#standalone-3d-coverage-slices)
for metric and camera controls.

## View Coverage In The Visualizer

After opening the scenario, select the **Coverage** tab and enable **Show
Coverage Map**. Choose a metric and use the height selector to switch between
1.5 m and 30 m. See [Coverage
Inspection](../../../../docs/visualizer/analysis.md#coverage-inspection) for the
remaining overlay controls.

## What To Notice

- `path_loss_db/BaseStation` is the path loss from `BaseStation` to every grid
  cell. `best_path_loss_db` normally selects the lowest path loss across all
  transmitters. With only one transmitter, it contains the same values. The
  distinction becomes useful in the
  [multi-transmitter example](../multi_tx/README.md).
- The same map is generated at 1.5 m and 30 m, so street-level and elevated
  coverage can be compared directly.
- Received-power and SINR views are derived from the stored path gain and radio
  metadata rather than saved as separate solver outputs.
- The coverage workflow samples virtual receiver positions, so the YAML does
  not need receiver actors.

This scenario does not need the top-level `raytracing` block because it does
not generate propagation paths for named transmitter and receiver pairs. The
`coverage` block configures its separate grid solver. After the coverage map is
published, ORCHAV writes one topology frame so the Visualizer can load the
transmitter placement.

## Browse Scenarios

Up: [Generator](../../README.md) ·
[Multi-Transmitter Coverage](../multi_tx/README.md)
