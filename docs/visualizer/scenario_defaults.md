# Visualizer Scenario Defaults

Scenario YAML can define the Visualizer's initial camera, filtering, coloring,
visibility, panel availability, and TX/RX marker appearance.

Use [Scenario Authoring](../generator/scenario_authoring.md) for the complete
scenario structure, the [Shared Data Layer](../shared/README.md) to choose a
[frame provider](../reference/glossary.md#frame-provider) and data mode, and the
[Scenario YAML Reference](../reference/scenario_yaml.md) for exact typed YAML
fields and defaults.

## Runnable Examples

- [Metrics Evolution](../../scenarios/visualizer/metrics_evolution/README.md)
  sets the initial camera, selected pair, delay coloring, and bounce-point
  visibility with `view_defaults`.
- [Multi-Device
  Trajectory](../../scenarios/visualizer/multi_device_trajectory/README.md)
  configures custom TX and RX marker appearance under `visualizer`.
- [Statistics](../../scenarios/visualizer/statistics/README.md) explicitly
  enables static scene batching for its dense urban scene.

Each scenario page provides the generation and Visualizer commands. The
sections below explain what those settings mean and when a workspace snapshot
overrides them.

## View Defaults

`view_defaults` describes how a scenario should first appear. Users can still
change these values from the application. Doing so does not rewrite
`scenario.yaml`.

```yaml
view_defaults:
  camera_view: isometric
  camera_dist: 80
  fov: 60
  color_mode: path_loss
  selected_tx: all
  selected_rx: all
  mpc_visibility:
    enabled: true
    paths: true
    bounce_points: false
```

The named camera view and distance establish the initial framing. TX/RX
selection limits the displayed pairs, and `color_mode` selects the initial MPC
legend. If ORCHAV resumes a workspace snapshot, the saved camera, filters, and
visibility replace the corresponding initial values. Start with `--no-resume`
when checking the YAML defaults themselves. See [Workspace
Snapshots](workspace_snapshots.md) for the complete restore behavior.

### MPC Visibility

`mpc_visibility.enabled` is the master MPC-layer switch. `paths` and
`bounce_points` are independent preferences evaluated while that layer is
enabled. Bounce points are physical path-interaction points, not every vertex
used to draw a path.

### Color Modes

| Mode | Meaning |
|------|---------|
| `reflection_order` | Color each MPC by its number of interactions. |
| `mpc_type` | Color each MPC by the path type carried by the frame. |
| `delay` | Color by propagation delay. |
| `path_loss` | Color by path loss when path metrics are available. |

### Renderer-Dependent Defaults

`view_defaults.merge_scene_meshes` controls static scene batching: omit it for
automatic renderer policy, set it to `true` to force batching of compatible
static meshes, or set it to `false` to disable batching. See
[Renderers](renderers.md) for the rendering behavior.

Exact `view_defaults` fields are listed in the
[Scenario YAML Reference](../reference/scenario_yaml.md#visualizer-defaults).

## Component-Specific Settings

The root `visualizer` block configures Visualizer components rather than the
shared scene, actor, or frame model.

### Statistics Panel

The Statistics panel is enabled by default. Disable it for a scenario that
should not create the panel or start whole-scenario statistics processing:

```yaml
visualizer:
  panels:
    statistics:
      enabled: false
```

This changes component availability, whereas hiding a panel interactively is
only a workspace choice.

### TX And RX Markers

`node_markers` can set shared defaults and then override TX or RX appearance.
The normal marker is a centered sphere with a size of 0.3 m.

```yaml
visualizer:
  node_markers:
    default:
      center: true
    tx:
      shape: box
      size: 2.0
    rx:
      shape: mesh
      mesh_path: meshes/receiver_marker.obj
      marker_size: 4.0
      center: true
```

| Key | Meaning |
|-----|---------|
| `shape` | `sphere`, `box`, or `mesh`. An unavailable custom mesh falls back to a sphere. |
| `size`, `marker_size` | Equivalent positive marker-size keys. |
| `mesh_path` | Custom OBJ or PLY marker path. Relative paths start at the scenario directory. |
| `center` | Center the marker's bounds on the actor position. The default is `true`. |
| `scale` | Additional positive scale multiplier for a custom mesh. |

Settings under `default` apply to both roles. A nested `tx` or `rx` value
overrides the corresponding default.

## Related Settings

| Task | Primary reference |
|------|-------------------|
| Choose local files, Live Generator, or Remote HDF5 | [Shared Data Layer](../shared/README.md) |
| Choose or tune a renderer backend | [Renderers](renderers.md) |
| Configure Generator-side ray tracing | [Generator Configuration](../generator/configuration.md) |
| Configure application-level endpoints | [Application Configuration](../reference/application_configuration.md) |
| Inspect stored HDF5 frame data | [Frame Inspection](../shared/frame_inspection.md) |
| Look up a typed YAML field | [Scenario YAML Reference](../reference/scenario_yaml.md) |

---

Up: [Visualizer](README.md) | Related: [Scenario YAML Reference](../reference/scenario_yaml.md) | [Workspace Snapshots](workspace_snapshots.md) | [Renderers](renderers.md)
