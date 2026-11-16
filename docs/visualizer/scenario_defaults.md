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

An explicit `camera_pose` takes precedence over the named view. It sets only
the initial camera. Orbit, pan, zoom and the Camera panel remain available.
Optional `playback` settings control display cadence, independently of the
scenario's sampling rate. Playback starts paused. Sensing-only navigation uses x1
over saved results and retains the authored stride for full-timeline navigation
when **Play sensing frames only** is cleared.

```yaml
view_defaults:
  camera_pose:
    eye: [-6, -129, 16]
    lookat: [-38, -55, 9]
    up: [0, 0, 1]
    fov: 60
  playback: {mode: fixed_fps, fps: 25, stride: 32}
  appearance_recipe: appearance.json
```

`appearance_recipe` references a local version-1
[appearance recipe](appearance.md#customize-and-save-a-look), resolved using the scenario path
policy. It uses the same material, texture, lighting and shadow controls as
the normal pygfx UI. It does not download sky assets, and an explicit
`--disable-textures` remains authoritative. Other renderers retain their own
appearance. For initial sensing overlays, use the `view_defaults.sensing` fields
in the [YAML reference](../reference/scenario_yaml.md#visualizer-defaults). Their
meaning follows the [sensing panel](sensing_dashboard.md). They do not change
detection or tracking.

### Authored Demo Tours

`view_defaults.demo_tour` makes **Scenario camera tour** available in the pygfx
[Demo panel](presentation_tools.md#demo-mode). It does not start automatically.
Views use the existing [camera sequence](camera.md) format. Coordinates are in
scene metres, and shot `hold` and `move` values are seconds.

```yaml
view_defaults:
  demo_tour:
    label: Plaza tour
    views:
      overview:
        pose: {eye: [-5, -142, 42], lookat: [-36, -59, 10], up: [0, 0, 1], fov: 56}
      street:
        pose: {eye: [-14, -98, 6], lookat: [-36, -45, 11], up: [0, 0, 1], fov: 65}
    shots:
      - {view: overview, hold: 5, move: 0}
      - {view: street, hold: 5, move: 4}
```

The first view is applied immediately. Later shots travel to their named view
before holding it. If the sequence does not end at its first view, a four-second
return transition closes the loop. Each timing must be finite and between
0 and 120 seconds. Tours accept 2–32 named views and 2–64 authored shots.

Optional `actor` selectors use the camera service's saved-view identity format. Their
`pose` coordinates are offsets from that actor's current recorded position.
Omit `actor` for fixed scene views. Camera movement never interpolates scientific
frame data. Takeover leaves the camera at its current position and restores the
user's named views, sequence and camera history. The separate **Exhibition
layout** option is selected interactively in Demo.

### Authored Dusk Lighting

`view_defaults.dusk` makes **Scenario blue hour** available in Rendering and
enables **Blue hour** and **Sunset cycle** in Demo's **Scenario lighting**
selector. Opening a scenario does not enable it. The recipe accepts a `label`,
`facade_lights`, `targets` and `window_accents`.

Each facade light has world-space `position` and `lookat` coordinates in metres,
an RGB `color` in [0, 1], `intensity`, `distance` in metres and `angle_deg`.
Each target entry uses its render `object_id`, a target-local `light`, and an
optional emission palette (`palette_slots` and `emissive_slots`). Target-local
coordinates follow the recorded mesh transform. At most six shadowless local
lights are permitted, with at most four facade lights and four targets.

The included [Paris recipe](../../scenarios/showcases/paris_opera_multitarget/scripts/dusk_look.py)
shows the complete configuration. Its window accents require the matching
Haussmann atlas. Other facade textures remain unchanged. See
[presentation controls](presentation_tools.md#authored-scenario-lighting) for
selecting and cycling authored lighting.

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

### Visual Material Profiles

Use `view_defaults.visual_profiles` to give a target a visual preset independently
of its radio material:

```yaml
view_defaults:
  visual_profiles:
    - match:
        name: Cube
        type: target
      preset: Matte
```

Names use case-insensitive regular expressions with full-name matching. The first
matching valid rule wins. Keep other rules when adding a profile. This changes
appearance only, not geometry, ray tracing or saved sensing results. Global
[scene appearance](appearance.md) are separate from these per-object rules.

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

| Goal | Primary reference |
|------|-------------------|
| Choose local files, Live Generator, or Remote HDF5 | [Shared Data Layer](../shared/README.md) |
| Choose or tune a renderer backend | [Renderers](renderers.md) |
| Configure Generator-side ray tracing | [Generator Configuration](../generator/configuration.md) |
| Configure application-level endpoints | [Application Configuration](../reference/application_configuration.md) |
| Inspect stored HDF5 frame data | [Frame Inspection](../shared/frame_inspection.md) |
| Look up a typed YAML field | [Scenario YAML Reference](../reference/scenario_yaml.md) |

---

Up: [Visualizer](README.md) | Related: [Scenario YAML Reference](../reference/scenario_yaml.md) | [Workspace Snapshots](workspace_snapshots.md) | [Renderers](renderers.md)
