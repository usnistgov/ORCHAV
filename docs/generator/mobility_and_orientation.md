# Mobility and Orientation

Every actor pose combines two independent specifications:

| Specification | Controls |
|---|---|
| Mobility | Position and velocity at each timeline sample |
| Orientation | The direction the actor faces at each timeline sample |

ORCHAV samples both specifications on the shared scenario timeline. Generation,
Scenario Builder preview, and motion plots therefore use the same positions and
orientations. Position-like values are meters, speed is meters per second, time
is seconds, and authored angles are degrees.

## Mobility Models

Use the thumbnails to choose a path shape. Each shows the samples used by the Generator and Scenario Builder.

| Type | Example | Behavior |
|---|---|---|
| `stationary` | <img src="../assets/generator/mobility_orientation/mobility_stationary.png" width="170" alt="One fixed position used at every timeline sample."> | Fixed `position_m`. |
| `linear` | <img src="../assets/generator/mobility_orientation/mobility_linear.png" width="170" alt="A straight path between its start and end positions."> | Straight path from `start_m` to `end_m`. |
| `waypoint` | <img src="../assets/generator/mobility_orientation/mobility_waypoint.png" width="170" alt="A path passing through four ordered waypoints."> | Ordered `points_m`, with `linear` or `catmull_rom` interpolation. |
| `circular` | <img src="../assets/generator/mobility_orientation/mobility_circular.png" width="170" alt="A complete horizontal circular orbit."> | Horizontal circle with radius, start angle, direction, and turns. |
| `survey` | <img src="../assets/generator/mobility_orientation/mobility_survey.png" width="170" alt="A rotated lawnmower survey route over a rectangular area."> | Rotated lawnmower path over a rectangular area. |
| `grid_scan` | <img src="../assets/generator/mobility_orientation/mobility_grid_scan.png" width="170" alt="A connected snake traversal through a three-dimensional grid."> | Raster or snake traversal through a 3D grid. |
| `oscillating` | <img src="../assets/generator/mobility_orientation/mobility_oscillating.png" width="170" alt="Vertical position varying sinusoidally around a fixed center."> | Sinusoidal displacement along an arbitrary axis. |
| `pendulum` | <img src="../assets/generator/mobility_orientation/mobility_pendulum.png" width="170" alt="A bounded pendulum arc around a pivot."> | Periodic arc around a pivot in a selected plane. |
| `figure8` | <img src="../assets/generator/mobility_orientation/mobility_figure8.png" width="170" alt="A horizontal figure-eight trajectory."> | Figure-eight path in the `xy`, `xz`, or `yz` plane. |
| `spiral` | <img src="../assets/generator/mobility_orientation/mobility_spiral.png" width="170" alt="A helical path climbing between two altitudes."> | Helical path between two altitudes. |
| `random_sampling` | <img src="../assets/generator/mobility_orientation/mobility_random_sampling.png" width="170" alt="Independent colored observations scattered inside three-dimensional bounds, without a connecting path."> | Independent uniform or Poisson-disk samples, with an optional exact first observation and no physical velocity. |
| `sampled` | <img src="../assets/generator/mobility_orientation/mobility_sampled.png" width="170" alt="Seven exact positions authored for seven timeline samples."> | Exact `positions_m` authored in YAML, one for each timeline step. |
| `gauss_markov` | <img src="../assets/generator/mobility_orientation/mobility_gauss_markov.png" width="170" alt="A smooth seeded stochastic path bounded in the horizontal plane."> | Seeded correlated motion in XY. Z remains at the initial altitude. |
| `random_waypoint` | <img src="../assets/generator/mobility_orientation/mobility_random_waypoint.png" width="170" alt="A seeded three-dimensional path between random destinations."> | Seeded destinations, speeds, and optional pauses inside bounds. |
| `manhattan_grid` | <img src="../assets/generator/mobility_orientation/mobility_manhattan_grid.png" width="170" alt="A seeded path that moves and turns along a rectangular street grid."> | Seeded movement and turns on a rectangular street grid. |
| `network_route` | <img src="../assets/generator/mobility_orientation/mobility_network_route.png" width="170" alt="A selected route highlighted over a cached local street graph."> | Route sampled from a cached local network graph. |
| `mesh_sequence` | <img src="../assets/generator/mobility_orientation/mobility_mesh_sequence.png" width="170" alt="A target path prepared from position samples loaded from a local file."> | Target-only positions loaded from a local data file. Target mesh geometry is configured separately. |
| `group_member` | <img src="../assets/generator/mobility_orientation/mobility_group_member.png" width="170" alt="A group member following a curved reference path with a heading-relative offset."> | A named group trajectory plus a local heading-frame offset. |

## Configure a route

For a receiver walking along a curved path, set its mobility to `waypoint`. The points use scene coordinates in metres:

```yaml
mobility:
  type: waypoint
  interpolation: catmull_rom
  points_m: [[0.0, 0.0, 1.5], [4.0, 3.0, 1.5], [9.0, 1.0, 1.5]]
```

Use a seed for stochastic models so preview and generation reproduce the same positions. For externally sampled positions use `sampled`, one position per timeline step. Choose a `network_route` with a local graph for street-constrained travel. Field names and constraints for every model are in the [mobility reference](../reference/scenario_yaml.md#mobility).

## Timeline

The root timeline determines when every mobility and orientation model is
sampled:

```yaml
timeline:
  steps: 61
  duration_s: 6.0
```

Samples include both endpoints. A moving scenario requires at least two
samples and positive duration. A fully stationary scenario may use one sample
and zero duration.

## Traversal

Finite path models traverse the complete authored path over
`timeline.duration_s` by default. Add `constant_speed` only when physical speed
should determine progress:

```yaml
mobility:
  type: waypoint
  points_m:
    - [0.0, 0.0, 1.5]
    - [10.0, 0.0, 1.5]
    - [10.0, 8.0, 1.5]
  traversal:
    type: constant_speed
    speed_mps: 1.4
    after_end: ping_pong
```

`after_end` is `hold`, `loop`, or `ping_pong`. Periodic and stochastic models
define their own time behavior.

## Optional Groups

A group owns a standalone reference mobility. An actor joins only by choosing
`group_member`. Ordinary actors do not declare a group.

```yaml
groups:
  - name: Convoy
    mobility:
      type: waypoint
      points_m:
        - [-30.0, 0.0, 2.0]
        - [0.0, 15.0, 2.0]
        - [30.0, 0.0, 2.0]

actors:
  rx:
    - name: LeadRX
      mobility:
        type: group_member
        group: Convoy
    - name: FollowRX
      mobility:
        type: group_member
        group: Convoy
        offset_m:
          right: 3.0
          forward: -5.0
          up: 0.0
```

Offsets are expressed as right, forward, and up in the group's local heading
frame. A group may add bounded seeded deviation. Groups require at least two
members and cannot reference another group.

## Orientation Models

Orientation is optional and defaults to fixed zero yaw, pitch, and roll.
Angles follow [Sionna RT](https://nvlabs.github.io/sionna/rt/)'s right-handed
Z/Y/X convention. The actor's forward axis
is +X and world up is +Z, so positive pitch turns +X toward -Z while negative
pitch turns it toward +Z.

The examples show the actor's +X forward direction or its yaw, pitch and roll
angles over time.

| Type | Example | Behavior |
|---|---|---|
| `fixed` | <img src="../assets/generator/mobility_orientation/orientation_fixed.png" width="170" alt="An actor with a constant 180-degree yaw heading."> | Constant yaw, pitch, and roll. |
| `keyframes` | <img src="../assets/generator/mobility_orientation/orientation_keyframes.png" width="170" alt="Yaw, pitch, and roll prepared between three timed orientation keyframes."> | Time-stamped orientations joined by smooth rotation. |
| `align_motion` | <img src="../assets/generator/mobility_orientation/orientation_align_motion.png" width="170" alt="Forward arrows following the direction of travel along a curved path."> | Points the actor's +X forward axis along physical velocity. |
| `look_at` | <img src="../assets/generator/mobility_orientation/orientation_look_at.png" width="170" alt="A stationary actor turning to track another actor along a rising path."> | Points +X toward another actor or a fixed point. |
| `spin` | <img src="../assets/generator/mobility_orientation/orientation_spin.png" width="170" alt="Yaw increasing at a constant 30 degrees per second while pitch and roll remain fixed."> | Rotates one selected yaw, pitch, or roll axis at a constant rate. |
| `random` | <img src="../assets/generator/mobility_orientation/orientation_random.png" width="170" alt="Seeded random yaw and pitch values held for half-second intervals."> | Seeded bounded samples, optionally held for `update_interval_s`. |

Keyframes use spherical linear interpolation (SLERP) to rotate smoothly
between the authored orientations. Configure their yaw, pitch and roll in degrees.

Choose `align_motion` when the actor should face its direction of travel. Set `allow_pitch: false` to keep it level. Use `fixed` to hold a heading, or `keyframes` to interpolate between orientations at named times.

### Look At

Track another actor by globally unique name:

```yaml
orientation:
  type: look_at
  actor: PatrolTarget
  allow_pitch: true
```

Or track a fixed point:

```yaml
orientation:
  type: look_at
  point_m: [0.0, 0.0, 10.0]
```

Exactly one of `actor` and `point_m` is required. Smoothing, rate limits,
offsets, and yaw/pitch limits are optional.

For optional smoothing, angular-rate limits, spin rates and random-orientation seeds, see the [orientation reference](../reference/scenario_yaml.md#orientation).

## Target Asset Alignment

A catalog target can declare its modeled front direction in
`target_metadata.json`. ORCHAV combines that asset alignment with the actor's
orientation before generation or preview. Keep asset-specific corrections with
the mesh metadata, as in the
[bundled walking target](../../libraries/targets/digital_twin_update1/README.md).
For imported animated assets, use the
[world and body conventions](motion/animated_targets.md#world-and-body-coordinates).

## Road-network Trajectories

Use `network_route` to follow a cached OpenStreetMap street graph instead of
interpolating a straight line between arbitrary world coordinates. Obtain
`street_network.graphml` through [OSM scene preparation](geography/osm_scene_generation.md)
and select node IDs present in that graph:

```yaml
mobility:
  type: network_route
  graph_path: street_network.graphml
  travel_mode: car
  start_node: '1749655963'
  end_node: '1749656095'
  altitude_m: 1.0
  traversal:
    type: constant_speed
    speed_mps: 10.0
orientation:
  type: align_motion
  allow_pitch: false
```

The example node IDs illustrate the syntax. Use nodes from your own graph.
`travel_mode` selects permitted edges, `altitude_m` sets the actor's height,
and `constant_speed` samples the route at 10 m/s on the scenario timeline.
`align_motion` follows the route tangent while `allow_pitch: false` keeps the
actor level. The graph must use the scene's coordinate frame. Choose distinct
routes for multiple actors, then configure sensing through the same scenario settings used for other
mobility models.

## Runnable Examples

- [Actor Mobility](../../scenarios/generator/mobility_and_orientation/actor_mobility/README.md)
- [Actor Orientation](../../scenarios/generator/mobility_and_orientation/actor_orientation/README.md)
- [Mesh Targets](../../scenarios/generator/targets/mesh_targets/README.md)
- [Multi-Device Trajectory](../../scenarios/visualizer/multi_device_trajectory/README.md)

---

Up: [Generator](README.md) | Previous: [Simulation Configuration](configuration.md) | Next: [Summary and Coverage Figures](generated_figures.md)
