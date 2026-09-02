# Target Orientation

[Scenarios](../../../README.md) > [Generator](../../README.md) > Target Orientation

This scenario is a compact visual check for
[target](../../../../docs/reference/glossary.md#target) and node orientation. It
uses three human targets. Two walkers face their direction of travel, and one
stationary human tracks a moving walker with yaw-only `look_at` orientation.
The transmitter and receiver also point at different moving walkers so actor
`look_at` orientation can be checked in the same Visualizer run.

To run it, see [Running](#running).

## Actors and Checks

| Element | Role |
| --- | --- |
| `FollowWalkerEastTX` | Fixed transmitter with `look_at` orientation toward `WalkerEast` |
| `FollowWalkerWestRX` | Fixed receiver with `look_at` orientation toward `WalkerWest` |
| `WalkerEast` | Human target moving east with `align_motion` orientation |
| `WalkerWest` | Human target moving west with `align_motion` orientation |
| `TrackingYawHuman` | Human target with yaw-only `look_at` orientation toward `WalkerEast` |

## Configuration Walkthrough

```yaml
actors:
  tx:
    - name: FollowWalkerEastTX
      mobility:
        type: stationary
        position_m: [-24, -14, 7]
      orientation:
        type: look_at
        actor: WalkerEast

  rx:
    - name: FollowWalkerWestRX
      mobility:
        type: stationary
        position_m: [24, -14, 2]
      orientation:
        type: look_at
        actor: WalkerWest

  targets:
    - name: WalkerEast
      asset:
        source: directory
        path: libraries/targets/nist_human_walking
        pattern: "fitted_*.ply"
      mobility:
        type: linear
        start_m: [-18, 8, 0.85]
        end_m: [18, 8, 0.85]
      orientation:
        type: align_motion
        allow_pitch: false

    - name: TrackingYawHuman
      asset:
        source: directory
        path: libraries/targets/nist_human_walking
        pattern: "fitted_*.ply"
      mobility:
        type: stationary
        position_m: [0, -10, 0.85]
      orientation:
        type: look_at
        actor: WalkerEast
        allow_pitch: false
```

The bundled NIST human mesh declares a target-front offset for `align_motion`
and `look_at` orientations. `TrackingYawHuman` disables pitch so the target-tracking
case is a pure yaw check.

## Running

```bash
orchav-validate scenarios/generator/targets/target_orientation
orchav-generator scenarios/generator/targets/target_orientation
orchav-visualizer --scenario scenarios/generator/targets/target_orientation
```

## What To Notice

- `WalkerEast` and `WalkerWest` should face along their linear paths.
- `TrackingYawHuman` should rotate in yaw while staying fixed and tracking
  `WalkerEast`.
- `FollowWalkerEastTX` should point at `WalkerEast`.
- `FollowWalkerWestRX` should point at `WalkerWest`.
- Enable the target orientation axes to see each target's local frame. The axes
  should not change target or node motion. On Windows and Linux, you can compare
  the same orientation in pygfx and Open3D. macOS v0.1 uses pygfx.

## Outputs

Running the scenario writes default HDF5 frame chunks under `frames/` and
summary figures under `summary/`. The orientation summary is useful for spotting
unexpected yaw wrapping before opening the Visualizer.

```bash
orchav-inspect scenarios/generator/targets/target_orientation --frame 0
```

## Browse Scenarios

> **Scenario path:** [All scenarios](../../../README.md) | [Generator](../../README.md) |
> Current: **Target Orientation**
>
> Track: **Targets** | Previous: [Mesh Targets](../mesh_targets/README.md) | Next: [Target Diffuse Scattering](../target_diffuse_scattering/README.md)
