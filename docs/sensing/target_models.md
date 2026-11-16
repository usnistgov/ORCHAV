# Choose a target representation

Use this guide when deciding **what physically produces the target's radio
return**. ORCHAV supports two distinct modeling routes: mesh/material
propagation through Sionna RT's ordinary path solver, and Sionna RT 2.2's
**native radar cross section (RCS) scattering-point targets**. ORCHAV feeds
either kind of simulated channel into its own sensing acquisition, detector
and tracker. A native RCS target is therefore *not* a separate ORCHAV
detection algorithm.

The choice of RF representation is separate from whether a target is stationary,
moves rigidly, or changes shape. For range, Doppler, tracking and experimental
limitations shared by both routes, see [sensing concepts](concepts.md).

## Choose A Representation

| Target representation | What computes the return | Choose it when |
|---|---|---|
| `mesh_material` (mesh-based sensing) | Sionna RT's propagation solver traces interactions with the target's actual mesh surfaces and assigned radio material. | Surface geometry, orientation, material contacts, occlusion or changing body shape matter. No RCS model is assumed or calibrated from the mesh alone. |
| `constant_rcs` (native Sionna RT 2.2) | `RCSSolver` uses one scattering point with a specified RCS; a box or mesh can define an absorbing proxy. | A controlled, rigid scattering-point response is more appropriate than detailed surface paths. |
| `tr38901` (native Sionna RT 2.2) | `RCSSolver` uses Sionna's supported 3GPP TR 38.901 RCS presets with one or multiple scattering points. | You want the prescribed angular/RCS model for a human, UAV, vehicle or AGV instead of mesh-material scattering. |

For a native RCS target, the proxy mesh or box controls blockage and display,
**not the target's scattering response**. Its response comes from the RCS model.
Conversely, a mesh/material target returns energy from the traced surface
interactions; assigning a visually human-shaped mesh does not provide a
validated human RCS automatically. Sionna RT 2.2 supplies the native RCS
models and `RCSSolver`; ORCHAV integrates those paths with its channel
acquisition, visualization, detection and tracking. See
[Sionna's RCS tutorial](https://nvlabs.github.io/sionna/rt/tutorials/RCS.html)
and [native RCS installation](../getting_started/installation.md#native-rcs-targets).

## Choose how the target moves

| Motion | How to configure it | Applicable representations |
|---|---|---|
| Stationary | `mobility.type: stationary` | Mesh or native RCS |
| Rigid translation or route | Ordinary actor `mobility` and `orientation` (for example, linear or waypoint) | Mesh or native RCS; no animated-target manifest is needed |
| Changing mesh surface / articulated motion | **Simplest:** an [ordered PLY folder](../generator/motion/animated_targets.md#simple-example-a-folder-of-ply-poses). Optionally use a [timed manifest](../generator/motion/animated_targets.md#alternative-a-timed-motion-manifest) for imported motion with explicit pose timing. | Mesh/material only in the current integration |

A moving rigid mesh is not the same thing as an articulated mesh animation.
The included [mesh-target scenario](../../scenarios/generator/targets/mesh_targets/README.md)
shows propagation involving ordinary meshes. The
[marching-person sensing scenario](../../scenarios/sensing/targets_motion/marching_person/README.md)
starts with a rigid human mesh and only later introduces articulated poses.
The native RCS integration supports rigid translation and explicitly assigned
velocity; it does not synthesize limb or rotor micro-Doppler from a model's
scattering points.

## Walkthroughs

Start with [stationary cube mesh sensing](../../scenarios/sensing/cube_radar/stationary/README.md)
for a surface reflection. For native RCS, compare
[human model 1](../../scenarios/sensing/foundations/RCS/08_human_model1/README.md)
(one scattering point) and the
[five-point vehicle](../../scenarios/sensing/foundations/RCS/06_vehicle_multi/README.md).
Both native examples show visualization, range–Doppler processing, detection
and tracking; the
[native RCS catalog](../../scenarios/sensing/foundations/RCS/README.md)
introduces the other supported presets.

The [human-response video](../visualizer/demo_video.md#human-target-response-rcs-and-articulated-mesh)
compares one rigid native RCS recording with one articulated mesh recording.
It illustrates those **specific configurations**, not an isolated or
calibrated head-to-head comparison of all RCS and mesh approaches.

## Configuration

The following is a target entry under `actors.targets`. Mobility and orientation
retain their ordinary ORCHAV meaning:

```yaml
- name: control
  representation: constant_rcs
  geometry:
    type: box
    length_m: 0.3
    width_m: 0.3
    height_m: 0.3
  constant_rcs:
    sigma_m2: 1.0
    xpr_db: null
  mobility:
    type: stationary
    position_m: [40, 0, 1]
```

`sigma_m2` is a nonnegative linear area in square meters. For a dBsm value `x`,
the equivalent area is `10 ** (x / 10)`. `xpr_db: null` preserves polarization.
Zero is a different polarization assumption, not an absent value.

Choose a TR 38.901 preset by object and model type:

```yaml
- name: person
  representation: tr38901
  geometry: {type: box}
  tr38901:
    object_type: human
    model_type: 2
    random_sigma_s: false
    random_xpr: false
    random_phases: false
  mobility:
    type: linear
    start_m: [40, 0, 1]
    end_m: [39, 0, 1]
  orientation:
    type: fixed
    yaw_deg: 0
```

**Model 1 and model 2 describe the RCS response, not the number of scattering
points.** Model 1 has an angle-independent monostatic RCS response. Model 2 has
an angle-dependent response. A large-UAV model 2 target still has one point.
Vehicle point count is selected by the object preset. These definitions follow
the [upstream native RCS API](https://nvlabs.github.io/sionna/rt/api/rcs.html).

Supported combinations are validated before scene construction:

| `object_type` | `model_type` | Points |
|---|---|---|
| `uav-small-size` | 1 | 1 |
| `human` | 1 or 2 | 1 |
| `uav-large-size` | 2 | 1 |
| `vehicle-single-sp` | 2 | 1 |
| `vehicle-multi-sp` | 2 | 5 |
| `agv-single-sp` | 2 | 1 |
| `agv-multi-sp` | 2 | 5 |

The generated [scenario reference](../reference/scenario_yaml.md#target-constant-rcs)
lists the native model/geometry fields and their defaults. The
[solver reference](../reference/scenario_yaml.md#native-rcs-solver) lists the
separate native ray budgets.

An empty native box specification uses the upstream dimensions. Specifying one
dimension overrides that dimension only. A mesh proxy uses
`geometry: {type: mesh, path: proxy.ply, scale: 1.0}`. It cannot also specify box
dimensions. The mesh's positive local bounding-box extents determine model
dimensions/point placement. Target position is the **center**, not a foot or
ground-contact coordinate. Local +x is front, y is width, z is height.
Scaling geometry/points does not automatically change the declared RCS table
or constant sigma.

Native entries cannot also contain `asset`, materials, or the other RCS model
block. Existing mesh entries need no new fields.

## Solver And Acquisition

```yaml
raytracing:
  rcs:
    channel: combined
    max_depth: 3
    samples_per_sp: 100000
    buffer_size_per_sp: 100000
    los: true
    specular_reflection: true
    refraction: true
    seed: 42
```

`combined` joins native target paths with the background paths computed by
`PathSolver`. Use `target_only` or `background_only` to inspect either channel.
Native `max_depth` includes the sensing event: one gives direct TX–target–RX
legs. Diffuse reflection and diffraction are unsupported for native RCS legs.

The native route requires one TX/RX pair and `synthetic_array: true`. Start with
monostatic SISO, `coherent_cpi` and `cfr_ifft`, as used in the included examples.
Use a polarized Mitsuba variant (`llvm_ad_mono_polarized` or
`cuda_ad_mono_polarized`) before constructing native targets.

Set target velocity explicitly through the motion configuration. Native
Doppler models rigid translation. Preset multipoint targets do not add rotation,
limb or rotor micro-Doppler. Coherent acquisition freezes geometry within each
CPI. Native visibility excludes the target's own bounding-box region when
connecting its scattering points, so it does not model articulated self-occlusion.
Keep random RCS, polarization and phase components disabled for deterministic
controls. A fixed seed alone does not ensure smooth temporal variation.

## Replay And Inspect

Frames retain the native model, target/point identities, pose and solve-time
state. The Visualizer displays these with the stored sensing results. CPU
reprocessing uses the saved acquisition without a new ray-tracing solve.
Changing target geometry or physical settings requires regeneration.

During analytical-Doppler playback, the proxy and scattering-point markers stay
at the pose used for the current CPI's ray-tracing solve. They advance at the
next solve, keeping their displayed blockage geometry aligned with the paths.
Sequential playback updates the pose on every independently solved pulse.
Existing recordings use their saved solve pose or referenced source frame.
The recorded timeline positions and sensing observations are preserved.


---

[Sensing](README.md) | [Sionna integration](concepts.md) |
[Supported modes](concepts.md#acquisition-modes-and-timing)
